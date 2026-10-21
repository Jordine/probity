"""Offline recovery tests: real frozen driver/client, mocked catalog and HTTP."""

import asyncio
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark import grid5_generate as gen
from probity_benchmark.preflight import load_problems

SCRIPT = Path(__file__).resolve().parents[1] / "runs/iteration5/coordinator/preparation_resume_v2.py"
SPEC = importlib.util.spec_from_file_location("grid5_preparation_resume_tested", SCRIPT)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class World:
    def __init__(self, root):
        self.root = root
        self.run = root / "runs/iteration5"
        self.meta = self.run / "coordinator"
        self.meta.mkdir(parents=True)
        self.source = self.meta / "source.jsonl"
        self.rows = [{"synthetic_story": "Ada visited the observatory. Bea stayed home.",
            "question": "Who visited the observatory?", "option_a": "Ada", "option_b": "Bea",
            "correct_answer": "A", "split": "dev", "schedule_parent_index": 0}]
        self.source.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        self.problems, self.data_audit = load_problems(self.source)
        self.catalog = {model: {"id": model, "supported_parameters": ["temperature", "structured_outputs"],
            "pricing": {"prompt": "0.00000026", "completion": "0.0000026", "request": "0"},
            "top_provider": {"context_length": 262144}, "context_length": 262144}
            for model in gen.MODELS}
        self.live_catalog = copy.deepcopy(self.catalog)
        self.jobs = resume.make_jobs(self.problems, self.rows, self.catalog)
        self.keys = list(self.jobs)
        self.posts = []
        self.timeout_remaining = Counter()
        self.ledger = {"cap_usd": 45.0, "entries": {}}
        self.old = self.run / "preparations_v1"
        self.write_batch(self.old, [self.receipt(self.keys[0]), self.receipt(self.keys[1]),
                                   self.receipt(self.keys[2], timeout=True)])

    def payload(self, request):
        parsed = {"evidence_items": [{"claim": "Claim " + str(i), "quotation": "Quote " + str(i)}
                                     for i in range(3)],
                  "anticipated_challenge": "An opponent may disagree.",
                  "response_to_challenge": "The evidence supports this account."}
        return {"id": "mock-" + gen.sha256(json.dumps(request, sort_keys=True)),
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(parsed)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 100, "cost": .001}}

    def receipt(self, key, timeout=False):
        job = copy.deepcopy(self.jobs[key])
        row = {**job, "model": key[1], "attempt": 0, "reserved_usd": .05,
               "valid": not timeout, "cost_usd": .05 if timeout else .001,
               "cost_basis": "reservation_upper_bound" if timeout else "provider_reported"}
        if timeout:
            row.update(error_type="TimeoutError", error="", duration_seconds=1800)
        else:
            payload = self.payload(row["request"])
            row.update(http_status=200, response_body=json.dumps(payload), usage=payload["usage"],
                finish_reason="stop", parsed=json.loads(payload["choices"][0]["message"]["content"]))
        return row

    def clone(self, receipt, source):
        return {**copy.deepcopy(receipt), "reused_from": str(source / "requests.jsonl"),
                "original_cost_usd": receipt["cost_usd"], "cost_usd": 0,
                "cost_basis": "previously_accounted_receipt"}

    def write_batch(self, directory, receipts):
        directory.mkdir()
        (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in receipts))
        records = []
        for r in receipts:
            if not r["valid"]:
                continue
            records.append({"problem_id": r["tag"]["problem_id"], "model": r["model"],
                "speaker": r["tag"]["speaker"], "assigned_answer": "A" if r["tag"]["speaker"] == "Red" else "B",
                "swapped": False, "source_story_sha256": gen.sha256(self.problems[0].story),
                "request_sha256": r["request_sha256"], "parsed": r["parsed"], "finish_reason": "stop",
                "max_tokens": 16384, "completion_tokens": r["usage"]["completion_tokens"],
                "reasoning_tokens": 0, "reasoning": {"enabled": True}})
        (directory / "preparations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        failures = [r for r in receipts if not r["valid"]]
        cost = sum(r["cost_usd"] for r in receipts)
        errors = [f"Invalid request {r['tag']}: {r.get('error', '')}" for r in failures]
        if len(records) < len(self.jobs):
            errors.append("pilot stopped or budget reservation unavailable")
        (directory / "summary.json").write_text(json.dumps({"errors": errors, "completed": len(records),
            "expected": len(self.jobs), "cost_usd": cost, "dataset_audit": self.data_audit}))
        (directory / "api_manifest.json").write_text(json.dumps({"models": self.catalog}))
        self.ledger["entries"]["api:" + str(directory)] = {"status": "settled", "amount_usd": cost,
            "kind": "api", "original_reservation_usd": 11, "reservation_exceeded": False}
        (self.run / "budget.json").write_text(json.dumps(self.ledger))

    def history(self, directories=None):
        return resume.audit_history(directories or [self.old], self.jobs,
                                    json.loads((self.run / "budget.json").read_text()))

    def session(self, **kwargs):
        world = self

        class Response:
            status = 200

            def __init__(self, body):
                self.body = body

            async def __aenter__(self):
                await asyncio.sleep(0)  # Let already-dispatched calls drain after a timeout.
                h = gen.sha256(json.dumps(self.body, sort_keys=True))
                if world.timeout_remaining[h]:
                    world.timeout_remaining[h] -= 1
                    raise TimeoutError()
                return self

            async def __aexit__(self, *args):
                pass

            async def text(self):
                return json.dumps(world.payload(self.body))

        class Session:
            def post(self, url, json):
                world.posts.append(copy.deepcopy(json))
                return Response(json)

            async def close(self):
                pass

        return Session()

    def patched(self):
        stack = ExitStack()
        for name, value in {"ROOT": self.root, "RUN": self.run, "META": self.meta,
                            "SOURCE": self.source, "EXPECTED": len(self.jobs)}.items():
            stack.enter_context(patch.object(resume, name, value))
        stack.enter_context(patch.object(resume, "verify_freezes"))
        stack.enter_context(patch.object(gen, "frozen_inventory", return_value=(self.problems, self.rows, self.data_audit, {})))
        stack.enter_context(patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-test-only",
                            "PROBITY_PILOT_LEDGER": str(self.run / "budget.json")}, clear=True))
        stack.enter_context(patch("probity_benchmark.api.urllib.request.urlopen",
            side_effect=lambda *a, **kw: io.StringIO(json.dumps({"data": list(self.live_catalog.values())}))))
        stack.enter_context(patch("probity_benchmark.api.aiohttp.ClientSession", self.session))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def factory(self):
        history = self.history()
        factory = resume.AuditedFactory(self.jobs, history, lambda: None)
        client = factory(self.run / "preparations_v2", list(gen.MODELS), 9.70,
            max_tokens=16384, reasoning={"enabled": True}, schema_retries=2,
            reuse_receipts=list(reversed(history.receipts)), request_timeout_seconds=1800)
        return factory, client


class PreparationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = World(Path(self.temp.name))

    def test_real_driver_timeout_drain_then_missing_only_completion(self):
        w = self.w
        timed_hash = w.jobs[w.keys[2]]["request_sha256"]
        w.timeout_remaining[timed_hash] = 1
        with w.patched():
            marker = asyncio.run(resume.run("approved-offline-freeze"))
        self.assertTrue(marker["complete"])
        self.assertEqual(marker["preparations_path"], "runs/iteration5/preparations_v3/preparations.jsonl")
        self.assertEqual(len(w.posts), 3)  # timeout+drained peer, then only the timed-out job
        counts = Counter(gen.sha256(json.dumps(b, sort_keys=True)) for b in w.posts)
        self.assertEqual(counts, {timed_hash: 2, w.jobs[w.keys[3]]["request_sha256"]: 1})
        history = w.history([w.old, w.run / "preparations_v2", w.run / "preparations_v3"])
        self.assertFalse(history.missing)
        self.assertEqual(history.attempts[w.keys[2]], 3)  # includes the original timeout
        for directory in [w.run / "preparations_v2", w.run / "preparations_v3"]:
            audit = resume.read(directory / "precall_contract_audit_v2.json")
            self.assertEqual(audit["paid_posts_before_audit"], 0)
            self.assertEqual(audit["all_requests_checked"], 4)
        for row in gen.read_rows(w.run / "preparations_v3/requests.jsonl"):
            if resume.job_key(row["tag"]) in w.keys[:2]:
                self.assertEqual(row["reused_from"], str(w.old / "requests.jsonl"))
                self.assertEqual(row["cost_usd"], 0)
        self.assertTrue(all(e["status"] == "settled" for e in resume.read(w.run / "budget.json")["entries"].values()))
        self.assertLess(marker["total_preparation_cost_usd"], 11)

    def test_catalog_price_drift_rejected_before_any_post_and_zero_settled(self):
        self.w.live_catalog[gen.MODELS[0]]["pricing"]["prompt"] = "0.1"
        with self.w.patched(), self.assertRaisesRegex(ValueError, "live catalog"):
            self.w.factory()
        self.assertFalse(self.w.posts)
        entry = resume.read(self.w.run / "budget.json")["entries"]["api:" + str(self.w.run / "preparations_v2")]
        self.assertEqual((entry["status"], entry["amount_usd"]), ("settled", 0))

    def test_missing_cache_rejected_before_any_post(self):
        original = resume.BudgetedChat

        def bad_client(*args, **kwargs):
            client = original(*args, **kwargs)
            client.reusable.pop(next(iter(client.reusable)))
            return client

        with self.w.patched(), patch.object(resume, "BudgetedChat", bad_client):
            with self.assertRaisesRegex(ValueError, "miss the exact receipt cache"):
                self.w.factory()
        self.assertFalse(self.w.posts)

    def test_context_reduction_rejected_before_any_post(self):
        self.w.live_catalog[gen.MODELS[0]]["top_provider"]["context_length"] = 100
        with self.w.patched(), self.assertRaisesRegex(ValueError, "context allowance"):
            self.w.factory()
        self.assertFalse(self.w.posts)

    def test_changed_single_call_body_rejected_before_post(self):
        with self.w.patched():
            _, client = self.w.factory()
            job = copy.deepcopy(self.w.jobs[self.w.keys[2]])
            job["request"]["messages"][0]["content"] += " changed"
            async def call():
                async with client:
                    await client.complete(job["tag"]["model"], job["request"]["messages"],
                        gen.preparation_schema(), job["tag"])
            with self.assertRaisesRegex(ValueError, "per-call request/tag"):
                asyncio.run(call())
        self.assertFalse(self.w.posts)

    def test_attempt_guard_prevents_fourth_post_even_inside_original_schema_retry(self):
        with self.w.patched():
            factory, client = self.w.factory()
            key = self.w.keys[2]
            factory.history.attempts[key] = 3
            job = self.w.jobs[key]
            async def call():
                async with client:
                    await client.complete(key[1], job["request"]["messages"], gen.preparation_schema(), job["tag"], _attempt=2)
            with self.assertRaisesRegex(RuntimeError, "attempt limit"):
                asyncio.run(call())
        self.assertFalse(self.w.posts)

    def test_three_failed_transport_attempts_stop_without_fourth_post(self):
        self.w.timeout_remaining[self.w.jobs[self.w.keys[2]]["request_sha256"]] = 3
        with self.w.patched(), self.assertRaisesRegex(RuntimeError, "transport recovery exhausted"):
            asyncio.run(resume.run("approved-offline-freeze"))
        self.assertEqual(sum(gen.sha256(json.dumps(b, sort_keys=True)) ==
            self.w.jobs[self.w.keys[2]]["request_sha256"] for b in self.w.posts), 2)
        self.assertFalse((self.w.run / "preparations_v4").exists())
        self.assertFalse((self.w.meta / "preparations_recovery_ready_v2.json").exists())

    def test_clone_raw_response_mutation_is_rejected(self):
        receipts = gen.read_rows(self.w.old / "requests.jsonl")
        clones = [self.w.clone(r, self.w.old) for r in receipts if r["valid"]]
        clones[0]["response_body"] += " "
        later = self.w.run / "preparations_v2"
        self.w.write_batch(later, clones + [self.w.receipt(self.w.keys[2], timeout=True)])
        with self.assertRaisesRegex(ValueError, "cache clone differs"):
            self.w.history([self.w.old, later])

    def test_accepted_job_cannot_be_paid_for_again_even_identical_output(self):
        later = self.w.run / "preparations_v2"
        self.w.write_batch(later, [self.w.receipt(self.w.keys[0])])
        with self.assertRaisesRegex(ValueError, "paid for again"):
            self.w.history([self.w.old, later])

    def test_provider_block_is_not_transport_recoverable(self):
        p = self.w.old / "requests.jsonl"
        rows = gen.read_rows(p)
        rows[-1].update(error_type="RuntimeError", http_status=200, finish_reason="error",
            response_body=json.dumps({"choices": [{"error": {"message": "DataInspectionFailed"}}]}))
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "nontransport failure"):
            self.w.history()

    def test_unsettled_prior_batch_prevents_continuation(self):
        ledger = resume.read(self.w.run / "budget.json")
        next(iter(ledger["entries"].values()))["status"] = "reserved"
        with self.assertRaisesRegex(ValueError, "not safely settled"):
            resume.audit_history([self.w.old], self.w.jobs, ledger)

    def test_unknown_timeout_charge_cannot_be_discarded(self):
        ledger = resume.read(self.w.run / "budget.json")
        next(iter(ledger["entries"].values()))["amount_usd"] -= .05
        with self.assertRaisesRegex(ValueError, "charges, receipts"):
            resume.audit_history([self.w.old], self.w.jobs, ledger)

    def test_original_envelope_not_reset_between_batches(self):
        self.assertEqual(resume.batch_budget(SimpleNamespace(cost=1.2943549709999993)), 9.70)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            resume.batch_budget(SimpleNamespace(cost=10.994))

    def test_transport_classifier_requires_no_http_response(self):
        row = self.w.receipt(self.w.keys[2], timeout=True)
        self.assertTrue(resume.transport_only(row))
        self.assertFalse(resume.transport_only({**row, "http_status": 429}))
        self.assertFalse(resume.transport_only({**row, "response_body": ""}))


if __name__ == "__main__":
    unittest.main()
