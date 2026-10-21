"""CPU-only exact-quarantine recovery tests; no real provider or API key."""

import asyncio
from collections import Counter
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).resolve().parents[1]
resume = module("reviewed_preparation_tested", ROOT / "runs/iteration5/coordinator/preparation_resume_v3.py")
old_tests = module("frozen_preparation_test_helpers", ROOT / "tests/test_grid5_preparation_resume.py")
gen = resume.generation


class ReviewedWorld(old_tests.World):
    def __init__(self, root):
        super().__init__(root)
        originals = gen.read_rows(self.old / "requests.jsonl")
        cached = [r for r in originals if r["valid"]]
        self.write_batch(self.run / "preparations_v2", [self.clone(r, self.old) for r in cached]
                         + [self.receipt(self.keys[2], timeout=True)])
        mismatch = self.receipt(self.keys[3])
        payload = json.loads(mismatch["response_body"])
        payload["usage"].update(completion_tokens=82150, cost=.128880375)
        payload["usage"]["completion_tokens_details"] = {"reasoning_tokens": 81920}
        mismatch.update(valid=False, error_type="ValueError", error="provider exceeded reservation",
                        cost_usd=.128880375, usage=payload["usage"], response_body=json.dumps(payload))
        mismatch.pop("parsed")
        mismatch.pop("finish_reason")
        self.write_batch(self.run / "preparations_v3", [self.clone(r, self.old) for r in cached]
                         + [self.receipt(self.keys[2]), mismatch])
        self.directories = [self.run / f"preparations_v{v}" for v in range(1, 4)]
        # The old test helper writes zero instead of the real driver's absent
        # reasoning-token detail. Match the real driver for whole-record checks.
        for directory in self.directories:
            path = directory / "preparations.jsonl"
            records = gen.read_rows(path)
            for record in records:
                record["reasoning_tokens"] = None
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
        self.qpath = self.run / "preparations_v3/requests.jsonl"
        self.qline = self.qpath.read_bytes().splitlines()[-1]
        self.qrow = json.loads(self.qline)
        self.inventory = self.meta / "inventory.json"
        self.inventory.write_text("{}")
        for name in ("execution_code_freeze_v1.json", "execution_recovery_freeze_v2.json",
                     "preparation_liability_policy_v3.json"):
            (self.meta / name).write_text("{}")

    def patched(self, liability=.25):
        stack = ExitStack()
        stack.enter_context(patch.object(old_tests, "resume", resume))
        stack.enter_context(super().patched())
        values = {"INVENTORY": self.inventory, "RETAINED_PRIOR": 3,
            "RECOVERY_FREEZE": self.meta / "execution_recovery_freeze_v3.json",
            "HISTORY_MANIFEST": self.meta / "preparation_reviewed_history_v3.json",
            "LIABILITY_POLICY": self.meta / "preparation_liability_policy_v3.json",
            "QUARANTINE_LINE": 4, "QUARANTINE_LINE_SHA": gen.sha256(self.qline.decode()),
            "QUARANTINE_REQUEST_SHA": self.qrow["request_sha256"]}
        for key, value in values.items():
            stack.enter_context(patch.object(resume, key, value))
        if liability is not None:
            fake_guard = SimpleNamespace(
                ensure_historical_topups=lambda *args: liability,
                wrap_client=lambda client, *args: client)
            stack.enter_context(patch.object(resume, "load_liability", return_value=fake_guard))
        else:
            stack.enter_context(patch.object(resume, "load_liability",
                side_effect=ValueError("reviewed liability policy is not approved")))
        return stack

    def freeze_manifest(self):
        manifest = resume.stopped_history(self.jobs)
        resume.atomic(resume.HISTORY_MANIFEST, manifest, immutable=True)
        return manifest

    def history(self, directories=None):
        return resume.audit_history(directories or self.directories, self.jobs,
                                    resume.read(self.run / "budget.json"))

    def factory(self):
        history = self.history()
        factory = resume.AuditedFactory(self.jobs, history, lambda: None)
        client = factory(self.run / "preparations_v4", list(gen.MODELS), 9.0,
            max_tokens=16384, reasoning={"enabled": True}, schema_retries=2,
            reuse_receipts=list(reversed(history.receipts)), request_timeout_seconds=1800)
        return factory, client


class ReviewedPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.w = ReviewedWorld(Path(temporary.name))

    def test_identical_missing_only_retry_preserves_all_canonical_records_and_charges(self):
        w = self.w
        with w.patched():
            manifest = w.freeze_manifest()
            old_hashes = {str(path): gen.file_hash(path) for d in w.directories for path in d.iterdir()}
            marker = asyncio.run(resume.run("offline-review-freeze"))
            history = w.history(w.directories + [w.run / "preparations_v4"])
            resume.verify_canonical(history, manifest)
            self.assertEqual({str(Path(p)): gen.file_hash(Path(p)) for p in old_hashes}, old_hashes)
            self.assertEqual(len(w.posts), 1)
            self.assertEqual(w.posts[0], w.jobs[w.keys[3]]["request"])
            self.assertEqual(history.attempts[w.keys[3]], 2)
            self.assertFalse(history.missing)
            self.assertEqual(marker["retained_prior_accepted"], 3)
            self.assertEqual(marker["reviewed_quarantine_count"], 1)
            self.assertEqual(marker["transport_failures_retained"], 2)
            self.assertEqual(marker["additional_preparation_liability_usd"], .25)
            self.assertAlmostEqual(marker["total_preparation_cost_usd"], manifest["cost_usd"] + .001)
            audit = resume.read(w.run / "preparations_v4/precall_contract_audit_v2.json")
            self.assertEqual((audit["all_requests_checked"], audit["canonical_cache_hits_verified"],
                              audit["paid_posts_before_audit"]), (4, 3, 0))
            self.assertFalse(w.qrow["valid"])
            self.assertFalse(resume.transport_only(w.qrow))

    def test_exact_exception_copy_in_new_batch_is_not_admitted(self):
        w = self.w
        w.write_batch(w.run / "preparations_v4", [w.qrow])
        with w.patched(), self.assertRaisesRegex(ValueError, "new or unreviewed nontransport"):
            w.history(w.directories + [w.run / "preparations_v4"])

    def test_modified_historical_raw_line_is_rejected_before_any_call(self):
        w = self.w
        with w.patched():
            w.freeze_manifest()
            lines = w.qpath.read_text().splitlines()
            lines[-1] += " "
            w.qpath.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "nontransport"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertFalse(w.posts)

    def test_changed_request_cannot_match_by_tag_only(self):
        w = self.w
        lines = w.qpath.read_text().splitlines()
        row = json.loads(lines[-1])
        row["request"]["temperature"] = .1
        lines[-1] = json.dumps(row)
        w.qpath.write_text("\n".join(lines) + "\n")
        with w.patched(), self.assertRaisesRegex(ValueError, "frozen exact preparation"):
            w.history()

    def test_actual_quarantine_charge_cannot_be_reduced_to_reservation(self):
        w = self.w
        ledger = resume.read(w.run / "budget.json")
        ledger["entries"]["api:" + str(w.directories[-1])]["amount_usd"] -= .078880375
        with w.patched(), self.assertRaisesRegex(ValueError, "charges, receipts"):
            resume.audit_history(w.directories, w.jobs, ledger)

    def test_whole_preparation_record_preserved_not_just_parsed_output(self):
        w = self.w
        path = w.run / "preparations_v3/preparations.jsonl"
        rows = gen.read_rows(path)
        rows[0]["assigned_answer"] = "B"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with w.patched(), self.assertRaisesRegex(ValueError, "earliest accepted record"):
            w.history()

    def test_paid_duplicate_of_accepted_generation_is_rejected(self):
        w = self.w
        w.write_batch(w.run / "preparations_v4", [w.receipt(w.keys[0])])
        with w.patched(), self.assertRaisesRegex(ValueError, "paid for again"):
            w.history(w.directories + [w.run / "preparations_v4"])

    def test_live_price_change_fails_before_post(self):
        w = self.w
        w.live_catalog[gen.MODELS[0]]["pricing"]["prompt"] = "0.1"
        with w.patched(), self.assertRaisesRegex(ValueError, "live catalog"):
            w.factory()
        self.assertFalse(w.posts)

    def test_reduced_live_context_fails_before_post(self):
        w = self.w
        w.live_catalog[gen.MODELS[0]]["top_provider"]["context_length"] = 100
        with w.patched(), self.assertRaisesRegex(ValueError, "context allowance"):
            w.factory()
        self.assertFalse(w.posts)

    def test_missing_exact_cache_fails_before_post(self):
        w = self.w
        original = resume.base.BudgetedChat
        def altered(*args, **kwargs):
            client = original(*args, **kwargs)
            client.reusable.pop(next(iter(client.reusable)))
            return client
        with w.patched(), patch.object(resume.base, "BudgetedChat", altered):
            with self.assertRaisesRegex(ValueError, "exact receipt cache"):
                w.factory()
        self.assertFalse(w.posts)

    def test_new_overrun_stops_after_drain_and_retains_actual_charge(self):
        w = self.w
        original = w.payload
        def overrun(request):
            body = original(request)
            body["usage"].update(completion_tokens=82150, cost=.128880375)
            return body
        with w.patched(), patch.object(w, "payload", overrun):
            w.freeze_manifest()
            with self.assertRaisesRegex(RuntimeError, "new or unreviewed nontransport"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertEqual(len(w.posts), 1)
        receipt = gen.read_rows(w.run / "preparations_v4/requests.jsonl")[-1]
        self.assertFalse(receipt["valid"])
        self.assertAlmostEqual(receipt["cost_usd"], .128880375)
        self.assertFalse((w.run / "preparations_v5").exists())

    def test_new_schema_failure_cannot_trigger_unreviewed_paid_retry(self):
        w = self.w
        original = w.payload
        def malformed(request):
            body = original(request)
            body["choices"][0]["message"]["content"] = "{not JSON}"
            return body
        with w.patched(), patch.object(w, "payload", malformed):
            w.freeze_manifest()
            with self.assertRaisesRegex(RuntimeError, "new or unreviewed nontransport"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertEqual(len(w.posts), 1)
        self.assertFalse((w.run / "preparations_v5").exists())

    def test_original_three_attempt_ceiling_includes_reviewed_failure(self):
        w = self.w
        w.timeout_remaining[w.jobs[w.keys[3]]["request_sha256"]] = 1
        with w.patched():
            w.freeze_manifest()
            marker = asyncio.run(resume.run("offline-review-freeze"))
            history = w.history(w.directories + [w.run / "preparations_v4", w.run / "preparations_v5"])
        self.assertEqual(len(w.posts), 2)
        self.assertEqual(history.attempts[w.keys[3]], 3)
        self.assertTrue(marker["complete"])

    def test_third_failed_attempt_prevents_fourth_paid_post(self):
        w = self.w
        w.timeout_remaining[w.jobs[w.keys[3]]["request_sha256"]] = 3
        with w.patched():
            w.freeze_manifest()
            with self.assertRaisesRegex(RuntimeError, "recovery exhausted"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertEqual(len(w.posts), 2)
        self.assertFalse((w.run / "preparations_v6").exists())

    def test_original_batch_count_is_not_reset(self):
        w = self.w
        with w.patched(), patch.object(resume, "MAX_NEW_BATCHES", 2):
            w.freeze_manifest()
            with self.assertRaisesRegex(RuntimeError, "six original recovery batches"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertFalse(w.posts)

    def test_separate_liability_is_subtracted_and_cannot_reset_prep_cap(self):
        self.assertEqual(resume.batch_budget(SimpleNamespace(cost=3.617581936), 1.5), 5.88)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            resume.batch_budget(SimpleNamespace(cost=10.0), 1.01)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            resume.batch_budget(SimpleNamespace(cost=3.6), -.1)

    def test_no_paid_calls_until_liability_policy_is_approved(self):
        w = self.w
        with w.patched(liability=None):
            w.freeze_manifest()
            with self.assertRaisesRegex(ValueError, "liability policy"):
                asyncio.run(resume.run("offline-review-freeze"))
        self.assertFalse(w.posts)
        self.assertFalse((w.run / "preparations_v4").exists())

    def test_missing_q38_finishes_before_q35_while_cache_hits_stay_immediate(self):
        tags = [{"problem_id": "p", "model": model, "speaker": speaker}
                for model, speaker in [(gen.MODELS[0], "Red"), (gen.MODELS[1], "Red"),
                                       (gen.MODELS[0], "Blue")]]
        history = SimpleNamespace(missing={resume.job_key(t) for t in tags[:2]})
        client = SimpleNamespace(records=[], stopped=False)
        events = []
        async def scenario():
            release = asyncio.Event()
            async def original(model, messages, schema, tag, **options):
                events.append((tag["model"], tag["speaker"], "start"))
                if tag is tags[1]:
                    await release.wait()
                events.append((tag["model"], tag["speaker"], "finish"))
                return tag
            guarded = resume.ordered_completion(client, history, original)
            tasks = [asyncio.create_task(guarded(t["model"], [], {}, t)) for t in tags]
            await asyncio.sleep(0)
            self.assertNotIn((gen.MODELS[0], "Red", "start"), events)
            self.assertIn((gen.MODELS[0], "Blue", "finish"), events)
            release.set()
            await asyncio.gather(*tasks)
        asyncio.run(scenario())
        self.assertLess(events.index((gen.MODELS[1], "Red", "finish")),
                        events.index((gen.MODELS[0], "Red", "start")))

    def test_q38_failure_releases_deferred_q35_for_a_drained_stop(self):
        tags = [{"problem_id": "p", "model": model, "speaker": "Red"} for model in gen.MODELS]
        history = SimpleNamespace(missing={resume.job_key(t) for t in tags})
        client = SimpleNamespace(records=[], stopped=False)
        async def original(model, *args, **kwargs):
            if model == gen.MODELS[1]:
                client.records.append({"valid": False, "error_type": "ValueError"})
                raise ValueError("technical failure")
            self.fail("Deferred Q35 must not dispatch after a Q38 technical failure")
        async def scenario():
            guarded = resume.ordered_completion(client, history, original)
            return await asyncio.wait_for(asyncio.gather(*[
                guarded(t["model"], [], {}, t) for t in tags], return_exceptions=True), timeout=1)
        outcomes = asyncio.run(scenario())
        self.assertIsInstance(outcomes[0], RuntimeError)
        self.assertIsInstance(outcomes[1], ValueError)
        self.assertTrue(client.stopped)

    def test_recursive_q38_return_does_not_release_q35_before_outer_return(self):
        tags = [{"problem_id": "p", "model": model, "speaker": "Red"} for model in gen.MODELS]
        history = SimpleNamespace(missing={resume.job_key(t) for t in tags})
        client = SimpleNamespace(records=[], stopped=False)
        async def scenario():
            inner_finished, outer_release = asyncio.Event(), asyncio.Event()
            q35_started = []
            async def original(model, messages, schema, tag, **options):
                if model == gen.MODELS[0]:
                    q35_started.append(True)
                elif not options.get("_attempt"):
                    await client.complete(model, messages, schema, tag, _attempt=1)
                    inner_finished.set()
                    await outer_release.wait()
                return tag
            client.complete = resume.ordered_completion(client, history, original)
            tasks = [asyncio.create_task(client.complete(t["model"], [], {}, t)) for t in tags]
            await inner_finished.wait()
            await asyncio.sleep(0)
            self.assertFalse(q35_started)
            outer_release.set()
            await asyncio.gather(*tasks)
            self.assertTrue(q35_started)
        asyncio.run(scenario())

    def test_stopped_q38_wakes_q35_while_other_q38_is_still_draining(self):
        tags = [{"problem_id": "p", "model": model, "speaker": speaker}
                for model, speaker in [(gen.MODELS[0], "Red"), (gen.MODELS[1], "Red"),
                                       (gen.MODELS[1], "Blue")]]
        history = SimpleNamespace(missing={resume.job_key(t) for t in tags})
        client = SimpleNamespace(records=[], stopped=False)
        async def scenario():
            draining, release = asyncio.Event(), asyncio.Event()
            async def original(model, messages, schema, tag, **options):
                if model == gen.MODELS[0]:
                    self.fail("Stopped uncached Q35 must be rejected before a lease or POST")
                if tag["speaker"] == "Red":
                    await draining.wait()
                    client.stopped = True
                    raise TimeoutError()
                draining.set()
                await release.wait()
                return tag
            guarded = resume.ordered_completion(client, history, original)
            tasks = [asyncio.create_task(guarded(t["model"], [], {}, t)) for t in tags]
            try:
                with self.assertRaisesRegex(RuntimeError, "pilot stopped"):
                    await asyncio.wait_for(tasks[0], timeout=1)
                self.assertFalse(tasks[2].done())
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.run(scenario())

    def test_real_financial_proxy_keeps_outer_order_guard_and_settles_raw_charge(self):
        w = self.w
        guard = module("integrated_liability_guard_tested", resume.LIABILITY_GUARD)
        policy_path = w.meta / "preparation_liability_policy_v3.json"
        policy_path.write_text(json.dumps({
            "schema": "grid5_preparation_liability_policy_v3", "ready": True,
            "preparation_cap_usd": 11, "campaign_cap_usd": 45, "concurrency": 4,
            "global_context_tokens": {m: 262144 for m in gen.MODELS},
            "catalog_path": str(w.old / "api_manifest.json"),
            "catalog_sha256": gen.file_hash(w.old / "api_manifest.json"),
            "run_directory": str(w.run), "historical_unknowns": []}))
        seen = []
        original_order = resume.ordered_completion
        def traced_order(client, history, original):
            wrapped = original_order(client, history, original)
            async def traced(model, messages, schema, tag, **options):
                seen.append(copy.deepcopy(tag))
                return await wrapped(model, messages, schema, tag, **options)
            return traced
        # Historical-top-up arithmetic has independent guard tests; this case
        # integrates the real proxy/POST leases/drain/settlement with the frozen
        # client and our outer request scheduling on a compact four-job fixture.
        with w.patched(), patch.object(resume, "load_liability", return_value=guard), \
                patch.object(guard, "ensure_historical_topups", side_effect=lambda *args:
                    guard.validate_liability_ledger(w.run / "budget.json", policy_path)["additional_liability_usd"]), \
                patch.object(guard, "CONTEXTS", {m: 262144 for m in gen.MODELS}), \
                patch.object(guard, "PRICES", {m: ("0.00000026", "0.0000026") for m in gen.MODELS}), \
                patch.object(resume, "ordered_completion", traced_order):
            # Fund this compact fixture's two historical unknown attempts so the
            # real constructor can enforce its new funded-ledger precondition.
            entries = {}
            for directory in w.directories:
                entries.update(guard.expected_topups(directory / "requests.jsonl"))
            ledger = resume.read(w.run / "budget.json")
            ledger["entries"].update(entries)
            (w.run / "budget.json").write_text(json.dumps(ledger))
            policy = resume.read(policy_path)
            policy["historical_unknowns"] = [
                {**{name: value[name] for name in ("receipt_path", "receipt_line",
                    "receipt_line_sha256", "request_sha256")},
                 "ledger_key": key, "additional_liability_usd": value["amount_usd"]}
                for key, value in entries.items()]
            policy_path.write_text(json.dumps(policy))
            w.freeze_manifest()
            marker = asyncio.run(resume.run("offline-review-freeze"))
        self.assertTrue(marker["complete"])
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(w.posts), 1)
        proof = resume.read(w.run / "preparations_v4/liability_settlement_v3.json")
        self.assertEqual((proof["posts"], proof["raw_receipt_charge_usd"],
                          proof["additional_unknown_liability_usd"]), (1, .001, 0))
        entry = resume.read(w.run / "budget.json")["entries"]["api:" + str(w.run / "preparations_v4")]
        self.assertEqual((entry["status"], entry["amount_usd"]), ("settled", .001))


if __name__ == "__main__":
    unittest.main()
