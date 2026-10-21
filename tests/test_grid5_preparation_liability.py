"""Real frozen client with fake transport and temporary-only budget ledgers."""

import asyncio
from contextlib import ExitStack
import copy
from decimal import Decimal
import importlib.util
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark import api

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("liability_guard_tested",
    ROOT / "runs/iteration5/coordinator/preparation_liability_guard_v3.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class World:
    def __init__(self, directory, count=6, model="qwen/qwen3.8-27b"):
        self.run = Path(directory) / "runs/iteration5"
        self.run.mkdir(parents=True)
        self.ledger = self.run / "budget.json"
        guard.atomic(self.ledger, {"cap_usd": 45, "entries": {}})
        self.catalog = {m: {"id": m, "context_length": c,
            "top_provider": {"context_length": 262144, "max_completion_tokens": 81920},
            "pricing": dict(zip(("prompt", "completion"), guard.PRICES[m])),
            "supported_parameters": ["temperature", "structured_outputs"]}
            for m, c in guard.CONTEXTS.items()}
        self.catalog_path = self.run / "catalog.json"
        guard.atomic(self.catalog_path, {"models": self.catalog})
        self.policy = self.run / "policy.json"
        guard.atomic(self.policy, {"schema": "grid5_preparation_liability_policy_v3", "ready": True,
            "preparation_cap_usd": 11, "campaign_cap_usd": 45, "concurrency": 4,
            "global_context_tokens": guard.CONTEXTS, "catalog_path": str(self.catalog_path),
            "catalog_sha256": guard.digest(self.catalog_path.read_bytes()),
            "run_directory": str(self.run), "historical_unknowns": []})
        self.schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"], "additionalProperties": False}
        self.jobs = {}
        for i in range(count):
            tag = {"problem_id": str(i), "model": model, "speaker": "Red", "stage": "reasoned_preparation"}
            prices = self.catalog[model]["pricing"]
            body = {"model": model, "messages": [{"role": "user", "content": "job " + str(i)}],
                    "max_tokens": 16384, "reasoning": {"enabled": True}, "usage": {"include": True},
                    "provider": {"require_parameters": True, "max_price": {
                        "prompt": float(prices["prompt"]) * 1e6,
                        "completion": float(prices["completion"]) * 1e6, "request": 0.0}},
                    "temperature": 0, "response_format": {"type": "json_schema", "json_schema": {
                        "name": "diagnostic", "strict": True, "schema": self.schema}}}
            self.jobs[(str(i), model, "Red")] = {"tag": tag, "request": body,
                                                   "request_sha256": guard.body_hash(body)}
        self.posts, self.active, self.peak = [], 0, 0
        self.behavior = "normal"
        self.gate = None
        self.close_fails = False

    def payload(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "cost": .001}}

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-only-no-network",
                                                    "PROBITY_PILOT_LEDGER": str(self.ledger)}))
        stack.enter_context(patch.object(api.urllib.request, "urlopen", side_effect=lambda *a, **k:
            io.BytesIO(json.dumps({"data": list(self.catalog.values())}).encode())))
        stack.enter_context(patch.object(api.aiohttp, "ClientSession", side_effect=lambda **k: self.session()))
        return stack

    def session(self):
        world = self
        class Response:
            status = 200
            async def __aenter__(self):
                world.active += 1
                world.peak = max(world.active, world.peak)
                if world.behavior == "timeout":
                    world.active -= 1
                    raise TimeoutError()
                if world.gate is not None:
                    await world.gate.wait()
                await asyncio.sleep(.01)
                return self
            async def text(self):
                data = world.payload()
                if world.behavior == "schema_once" and len(world.posts) == 1:
                    data["choices"][0]["message"]["content"] = "broken JSON"
                if world.behavior == "old_allowance_overrun":
                    data["usage"].update(completion_tokens=82150, cost=.128880375)
                return json.dumps(data)
            async def __aexit__(self, *args):
                world.active -= 1
        class Session:
            def post(self, url, **kwargs):
                world.posts.append(copy.deepcopy(kwargs["json"]))
                return Response()
            async def close(self):
                if world.close_fails:
                    raise OSError("mock close failure")
        return Session()

    def client(self, budget=3):
        raw = api.BudgetedChat(self.run / "preparations_v4", list(self.catalog), budget,
            max_tokens=16384, reasoning={"enabled": True}, schema_retries=2)
        proxy = guard.wrap_client(raw, self.jobs, lambda: None, self.policy, SimpleNamespace(cost=0))
        return raw, proxy

    async def call(self, raw, job):
        b = job["request"]
        return await raw.complete(b["model"], b["messages"], self.schema, job["tag"])

    def historical_timeouts(self):
        path = self.run / "preparations_v1/requests.jsonl"
        path.parent.mkdir()
        rows, proofs = [], []
        for job in list(self.jobs.values())[:5]:
            body = job["request"]
            prompt = len(json.dumps([body["messages"], self.schema], ensure_ascii=False).encode()) + 2048
            charge = prompt * .00000026 + 16384 * .0000026
            row = {"tag": job["tag"], "model": body["model"], "request": body,
                   "request_sha256": job["request_sha256"], "reserved_usd": charge,
                   "valid": False, "attempt": 0, "error": "", "error_type": "TimeoutError",
                   "cost_usd": charge, "cost_basis": "reservation_upper_bound"}
            raw = json.dumps(row).encode()
            rows.append(raw)
            proofs.append({"receipt_path": str(path), "receipt_line": len(rows),
                "receipt_line_sha256": guard.digest(raw), "request_sha256": job["request_sha256"],
                "ledger_key": "unknown_provider_liability:" + str(path) + ":" + str(len(rows)),
                "additional_liability_usd": .638976})
        path.write_bytes(b"\n".join(rows) + b"\n")
        policy = guard.read(self.policy)
        policy["historical_unknowns"] = proofs
        guard.atomic(self.policy, policy)
        cost = sum(json.loads(raw)["cost_usd"] for raw in rows)
        ledger = guard.read(self.ledger)
        ledger["entries"]["api:" + str(path.parent)] = {
            "kind": "api", "status": "settled", "amount_usd": cost,
            "original_reservation_usd": 11, "reservation_exceeded": False}
        guard.atomic(self.ledger, ledger)
        return SimpleNamespace(cost=cost)


class LiabilityTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)

    def test_semantics_independent_bounds_match_frozen_prices(self):
        w = World(self.path, 1)
        b = next(iter(w.jobs.values()))["request"]
        expected = (Decimal(len(json.dumps([b["messages"], w.schema], ensure_ascii=False).encode()) + 2048)
                    * Decimal(".00000024") + Decimal("2.2"))
        self.assertEqual(guard.request_bound(b), expected)
        changed = copy.deepcopy(b)
        changed["max_tokens"] = 8192
        with self.assertRaises(ValueError):
            guard.request_bound(changed)

    def test_waits_for_settlement_and_runs_exact_bodies_under_single_request_headroom(self):
        w = World(self.path, 4)
        async def run():
            raw, proxy = w.client(3)
            async with proxy as entered:
                self.assertIs(entered, raw)
                await asyncio.wait_for(asyncio.gather(*(w.call(entered, j) for j in w.jobs.values())), 3)
            self.assertEqual(w.peak, 1)
            self.assertEqual(w.posts, [j["request"] for j in w.jobs.values()])
            result = guard.validate_liability_ledger(w.ledger, w.policy)
            self.assertAlmostEqual(result["preparation_committed_usd"], .004)
        with w.patches():
            asyncio.run(run())

    def test_max_four_concurrent_leases_even_with_large_available_budget(self):
        w = World(self.path, 6, "qwen/qwen3.5-27b")
        async def run():
            _, proxy = w.client(8)
            async with proxy as raw:
                await asyncio.gather(*(w.call(raw, j) for j in w.jobs.values()))
        with w.patches():
            asyncio.run(run())
        self.assertEqual(w.peak, 4)

    def test_schema_retry_releases_prior_attempt_after_receipt_without_deadlock(self):
        w = World(self.path, 1)
        w.behavior = "schema_once"
        async def run():
            _, proxy = w.client(2.21)
            async with proxy as raw:
                result = await asyncio.wait_for(w.call(raw, next(iter(w.jobs.values()))), 3)
                self.assertTrue(result["valid"])
            self.assertEqual([r["attempt"] for r in raw.records], [0, 1])
            self.assertEqual(len(w.posts), 2)
            self.assertEqual(proxy.completed_posts, 2)
        with w.patches():
            asyncio.run(run())

    def test_timeout_settles_raw_charge_and_separate_unknown_topup_atomically(self):
        w = World(self.path, 1, "qwen/qwen3.5-27b")
        w.behavior = "timeout"
        async def run():
            _, proxy = w.client(1)
            async with proxy as raw:
                with self.assertRaises(RuntimeError):
                    await w.call(raw, next(iter(w.jobs.values())))
            ledger = guard.read(w.ledger)
            entry = ledger["entries"][raw.budget_key]
            self.assertEqual(entry["amount_usd"], raw.records[0]["cost_usd"])
            extra = [x for x in ledger["entries"].values() if x["kind"] == "unknown_provider_liability"]
            self.assertEqual(len(extra), 1)
            self.assertAlmostEqual(extra[0]["amount_usd"], .638976)
            report = guard.validate_liability_ledger(w.ledger, w.policy)
            self.assertAlmostEqual(report["preparation_committed_usd"], float(guard.request_bound(w.posts[0])))
        with w.patches():
            asyncio.run(run())

    def test_known_allowance_overrun_remains_invalid_and_actual_cost_is_recorded(self):
        w = World(self.path, 1, "qwen/qwen3.5-27b")
        w.behavior = "old_allowance_overrun"
        async def run():
            _, proxy = w.client(1)
            async with proxy as raw:
                with self.assertRaises(RuntimeError):
                    await w.call(raw, next(iter(w.jobs.values())))
            self.assertFalse(raw.records[0]["valid"])
            self.assertEqual(raw.records[0]["error"], "provider exceeded reservation")
            self.assertAlmostEqual(guard.read(w.ledger)["entries"][raw.budget_key]["amount_usd"], .128880375)
            self.assertEqual(proxy.extra, 0)
        with w.patches():
            asyncio.run(run())

    def test_insufficient_alone_halts_before_post_and_creates_no_fake_receipt(self):
        w = World(self.path, 1)
        async def run():
            _, proxy = w.client(1)
            (proxy.out / "requests.jsonl").write_text("")
            async with proxy as raw:
                with self.assertRaisesRegex(RuntimeError, guard.STOP):
                    await w.call(raw, next(iter(w.jobs.values())))
            self.assertFalse(w.posts)
            self.assertFalse(raw.records)
            self.assertEqual(guard.read(w.ledger)["entries"][raw.budget_key]["amount_usd"], 0)
        with w.patches():
            asyncio.run(run())

    def test_cancellation_retains_full_batch_reservation(self):
        w = World(self.path, 1)
        async def run():
            w.gate = asyncio.Event()
            raw, proxy = w.client(3)
            with self.assertRaisesRegex(ValueError, "incomplete preparation drain"):
                async with proxy:
                    task = asyncio.create_task(w.call(raw, next(iter(w.jobs.values()))))
                    while not w.posts:
                        await asyncio.sleep(0)
                    task.cancel()
                    await task
            entry = guard.read(w.ledger)["entries"][raw.budget_key]
            self.assertEqual((entry["status"], entry["amount_usd"]), ("reserved", 3))
            self.assertFalse((raw.out / "liability_settlement_v3.json").exists())
        with w.patches():
            asyncio.run(run())

    def test_receipt_write_failure_keeps_full_batch_reservation(self):
        w = World(self.path, 1)
        original_open = Path.open
        def failing(path, mode="r", *args, **kwargs):
            if path.name == "requests.jsonl" and mode == "a":
                raise OSError("mock receipt disk failure")
            return original_open(path, mode, *args, **kwargs)
        async def run():
            raw, proxy = w.client(3)
            with self.assertRaises(ValueError):
                async with proxy:
                    await w.call(raw, next(iter(w.jobs.values())))
            self.assertEqual(guard.read(w.ledger)["entries"][raw.budget_key]["status"], "reserved")
        with w.patches(), patch.object(Path, "open", failing):
            asyncio.run(run())

    def test_session_close_failure_retains_batch(self):
        w = World(self.path, 1)
        w.close_fails = True
        async def run():
            raw, proxy = w.client(3)
            with self.assertRaises(OSError):
                async with proxy:
                    await w.call(raw, next(iter(w.jobs.values())))
            self.assertEqual(guard.read(w.ledger)["entries"][raw.budget_key]["status"], "reserved")
        with w.patches():
            asyncio.run(run())

    def test_corrupt_cached_disk_state_wakes_all_admission_waiters(self):
        w = World(self.path, 3)
        async def run():
            raw, proxy = w.client(3)
            # Exercise admission directly: one lease holds headroom, one waits,
            # then a third observes a malformed durable file.
            jobs = list(w.jobs.values())
            holder = asyncio.create_task(proxy.acquire(jobs[0]["request"], jobs[0]["tag"], 0))
            await holder
            waiter = asyncio.create_task(proxy.acquire(jobs[1]["request"], jobs[1]["tag"], 0))
            await asyncio.sleep(0)
            (raw.out / "requests.jsonl").write_text("torn")
            with self.assertRaises(ValueError):
                await proxy.acquire(jobs[2]["request"], jobs[2]["tag"], 0)
            with self.assertRaises(ValueError):
                await asyncio.wait_for(waiter, 1)
            self.assertTrue(proxy.broken)
            self.assertTrue(raw.stopped)
            self.assertFalse(w.posts)
        with w.patches():
            asyncio.run(run())

    def test_validator_allows_other_stage_reservations_but_rejects_prep_reservation(self):
        w = World(self.path, 1)
        ledger = guard.read(w.ledger)
        ledger["entries"]["gpu:fixture"] = {"kind": "gpu", "status": "reserved", "amount_usd": 15}
        guard.atomic(w.ledger, ledger)
        self.assertTrue(guard.validate_liability_ledger(w.ledger, w.policy)["complete"])
        with w.patches():
            w.client(3)
        with self.assertRaisesRegex(ValueError, "outstanding reservation"):
            guard.validate_liability_ledger(w.ledger, w.policy)

    def test_settlement_rechecks_campaign_cap_and_retains_reservation_on_failure(self):
        w = World(self.path, 1)
        async def run():
            raw, proxy = w.client(3)
            with self.assertRaisesRegex(ValueError, "cap exceeded"):
                async with proxy:
                    await w.call(raw, next(iter(w.jobs.values())))
                    ledger = guard.read(w.ledger)
                    ledger["entries"]["other"] = {"kind": "other", "status": "reserved", "amount_usd": 45}
                    guard.atomic(w.ledger, ledger)
            self.assertEqual(guard.read(w.ledger)["entries"][raw.budget_key]["status"], "reserved")
        with w.patches():
            asyncio.run(run())

    def test_five_historical_topups_are_idempotent_and_preserve_original_settled_entries(self):
        w = World(self.path, 5, "qwen/qwen3.5-27b")
        history = w.historical_timeouts()
        old = copy.deepcopy(guard.read(w.ledger)["entries"])
        raw_bytes = (w.run / "preparations_v1/requests.jsonl").read_bytes()
        for _ in range(2):
            allowance = guard.ensure_historical_topups(history, w.jobs, w.ledger, w.policy)
            self.assertAlmostEqual(allowance, 3.19488)
        ledger = guard.read(w.ledger)
        self.assertEqual(len(ledger["entries"]), 6)
        self.assertEqual({k: ledger["entries"][k] for k in old}, old)
        self.assertEqual((w.run / "preparations_v1/requests.jsonl").read_bytes(), raw_bytes)
        # Unrelated paid-stage reservations are permitted after preparation drain.
        ledger["entries"]["gpu:test"] = {"kind": "gpu", "status": "reserved", "amount_usd": 15}
        guard.atomic(w.ledger, ledger)
        self.assertEqual(guard.validate_liability_ledger(w.ledger, w.policy)["unknown_liability_entries"], 5)

    def test_historical_topup_cap_failure_is_atomic_and_does_not_mutate_ledger(self):
        w = World(self.path, 5, "qwen/qwen3.5-27b")
        history = w.historical_timeouts()
        ledger = guard.read(w.ledger)
        ledger["entries"]["other"] = {"kind": "other", "status": "reserved", "amount_usd": 42}
        guard.atomic(w.ledger, ledger)
        before = w.ledger.read_bytes()
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            guard.ensure_historical_topups(history, w.jobs, w.ledger, w.policy)
        self.assertEqual(w.ledger.read_bytes(), before)

    def test_historical_entry_tampering_or_missing_entry_blocks_downstream(self):
        w = World(self.path, 5, "qwen/qwen3.5-27b")
        history = w.historical_timeouts()
        guard.ensure_historical_topups(history, w.jobs, w.ledger, w.policy)
        original = guard.read(w.ledger)
        key = next(k for k, v in original["entries"].items() if v["kind"] == "unknown_provider_liability")
        for change in ("remove", "reduce"):
            altered = copy.deepcopy(original)
            if change == "remove":
                del altered["entries"][key]
            else:
                altered["entries"][key]["amount_usd"] = 0
            guard.atomic(w.ledger, altered)
            with self.assertRaisesRegex(ValueError, "liabilities are missing"):
                guard.validate_liability_ledger(w.ledger, w.policy)

    def test_cache_hit_uses_no_lease_post_or_new_charge(self):
        w = World(self.path, 1)
        async def run():
            raw, proxy = w.client(.01)
            job = next(iter(w.jobs.values()))
            payload = w.payload()
            prior = {"tag": job["tag"], "model": job["request"]["model"], "request": job["request"],
                "request_sha256": job["request_sha256"], "valid": True, "attempt": 0,
                "finish_reason": "stop", "http_status": 200, "response_body": json.dumps(payload),
                "parsed": {"answer": "ok"}, "usage": payload["usage"],
                "cost_usd": .001, "cost_basis": "provider_reported"}
            raw.reusable[(job["request_sha256"], json.dumps(job["tag"], sort_keys=True))] = ("fixture", prior)
            async with proxy:
                result = await w.call(raw, job)
            self.assertEqual(result["cost_usd"], 0)
            self.assertEqual((proxy.posted, proxy.completed_posts), (0, 0))
            self.assertFalse(w.posts)
            self.assertEqual(guard.read(w.ledger)["entries"][raw.budget_key]["amount_usd"], 0)
        with w.patches():
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
