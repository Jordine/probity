import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark import grid4_recover as recovery
from probity_benchmark.grid4_generate import atomic_json, validate_saved_response
from probity_benchmark.preflight import sha256


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def text(self):
        return json.dumps(self.payload)


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, *, json):
        self.calls.append(copy.deepcopy(json))
        return FakeResponse(self.payload)

    async def close(self):
        pass


class GridRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        schema = {"type": "object", "properties": {
            "argument": {"type": "string"}, "answer": {"enum": ["A", "B"]}},
            "required": ["argument", "answer"], "additionalProperties": False}
        body = {"model": "model", "messages": [{"role": "user", "content": "Full original context"}],
                "max_tokens": 8192, "reasoning": {"enabled": False},
                "provider": {"max_price": {"prompt": .1, "completion": 1., "request": 0.}},
                "response_format": {"json_schema": {"schema": schema}}}
        self.request_hash = sha256(json.dumps(body, sort_keys=True))
        payload = {"choices": [{"finish_reason": "length", "message": {
            "content": " " * 8192}}], "usage": {
            "prompt_tokens": 100, "completion_tokens": 8192, "cost": .01}}
        self.failed = {"model": "model", "request": body, "request_sha256": self.request_hash,
                       "http_status": 200, "response_body": json.dumps(payload), "valid": False,
                       "finish_reason": "length", "cost_usd": .01, "reserved_usd": .1,
                       "input_token_upper_bound": 10000, "assigned_answer": "A",
                       "tag": {"episode": "episode", "turn_index": 0, "stage": "debate",
                               "speaker": "Red", "round": 1}, "budget_key": "original"}
        self.intent = {k: self.failed[k] for k in ("request_sha256", "reserved_usd", "budget_key", "tag")}
        self.manifest = {"source_code_sha256": "frozen-generator",
                         "request_templates": {"model": {k: v for k, v in body.items() if k != "messages"}},
                         "model_specs": {"model": {"context_length": 100000, "top_provider": {}}}}
        self.p = recovery.paths(self.out, self.request_hash)
        atomic_json(self.out / "manifest.json", self.manifest)
        atomic_json(self.p["receipt"], self.failed)
        atomic_json(self.p["intent"], self.intent)
        self.plan = recovery.prepare(self.out, self.request_hash)
        self.accepted = copy.deepcopy(self.failed)
        self.success_payload = {"choices": [{"finish_reason": "stop", "message": {
            "content": '{"argument":"I concede the assigned position.","answer":"B"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": .002}}
        self.accepted.update(response_body=json.dumps(self.success_payload), budget_key="retry")
        validate_saved_response(self.accepted)
        self.accepted_intent = {k: self.accepted[k] for k in (
            "request_sha256", "reserved_usd", "budget_key", "tag")}

    def retain_attempt(self, receipt=None):
        receipt = receipt or self.accepted
        atomic_json(self.p["attempt"] / "requests" / (self.request_hash + ".json"), receipt)
        atomic_json(self.p["attempt"] / "intents" / (self.request_hash + ".json"), self.accepted_intent)
        recovery.retain_retry_accounting(self.p, self.request_hash)

    def test_failed_evidence_is_exact_and_only_loop_failures_qualify(self):
        self.assertEqual(self.p["receipt"].read_bytes(), self.p["original_receipt"].read_bytes())
        self.assertEqual(self.p["intent"].read_bytes(), self.p["original_intent"].read_bytes())
        self.assertEqual(recovery.prepare(self.out, self.request_hash), self.plan)
        self.assertEqual(self.plan["allowed_new_attempts"], 1)
        with self.assertRaisesRegex(ValueError, "whitespace-loop"):
            recovery.loop_diagnostic(self.accepted)
        nonloop = copy.deepcopy(self.failed)
        payload = json.loads(nonloop["response_body"])
        payload["choices"][0]["message"]["content"] = "ordinary unfinished answer"
        nonloop["response_body"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "whitespace-loop"):
            recovery.loop_diagnostic(nonloop)

    def test_promotion_retains_concession_and_both_costs_idempotently(self):
        self.retain_attempt()
        promotion = recovery.promote_receipt(self.out, self.request_hash, self.accepted, self.accepted_intent)
        self.assertEqual(recovery.promote_receipt(
            self.out, self.request_hash, self.accepted, self.accepted_intent), promotion)
        self.assertAlmostEqual(recovery.committed(self.out), .012)
        self.assertEqual(self.p["archived_receipt"].read_bytes(), self.p["original_receipt"].read_bytes())
        self.assertEqual(json.loads(self.p["receipt"].read_text())["parsed"]["answer"], "B")
        self.assertFalse(self.p["retry_intent"].exists())
        self.assertTrue(self.p["completed"].exists())

    def test_every_interrupted_promotion_preserves_liability_and_resumes(self):
        # Simulate process death after each durable write in the transaction.
        for break_after in range(1, 9):
            with self.subTest(break_after=break_after):
                temporary = tempfile.TemporaryDirectory()
                try:
                    out = Path(temporary.name)
                    import shutil
                    shutil.copytree(self.out, out, dirs_exist_ok=True)
                    p = recovery.paths(out, self.request_hash)
                    for kind, value in (("receipt", self.accepted), ("intent", self.accepted_intent)):
                        atomic_json(p["retry_" + kind], value)
                    calls = 0

                    def interrupted(path, value, **kwargs):
                        nonlocal calls
                        atomic_json(path, value, **kwargs)
                        calls += 1
                        self.assertGreaterEqual(recovery.committed(out) + 1e-12, .012)
                        if calls == break_after:
                            raise OSError("simulated interruption")

                    with patch.object(recovery, "atomic_json", interrupted):
                        with self.assertRaisesRegex(OSError, "simulated interruption"):
                            recovery.promote_receipt(out, self.request_hash, self.accepted, self.accepted_intent)
                    recovery.promote_receipt(out, self.request_hash, self.accepted, self.accepted_intent)
                    self.assertAlmostEqual(recovery.committed(out), .012)
                    self.assertEqual(p["archived_receipt"].read_bytes(), p["original_receipt"].read_bytes())
                finally:
                    temporary.cleanup()

    def test_changed_request_or_mismatched_attempt_accounting_rejected(self):
        changed = copy.deepcopy(self.accepted)
        changed["request"]["max_tokens"] = 16384
        with self.assertRaises(ValueError):
            recovery.promote_receipt(self.out, self.request_hash, changed, self.accepted_intent)
        mismatch = {**self.accepted_intent, "budget_key": "unrelated"}
        with self.assertRaisesRegex(ValueError, "changed original request"):
            recovery.promote_receipt(self.out, self.request_hash, self.accepted, mismatch)
        self.assertFalse(self.p["promotion"].exists())

    def run_with_session(self, payload, expect_failure=False):
        session = FakeSession(payload)
        ledger = self.out / "budget.json"
        atomic_json(ledger, {"cap_usd": 1., "entries": {"original": {
            "status": "settled", "amount_usd": .01, "original_reservation_usd": .1}}})
        episode = {"episode_id": "episode", "problem_id": "problem", "model": "model",
                   "turns": [], "turn_schedule": [{"round": 1, "speaker": "Red", "assigned_answer": "A"}]}
        args = SimpleNamespace(out=self.out, failed_request=self.request_hash, budget=.2,
                               timeout=10, ledger=ledger)

        async def enter(client):
            client.session = session
            return client

        with patch.object(recovery, "load_run", return_value=(self.manifest, {"problem": None}, [episode])), \
                patch.object(recovery, "feedback_messages", return_value=self.failed["request"]["messages"]), \
                patch.object(recovery, "publish_turn", return_value="score-job"), \
                patch.object(recovery, "status", return_value={}), \
                patch.object(recovery.FrozenActionClient, "__aenter__", enter):
            for _ in range(2):
                if expect_failure:
                    with self.assertRaises(RuntimeError):
                        asyncio.run(recovery.run(args))
                else:
                    result = asyncio.run(recovery.run(args))
                    self.assertEqual(result["technical_retry_count"], 1)
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(session.calls[0], self.failed["request"])
        entries = json.loads(ledger.read_text())["entries"]
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e["status"] == "settled" for e in entries.values()))
        return entries

    def test_successful_retry_makes_exactly_one_call_and_accounts_both_attempts(self):
        entries = self.run_with_session(self.success_payload)
        self.assertAlmostEqual(sum(e["amount_usd"] for e in entries.values()), .012)
        self.assertAlmostEqual(recovery.committed(self.out), .012)

    def test_second_failure_is_retained_and_never_retried(self):
        failed_payload = json.loads(self.failed["response_body"])
        failed_payload["usage"]["cost"] = .002
        entries = self.run_with_session(failed_payload, expect_failure=True)
        self.assertAlmostEqual(sum(e["amount_usd"] for e in entries.values()), .012)
        self.assertAlmostEqual(recovery.committed(self.out), .012)
        self.assertFalse(self.p["completed"].exists())
        self.assertEqual(self.p["receipt"].read_bytes(), self.p["original_receipt"].read_bytes())


if __name__ == "__main__":
    unittest.main()
