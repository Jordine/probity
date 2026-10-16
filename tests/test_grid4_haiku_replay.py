import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark import grid4_haiku_replay as replay
from probity_benchmark.grid4_evaluate import api_request_body


class HaikuReplayTests(unittest.TestCase):
    def client(self):
        return SimpleNamespace(models={replay.MODEL: {
            "pricing": {"prompt": "0.0000001", "completion": "0.0000005", "request": "0"},
            "supported_parameters": ["structured_outputs"]}},
            max_tokens=8192, reasoning={"enabled": False}, reasoning_by_model={},
            request_timeout_seconds=300., schema_retries=0, reusable={}, unavailable={},
            structured_output=True)

    def fixture(self):
        client = self.client()
        job = {"job_id": "frozen-job", "messages": [{"role": "user", "content": "Complete original transcript."}]}
        body = api_request_body(client, replay.MODEL, job["messages"])
        frozen = {"job_id": job["job_id"], "request": body, "request_sha256": replay.request_hash(body)}
        return client, job, frozen

    def test_preserves_complete_canonical_original_body_without_temperature(self):
        client, job, original = self.fixture()
        replay.validate_client_requests(client, [job], [original])
        self.assertNotIn("temperature", original["request"])
        rebuilt = api_request_body(client, replay.MODEL, job["messages"])
        self.assertEqual(json.dumps(rebuilt, sort_keys=True), json.dumps(original["request"], sort_keys=True))
        self.assertEqual(rebuilt["provider"]["max_price"]["prompt"], .09999999999999999)

    def test_any_body_runtime_or_cache_drift_is_rejected_before_dispatch(self):
        mutations = [
            lambda c: c.models[replay.MODEL]["supported_parameters"].append("temperature"),
            lambda c: c.models[replay.MODEL]["pricing"].update(prompt="0.0000002"),
            lambda c: setattr(c, "max_tokens", 4096),
            lambda c: setattr(c, "reasoning", {"enabled": True}),
            lambda c: setattr(c, "request_timeout_seconds", 30),
            lambda c: setattr(c, "schema_retries", 1),
            lambda c: c.reusable.update(prior="receipt"),
            lambda c: setattr(c, "structured_output", False),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                client, job, original = self.fixture()
                mutate(client)
                with self.assertRaises(ValueError):
                    replay.validate_client_requests(client, [job], [original])

    def test_prompt_change_rejected_even_if_expected_request_hash_field_unchanged(self):
        client, job, original = self.fixture()
        changed = copy.deepcopy(job)
        changed["messages"][0]["content"] += " shortened/reworded"
        with self.assertRaisesRegex(ValueError, "differs from frozen original"):
            replay.validate_client_requests(client, [changed], [original])

    def test_binding_keeps_wrong_original_answer_and_full_raw_provider_receipt(self):
        _, job, original = self.fixture()
        receipt = {"valid": True, "finish_reason": "stop", "request": original["request"],
                   "request_sha256": original["request_sha256"], "tag": {"job": job["job_id"]},
                   "parsed": {"answer": "B"}, "usage": {"completion_tokens": 5},
                   "response_body": json.dumps({"provider": "original-provider", "choices": [
                       {"finish_reason": "stop", "message": {"content": '{"answer":"B"}'}}]})}
        outcome = {"request_sha256": original["request_sha256"], "correct": False, "answer": "B"}
        bound = replay.bind_original_requests([job], {job["job_id"]: outcome}, [receipt])
        self.assertEqual(bound[0]["original_receipt"]["response_body"], receipt["response_body"])
        self.assertFalse(bound[0]["original_outcome"]["correct"])

    def test_selection_requires_all_cells_and_balanced_24_families(self):
        jobs, pairs = [], []
        for family in range(24):
            for model in ("qwen/qwen3.5-27b", "qwen/qwen3.8-27b"):
                pair = {"family": str(family), "participant_model": model, "jobs": {}}
                for cell, kind in (("A", "none"), ("C", "adapted_primary:real")):
                    identity = model + str(family) + cell
                    jobs.append({"job_id": identity, "kind": kind})
                    pair["jobs"][cell] = identity
                pairs.append(pair)
        replay.validate_selection(jobs, pairs)
        with self.assertRaisesRegex(ValueError, "all 96"):
            replay.validate_selection(jobs[:-1], pairs)
        changed = copy.deepcopy(pairs)
        changed[0]["family"] = changed[2]["family"]
        with self.assertRaisesRegex(ValueError, "one trajectory"):
            replay.validate_selection(jobs, changed)

    def test_started_run_cannot_reissue_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "started.json").write_text("{}")
            args = SimpleNamespace(out=out, concurrency=4)
            with patch.object(replay, "validate_plan", return_value=({}, [], [], [])), \
                    patch.dict(os.environ, {"PROBITY_PILOT_LEDGER": "/fake/ledger"}), \
                    patch.object(replay, "BudgetedChat") as client:
                with self.assertRaisesRegex(ValueError, "never automatically repeat"):
                    asyncio.run(replay.run(args))
                client.assert_not_called()

    def test_fresh_receipt_guard_rejects_retries_cache_body_drift_and_old_response_id(self):
        _, job, original = self.fixture()
        original["original_receipt"] = {"response_body": json.dumps({"id": "prior-generation"})}
        receipt = {"tag": {"job": job["job_id"]}, "valid": True, "attempt": 0,
                   "request": original["request"], "request_sha256": original["request_sha256"],
                   "response_body": json.dumps({"id": "fresh-generation"})}
        _, evidence = replay.validate_fresh_receipts([job], [original], [receipt])
        self.assertEqual(evidence["new_provider_response_ids_observed"], 1)
        mutations = [lambda r: r.update(attempt=1), lambda r: r.update(reused_from="cache"),
                     lambda r: r.update(request_sha256="different"),
                     lambda r: r["request"].update(temperature=0),
                     lambda r: r.update(response_body=json.dumps({"id": "prior-generation"}))]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(receipt)
                mutate(changed)
                with self.assertRaises(ValueError):
                    replay.validate_fresh_receipts([job], [original], [changed])


if __name__ == "__main__":
    unittest.main()
