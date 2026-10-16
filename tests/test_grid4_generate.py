import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest

from probity_benchmark.grid4_generate import (
    FEEDBACK_INSTRUCTION, SIGNAL_EXPLANATION, FrozenActionClient,
    atomic_json, base_messages, canonical_hash, disclosure_audit,
    feedback_messages, feedback_packet, initial_episode, probe_request,
    validate_saved_response, validate_score,
)
from probity_benchmark.preflight import Problem, sha256


class GridFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("problem", "Full source with unique evidence.", "Who?", ("Ada", "Bo"), "A", 1)
        prep = {"evidence_items": [{"claim": "Ada arrived", "quotation": "Ada arrived"}],
                "anticipated_challenge": "Why?", "response_to_challenge": "The record.",
                "request_sha256": "prep"}
        self.original = {
            "episode_id": "original", "protocol": "explicit_reasoned_evidence_sequential",
            "problem_id": "problem", "parent_problem_id": "family", "model": "model",
            "split": "test", "world": "original", "swapped": False, "gold": "A",
            "knowledge": "A", "knowledge_correct": True, "complete": True,
            "private_preparations": {"Red": prep, "Blue": copy.deepcopy(prep)},
            "turns": [{"speaker": "Red", "round": 1, "assigned_answer": "A", "answer": "A", "argument": "Ada arrived."},
                      {"speaker": "Blue", "round": 1, "assigned_answer": "B", "answer": "B", "argument": "Bo arrived."}],
        }
        self.detector = {"head_sha256": "head", "reference_sha256": "reference",
                         "references": {"model": {"scores": [0., 1., 2., 3.]}}}
        self.episode = initial_episode(self.original, self.detector)

    def test_first_action_preserves_base_prompt_and_no_scores(self):
        original_messages = base_messages(self.problem, self.original, 0)
        messages = feedback_messages(self.problem, self.episode)
        self.assertEqual(messages[0]["content"], original_messages[0]["content"] + FEEDBACK_INSTRUCTION)
        self.assertTrue(messages[1]["content"].startswith(original_messages[1]["content"]))
        self.assertIn('"units": []', messages[1]["content"])
        self.assertEqual(self.episode["private_preparations"], self.original["private_preparations"])
        self.assertEqual(self.episode["matched_no_feedback_episode_id"], "original")
        self.assertNotEqual(self.episode["episode_id"], "original")

    def test_feedback_must_be_past_and_already_scored(self):
        self.episode["turns"].append(copy.deepcopy(self.original["turns"][0]))
        with self.assertRaisesRegex(ValueError, "unscored"):
            feedback_messages(self.problem, self.episode)
        self.episode["turns"][0]["probe_feedback"] = {"percentile": 37.5}
        messages = feedback_messages(self.problem, self.episode)
        self.assertIn('"turn": 1', messages[1]["content"])
        self.assertNotIn('"turn": 2', messages[1]["content"])
        self.assertIn(SIGNAL_EXPLANATION, messages[1]["content"])
        self.assertNotIn(self.original["turns"][1]["argument"], messages[1]["content"])

    def test_packet_matches_existing_whole_response_display(self):
        turns = copy.deepcopy(self.original["turns"])
        turns[0]["probe_feedback"] = {"percentile": 10.0}
        turns[1]["probe_feedback"] = {"percentile": 90.0}
        packet = feedback_packet(turns)
        self.assertEqual(packet["mean_signal_by_speaker"], {"Red": 10.0, "Blue": 90.0})
        self.assertEqual(packet["detector_only_recommendation"], "A")

    def test_disclosure_audit_does_not_edit_public_text(self):
        text = "My opponent's probe percentile is 93."
        audit = disclosure_audit(text)
        self.assertTrue(audit["needs_review"])
        self.assertEqual(audit["argument_sha256"], sha256(text))
        self.assertFalse(disclosure_audit("The source names Ada.")["needs_review"])

    def record(self):
        body = {"model": "model", "messages": [{"role": "user", "content": "complete context"}],
                "max_tokens": 8192, "reasoning": {"enabled": False},
                "provider": {"max_price": {"prompt": 0.1, "completion": 1., "request": 0.}},
                "response_format": {"json_schema": {"schema": {
                    "type": "object", "properties": {"argument": {"type": "string"}, "answer": {"enum": ["A", "B"]}},
                    "required": ["argument", "answer"], "additionalProperties": False}}}}
        payload = {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": '{"argument":"Ada arrived.","answer":"A"}', "reasoning": None}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 12, "cost": 0.0001}}
        return {"model": "model", "request": body, "request_sha256": sha256(json.dumps(body, sort_keys=True)),
                "http_status": 200, "response_body": json.dumps(payload), "valid": None,
                "reserved_usd": .1, "input_token_upper_bound": 1000, "assigned_answer": "A"}

    def test_raw_receipt_recovery_retains_every_byte(self):
        record = self.record()
        raw = record["response_body"]
        validate_saved_response(record)
        self.assertTrue(record["valid"])
        self.assertEqual(record["response_body"], raw)
        self.assertEqual(record["parsed"]["answer"], "A")
        self.assertEqual(record["response_body_sha256"], sha256(raw))

    def test_length_finish_is_rejected(self):
        record = self.record()
        payload = json.loads(record["response_body"])
        payload["choices"][0]["finish_reason"] = "length"
        record["response_body"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_saved_response(record)

    def test_role_concession_is_retained_as_outcome(self):
        record = self.record()
        payload = json.loads(record["response_body"])
        payload["choices"][0]["message"]["content"] = '{"argument":"Bo arrived.","answer":"B"}'
        record["response_body"] = json.dumps(payload)
        validate_saved_response(record)
        self.assertTrue(record["valid"])
        self.assertEqual(record["parsed"]["answer"], "B")
        self.assertEqual(record["assigned_answer"], "A")
        turns = copy.deepcopy(self.original["turns"])
        turns[0].update(answer="B", probe_feedback={"percentile": 10.})
        turns[1].update(probe_feedback={"percentile": 90.})
        self.assertEqual(feedback_packet(turns)["detector_only_recommendation"], "A")

    def test_score_binds_full_context_argument_and_frozen_reference(self):
        receipt = self.record()
        validate_saved_response(receipt)
        self.episode["turns"] = [{**self.original["turns"][0], "request_sha256": receipt["request_sha256"]}]
        request = probe_request("/tmp/grid4", {"detector": self.detector}, self.episode, 0, receipt)
        result = {k: request[k] for k in (
            "job_id", "episode_id", "matched_no_feedback_episode_id", "turn_index", "model",
            "argument_sha256", "source_request_sha256", "response_body_sha256", "head_sha256", "reference_sha256")}
        result.update(schema="grid4_probe_result_v1", raw_score=1., percentile=37.5,
                      score_request_sha256=canonical_hash(request))
        self.assertEqual(validate_score(request, result, self.detector), result)
        changed = copy.deepcopy(request)
        changed["messages"][0]["content"] += " feedback removed"
        with self.assertRaisesRegex(ValueError, "exact full"):
            validate_score(changed, result, self.detector)
        result["percentile"] = 50.
        with self.assertRaisesRegex(ValueError, "frozen"):
            validate_score(request, result, self.detector)

    def test_unresolved_intent_cannot_trigger_second_paid_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "intents").mkdir()
            (out / "requests").mkdir()
            receipt = self.record()
            template = {k: v for k, v in receipt["request"].items() if k != "messages"}
            manifest = {"request_templates": {"model": template}}
            atomic_json(out / "intents" / (receipt["request_sha256"] + ".json"), {"reserved_usd": .1})
            client = FrozenActionClient(out, manifest, 1.)
            # No session exists: the guard must run before any network access.
            with self.assertRaisesRegex(RuntimeError, "unresolved paid-call intent"):
                asyncio.run(client.complete(self.episode, receipt["request"]["messages"]))

    def test_complete_receipt_reuse_needs_no_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "requests").mkdir()
            (out / "intents").mkdir()
            receipt = self.record()
            template = {k: v for k, v in receipt["request"].items() if k != "messages"}
            atomic_json(out / "requests" / (receipt["request_sha256"] + ".json"), receipt)
            client = FrozenActionClient(out, {"request_templates": {"model": template}}, 1.)
            result = asyncio.run(client.complete(self.episode, receipt["request"]["messages"]))
            self.assertTrue(result["valid"])
            self.assertEqual(result["response_body"], receipt["response_body"])

    def test_immutable_queue_does_not_overwrite_different_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            atomic_json(path, {"context": "full"}, immutable=True)
            atomic_json(path, {"context": "full"}, immutable=True)
            with self.assertRaisesRegex(ValueError, "immutable"):
                atomic_json(path, {"context": "truncated"}, immutable=True)


if __name__ == "__main__":
    unittest.main()
