import copy
import hashlib
import json
import unittest

from probity_benchmark.grid6_protocol import (
    MODEL, POLICIES, append_turn, continuation_tag, digest, make_branch,
    next_messages, next_request, state_sha256, validate_branch, visible_packet,
)
from probity_benchmark.preflight import Problem


class Grid6CausalityTests(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("problem", "Full source: Ada arrived.", "Who arrived?",
                               ("Ada", "Bo"), "A", 1)
        prep = {"evidence_items": [], "anticipated_challenge": "A challenge.",
                "response_to_challenge": "A response.", "request_sha256": "prep"}
        self.original = {
            "episode_id": "original", "problem_id": "problem",
            "parent_problem_id": "parent", "model": MODEL, "complete": True,
            "protocol": "explicit_reasoned_evidence_sequential",
            "gold": "A", "swapped": False,
            "private_preparations": {"Red": prep, "Blue": copy.deepcopy(prep)},
            "turns": [],
        }
        for i in range(6):
            text = f"Argument {i}."
            self.original["turns"].append({
                "speaker": "Red" if i % 2 == 0 else "Blue", "round": i // 2 + 1,
                "assigned_answer": "A" if i % 2 == 0 else "B",
                "answer": "A" if i % 2 == 0 else "B", "argument": text,
                "request_sha256": f"request{i}",
                "probe_feedback": {"turn_index": i, "source_request_sha256": f"request{i}",
                    "argument_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "detector_identity": "detector", "percentile": 10. if i % 2 == 0 else 90.},
            })

    def branch(self, policy):
        return make_branch(self.original, policy, "detector", "family", self.problem)

    def test_all_policies_share_exact_prefix_without_future_arguments(self):
        branches = [self.branch(p) for p in POLICIES]
        self.assertEqual(len({b["shared_prefix_sha256"] for b in branches}), 1)
        self.assertEqual(len({b["branch_id"] for b in branches}), len(POLICIES))
        for branch in branches:
            self.assertEqual(len(branch["turns"]), 2)
            messages = next_messages(self.problem, branch)
            self.assertIn("Argument 0.", messages[1]["content"])
            self.assertIn("Argument 1.", messages[1]["content"])
            for i in range(2, 6):
                self.assertNotIn(f"Argument {i}.", str(messages))
        self.assertEqual(len(self.original["turns"]), 6)

    def test_components_are_actually_removed(self):
        turns = self.branch("bundle")["turns"]
        self.assertEqual(set(visible_packet(turns, "turn_scores", "seed")),
                         {"granularity", "units"})
        self.assertEqual(set(visible_packet(turns, "speaker_means", "seed")),
                         {"mean_signal_by_speaker"})
        self.assertEqual(visible_packet(turns, "recommendation", "seed"),
                         {"detector_only_recommendation": "A"})
        self.assertIsNone(visible_packet(turns, "none", "seed"))

    def test_neutral_contains_no_real_values(self):
        turns = self.branch("neutral")["turns"]
        for t in turns:
            del t["probe_feedback"]
        p = visible_packet(turns, "neutral", "seed")
        self.assertTrue(all(u["signal_percentile"] is None for u in p["units"]))
        self.assertTrue(all(v is None for v in p["mean_signal_by_speaker"].values()))
        self.assertIsNone(p["detector_only_recommendation"])

    def test_missing_or_misbound_scores_rejected(self):
        for field, value in (("turn_index", 5), ("source_request_sha256", "future"),
                             ("argument_sha256", "wrong"), ("percentile", float("nan"))):
            branch = self.branch("bundle")
            # Use suffix-like list for isolated binding test, without prefix mutation check.
            branch["turns"][0]["probe_feedback"][field] = value
            with self.assertRaises(ValueError):
                visible_packet(branch["turns"], "bundle", "seed")
        branch = self.branch("bundle")
        del branch["turns"][0]["probe_feedback"]
        with self.assertRaisesRegex(ValueError, "unscored"):
            visible_packet(branch["turns"], "bundle", "seed")

    def test_prefix_mutation_rejected(self):
        branch = self.branch("bundle")
        branch["turns"][0]["argument"] += " Changed"
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            next_messages(self.problem, branch)

    def test_wrong_detector_rejected(self):
        branch = self.branch("bundle")
        branch["detector_identity"] = "different"
        with self.assertRaisesRegex(ValueError, "immutable branch contract"):
            next_messages(self.problem, branch)

    def test_shuffle_is_deterministic_and_uses_only_visible_scores(self):
        turns = self.branch("bundle")["turns"]
        before = copy.deepcopy(turns)
        a = visible_packet(turns, "shuffled_bundle", "fixed")
        b = visible_packet(turns, "shuffled_bundle", "fixed")
        self.assertEqual(a, b)
        self.assertEqual(sorted(u["signal_percentile"] for u in a["units"]), [10., 90.])
        self.assertEqual(turns, before)

    def test_no_feedback_has_no_feedback_instruction(self):
        messages = next_messages(self.problem, self.branch("none"))
        self.assertNotIn("PRIVATE PAST-TURN", str(messages))
        self.assertNotIn("Private experimental feedback", str(messages))

    def receipt(self, branch, *, answer="B", argument="A fresh argument."):
        body = next_request(self.problem, branch)
        payload = {"id": f"test-event-{branch['branch_id']}-{len(branch['turns'])}",
                   "created": 1792360000 + len(branch["turns"]), "provider": "test-provider",
                   "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                        "content": json.dumps({"argument": argument, "answer": answer})}}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        return {"model": MODEL, "request": body,
                "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
                "tag": continuation_tag(branch), "valid": True, "http_status": 200,
                "finish_reason": "stop", "response_body": json.dumps(payload),
                "parsed": {"argument": argument, "answer": answer}, "usage": payload["usage"]}

    def score(self, branch, receipt):
        return {"turn_index": len(branch["turns"]),
                "source_request_sha256": receipt["request_sha256"],
                "argument_sha256": hashlib.sha256(receipt["parsed"]["argument"].encode()).hexdigest(),
                "detector_identity": branch["detector_identity"],
                "messages_sha256": digest(receipt["request"]["messages"]),
                "response_body_sha256": hashlib.sha256(receipt["response_body"].encode()).hexdigest(),
                "branch_id": branch["branch_id"], "percentile": 50.}

    def test_old_suffix_append_rejected_for_every_policy(self):
        for policy in POLICIES:
            branch = self.branch(policy)
            branch["turns"].append(copy.deepcopy(self.original["turns"][2]))
            with self.assertRaisesRegex(ValueError, "authenticated receipt"):
                next_messages(self.problem, branch)

    def test_wrong_problem_or_source_rejected(self):
        other = Problem("unrelated", "Other story.", "Other question?", ("C", "D"), "A", 1)
        with self.assertRaisesRegex(ValueError, "source problem changed"):
            next_messages(other, self.branch("bundle"))
        changed = Problem(self.problem.problem_id, "Changed story with same ID.", self.problem.question,
                          self.problem.options, self.problem.gold, self.problem.source_line)
        with self.assertRaisesRegex(ValueError, "source problem changed"):
            next_messages(changed, self.branch("bundle"))

    def test_static_contract_mutations_rejected(self):
        mutations = {
            "policy": lambda b: b.update(policy="none"),
            "preparation": lambda b: b["private_preparations"]["Red"].update(anticipated_challenge="Changed"),
            "detector": lambda b: b.update(detector_identity="changed"),
            "seed": lambda b: b.update(shuffle_seed="changed"),
            "schedule": lambda b: b["turn_schedule"][2].update(assigned_answer="B"),
            "provider": lambda b: b["generation_parameters"].update(provider={"only": ["other"]}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                branch = self.branch("bundle"); mutate(branch)
                with self.assertRaisesRegex(ValueError, "immutable branch contract"):
                    next_messages(self.problem, branch)

    def test_old_live_feedback_prefix_rejected(self):
        original = copy.deepcopy(self.original)
        original["protocol"] = "grid4_adapted_feedback_v1"
        with self.assertRaisesRegex(ValueError, "no-feedback source"):
            make_branch(original, "bundle", "detector", "family", self.problem)

    def test_full_six_turn_chains_keep_answer_deviations(self):
        for policy in POLICIES:
            branch = self.branch(policy)
            before = copy.deepcopy(branch)
            for i in range(2, 6):
                receipt = self.receipt(branch, answer="B", argument=f"New argument {i}.")
                score = self.score(branch, receipt)
                branch = append_turn(self.problem, branch, receipt, score)
                self.assertEqual(branch["turns"][-1]["answer"], "B")
                self.assertEqual(branch["turns"][-1]["assigned_answer"], "A" if i % 2 == 0 else "B")
                self.assertEqual(len(branch["turns"]), i + 1)
                validate_branch(self.problem, branch)
            self.assertTrue(branch["complete"])
            self.assertEqual(branch["turns"][:2], before["turns"])
            with self.assertRaisesRegex(ValueError, "no next"):
                next_messages(self.problem, branch)

    def test_wrong_branch_and_state_receipts_rejected(self):
        branch = self.branch("bundle"); receipt = self.receipt(branch)
        for field in ("branch_id", "prior_state_sha256", "turn_index"):
            changed = copy.deepcopy(receipt); changed["tag"][field] = "unrelated"
            with self.assertRaisesRegex(ValueError, "another branch or prior state"):
                append_turn(self.problem, branch, changed, self.score(branch, receipt))

    def test_receipt_settings_stop_and_parsed_tampering_rejected(self):
        branch = self.branch("none"); receipt = self.receipt(branch)
        for field, value in (("temperature", 1), ("reasoning", {"enabled": True}),
                             ("provider", {"only": ["other"]})):
            changed = copy.deepcopy(receipt); changed["request"][field] = value
            with self.assertRaisesRegex(ValueError, "settings differ"):
                append_turn(self.problem, branch, changed)
        changed = copy.deepcopy(receipt); changed["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "incomplete generation"):
            append_turn(self.problem, branch, changed)
        changed = copy.deepcopy(receipt); changed["parsed"]["argument"] = "Wrong output"
        with self.assertRaisesRegex(ValueError, "parsed response"):
            append_turn(self.problem, branch, changed)

    def test_score_same_argument_wrong_context_rejected(self):
        branch = self.branch("bundle"); receipt = self.receipt(branch)
        for field in ("messages_sha256", "response_body_sha256", "detector_identity", "branch_id"):
            score = self.score(branch, receipt); score[field] = "different"
            with self.assertRaisesRegex(ValueError, "another context"):
                append_turn(self.problem, branch, receipt, score)
        with self.assertRaisesRegex(ValueError, "requires a score"):
            append_turn(self.problem, branch, receipt)

    def test_authenticated_turn_mutation_rejected(self):
        branch = self.branch("none")
        branch = append_turn(self.problem, branch, self.receipt(branch))
        branch["turns"][-1]["argument"] += " tampered"
        with self.assertRaisesRegex(ValueError, "differs from authenticated"):
            next_messages(self.problem, branch)

    def test_known_historical_envelope_rejected_but_identical_text_allowed(self):
        branch = self.branch("none")
        receipt = self.receipt(branch, argument="Argument 2.")
        # Same output text is a valid observed outcome.
        accepted = append_turn(self.problem, branch, receipt)
        self.assertEqual(accepted["turns"][2]["argument"], self.original["turns"][2]["argument"])
        historical_hash = hashlib.sha256(receipt["response_body"].encode()).hexdigest()
        protected = make_branch(self.original, "none", "detector", "family", self.problem,
                                historical_response_body_sha256=[historical_hash])
        forged = copy.deepcopy(receipt); forged["tag"] = continuation_tag(protected)
        with self.assertRaisesRegex(ValueError, "historical response"):
            append_turn(self.problem, protected, forged)

    def test_same_historical_request_hash_allows_a_fresh_response_event(self):
        first = self.branch("none")
        future_request = self.receipt(first)["request_sha256"]
        original = copy.deepcopy(self.original)
        original["turns"][2]["request_sha256"] = future_request
        branch = make_branch(original, "none", "detector", "family", self.problem)
        receipt = self.receipt(branch)
        self.assertEqual(receipt["request_sha256"], future_request)
        accepted = append_turn(self.problem, branch, receipt)
        self.assertEqual(len(accepted["turns"]), 3)


if __name__ == "__main__":
    unittest.main()
