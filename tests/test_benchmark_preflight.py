import json
from pathlib import Path
import tempfile
import unittest

from probity_benchmark.preflight import (
    Problem,
    load_problems,
    parse_answer,
    select_pilot,
)
from probity_benchmark.run_baselines import summarize


class BenchmarkPreflightTests(unittest.TestCase):
    def setUp(self):
        self.p = Problem(
            "id",
            "PRIVATE STORY SENTINEL",
            "Who opened the door?",
            ("Alice", "Bob"),
            "A",
            1,
        )

    def test_question_only_view_cannot_leak_story_or_gold(self):
        messages, gold = self.p.view(False, False)
        text = json.dumps(messages)
        self.assertNotIn(self.p.story, text)
        self.assertNotIn("gold", text)
        self.assertEqual(gold, "A")

    def test_both_answer_orders_preserve_ground_truth(self):
        messages, gold = self.p.view(True, True)
        self.assertIn("A: Bob\nB: Alice", messages[-1]["content"])
        self.assertEqual(gold, "B")
        self.assertIn(self.p.story, messages[-1]["content"])

    def test_ambiguous_answers_fail_instead_of_guessing(self):
        for text in ["A or B", "Answer A", "", "AB", "B because A is wrong"]:
            with self.assertRaises(ValueError):
                parse_answer(text)
        self.assertEqual(parse_answer(" B\n"), "B")

    def test_invalid_label_is_not_silently_mapped_to_b(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "data.jsonl"
            f.write_text(
                json.dumps(
                    dict(
                        synthetic_story="full",
                        question="q",
                        option_a="a",
                        option_b="b",
                        correct_answer="C",
                    )
                )
                + "\n"
            )
            rows, audit = load_problems(f)
            self.assertEqual(rows, [])
            self.assertEqual(len(audit["errors"]), 1)
            self.assertEqual(audit["nonempty_lines"], 1)
            self.assertEqual(audit["bytes_read"], audit["bytes_on_disk"])

    def test_duplicate_story_not_counted_as_independent(self):
        other = Problem("id2", self.p.story, "Different?", ("X", "Y"), "B", 2)
        with self.assertRaises(ValueError):
            select_pilot([self.p, other], 2, 42)

    def test_summary_requires_complete_order_pairs(self):
        rows = [
            dict(
                model="m",
                with_story=False,
                swapped=False,
                valid=True,
                problem_id="p",
                correct=True,
                finish_reason="stop",
            )
        ]
        cell = summarize(rows)["cells"][0]
        self.assertIsNone(cell["accuracy"])
        rows.append(dict(rows[0], swapped=True, correct=False))
        cell = summarize(rows)["cells"][0]
        self.assertEqual(cell["accuracy"], 0.5)
        self.assertEqual(cell["complete_story_pairs"], 1)

    def test_duplicate_order_cannot_masquerade_as_a_pair(self):
        row = dict(
            model="m",
            with_story=False,
            swapped=False,
            valid=True,
            problem_id="p",
            correct=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate answer order"):
            summarize([row, dict(row)])

    def test_unknown_request_cost_is_distinct_from_billed_cost(self):
        summary = summarize(
            [
                dict(
                    model="m",
                    with_story=False,
                    valid=False,
                    cost_usd=0.01,
                    cost_basis="reservation_upper_bound",
                )
            ]
        )
        self.assertEqual(summary["accounted_cost_upper_usd"], 0.01)
        self.assertEqual(summary["provider_reported_cost_usd"], 0)
        self.assertEqual(summary["unpriced_request_count"], 1)


if __name__ == "__main__":
    unittest.main()
