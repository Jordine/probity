import copy
import unittest

from probity_benchmark.judge3_export_tables import project_analysis


class JudgeTableExports(unittest.TestCase):
    def fixture(self):
        common = {
            "protocol": "frozen",
            "split": "test",
            "participant": "all",
            "episodes": 4,
            "parent_groups": 2,
        }
        comparison = {
            **common,
            "condition": "local3_sentence:real",
            "reference": "local3_sentence:neutral",
            "mean_difference": 0.5,
            "parent_bootstrap_95_ci": [0.0, 1.0],
            "improved": 2,
            "harmed": 0,
        }
        report = {
            "expected": 12,
            "completed": 12,
            "finish_reasons": {"stop": 12},
            "cells": [
                {
                    **common,
                    "condition": "none",
                    "accuracy": 0.25,
                    "correct": 1,
                    "parent_bootstrap_95_ci": [0, 0.5],
                },
                {
                    **common,
                    "condition": "local3_sentence:neutral",
                    "accuracy": 0.5,
                    "correct": 2,
                    "parent_bootstrap_95_ci": [0, 1],
                },
                {
                    **common,
                    "condition": "local3_sentence:real",
                    "accuracy": 1.0,
                    "correct": 4,
                    "parent_bootstrap_95_ci": [1, 1],
                },
            ],
            "comparisons": [comparison],
        }
        repeat = {
            "repeated_prompt_groups": 1,
            "logical_jobs_in_repeated_groups": 2,
            "groups_with_answer_disagreement": 1,
            "matching_answer_pairs": 0,
            "all_answer_pairs": 1,
            "pairwise_agreement": 0.0,
            "scope": "descriptive only",
            "groups": [
                {
                    "messages_sha256": "same-message",
                    "answer_counts": {"A": 1, "B": 1},
                    "answer_disagreement": True,
                    "complete_request_hashes": ["same-request"],
                    "jobs": [
                        {
                            "job_id": "one",
                            "kind": "local3_sentence:neutral",
                            "answer": "A",
                            "request_sha256": "same-request",
                            "provider_as_reported": "provider-1",
                        },
                        {
                            "job_id": "two",
                            "kind": "local3_sentence:neutral",
                            "answer": "B",
                            "request_sha256": "same-request",
                            "provider_as_reported": "provider-2",
                        },
                    ],
                }
            ],
        }
        audit = {
            "judge": "test-judge",
            "expected": 12,
            "completed": 12,
            "families": 2,
            "primary_comparisons": [comparison],
            "prompt_repeatability": {k: v for k, v in repeat.items() if k != "groups"},
        }
        return report, audit, repeat

    def test_denominators_effects_and_routing_are_preserved(self):
        report, audit, repeat = self.fixture()
        before = copy.deepcopy((report, audit, repeat))
        tables = project_analysis(report, audit, repeat)
        row = tables["real_minus_neutral"][0]
        self.assertEqual((row["n_presentations"], row["n_families"]), (4, 2))
        self.assertEqual(
            (row["mean_difference"], row["fixes"], row["harms"]), (0.5, 2, 0)
        )
        self.assertEqual(
            (row["family_bootstrap_ci_low"], row["family_bootstrap_ci_high"]), (0, 1)
        )
        group = tables["exact_prompt_groups"][0]
        self.assertTrue(group["logical_answer_disagreement"])
        self.assertEqual(group["complete_request_hash_count"], 1)
        self.assertEqual(
            group["providers_as_reported_json"], '["provider-1", "provider-2"]'
        )
        self.assertEqual((report, audit, repeat), before)

    def test_unmeasured_repeats_are_not_zero_noise_and_changed_audits_fail(self):
        report, audit, repeat = self.fixture()
        repeat.update(
            repeated_prompt_groups=0,
            logical_jobs_in_repeated_groups=0,
            groups_with_answer_disagreement=0,
            matching_answer_pairs=0,
            all_answer_pairs=0,
            pairwise_agreement=None,
            groups=[],
        )
        audit["prompt_repeatability"] = {
            k: v for k, v in repeat.items() if k != "groups"
        }
        tables = project_analysis(report, audit, repeat)
        self.assertEqual(
            tables["exact_prompt_summary"][0]["measurement_status"],
            "generation_provenance_not_supplied_unmeasured",
        )
        self.assertIsNone(
            tables["exact_prompt_summary"][0]["generation_pairwise_agreement"]
        )
        self.assertEqual(tables["exact_prompt_groups"], [])
        audit["primary_comparisons"] = []
        with self.assertRaisesRegex(ValueError, "paired effects"):
            project_analysis(report, audit, repeat)


if __name__ == "__main__":
    unittest.main()
