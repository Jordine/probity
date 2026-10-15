"""Offline construction failures must block future paired-world inference."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

import build_scaffold as scaffold


class CounterfactualContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in scaffold.SOURCE.read_text().splitlines()]
        cls.family = json.loads(scaffold.FAMILY_MAP.read_text())["by_parent"]
        audit = json.loads(scaffold.CONSUMPTION_AUDIT.read_text())
        cls.consumed = {row["problem_id"] for group in audit["families"].values() for row in group if row["previously_consumed"]}

    def pair(self, parent=112):
        return deepcopy([row for row in self.rows if row["parent_source_line"] == parent])

    def test_complete_existing_development_pairs(self):
        for parent in (90, 112):
            self.assertEqual(len(scaffold.validate_pair(self.pair(parent), self.consumed, self.family)), 2)

    def test_same_gold_cannot_masquerade_as_balanced_pair(self):
        pair = self.pair()
        pair[1]["correct_answer"] = pair[0]["correct_answer"]
        with self.assertRaisesRegex(ValueError, "Gold must flip"):
            scaffold.validate_pair(pair, self.consumed, self.family)

    def test_option_wording_change_is_information_leak(self):
        pair = self.pair()
        pair[1]["option_a"] += " (the coordinator)"
        with self.assertRaisesRegex(ValueError, "fixed field"):
            scaffold.validate_pair(pair, self.consumed, self.family)

    def test_partial_name_edit_with_valid_digest_is_rejected(self):
        pair = self.pair()
        pair[1]["synthetic_story"] = pair[0]["synthetic_story"].replace("Isaac Hunt", "Peter Ford", 1)
        pair[1]["story_sha256"] = scaffold.digest(pair[1]["synthetic_story"])
        with self.assertRaisesRegex(ValueError, "complete simultaneous"):
            scaffold.validate_pair(pair, self.consumed, self.family)

    def test_missing_consumption_proof_blocks_pair(self):
        with self.assertRaisesRegex(ValueError, "previously consumed"):
            scaffold.validate_pair(self.pair(), set(), self.family)

    def test_unapproved_parent_blocks_pair(self):
        pair = self.pair()
        for row in pair:
            row["parent_source_line"] = -1
        with self.assertRaisesRegex(ValueError, "preselected"):
            scaffold.validate_pair(pair, self.consumed, self.family)

    def test_alternate_world_context_injection_is_rejected(self):
        pair = self.pair()
        messages, _ = scaffold.control_messages(pair[0], True, False)
        scaffold.validate_single_source(messages, pair[0], pair[1])
        messages.append({"role": "user", "content": pair[1]["synthetic_story"]})
        with self.assertRaisesRegex(ValueError, "Alternate source leaked"):
            scaffold.validate_single_source(messages, pair[0], pair[1])

    def test_no_story_world_indifference_and_coupled_scoring(self):
        for parent in (90, 112):
            pair = self.pair(parent)
            for swapped in (False, True):
                first, gold0 = scaffold.control_messages(pair[0], False, swapped)
                second, gold1 = scaffold.control_messages(pair[1], False, swapped)
                self.assertEqual(first, second)
                for answer in ("A", "B"):
                    self.assertEqual(scaffold.paired_no_story_accuracy(answer, [gold0, gold1]), 0.5)
                with self.assertRaises(ValueError):
                    scaffold.paired_no_story_accuracy("A", [gold0, gold0])
                with self.assertRaises(ValueError):
                    scaffold.paired_no_story_accuracy("invalid", [gold0, gold1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
