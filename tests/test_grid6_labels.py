import copy
import unittest

from probity_benchmark.grid6_labels import PRIMARY, SECONDARY, assemble, resolve


def check(verdict="supported", valid=True):
    return {"index": 0, "exact_text": "Ada arrived.", "verdict": verdict,
            "claim_kind": "checkable_fact", "local_label_valid": valid,
            "story_paragraph_ids": [0]}


class LabelTierTests(unittest.TestCase):
    def setUp(self):
        self.unit = {"display_text": "Ada arrived.", "start": 0, "end": 12}
        self.rows = [{"id": "turn", "source_problem": "source", "source_story_sha256": "story",
                      "request_sha256": "generation", "units": [self.unit]}]
        self.meta = [{"job_id": "job", "units": [{"index": 0, "turn_id": "turn",
                      "unit_index": 0, "start": 0, "end": 12}]}]

    def annotation(self, model, verdict):
        return {"job_id": "job", "model": model, "request_sha256": model,
                "parsed": {"checks": [check(verdict)]}}

    def test_single_reader_does_not_become_consensus(self):
        rows, summary = assemble(self.rows, self.meta, [],
                                 [self.annotation(PRIMARY, "contradicted")], [PRIMARY, SECONDARY])
        self.assertEqual(rows[0]["primary_label"], 1)
        self.assertIsNone(rows[0]["consensus_label"])
        self.assertEqual(rows[0]["consensus_status"], "not_dual_annotated")
        self.assertTrue(summary["primary_complete"])

    def test_missing_and_unknown_are_not_negatives(self):
        result = resolve(self.unit, {})
        self.assertFalse(result["primary_annotated"])
        self.assertIsNone(result["primary_label"])
        result = resolve(self.unit, {PRIMARY: {"check": check("unknown")}})
        self.assertTrue(result["primary_annotated"])
        self.assertIsNone(result["primary_label"])
        result = resolve(self.unit, {PRIMARY: {"check": check("supported", False)}})
        self.assertIsNone(result["primary_label"])

    def test_disagreement_preserves_primary_but_not_consensus(self):
        rows, _ = assemble(self.rows, self.meta, [], [self.annotation(PRIMARY, "supported"),
                           self.annotation(SECONDARY, "contradicted")], [PRIMARY, SECONDARY])
        self.assertEqual(rows[0]["primary_label"], 0)
        self.assertIsNone(rows[0]["consensus_label"])
        self.assertEqual(rows[0]["consensus_status"], "verdict_disagreement")

    def test_legacy_and_new_have_same_primary_interpretation(self):
        legacy = [{"id": "turn", "units": [self.unit], "annotations": [
            {"unit_index": 0, "unit": self.unit, "checks": [check("supported"), check("unknown")],
             "request_sha256": ["old_primary", "old_secondary"]}]}]
        old, _ = assemble(self.rows, [], legacy, [], [PRIMARY, SECONDARY])
        new, _ = assemble(self.rows, self.meta, [], [self.annotation(PRIMARY, "supported")], [PRIMARY, SECONDARY])
        self.assertEqual(old[0]["primary_label"], new[0]["primary_label"])
        self.assertIsNone(old[0]["consensus_label"])
        self.assertIsNone(new[0]["consensus_label"])

    def test_incomplete_job_rejected(self):
        a = self.annotation(PRIMARY, "supported")
        a["parsed"]["checks"] = []
        with self.assertRaisesRegex(ValueError, "incomplete"):
            assemble(self.rows, self.meta, [], [a], [PRIMARY, SECONDARY])

    def test_misbound_text_offsets_and_duplicate_verdict_rejected(self):
        a = self.annotation(PRIMARY, "supported")
        a["parsed"]["checks"][0]["exact_text"] = "Bo arrived."
        with self.assertRaisesRegex(ValueError, "exact unit"):
            assemble(self.rows, self.meta, [], [a], [PRIMARY, SECONDARY])
        meta = copy.deepcopy(self.meta)
        meta[0]["units"][0]["start"] = 1
        with self.assertRaisesRegex(ValueError, "offsets"):
            assemble(self.rows, meta, [], [self.annotation(PRIMARY, "supported")], [PRIMARY, SECONDARY])
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            assemble(self.rows, self.meta, [], [self.annotation(PRIMARY, "supported"),
                     self.annotation(PRIMARY, "contradicted")], [PRIMARY, SECONDARY])


    def test_partial_overlay_preserves_exact_raw_provenance(self):
        meta = copy.deepcopy(self.meta)
        meta[0]["job_id"] = "partial-alias"
        meta[0]["units"][0]["index"] = 7
        annotation = self.annotation(PRIMARY, "supported")
        annotation["job_id"] = "partial-alias"
        annotation["parsed"]["checks"][0]["index"] = 7
        annotation["provenance"] = {"receipt_path": "raw.jsonl", "receipt_line": 3,
                                    "receipt_sha256": "raw-sha", "original_job_id": "job"}
        rows, summary = assemble(self.rows, meta, [], [annotation, copy.deepcopy(annotation)], [PRIMARY, SECONDARY])
        self.assertTrue(summary["primary_complete"])
        self.assertEqual(rows[0]["checks_by_model"][PRIMARY]["provenance"], annotation["provenance"])
        conflicting = copy.deepcopy(annotation)
        conflicting["provenance"]["receipt_line"] = 4
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            assemble(self.rows, meta, [], [annotation, conflicting], [PRIMARY, SECONDARY])


    def test_administrative_length_waiver_remains_visible_and_unit_bound(self):
        annotation = self.annotation(PRIMARY, "supported")
        annotation["provenance"] = {"receipt_sha256": "raw"}
        annotation["unit_waivers"] = {"0": {"turn_id": "turn", "unit_index": 0, "field": "reason"}}
        rows, summary = assemble(self.rows, self.meta, [], [annotation], [PRIMARY, SECONDARY])
        self.assertEqual(summary["administrative_waived_checks"], 1)
        self.assertEqual(rows[0]["checks_by_model"][PRIMARY]["provenance"]["administrative_waiver"]["field"], "reason")
        annotation["unit_waivers"]["0"]["unit_index"] = 1
        with self.assertRaisesRegex(ValueError, "waiver unit"):
            assemble(self.rows, self.meta, [], [annotation], [PRIMARY, SECONDARY])

if __name__ == "__main__":
    unittest.main()
