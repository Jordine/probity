import json
import unittest

from probity_benchmark.annotate_debates import reconcile


class SilverAnnotations(unittest.TestCase):
    def test_escaped_unicode_claim_maps_to_raw_json(self):
        text = 'Café said "blue".'
        raw = json.dumps({"argument": text, "answer": "A"}, ensure_ascii=True)
        first = [
            {
                "text": text,
                "verdict": "supported",
                "evidence_quotes": [text],
                "reason": "direct",
            }
        ]
        second = [
            {
                "index": 0,
                "verdict": "supported",
                "evidence_quotes": [text],
                "reason": "direct",
            }
        ]
        spans, audit = reconcile(first, second, raw, text)
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(json.loads('"' + raw[span["start"] : span["end"]] + '"'), text)
        self.assertTrue(audit[0]["accepted"])

    def test_disagreement_and_fabricated_quotes_are_not_negatives(self):
        raw = json.dumps({"argument": "The value is nine.", "answer": "A"})
        first = [
            {
                "text": "The value is nine.",
                "verdict": "contradicted",
                "evidence_quotes": ["The value is eight."],
                "reason": "different",
            }
        ]
        second = [
            {
                "index": 0,
                "verdict": "supported",
                "evidence_quotes": ["invented evidence"],
                "reason": "unsupported",
            }
        ]
        spans, audit = reconcile(first, second, raw, "The value is eight.")
        self.assertEqual(spans, [])
        self.assertIn("label_disagreement", audit[0]["rejection_reasons"])
        self.assertIn("unverified_evidence_quote", audit[0]["rejection_reasons"])

    def test_missing_verifier_claim_stops(self):
        with self.assertRaises(ValueError):
            reconcile([{"text": "x"}], [], '{"argument":"x"}', "x")

    def test_shared_interpretation_is_not_a_factual_label(self):
        claim = {
            "text": "X is the main success story.",
            "verdict": "contradicted",
            "claim_kind": "interpretation_or_advocacy",
            "evidence_quotes": ["Y is larger."],
            "reason": "framing",
        }
        check = {
            "index": 0,
            "verdict": "contradicted",
            "claim_kind": "checkable_fact",
            "evidence_quotes": ["Y is larger."],
            "reason": "framing",
        }
        spans, audit = reconcile(
            [claim], [check], json.dumps({"argument": claim["text"]}), "Y is larger."
        )
        self.assertEqual(spans, [])
        self.assertIn("interpretation_or_advocacy", audit[0]["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
