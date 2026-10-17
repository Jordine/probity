"""Offline regressions for a future parser, never current experiment outputs."""

import copy
import json
import random
import unittest
from probity_benchmark.local3_future_units import (
    make_payload,
    quote_aware_partition,
    units_from_raw,
    validate_annotation,
)


def fixture():
    source = "Ari gave Bea a compass."
    argument = "Bea gave Ari a compass."
    payload, _, _ = make_payload(source, json.dumps({"argument": argument}))
    atom = {
        "kind": "provider_recipient",
        "scope": "asserted",
        "argument_spans": [{"start": 0, "end": len(argument), "text": argument}],
        "roles": {
            "provider": "Bea",
            "recipient": "Ari",
            "transferred_object": "compass",
            "speaker": None,
            "attributed_content": None,
        },
        "verdict": "contradicted",
        "source_evidence": [
            {"paragraph_id": 0, "start": 0, "end": len(source), "text": source}
        ],
        "reason": "The source has the opposite provider and recipient.",
    }
    response = {
        "checks": [
            {
                "unit_index": 0,
                "exact_text": argument,
                "enumeration_complete": True,
                "propositions": [atom],
                "whole_unit_verdict": "contradicted",
                "reason": "One relation, complete scope.",
            }
        ]
    }
    return (payload, response)


class FutureUnitTests(unittest.TestCase):

    def test_complete_expected_boundaries(self):
        for text, expected in [
            ("First.  Second!\nThird?\t", ["First.  ", "Second!\n", "Third?\t"]),
            (
                "Dr. Vale met Mrs. Moss. They left.",
                ["Dr. Vale met Mrs. Moss. ", "They left."],
            ),
            (
                'Mira said, "Stay here. Wait for dawn." Leo left.',
                ['Mira said, "Stay here. Wait for dawn." ', "Leo left."],
            ),
            (
                "Mira said, “Wait. Stay.” Leo left.",
                ["Mira said, “Wait. Stay.” ", "Leo left."],
            ),
            (
                "Mira said, “He told me ‘Wait. Stay.’ yesterday.” Leo left.",
                ["Mira said, “He told me ‘Wait. Stay.’ yesterday.” ", "Leo left."],
            ),
            ('"Wait!" she said. Leo left.', ['"Wait!" she said. ', "Leo left."]),
            (
                "The workers' tools weren't lost. Ari’s tool wasn’t lost.",
                ["The workers' tools weren't lost. ", "Ari’s tool wasn’t lost."],
            ),
            (
                "The value was 3.14. J. R. Vale agreed.",
                ["The value was 3.14. ", "J. R. Vale agreed."],
            ),
            ("It worked (really!). Next.", ["It worked (really!). ", "Next."]),
            ("A note: ‘Stay. Wait.’ Next.", ["A note: ‘Stay. Wait.’ ", "Next."]),
            ("   ", ["   "]),
            ("", []),
        ]:
            with self.subTest(text=text, expected=expected):
                spans, _ = quote_aware_partition(text)
                assert [text[a:b] for a, b in spans] == expected
                assert "".join((text[a:b] for a, b in spans)) == text

    def test_unclosed_quote_retains_complete_remainder_and_reports_position(self):
        text = 'First. She said, "Stay. Wait.\nEverything after remains.'
        spans, events = quote_aware_partition(text)
        assert [text[a:b] for a, b in spans] == [
            "First. ",
            'She said, "Stay. Wait.\nEverything after remains.',
        ]
        assert events == [
            {
                "kind": "unclosed_quote",
                "offset": text.index('"'),
                "expected_closer": '"',
            }
        ]

    def test_unicode_json_offsets_preserve_escape_and_surrogate_pairs(self):
        argument = "Dr. Vale said, “Wait. Stay.”\nA path \\ and emoji 😀 remain.\t"
        raw = json.dumps({"argument": argument, "answer": "B"}, ensure_ascii=True)
        returned, units, _ = units_from_raw(raw)
        assert returned == argument
        assert "".join((u["text"] for u in units)) == argument
        for unit in units:
            assert raw[unit["raw_start"] : unit["raw_end"]] == unit["raw_text"]
            assert json.loads('"' + unit["raw_text"] + '"') == unit["text"]
        assert all((a["raw_end"] == b["raw_start"] for a, b in zip(units, units[1:])))

    def test_random_punctuation_never_loses_or_duplicates_text(self):
        rng = random.Random(391)
        alphabet = "abcXYZ.!?\n\t \"'‘’“”()😀"
        for _ in range(80):
            text = "".join(rng.choices(alphabet, k=rng.randrange(1, 400)))
            spans, _ = quote_aware_partition(text)
            assert spans[0][0] == 0 and spans[-1][1] == len(text)
            assert all((a[1] == b[0] for a, b in zip(spans, spans[1:])))
            assert "".join((text[a:b] for a, b in spans)) == text

    def test_atomic_roles_and_lossless_long_reason_are_accepted(self):
        payload, response = fixture()
        response["checks"][0]["propositions"][0]["reason"] = (
            "Complete reasoning. " * 100
        )
        before = copy.deepcopy(response)
        validate_annotation(response, payload)
        assert response == before

    def test_reject_invalid_mechanical_annotation_contract(self):
        for mutation in [
            "unit_text",
            "unit_index",
            "atomic_span",
            "evidence_text",
            "evidence_index",
            "missing_role",
            "scope",
            "missing_evidence",
            "incomplete_enumeration",
        ]:
            with self.subTest(mutation=mutation):
                payload, response = fixture()
                check = response["checks"][0]
                atom = check["propositions"][0]
                if mutation == "unit_text":
                    check["exact_text"] += " changed"
                elif mutation == "unit_index":
                    check["unit_index"] = 1
                elif mutation == "atomic_span":
                    atom["argument_spans"][0]["end"] -= 1
                elif mutation == "evidence_text":
                    atom["source_evidence"][0]["text"] = "Wrong evidence."
                elif mutation == "evidence_index":
                    atom["source_evidence"][0]["paragraph_id"] = 1
                elif mutation == "missing_role":
                    atom["roles"]["recipient"] = None
                elif mutation == "scope":
                    atom["scope"] = "hypothetical"
                elif mutation == "missing_evidence":
                    atom["source_evidence"] = []
                else:
                    check["enumeration_complete"] = False
                with self.assertRaises(ValueError):
                    validate_annotation(response, payload)

    def test_accurate_quote_with_false_attribution_keeps_mixed_unit_unknown(self):
        source = 'Ada said, "Wait here."'
        argument = 'Ben said, "Wait here."'
        payload, _, _ = make_payload(source, json.dumps({"argument": argument}))
        evidence = [{"paragraph_id": 0, "start": 0, "end": len(source), "text": source}]
        roles = {
            "provider": None,
            "recipient": None,
            "transferred_object": None,
            "speaker": None,
            "attributed_content": "Wait here.",
        }
        quote_start = argument.index("Wait")
        quote = {
            "kind": "quoted_content",
            "scope": "reported_speech",
            "argument_spans": [
                {
                    "start": quote_start,
                    "end": quote_start + len("Wait here."),
                    "text": "Wait here.",
                }
            ],
            "roles": roles,
            "verdict": "supported",
            "source_evidence": evidence,
            "reason": "These exact words appear in source reporting.",
        }
        attribution = {
            "kind": "speaker_attribution",
            "scope": "asserted",
            "argument_spans": [{"start": 0, "end": len(argument), "text": argument}],
            "roles": {**roles, "speaker": "Ben"},
            "verdict": "contradicted",
            "source_evidence": evidence,
            "reason": "The source attributes these words to Ada, not Ben.",
        }
        response = {
            "checks": [
                {
                    "unit_index": 0,
                    "exact_text": argument,
                    "enumeration_complete": True,
                    "propositions": [quote, attribution],
                    "whole_unit_verdict": "unknown",
                    "reason": "Quotation content and speaker attribution differ.",
                }
            ]
        }
        validate_annotation(response, payload)
        response["checks"][0]["whole_unit_verdict"] = "contradicted"
        with self.assertRaises(ValueError):
            validate_annotation(response, payload)

    def test_complete_source_payload_has_no_gold_roles_or_probe_fields(self):
        source = (
            "paragraph zero\n\n" + "full source text " * 2000 + "\n\nlast paragraph"
        )
        argument = "Dr. Vale arrived. The last line remains."
        payload, _, _ = make_payload(source, json.dumps({"argument": argument}))
        assert (
            "\n\n".join((p["text"] for p in payload["SOURCE_STORY_PARAGRAPHS"]))
            == source
        )
        assert payload["COMPLETE_ARGUMENT"] == argument
        assert set(payload) == {"SOURCE_STORY_PARAGRAPHS", "COMPLETE_ARGUMENT", "UNITS"}


if __name__ == "__main__":
    unittest.main()
