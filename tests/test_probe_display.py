import json
import unittest

from probity_benchmark.evaluate_debate_transfer import argument_mapping, segments


class ProbeDisplayTests(unittest.TestCase):
    def test_decoded_argument_maps_to_complete_raw_json_occurrences(self):
        samples = [
            "Café blue blue. 3.2 credits.",
            'Quoted "words" and a backslash \\ here.\nNext sentence.',
            "Emoji 😀; tabs\tand\r\nnewlines.",
        ]
        for text in samples:
            for ascii_only in (True, False):
                raw = json.dumps(
                    {"answer": "A", "argument": text}, ensure_ascii=ascii_only
                )
                decoded, mapping = argument_mapping(raw)
                self.assertEqual(decoded, text)
                self.assertEqual(len(mapping), len(text))
                for a, b in segments(text):
                    encoded = raw[mapping[a][0] : mapping[b - 1][1]]
                    self.assertEqual(json.loads('"' + encoded + '"'), text[a:b])
                self.assertEqual(
                    "".join(text[a:b] for a, b in segments(text, clauses=True)), text
                )

    def test_json_scaffolding_is_outside_argument_intervals(self):
        raw = '{"argument": "Blue. Blue.", "answer": "A"}'
        text, mapping = argument_mapping(raw)
        self.assertEqual(raw[mapping[0][0] : mapping[-1][1]], text)
        self.assertEqual([text[a:b] for a, b in segments(text)], ["Blue. ", "Blue."])

    def test_invalid_or_ambiguous_json_is_rejected(self):
        for raw in [
            '{"argument":"A",}',
            '{"argument":"A","argument":"B"}',
            '{"answer":"A"}',
        ]:
            with self.assertRaises(ValueError):
                argument_mapping(raw)


if __name__ == "__main__":
    unittest.main()
