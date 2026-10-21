import unittest

from probity_benchmark.grid6_factual_behavior import summarize


class FactualBehaviorTests(unittest.TestCase):
    def test_unknowns_do_not_become_truthful(self):
        labels = [{'turn_id': 'a', 'primary_label': 1, 'primary_annotated': True},
                  {'turn_id': 'b', 'primary_label': None, 'primary_annotated': True}]
        meta = {turn: {'family': 'f', 'matched_question': 'q', 'turn_index': 0,
                       'arm': arm, 'role': 'wrong_advocate'} for turn, arm in [('a', 'A'), ('b', 'B')]}
        result, _ = summarize(labels, meta, 'primary_label', bootstraps=20)
        self.assertEqual(result['metrics']['contradiction_fraction']['families'], 0)
        self.assertEqual(result['metrics']['eligible_coverage']['B_minus_A'], -1)

    def test_paired_zero_difference(self):
        labels = [{'turn_id': turn, 'primary_label': value, 'primary_annotated': True}
                  for turn in ['a', 'b'] for value in [0, 1, None]]
        meta = {turn: {'family': 'f', 'matched_question': 'q', 'turn_index': 0,
                       'arm': arm, 'role': 'wrong_advocate'} for turn, arm in [('a', 'A'), ('b', 'B')]}
        result, _ = summarize(labels, meta, 'primary_label', bootstraps=20)
        self.assertEqual(result['metrics']['contradiction_fraction']['B_minus_A'], 0)
        self.assertEqual(result['metrics']['contradiction_fraction']['delta_ci95'], [0, 0])


if __name__ == '__main__':
    unittest.main()
