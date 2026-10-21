import unittest

from probity_benchmark.grid6_evaluate import estimate, join_units


class ProbeEvaluationTests(unittest.TestCase):
    def test_response_broadcast_has_no_localization(self):
        rows = [{'family': f, 'turn_id': f, 'primary_label': label,
                 'scores': {'response_broadcast': score}}
                for f, score in [('a', .1), ('b', .9)] for label in [0, 1]]
        result = estimate(rows, 'response_broadcast', 'primary_label', 20)
        self.assertEqual(result['auroc'], .5)
        self.assertEqual(result['within_response_family_macro_auroc'], .5)

    def test_missing_labels_are_not_negatives(self):
        rows = [{'family': 'a', 'turn_id': 'a', 'primary_label': label, 'scores': {'p': score}}
                for label, score in [(0, 0), (1, 1), (None, 10)]]
        result = estimate(rows, 'p', 'primary_label', 0)
        self.assertEqual(result['units'], 2)
        self.assertEqual(result['auroc'], 1)

    def test_family_weight_not_sentence_weight(self):
        rows = [{'family': 'a', 'turn_id': 'a', 'primary_label': 1, 'scores': {'p': 1}}] * 20
        rows += [{'family': 'b', 'turn_id': 'b', 'primary_label': 0, 'scores': {'p': 0}}]
        self.assertAlmostEqual(estimate(rows, 'p', 'primary_label', 0)['positive_prevalence'], .5)

    def test_wrong_context_cannot_join(self):
        labels = [{'turn_id': 'e:0', 'unit_index': 0, 'generation_request_sha256': 'a'}]
        scores = [{'episode_id': 'e', 'turn_index': 0, 'source_request_sha256': 'b',
                   'replay_contract_sha256': 'contract', 'head_provenance': {'p': {'layer': 35}},
                   'units': [{'scores': {'p': .5}}], 'response_scores': {}}]
        with self.assertRaisesRegex(ValueError, 'context'):
            join_units(labels, scores, {'e:0': {}})

    def test_sentence_join_uses_offsets_not_clause_index(self):
        unit = {'start': 17, 'end': 27, 'argument_start': 0, 'argument_end': 10,
                'text': 'A sentence', 'display_text': 'A sentence', 'granularities': ['sentence']}
        label = {'turn_id': 'e:0', 'unit_index': 0, 'generation_request_sha256': 'request',
                 'unit': unit, 'primary_label': 1, 'consensus_label': None}
        other = {**unit, 'start': 18, 'end': 20, 'granularities': ['clause'], 'scores': {'p': 999}}
        score = {'episode_id': 'e', 'turn_index': 0, 'source_request_sha256': 'request',
                 'replay_contract_sha256': 'contract', 'head_provenance': {'p': {'layer': 35}},
                 'response_body_sha256': 'response', 'family': 'f', 'arm': 'B',
                 'units': [other, {**unit, 'scores': {'p': .7}}], 'response_scores': {}}
        meta = {'e:0': {'family': 'f', 'arm': 'B', 'response_body_sha256': 'response'}}
        joined, missing = join_units([label], [score], meta)
        self.assertEqual(missing, 0)
        self.assertEqual(joined[0]['scores']['p'], .7)
        score['response_body_sha256'] = 'different-response'
        with self.assertRaisesRegex(ValueError, 'response'):
            join_units([label], [score], meta)
        score['response_body_sha256'] = 'response'
        with self.assertRaisesRegex(ValueError, 'duplicate labeled'):
            join_units([label, label], [score], meta)
        other_score = {**score, 'episode_id': 'f', 'replay_contract_sha256': 'different-contract'}
        with self.assertRaisesRegex(ValueError, 'contracts'):
            join_units([label], [score, other_score], {**meta, 'f:0': meta['e:0']})


if __name__ == '__main__':
    unittest.main()
