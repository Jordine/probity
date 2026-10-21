import unittest

from probity_benchmark.grid6_live_evaluate import attach_metadata


class LiveProbePopulationTests(unittest.TestCase):
    def test_requires_suffix_and_exact_response_identity(self):
        score = {'episode_id': 'branch', 'turn_index': 2, 'response_body_sha256': 'raw'}
        meta = {'branch:2': {'response_body_sha256': 'raw', 'family': 'story', 'policy': 'neutral'}}
        result = attach_metadata(score, meta)
        self.assertEqual(result['arm'], 'neutral')
        self.assertNotIn('arm', score)
        with self.assertRaisesRegex(ValueError, 'response differs'):
            attach_metadata({**score, 'response_body_sha256': 'another'}, meta)
        with self.assertRaisesRegex(ValueError, 'outside frozen'):
            attach_metadata({**score, 'episode_id': 'prompt-experiment'}, meta)
        with self.assertRaisesRegex(ValueError, 'shared prefix'):
            attach_metadata({**score, 'turn_index': 1}, {'branch:1': meta['branch:2']})


if __name__ == '__main__':
    unittest.main()
