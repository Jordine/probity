import unittest

from probity_benchmark.judge_assistance import paired_stats, public_speaker_answers


class JudgeAssistance(unittest.TestCase):
    def test_recommendation_uses_public_answer_not_private_assignment(self):
        turns = [
            {"speaker": "Blue", "answer": "A", "assigned_answer": "B"},
            {"speaker": "Red", "answer": "B", "assigned_answer": "A"},
        ]
        self.assertEqual(public_speaker_answers(turns), {"Blue": "A", "Red": "B"})
        turns.append({"speaker": "Blue", "answer": "B", "assigned_answer": "B"})
        with self.assertRaises(ValueError):
            public_speaker_answers(turns)

    def test_pairing_resamples_parents_not_repeated_episodes(self):
        rows = [{"group": "a", "difference": 1} for _ in range(8)]
        rows.append({"group": "b", "difference": -1})
        stats = paired_stats(rows)
        self.assertEqual(stats["mean_difference"], 0)
        self.assertEqual(stats["parent_groups"], 2)
        self.assertEqual(stats["episodes"], 9)


if __name__ == "__main__":
    unittest.main()
