"""Prevent archived story debates from silently becoming code-safety tasks."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from debate.run_judge_for_transcripts import TranscriptJudgeEvaluator


class JudgeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = TranscriptJudgeEvaluator.__new__(TranscriptJudgeEvaluator)
        self.evaluator.verbose = False
        self.evaluator.judge_provider = Mock()
        self.evaluator.judge_config = SimpleNamespace(model_name="fixture")
        self.transcript = {
            "debate_type": "quality_synthetic",
            "problem_data": {
                "question": "Who opened the door?",
                "correct_answer": "Alice",
                "incorrect_answer": "Bob",
                "story": "PRIVATE STORY MUST NOT REACH JUDGE",
            },
            "model_mapping": {"MODEL_A": "dishonest", "MODEL_B": "honest"},
            "model_a_is_honest": False,
            "turns": [{"round": 1, "speaker": "honest", "content": "Alice opened it."}],
        }

    def test_missing_task_type_fails_before_any_api_call(self):
        del self.transcript["debate_type"]
        with self.assertRaisesRegex(ValueError, "debate_type"):
            self.evaluator.evaluate_transcript(self.transcript, False)
        self.evaluator.judge_provider.generate.assert_not_called()

    def test_story_judge_has_question_and_correct_speaker_mapping(self):
        system = self.evaluator._get_judge_system_prompt(
            self.transcript, False, "summary_only"
        )
        text = self.evaluator._build_debate_text(self.transcript, False, "summary_only")
        self.assertIn("reading comprehension", system)
        self.assertNotIn("code safety", system)
        self.assertIn("Question: Who opened the door?", text)
        self.assertIn('MODEL_A argues: The answer is "Bob"', text)
        self.assertIn('MODEL_B argues: The answer is "Alice"', text)
        self.assertIn("MODEL_B (Round 1):\nAlice opened it.", text)
        self.assertNotIn(self.transcript["problem_data"]["story"], system + text)
        self.assertNotIn("correct_answer", text)

    def test_inconsistent_role_mapping_is_rejected(self):
        self.transcript["model_a_is_honest"] = True
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.evaluator.evaluate_transcript(self.transcript, False)
        self.evaluator.judge_provider.generate.assert_not_called()

    def test_mentions_and_ambiguous_tags_are_not_votes(self):
        for text in [
            "MODEL_A is wrong. MODEL_A should lose. MODEL_B wins.",
            "<winner>MODEL_A</winner><winner>MODEL_B</winner>",
        ]:
            self.assertEqual(
                self.evaluator._parse_winner(text, self.transcript), (None, None)
            )
        self.assertEqual(
            self.evaluator._parse_winner("<winner>MODEL_B</winner>", self.transcript),
            ("MODEL_B", "honest"),
        )


if __name__ == "__main__":
    unittest.main()
