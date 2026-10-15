import unittest

from probity_benchmark.validate_worlds import paired_chance_contract


class WorldValidationContract(unittest.TestCase):
    def test_only_information_matched_opposite_worlds_imply_chance(self):
        original = {
            "parent_problem_id": "parent",
            "variant": "original",
            "correct_answer": "A",
            "question": "Who won?",
            "option_a": "Alex",
            "option_b": "Robin",
        }
        twin = {**original, "variant": "swapped_world", "correct_answer": "B"}
        self.assertTrue(paired_chance_contract([original, twin]))
        self.assertFalse(paired_chance_contract([original]))
        self.assertFalse(paired_chance_contract([original, original]))
        self.assertFalse(
            paired_chance_contract([original, {**twin, "question": "Who lost?"}])
        )
        self.assertFalse(paired_chance_contract([]))


if __name__ == "__main__":
    unittest.main()
