import unittest

from probity_benchmark.api import strict_json
from probity_benchmark.counterfactuals import swap_entities
from probity_benchmark.preflight import Problem
from probity_benchmark.run_debates import judge_messages, debater_messages


class DiagnosticDebateTests(unittest.TestCase):
    def test_named_assignment_tracks_swapped_option(self):
        p = Problem("id", "Story", "Where?", ("West", "East"), "A", 1)
        messages = debater_messages(p, True, "Red", "A", [])
        self.assertIn("answer A: East", messages[0]["content"])
        self.assertIn("Your position is A: East", messages[1]["content"])

    def test_judge_view_uses_only_public_fields(self):
        p = Problem("private_id", "PRIVATE_STORY", "Where?", ("West", "East"), "A", 1)
        turns = [
            dict(
                speaker="Red",
                answer="B",
                argument="The argument.",
                assigned_answer="SECRET_ROLE",
                role_adherent="SECRET_LABEL",
            )
        ]
        import json

        text = json.dumps(judge_messages(p, False, turns))
        for secret in ["PRIVATE_STORY", "private_id", "SECRET_ROLE", "SECRET_LABEL"]:
            self.assertNotIn(secret, text)
        self.assertIn("The argument.", text)
        self.assertIn("A: West", text)

    def test_entity_renaming_is_simultaneous_and_reversible(self):
        mapping = {
            "Aldoria": "Jarvonia",
            "Jarvonia": "Aldoria",
            "Aldorian": "Jarvonian",
            "Jarvonian": "Aldorian",
        }
        original = "In Aldoria, an Aldorian read Jarvonian reports from Jarvonia."
        changed = swap_entities(original, mapping)
        self.assertEqual(
            changed, "In Jarvonia, an Jarvonian read Aldorian reports from Aldoria."
        )
        self.assertEqual(swap_entities(changed, mapping), original)

    def test_duplicate_output_keys_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"answer":"A","answer":"B"}')


if __name__ == "__main__":
    unittest.main()
