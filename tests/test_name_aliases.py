import unittest

from probity_benchmark.build_paired_worlds import unmapped_name_forms
from probity_benchmark.counterfactuals import swap_entities


class NameAliases(unittest.TestCase):
    def test_country_swap_cannot_leave_its_adjectives_unchanged(self):
        source = "Zelvaria borders Prandoria. Zelvarian soil meets Prandorian soil."
        names = {"Zelvaria": "Prandoria", "Prandoria": "Zelvaria"}
        self.assertEqual(
            unmapped_name_forms(source, names), ["Prandorian", "Zelvarian"]
        )
        names.update(Zelvarian="Prandorian", Prandorian="Zelvarian")
        self.assertEqual(unmapped_name_forms(source, names), [])
        changed = swap_entities(source, names)
        self.assertIn("Prandorian soil meets Zelvarian soil", changed)
        self.assertEqual(swap_entities(changed, names), source)

    def test_hyphens_and_possessives_keep_word_boundary_swaps(self):
        source = "Victorville-style interfaces on Northfield's waterfront."
        names = {"Victorville": "Northfield", "Northfield": "Victorville"}
        self.assertEqual(unmapped_name_forms(source, names), [])
        self.assertEqual(
            swap_entities(source, names),
            "Northfield-style interfaces on Victorville's waterfront.",
        )


if __name__ == "__main__":
    unittest.main()
