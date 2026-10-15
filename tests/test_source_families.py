import unittest

from probity_benchmark.source_families import family_id, normalize_family, validate_disjoint_splits


class SourceFamilies(unittest.TestCase):
    def test_source_title_survives_formatting_changes(self):
        self.assertEqual(family_id("The FIRST\r\nOne"), family_id(" the first one "))
        self.assertEqual(family_id("Ｃafé"), family_id("café"))
        self.assertNotEqual(family_id("The First One"), family_id("The Last One"))

    def test_distinct_question_hashes_cannot_hide_shared_source(self):
        index = {"by_parent": {"question_a": "family_a", "question_b": "family_a", "question_c": "family_b"}}
        with self.assertRaisesRegex(ValueError, "cross splits"):
            validate_disjoint_splits({"train": ["question_a"], "test": ["question_b"]}, index)
        counts = validate_disjoint_splits({"train": ["question_a", "question_b"], "test": ["question_c"]}, index)
        self.assertEqual(counts, {"train": 1, "test": 1})

    def test_missing_original_family_is_not_silently_treated_as_independent(self):
        for name in [None, "", "  ", 7]:
            with self.assertRaises(ValueError):
                normalize_family(name)


if __name__ == "__main__":
    unittest.main()
