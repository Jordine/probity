import copy
import unittest

from probity_benchmark.grid6_rental import cost_bound, owned, validate


class RentalScopeTests(unittest.TestCase):
    def setUp(self):
        self.intent = {"label": "probity-iteration6-test", "existing_instance_ids": [10],
                       "hourly_upper_usd": 1.5, "max_runtime_hours": 6,
                       "traffic_reserve_usd": .5, "reservation_usd": 12,
                       "created_at_unix": 1000, "destroy_at_unix": 22600}

    def test_exact_owned_label_excludes_other_projects(self):
        rows = [{"id": 10, "label": "ee_known_issue_seeds"},
                {"id": 11, "label": "probity-iteration6-test-other"},
                {"id": 12, "label": "probity-iteration6-test"}]
        self.assertEqual([r["id"] for r in owned(self.intent, rows)], [12])

    def test_relabelled_preexisting_instance_never_destroyed(self):
        with self.assertRaisesRegex(ValueError, "preexisting"):
            owned(self.intent, [{"id": 10, "label": self.intent["label"]}])

    def test_fixed_deadline_and_traffic_reservation(self):
        validate(self.intent)
        self.assertAlmostEqual(cost_bound(1.5, 6, .5), 9.65)
        for field, value in [("destroy_at_unix", 22601), ("reservation_usd", 9),
                             ("reservation_usd", float("nan")), ("max_runtime_hours", 9),
                             ("hourly_upper_usd", 2), ("traffic_reserve_usd", -1)]:
            intent = copy.deepcopy(self.intent)
            intent[field] = value
            with self.assertRaises(ValueError, msg=field):
                validate(intent)


if __name__ == "__main__":
    unittest.main()
