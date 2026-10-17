import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from probity_benchmark.budget import PilotBudget


class PilotBudgetTests(unittest.TestCase):
    def test_concurrent_reservations_cannot_spend_same_balance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"cap_usd": 1, "entries": {}}))

            def reserve(key):
                try:
                    PilotBudget(path).reserve(key, 0.75, "test")
                    return True
                except RuntimeError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(reserve, ["a", "b"]))
            self.assertEqual(sum(outcomes), 1)
            ledger = json.loads(path.read_text())
            self.assertEqual(PilotBudget.committed(ledger), 0.75)
            key = next(iter(ledger["entries"]))
            PilotBudget(path).settle(key, 0.25, "actual")
            PilotBudget(path).reserve("next", 0.75, "test")
            self.assertEqual(PilotBudget.committed(json.loads(path.read_text())), 1)

    def test_overrun_is_recorded_instead_of_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps({"cap_usd": 1, "entries": {}}))
            book = PilotBudget(path)
            book.reserve("a", 0.5, "test")
            with self.assertRaises(RuntimeError):
                book.settle("a", 0.6, "provider overrun")
            self.assertEqual(
                json.loads(path.read_text())["entries"]["a"]["amount_usd"], 0.6
            )


if __name__ == "__main__":
    unittest.main()
