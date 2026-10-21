"""Offline integration audit for private iteration5 rental wiring.

Every provider and process-launch function is mocked; any unexpected subprocess
is a test failure. The real process-safe budget operates only in temporary files.
"""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "runs/iteration5/coordinator/rental_v1.py"
SPEC = importlib.util.spec_from_file_location("grid5_rental_integration_tested", SCRIPT)
rental = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rental)


class World:
    def __init__(self, folder):
        self.root = Path(folder)
        self.meta = self.root / "runs/iteration5/coordinator"
        (self.meta / "logs").mkdir(parents=True)
        self.ledger = self.meta.parent / "budget.json"
        self.ledger.write_text(json.dumps({"cap_usd": 45, "entries": {}}))
        self.ready = self.meta / "ready.json"
        self.ready.write_text(json.dumps({"ready_for_rental": True, "sha256": {}}))
        self.receipt = self.meta / "rental_v1.json"
        self.intent = self.meta / "rental_intent_v1.json"
        self.now = 1000.0
        self.calls = []
        self.creates = 0
        self.destroyed = []
        self.created = False
        self.create_mode = "success"
        self.startup_mode = "success"
        self.watchdog_mode = "success"
        self.empty_polls_after_create = 0
        self.guard_state = None
        self.offer = {"id": 99, "dph_total": 1.0, "inet_down_cost": .001,
                      "inet_up_cost": 0, "gpu_name": "RTX PRO 6000 WS", "num_gpus": 1,
                      "gpu_ram": 96000, "disk_space": 240, "duration": 100000}

    def append(self, path, row):
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def active_row(self):
        return {"id": 22, "label": rental.LABEL}

    def provider(self, *args, **kwargs):
        self.calls.append(args)
        if args[:2] == ("search", "offers"):
            return [self.offer]
        if args == ("show", "instances"):
            rows = [{"id": 11, "label": "preexisting-unrelated"}]
            if self.created and not self.destroyed:
                if self.empty_polls_after_create:
                    self.empty_polls_after_create -= 1
                else:
                    rows.append(self.active_row())
            return rows
        if args[:2] == ("create", "instance"):
            self.creates += 1
            self.created = True
            self.assert_startup_preceded_create()
            if self.create_mode == "timeout":
                raise subprocess.TimeoutExpired("mock provider create", 120)
            if self.create_mode == "abort_during_create":
                self.intent.with_suffix(".abort.json").write_text(json.dumps({
                    "intent_sha256": rental.intent_guard.digest(self.intent.read_bytes()),
                    "abort": True, "reason": "mock independent cleanup latch"}))
            return {"success": True, "new_contract": 22}
        if args[:2] == ("destroy", "instance"):
            if args[2] != "22":
                raise AssertionError("attempt to destroy a non-owned mock instance")
            self.destroyed.append(22)
            return {"success": True}
        raise AssertionError("unexpected provider call: " + repr(args))

    def assert_startup_preceded_create(self):
        rows = rental.intent_guard.read_jsonl(self.intent.with_suffix(".guard.jsonl"))
        expected = rental.intent_guard.digest(self.intent.read_bytes())
        assert any(r.get("guard_started") is True and r.get("intent_sha256") == expected for r in rows)
        state = json.loads(self.ledger.read_text())["entries"]["gpu:iteration5:v1"]
        assert state["status"] == "reserved" and state["amount_usd"] == 15

    def launch(self, command, session, log):
        self.calls.append(("launch", session))
        value = json.loads(self.intent.read_text())
        intent_hash = rental.intent_guard.digest(self.intent.read_bytes())
        if session.endswith("intent-guard"):
            if self.startup_mode == "fail":
                raise subprocess.CalledProcessError(1, "mock tmux")
            self.guard_state = rental.intent_guard.GuardState(value)
            row = {"guard_started": True, "intent_sha256": intent_hash,
                   "destroy_at_unix": value["destroy_at_unix"]}
            if self.startup_mode == "wrong_hash":
                row["intent_sha256"] = "not-the-immutable-intent"
            self.append(self.intent.with_suffix(".guard.jsonl"), row)
            return
        if session != "probity-iteration5-watchdog":
            raise AssertionError("unexpected session")
        if self.watchdog_mode == "fail":
            raise subprocess.CalledProcessError(1, "mock tmux")
        raw = self.receipt.read_bytes()
        receipt_hash = rental.intent_guard.digest(raw)
        row = {"watchdog_started": True, "instance_id": 22, "receipt_sha256": receipt_hash,
               "destroy_at_unix": value["destroy_at_unix"], "at_unix": self.now}
        if self.watchdog_mode == "wrong_hash":
            row["receipt_sha256"] = "not-the-creation-receipt"
        self.append(self.receipt.with_suffix(".cleanup.jsonl"), row)
        if self.watchdog_mode == "success":
            handoff = rental.intent_guard.confirmed_handoff(value, raw, [row], {22})
            result = self.guard_state.step([self.active_row()], self.now, True,
                unresolved=(self.meta / "rental_create_unresolved_v1.json").exists(), handoff=handoff)
            assert result["handoff"] is not None and result["destroy_ids"] == []
            self.append(self.intent.with_suffix(".guard.jsonl"), {
                "handoff_confirmed": True, "intent_sha256": intent_hash, **result["handoff"]})

    def sleep(self, seconds):
        self.now += seconds

    def entry(self):
        return json.loads(self.ledger.read_text())["entries"]["gpu:iteration5:v1"]

    def install(self, stack):
        for name, value in {"ROOT": self.root, "META": self.meta, "RECEIPT": self.receipt,
                            "LEDGER": self.ledger, "vast": self.provider, "launch_guard": self.launch,
                            "time": SimpleNamespace(time=lambda: self.now, sleep=self.sleep)}.items():
            stack.enter_context(patch.object(rental, name, value))
        stack.enter_context(patch.object(rental.subprocess, "run", side_effect=AssertionError("unexpected real process")))
        stack.enter_context(redirect_stdout(io.StringIO()))

    def seed_receipt(self):
        value = {"schema": "probity_iteration5_rental_v1", "created_at_unix": 1000,
                 "destroy_at_unix": 33400, "max_runtime_hours": 9, "cleanup_allowance_hours": .1,
                 "hourly_upper_usd": 1, "traffic_reserve_usd": .5, "reservation_usd": 15,
                 "label": rental.LABEL, "existing_instance_ids": [11], "budget_key": "gpu:iteration5:v1"}
        self.intent.write_text(json.dumps(value))
        self.receipt.write_text(json.dumps({**value, "instance_id": 22}))
        rental.PilotBudget(self.ledger).reserve("gpu:iteration5:v1", 15, "gpu_rental")
        return value


class RentalIntegrationTests(unittest.TestCase):
    def world(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        temporary = stack.enter_context(TemporaryDirectory())
        world = World(temporary)
        world.install(stack)
        return world

    def test_success_is_guarded_before_create_and_hash_bound_afterwards(self):
        world = self.world()
        rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 1)
        self.assertEqual(world.destroyed, [])
        self.assertTrue(world.receipt.exists())
        self.assertEqual(world.entry()["status"], "reserved")
        self.assertLess(world.calls.index(("launch", "probity-iteration5-intent-guard")),
                        next(i for i, call in enumerate(world.calls) if call[:2] == ("create", "instance")))

    def test_startup_failure_never_creates_and_only_then_settles_zero(self):
        world = self.world()
        world.startup_mode = "fail"
        with self.assertRaises(subprocess.CalledProcessError):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 0)
        self.assertEqual(world.entry()["status"], "settled")
        self.assertEqual(world.entry()["amount_usd"], 0)

    def test_wrong_startup_hash_cannot_authorize_create(self):
        world = self.world()
        world.startup_mode = "wrong_hash"
        with self.assertRaisesRegex(RuntimeError, "startup was not confirmed"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 0)
        self.assertEqual(world.entry()["amount_usd"], 0)

    def test_timeout_resolves_late_exact_owner_without_second_create(self):
        world = self.world()
        world.create_mode, world.empty_polls_after_create = "timeout", 2
        rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 1)
        self.assertEqual(world.destroyed, [])
        self.assertEqual(json.loads(world.receipt.read_text())["instance_id"], 22)
        self.assertEqual(world.entry()["status"], "reserved")

    def test_unresolved_create_retains_full_reservation_and_guard_evidence(self):
        world = self.world()
        world.create_mode, world.empty_polls_after_create = "timeout", 100
        with self.assertRaisesRegex(RuntimeError, "Unresolved creation"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 1)
        self.assertFalse(world.receipt.exists())
        self.assertEqual(world.entry()["amount_usd"], 15)
        self.assertEqual(world.entry()["status"], "reserved")
        # The immutable, running intent guard remains able to identify a later
        # appearance even after the creator has given up resolving the response.
        result = world.guard_state.step([world.active_row()], world.now + 60, False, unresolved=True)
        self.assertEqual(result["destroy_ids"], [22])
        self.assertIsNone(result["handoff"])

    def test_abort_during_create_preserves_ownership_for_final_cleanup(self):
        world = self.world()
        world.create_mode = "abort_during_create"
        with self.assertRaisesRegex(RuntimeError, "latched cleanup"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertTrue(world.receipt.exists())
        self.assertEqual(json.loads(world.receipt.read_text())["instance_id"], 22)
        self.assertEqual(world.entry()["status"], "reserved")
        self.assertNotIn(("launch", "probity-iteration5-watchdog"), world.calls)

    def test_wrong_watchdog_hash_is_destroyed_without_handoff(self):
        world = self.world()
        world.watchdog_mode = "wrong_hash"
        with self.assertRaisesRegex(RuntimeError, "watchdog startup was not confirmed"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 1)
        self.assertEqual(world.destroyed, [22])
        self.assertEqual(world.entry()["status"], "reserved")

    def test_missing_guard_handoff_does_not_report_success(self):
        world = self.world()
        world.watchdog_mode = "no_handoff"
        with self.assertRaisesRegex(RuntimeError, "did not confirm.*handoff"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertTrue(world.receipt.exists())
        self.assertEqual(world.entry()["status"], "reserved")

    def test_global_budget_rejection_precedes_guard_and_create(self):
        world = self.world()
        world.ledger.write_text(json.dumps({"cap_usd": 10, "entries": {}}))
        with self.assertRaisesRegex(RuntimeError, "cumulative pilot budget"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.creates, 0)
        self.assertFalse(world.intent.exists())
        self.assertFalse(any(call[0] == "launch" for call in world.calls))

    def test_existing_intent_cannot_create_again(self):
        world = self.world()
        world.intent.write_text("{}")
        with self.assertRaisesRegex(ValueError, "never another create"):
            rental.rent(SimpleNamespace(ready_manifest=world.ready))
        self.assertEqual(world.calls, [])

    def test_settlement_uses_stable_guard_absence_not_resume_wall_clock(self):
        world = self.world()
        world.seed_receipt()
        expected = rental.intent_guard.digest(world.intent.read_bytes())
        events = [
            {"event": "owned_instance_observed", "at_unix": 1100},
            {"event": "owned_instance_verified_absent", "verified_absent": True,
             "earliest_verified_absent_at_unix": 2000},
            {"event": "owned_instance_reappeared", "at_unix": 3000},
            {"event": "owned_instance_verified_absent", "verified_absent": True,
             "earliest_verified_absent_at_unix": 4600},
        ]
        for event in events:
            world.append(world.intent.with_suffix(".guard.jsonl"),
                         {"intent_sha256": expected, "instance_id": 22, **event})
        world.now = 1000 + 12 * 3600
        rental.destroy(SimpleNamespace(reason="mock reconciliation"))
        result = json.loads((world.meta / "rental_settlement_v1.json").read_text())
        self.assertEqual(result["verified_absent_at_unix"], 4600)
        self.assertEqual(result["amount_usd"], 1.5)
        self.assertEqual(world.entry()["amount_usd"], 1.5)

    def test_settlement_matches_already_settled_ledger_after_output_write_crash(self):
        world = self.world()
        world.seed_receipt()
        # Simulate a crash just after the first destroy settled the ledger but
        # before it durably wrote rental_settlement_v1.json.
        rental.PilotBudget(world.ledger).settle("gpu:iteration5:v1", 1.5, "first verified cleanup")
        world.now = 1000 + 2 * 3600
        rental.destroy(SimpleNamespace(reason="resume after output write crash"))
        result = json.loads((world.meta / "rental_settlement_v1.json").read_text())
        self.assertEqual(result["amount_usd"], world.entry()["amount_usd"])

    def test_partial_watchdog_tail_does_not_block_settlement(self):
        world = self.world()
        world.seed_receipt()
        receipt_hash = rental.intent_guard.digest(world.receipt.read_bytes())
        world.append(world.receipt.with_suffix(".cleanup.jsonl"), {
            "verified_absent": True, "instance_id": 22, "receipt_sha256": receipt_hash, "at_unix": 4600})
        with world.receipt.with_suffix(".cleanup.jsonl").open("a") as handle:
            handle.write('{"at_unix":')
        world.now = 1000 + 2 * 3600
        rental.destroy(SimpleNamespace(reason="concurrent live cleanup-log append"))
        self.assertEqual(world.entry()["amount_usd"], 1.5)

    def test_actual_output_write_crash_reuses_pre_settlement_absence_journal(self):
        world = self.world()
        world.seed_receipt()
        world.now = 4600
        final = world.meta / "rental_settlement_v1.json"
        original_save = rental.save
        def fail_final_only(path, value):
            if path == final:
                raise OSError("mock disk interruption after ledger settlement")
            return original_save(path, value)
        with patch.object(rental, "save", side_effect=fail_final_only):
            with self.assertRaisesRegex(OSError, "mock disk interruption"):
                rental.destroy(SimpleNamespace(reason="first verified cleanup"))
        journal = world.meta / "rental_absence_verified_v1.json"
        original_journal = journal.read_bytes()
        self.assertEqual(world.entry()["status"], "settled")
        self.assertEqual(world.entry()["amount_usd"], 1.5)
        self.assertFalse(final.exists())
        world.now = 1000 + 2 * 3600
        rental.destroy(SimpleNamespace(reason="resume after interrupted final receipt write"))
        result = json.loads(final.read_text())
        self.assertEqual(result["amount_usd"], 1.5)
        self.assertEqual(result["verified_absent_at_unix"], 4600)
        self.assertEqual(result["reconciled_at_unix"], 8200)
        self.assertEqual(journal.read_bytes(), original_journal)
        self.assertNotIn("ledger_recovered_without_original_absence_journal", result)


if __name__ == "__main__":
    unittest.main()
