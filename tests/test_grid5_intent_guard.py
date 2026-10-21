"""CPU-only provider/process mocks for pre-create rental ownership cleanup."""

import fcntl
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "runs/iteration5/coordinator/intent_guard_v1.py"
SPEC = importlib.util.spec_from_file_location("grid5_intent_guard_tested", SCRIPT)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def intent(**updates):
    value = {"schema": "probity_iteration5_rental_v1", "created_at_unix": 0,
             "destroy_at_unix": 72, "max_runtime_hours": .02, "cleanup_allowance_hours": .01,
             "hourly_upper_usd": 1.0, "traffic_reserve_usd": .5, "reservation_usd": 15,
             "label": "probity-iteration5-20261016", "existing_instance_ids": [11],
             "budget_key": "gpu:iteration5:v1", "offer_id": 99}
    value.update(updates)
    return value


def receipt_and_start(value, iid=22):
    raw = json.dumps({**value, "instance_id": iid}).encode()
    row = {"watchdog_started": True, "instance_id": iid, "receipt_sha256": guard.digest(raw),
           "destroy_at_unix": value["destroy_at_unix"], "at_unix": 1}
    return raw, row


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, delay):
        self.value += delay


class IntentGuardTests(unittest.TestCase):
    def test_fixed_cost_and_ownership_envelope(self):
        guard.validate_intent(intent())
        for changes in ({"hourly_upper_usd": 1.7}, {"reservation_usd": 16},
                        {"existing_instance_ids": [11, 11]}, {"label": "unrelated-task"},
                        {"destroy_at_unix": 999}, {"traffic_reserve_usd": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                guard.validate_intent(intent(**changes))

    def test_early_empty_does_not_complete_and_late_create_is_scoped(self):
        value = intent()
        state = guard.GuardState(value)
        first = state.step([], 0, False)
        self.assertIsNone(first["handoff"])
        self.assertEqual(first["destroy_ids"], [])
        self.assertEqual(state.owned, {})
        rows = [{"id": 11, "label": value["label"]}, {"id": 12, "label": "unrelated"},
                {"id": 22, "label": value["label"]}]
        second = state.step(rows, 45, False)
        self.assertEqual(second["destroy_ids"], [22])
        self.assertEqual(set(state.owned), {22})

    def test_creator_liveness_and_setup_timeout(self):
        value = intent(destroy_at_unix=3600, max_runtime_hours=1)
        state = guard.GuardState(value)
        rows = [{"id": 22, "label": value["label"]}]
        self.assertEqual(state.step(rows, 1199, True)["destroy_ids"], [])
        self.assertEqual(state.step(rows, 1200, True)["destroy_ids"], [22])
        self.assertEqual(state.abort_reason, "setup_timeout")

    def test_handoff_exact_identity_and_full_intent(self):
        value = intent()
        raw, start = receipt_and_start(value)
        accepted = guard.confirmed_handoff(value, raw, [start], {22})
        self.assertEqual(accepted["instance_id"], 22)
        for bad in ({**start, "receipt_sha256": "wrong"}, {**start, "instance_id": 23},
                    {**start, "destroy_at_unix": 73}):
            self.assertIsNone(guard.confirmed_handoff(value, raw, [bad], {22}))
        changed = json.loads(raw)
        changed["offer_id"] = 100
        changed_raw = json.dumps(changed).encode()
        self.assertIsNone(guard.confirmed_handoff(value, changed_raw,
            [{**start, "receipt_sha256": guard.digest(changed_raw)}], {22}))
        self.assertIsNone(guard.confirmed_handoff(value, raw, [start], {22, 23}))
        self.assertIsNone(guard.confirmed_handoff(value, raw, [start], set()))

    def test_unresolved_marker_and_latched_abort_prevent_handoff(self):
        value = intent()
        rows = [{"id": 22, "label": value["label"]}]
        raw, start = receipt_and_start(value)
        handoff = guard.confirmed_handoff(value, raw, [start], {22})
        state = guard.GuardState(value)
        result = state.step(rows, 1, True, unresolved=True)
        self.assertIsNone(result["handoff"])
        self.assertEqual(result["destroy_ids"], [22])
        self.assertIsNone(state.step(rows, 2, True, handoff=handoff)["handoff"])

    def test_confirmed_handoff_resolves_old_timeout_marker_before_abort(self):
        value = intent()
        raw, start = receipt_and_start(value)
        handoff = guard.confirmed_handoff(value, raw, [start], {22})
        state = guard.GuardState(value)
        result = state.step([{"id": 22, "label": value["label"]}], 1, True,
                            unresolved=True, handoff=handoff)
        self.assertEqual(result["handoff"]["instance_id"], 22)
        self.assertIsNone(state.abort_reason)
        self.assertEqual(result["destroy_ids"], [])

    def test_changed_intent_cannot_hand_off(self):
        value = intent()
        raw, start = receipt_and_start(value)
        state = guard.GuardState(value)
        result = state.step([{"id": 22, "label": value["label"]}], 1, True,
                            intent_changed=True,
                            handoff=guard.confirmed_handoff(value, raw, [start], {22}))
        self.assertIsNone(result["handoff"])
        self.assertEqual(result["destroy_ids"], [22])
        self.assertEqual(state.abort_reason, "intent_changed")

    def test_reappearance_revokes_earliest_absence(self):
        value, events = intent(), []
        state = guard.GuardState(value)
        rows = [{"id": 22, "label": value["label"]}]
        for now, inventory in [(0, rows), (15, []), (30, []), (45, rows)]:
            events += state.step(inventory, now, False)["events"]
        events = [{"intent_sha256": "intent", **e} for e in events]
        self.assertEqual(guard.verified_absences(events, "intent"), {})
        for now in (60, 75):
            events += [{"intent_sha256": "intent", **e} for e in state.step([], now, False)["events"]]
        self.assertEqual(guard.verified_absences(events, "intent"), {22: 75})
        self.assertEqual(guard.verified_absences(events, "other"), {})

    def test_changed_live_label_never_destroyed_or_counted_absent(self):
        value = intent()
        state = guard.GuardState(value)
        state.step([{"id": 22, "label": value["label"]}], 0, False)
        for now in (15, 30, 45):
            result = state.step([{"id": 22, "label": "another-task"}], now, False)
            self.assertEqual(result["destroy_ids"], [])
            self.assertTrue(any(e["event"] == "ownership_changed" for e in result["events"]))
            self.assertIsNone(state.owned[22]["verified_absent_at_unix"])

    def test_invalid_inventory_cannot_establish_absence(self):
        for rows in ({"error": "provider down"}, [{"id": True}], [{"id": 1}, {"id": 1}]):
            with self.assertRaises(ValueError):
                guard.parse_inventory(rows)

    def test_full_guard_waits_for_late_instance_and_keeps_monitoring(self):
        with TemporaryDirectory() as temporary:
            folder, clock, value = Path(temporary), Clock(), intent()
            path, receipt = folder / "rental_intent_v1.json", folder / "rental_v1.json"
            path.write_text(json.dumps(value))
            calls, destroyed = [], []
            def provider(argv, **kwargs):
                calls.append((clock.now(), argv))
                self.assertNotIn("create", argv)
                if argv[1:3] == ["show", "instances"]:
                    rows = [{"id": 11, "label": value["label"]}, {"id": 12, "label": "unrelated"}]
                    if clock.now() >= 45 and not destroyed:
                        rows.append({"id": 22, "label": value["label"]})
                    return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
                self.assertEqual(argv[1:4], ["destroy", "instance", "22"])
                destroyed.append(22)
                return SimpleNamespace(returncode=0, stdout='{"success":true}', stderr="")
            def identity(pid):
                return {"pid": pid, "start_ticks": 1} if clock.now() == 0 else None
            code = guard.run_guard(path, receipt, 42, run=provider, now=clock.now,
                                   sleep=clock.sleep, identity=identity)
            self.assertEqual(code, 0)
            self.assertEqual(destroyed, [22])
            self.assertEqual(clock.now(), 108)
            events = guard.read_jsonl(path.with_suffix(".guard.jsonl"))
            self.assertTrue(events[0]["guard_started"])
            self.assertTrue(events[-1]["all_observed_owned_instances_verified_absent"])
            self.assertEqual(guard.verified_absences(events, guard.digest(path.read_bytes())), {22: 75.0})
            self.assertTrue(any(t > 75 for t, _ in calls))
            abort = json.loads(path.with_suffix(".abort.json").read_text())
            self.assertEqual(abort["intent_sha256"], guard.digest(path.read_bytes()))
            self.assertFalse((folder / "budget.json").exists())

    def test_full_guard_hands_off_even_if_creator_exited_after_watchdog_start(self):
        with TemporaryDirectory() as temporary:
            folder, clock, value = Path(temporary), Clock(), intent()
            path, receipt = folder / "rental_intent_v1.json", folder / "rental_v1.json"
            path.write_text(json.dumps(value))
            raw, start = receipt_and_start(value)
            receipt.write_bytes(raw)
            receipt.with_suffix(".cleanup.jsonl").write_text(json.dumps(start) + "\n")
            def provider(argv, **kwargs):
                self.assertEqual(argv, ["vastai", "show", "instances", "--raw"])
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps([{"id": 22, "label": value["label"]}]), stderr="")
            self.assertEqual(guard.run_guard(path, receipt, 42, run=provider, now=clock.now,
                sleep=clock.sleep, identity=lambda _: None), 0)
            events = guard.read_jsonl(path.with_suffix(".guard.jsonl"))
            self.assertEqual(events[-1]["event"], "guard_handoff")
            self.assertFalse(path.with_suffix(".abort.json").exists())

    def test_provider_failures_remain_unresolved_without_fake_absence(self):
        with TemporaryDirectory() as temporary:
            folder, clock = Path(temporary), Clock()
            path = folder / "rental_intent_v1.json"
            path.write_text(json.dumps(intent()))
            calls = []
            def provider(argv, **kwargs):
                calls.append(argv)
                return SimpleNamespace(returncode=1, stdout="", stderr="provider down")
            self.assertEqual(guard.run_guard(path, folder / "rental_v1.json", 42,
                run=provider, now=clock.now, sleep=clock.sleep, identity=lambda _: None), 1)
            self.assertGreater(len(calls), 1)
            events = guard.read_jsonl(path.with_suffix(".guard.jsonl"))
            self.assertTrue(events[-1]["manual_cleanup_required"])
            self.assertTrue(events[-1]["creation_liability_unresolved"])
            self.assertFalse(any(e.get("verified_absent") for e in events))

    def test_second_guard_lock_fails_before_provider_call(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "rental_intent_v1.json"
            path.write_text(json.dumps(intent()))
            with path.with_suffix(".guard.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                def provider(*args, **kwargs):
                    self.fail("duplicate guard must not contact provider")
                with self.assertRaises(BlockingIOError):
                    guard.run_guard(path, folder / "rental_v1.json", 42, run=provider,
                                    now=lambda: 0, identity=lambda _: None)


if __name__ == "__main__":
    unittest.main()
