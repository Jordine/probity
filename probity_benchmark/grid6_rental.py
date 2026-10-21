"""Single-attempt Vast rental with pre-create deadline guard and shared ledger.

Only the new exact campaign label may be destroyed. The guard is a detached
process; its deadline cannot be extended by this module. No model work here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .budget import PilotBudget

ROOT = Path(__file__).resolve().parents[1]
SDK = "/root/.local/share/pipx/venvs/vastai/bin/python"
IMAGE = "pytorch/pytorch@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def inventory():
    result = subprocess.run(["vastai", "show", "instances-v1", "--all", "--raw"],
                            capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(result.stdout)
    rows = result.get("instances")
    if (result.get("success") is not True or result.get("next_token") is not None
            or not isinstance(rows, list) or result.get("total_instances") != len(rows)
            or any(type(r.get("id")) is not int for r in rows)
            or len({r["id"] for r in rows}) != len(rows)):
        raise ValueError("unverified provider inventory")
    return rows


def owned(intent, rows):
    previous = set(intent["existing_instance_ids"])
    matches = [r for r in rows if r.get("label") == intent["label"]]
    if any(r["id"] in previous for r in matches):
        raise ValueError("campaign label overlaps a preexisting instance")
    return matches


def cost_bound(rate, hours, traffic):
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (rate, hours, traffic)) or not 0 < rate <= 1.8 or not 0 < hours <= 8:
        raise ValueError("invalid rental cost/runtime bound")
    return rate * (hours + 0.1) + traffic


def validate(intent):
    if not intent["label"].startswith("probity-iteration6-"):
        raise ValueError("wrong campaign label")
    if (type(intent["reservation_usd"]) not in (int, float)
            or not math.isfinite(intent["reservation_usd"])
            or not 0 < intent["reservation_usd"] <= 25):
        raise ValueError("invalid campaign GPU reservation")
    if cost_bound(intent["hourly_upper_usd"], intent["max_runtime_hours"],
                  intent["traffic_reserve_usd"]) > intent["reservation_usd"]:
        raise ValueError("reservation does not cover fixed deadline")
    if abs(intent["destroy_at_unix"] - intent["created_at_unix"]
           - intent["max_runtime_hours"] * 3600) > 1e-5:
        raise ValueError("deadline changed")


def guard(intent_path):
    raw = intent_path.read_bytes()
    intent = json.loads(raw)
    validate(intent)
    directory = intent_path.parent
    digest = hashlib.sha256(raw).hexdigest()
    save(directory / "guard_ready.json", {"pid": os.getpid(), "intent_sha256": digest,
                                         "at_unix": time.time()})
    while True:
        try:
            if intent_path.read_bytes() != raw:
                reason = "intent_mutated"
            elif (directory / "stop_requested.json").exists():
                reason = "coordinator_requested_cleanup"
            elif time.time() >= intent["destroy_at_unix"]:
                reason = "fixed_deadline"
            elif not (directory / "rental.json").exists() and time.time() > intent["created_at_unix"] + 900:
                reason = "creation_unresolved_15_minutes"
            else:
                reason = None
            rows = inventory()
            matches = owned(intent, rows)
            if len(matches) > 1:
                reason = "duplicate_owned_instances"
            if reason:
                for row in matches:
                    result = subprocess.run(["vastai", "destroy", "instance", str(row["id"]),
                                             "--yes", "--raw"], capture_output=True, timeout=60)
                    with (directory / "guard_events.jsonl").open("a") as f:
                        f.write(json.dumps({"at_unix": time.time(), "reason": reason,
                                            "destroy_id": row["id"], "returncode": result.returncode}) + "\n")
                if not owned(intent, inventory()):
                    save(directory / "absence.json", {"verified_absent": True,
                         "at_unix": time.time(), "reason": reason, "intent_sha256": digest})
                    return
            save(directory / "guard_status.json", {"at_unix": time.time(), "owned_ids": [r["id"] for r in matches],
                                                    "cleanup_reason": reason, "deadline_unix": intent["destroy_at_unix"]})
        except Exception as exc:
            with (directory / "guard_events.jsonl").open("a") as f:
                f.write(json.dumps({"at_unix": time.time(), "error_type": type(exc).__name__}) + "\n")
        time.sleep(20)


def rent(args):
    if args.out.exists():
        raise ValueError("new rental directory required; existing intents must be reconciled")
    ready = json.loads(args.ready.read_text())
    if ready.get("ready_for_rental") is not True:
        raise ValueError("reviewed bootstrap/scorer readiness required")
    for path, expected in ready["files_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("rental readiness changed: " + path)
    snapshot = json.loads(args.offers.read_text())
    if time.time() - snapshot["at_unix"] > 1800:
        raise ValueError("offer snapshot older than 30 minutes")
    offer = next(o for o in snapshot["offers"] if o["id"] == args.offer_id)
    rate = float(offer["dph_total"])
    traffic = max(.5, 150 * float(offer["inet_down_cost"]) + 5 * float(offer["inet_up_cost"]))
    if (offer["num_gpus"] != 1 or offer["gpu_ram"] < 90000
            or offer.get("reliability2", 0) < .98 or offer.get("cuda_max_good", 0) < 12.8
            or offer.get("disk_space", 0) < 160
            or offer["gpu_name"] != "RTX PRO 6000 WS" or offer["duration"] < (args.hours + .5) * 3600
            or cost_bound(rate, args.hours, traffic) > args.reservation):
        raise ValueError("offer does not fit intended GPU/runtime/cost")
    before = inventory()
    if any(r.get("label") == args.label for r in before):
        raise ValueError("label already exists; never duplicate rental")
    args.out.mkdir(parents=True)
    save(args.out / "instances_before.json", before)
    key = "gpu:" + str(args.out.resolve())
    PilotBudget(args.ledger).reserve(key, args.reservation, "gpu_rental")
    now = time.time()
    intent = {"schema": "iteration6_rental_v1", "created_at_unix": now,
              "destroy_at_unix": now + args.hours * 3600, "max_runtime_hours": args.hours,
              "hourly_upper_usd": rate, "traffic_reserve_usd": traffic,
              "reservation_usd": args.reservation, "offer_id": args.offer_id,
              "label": args.label, "existing_instance_ids": [r["id"] for r in before],
              "budget_key": key, "ledger": str(args.ledger.resolve()), "disk_gb": 160,
              "image": IMAGE, "ready_sha256": hashlib.sha256(args.ready.read_bytes()).hexdigest()}
    validate(intent)
    intent_path = args.out / "intent.json"
    save(intent_path, intent)
    with (args.out / "guard.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "probity_benchmark.grid6_rental",
                                    "guard", "--intent", str(intent_path.resolve())], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if (args.out / "guard_ready.json").exists():
            break
        if process.poll() is not None:
            break
        time.sleep(.5)
    guard_ready = args.out / "guard_ready.json"
    if (not guard_ready.exists() or process.poll() is not None
            or json.loads(guard_ready.read_text()).get("intent_sha256")
            != hashlib.sha256(intent_path.read_bytes()).hexdigest()):
        PilotBudget(args.ledger).settle(key, 0, "No provider create was attempted; guard did not start.")
        raise RuntimeError("guard did not start")
    # A single SDK attempt. Parse only sanitized fields; credentials never leave SDK.
    helper = """import json,sys
from vastai import VastAI
config=json.load(sys.stdin)
try:
 r=VastAI(retry=1,raw=True,quiet=True).create_instance(config['offer'], image=config['image'], disk=160, label=config['label'], cancel_unavail=True, runtype='ssh_direc ssh_proxy')
 print(json.dumps({'success':bool(r.get('success')),'new_contract':r.get('new_contract')}))
except Exception as e:
 print(json.dumps({'success':False,'error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}))
"""
    save(args.out / "create_started.json", {"at_unix": time.time(), "single_attempt": True})
    try:
        result = subprocess.run([SDK, "-c", helper], input=json.dumps({"offer": args.offer_id,
                                  "image": IMAGE, "label": args.label}),
                                capture_output=True, text=True, timeout=125)
        response = json.loads(result.stdout)
    except Exception as exc:
        response = {"success": False, "unresolved": True, "error_type": type(exc).__name__}
    save(args.out / "create_response.json", response)
    hint = response.get("new_contract")
    for _ in range(12):
        matches = owned(intent, inventory())
        if len(matches) > 1:
            save(args.out / "stop_requested.json", {"reason": "multiple contracts", "at_unix": time.time()})
            raise RuntimeError("multiple new contracts; guard cleaning")
        if matches:
            instance = matches[0]
            if hint is not None and instance["id"] != int(hint):
                raise RuntimeError("contract hint differs from inventory")
            actual_rate = instance.get("dph_total")
            if (instance.get("gpu_ram", 0) < 90000 or instance.get("num_gpus") != 1
                    or actual_rate is None or float(actual_rate) > rate + 1e-6):
                save(args.out / "stop_requested.json", {"reason": "instance shape/rate mismatch", "at_unix": time.time()})
                raise RuntimeError("rental differs from reserved offer; cleaning")
            save(args.out / "rental.json", {**intent, "instance_id": instance["id"],
                                            "instance": instance})
            print(json.dumps({"instance_id": instance["id"], "hourly_usd": rate,
                              "reserved_usd": args.reservation, "deadline": intent["destroy_at_unix"]}))
            return
        time.sleep(5)
    raise RuntimeError("creation unresolved; reservation and independent guard retained")


def release(directory):
    if (directory / "settlement.json").exists():
        print((directory / "settlement.json").read_text())
        return
    intent = json.loads((directory / "intent.json").read_text())
    validate(intent)
    save(directory / "stop_requested.json", {"at_unix": time.time(), "reason": "work complete"})
    for row in owned(intent, inventory()):
        subprocess.run(["vastai", "destroy", "instance", str(row["id"]), "--yes", "--raw"],
                       capture_output=True, timeout=60, check=True)
    for _ in range(12):
        if not owned(intent, inventory()):
            break
        time.sleep(3)
    else:
        raise RuntimeError("rental absence not verified; reservation retained")
    now = time.time()
    # The independent guard owns the fixed deadline. If reconciliation happens
    # later, do not charge the bookkeeping delay as GPU runtime after the
    # immutable destruction deadline. Absence is still verified live above.
    billed_until = min(now, intent["destroy_at_unix"])
    elapsed_hours = (billed_until - intent["created_at_unix"]) / 3600
    # Conservative: reserve all traffic and the full bounded rental window.
    amount = elapsed_hours * intent["hourly_upper_usd"] + intent["traffic_reserve_usd"]
    if not (directory / "rental.json").exists():
        response_path = directory / "create_response.json"
        response = json.loads(response_path.read_text()) if response_path.exists() else {}
        # A provider-declared failed create plus a live, exact-label absence is a
        # resolved zero-cost attempt. Timeouts and malformed/missing responses
        # remain ambiguous and retain the full liability bound.
        if response.get("success") is False and response.get("unresolved") is not True:
            amount = 0
        else:
            amount = intent["reservation_usd"]  # ambiguous create: retain full liability
    ledger = PilotBudget(intent["ledger"])
    current = json.loads(Path(intent["ledger"]).read_text())["entries"][intent["budget_key"]]
    if current["status"] == "reserved":
        ledger.settle(intent["budget_key"], amount, "Verified owned-instance absence; elapsed billing upper bound plus full traffic reserve.")
    save(directory / "settlement.json", {"verified_absent": True, "at_unix": now,
                                         "amount_usd": amount, "elapsed_hours": elapsed_hours})
    print(json.dumps({"verified_absent": True, "accounted_usd": amount}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rent_p = sub.add_parser("rent")
    for name in ("out", "ledger", "offers", "ready"):
        rent_p.add_argument("--" + name, type=Path, required=True)
    rent_p.add_argument("--offer-id", type=int, required=True)
    rent_p.add_argument("--label", required=True)
    rent_p.add_argument("--hours", type=float, default=6)
    rent_p.add_argument("--reservation", type=float, default=12)
    guard_p = sub.add_parser("guard")
    guard_p.add_argument("--intent", type=Path, required=True)
    release_p = sub.add_parser("release")
    release_p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "rent":
        rent(args)
    elif args.command == "guard":
        guard(args.intent)
    else:
        release(args.out)


if __name__ == "__main__":
    main()
