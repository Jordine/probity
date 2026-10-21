"""Fixed-deadline iteration-five cleanup, scoped to a verified owned rental.

No budget mutation and no inference. The immutable creation receipt records the
instance, exact task label, pre-existing instances, rate and fixed cost bounds.
The live label is checked before every destroy attempt; unrelated instances are
never modified. Cleanup receipts distinguish a request from verified absence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time


def validate_receipt(receipt):
    instance = receipt.get("instance_id")
    if type(instance) is not int or instance <= 0:
        raise ValueError("positive integer task instance required")
    if not re.fullmatch(r"probity-iteration5-\d{8}(?:-[a-z0-9-]{1,32})?", receipt.get("label", "")):
        raise ValueError("iteration5 task label required")
    previous = receipt.get("existing_instance_ids")
    if not isinstance(previous, list) or any(type(x) is not int for x in previous) or instance in previous:
        raise ValueError("task rental overlaps unrelated-instance snapshot")
    keys = ("created_at_unix", "destroy_at_unix", "max_runtime_hours", "cleanup_allowance_hours",
            "hourly_upper_usd", "traffic_reserve_usd", "reservation_usd")
    if any(type(receipt.get(k)) not in (int, float) or not math.isfinite(receipt[k]) for k in keys):
        raise ValueError("finite creation/deadline/cost receipt required")
    if not 0 < receipt["max_runtime_hours"] <= 9 or not 0 < receipt["cleanup_allowance_hours"] <= .25:
        raise ValueError("iteration5 runtime/cleanup allowance exceeds frozen envelope")
    delta = receipt["destroy_at_unix"] - receipt["created_at_unix"]
    if not 0 < delta <= receipt["max_runtime_hours"] * 3600 + 1e-6:
        raise ValueError("destroy deadline exceeds fixed runtime")
    if not 0 < receipt["hourly_upper_usd"] <= 1.6 or receipt["traffic_reserve_usd"] < 0:
        raise ValueError("rental hourly rate outside iteration5 envelope")
    bound = (receipt["max_runtime_hours"] + receipt["cleanup_allowance_hours"]) * receipt["hourly_upper_usd"] + receipt["traffic_reserve_usd"]
    if not 0 < receipt["reservation_usd"] <= 15 or bound > receipt["reservation_usd"] + 1e-9:
        raise ValueError("rental bound exceeds recorded GPU reservation")
    return receipt


def parse_instances(value):
    if isinstance(value, dict) and isinstance(value.get("instances"), list):
        value = value["instances"]
    if not isinstance(value, list) or any(not isinstance(x, dict) or type(x.get("id")) is not int for x in value):
        raise ValueError("invalid provider instance inventory; absence is not established")
    if len({x["id"] for x in value}) != len(value):
        raise ValueError("duplicate provider instance identity")
    return value


def owned_instance(receipt, instances):
    matches = [x for x in parse_instances(instances) if x["id"] == receipt["instance_id"]]
    if not matches:
        return None
    if matches[0].get("label") != receipt["label"]:
        raise ValueError("live rental label differs; refusing destruction")
    return matches[0]


def destroy_once(receipt, now, run=subprocess.run, timeout=30):
    """One bounded attempt; injectable subprocess runner supports CPU-only tests."""
    validate_receipt(receipt)
    if now < receipt["destroy_at_unix"]:
        raise ValueError("fixed rental deadline has not arrived")
    show = ["vastai", "show", "instances", "--raw"]
    def inventory():
        response = run(show, capture_output=True, text=True, timeout=timeout)
        if response.returncode:
            raise RuntimeError("provider inventory failed; task ownership not verified")
        return json.loads(response.stdout)
    if owned_instance(receipt, inventory()) is None:
        return {"verified_absent": True, "destroy_requested": False}
    response = run(["vastai", "destroy", "instance", str(receipt["instance_id"]), "--yes", "--raw"],
                   capture_output=True, text=True, timeout=timeout)
    result = {"destroy_requested": True, "destroy_returncode": response.returncode,
              "stdout_sha256": hashlib.sha256(response.stdout.encode()).hexdigest(),
              "stderr_sha256": hashlib.sha256(response.stderr.encode()).hexdigest(),
              "verified_absent": False}
    # Success is an independently verified absence, not merely a zero exit code.
    if response.returncode == 0:
        result["verified_absent"] = owned_instance(receipt, inventory()) is None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    receipt = validate_receipt(json.loads(raw))
    digest = hashlib.sha256(raw).hexdigest()
    log_path = args.receipt.with_suffix(".cleanup.jsonl")
    def record(value):
        row = {"at_unix": time.time(), "instance_id": receipt["instance_id"], "receipt_sha256": digest, **value}
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    record({"watchdog_started": True, "destroy_at_unix": receipt["destroy_at_unix"]})
    while time.time() < receipt["destroy_at_unix"]:
        if hashlib.sha256(args.receipt.read_bytes()).hexdigest() != digest:
            raise ValueError("rental receipt changed after watchdog start")
        time.sleep(min(30, max(0, receipt["destroy_at_unix"] - time.time())))
    cleanup_deadline = receipt["destroy_at_unix"] + 3600 * receipt["cleanup_allowance_hours"]
    attempt = 0
    while time.time() < cleanup_deadline:
        if hashlib.sha256(args.receipt.read_bytes()).hexdigest() != digest:
            raise ValueError("rental receipt changed before destruction")
        remaining = cleanup_deadline - time.time()
        try:
            result = destroy_once(receipt, time.time(), timeout=min(30, max(.1, remaining / 3)))
        except ValueError:
            record({"attempt": attempt, "ownership_or_receipt_error": True})
            raise
        except Exception as exc:
            result = {"verified_absent": False, "error_type": type(exc).__name__}
        record({"attempt": attempt, **result})
        if result["verified_absent"]:
            return
        attempt += 1
        time.sleep(min(15, max(0, cleanup_deadline - time.time())))
    record({"cleanup_failed": True, "manual_cleanup_required": True})
    raise RuntimeError("task rental absence not verified within cleanup allowance")


if __name__ == "__main__":
    main()
