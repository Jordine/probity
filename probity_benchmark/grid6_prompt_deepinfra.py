"""Transparent transport routing for unfinished Qwen prompt events.

The canonical scientific request remains in ``receipt.request`` and is what the
frozen prompt compiler validates. ``transport_request`` differs only by the
explicit DeepInfra route and is retained with its own hash.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from . import grid6_live as live
from .budget import PilotBudget

ROUTE = ["deepinfra/fp8"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def transport_request(canonical):
    body = copy.deepcopy(canonical)
    body["provider"].update(order=ROUTE, allow_fallbacks=False)
    return body


async def execute(out, job, manifest, ledger, session):
    """Drop-in replacement for grid6_prompt.execute for Qwen calls only."""
    from . import grid6_prompt as prompt
    if job["request"]["model"] != "qwen/qwen3.5-27b":
        return await ORIGINAL_EXECUTE(out, job, manifest, ledger, session)
    out = Path(out)
    if prompt.recover(out, job, ledger):
        return 0
    key = job["event_id"]
    intent = out / "intents" / f"{key}.json"
    if intent.exists():
        raise RuntimeError("unknown in-flight prompt event; no automatic retry")
    bound, upper = live.bound_request(job["request"], manifest["model_specs"][job["request"]["model"]])
    if live.stage_committed(out) + upper > 3:
        raise RuntimeError("hard$3 prompt APIcap exhausted")
    budget_key = "grid6-prompt:" + str(out.resolve()) + ":" + key
    if budget_key in live.read(ledger)["entries"]:
        raise RuntimeError("orphan prompt reservation")
    routed = transport_request(job["request"])
    record = {"tag": job, "model": job["request"]["model"],
              "request": job["request"], "request_sha256": digest(job["request"]),
              "transport_request": routed, "transport_request_sha256": digest(routed),
              "provider_route_amendment": {"order": ROUTE, "allow_fallbacks": False},
              "budget_key": budget_key, "reserved_usd": upper,
              "input_token_upper_bound": bound, "valid": None, "attempt": 0,
              "created_at_unix": time.time()}
    PilotBudget(ledger).reserve(budget_key, upper, "api")
    live.durable(intent, record)
    live.durable(out / "jobs" / f"{key}.json", job)
    started = time.monotonic()
    try:
        async with session.post("https://openrouter.ai/api/v1/chat/completions", json=routed) as response:
            record.update(http_status=response.status, response_body=await response.text())
        record["duration_seconds"] = time.monotonic() - started
        live.durable(out / "raw_responses" / f"{key}.json", record)
        record = live.finish_raw(record)
    except Exception as exc:
        record.update(valid=False, error=str(exc), error_type=type(exc).__name__,
                      cost_usd=upper, cost_basis="reservation_upper_bound")
    live.durable(out / "requests" / f"{key}.json", record)
    live.settle_receipt(record, ledger)
    if not record["valid"]:
        raise RuntimeError("routed prompt event failed; raw retained, no retry")
    return 1


async def recover_failed(out, event, ledger, recovery_dir, session):
    """Retry one unconsumed failed Qwen event once on the pinned route."""
    from . import grid6_prompt as prompt
    out, recovery_dir = Path(out), Path(recovery_dir)
    if recovery_dir.exists():
        raise ValueError("new recovery directory required")
    path = out / "requests" / f"{event}.json"
    intent = out / "intents" / f"{event}.json"
    raw = out / "raw_responses" / f"{event}.json"
    old = live.read(path)
    if old.get("valid") is not False or old["request"]["model"] != "qwen/qwen3.5-27b":
        raise ValueError("not a failed Qwen prompt event")
    upper = old["reserved_usd"]
    if live.stage_committed(out) + upper > 3 or live.read(ledger)["cap_usd"] != 60:
        raise ValueError("budget gate failed")
    routed = transport_request(old["request"])
    recovery_dir.mkdir()
    plan = {"schema": "grid6_prompt_deepinfra_recovery_v1", "event_id": event,
            "canonical_request_sha256": old["request_sha256"],
            "transport_request_sha256": digest(routed), "route": ROUTE,
            "allow_fallbacks": False, "attempts": 1,
            "old_receipt_sha256": live.file_hash(path),
            "old_intent_sha256": live.file_hash(intent),
            "old_raw_sha256": live.file_hash(raw) if raw.exists() else None,
            "authorization": "User authorized iterative fixes and a $60 campaign cap."}
    live.durable(recovery_dir / "plan.json", plan)
    attempt = int(old.get("attempt", 0)) + 1
    alias = event + f"_attempt{attempt - 1}_failed.json"
    live.durable(out / "intents" / alias, live.read(intent))
    path.rename(out / "requests" / alias)
    if raw.exists():
        raw.rename(out / "raw_responses" / alias)
    budget_key = old["budget_key"] + f":deepinfra_attempt{attempt}"
    PilotBudget(ledger).reserve(budget_key, upper, "api")
    record = {key: copy.deepcopy(old[key]) for key in
              ["tag", "model", "request", "request_sha256", "reserved_usd", "input_token_upper_bound"]}
    record.update(transport_request=routed, transport_request_sha256=digest(routed),
                  provider_route_amendment={"order": ROUTE, "allow_fallbacks": False},
                  budget_key=budget_key, valid=None, attempt=attempt, created_at_unix=time.time(),
                  recovery_plan_sha256=live.file_hash(recovery_dir / "plan.json"))
    live.durable(intent, record, immutable=False)
    started = time.monotonic()
    try:
        async with session.post("https://openrouter.ai/api/v1/chat/completions", json=routed) as response:
            record.update(http_status=response.status, response_body=await response.text())
        record["duration_seconds"] = time.monotonic() - started
        live.durable(raw, record)
        record = live.finish_raw(record)
    except Exception as exc:
        record.update(valid=False, error=str(exc), error_type=type(exc).__name__,
                      cost_usd=upper, cost_basis="reservation_upper_bound")
    live.durable(path, record)
    live.settle_receipt(record, ledger)
    result = {"at_unix": time.time(), "valid": record["valid"],
              "cost_usd": record["cost_usd"], "attempt": attempt,
              "provider": json.loads(record.get("response_body", "{}")).get("provider"),
              "canonical_request_unchanged": record["request"] == old["request"], "no_retry": True}
    live.durable(recovery_dir / "result.json", result)
    if not record["valid"]:
        raise RuntimeError("routed recovery failed")
    prompt.recover(out, old["tag"], ledger)
    return result


# Set by the coordinator wrapper before execute can be called. Keeping the
# original function avoids routing the Haiku judge through a Qwen endpoint.
ORIGINAL_EXECUTE = None
