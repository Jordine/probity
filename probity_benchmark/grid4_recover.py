"""One explicit identical-request retry for a recorded whitespace-loop failure.

The failed attempt remains immutable and charged. This helper never repairs or
truncates response text, retries successful answers, or changes model settings.
It leaves the frozen generator unchanged and promotes the accepted retry only
after writing a durable technical-recovery journal.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .api import strict_json
from .grid4_generate import (
    FrozenActionClient, atomic_json, disclosure_audit, feedback_messages,
    file_hash, load_run, probe_request, run_lock, status, validate_saved_response,
)
from .preflight import sha256


def paths(out, request_hash):
    root = Path(out) / "recoveries" / request_hash
    return {
        "root": root, "plan": root / "plan.json", "attempt": root / "attempt1",
        "original_receipt": root / "original_failed_receipt.json",
        "original_intent": root / "original_failed_intent.json",
        "receipt": Path(out) / "requests" / (request_hash + ".json"),
        "intent": Path(out) / "intents" / (request_hash + ".json"),
        "archived_receipt": Path(out) / "requests" / (request_hash + "__failed_attempt_0.json"),
        "archived_intent": Path(out) / "intents" / (request_hash + "__failed_attempt_0.json"),
        "promotion": root / "promotion.json", "completed": root / "promotion_completed.json",
        "retry_receipt": Path(out) / "requests" / (request_hash + "__recovery_attempt_1.json"),
        "retry_intent": Path(out) / "intents" / (request_hash + "__recovery_attempt_1.json"),
    }


def loop_diagnostic(receipt):
    payload = strict_json(receipt["response_body"])
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    runs = list(re.finditer(r"([\r\n\t ])\1*", content))
    longest = max(runs, key=lambda match: len(match.group()), default=None)
    whitespace = sum(c.isspace() for c in content)
    diagnostic = {
        "content_characters": len(content), "whitespace_characters": whitespace,
        "nonwhitespace_characters": len(content) - whitespace,
        "longest_identical_whitespace_run": 0 if longest is None else len(longest.group()),
        "longest_run_character_codepoint": None if longest is None else ord(longest.group()[0]),
        "longest_run_start": None if longest is None else longest.start(),
        "reasoning_characters": len(choice["message"].get("reasoning") or ""),
        "completion_tokens": payload["usage"]["completion_tokens"],
        "provider": payload.get("provider"), "system_fingerprint": payload.get("system_fingerprint"),
        "response_body_sha256": sha256(receipt["response_body"]),
    }
    if (receipt.get("valid") is not False or receipt.get("http_status") != 200
            or receipt.get("finish_reason") != "length"
            or choice.get("finish_reason") != "length"
            or not content or whitespace / len(content) < .9
            or diagnostic["longest_identical_whitespace_run"] < 1024):
        raise ValueError("not the explicitly supported technical whitespace-loop failure")
    return diagnostic


def committed(out):
    total = 0.0
    for path in (Path(out) / "intents").glob("*.json"):
        intent = strict_json(path.read_text())
        receipt_path = Path(out) / "requests" / path.name
        total += (strict_json(receipt_path.read_text()).get("cost_usd", intent["reserved_usd"])
                  if receipt_path.exists() else intent["reserved_usd"])
    return total


def retry_liability(p):
    if not p["retry_intent"].exists():
        return 0.0
    intent = strict_json(p["retry_intent"].read_text())
    return (strict_json(p["retry_receipt"].read_text()).get("cost_usd", intent["reserved_usd"])
            if p["retry_receipt"].exists() else intent["reserved_usd"])


def retain_retry_accounting(p, request_hash):
    """Expose completed or ambiguous retry liability in the main run ceiling."""
    for name in ("receipt", "intent"):
        source = p["attempt"] / ("requests" if name == "receipt" else "intents") / (request_hash + ".json")
        if source.exists():
            atomic_json(p["retry_" + name], strict_json(source.read_text()))


def prepare(out, request_hash):
    p = paths(out, request_hash)
    if p["plan"].exists():
        return strict_json(p["plan"].read_text())
    failed = strict_json(p["receipt"].read_text())
    intent = strict_json(p["intent"].read_text())
    diagnostic = loop_diagnostic(failed)
    if failed["request_sha256"] != request_hash or intent["request_sha256"] != request_hash:
        raise ValueError("failed action identity mismatch")
    if sha256(json.dumps(failed["request"], sort_keys=True)) != request_hash:
        raise ValueError("failed full request hash mismatch")
    manifest = strict_json((Path(out) / "manifest.json").read_text())
    original_bytes = {"receipt": p["receipt"].read_bytes(), "intent": p["intent"].read_bytes()}
    # Preserve the original JSON object and exact source-file digests. Existing
    # files already use this atomic serializer, so the archived bytes also match.
    atomic_json(p["original_receipt"], failed, immutable=True)
    atomic_json(p["original_intent"], intent, immutable=True)
    if (p["original_receipt"].read_bytes() != original_bytes["receipt"]
            or p["original_intent"].read_bytes() != original_bytes["intent"]):
        raise ValueError("archiving changed original receipt or intent bytes")
    plan = {
        "schema": "grid4_identical_technical_retry_v1", "request_sha256": request_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_id": failed["tag"]["episode"], "turn_index": failed["tag"]["turn_index"],
        "failed_receipt_sha256": file_hash(p["original_receipt"]),
        "failed_intent_sha256": file_hash(p["original_intent"]),
        "failed_cost_usd": failed["cost_usd"], "diagnostic": diagnostic,
        "manifest_sha256": file_hash(Path(out) / "manifest.json"),
        "frozen_generator_sha256": manifest["source_code_sha256"],
        "recovery_source_sha256": file_hash(__file__), "allowed_new_attempts": 1,
        "request_change": "none; full request byte-equivalent under canonical JSON, including max_tokens8192",
        "selection_rule": "Technical finish/whitespace failure only. First complete schema-valid response retained regardless of assigned answer, prose quality, or probe/judge scores.",
        "authorization": "Coordinator explicitly approved one identical-body technical retry before any replacement result was observed.",
        "accounting": "Original failed attempt and recovery attempt both remain in cumulative generation ceiling and global ledger; archived failed intent/receipt also remain in top-level accounting.",
    }
    atomic_json(p["plan"], plan, immutable=True)
    for name in ("requests", "intents"):
        (p["attempt"] / name).mkdir(parents=True, exist_ok=True)
    return plan


def validate_plan(out, request_hash):
    p = paths(out, request_hash)
    plan = strict_json(p["plan"].read_text())
    if file_hash(__file__) != plan["recovery_source_sha256"]:
        raise ValueError("recovery source changed after explicit retry freeze")
    if file_hash(Path(out) / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("generation manifest changed after retry freeze")
    for key, hash_key in (("original_receipt", "failed_receipt_sha256"),
                          ("original_intent", "failed_intent_sha256")):
        if file_hash(p[key]) != plan[hash_key]:
            raise ValueError("immutable failed evidence changed")
    return p, plan


def promote_receipt(out, request_hash, accepted, accepted_intent):
    """Idempotent promotion; every intermediate state conservatively accounts cost."""
    p, plan = validate_plan(out, request_hash)
    failed = strict_json(p["original_receipt"].read_text())
    failed_intent = strict_json(p["original_intent"].read_text())
    validate_saved_response(accepted)
    if (accepted["request_sha256"] != request_hash or accepted["request"] != failed["request"]
            or accepted_intent["request_sha256"] != request_hash
            or accepted["tag"] != failed["tag"]
            or accepted_intent["budget_key"] != accepted["budget_key"]
            or accepted_intent["reserved_usd"] != accepted["reserved_usd"]):
        raise ValueError("technical retry changed original request")
    if accepted["budget_key"] == failed["budget_key"]:
        raise ValueError("retry requires an independently accounted new attempt")
    promotion = {
        "schema": "grid4_receipt_promotion_v1", "request_sha256": request_hash,
        "plan_sha256": file_hash(p["plan"]),
        "failed_receipt_sha256": plan["failed_receipt_sha256"],
        "accepted_receipt_sha256": sha256(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n"),
        "failed_cost_usd": failed["cost_usd"], "accepted_cost_usd": accepted["cost_usd"],
        "combined_attempt_cost_usd": failed["cost_usd"] + accepted["cost_usd"],
        "failed_budget_key": failed["budget_key"], "accepted_budget_key": accepted["budget_key"],
        "source_request_identical": True,
        "archived_receipt": str(p["archived_receipt"]), "accepted_receipt": str(p["receipt"]),
    }
    atomic_json(p["promotion"], promotion, immutable=True)
    current = strict_json(p["receipt"].read_text())
    if current != failed and current != accepted:
        raise ValueError("canonical receipt is neither the original failure nor planned accepted attempt")
    current_intent = strict_json(p["intent"].read_text())
    if current_intent != failed_intent and current_intent != accepted_intent:
        raise ValueError("canonical intent changed outside the planned promotion")
    # Archive liability first. A crash here temporarily overcounts old cost,
    # never loses it; rerunning promotion finishes without another paid call.
    atomic_json(p["archived_receipt"], failed, immutable=True)
    atomic_json(p["archived_intent"], failed_intent, immutable=True)
    atomic_json(p["receipt"], accepted)
    atomic_json(p["intent"], accepted_intent)
    # The accepted attempt now occupies the canonical slot. Move its temporary
    # accounting aliases out of the top-level total, retaining their full data.
    # Removing receipt first leaves a reservation overcount if interrupted.
    for name in ("receipt", "intent"):
        alias = p["retry_" + name]
        if alias.exists():
            saved = p["root"] / ("accounting_" + name + ".json")
            value = strict_json(alias.read_text())
            atomic_json(saved, value, immutable=True)
            if file_hash(alias) != file_hash(saved):
                raise ValueError("retry accounting archive mismatch")
            alias.unlink()
    atomic_json(p["completed"], {
        **promotion, "canonical_receipt_sha256": file_hash(p["receipt"]),
        "top_level_combined_attempt_cost_usd": (
            strict_json(p["receipt"].read_text())["cost_usd"]
            + strict_json(p["archived_receipt"].read_text())["cost_usd"]),
    }, immutable=True)
    return promotion


def publish_turn(out, manifest, episode, receipt):
    index = receipt["tag"]["turn_index"]
    if len(episode["turns"]) == index:
        step = episode["turn_schedule"][index]
        turn = {**step, **receipt["parsed"],
                "role_adherent": receipt["parsed"]["answer"] == step["assigned_answer"],
                "advocates_gold": receipt["parsed"]["answer"] == episode["gold"],
                "request_sha256": receipt["request_sha256"], "visible_prior_turns": index,
                "feedback_prior_request_sha256": [t["request_sha256"] for t in episode["turns"]],
                "disclosure_audit": disclosure_audit(receipt["parsed"]["argument"]),
                "technical_recovery": receipt["technical_recovery"]}
        episode["turns"].append(turn)
        atomic_json(Path(out) / f"episode_{episode['episode_id']}.json", episode)
    else:
        turn = episode["turns"][index]
        if (turn["request_sha256"] != receipt["request_sha256"]
                or turn["argument"] != receipt["parsed"]["argument"]
                or turn["answer"] != receipt["parsed"]["answer"]):
            raise ValueError("existing turn differs from accepted technical retry")
    request = probe_request(out, manifest, episode, index, receipt)
    atomic_json(Path(out) / "queue" / "requests" / (request["job_id"] + ".json"), request, immutable=True)
    return request["job_id"]


async def run(args):
    p, plan = validate_plan(args.out, args.failed_request)
    manifest, problems, episodes = load_run(args.out)
    episode = next(e for e in episodes if e["episode_id"] == plan["episode_id"])
    failed = strict_json(p["original_receipt"].read_text())
    if not math.isfinite(args.budget) or args.budget <= 0:
        raise ValueError("finite positive cumulative generation budget required")
    if not p["completed"].exists():
        if len(episode["turns"]) != plan["turn_index"]:
            raise ValueError("failed turn is no longer the next action")
        messages = feedback_messages(problems[episode["problem_id"]], episode)
        if messages != failed["request"]["messages"]:
            raise ValueError("retry causal input differs from original failed request")
        if p["promotion"].exists():
            # A prior interrupted promotion already has a paid, complete result.
            # Do not invoke the client or recompute a reduced retry allowance.
            receipt = strict_json((p["attempt"] / "requests" / (args.failed_request + ".json")).read_text())
        else:
            available = args.budget - (committed(args.out) - retry_liability(p))
            if available <= 0:
                raise ValueError("cumulative ceiling has no recovery allowance")
            if not p["retry_intent"].exists():
                # Write the main-run reservation before the helper can call the
                # provider. A crash cannot hide this attempt from later waves.
                if failed["reserved_usd"] > available:
                    raise ValueError("cumulative ceiling cannot reserve identical technical retry")
                atomic_json(p["retry_intent"], {
                    "request_sha256": args.failed_request,
                    "reserved_usd": failed["reserved_usd"],
                    "budget_key": "pending-grid4-recovery:" + str(p["attempt"].resolve()),
                    "tag": failed["tag"], "plan_sha256": file_hash(p["plan"]),
                }, immutable=True)
            try:
                async with FrozenActionClient(p["attempt"], manifest, available, 1,
                                              args.timeout, args.ledger) as client:
                    receipt = await client.complete(episode, messages)
            finally:
                # Also retain a second failure or unresolved intent. Neither
                # permits a third request or disappears from budget accounting.
                retain_retry_accounting(p, args.failed_request)
            receipt["attempt"] = 1
            receipt["technical_recovery"] = {
                "plan_sha256": file_hash(p["plan"]),
                "original_failed_receipt_sha256": plan["failed_receipt_sha256"],
                "original_failed_cost_usd": plan["failed_cost_usd"],
                "identical_request": True, "selection_basis": "technical_whitespace_loop_only",
            }
            atomic_json(p["attempt"] / "requests" / (args.failed_request + ".json"), receipt)
            retain_retry_accounting(p, args.failed_request)
        intent = strict_json((p["attempt"] / "intents" / (args.failed_request + ".json")).read_text())
        promote_receipt(args.out, args.failed_request, receipt, intent)
    else:
        receipt = strict_json(p["receipt"].read_text())
    job_id = publish_turn(args.out, manifest, episode, receipt)
    result = status(args.out, manifest, episodes)
    result.update(recovered_episode_id=episode["episode_id"], score_job_id=job_id,
                  technical_retry_count=1, combined_attempt_cost_usd=(
                      plan["failed_cost_usd"] + receipt["cost_usd"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--failed-request", required=True)
    parser.add_argument("--budget", type=float, default=.85)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    with run_lock(args.out):
        if args.command == "prepare":
            result = prepare(args.out, args.failed_request)
        else:
            if args.ledger is None:
                raise ValueError("global ledger required for explicit technical retry")
            result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
