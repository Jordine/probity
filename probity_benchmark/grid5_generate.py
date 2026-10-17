"""Dataset-only extension of the frozen six-turn generation protocol.

Knowledge mistakes and public role deviations are recorded outcomes. Technical
completion, request identity, source, preparation, and causal-feedback checks
remain mandatory. The original modules are imported without modification.
"""

import argparse
import asyncio
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import jsonschema

from .api import ANSWER, BudgetedChat, assert_no_exposed_truncation, object_schema, strict_json
from .environment3_evidence import evidence_messages
from .environment3_packets import judge_messages, load_external_preparations
from .environment3_prepared import preparation_messages, preparation_schema
from . import grid4_generate as engine
from .preflight import load_problems, sha256


PROTOCOL = "explicit_reasoned_evidence_sequential"
MODELS = ("qwen/qwen3.5-27b", "qwen/qwen3.8-27b")
JUDGE = "anthropic/claude-haiku-5.5"
PUBLIC_SCHEMA = object_schema({"argument": {"type": "string", "minLength": 1}, "answer": ANSWER})
KNOWLEDGE_SCHEMA = object_schema({"answer": ANSWER})
DIAGNOSTIC_JUDGE_SCHEMA = object_schema({"answer": ANSWER, "explanation": {"type": "string"}})


def file_hash(path):
    return engine.file_hash(path)


def read_rows(path):
    return [strict_json(line) for line in Path(path).read_text().splitlines() if line]


def frozen_inventory(source, inventory_path, expected_hash):
    if file_hash(inventory_path) != expected_hash:
        raise ValueError("inventory hash mismatch")
    inventory = strict_json(Path(inventory_path).read_text())
    if inventory.get("status") not in ("frozen", "frozen_after_source_review"):
        raise ValueError("source inventory must be explicitly frozen")
    declared = inventory["source_outputs"]["new142"]
    if file_hash(source) != declared["sha256"]:
        raise ValueError("new-source content differs from frozen inventory")
    problems, audit = load_problems(Path(source))
    rows = read_rows(source)
    if audit["errors"] or audit["duplicate_problem_hashes"] or len(rows) != len(problems):
        raise ValueError("invalid complete source corpus")
    selected = [r for r in inventory["rows"] if not r["prior24"]]
    by_problem = {r["problem_id"]: r for r in selected}
    if len(by_problem) != len(selected) or set(by_problem) != {p.problem_id for p in problems}:
        raise ValueError("new-source inclusion differs from frozen inventory")
    for problem, row in zip(problems, rows):
        planned = by_problem[problem.problem_id]
        for field in ("parent_problem_id", "parent_source_line", "family_id", "split", "schedule_parent_index"):
            if row[field] != planned[field]:
                raise ValueError("source metadata differs from inventory: " + field)
        if row["split"] != "dev" or row.get("campaign_role") != "exploratory_dataset_extension":
            raise ValueError("new rows must explicitly be exploratory extension inputs")
        if sha256(problem.story) != planned["source_sha256"] or sha256(problem.question) != planned["question_sha256"]:
            raise ValueError("source/question identity differs from inventory")
    return problems, rows, audit, by_problem


def schedule(row):
    index = row["schedule_parent_index"]
    order = ["Red", "Blue"] if (index ^ (index // 2)) % 2 == 0 else ["Blue", "Red"]
    return bool(index % 2), [
        {"speaker": speaker, "round": round_number, "assigned_answer": "A" if speaker == "Red" else "B"}
        for round_number in range(1, 4) for speaker in order
    ]


def audit_receipt(receipt):
    """Independently validate an accepted receipt without changing any field."""
    body = receipt["request"]
    if sha256(json.dumps(body, sort_keys=True)) != receipt["request_sha256"]:
        raise ValueError("receipt request hash mismatch")
    payload = strict_json(receipt["response_body"])
    if receipt.get("valid") is not True or receipt.get("http_status") != 200 or "error" in payload:
        raise ValueError("receipt is not an accepted provider response")
    if len(payload["choices"]) != 1:
        raise ValueError("expected one complete choice")
    choice, usage = payload["choices"][0], payload["usage"]
    if choice.get("finish_reason") != "stop" or receipt.get("finish_reason") != "stop":
        raise ValueError("incomplete generation")
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens")):
        raise ValueError("invalid provider token counts")
    if usage["completion_tokens"] >= body["max_tokens"] or receipt["usage"] != usage:
        raise ValueError("capped generation or mismatched usage")
    assert_no_exposed_truncation(payload)
    parsed = strict_json(choice["message"]["content"])
    jsonschema.validate(parsed, body["response_format"]["json_schema"]["schema"])
    if parsed != receipt["parsed"] or receipt["model"] != body["model"]:
        raise ValueError("parsed response or request model mismatch")
    return receipt


def external_preparations(problems, rows, path):
    external = load_external_preparations(Path(path))
    expected = {(p.problem_id, model, speaker) for p in problems for model in MODELS for speaker in ("Red", "Blue")}
    if set(external) != expected:
        raise ValueError("external preparations do not exactly cover requested episodes")
    for problem, row in zip(problems, rows):
        swapped, _ = schedule(row)
        for model in MODELS:
            for speaker in ("Red", "Blue"):
                record, receipt = external[(problem.problem_id, model, speaker)]
                audit_receipt(receipt)
                assigned = "A" if speaker == "Red" else "B"
                if (receipt["request"]["messages"] != preparation_messages(problem, swapped, speaker, assigned, explicit_truth=True)
                        or record["swapped"] != swapped or record["assigned_answer"] != assigned
                        or record["source_story_sha256"] != sha256(problem.story)):
                    raise ValueError("external private preparation input mismatch")
                if (receipt["model"] != model or receipt["request"]["max_tokens"] != 16384
                        or receipt["request"].get("temperature") != 0
                        or receipt["request"]["reasoning"] != {"enabled": True}
                        or receipt["request"]["response_format"]["json_schema"]["schema"] != preparation_schema()
                        or receipt["tag"] != {"problem_id": problem.problem_id, "model": model,
                                              "speaker": speaker, "stage": "reasoned_preparation"}):
                    raise ValueError("external preparation model/settings/tag mismatch")
    return external


def expected_api_body(api, model, messages, schema):
    """Mirror the unchanged BudgetedChat request builder for pre-call auditing."""
    spec = api.models[model]
    price = {k: float(spec["pricing"].get(k, 0)) for k in ("prompt", "completion", "request")}
    body = {"model": model, "messages": messages, "max_tokens": api.max_tokens,
            "reasoning": api.reasoning_by_model.get(model, api.reasoning), "usage": {"include": True},
            "provider": {"require_parameters": True, "max_price": {
                "prompt": price["prompt"] * 1e6, "completion": price["completion"] * 1e6, "request": price["request"]}},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "diagnostic", "strict": True, "schema": schema}}}
    if "temperature" in spec["supported_parameters"]:
        body["temperature"] = 0
    return body


def reference_templates(path):
    result = {}
    for receipt in read_rows(path):
        if not receipt.get("valid"):
            continue
        stage = receipt["tag"].get("stage")
        if stage not in ("knowledge", "debate", "judge"):
            continue
        audit_receipt(receipt)
        key = (receipt["model"], stage)
        template = {k: v for k, v in receipt["request"].items() if k != "messages"}
        if key in result and result[key] != template:
            raise ValueError("historical request settings vary within a stage/model")
        result[key] = template
    expected = {(m, s) for m in MODELS for s in ("knowledge", "debate")} | {(JUDGE, "judge")}
    if set(result) != expected:
        raise ValueError("historical request reference lacks exact required stage/model coverage")
    return result


def audit_api_templates(api, templates):
    for (model, stage), template in templates.items():
        schema = {"knowledge": KNOWLEDGE_SCHEMA, "debate": PUBLIC_SCHEMA, "judge": DIAGNOSTIC_JUDGE_SCHEMA}[stage]
        body = expected_api_body(api, model, [], schema)
        body.pop("messages")
        if body != template:
            raise ValueError("API settings differ from frozen historical contract: " + model + "/" + stage)


async def baseline_episode(api, problem, row, model, external, out):
    """Exact old requests; retain knowledge mistakes and every actual public answer."""
    swapped, turns_schedule = schedule(row)
    _, gold = problem.view(False, swapped)
    eid = sha256(problem.problem_id + model + PROTOCOL + "3")
    context, _ = problem.view(True, swapped, structured=True)
    knowledge = audit_receipt(await api.complete(model, context, KNOWLEDGE_SCHEMA,
                                                {"episode": eid, "stage": "knowledge"}))
    preparations = {}
    for step in turns_schedule[:2]:
        speaker = step["speaker"]
        _, receipt = external[(problem.problem_id, model, speaker)]
        preparations[speaker] = {**receipt["parsed"], "request_sha256": receipt["request_sha256"]}
    episode = {
        "episode_id": eid, "problem_id": problem.problem_id, "source_line": problem.source_line,
        "parent_problem_id": row["parent_problem_id"], "parent_source_line": row["parent_source_line"],
        "split": row["split"], "world": row.get("variant", "original"), "protocol": PROTOCOL,
        "model": model, "gold": gold, "swapped": swapped, "knowledge": knowledge["parsed"]["answer"],
        "knowledge_correct": knowledge["parsed"]["answer"] == gold, "alternate_problem_id": None,
        "private_preparations": preparations, "turns": [], "complete": False,
    }
    path = Path(out) / f"episode_{eid}.json"
    engine.atomic_json(path, episode)
    for step in turns_schedule:
        speaker = step["speaker"]
        messages = evidence_messages(problem, swapped, speaker, step["assigned_answer"], episode["turns"],
            json.dumps({k: v for k, v in preparations[speaker].items() if k != "request_sha256"}, ensure_ascii=False),
            explicit_truth=True)
        receipt = audit_receipt(await api.complete(model, messages, PUBLIC_SCHEMA, {
            "episode": eid, "stage": "debate", "round": step["round"], "speaker": speaker}))
        parsed = receipt["parsed"]
        episode["turns"].append({"speaker": speaker, "round": step["round"], **parsed,
            "assigned_answer": step["assigned_answer"], "role_adherent": parsed["answer"] == step["assigned_answer"],
            "advocates_gold": parsed["answer"] == gold, "request_sha256": receipt["request_sha256"],
            "visible_prior_turns": len(episode["turns"])})
        engine.atomic_json(path, episode)
    vote = audit_receipt(await api.complete(JUDGE, judge_messages(problem, swapped, episode["turns"]),
                                           DIAGNOSTIC_JUDGE_SCHEMA, {"episode": eid, "stage": "judge"}))
    episode.update(judge=vote["parsed"], judge_correct=vote["parsed"]["answer"] == gold, complete=True)
    engine.atomic_json(path, episode)
    return episode


async def run_baseline(args):
    problems, rows, audit, planned = frozen_inventory(args.source, args.inventory, args.inventory_sha256)
    external = external_preparations(problems, rows, args.prepared_records)
    templates = reference_templates(args.request_reference)
    async with BudgetedChat(args.out, list(MODELS) + [JUDGE], args.budget, max_tokens=8192,
                            reasoning={"enabled": False}, schema_retries=2, concurrency=args.concurrency,
                            request_timeout_seconds=300, reuse_receipts=args.reuse_receipts) as api:
        audit_api_templates(api, templates)
        manifest = {
            "schema": "grid5_no_feedback_generation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL, "rounds": 3, "dataset_audit": audit,
            "source_sha256": file_hash(__file__), "source": str(Path(args.source).resolve()),
            "inventory": str(Path(args.inventory).resolve()), "inventory_sha256": args.inventory_sha256,
            "request_reference": str(Path(args.request_reference).resolve()), "request_reference_sha256": file_hash(args.request_reference),
            "external_preparations": {"path": str(Path(args.prepared_records).resolve()),
                "sha256": file_hash(args.prepared_records), "receipts_sha256": file_hash(Path(args.prepared_records).with_name("requests.jsonl")),
                "reasoning_enabled": True},
            "expected_episodes": len(problems) * len(MODELS), "questions": len(problems),
            "families": len({r["family_id"] for r in planned.values()}),
            "input_partition": "dev", "campaign_role": "exploratory_dataset_extension",
            "public_reasoning_enabled": False, "public_provider_structured_output": True,
            "historical_raw_haiku_diagnostic": "Preserved answer+explanation request; separate from answer-only grid judgments.",
            "outcome_policy": "Retain knowledge_correct=false and public actual-answer role deviations; no outcome-based retries, rewrites or exclusions.",
            "technical_policy": "Unchanged BudgetedChat completion/context/schema/exposed-truncation gates and up to two technical schema retries; all attempts retained.",
            "message_source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in (
                "api.py", "environment3_packets.py", "environment3_prepared.py", "environment3_evidence.py")},
        }
        engine.atomic_json(args.out / "manifest.json", manifest, immutable=True)
        outcomes = await asyncio.gather(*(baseline_episode(api, p, row, model, external, args.out)
            for p, row in zip(problems, rows) for model in MODELS), return_exceptions=True)
        completed = [x for x in outcomes if not isinstance(x, BaseException)]
        summary = {"episodes": len(completed), "expected_episodes": len(problems) * len(MODELS),
                   "questions": len(problems), "families": manifest["families"],
                   "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
                   "requests": len(api.records), "finish_reasons": dict(Counter(r.get("finish_reason") for r in api.records)),
                   "knowledge_correct": sum(e["knowledge_correct"] for e in completed),
                   "knowledge_incorrect_retained": sum(not e["knowledge_correct"] for e in completed),
                   "role_deviations_retained": sum(not t["role_adherent"] for e in completed for t in e["turns"]),
                   "turns": sum(len(e["turns"]) for e in completed), "accounted_cost_upper_usd": api.spent,
                   "all_receipts_valid": all(r["valid"] for r in api.records)}
        engine.atomic_json(args.out / "summary.json", summary)
    if summary["errors"] or summary["episodes"] != summary["expected_episodes"]:
        raise RuntimeError("technical generation failures retained; incomplete full coverage: " + json.dumps(summary))
    return summary


def initialize_feedback(args):
    out = Path(args.out)
    if (out / "manifest.json").exists():
        raise ValueError("run already initialized; use advance or status")
    source, original_dir, bundle_path = Path(args.source).resolve(), Path(args.episodes).resolve(), Path(args.bundle).resolve()
    problem_list, source_rows, audit, planned = frozen_inventory(source, args.inventory, args.inventory_sha256)
    problems = {p.problem_id: p for p in problem_list}
    rows = {p.problem_id: r for p, r in zip(problem_list, source_rows)}
    bundle = strict_json(bundle_path.read_text())
    if (bundle["detector"] != "adapted_primary" or bundle["granularity"] != "response"
            or bundle["target"] != "wrong_answer_advocacy" or bundle["orientation"] != "higher_more_unreliable"):
        raise ValueError("requires frozen adapted-primary role-proxy bundle")
    old_manifest = strict_json((original_dir / "manifest.json").read_text())
    old_api = strict_json((original_dir / "api_manifest.json").read_text())
    if old_manifest["protocol"] != PROTOCOL or old_manifest["dataset_audit"]["dataset_sha256"] != file_hash(source):
        raise ValueError("original source/protocol differs from frozen inputs")
    if old_manifest.get("inventory_sha256") != args.inventory_sha256:
        raise ValueError("baseline inventory differs from feedback inventory")
    by_hash = {r["request_sha256"]: r for r in read_rows(original_dir / "requests.jsonl") if r["valid"]}
    templates, episodes, hashes = {}, [], {}
    for path in sorted(original_dir.glob("episode_*.json")):
        old = strict_json(path.read_text())
        if not old["complete"]:
            raise ValueError("original matched episode incomplete")
        problem = problems[old["problem_id"]]
        source_row = rows[problem.problem_id]
        if old["episode_id"] != sha256(problem.problem_id + old["model"] + PROTOCOL + "3"):
            raise ValueError("original episode identity changed")
        if any(old[k] != source_row[k] for k in ("parent_problem_id", "parent_source_line", "split")):
            raise ValueError("original source metadata changed")
        swapped, steps = schedule(rows[problem.problem_id])
        gold = problem.view(False, swapped)[1]
        if old["swapped"] != swapped or old["gold"] != gold:
            raise ValueError("original option mapping mismatch")
        if type(old["knowledge_correct"]) is not bool or old["knowledge_correct"] != (old["knowledge"] == gold):
            raise ValueError("knowledge diagnostic metadata mismatch")
        knowledge = [r for r in by_hash.values() if r["model"] == old["model"] and
                     r["tag"] == {"episode": old["episode_id"], "stage": "knowledge"}]
        if len(knowledge) != 1:
            raise ValueError("missing or repeated original knowledge receipt")
        diagnostic = audit_receipt(knowledge[0])
        if (diagnostic["request"]["messages"] != problem.view(True, swapped, structured=True)[0]
                or diagnostic["parsed"]["answer"] != old["knowledge"]):
            raise ValueError("knowledge diagnostic differs from original receipt")
        if len(old["turns"]) != 6:
            raise ValueError("six complete public turns required")
        for index, (turn, step) in enumerate(zip(old["turns"], steps)):
            if any(turn[k] != v for k, v in step.items()):
                raise ValueError("original six-turn schedule changed")
            receipt = audit_receipt(by_hash[turn["request_sha256"]])
            if (receipt["model"] != old["model"] or receipt["tag"] != {
                    "episode": old["episode_id"], "stage": "debate", "round": step["round"], "speaker": step["speaker"]}):
                raise ValueError("original public receipt model/tag mismatch")
            if receipt["request"]["messages"] != engine.base_messages(problem, old, index):
                raise ValueError("base generation prompt cannot be reproduced exactly")
            if any(receipt["parsed"][k] != turn[k] for k in ("argument", "answer")):
                raise ValueError("original transcript/receipt mismatch")
            if turn["role_adherent"] != (turn["answer"] == turn["assigned_answer"]):
                raise ValueError("role diagnostic metadata mismatch")
            if turn["advocates_gold"] != (turn["answer"] == gold) or turn["visible_prior_turns"] != index:
                raise ValueError("public history diagnostic metadata mismatch")
            template = {k: copy.deepcopy(v) for k, v in receipt["request"].items() if k != "messages"}
            if template["reasoning"] != {"enabled": False} or template.get("temperature") != 0 or template["max_tokens"] != 8192:
                raise ValueError("unexpected original public settings")
            if templates.setdefault(old["model"], template) != template:
                raise ValueError("original request settings vary within model")
        episodes.append(engine.initial_episode(old, bundle))
        hashes[str(path)] = file_hash(path)
    pairs = [(e["problem_id"], e["model"]) for e in episodes]
    if len(pairs) != len(set(pairs)) or set(pairs) != {(p, m) for p in problems for m in MODELS}:
        raise ValueError("source and original model/episode coverage differ")
    manifest = {
        "schema": "grid4_generation_manifest_v1", "protocol": engine.PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "source_sha256": file_hash(source),
        "original_episodes": str(original_dir), "original_episode_sha256": hashes,
        "original_receipts_sha256": file_hash(original_dir / "requests.jsonl"),
        "original_api_manifest_sha256": file_hash(original_dir / "api_manifest.json"),
        "original_manifest_sha256": file_hash(original_dir / "manifest.json"),
        "bundle": str(bundle_path), "bundle_sha256": file_hash(bundle_path),
        "detector": {k: copy.deepcopy(v) for k, v in bundle.items() if k != "rows"},
        "request_templates": templates, "model_specs": {m: old_api["models"][m] for m in templates},
        "episode_ids": [e["episode_id"] for e in episodes], "episodes": len(episodes), "questions": len(problems),
        "families": len({r["family_id"] for r in planned.values()}), "dataset_audit": audit,
        "feedback_instruction": engine.FEEDBACK_INSTRUCTION, "signal_explanation": engine.SIGNAL_EXPLANATION,
        "source_code_sha256": file_hash(engine.__file__),
        "extension_initializer": str(Path(__file__).resolve()), "extension_initializer_sha256": file_hash(__file__),
        "extension_inventory": str(Path(args.inventory).resolve()), "extension_inventory_sha256": args.inventory_sha256,
        "outcome_policy": "Knowledge mistakes and actual-answer role deviations remain outcomes; no exclusion or reroll.",
        "causal_contract": "Every action uses all and only prior scored public actions in its own trajectory; exact feedback-containing requests are scored.",
        "shared_trajectory": "A/C share original episodes; B/D share feedback episodes.",
        "restart_contract": "Unchanged grid4 durable intents, receipt replay and no automatic retries.",
    }
    engine.atomic_json(out / "manifest.json", manifest, immutable=True)
    for episode in episodes:
        engine.atomic_json(out / f"episode_{episode['episode_id']}.json", episode, immutable=True)
    for directory in ("requests", "intents", "queue/requests", "queue/results", "disclosure_adjudications"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    # Prove compatibility with the unchanged executor before any paid advance.
    engine.load_run(out)
    return {"episodes": len(episodes), "families": manifest["families"], "questions": len(problems), "out": str(out)}


def audit_extension_run(out):
    manifest = strict_json((Path(out) / "manifest.json").read_text())
    if manifest.get("extension_initializer_sha256") != file_hash(__file__):
        raise ValueError("extension initializer changed after freeze")
    frozen_inventory(manifest["source"], manifest["extension_inventory"], manifest["extension_inventory_sha256"])
    for name, expected_key in (("manifest.json", "original_manifest_sha256"),
                               ("api_manifest.json", "original_api_manifest_sha256"),
                               ("requests.jsonl", "original_receipts_sha256")):
        if file_hash(Path(manifest["original_episodes"]) / name) != manifest[expected_key]:
            raise ValueError("original baseline artifact changed: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline")
    initialize = commands.add_parser("feedback-init")
    for command in (baseline, initialize):
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--inventory", type=Path, required=True)
        command.add_argument("--inventory-sha256", required=True)
        command.add_argument("--out", type=Path, required=True)
    baseline.add_argument("--prepared-records", type=Path, required=True)
    baseline.add_argument("--budget", type=float, required=True)
    baseline.add_argument("--concurrency", type=int, default=4)
    baseline.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    baseline.add_argument("--request-reference", type=Path, default=Path("runs/iteration3/final/debates_v1/requests.jsonl"))
    initialize.add_argument("--episodes", type=Path, required=True)
    initialize.add_argument("--bundle", type=Path, required=True)
    step = commands.add_parser("advance")
    step.add_argument("--out", type=Path, required=True)
    step.add_argument("--model", action="append")
    step.add_argument("--budget", type=float, required=True)
    step.add_argument("--ledger", type=Path)
    step.add_argument("--concurrency", type=int, default=4)
    step.add_argument("--timeout", type=float, default=300)
    step.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "baseline":
        result = asyncio.run(run_baseline(args))
    else:
        with engine.run_lock(args.out):
            if args.command == "feedback-init":
                result = initialize_feedback(args)
            else:
                audit_extension_run(args.out)
                if args.command == "advance":
                    result = asyncio.run(engine.advance(args))
                else:
                    manifest, _, episodes = engine.load_run(args.out)
                    result = engine.status(args.out, manifest, episodes)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
