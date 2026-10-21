"""Offline, causal display compiler for fresh shared-prefix debate branches.

This module makes no provider calls. It cannot reuse a historical suffix after
changing feedback. All input artifacts are read only; output jobs are drafts.
"""

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import random

from .environment3_evidence import evidence_messages
from .grid4_generate import FEEDBACK_INSTRUCTION, SIGNAL_EXPLANATION, feedback_packet
from .judge3_protocol import percentile
from .preflight import load_problems
from .grid5_generate import PUBLIC_SCHEMA, audit_receipt


MODEL = "qwen/qwen3.5-27b"
PROTOCOL = "grid6_after_round1_v2"
SELECTION_PROTOCOL = "grid6_after_round1_v1"
POLICIES = ("none", "neutral", "bundle", "turn_scores", "speaker_means",
            "recommendation", "shuffled_bundle")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def publish(path, value):
    """Idempotent immutable publication, rejecting an incompatible rerun."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"refusing to overwrite different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


STATIC_FIELDS = (
    "schema", "parent_episode_id", "family_id", "problem_id", "parent_problem_id",
    "model", "gold", "swapped", "private_preparations", "policy",
    "detector_identity", "shared_prefix_sha256", "turn_schedule", "shuffle_seed",
    "problem_sha256", "source_episode_sha256", "generation_parameters",
    "historical_suffix_request_sha256", "historical_response_body_sha256",
)


def generation_parameters():
    """Full request template; replacing provider routing creates a new contract."""
    return {"model": MODEL, "temperature": 0, "max_tokens": 8192,
            "reasoning": {"enabled": False}, "usage": {"include": True},
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "diagnostic", "strict": True, "schema": PUBLIC_SCHEMA}}}


def make_branch(original, policy, detector_identity, family_id, problem,
                *, request_parameters=None, source_episode_sha256=None,
                historical_response_body_sha256=()):
    if policy not in POLICIES:
        raise ValueError("unknown feedback policy")
    if (original["model"] != MODEL or not original["complete"]
            or original.get("protocol") != "explicit_reasoned_evidence_sequential"):
        raise ValueError("requires completed Qwen3.5 no-feedback source trajectory")
    if problem.problem_id != original["problem_id"] or problem.view(False, original["swapped"])[1] != original["gold"]:
        raise ValueError("source problem differs from original trajectory")
    turns = original["turns"]
    if (len(turns) != 6 or {t["speaker"] for t in turns[:2]} != {"Red", "Blue"}
            or any(t["round"] != 1 for t in turns[:2])):
        raise ValueError("requires six turns and a complete first round")
    parameters = copy.deepcopy(request_parameters or generation_parameters())
    if ("messages" in parameters or parameters.get("model") != MODEL
            or parameters.get("reasoning") != {"enabled": False}
            or parameters.get("temperature") != 0
            or parameters.get("max_tokens") != 8192
            or parameters.get("response_format") != generation_parameters()["response_format"]
            or not isinstance(parameters.get("provider"), dict)):
        raise ValueError("invalid frozen generation parameters")
    branch = {k: copy.deepcopy(original[k]) for k in (
        "problem_id", "parent_problem_id", "model", "gold", "swapped",
        "private_preparations")}
    branch.update(
        schema=PROTOCOL, parent_episode_id=original["episode_id"],
        family_id=family_id, policy=policy, detector_identity=detector_identity,
        shared_prefix_sha256=digest(turns[:2]),
        turns=copy.deepcopy(turns[:2]), complete=False,
        turn_schedule=[{k: t[k] for k in ("speaker", "round", "assigned_answer")}
                       for t in turns],
        shuffle_seed=digest([SELECTION_PROTOCOL, original["episode_id"], "control"]),
        problem_sha256=digest(asdict(problem)),
        source_episode_sha256=source_episode_sha256 or digest(original),
        generation_parameters=parameters,
        historical_suffix_request_sha256=[t["request_sha256"] for t in turns[2:]],
        historical_response_body_sha256=sorted(set(historical_response_body_sha256)),
        continuation_receipts=[],
    )
    branch["contract_sha256"] = digest({k: branch[k] for k in STATIC_FIELDS})
    branch["branch_id"] = digest([PROTOCOL, branch["contract_sha256"]])
    return branch


def visible_packet(turns, policy, shuffle_seed):
    if policy not in POLICIES:
        raise ValueError("unknown feedback policy")
    if policy == "none":
        return None
    # Neutral geometry uses no scores, so it also works without a live scorer.
    if policy == "neutral":
        return {"granularity": "response", "units": [
            {"turn": i + 1, "speaker": t["speaker"], "unit": 1,
             "signal_percentile": None} for i, t in enumerate(turns)],
            "mean_signal_by_speaker": {t["speaker"]: None for t in turns},
            "detector_only_recommendation": None}
    scored = copy.deepcopy(turns)
    for i, turn in enumerate(scored):
        score = turn.get("probe_feedback")
        if not score:
            raise ValueError("unscored prior turn")
        value = score.get("percentile")
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 100):
            raise ValueError("invalid percentile")
        if (score.get("source_request_sha256") != turn["request_sha256"]
                or score.get("argument_sha256") != hashlib.sha256(
                    turn["argument"].encode()).hexdigest()
                or score.get("turn_index") != i):
            raise ValueError("score is not bound to its completed prior turn")
    if policy == "shuffled_bundle":
        values = [t["probe_feedback"]["percentile"] for t in scored]
        random.Random(digest([shuffle_seed, len(scored)])).shuffle(values)
        for t, v in zip(scored, values):
            t["probe_feedback"]["percentile"] = v
    packet = feedback_packet(scored)
    if policy in ("bundle", "shuffled_bundle"):
        return packet
    if policy == "turn_scores":
        return {"granularity": "response", "units": packet["units"]}
    if policy == "speaker_means":
        return {"mean_signal_by_speaker": packet["mean_signal_by_speaker"]}
    return {"detector_only_recommendation": packet["detector_only_recommendation"]}


def _messages(problem, branch):
    """Compile only after the public validation boundary has checked state."""
    turns = branch["turns"]
    index = len(turns)
    if branch["complete"] or not 2 <= index < len(branch["turn_schedule"]):
        raise ValueError("branch has no next continuation turn")
    step = branch["turn_schedule"][index]
    prep = {k: v for k, v in branch["private_preparations"][step["speaker"]].items()
            if k != "request_sha256"}
    messages = evidence_messages(problem, branch["swapped"], step["speaker"],
                                 step["assigned_answer"], turns,
                                 json.dumps(prep, ensure_ascii=False), explicit_truth=True)
    packet = visible_packet(turns, branch["policy"], branch["shuffle_seed"])
    if packet is not None:
        if branch["policy"] != "neutral":
            for turn in turns:
                if turn["probe_feedback"].get("detector_identity") != branch["detector_identity"]:
                    raise ValueError("score belongs to another branch detector")
        messages[0]["content"] += FEEDBACK_INSTRUCTION
        messages[-1]["content"] += (
            "\n\nBEGIN PRIVATE PAST-TURN PROBE FEEDBACK\n" + SIGNAL_EXPLANATION
            + "\n" + json.dumps(packet, ensure_ascii=False)
            + "\nEND PRIVATE PAST-TURN PROBE FEEDBACK")
    return messages


def state_sha256(branch):
    return digest({"branch_id": branch["branch_id"], "turns": branch["turns"]})


def continuation_tag(branch):
    return {"branch_id": branch["branch_id"], "turn_index": len(branch["turns"]),
            "prior_state_sha256": state_sha256(branch), "stage": "grid6_continuation"}


def _accepted_turn(problem, branch, receipt, score_result):
    """Verify full request, response, causal branch tag, and optional score receipt."""
    expected = {**copy.deepcopy(branch["generation_parameters"]),
                "messages": _messages(problem, branch)}
    if receipt.get("request") != expected:
        raise ValueError("continuation receipt request/context/settings differ")
    if receipt.get("tag") != continuation_tag(branch):
        raise ValueError("continuation receipt belongs to another branch or prior state")
    audit_receipt(receipt)
    response_hash = hashlib.sha256(receipt["response_body"].encode()).hexdigest()
    if response_hash in branch["historical_response_body_sha256"]:
        raise ValueError("historical response receipt cannot be a new continuation")
    if any(r["receipt"]["response_body"] == receipt["response_body"] for r in branch["continuation_receipts"]):
        raise ValueError("provider response receipt already used in this branch")
    index = len(branch["turns"])
    step = branch["turn_schedule"][index]
    turn = {**copy.deepcopy(step), **copy.deepcopy(receipt["parsed"]),
            "request_sha256": receipt["request_sha256"], "visible_prior_turns": index,
            "generation_branch_id": branch["branch_id"],
            "generation_prior_state_sha256": state_sha256(branch),
            "generation_receipt_sha256": digest(receipt),
            "response_body_sha256": response_hash}
    if score_result is not None:
        expected_score = {
            "turn_index": index, "source_request_sha256": receipt["request_sha256"],
            "argument_sha256": hashlib.sha256(turn["argument"].encode()).hexdigest(),
            "detector_identity": branch["detector_identity"],
            "messages_sha256": digest(expected["messages"]),
            "response_body_sha256": response_hash,
            "branch_id": branch["branch_id"],
        }
        if any(score_result.get(k) != v for k, v in expected_score.items()):
            raise ValueError("score result belongs to another context/response/detector")
        turn["probe_feedback"] = copy.deepcopy(score_result)
        visible_packet([*branch["turns"], turn], "bundle", branch["shuffle_seed"])
    elif branch["policy"] not in ("none", "neutral"):
        raise ValueError("live feedback requires a score before accepting the turn")
    return turn


def validate_branch(problem, branch):
    if branch.get("schema") != PROTOCOL:
        raise ValueError("unsupported branch protocol")
    if digest({k: branch[k] for k in STATIC_FIELDS}) != branch.get("contract_sha256"):
        raise ValueError("immutable branch contract changed")
    if digest([PROTOCOL, branch["contract_sha256"]]) != branch.get("branch_id"):
        raise ValueError("branch identity changed")
    if digest(asdict(problem)) != branch["problem_sha256"] or problem.problem_id != branch["problem_id"]:
        raise ValueError("source problem changed")
    turns = branch["turns"]
    if not 2 <= len(turns) <= len(branch["turn_schedule"]):
        raise ValueError("invalid branch length")
    if digest(turns[:2]) != branch["shared_prefix_sha256"]:
        raise ValueError("shared prefix changed")
    if branch["complete"] != (len(turns) == len(branch["turn_schedule"])):
        raise ValueError("branch completion flag differs from retained turns")
    if len(branch["continuation_receipts"]) != len(turns) - 2:
        raise ValueError("continuation requires authenticated receipt; historical suffix prohibited")
    for i, turn in enumerate(turns):
        if any(turn[k] != branch["turn_schedule"][i][k]
               for k in ("speaker", "round", "assigned_answer")):
            raise ValueError("turn schedule changed")
    prior = {**branch, "turns": copy.deepcopy(turns[:2]),
             "continuation_receipts": [], "complete": False}
    for index, stored in enumerate(branch["continuation_receipts"], 2):
        accepted = _accepted_turn(problem, prior, stored["receipt"], stored["score_result"])
        if accepted != turns[index]:
            raise ValueError("retained continuation differs from authenticated receipt")
        prior["turns"].append(accepted)
        prior["continuation_receipts"].append(stored)
        prior["complete"] = len(prior["turns"]) == len(prior["turn_schedule"])
    return branch


def next_messages(problem, branch):
    validate_branch(problem, branch)
    return _messages(problem, branch)


def next_request(problem, branch):
    return {**copy.deepcopy(branch["generation_parameters"]),
            "messages": next_messages(problem, branch)}


def append_turn(problem, branch, receipt, score_result=None):
    """Return a validated new state; never mutate source state or supplied receipts.

    Request hashes identify inputs, so a fresh no-feedback request may legitimately
    equal a historical request. Historical response hashes and authenticated causal
    receipt provenance identify the generation event that must not be reused.
    """
    validate_branch(problem, branch)
    result = copy.deepcopy(branch)
    turn = _accepted_turn(problem, result, receipt, score_result)
    result["turns"].append(turn)
    result["continuation_receipts"].append({"receipt": copy.deepcopy(receipt),
                                            "score_result": copy.deepcopy(score_result)})
    result["complete"] = len(result["turns"]) == len(result["turn_schedule"])
    validate_branch(problem, result)
    return result


def compile_manifest(root, out, family_count=24):
    root, out = Path(root), Path(out)
    source = root / "runs/iteration5/coordinator/all166_source_frozen_v2.jsonl"
    package_path = root / "runs/iteration5/judge_grid_v1/package.json"
    completion_path = root / "runs/iteration5/coordinator/final_analysis_complete_v1.json"
    package, completion = (json.loads(p.read_text()) for p in (package_path, completion_path))
    if hashlib.sha256(package_path.read_bytes()).hexdigest() != completion["package_sha256"]:
        raise ValueError("completed package lineage changed")
    def canonical(path):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if package["input_sha256"].get(str(path.resolve())) != actual:
            raise ValueError("frozen source/episode/bundle differs from completed package")
        return actual
    canonical(source)
    raw_dir = root / "runs/iteration6/raw_audit"
    raw_manifest = json.loads((raw_dir / "manifest.json").read_text())
    raw_index_path = raw_dir / "turn_index.jsonl"
    if hashlib.sha256(raw_index_path.read_bytes()).hexdigest() != raw_manifest["outputs_sha256"]["turn_index.jsonl"]:
        raise ValueError("raw-context receipt index changed")
    raw_index = {}
    for line in raw_index_path.read_text().splitlines():
        item = json.loads(line)
        key = item["episode_id"], item["turn_index"]
        if key in raw_index:
            raise ValueError("duplicate raw receipt index")
        raw_index[key] = item
    checked_receipt_files = set()
    problems, audit = load_problems(source)
    if audit["errors"]:
        raise ValueError("invalid frozen source")
    problems = {p.problem_id: p for p in problems}
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    by_parent = {r["parent_problem_id"]: r for r in rows}
    families = sorted({r["family_id"] for r in rows},
                      key=lambda f: digest([SELECTION_PROTOCOL, "screen", f]))
    if not 1 <= family_count <= len(families):
        raise ValueError("invalid family count")
    # One prospectively chosen question per selected family for screening.
    parents = {min((p for p, r in by_parent.items() if r["family_id"] == f),
                   key=lambda p: digest([SELECTION_PROTOCOL, "question", p]))
               for f in families[:family_count]}
    bundle_path = root / "runs/iteration5/response_bundles/adapted_primary_baseline_v1.json"
    canonical(bundle_path)
    bundle = json.loads(bundle_path.read_text())
    detector = digest([bundle["head_sha256"], bundle["reference_sha256"]])
    scores = {(r["episode_id"], r["turn_index"]): r for r in bundle["rows"]}
    reference = bundle["references"][MODEL]["scores"]
    jobs, included = [], set()
    for directory in ("runs/iteration3/final/debates_v1", "runs/iteration5/debates_v4"):
        for path in sorted((root / directory).glob("episode_*.json")):
            episode = json.loads(path.read_text())
            parent = episode["parent_problem_id"]
            if episode["model"] != MODEL or parent not in parents:
                continue
            if parent in included:
                raise ValueError("duplicate canonical source episode")
            included.add(parent)
            episode_hash = canonical(path)
            historical_responses = []
            for i, turn in enumerate(episode["turns"]):
                indexed = raw_index[(episode["episode_id"], i)]
                if indexed["arm"] != "A" or indexed["request_sha256"] != turn["request_sha256"]:
                    raise ValueError("raw receipt source identity differs")
                receipt_path = root / indexed["receipt"]["path"]
                if receipt_path not in checked_receipt_files:
                    if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != raw_manifest["inputs_sha256"][indexed["receipt"]["path"]]:
                        raise ValueError("audited raw provider receipts changed")
                    checked_receipt_files.add(receipt_path)
                historical_responses.append(indexed["response_body_sha256"])
            for i, turn in enumerate(episode["turns"][:2]):
                score = scores[(episode["episode_id"], i)]
                ah = hashlib.sha256(turn["argument"].encode()).hexdigest()
                if score["argument_sha256"] != ah or len(score["scores"]) != 1:
                    raise ValueError("prefix scalar differs from source text")
                turn["probe_feedback"] = {
                    "turn_index": i, "source_request_sha256": turn["request_sha256"],
                    "argument_sha256": ah, "detector_identity": detector,
                    "percentile": percentile(score["scores"][0], reference),
                    "score_bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                    "origin": "original no-feedback prefix; detector scores not shown during prefix",
                }
            for policy in POLICIES:
                branch = make_branch(episode, policy, detector, by_parent[parent]["family_id"],
                                     problems[episode["problem_id"]], source_episode_sha256=episode_hash,
                                     historical_response_body_sha256=historical_responses)
                messages = next_messages(problems[episode["problem_id"]], branch)
                job = {"branch": branch, "first_continuation_messages": messages,
                       "messages_sha256": digest(messages),
                       "source_episode_path": str(path.relative_to(root)),
                       "source_episode_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "status": "prepared_not_generated"}
                publish(out / "branches" / (branch["branch_id"] + ".json"), job)
                jobs.append({"branch_id": branch["branch_id"], "policy": policy,
                             "parent_problem_id": parent, "family_id": branch["family_id"],
                             "shared_prefix_sha256": branch["shared_prefix_sha256"],
                             "first_messages_sha256": digest(messages)})
    if included != parents:
        raise ValueError("missing canonical source episodes")
    manifest = {"schema": PROTOCOL, "status": "prepared_not_generated",
                "paid_calls": 0, "families": family_count, "questions": len(parents),
                "policies": list(POLICIES), "branches": len(jobs),
                "new_public_turns_needed": len(jobs) * 4,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "compiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
                "raw_audit_manifest_sha256": hashlib.sha256((raw_dir / "manifest.json").read_bytes()).hexdigest(),
                "selection_protocol": SELECTION_PROTOCOL,
                "provider_contract": "Unpinned provider route; full request parameters are frozen per branch. Pinning a route requires a new branch contract.",
                "receipt_authentication_limit": "Receipt consistency and known historical reuse are checked. Provider event freshness requires a trusted caller-owned durable executor; hashes are not cryptographic proof of a new network call.",
                "screening": "hash-selected families and one question per family; historically exposed",
                "remaining_families": families[family_count:], "jobs": jobs,
                "execution_gates": ["bounded new campaign budget", "provider/runtime contract",
                                    "incremental generation runner", "score parity", "annotation ready"]}
    publish(out / "manifest.json", manifest)
    return {k: v for k, v in manifest.items() if k not in ("jobs", "remaining_families")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(compile_manifest(args.root, args.out, args.families), indent=2))


if __name__ == "__main__":
    main()
