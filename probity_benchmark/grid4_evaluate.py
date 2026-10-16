"""Complete the paired feedback grid without regenerating frozen A/C judgments.

A/C use one no-feedback transcript; B/D use one new feedback transcript on the
same story and assignment. A and B are raw no-score judgments. The original
matched-neutral control is deliberately not substituted for A.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import re

from .api import ANSWER, BudgetedChat, object_schema, strict_json
from .judge3_final import audit_completion, check_package as check_prior_package
from .judge3_final import file_hash, read_rows
from .judge3_protocol import base_job, bundle_jobs
from .judge_assistance import paired_stats
from .preflight import load_problems, sha256


SCHEMA = "grid4_paired_feedback_v1"
PAIR_CONTRASTS = {"B-A": ("B", "A"), "C-A": ("C", "A"),
                  "D-B": ("D", "B"), "D-A": ("D", "A")}
DISCLOSURE = re.compile(
    r"\b(?:probe|detector|percentile|neural analysis)\b|"
    r"\bscore\w*\b[^.!?\n]{0,35}(?:\d|%)", re.IGNORECASE
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def keyed(rows, key="job_id"):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate " + key)
    return result


def load_episodes(directory):
    return keyed([json.loads(p.read_text()) for p in sorted(
        Path(directory).glob("episode_*.json"))], "episode_id")


def pair_episodes(prior, feedback, families):
    """Require every frozen episode once, with identical non-treatment design."""
    if set(prior) != set(families) or len(feedback) != len(prior):
        raise ValueError("episode inclusion differs from the frozen design")
    fallback = {}
    for old in prior.values():
        key = (old["parent_problem_id"], old["model"])
        if key in fallback:
            raise ValueError("ambiguous story/model pairing")
        fallback[key] = old["episode_id"]
    pairs, seen = [], set()
    for new in sorted(feedback.values(), key=lambda e: e["episode_id"]):
        old_id = new.get("matched_no_feedback_episode_id") or fallback.get(
            (new["parent_problem_id"], new["model"]))
        if old_id not in prior or old_id in seen or new["episode_id"] in prior:
            raise ValueError("feedback episode has missing, repeated, or old identity")
        old = prior[old_id]
        for field in ("problem_id", "parent_problem_id", "model", "split",
                      "world", "swapped", "gold"):
            if old[field] != new[field]:
                raise ValueError("paired episode changed " + field)
        if not old["complete"] or not new["complete"]:
            raise ValueError("incomplete paired trajectory")
        if len(old["turns"]) != len(new["turns"]):
            raise ValueError("feedback trajectory changed turn count")
        for a, b in zip(old["turns"], new["turns"]):
            for field in ("speaker", "assigned_answer", "round"):
                if a.get(field) != b.get(field):
                    raise ValueError("paired trajectory changed turn " + field)
            if b.get("answer") not in ("A", "B"):
                raise ValueError("invalid actual public answer")
        seen.add(old_id)
        pairs.append({
            "no_feedback_episode_id": old_id,
            "feedback_episode_id": new["episode_id"],
            "family": families[old_id], "participant_model": old["model"],
            "parent_problem_id": old["parent_problem_id"], "gold": old["gold"],
            "no_feedback_argument_sha256": [sha256(t["argument"]) for t in old["turns"]],
            "feedback_argument_sha256": [sha256(t["argument"]) for t in new["turns"]],
            "feedback_role_switches": [{"turn_index": i, "speaker": t["speaker"],
                                        "assigned_answer": t["assigned_answer"], "answer": t["answer"]}
                                       for i, t in enumerate(new["turns"])
                                       if t["answer"] != t["assigned_answer"]],
        })
    if seen != set(prior):
        raise ValueError("missing paired trajectory")
    return sorted(pairs, key=lambda p: p["no_feedback_episode_id"])


def disclosure_audit(episodes, reviewed=None):
    """Hold suspicious public text for review; never remove or rewrite prose."""
    reviewed = reviewed or {}
    result = []
    for eid, episode in sorted(episodes.items()):
        for index, turn in enumerate(episode["turns"]):
            matches = [m.group() for m in DISCLOSURE.finditer(turn["argument"])]
            upstream = turn.get("disclosure_audit", {})
            if not matches and not upstream.get("needs_review"):
                continue
            digest = sha256(turn["argument"])
            review = reviewed.get(digest)
            if (not isinstance(review, dict) or type(review.get("discloses_scores")) is not bool
                    or not review.get("reason")):
                raise ValueError("public feedback disclosure needs review: " + eid + ":" + str(index))
            result.append({"episode_id": eid, "turn_index": index,
                           "argument_sha256": digest, "matches": matches,
                           "upstream_audit": upstream, "review": review})
    return result


def load_disclosure_reviews(path):
    if path is None or not path.exists():
        return {}, []
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    reviews = {}
    for filename in files:
        data = json.loads(filename.read_text())
        if "reviewed_argument_sha256" in data:
            values = data["reviewed_argument_sha256"]
        else:
            if data["decision"] not in ("not_feedback_disclosure", "retain_with_disclosure"):
                raise ValueError("unknown public disclosure adjudication")
            values = {data["argument_sha256"]: {
                "discloses_scores": data["decision"] == "retain_with_disclosure",
                "reason": data["reason"]}}
        for digest, review in values.items():
            if digest in reviews and reviews[digest] != review:
                raise ValueError("conflicting disclosure adjudications")
            reviews[digest] = review
    return reviews, files


def construct_jobs(prior_jobs, prior_episodes, feedback_episodes, families, problems,
                   bundle, prior_bundle):
    """Preserve old jobs byte-for-byte and existing D display semantics."""
    for field in ("detector", "target", "granularity", "orientation", "head_sha256",
                  "reference_sha256", "references"):
        if bundle[field] != prior_bundle[field]:
            raise ValueError("feedback signal changed frozen " + field)
    if bundle["granularity"] != "response":
        raise ValueError("this grid freezes the adapted whole-response signal")
    pairs = pair_episodes(prior_episodes, feedback_episodes, families)
    original = {}
    for job in prior_jobs:
        cell = "A" if job["kind"] == "none" else (
            "C" if job["kind"] == bundle["detector"] + ":real" else None)
        if cell:
            identity = (job["episode_id"], cell)
            if identity in original:
                raise ValueError("duplicate prior grid cell")
            original[identity] = job
    if set(original) != {(eid, c) for eid in families for c in ("A", "C")}:
        raise ValueError("prior package lacks a frozen A/C presentation")
    bases, original_bases, assigned_feedback = {}, {}, {}
    for pair in pairs:
        old_id, new_id = pair["no_feedback_episode_id"], pair["feedback_episode_id"]
        old, new = prior_episodes[old_id], feedback_episodes[new_id]
        original_base = original[(old_id, "A")]
        rebuilt = base_job(problems[old["problem_id"]], old,
                           original_base["environment_protocol"])
        if rebuilt != original_base:
            raise ValueError("original A prompt or metadata no longer reproduces exactly")
        original_bases[old_id] = rebuilt
        # A concession can be caused by feedback. Keep every such trajectory,
        # show its actual answer fields verbatim, and use the original assigned
        # speaker roles only for the frozen role-proxy recommendation.
        assigned = copy.deepcopy(new)
        for turn in assigned["turns"]:
            turn["answer"] = turn["assigned_answer"]
        assigned_feedback[new_id] = assigned
        base = base_job(problems[new["problem_id"]], assigned,
                        original_base["environment_protocol"])
        messages, _ = problems[new["problem_id"]].view(False, new["swapped"], structured=True)
        transcript = "\n\n".join(
            f"Turn {i + 1}. {turn['speaker']} (answer {turn['answer']}):\n" + turn["argument"]
            for i, turn in enumerate(new["turns"]))
        base["messages"][1]["content"] = messages[1]["content"] + "\n\nPublic debate:\n" + transcript
        bases[new_id] = base
    original_signals, _ = bundle_jobs(prior_bundle, prior_episodes, original_bases)
    for job in original_signals:
        if job["condition"] == "real" and job != original[(job["episode_id"], "C")]:
            raise ValueError("original C scores or prompt no longer reproduce exactly")
    # Existing compiler supplies the frozen response display and recommendation.
    # Neutral/shuffled jobs are constructed locally, but are never paid jobs here.
    score_jobs, _ = bundle_jobs(bundle, assigned_feedback, bases)
    real = {j["episode_id"]: j for j in score_jobs if j["condition"] == "real"}
    if set(real) != set(feedback_episodes):
        raise ValueError("feedback scores do not cover every frozen episode")
    old_jobs, new_jobs = [], []
    for pair in pairs:
        old_id, new_id = pair["no_feedback_episode_id"], pair["feedback_episode_id"]
        pair["jobs"] = {}
        for cell in ("A", "C"):
            job = copy.deepcopy(original[(old_id, cell)])
            old_jobs.append(job)
            pair["jobs"][cell] = job["job_id"]
        for cell, source in (("B", bases[new_id]), ("D", real[new_id])):
            job = copy.deepcopy(source)
            job.update(grid_cell=cell, matched_no_feedback_episode_id=old_id,
                       source_family=pair["family"],
                       feedback_protocol=feedback_episodes[new_id].get("protocol"),
                       role_mapping="original assigned speaker-to-answer mapping",
                       feedback_role_switches=pair["feedback_role_switches"])
            # Retain the neutral ID for display provenance, but identify the actual
            # raw no-score comparator separately instead of relabelling it neutral.
            job["grid_raw_baseline_job_id"] = bases[new_id]["job_id"]
            new_jobs.append(job)
            pair["jobs"][cell] = job["job_id"]
    keyed(old_jobs + new_jobs)
    return old_jobs, new_jobs, pairs


def prepare(args):
    if args.out.exists():
        raise ValueError("immutable new package directory required")
    prior, old_jobs_path = check_prior_package(args.prior_package, args.prior_package_sha256)
    if prior["episodes"] != args.expected_episodes or prior["families"] != args.expected_families:
        raise ValueError("prior inclusion does not match the preregistered count")
    episodes = load_episodes(args.prior_episodes)
    episodes = {eid: e for eid, e in episodes.items() if eid in prior["episode_families"]}
    feedback = load_episodes(args.feedback_episodes)
    problems, audit = load_problems(args.dataset)
    if audit["errors"] or audit["duplicate_problem_hashes"]:
        raise ValueError("invalid or duplicate original dataset")
    bundle, original_bundle = (json.loads(p.read_text()) for p in
                               (args.feedback_bundle, args.prior_bundle))
    reviews, review_paths = load_disclosure_reviews(
        args.disclosure_review or args.feedback_episodes / "disclosure_adjudications")
    disclosures = disclosure_audit(feedback, reviews)
    old, jobs, pairs = construct_jobs(
        read_rows(old_jobs_path), episodes, feedback, prior["episode_families"],
        {p.problem_id: p for p in problems}, bundle, original_bundle)
    compatibility = []
    if len(args.compatible_prior_package) != len(args.compatible_prior_package_sha256):
        raise ValueError("compatible prior packages need corresponding reviewed hashes")
    for directory, digest in zip(args.compatible_prior_package, args.compatible_prior_package_sha256):
        alternate, alternate_jobs_path = check_prior_package(directory, digest)
        alternate_jobs = keyed(read_rows(alternate_jobs_path))
        if (alternate["episode_families"] != prior["episode_families"]
                or any(alternate_jobs.get(j["job_id"]) != j for j in old)):
            raise ValueError("alternate prior package changed frozen A/C")
        compatibility.append({"directory": str(directory.resolve()), "package_sha256": digest,
                              "jobs_sha256": alternate["jobs_sha256"]})
    counts = Counter((p["family"], p["participant_model"]) for p in pairs)
    models = sorted({p["participant_model"] for p in pairs})
    if set(counts.values()) != {1} or len(counts) != args.expected_families * len(models):
        raise ValueError("unequal frozen participant coverage by source family")
    args.out.mkdir(parents=True)
    write_rows(args.out / "prior_jobs.jsonl", old)
    write_rows(args.out / "jobs.jsonl", jobs)
    write_json(args.out / "pairs.json", pairs)
    inputs = [args.prior_package / "package.json", old_jobs_path, args.dataset,
              args.prior_bundle, args.feedback_bundle]
    inputs.extend(review_paths)
    for alternate in compatibility:
        inputs.append(Path(alternate["directory"]) / "package.json")
    for directory, selected in ((args.prior_episodes, episodes), (args.feedback_episodes, feedback)):
        inputs.extend(directory / ("episode_" + eid + ".json") for eid in selected)
    manifest = {
        "schema": SCHEMA, "episodes": len(pairs), "families": args.expected_families,
        "participant_models": models, "detector": bundle["detector"],
        "head_sha256": bundle["head_sha256"], "reference_sha256": bundle["reference_sha256"],
        "prior_package_sha256": args.prior_package_sha256,
        "prior_full_jobs_sha256": prior["jobs_sha256"],
        "compatible_prior_packages": compatibility,
        "prior_jobs_sha256": file_hash(args.out / "prior_jobs.jsonl"),
        "jobs_sha256": file_hash(args.out / "jobs.jsonl"),
        "pairs_sha256": file_hash(args.out / "pairs.json"), "jobs_expected": len(jobs),
        "public_disclosure_review": disclosures,
        "verified_disclosure_episodes": sorted({d["episode_id"] for d in disclosures
                                                 if d["review"]["discloses_scores"]}),
        "verified_disclosure_turns": sum(d["review"]["discloses_scores"] for d in disclosures),
        "disclosure_policy": "All verified leaks retained in the full paired intention-to-treat grid and labelled protocol deviations. A clean paired subset, when present, is secondary only.",
        "feedback_role_switches": sum(len(p["feedback_role_switches"]) for p in pairs),
        "feedback_episodes_with_role_switch": sum(bool(p["feedback_role_switches"]) for p in pairs),
        "role_switch_policy": "Retain every trajectory and display actual public answers and prose unchanged. Score-only recommendations use original assigned roles, even if a speaker concedes.",
        "input_sha256": {str(p.resolve()): file_hash(p) for p in inputs},
        "runtime_source_sha256": {str(Path(p).resolve()): file_hash(p) for p in (
            __file__, "probity_benchmark/api.py", "probity_benchmark/judge3_protocol.py",
            "probity_benchmark/run_local_judge.py")},
        "estimands": [*PAIR_CONTRASTS, "(D-B)-(C-A)"],
        "baseline": "A is raw no-score, not matched neutral. C and D use real scores.",
        "independent_unit": "source family, retaining both participant models together",
        "selection": "All prior frozen story/model episodes retained before new judge outcomes.",
        "scope": "One new feedback trajectory per original story/model assignment; feedback can change the prose. No fresh no-feedback regeneration or independent trajectory repeats.",
    }
    write_json(args.out / "package.json", manifest)
    print(json.dumps({"package_sha256": file_hash(args.out / "package.json"),
                      "new_jobs": len(jobs), "reused_A_C_jobs": len(old),
                      "episode_pairs": len(pairs), "families": args.expected_families}))


def check_package(directory, expected_hash, check_runtime=False):
    if file_hash(directory / "package.json") != expected_hash:
        raise ValueError("grid package changed after freeze")
    package = json.loads((directory / "package.json").read_text())
    if package["schema"] != SCHEMA:
        raise ValueError("unknown grid package")
    for filename, key in (("jobs.jsonl", "jobs_sha256"), ("prior_jobs.jsonl", "prior_jobs_sha256"),
                          ("pairs.json", "pairs_sha256")):
        if file_hash(directory / filename) != package[key]:
            raise ValueError("frozen grid input changed: " + filename)
    if check_runtime:
        for filename, digest in package["runtime_source_sha256"].items():
            if file_hash(filename) != digest:
                raise ValueError("frozen runner changed: " + filename)
    return package


def api_request_body(api, judge, messages):
    """Mirror BudgetedChat's complete request for exact dedup; assert at receipt."""
    spec = api.models[judge]
    price = {k: float(spec["pricing"].get(k, 0)) for k in ("prompt", "completion", "request")}
    body = {"model": judge, "messages": messages, "max_tokens": api.max_tokens,
            "reasoning": api.reasoning_by_model.get(judge, api.reasoning),
            "usage": {"include": True}, "provider": {"require_parameters": True,
            "max_price": {"prompt": price["prompt"] * 1e6,
                          "completion": price["completion"] * 1e6, "request": price["request"]}}}
    if api.structured_output:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "diagnostic", "strict": True, "schema": object_schema({"answer": ANSWER})}}
    if "temperature" in spec["supported_parameters"]:
        body["temperature"] = 0
    return body


def prior_jobs_hash(package, directory):
    digest = json.loads((directory / "manifest.json").read_text())["jobs_sha256"]
    allowed = {package["prior_full_jobs_sha256"],
               *(p["jobs_sha256"] for p in package["compatible_prior_packages"])}
    if digest not in allowed:
        raise ValueError("prior results do not use a reviewed compatible package")
    return digest


def exact_request_groups(jobs, body_for_job):
    grouped = {}
    for job in sorted(jobs, key=lambda j: j["job_id"]):
        body = body_for_job(job)
        key = sha256(json.dumps(body, sort_keys=True))
        if key not in grouped:
            grouped[key] = {"request": body, "jobs": []}
        if grouped[key]["request"] != body:
            raise ValueError("request hash collision")
        grouped[key]["jobs"].append(job)
    return grouped


async def run_api(args):
    package = check_package(args.package, args.frozen_package_sha256, True)
    if not os.environ.get("PROBITY_PILOT_LEDGER"):
        raise ValueError("shared pilot budget ledger is mandatory")
    if args.out.exists():
        raise ValueError("immutable new result directory required")
    jobs = read_rows(args.package / "jobs.jsonl")
    _, old_audit = audit_results(args.prior_results,
                                keyed(read_rows(args.package / "prior_jobs.jsonl")),
                                prior_jobs_hash(package, args.prior_results), True)
    if old_audit["runtime_identity"]["type"] != "api":
        raise ValueError("API B/D cannot be joined with native A/C")
    reasoning = {"effort": args.reasoning_effort} if args.reasoning_effort else {"enabled": False}
    results = []
    async with BudgetedChat(args.out, [args.judge], args.budget, schema_retries=2,
                            concurrency=args.concurrency, reuse_receipts=args.reuse_receipts,
                            max_tokens=args.max_output_tokens, reasoning=reasoning,
                            request_timeout_seconds=args.request_timeout_seconds) as api:
        groups = exact_request_groups(jobs, lambda j: api_request_body(api, args.judge, j["messages"]))
        for group in groups.values():
            signature = {k: v for k, v in group["request"].items() if k != "messages"}
            if signature != old_audit["runtime_identity"]["request_settings"]:
                raise ValueError("new API request settings differ from A/C; no judge calls issued")
        if {"reasoning": reasoning, "max_tokens": args.max_output_tokens,
            "request_timeout_seconds": args.request_timeout_seconds} != old_audit["runtime_identity"]["runtime"]:
            raise ValueError("new API runtime differs from A/C; no judge calls issued")
        write_json(args.out / "manifest.json", {
            "schema": SCHEMA, "package_sha256": args.frozen_package_sha256,
            "jobs_sha256": package["jobs_sha256"], "judge_model": args.judge,
            "jobs_expected": len(jobs), "unique_requests_expected": len(groups),
            "repeats": 1, "source_sha256": file_hash(__file__),
            "runtime": {"reasoning": reasoning, "max_tokens": args.max_output_tokens,
                        "request_timeout_seconds": args.request_timeout_seconds},
            "reuse_scope": "Byte-identical full API requests share one receipt. Aliases and cached results are not independent stochastic replicates.",
        })

        async def one(request_hash, group):
            canonical = group["jobs"][0]
            receipt = await api.complete(args.judge, canonical["messages"], object_schema({"answer": ANSWER}),
                                         {"job": canonical["job_id"], "kind": canonical["kind"]})
            if receipt["request"] != group["request"] or receipt["request_sha256"] != request_hash:
                api.stopped = True
                raise ValueError("request builder diverged from exact dedup identity")
            try:
                completion = audit_completion(receipt)
            except ValueError:
                api.stopped = True
                raise
            answer = receipt["parsed"]["answer"]
            for job in group["jobs"]:
                row = {**{k: v for k, v in job.items() if k != "messages"},
                       "judge_model": args.judge, "answer": answer, "correct": answer == job["gold"],
                       "request_sha256": request_hash, "receipt_job_id": canonical["job_id"],
                       "exact_request_alias": job["job_id"] != canonical["job_id"],
                       "receipt_reused": bool(receipt.get("reused_from")),
                       "valid": True, "finish_reason": "stop", "completion_audit": completion}
                results.append(row)
                with (args.out / "evaluations.jsonl").open("a") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

        outcomes = await asyncio.gather(*(one(k, g) for k, g in groups.items()), return_exceptions=True)
        summary = {"expected": len(jobs), "completed": len(results),
                   "unique_requests_expected": len(groups), "receipt_rows": len(api.records),
                   "exact_request_aliases": sum(r["exact_request_alias"] for r in results),
                   "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
                   "cost_upper_usd": api.spent,
                   "finish_reasons": dict(Counter(r.get("finish_reason") for r in api.records))}
        write_json(args.out / "summary.json", summary)
        print(json.dumps(summary))
        if summary["errors"] or len(results) != len(jobs):
            raise RuntimeError("incomplete grid judgments; retained receipts, no aggregate")


def audit_results(directory, jobs, expected_jobs_hash, allow_extra=False):
    """Bind outcomes to frozen messages and exact declared native/API runtime."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["jobs_sha256"] != expected_jobs_hash:
        raise ValueError("judge run used different frozen jobs")
    all_rows = read_rows(directory / "evaluations.jsonl")
    by_id = keyed(all_rows)
    if len(all_rows) != manifest["jobs_expected"]:
        raise ValueError("incomplete judge result file")
    if not set(jobs) <= set(by_id) or (not allow_extra and set(jobs) != set(by_id)):
        raise ValueError("missing or extra grid result")
    api = (directory / "requests.jsonl").exists()
    receipts = {}
    signatures, providers, reported_models, fingerprints = set(), Counter(), Counter(), Counter()
    if api:
        for receipt in read_rows(directory / "requests.jsonl"):
            if receipt.get("valid"):
                key = (receipt["request_sha256"], receipt["tag"]["job"])
                if key in receipts:
                    raise ValueError("ambiguous valid API receipt")
                receipts[key] = receipt
        identity = {"type": "api", "model": manifest["judge_model"], "runtime": manifest["runtime"]}
    else:
        contract = manifest["contract"]
        if not contract["answer_only"]:
            raise ValueError("native judge must retain answer-only contract")
        identity = {"type": "native", "model": contract["model"], "contract": contract}
    result = {}
    for job_id, job in jobs.items():
        row = by_id[job_id]
        if any(row.get(k) != v for k, v in job.items() if k != "messages"):
            raise ValueError("evaluation metadata differs from frozen job")
        answer = row.get("answer", row.get("parsed", {}).get("answer"))
        if (row.get("valid") is not True or row.get("finish_reason") != "stop"
                or answer not in ("A", "B") or row.get("correct") is not (answer == job["gold"])):
            raise ValueError("invalid or inconsistent judge answer")
        if api:
            receipt_id = row.get("receipt_job_id", job_id)
            receipt = receipts[(row["request_sha256"], receipt_id)]
            audit_completion(receipt)
            request = receipt["request"]
            if (request["messages"] != job["messages"]
                    or sha256(json.dumps(request, sort_keys=True)) != row["request_sha256"]
                    or receipt["parsed"]["answer"] != answer
                    or receipt["model"] != identity["model"]
                    or request["model"] != identity["model"]):
                raise ValueError("API receipt differs from frozen judge outcome")
            if any(request[k] != manifest["runtime"][k] for k in ("reasoning", "max_tokens")):
                raise ValueError("API request differs from declared runtime")
            signatures.add(json.dumps({k: v for k, v in request.items() if k != "messages"}, sort_keys=True))
            payload = json.loads(receipt["response_body"])
            if strict_json(payload["choices"][0]["message"]["content"]) != {"answer": answer}:
                raise ValueError("complete provider output differs from parsed answer")
            providers[str(payload.get("provider"))] += 1
            reported_models[str(payload.get("model"))] += 1
            fingerprints[str(payload.get("system_fingerprint"))] += 1
        else:
            if (strict_json(row["raw"]) != {"answer": answer} or row["parsed"] != {"answer": answer}
                    or not 0 < row["completion_tokens"] < contract["max_new_tokens"]
                    or row["input_tokens"] + contract["max_new_tokens"] > manifest["context_limit"]):
                raise ValueError("native completion or context bound failed")
        result[job_id] = row
    if api:
        if len(signatures) != 1:
            raise ValueError("multiple API request runtimes within grid cells")
        identity["request_settings"] = json.loads(next(iter(signatures)))
    audit = {"runtime_identity": identity, "logical_grid_rows": len(result),
             "providers_as_reported": dict(providers), "models_as_reported": dict(reported_models),
             "fingerprints_as_reported": dict(fingerprints),
             "manifest_sha256": file_hash(directory / "manifest.json"),
             "evaluations_sha256": file_hash(directory / "evaluations.jsonl"),
             "unique_grid_request_hashes": len({r["request_sha256"] for r in result.values()}),
             "cached_or_aliased_grid_rows": sum(bool(r.get("reused") or r.get("receipt_reused")
                                                   or r.get("exact_request_alias")) for r in result.values()),
             "scope": "Native identity requires the complete frozen model/template/dtype/generation/source contract. This contract does not encode GPU hardware, driver, library versions or all CUDA determinism settings; matching it is not proof of identical numerical execution. API settings match exactly; provider routing, model revision, hardware and missing fingerprints remain unknown."}
    if api:
        audit["requests_sha256"] = file_hash(directory / "requests.jsonl")
    return result, audit


def grid_statistics(pairs, outcomes, judge):
    """Paired family-macro effects; correctness pairs are episode counts."""
    accuracies, contrasts, paired_rows = [], [], []
    for pair in pairs:
        row = {"judge": judge, "family": pair["family"],
               "participant_model": pair["participant_model"],
               "no_feedback_episode_id": pair["no_feedback_episode_id"],
               "feedback_episode_id": pair["feedback_episode_id"],
               "feedback_role_switches": pair.get("feedback_role_switches", []),
               **{cell: bool(outcomes[job]["correct"]) for cell, job in pair["jobs"].items()}}
        paired_rows.append(row)
    models = sorted({p["participant_model"] for p in paired_rows})
    for model in ["all", *models]:
        selected = [p for p in paired_rows if model == "all" or p["participant_model"] == model]
        for cell in ("A", "B", "C", "D"):
            stats = paired_stats([{"group": p["family"], "difference": int(p[cell])} for p in selected])
            accuracies.append({"judge": judge, "participant_model": model, "cell": cell,
                               "episodes": len(selected), "families": stats["parent_groups"],
                               "correct": sum(p[cell] for p in selected),
                               "accuracy": sum(p[cell] for p in selected) / len(selected),
                               "family_macro_accuracy": stats["mean_difference"],
                               "family_bootstrap_95_ci": stats["parent_bootstrap_95_ci"]})
        for name, (positive, negative) in PAIR_CONTRASTS.items():
            stats = paired_stats([{"group": p["family"], "difference": int(p[positive]) - int(p[negative])}
                                  for p in selected])
            contrasts.append({"judge": judge, "participant_model": model, "contrast": name,
                              "episodes": stats["episodes"], "families": stats["parent_groups"],
                              "difference": stats["mean_difference"],
                              "family_bootstrap_95_ci": stats["parent_bootstrap_95_ci"],
                              "fixes": stats["improved"], "harms": stats["harmed"],
                              "both_correct": sum(p[positive] and p[negative] for p in selected),
                              "both_wrong": sum(not p[positive] and not p[negative] for p in selected)})
        stats = paired_stats([{"group": p["family"], "difference":
                               (int(p["D"]) - int(p["B"])) - (int(p["C"]) - int(p["A"]))}
                              for p in selected])
        contrasts.append({"judge": judge, "participant_model": model, "contrast": "(D-B)-(C-A)",
                          "episodes": stats["episodes"], "families": stats["parent_groups"],
                          "difference": stats["mean_difference"],
                          "family_bootstrap_95_ci": stats["parent_bootstrap_95_ci"],
                          "positive_interaction_episodes": stats["improved"],
                          "negative_interaction_episodes": stats["harmed"]})
    return accuracies, contrasts, paired_rows


def analyse(args):
    package = check_package(args.package, args.frozen_package_sha256)
    if args.out.exists() or len(args.prior_results) != len(args.feedback_results):
        raise ValueError("new output and one old/new result pair per judge required")
    old_jobs = keyed(read_rows(args.package / "prior_jobs.jsonl"))
    new_jobs = keyed(read_rows(args.package / "jobs.jsonl"))
    pairs = json.loads((args.package / "pairs.json").read_text())
    audits, accuracies, contrasts, paired = [], [], [], []
    secondary_accuracy, secondary_contrasts = [], []
    judges = set()
    for old_path, new_path in zip(args.prior_results, args.feedback_results):
        old, old_audit = audit_results(old_path, old_jobs, prior_jobs_hash(package, old_path), True)
        new, new_audit = audit_results(new_path, new_jobs, package["jobs_sha256"])
        if old_audit["runtime_identity"] != new_audit["runtime_identity"]:
            raise ValueError("A/C and B/D judge runtime contracts differ")
        judge = old_audit["runtime_identity"]["model"]
        if judge in judges:
            raise ValueError("repeats are not independent judges")
        judges.add(judge)
        a, c, p = grid_statistics(pairs, {**old, **new}, judge)
        accuracies.extend(a); contrasts.extend(c); paired.extend(p)
        if package["verified_disclosure_episodes"]:
            clean = [p for p in pairs if p["feedback_episode_id"] not in package["verified_disclosure_episodes"]]
            if clean:
                a, c, _ = grid_statistics(clean, {**old, **new}, judge)
                secondary_accuracy.extend(a); secondary_contrasts.extend(c)
        audits.append({"judge": judge, "A_C": old_audit, "B_D": new_audit})
    args.out.mkdir(parents=True)
    report = {"schema": SCHEMA, "package_sha256": args.frozen_package_sha256,
              "episodes_per_judge": package["episodes"], "families": package["families"],
              "detector": package["detector"], "baseline": package["baseline"],
              "feedback_role_switches": package["feedback_role_switches"],
              "feedback_episodes_with_role_switch": package["feedback_episodes_with_role_switch"],
              "role_switch_policy": package["role_switch_policy"],
              "verified_disclosure_episodes": package["verified_disclosure_episodes"],
              "verified_disclosure_turns": package["verified_disclosure_turns"],
              "disclosure_policy": package["disclosure_policy"],
              "accuracy": accuracies, "contrasts": contrasts, "runtime_audits": audits,
              "secondary_clean_subset_accuracy": secondary_accuracy,
              "secondary_clean_subset_contrasts": secondary_contrasts,
              "independent_unit": package["independent_unit"],
              "limits": [package["scope"],
                         "Family bootstrap 95% intervals are unadjusted descriptive pilot intervals; perfect samples can have degenerate intervals.",
                         "Reused A/C and fresh B/D were collected at different times; unchanged API model labels cannot guarantee unchanged provider weights or routing.",
                         "The native runner contract does not include GPU variant, driver, library versions or every CUDA determinism setting. Separate runtime evidence and fresh replay diagnostics are needed to assess cross-run numerical drift.",
                         "This grid compares raw no-score A/B with real-score C/D, not matched-neutral formatting controls.",
                         "The detector is a wrong-answer advocacy proxy, not a verified local-lie or intent label."]}
    write_json(args.out / "analysis.json", report)
    write_rows(args.out / "paired_episodes.jsonl", paired)
    print(json.dumps({"judges": sorted(judges), "episodes_per_judge": package["episodes"],
                      "families": package["families"], "out": str(args.out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("prior-package", "prior-episodes", "feedback-episodes", "dataset",
                 "prior-bundle", "feedback-bundle"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--prior-package-sha256", required=True)
    prep.add_argument("--compatible-prior-package", type=Path, nargs="*", default=[])
    prep.add_argument("--compatible-prior-package-sha256", nargs="*", default=[])
    prep.add_argument("--disclosure-review", type=Path)
    prep.add_argument("--expected-episodes", type=int, default=48)
    prep.add_argument("--expected-families", type=int, default=24)
    run = sub.add_parser("run-api")
    run.add_argument("--judge", default="anthropic/claude-haiku-5.5")
    run.add_argument("--prior-results", type=Path, required=True)
    run.add_argument("--budget", type=float, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    run.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--request-timeout-seconds", type=float, default=300)
    analysis = sub.add_parser("analyse")
    analysis.add_argument("--prior-results", type=Path, nargs="+", required=True)
    analysis.add_argument("--feedback-results", type=Path, nargs="+", required=True)
    for command in (run, analysis):
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--frozen-package-sha256", required=True)
    for command in (prep, run, analysis):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-api":
        asyncio.run(run_api(args))
    else:
        analyse(args)


if __name__ == "__main__":
    main()
