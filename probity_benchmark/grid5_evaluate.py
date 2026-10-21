"""Versioned corpus expansion of the frozen iteration-4 feedback grid.

Preparation is offline. Questions remain paired, while inference clusters by
original source family and explicitly separates prior exposure. Existing prompt
builders, native execution, heads, normalization references and old outcomes are
not changed. A/C can be compiled and judged before B/D generation is complete.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path

from .api import ANSWER, BudgetedChat, object_schema
from . import grid4_evaluate as grid4
from .judge3_final import audit_completion, file_hash, read_rows
from .judge3_protocol import base_job, bundle_jobs, validate_bundle
from .preflight import load_problems, sha256


SCHEMA = "grid5_corpus_evaluation_v1"
INVENTORY_SCHEMA = "probity_iteration5_source_inventory_v1"
DIAGNOSTICS_SCHEMA = "grid5_secondary_diagnostics_v1"
SOURCE_STATUSES = ("supported", "ambiguous", "contradicted", "unreviewed")
MODELS = ("qwen/qwen3.5-27b", "qwen/qwen3.8-27b")
PROTOCOL = "explicit_reasoned_evidence_sequential"
SIGNAL_FIELDS = ("detector", "target", "granularity", "orientation",
                 "head_sha256", "reference_sha256", "references")
EXPOSURE_FLAGS = ("previous_training_family", "previous_development_family",
                  "previous_tuning_or_diagnostic_family", "previously_evaluated_family",
                  "family_untouched_before_iteration5")
EXPOSURE_STRATA = ("training", "development_reference", "other_tuning",
                   "previously_evaluated", "untouched")
RUNTIME_FILES = ("grid5_evaluate.py", "grid4_evaluate.py", "judge3_final.py",
                 "judge3_protocol.py", "judge_assistance.py", "api.py",
                 "preflight.py", "run_local_judge.py")


def exposure_stratum(row):
    flags = [row[key] for key in EXPOSURE_FLAGS]
    if any(type(value) is not bool for value in flags):
        raise ValueError("exposure flags must be explicit booleans")
    if flags[-1] != (not any(flags[:-1])):
        raise ValueError("untouched exposure conflicts with prior family use")
    return next(name for name, active in zip(EXPOSURE_STRATA, flags) if active)


def check_inventory(inventory, problems, source_rows, expected_questions,
                    expected_families, models):
    """Validate explicit family exposure independently of legacy split labels."""
    if (inventory.get("schema") != INVENTORY_SCHEMA
            or not str(inventory.get("status", "")).startswith("frozen")):
        raise ValueError("a frozen, source-reviewed exposure inventory is required")
    if not models or len(models) != len(set(models)):
        raise ValueError("unique participant models required")
    rows = grid4.keyed(inventory["rows"], "parent_problem_id")
    if len(rows) != expected_questions or len(problems) != expected_questions:
        raise ValueError("question inclusion differs from the frozen inventory")
    if len({r["family_id"] for r in rows.values()}) != expected_families:
        raise ValueError("source-family count differs from the frozen inventory")
    if {r["problem_id"] for r in rows.values()} != set(problems):
        raise ValueError("inventory canonical problem identities differ from source")
    source_by_parent = grid4.keyed(source_rows, "parent_problem_id")
    if set(source_by_parent) != set(rows):
        raise ValueError("source parent question identities differ from inventory")
    family_flags = {}
    for parent, row in rows.items():
        if not row.get("family_id") or not row.get("family_name"):
            raise ValueError("explicit original source family identity required")
        stratum = exposure_stratum(row)
        flags = tuple(row[key] for key in EXPOSURE_FLAGS)
        previous = family_flags.setdefault(row["family_id"], flags)
        if previous != flags:
            raise ValueError("questions from one family have inconsistent exposure")
        if (not isinstance(row.get("historical_exposure_labels"), list)
                or not all(isinstance(x, str) for x in row["historical_exposure_labels"])):
            raise ValueError("historical exposure labels must be retained")
        if type(row.get("prior24")) is not bool:
            raise ValueError("exact prior-question reuse must be explicit")
        if row["prior24"] and (stratum == "untouched" or not row["previously_evaluated_family"]):
            raise ValueError("cached question cannot be an untouched family")
        source = source_by_parent[parent]
        problem = problems[row["problem_id"]]
        if (source.get("parent_source_line") != row["parent_source_line"]
                or source.get("split") != row["split"]
                or row.get("source_sha256") != sha256(problem.story)
                or row.get("question_sha256") != sha256(problem.question)):
            raise ValueError("source content or parent provenance differs from inventory")
        if row["prior24"] and set(row.get("cached", {})) != set(models):
            raise ValueError("cached question lacks both model identities")
        if not row["prior24"] and row.get("cached"):
            raise ValueError("new question unexpectedly claims cached identities")
    counts = inventory["counts"]
    derived = {
        "questions": len(rows), "families": len(family_flags),
        "reused_prior_questions": sum(r["prior24"] for r in rows.values()),
        "new_questions": sum(not r["prior24"] for r in rows.values()),
        "untouched_question_rows_before_iteration5": sum(
            r["family_untouched_before_iteration5"] for r in rows.values()),
        "untouched_families_before_iteration5": len({r["family_id"] for r in rows.values()
                                                     if r["family_untouched_before_iteration5"]}),
    }
    if any(counts.get(key) != value for key, value in derived.items()):
        raise ValueError("inventory exposure counts are inconsistent")
    return rows


def load_inventory(path, expected_hash, source, expected_questions, expected_families, models):
    if file_hash(path) != expected_hash:
        raise ValueError("inventory changed after its reviewed freeze")
    inventory = json.loads(path.read_text())
    if inventory["source_outputs"]["all166"]["sha256"] != file_hash(source):
        raise ValueError("full source changed after inventory freeze")
    problems, audit = load_problems(source)
    if audit["errors"] or audit["duplicate_problem_hashes"]:
        raise ValueError("invalid or duplicate source questions")
    problems = {p.problem_id: p for p in problems}
    rows = check_inventory(inventory, problems, read_rows(source), expected_questions,
                           expected_families, models)
    return inventory, rows, problems, audit


def load_episode_sources(directories):
    episodes, paths = {}, {}
    for directory in directories:
        for path in sorted(Path(directory).glob("episode_*.json")):
            episode = json.loads(path.read_text())
            eid = episode["episode_id"]
            if eid in episodes:
                raise ValueError("duplicate episode across source directories")
            episodes[eid], paths[eid] = episode, path.resolve()
    if not episodes:
        raise ValueError("no complete episode sources supplied")
    return episodes, paths


def check_episode_coverage(episodes, inventory_rows, models, feedback=False, paths=None,
                           inventory=None):
    expected = {(parent, model) for parent in inventory_rows for model in models}
    seen, families = set(), {}
    for eid, episode in episodes.items():
        key = (episode["parent_problem_id"], episode["model"])
        if key not in expected or key in seen:
            raise ValueError("missing, extra or repeated question/model episode")
        row = inventory_rows[key[0]]
        if (episode["problem_id"] != row["problem_id"]
                or episode["parent_source_line"] != row["parent_source_line"]
                or episode["split"] != row["split"] or episode["world"] != "original"):
            raise ValueError("episode changed the frozen source or question identity")
        if (episode.get("complete") is not True or type(episode.get("knowledge_correct")) is not bool
                or len(episode.get("turns", [])) != 6):
            raise ValueError("incomplete six-turn episode or missing knowledge diagnostic")
        if any(t.get("answer") not in ("A", "B") or t.get("assigned_answer") not in ("A", "B")
               or not t.get("argument") for t in episode["turns"]):
            raise ValueError("invalid complete public turn")
        if (episode["gold"] != row["displayed_gold"]
                or episode["turns"][0]["speaker"] != row["first_speaker"]):
            raise ValueError("episode changed the frozen answer/order schedule")
        first = row["first_speaker"]
        if first not in ("Red", "Blue"):
            raise ValueError("unknown opening speaker")
        second = "Blue" if first == "Red" else "Red"
        if any(t["speaker"] != (first, second)[i % 2] or t.get("round") != i // 2 + 1
               for i, t in enumerate(episode["turns"])):
            raise ValueError("episode changed the three-round sequential schedule")
        if row["prior24"]:
            prefix = "feedback" if feedback else "no_feedback"
            cached = row["cached"][episode["model"]]
            if eid != cached[prefix + "_episode_id"]:
                raise ValueError("cached episode identity was regenerated or replaced")
            if paths is not None and inventory is not None:
                expected_hash = inventory["evidence"][cached[prefix + "_path"]]["sha256"]
                if file_hash(paths[eid]) != expected_hash:
                    raise ValueError("cached episode bytes changed")
        seen.add(key)
        families[eid] = row["family_id"]
    if seen != expected:
        raise ValueError("missing question/model episodes; no partial inclusion allowed")
    return families


def check_frozen_signal(bundle, reference, episodes, inventory_rows):
    if any(bundle.get(key) != reference.get(key) for key in SIGNAL_FIELDS):
        raise ValueError("signal changed the frozen head or development reference")
    if (bundle["detector"] != "adapted_primary" or bundle["granularity"] != "response"
            or bundle["target"] != "wrong_answer_advocacy"):
        raise ValueError("dataset-only expansion requires the adapted response probe")
    _, covered = validate_bundle(bundle, episodes)
    if covered != set(episodes) or set(bundle["references"]) != {e["model"] for e in episodes.values()}:
        raise ValueError("frozen signal does not cover every included model and turn")
    reference_families = set()
    for ref in reference["references"].values():
        for parent in ref["parents"]:
            if parent not in inventory_rows:
                raise ValueError("normalization parent missing from exposure inventory")
            row = inventory_rows[parent]
            if row["family_untouched_before_iteration5"] or not row["previous_development_family"]:
                raise ValueError("normalization family must be explicitly development-exposed")
            reference_families.add(row["family_id"])
    return {"head_sha256": reference["head_sha256"],
            "reference_sha256": reference["reference_sha256"],
            "normalization_reference_families": sorted(reference_families),
            "reference_overlap_policy": "Retain these exposed families in exploratory coverage only; no fitting, refitting, renormalization or threshold selection on expansion outcomes.",
            "turns": len(bundle["rows"])}


def merge_score_bundles(cached, new, expected_cached=288, expected_new=1704):
    """Append new scalar rows without rewriting old values or frozen metadata."""
    if {k: v for k, v in cached.items() if k != "rows"} != {k: v for k, v in new.items() if k != "rows"}:
        raise ValueError("new score export changed frozen bundle metadata")
    if len(cached["rows"]) != expected_cached or len(new["rows"]) != expected_new:
        raise ValueError("old/new scalar coverage differs from frozen counts")
    combined = copy.deepcopy(cached)
    combined["rows"].extend(copy.deepcopy(new["rows"]))
    keys = {(r["episode_id"], r["turn_index"]) for r in combined["rows"]}
    if len(keys) != len(combined["rows"]):
        raise ValueError("duplicate or overlapping old/new scalar identities")
    return combined


def pair_exposure(row):
    return {"matched_question_id": row["parent_problem_id"],
            "family": row["family_id"], "family_name": row["family_name"],
            "exposure_stratum": exposure_stratum(row),
            "historical_exposure_labels": row["historical_exposure_labels"],
            "exposure_flags": {key: row[key] for key in EXPOSURE_FLAGS},
            "prior24_question": row["prior24"],
            "generation_provenance": "reused_iteration3_and_4" if row["prior24"] else "new_iteration5"}


def role_switches(episode):
    return [{"turn_index": i, "speaker": t["speaker"], "assigned_answer": t["assigned_answer"],
             "answer": t["answer"]} for i, t in enumerate(episode["turns"])
            if t["answer"] != t["assigned_answer"]]


def render_bases(episodes, problems):
    """Reuse the old renderer while retaining failed eligibility and concessions.

    The temporary copy bypasses legacy admission gates; it is never written as
    an episode or eligibility artifact. Only role-proxy recommendations use its
    assigned answers. Final-judge text contains every actual public answer.
    For adherent, knowledgeable old episodes the output equals base_job exactly.
    """
    bases, assigned = {}, {}
    for eid, episode in episodes.items():
        view = copy.deepcopy(episode)
        view["knowledge_correct"] = True
        for turn in view["turns"]:
            turn["answer"] = turn["assigned_answer"]
        problem = problems[episode["problem_id"]]
        base = base_job(problem, view, PROTOCOL)
        messages, _ = problem.view(False, episode["swapped"], structured=True)
        transcript = "\n\n".join(
            f"Turn {i + 1}. {turn['speaker']} (answer {turn['answer']}):\n" + turn["argument"]
            for i, turn in enumerate(episode["turns"]))
        base["messages"][1]["content"] = messages[1]["content"] + "\n\nPublic debate:\n" + transcript
        bases[eid], assigned[eid] = base, view
    return bases, assigned


def baseline_jobs(episodes, problems, bundle, inventory_rows, original_jobs):
    bases, assigned = render_bases(episodes, problems)
    signals, _ = bundle_jobs(bundle, assigned, bases)
    real = {j["episode_id"]: j for j in signals if j["condition"] == "real"}
    originals = grid4.keyed(original_jobs)
    jobs, pairs, retained = [], [], set()
    for eid in sorted(episodes):
        episode = episodes[eid]
        row = inventory_rows[episode["parent_problem_id"]]
        pair = {**pair_exposure(row), "no_feedback_episode_id": eid,
                "participant_model": episode["model"],
                "parent_problem_id": episode["parent_problem_id"], "gold": episode["gold"],
                "no_feedback_knowledge_correct": episode["knowledge_correct"],
                "no_feedback_role_switches": role_switches(episode),
                "jobs": {}}
        for cell, job in (("A", bases[eid]), ("C", real[eid])):
            if row["prior24"]:
                if originals.get(job["job_id"]) != job:
                    raise ValueError("cached A/C request or job metadata changed")
                retained.add(job["job_id"])
            elif job["job_id"] in originals:
                raise ValueError("new question collides with cached A/C identity")
            jobs.append(job)
            pair["jobs"][cell] = job["job_id"]
        pairs.append(pair)
    if retained != set(originals):
        raise ValueError("original A/C inclusion was changed")
    grid4.keyed(jobs)
    return jobs, pairs


def feedback_jobs(prior, feedback, families, problems, bundle, prior_jobs, inventory_rows):
    """Original grid-4 job semantics, with eligibility diagnostics never filtering."""
    pairs = grid4.pair_episodes(prior, feedback, families)
    originals = {(job["episode_id"], "A" if job["kind"] == "none" else "C"): job
                 for job in prior_jobs}
    if len(originals) != len(prior_jobs) or set(originals) != {(eid, c) for eid in prior for c in "AC"}:
        raise ValueError("baseline lacks unique A/C jobs")
    bases, assigned = render_bases(feedback, problems)
    signals, _ = bundle_jobs(bundle, assigned, bases)
    real = {job["episode_id"]: job for job in signals if job["condition"] == "real"}
    jobs = []
    for pair in pairs:
        old_id, new_id = pair["no_feedback_episode_id"], pair["feedback_episode_id"]
        pair.update(pair_exposure(inventory_rows[pair["parent_problem_id"]]))
        pair.update(no_feedback_knowledge_correct=prior[old_id]["knowledge_correct"],
                    feedback_knowledge_correct=feedback[new_id]["knowledge_correct"],
                    no_feedback_role_switches=role_switches(prior[old_id]),
                    jobs={cell: originals[(old_id, cell)]["job_id"] for cell in "AC"})
        for cell, source in (("B", bases[new_id]), ("D", real[new_id])):
            job = copy.deepcopy(source)
            job.update(grid_cell=cell, matched_no_feedback_episode_id=old_id,
                       source_family=pair["family"], feedback_protocol=feedback[new_id].get("protocol"),
                       role_mapping="original assigned speaker-to-answer mapping",
                       feedback_role_switches=pair["feedback_role_switches"],
                       grid_raw_baseline_job_id=bases[new_id]["job_id"])
            jobs.append(job)
            pair["jobs"][cell] = job["job_id"]
    grid4.keyed(prior_jobs + jobs)
    return jobs, pairs


def input_hashes(paths):
    return {str(Path(path).resolve()): file_hash(path) for path in paths}


def runtime_hashes():
    return input_hashes(Path(__file__).parent / name for name in RUNTIME_FILES)


def write_package(out, manifest, jobs, pairs, all_jobs=None):
    if out.exists():
        raise ValueError("immutable new package directory required")
    out.mkdir(parents=True)
    grid4.write_rows(out / "jobs.jsonl", jobs)
    grid4.write_json(out / "pairs.json", pairs)
    manifest.update(schema=SCHEMA, jobs_sha256=file_hash(out / "jobs.jsonl"),
                    pairs_sha256=file_hash(out / "pairs.json"), jobs_expected=len(jobs),
                    runtime_source_sha256=runtime_hashes())
    if all_jobs is not None:
        grid4.keyed(all_jobs)
        grid4.write_rows(out / "all_jobs.jsonl", all_jobs)
        manifest.update(all_jobs_sha256=file_hash(out / "all_jobs.jsonl"),
                        all_jobs_expected=len(all_jobs))
    grid4.write_json(out / "package.json", manifest)
    result = {"stage": manifest["stage"], "package_sha256": file_hash(out / "package.json"),
              "questions": manifest["questions"], "families": manifest["families"],
              "matched_units": len(pairs), "jobs": len(jobs), "paid_calls": 0}
    print(json.dumps(result))
    return result


def prepare_baseline(args):
    if args.out.exists():
        raise ValueError("immutable new package directory required")
    models = tuple(args.models)
    inventory, rows, problems, source_audit = load_inventory(
        args.inventory, args.inventory_sha256, args.source,
        args.expected_questions, args.expected_families, models)
    original = grid4.check_package(args.original_grid_package,
                                   args.original_grid_package_sha256)
    reference = json.loads(args.reference_bundle.read_text())
    if file_hash(args.reference_bundle) != args.reference_bundle_sha256:
        raise ValueError("original reference artifact changed")
    if args.reference_bundle_sha256 not in original["input_sha256"].values():
        raise ValueError("reference artifact was not an input to the completed grid")
    if any(original[key] != reference[key] for key in ("detector", "head_sha256", "reference_sha256")):
        raise ValueError("reference differs from completed iteration-4 grid")
    episodes, paths = load_episode_sources(args.episodes)
    families = check_episode_coverage(episodes, rows, models, paths=paths, inventory=inventory)
    bundle = json.loads(args.bundle.read_text())
    signal = check_frozen_signal(bundle, reference, episodes, rows)
    original_jobs = read_rows(args.original_grid_package / "prior_jobs.jsonl")
    jobs, pairs = baseline_jobs(episodes, problems, bundle, rows, original_jobs)
    files = [args.inventory, args.source, args.bundle, args.reference_bundle,
             args.original_grid_package / "package.json",
             args.original_grid_package / "prior_jobs.jsonl",
             args.original_grid_package / "jobs.jsonl", *paths.values()]
    manifest = {
        "stage": "baseline", "questions": len(rows), "families": args.expected_families,
        "episodes": len(episodes), "participant_models": list(models),
        "inventory": str(args.inventory.resolve()), "inventory_sha256": args.inventory_sha256,
        "source": str(args.source.resolve()), "source_audit": source_audit,
        "episode_paths": {eid: str(path) for eid, path in paths.items()},
        "episode_families": families, "bundle": str(args.bundle.resolve()),
        "reference_bundle": str(args.reference_bundle.resolve()),
        "reference_bundle_sha256": args.reference_bundle_sha256,
        "original_grid_package": str(args.original_grid_package.resolve()),
        "original_grid_package_sha256": args.original_grid_package_sha256,
        "original_primary_job_ids": sorted(j["job_id"] for j in original_jobs),
        "signal": signal, "input_sha256": input_hashes(files),
        "analysis_contract": "Family-macro primary; question-weighted secondary. All 166 question rows are exploratory corpus coverage, not 166 independent or unseen families. Exposure is frozen before expansion outcomes, independent of legacy split metadata.",
        "scope": "One frozen no-feedback trajectory per question/model; exact old A/C retained. Complete failed-knowledge or role-switch trajectories remain included with actual diagnostics. No neutral/shuffled/oracle jobs added to the requested grid.",
        "eligibility_rendering": "An internal, unsaved compatibility copy bypasses the legacy knowledge/opposing-public-answer admission gates. Actual public answer fields and arguments are restored verbatim, while original assigned roles determine only the proxy recommendation. Actual knowledge and role diagnostics remain unchanged in pairs.json and original episodes.",
        "secondary_diagnostics_schema": DIAGNOSTICS_SCHEMA,
    }
    return write_package(args.out, manifest, jobs, pairs)


def check_package(directory, expected_hash, check_runtime=False):
    if file_hash(directory / "package.json") != expected_hash:
        raise ValueError("expansion package changed after review")
    package = json.loads((directory / "package.json").read_text())
    if package.get("schema") != SCHEMA or package.get("stage") not in ("baseline", "feedback"):
        raise ValueError("unknown expansion package")
    for name, key in (("jobs.jsonl", "jobs_sha256"), ("pairs.json", "pairs_sha256")):
        if file_hash(directory / name) != package[key]:
            raise ValueError("frozen expansion output changed: " + name)
    if "all_jobs_sha256" in package and file_hash(directory / "all_jobs.jsonl") != package["all_jobs_sha256"]:
        raise ValueError("frozen combined native jobs changed")
    for filename, digest in package["input_sha256"].items():
        if file_hash(filename) != digest:
            raise ValueError("frozen expansion input changed: " + filename)
    if check_runtime:
        for filename, digest in package["runtime_source_sha256"].items():
            if file_hash(filename) != digest:
                raise ValueError("frozen expansion runtime changed: " + filename)
    return package


def prepare_feedback(args):
    if args.out.exists():
        raise ValueError("immutable new package directory required")
    baseline = check_package(args.baseline_package, args.baseline_package_sha256, True)
    if baseline["stage"] != "baseline":
        raise ValueError("A/C baseline package required")
    inventory, rows, problems, _ = load_inventory(
        Path(baseline["inventory"]), baseline["inventory_sha256"], Path(baseline["source"]),
        baseline["questions"], baseline["families"], baseline["participant_models"])
    prior = {eid: json.loads(Path(path).read_text()) for eid, path in baseline["episode_paths"].items()}
    feedback, paths = load_episode_sources(args.feedback_episodes)
    check_episode_coverage(feedback, rows, baseline["participant_models"], True, paths, inventory)
    bundle = json.loads(args.feedback_bundle.read_text())
    prior_bundle = json.loads(Path(baseline["bundle"]).read_text())
    signal = check_frozen_signal(bundle, prior_bundle, feedback, rows)
    original_dir = Path(baseline["original_grid_package"])
    original = grid4.check_package(original_dir, baseline["original_grid_package_sha256"])
    old_jobs, _ = baseline_jobs(prior, problems, prior_bundle, rows,
                                read_rows(original_dir / "prior_jobs.jsonl"))
    if grid4.keyed(old_jobs) != grid4.keyed(read_rows(args.baseline_package / "jobs.jsonl")):
        raise ValueError("feedback preparation changed A/C jobs")
    jobs, pairs = feedback_jobs(prior, feedback, baseline["episode_families"], problems,
                                bundle, old_jobs, rows)
    original_jobs = grid4.keyed(read_rows(original_dir / "jobs.jsonl"))
    new_by_id = grid4.keyed(jobs)
    if any(new_by_id.get(jid) != job for jid, job in original_jobs.items()):
        raise ValueError("cached B/D request or metadata changed")
    reviews, review_files = {}, []
    review_sources = args.disclosure_review or [Path(p) / "disclosure_adjudications"
                                               for p in args.feedback_episodes]
    for directory in review_sources:
        more, files = grid4.load_disclosure_reviews(directory)
        if any(key in reviews and reviews[key] != value for key, value in more.items()):
            raise ValueError("conflicting disclosure reviews across runs")
        reviews.update(more)
        review_files.extend(files)
    disclosures = grid4.disclosure_audit(feedback, reviews)
    files = [args.baseline_package / "package.json", args.baseline_package / "jobs.jsonl",
             args.baseline_package / "pairs.json", args.feedback_bundle,
             *paths.values(), *review_files]
    manifest = {
        **{k: baseline[k] for k in ("questions", "families", "episodes", "participant_models",
                                    "inventory", "inventory_sha256", "source", "reference_bundle",
                                    "reference_bundle_sha256", "original_grid_package",
                                    "original_grid_package_sha256", "analysis_contract",
                                    "eligibility_rendering", "secondary_diagnostics_schema")},
        "stage": "feedback", "baseline_package": str(args.baseline_package.resolve()),
        "baseline_package_sha256": args.baseline_package_sha256,
        "bundle": str(args.feedback_bundle.resolve()), "signal": signal,
        "input_sha256": {**baseline["input_sha256"], **input_hashes(files)},
        "original_primary_job_ids": sorted(original_jobs),
        "public_disclosure_review": disclosures,
        "disclosure_flagged_episodes": sorted({d["episode_id"] for d in disclosures
                                               if d["review"]["discloses_scores"]}),
        "disclosure_interpretation": "Review dispositions can include conservative automatic retain-with-disclosure flags. These are not necessarily verified semantic leaks. Every flagged trajectory remains in primary results with unchanged text.",
        "feedback_role_switches": sum(len(p["feedback_role_switches"]) for p in pairs),
        "protocol_deviation_policy": "Retain all declared pairs and actual public answers. Original assigned roles determine only the role-proxy recommendation. Disclosure or concession is a reported deviation, not an exclusion.",
        "scope": "Same frozen protocol and signal; one no-feedback and one feedback trajectory per question/model. Old 24 questions reuse both trajectories and primary outcomes. New questions use newly generated trajectories; generation timing/routing is not randomized by this compiler.",
    }
    return write_package(args.out, manifest, jobs, pairs, all_jobs=old_jobs + jobs)


def original_results(package, directory, stage):
    original_dir = Path(package["original_grid_package"])
    original = grid4.check_package(original_dir, package["original_grid_package_sha256"])
    jobs_file = "prior_jobs.jsonl" if stage == "baseline" else "jobs.jsonl"
    jobs = grid4.keyed(read_rows(original_dir / jobs_file))
    expected_hash = (grid4.prior_jobs_hash(original, directory) if stage == "baseline"
                     else original["jobs_sha256"])
    return grid4.audit_results(directory, jobs, expected_hash, allow_extra=stage == "baseline")


def selected_receipts(directory, outcomes):
    receipts = {(r["request_sha256"], r["tag"]["job"]): r
                for r in read_rows(directory / "requests.jsonl") if r.get("valid")}
    return {jid: receipts[(row["request_sha256"], row.get("receipt_job_id", jid))]
            for jid, row in outcomes.items()}


def execution_groups(jobs, body_for_job, reusable, original_receipts):
    """Retain each historical outcome; deduplicate only the new presentations."""
    by_id = grid4.keyed(jobs)
    if not set(original_receipts) <= set(by_id):
        raise ValueError("cached primary jobs missing from expansion")
    groups = []
    for jid, receipt in sorted(original_receipts.items()):
        job = by_id[jid]
        body = body_for_job(job)
        key = sha256(json.dumps(body, sort_keys=True))
        tag = {"job": jid, "kind": job["kind"]}
        cached = reusable.get((key, json.dumps(tag, sort_keys=True)))
        if (body != receipt["request"] or key != receipt["request_sha256"]
                or cached is None or cached[1]["request"] != body
                or cached[1]["response_body"] != receipt["response_body"]):
            raise ValueError("exact original primary receipt unavailable; refuse replay or replacement")
        groups.append({"request_sha256": key, "request": body, "jobs": [job]})
    fresh = [j for j in jobs if j["job_id"] not in original_receipts]
    for key, group in grid4.exact_request_groups(fresh, body_for_job).items():
        eligible = [j for j in group["jobs"] if
                    (key, json.dumps({"job": j["job_id"], "kind": j["kind"]}, sort_keys=True)) in reusable]
        if eligible:
            canonical = eligible[0]
            group["jobs"] = [canonical, *[j for j in group["jobs"] if j != canonical]]
        groups.append({"request_sha256": key, **group})
    return groups


async def run_api(args):
    package = check_package(args.package, args.frozen_package_sha256, True)
    if args.out.exists() or not os.environ.get("PROBITY_PILOT_LEDGER"):
        raise ValueError("new output directory and shared budget ledger required")
    _, reference_audit = original_results(package, args.runtime_reference, "baseline")
    cached_outcomes, cached_audit = original_results(package, args.cached_results, package["stage"])
    identity = reference_audit["runtime_identity"]
    if identity["type"] != "api" or cached_audit["runtime_identity"] != identity:
        raise ValueError("cached and reference API judge runtimes differ")
    originals = selected_receipts(args.cached_results, cached_outcomes)
    jobs = read_rows(args.package / "jobs.jsonl")
    reasoning = {"effort": args.reasoning_effort} if args.reasoning_effort else {"enabled": False}
    runtime = {"reasoning": reasoning, "max_tokens": args.max_output_tokens,
               "request_timeout_seconds": args.request_timeout_seconds}
    if identity["model"] != args.judge or runtime != identity["runtime"]:
        raise ValueError("new judge runtime differs from original A/C")
    results = []
    async with BudgetedChat(args.out, [args.judge], args.budget, schema_retries=0,
                            concurrency=args.concurrency, reuse_receipts=args.reuse_receipts,
                            max_tokens=args.max_output_tokens, reasoning=reasoning,
                            request_timeout_seconds=args.request_timeout_seconds) as api:
        groups = execution_groups(
            jobs, lambda j: grid4.api_request_body(api, args.judge, j["messages"]),
            api.reusable, originals)
        if any({k: v for k, v in g["request"].items() if k != "messages"}
               != identity["request_settings"] for g in groups):
            raise ValueError("new complete request settings differ; no judge calls issued")
        grid4.write_json(args.out / "manifest.json", {
            "schema": SCHEMA, "stage": package["stage"],
            "package_sha256": args.frozen_package_sha256, "jobs_sha256": package["jobs_sha256"],
            "judge_model": args.judge, "jobs_expected": len(jobs), "repeats": 1,
            "runtime": runtime, "source_sha256": file_hash(__file__),
            "schema_retries": 0,
            "original_primary_rows": len(originals), "request_groups_expected": len(groups),
            "reference_runtime_audit": reference_audit,
            "cached_primary_audit": cached_audit,
            "reuse_scope": "Exact full request and canonical job tag; original primary response bodies are immutable. New identical requests may share a receipt. Reuse and aliases are not independent repetitions.",
        })

        async def one(group):
            canonical = group["jobs"][0]
            receipt = await api.complete(args.judge, canonical["messages"], object_schema({"answer": ANSWER}),
                                         {"job": canonical["job_id"], "kind": canonical["kind"]})
            if (receipt["request"] != group["request"]
                    or receipt["request_sha256"] != group["request_sha256"]):
                api.stopped = True
                raise ValueError("complete request differs from compiled identity")
            try:
                completion = audit_completion(receipt)
            except ValueError:
                api.stopped = True
                raise
            answer = receipt["parsed"]["answer"]
            for job in group["jobs"]:
                row = {**{k: v for k, v in job.items() if k != "messages"},
                       "judge_model": args.judge, "answer": answer, "correct": answer == job["gold"],
                       "request_sha256": group["request_sha256"],
                       "receipt_job_id": canonical["job_id"],
                       "exact_request_alias": canonical["job_id"] != job["job_id"],
                       "receipt_reused": bool(receipt.get("reused_from")),
                       "valid": True, "finish_reason": "stop", "completion_audit": completion}
                results.append(row)
                with (args.out / "evaluations.jsonl").open("a") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")

        outcomes = await asyncio.gather(*(one(group) for group in groups), return_exceptions=True)
        summary = {"expected": len(jobs), "completed": len(results),
                   "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
                   "receipt_rows": len(api.records), "cost_upper_usd": api.spent,
                   "cached_rows": sum(r["receipt_reused"] for r in results),
                   "finish_reasons": dict(Counter(r.get("finish_reason") for r in api.records))}
        grid4.write_json(args.out / "summary.json", summary)
        print(json.dumps(summary))
        if summary["errors"] or len(results) != len(jobs):
            raise RuntimeError("incomplete expansion judgments; no aggregate released")


def validate_secondary_diagnostics(diagnostics, inventory_sha256, pairs):
    if (diagnostics.get("schema") != DIAGNOSTICS_SCHEMA
            or diagnostics.get("inventory_sha256") != inventory_sha256
            or diagnostics.get("outcome_blind_to_judges") is not True):
        raise ValueError("secondary diagnostics require frozen inventory and judge-blind provenance")
    rows = grid4.keyed(diagnostics["rows"], "matched_question_id")
    questions = {p["matched_question_id"] for p in pairs}
    models = {p["participant_model"] for p in pairs}
    if set(rows) != questions:
        raise ValueError("diagnostics must account for every frozen question")
    for row in rows.values():
        if (row.get("source_status") not in SOURCE_STATUSES or not row.get("source_reason")
                or set(row.get("eligibility_by_model", {})) != models
                or set(row.get("eligibility_reason_by_model", {})) != models):
            raise ValueError("incomplete source validity or model eligibility diagnostic")
        if any(value is not None and type(value) is not bool
               for value in row["eligibility_by_model"].values()):
            raise ValueError("model eligibility must be true, false or unknown/null")
        if any(not isinstance(reason, str) or not reason
               for reason in row["eligibility_reason_by_model"].values()):
            raise ValueError("eligibility diagnostic requires explicit reasons")
        if any(key in row for key in ("gold", "alternative_gold", "correct_answer")):
            raise ValueError("alternative keys require a separate explicit sensitivity analysis")
    return rows


def audit_stage_results(directory, jobs, stage_hash, full_package):
    """Allow one unchanged native run over the bound A/C+B/D job union."""
    manifest = json.loads((directory / "manifest.json").read_text())
    combined = ("contract" in manifest
                and manifest["jobs_sha256"] == full_package.get("all_jobs_sha256"))
    if combined and manifest["jobs_expected"] != full_package["all_jobs_expected"]:
        raise ValueError("combined native run has incomplete declared coverage")
    return grid4.audit_results(directory, jobs,
                               full_package["all_jobs_sha256"] if combined else stage_hash,
                               allow_extra=combined)


def stratified_statistics(pairs, outcomes, judge, diagnostics=None):
    """Preserve paired clusters; explicitly name the macro and micro estimands."""
    cohorts = {"all": pairs}
    cohorts["exposure:previously_used"] = [p for p in pairs
        if not p["exposure_flags"]["family_untouched_before_iteration5"]]
    for name in EXPOSURE_STRATA:
        cohorts["exposure:" + name] = [p for p in pairs if p["exposure_stratum"] == name]
    cohorts["questions:prior24"] = [p for p in pairs if p["prior24_question"]]
    cohorts["questions:new142"] = [p for p in pairs if not p["prior24_question"]]
    for flag in EXPOSURE_FLAGS:
        cohorts["flag:" + flag] = [p for p in pairs if p["exposure_flags"][flag]]
    if diagnostics is not None:
        supported = [p for p in pairs if diagnostics[p["matched_question_id"]]["source_status"] == "supported"]
        cohorts["secondary:source_supported"] = supported
        cohorts["secondary:source_supported_and_eligible"] = [p for p in supported
            if diagnostics[p["matched_question_id"]]["eligibility_by_model"][p["participant_model"]] is True
            and p["no_feedback_knowledge_correct"] and p["feedback_knowledge_correct"]]
    accuracy, contrasts, paired = [], [], []
    for cohort, selected in cohorts.items():
        if not selected:
            continue
        a, c, p = grid4.grid_statistics(selected, outcomes, judge)
        by_episode = {row["no_feedback_episode_id"]: row for row in p}
        for row in a:
            chosen = [item for item in selected if row["participant_model"] == "all"
                      or item["participant_model"] == row["participant_model"]]
            row.update(cohort=cohort, questions=len({x["matched_question_id"] for x in chosen}),
                       question_weighted_accuracy=row["accuracy"],
                       accuracy=row["family_macro_accuracy"],
                       primary_estimator="equal-weight original source-family mean")
        for row in c:
            chosen = [item for item in selected if row["participant_model"] == "all"
                      or item["participant_model"] == row["participant_model"]]
            values = []
            for item in chosen:
                observed = by_episode[item["no_feedback_episode_id"]]
                if row["contrast"] in grid4.PAIR_CONTRASTS:
                    positive, negative = grid4.PAIR_CONTRASTS[row["contrast"]]
                    values.append(int(observed[positive]) - int(observed[negative]))
                else:
                    values.append(int(observed["D"]) - int(observed["B"])
                                  - int(observed["C"]) + int(observed["A"]))
            row.update(cohort=cohort, questions=len({x["matched_question_id"] for x in chosen}),
                       family_macro_difference=row["difference"],
                       question_weighted_difference=sum(values) / len(values),
                       primary_estimator="equal-weight original source-family paired difference")
        accuracy.extend(a)
        contrasts.extend(c)
        if cohort == "all":
            for row, pair in zip(p, selected):
                row.update({key: pair[key] for key in ("matched_question_id", "exposure_stratum",
                           "exposure_flags", "historical_exposure_labels", "prior24_question",
                           "generation_provenance", "no_feedback_knowledge_correct",
                           "feedback_knowledge_correct", "no_feedback_role_switches")})
                if diagnostics is not None:
                    row["secondary_diagnostics"] = diagnostics[pair["matched_question_id"]]
            paired = p
    return accuracy, contrasts, paired


def analyse(args):
    package = check_package(args.package, args.frozen_package_sha256)
    if (package["stage"] != "feedback" or args.out.exists()
            or len(args.baseline_results) != len(args.feedback_results)):
        raise ValueError("complete feedback package, paired judge runs and new output required")
    baseline_dir = Path(package["baseline_package"])
    baseline = check_package(baseline_dir, package["baseline_package_sha256"])
    old_jobs = grid4.keyed(read_rows(baseline_dir / "jobs.jsonl"))
    new_jobs = grid4.keyed(read_rows(args.package / "jobs.jsonl"))
    pairs = json.loads((args.package / "pairs.json").read_text())
    diagnostics, diagnostics_receipt = None, None
    if bool(args.secondary_diagnostics) != bool(args.secondary_diagnostics_sha256):
        raise ValueError("secondary diagnostics need a reviewed hash and path together")
    if args.secondary_diagnostics:
        if file_hash(args.secondary_diagnostics) != args.secondary_diagnostics_sha256:
            raise ValueError("secondary diagnostics changed after review")
        diagnostics = validate_secondary_diagnostics(
            json.loads(args.secondary_diagnostics.read_text()), package["inventory_sha256"], pairs)
        diagnostics_receipt = {"path": str(args.secondary_diagnostics.resolve()),
                               "sha256": args.secondary_diagnostics_sha256,
                               "source_status_counts": dict(Counter(r["source_status"] for r in diagnostics.values())),
                               "policy": "All questions retain provided keys in primary results. Source-supported and source-supported-and-eligible cohorts are secondary. Eligibility additionally requires actual initial knowledge diagnostics to pass in both paired episodes. Unknown is not a pass. No alternative gold keys are substituted."}
    accuracies, contrasts, paired, audits, judges = [], [], [], [], set()
    for ac_dir, bd_dir in zip(args.baseline_results, args.feedback_results):
        ac, old_audit = audit_stage_results(ac_dir, old_jobs, baseline["jobs_sha256"], package)
        bd, new_audit = audit_stage_results(bd_dir, new_jobs, package["jobs_sha256"], package)
        if old_audit["runtime_identity"] != new_audit["runtime_identity"]:
            raise ValueError("A/C and B/D judge runtime identities differ")
        judge = old_audit["runtime_identity"]["model"]
        if judge in judges:
            raise ValueError("repetitions cannot be presented as independent judges")
        judges.add(judge)
        a, c, p = stratified_statistics(pairs, {**ac, **bd}, judge, diagnostics)
        accuracies.extend(a); contrasts.extend(c); paired.extend(p)
        audits.append({"judge": judge, "A_C": old_audit, "B_D": new_audit})
    args.out.mkdir(parents=True)
    grid4.write_json(args.out / "analysis.json", {
        "schema": SCHEMA, "package_sha256": args.frozen_package_sha256,
        "questions": package["questions"], "families": package["families"],
        "matched_units_per_judge": len(pairs), "accuracy": accuracies,
        "contrasts": contrasts, "runtime_audits": audits,
        "bootstrap": {"unit": "original source family; keep every question/model/cell together",
                      "resamples": 10000, "seed": 2026101422, "interval": "unadjusted descriptive 95%"},
        "analysis_contract": package["analysis_contract"], "scope": package["scope"],
        "feedback_role_switches": package["feedback_role_switches"],
        "public_disclosure_review": package["public_disclosure_review"],
        "disclosure_interpretation": package["disclosure_interpretation"],
        "secondary_diagnostics": diagnostics_receipt,
        "limits": ["Exposure flag cohorts overlap; mutually exclusive exposure strata use training, development/reference, other tuning, prior evaluation, untouched precedence.",
                   "Cached jobs are primary historical observations, not new repeated trials. The post-result Haiku A/C replay is not substituted.",
                   "Family-macro primary and question-weighted secondary estimates differ when families have multiple questions.",
                   "The intervention bundles awareness, response percentiles, speaker means and a recommendation; no new neutral/shuffled feedback or independent generation repetitions.",
                   "The probe is a wrong-answer advocacy proxy under explicit instructions, not a validated local truth or hidden-intent detector.",
                   "Original questions reuse older trajectories; API routing, time and unreported serving details may differ. Native runtime contracts do not encode all hardware and numerical settings."]})
    grid4.write_rows(args.out / "paired_episodes.jsonl", paired)
    print(json.dumps({"judges": sorted(judges), "matched_units_per_judge": len(pairs),
                      "families": package["families"], "out": str(args.out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("prepare-baseline")
    for name in ("inventory", "source", "bundle", "reference-bundle", "original-grid-package"):
        baseline.add_argument("--" + name, type=Path, required=True)
    for name in ("inventory-sha256", "reference-bundle-sha256", "original-grid-package-sha256"):
        baseline.add_argument("--" + name, required=True)
    baseline.add_argument("--episodes", type=Path, nargs="+", required=True)
    baseline.add_argument("--models", nargs="+", default=list(MODELS))
    baseline.add_argument("--expected-questions", type=int, default=166)
    baseline.add_argument("--expected-families", type=int, default=116)
    feedback = sub.add_parser("prepare-feedback")
    feedback.add_argument("--baseline-package", type=Path, required=True)
    feedback.add_argument("--baseline-package-sha256", required=True)
    feedback.add_argument("--feedback-episodes", type=Path, nargs="+", required=True)
    feedback.add_argument("--feedback-bundle", type=Path, required=True)
    feedback.add_argument("--disclosure-review", type=Path, nargs="*")
    run = sub.add_parser("run-api")
    run.add_argument("--judge", default="anthropic/claude-haiku-5.5")
    run.add_argument("--runtime-reference", type=Path, required=True)
    run.add_argument("--cached-results", type=Path, required=True)
    run.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    run.add_argument("--budget", type=float, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--request-timeout-seconds", type=float, default=300)
    analysis = sub.add_parser("analyse")
    analysis.add_argument("--baseline-results", type=Path, nargs="+", required=True)
    analysis.add_argument("--feedback-results", type=Path, nargs="+", required=True)
    analysis.add_argument("--secondary-diagnostics", type=Path)
    analysis.add_argument("--secondary-diagnostics-sha256")
    for command in (run, analysis):
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--frozen-package-sha256", required=True)
    for command in (baseline, feedback, run, analysis):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-baseline":
        prepare_baseline(args)
    elif args.command == "prepare-feedback":
        prepare_feedback(args)
    elif args.command == "run-api":
        asyncio.run(run_api(args))
    else:
        analyse(args)


if __name__ == "__main__":
    main()
