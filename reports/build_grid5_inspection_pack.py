"""Private, offline paired-transcript inspection; never an evaluated treatment.

Requires reviewed complete-package, completion-marker and fixed-selection hashes.
No generation, tokenizer/model download, analysis recomputation or publication.
Outputs are restricted to runs/iteration5/reader_reviews, in a fresh directory.
"""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path

from probity_benchmark import grid4_evaluate as grid4
from probity_benchmark import grid5_evaluate as evaluate
from probity_benchmark import grid5_generate as generation
from probity_benchmark import grid5_score as score
from probity_benchmark.environment3_evidence import evidence_messages
from probity_benchmark.preflight import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/iteration5"
LABEL = "Private inspection view — not an evaluated treatment"
CELLS = {"A": ("baseline", "No feedback; judge without probe packet"),
         "B": ("feedback", "Live feedback; judge without probe packet"),
         "C": ("baseline", "No feedback; judge with probe packet"),
         "D": ("feedback", "Live feedback; judge with probe packet")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keyed(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, "duplicate inspection join identity: " + str(identity))
        result[identity] = row
    return result


def join_turn(episode, index, request, result, reference, arm):
    """Validate the actual response, complete replay context, score and episode join."""
    score.check_result(result, request, reference)
    receipt = generation.audit_receipt(request["source_receipt"])
    turn = episode["turns"][index]
    expected = {"episode_id": episode["episode_id"], "turn_index": index,
                "model": episode["model"], "parent_problem_id": episode["parent_problem_id"],
                "split": episode["split"], "speaker": turn["speaker"],
                "source_request_sha256": turn["request_sha256"],
                "argument": turn["argument"]}
    require(all(request.get(k) == v for k, v in expected.items()), "episode/request turn join differs")
    require(all(receipt["parsed"].get(k) == turn[k] for k in ("argument", "answer")),
            "parsed public response differs from episode")
    require(turn["visible_prior_turns"] == index, "visible prior-turn count differs")
    if arm == "feedback":
        require(request["matched_no_feedback_episode_id"] == episode["matched_no_feedback_episode_id"],
                "feedback matched episode differs")
        require(turn["feedback_prior_request_sha256"] ==
                [t["request_sha256"] for t in episode["turns"][:index]], "causal feedback chain differs")
        require(turn["probe_feedback"] == result, "displayed feedback scalar differs from replay result")
    else:
        require(request["matched_no_feedback_episode_id"] == episode["episode_id"],
                "baseline score paired to another episode")
    return {"turn_index": index, "episode_turn": copy.deepcopy(turn),
            "submitted_messages": copy.deepcopy(receipt["request"]["messages"]),
            "parsed_public_response": copy.deepcopy(receipt["parsed"]),
            "raw_provider_response_body": receipt["response_body"],
            "generation_receipt": copy.deepcopy(receipt),
            "score_request": copy.deepcopy(request), "score_result": copy.deepcopy(result)}


def join_cell(cell, pair, episode, job, api_row, api_receipt, native_row):
    require(job["job_id"] == pair["jobs"][cell] and job["episode_id"] == episode["episode_id"],
            "judge cell joined to wrong episode/job")
    require(job["participant_model"] == pair["participant_model"], "judge participant differs")
    require(job["turn_argument_sha256"] == [sha256(t["argument"]) for t in episode["turns"]],
            "judge transcript hashes differ")
    require((job["kind"] == "none") == (cell in "AB"), "judge packet condition differs")
    require(all(row["job_id"] == job["job_id"] for row in (api_row, native_row)), "judge result job differs")
    generation.audit_receipt(api_receipt)
    require(api_receipt["request"]["messages"] == job["messages"], "submitted judge messages differ")
    require(api_receipt["request_sha256"] == api_row["request_sha256"] and
            api_receipt["tag"]["job"] == api_row.get("receipt_job_id", job["job_id"]),
            "canonical/aliased judge receipt differs")
    require(api_receipt["parsed"]["answer"] == api_row["answer"], "judge parsed answer differs")
    return {"cell": cell, "description": CELLS[cell][1], "compiled_job": copy.deepcopy(job),
            "exact_api_submitted_messages": copy.deepcopy(api_receipt["request"]["messages"]),
            "api_result": copy.deepcopy(api_row), "api_receipt": copy.deepcopy(api_receipt),
            "native_result": copy.deepcopy(native_row),
            "native_message_note": "Same compiled messages; the native runner applies its frozen chat template. "
                "Serialized native prompt text is not retained in the result receipt and is not reconstructed here."}


def private_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to((RUN / "reader_reviews").resolve()) and path != RUN / "reader_reviews",
            "raw inspection output must remain inside private iteration5/reader_reviews")
    require(not path.exists(), "fresh inspection output directory required")
    return path


def collect(args):
    """Read existing evidence only. Every selected join must validate before output."""
    out = private_output(args.out)
    meta = RUN / "coordinator"
    selection_path = meta / "qualitative_reader_selection_v1.json"
    completion_path = meta / "final_analysis_complete_v1.json"
    require(digest(selection_path) == args.selection_sha256, "reviewed fixed selection changed")
    require(digest(completion_path) == args.completion_sha256, "reviewed completion marker changed")
    selection, completion = read(selection_path), read(completion_path)
    require(completion["package_sha256"] == args.package_sha256, "completion/package identity differs")
    completed_paths = {"analysis_sha256": RUN / "analysis_v1/analysis.json",
                        "diagnostics_sha256": meta / "secondary_diagnostics_v1.json",
                        "public_summary_sha256": ROOT / "reports/data/iteration5/grid_summary_v1.json",
                        "plot_manifest_sha256": ROOT / "reports/figures/iteration5/grid_v1/manifest.json"}
    for field, path in completed_paths.items():
        require(digest(path) == completion[field], "completion-bound artifact changed: " + field)
    package_dir = RUN / "judge_grid_v1"
    package = evaluate.check_package(package_dir, args.package_sha256, True)
    require(package["stage"] == "feedback", "complete A/B/C/D package required")
    baseline_dir = Path(package["baseline_package"])
    baseline = evaluate.check_package(baseline_dir, package["baseline_package_sha256"], True)
    inventory, inventory_rows, problems, _ = evaluate.load_inventory(
        Path(package["inventory"]), package["inventory_sha256"], Path(package["source"]),
        166, 116, package["participant_models"])
    require(selection["inventory_sha256"] == package["inventory_sha256"], "selection inventory differs")
    chosen = selection["primary_selection"]
    require(len(chosen) == 4 and len(keyed(chosen, lambda r: (r["matched_question_id"], r["participant_model"]))) == 4,
            "fixed four-unit qualitative sample required")
    pairs = keyed(read(package_dir / "pairs.json"), lambda r: (r["matched_question_id"], r["participant_model"]))
    selected_pairs = [pairs[(r["matched_question_id"], r["participant_model"])] for r in chosen]
    ids = {p[k] for p in selected_pairs for k in ("no_feedback_episode_id", "feedback_episode_id")}
    require(len(ids) == 8, "eight distinct selected trajectories required")
    episodes, episode_paths = {}, {}
    for name in package["input_sha256"]:
        path = Path(name)
        if path.name.startswith("episode_") and path.suffix == ".json":
            value = read(path)
            if value["episode_id"] in ids:
                require(value["episode_id"] not in episodes, "duplicate selected episode input")
                episodes[value["episode_id"]], episode_paths[value["episode_id"]] = value, path
    require(set(episodes) == ids, "missing selected episode input")
    jobs = keyed(generation.read_rows(package_dir / "all_jobs.jsonl"), lambda j: j["job_id"])
    native, native_audit = grid4.audit_results(RUN / "native_judge_v1", jobs, package["all_jobs_sha256"])
    api_rows, api_receipts, api_audits, judge_paths = {}, {}, {}, []
    for name, pkg, directory in (("AC", baseline, RUN / "haiku_ac_v1"), ("BD", package, RUN / "haiku_bd_v1")):
        stage_jobs = {j["job_id"]: j for j in generation.read_rows((baseline_dir if name == "AC" else package_dir) / "jobs.jsonl")}
        rows, audit = grid4.audit_results(directory, stage_jobs, pkg["jobs_sha256"])
        require(not set(api_rows) & set(rows), "overlapping A/C and B/D judge identities")
        api_rows.update(rows)
        for receipt in generation.read_rows(directory / "requests.jsonl"):
            if receipt.get("valid"):
                key = receipt["request_sha256"], receipt["tag"]["job"]
                require(key not in api_receipts, "ambiguous canonical judge receipt")
                api_receipts[key] = receipt
        api_audits[name] = audit
        judge_paths.extend(directory / name for name in ("manifest.json", "evaluations.jsonl", "requests.jsonl"))
    reference_path = Path(package["reference_bundle"])
    require(digest(reference_path) == package["reference_bundle_sha256"], "reference bundle changed")
    reference = read(reference_path)
    bundle_rows = {arm: keyed(read(Path(pkg["bundle"]))["rows"],
                             lambda row: (row["episode_id"], row["turn_index"]))
                   for arm, pkg in (("baseline", baseline), ("feedback", package))}
    queue = RUN / "score_queue_v1"
    plan = score.load_plan(queue)
    require(plan["reference_artifact_sha256"] == digest(reference_path), "queue reference differs")
    joined_scores, selected_paths = {}, []
    # Stream full request files rather than retaining the entire corpus in RAM.
    for item_path in sorted((queue / "jobs").glob("*.json")):
        request_path = queue / "requests" / item_path.name
        request = read(request_path)
        if request["episode_id"] not in ids:
            continue
        item = read(item_path)
        score.validate_item(item, request, plan, digest(queue / "plan.json"), reference)
        result_path = queue / "results" / item_path.name
        key = request["episode_id"], request["turn_index"]
        require(key not in joined_scores, "duplicate selected score turn")
        joined_scores[key] = item["arm"], request, read(result_path)
        selected_paths.extend((item_path, request_path, result_path))
    require(set(joined_scores) == {(eid, i) for eid in ids for i in range(6)}, "selected scalar coverage incomplete")
    ready_path = meta / "preparations_recovery_ready_v4.json"
    ready = read(ready_path)
    require(ready.get("complete") is True and ready.get("expected_preparations") == 568,
            "complete preparation readiness required")
    prep_path = ROOT / ready["preparations_path"]
    for name, field in (("preparations.jsonl", "preparations_sha256"), ("requests.jsonl", "requests_sha256"),
                        ("summary.json", "summary_sha256")):
        require(digest(prep_path.with_name(name)) == ready[field], "complete preparation evidence changed")
    new_problems, new_rows, _, _ = generation.frozen_inventory(
        meta / "new142_source_frozen_v2.jsonl", Path(package["inventory"]), package["inventory_sha256"])
    preparations = generation.external_preparations(new_problems, new_rows, prep_path)
    units = []
    for selected, pair in zip(chosen, selected_pairs):
        row = inventory_rows[pair["parent_problem_id"]]
        problem = problems[row["problem_id"]]
        require(not row["prior24"] and selected["problem_id"] == problem.problem_id and
                selected["family"] == row["family_id"] == pair["family"] and
                selected["source_sha256"] == sha256(problem.story) and
                selected["untouched_before_expansion"] == row["family_untouched_before_iteration5"],
                "selected source/family/exposure join differs")
        arms = {}
        for arm, key in (("baseline", "no_feedback_episode_id"), ("feedback", "feedback_episode_id")):
            episode = episodes[pair[key]]
            require(episode["problem_id"] == problem.problem_id and episode["model"] == pair["participant_model"]
                    and episode.get("complete") is True and len(episode["turns"]) == 6, "selected episode/source identity differs")
            turns = []
            for i in range(6):
                actual_arm, request, result = joined_scores[(episode["episode_id"], i)]
                require(actual_arm == arm, "selected score arm differs")
                joined = join_turn(episode, i, request, result, reference, arm)
                frozen_row = bundle_rows[arm][(episode["episode_id"], i)]
                require(frozen_row["scores"] == [result["raw_score"]] and
                        frozen_row["argument_sha256"] == request["argument_sha256"] and
                        frozen_row["spans"] == [[0, len(request["argument"])]],
                        "selected replay scalar differs from compiled judge bundle")
                if arm == "feedback":
                    expected_messages = generation.engine.feedback_messages(problem, {**episode, "turns": episode["turns"][:i]})
                else:
                    turn = episode["turns"][i]
                    preparation = {k: v for k, v in episode["private_preparations"][turn["speaker"]].items()
                                   if k != "request_sha256"}
                    expected_messages = evidence_messages(problem, episode["swapped"], turn["speaker"],
                        turn["assigned_answer"], episode["turns"][:i], json.dumps(preparation, ensure_ascii=False), explicit_truth=True)
                require(joined["submitted_messages"] == expected_messages, "submitted debater causal context differs")
                turns.append(joined)
            plans = {}
            for speaker, parsed in episode["private_preparations"].items():
                record, receipt = preparations[(problem.problem_id, episode["model"], speaker)]
                require(parsed == {**receipt["parsed"], "request_sha256": receipt["request_sha256"]},
                        "episode private preparation differs from retained response")
                plans[speaker] = {"record": record, "receipt": receipt}
            arms[arm] = {"episode": episode, "turns": turns, "private_preparations": plans}
        cells = {}
        for cell, (arm, _) in CELLS.items():
            job = jobs[pair["jobs"][cell]]
            api_row = api_rows[job["job_id"]]
            receipt = api_receipts[(api_row["request_sha256"], api_row.get("receipt_job_id", job["job_id"]))]
            cells[cell] = join_cell(cell, pair, arms[arm]["episode"], job, api_row, receipt, native[job["job_id"]])
        units.append({"selection": selected, "pair": pair,
                      "source_and_roles_not_a_judge_view": {"story": problem.story, "question": problem.question,
                          "original_options": list(problem.options), "displayed_gold": row["displayed_gold"],
                          "displayed_options": list(reversed(problem.options)) if arms["baseline"]["episode"]["swapped"] else list(problem.options),
                          "swapped": arms["baseline"]["episode"]["swapped"], "inventory_row": row},
                      "arms": arms, "cells": cells})
    paths = [selection_path, completion_path, package_dir / "package.json", baseline_dir / "package.json",
             package_dir / "all_jobs.jsonl", package_dir / "jobs.jsonl", package_dir / "pairs.json",
             baseline_dir / "jobs.jsonl", baseline_dir / "pairs.json",
             RUN / "native_judge_v1/manifest.json", RUN / "native_judge_v1/evaluations.jsonl",
             queue / "plan.json", reference_path, ready_path, prep_path, prep_path.with_name("requests.jsonl"),
             prep_path.with_name("summary.json"), *completed_paths.values(), *judge_paths,
             *episode_paths.values(), *selected_paths]
    provenance = {**package["input_sha256"], **{str(p): digest(p) for p in paths}}
    return out, {"schema": "grid5_private_inspection_pack_v1", "label": LABEL, "units": units,
                 "selection_rule": selection["selection_rule"], "selection_limits": selection["limits"],
                 "judge_audits": {"native": native_audit, "api": api_audits},
                 "input_sha256": provenance, "builder_sha256": digest(__file__),
                 "native_prompt_limit": "Compiled messages are exact; serialized native prompt text is not retained. "
                    "No new tokenizer replay or equivalence claim is made.",
                 "reasoning_limit": "Raw responses preserve only provider-exposed reasoning. Structured private plans, "
                    "public argument text and complete raw provider envelopes are separately identified."}


def block(title, value, opened=False):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return '<details' + (' open' if opened else '') + '><summary>' + html.escape(title) + \
        '</summary><pre><code>' + html.escape(text) + '</code></pre></details>'


def render(report):
    """Static HTML only; every source/provider string is escaped, no scripts."""
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>Private paired transcript inspection</title><style>body{font:16px/1.55 system-ui;margin:2rem;color:#172536}'
        'header{max-width:75rem}a{color:#175e96}article{border-top:3px solid #334b65;margin-top:3rem}'
        '.arms{display:grid;grid-template-columns:1fr 1fr;gap:2rem}.arm{min-width:0}'
        '.turn{border-top:1px solid #ccd5de;padding:1rem 0}.text,pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'pre{font:13px/1.5 monospace;background:#f3f5f7;padding:1rem}summary{cursor:pointer;font-weight:600}'
        '.note{color:#45576b}.warning{border-left:4px solid #a65514;padding-left:1rem}'
        '@media(max-width:900px){.arms{display:block}body{margin:1rem}}</style><header><h1>' + html.escape(LABEL) + '</h1>',
        '<p class="warning">Contains full private source stories, plans and raw responses. Keep this artifact private. '
        'These panels were not shown to judges; exact evaluated messages are under A/B/C/D below.</p>',
        '<p>' + html.escape(report['selection_rule']) + '</p><p>' + html.escape(report['selection_limits']) + '</p>',
        '<p class="note">Source keys and assigned roles are analyst metadata, not sentence-level truth labels. '
        'Probe scores are associations with the frozen detector target, not probabilities of lying.</p>',
        '<p>' + html.escape(report['reasoning_limit']) + '</p><nav>']
    for n, unit in enumerate(report['units'], 1):
        parts.append(f'<a href="#unit-{n}">Unit {n}: ' + html.escape(unit['pair']['participant_model']) + '</a> · ')
    parts.append('</nav></header>')
    for n, unit in enumerate(report['units'], 1):
        parts.append(f'<article id="unit-{n}"><h2>Unit {n} — ' + html.escape(unit['pair']['participant_model']) + '</h2>')
        parts.append(block('Source, question, original option order and role metadata — NOT a judge view', unit['source_and_roles_not_a_judge_view']))
        parts.append('<div class="arms">')
        for arm, data in unit['arms'].items():
            parts.append('<section class="arm"><h3>' + ('No-feedback transcript (A/C)' if arm == 'baseline' else 'Live-feedback transcript (B/D)') + '</h3>')
            for speaker, plan in data['private_preparations'].items():
                parts.append(block(speaker + ': parsed private plan', plan['receipt']['parsed']))
                parts.append(block(speaker + ': complete private preparation request and raw provider response', plan['receipt']))
            for joined in data['turns']:
                t, result = joined['episode_turn'], joined['score_result']
                title = f"Turn {joined['turn_index'] + 1} · {t['speaker']} · round {t['round']} · assigned {t['assigned_answer']} · returned {t['answer']}"
                parts.append('<div class="turn"><h4>' + html.escape(title) + '</h4><div class="text">' + html.escape(t['argument']) + '</div>')
                parts.append('<p class="note">Post-turn scalar: ' + html.escape(str(result['raw_score'])) + '; percentile: ' + html.escape(str(result['percentile'])) + '. This score is not feedback available before this same turn.</p>')
                parts.append(block('Exactly what this debater was sent — complete submitted request.messages', joined['submitted_messages']))
                parts.append(block('Parsed public JSON response', joined['parsed_public_response']))
                parts.append(block('Complete raw provider response body (including exposed reasoning, if any)', joined['raw_provider_response_body']))
                parts.append(block('Scalar replay, context/token/head provenance and causal prior-turn metadata',
                    {'result': result, 'episode_turn': t, 'score_request_sha256': result['score_request_sha256']}))
                parts.append('</div>')
            parts.append('</section>')
        parts.append('</div><h3>Exact final-judge views and responses</h3>')
        for cell, data in unit['cells'].items():
            parts.append('<section><h4>' + html.escape(cell + ' — ' + data['description']) + '</h4>')
            parts.append(block('Exact submitted API judge messages (also the compiled native messages)', data['exact_api_submitted_messages']))
            parts.append('<p class="note">' + html.escape(data['native_message_note']) + '</p>')
            parts.append(block('API result, exact-request reuse metadata and full provider receipt', {'result': data['api_result'], 'receipt': data['api_receipt']}))
            parts.append(block('Native result and raw answer-only response', data['native_result']))
            parts.append('</section>')
        parts.append('</article>')
    parts.append(block('Input hashes and audit provenance', {k: v for k, v in report.items() if k != 'units'}))
    parts.append('</html>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'completion', 'selection'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    out, report = collect(parser.parse_args())
    page = render(report)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'evidence.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'index.html').write_text(page)
    manifest = {'schema': report['schema'], 'label': LABEL, 'private_only': True,
                'units': 4, 'transcripts': 8, 'public_turns': 48, 'judge_cells': 16,
                'builder_sha256': report['builder_sha256'], 'input_sha256': report['input_sha256'],
                'output_sha256': {n: digest(out / n) for n in ('evidence.json', 'index.html')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'private_inspection_pack': str(out), 'units': 4, 'transcripts': 8, 'paid_calls': 0}))


if __name__ == '__main__':
    main()
