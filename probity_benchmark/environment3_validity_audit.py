"""Read-only full-corpus provenance and quotation diagnostics.

This audit does not inspect judge answers or probe scores. Quotation matches
measure literal source copying, not factual truth; absent matches are reviewed
or left uncertain and never used to relabel a speaker's complete response.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .api import assert_no_exposed_truncation
from .environment3_evidence import evidence_messages
from .environment3_prepared import preparation_messages
from .environment3_preparation_gate import audit_receipts
from .environment3_prepared import preparation_schema
from .preflight import load_problems, sha256

QUOTE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')


def read_jsonl(path):
    data = path.read_bytes()
    if len(data) != path.stat().st_size or not data.endswith(b"\n"):
        raise ValueError("incomplete JSONL read or unterminated final record")
    return [json.loads(line) for line in data.decode().splitlines()]


def complete_receipt(row):
    if not row.get("valid") or row.get("finish_reason") != "stop":
        raise ValueError("incomplete or invalid generation receipt")
    if sha256(json.dumps(row["request"], sort_keys=True)) != row["request_sha256"]:
        raise ValueError("request bytes disagree with recorded hash")
    payload = json.loads(row["response_body"])
    assert_no_exposed_truncation(payload)
    message = payload["choices"][0]["message"]
    if json.loads(message["content"]) != row["parsed"]:
        raise ValueError("parsed response disagrees with complete raw JSON")
    allowance = row["request"]["max_tokens"]
    if row["usage"].get("completion_tokens", 0) >= allowance:
        raise ValueError("output token count reaches its allowance")
    return message


def quoted_spans(argument, story):
    spans = []
    for match in QUOTE.finditer(argument):
        group = 1 if match.group(1) is not None else 2
        text = match.group(group)
        start, end = match.span(group)
        assert argument[start:end] == text
        spans.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "exact_source_substring": text in story,
            }
        )
    return spans


def validate_turn(problem, episode, turn, index, prior, receipt):
    message = complete_receipt(receipt)
    assigned = "A" if turn["speaker"] == "Red" else "B"
    expected_tag = {
        "episode": episode["episode_id"],
        "stage": "debate",
        "round": turn["round"],
        "speaker": turn["speaker"],
    }
    if receipt["tag"] != expected_tag or receipt["model"] != episode["model"]:
        raise ValueError("turn does not identify its exact request and job tag")
    plan = {
        k: v
        for k, v in episode["private_preparations"][turn["speaker"]].items()
        if k != "request_sha256"
    }
    expected_messages = evidence_messages(
        problem,
        episode["swapped"],
        turn["speaker"],
        assigned,
        prior,
        json.dumps(plan, ensure_ascii=False),
        explicit_truth=True,
    )
    if receipt["request"]["messages"] != expected_messages:
        raise ValueError("source, plan, public history or instructions changed")
    if problem.story not in receipt["request"]["messages"][1]["content"]:
        raise ValueError("complete authentic source missing")
    if turn["visible_prior_turns"] != index:
        raise ValueError("sequential history count mismatch")
    if (
        receipt["parsed"] != {"argument": turn["argument"], "answer": turn["answer"]}
        or turn["assigned_answer"] != assigned
        or turn["role_adherent"] != (turn["answer"] == assigned)
        or turn["advocates_gold"] != (turn["answer"] == episode["gold"])
    ):
        raise ValueError("public turn differs from complete output or assignment")
    return {
        "episode_id": episode["episode_id"],
        "parent_problem_id": episode["parent_problem_id"],
        "parent_source_line": episode["parent_source_line"],
        "model": episode["model"],
        "turn_index": index,
        "speaker": turn["speaker"],
        "assigned_gold": assigned == episode["gold"],
        "request_sha256": receipt["request_sha256"],
        "argument_sha256": sha256(turn["argument"]),
        "argument_characters": len(turn["argument"]),
        "full_source_plan_history_identical": True,
        "field_adheres": turn["answer"] == assigned,
        "exposed_public_reasoning_characters": len(message.get("reasoning") or ""),
        "quoted_spans": quoted_spans(turn["argument"], problem.story),
    }


def run(args):
    if args.out.exists():
        raise ValueError("immutable new audit file required")
    problems, source_audit = load_problems(args.dataset)
    source_rows = read_jsonl(args.dataset)
    if source_audit["errors"] or len(source_rows) != len(problems):
        raise ValueError("invalid source corpus")
    source_map = {p.problem_id: (p, r) for p, r in zip(problems, source_rows)}
    summary = json.loads((args.episodes / "summary.json").read_text())
    if summary["errors"] or not summary["all_receipts_valid"]:
        raise ValueError("complete generation gate required")
    receipts = read_jsonl(args.episodes / "requests.jsonl")
    for receipt in receipts:
        complete_receipt(receipt)
    keyed = {
        (r["request_sha256"], json.dumps(r["tag"], sort_keys=True)): r for r in receipts
    }
    episodes = [
        json.loads(path.read_text())
        for path in sorted(args.episodes.glob("episode_*.json"))
    ]
    if len(episodes) != summary["expected_episodes"] or len(
        {e["episode_id"] for e in episodes}
    ) != len(episodes):
        raise ValueError("missing or duplicate complete episode")
    raw_preparations = read_jsonl(args.preparations / "requests.jsonl")
    preparation_gate = getattr(args, "preparation_completion_audit", None)
    failed_preparations = []
    if preparation_gate:
        gate = json.loads(preparation_gate.read_text())
        if (
            gate["status"] != "all_96_canonical_preparation_jobs_complete"
            or gate["accepted_jobs"] != 96
            or gate["raw_receipts"] != len(raw_preparations)
        ):
            raise ValueError("complete canonical preparation gate required")
        for name, digest in gate["input_sha256"].items():
            if sha256(Path(name).read_text()) != digest:
                raise ValueError("preparation gate input changed: " + name)
        preparations, failed_preparations = audit_receipts(
            raw_preparations,
            [r["tag"] for r in gate["accepted_receipts"]],
            preparation_schema(),
        )
        if [
            {"tag": r["tag"], "request_sha256": r["request_sha256"]}
            for r in preparations
        ] != gate["accepted_receipts"] or failed_preparations != gate[
            "retained_schema_failures"
        ]:
            raise ValueError("preparation attempts differ from audited gate")
    else:
        preparations = raw_preparations
    prepared_by_hash = {r["request_sha256"]: r for r in preparations}
    for receipt in preparations:
        complete_receipt(receipt)
    records = []
    for episode in episodes:
        if (
            not episode["complete"]
            or episode["protocol"] != "explicit_reasoned_evidence_sequential"
        ):
            raise ValueError("wrong protocol or incomplete episode")
        problem, source = source_map[episode["problem_id"]]
        schedule = source["schedule_parent_index"]
        swapped = bool(schedule % 2)
        _, gold = problem.view(False, swapped)
        if episode["gold"] != gold or episode["swapped"] != swapped:
            raise ValueError("source/option mapping mismatch")
        order = (
            ["Red", "Blue"]
            if (schedule ^ (schedule // 2)) % 2 == 0
            else ["Blue", "Red"]
        )
        if [(t["round"], t["speaker"]) for t in episode["turns"]] != [
            (round_number, speaker) for round_number in range(1, 4) for speaker in order
        ]:
            raise ValueError("round count or complete speaker order mismatch")
        for speaker, plan in episode["private_preparations"].items():
            receipt = prepared_by_hash[plan["request_sha256"]]
            assigned = "A" if speaker == "Red" else "B"
            if receipt["model"] != episode["model"] or receipt["tag"] != {
                "problem_id": problem.problem_id,
                "model": episode["model"],
                "speaker": speaker,
                "stage": "reasoned_preparation",
            }:
                raise ValueError(
                    "private preparation model or complete job tag differs"
                )
            if receipt["request"]["messages"] != preparation_messages(
                problem, swapped, speaker, assigned, explicit_truth=True
            ) or receipt["parsed"] != {
                k: v for k, v in plan.items() if k != "request_sha256"
            }:
                raise ValueError(
                    "private plan differs from complete preparation receipt"
                )
        prior = []
        for index, turn in enumerate(episode["turns"]):
            tag = {
                "episode": episode["episode_id"],
                "stage": "debate",
                "round": turn["round"],
                "speaker": turn["speaker"],
            }
            receipt = keyed[(turn["request_sha256"], json.dumps(tag, sort_keys=True))]
            records.append(validate_turn(problem, episode, turn, index, prior, receipt))
            prior.append(turn)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": len(episodes),
        "turns": len(records),
        "preparation_accepted_jobs": len(preparations),
        "preparation_raw_attempts_retained": len(raw_preparations),
        "preparation_schema_failures_retained": failed_preparations,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in receipts)),
        "all_source_plan_history_and_output_bytes_match": True,
        "all_schedule_and_option_maps_match": True,
        "field_adherent_turns": sum(r["field_adheres"] for r in records),
        "turns_with_exposed_public_reasoning": sum(
            bool(r["exposed_public_reasoning_characters"]) for r in records
        ),
        "rows": records,
        "claim_limit": "Quoted spans are exact byte-match diagnostics, not factual labels; attribution, negation, paraphrase and true facts embedded in misleading responses require independent source review. Speaker roles are never atomic truth labels. Judge and probe outcomes are not accessed.",
        "file_sha256": {
            str(path): sha256(path.read_text())
            for path in [
                args.dataset,
                args.episodes / "requests.jsonl",
                args.episodes / "summary.json",
                args.preparations / "requests.jsonl",
                Path(__file__),
            ]
            + ([preparation_gate] if preparation_gate else [])
        },
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "file_sha256")},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ["dataset", "episodes", "preparations", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preparation-completion-audit", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
