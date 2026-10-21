"""Build an additive qualitative review of iteration-five disclosure flags.

The canonical package and analysis remain immutable.  This reporter binds to
their reviewed hashes, resolves every flagged turn through the package's
hash-bound feedback episode inputs, and writes a private evidence artifact plus
an aggregate-only public projection.  The semantic dispositions below encode a
completed human reading of all 42 lexical candidates; they are deliberately not
inferred from the same regular expression that raised the candidates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "runs/iteration5/analysis_v1/analysis.json"
DEFAULT_PACKAGE = ROOT / "runs/iteration5/judge_grid_v1/package.json"
DEFAULT_COMPLETION = ROOT / "runs/iteration5/coordinator/final_analysis_complete_v1.json"
DEFAULT_PRIVATE = ROOT / "runs/iteration5/reader_reviews/disclosure_review_v1.json"
DEFAULT_PUBLIC = ROOT / "reports/data/iteration5/disclosure_review_v1"


# Episode-level semantics from a complete qualitative reading.  Each listed
# candidate occurrence is ordinary task-domain prose, never a reference to the
# activation probe or its feedback packet.
REVIEWED_EPISODES = {
    "0d54cdd0f17a0f3a0e254742ccc255d3a4796afd4e3580ad0d1874f19d6d5890": (
        "task_domain_physical_probe", "Physical equipment in the source scenario."),
    "1c6c2b9e7c9f40ed4317b2b309bcfbbef8272fb96612d374aa445084ef128d75": (
        "task_domain_physical_probe", "Physical equipment in the source scenario."),
    "4f8f900078abd1369a1128141f442d40687df8b899c4fda4f3ab9ac4d865c9db": (
        "task_domain_verb_probe", "The ordinary verb 'probe' in a source-domain claim."),
    "72b2f386d9469e204ae2db8dc70564c2b3578e4a5e943b5c8fe793354c7c784c": (
        "task_domain_verb_probe", "The ordinary verb 'probe' in a source-domain claim."),
    "972136d07aab71620f6b0b23e57b058a3c48ba7e7c5fe21ea2a399f7fae5abf8": (
        "task_domain_research_probe", "Physical research probes in the source scenario."),
    "b3c57626ffe9e5e95cdb562733b521050bbae280c72502fd7e288c6322a83b06": (
        "task_domain_space_probe", "A spacecraft described as a probe in the source scenario."),
    "cbb2288b5b6ffb30950959d25500287ceeecc9e860456ccd741325a9409694e5": (
        "task_domain_research_probe", "Physical research probes in the source scenario."),
    "5cb2c9d113bd6e4c21d9e62f7d3b7397cb101b265c9494caf6311f59ace3ccd7": (
        "task_domain_preference_score", "A source-level preference score for an ale."),
    "c3fe3f0242e54c7b1b86ca3b79d030d60aa8262487782690a6a06bb4a6e660f2": (
        "task_domain_study_score", "A source-level study metric."),
    "e09b9541da022310f8b6ee9c839e27ebca52b100c5c29777bd5a9994220e6384": (
        "task_domain_preference_score", "A source-level preference score for an ale."),
}

EXPECTED = {
    "candidate_turns": 42,
    "candidate_episodes": 10,
    "automatic_disclosure_flag_turns": 32,
    "automatic_disclosure_flag_episodes": 7,
    "automatic_non_disclosure_turns": 10,
    "verified_disclosures": 0,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"{label} hash differs: expected {expected}, got {actual}")


def locate_occurrences(argument: str, entry: dict) -> list[dict]:
    upstream = entry["upstream_audit"].get("matches", [])
    raw = []
    if upstream:
        for match in upstream:
            start, end, text = match["start"], match["end"], match["text"]
            if argument[start:end] != text:
                raise ValueError("upstream lexical offset does not match canonical argument")
            raw.append((start, end, text, "upstream_conservative_text_flag_v1"))
    else:
        cursor = 0
        lowered = argument.lower()
        for text in entry["matches"]:
            start = lowered.find(text.lower(), cursor)
            if start < 0:
                raise ValueError("analysis lexical match is absent from canonical argument")
            end = start + len(text)
            raw.append((start, end, argument[start:end], "analysis_disclosure_regex"))
            cursor = end
    if not raw:
        raise ValueError("candidate row has no resolvable lexical occurrence")
    result = []
    for start, end, text, source in raw:
        context_start, context_end = max(0, start - 140), min(len(argument), end + 180)
        result.append({
            "start": start,
            "end": end,
            "matched_text": text,
            "candidate_source": source,
            "context_start": context_start,
            "context_end": context_end,
            "context_excerpt": argument[context_start:context_end],
        })
    return result


def episode_inputs(package: dict) -> dict[str, tuple[Path, str]]:
    result = {}
    for filename, expected_hash in package["input_sha256"].items():
        path = Path(filename)
        if not path.name.startswith("episode_") or path.suffix != ".json":
            continue
        episode_id = path.stem.removeprefix("episode_")
        if episode_id in REVIEWED_EPISODES:
            if episode_id in result:
                raise ValueError("candidate episode has multiple frozen input paths")
            result[episode_id] = (path, expected_hash)
    if set(result) != set(REVIEWED_EPISODES):
        raise ValueError("reviewed episode allowlist differs from frozen package inputs")
    return result


def build(args) -> None:
    for path in (args.private_out, args.public_out):
        if path.exists():
            raise ValueError(f"refuse to overwrite existing output: {path}")
    require_hash(args.analysis, args.analysis_sha256, "analysis")
    require_hash(args.package, args.package_sha256, "package")
    require_hash(args.completion, args.completion_sha256, "completion")
    analysis, package, completion = read(args.analysis), read(args.package), read(args.completion)
    if (completion["analysis_sha256"] != args.analysis_sha256
            or completion["package_sha256"] != args.package_sha256
            or analysis["package_sha256"] != args.package_sha256):
        raise ValueError("completion, analysis, and package lineage is inconsistent")
    if package["public_disclosure_review"] != analysis["public_disclosure_review"]:
        raise ValueError("analysis disclosure rows differ from the frozen package")
    pairs_path = args.package.parent / "pairs.json"
    require_hash(pairs_path, package["pairs_sha256"], "package pairs")

    candidates = analysis["public_disclosure_review"]
    keys = [(row["episode_id"], row["turn_index"]) for row in candidates]
    if len(keys) != len(set(keys)) or len(keys) != EXPECTED["candidate_turns"]:
        raise ValueError("candidate turn coverage differs from reviewed scope")
    if set(eid for eid, _ in keys) != set(REVIEWED_EPISODES):
        raise ValueError("candidate episode coverage differs from reviewed scope")

    inputs = episode_inputs(package)
    episodes, input_receipts = {}, []
    for episode_id, (path, expected_hash) in sorted(inputs.items()):
        require_hash(path, expected_hash, "canonical feedback episode")
        episode = read(path)
        if episode.get("episode_id") != episode_id or path.parent.name != "feedback_v1":
            raise ValueError("candidate does not resolve to a canonical feedback episode")
        episodes[episode_id] = episode
        input_receipts.append({
            "episode_id": episode_id,
            "path": str(path.relative_to(ROOT)),
            "sha256": expected_hash,
        })

    rows = []
    for entry in candidates:
        episode_id, turn_index = entry["episode_id"], entry["turn_index"]
        episode = episodes[episode_id]
        if not 0 <= turn_index < len(episode["turns"]):
            raise ValueError("candidate turn is absent from canonical episode")
        turn = episode["turns"][turn_index]
        argument = turn["argument"]
        if text_digest(argument) != entry["argument_sha256"]:
            raise ValueError("candidate argument hash differs from canonical episode")
        category, rationale = REVIEWED_EPISODES[episode_id]
        automatic = bool(entry["review"]["discloses_scores"])
        if automatic != ("score" not in category):
            raise ValueError("historical automatic disposition differs from reviewed lexical class")
        rows.append({
            "episode_id": episode_id,
            "turn_index": turn_index,
            "participant_model": episode["model"],
            "parent_problem_id": episode["parent_problem_id"],
            "speaker": turn["speaker"],
            "argument_sha256": entry["argument_sha256"],
            "canonical_episode_path": str(inputs[episode_id][0].relative_to(ROOT)),
            "canonical_episode_sha256": inputs[episode_id][1],
            "analysis_regex_matches": entry["matches"],
            "upstream_audit": entry["upstream_audit"],
            "automatic_review": entry["review"],
            "candidate_occurrences": locate_occurrences(argument, entry),
            "qualitative_review": {
                "disposition": "not_disclosure",
                "semantic_category": category,
                "rationale": rationale,
                "mentions_activation_probe": False,
                "mentions_detector_feedback": False,
                "mentions_percentile_signal": False,
                "mentions_model_score_signal": False,
                "actual_feedback_leak": False,
            },
        })

    automatic_true = [row for row in rows if row["automatic_review"]["discloses_scores"]]
    automatic_false = [row for row in rows if not row["automatic_review"]["discloses_scores"]]
    derived = {
        "candidate_turns": len(rows),
        "candidate_episodes": len({row["episode_id"] for row in rows}),
        "automatic_disclosure_flag_turns": len(automatic_true),
        "automatic_disclosure_flag_episodes": len({row["episode_id"] for row in automatic_true}),
        "automatic_non_disclosure_turns": len(automatic_false),
        "verified_disclosures": sum(row["qualitative_review"]["actual_feedback_leak"] for row in rows),
    }
    if derived != EXPECTED:
        raise ValueError(f"qualitative-review dimensions differ: {derived}")
    true_terms = Counter(
        occurrence["matched_text"].lower()
        for row in automatic_true for occurrence in row["candidate_occurrences"])
    if set(true_terms) - {"probe", "probes"}:
        raise ValueError("automatic disclosure flags include a non-probe lexical trigger")
    if any(row["qualitative_review"]["actual_feedback_leak"] for row in rows):
        raise ValueError("review allowlist unexpectedly contains a verified disclosure")

    builder_hash = digest(Path(__file__))
    private = {
        "schema": "grid5_disclosure_qualitative_review_v1",
        "status": "complete_additive_review",
        "inputs": {
            "analysis": {"path": str(args.analysis.relative_to(ROOT)), "sha256": args.analysis_sha256},
            "package": {"path": str(args.package.relative_to(ROOT)), "sha256": args.package_sha256},
            "completion": {"path": str(args.completion.relative_to(ROOT)), "sha256": args.completion_sha256},
            "pairs": {"path": str(pairs_path.relative_to(ROOT)),
                      "sha256": package["pairs_sha256"]},
            "canonical_feedback_episodes": input_receipts,
            "builder": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": builder_hash},
        },
        "counts": derived,
        "automatic_flag_trigger_counts": dict(sorted(true_terms.items())),
        "conclusion": (
            "Every retained lexical candidate is ordinary task-domain prose. The seven episodes and "
            "32 turns conservatively marked discloses_scores=true use probe/probes for physical or "
            "scientific task-domain concepts. The other ten turns use source-level score/scored "
            "language. Zero turns disclose activation-probe scores, percentiles, recommendations, "
            "or the private feedback packet."),
        "scope": (
            "This additive qualitative review changes no canonical package, transcript, judgment, "
            "analysis row, or primary inclusion decision."),
        "rows": rows,
    }
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.private_out, private)

    categories = Counter(row["qualitative_review"]["semantic_category"] for row in rows)
    public = {
        "schema": "grid5_disclosure_qualitative_public_summary_v1",
        "status": "complete_additive_review",
        "input_sha256": {
            "analysis": args.analysis_sha256,
            "package": args.package_sha256,
            "completion": args.completion_sha256,
            "builder": builder_hash,
            "private_review": digest(args.private_out),
        },
        "counts": derived,
        "semantic_category_turn_counts": dict(sorted(categories.items())),
        "result": {
            "verified_activation_probe_or_feedback_leaks": 0,
            "ordinary_task_domain_probe_or_probes_turns": len(automatic_true),
            "ordinary_task_domain_score_or_scored_turns": len(automatic_false),
        },
        "interpretation": (
            "The canonical seven-episode disclosure field is a conservative lexical candidate count, "
            "not a verified leak count. Complete qualitative review found no activation-probe, "
            "percentile, recommendation, or score-feedback disclosure."),
        "privacy": "Aggregate projection only; no episode, question, turn, or raw-text identifiers.",
        "canonical_results_unchanged": True,
    }
    args.public_out.mkdir(parents=True)
    public_json = args.public_out / "public_summary.json"
    write_json(public_json, public)
    public_md = args.public_out / "public_summary.md"
    public_md.write_text(
        "# Iteration 5 disclosure-candidate qualitative review\n\n"
        "The canonical automatic audit retained **42 lexical-candidate turns across 10 episodes**. "
        "It conservatively marked 32 turns across seven episodes because they contained the ordinary "
        "task-domain words `probe` or `probes`; the other 10 turns contained source-level `score` or "
        "`scored` language and were already marked non-disclosures.\n\n"
        "Complete qualitative review found **zero activation-probe, percentile, recommendation, or "
        "feedback-score leaks**. The seven-episode canonical field is therefore a lexical-candidate "
        "count, not a verified disclosure count.\n\n"
        "This is an additive review. It does not alter transcripts, judgments, inclusion, the frozen "
        "analysis, or any primary result. The public projection contains aggregates only; exact "
        "episode, turn, hash, matched-text, and context evidence remains in the private review.\n"
    )
    manifest = {
        "schema": "grid5_disclosure_qualitative_public_manifest_v1",
        "builder_sha256": builder_hash,
        "private_review_sha256": digest(args.private_out),
        "files": {
            "public_summary.json": digest(public_json),
            "public_summary.md": digest(public_md),
        },
    }
    write_json(args.public_out / "manifest.json", manifest)
    print(json.dumps({"private": str(args.private_out), "public": str(args.public_out), **derived}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--analysis-sha256", required=True)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--private-out", type=Path, default=DEFAULT_PRIVATE)
    parser.add_argument("--public-out", type=Path, default=DEFAULT_PUBLIC)
    args = parser.parse_args()
    for key in ("analysis", "package", "completion", "private_out", "public_out"):
        path = getattr(args, key)
        setattr(args, key, path if path.is_absolute() else (ROOT / path).resolve())
    build(args)


if __name__ == "__main__":
    main()
