"""Private final inspection pack adapter for canonical ``debates_v4``.

Iteration five's reviewed baseline recovery completed in ``debates_v4``.  The
earlier v5 reader adapter was prepared before that recovery and hard-coded
``debates_v1`` only in its route-provenance check.  This offline additive
adapter preserves every v5/full-builder gate and changes that one canonical
baseline directory expectation.
"""

import argparse
import json
import os
from pathlib import Path

from reports import build_grid5_inspection_pack as original
from reports import build_grid5_inspection_pack_v5 as v5


ROOT, RUN, META = v5.ROOT, v5.RUN, v5.META


def validate_route_v33(route, ready, freeze_sha, root=ROOT):
    require = original.require
    require(route.get("schema") == "grid5_analysis_route_provenance_v5"
            and route.get("recovery_freeze_sha256") == freeze_sha
            and route.get("route_proposal_sha256") == ready["route_proposal_sha256"]
            and route.get("routed_missing_requests") == ready["routed_missing_requests"],
            "final route provenance differs from reviewed v5 readiness")
    affected = {(e["tag"]["problem_id"], e["tag"]["model"])
                for e in ready["routed_missing_requests"]}
    require(len(affected) == 4 and len(ready["routed_missing_requests"]) == 8,
            "exact four-question/eight-plan routing stratum required")
    seen, hashes = set(), {}
    trajectories = route.get("affected_trajectories", [])
    require(len(trajectories) == 8, "all eight affected trajectories required")
    for row in trajectories:
        arm = row["arm"]
        require(arm in ("baseline", "feedback"), "unknown route provenance arm")
        rel = Path(row["path"])
        directory = "debates_v4" if arm == "baseline" else "feedback_v1"
        require(not rel.is_absolute() and ".." not in rel.parts
                and rel.parent == Path("runs/iteration5") / directory
                and rel.name.startswith("episode_") and rel.suffix == ".json",
                "unexpected trajectory path")
        path = root / rel
        require(not path.is_symlink() and original.digest(path) == row["sha256"],
                "affected trajectory changed")
        episode = original.read(path)
        identity = (arm, row["problem_id"], row["model"])
        require(identity not in seen and (row["problem_id"], row["model"]) in affected,
                "duplicate or unrelated route trajectory")
        seen.add(identity)
        require(all(episode[k] == row[k] for k in ("episode_id", "problem_id", "model"))
                and episode.get("complete") is True and len(episode["turns"]) == 6,
                "route trajectory identity/completion differs")
        hashes[str(path)] = row["sha256"]
    require(seen == {(arm, *key) for arm in ("baseline", "feedback") for key in affected},
            "incomplete route trajectory stratum")
    analyses = {str(p.relative_to(root)): original.digest(p)
                for p in (root / "runs/iteration5/analysis_v1").rglob("*") if p.is_file()}
    require(analyses and route.get("analysis_sha256") == analyses,
            "complete final analysis provenance differs")
    return hashes


def collect(args):
    # The old nested freeze validator correctly notices that the compiler was
    # repaired after v5.  Re-verify the same lineage here while allowing only
    # that reviewed replacement, bound to the completed native readiness.
    original_freeze = original.read(META / "execution_code_freeze_v1.json")
    prior_freeze = original.read(META / "execution_recovery_freeze_v4.json")
    recovery_freeze = original.read(META / "execution_recovery_freeze_v5.json")
    original.require(original.digest(META / "execution_code_freeze_v1.json")
                     == v5.load_prework().ORIGINAL_FREEZE_SHA,
                     "original execution freeze receipt changed")
    original.require(original.digest(META / "execution_recovery_freeze_v4.json")
                     == v5.load_prework().PRIOR_FREEZE_SHA,
                     "prior recovery freeze receipt changed")
    original.require(original.digest(META / "execution_recovery_freeze_v5.json")
                     == args.recovery_freeze_sha256,
                     "v5 recovery freeze receipt changed")
    compiler_key = "runs/iteration5/coordinator/compile_packages_v1.py"
    for rel, expected in original_freeze["sha256"].items():
        if rel != compiler_key:
            original.require(original.digest(ROOT / rel) == expected,
                             "non-compiler original freeze member changed: " + rel)
    native_ready = original.read(META / "native_jobs_ready_v1.json")
    original.require(original.digest(ROOT / compiler_key) == native_ready["compiler_sha256"],
                     "reviewed canonical compiler lineage changed")
    for freeze, label in ((prior_freeze, "v4"), (recovery_freeze, "v5")):
        for rel, expected in freeze["sha256"].items():
            original.require(original.digest(ROOT / rel) == expected,
                             label + " recovery freeze member changed: " + rel)
    freezes = {
        "original_execution_freeze_sha256": v5.load_prework().ORIGINAL_FREEZE_SHA,
        "prior_recovery_freeze_sha256": v5.load_prework().PRIOR_FREEZE_SHA,
        "recovery_freeze_sha256": args.recovery_freeze_sha256,
    }
    prework = v5.load_prework()
    ready = original.read(META / v5.READY_NAME)
    _, audit = prework.audit_recovery(ready, freezes)
    original.require(audit["complete_preparations"] == 568
                     and audit["unchanged_prior_preparations"] == 560,
                     "complete v5 preparation audit required")
    route = original.read(META / v5.ROUTE_NAME)
    affected_hashes = validate_route_v33(route, ready, args.recovery_freeze_sha256)
    builder = v5.adapted_collect()
    out, report = builder(args)
    original.require(all(report["input_sha256"].get(k) == value
                         for k, value in affected_hashes.items()),
                     "affected trajectories not bound into complete benchmark package")
    report["schema"] = "grid5_private_inspection_pack_v5"
    report["adapter_sha256"] = original.digest(v5.__file__)
    report["version_adapter"] = {
        "frozen_builder_sha256": v5.FROZEN_BUILDER_SHA,
        "sole_original_function_change":
            "preparations_recovery_ready_v4.json -> preparations_recovery_ready_v5.json",
        "v5_preparation_audit": audit,
        "analysis_route_provenance": route,
        "interpretation": "The eight routed plans form a failure-selected operational amendment, not a randomized provider comparison. Private diagnostics never enter this pack as canonical preparations.",
    }
    for path in (META / v5.READY_NAME, META / v5.ROUTE_NAME,
                 META / "execution_recovery_freeze_v5.json",
                 META / "prework_resume_v5.py", Path(v5.__file__)):
        report["input_sha256"][str(path)] = original.digest(path)
    report["schema"] = "grid5_private_inspection_pack_v33"
    report["final_adapter_sha256"] = original.digest(__file__)
    report["final_version_adapter"] = {
        "parent_adapter_sha256": original.digest(v5.__file__),
        "sole_parent_change": "canonical baseline route path debates_v1 -> debates_v4",
        "reason": "The reviewed baseline recovery completed the canonical expansion in debates_v4.",
        "stale_original_freeze_exception": {
            "path": compiler_key,
            "original_sha256": original_freeze["sha256"][compiler_key],
            "canonical_sha256": native_ready["compiler_sha256"],
        },
    }
    report["input_sha256"][str(Path(__file__))] = original.digest(__file__)
    return out, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "completion", "selection", "preparation-ready",
                 "route-provenance", "recovery-freeze"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out, report = collect(args)
    page = original.render(report)
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (out / "index.html").write_text(page)
    manifest = {
        "schema": report["schema"], "label": original.LABEL, "private_only": True,
        "units": 4, "transcripts": 8, "public_turns": 48, "judge_cells": 16,
        "builder_sha256": report["builder_sha256"],
        "adapter_sha256": report["adapter_sha256"],
        "final_adapter_sha256": report["final_adapter_sha256"],
        "input_sha256": report["input_sha256"],
        "output_sha256": {name: original.digest(out / name)
                          for name in ("evidence.json", "index.html")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"private_inspection_pack": str(out), "paid_calls": 0,
                      "canonical_baseline": "debates_v4"}))


if __name__ == "__main__":
    main()
