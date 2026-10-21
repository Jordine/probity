"""Build the final iteration-five recovery report with final v9 rental binding.

This additive wrapper preserves the reviewed preparation/accounting report and
adds direct evidence for the rental that produced the completed native judge
outputs.  It is offline and makes no API or provider calls.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from reports import build_grid5_recovery_report as v1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build(root):
    root = Path(root)
    public, private = v1.build(root)
    run = root / "runs/iteration5"
    meta = run / "coordinator"
    paths = {
        "rental": meta / "rental_v9.json",
        "settlement": meta / "rental_settlement_v9.json",
        "absence": meta / "rental_absence_verified_v9.json",
        "native_complete": meta / "remote_native_evidence_v30/native_judge_v30_complete.json",
        "native_manifest": run / "native_judge_v1/manifest.json",
        "native_evaluations": run / "native_judge_v1/evaluations.jsonl",
        "native_summary": run / "native_judge_v1/summary.json",
        "final_completion": meta / "final_analysis_complete_v1.json",
        "analysis_recovery_readiness": meta / "finish_analysis_recovery_ready_v32.json",
    }
    values = {name: read(path) for name, path in paths.items() if path.suffix == ".json"}
    rental, settlement, absence = values["rental"], values["settlement"], values["absence"]
    native, manifest = values["native_complete"], values["native_manifest"]
    if (settlement != absence or settlement["instance_id"] != rental["instance_id"]
            or not settlement.get("verified_absent_at_unix")):
        raise ValueError("final v9 rental is not exactly absent and settled")
    if (native.get("complete") is not True or native.get("jobs") != 1328
            or native.get("reused") != 192 or native.get("fresh") != 1136):
        raise ValueError("final native completion receipt is incomplete")
    for name, field in (("native_manifest", "manifest_sha256"),
                        ("native_evaluations", "evaluations_sha256"),
                        ("native_summary", "summary_sha256")):
        if sha(paths[name]) != native[field]:
            raise ValueError("final native output differs from completion receipt: " + name)
    if (manifest.get("jobs_expected") != native["jobs"]
            or manifest.get("jobs_sha256") != read(meta / "native_jobs_ready_v1.json")["jobs_sha256"]):
        raise ValueError("final native manifest differs from reviewed job inventory")
    ledger = read(run / "budget.json")
    entry = ledger["entries"].get("gpu:iteration5:v9")
    if (not entry or entry.get("status") != "settled"
            or abs(float(entry["amount_usd"]) - float(settlement["amount_usd"])) > 1e-9):
        raise ValueError("final v9 rental settlement differs from ledger")
    active = {key: row for key, row in ledger["entries"].items() if row.get("status") != "settled"}
    if active:
        raise ValueError("campaign still has active budget entries")
    public["schema"] = "grid5_public_recovery_report_v2"
    public["as_of_utc"] = datetime.now(timezone.utc).isoformat()
    public["final_native_release"] = {
        "instance_id": rental["instance_id"],
        "label": rental["label"],
        "verified_absent_at_unix": settlement["verified_absent_at_unix"],
        "settled_accounting_usd": settlement["amount_usd"],
        "native_decisions": native["jobs"],
        "exact_cache_hits": native["reused"],
        "fresh_decisions": native["fresh"],
        "manifest_sha256": native["manifest_sha256"],
        "evaluations_sha256": native["evaluations_sha256"],
        "completion_marker_sha256": sha(paths["final_completion"]),
        "analysis_recovery_readiness_sha256": sha(paths["analysis_recovery_readiness"]),
    }
    public["status"] = "complete_with_verified_release_marker"
    bound = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size}
             for path in paths.values()}
    private["schema"] = "grid5_private_recovery_audit_v2"
    private["public_summary"] = public
    private["final_native_inputs"] = bound
    private["builder_v1_sha256"] = sha(v1.__file__)
    private["builder_sha256"] = sha(__file__)
    return public, private


def markdown(report):
    text = v1.markdown(report)
    native = report["final_native_release"]
    text += (
        "\n## Final native judging and rental release\n\n"
        f"The final native judge produced {native['native_decisions']} decisions: "
        f"{native['exact_cache_hits']} exact cache hits and {native['fresh_decisions']} fresh decisions. "
        f"Rental `{native['instance_id']}` is independently verified absent and its campaign-ledger "
        f"accounting is settled at ${native['settled_accounting_usd']:.6f}.\n"
    )
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    public, private = build(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("private_audit.json", private), ("public_summary.json", public)):
        (args.out / name).write_text(json.dumps(value, indent=2) + "\n")
    (args.out / "public_summary.md").write_text(markdown(public))
    print(json.dumps({"status": public["status"], "totals": public["totals"],
                      "final_native_release": public["final_native_release"]}))


if __name__ == "__main__":
    main()
