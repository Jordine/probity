"""Export the already locked public sample, omitting all judge/probe fields."""
import hashlib
import json
from pathlib import Path
from probity_benchmark.preflight import load_problems

base = Path("runs/iteration3")
plan_path = base / "environment/fresh_validity/manual_sample_plan_v1.json"
plan = json.loads(plan_path.read_text())
source = base / "coordinator/holdout_families_source_v5.jsonl"
problems, audit = load_problems(source)
assert not audit["errors"]
source_rows = list(map(json.loads, source.read_text().splitlines()))
source_map = {r["parent_source_line"]: (p, r) for p, r in zip(problems, source_rows)}
corpus = base / "final/debates_v1"
summary = json.loads((corpus / "summary.json").read_text())
assert not summary["errors"] and summary["episodes"] == summary["expected_episodes"] == 48
selected = []
for path in sorted(corpus.glob("episode_*.json")):
    episode = json.loads(path.read_text())
    if episode["parent_source_line"] in plan["parent_source_lines"]:
        assert episode["complete"] and len(episode["turns"]) == 6
        selected.append((path, episode))
assert len(selected) == plan["expected_episodes"] == 8
assert sum(len(e["turns"]) for _, e in selected) == plan["expected_turns"] == 48
out = base / "final/manual_public_sample_v1"
assert not out.exists()
out.mkdir()
for parent in plan["parent_source_lines"]:
    problem, source_row = source_map[parent]
    entries = [(p, e) for p, e in selected if e["parent_source_line"] == parent]
    assert len(entries) == 2 and len({e["model"] for _, e in entries}) == 2
    text = f"Parent source {parent}\nQuestion: {problem.question}\nOriginal options: {problem.options}\nOriginal gold: {problem.gold}\n\nCOMPLETE AUTHENTIC SOURCE\n{problem.story}\n\n"
    for path, episode in entries:
        text += f"COMPLETE PUBLIC EPISODE {episode['episode_id']}\nModel: {episode['model']}\nSwapped options: {episode['swapped']}\nDisplayed gold: {episode['gold']}\n"
        for i, turn in enumerate(episode["turns"]):
            text += f"\nTurn {i}; round {turn['round']}; speaker {turn['speaker']}; assigned {turn['assigned_answer']}; answer field {turn['answer']}\n{turn['argument']}\n"
    (out / f"source_{parent}_and_public.txt").write_text(text)
manifest = {
    "selection_plan": str(plan_path),
    "selection_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "episodes": len(selected),
    "turns": sum(len(e["turns"]) for _, e in selected),
    "all_selected_arguments_complete": True,
    "judge_and_probe_fields_exported": False,
    "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, plan_path] + [p for p, _ in selected]},
    "exports_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.txt")},
    "limit": "This fixed manual sample checks qualitative source fidelity; speaker role is not a local truth label. Selection cannot change after outputs are read.",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"episodes": manifest["episodes"], "turns": manifest["turns"], "out": str(out)}, indent=2))
