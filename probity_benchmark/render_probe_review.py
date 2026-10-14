"""Render complete, source-linked probe reviews from portable score exports."""

import argparse
import html
import json
from pathlib import Path

import numpy as np

from .preflight import load_problems


def render_review(records, episodes, problems):
    def escape(value):
        return html.escape(str(value), quote=True)

    pages = []
    for row in records:
        episode = episodes[row["episode_id"]]
        problem = problems[episode["problem_id"]]
        options = problem.options[::-1] if episode["swapped"] else problem.options
        views = []
        for name in ["sentence", "clause"]:
            units = row[name + "_scores"]
            if "".join(unit["text"] for unit in units) != row["argument"]:
                raise ValueError("display must preserve every argument character")
            low, high = np.quantile([unit["token_head"] for unit in units], [0.1, 0.9])
            spans = []
            for unit in units:
                intensity = float(
                    np.clip((unit["token_head"] - low) / max(high - low, 1e-12), 0, 1)
                )
                tooltip = escape(
                    json.dumps(
                        {
                            k: unit[k]
                            for k in [
                                "token_head",
                                "pooled_head",
                                "raw_start",
                                "raw_end",
                            ]
                        }
                    )
                )
                spans.append(
                    f'<span title="{tooltip}" style="background:rgba(215,75,65,{0.06 + 0.38 * intensity:.3f})">{escape(unit["text"])}</span>'
                )
            views.append(f'<div class="argument {name}-view">{"".join(spans)}</div>')
        peak = max(row["eight_token_windows"], key=lambda value: value["token_mean"])
        if row["raw_response"][peak["raw_start"] : peak["raw_end"]] != peak["text"]:
            raise ValueError("peak phrase offsets disagree with full completion")
        claims = "".join(
            f'<tr><td>{escape(claim.get("display_text", claim["text"]))}</td><td>{"Contradicted" if claim["label"] else "Supported"} (silver)</td><td>{escape(json.dumps(claim["raw_scores"]))}</td></tr>'
            for claim in row["reviewed_claims"]
        )
        turn = episode["turns"][row["turn_index"]]
        pages.append(
            f'<section id="{escape(row["id"])}"><h2>{escape(row["id"])}</h2>'
            f'<p>{escape(row["split"])} · {escape(row["participant_model"])} · speaker {escape(turn["speaker"])} · turn {row["turn_index"] + 1} · <a href="#source-{escape(problem.problem_id)}">Complete source story</a></p>'
            f"<p><strong>Question:</strong> {escape(problem.question)}<br>A: {escape(options[0])}<br>B: {escape(options[1])}</p>"
            + "".join(views)
            + f'<details><summary>Highest-scoring eight-token window</summary><p>This is the complete raw-JSON substring for the selected window, not a factual verdict. Full argument above; all windows are in the score export.</p><pre>{escape(peak["text"])}</pre><p>Mean local token score: {peak["token_mean"]:.6g}; raw offsets [{peak["raw_start"]}, {peak["raw_end"]}).</p></details>'
            + f'<details><summary>Separate whole-response and whole-argument scores</summary><pre>{escape(json.dumps({"whole_response":row["whole_response_scores"],"whole_argument_local":row["whole_argument_local_scores"]},indent=2))}</pre></details>'
            + f'<details><summary>Full JSON completion</summary><pre>{escape(row["raw_response"])}</pre></details>'
            + f"<details><summary>Reviewed claims; all other text is unlabelled</summary><table><thead><tr><th>Complete claim</th><th>Silver label</th><th>Raw local scores</th></tr></thead><tbody>{claims}</tbody></table></details>"
            + f'<details><summary>Task answer and role metadata</summary><p>Correct option: {escape(episode["gold"])}. Assigned option: {escape(turn["assigned_answer"])}. Assignment is not a claim-level label.</p></details></section>'
        )
    sources = "".join(
        f'<details id="source-{escape(pid)}"><summary>Complete source {escape(pid)}</summary><pre>{escape(problems[pid].story)}</pre></details>'
        for pid in sorted(
            {episodes[row["episode_id"]]["problem_id"] for row in records}
        )
    )
    return (
        '<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Probity complete probe review</title>'
        "<style>body{max-width:1000px;margin:auto;padding:2rem;font:16px/1.55 system-ui}pre,.argument{white-space:pre-wrap;overflow-wrap:anywhere}h2{overflow-wrap:anywhere;font-size:1.1rem}section{margin:3rem 0}td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left;overflow-wrap:anywhere}details{margin:1rem 0}table{width:100%;table-layout:fixed}.clause-view{display:none}body.clauses .sentence-view{display:none}body.clauses .clause-view{display:block}.toolbar{position:sticky;top:0;background:white;padding:1rem;border-bottom:1px solid #ddd;z-index:1}</style>"
        "<h1>Complete probe review</h1><p>Every argument is shown in full. Shading compares the local token-head score within that response; it is not a factual verdict or calibrated probability. Hover over a unit for raw token-head and pooled-head scores. Phrase windows and sentence/clause partitions are defined independently of claim labels. Whole-response and DYL scores remain separate. Labels are silver, not exhaustive human ground truth. Features replay complete API text in the native model.</p>"
        '<div class="toolbar"><label>Shading units: <select id="units"><option value="sentence">Sentences</option><option value="clause">Clauses</option></select></label></div>'
        + sources
        + "".join(pages)
        + '<script>document.getElementById("units").addEventListener("change",function(){document.body.classList.toggle("clauses",this.value==="clause")});document.querySelectorAll("section a").forEach(function(a){a.addEventListener("click",function(){document.getElementById(a.getAttribute("href").substring(1)).open=true})});</script></html>'
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    records = [json.loads(line) for line in args.scores.read_text().splitlines()]
    episodes = {
        row["episode_id"]: row
        for row in (
            json.loads(p.read_text()) for p in args.episodes.glob("episode_*.json")
        )
    }
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid source dataset")
    document = render_review(records, episodes, {p.problem_id: p for p in problems})
    args.out.write_text(document)
    print(json.dumps({"rows": len(records), "html_bytes": args.out.stat().st_size}))


if __name__ == "__main__":
    main()
