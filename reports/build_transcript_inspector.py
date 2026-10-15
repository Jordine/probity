"""Build a portable, complete-text inspection artifact from frozen score bundles.

This report does not alter judge prompts or compute experimental outcomes.
Run with PYTHONPATH=. python3 reports/build_transcript_inspector.py --help.
"""

import argparse
import hashlib
import json
from pathlib import Path

from probity_benchmark.judge3_protocol import percentile, validate_bundle
from probity_benchmark.preflight import load_problems

HTML = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Probity transcript inspector</title>
<style>
:root{font-family:system-ui,sans-serif;color:#18232e;background:#f6f7f9;line-height:1.5}
*{box-sizing:border-box}body{margin:0}header{padding:22px 28px;background:#14283c;color:white}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:0 0 10px}p{margin:8px 0}
header p{max-width:960px;font-size:14px;color:#d6e1ec}.toolbar{display:flex;gap:16px;flex-wrap:wrap;padding:16px 28px;background:white;border-bottom:1px solid #cbd2da}
label{display:flex;flex-direction:column;font-size:13px;gap:4px}select,button{font:inherit;padding:8px;border:1px solid #aeb9c5;border-radius:5px;background:white;color:#18232e}
select{max-width:100%}.toggle{flex-direction:row;align-items:center}.layout{display:grid;grid-template-columns:minmax(280px,36%) minmax(0,1fr);gap:20px;padding:20px 28px}
section,.turn{background:white;border:1px solid #d8dee5;border-radius:8px;padding:18px;margin-bottom:16px}.muted{font-size:13px;color:#526171}
.source,.argument,.private{white-space:pre-wrap;overflow-wrap:anywhere}.source{font-size:14px}.argument{line-height:1.8;font-size:16px}.private{font-size:13px}
.turn{scroll-margin:12px}.turn h2{display:flex;justify-content:space-between;gap:10px}.tag{font-weight:400;font-size:12px;color:#526171}
.unit{border-radius:2px;cursor:help}.unit:focus{outline:2px solid #1d5b8e}.detail{font-size:13px;margin-top:14px;border-top:1px solid #e0e4e9;padding-top:10px;white-space:pre-wrap}
summary{cursor:pointer;font-weight:600}.legend{height:12px;max-width:300px;background:linear-gradient(to right,#fff,#fbdfab,#e79542);margin:8px 0}
.peak{display:block;text-align:left;width:100%;margin:8px 0;white-space:pre-wrap;font-size:13px}.gold{border-left:3px solid #427482;padding-left:10px}.hidden{display:none}a{color:#215f99}
@media(max-width:800px){header,.toolbar{padding:16px}.layout{display:block;padding:16px}section,.turn{padding:14px}select{width:100%}.toolbar label{width:100%}}
</style>
<header><h1>Probity transcript inspector</h1><p>Complete source stories, public arguments and fixed probe spans. Scores indicate association with the stated detector target; they are not probabilities of a lie or evidence of intent. This inspection view is separate from the evaluated judge display.</p></header>
<div class="toolbar">
<label>Debate<select id="episode"></select></label>
<label>Detector<select id="detector"></select></label>
<label>Color<select id="color"><option value="percentile">Development-reference percentile</option><option value="none">No highlighting</option></select></label>
<label class="toggle"><input type="checkbox" id="reveal">Show answer key and private preparations</label>
</div>
<main class="layout"><aside>
<section><h2>Question</h2><div id="question"></div><div id="options"></div><p id="gold" class="gold hidden"></p></section>
<section><h2>Signal</h2><p id="signal"></p><div class="legend"></div><p class="muted">0th → 100th percentile in the frozen development reference. Click a highlighted unit for its complete text, raw score and character coordinates. Every character remains displayed.</p><details><summary>Highest-scored units</summary><p class="muted">Convenience navigation only; these units were not selected for judge evaluation or truth labeling.</p><div id="peaks"></div></details></section>
<section><details><summary>Complete source story</summary><p id="source" class="source"></p></details></section>
<section><details><summary>Artifact provenance</summary><pre id="provenance" class="private"></pre></details></section>
</aside><div id="turns"></div></main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("data").textContent);
const el=id=>document.getElementById(id);
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
for(const e of D.episodes){const o=node("option",`Source ${e.parent_source_line} · ${e.model} · ${e.world}`);o.value=e.episode_id;el("episode").append(o);}
for(const b of D.detectors){const o=node("option",b.detector.replaceAll("_"," "));o.value=b.detector;el("detector").append(o);}
function render(){
 const e=D.episodes.find(x=>x.episode_id===el("episode").value), s=D.sources[e.problem_id];
 const b=D.detectors.find(x=>x.detector===el("detector").value), rows=b.rows[e.episode_id];
 const reveal=el("reveal").checked, opts=e.swapped?[s.options[1],s.options[0]]:s.options;
 el("question").textContent=s.question;el("options").replaceChildren(node("p","A: "+opts[0]),node("p","B: "+opts[1]));
 el("gold").textContent="Source key: "+e.gold+". Assigned-answer roles do not label every statement as true or false.";el("gold").classList.toggle("hidden",!reveal);
 el("source").textContent=s.story;
 el("signal").textContent=rows?`${b.granularity} units · target: ${b.target.replaceAll("_"," ")}. ${b.limitation||""}`:"No model-matched checkpoint for this participant. No scores are substituted from another model.";
 el("provenance").textContent=JSON.stringify({report:D.provenance,episode:e.episode_id,head_sha256:b.head_sha256,reference_sha256:b.reference_sha256,reference_population:b.reference_population},null,2);
 el("turns").replaceChildren();el("peaks").replaceChildren();const ranked=[];
 e.turns.forEach((t,index)=>{
  const card=node("article",undefined,"turn");card.id="turn-"+index;
  const title=node("h2",`${t.speaker} · round ${t.round}`);title.append(node("span",`Turn ${index+1} · assigned ${t.assigned_answer}`,"tag"));card.append(title);
  const paragraph=node("div",undefined,"argument"), detail=node("div","Select a scored unit to inspect it.","detail");
  if(rows){rows[index].forEach((u,unit)=>{
   const span=node("span",u.text,"unit");span.tabIndex=0;
   if(el("color").value==="percentile")span.style.backgroundColor=`rgba(229,137,37,${0.65*u.percentile/100})`;
   span.title=`Percentile ${u.percentile.toFixed(3)}; raw score ${u.score.toPrecision(6)}`;
   const show=()=>{detail.textContent=`Unit ${unit+1} · Unicode character offsets [${u.start}, ${u.end})\nPercentile ${u.percentile.toFixed(3)} · raw score ${u.score.toPrecision(8)}\n\n${u.text}`;};
   span.addEventListener("click",show);span.addEventListener("keydown",event=>{if(event.key==="Enter")show();});paragraph.append(span);
   ranked.push({u,index,unit,show,card});
  });}else paragraph.textContent=t.argument;
  card.append(paragraph);if(rows)card.append(detail);
  if(reveal&&e.private_preparations?.[t.speaker]){
   const p=node("details"),summary=node("summary","Complete private preparation for "+t.speaker);p.append(summary,node("pre",JSON.stringify(e.private_preparations[t.speaker],null,2),"private"));card.append(p);
  }
  el("turns").append(card);
 });
 ranked.sort((a,b)=>b.u.percentile-a.u.percentile||a.index-b.index||a.unit-b.unit);
 for(const r of ranked.slice(0,3)){const button=node("button",`Turn ${r.index+1}, unit ${r.unit+1} · ${r.u.percentile.toFixed(1)}th percentile\n${r.u.text}`,"peak");button.addEventListener("click",()=>{r.show();r.card.scrollIntoView({behavior:"smooth",block:"start"});});el("peaks").append(button);}
 if(!ranked.length)el("peaks").textContent="No model-matched scores.";
}
for(const id of ["episode","detector","color","reveal"])el(id).addEventListener("change",render);
render();
</script></html>"""


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    if args.out.exists():
        raise ValueError("new report output required")
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid source dataset")
    by_problem = {p.problem_id: p for p in problems}
    episodes = {}
    for path in sorted(args.episodes.glob("episode_*.json")):
        episode = json.loads(path.read_text())
        if not episode.get("complete") or episode["episode_id"] in episodes:
            raise ValueError("complete unique episodes required")
        if episode["problem_id"] not in by_problem:
            raise ValueError("episode source missing")
        episodes[episode["episode_id"]] = episode
    if not episodes:
        raise ValueError("empty episode corpus")
    detectors = []
    for path in args.bundles:
        bundle = json.loads(path.read_text())
        keyed, covered = validate_bundle(bundle, episodes)
        participant_models = set(bundle["references"])
        if "participant_models" in bundle and set(bundle["participant_models"]) != participant_models:
            raise ValueError("participant metadata differs from reference coverage")
        expected = {
            k for k, e in episodes.items() if e["model"] in participant_models
        }
        if covered != expected:
            raise ValueError("incomplete detector coverage")
        display = {
            k: bundle.get(k)
            for k in (
                "detector",
                "granularity",
                "target",
                "head_sha256",
                "reference_sha256",
                "reference_population",
                "limitation",
            )
        }
        display["rows"] = {}
        for episode_id in sorted(covered):
            episode = episodes[episode_id]
            reference = bundle["references"][episode["model"]]["scores"]
            turns = []
            for i, turn in enumerate(episode["turns"]):
                row = keyed[(episode_id, i)]
                units = [
                    {
                        "text": turn["argument"][start:end],
                        "start": start,
                        "end": end,
                        "score": score,
                        "percentile": percentile(score, reference),
                    }
                    for (start, end), score in zip(row["spans"], row["scores"])
                ]
                if "".join(u["text"] for u in units) != turn["argument"]:
                    raise ValueError(
                        "display does not preserve every argument character"
                    )
                turns.append(units)
            display["rows"][episode_id] = turns
        detectors.append(display)
    report = {
        "sources": {
            k: {"story": p.story, "question": p.question, "options": list(p.options)}
            for k, p in by_problem.items()
        },
        "episodes": sorted(
            episodes.values(),
            key=lambda e: (e["parent_source_line"], e["model"], e["world"]),
        ),
        "detectors": detectors,
        "provenance": {
            "dataset_sha256": file_hash(args.dataset),
            "score_bundle_sha256": {str(p): file_hash(p) for p in args.bundles},
            "builder_sha256": file_hash(Path(__file__)),
            "episodes": len(episodes),
            "full_text_preserved": True,
            "status": "Inspection artifact only; peak-unit navigation was not an evaluated judge treatment.",
        },
    }
    # Keep generated model/source text inert, including literal closing script tags.
    payload = (
        json.dumps(report, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    assert json.loads(payload) == report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(HTML.replace("__DATA__", payload))
    receipt = {
        **report["provenance"],
        "html_sha256": file_hash(args.out),
        "html_bytes": args.out.stat().st_size,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
