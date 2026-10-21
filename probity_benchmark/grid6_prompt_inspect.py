"""Render complete exact-context records for fixed prompt-screen families."""
import argparse
import hashlib
import html
import json
import os
from collections import defaultdict
from pathlib import Path

from . import grid6_prompt as prompt
from .grid6_live import file_hash, read


def pre(value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    return "<pre>" + html.escape(text) + "</pre>"


def details(title, value):
    return "<details><summary>" + html.escape(title) + "</summary>" + pre(value) + "</details>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new inspector output required")
    manifest, problems, branches, _ = prompt.load(args.run)
    by_family = defaultdict(list)
    for branch in branches:
        by_family[branch["family_id"]].append(branch)
    chosen = manifest["families"][: args.families]
    args.out.mkdir(parents=True)
    style = "<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em;color:#17202a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f7;padding:1em;font:13px monospace}details{margin:1em 0}summary{cursor:pointer}article{border-top:2px solid #ccd1d1;margin-top:2em}.note{background:#fff7dd;padding:1em}a{color:#15568b}</style>"
    pages, hashes = [], {}
    for family in chosen:
        for branch in sorted(by_family[family], key=lambda row: (row["level"], row["policy"])):
            problem = problems[branch["problem_id"]]
            body = [
                "<!doctype html><meta charset=\"utf-8\">",
                style,
                "<h1>" + html.escape(branch["level"] + " · " + branch["policy"]) + "</h1>",
                "<p class=\"note\">Every model-visible message and raw provider response is shown. The receipt's request field is the canonical scientific request. When transport_request is present, it is the actual on-wire body; only provider routing differs, while messages, schema and sampling remain identical. The feedback packet is inside the public request. Private preparation calls requested reasoning and preserve whatever reasoning the provider returned. The public call receives only the parsed plan, not that reasoning. Public calls disabled reasoning. Provider-internal state that was never returned is unavailable.</p>",
                details("Complete source problem", problem.__dict__),
                details("Branch state and all resulting probe scores", branch),
            ]
            plans = []
            for plan in manifest["plans"]:
                if plan["pair_id"] != branch["pair_id"]:
                    continue
                job = prompt.plan_job(manifest, problem, plan)
                receipt = prompt.recover(args.run, job)
                path = args.run / "requests" / (job["event_id"] + ".json")
                hashes[str(path)] = file_hash(path)
                plans.append({"speaker": plan["speaker"], "exact_receipt": receipt})
            body.append(details("Private preparation calls: request, returned plan, provider reasoning, receipt", plans))
            for index, turn in enumerate(branch["turns"]):
                path = args.run / "requests" / (turn["event_id"] + ".json")
                receipt = read(path)
                hashes[str(path)] = file_hash(path)
                raw = json.loads(receipt["response_body"])
                body.extend(
                    [
                        "<article><h2>Turn " + str(index + 1) + ": " + html.escape(turn["speaker"]) + "</h2>",
                        pre(turn["argument"]),
                        details("Canonical scientific request and settings", receipt["request"]),
                        details("Actual submitted body (transport override when present)", receipt.get("transport_request", receipt["request"])),
                        details("Raw provider response body", raw),
                        details("Full receipt and hashes", receipt),
                        details("Accepted turn and resulting live probe score", turn),
                        "</article>",
                    ]
                )
            filename = branch["branch_id"] + ".html"
            (args.out / filename).write_text("\n".join(body))
            pages.append({"family": family, "level": branch["level"], "policy": branch["policy"], "page": filename})
    index = [
        "<!doctype html><meta charset=\"utf-8\">",
        style,
        "<h1>Prompt screen: exact model-visible records</h1>",
        "<p>The first two frozen-manifest families are shown, with all four instruction × feedback cells and all six turns. These examples were selected without inspecting outcomes and are not prevalence estimates.</p><ul>",
    ]
    for page in pages:
        index.append("<li><a href=\"" + page["page"] + "\">" + html.escape(page["level"] + " · " + page["policy"] + " · " + page["family"][:12]) + "</a></li>")
    index.extend(["</ul>", "<p>Full private archive: <a href=\"" + html.escape(os.path.relpath(args.run, args.out)) + "\">production run directory</a>.</p>"])
    (args.out / "index.html").write_text("\n".join(index))
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "selection": "first families in frozen prompt manifest",
                "selected_families": chosen,
                "pages": pages,
                "production_manifest_sha256": file_hash(args.run / "manifest.json"),
                "receipt_sha256": hashes,
                "script_sha256": file_hash(__file__),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"families": len(chosen), "pages": len(pages)}, indent=2))


if __name__ == "__main__":
    main()
