"""Exact response-matched context twins: every claim changes truth value."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .build_relational_corpus import build
from .preflight import sha256


def build_twins():
    rng = random.Random(20261016)
    rows = []
    for base in build():
        if base["variant"] != "mixed_0":
            continue
        facts = base["facts"]
        a, b = facts["names"]
        p, _ = facts["managers"]
        city_a, _ = facts["cities"]
        old_a, _ = facts["old"]
        new_a, new_b = facts["new"]
        bigger, smaller = (a, b) if new_a > new_b else (b, a)
        faster, slower = (a, b) if facts["growth"][0] > facts["growth"][1] else (b, a)
        open_company = facts["open"]
        closed_company = b if open_company == a else a
        pairs = [
            (
                f"{a}'s final valuation is {new_a} million credits.",
                f"{b}'s final valuation is {new_a} million credits.",
                "temporal",
            ),
            (
                f"{b}'s final valuation is {new_b} million credits.",
                f"{a}'s final valuation is {new_b} million credits.",
                "temporal",
            ),
            (
                f"{bigger} ended the year with a higher valuation than {smaller}.",
                f"{smaller} ended the year with a higher valuation than {bigger}.",
                "comparison",
            ),
            (
                f"{faster}'s valuation change, measured as final minus initial, is greater than {slower}'s.",
                f"{slower}'s valuation change, measured as final minus initial, is greater than {faster}'s.",
                "change",
            ),
            (
                f"{open_company} released all its software openly.",
                f"{closed_company} released all its software openly.",
                "licensing",
            ),
            (f"{p} manages research at {a}.", f"{p} manages research at {b}.", "role"),
            (
                f"The audit places {a}'s current headquarters in {city_a}.",
                f"The audit places {b}'s current headquarters in {city_a}.",
                "headquarters",
            ),
            (
                f"The old newspaper's location report concerned {a}.",
                f"The old newspaper's location report concerned {b}.",
                "attribution",
            ),
            (
                f"{a}'s initial valuation was {old_a} million credits.",
                f"{b}'s initial valuation was {old_a} million credits.",
                "history",
            ),
        ]
        rng.shuffle(pairs)
        mask = [0] * 5 + [1] * 4
        rng.shuffle(mask)
        parent = base["group"].replace("relational2-", "context-twin-")
        for variant in range(2):
            labels = mask if variant == 0 else [1 - x for x in mask]
            response, spans = "", []
            for (truth, falsehood, family), label in zip(pairs, labels):
                claim = falsehood if label else truth
                response += " " if response else ""
                start = len(response)
                response += claim
                spans.append(
                    {
                        "start": start,
                        "end": len(response),
                        "text": claim,
                        "label": label,
                        "family": family,
                    }
                )
            for world in range(2):
                messages = [dict(m) for m in base["messages"]]
                if world:
                    marker = "__CONTEXT_TWIN_COMPANY_SWAP__"
                    text = messages[1]["content"]
                    if marker in text:
                        raise ValueError("swap marker already present")
                    swapped = text.replace(a, marker).replace(b, a).replace(marker, b)
                    restored = (
                        swapped.replace(a, marker).replace(b, a).replace(marker, b)
                    )
                    if restored != text:
                        raise ValueError("context swap is not exactly reversible")
                    messages[1]["content"] = swapped
                rows.append(
                    {
                        "id": f"{parent}-{variant}-{world}",
                        "group": parent,
                        "split": base["split"],
                        "source": "controlled_relational",
                        "messages": messages,
                        "response": response,
                        "variant": f"mixed_{variant}",
                        "world": world,
                        "spans": [{**s, "label": s["label"] ^ world} for s in spans],
                        "label": 1,
                        "cache_tokens": False,
                        "facts": facts,
                        "context_company_swap": bool(world),
                        "label_status": "Mechanically labelled factual consistency. Every identical response occurs under opposite source contexts with complementary claim labels. Same template families across splits; no intent claim.",
                    }
                )
    for i in range(0, len(rows), 2):
        first, second = rows[i : i + 2]
        if first["response"] != second["response"] or first["group"] != second["group"]:
            raise ValueError("matched response accounting failed")
        for a, b in zip(first["spans"], second["spans"]):
            if a["text"] != b["text"] or a["label"] == b["label"]:
                raise ValueError("each exact claim must have opposite labels")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new corpus required")
    rows = build_twins()
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "rows": len(rows),
        "groups": len({r["group"] for r in rows}),
        "sha256": sha256(args.out.read_text()),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "text_only_control": "Any deterministic context-blind score has pooled AUROC exactly 0.5 within each split because every exact response/claim has complementary labels under paired source contexts.",
        "claim_limit": "Controlled factual-consistency diagnostic; no generated-deception or natural-language generalization claim.",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
