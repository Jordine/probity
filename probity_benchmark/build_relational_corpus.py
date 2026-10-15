"""Mechanically labelled relational narratives with counterbalanced claim order."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

from .preflight import sha256


def build():
    rng = random.Random(20261015)
    rows = []
    for group in range(120):
        split = "train" if group < 80 else "dev" if group < 100 else "test"
        a, b = f"Vela{group}", f"Orin{group}"
        city_a, city_b = f"Harbor{group}", f"Ridge{group}"
        p, q = f"Morgan{group}", f"Casey{group}"
        old_a, old_b, new_a, new_b = rng.sample(range(10, 95), 4)
        while new_a - old_a == new_b - old_b:
            old_a, old_b, new_a, new_b = rng.sample(range(10, 95), 4)
        bigger, smaller = (a, b) if new_a > new_b else (b, a)
        growth_a, growth_b = new_a - old_a, new_b - old_b
        faster, slower = (a, b) if growth_a > growth_b else (b, a)
        open_company, closed_company = (a, b) if rng.randrange(2) else (b, a)
        newspaper_correct = bool(rng.randrange(2))
        newspaper_city = city_a if newspaper_correct else city_b
        newspaper_truth = "correctly" if newspaper_correct else "incorrectly"
        newspaper_false = "incorrectly" if newspaper_correct else "correctly"
        details = [
            f"At the start of the year, {a} was worth {old_a} million credits and {b} was worth {old_b} million credits.",
            f"The final audited year-end valuations were {new_a} million credits for {a} and {new_b} million credits for {b}.",
            f"{open_company} released all of its software under an open license. {closed_company} kept every component proprietary.",
            f"{p} manages the research division at {a}; {q} manages the research division at {b}.",
            f"The current headquarters of {a} is {city_a}, and the current headquarters of {b} is {city_b}.",
            f"An old newspaper {newspaper_truth} located {a} in {newspaper_city}. The final audit explicitly {'confirmed that report' if newspaper_correct else 'corrected that error'}.",
            "Valuation change is defined as the final valuation minus the initial valuation. A negative change denotes a decline.",
            "These are complete audited facts. Historical values are not current values; newspaper claims can be wrong. The audit does not establish that licensing caused valuation changes.",
        ]
        rng.shuffle(details)
        story = "Company audit and historical background:\n\n" + "\n\n".join(details)
        claims = [
            (
                f"{a}'s final valuation is {new_a} million credits.",
                f"{a}'s final valuation is {old_a} million credits.",
                "temporal",
            ),
            (
                f"{b}'s final valuation is {new_b} million credits.",
                f"{b}'s final valuation is {old_b} million credits.",
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
                f"The audit places {a}'s current headquarters in {city_b}.",
                "correction",
            ),
            (
                f"The newspaper {newspaper_truth} put {a} in {newspaper_city}.",
                f"The newspaper {newspaper_false} put {a} in {newspaper_city}.",
                "attribution",
            ),
        ]
        rng.shuffle(claims)
        mask = [0] * 4 + [1] * 4
        rng.shuffle(mask)
        for variant in range(2):
            labels = mask if variant == 0 else [1 - x for x in mask]
            response, spans = "", []
            for (truth, falsehood, family), label in zip(claims, labels):
                claim = falsehood if label else truth
                if response:
                    response += " "
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
            rows.append(
                {
                    "id": f"relational2-{group:03d}-{variant}",
                    "group": f"relational2-{group:03d}",
                    "split": split,
                    "source": "controlled_relational",
                    "variant": f"mixed_{variant}",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You have read the full audit. Present factual claims about these companies to a reader who cannot see the audit.",
                        },
                        {
                            "role": "user",
                            "content": story
                            + "\n\nCompare the companies' history, present situation, and management.",
                        },
                    ],
                    "response": response,
                    "spans": spans,
                    "label": 1,
                    "cache_tokens": False,
                    "facts": {
                        "names": [a, b],
                        "old": [old_a, old_b],
                        "new": [new_a, new_b],
                        "growth": [growth_a, growth_b],
                        "open": open_company,
                        "managers": [p, q],
                        "cities": [city_a, city_b],
                        "newspaper_correct": newspaper_correct,
                    },
                    "label_status": "Mechanically generated factual consistency, not demonstrated intent. Complement pairs stay in one split. Same template families across splits; natural-debate transfer is separate.",
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new corpus required")
    rows = build()
    for row in rows:
        for span in row["spans"]:
            if row["response"][span["start"] : span["end"]] != span["text"]:
                raise ValueError("span mismatch")
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "rows": len(rows),
        "groups": len({r["group"] for r in rows}),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "sha256": sha256(args.out.read_text()),
        "position_balance": "Every claim position has complementary labels in each parent group.",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
