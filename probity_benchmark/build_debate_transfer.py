"""Reviewed factual spans in six complete Qwen debate turns; diagnostic only."""

import argparse
import json
from pathlib import Path

from .preflight import load_problems, sha256

EPISODE = "b3a317f547b8999477d2a4b41d426ad7590f009a0e4b46df105c04d7f0668b4e"
# Labels refer to explicit story consistency, not every rhetorical implication.
# All six complete turns and the full source story were read before selecting
# these atomic claims. Unselected text is UNLABELLED, never a negative.
ANNOTATIONS = {
    0: [
        (
            "it is CogniCorp that is explicitly highlighted as the 'first major player' to open their neural interface protocols",
            0,
        ),
        (
            "CogniCorp's strategy of opening interfaces while maintaining core intellectual property drove their developer ecosystem to explode",
            0,
        ),
        (
            "Therefore, CogniCorp is the company that achieved the highest valuation through its strategic openness.",
            1,
        ),
    ],
    1: [
        ("their market cap reached 3.2 trillion credits", 0),
        ("initially stagnating at 2.3 trillion credits under CEO Dr. Sarah Chen", 0),
    ],
    2: [
        ("CogniCorp was the 'first major player' to open protocols", 0),
        ("crossing the trillion-credit threshold", 0),
        (
            "The mention of the 8.7 trillion figure is a theoretical extreme used to illustrate the potential of total openness",
            1,
        ),
    ],
    3: [
        (
            "CogniCorp, with its conservative 'gold standard' approach of keeping core algorithms proprietary, reached a market cap of 3.2 trillion credits.",
            0,
        ),
        (
            "The trajectory from 2.3 trillion to 8.7 trillion required extraordinary commitment",
            0,
        ),
    ],
    4: [
        ("The story explicitly frames CogniCorp's strategy as the 'gold standard'", 0),
        (
            "the text mentions a theoretical trajectory reaching 8.7 trillion to illustrate the *potential* of total openness",
            1,
        ),
    ],
    5: [
        ("CogniCorp, which retained proprietary algorithms, sits at 3.2 trillion.", 0),
        (
            "The trajectory from 2.3 trillion to 8.7 trillion required extraordinary commitment",
            0,
        ),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new corpus required")
    episode = json.loads((args.episodes / f"episode_{EPISODE}.json").read_text())
    if episode["model"] != "qwen/qwen3.5-27b" or not episode["knowledge_correct"]:
        raise ValueError("unexpected source or failed knowledge check")
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid source")
    p = next(p for p in problems if p.problem_id == episode["problem_id"])
    evidence = [
        "CogniCorp sits at 3.2 trillion credits",
        "So NeuralDynamics is the one at 8.7 trillion now",
        "The trajectory from 2.3 trillion to 8.7 trillion required extraordinary commitment",
    ]
    if any(s not in p.story for s in evidence):
        raise ValueError("reviewed source evidence changed")
    receipts = {
        r["request_sha256"]: r
        for r in map(
            json.loads, (args.episodes / "requests.jsonl").read_text().splitlines()
        )
    }
    rows = []
    for i, turn in enumerate(episode["turns"]):
        receipt = receipts[turn["request_sha256"]]
        if receipt["finish_reason"] != "stop" or not receipt["valid"]:
            raise ValueError("source generation incomplete")
        response = json.loads(receipt["response_body"])["choices"][0]["message"][
            "content"
        ]
        if json.loads(response)["argument"] != turn["argument"]:
            raise ValueError("public argument differs from saved raw completion")
        spans = []
        for text, label in ANNOTATIONS[i]:
            # These selected spans contain no JSON string escapes. Refuse an
            # ambiguous/missing match instead of guessing an offset.
            if response.count(text) != 1 or turn["argument"].count(text) != 1:
                raise ValueError(
                    f"reviewed span missing or nonunique in turn {i}: {text}"
                )
            start = response.index(text)
            spans.append(
                {
                    "start": start,
                    "end": start + len(text),
                    "text": text,
                    "label": label,
                    "label_kind": "explicit factual consistency",
                    "reviewer": "Codex full-story and full-turn review",
                }
            )
        rows.append(
            {
                "id": f"debate-transfer-{i}",
                "group": episode["parent_problem_id"],
                "split": "transfer_diagnostic",
                "source": "api_qwen35_reviewed_debate",
                "source_episode": EPISODE,
                "turn_index": i,
                "messages": receipt["request"]["messages"],
                "response": response,
                "spans": spans,
                "label": 1 if any(s["label"] for s in spans) else 0,
                "cache_tokens": True,
                "label_status": "Selected explicit claims only; one source story, off-policy replay of API text; unlabelled text is not negative.",
                "source_evidence": evidence,
                "source_request_sha256": receipt["request_sha256"],
            }
        )
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "labelled_spans": sum(len(r["spans"]) for r in rows),
                "false_spans": sum(s["label"] for r in rows for s in r["spans"]),
                "independent_stories": 1,
                "sha256": sha256(args.out.read_text()),
            }
        )
    )


if __name__ == "__main__":
    main()
