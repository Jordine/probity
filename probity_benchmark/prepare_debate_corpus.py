"""Preserve complete generated debate turns for label-independent feature caching."""

import argparse
import json
from pathlib import Path

from .evaluate_debate_transfer import argument_mapping
from .preflight import sha256


def returned_reasoning(receipt):
    """Fail closed when a reasoning-enabled action lacks a replayable trace."""
    enabled = receipt["request"].get("reasoning", {}).get("enabled", False)
    message = json.loads(receipt["response_body"])["choices"][0]["message"]
    reasoning = message.get("reasoning")
    details = message.get("reasoning_details") or []
    if not enabled:
        if reasoning or details:
            raise ValueError("unexpected reasoning on a thinking-disabled action")
        return None
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("thinking-enabled action has no complete returned trace")
    if not details or any(d.get("type") != "reasoning.text" for d in details):
        raise ValueError("opaque or missing reasoning details cannot be replayed")
    if "".join(d.get("text", "") for d in details) != reasoning:
        raise ValueError("reasoning text/details disagree")
    return reasoning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output path required")
    summary = json.loads((args.episodes / "summary.json").read_text())
    if summary["errors"] or summary["episodes"] != summary["expected_episodes"]:
        raise ValueError("debates incomplete")
    receipts = {
        r["request_sha256"]: r
        for r in map(
            json.loads, (args.episodes / "requests.jsonl").read_text().splitlines()
        )
        if r["valid"]
    }
    rows = []
    for path in sorted(args.episodes.glob("episode_*.json")):
        episode = json.loads(path.read_text())
        if episode["model"] != args.participant:
            continue
        for i, turn in enumerate(episode["turns"]):
            receipt = receipts[turn["request_sha256"]]
            if receipt["finish_reason"] != "stop":
                raise ValueError("incomplete generation")
            response = json.loads(receipt["response_body"])["choices"][0]["message"][
                "content"
            ]
            argument, _ = argument_mapping(response)
            if argument != turn["argument"]:
                raise ValueError("raw/public argument mismatch")
            rows.append(
                {
                    "id": "debate2-" + episode["episode_id"] + f"-{i}",
                    "group": episode["parent_problem_id"],
                    "split": episode["split"],
                    "source": "api_debate_silver",
                    "participant_model": episode["model"],
                    "source_episode": episode["episode_id"],
                    "turn_index": i,
                    "parent_source_line": episode["parent_source_line"],
                    "world": episode["world"],
                    "messages": receipt["request"]["messages"],
                    "response": response,
                    "response_reasoning": returned_reasoning(receipt),
                    "spans": [],
                    "label": None,
                    "cache_tokens": True,
                    "source_request_sha256": receipt["request_sha256"],
                    "label_status": "No labels: extraction precedes independent annotation.",
                }
            )
    if not rows:
        raise ValueError("no participant turns")
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "sha256": sha256(args.out.read_text()),
                "participant": args.participant,
            }
        )
    )


if __name__ == "__main__":
    main()
