import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probity_benchmark.environment3_packets import load_external_preparations, run


class FakeChat:
    calls = []
    knowledge_answer = "A"

    def __init__(self, out, models, budget, **kwargs):
        out.mkdir()
        self.records = []
        self.spent = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def complete(self, model, messages, schema, tag):
        self.calls.append((tag, messages))
        if tag["stage"] == "knowledge":
            parsed = {"answer": self.knowledge_answer}
        elif tag["stage"] == "preparation":
            parsed = {
                "evidence_items": [
                    {"claim": f"Claim {i}.", "quotation": f"Quote {i}."}
                    for i in range(3)
                ],
                "anticipated_challenge": "A conflicting attribution.",
                "response_to_challenge": "A specific answer.",
            }
        elif tag["stage"] == "debate":
            parsed = {
                "answer": "A" if tag["speaker"] == "Red" else "B",
                "argument": f"Complete public argument from {tag['speaker']}.",
            }
        else:
            parsed = {"answer": "A", "explanation": "A full judgment."}
        receipt = {
            "parsed": parsed,
            "valid": True,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 50, "prompt_tokens": 200},
            "request_sha256": str(len(self.calls)),
        }
        self.records.append(receipt)
        return receipt


class EnvironmentProtocolTests(unittest.IsolatedAsyncioTestCase):
    def make_args(self, directory, **overrides):
        dataset = directory / "source.jsonl"
        row = {
            "synthetic_story": "BEGIN_AUTHENTIC_SENTINEL\nMira visited Orion.\nEND_AUTHENTIC_SENTINEL",
            "question": "Where did Mira visit?",
            "option_a": "Orion",
            "option_b": "Vega",
            "correct_answer": "A",
            "parent_problem_id": "parent-one",
            "parent_source_line": 7,
            "schedule_parent_index": 0,
            "split": "test",
        }
        dataset.write_text(json.dumps(row) + "\n")
        fields = dict(
            dataset=dataset,
            out=directory / "result",
            models=["qwen/example"],
            judge="judge/example",
            budget=1,
            reuse_receipts=[],
            rounds=2,
            protocol="prepared_simultaneous",
            unpaired=True,
            expected_split="test",
        )
        fields.update(overrides)
        return argparse.Namespace(**fields), row

    def setUp(self):
        FakeChat.calls = []
        FakeChat.knowledge_answer = "A"

    async def test_independent_story_preserved_and_rounds_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, row = self.make_args(Path(tmp))
            with patch("probity_benchmark.environment3_packets.BudgetedChat", FakeChat):
                await run(args)
            episode = json.loads(next(args.out.glob("episode_*.json")).read_text())
            self.assertEqual(episode["split"], "test")
            self.assertIsNone(episode["alternate_problem_id"])
            self.assertEqual(
                [t["visible_prior_turns"] for t in episode["turns"]], [0, 0, 2, 2]
            )
            for tag, messages in FakeChat.calls:
                body = "\n".join(m["content"] for m in messages)
                if tag["stage"] in ("knowledge", "preparation", "debate"):
                    self.assertIn(row["synthetic_story"], body)
                if tag["stage"] == "judge":
                    self.assertNotIn("AUTHENTIC_SENTINEL", body)
                    self.assertNotIn("anticipated_challenge", body)
            self.assertEqual(len(episode["private_preparations"]), 2)

    async def test_failed_knowledge_prevents_all_dependent_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.make_args(Path(tmp))
            FakeChat.knowledge_answer = "B"
            with patch("probity_benchmark.environment3_packets.BudgetedChat", FakeChat):
                with self.assertRaisesRegex(RuntimeError, "incomplete run"):
                    await run(args)
            self.assertEqual([t[0]["stage"] for t in FakeChat.calls], ["knowledge"])
            self.assertFalse(list(args.out.glob("episode_*.json")))

    async def test_packet_cannot_silently_use_unpaired_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.make_args(Path(tmp), protocol="packet_sequential")
            with self.assertRaisesRegex(ValueError, "require paired"):
                await run(args)

    def test_empty_reasoned_batch_cannot_fall_back_to_unreasoned(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "preparations.jsonl"
            path.write_text("")
            path.with_name("summary.json").write_text(
                json.dumps({"errors": [], "completed": 0, "expected": 0})
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_external_preparations(path)

    def test_external_usage_must_match_the_original_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "preparations.jsonl"
            record = {
                "problem_id": "p",
                "model": "m",
                "speaker": "Red",
                "request_sha256": "r",
                "parsed": {"evidence_items": []},
                "finish_reason": "stop",
                "completion_tokens": 20,
                "max_tokens": 16384,
            }
            receipt = {
                "request_sha256": "r",
                "parsed": record["parsed"],
                "valid": True,
                "finish_reason": "stop",
                "usage": {"completion_tokens": 20},
                "request": {"reasoning": {"enabled": True}, "max_tokens": 16384},
            }
            path.write_text(json.dumps(record) + "\n")
            path.with_name("summary.json").write_text(
                json.dumps({"errors": [], "completed": 1, "expected": 1})
            )
            path.with_name("requests.jsonl").write_text(json.dumps(receipt) + "\n")
            self.assertEqual(set(load_external_preparations(path)), {("p", "m", "Red")})
            record["completion_tokens"] = 10
            path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "receipt failed"):
                load_external_preparations(path)


if __name__ == "__main__":
    unittest.main()
