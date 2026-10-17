import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from probity_benchmark.controls3 import analyse
from probity_benchmark.preflight import sha256


class ControlIntegrity(unittest.TestCase):
    def fixture(self, root):
        package = root / "package"
        package.mkdir()
        jobs = [
            {
                "job_id": "job",
                "kind": "no_story",
                "group": "family",
                "swapped": False,
                "short_options": True,
                "gold": "A",
                "messages": [{"role": "user", "content": "Choose."}],
            }
        ]
        text = json.dumps(jobs[0]) + "\n"
        (package / "jobs.jsonl").write_text(text)
        row = {
            **{k: v for k, v in jobs[0].items() if k != "messages"},
            "valid": True,
            "finish_reason": "stop",
            "correct": True,
            "parsed": {"answer": "A"},
            "completion_tokens": 8,
        }
        evaluations = root / "evaluations.jsonl"
        evaluations.write_text(json.dumps(row) + "\n")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "jobs_sha256": sha256(text),
                    "contract": {
                        "answer_only": True,
                        "model": "test/native",
                        "max_new_tokens": 8192,
                    },
                }
            )
        )
        return (
            SimpleNamespace(
                package=package, evaluations=evaluations, out=root / "summary.json"
            ),
            row,
        )

    def test_empty_results_are_not_a_successful_control_run(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            args.evaluations.write_text("")
            with self.assertRaisesRegex(ValueError, "empty control"):
                analyse(args)

    def test_native_identity_comes_from_recorded_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(Path(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                analyse(args)
            result = json.loads(args.out.read_text())
            self.assertEqual({c["model"] for c in result["cells"]}, {"test/native"})

    def test_group_tampering_and_exposed_forcing_are_rejected(self):
        for replacement, error in (
            ({"group": "other"}, "metadata"),
            ({"forced": True}, "incomplete native"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                args, row = self.fixture(Path(directory))
                args.evaluations.write_text(json.dumps({**row, **replacement}) + "\n")
                with self.assertRaisesRegex(ValueError, error):
                    analyse(args)

    def test_entire_missing_api_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, row = self.fixture(root)
            row.update(model="test/first", request_sha256="request")
            args.evaluations.write_text(json.dumps(row) + "\n")
            receipt = {
                "request_sha256": "request",
                "tag": {"job_id": "job"},
                "valid": True,
                "finish_reason": "stop",
                "model": "test/first",
                "parsed": row["parsed"],
                "request": {
                    "messages": [{"role": "user", "content": "Choose."}],
                    "max_tokens": 8192,
                },
                "usage": {"completion_tokens": 8},
                "response_body": json.dumps({"choices": [{"finish_reason": "stop"}]}),
            }
            (root / "requests.jsonl").write_text(json.dumps(receipt) + "\n")
            (root / "api_manifest.json").write_text(
                json.dumps({"models": {"test/first": {}, "test/missing": {}}})
            )
            with self.assertRaisesRegex(ValueError, "missing or unexpected"):
                analyse(args)


if __name__ == "__main__":
    unittest.main()
