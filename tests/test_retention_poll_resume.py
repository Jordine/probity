import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "retention_poll_resume", Path(__file__).resolve().parents[1]
    / "runs/iteration4/coordinator/retention_poll_resume.py")
poller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(poller)


class RetentionPollTests(unittest.TestCase):
    def test_allowlist_excludes_connection_credentials_requests_and_other_inputs(self):
        for folder in poller.ALLOWED:
            with self.subTest(folder=folder), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "source"
                destination = Path(temporary) / "destination"
                source.mkdir()
                destination.mkdir()
                dangerous = ["connection.json", "connection_resume_proxy_v1.json", ".env", "auth.json",
                             "rental_resume_v1.json", "requests/input.json", "raw_activations/tensor.pt"]
                expected = {
                    "coordinator": ["native_judge.log", "runtime_ready.json", "all_remote_stages_complete.json"],
                    "native_judge_v1": ["manifest.json", "evaluations.jsonl", "summary.json"],
                    "native_replay_v1": ["manifest.json", "evaluations.jsonl", "summary.json"],
                    "qwen35_replay_v1": ["results/a.json", "parity_receipts/b.json", "worker_qwen35.json"],
                }[folder]
                for name in dangerous + expected:
                    p = source / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text("example")
                subprocess.run(["rsync", "-az", *poller.filter_args(folder),
                                str(source) + "/", str(destination) + "/"], check=True)
                self.assertEqual(sorted(str(p.relative_to(destination)) for p in destination.rglob("*")
                                        if p.is_file()), sorted(expected))

    def test_failed_cleanup_does_not_end_retention(self):
        with tempfile.TemporaryDirectory() as temporary:
            meta = Path(temporary)
            receipt = {"instance_id": 123}
            path = meta / "rental_resume_manual_cleanup_v1.json"
            path.write_text(json.dumps({"returncode": 1}))
            self.assertFalse(poller.cleanup_seen(meta, receipt))
            path.write_text(json.dumps({"returncode": 0}))
            self.assertTrue(poller.cleanup_seen(meta, receipt))

    def test_transfer_uses_no_delete_and_only_expected_remote_folder(self):
        command = poller.transfer_command({"key": "/fake/key with spaces", "port": 123,
                                          "host": "example.invalid"}, "native_judge_v1", Path("/tmp/results"))
        self.assertNotIn("--delete", command)
        self.assertIn("root@example.invalid:/workspace/probity/runs/iteration4/native_judge_v1/", command)
        self.assertIn("'/fake/key with spaces'", command[command.index("-e") + 1])


if __name__ == "__main__":
    unittest.main()
