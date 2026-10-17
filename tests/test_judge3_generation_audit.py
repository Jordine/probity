import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from probity_benchmark.judge3_export_tables import main as export_tables
from probity_benchmark.judge3_export_tables import project_analysis
from probity_benchmark.judge3_final import analyse, file_hash
from probity_benchmark.judge3_generation_audit import (
    audit,
    generation_repeatability,
    prepare_native_view,
    read_api_receipts,
)


class JudgeGenerationAudit(unittest.TestCase):
    def native_fixture(self):
        messages = [{"role": "user", "content": "Identical complete prompt"}]
        jobs = {
            name: {
                "job_id": name,
                "kind": "none",
                "environment_protocol": "fixture",
                "split": "test",
                "participant_model": "fixture-participant",
                "group": "one-family",
                "gold": "A",
                "messages": messages,
            }
            for name in ("first", "cached")
        }
        rows = [
            {
                **{key: value for key, value in job.items() if key != "messages"},
                "parsed": {"answer": "A"},
                "raw": '{"answer":"A"}',
                "correct": True,
                "valid": True,
                "finish_reason": "stop",
                "completion_tokens": 8,
                "request_sha256": "same-native-cache-key",
                "reused": index == 1,
            }
            for index, job in enumerate(jobs.values())
        ]
        manifest = {
            "contract": {
                "model": "fixture-native",
                "answer_only": True,
                "max_new_tokens": 8192,
            }
        }
        return jobs, rows, manifest

    def test_cached_native_is_unmeasured_and_analysis_view_preserves_originals(self):
        jobs, rows, manifest = self.native_fixture()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            package, original, view = (
                root / name for name in ("package", "original", "view")
            )
            (package / "compiled").mkdir(parents=True)
            original.mkdir()
            jobs_path = package / "compiled/jobs.jsonl"
            jobs_path.write_text(
                "".join(json.dumps(job) + "\n" for job in jobs.values())
            )
            (package / "compiled/manifest.json").write_text("{}\n")
            manifest["jobs_sha256"] = file_hash(jobs_path)
            (original / "manifest.json").write_text(json.dumps(manifest))
            (original / "evaluations.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            package_data = {
                "jobs_path": "compiled/jobs.jsonl",
                "jobs_sha256": file_hash(jobs_path),
                "compiled_manifest_sha256": file_hash(
                    package / "compiled/manifest.json"
                ),
                "jobs": 2,
                "families": 1,
                "limits": ["Synthetic fixture"],
            }
            (package / "package.json").write_text(json.dumps(package_data))
            package_hash = file_hash(package / "package.json")
            original_bytes = {
                path.name: path.read_bytes() for path in original.iterdir()
            }
            args = SimpleNamespace(
                package=package,
                frozen_package_sha256=package_hash,
                results=original,
                out=root / "old_failed_analysis",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(KeyError, "answer"):
                    analyse(args)
                prepare_native_view(SimpleNamespace(**{**vars(args), "out": view}))
                analyse(
                    SimpleNamespace(
                        **{**vars(args), "results": view, "out": root / "analysis"}
                    )
                )
                audit(
                    SimpleNamespace(
                        **{
                            **vars(args),
                            "analysis": root / "analysis",
                            "out": root / "generation.json",
                        }
                    )
                )
            derived = [
                json.loads(line)
                for line in (view / "evaluations.jsonl").read_text().splitlines()
            ]
            for old, new in zip(rows, derived):
                self.assertEqual(new["answer"], old["parsed"]["answer"])
                self.assertEqual(
                    {key: value for key, value in new.items() if key != "answer"}, old
                )
            self.assertEqual(
                {path.name: path.read_bytes() for path in original.iterdir()},
                original_bytes,
            )
            self.assertEqual(
                (view / "manifest.json").read_bytes(), original_bytes["manifest.json"]
            )
            generation = json.loads((root / "generation.json").read_text())
            summary = generation["summary"]
            self.assertEqual(summary["logical_jobs_total"], 2)
            self.assertEqual(summary["fresh_generations_in_run"], 1)
            self.assertEqual(summary["reused_logical_jobs"], 1)
            self.assertEqual(summary["generation_all_answer_pairs"], 0)
            self.assertIsNone(summary["generation_pairwise_agreement"])
            self.assertEqual(
                summary["measurement_status"],
                "cached_repeats_only_sampling_repeatability_unmeasured",
            )
            report, receipt_audit, repeats = [
                json.loads((root / "analysis" / name).read_text())
                for name in (
                    "analysis.json",
                    "receipt_audit.json",
                    "prompt_repeatability.json",
                )
            ]
            tables = project_analysis(report, receipt_audit, repeats, generation)
            noise = tables["exact_prompt_summary"][0]
            self.assertEqual(noise["logical_pairwise_agreement"], 1.0)
            self.assertIsNone(noise["generation_pairwise_agreement"])
            self.assertIsNone(
                tables["exact_prompt_groups"][0]["generation_answer_disagreement"]
            )
            spec = root / "export_spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema": "judge3_table_export_spec_v1",
                        "runs": [
                            {
                                "package": str(package),
                                "package_sha256": package_hash,
                                "results": str(view),
                                "original_results": str(original),
                                "analysis": str(root / "analysis"),
                                "generation_audit": str(root / "generation.json"),
                            }
                        ],
                    }
                )
            )
            with patch(
                "sys.argv",
                ["export", "--spec", str(spec), "--out", str(root / "tables")],
            ), contextlib.redirect_stdout(io.StringIO()):
                export_tables()
            with (root / "tables/exact_prompt_summary.csv").open() as stream:
                exported = next(csv.DictReader(stream))
            self.assertEqual(exported["logical_pairwise_agreement"], "1.0")
            self.assertEqual(exported["generation_pairwise_agreement"], "")
            self.assertEqual(exported["fresh_generations_in_run"], "1")
            self.assertEqual(
                {path.name: path.read_bytes() for path in original.iterdir()},
                original_bytes,
            )

    def test_recovered_api_receipts_remain_distinct_generations_not_new_calls(self):
        jobs, native_rows, _ = self.native_fixture()
        request = {"messages": jobs["first"]["messages"], "max_tokens": 8192}
        request_hash = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        rows, receipts = [], []
        for index, (job_id, job) in enumerate(jobs.items()):
            value = "A" if index == 0 else "B"
            rows.append(
                {
                    **native_rows[index],
                    "answer": value,
                    "parsed": {"answer": value},
                    "correct": value == "A",
                    "request_sha256": request_hash,
                }
            )
            receipts.append(
                {
                    "request": request,
                    "request_sha256": request_hash,
                    "tag": {"job": job_id, "kind": job["kind"]},
                    "model": "fixture-api",
                    "valid": True,
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 8},
                    "parsed": {"answer": value},
                    "response_body": json.dumps(
                        {
                            "id": "distinct-response-" + str(index),
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {
                                        "content": json.dumps({"answer": value})
                                    },
                                }
                            ],
                        }
                    ),
                }
            )
        with TemporaryDirectory() as tmp:
            earlier, current = Path(tmp) / "earlier.jsonl", Path(tmp) / "current.jsonl"
            earlier.write_text(json.dumps(receipts[1]) + "\n")
            receipts[1] = {**receipts[1], "reused_from": str(earlier)}
            current.write_text("".join(json.dumps(row) + "\n" for row in receipts))
            inputs = {}
            indexed, origins = read_api_receipts(
                current, jobs, rows, "fixture-api", inputs
            )
            report = generation_repeatability(
                jobs, rows, {"judge_model": "fixture-api"}, indexed, origins
            )
        summary = report["summary"]
        self.assertEqual(summary["fresh_generations_in_run"], 1)
        self.assertEqual(summary["reused_logical_jobs"], 1)
        self.assertEqual(summary["generation_receipts_represented"], 2)
        self.assertEqual(summary["generation_all_answer_pairs"], 1)
        self.assertEqual(summary["generation_pairwise_agreement"], 0.0)
        self.assertEqual(len(inputs), 2)

    def test_native_reuse_provenance_is_required_and_unknown_origins_stay_unknown(self):
        jobs, rows, manifest = self.native_fixture()
        missing = copy.deepcopy(rows)
        missing[0].pop("reused")
        with self.assertRaisesRegex(ValueError, "identify cache reuse"):
            generation_repeatability(jobs, missing, manifest)
        for row in rows:
            row["reused"] = True
        report = generation_repeatability(jobs, rows, manifest)
        self.assertEqual(report["summary"]["unresolved_native_cached_origins"], 1)
        self.assertEqual(report["summary"]["fresh_generations_in_run"], 0)
        self.assertIsNone(report["summary"]["generation_pairwise_agreement"])


if __name__ == "__main__":
    unittest.main()
