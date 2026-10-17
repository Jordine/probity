import unittest
import copy
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from probity_benchmark.judge3_ablate import make_job, rename_speakers, selected_turns
from probity_benchmark.judge3_protocol import (
    base_job,
    bundle_jobs,
    cross_parent_permutation,
    partition,
    validate_bundle,
)
from probity_benchmark.preflight import sha256
from probity_benchmark.judge3_final import (
    analyse as analyse_final,
    audit_completion,
    check_families,
    check_signal,
    file_hash,
)
from probity_benchmark.judge3_compare import interaction_rows, shared_jobs
from probity_benchmark.judge3_context import compare_context


class DummyProblem:
    def view(self, with_story, swapped, structured=True):
        return [
            {"role": "system", "content": "original"},
            {"role": "user", "content": "Question?\nA: alpha\nB: beta"},
        ], "A"


class JudgeControls(unittest.TestCase):
    def episode(self):
        return {
            "complete": True,
            "episode_id": "episode",
            "parent_problem_id": "parent",
            "split": "dev",
            "model": "qwen",
            "world": "original",
            "gold": "A",
            "swapped": False,
            "turns": [
                {
                    "speaker": speaker,
                    "round": round_number,
                    "answer": answer,
                    "assigned_answer": "B" if answer == "A" else "A",
                    "argument": f"{speaker} says every word matters. Red's and Blue’s arguments are complete.",
                    "advocates_gold": answer == "A",
                }
                for round_number in [1, 2, 3]
                for speaker, answer in [("Red", "A"), ("Blue", "B")]
            ],
        }

    def test_complete_unit_preservation_and_public_answers(self):
        episode = self.episode()
        job = make_job(DummyProblem(), episode, "final_round")
        self.assertEqual(selected_turns(episode, "final_round"), [4, 5])
        self.assertEqual(len(job["selected_turns"]), 2)
        for turn in episode["turns"][4:]:
            self.assertIn(turn["argument"], job["messages"][1]["content"])
            self.assertIn(
                turn["speaker"] + " (answer " + turn["answer"] + "):",
                job["messages"][1]["content"],
            )
        for record in job["selected_turns"]:
            self.assertEqual(
                record["source_argument_sha256"], record["presented_argument_sha256"]
            )
        episode["gold"] = "B"
        with self.assertRaises(ValueError):
            make_job(DummyProblem(), episode, "full")

    def test_renaming_is_simultaneous_and_preserves_possessive_references(self):
        text = "Red's claim disputes Blue’s claim. Redwood is unrelated."
        self.assertEqual(
            rename_speakers(text, {"Red": "Blue", "Blue": "Red"}),
            "Blue's claim disputes Red’s claim. Redwood is unrelated.",
        )
        job = make_job(DummyProblem(), self.episode(), "renamed_speakers")
        body = job["messages"][1]["content"]
        self.assertIn("Speaker 1 (answer A)", body)
        self.assertIn("Speaker 1's and Speaker 2’s", body)

    def test_cross_parent_shuffle_preserves_exact_multiset(self):
        items = [
            {"key": f"{group}-{i}", "group": group, "score": i}
            for group, count in [("a", 4), ("b", 3), ("c", 2)]
            for i in range(count)
        ]
        for seed in range(20):
            shuffled = cross_parent_permutation(items, seed)
            self.assertEqual(
                Counter(r["key"] for r in items),
                Counter(r["key"] for r in shuffled.values()),
            )
            self.assertTrue(
                all(row["group"] != shuffled[row["key"]]["group"] for row in items)
            )
        with self.assertRaises(ValueError):
            cross_parent_permutation(
                items
                + [
                    {"key": "a-extra-1", "group": "a"},
                    {"key": "a-extra-2", "group": "a"},
                ],
                1,
            )

    def bundle_fixture(self):
        episodes, bases, rows = {}, {}, []
        for i in range(4):
            episode = self.episode()
            episode.update(
                episode_id=f"episode-{i}",
                parent_problem_id=f"parent-{i}",
                knowledge_correct=True,
            )
            episodes[episode["episode_id"]] = episode
            bases[episode["episode_id"]] = base_job(DummyProblem(), episode, "fixture")
            for index, turn in enumerate(episode["turns"]):
                spans = partition(turn["argument"], "sentence")
                rows.append(
                    {
                        "episode_id": episode["episode_id"],
                        "turn_index": index,
                        "argument_sha256": sha256(turn["argument"]),
                        "spans": [list(s) for s in spans],
                        "scores": [float(i + index + j) for j in range(len(spans))],
                    }
                )
        bundle = {
            "schema": "judge3_score_bundle_v1",
            "detector": "local",
            "target": "local_factual_contradiction",
            "granularity": "sentence",
            "orientation": "higher_more_unreliable",
            "head_sha256": "frozen-head",
            "reference_sha256": "fixed-dev-reference",
            "references": {
                "qwen": {
                    "split": "dev",
                    "parents": ["previous-parent"],
                    "scores": [-1, 0, 1, 2, 3],
                }
            },
            "rows": rows,
        }
        return bundle, episodes, bases

    def test_local_rendering_rejects_annotation_selected_segments(self):
        bundle, episodes, bases = self.bundle_fixture()
        jobs, _ = bundle_jobs(bundle, episodes, bases)
        self.assertEqual(len(jobs), 12)
        job = next(j for j in jobs if j["kind"] == "local:real")
        self.assertNotIn("reviewed_claims", job["messages"][-1]["content"])
        altered = copy.deepcopy(bundle)
        altered["rows"][0]["spans"].pop()
        altered["rows"][0]["scores"].pop()
        with self.assertRaises(ValueError):
            validate_bundle(altered, episodes)
        altered = copy.deepcopy(bundle)
        altered["rows"][0]["argument_sha256"] = "different-transcript"
        with self.assertRaises(ValueError):
            validate_bundle(altered, episodes)
        episodes["episode-0"]["split"] = "test"
        bundle["references"]["qwen"]["parents"].append("parent-0")
        with self.assertRaises(ValueError):
            validate_bundle(bundle, episodes)

    def test_detector_scope_allows_model_specific_weights_but_not_missing_episodes(
        self,
    ):
        bundle, episodes, _ = self.bundle_fixture()
        for i, episode in enumerate(episodes.values()):
            episode["model"] = "model35" if i < 2 else "model38"
        bundle["references"] = {"model35": bundle["references"]["qwen"]}
        bundle["rows"] = [
            row
            for row in bundle["rows"]
            if episodes[row["episode_id"]]["model"] == "model35"
        ]
        index = {"by_parent": {f"parent-{i}": f"family-{i}" for i in range(4)}}
        index["by_parent"]["previous-parent"] = "reference-family"
        families = {
            eid: index["by_parent"][e["parent_problem_id"]]
            for eid, e in episodes.items()
        }
        spec = {
            "participant_models": ["model35"],
            "head_sha256": bundle["head_sha256"],
            "reference_sha256": bundle["reference_sha256"],
        }
        self.assertEqual(
            check_signal(spec, bundle, episodes, index, families)["episodes"], 2
        )
        missing = copy.deepcopy(bundle)
        missing["rows"] = [
            row for row in missing["rows"] if row["episode_id"] != "episode-0"
        ]
        with self.assertRaisesRegex(ValueError, "every episode"):
            check_signal(spec, missing, episodes, index, families)
        index["by_parent"]["previous-parent"] = "family-0"
        with self.assertRaisesRegex(ValueError, "family leaked"):
            check_signal(spec, bundle, episodes, index, families)

    def test_final_independence_gate_rejects_multiple_questions_from_one_family(self):
        episodes = {}
        for parent in range(2):
            for model in ["model35", "model38"]:
                eid = f"{parent}-{model}"
                episodes[eid] = {
                    "parent_problem_id": f"parent-{parent}",
                    "model": model,
                }
        index = {"by_parent": {"parent-0": "family-0", "parent-1": "family-1"}}
        self.assertEqual(
            len(
                set(
                    check_families(
                        episodes, index, 4, 2, ["model35", "model38"]
                    ).values()
                )
            ),
            2,
        )
        index["by_parent"]["parent-1"] = "family-0"
        with self.assertRaisesRegex(ValueError, "one question"):
            check_families(episodes, index, 4, 1, ["model35", "model38"])

    def test_frozen_reference_arrays_cannot_change_behind_unchanged_hash_labels(self):
        bundle, episodes, _ = self.bundle_fixture()
        index = {"by_parent": {f"parent-{i}": f"family-{i}" for i in range(4)}}
        index["by_parent"]["previous-parent"] = "reference-family"
        families = {
            eid: index["by_parent"][e["parent_problem_id"]]
            for eid, e in episodes.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "references.json"
            artifact = {k: v for k, v in bundle.items() if k != "rows"}
            artifact["schema"] = "judge3_reference_bundle_v1"
            path.write_text(json.dumps(artifact))
            spec = {
                "participant_models": ["qwen"],
                "head_sha256": bundle["head_sha256"],
                "reference_sha256": bundle["reference_sha256"],
                "reference_artifact": str(path),
                "reference_artifact_sha256": file_hash(path),
            }
            self.assertEqual(
                check_signal(spec, bundle, episodes, index, families)["episodes"], 4
            )
            changed = copy.deepcopy(bundle)
            changed["references"]["qwen"]["scores"][0] = 1000
            with self.assertRaisesRegex(ValueError, "frozen reference references"):
                check_signal(spec, changed, episodes, index, families)
            artifact["references"]["qwen"]["scores"][0] = 1000
            path.write_text(json.dumps(artifact))
            with self.assertRaisesRegex(ValueError, "reference artifact changed"):
                check_signal(spec, bundle, episodes, index, families)

    def test_final_analysis_binds_api_receipts_to_complete_frozen_messages(self):
        _, _, bases = self.bundle_fixture()
        jobs = list(bases.values())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            package, results = directory / "package", directory / "results"
            (package / "compiled").mkdir(parents=True)
            results.mkdir()
            job_path = package / "compiled/jobs.jsonl"
            job_path.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            compiled = package / "compiled/manifest.json"
            compiled.write_text("{}\n")
            manifest = {
                "jobs_path": "compiled/jobs.jsonl",
                "jobs_sha256": file_hash(job_path),
                "compiled_manifest_sha256": file_hash(compiled),
                "families": 4,
                "limits": [],
            }
            (package / "package.json").write_text(json.dumps(manifest))
            (results / "manifest.json").write_text(
                json.dumps(
                    {"jobs_sha256": manifest["jobs_sha256"], "judge_model": "fixture"}
                )
            )
            rows, receipts = [], []
            for i, job in enumerate(jobs):
                answer = "A" if i % 2 else "B"
                request = {"messages": job["messages"], "max_tokens": 8192}
                request_hash = sha256(json.dumps(request, sort_keys=True))
                rows.append(
                    {
                        **{k: v for k, v in job.items() if k != "messages"},
                        "request_sha256": request_hash,
                        "answer": answer,
                        "correct": answer == job["gold"],
                        "valid": True,
                        "finish_reason": "stop",
                    }
                )
                receipts.append(
                    {
                        "request_sha256": request_hash,
                        "tag": {"job": job["job_id"], "kind": job["kind"]},
                        "model": "fixture",
                        "valid": True,
                        "finish_reason": "stop",
                        "parsed": {"answer": answer},
                        "request": request,
                        "usage": {"completion_tokens": 8},
                        "response_body": json.dumps(
                            {"choices": [{"native_finish_reason": "end_turn"}]}
                        ),
                    }
                )
            (results / "evaluations.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            request_path = results / "requests.jsonl"
            request_path.write_text("".join(json.dumps(r) + "\n" for r in receipts))
            args = SimpleNamespace(
                package=package,
                frozen_package_sha256=file_hash(package / "package.json"),
                results=results,
                out=directory / "analysis",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                analyse_final(args)
            self.assertEqual(
                json.loads((args.out / "receipt_audit.json").read_text())["completed"],
                4,
            )
            repeatability = json.loads(
                (args.out / "prompt_repeatability.json").read_text()
            )
            self.assertEqual(repeatability["repeated_prompt_groups"], 1)
            self.assertEqual(repeatability["groups_with_answer_disagreement"], 1)
            self.assertEqual(repeatability["pairwise_agreement"], 1 / 3)
            self.assertEqual(
                len(repeatability["groups"][0]["complete_request_hashes"]), 1
            )
            receipts[0]["request"]["messages"] = [
                {"role": "user", "content": "Different condition"}
            ]
            request_path.write_text("".join(json.dumps(r) + "\n" for r in receipts))
            args.out = directory / "rejected"
            with self.assertRaisesRegex(ValueError, "receipt differs"):
                analyse_final(args)

    def test_normalized_stop_does_not_hide_native_or_reasoning_truncation(self):
        receipt = {
            "valid": True,
            "finish_reason": "stop",
            "request": {"max_tokens": 8192},
            "usage": {
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 200},
            },
            "response_body": json.dumps(
                {"choices": [{"native_finish_reason": "end_turn"}]}
            ),
        }
        self.assertEqual(audit_completion(receipt)["reasoning_tokens"], 200)
        receipt["response_body"] = json.dumps(
            {"choices": [{"native_finish_reason": "max_tokens"}]}
        )
        with self.assertRaisesRegex(ValueError, "truncated judge"):
            audit_completion(receipt)
        receipt["response_body"] = json.dumps(
            {"choices": [{"message": {"reasoning_details": [{"forced": True}]}}]}
        )
        with self.assertRaisesRegex(ValueError, "truncated judge"):
            audit_completion(receipt)
        receipt["response_body"] = json.dumps({"choices": [{}]})
        receipt["usage"]["completion_tokens_details"]["reasoning_tokens"] = 8192
        with self.assertRaisesRegex(ValueError, "reached its allowance"):
            audit_completion(receipt)

    def test_cross_judge_interaction_uses_matched_neutral_and_preserves_pairing(self):
        job = {
            "condition": "real",
            "baseline_condition": "probe:neutral",
            "detector": "probe",
            "job_id": "real",
            "baseline_job_id": "neutral",
            "unformatted_baseline_job_id": "raw",
            "environment_protocol": "fixed",
            "split": "test",
            "participant_model": "participant",
            "episode_id": "episode",
        }
        first = {
            "real": {"correct": True},
            "neutral": {"correct": False},
            "raw": {"correct": True},
        }
        second = {
            "real": {"correct": True},
            "neutral": {"correct": True},
            "raw": {"correct": False},
        }
        rows = interaction_rows([job], first, second, {"episode": "same-source-family"})
        self.assertEqual(
            rows[("fixed", "test", "all", "probe")],
            [{"group": "same-source-family", "difference": -1}],
        )
        job["baseline_condition"] = "none"
        with self.assertRaisesRegex(ValueError, "matched neutral"):
            interaction_rows([job], first, second, {"episode": "same-source-family"})

    def test_reduced_judge_package_requires_exact_shared_inputs_and_provenance(self):
        package = {
            "episode_families": {"episode": "family"},
            "signals": [{"detector": "local", "head_sha256": "frozen"}],
        }
        jobs = [
            {
                "job_id": condition,
                "kind": condition,
                "episode_id": "episode",
                "messages": [{"role": "user", "content": "Complete transcript"}],
            }
            for condition in ["none", "oracle", "extra"]
        ]
        jobs += [
            {
                "job_id": "local:" + condition,
                "kind": "local:" + condition,
                "condition": condition,
                "episode_id": "episode",
                "detector": "local",
                "baseline_job_id": "local:neutral",
                "head_sha256": "frozen",
                "messages": [{"role": "user", "content": "Complete transcript"}],
            }
            for condition in ["neutral", "real", "shuffled"]
        ]
        subset = copy.deepcopy([j for j in jobs if j["kind"] != "extra"])
        self.assertEqual(len(shared_jobs(package, jobs, package, subset)), 5)
        subset[0]["messages"][0]["content"] = "Changed transcript"
        with self.assertRaisesRegex(ValueError, "messages or score provenance"):
            shared_jobs(package, jobs, package, subset)
        subset = copy.deepcopy([j for j in jobs if j["kind"] != "extra"])
        different = copy.deepcopy(package)
        different["signals"][0]["head_sha256"] = "new-head"
        with self.assertRaisesRegex(ValueError, "detector provenance"):
            shared_jobs(package, jobs, different, subset)
        different = copy.deepcopy(package)
        different["episode_families"]["episode"] = "other-family"
        with self.assertRaisesRegex(ValueError, "episode-family sets"):
            shared_jobs(package, jobs, different, subset)

    def test_private_context_check_rejects_lost_plan_despite_complete_source(self):
        original = [
            {"role": "system", "content": "Advocate assignment"},
            {"role": "user", "content": "Complete authentic story"},
            {"role": "assistant", "content": "Complete private plan"},
            {"role": "user", "content": "Complete public history"},
        ]
        diagnostic = [
            {"role": "system", "content": "Truth diagnostic"},
            *original[1:],
            {"role": "user", "content": "Answer truthfully"},
        ]
        compare_context(original, diagnostic, "Complete authentic story")
        diagnostic.pop(2)
        with self.assertRaisesRegex(ValueError, "complete non-system context"):
            compare_context(original, diagnostic, "Complete authentic story")


if __name__ == "__main__":
    unittest.main()
