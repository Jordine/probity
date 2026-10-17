import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from probity_benchmark.grid4_evaluate import (
    api_request_body, audit_results, construct_jobs, disclosure_audit,
    exact_request_groups, grid_statistics, keyed, pair_episodes, write_json, write_rows,
)
from probity_benchmark.judge3_protocol import base_job, bundle_jobs
from probity_benchmark.preflight import sha256


class Problem:
    def view(self, with_story, swapped, structured=True):
        return [{"role": "system", "content": "old"},
                {"role": "user", "content": "Question? A: alpha B: beta"}], "A"


def fixtures():
    prior, feedback, families, problems = {}, {}, {}, {}
    for family in range(2):
        for model in ("qwen35", "qwen38"):
            eid = f"old-{family}-{model}"
            episode = {"episode_id": eid, "parent_problem_id": f"parent-{family}",
                       "problem_id": f"problem-{family}", "model": model,
                       "split": "test", "world": "original", "gold": "A", "swapped": False,
                       "complete": True, "knowledge_correct": True, "turns": [
                           {"speaker": speaker, "round": 1, "answer": answer,
                            "assigned_answer": answer, "argument": f"{speaker} cites the old story."}
                           for speaker, answer in (("Red", "A"), ("Blue", "B"))]}
            prior[eid] = episode
            families[eid] = f"family-{family}"
            problems[episode["problem_id"]] = Problem()
            new = copy.deepcopy(episode)
            new.update(episode_id=eid.replace("old", "new"), matched_no_feedback_episode_id=eid,
                       protocol="feedback_v1")
            for turn in new["turns"]:
                turn["argument"] = turn["argument"].replace("old story", "new evidence")
            feedback[new["episode_id"]] = new

    def bundle(episodes):
        return {"schema": "judge3_score_bundle_v1", "detector": "adapted_primary",
                "target": "wrong_answer_advocacy", "granularity": "response",
                "orientation": "higher_more_unreliable", "head_sha256": "head",
                "reference_sha256": "reference", "references": {
                    model: {"split": "dev", "parents": ["training-parent"], "scores": [0., 1.]}
                    for model in ("qwen35", "qwen38")}, "rows": [
                        {"episode_id": eid, "turn_index": index,
                         "argument_sha256": sha256(turn["argument"]),
                         "spans": [[0, len(turn["argument"])]], "scores": [float(index)]}
                        for eid, ep in episodes.items() for index, turn in enumerate(ep["turns"])]}
    prior_bundle, feedback_bundle = bundle(prior), bundle(feedback)
    bases = {eid: base_job(problems[ep["problem_id"]], ep, "old_protocol") for eid, ep in prior.items()}
    score_jobs, _ = bundle_jobs(prior_bundle, prior, bases)
    old_jobs = list(bases.values()) + score_jobs
    return old_jobs, prior, feedback, families, problems, feedback_bundle, prior_bundle


class GridEvaluationTests(unittest.TestCase):
    def test_frozen_grid_uses_raw_A_real_C_and_same_B_D_transcript(self):
        data = fixtures()
        old, jobs, pairs = construct_jobs(*data)
        old_by_id, new_by_id = keyed(old), keyed(jobs)
        original_by_id = keyed(data[0])
        self.assertEqual(len(pairs), 4)
        self.assertEqual(len(old), 8)
        self.assertEqual(len(jobs), 8)
        for pair in pairs:
            a, c = (old_by_id[pair["jobs"][x]] for x in ("A", "C"))
            b, d = (new_by_id[pair["jobs"][x]] for x in ("B", "D"))
            self.assertEqual(a, original_by_id[a["job_id"]])
            self.assertEqual(c, original_by_id[c["job_id"]])
            self.assertEqual(a["kind"], "none")
            self.assertEqual(c["condition"], "real")
            self.assertEqual(b["episode_id"], d["episode_id"])
            self.assertEqual(b["turn_argument_sha256"], d["turn_argument_sha256"])
            self.assertEqual(b["messages"][0], d["messages"][0])
            self.assertTrue(d["messages"][1]["content"].startswith(b["messages"][1]["content"]))
            self.assertNotIn("signal_percentile", b["messages"][1]["content"])
            self.assertIn("signal_percentile", d["messages"][1]["content"])

    def test_missing_or_repeated_pair_fails_before_outcomes(self):
        _, prior, feedback, families, *_ = fixtures()
        missing = dict(feedback)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "inclusion"):
            pair_episodes(prior, missing, families)
        first, second = list(feedback)[:2]
        feedback[second]["matched_no_feedback_episode_id"] = feedback[first]["matched_no_feedback_episode_id"]
        with self.assertRaises(ValueError):
            pair_episodes(prior, feedback, families)

    def test_feedback_concession_retained_and_actual_answer_displayed(self):
        data = fixtures()
        episode = next(iter(data[2].values()))
        episode["turns"][0]["answer"] = "B"
        episode["turns"][0]["role_adherent"] = False
        _, jobs, pairs = construct_jobs(*data)
        relevant = [j for j in jobs if j["episode_id"] == episode["episode_id"]]
        self.assertEqual(len(relevant), 2)
        for job in relevant:
            self.assertIn("Turn 1. Red (answer B):", job["messages"][1]["content"])
            self.assertIn(episode["turns"][0]["argument"], job["messages"][1]["content"])
            self.assertEqual(len(job["feedback_role_switches"]), 1)
        d = next(j for j in relevant if j["grid_cell"] == "D")
        # Red has the low signal but was originally assigned A, despite conceding B.
        self.assertIn('"detector_only_recommendation": "A"', d["messages"][1]["content"])
        self.assertEqual(len(pairs), 4)

    def test_option_mapping_or_speaker_order_cannot_change(self):
        for mutate in (lambda e: e.update(swapped=True), lambda e: e["turns"].reverse()):
            _, prior, feedback, families, *_ = fixtures()
            mutate(next(iter(feedback.values())))
            with self.assertRaisesRegex(ValueError, "changed"):
                pair_episodes(prior, feedback, families)

    def test_score_head_reference_or_original_score_change_rejected(self):
        for field in ("head_sha256", "reference_sha256"):
            data = fixtures()
            data[-2][field] = "new"
            with self.assertRaisesRegex(ValueError, "frozen"):
                construct_jobs(*data)
        data = fixtures()
        data[-1]["rows"][0]["scores"] = [1.]
        with self.assertRaisesRegex(ValueError, "original C"):
            construct_jobs(*data)

    def test_missing_feedback_turn_score_rejected(self):
        data = fixtures()
        data[-2]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "selected turns"):
            construct_jobs(*data)

    def test_disclosure_requires_text_bound_review_and_never_changes_text(self):
        _, _, feedback, *_ = fixtures()
        turn = next(iter(feedback.values()))["turns"][0]
        turn["argument"] = "The detector gave my rival a 97 percentile score."
        original = turn["argument"]
        with self.assertRaisesRegex(ValueError, "disclosure"):
            disclosure_audit(feedback)
        leak = disclosure_audit(feedback, {sha256(original): {"discloses_scores": True, "reason": "actual leak"}})
        self.assertTrue(leak[0]["review"]["discloses_scores"])
        self.assertEqual(turn["argument"], original)
        turn["argument"] = "The story's smoke detector sounded."
        reviewed = {sha256(turn["argument"]): {"discloses_scores": False, "reason": "source object, no probe reference"}}
        audit = disclosure_audit(feedback, reviewed)
        self.assertEqual(len(audit), 1)
        self.assertEqual(turn["argument"], "The story's smoke detector sounded.")

    def test_exact_request_dedup_includes_runtime_and_complete_messages(self):
        jobs = [{"job_id": k, "messages": [{"role": "user", "content": text}], "runtime": runtime}
                for k, text, runtime in (("a", "complete", 8), ("b", "complete", 8),
                                         ("c", "complete", 9), ("d", "complete tail", 8))]
        groups = exact_request_groups(jobs, lambda j: {"messages": j["messages"], "max_tokens": j["runtime"]})
        self.assertEqual(sorted(len(g["jobs"]) for g in groups.values()), [1, 1, 2])
        self.assertEqual(next(g["jobs"][0]["job_id"] for g in groups.values() if len(g["jobs"]) == 2), "a")

    def test_api_body_has_full_runtime_and_answer_schema(self):
        api = SimpleNamespace(models={"judge": {"pricing": {"prompt": ".0001", "completion": ".0002"},
                                               "supported_parameters": ["temperature"]}},
                              max_tokens=8192, reasoning={"enabled": False}, reasoning_by_model={},
                              structured_output=True)
        body = api_request_body(api, "judge", [{"role": "user", "content": "text"}])
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"]["required"], ["answer"])

    def test_D_minus_A_and_interaction_use_paired_family_cluster(self):
        outcomes, pairs = {}, []
        values = [(False, True, True, True), (False, False, True, True),
                  (True, False, False, True), (True, True, True, False)]
        for i, cells in enumerate(values):
            pair = {"family": str(i // 2), "participant_model": "q" + str(i % 2),
                    "no_feedback_episode_id": str(i), "feedback_episode_id": "new" + str(i),
                    "jobs": {cell: str(i) + cell for cell in "ABCD"}}
            pairs.append(pair)
            outcomes.update({str(i) + cell: {"correct": value} for cell, value in zip("ABCD", cells)})
        accuracy, contrasts, _ = grid_statistics(pairs, outcomes, "judge")
        pooled = {r["contrast"]: r for r in contrasts if r["participant_model"] == "all"}
        self.assertEqual(pooled["D-A"]["difference"], .25)
        self.assertEqual(pooled["D-A"]["fixes"], 2)
        self.assertEqual(pooled["D-A"]["harms"], 1)
        self.assertEqual(pooled["D-A"]["families"], 2)
        self.assertEqual(pooled["D-A"]["episodes"], 4)
        self.assertEqual(pooled["D-A"]["family_bootstrap_95_ci"], [-.5, 1.])
        self.assertEqual(pooled["(D-B)-(C-A)"]["difference"], 0.)
        self.assertEqual(pooled["D-A"]["difference"], pooled["B-A"]["difference"] + pooled["D-B"]["difference"])
        self.assertEqual(len(accuracy), 12)

    def test_native_result_requires_frozen_metadata_normal_stop_and_bounds(self):
        job = {"job_id": "one", "messages": [{"role": "user", "content": "complete"}], "gold": "A"}
        row = {"job_id": "one", "gold": "A", "parsed": {"answer": "A"}, "raw": '{"answer":"A"}',
               "valid": True, "finish_reason": "stop", "correct": True, "input_tokens": 10,
               "completion_tokens": 3, "request_sha256": "key"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {"jobs_sha256": "jobs", "jobs_expected": 1, "context_limit": 18,
                        "contract": {"model": "native", "answer_only": True, "max_new_tokens": 8}}
            write_json(path / "manifest.json", manifest)
            write_rows(path / "evaluations.jsonl", [row])
            accepted, audit = audit_results(path, {"one": job}, "jobs")
            self.assertEqual(accepted["one"]["correct"], True)
            self.assertEqual(audit["runtime_identity"]["type"], "native")
            for change in ({"finish_reason": "length"}, {"completion_tokens": 8}, {"gold": "B"}, {"correct": False}):
                write_rows(path / "evaluations.jsonl", [{**row, **change}])
                with self.assertRaises(ValueError):
                    audit_results(path, {"one": job}, "jobs")

    def test_api_alias_uses_same_full_request_receipt(self):
        jobs = {key: {"job_id": key, "kind": "none", "gold": "A",
                      "messages": [{"role": "user", "content": "same complete prompt"}]} for key in ("a", "b")}
        request = {"model": "api", "messages": jobs["a"]["messages"], "max_tokens": 8192,
                   "reasoning": {"enabled": False}}
        request_hash = sha256(json.dumps(request, sort_keys=True))
        receipt = {"request": request, "request_sha256": request_hash, "model": "api", "tag": {"job": "a"},
                   "valid": True, "finish_reason": "stop", "parsed": {"answer": "A"},
                   "usage": {"completion_tokens": 3}, "response_body": json.dumps({
                       "model": "api", "choices": [{"native_finish_reason": "stop",
                           "message": {"content": '{"answer":"A"}'}}]})}
        rows = [{**{k: v for k, v in j.items() if k != "messages"}, "answer": "A", "correct": True,
                 "request_sha256": request_hash, "receipt_job_id": "a", "valid": True, "finish_reason": "stop"}
                for j in jobs.values()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "manifest.json", {"jobs_sha256": "jobs", "jobs_expected": 2,
                "judge_model": "api", "runtime": {"reasoning": {"enabled": False}, "max_tokens": 8192}})
            write_rows(path / "requests.jsonl", [receipt])
            write_rows(path / "evaluations.jsonl", rows)
            _, audit = audit_results(path, jobs, "jobs")
            self.assertEqual(audit["unique_grid_request_hashes"], 1)
            jobs["b"]["messages"][0]["content"] = "different prompt"
            with self.assertRaisesRegex(ValueError, "receipt"):
                audit_results(path, jobs, "jobs")


if __name__ == "__main__":
    unittest.main()
