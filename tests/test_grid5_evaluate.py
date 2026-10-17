import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from probity_benchmark import grid4_evaluate as old
from probity_benchmark import grid5_evaluate as grid
from probity_benchmark.judge3_protocol import base_job, bundle_jobs
from probity_benchmark.preflight import Problem, sha256


def fixtures():
    models = grid.MODELS
    source, rows, problems, prior, feedback, families = [], [], {}, {}, {}, {}
    for index, family in enumerate((0, 1, 0, 2)):
        parent = f"parent-{index}"
        story, question = f"Source family {family} story.", f"Question {index}?"
        fields = [story, question, "alpha", "beta", "A"]
        pid = sha256(json.dumps(fields, ensure_ascii=False))
        problem = Problem(pid, story, question, ("alpha", "beta"), "A", index + 1)
        problems[pid] = problem
        cached = index < 2
        split = "test" if cached else "dev"
        source.append(dict(zip(("synthetic_story", "question", "option_a", "option_b", "correct_answer"), fields),
                           parent_problem_id=parent, parent_source_line=index + 1, split=split))
        row = {"parent_problem_id": parent, "problem_id": pid, "parent_source_line": index + 1,
               "family_id": f"family-{family}", "family_name": f"Family {family}",
               "prior24": cached, "split": split, "historical_exposure_labels": ["previous use"],
               "previous_training_family": False, "previous_development_family": family == 2,
               "previous_tuning_or_diagnostic_family": family == 2,
               "previously_evaluated_family": family in (0, 1),
               "family_untouched_before_iteration5": False, "source_sha256": sha256(story),
               "question_sha256": sha256(question), "displayed_gold": "A", "first_speaker": "Red",
               "cached": {}}
        for model_index, model in enumerate(models):
            eid = f"q{index}-m{model_index}"
            episode = {"episode_id": eid, "parent_problem_id": parent, "problem_id": pid,
                       "parent_source_line": index + 1, "model": model, "split": split,
                       "world": "original", "gold": "A", "swapped": False,
                       "complete": True, "knowledge_correct": True, "turns": [
                           {"speaker": ("Red", "Blue")[i % 2], "round": i // 2 + 1,
                            "answer": ("A", "B")[i % 2], "assigned_answer": ("A", "B")[i % 2],
                            "argument": f"Question {index}, model {model_index}, turn {i}: cites old evidence."}
                           for i in range(6)]}
            new = copy.deepcopy(episode)
            new.update(episode_id="feedback-" + eid, matched_no_feedback_episode_id=eid,
                       protocol="feedback_v1")
            for turn in new["turns"]:
                turn["argument"] = turn["argument"].replace("old evidence", "new evidence")
            prior[eid], feedback[new["episode_id"]] = episode, new
            families[eid] = row["family_id"]
            if cached:
                row["cached"][model] = {"no_feedback_episode_id": eid,
                                        "feedback_episode_id": new["episode_id"]}
        rows.append(row)

    def bundle(episodes):
        return {"schema": "judge3_score_bundle_v1", "detector": "adapted_primary",
                "target": "wrong_answer_advocacy", "granularity": "response",
                "orientation": "higher_more_unreliable", "head_sha256": "head",
                "reference_sha256": "reference", "references": {
                    model: {"split": "dev", "parents": ["parent-3"], "scores": [0., 1.]}
                    for model in models}, "rows": [
                        {"episode_id": eid, "turn_index": index,
                         "argument_sha256": sha256(turn["argument"]),
                         "spans": [[0, len(turn["argument"])]], "scores": [float(index % 2)]}
                        for eid, episode in episodes.items() for index, turn in enumerate(episode["turns"])]}

    original = {eid: e for eid, e in prior.items() if e["split"] == "test"}
    original_feedback = {eid: e for eid, e in feedback.items() if e["split"] == "test"}
    reference = bundle(original)
    bases = {eid: base_job(problems[e["problem_id"]], e, grid.PROTOCOL) for eid, e in original.items()}
    signals, _ = bundle_jobs(reference, original, bases)
    original_ac, original_bd, original_pairs = old.construct_jobs(
        list(bases.values()) + signals, original, original_feedback,
        {eid: families[eid] for eid in original}, problems, bundle(original_feedback), reference)
    inventory = {"schema": grid.INVENTORY_SCHEMA, "status": "frozen_after_source_review",
                 "rows": rows, "counts": {"questions": 4, "families": 3,
                 "reused_prior_questions": 2, "new_questions": 2,
                 "untouched_question_rows_before_iteration5": 0,
                 "untouched_families_before_iteration5": 0}}
    return SimpleNamespace(models=models, source=source, inventory=inventory,
                           rows={r["parent_problem_id"]: r for r in rows}, problems=problems,
                           prior=prior, feedback=feedback, families=families,
                           bundle=bundle(prior), feedback_bundle=bundle(feedback), reference=reference,
                           original_ac=original_ac, original_bd=original_bd, original_pairs=original_pairs)


def write_fixture(root, fixture):
    baseline_dir, feedback_dir, original_dir = [root / name for name in ("episodes", "feedback", "original_grid")]
    for path in (baseline_dir, feedback_dir, original_dir):
        path.mkdir()
    inventory = copy.deepcopy(fixture.inventory)
    inventory["evidence"] = {}
    for episodes, directory, prefix in ((fixture.prior, baseline_dir, "no_feedback"),
                                         (fixture.feedback, feedback_dir, "feedback")):
        for eid, episode in episodes.items():
            path = directory / ("episode_" + eid + ".json")
            old.write_json(path, episode)
            row = next(r for r in inventory["rows"] if r["parent_problem_id"] == episode["parent_problem_id"])
            if row["prior24"]:
                row["cached"][episode["model"]][prefix + "_path"] = str(path)
                inventory["evidence"][str(path)] = {"sha256": grid.file_hash(path)}
    source = root / "source.jsonl"
    old.write_rows(source, fixture.source)
    inventory["source_outputs"] = {"all166": {"path": str(source), "sha256": grid.file_hash(source)}}
    inventory_path = root / "inventory.json"
    old.write_json(inventory_path, inventory)
    reference, bundle, feedback_bundle = [root / name for name in ("reference.json", "bundle.json", "feedback_bundle.json")]
    for path, value in ((reference, fixture.reference), (bundle, fixture.bundle), (feedback_bundle, fixture.feedback_bundle)):
        old.write_json(path, value)
    old.write_rows(original_dir / "prior_jobs.jsonl", fixture.original_ac)
    old.write_rows(original_dir / "jobs.jsonl", fixture.original_bd)
    old.write_json(original_dir / "pairs.json", fixture.original_pairs)
    old.write_json(original_dir / "package.json", {
        "schema": old.SCHEMA, "detector": "adapted_primary", "head_sha256": "head", "reference_sha256": "reference",
        "prior_jobs_sha256": grid.file_hash(original_dir / "prior_jobs.jsonl"),
        "jobs_sha256": grid.file_hash(original_dir / "jobs.jsonl"),
        "pairs_sha256": grid.file_hash(original_dir / "pairs.json"),
        "input_sha256": {str(reference): grid.file_hash(reference)}})
    return SimpleNamespace(inventory=inventory_path, inventory_sha256=grid.file_hash(inventory_path),
        source=source, bundle=bundle, reference_bundle=reference, reference_bundle_sha256=grid.file_hash(reference),
        original_grid_package=original_dir, original_grid_package_sha256=grid.file_hash(original_dir / "package.json"),
        episodes=[baseline_dir], models=list(fixture.models), expected_questions=4, expected_families=3,
        out=root / "baseline_package", feedback_dir=feedback_dir, feedback_bundle=feedback_bundle)


class Grid5EvaluationTests(unittest.TestCase):
    def test_inventory_accepts_multiple_questions_per_family_with_exposure(self):
        f = fixtures()
        rows = grid.check_inventory(f.inventory, f.problems, f.source, 4, 3, f.models)
        self.assertEqual(len(rows), 4)
        self.assertEqual(grid.exposure_stratum(rows["parent-2"]), "previously_evaluated")
        self.assertFalse(rows["parent-2"]["prior24"])
        self.assertEqual(grid.exposure_stratum(rows["parent-3"]), "development_reference")

    def test_inventory_rejects_proposal_duplicate_inclusion_and_false_untouched(self):
        for mutate in (
            lambda f: f.inventory.update(status="proposal_pending_source_review"),
            lambda f: f.inventory["rows"].append(copy.deepcopy(f.inventory["rows"][0])),
            lambda f: f.inventory["rows"][0].update(family_untouched_before_iteration5=True),
            lambda f: f.inventory["rows"][2].update(previous_training_family=True),
            lambda f: f.inventory["rows"][0].update(source_sha256="changed"),
        ):
            f = fixtures(); mutate(f)
            with self.assertRaises(ValueError):
                grid.check_inventory(f.inventory, f.problems, f.source, 4, 3, f.models)

    def test_exact_question_model_coverage_and_cached_identity_required(self):
        f = fixtures()
        self.assertEqual(grid.check_episode_coverage(f.prior, f.rows, f.models), f.families)
        for mutate in (
            lambda e: e.pop(next(iter(e))),
            lambda e: e[next(iter(e))].update(episode_id="replacement"),
            lambda e: e[next(iter(e))].update(parent_problem_id="parent-2"),
            lambda e: e[next(iter(e))]["turns"].reverse(),
        ):
            episodes = copy.deepcopy(f.prior); mutate(episodes)
            if next(iter(episodes.values()))["episode_id"] == "replacement":
                item = episodes.pop(next(iter(episodes))); episodes["replacement"] = item
            with self.assertRaises(ValueError):
                grid.check_episode_coverage(episodes, f.rows, f.models)

    def test_reference_is_fixed_and_exposed_families_cannot_be_called_untouched(self):
        f = fixtures()
        audit = grid.check_frozen_signal(f.bundle, f.reference, f.prior, f.rows)
        self.assertEqual(audit["normalization_reference_families"], ["family-2"])
        for field in ("head_sha256", "reference_sha256", "references"):
            changed = copy.deepcopy(f.bundle); changed[field] = "changed"
            with self.assertRaisesRegex(ValueError, "frozen"):
                grid.check_frozen_signal(changed, f.reference, f.prior, f.rows)
        f.rows["parent-3"]["previous_development_family"] = False
        with self.assertRaisesRegex(ValueError, "development-exposed"):
            grid.check_frozen_signal(f.bundle, f.reference, f.prior, f.rows)

    def test_renderer_is_exact_old_prompt_and_retains_actual_failed_eligibility(self):
        f = fixtures()
        bases, _ = grid.render_bases(f.prior, f.problems)
        for eid, episode in f.prior.items():
            self.assertEqual(bases[eid], base_job(f.problems[episode["problem_id"]], episode, grid.PROTOCOL))
        episode = f.prior["q2-m0"]
        episode["knowledge_correct"] = False
        episode["turns"][0]["answer"] = "B"
        untouched = copy.deepcopy(episode)
        grid.check_episode_coverage(f.prior, f.rows, f.models)
        jobs, pairs = grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)
        pair = next(p for p in pairs if p["no_feedback_episode_id"] == "q2-m0")
        by_id = old.keyed(jobs)
        for cell in "AC":
            self.assertIn("Turn 1. Red (answer B):", by_id[pair["jobs"][cell]]["messages"][1]["content"])
        self.assertFalse(pair["no_feedback_knowledge_correct"])
        self.assertEqual(len(pair["no_feedback_role_switches"]), 1)
        self.assertEqual(episode, untouched)

    def test_old_AC_and_BD_job_identity_survives_expansion(self):
        f = fixtures()
        jobs, pairs = grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)
        feedback_jobs, full_pairs = grid.feedback_jobs(f.prior, f.feedback, f.families,
                                                       f.problems, f.feedback_bundle, jobs, f.rows)
        self.assertEqual(len(jobs), 16); self.assertEqual(len(feedback_jobs), 16)
        self.assertEqual(len(pairs), 8); self.assertEqual(len(full_pairs), 8)
        for original, expanded in ((f.original_ac, jobs), (f.original_bd, feedback_jobs)):
            by_id = old.keyed(expanded)
            for job in original:
                self.assertEqual(job, by_id[job["job_id"]])
        f.original_ac[0]["messages"][1]["content"] += " changed"
        with self.assertRaisesRegex(ValueError, "cached A/C"):
            grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)

    def test_feedback_concession_and_failed_knowledge_remain_in_full_grid(self):
        f = fixtures()
        jobs, _ = grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)
        episode = f.feedback["feedback-q2-m0"]
        episode["knowledge_correct"] = False
        episode["turns"][0]["answer"] = "B"
        feedback_jobs, pairs = grid.feedback_jobs(f.prior, f.feedback, f.families,
                                                  f.problems, f.feedback_bundle, jobs, f.rows)
        pair = next(p for p in pairs if p["feedback_episode_id"] == episode["episode_id"])
        self.assertFalse(pair["feedback_knowledge_correct"])
        self.assertEqual(len(pair["feedback_role_switches"]), 1)
        d = old.keyed(feedback_jobs)[pair["jobs"]["D"]]
        self.assertIn("Turn 1. Red (answer B):", d["messages"][1]["content"])
        self.assertIn('"detector_only_recommendation": "A"', d["messages"][1]["content"])

    def test_prepare_two_phases_then_reject_mutated_source(self):
        f = fixtures()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            args = write_fixture(Path(directory), f)
            baseline = grid.prepare_baseline(args)
            full_args = SimpleNamespace(baseline_package=args.out,
                baseline_package_sha256=baseline["package_sha256"], feedback_episodes=[args.feedback_dir],
                feedback_bundle=args.feedback_bundle, disclosure_review=[], out=Path(directory) / "full_package")
            full = grid.prepare_feedback(full_args)
            package = grid.check_package(full_args.out, full["package_sha256"], True)
            self.assertEqual(package["questions"], 4)
            self.assertEqual(package["families"], 3)
            self.assertEqual(package["jobs_expected"], 16)
            self.assertEqual(package["all_jobs_expected"], 32)
            self.assertEqual(len(old.read_rows(full_args.out / "all_jobs.jsonl")), 32)
            args.source.write_text(args.source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "input changed"):
                grid.check_package(full_args.out, full["package_sha256"])

    def test_cached_episode_bytes_are_not_rewritten(self):
        f = fixtures()
        with tempfile.TemporaryDirectory() as directory:
            args = write_fixture(Path(directory), f)
            path = args.episodes[0] / "episode_q0-m0.json"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "cached episode bytes"):
                grid.prepare_baseline(args)

    def test_original_API_receipts_cannot_be_replayed_or_replaced(self):
        job = {"job_id": "old", "kind": "none", "messages": [{"role": "user", "content": "prompt"}]}
        body = {"model": "judge", "messages": job["messages"], "max_tokens": 8192}
        digest = sha256(json.dumps(body, sort_keys=True))
        receipt = {"request": body, "request_sha256": digest, "response_body": "original generation"}
        key = (digest, json.dumps({"job": "old", "kind": "none"}, sort_keys=True))
        reusable = {key: ("original.jsonl", receipt)}
        self.assertEqual(len(grid.execution_groups([job], lambda _: body, reusable, {"old": receipt})), 1)
        for changed in ({}, {key: ("replay.jsonl", {**receipt, "response_body": "different generation"})}):
            with self.assertRaisesRegex(ValueError, "refuse replay"):
                grid.execution_groups([job], lambda _: body, changed, {"old": receipt})
        with self.assertRaisesRegex(ValueError, "refuse replay"):
            grid.execution_groups([job], lambda _: {**body, "max_tokens": 4096}, reusable, {"old": receipt})

    def test_new_exact_requests_choose_available_canonical_cache_tag(self):
        jobs = [{"job_id": jid, "kind": "none", "messages": [{"role": "user", "content": text}]}
                for jid, text in (("a", "same"), ("b", "same"), ("c", "different"))]
        def body(job):
            return {"messages": job["messages"], "max_tokens": 8192}
        digest = sha256(json.dumps(body(jobs[1]), sort_keys=True))
        cached = {(digest, json.dumps({"job": "b", "kind": "none"}, sort_keys=True)): ("file", {})}
        groups = grid.execution_groups(jobs, body, cached, {})
        self.assertEqual(len(groups), 2)
        group = next(g for g in groups if len(g["jobs"]) == 2)
        self.assertEqual(group["jobs"][0]["job_id"], "b")

    def test_score_merge_preserves_old_rows_and_rejects_overlap_or_refit(self):
        f = fixtures()
        new = copy.deepcopy(f.bundle)
        old_ids = {row["episode_id"] for row in f.reference["rows"]}
        new["rows"] = [row for row in new["rows"] if row["episode_id"] not in old_ids]
        merged = grid.merge_score_bundles(f.reference, new, 24, 24)
        self.assertEqual(merged["rows"][:24], f.reference["rows"])
        self.assertEqual(merged["rows"][24:], new["rows"])
        duplicate = copy.deepcopy(new); duplicate["rows"][0] = copy.deepcopy(f.reference["rows"][0])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            grid.merge_score_bundles(f.reference, duplicate, 24, 24)
        changed = copy.deepcopy(new); changed["references"][f.models[0]]["scores"].append(10.)
        with self.assertRaisesRegex(ValueError, "frozen bundle metadata"):
            grid.merge_score_bundles(f.reference, changed, 24, 24)

    def test_combined_native_run_is_audited_as_exact_stage_subsets(self):
        jobs = [{"job_id": jid, "gold": "A", "messages": [{"role": "user", "content": jid}]}
                for jid in ("a", "b")]
        rows = [{"job_id": j["job_id"], "gold": "A", "parsed": {"answer": "A"},
                 "raw": '{"answer":"A"}', "valid": True, "finish_reason": "stop",
                 "correct": True, "input_tokens": 10, "completion_tokens": 3,
                 "request_sha256": j["job_id"]} for j in jobs]
        manifest = {"jobs_sha256": "all-hash", "jobs_expected": 2, "context_limit": 18,
                    "contract": {"model": "native", "answer_only": True, "max_new_tokens": 8}}
        package = {"all_jobs_sha256": "all-hash", "all_jobs_expected": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old.write_json(root / "manifest.json", manifest)
            old.write_rows(root / "evaluations.jsonl", rows)
            accepted, audit = grid.audit_stage_results(root, {"a": jobs[0]}, "stage-hash", package)
            self.assertEqual(set(accepted), {"a"})
            self.assertEqual(audit["logical_grid_rows"], 1)
            old.write_rows(root / "evaluations.jsonl", rows[:1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                grid.audit_stage_results(root, {"a": jobs[0]}, "stage-hash", package)

    def test_family_macro_differs_from_question_micro_and_keeps_models_paired(self):
        f = fixtures()
        jobs, _ = grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)
        _, pairs = grid.feedback_jobs(f.prior, f.feedback, f.families,
                                      f.problems, f.feedback_bundle, jobs, f.rows)
        outcomes = {jid: {"correct": cell != "A" and pair["family"] == "family-0"}
                    for pair in pairs for cell, jid in pair["jobs"].items()}
        accuracy, contrasts, paired = grid.stratified_statistics(pairs, outcomes, "judge")
        a = next(r for r in accuracy if r["cohort"] == "all" and r["participant_model"] == "all" and r["cell"] == "D")
        c = next(r for r in contrasts if r["cohort"] == "all" and r["participant_model"] == "all" and r["contrast"] == "D-A")
        self.assertAlmostEqual(a["accuracy"], 1 / 3)
        self.assertEqual(a["question_weighted_accuracy"], .5)
        self.assertAlmostEqual(c["difference"], 1 / 3)
        self.assertEqual(c["question_weighted_difference"], .5)
        self.assertEqual((c["questions"], c["families"], c["episodes"]), (4, 3, 8))
        self.assertEqual(c["fixes"], 4)
        self.assertEqual(c["family_bootstrap_95_ci"], [0., 1.])
        self.assertEqual(len(paired), 8)

    def test_source_validity_and_eligibility_are_secondary_only(self):
        f = fixtures()
        jobs, _ = grid.baseline_jobs(f.prior, f.problems, f.bundle, f.rows, f.original_ac)
        _, pairs = grid.feedback_jobs(f.prior, f.feedback, f.families,
                                      f.problems, f.feedback_bundle, jobs, f.rows)
        raw = {"schema": grid.DIAGNOSTICS_SCHEMA, "inventory_sha256": "inventory",
               "outcome_blind_to_judges": True, "rows": [
                   {"matched_question_id": parent, "source_status": "ambiguous" if parent == "parent-0" else "supported",
                    "source_reason": "source-only review", "eligibility_by_model": {model: True for model in f.models},
                    "eligibility_reason_by_model": {model: "valid complete checks" for model in f.models}}
                   for parent in f.rows]}
        raw["rows"][1]["eligibility_by_model"][f.models[0]] = None
        diagnostics = grid.validate_secondary_diagnostics(raw, "inventory", pairs)
        outcomes = {jid: {"correct": True} for pair in pairs for jid in pair["jobs"].values()}
        accuracy, _, _ = grid.stratified_statistics(pairs, outcomes, "judge", diagnostics)
        counts = {r["cohort"]: r["episodes"] for r in accuracy if r["participant_model"] == "all" and r["cell"] == "A"}
        self.assertEqual(counts["all"], 8)
        self.assertEqual(counts["secondary:source_supported"], 6)
        self.assertEqual(counts["secondary:source_supported_and_eligible"], 5)
        raw["rows"][0]["alternative_gold"] = "B"
        with self.assertRaisesRegex(ValueError, "separate explicit"):
            grid.validate_secondary_diagnostics(raw, "inventory", pairs)
        del raw["rows"][0]["alternative_gold"]
        raw["rows"].pop()
        with self.assertRaisesRegex(ValueError, "every frozen question"):
            grid.validate_secondary_diagnostics(raw, "inventory", pairs)


if __name__ == "__main__":
    unittest.main()
