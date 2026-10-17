import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from probity_benchmark import grid4_generate as engine
from probity_benchmark import grid5_generate as grid
from probity_benchmark.preflight import load_problems, sha256


ROOT = Path(__file__).resolve().parents[1]


class FakeAPI:
    def __init__(self, wrong_knowledge=False, role_deviation=False, malformed=None):
        self.models = {m: {"pricing": {"prompt": "0.00000026", "completion": "0.0000026"},
                          "supported_parameters": ["temperature"], "context_length": 262144,
                          "top_provider": {"context_length": 262144}}
                       for m in list(grid.MODELS) + [grid.JUDGE]}
        self.max_tokens = 8192
        self.reasoning = {"enabled": False}
        self.reasoning_by_model = {}
        self.records = []
        self.wrong_knowledge, self.role_deviation, self.malformed = wrong_knowledge, role_deviation, malformed

    async def complete(self, model, messages, schema, tag):
        if tag["stage"] == "knowledge":
            parsed = {"answer": "B" if self.wrong_knowledge else "A"}
        elif tag["stage"] == "debate":
            answer = "A" if tag["speaker"] == "Red" else "B"
            if self.role_deviation and tag["round"] == 1 and tag["speaker"] == "Red":
                answer = "B"
            parsed = {"argument": "Complete argument with unicode Ω and exact quotation.", "answer": answer}
        else:
            parsed = {"answer": "B", "explanation": "Historical diagnostic kept intact."}
        body = grid.expected_api_body(self, model, messages, schema)
        payload = {"id": "fake-" + str(len(self.records)),
                   "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(parsed)}}],
                   "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130, "cost": .0001}}
        if self.malformed and tag["stage"] == "debate":
            self.malformed(payload)
        record = {"tag": copy.deepcopy(tag), "model": model, "request": body,
                  "request_sha256": sha256(json.dumps(body, sort_keys=True)), "valid": True,
                  "http_status": 200, "response_body": json.dumps(payload), "usage": payload["usage"],
                  "finish_reason": payload["choices"][0]["finish_reason"], "parsed": parsed,
                  "cost_usd": .0001, "cost_basis": "provider_reported"}
        self.records.append(record)
        return record


class ExtensionGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        row = {"synthetic_story": "The complete source states Ada arrived.", "question": "Who arrived?",
               "option_a": "Ada", "option_b": "Bo", "correct_answer": "A",
               "parent_problem_id": "parent", "parent_source_line": 1, "schedule_parent_index": 1000,
               "family_id": "family", "family_name": "family", "variant": "original", "split": "dev",
               "campaign_role": "exploratory_dataset_extension"}
        self.source = self.root / "source.jsonl"
        self.source.write_text(json.dumps(row) + "\n")
        self.problem = load_problems(self.source)[0][0]
        self.row = row
        item = {**row, "problem_id": self.problem.problem_id, "prior24": False,
                "source_sha256": sha256(row["synthetic_story"]), "question_sha256": sha256(row["question"])}
        self.inventory = self.root / "inventory.json"
        self.inventory.write_text(json.dumps({"status": "frozen", "rows": [item],
            "source_outputs": {"new142": {"path": str(self.source), "sha256": grid.file_hash(self.source)}}}))
        prep = {"evidence_items": [{"claim": "Ada arrived.", "quotation": "Ada arrived."}] * 3,
                "anticipated_challenge": "Who arrived?", "response_to_challenge": "The source says Ada."}
        self.external = {(self.problem.problem_id, m, s): ({}, {"parsed": copy.deepcopy(prep), "request_sha256": "prep-" + s})
                         for m in grid.MODELS for s in ("Red", "Blue")}

    def tearDown(self):
        self.temp.cleanup()

    def run_episode(self, api, model=None, out=None):
        directory = out or self.root
        return asyncio.run(grid.baseline_episode(api, self.problem, self.row, model or grid.MODELS[0], self.external, directory))

    def test_wrong_knowledge_and_role_deviation_remain_complete_outcomes(self):
        api = FakeAPI(wrong_knowledge=True, role_deviation=True)
        e = self.run_episode(api)
        self.assertTrue(e["complete"])
        self.assertFalse(e["knowledge_correct"])
        self.assertFalse(e["turns"][0]["role_adherent"])
        self.assertEqual(e["turns"][0]["assigned_answer"], "A")
        self.assertEqual(e["turns"][0]["answer"], "B")
        self.assertEqual(len(e["turns"]), 6)
        self.assertEqual(len(api.records), 8)
        self.assertIn("Red (answer B):", api.records[2]["request"]["messages"][-1]["content"])
        self.assertIn("Red (answer B):", api.records[-1]["request"]["messages"][-1]["content"])
        self.assertEqual(api.records[-1]["request"]["response_format"]["json_schema"]["schema"], grid.DIAGNOSTIC_JUDGE_SCHEMA)

    def test_technical_truncation_cannot_continue_or_become_complete(self):
        api = FakeAPI(malformed=lambda p: p["choices"][0].update(finish_reason="length"))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_episode(api)
        self.assertEqual(len(api.records), 2)
        saved = json.loads(next(self.root.glob("episode_*.json")).read_text())
        self.assertFalse(saved["complete"])
        self.assertEqual(saved["turns"], [])

    def test_exposed_truncation_and_invalid_answers_remain_rejected(self):
        for mutate in [lambda p: p["choices"][0]["message"].update(truncated=True),
                       lambda p: p["choices"][0]["message"].update(content='{"argument":"text","answer":"C"}')]:
            with self.subTest(mutate=mutate):
                api = FakeAPI(malformed=mutate)
                with self.assertRaises((ValueError, grid.jsonschema.ValidationError)):
                    self.run_episode(api)
                self.assertEqual(len(api.records), 2)

    def test_inventory_requires_freeze_and_exact_inputs(self):
        h = grid.file_hash(self.inventory)
        problems, _, _, _ = grid.frozen_inventory(self.source, self.inventory, h)
        self.assertEqual(len(problems), 1)
        value = json.loads(self.inventory.read_text())
        value["status"] = "proposal"
        self.inventory.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "explicitly frozen"):
            grid.frozen_inventory(self.source, self.inventory, grid.file_hash(self.inventory))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            grid.frozen_inventory(self.source, self.inventory, h)

    def test_api_price_or_temperature_contract_drift_fails_before_calls(self):
        api = FakeAPI()
        templates = {}
        for m, stage, schema in [(m, s, schema) for m in grid.MODELS for s, schema in
                                [("knowledge", grid.KNOWLEDGE_SCHEMA), ("debate", grid.PUBLIC_SCHEMA)]] + [(grid.JUDGE, "judge", grid.DIAGNOSTIC_JUDGE_SCHEMA)]:
            body = grid.expected_api_body(api, m, [], schema)
            body.pop("messages")
            templates[(m, stage)] = body
        grid.audit_api_templates(api, templates)
        api.models[grid.MODELS[0]]["pricing"]["prompt"] = "0.1"
        with self.assertRaisesRegex(ValueError, "frozen historical contract"):
            grid.audit_api_templates(api, templates)
        self.assertEqual(api.records, [])

    def feedback_fixture(self):
        directory = self.root / "baseline"
        directory.mkdir()
        api = FakeAPI(wrong_knowledge=True, role_deviation=True)
        for model in grid.MODELS:
            self.run_episode(api, model, directory)
        (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in api.records))
        (directory / "api_manifest.json").write_text(json.dumps({"models": api.models}))
        (directory / "manifest.json").write_text(json.dumps({"protocol": grid.PROTOCOL,
            "inventory_sha256": grid.file_hash(self.inventory), "dataset_audit": {"dataset_sha256": grid.file_hash(self.source)}}))
        bundle = self.root / "bundle.json"
        bundle.write_text(json.dumps({"detector": "adapted_primary", "granularity": "response",
            "target": "wrong_answer_advocacy", "orientation": "higher_more_unreliable",
            "head_sha256": "head", "reference_sha256": "reference", "rows": [],
            "references": {m: {"scores": [0., 1.]} for m in grid.MODELS}}))
        return SimpleNamespace(source=self.source, inventory=self.inventory, inventory_sha256=grid.file_hash(self.inventory),
                               episodes=directory, bundle=bundle, out=self.root / "feedback")

    def test_feedback_accepts_diagnostic_failures_and_unchanged_engine_loads(self):
        args = self.feedback_fixture()
        result = grid.initialize_feedback(args)
        self.assertEqual(result["families"], 1)
        self.assertEqual(result["episodes"], 2)
        manifest, problems, episodes = engine.load_run(args.out)
        self.assertEqual(manifest["source_code_sha256"], grid.file_hash(engine.__file__))
        self.assertTrue(all(not e["knowledge_correct"] for e in episodes))
        grid.audit_extension_run(args.out)
        dry = asyncio.run(engine.advance(SimpleNamespace(out=args.out, model=None, dry_run=True)))
        self.assertEqual(dry["actions_ready"], 2)
        for e in episodes:
            old = json.loads((args.episodes / f"episode_{e['matched_no_feedback_episode_id']}.json").read_text())
            self.assertEqual(e, engine.initial_episode(old, manifest["detector"]))
            self.assertEqual(engine.feedback_messages(problems[e["problem_id"]], e),
                             engine.feedback_messages(self.problem, engine.initial_episode(old, manifest["detector"])))

    def test_feedback_rejects_changed_baseline_context(self):
        args = self.feedback_fixture()
        path = next(args.episodes.glob("episode_*.json"))
        old = json.loads(path.read_text())
        old["turns"][0]["argument"] += " changed"
        path.write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "transcript/receipt mismatch"):
            grid.initialize_feedback(args)
        self.assertFalse((args.out / "manifest.json").exists())


@unittest.skipUnless((ROOT / "runs/iteration3/final/debates_v1/requests.jsonl").exists(), "private historical receipts are optional")
class CachedIdentityTests(unittest.TestCase):
    def test_all_48_cached_baselines_reproduce_exact_requests_and_outputs(self):
        source = ROOT / "runs/iteration3/coordinator/holdout_families_source_v5.jsonl"
        problems, _ = load_problems(source)
        rows = grid.read_rows(source)
        external = grid.external_preparations(problems, rows, ROOT / "runs/iteration3/final/preparations_v3/preparations.jsonl")
        olddir = ROOT / "runs/iteration3/final/debates_v1"
        old_records = grid.read_rows(olddir / "requests.jsonl")
        by_tag = {(r["model"], json.dumps(r["tag"], sort_keys=True)): r for r in old_records}

        class ReuseAPI:
            def __init__(self):
                self.models = json.loads((olddir / "api_manifest.json").read_text())["models"]
                self.max_tokens = 8192
                self.reasoning = {"enabled": False}
                self.reasoning_by_model = {}
                self.calls = 0

            async def complete(inner, model, messages, schema, tag):
                record = by_tag[(model, json.dumps(tag, sort_keys=True))]
                body = grid.expected_api_body(inner, model, messages, schema)
                self.assertEqual(body, record["request"])
                self.assertEqual(sha256(json.dumps(body, sort_keys=True)), record["request_sha256"])
                inner.calls += 1
                return copy.deepcopy(record)

        api = ReuseAPI()
        grid.audit_api_templates(api, grid.reference_templates(olddir / "requests.jsonl"))
        with tempfile.TemporaryDirectory() as out:
            for problem, row in zip(problems, rows):
                for model in grid.MODELS:
                    result = asyncio.run(grid.baseline_episode(api, problem, row, model, external, Path(out)))
                    old = json.loads((olddir / f"episode_{result['episode_id']}.json").read_text())
                    self.assertEqual(result, old)
        self.assertEqual(api.calls, 384)


if __name__ == "__main__":
    unittest.main()
