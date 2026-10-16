"""Feedback scores must replay the exact action and retain frozen normalization."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from probity_benchmark.grid4_score import (
    IDENTITY_FIELDS,
    LAYERS,
    MODELS,
    ScalarRuntime,
    atomic_json,
    canonical_hash,
    export_bundle,
    project_response,
    validate_request,
    validate_result,
)
from probity_benchmark.judge3_protocol import percentile
from probity_benchmark.preflight import sha256


def fixture():
    model = "qwen/qwen3.5-27b"
    bundle = {
        "schema": "judge3_score_bundle_v1",
        "detector": "adapted_primary",
        "granularity": "response",
        "orientation": "higher_more_unreliable",
        "target": "wrong_answer_advocacy",
        "participant_models": list(MODELS),
        "head_artifacts": {
            m: {
                "adapted": "a" * 64,
                "model_contract": {
                    "model": spec["model"],
                    "model_revision": spec["model_revision"],
                    "layers": list(LAYERS),
                },
            }
            for m, spec in MODELS.items()
        },
        "references": {
            m: {"split": "dev", "parents": ["dev-story"], "scores": [-1, 0, 1]}
            for m in MODELS
        },
        "rows": [],
    }
    bundle["head_sha256"] = sha256(json.dumps(bundle["head_artifacts"], sort_keys=True))
    bundle["reference_sha256"] = sha256(json.dumps(bundle["references"], sort_keys=True))
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Complete source and prior feedback."}],
        "max_tokens": 8192,
        "reasoning": {"enabled": False},
    }
    response = '{"argument": "Caf\u00e9 claim. Another claim.", "answer": "A"}'
    receipt = {
        "request": body,
        "request_sha256": sha256(json.dumps(body, sort_keys=True)),
        "valid": True,
        "finish_reason": "stop",
        "usage": {"completion_tokens": 25},
        "response_body": json.dumps({
            "choices": [{"message": {"content": response, "reasoning": None},
                         "finish_reason": "stop", "native_finish_reason": "stop"}]
        }),
    }
    argument = json.loads(response)["argument"]
    request = {
        "schema": "grid4_probe_request_v1", "job_id": "job-1",
        "model": model, "episode_id": "feedback-1",
        "matched_no_feedback_episode_id": "original-1", "turn_index": 0,
        "parent_problem_id": "test-story", "split": "test", "speaker": "X",
        "argument": argument, "argument_sha256": sha256(argument),
        "messages": copy.deepcopy(body["messages"]), "response": response,
        "response_reasoning": None,
        "source_request_sha256": receipt["request_sha256"],
        "response_body_sha256": sha256(receipt["response_body"]),
        "source_receipt": receipt,
        "head_sha256": bundle["head_sha256"], "reference_sha256": bundle["reference_sha256"],
    }
    result = {
        "schema": "grid4_probe_result_v1",
        **{k: request[k] for k in IDENTITY_FIELDS},
        "score_request_sha256": canonical_hash(request),
        "raw_score": 0.1, "percentile": percentile(0.1, [-1, 0, 1]),
        "provenance": {
            **{k: MODELS[model][k] for k in ("model", "model_revision", "layer", "chat_template_sha256")},
            "adapted_heads_sha256": "a" * 64,
            "head_sha256": bundle["head_sha256"], "reference_sha256": bundle["reference_sha256"],
        },
    }
    return request, result, bundle


class GridScoringContractTests(unittest.TestCase):
    def test_complete_receipt_is_valid(self):
        request, result, bundle = fixture()
        self.assertEqual(validate_request(request, bundle), canonical_hash(request))
        self.assertEqual(validate_result(result, request, bundle), result)

    def test_same_argument_different_context_cannot_reuse_scalar(self):
        request, result, bundle = fixture()
        changed = copy.deepcopy(request)
        changed["messages"][0]["content"] += " New prior probe feedback."
        changed["source_receipt"]["request"]["messages"] = copy.deepcopy(changed["messages"])
        digest = sha256(json.dumps(changed["source_receipt"]["request"], sort_keys=True))
        changed["source_request_sha256"] = digest
        changed["source_receipt"]["request_sha256"] = digest
        validate_request(changed, bundle)
        with self.assertRaisesRegex(ValueError, "different complete request"):
            validate_result(result, changed, bundle)

    def test_changed_body_argument_context_or_reasoning_fails(self):
        request, _, bundle = fixture()
        changes = [
            ("response", "{}"),
            ("argument", "Another argument"),
            ("response_reasoning", "Hidden reasoning lost from receipt"),
            ("response_body_sha256", "0" * 64),
            ("reference_sha256", "0" * 64),
            ("model", "qwen/qwen3.8-27b"),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                changed = copy.deepcopy(request)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_request(changed, bundle)

    def test_preserves_complete_reasoning_and_rejects_opaque_trace(self):
        request, _, bundle = fixture()
        receipt = request["source_receipt"]
        receipt["request"]["reasoning"]["enabled"] = True
        trace = "\n  Full trace with boundary whitespace. \n"
        request["response_reasoning"] = trace
        payload = json.loads(receipt["response_body"])
        message = payload["choices"][0]["message"]
        message["reasoning"] = trace
        message["reasoning_details"] = [{"type": "reasoning.text", "text": trace}]
        receipt["response_body"] = json.dumps(payload)
        request["response_body_sha256"] = sha256(receipt["response_body"])
        request["source_request_sha256"] = receipt["request_sha256"] = sha256(json.dumps(receipt["request"], sort_keys=True))
        validate_request(request, bundle)
        message["reasoning_details"][0]["type"] = "reasoning.encrypted"
        receipt["response_body"] = json.dumps(payload)
        request["response_body_sha256"] = sha256(receipt["response_body"])
        with self.assertRaisesRegex(ValueError, "opaque"):
            validate_request(request, bundle)

    def test_normal_stop_cannot_hide_native_truncation(self):
        request, _, bundle = fixture()
        payload = json.loads(request["source_receipt"]["response_body"])
        payload["choices"][0]["native_finish_reason"] = "length"
        request["source_receipt"]["response_body"] = json.dumps(payload)
        request["response_body_sha256"] = sha256(request["source_receipt"]["response_body"])
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_request(request, bundle)

    def test_frozen_reference_and_percentiles_cannot_drift(self):
        request, result, bundle = fixture()
        changed = copy.deepcopy(bundle)
        changed["references"][request["model"]]["scores"].append(5)
        with self.assertRaisesRegex(ValueError, "content/hash"):
            validate_request(request, changed)
        result["percentile"] += 1
        with self.assertRaisesRegex(ValueError, "percentile"):
            validate_result(result, request, bundle)

    def test_final_bundle_keeps_frozen_metadata_and_requires_all_turns(self):
        request, result, bundle = fixture()
        episode = {
            "episode_id": request["episode_id"], "model": request["model"],
            "parent_problem_id": request["parent_problem_id"], "split": "test", "complete": True,
            "turns": [{"argument": request["argument"], "speaker": request["speaker"],
                       "request_sha256": request["source_request_sha256"]}],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            atomic_json(root / "episodes/episode_1.json", episode)
            atomic_json(root / "queue/requests/job-1.json", request)
            atomic_json(root / "queue/results/job-1.json", result)
            atomic_json(root / "reference.json", bundle)
            export_bundle(root / "episodes", root / "queue", root / "reference.json", root / "out.json")
            actual = json.loads((root / "out.json").read_text())
            self.assertEqual({k: v for k, v in actual.items() if k != "rows"},
                             {k: v for k, v in bundle.items() if k != "rows"})
            self.assertEqual(actual["rows"][0]["scores"], [0.1])
            episode["turns"].append(copy.deepcopy(episode["turns"][0]))
            (root / "episodes/episode_1.json").write_text(json.dumps(episode))
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                export_bundle(root / "episodes", root / "queue", root / "reference.json", root / "missing.json")


class ProjectionTests(unittest.TestCase):
    def test_scalar_matches_original_six_layer_pooling_and_frozen_projection(self):
        generator = torch.Generator().manual_seed(42)
        response = torch.randn(6, 173, 5120, generator=generator).to(torch.bfloat16)
        head = {"weight": torch.randn(5120, generator=generator).double(), "bias": -0.6129954596585053}
        old_features = response.float().mean(1)
        for layer_index in range(6):
            old = float(old_features[layer_index].double() @ head["weight"].double() + head["bias"])
            self.assertEqual(project_response(response[layer_index], head), old)

    def test_capture_only_selected_response_tokens_and_discard_after_score(self):
        runtime = object.__new__(ScalarRuntime)
        runtime.context_length = 16
        runtime.positions, runtime.response_tokens = None, None
        tensor = torch.arange(10 * 5120).reshape(1, 10, 5120).to(torch.bfloat16)
        def forward(**kwargs):
            self.assertFalse(kwargs["use_cache"])
            self.assertTrue(bool(kwargs["attention_mask"].all()))
            self.assertEqual(kwargs["input_ids"].shape, (1, 10))
            runtime.capture_response(None, None, (tensor,))
            self.assertEqual(runtime.response_tokens.shape, (3, 5120))
            self.assertTrue(torch.equal(runtime.response_tokens, tensor[0, [4, 5, 6]]))
        class Model:
            device = "cpu"
            def __call__(self, **kwargs):
                return forward(**kwargs)
        runtime.model = Model()
        head = {"weight": torch.ones(5120).double(), "bias": 0.0}
        result = runtime.score({"ids": list(range(10)), "response_positions": [4, 5, 6]}, head)
        self.assertEqual(result, project_response(tensor[0, [4, 5, 6]], head))
        self.assertIsNone(runtime.response_tokens)
        self.assertIsNone(runtime.positions)
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            runtime.score({"ids": list(range(17)), "response_positions": [4, 5, 6]}, head)


if __name__ == "__main__":
    unittest.main()
