"""Score complete feedback actions with the frozen iteration-3 response probe.

The watch-directory worker keeps one pinned participant model resident. Each
action is replayed once, unpadded, using the *same complete Yes.-followup input*
as extract_iteration. Only the selected layer's final-response tokens reach
CPU memory; they are pooled and discarded immediately. No activation file is
written. CPU float32 pooling and float64 projection match score_frozen_response.

The scalar cache is keyed by the complete request object, including the raw
provider receipt, detector artifacts, and source turn identity. Never reuse a
score merely because two visible arguments happen to have the same text.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time

import torch

from .evaluate_debate_transfer import argument_mapping
from .judge3_final import audit_completion
from .judge3_protocol import percentile, validate_bundle
from .preflight import sha256
from .prepare_debate_corpus import returned_reasoning
from .qwen_extract import LAYERS, Runtime, serialize


MODELS = {
    "qwen/qwen3.5-27b": {
        "short": "qwen35",
        "model": "Qwen/Qwen3.5-27B",
        "model_revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "layer": 43,
        "chat_template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    },
    "qwen/qwen3.8-27b": {
        "short": "qwen38",
        "model": "Qwen/Qwen3.8-27B",
        "model_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "layer": 35,
        "chat_template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    },
}
IDENTITY_FIELDS = (
    "job_id",
    "model",
    "episode_id",
    "matched_no_feedback_episode_id",
    "turn_index",
    "argument_sha256",
    "source_request_sha256",
    "response_body_sha256",
    "head_sha256",
    "reference_sha256",
)
PARITY_SCORE_ATOL = 1e-4
SCOPE = (
    "Complete unpadded teacher-forced API-text replay, including every returned "
    "reasoning character when present. No live-generation, opaque provider "
    "serialization, padded-execution or unreturned-reasoning equivalence claim."
)


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    """Publish complete files only; an existing file is immutable."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("refusing to replace different immutable artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def validate_reference(bundle):
    if (bundle.get("schema"), bundle.get("detector"), bundle.get("granularity"),
        bundle.get("orientation"), bundle.get("target")) != (
        "judge3_score_bundle_v1", "adapted_primary", "response",
        "higher_more_unreliable", "wrong_answer_advocacy",
    ):
        raise ValueError("the frozen adapted-response bundle is required")
    # These match response3_bundle's original encoding, not canonical_hash.
    for field in ("head", "reference"):
        source = bundle["head_artifacts" if field == "head" else "references"]
        if sha256(json.dumps(source, sort_keys=True)) != bundle[field + "_sha256"]:
            raise ValueError("frozen bundle content/hash mismatch")
    if set(bundle["participant_models"]) != set(MODELS):
        raise ValueError("frozen participant coverage changed")
    for model, spec in MODELS.items():
        contract = bundle["head_artifacts"][model]["model_contract"]
        if any(contract[k] != spec[k] for k in ("model", "model_revision")):
            raise ValueError("model revision differs from iteration 3")
        if list(contract["layers"]) != list(LAYERS):
            raise ValueError("frozen extraction layer contract changed")
        ref = bundle["references"][model]
        if ref["split"] != "dev" or not ref["parents"] or not ref["scores"]:
            raise ValueError("missing fixed development reference")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in ref["scores"]):
            raise ValueError("nonfinite fixed development reference")


def validate_request(request, bundle, model=None):
    validate_reference(bundle)
    if request.get("schema") != "grid4_probe_request_v1":
        raise ValueError("unknown scoring request schema")
    if model is not None and request["model"] != model:
        raise ValueError("request sent to a different participant model")
    if request["model"] not in MODELS:
        raise ValueError("unsupported participant")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", request["job_id"]):
        raise ValueError("unsafe job identity")
    if type(request["turn_index"]) is not int or request["turn_index"] < 0:
        raise ValueError("invalid turn index")
    for field in IDENTITY_FIELDS:
        if field not in request:
            raise ValueError("missing score identity: " + field)
    for field in ("head_sha256", "reference_sha256"):
        if request[field] != bundle[field]:
            raise ValueError("request detector differs from frozen C detector")
    if (request["split"] == "test" and request["parent_problem_id"] in
            bundle["references"][request["model"]]["parents"]):
        raise ValueError("test family leaked into score normalization")
    receipt = request["source_receipt"]
    audit_completion(receipt)
    body = receipt["request"]
    body_hash = sha256(json.dumps(body, sort_keys=True))
    if body_hash != receipt["request_sha256"] or body_hash != request["source_request_sha256"]:
        raise ValueError("source request hash mismatch")
    if body["model"] != request["model"] or body["messages"] != request["messages"]:
        raise ValueError("source model or complete action context changed")
    if sha256(receipt["response_body"]) != request["response_body_sha256"]:
        raise ValueError("source response body hash mismatch")
    message = json.loads(receipt["response_body"])["choices"][0]["message"]
    if message["content"] != request["response"]:
        raise ValueError("complete response differs from receipt")
    if returned_reasoning(receipt) != request["response_reasoning"]:
        raise ValueError("returned reasoning differs from receipt")
    argument, _ = argument_mapping(request["response"])
    if not argument or argument != request["argument"] or sha256(argument) != request["argument_sha256"]:
        raise ValueError("public argument differs from complete response")
    return canonical_hash(request)


def project_response(response_tokens, head):
    """Identical dtype/order to extract_iteration + score_frozen_response."""
    if response_tokens.ndim != 2 or not response_tokens.shape[0]:
        raise ValueError("empty or malformed response activations")
    if response_tokens.shape[1] != head["weight"].numel():
        raise ValueError("head/activation dimension mismatch")
    if response_tokens.device.type != "cpu":
        raise ValueError("exact original CPU pooling required")
    if not bool(torch.isfinite(response_tokens).all()):
        raise ValueError("nonfinite response activations")
    mean = response_tokens.float().mean(0)
    score = float(mean.double() @ head["weight"].double() + head["bias"])
    if not math.isfinite(score):
        raise ValueError("nonfinite frozen score")
    return score


def load_head(path, bundle, model):
    if file_hash(path) != bundle["head_artifacts"][model]["adapted"]:
        raise ValueError("head artifact changed")
    heads = torch.load(path, map_location="cpu", weights_only=True)
    contract = heads["__contract__"]
    spec = MODELS[model]
    if contract.get("selected_feature") != "response_mean":
        raise ValueError("selected feature differs from frozen primary")
    for field in ("model", "model_revision"):
        if contract[field] != spec[field]:
            raise ValueError("head model contract mismatch")
    if list(contract["layers"]) != list(LAYERS):
        raise ValueError("head layer contract mismatch")
    head = heads["response_mean"]
    if head["layer"] != spec["layer"] or tuple(head["weight"].shape) != (5120,):
        raise ValueError("frozen response head changed")
    if not bool(torch.isfinite(head["weight"]).all()) or not math.isfinite(head["bias"]):
        raise ValueError("invalid frozen head")
    return head


class ScalarRuntime(Runtime):
    """Reuse the pinned loader, but never copy whole-context hidden states."""

    def __init__(self, model, head):
        spec = MODELS[model]
        super().__init__(spec["model"], spec["model_revision"])
        if sha256(self.tokenizer.chat_template) != spec["chat_template_sha256"]:
            raise ValueError("chat template differs from original C extraction")
        for handle in self.handles:
            handle.remove()
        self.handles = [self.model.model.layers[head["layer"]].register_forward_hook(self.capture_response)]
        self.positions = None
        self.response_tokens = None

    def capture_response(self, module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 5120:
            raise ValueError("unexpected selected decoder output")
        if self.response_tokens is not None or self.positions is None:
            raise ValueError("unexpected selected-layer hook invocation")
        # Index on GPU before transfer. No prompt/DYL state is copied or saved.
        self.response_tokens = value[0, self.positions].detach().to("cpu", copy=True)

    @torch.inference_mode()
    def score(self, serialized, head):
        ids, positions = serialized["ids"], serialized["response_positions"]
        if not ids or len(ids) > self.context_length:
            raise ValueError("empty or oversized complete input; refusing truncation")
        if not positions or min(positions) < 0 or max(positions) >= len(ids):
            raise ValueError("invalid response token positions")
        self.positions, self.response_tokens = positions, None
        device = self.model.device
        input_ids = torch.tensor([ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                logits_to_keep=1,
            )
            if self.response_tokens is None or self.response_tokens.shape[0] != len(positions):
                raise ValueError("selected token capture incomplete")
            return project_response(self.response_tokens, head)
        finally:
            self.positions, self.response_tokens = None, None


class Scorer:
    def __init__(self, model, bundle_path, head_path):
        self.model = model
        self.bundle = json.loads(Path(bundle_path).read_text())
        validate_reference(self.bundle)
        self.head = load_head(head_path, self.bundle, model)
        self.runtime = ScalarRuntime(model, self.head)
        self.provenance = {
            "model": MODELS[model]["model"],
            "model_revision": MODELS[model]["model_revision"],
            "selected_feature": "response_mean",
            "layer": self.head["layer"],
            "adapted_heads_sha256": file_hash(head_path),
            "reference_artifact_sha256": file_hash(bundle_path),
            "head_sha256": self.bundle["head_sha256"],
            "reference_sha256": self.bundle["reference_sha256"],
            "scorer_sha256": file_hash(__file__),
            "original_runtime_sha256": file_hash(Path(__file__).with_name("qwen_extract.py")),
            "chat_template_sha256": sha256(self.runtime.tokenizer.chat_template),
            "gpu": torch.cuda.get_device_name(0),
            "versions": {n: importlib.metadata.version(n) for n in (
                "torch", "transformers", "flash-linear-attention", "causal-conv1d")},
            "dtype": "bfloat16",
            "allow_bf16_reduced_precision_reduction": False,
            "execution": "one complete unpadded Yes.-followup replay per action; no cached prefix/padding",
            "pooling": "All final JSON response tokens, excluding prior messages and returned reasoning; float32 CPU mean then float64 frozen linear projection, identical to original scoring.",
            "persistent_output": "scalar scores, hashes and counts only; no activations",
            "scope": SCOPE,
        }

    def score(self, request):
        request_hash = validate_request(request, self.bundle, self.model)
        started = time.monotonic()
        serial = serialize(self.runtime.tokenizer, request["messages"], request["response"],
                           "Yes.", reasoning=request["response_reasoning"])
        raw_score = self.runtime.score(serial, self.head)
        return {
            "schema": "grid4_probe_result_v1",
            **{key: request[key] for key in IDENTITY_FIELDS},
            "score_request_sha256": request_hash,
            "raw_score": raw_score,
            "percentile": percentile(raw_score, self.bundle["references"][self.model]["scores"]),
            "input_tokens": len(serial["ids"]),
            "response_tokens": len(serial["response_positions"]),
            "serialized_yes_sha256": sha256(serial["text"]),
            "returned_reasoning_sha256": serial["returned_reasoning_sha256"],
            "returned_reasoning_range": serial["returned_reasoning_range"],
            "termination": "teacher_forced_complete_sequence",
            "elapsed_seconds": time.monotonic() - started,
            "provenance": self.provenance,
        }


def validate_result(result, request, bundle):
    digest = validate_request(request, bundle)
    if result.get("schema") != "grid4_probe_result_v1" or result.get("score_request_sha256") != digest:
        raise ValueError("score cache is for a different complete request")
    if any(result.get(k) != request[k] for k in IDENTITY_FIELDS):
        raise ValueError("score identity mismatch")
    if type(result["raw_score"]) not in (float, int) or not math.isfinite(result["raw_score"]):
        raise ValueError("nonfinite score cache")
    reference = bundle["references"][request["model"]]["scores"]
    if result["percentile"] != percentile(result["raw_score"], reference):
        raise ValueError("cached percentile differs from frozen reference")
    provenance = result["provenance"]
    for key in ("head_sha256", "reference_sha256"):
        if provenance[key] != bundle[key]:
            raise ValueError("cached scorer provenance changed")
    spec = MODELS[request["model"]]
    if any(provenance[k] != spec[k] for k in ("model", "model_revision", "layer", "chat_template_sha256")):
        raise ValueError("cached scorer model contract changed")
    if provenance["adapted_heads_sha256"] != bundle["head_artifacts"][request["model"]]["adapted"]:
        raise ValueError("cached scorer used a different head")
    return result


def prepare_parity(episodes_dir, bundle_path, out):
    """Choose shortest and longest complete old action contexts per model."""
    bundle = json.loads(Path(bundle_path).read_text())
    validate_reference(bundle)
    receipts = {}
    for line in (Path(episodes_dir) / "requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("valid"):
            receipts[row["request_sha256"]] = row
    old_scores = {(r["episode_id"], r["turn_index"]): r for r in bundle["rows"]}
    candidates = {m: [] for m in MODELS}
    for path in sorted(Path(episodes_dir).glob("episode_*.json")):
        episode = json.loads(path.read_text())
        for index, turn in enumerate(episode["turns"]):
            receipt = receipts[turn["request_sha256"]]
            response = json.loads(receipt["response_body"])["choices"][0]["message"]["content"]
            request = {
                "schema": "grid4_probe_request_v1",
                "job_id": "parity-" + episode["episode_id"] + "-" + str(index),
                "episode_id": episode["episode_id"],
                "matched_no_feedback_episode_id": episode["episode_id"],
                "turn_index": index, "model": episode["model"],
                "parent_problem_id": episode["parent_problem_id"], "split": episode["split"],
                "speaker": turn["speaker"], "argument": turn["argument"],
                "argument_sha256": sha256(turn["argument"]),
                "messages": receipt["request"]["messages"], "response": response,
                "response_reasoning": returned_reasoning(receipt),
                "source_request_sha256": turn["request_sha256"],
                "response_body_sha256": sha256(receipt["response_body"]),
                "source_receipt": receipt,
                "head_sha256": bundle["head_sha256"], "reference_sha256": bundle["reference_sha256"],
            }
            validate_request(request, bundle)
            score = old_scores[(episode["episode_id"], index)]["scores"][0]
            candidates[episode["model"]].append({
                "request": request, "expected_raw_score": score,
                "expected_percentile": percentile(score, bundle["references"][episode["model"]]["scores"]),
            })
    for model, rows in candidates.items():
        rows.sort(key=lambda r: len(json.dumps(r["request"]["messages"])) + len(r["request"]["response"]))
        if len(rows) < 2:
            raise ValueError("at least two original actions per participant required")
        atomic_json(Path(out) / (MODELS[model]["short"] + ".json"), {
            "schema": "grid4_probe_parity_v1", "model": model,
            "reference_artifact_sha256": file_hash(bundle_path),
            "absolute_score_tolerance": PARITY_SCORE_ATOL,
            "require_identical_percentile": True, "cases": [rows[0], rows[-1]],
        })


def run_parity(scorer, parity_path):
    plan = json.loads(Path(parity_path).read_text())
    if (plan["schema"] != "grid4_probe_parity_v1" or plan["model"] != scorer.model
            or plan["reference_artifact_sha256"] != scorer.provenance["reference_artifact_sha256"]):
        raise ValueError("parity plan differs from scorer contract")
    if plan["absolute_score_tolerance"] != PARITY_SCORE_ATOL or plan["require_identical_percentile"] is not True:
        raise ValueError("parity acceptance rule changed")
    if len(plan["cases"]) < 2 or len({c["request"]["job_id"] for c in plan["cases"]}) < 2:
        raise ValueError("at least two distinct original actions required for parity")
    rows = []
    for case in plan["cases"]:
        result = scorer.score(case["request"])
        delta = abs(result["raw_score"] - case["expected_raw_score"])
        rows.append({"job_id": result["job_id"], "expected_raw_score": case["expected_raw_score"],
                     "expected_percentile": case["expected_percentile"], "result": result,
                     "absolute_score_delta": delta,
                     "passed": delta <= PARITY_SCORE_ATOL and result["percentile"] == case["expected_percentile"]})
    return {"schema": "grid4_probe_parity_receipt_v1", "plan_sha256": file_hash(parity_path),
            "passed": all(r["passed"] for r in rows), "absolute_score_tolerance": PARITY_SCORE_ATOL,
            "require_identical_percentile": True, "cases": rows}


def serve(args):
    scorer = Scorer(args.model, args.reference, args.heads)
    parity = run_parity(scorer, args.parity)
    # Every process lifetime repeats the gate. Each attempt remains immutable,
    # while the reusable worker identity excludes timing and repeat drift.
    parity_path = args.queue / "parity_receipts" / (MODELS[args.model]["short"] + "_" + str(time.time_ns()) + ".json")
    atomic_json(parity_path, parity)
    if not parity["passed"]:
        raise ValueError("original-score parity failed; no feedback action scored")
    manifest = {"schema": "grid4_probe_worker_v1", "model": args.model,
                "parity_plan_sha256": file_hash(args.parity), "provenance": scorer.provenance}
    atomic_json(args.queue / ("worker_" + MODELS[args.model]["short"] + ".json"), manifest)
    (args.queue / "results").mkdir(parents=True, exist_ok=True)
    started = last_new = time.monotonic()
    while True:
        requests = []
        for path in sorted((args.queue / "requests").glob("*.json")):
            request = json.loads(path.read_text())
            if request["model"] == args.model:
                if path.stem != request["job_id"]:
                    raise ValueError("queue filename differs from request identity")
                requests.append(request)
        done = 0
        for request in requests:
            dest = args.queue / "results" / (request["job_id"] + ".json")
            if dest.exists():
                validate_result(json.loads(dest.read_text()), request, scorer.bundle)
            else:
                result = scorer.score(request)
                validate_result(result, request, scorer.bundle)
                atomic_json(dest, result)
                last_new = time.monotonic()
                print(json.dumps({"job_id": request["job_id"], "input_tokens": result["input_tokens"],
                                  "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
            done += 1
        if args.expected_results is not None and done > args.expected_results:
            raise ValueError("queue exceeds fixed expected result count")
        if args.once or done == args.expected_results:
            print(json.dumps({"complete": True, "results": done, "model": args.model}), flush=True)
            return
        if time.monotonic() - started >= args.max_seconds or time.monotonic() - last_new >= args.idle_timeout:
            raise TimeoutError("bounded scoring worker deadline reached")
        time.sleep(args.poll_seconds)


def export_bundle(episodes_dir, queue, reference_path, out):
    reference = json.loads(Path(reference_path).read_text())
    validate_reference(reference)
    episodes = {}
    for path in sorted(Path(episodes_dir).glob("episode_*.json")):
        episode = json.loads(path.read_text())
        if not episode.get("complete") or episode["episode_id"] in episodes:
            raise ValueError("incomplete or duplicate feedback episode")
        episodes[episode["episode_id"]] = episode
    if not episodes:
        raise ValueError("no complete feedback episodes")
    results = {}
    for path in sorted((Path(queue) / "results").glob("*.json")):
        result = json.loads(path.read_text())
        request = json.loads((Path(queue) / "requests" / path.name).read_text())
        validate_result(result, request, reference)
        key = (request["episode_id"], request["turn_index"])
        if key in results:
            raise ValueError("duplicate scored turn")
        results[key] = (request, result)
    expected = {(e["episode_id"], i) for e in episodes.values() for i in range(len(e["turns"]))}
    if set(results) != expected:
        raise ValueError("scalar score cache does not exactly cover feedback episodes")
    output = copy.deepcopy(reference)
    output["rows"] = []
    for (episode_id, index), (request, result) in sorted(results.items()):
        episode = episodes[episode_id]
        turn = episode["turns"][index]
        if (request["model"] != episode["model"] or request["parent_problem_id"] != episode["parent_problem_id"]
                or request["split"] != episode["split"] or request["speaker"] != turn["speaker"]
                or request["source_request_sha256"] != turn["request_sha256"]
                or request["argument_sha256"] != sha256(turn["argument"])):
            raise ValueError("scored request does not match final feedback turn")
        output["rows"].append({"episode_id": episode_id, "turn_index": index,
                               "argument_sha256": request["argument_sha256"],
                               "spans": [[0, len(turn["argument"])]], "scores": [result["raw_score"]]})
    _, covered = validate_bundle(output, episodes)
    if covered != set(episodes):
        raise ValueError("incomplete final score bundle")
    atomic_json(out, output)
    return {"episodes": len(covered), "turns": len(output["rows"]), "sha256": file_hash(out),
            "head_sha256": output["head_sha256"], "reference_sha256": output["reference_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-parity")
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("serve")
    participants = p.add_mutually_exclusive_group(required=True)
    participants.add_argument("--model", choices=list(MODELS))
    participants.add_argument("--participant", choices=[s["short"] for s in MODELS.values()])
    p.add_argument("--heads", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--parity", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--expected-results", type=int)
    p.add_argument("--once", action="store_true")
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--idle-timeout", type=float, default=1800)
    p.add_argument("--poll-seconds", type=float, default=1)
    p = commands.add_parser("export-bundle")
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-parity":
        prepare_parity(args.episodes, args.reference, args.out)
    elif args.command == "serve":
        if args.participant:
            args.model = next(m for m, s in MODELS.items() if s["short"] == args.participant)
        if args.poll_seconds <= 0 or args.max_seconds <= 0 or args.idle_timeout <= 0:
            raise ValueError("worker durations must be positive")
        if args.expected_results is not None and args.expected_results <= 0:
            raise ValueError("expected result count must be positive")
        serve(args)
    else:
        print(json.dumps(export_bundle(args.episodes, args.queue, args.reference, args.out)))


if __name__ == "__main__":
    main()
