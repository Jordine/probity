"""Dataset-extension queue around the unchanged iteration-four scalar scorer.

Transport metadata stays outside the scoring request. Baseline and feedback
actions share one resident model; newly ready feedback has priority. Only JSON
requests, scalars and provenance are written, never hidden-state tensors.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import time

from . import grid4_score as frozen
from .prepare_debate_corpus import returned_reasoning
from .preflight import sha256


SCHEMA = "grid5_scalar_queue_v1"
ARMS = ("baseline", "feedback")
REFERENCE_SHA256 = "1af6785e0fb48bc8e1c4d09b6fdc744116c985394dcb043639f6dc001e3d7f88"
HEAD_SHA256 = {
    "qwen/qwen3.5-27b": "575e1b0256dc3fb0d3b9b5513d72f81f576f96e2b41ef9b950582957fccea643",
    "qwen/qwen3.8-27b": "045a9bc3fc76f17b1710a05f30563371647bd8a548c33c680ad57c5cfbcfc2d8",
}
SOURCE_SHA256 = {
    "grid4_score.py": "e4c3fa91eff2c0c9d3e7feccdcc4ad106364af138a521b79cfd7be8fcdc4d3e6",
    "qwen_extract.py": "e40bddb9fb594171d3d012803b05e0898597df9e83783df6a9d4d58380cd53ca",
    "run_local_judge.py": "e798394d5436ba81dffc38dd105fc8935bd9d5fd9cc46fc4574af4966815b499",
}
NATIVE_SPEC_SHA256 = "5d8d775bbb1b8c0e0892ab2b894ac4933971d7a66fee2552cefea62b00273f94"
PARITY_SHA256 = {
    "qwen/qwen3.5-27b": "21742fadce559de8fc7d911b886198dc6348dbc2776f52dc46fdf63401fe4e93",
    "qwen/qwen3.8-27b": "d3b360b0043d605e0150d84883802a7aeeaa79ac2884a02636e0c3442a4c537b",
}
VERSIONS = {"torch": "2.9.1+cu128", "transformers": "5.9.0",
            "flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0"}
DEFAULT_REFERENCE = Path("runs/iteration3/final/response_bundles/adapted_primary.json")
DEFAULT_NATIVE_SPEC = Path("runs/iteration2/local_judge_revision.json")


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns " + str(path)) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def assert_frozen_assets(reference, heads, native_spec=DEFAULT_NATIVE_SPEC):
    for name, expected in SOURCE_SHA256.items():
        if frozen.file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError("frozen inference source changed: " + name)
    if frozen.file_hash(reference) != REFERENCE_SHA256:
        raise ValueError("frozen reference artifact changed")
    if frozen.file_hash(native_spec) != NATIVE_SPEC_SHA256:
        raise ValueError("native judge revision artifact changed")
    for model, path in heads.items():
        if model not in HEAD_SHA256 or frozen.file_hash(path) != HEAD_SHA256[model]:
            raise ValueError("frozen adapted head changed")
    bundle = read(reference)
    frozen.validate_reference(bundle)
    return bundle


def cache_bundle_receipt(path, reference):
    value = read(path)
    for key in ("head_sha256", "reference_sha256", "references", "head_artifacts"):
        if value[key] != reference[key]:
            raise ValueError("cached old scores changed frozen " + key)
    rows = value["rows"]
    identities = {(r["episode_id"], r["turn_index"]) for r in rows}
    if len(rows) != 288 or len(identities) != 288:
        raise ValueError("complete old288-action score bundle required")
    if any(len(r["scores"]) != 1 or not math.isfinite(r["scores"][0]) for r in rows):
        raise ValueError("invalid cached old scalar")
    return {"source_path": str(Path(path).resolve()), "sha256": frozen.file_hash(path),
            "rows": len(rows), "purpose": "Old scores retained for exact reuse, never queued as new work."}


def initialize(queue, reference, heads, baseline, feedback, native_spec=DEFAULT_NATIVE_SPEC,
               cached_feedback=Path("runs/iteration4/feedback_v1/adapted_primary.json")):
    queue = Path(queue)
    if (queue / "plan.json").exists():
        raise ValueError("queue already initialized")
    if set(heads) != set(frozen.MODELS):
        raise ValueError("both frozen participant models required")
    if any(type(n) is not int or n < 0 for n in (baseline, feedback)) or baseline + feedback == 0:
        raise ValueError("positive total and nonnegative integer counts required")
    bundle = assert_frozen_assets(reference, heads, native_spec)
    plan = {
        "schema": SCHEMA, "adapter_sha256": frozen.file_hash(__file__),
        "reference_artifact_sha256": frozen.file_hash(reference),
        "head_artifact_sha256": {m: frozen.file_hash(p) for m, p in heads.items()},
        "native_judge_spec_sha256": frozen.file_hash(native_spec),
        "frozen_inference_sources": SOURCE_SHA256, "parity_plan_sha256": PARITY_SHA256,
        "expected": {m: {"baseline": baseline, "feedback": feedback} for m in heads},
        "cached_iteration4_scores": {"baseline": cache_bundle_receipt(reference, bundle),
                                     "feedback": cache_bundle_receipt(cached_feedback, bundle)},
        "scope": "Dataset-only extension; unchanged selected head, complete Yes-context replay and CPU pooling. No activations persisted.",
        "split_policy": "Preserve the independently frozen exposure/split metadata. Full166 is exploratory; this queue never changes a request split to bypass a reference-overlap check.",
    }
    for name in ("requests", "jobs", "results", "parity_receipts", "worker_receipts"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    frozen.atomic_json(queue / "plan.json", plan)
    return plan


def load_plan(queue):
    plan = read(Path(queue) / "plan.json")
    if plan["schema"] != SCHEMA or plan["adapter_sha256"] != frozen.file_hash(__file__):
        raise ValueError("queue adapter changed after freeze")
    if plan["frozen_inference_sources"] != SOURCE_SHA256 or plan["parity_plan_sha256"] != PARITY_SHA256:
        raise ValueError("queue frozen source/parity contract changed")
    if set(plan["expected"]) != set(frozen.MODELS):
        raise ValueError("queue participant coverage changed")
    for counts in plan["expected"].values():
        if set(counts) != set(ARMS) or any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) <= 0:
            raise ValueError("invalid queue expected counts")
    return plan


def baseline_request(episode, index, receipt, bundle, receipt_path=None):
    if not episode.get("complete") or not 0 <= index < len(episode["turns"]):
        raise ValueError("complete baseline episode/turn required")
    turn = episode["turns"][index]
    if receipt["request_sha256"] != turn["request_sha256"]:
        raise ValueError("baseline receipt belongs to another turn")
    message = json.loads(receipt["response_body"])["choices"][0]["message"]
    request = {
        "schema": "grid4_probe_request_v1",
        "job_id": "grid5-baseline-" + sha256(episode["episode_id"] + ":" + str(index) + ":" + turn["request_sha256"]),
        "episode_id": episode["episode_id"], "matched_no_feedback_episode_id": episode["episode_id"],
        "turn_index": index, "model": episode["model"],
        "parent_problem_id": episode["parent_problem_id"], "split": episode["split"],
        "speaker": turn["speaker"], "argument": turn["argument"], "argument_sha256": sha256(turn["argument"]),
        "messages": copy.deepcopy(receipt["request"]["messages"]), "response": message["content"],
        "response_reasoning": returned_reasoning(receipt), "source_request_sha256": turn["request_sha256"],
        "response_body_sha256": sha256(receipt["response_body"]), "source_receipt": copy.deepcopy(receipt),
        "head_sha256": bundle["head_sha256"], "reference_sha256": bundle["reference_sha256"],
    }
    if receipt_path is not None:
        request["source_receipt_path"] = str(Path(receipt_path).resolve())
    frozen.validate_request(request, bundle)
    return request


def validate_item(item, request, plan, plan_hash, bundle):
    if item.get("schema") != "grid5_scalar_transport_v1" or item["plan_sha256"] != plan_hash:
        raise ValueError("transport plan identity changed")
    digest = frozen.validate_request(request, bundle)
    if item["request_sha256"] != digest or item["job_id"] != request["job_id"] or item["model"] != request["model"]:
        raise ValueError("transport/complete request mismatch")
    arm = item["arm"]
    if arm not in ARMS or request["model"] not in plan["expected"]:
        raise ValueError("unsupported transport arm/model")
    self_paired = request["episode_id"] == request["matched_no_feedback_episode_id"]
    if self_paired != (arm == "baseline"):
        raise ValueError("baseline/feedback episode pairing mismatch")


def read_items(queue, plan, bundle, cache=None):
    """Read new immutable envelopes once; reject any subsequent file changes."""
    queue, cache = Path(queue), {} if cache is None else cache
    plan_hash = frozen.file_hash(queue / "plan.json")
    items, counts, identities = [], {m: {a: 0 for a in ARMS} for m in plan["expected"]}, set()
    paths = sorted((queue / "jobs").glob("*.json"))
    if not set(cache) <= {p.stem for p in paths}:
        raise ValueError("published transport item disappeared")
    for path in paths:
        req_path = queue / "requests" / path.name
        stamp = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (path, req_path))
        if path.stem in cache:
            old_stamp, item, request = cache[path.stem]
            if old_stamp != stamp:
                raise ValueError("published transport request changed")
        else:
            item, request = read(path), read(req_path)
            if path.stem != request["job_id"]:
                raise ValueError("transport filename differs from job")
            validate_item(item, request, plan, plan_hash, bundle)
            cache[path.stem] = stamp, item, request
        identity = request["model"], request["episode_id"], request["turn_index"]
        if identity in identities:
            raise ValueError("duplicate episode/turn in transport queue")
        identities.add(identity)
        counts[item["model"]][item["arm"]] += 1
        if counts[item["model"]][item["arm"]] > plan["expected"][item["model"]][item["arm"]]:
            raise ValueError("transport exceeds expected arm count")
        items.append((item, request))
    return items


def publish_many(queue, requests, arm, bundle, cache=None):
    """Validate one batch before publishing; avoid rereading all long contexts per item."""
    requests = list(requests)
    if not requests:
        return []
    queue = Path(queue)
    with lock(queue / ".publish.lock"):
        plan = load_plan(queue)
        plan_hash = frozen.file_hash(queue / "plan.json")
        existing = read_items(queue, plan, bundle, cache)
        by_job = {i["job_id"]: (i, r) for i, r in existing}
        by_turn = {(r["model"], r["episode_id"], r["turn_index"]): r["job_id"] for _, r in existing}
        counts = {m: {a: sum(i["model"] == m and i["arm"] == a for i, _ in existing)
                      for a in ARMS} for m in plan["expected"]}
        pending, accepted = [], []
        for request in requests:
            if Path(request["job_id"]).name != request["job_id"] or request["job_id"] in ("", ".", ".."):
                raise ValueError("unsafe job filename")
            item = {"schema": "grid5_scalar_transport_v1", "plan_sha256": plan_hash,
                    "job_id": request["job_id"], "model": request["model"], "arm": arm,
                    "request_sha256": frozen.canonical_hash(request)}
            validate_item(item, request, plan, plan_hash, bundle)
            accepted.append(item)
            if request["job_id"] in by_job:
                if by_job[request["job_id"]] != (item, request):
                    raise ValueError("different request under an existing job")
                continue
            identity = request["model"], request["episode_id"], request["turn_index"]
            if identity in by_turn:
                raise ValueError("duplicate episode/turn under a different job")
            counts[item["model"]][arm] += 1
            if counts[item["model"]][arm] > plan["expected"][item["model"]][arm]:
                raise ValueError("transport exceeds expected arm count")
            by_job[item["job_id"]] = item, request
            by_turn[identity] = item["job_id"]
            pending.append((item, request))
        for item, request in pending:
            frozen.atomic_json(queue / "requests" / (request["job_id"] + ".json"), request)
            # The immutable envelope is the publication marker, after the full request.
            frozen.atomic_json(queue / "jobs" / (request["job_id"] + ".json"), item)
        return accepted


def publish(queue, request, arm, bundle):
    return publish_many(queue, [request], arm, bundle)[0]


def check_result(result, request, bundle):
    frozen.validate_result(result, request, bundle)
    p = result["provenance"]
    if (p.get("scorer_sha256") != SOURCE_SHA256["grid4_score.py"]
            or p.get("original_runtime_sha256") != SOURCE_SHA256["qwen_extract.py"]
            or p.get("reference_artifact_sha256") != REFERENCE_SHA256
            or p.get("dtype") != "bfloat16" or p.get("allow_bf16_reduced_precision_reduction") is not False
            or p.get("versions") != VERSIONS):
        raise ValueError("cached scalar inference provenance changed")
    if result.get("termination") != "teacher_forced_complete_sequence":
        raise ValueError("incomplete scalar replay")
    if (type(result.get("input_tokens")) is not int or type(result.get("response_tokens")) is not int
            or not 0 < result["response_tokens"] <= result["input_tokens"]):
        raise ValueError("invalid scalar token accounting")
    return result


def next_job(items, done, model):
    pending = [(i, r) for i, r in items if i["model"] == model and i["job_id"] not in done]
    return min(pending, key=lambda v: (v[0]["arm"] != "feedback", v[1]["turn_index"], v[0]["job_id"])) if pending else None


def serve(args):
    queue = Path(args.queue)
    plan = load_plan(queue)
    plan_hash = frozen.file_hash(queue / "plan.json")
    bundle = assert_frozen_assets(args.reference, {args.model: args.heads}, args.native_spec)
    if (plan["reference_artifact_sha256"] != frozen.file_hash(args.reference)
            or plan["head_artifact_sha256"][args.model] != frozen.file_hash(args.heads)
            or plan["native_judge_spec_sha256"] != frozen.file_hash(args.native_spec)
            or frozen.file_hash(args.parity) != plan["parity_plan_sha256"][args.model]):
        raise ValueError("worker artifacts differ from frozen queue plan")
    if any(not math.isfinite(x) or x <= 0 for x in (args.max_seconds, args.idle_timeout, args.poll_seconds)):
        raise ValueError("positive finite worker time bounds required")
    with lock(queue / ".gpu_worker.lock"):
        started = time.monotonic()
        scorer = frozen.Scorer(args.model, args.reference, args.heads)
        parity = frozen.run_parity(scorer, args.parity)
        attempt = frozen.MODELS[args.model]["short"] + "_" + str(time.time_ns())
        frozen.atomic_json(queue / "parity_receipts" / (attempt + ".json"), parity)
        if not parity["passed"]:
            raise ValueError("original-score parity failed; no new action scored")
        for case in parity["cases"]:
            check_result(case["result"], next(c["request"] for c in read(args.parity)["cases"]
                                             if c["request"]["job_id"] == case["job_id"]), bundle)
        frozen.atomic_json(queue / "worker_receipts" / (attempt + ".json"), {
            "schema": "grid5_scalar_worker_v1", "plan_sha256": frozen.file_hash(queue / "plan.json"),
            "adapter_sha256": frozen.file_hash(__file__), "model": args.model,
            "provenance": scorer.provenance, "parity_receipt_sha256": frozen.file_hash(queue / "parity_receipts" / (attempt + ".json"))})
        cache, done, result_stamps = {}, set(), {}
        last_new = time.monotonic()
        while True:
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError("scoring worker reached fixed deadline")
            if frozen.file_hash(queue / "plan.json") != plan_hash:
                raise ValueError("queue plan changed during worker lifetime")
            items = read_items(queue, plan, bundle, cache)
            if not {p.stem for p in (queue / "results").glob("*.json")} <= {i["job_id"] for i, _ in items}:
                raise ValueError("orphan scalar result has no published request")
            for item, request in items:
                if item["model"] != args.model:
                    continue
                path = queue / "results" / (item["job_id"] + ".json")
                if path.exists():
                    stamp = path.stat().st_size, path.stat().st_mtime_ns
                    if item["job_id"] not in done:
                        check_result(read(path), request, bundle)
                        done.add(item["job_id"])
                        result_stamps[item["job_id"]] = stamp
                    elif stamp != result_stamps[item["job_id"]]:
                        raise ValueError("completed scalar result changed")
                elif item["job_id"] in done:
                    raise ValueError("completed scalar result disappeared")
            if len(done) == sum(plan["expected"][args.model].values()):
                summary = {"schema": "grid5_scalar_worker_complete_v1", "model": args.model,
                           "plan_sha256": frozen.file_hash(queue / "plan.json"),
                           "completed": {a: sum(i["arm"] == a and i["job_id"] in done for i, _ in items) for a in ARMS},
                           "elapsed_seconds": time.monotonic() - started, "attempt": attempt}
                frozen.atomic_json(queue / "worker_receipts" / (attempt + "_complete.json"), summary)
                print(json.dumps(summary), flush=True)
                return summary
            candidate = next_job(items, done, args.model)
            if candidate:
                item, request = candidate
                result = scorer.score(request)
                check_result(result, request, bundle)
                frozen.atomic_json(queue / "results" / (item["job_id"] + ".json"), result)
                last_new = time.monotonic()
                print(json.dumps({"job_id": item["job_id"], "arm": item["arm"],
                                  "input_tokens": result["input_tokens"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
                # Recheck the queue before each baseline action so live feedback
                # arriving during an offline batch cannot wait behind that batch.
                continue
            if time.monotonic() - last_new >= args.idle_timeout:
                raise TimeoutError("scoring worker reached idle timeout")
            time.sleep(args.poll_seconds)


def export_arm(queue, episodes_dir, reference, arm, out):
    """Export only one new arm, retaining the original score-bundle format."""
    queue = Path(queue)
    plan, bundle = load_plan(queue), read(reference)
    if frozen.file_hash(reference) != plan["reference_artifact_sha256"]:
        raise ValueError("export reference differs from queue plan")
    episodes = {}
    for path in sorted(Path(episodes_dir).glob("episode_*.json")):
        episode = read(path)
        if not episode.get("complete") or episode["episode_id"] in episodes:
            raise ValueError("incomplete or duplicate episode")
        episodes[episode["episode_id"]] = episode
    if not episodes:
        raise ValueError("no complete episodes")
    expected = {(e["episode_id"], n) for e in episodes.values() for n in range(len(e["turns"]))}
    selected = [(i, r) for i, r in read_items(queue, plan, bundle) if i["arm"] == arm]
    if {(r["episode_id"], r["turn_index"]) for _, r in selected} != expected:
        raise ValueError("scalar queue does not exactly cover requested arm episodes")
    output = copy.deepcopy(bundle)
    output["rows"] = []
    for item, request in sorted(selected, key=lambda v: (v[1]["episode_id"], v[1]["turn_index"])):
        episode, index = episodes[request["episode_id"]], request["turn_index"]
        turn = episode["turns"][index]
        if (request["model"] != episode["model"] or request["parent_problem_id"] != episode["parent_problem_id"]
                or request["split"] != episode["split"] or request["speaker"] != turn["speaker"]
                or request["source_request_sha256"] != turn["request_sha256"]
                or request["argument_sha256"] != sha256(turn["argument"])):
            raise ValueError("scored request does not match final episode turn")
        result = check_result(read(queue / "results" / (item["job_id"] + ".json")), request, bundle)
        output["rows"].append({"episode_id": episode["episode_id"], "turn_index": index,
                               "argument_sha256": request["argument_sha256"],
                               "spans": [[0, len(turn["argument"])]], "scores": [result["raw_score"]]})
    _, covered = frozen.validate_bundle(output, episodes)
    if covered != set(episodes):
        raise ValueError("incomplete final score bundle")
    frozen.atomic_json(out, output)
    return {"arm": arm, "episodes": len(covered), "turns": len(output["rows"]), "sha256": frozen.file_hash(out)}


def tokenizer_preflight(queue, reference, out, participant=None, local_only=False):
    """CPU-only: download tokenizer/config files, never instantiate model weights."""
    from transformers import AutoConfig, AutoTokenizer
    queue, bundle = Path(queue), read(reference)
    plan = load_plan(queue)
    if frozen.file_hash(reference) != plan["reference_artifact_sha256"]:
        raise ValueError("tokenizer reference differs from queue plan")
    items = read_items(queue, plan, bundle)
    rows, models = [], {}
    for model, spec in frozen.MODELS.items():
        if participant and participant != spec["short"]:
            continue
        selected = [(i, r) for i, r in items if i["model"] == model]
        if not selected:
            raise ValueError("no published requests for requested tokenizer model")
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"],
                                                  local_files_only=local_only)
        config = AutoConfig.from_pretrained(spec["model"], revision=spec["model_revision"],
                                            local_files_only=local_only).get_text_config()
        if sha256(tokenizer.chat_template) != spec["chat_template_sha256"]:
            raise ValueError("preflight tokenizer template differs from frozen runtime")
        limit = config.max_position_embeddings
        models[model] = {"revision": spec["model_revision"], "chat_template_sha256": spec["chat_template_sha256"],
                         "context_limit": limit}
        for item, request in selected:
            serial = frozen.serialize(tokenizer, request["messages"], request["response"], "Yes.",
                                      reasoning=request["response_reasoning"])
            if not 0 < len(serial["ids"]) <= limit:
                raise ValueError("complete teacher-forced action exceeds pinned context limit")
            # serialize itself enforces complete roundtrip, action-prefix identity,
            # exact response boundaries, preserved reasoning, and DYL period offset.
            rows.append({"job_id": item["job_id"], "model": model, "arm": item["arm"],
                         "request_sha256": item["request_sha256"], "input_tokens": len(serial["ids"]),
                         "response_tokens": len(serial["response_positions"]),
                         "serialized_yes_sha256": sha256(serial["text"]),
                         "returned_reasoning_sha256": serial["returned_reasoning_sha256"]})
    result = {"schema": "grid5_tokenizer_preflight_v1", "plan_sha256": frozen.file_hash(queue / "plan.json"),
              "passed": True, "models": models, "checked_requests": len(rows),
              "expected_total_requests": sum(sum(v.values()) for v in plan["expected"].values()),
              "scope": "All currently published requests; future feedback turns are checked identically inside frozen Scorer.score. No weights or activations loaded.",
              "rows": rows}
    frozen.atomic_json(out, result)
    return {k: v for k, v in result.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--baseline-per-model", type=int, default=852)
    init.add_argument("--feedback-per-model", type=int, default=852)
    init.add_argument("--cached-feedback", type=Path, default=Path("runs/iteration4/feedback_v1/adapted_primary.json"))
    for short in ("qwen35", "qwen38"):
        init.add_argument("--" + short + "-heads", type=Path, default=Path(f"runs/iteration2/fitted_response_{short}/heads.pt"))
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--requests", type=Path, required=True)
    ingest.add_argument("--arm", choices=ARMS, required=True)
    baseline = sub.add_parser("build-baseline")
    baseline.add_argument("--episodes", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--episodes", type=Path, required=True)
    export.add_argument("--arm", choices=ARMS, required=True)
    export.add_argument("--out", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--participant", choices=[v["short"] for v in frozen.MODELS.values()])
    preflight.add_argument("--local-only", action="store_true")
    worker = sub.add_parser("serve")
    worker.add_argument("--participant", choices=[v["short"] for v in frozen.MODELS.values()], required=True)
    worker.add_argument("--heads", type=Path, required=True)
    worker.add_argument("--parity", type=Path, required=True)
    worker.add_argument("--max-seconds", type=float, required=True)
    worker.add_argument("--idle-timeout", type=float, default=1800)
    worker.add_argument("--poll-seconds", type=float, default=1)
    for command in (init, ingest, baseline, worker, export, preflight):
        command.add_argument("--queue", type=Path, required=True)
        command.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    for command in (init, worker):
        command.add_argument("--native-spec", type=Path, default=DEFAULT_NATIVE_SPEC)
    args = parser.parse_args()
    if args.command == "init":
        heads = {m: getattr(args, spec["short"] + "_heads") for m, spec in frozen.MODELS.items()}
        result = initialize(args.queue, args.reference, heads, args.baseline_per_model, args.feedback_per_model,
                            args.native_spec, args.cached_feedback)
    elif args.command == "serve":
        args.model = next(m for m, v in frozen.MODELS.items() if v["short"] == args.participant)
        serve(args)
        return
    elif args.command == "export":
        result = export_arm(args.queue, args.episodes, args.reference, args.arm, args.out)
    elif args.command == "preflight":
        result = tokenizer_preflight(args.queue, args.reference, args.out, args.participant, args.local_only)
    else:
        bundle = read(args.reference)
        if frozen.file_hash(args.reference) != load_plan(args.queue)["reference_artifact_sha256"]:
            raise ValueError("transport reference differs from plan")
        requests = []
        if args.command == "ingest":
            paths = [args.requests] if args.requests.is_file() else sorted(args.requests.glob("*.json"))
            requests = [read(path) for path in paths]
            publish_many(args.queue, requests, args.arm, bundle)
        else:
            receipt_path = args.episodes / "requests.jsonl"
            receipts = {}
            for line in receipt_path.read_text().splitlines():
                receipt = json.loads(line)
                if receipt.get("valid"):
                    key = receipt["request_sha256"]
                    if key in receipts and receipts[key]["response_body"] != receipt["response_body"]:
                        raise ValueError("ambiguous baseline receipt for exact request")
                    receipts[key] = receipt
            for path in sorted(args.episodes.glob("episode_*.json")):
                episode = read(path)
                for index, turn in enumerate(episode["turns"]):
                    request = baseline_request(episode, index, receipts[turn["request_sha256"]], bundle, receipt_path)
                    requests.append(request)
            publish_many(args.queue, requests, "baseline", bundle)
        result = {"published_requests": len(requests)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
