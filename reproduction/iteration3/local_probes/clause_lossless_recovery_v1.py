"""Descriptive-only lossless reason-overflow recovery; frozen modules untouched."""

import argparse
import asyncio
from collections import Counter
import copy
import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema

from probity_benchmark import api
from probity_benchmark import local3_final_runner as runner

BASE = Path("runs/iteration3")
LOCAL = BASE / "local_probes"
FINAL = BASE / "final"
HANDOFF = LOCAL / "final_handoff_v3"
runner.HANDOFF = HANDOFF
from probity_benchmark import local3_secondary_clauses as secondary
from probity_benchmark.local3_annotate_units import chunks, load_inputs, reconcile_unit
from probity_benchmark.local3_verify_twins import validate_checks

secondary.HANDOFF = HANDOFF
OLD = FINAL / "local_opening_clause_annotations_v1"
NEW = FINAL / "local_opening_clause_annotations_v2"
PREP = LOCAL / "clause_lossless_recovery_preparation_v1"
PLAN = PREP / "plan.json"
CANONICAL = PREP / "canonical_earliest_qwen_receipt.json"
AUDIT = FINAL / "local_clause_annotation_completion_audit_v2.json"
MODELS = ["qwen/qwen3.8-27b", "anthropic/claude-opus-5.5"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(model, tag):
    return json.dumps([model, tag], sort_keys=True)


def body_for(models, model, messages, schema):
    spec = models[model]
    prices = {
        name: float(spec["pricing"].get(name, 0))
        for name in ["prompt", "completion", "request"]
    }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 32768,
        "reasoning": secondary.ANNOTATION_REASONING[model],
        "usage": {"include": True},
        "provider": {
            "require_parameters": True,
            "max_price": {
                "prompt": prices["prompt"] * 1e6,
                "completion": prices["completion"] * 1e6,
                "request": prices["request"],
            },
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "diagnostic",
                "strict": True,
                "schema": schema,
            },
        },
    }
    if "temperature" in spec["supported_parameters"]:
        body["temperature"] = 0
    return body


def validate_payload(record, allow_reason_overflow=False):
    if (
        record["http_status"] != 200
        or record["finish_reason"] != "stop"
        or record["usage"]["completion_tokens"] >= 32768
    ):
        raise ValueError("complete normal-stop response required")
    response = json.loads(record["response_body"])
    api.assert_no_exposed_truncation(response)
    if len(response["choices"]) != 1:
        raise ValueError("one complete response required")
    parsed = api.strict_json(response["choices"][0]["message"]["content"])
    schema = record["request"]["response_format"]["json_schema"]["schema"]
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(parsed))
    if allow_reason_overflow:
        if not errors or any(
            error.validator != "maxLength"
            or error.validator_value != 600
            or list(error.absolute_path)[0] != "checks"
            or len(list(error.absolute_path)) != 3
            or list(error.absolute_path)[2] != "reason"
            for error in errors
        ):
            raise ValueError("only explanatory reason maxLength600 may differ")
        schema = copy.deepcopy(schema)
        del schema["properties"]["checks"]["items"]["properties"]["reason"]["maxLength"]
    jsonschema.validate(parsed, schema)
    payload = json.loads(record["request"]["messages"][1]["content"])
    claims = [
        claim
        for argument in payload["ARGUMENTS_TO_ASSESS"]
        for claim in argument["EXACT_SUBSTRINGS_TO_ASSESS"]
    ]
    exact = {claim["index"]: claim["text"] for claim in claims}
    validate_checks(parsed, len(claims), len(payload["SOURCE_STORY_PARAGRAPHS"]))
    if any(check["exact_text"] != exact[check["index"]] for check in parsed["checks"]):
        raise ValueError("exact full unit text changed")
    return parsed, errors


def fixed_jobs():
    freeze = json.loads((FINAL / "global_freeze_v2.json").read_text())
    for path, expected in freeze["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError("frozen experimental source changed: " + path)
    secondary.verify_secondary_lock()
    handoff = json.loads((HANDOFF / "manifest.json").read_text())
    episodes, _ = runner.verify(handoff)
    selected = secondary.opening_ids(episodes)
    rows, sources = load_inputs(
        Path(handoff["dataset"]), [Path(handoff["episodes"])], ["clause"]
    )
    rows = [row for row in rows if row["id"] in selected]
    models = json.loads((OLD / "api_manifest.json").read_text())["models"]
    jobs = {}
    for problem, index, items in chunks(rows, 48):
        paragraphs = sources[problem].split("\n\n")
        payload, _ = secondary.complete_payload(paragraphs, items)
        messages = [
            {"role": "system", "content": secondary.CLAUSE_RULES},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        schema = api.object_schema(
            {
                "checks": {
                    "type": "array",
                    "items": secondary.MIXED_CHECK,
                    "minItems": len(items),
                    "maxItems": len(items),
                }
            }
        )
        tag = {
            "stage": "blind_secondary_fixed_opening_clauses",
            "source_problem": problem,
            "chunk": index,
        }
        for model in MODELS:
            jobs[key(model, tag)] = {
                "model": model,
                "tag": tag,
                "request": body_for(models, model, messages, schema),
                "units": len(items),
            }
    if (
        len(rows) != 48
        or sum(len(row["units"]) for row in rows) != 771
        or len(jobs) != 50
    ):
        raise ValueError("fixed complete clause population changed")
    return rows, sources, models, jobs


def prepare():
    if PREP.exists():
        raise ValueError("immutable preparation required")
    _, _, models, jobs = fixed_jobs()
    records = [
        json.loads(line) for line in (OLD / "requests.jsonl").read_text().splitlines()
    ]
    valid, failed = {}, []
    for index, record in enumerate(records):
        identity = key(record["model"], record["tag"])
        if identity not in jobs or record["request"] != jobs[identity]["request"]:
            raise ValueError("original request differs from exact frozen job")
        if record["request_sha256"] != api.sha256(
            json.dumps(record["request"], sort_keys=True)
        ):
            raise ValueError("original request hash mismatch")
        if record["valid"]:
            validate_payload(record)
            if identity in valid:
                raise ValueError("duplicate successful original job")
            valid[identity] = record
        else:
            parsed, errors = validate_payload(record, True)
            failed.append((index, identity, record, parsed, errors))
    if len(valid) != 48 or len(failed) != 2 or len({item[1] for item in failed}) != 1:
        raise ValueError("recovery scope differs from the reviewed single Qwen failure")
    index, identity, original, parsed, errors = failed[0]
    if original["model"] != MODELS[0] or original["attempt"] != 0:
        raise ValueError("earliest complete original Qwen attempt required")
    content = json.loads(original["response_body"])["choices"][0]["message"]["content"]
    if any(
        json.loads(item[2]["response_body"])["choices"][0]["message"]["content"]
        != content
        for item in failed
    ):
        raise ValueError("observed failed payload identity changed")
    missing = set(jobs) - set(valid) - {identity}
    if len(missing) != 1:
        raise ValueError("exactly one genuinely missing job required")
    missing_identity = missing.pop()
    missing_job = jobs[missing_identity]
    if missing_job["model"] != MODELS[1] or missing_job["tag"] != original["tag"]:
        raise ValueError("only matching missing Opus annotation may be requested")
    price = {
        name: float(models[MODELS[1]]["pricing"].get(name, 0))
        for name in ["prompt", "completion", "request"]
    }
    body = missing_job["request"]
    bound = (
        len(
            json.dumps(
                [body["messages"], body["response_format"]["json_schema"]["schema"]],
                ensure_ascii=False,
            ).encode()
        )
        + 2048
    )
    upper = bound * price["prompt"] + 32768 * price["completion"] + price["request"]
    PREP.mkdir()
    canonical = {
        "schema": "local3_lossless_reason_overflow_canonical_receipt_v1",
        "scope": "Descriptive-only preservation of earliest complete original response. Full reason retained; no labels, text, indices, evidence or output characters changed.",
        "source_requests_sha256": digest(OLD / "requests.jsonl"),
        "source_receipt_index": index,
        "original_receipt": original,
        "parsed_losslessly": parsed,
        "content_sha256": api.sha256(content),
        "only_schema_errors": [
            {
                "path": list(error.absolute_path),
                "validator": error.validator,
                "limit": error.validator_value,
                "actual_characters": len(error.instance),
            }
            for error in errors
        ],
    }
    CANONICAL.write_text(json.dumps(canonical, ensure_ascii=False, indent=2) + "\n")
    paths = [
        OLD / name
        for name in [
            "requests.jsonl",
            "summary.json",
            "api_manifest.json",
            "manifest.json",
            "operational_amendment_receipt.json",
        ]
    ]
    paths += [
        FINAL / "global_freeze_v2.json",
        FINAL / "clause_annotation_release_v1.json",
        FINAL / "local_annotation_payload_recount_v2.json",
        LOCAL / "annotation_allowance_amendment_v2.json",
        HANDOFF / "manifest.json",
        CANONICAL,
    ]
    plan = {
        "schema": "local3_clause_lossless_recovery_plan_v1",
        "status": "prepared_not_released",
        "adapter_sha256": digest(__file__),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_sha256": {str(path): digest(path) for path in paths},
        "original_directory": str(OLD),
        "recovered_directory": str(NEW),
        "expected_jobs": jobs,
        "valid_reuse_jobs": sorted(valid),
        "canonical_reuse_job": identity,
        "missing_paid_job": missing_identity,
        "new_cap_usd": 1.61,
        "maximum_new_paid_attempts": 2,
        "full_per_call_reservation_usd": upper,
        "full_two_attempt_reservation_usd": 2 * upper,
        "input_token_bound": bound,
        "max_tokens": 32768,
        "timeout_seconds": 1200,
        "original_stage_cost_usd": sum(record["cost_usd"] for record in records),
        "combined_cap_usd": sum(record["cost_usd"] for record in records) + 1.61,
        "scope": "Secondary descriptive diagnostic only. Primary sentence evaluation untouched. Reuse48 original valid jobs and one losslessly canonicalized earliest Qwen response; request only the matching never-completed Opus job with byte-equivalent original body and unchanged one schema retry. Every source, unit, batch, prompt, requested schema, model, reasoning setting and acceptance rule stays fixed. Explanatory reason length is relaxed locally only for the exact bound historical response; full reason retained.",
    }
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps(
            {
                "plan": str(PLAN),
                "plan_sha256": digest(PLAN),
                "canonical_sha256": digest(CANONICAL),
                "adapter_sha256": digest(__file__),
                "full_per_call_reservation_usd": upper,
                "new_cap_usd": plan["new_cap_usd"],
                "combined_cap_usd": plan["combined_cap_usd"],
            },
            indent=2,
        )
    )


def verify_plan():
    plan = json.loads(PLAN.read_text())
    if digest(__file__) != plan["adapter_sha256"]:
        raise ValueError("prepared adapter changed")
    for path, expected in plan["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("prepared recovery input changed: " + path)
    return plan


def guard(plan, models, model, messages, schema, tag, attempt):
    identity = key(model, tag)
    if identity not in plan["expected_jobs"]:
        raise ValueError("unapproved annotation job")
    body = body_for(models, model, messages, schema)
    if body != plan["expected_jobs"][identity]["request"]:
        raise ValueError("original complete request body changed")
    if attempt > (1 if identity == plan["missing_paid_job"] else 0):
        raise ValueError("unapproved retry or reuse attempt")
    return identity, body


def run(release_path):
    plan = verify_plan()
    release = json.loads(release_path.read_text())
    if (
        release.get("status") != "released"
        or release.get("plan_sha256") != digest(PLAN)
        or release.get("adapter_sha256") != digest(__file__)
        or release.get("budget_usd") != plan["new_cap_usd"]
        or release.get("descriptive_only") is not True
    ):
        raise ValueError("explicit hash-bound root recovery release required")
    canonical = json.loads(CANONICAL.read_text())
    original_class = secondary.BudgetedChat

    class GuardedChat(original_class):
        def __init__(self, out, models, budget, **kwargs):
            if (
                out != NEW
                or models != MODELS
                or budget != plan["new_cap_usd"]
                or kwargs["schema_retries"] != 1
                or kwargs["max_tokens"] != 16384
                or kwargs["concurrency"] != 2
                or kwargs["reuse_receipts"] != [OLD / "requests.jsonl"]
            ):
                raise ValueError("unchanged frozen annotation invocation required")
            kwargs.update(max_tokens=32768, request_timeout_seconds=1200)
            super().__init__(out, models, budget, **kwargs)
            self.recovery_calls = Counter()
            (out / "lossless_recovery_release_receipt.json").write_text(
                json.dumps(
                    {
                        "root_release_path": str(release_path),
                        "root_release_sha256": digest(release_path),
                        "plan_sha256": digest(PLAN),
                        "adapter_sha256": digest(__file__),
                        "descriptive_only": True,
                    },
                    indent=2,
                )
                + "\n"
            )

        async def complete(
            self,
            model,
            messages,
            schema,
            tag,
            *,
            schema_name="diagnostic",
            _attempt=0,
            response_validator=None,
        ):
            if schema_name != "diagnostic":
                raise ValueError("frozen schema name changed")
            identity, body = guard(
                plan, self.models, model, messages, schema, tag, _attempt
            )
            self.recovery_calls[identity] += 1
            if self.recovery_calls[identity] > (
                2 if identity == plan["missing_paid_job"] else 1
            ):
                raise ValueError("unexpected duplicate annotation call")
            if identity == plan["canonical_reuse_job"]:
                original = canonical["original_receipt"]
                if body != original["request"]:
                    raise ValueError("canonical request body changed")
                parsed, _ = validate_payload(original, True)
                if parsed != canonical["parsed_losslessly"]:
                    raise ValueError("canonical response changed")
                if response_validator is not None:
                    response_validator(parsed)
                record = {
                    **original,
                    "valid": True,
                    "parsed": parsed,
                    "cost_usd": 0.0,
                    "original_cost_usd": original["cost_usd"],
                    "cost_basis": "previously_accounted_lossless_descriptive_reason_amendment",
                    "reused_from": str(CANONICAL),
                    "canonical_source_sha256": digest(CANONICAL),
                    "valid_under_original_reason_limit": False,
                    "descriptive_only_lossless_reason_overflow": True,
                }
                self.records.append(record)
                with (self.out / "requests.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                return record
            if identity in plan["valid_reuse_jobs"]:
                reuse_key = (
                    api.sha256(json.dumps(body, sort_keys=True)),
                    json.dumps(tag, sort_keys=True),
                )
                if reuse_key not in self.reusable:
                    raise ValueError(
                        "valid original job must reuse; paid duplication forbidden"
                    )
            elif identity != plan["missing_paid_job"]:
                raise ValueError("no additional paid job authorized")
            return await super().complete(
                model,
                messages,
                schema,
                tag,
                schema_name=schema_name,
                _attempt=_attempt,
                response_validator=response_validator,
            )

    secondary.BudgetedChat = GuardedChat
    try:
        args = SimpleNamespace(
            out=NEW,
            max_units=48,
            budget=plan["new_cap_usd"],
            concurrency=2,
            reuse_receipts=[OLD / "requests.jsonl"],
            plan_only=False,
        )
        asyncio.run(secondary.run(args))
    finally:
        secondary.BudgetedChat = original_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        if args.release is None:
            raise ValueError("root release required")
        run(args.release)


if __name__ == "__main__":
    main()
