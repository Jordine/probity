"""Offline contract checks; fake responses are temporary and never truth data."""

import asyncio
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from probity_benchmark.api import BudgetedChat, object_schema

BASE = Path("runs/iteration3/local_probes")
spec = importlib.util.spec_from_file_location(
    "allowance_adapter", BASE / "run_final_annotations_allowance_v2.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class FakeChat:
    instances = []

    def __init__(self, out, models, budget, **kwargs):
        assert kwargs["max_tokens"] == 32768
        assert kwargs["request_timeout_seconds"] == 1200
        self.request_timeout_seconds = kwargs["request_timeout_seconds"]
        self.max_tokens = kwargs["max_tokens"]
        self.out, self.budget = out, budget
        self.out.mkdir(parents=True)
        self.records, self.spent = [], 0
        self.calls = []
        self.models = {model: RECOUNT["models"][model] for model in models}
        self.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def complete(self, model, messages, schema, tag, **kwargs):
        payload = json.loads(messages[1]["content"])
        if "ARGUMENTS_TO_ASSESS" in payload:
            claims = [
                item
                for group in payload["ARGUMENTS_TO_ASSESS"]
                for item in group["EXACT_SUBSTRINGS_TO_ASSESS"]
            ]
        else:
            claims = payload["EXACT_SUBSTRINGS_TO_ASSESS"]
        parsed = {
            "checks": [
                {
                    "index": item["index"],
                    "exact_text": item["text"],
                    "reason": "Offline transport fixture; no model inference.",
                    "story_paragraph_ids": [],
                    "claim_kind": "interpretation_or_advocacy",
                    "local_label_valid": False,
                    "verdict": "unknown",
                }
                for item in claims
            ]
        }
        kwargs["response_validator"](parsed)
        payload_sha = adapter.payload_digest(messages, schema)
        record = {
            "model": model,
            "tag": tag,
            "parsed": parsed,
            "request_sha256": payload_sha,
            "finish_reason": "stop",
            "cost_usd": 0,
        }
        self.records.append(record)
        self.calls.append((model, tag["source_problem"], tag["chunk"], payload_sha))
        self.last_inputs = model, messages, schema, tag, kwargs
        return record


class FakeHTTPResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def text(self):
        return json.dumps(
            {
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"ok":true}'}}
                ],
            }
        )


class FakeHTTPSession:
    def post(self, url, json):
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        self.body = json
        return FakeHTTPResponse()


async def check_api_body(directory, model, allowance):
    client = BudgetedChat.__new__(BudgetedChat)
    client.models = {
        model: {
            "pricing": RECOUNT["models"][model]["pricing"],
            "top_provider": {
                "context_length": RECOUNT["models"][model]["context_length"]
            },
            "supported_parameters": ["temperature"],
        }
    }
    client.out = directory
    directory.mkdir()
    client.max_tokens, client.budget = allowance, 20
    client.reasoning = {"enabled": False}
    client.reasoning_by_model = AMENDMENT["reasoning_by_model"]
    client.structured_output = True
    client.unavailable, client.reusable = {}, {}
    client.sem, client.lock = asyncio.Semaphore(1), asyncio.Lock()
    client.spent, client.reserved, client.stopped = 0.0, 0.0, False
    client.records, client.schema_retries = [], 1
    client.session = FakeHTTPSession()
    receipt = await client.complete(
        model,
        [{"role": "user", "content": "Offline synthetic transport fixture."}],
        object_schema({"ok": {"type": "boolean"}}),
        {"stage": "offline_transport_fixture"},
    )
    assert receipt["valid"] and receipt["cost_usd"] == 0
    assert client.reserved == 0
    return client.session.body, receipt["reserved_usd"]


async def check_stage(directory, release_path, stage):
    amendment = copy.deepcopy(AMENDMENT)
    amendment["stages"][stage]["out"] = str(directory)
    stage_spec = amendment["stages"][stage]
    module = importlib.import_module(
        "probity_benchmark."
        + (
            "local3_annotate_units"
            if stage == "sentence"
            else "local3_secondary_clauses"
        )
    )
    module.BudgetedChat, original = (
        adapter.make_allowance_client(FakeChat, amendment, release_path, stage),
        module.BudgetedChat,
    )
    args = SimpleNamespace(
        dataset=Path(amendment["dataset"]),
        episodes=[Path(amendment["episodes"])],
        out=directory,
        global_freeze=Path(amendment["global_freeze"]),
        granularities=["sentence"],
        max_units=48,
        budget=stage_spec["budget_usd"],
        concurrency=stage_spec["concurrency"],
        reuse_receipts=[],
        plan_only=False,
    )
    try:
        await module.run(args)
    finally:
        module.BudgetedChat = original
    summary = json.loads((directory / "summary.json").read_text())
    assert (
        summary["expected_units"] == summary["completed_units"] == stage_spec["units"]
    )
    assert summary["expected_rows"] == summary["completed_rows"] == stage_spec["rows"]
    assert summary["accepted_labels"] == {} and summary["errors"] == []
    assert summary["cost_upper_usd"] == 0
    instance = FakeChat.instances[-1]
    expected_calls = {
        (model, job["source_problem"], job["chunk"], job["payload_sha256"])
        for job in RECOUNT["stages"][stage]["jobs"]
        for model in AMENDMENT["models"]
    }
    assert len(instance.calls) == len(expected_calls)
    assert set(instance.calls) == expected_calls
    model, messages, schema, tag, kwargs = instance.last_inputs
    changed_schema = {**schema, "title": "intentional offline mismatch"}
    try:
        await instance.complete(model, messages, changed_schema, tag, **kwargs)
    except ValueError as exc:
        assert "payload changed" in str(exc)
    else:
        raise AssertionError("altered schema was accepted")
    return {
        "stage": stage,
        "complete_payloads_checked": len(expected_calls),
        "rows": stage_spec["rows"],
        "units": stage_spec["units"],
        "changed_payload_rejected": True,
    }


async def run(directory):
    release = directory / "offline_release_fixture.json"
    release.write_text(json.dumps({"offline_fixture_only": True}))
    stages = []
    for stage in ["sentence", "clause"]:
        stages.append(await check_stage(directory / stage, release, stage))
    body_checks = []
    for index, model in enumerate(AMENDMENT["models"]):
        old, old_upper = await check_api_body(directory / f"old{index}", model, 16384)
        new, new_upper = await check_api_body(directory / f"new{index}", model, 32768)
        assert old["max_tokens"] == 16384 and new["max_tokens"] == 32768
        assert {k: v for k, v in old.items() if k != "max_tokens"} == {
            k: v for k, v in new.items() if k != "max_tokens"
        }
        expected_delta = 16384 * float(
            RECOUNT["models"][model]["pricing"]["completion"]
        )
        assert abs(new_upper - old_upper - expected_delta) < 1e-12
        body_checks.append(
            {
                "model": model,
                "only_max_tokens_changed": True,
                "exact_reservation_increase_usd": expected_delta,
            }
        )
    try:
        adapter.verify_release(None, "sentence", AMENDMENT)
    except ValueError as exc:
        assert "explicit root" in str(exc)
    else:
        raise AssertionError("missing root release accepted")
    try:
        adapter.verify_release(release, "clause", AMENDMENT)
    except ValueError as exc:
        assert "not been released" in str(exc)
    else:
        raise AssertionError("unreleased secondary accepted")
    return {
        "stages": stages,
        "body_checks": body_checks,
        "missing_release_rejected": True,
        "unreleased_secondary_rejected": True,
    }


if __name__ == "__main__":
    AMENDMENT = adapter.load_amendment()
    RECOUNT = json.loads(Path(AMENDMENT["payload_recount"]).read_text())
    out = BASE / "annotation_allowance_offline_checks_v2.json"
    if out.exists():
        raise ValueError("new immutable check receipt required")
    with tempfile.TemporaryDirectory(prefix="probity-allowance-offline-") as name:
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("network forbidden")
        ), patch(
            "aiohttp.ClientSession", side_effect=AssertionError("network forbidden")
        ):
            results = asyncio.run(run(Path(name)))
    results.update(
        schema="local3_annotation_allowance_offline_checks_v2",
        amendment_sha256=adapter.digest(adapter.AMENDMENT),
        adapter_sha256=adapter.digest(BASE / "run_final_annotations_allowance_v2.py"),
        check_source_sha256=adapter.digest(__file__),
        paid_calls=0,
        model_inference_performed=False,
        temporary_fake_unknown_annotations_removed=True,
        scope="All 154 complete frozen payloads exercised through unchanged annotation modules using temporary fake transport. Four synthetic API-body checks confirm only max_tokens changes in outgoing bodies and reservation arithmetic follows the increased allowance. Both constructor fixtures require the reviewed 1200-second transport timeout. No real labels, fitting or selection occurred.",
    )
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
