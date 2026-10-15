"""Apply the approved pre-label output allowance without changing frozen prompts.

The frozen annotation modules run unchanged. This adapter replaces only their
BudgetedChat constructor's max_tokens and transport timeout, plus the approved
stage cap. Every dispatched payload must match the complete-corpus recount.
"""

import argparse
import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

BASE = Path("runs/iteration3/local_probes")
AMENDMENT = BASE / "annotation_allowance_amendment_v2.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_digest(messages, schema):
    return hashlib.sha256(
        json.dumps([messages, schema], ensure_ascii=False).encode()
    ).hexdigest()


def load_amendment():
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["adapter_sha256"] != digest(__file__):
        raise ValueError("operational adapter differs from reviewed amendment")
    if (
        amendment["original_max_tokens"] != 16384
        or amendment["effective_max_tokens"] != 32768
        or amendment["original_request_timeout_seconds"] != 300
        or amendment["effective_request_timeout_seconds"] != 1200
        or amendment["cumulative_cap_usd"] != 75
        or amendment["stages"]["sentence"]["budget_usd"] != 20
        or amendment["stages"]["clause"]["budget_usd"] != 5
    ):
        raise ValueError("operational allowance or planning caps changed")
    for name, expected in amendment["input_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"amendment input changed: {name}")
    freeze = json.loads(Path(amendment["global_freeze"]).read_text())
    for name, expected in freeze["source_hashes"].items():
        if digest(name) != expected:
            raise ValueError(f"frozen experiment source changed: {name}")
    # Reuse the already frozen path-only verifier, including all original locks.
    adapter_path = BASE / "run_final_local_v3.py"
    spec = importlib.util.spec_from_file_location(
        "local3_frozen_path_adapter", adapter_path
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    old_argv = sys.argv
    try:
        sys.argv = [str(adapter_path), "verify"]
        adapter.main()
    finally:
        sys.argv = old_argv
    return amendment


def verify_release(path, stage, amendment):
    if path is None:
        raise ValueError("explicit root stage-release receipt required")
    release = json.loads(path.read_text())
    if (
        release.get("schema") != "local3_annotation_stage_release_v1"
        or release.get("status") != "released"
        or release.get("amendment_sha256") != digest(AMENDMENT)
        or release.get("adapter_sha256") != digest(__file__)
        or stage not in release.get("approved_stages", [])
    ):
        raise ValueError("stage has not been released for this exact amendment")
    validity = release.get("validity_receipts_sha256", {})
    if not validity:
        raise ValueError("root validity-review receipt must accompany release")
    for name, expected in validity.items():
        if digest(name) != expected:
            raise ValueError("released validity-review receipt changed")
    ledger_path = Path(os.environ.get("PROBITY_PILOT_LEDGER", ""))
    if ledger_path.resolve() != Path(amendment["ledger"]).resolve():
        raise ValueError("the shared cumulative budget ledger is required")
    if (
        json.loads(ledger_path.read_text())["cap_usd"]
        != amendment["cumulative_cap_usd"]
    ):
        raise ValueError("shared cumulative ceiling changed")
    return release


def make_allowance_client(base_class, amendment, release_path, stage):
    stage_spec = amendment["stages"][stage]
    recount = json.loads(Path(amendment["payload_recount"]).read_text())
    jobs = {
        (job["source_problem"], job["chunk"]): job
        for job in recount["stages"][stage]["jobs"]
    }
    expected_keys = {
        "max_tokens",
        "schema_retries",
        "concurrency",
        "reuse_receipts",
        "reasoning_by_model",
    }

    class AllowanceChat(base_class):
        def __init__(self, out, models, budget, **kwargs):
            if (
                Path(out) != Path(stage_spec["out"])
                or models != amendment["models"]
                or budget != stage_spec["budget_usd"]
                or set(kwargs) != expected_keys
                or kwargs["max_tokens"] != amendment["original_max_tokens"]
                or kwargs["schema_retries"] != 1
                or kwargs["concurrency"] != stage_spec["concurrency"]
                or kwargs["reuse_receipts"] != []
                or kwargs["reasoning_by_model"] != amendment["reasoning_by_model"]
            ):
                raise ValueError("frozen annotation constructor contract changed")
            kwargs["max_tokens"] = amendment["effective_max_tokens"]
            kwargs["request_timeout_seconds"] = amendment[
                "effective_request_timeout_seconds"
            ]
            super().__init__(out, models, budget, **kwargs)
            (Path(out) / "operational_amendment_receipt.json").write_text(
                json.dumps(
                    {
                        "schema": "local3_annotation_operational_receipt_v2",
                        "stage": stage,
                        "amendment_path": str(AMENDMENT),
                        "amendment_sha256": digest(AMENDMENT),
                        "adapter_sha256": digest(__file__),
                        "root_release_path": str(release_path),
                        "root_release_sha256": digest(release_path),
                        "original_max_tokens": amendment["original_max_tokens"],
                        "effective_max_tokens": self.max_tokens,
                        "original_request_timeout_seconds": amendment[
                            "original_request_timeout_seconds"
                        ],
                        "effective_request_timeout_seconds": self.request_timeout_seconds,
                        "stage_cap_usd": budget,
                        "payload_recount_sha256": digest(amendment["payload_recount"]),
                        "scope": "Only output allowance, transport timeout and approved stage cap changed. Complete messages, schema, batch membership, reasoning, model identities, retries and concurrency are the frozen originals.",
                    },
                    indent=2,
                )
                + "\n"
            )

        async def complete(self, model, messages, schema, tag, **kwargs):
            if (
                model not in amendment["models"]
                or set(tag) != {"stage", "source_problem", "chunk"}
                or tag["stage"] != stage_spec["request_stage"]
            ):
                raise ValueError("frozen annotation job identity changed")
            job = jobs[tag["source_problem"], tag["chunk"]]
            if payload_digest(messages, schema) != job["payload_sha256"]:
                raise ValueError("complete frozen annotation payload changed")
            # Live prices may decrease. A price increase requires another read-only
            # reconciliation rather than silently exceeding the reviewed envelope.
            actual = self.models[model]["pricing"]
            approved = recount["models"][model]["pricing"]
            for key in ["prompt", "completion", "request"]:
                if float(actual.get(key, 0)) > float(approved.get(key, 0)):
                    raise ValueError(
                        "catalog price exceeds reviewed annotation envelope"
                    )
            return await super().complete(model, messages, schema, tag, **kwargs)

    return AllowanceChat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sentence", "clause"])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    amendment = load_amendment()
    stage_spec = amendment["stages"][args.stage]
    if args.verify_only:
        print(
            json.dumps(
                {
                    "verified_amendment": str(AMENDMENT),
                    "stage": args.stage,
                    "stage_spec": stage_spec,
                    "paid_calls": 0,
                },
                indent=2,
            )
        )
        return
    verify_release(args.release, args.stage, amendment)
    module_name = (
        "local3_annotate_units"
        if args.stage == "sentence"
        else "local3_secondary_clauses"
    )
    module = importlib.import_module("probity_benchmark." + module_name)
    if args.stage == "clause":
        module.verify_secondary_lock()
    run_args = SimpleNamespace(
        dataset=Path(amendment["dataset"]),
        episodes=[Path(amendment["episodes"])],
        out=Path(stage_spec["out"]),
        global_freeze=Path(amendment["global_freeze"]),
        granularities=["sentence"],
        max_units=48,
        budget=stage_spec["budget_usd"],
        concurrency=stage_spec["concurrency"],
        reuse_receipts=[],
        plan_only=False,
    )
    original_class = module.BudgetedChat
    module.BudgetedChat = make_allowance_client(
        original_class, amendment, args.release, args.stage
    )
    try:
        asyncio.run(module.run(run_args))
    finally:
        module.BudgetedChat = original_class


if __name__ == "__main__":
    main()
