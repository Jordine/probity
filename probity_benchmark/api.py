"""Budgeted structured chat for diagnostic episodes, with complete receipts."""

import asyncio
import json
import math
import os
from pathlib import Path
import time
import urllib.request

import aiohttp
import jsonschema

from .preflight import sha256
from .budget import PilotBudget


def strict_json(text):
    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("duplicate JSON keys")
        return obj

    return json.loads(text, object_pairs_hook=unique)


class BudgetedChat:
    def __init__(
        self,
        out: Path,
        models: list[str],
        budget: float,
        max_tokens=8192,
        *,
        reuse_receipts=(),
        schema_retries=0,
        concurrency=4,
        reasoning=None,
    ):
        if out.exists() or not math.isfinite(budget) or budget <= 0:
            raise ValueError("new output directory and finite positive budget required")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY unavailable")
        catalog = json.load(
            urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30)
        )
        all_models = {m["id"]: m for m in catalog["data"]}
        self.models = {m: all_models[m] for m in models}
        for m in self.models.values():
            if "structured_outputs" not in m["supported_parameters"]:
                raise ValueError("structured output unsupported")
            for key in ["prompt", "completion", "request"]:
                value = float(m["pricing"].get(key, 0))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid pricing")
        out.mkdir(parents=True)
        self.out, self.budget, self.max_tokens = out, budget, max_tokens
        self.spent = self.reserved = 0.0
        self.stopped = False
        self.lock = asyncio.Lock()
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("positive integer concurrency required")
        self.sem = asyncio.Semaphore(concurrency)
        self.records = []
        self.schema_retries = schema_retries
        self.reasoning = {"enabled": False} if reasoning is None else dict(reasoning)
        self.reusable = {}
        self.unavailable = {}
        for source in reuse_receipts:
            for line in Path(source).read_text().splitlines():
                record = strict_json(line)
                if record["valid"] and record.get("finish_reason") == "stop":
                    self.reusable[record["request_sha256"]] = (str(source), record)
                elif record.get("finish_reason") == "error":
                    payload = json.loads(record.get("response_body", "{}"))
                    for choice in payload.get("choices", []):
                        if "DataInspectionFailed" in choice.get("error", {}).get(
                            "message", ""
                        ):
                            # Preserve a provider's content block as missing data.
                            # Do not retry or change routing to get around it.
                            self.unavailable[record["request_sha256"]] = (
                                str(source),
                                record,
                            )
        self.global_budget = (
            PilotBudget(os.environ["PROBITY_PILOT_LEDGER"])
            if os.environ.get("PROBITY_PILOT_LEDGER")
            else None
        )
        self.budget_key = "api:" + str(out.resolve())
        if self.global_budget is not None:
            self.global_budget.reserve(self.budget_key, budget, "api")
        (out / "api_manifest.json").write_text(
            json.dumps(
                {
                    "models": self.models,
                    "budget": budget,
                    "max_tokens": max_tokens,
                    "reasoning": self.reasoning,
                    "source_sha256": sha256(Path(__file__).read_text()),
                },
                indent=2,
            )
            + "\n"
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),
            headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        )
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        if self.global_budget is not None:
            self.global_budget.settle(
                self.budget_key,
                self.spent + self.reserved,
                "Recorded charges plus full reservations for unresolved in-flight calls; see requests.jsonl.",
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
        spec = self.models[model]
        price = {
            k: float(spec["pricing"].get(k, 0))
            for k in ["prompt", "completion", "request"]
        }
        # Include schema/tool overhead in both the context and cost reservation.
        bound = len(json.dumps([messages, schema], ensure_ascii=False).encode()) + 2048
        context = spec["top_provider"].get("context_length") or spec["context_length"]
        if bound + self.max_tokens > context:
            raise ValueError("full context plus allowance exceeds model limit")
        upper = (
            bound * price["prompt"]
            + self.max_tokens * price["completion"]
            + price["request"]
        )
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning,
            "usage": {"include": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            "provider": {
                "require_parameters": True,
                "max_price": {
                    "prompt": price["prompt"] * 1e6,
                    "completion": price["completion"] * 1e6,
                    "request": price["request"],
                },
            },
        }
        if "temperature" in spec["supported_parameters"]:
            body["temperature"] = 0
        request_hash = sha256(json.dumps(body, sort_keys=True))
        if request_hash in self.unavailable:
            source, prior = self.unavailable[request_hash]
            record = {
                **prior,
                "tag": tag,
                "reused_from": source,
                "original_cost_usd": prior["cost_usd"],
                "cost_usd": 0.0,
                "cost_basis": "previously_accounted_provider_block",
            }
            self.records.append(record)
            with (self.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            raise RuntimeError(
                "Previously blocked by provider content inspection; not retried"
            )
        if request_hash in self.reusable:
            source, prior = self.reusable[request_hash]
            if prior["request"] != body:
                raise ValueError("cached request hash/body mismatch")
            parsed = strict_json(
                json.loads(prior["response_body"])["choices"][0]["message"]["content"]
            )
            jsonschema.validate(parsed, schema)
            if response_validator is not None:
                response_validator(parsed)
            record = {
                **prior,
                "tag": tag,
                "parsed": parsed,
                "reused_from": source,
                "original_cost_usd": prior["cost_usd"],
                "cost_usd": 0.0,
                "cost_basis": "previously_accounted_receipt",
            }
            self.records.append(record)
            with (self.out / "requests.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return record
        async with self.sem:
            async with self.lock:
                if self.stopped or self.spent + self.reserved + upper > self.budget:
                    self.stopped = True
                    raise RuntimeError(
                        "pilot stopped or budget reservation unavailable"
                    )
                self.reserved += upper
            record = {
                "tag": tag,
                "model": model,
                "request": body,
                "request_sha256": sha256(json.dumps(body, sort_keys=True)),
                "reserved_usd": upper,
                "valid": False,
                "attempt": _attempt,
            }
            start = time.monotonic()
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions", json=body
                ) as response:
                    raw = await response.text()
                    record.update(http_status=response.status, response_body=raw)
                    payload = json.loads(raw)
                    if response.status != 200 or "error" in payload:
                        raise RuntimeError(f"HTTP {response.status}")
                usage = payload["usage"]
                record["usage"] = usage
                if not all(
                    type(usage.get(k)) is int and usage[k] >= 0
                    for k in ["prompt_tokens", "completion_tokens"]
                ):
                    raise ValueError("invalid token accounting")
                reported = usage.get("cost")
                cost = (
                    float(reported)
                    if reported is not None
                    else usage["prompt_tokens"] * price["prompt"]
                    + usage["completion_tokens"] * price["completion"]
                    + price["request"]
                )
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("invalid cost")
                record.update(
                    cost_usd=cost,
                    cost_basis=(
                        "provider_reported"
                        if reported is not None
                        else "catalog_estimate"
                    ),
                )
                if cost > upper or usage["prompt_tokens"] > bound:
                    raise ValueError("provider exceeded reservation")
                if len(payload["choices"]) != 1:
                    raise ValueError("expected one response")
                choice = payload["choices"][0]
                record["finish_reason"] = choice.get("finish_reason")
                if (
                    record["finish_reason"] != "stop"
                    or usage["completion_tokens"] >= self.max_tokens
                ):
                    raise ValueError("incomplete generation")
                obj = strict_json(choice["message"]["content"])
                jsonschema.validate(obj, schema)
                if response_validator is not None:
                    response_validator(obj)
                record.update(valid=True, parsed=obj)
            except Exception as exc:
                record["error"] = str(exc)
                record["error_type"] = type(exc).__name__
            record["duration_seconds"] = time.monotonic() - start
            async with self.lock:
                self.reserved -= upper
                record.setdefault("cost_usd", upper)
                record.setdefault("cost_basis", "reservation_upper_bound")
                self.spent += record["cost_usd"]
                retry_schema = (
                    not record["valid"]
                    and record.get("finish_reason") == "stop"
                    and record.get("error_type")
                    in ("JSONDecodeError", "ValidationError")
                    and _attempt < self.schema_retries
                )
                self.stopped |= not record["valid"] and not retry_schema
                self.records.append(record)
                with (self.out / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "tag": tag,
                            "finish_reason": record.get("finish_reason"),
                            "valid": record["valid"],
                            "accounted_cost_upper_usd": self.spent,
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )
        if retry_schema:
            return await self.complete(
                model,
                messages,
                schema,
                tag,
                schema_name=schema_name,
                _attempt=_attempt + 1,
                response_validator=response_validator,
            )
        if not record["valid"]:
            raise RuntimeError(f"Invalid request {tag}: {record['error']}")
        return record


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


ANSWER = {"type": "string", "enum": ["A", "B"]}
