"""Explicit recovery of frozen private plans across catalog price changes.

The frozen API and preparation driver remain unchanged. Previously completed
jobs retain their original request and response, including the old routing
price ceiling. Only the catalog-derived price ceiling may differ from the
request the driver would build today. Missing jobs use the ordinary API.
"""

import copy
import json
from pathlib import Path

import jsonschema

from .api import BudgetedChat, assert_no_exposed_truncation, strict_json
from .preflight import sha256


def without_price_ceiling(request):
    result = copy.deepcopy(request)
    provider = result.get("provider")
    if not isinstance(provider, dict) or "max_price" not in provider:
        raise ValueError("explicit provider price ceiling required")
    provider.pop("max_price")
    return result


def validate_receipt_hash(record):
    if (
        sha256(json.dumps(record["request"], sort_keys=True))
        != record["request_sha256"]
    ):
        raise ValueError("saved request hash/body mismatch")
    if record["model"] != record["request"]["model"]:
        raise ValueError("saved model/request mismatch")


class OriginalPreparationChat(BudgetedChat):
    """First valid source receipt wins; mismatched inference contracts fail closed."""

    def __init__(self, *args, **kwargs):
        sources = list(kwargs.get("reuse_receipts", ()))
        self.original_jobs = {}
        self.blocked_contracts = {}
        provenance = []
        duplicates = []
        for source in sources:
            path = Path(source)
            data = path.read_text()
            provenance.append({"path": str(path), "sha256": sha256(data)})
            for line_number, line in enumerate(data.splitlines(), 1):
                record = strict_json(line)
                validate_receipt_hash(record)
                key = json.dumps(record["tag"], sort_keys=True)
                if record.get("valid") and record.get("finish_reason") == "stop":
                    if key in self.original_jobs:
                        duplicates.append(
                            {
                                "source": str(path),
                                "line": line_number,
                                "tag": record["tag"],
                                "request_sha256": record["request_sha256"],
                                "reason": "earlier valid source receipt is canonical",
                            }
                        )
                    else:
                        self.original_jobs[key] = (str(path), record)
                elif record.get("finish_reason") == "error":
                    payload = json.loads(record.get("response_body", "{}"))
                    if any(
                        "DataInspectionFailed"
                        in choice.get("error", {}).get("message", "")
                        for choice in payload.get("choices", [])
                    ):
                        identity = json.dumps(
                            without_price_ceiling(record["request"]), sort_keys=True
                        )
                        self.blocked_contracts[identity] = (str(path), record)
        super().__init__(*args, **kwargs)
        (self.out / "recovery_manifest.json").write_text(
            json.dumps(
                {
                    "schema": "original_preparation_recovery_v1",
                    "adapter_sha256": sha256(Path(__file__).read_text()),
                    "frozen_api_sha256": sha256(
                        Path(__file__).with_name("api.py").read_text()
                    ),
                    "ordered_sources": provenance,
                    "canonical_complete_jobs": len(self.original_jobs),
                    "superseded_valid_receipts": duplicates,
                    "allowed_request_difference": "provider.max_price only",
                    "policy": "First valid receipt in ordered sources wins without reading its answer for selection. Original request, response and hash are retained. No complete job is regenerated. Missing jobs use the frozen API with current catalog ceilings.",
                },
                indent=2,
            )
            + "\n"
        )

    def current_request(self, model, messages, schema, schema_name):
        spec = self.models[model]
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "reasoning": self.reasoning_by_model.get(model, self.reasoning),
            "usage": {"include": True},
            "provider": {
                "require_parameters": True,
                "max_price": {
                    key: float(spec["pricing"].get(key, 0))
                    * (1 if key == "request" else 1e6)
                    for key in ("prompt", "completion", "request")
                },
            },
        }
        if self.structured_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            }
        if "temperature" in spec["supported_parameters"]:
            body["temperature"] = 0
        return body

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
        if tag.get("stage") != "reasoned_preparation":
            raise ValueError("this adapter is restricted to private preparation jobs")
        body = self.current_request(model, messages, schema, schema_name)
        identity = json.dumps(without_price_ceiling(body), sort_keys=True)
        if identity in self.blocked_contracts:
            raise RuntimeError(
                "Previously blocked by provider content inspection; not retried"
            )
        key = json.dumps(tag, sort_keys=True)
        if key not in self.original_jobs:
            return await super().complete(
                model,
                messages,
                schema,
                tag,
                schema_name=schema_name,
                _attempt=_attempt,
                response_validator=response_validator,
            )
        source, prior = self.original_jobs[key]
        if without_price_ceiling(prior["request"]) != without_price_ceiling(body):
            raise ValueError(
                "canonical completed job has a different inference contract"
            )
        payload = strict_json(prior["response_body"])
        assert_no_exposed_truncation(payload)
        if (
            len(payload["choices"]) != 1
            or payload["choices"][0].get("finish_reason") != "stop"
        ):
            raise ValueError("canonical receipt lacks a complete single response")
        usage = payload["usage"]
        if (
            not all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens")
            )
            or usage["completion_tokens"] >= self.max_tokens
        ):
            raise ValueError("canonical receipt has invalid or capped token accounting")
        if prior["usage"] != usage:
            raise ValueError("canonical receipt usage differs from original response")
        parsed = strict_json(payload["choices"][0]["message"]["content"])
        jsonschema.validate(parsed, schema)
        if response_validator is not None:
            response_validator(parsed)
        record = {
            **prior,
            "parsed": parsed,
            "reused_from": source,
            "original_cost_usd": prior["cost_usd"],
            "cost_usd": 0.0,
            "cost_basis": "previously_accounted_original_preparation",
            "recovery_contract": {
                "policy": "original_receipt_retained_across_catalog_price_change",
                "current_catalog_request_sha256": sha256(
                    json.dumps(body, sort_keys=True)
                ),
                "current_catalog_price_ceiling": body["provider"]["max_price"],
                "price_ceiling_changed": prior["request"]["provider"]["max_price"]
                != body["provider"]["max_price"],
            },
        }
        self.records.append(record)
        with (self.out / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def main():
    # The module itself remains byte-identical to the global freeze. Only this
    # explicit operational entrypoint supplies the audited recovery adapter.
    from . import environment3_reasoned_prepare as driver

    driver.BudgetedChat = OriginalPreparationChat
    driver.main()


if __name__ == "__main__":
    main()
