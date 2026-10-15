"""Audit complete provider-returned reasoning without modifying its bytes."""

import json

from .preflight import sha256


def audit_reasoning_receipt(receipt):
    if (
        not receipt["valid"]
        or receipt["finish_reason"] != "stop"
        or receipt["request"]["reasoning"] != {"enabled": True}
    ):
        raise ValueError("public thinking receipt is incomplete or disabled")
    payload = json.loads(receipt["response_body"])
    choice = payload["choices"][0]
    flags = []
    text_fields = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                here = path + [key]
                if key in ("forced", "truncated", "forced_final") and item:
                    flags.append({"path": here, "value": item})
                if key in (
                    "native_finish_reason",
                    "stop_reason",
                    "finish_reason",
                ) and item in (
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                    "truncated",
                ):
                    flags.append({"path": here, "value": item})
                if (
                    isinstance(item, str)
                    and item
                    and (
                        key in ("reasoning", "reasoning_content")
                        or (
                            "reasoning_details" in path
                            and key in ("text", "summary", "data")
                        )
                    )
                ):
                    text_fields.append(
                        {"path": here, "characters": len(item), "sha256": sha256(item)}
                    )
                visit(item, here)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + [index])

    visit(payload, [])
    if flags:
        raise ValueError(
            "provider exposes forced/truncated public reasoning: " + json.dumps(flags)
        )
    usage = receipt["usage"]
    reasoning_tokens = usage.get("completion_tokens_details", {}).get(
        "reasoning_tokens"
    )
    cap = receipt["request"]["max_tokens"]
    if usage["completion_tokens"] >= cap or (reasoning_tokens or 0) >= cap:
        raise ValueError("reported public thinking tokens reached the allowance")
    return {
        "native_finish_reason": choice.get("native_finish_reason"),
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": reasoning_tokens,
        "max_tokens": cap,
        "exposed_forced_or_truncated_flags": flags,
        "returned_reasoning_fields": text_fields,
        "returned_reasoning_available": bool(text_fields),
        "response_body_sha256": sha256(receipt["response_body"]),
        "scope": "All provider-returned fields retained verbatim in raw receipt. No claim about unexposed provider internals. Final-JSON-only probe replay omits the returned reasoning and is not the complete action context.",
    }
