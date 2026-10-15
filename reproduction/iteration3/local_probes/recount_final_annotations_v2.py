"""Read-only exact-payload audit and smoke-based cost forecast before release."""

import copy
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

from probity_benchmark.api import object_schema
from probity_benchmark.evaluate_debate_transfer import argument_mapping
from probity_benchmark.local3_annotate_units import (
    ANNOTATION_REASONING,
    UNIT_RULES,
    chunks,
    load_inputs,
)
from probity_benchmark.local3_secondary_clauses import (
    CLAUSE_RULES,
    complete_payload,
    opening_ids,
)
from probity_benchmark.local3_verify_twins import MIXED_CHECK

BASE = Path("runs/iteration3/local_probes")
OUT = Path("runs/iteration3/final/local_annotation_payload_recount_v2.json")


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def completion_forecast(prior, items):
    """Estimate complete output length without making or changing any labels."""
    old_checks = prior["parsed"]["checks"]
    old_serialized_bytes = len(encoded({"checks": old_checks}))
    tokens_per_serialized_byte = (
        prior["usage"]["completion_tokens"] / old_serialized_bytes
    )
    usual_checks, maximum_reason_checks = [], []
    for index, (_, _, unit) in enumerate(items):
        # Old verdicts are only byte-count templates; they are never predictions
        # or exported labels, and no resulting synthetic payload is submitted.
        template = copy.deepcopy(old_checks[index % len(old_checks)])
        template.update(index=index, exact_text=unit["display_text"])
        usual_checks.append(template)
        long_reason = copy.deepcopy(template)
        long_reason["reason"] = "x" * 600
        maximum_reason_checks.append(long_reason)
    expected = len(encoded({"checks": usual_checks})) * tokens_per_serialized_byte
    reason_ceiling = (
        len(encoded({"checks": maximum_reason_checks})) * tokens_per_serialized_byte
    )
    unit_scaled = len(items) * prior["usage"]["completion_tokens"] / len(old_checks)
    warning_projection = max(expected * 1.25, reason_ceiling)
    return {
        "allowance_tokens": 16384,
        "old_smoke_tokens_per_unit": prior["usage"]["completion_tokens"]
        / len(old_checks),
        "unit_scaled_expected_tokens": unit_scaled,
        "exact_text_adjusted_response_json_bytes": len(
            encoded({"checks": usual_checks})
        ),
        "exact_text_adjusted_expected_tokens": expected,
        "exact_text_adjusted_with_25_percent_margin_tokens": expected * 1.25,
        "all_reasons_at_schema_600_character_max_tokens": reason_ceiling,
        "warning_projection_tokens": warning_projection,
        "warning_projection_fraction_of_allowance": warning_projection / 16384,
        "headroom_warning": warning_projection >= 0.8 * 16384,
        "exceeds_allowance_projection": warning_projection >= 16384,
        "method_limit": "Byte-based projection calibrated on complete old-development JSON, adjusted for every exact current unit string. The 600-character scenario covers each reason field at its schema maximum; prose tokenization, hidden reasoning and formatting can differ. This is a warning forecast, not a truncation guarantee.",
    }


def payload_for(granularity, paragraphs, items):
    if granularity == "clause":
        return complete_payload(paragraphs, items)[0]
    return {
        "SOURCE_STORY_PARAGRAPHS": [
            {"id": i, "text": paragraph} for i, paragraph in enumerate(paragraphs)
        ],
        "EXACT_SUBSTRINGS_TO_ASSESS": [
            {
                "index": i,
                "text": unit["display_text"],
                "COMPLETE_ARGUMENT": argument_mapping(row["response"])[0],
            }
            for i, (row, _, unit) in enumerate(items)
        ],
    }


def main():
    if OUT.exists():
        raise ValueError("immutable new payload audit required")
    handoff = json.loads((BASE / "final_handoff_v3/manifest.json").read_text())
    import probity_benchmark.local3_final_runner as runner

    runner.HANDOFF = BASE / "final_handoff_v3"
    episodes, _ = runner.verify(handoff)
    smoke_path = BASE / "frozen_annotation_smoke_v1/requests.jsonl"
    smoke = {
        (row["model"], row["tag"]["granularity"]): row
        for row in map(json.loads, smoke_path.read_text().splitlines())
    }
    catalog = json.load(
        urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30)
    )
    models = {
        row["id"]: row for row in catalog["data"] if row["id"] in ANNOTATION_REASONING
    }
    if set(models) != set(ANNOTATION_REASONING):
        raise ValueError("both frozen annotator models must be available")
    result = {
        "schema": "local3_exact_payload_forecast_v2",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_sha256": sha(Path(handoff["dataset"]).read_bytes()),
        "global_freeze_sha256": sha(Path(handoff["global_freeze"]).read_bytes()),
        "smoke_receipts_sha256": sha(smoke_path.read_bytes()),
        "forecast_limit": "Payload bytes, schema, job tags and API byte-per-token reservation bounds are exact. Expected token usage/cost is only a planning heuristic from one old-development smoke call per model/granularity, not a guaranteed price or output-length bound. Live shared and per-stage budget guards remain mandatory.",
        "models": {
            model: {
                "pricing": spec["pricing"],
                "context_length": spec["top_provider"].get("context_length")
                or spec["context_length"],
            }
            for model, spec in models.items()
        },
        "stages": {},
        "completion_warning_threshold_fraction": 0.8,
    }
    for granularity, budget, concurrency in [("sentence", 8, 3), ("clause", 4, 2)]:
        rows, sources = load_inputs(
            Path(handoff["dataset"]), [Path(handoff["episodes"])], [granularity]
        )
        if granularity == "clause":
            selected = opening_ids(episodes)
            rows = [row for row in rows if row["id"] in selected]
            if len(rows) != 48:
                raise ValueError("all fixed wrong-advocate openings required")
        jobs = chunks(rows, 48)
        stage = {
            "rows": len(rows),
            "units": sum(len(row["units"]) for row in rows),
            "chunks": len(jobs),
            "expected_calls": 2 * len(jobs),
            "declared_budget_usd": budget,
            "concurrency": concurrency,
            "jobs": [],
            "expected_cost_usd": 0,
            "complete_input_rows_jsonl_bytes": sum(
                len(encoded(row)) + 1 for row in rows
            ),
            "projected_api_receipt_bytes": 0,
            "projected_annotation_check_and_evidence_bytes": 0,
            "annotation_all_source_evidence_bytes_upper_scenario": 0,
        }
        for problem, chunk_index, items in jobs:
            payload = payload_for(granularity, sources[problem].split("\n\n"), items)
            messages = [
                {
                    "role": "system",
                    "content": (
                        UNIT_RULES if granularity == "sentence" else CLAUSE_RULES
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            schema = object_schema(
                {
                    "checks": {
                        "type": "array",
                        "items": MIXED_CHECK,
                        "minItems": len(items),
                        "maxItems": len(items),
                    }
                }
            )
            input_bytes = len(encoded([messages, schema]))
            bound = input_bytes + 2048
            entry = {
                "source_problem": problem,
                "chunk": chunk_index,
                "units": len(items),
                "exact_payload_bytes": input_bytes,
                "payload_sha256": sha(encoded([messages, schema])),
                "api_prompt_token_reservation_bound": bound,
                "exact_unit_text_bytes": sum(
                    len(unit["display_text"].encode()) for _, _, unit in items
                ),
                "largest_exact_unit_text_bytes": max(
                    len(unit["display_text"].encode()) for _, _, unit in items
                ),
                "models": {},
            }
            for model, spec in models.items():
                prior = smoke[model, granularity]
                prior_schema = prior["request"]["response_format"]["json_schema"][
                    "schema"
                ]
                prior_bytes = len(encoded([prior["request"]["messages"], prior_schema]))
                expected_input = (
                    input_bytes * prior["usage"]["prompt_tokens"] / prior_bytes
                )
                output_forecast = completion_forecast(prior, items)
                expected_output = output_forecast["exact_text_adjusted_expected_tokens"]
                pricing = {
                    key: float(spec["pricing"].get(key, 0))
                    for key in ["prompt", "completion", "request"]
                }
                cost = (
                    expected_input * pricing["prompt"]
                    + expected_output * pricing["completion"]
                    + pricing["request"]
                )
                context = result["models"][model]["context_length"]
                if bound + 16384 > context:
                    raise ValueError("frozen complete payload exceeds API context gate")
                entry["models"][model] = {
                    "expected_input_tokens": expected_input,
                    "expected_output_tokens": expected_output,
                    "expected_cost_usd": cost,
                    "api_per_call_reservation_usd": bound * pricing["prompt"]
                    + 16384 * pricing["completion"]
                    + pricing["request"],
                    "context_reservation_headroom_tokens": context - bound - 16384,
                    "completion_forecast": output_forecast,
                }
                stage["expected_cost_usd"] += cost
                output_bytes = output_forecast[
                    "exact_text_adjusted_response_json_bytes"
                ]
                # Raw response text plus parsed JSON are both retained. Three
                # copies cover ordinary JSON escaping/metadata conservatively.
                stage["projected_api_receipt_bytes"] += (
                    input_bytes + 3 * output_bytes + 2048
                )
                old_evidence_count = sum(
                    len(check["story_paragraph_ids"])
                    for check in prior["parsed"]["checks"]
                ) / len(prior["parsed"]["checks"])
                paragraphs = sources[problem].split("\n\n")
                average_paragraph_bytes = sum(
                    len(encoded(paragraph)) for paragraph in paragraphs
                ) / len(paragraphs)
                stage["projected_annotation_check_and_evidence_bytes"] += (
                    output_bytes
                    + len(items) * old_evidence_count * average_paragraph_bytes
                )
                stage[
                    "annotation_all_source_evidence_bytes_upper_scenario"
                ] += output_bytes + len(items) * len(encoded(paragraphs))
            stage["jobs"].append(entry)
        stage["expected_cost_with_25_percent_margin_usd"] = (
            stage["expected_cost_usd"] * 1.25
        )
        stage["largest_prompt_token_reservation_bound"] = max(
            job["api_prompt_token_reservation_bound"] for job in stage["jobs"]
        )
        stage["completion_headroom_warnings"] = [
            {
                "source_problem": job["source_problem"],
                "chunk": job["chunk"],
                "model": model,
                "units": job["units"],
                "forecast": values["completion_forecast"],
            }
            for job in stage["jobs"]
            for model, values in job["models"].items()
            if values["completion_forecast"]["headroom_warning"]
        ]
        stage["maximum_completion_warning_projection_tokens"] = max(
            values["completion_forecast"]["warning_projection_tokens"]
            for job in stage["jobs"]
            for values in job["models"].values()
        )
        unit_metadata_bytes = sum(
            len(encoded(unit)) + 512 for row in rows for unit in row["units"]
        )
        stage["projected_annotation_text_artifact_bytes"] = int(
            stage["complete_input_rows_jsonl_bytes"]
            + stage["projected_api_receipt_bytes"]
            + 2
            * (
                unit_metadata_bytes
                + stage["projected_annotation_check_and_evidence_bytes"]
            )
        )
        stage["all_source_evidence_text_storage_scenario_bytes"] = int(
            stage["complete_input_rows_jsonl_bytes"]
            + stage["projected_api_receipt_bytes"]
            + 2
            * (
                unit_metadata_bytes
                + stage["annotation_all_source_evidence_bytes_upper_scenario"]
            )
        )
        result["stages"][granularity] = stage
    retained_segments = (
        result["stages"]["sentence"]["units"] + result["stages"]["clause"]["units"]
    )
    result["retention_storage_plan"] = {
        "retained_sentence_plus_opening_clause_segments": retained_segments,
        "exact_segment_tensor_bytes_per_segment": 6 * 5120 * 4 * 2,
        "response_argument_dyl_tensor_bytes_all_288_turns": 288
        * 6
        * 5120
        * (4 + 4 + 2 + 2),
        "raw_retained_tensor_bytes_before_metadata_and_compression": retained_segments
        * 6
        * 5120
        * 4
        * 2
        + 288 * 6 * 5120 * (4 + 4 + 2 + 2),
        "projected_annotation_text_artifact_bytes": sum(
            stage["projected_annotation_text_artifact_bytes"]
            for stage in result["stages"].values()
        ),
        "all_source_evidence_text_storage_scenario_bytes": sum(
            stage["all_source_evidence_text_storage_scenario_bytes"]
            for stage in result["stages"].values()
        ),
        "storage_limit": "Tensor byte count is exact for the frozen cache shapes/dtypes, excluding offsets/text/serialization overhead. Annotation text size is a planning estimate; the second scenario lets every check cite the entire source. Final all-unit score bundles, downstream judge artifacts and scientific plots require additional space. No archive transfer until remote xz preview, actual compressed size and SHA verification pass against then-current free disk.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(OUT),
                "retention_storage_plan": result["retention_storage_plan"],
                "stages": {
                    name: {
                        **{
                            key: value
                            for key, value in stage.items()
                            if key not in {"jobs", "completion_headroom_warnings"}
                        },
                        "completion_headroom_warning_count": len(
                            stage["completion_headroom_warnings"]
                        ),
                    }
                    for name, stage in result["stages"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
