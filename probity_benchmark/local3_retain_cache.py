"""Lossless, explicit retention of local fit and fixed-unit evaluation features.

Training keeps all labelled-span tensors and full frozen-reference segment rows.
Final caches keep all sentence units plus clauses in the 48 fixed openings.
All original texts, offsets, response/DYL tensors, layer precision and labels stay.
"""

import argparse
import copy
import hashlib
import io
import lzma
import zlib
import json
from pathlib import Path

import torch

from .local3_bundle import read_cache
from .local3_secondary_clauses import opening_ids
from .preflight import sha256


def retain(cache, policy, complete_segment_ids):
    row_id = cache["row"]["id"]
    selected = [
        i
        for i, segment in enumerate(cache["segments"])
        if row_id in complete_segment_ids
        or (policy == "final" and segment["kind"] == "sentence")
    ]
    if policy not in {"training", "final"}:
        raise ValueError("explicit training/final retention policy required")
    output = copy.copy(cache)
    output["token_features"] = None
    output["segments"] = [cache["segments"][i] for i in selected]
    for key in ["segment_mean", "segment_last"]:
        value = cache[key]
        if value is None or value.ndim != 3 or value.shape[1] != len(cache["segments"]):
            raise ValueError("complete segment feature source required")
        output[key] = value.index_select(
            1, torch.tensor(selected, dtype=torch.long)
        ).clone()
        if not torch.equal(output[key], value[:, selected]):
            raise ValueError("retained segment values changed")
    lengths = cache["segment_lengths"]
    output["segment_lengths"] = [lengths[i] for i in selected]
    output["segment_retention"] = {
        "source_segment_indices": selected,
        "all_original_segments": cache["segments"],
        "all_original_segment_lengths": lengths,
        "policy": policy,
        "all_sentence_units_retained": policy == "final"
        or row_id in complete_segment_ids,
        "all_clause_units_retained": row_id in complete_segment_ids,
        "omitted_full_token_tensor": cache.get("token_features") is not None,
        "all_other_tensors_exact": True,
    }
    return output


def verify_retained(original, retained):
    indices = retained["segment_retention"]["source_segment_indices"]
    for key, value in original.items():
        if key == "token_features":
            if retained[key] is not None:
                raise ValueError("full token cache unexpectedly retained")
        elif key in {"segment_mean", "segment_last"}:
            if not torch.equal(retained[key], value[:, indices]):
                raise ValueError("pooled segment archive changed exact values")
        elif key in {"segments", "segment_lengths"}:
            if retained[key] != [value[i] for i in indices]:
                raise ValueError("segment coordinate metadata changed")
        elif isinstance(value, torch.Tensor):
            if not torch.equal(retained[key], value):
                raise ValueError("nonsegment pooled tensor changed")
        elif retained[key] != value:
            raise ValueError("complete original record changed")


def tensor_record(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_receipts(source, retained):
    indices = retained["segment_retention"]["source_segment_indices"]
    records = {}
    for key, value in source.items():
        if not isinstance(value, torch.Tensor):
            continue
        target = retained.get(key)
        entry = {
            "source": tensor_record(value),
            "retained": (
                tensor_record(target) if isinstance(target, torch.Tensor) else None
            ),
        }
        if key in {"segment_mean", "segment_last"}:
            entry["source_retained_projection"] = tensor_record(value[:, indices])
            if entry["source_retained_projection"] != entry["retained"]:
                raise ValueError("retained projection tensor hash changed")
        elif key != "token_features" and entry["source"] != entry["retained"]:
            raise ValueError("retained full tensor hash changed")
        records[key] = entry
    return records


def preview(paths, policy, complete_ids, per_stratum):
    strata = {"full_segment_rows": [], "other_rows": []}
    for path in paths:
        key = "full_segment_rows" if path.stem in complete_ids else "other_rows"
        strata[key].append(path)
    results = {}
    for name, members in strata.items():
        if not members:
            continue
        count = min(per_stratum, len(members))
        indices = [i * (len(members) - 1) // max(1, count - 1) for i in range(count)]
        rows = []
        for i in indices:
            source = torch.load(members[i], map_location="cpu", weights_only=True)
            retained = retain(source, policy, complete_ids)
            verify_retained(source, retained)
            buffer = io.BytesIO()
            torch.save(retained, buffer)
            raw = buffer.getvalue()
            rows.append(
                {
                    "id": source["row"]["id"],
                    "torch_bytes": len(raw),
                    "zlib_level1_bytes": len(zlib.compress(raw, 1)),
                    "xz_level6_bytes": len(lzma.compress(raw, preset=6)),
                    "tensor_hashes": tensor_receipts(source, retained),
                }
            )
        results[name] = {
            "population_rows": len(members),
            "sampled_rows": len(rows),
            "estimated_torch_bytes": int(
                sum(r["torch_bytes"] for r in rows) / len(rows) * len(members)
            ),
            "estimated_compressed_bytes": int(
                sum(r["zlib_level1_bytes"] for r in rows) / len(rows) * len(members)
            ),
            "estimated_xz_bytes": int(
                sum(r["xz_level6_bytes"] for r in rows) / len(rows) * len(members)
            ),
            "samples": rows,
        }
    return {
        "policy": policy,
        "method": "Evenly spaced deterministic samples within reference/full-segment and other row strata; level-1 zlib and level-6 xz per-file estimates. Final tar compression can differ. No source file modified.",
        "estimated_torch_bytes": sum(
            r["estimated_torch_bytes"] for r in results.values()
        ),
        "estimated_compressed_bytes": sum(
            r["estimated_compressed_bytes"] for r in results.values()
        ),
        "estimated_xz_bytes": sum(r["estimated_xz_bytes"] for r in results.values()),
        "strata": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=["training", "final"], required=True)
    parser.add_argument("--reference-bundles", type=Path, nargs="*", default=[])
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--export-receipt", type=Path)
    parser.add_argument("--preview-per-stratum", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable archive output required")
    paths, manifest = read_cache(args.features)
    complete_ids = set()
    dependencies = {}
    if args.policy == "training":
        if not args.reference_bundles:
            raise ValueError(
                "frozen reference population required for training retention"
            )
        for path in args.reference_bundles:
            bundle = json.loads(path.read_text())
            dependencies[str(path)] = sha256(path.read_text())
            matching = [
                reference
                for participant, reference in bundle["references"].items()
                if participant.lower() == manifest["model"].lower()
            ]
            if len(matching) != 1:
                raise ValueError(
                    "one model-matched frozen reference population required"
                )
            complete_ids.update(matching[0]["rows"])
    else:
        if not args.episodes or not args.export_receipt:
            raise ValueError(
                "final episodes and complete score export receipt required"
            )
        receipt = json.loads(args.export_receipt.read_text())
        if (
            receipt.get("status") != "all_fixed_units_exported"
            or receipt.get("turns") != 288
        ):
            raise ValueError(
                "all frozen scores must be exported before final retention"
            )
        episodes = [
            json.loads(path.read_text())
            for path in sorted(args.episodes.glob("episode_*.json"))
        ]
        if len(episodes) != 48 or sum(len(e["turns"]) for e in episodes) != 288:
            raise ValueError("complete final episodes required")
        complete_ids = opening_ids(episodes)
        dependencies[str(args.export_receipt)] = sha256(args.export_receipt.read_text())
    args.out.mkdir(parents=True)
    if args.preview_per_stratum:
        if args.preview_per_stratum < 1:
            raise ValueError("positive preview size required")
        result = preview(paths, args.policy, complete_ids, args.preview_per_stratum)
        result.update(
            source_manifest_sha256=sha256(
                (args.features / "manifest.json").read_text()
            ),
            retention_source_sha256=sha256(Path(__file__).read_text()),
        )
        (args.out / "preview.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in result.items() if k != "strata"}, indent=2),
            flush=True,
        )
        return
    receipts, tensor_bytes = [], 0
    for path in paths:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        output = retain(cache, args.policy, complete_ids)
        dest = args.out / path.name
        torch.save(output, dest)
        verified = torch.load(dest, map_location="cpu", weights_only=True)
        verify_retained(cache, verified)
        tensor_bytes += sum(
            v.numel() * v.element_size()
            for v in output.values()
            if isinstance(v, torch.Tensor)
        )
        receipts.append(
            {
                "id": cache["row"]["id"],
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "archive_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "source_segments": len(cache["segments"]),
                "retained_segments": len(output["segments"]),
                "retained_indices": output["segment_retention"][
                    "source_segment_indices"
                ],
                "tensor_hashes": tensor_receipts(cache, verified),
                "bytes": dest.stat().st_size,
            }
        )
    source_manifest = (args.features / "manifest.json").read_text()
    (args.out / "source_manifest.json").write_text(source_manifest)
    manifest.update(
        {
            "archive_schema": "local3_exact_pooled_retention_v1",
            "source_manifest_sha256": sha256(source_manifest),
            "cache_complete_response_tokens": False,
            "cache_all_sentence_features": args.policy == "final",
            "cache_all_clause_features": False,
            "retention_policy": args.policy,
            "full_segment_response_ids": sorted(complete_ids),
            "dependencies_sha256": dependencies,
            "retention_source_sha256": sha256(Path(__file__).read_text()),
            "archive_scope": "All six original feature layers, original dtypes and every labelled-span/response/argument/DYL tensor retained exactly. Training retains complete sentence/clause tensors for frozen reference rows. Final retains all sentence tensors and all clauses of fixed wrong-advocate openings. All omitted unit coordinates and original full texts retained; already exported frozen all-unit scores remain available. Original manifest retained separately for exact normalizer provenance.",
        }
    )
    summary = json.loads((args.features / "summary.json").read_text())
    summary.update(
        {
            "bytes": sum(r["bytes"] for r in receipts),
            "retained_tensor_bytes": tensor_bytes,
            "source_segments": sum(r["source_segments"] for r in receipts),
            "retained_segments": sum(r["retained_segments"] for r in receipts),
        }
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "receipts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in receipts)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
