"""Public-safe reproduction inventory; no environment or machine identifiers."""

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["gpu", "vps"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-image")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable runtime snapshot required")
    packages = sorted(
        {
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
        },
        key=lambda item: (item[0].lower(), item[1]),
    )
    gpus = []
    if args.kind == "gpu":
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
        )
        for name, driver, memory in csv.reader(io.StringIO(result.stdout)):
            gpus.append(
                {
                    "model": name.strip(),
                    "driver_version": driver.strip(),
                    "vram_mib": int(memory.strip()),
                }
            )
    original_manifests = {}
    if args.kind == "gpu":
        for short in ["qwen35", "qwen38"]:
            path = Path(f"runs/iteration3/final/features_{short}/manifest.json")
            original_manifests[str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    output = {
        "schema": "local3_public_runtime_packages_v1",
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "role": args.kind,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build_string": sys.version,
        },
        "packages": [{"name": name, "version": version} for name, version in packages],
        "gpu_model_driver_vram_only": gpus,
        "pinned_base_image": args.base_image,
        "original_inference_manifest_sha256": original_manifests,
        "scope": "Post-run package inventory. Only distribution names/versions, Python version, GPU model/driver/VRAM, and a known pinned image are recorded. No environment variables, hostnames, IPs, credentials, package direct URLs or filesystem installation locations are collected. Inference settings remain those recorded in the original bound manifests; current process defaults are not represented as inference settings.",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "path": str(args.out),
                "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "bytes": args.out.stat().st_size,
                "packages": len(packages),
                "gpus": gpus,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
