#!/usr/bin/env python3
import argparse, json, os, sys, torch
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from transformer_lens import HookedTransformer

# ------------------------------------------------------------------ #
# local repo bootstrap
probity_root = Path(__file__).parent.parent
sys.path.insert(0, str(probity_root))
from probity.utils.qwen3_support import ensure_qwen3_transformerlens_support
ensure_qwen3_transformerlens_support()
# ------------------------------------------------------------------ #

from probity.collection.activation_store import ActivationStore
from probity.training.configs import (
    get_probe_config, get_probe_class,
    get_trainer_config, get_trainer_class
)
from probity.utils.caching import get_dataset_hash, smart_cache_activations
from probity.utils.dataset_loading import get_model_dtype
from probity.utils.eval_awareness_dataset import (
    load_eval_awareness_dataset,          # token-level  (old)
    load_snippet_avg_dataset              # snippet-level (new)
)
# ------------------------------------------------------------------ #

# ================================================================
# helper: train all probe types for ONE layer
# ================================================================
def train_all_probes_for_layer(
    layer: int,
    activation_store: ActivationStore,
    probe_types: List[str],
    args,
    model_name: str,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Dict]:
    hook_point = f"blocks.{layer}.hook_resid_pre"
    layer_results = {}

    for probe_type in probe_types:
        print(f"Training {probe_type} probe on layer {layer}")

        probe_config  = get_probe_config(
            probe_type, hidden_size, model_name, hook_point, layer, dtype
        )
        trainer_config = get_trainer_config(
            probe_type, device, args.batch_size
        )

        probe   = get_probe_class(probe_type)(probe_config).to(device)
        trainer = get_trainer_class(probe_type)(trainer_config)

        # ── prepare data ────────────────────────────────────────────────
        train_loader, val_loader = trainer.prepare_supervised_data(
            activation_store,
            "TARGET",
            # average_span = (args.sample_mode == "snippet_avg"), 
        )

        history = trainer.train(probe, train_loader, val_loader)

        # ── save probe ──────────────────────────────────────────────────
        save_dir  = Path(args.probe_save_dir) / probe_type
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"layer_{layer}_probe.pt"
        probe.save(str(save_path))

        layer_results[probe_type] = {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss":   history.get("val_loss", [None])[-1],
            "save_path": str(save_path),
        }
        print(f"Saved {probe_type} probe for layer {layer} → {save_path}")

        del probe
        torch.cuda.empty_cache()

    return layer_results


# ================================================================
# dataset discovery helpers (unchanged)
# ================================================================
def find_dataset_file(dataset_name: str) -> Optional[Path]:
    contrastive_dir = Path("./data/NTML-datasets/contrastive")
    if not contrastive_dir.exists():
        return None
    exact = contrastive_dir / f"{dataset_name}.json"
    if exact.exists():
        return exact
    for pat in (f"{dataset_name}*.json", f"*{dataset_name}*.json"):
        matches = list(contrastive_dir.glob(pat))
        if matches:
            return matches[0]
    return None


def list_available_datasets() -> List[str]:
    root = Path("./data/NTML-datasets/contrastive")
    if not root.exists():
        return []
    return sorted([p.stem for p in root.glob("*.json")])


# ================================================================
# CLI
# ================================================================
def parse_args():
    p = argparse.ArgumentParser("Train evaluation-awareness probes")
    p.add_argument("--model_name", required=True)
    p.add_argument("--train_dataset_dir")
    p.add_argument("--dataset_name")
    p.add_argument("--probe_types", nargs="+",
                   choices=["logistic", "linear", "pca",
                            "meandiff", "kmeans"],
                   default=["logistic", "pca", "meandiff"])
    p.add_argument("--layers", nargs="+", default=["all"])
    p.add_argument("--probe_save_dir", required=True)
    p.add_argument("--cache_dir", default="./cache/contrastive")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--activation_batch_size", type=int, default=16)
    # ── NEW ───────────────────────────────────────────────────────────
    p.add_argument("--sample_mode",
                   choices=["ratio", "snippet_avg"],
                   default="ratio",
                   help="ratio = current token-level loader; "
                        "snippet_avg = paper-style snippet loader")
    # ──────────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force_recache", action="store_true")
    p.add_argument("--list_datasets", action="store_true")
    return p.parse_args()


# ================================================================
def main():
    args = parse_args()

    # ---------- dataset path resolution -----------------------------
    if args.list_datasets:
        for d in list_available_datasets():
            print(" •", d)
        return

    if args.train_dataset_dir:
        dataset_path = Path(args.train_dataset_dir)
    elif args.dataset_name:
        dataset_path = find_dataset_file(args.dataset_name)
        if not dataset_path:
            raise SystemExit(f"Dataset '{args.dataset_name}' not found.")
    else:
        raise SystemExit("Must specify --train_dataset_dir or --dataset_name")

    # ---------- load dataset ----------------------------------------
    print("🚀 Contrastive NTML Probe Training")
    print("📄 Dataset:", dataset_path.name)
    print("🤖 Model:", args.model_name)
    print("Loading dataset ...")

    if args.sample_mode == "ratio":
        dataset = load_eval_awareness_dataset(str(dataset_path), args.model_name)
    else:  # snippet_avg
        dataset = load_snippet_avg_dataset(str(dataset_path), args.model_name)

    print("Dataset size:", len(dataset.examples))

    # ---------- load model ------------------------------------------
    model_dtype = get_model_dtype(args.model_name)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=args.device, dtype=model_dtype
    )
    hidden_size = model.cfg.d_model

    # ---------- layer list ------------------------------------------
    layers = list(range(model.cfg.n_layers)) if "all" in args.layers else [int(l) for l in args.layers]
    print("Training on layers:", layers)
    print("Model dtype:", model_dtype)

    # ---------- collect activations ---------------------------------
    activation_stores = smart_cache_activations(
        model,
        dataset,
        layers,
        args.cache_dir,
        args.activation_batch_size,
        args.device,
        model_dtype,
        args.force_recache,
    )
    del model
    torch.cuda.empty_cache()

    # ---------- train probes layer-by-layer --------------------------
    results = {}
    for layer in tqdm(layers, desc="Training layers"):
        hook_pt = f"blocks.{layer}.hook_resid_pre"
        layer_results = train_all_probes_for_layer(
            layer,
            activation_stores[hook_pt],
            args.probe_types,
            args,
            args.model_name,
            hidden_size,
            args.device,
            model_dtype,
        )
        results[layer] = layer_results

    # ---------- save summary ----------------------------------------
    summary_path = Path(args.probe_save_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(summary_path, "w"), indent=2)
    print("✓ Training complete – summary saved to", summary_path)


if __name__ == "__main__":
    main()