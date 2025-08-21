#!/usr/bin/env python3
"""
Evaluate previously-trained evaluation-awareness probes with the
conversation-level label  'metadata.eval_category'.
Label 1  →  eval_category == True
Label 0  →  eval_category == False
"""

import argparse, json, random, torch
from pathlib import Path
from typing import List, Dict, Tuple
import sys

probity_root = Path(__file__).parent.parent
sys.path.insert(0, str(probity_root))

from probity.utils.qwen3_support import ensure_qwen3_transformerlens_support
ensure_qwen3_transformerlens_support()

from probity.utils.eval_awareness_dataset import (
    CharacterPositions, Position, ProbingDataset, TokenizedProbingDataset,
    _prepare_chat_text
)
from transformers import AutoTokenizer
import re


# --------------- probity imports -----------------
from probity.datasets.base import (
    ProbingExample, ProbingDataset,
    CharacterPositions, Position
)
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.utils.qwen3_support import apply_qwen3_chat_template



from probity.utils.eval_awareness_dataset import (
    load_eval_awareness_evalflag     #  <— loader we just added
)
from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.probes import BaseProbe


from probity.probes import (
    BaseProbe,
    LogisticProbe, LogisticProbeConfig,
    PCAProbe,     PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe,  KMeansProbeConfig,
    LinearProbe,  LinearProbeConfig,
)

def _canonical_probe_key(raw: str) -> str:
    """
    Turn 'LogisticProbe' → 'logistic',
         'MeanDifferenceProbe' → 'meandiff',
         'linear' → 'linear', etc.
    """
    raw = raw.strip()

    # strip trailing "Probe"/"probe"
    raw = re.sub(r'probe$', '', raw, flags=re.I)

    # camel-case → snake-ish:  "MeanDifference" → "mean_difference"
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', raw)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    # remove underscores
    snake = snake.replace('_', '')

    # special alias
    if snake in {"meandifference"}:
        snake = "meandiff"

    return snake


# ------------------------------------------------------------------ #
# 2) mapping from canonical key → (ProbeClass, ConfigClass)
# ------------------------------------------------------------------ #
_PROBE_CLASS_MAP = {
    "logistic":   (LogisticProbe,      LogisticProbeConfig),
    "pca":        (PCAProbe,           PCAProbeConfig),
    "meandiff":   (MeanDifferenceProbe,MeanDiffProbeConfig),
    "kmeans":     (KMeansProbe,        KMeansProbeConfig),
    "linear":     (LinearProbe,        LinearProbeConfig),
}


def _rebuild_probe_from_dict(obj: dict) -> BaseProbe:
    ptype_raw = obj.get("probe_type", "")
    key       = _canonical_probe_key(ptype_raw)

    if key not in _PROBE_CLASS_MAP:
        raise ValueError(f"Unknown probe_type '{ptype_raw}' in saved file")

    ProbeCls, CfgCls = _PROBE_CLASS_MAP[key]

    # ---- config handling -----------------------------------------
    raw_cfg = obj.get("config")
    if raw_cfg is None:
        cfg = CfgCls()                               # default config

    elif isinstance(raw_cfg, CfgCls):
        cfg = raw_cfg                                # already a config object

    elif isinstance(raw_cfg, dict):
        cfg = CfgCls(**raw_cfg)                      # build from kwargs

    else:
        raise TypeError(
            f"`config` in saved file is {type(raw_cfg)}, "
            f"expected dict or {CfgCls.__name__}"
        )
    # --------------------------------------------------------------

    probe = ProbeCls(cfg)
    probe.load_state_dict(obj["state_dict"])
    return probe


# ────────────────────────────────────────────────────────────────────
def collect_probes(dir_: Path) -> dict[tuple[int, str], BaseProbe]:
    probes = {}

    for pf in dir_.rglob("layer_*_probe.*"):
        if pf.suffix not in {".json", ".pt"}:
            continue

        # layer number
        try:
            layer = int(pf.stem.split("_")[1])
        except (IndexError, ValueError):
            print(f"⚠️  could not parse layer from {pf.name} – skipped")
            continue

        probe_type_folder = pf.parent.name.lower()

        # -------- load file --------
        if pf.suffix == ".json":
            probe = BaseProbe.load_json(str(pf))

        else:  # ".pt"
            obj = torch.load(pf, map_location="cpu", weights_only=False)
            if isinstance(obj, BaseProbe):
                probe = obj
            elif isinstance(obj, dict):
                probe = _rebuild_probe_from_dict(obj)
            else:
                print(f"⚠️  {pf} is neither a BaseProbe nor a dict – skipped")
                continue

        probes[(layer, probe_type_folder)] = probe
        print(f"✓ loaded {probe_type_folder:10} layer {layer:2}  ({pf.name})")

    if not probes:
        raise RuntimeError(f"No probe files found in {dir_}")

    return probes

# ────────────────────────────────────────────────────────────────────
def load_eval_awareness_evalflag_sampled(
    json_path: str,
    model_name: str,
    max_len: int,
    n_per_class: int | None = None,
    seed: int = 0,
    enable_thinking_flag: bool = False,
) -> TokenizedProbingDataset:
    """
    Build the token-level dataset but keep **at most `n_per_class`
    conversations for each class**.  If `n_per_class` is None the whole
    file is processed (original behaviour).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(json_path) as f:
        raw = json.load(f)["results"]

    # ── decide which conversations we will keep ─────────────────────
    if n_per_class is not None:
        random.seed(seed)
        true_ids  = [item["id"] for item in raw
                     if item["metadata"].get("eval_category", False)]
        false_ids = [item["id"] for item in raw
                     if not item["metadata"].get("eval_category", False)]

        keep_true  = set(random.sample(true_ids,  min(n_per_class, len(true_ids))))
        keep_false = set(random.sample(false_ids, min(n_per_class, len(false_ids))))
        keep = keep_true | keep_false
    else:
        keep = None  # keep everything

    examples: List[ProbingExample] = []
    skipped_long, skipped_missing = 0, 0

    for entry in raw:
        cid = entry["id"]
        if keep is not None and cid not in keep:
            continue

        is_eval  = bool(entry["metadata"].get("eval_category", False))
        lbl_int  = 1 if is_eval else 0
        lbl_text = "eval_aware" if is_eval else "not_eval_aware"

        conv      = entry["reasoning_traces_results"]["full_conversations"][0]
        chat_text = _prepare_chat_text(tok, conv, enable_thinking_flag)

        m = re.search(r"<think>(.*?)</think>", chat_text, re.S)
        if not m:
            skipped_missing += 1
            continue
        think_s, think_e = m.span(1)

        enc       = tok(chat_text, add_special_tokens=False,
                        return_offsets_mapping=True)
        ids       = enc["input_ids"]
        offsets   = enc["offset_mapping"]

        if len(ids) > max_len:
            skipped_long += 1
            continue

        for tok_idx, (cs, ce) in enumerate(offsets):
            if think_s <= cs < think_e:
                examples.append(
                    ProbingExample(
                        text  = chat_text,
                        label = lbl_int,
                        label_text = lbl_text,
                        character_positions = CharacterPositions(
                            {"TARGET": Position(cs, ce)}
                        ),
                        group_id = f"{cid}_tok{tok_idx}",
                        attributes = {
                            "conversation_id": cid,
                            "token_idx": tok_idx,
                            "eval_category": is_eval,
                        },
                    )
                )

        # stop early once we reached the quota
        if n_per_class is not None:
            if (is_eval and cid in keep_true and
                    sum(e.label for e in examples if
                        e.attributes["conversation_id"] == cid) == 0):
                pass  # nothing special
            # we just rely on the outer loop finishing over 'keep'

    print(f"[loader] built {len(examples)} examples "
          f"(skipped {skipped_long} long, {skipped_missing} no-think)")
    ds = ProbingDataset(
        examples,
        dataset_attributes={"task": "eval_awareness_evalflag_sampled",
                            "source_file": json_path}
    )
    return TokenizedProbingDataset.from_probing_dataset(
        ds, tok, padding="max_length", max_length=max_len,
        truncation=True, add_special_tokens=False
    )

# ────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Eval-awareness probe evaluation (balanced sample)")
    p.add_argument("--model_name",   required=True)
    p.add_argument("--dataset_json", required=True)
    p.add_argument("--probe_dir",    required=True)
    p.add_argument("--save_dir",     required=True)
    p.add_argument("--samples", type=int, default=None,
                   help="Total #conversations to evaluate on "
                        "(balanced 50/50). Omit for full set.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for sampling")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_cli()
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)

    n_per_class = args.samples // 2 if args.samples else None
    ds = load_eval_awareness_evalflag_sampled(
        args.dataset_json,
        model_name = args.model_name,
        max_len    = 1024,
        n_per_class= n_per_class,
        seed       = args.seed,
    )

    texts  = [e.text  for e in ds.examples]
    labels = [e.label for e in ds.examples]
    print(f"Dataset: {len(labels)} token-examples "
          f"(pos {sum(labels)}, neg {len(labels)-sum(labels)})")

    probes = collect_probes(Path(args.probe_dir))
    evaluator = OptimizedBatchProbeEvaluator(args.model_name, args.device)
    results  = evaluator.evaluate_all_probes(texts, labels, probes)

    with open(out_dir / "evalflag_metrics.json", "w") as f:
        json.dump({f"{l}_{t}": r for (l, t), r in results.items()},
                  f, indent=2)
    print(f"Metrics saved → {out_dir/'evalflag_metrics.json'}")

    print("\n=== accuracy / AUROC ===")
    for (layer, typ), res in sorted(results.items()):
        print(f"layer {layer:<2}  {typ:<10}  "
              f"acc={res['metrics']['accuracy']:.4f}  "
              f"AUROC={res['metrics']['auroc']:.4f}")

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()