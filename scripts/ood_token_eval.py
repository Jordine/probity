"""OOD token-level evaluation using trained logistic probes."""
import torch, numpy as np, json, warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
warnings.filterwarnings("ignore")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROBE_DIR = Path("probes/token_level_qwen7b/logistic")
LAYERS = [7, 10, 14, 18, 21, 24]
MAX_LEN = 1024
BATCH_SIZE = 4

OOD_DATASETS = {
    "ai_liar": {
        "data": "data/deception_detection/ai_liar.jsonl",
        "labels": "deception-detection-labelled/deception_detection_labeled/ai_liar_labeled.json",
    },
    "sandbagging": {
        "data": "data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl",
        "labels": "deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu_labeled.json",
    },
}


def load_ood_with_labels(data_path, label_path, tokenizer, max_len):
    with open(label_path) as f:
        label_data = json.load(f)

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print("  Loaded {} examples, {} labels".format(len(examples), len(label_data)))

    results = []
    skipped = 0

    for idx, ex in enumerate(examples):
        if "messages" in ex:
            text = " ".join(m.get("content", "") for m in ex["messages"])
        elif "text" in ex:
            text = ex["text"]
        elif "prompt" in ex and "response" in ex:
            text = ex["prompt"] + " " + ex["response"]
        else:
            skipped += 1
            continue

        enc = tokenizer(text, max_length=max_len, truncation=True,
                       padding="max_length", return_tensors="pt",
                       return_offsets_mapping=True)

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        offsets = enc["offset_mapping"][0]

        label_info = label_data[idx] if idx < len(label_data) else None
        if label_info is None:
            skipped += 1
            continue

        spans = label_info.get("deceptive_spans") or label_info.get("lie_spans") or label_info.get("spans", [])

        char_labels = np.zeros(len(text))
        for span in spans:
            if isinstance(span, dict):
                start = span.get("start", span.get("char_start", 0))
                end = span.get("end", span.get("char_end", 0))
            elif isinstance(span, (list, tuple)) and len(span) >= 2:
                start, end = span[0], span[1]
            else:
                continue
            char_labels[start:end] = 1

        token_labels = np.zeros(max_len)
        for tok_idx in range(max_len):
            if attention_mask[tok_idx] == 0:
                continue
            off_start, off_end = offsets[tok_idx].tolist()
            if off_start == off_end:
                continue
            char_slice = char_labels[int(off_start):int(off_end)]
            if len(char_slice) > 0 and char_slice.mean() > 0.5:
                token_labels[tok_idx] = 1

        n_lie = int(token_labels.sum())
        n_truth = int(attention_mask.sum()) - n_lie

        if n_lie > 0 and n_truth > 0:
            results.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_labels": torch.tensor(token_labels, dtype=torch.float32),
            })

    print("  Usable (both classes): {}, skipped: {}".format(len(results), skipped))
    return results


def collect_activations(model, examples, layers, batch_size):
    hook_points = ["blocks.{}.hook_resid_pre".format(l) for l in layers]
    all_acts = {l: [] for l in layers}

    for batch_start in tqdm(range(0, len(examples), batch_size), desc="  Collecting"):
        batch = examples[batch_start:batch_start+batch_size]
        input_ids = torch.stack([e["input_ids"] for e in batch]).to("cuda")

        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids, names_filter=hook_points,
                return_type=None
            )

        for layer in layers:
            hp = "blocks.{}.hook_resid_pre".format(layer)
            all_acts[layer].append(cache[hp].cpu().float())

        del cache
        torch.cuda.empty_cache()

    return {l: torch.cat(v) for l, v in all_acts.items()}


def compute_metrics(scores_list, labels_list, masks_list):
    all_s, all_l, norm_aurocs = [], [], []
    pos_m, neg_m = [], []

    for scores, labels, mask in zip(scores_list, labels_list, masks_list):
        m = mask.bool()
        s = scores[m].numpy()
        l = labels[m].numpy().astype(int)
        all_s.append(s)
        all_l.append(l)

        n_pos, n_neg = l.sum(), len(l) - l.sum()
        if n_pos > 0 and n_neg > 0:
            std = s.std()
            if std > 1e-8:
                z = (s - s.mean()) / std
                try:
                    norm_aurocs.append(roc_auc_score(l, z))
                except:
                    pass
            pos_m.append(float(s[l==1].mean()))
            neg_m.append(float(s[l==0].mean()))

    all_s = np.concatenate(all_s)
    all_l = np.concatenate(all_l)
    try:
        glob = roc_auc_score(all_l, all_s)
    except:
        glob = 0.5
    mn = np.mean(norm_aurocs) if norm_aurocs else 0.5
    md = np.median(norm_aurocs) if norm_aurocs else 0.5
    sep = np.mean(pos_m) - np.mean(neg_m) if pos_m else 0
    return glob, mn, md, sep, len(norm_aurocs)


print("Loading model...")
from probity.utils.dataset_loading import get_model_dtype
model_dtype = get_model_dtype(MODEL)
model = HookedTransformer.from_pretrained_no_processing(
    MODEL, device="cuda", dtype=model_dtype)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load probes
probes = {}
for layer in LAYERS:
    path = PROBE_DIR / "layer_{}_probe.pt".format(layer)
    saved = torch.load(str(path), map_location="cpu", weights_only=False)
    w = saved["state_dict"]["linear.weight"].float()
    b = saved["state_dict"]["linear.bias"].float()
    probes[layer] = (w, b)

print("\n" + "="*80)
print("OOD TOKEN-LEVEL EVALUATION (Qwen2.5-7B logistic probes)")
print("="*80)

for ds_name, paths in OOD_DATASETS.items():
    print("\n--- {} ---".format(ds_name))
    examples = load_ood_with_labels(paths["data"], paths["labels"], tokenizer, MAX_LEN)

    if not examples:
        print("  No usable examples!")
        continue

    acts = collect_activations(model, examples, LAYERS, BATCH_SIZE)

    header = "  {:<10} {:>10} {:>10} {:>10} {:>8} {:>5}".format(
        "Layer", "GlobAUROC", "nAUROC", "Med nAUROC", "Sep", "N")
    print(header)
    print("  " + "-"*60)

    for layer in LAYERS:
        w, b = probes[layer]
        logits = (acts[layer] @ w.T).squeeze(-1) + b
        scores = torch.sigmoid(logits)

        s_list = [scores[i] for i in range(len(examples))]
        l_list = [examples[i]["token_labels"] for i in range(len(examples))]
        m_list = [examples[i]["attention_mask"].float() for i in range(len(examples))]

        glob, mn, md, sep, n_eval = compute_metrics(s_list, l_list, m_list)
        row = "  L{:<9} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f} {:>5}".format(
            layer, glob, mn, md, sep, n_eval)
        print(row)

    del acts
    torch.cuda.empty_cache()

print("\n" + "="*80)
print("nAUROC > 0.5 = above-chance localization")
