"""Compare NTML vs roleplay probes on debate transcripts.
Runs each probe on debate text through Qwen 7B and shows qualitative scores."""
import torch, numpy as np, json, warnings, glob
from pathlib import Path
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from probity.utils.dataset_loading import get_model_dtype
warnings.filterwarnings("ignore")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 1024
LAYER = 21

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load both probes
probes = {}
for name, path in [
    ("ntml", "probes/token_level_qwen7b/logistic/layer_21_probe.pt"),
    ("roleplay", "probes/token_level_roleplay/logistic/layer_21_probe.pt"),
    ("ranking", "probes/token_level_ranking/logistic/layer_21_probe.pt"),
]:
    if Path(path).exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        w = saved["state_dict"]["linear.weight"].float()
        b = saved["state_dict"]["linear.bias"].float()
        probes[name] = (w, b)
        print("Loaded %s probe" % name)

model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=get_model_dtype(MODEL))
hook = "blocks.%d.hook_resid_pre" % LAYER

# Load debate transcripts
transcript_files = sorted(glob.glob(
    "debate_transcripts_apollo_dec14_final/baseline_judge_run/transcripts/*.json"))[:10]

print("\nAnalyzing %d debate transcripts..." % len(transcript_files))
print("=" * 80)

for tf_idx, tf_path in enumerate(transcript_files[:5]):
    with open(tf_path) as f:
        transcript = json.load(f)

    mm = transcript["model_mapping"]  # e.g. MODEL_A=dishonest
    pd = transcript["problem_data"]
    question = pd.get("question", "?")
    correct = pd.get("correct_answer", "?")
    incorrect = pd.get("incorrect_answer", "?")

    print("\n--- Debate %d: %s ---" % (tf_idx, question[:60]))
    print("Correct: %s" % correct[:80])
    print("Incorrect: %s" % incorrect[:80])

    for turn in transcript["turns"][:4]:  # first 4 turns
        speaker = turn["speaker"]
        role = mm.get(speaker, "?")
        content = turn["content"]
        turn_num = turn["turn_number"]

        # Run through model
        enc = tokenizer(content, max_length=MAX_LEN, truncation=True,
                       padding="max_length", return_tensors="pt")
        with torch.no_grad():
            _, cache = model.run_with_cache(
                enc["input_ids"].cuda(),
                names_filter=[hook], return_type=None)
        acts = cache[hook].cpu().float()
        am = enc["attention_mask"][0]
        del cache; torch.cuda.empty_cache()

        print("\n  Turn %d [%s]:" % (turn_num, role))

        for probe_name, (w, b) in probes.items():
            logits = (acts @ w.T).squeeze(-1) + b
            scores = torch.sigmoid(logits)[0]
            valid_scores = [scores[ti].item() for ti in range(MAX_LEN) if am[ti] > 0]
            mean_s = np.mean(valid_scores)
            max_s = np.max(valid_scores)
            pct_high = np.mean([s > 0.5 for s in valid_scores]) * 100

            # Top 5 tokens
            ts = [(scores[ti].item(), ti) for ti in range(MAX_LEN) if am[ti] > 0]
            ts.sort(reverse=True)
            top5_tokens = []
            for s, ti in ts[:5]:
                tok = tokenizer.decode([enc["input_ids"][0][ti].item()])
                top5_tokens.append("%.2f:%s" % (s, tok.strip()))

            print("    %s: mean=%.3f max=%.3f %%>0.5=%.0f%% top=[%s]" % (
                probe_name, mean_s, max_s, pct_high, ", ".join(top5_tokens)))

        del acts

# Summary: can we distinguish dishonest from honest turns?
print("\n" + "=" * 80)
print("AGGREGATE: dishonest vs honest turn scores across all transcripts")
print("=" * 80)

for probe_name, (w, b) in probes.items():
    d_scores, h_scores = [], []

    for tf_path in transcript_files:
        with open(tf_path) as f:
            transcript = json.load(f)
        mm = transcript["model_mapping"]

        for turn in transcript["turns"]:
            content = turn["content"]
            role = mm.get(turn["speaker"], "?")

            enc = tokenizer(content, max_length=MAX_LEN, truncation=True,
                           padding="max_length", return_tensors="pt")
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    enc["input_ids"].cuda(),
                    names_filter=[hook], return_type=None)
            acts = cache[hook].cpu().float()
            am = enc["attention_mask"][0]
            del cache; torch.cuda.empty_cache()

            logits = (acts @ w.T).squeeze(-1) + b
            scores = torch.sigmoid(logits)[0]
            valid = [scores[ti].item() for ti in range(MAX_LEN) if am[ti] > 0]
            mean_s = np.mean(valid)

            if "dishonest" in role:
                d_scores.append(mean_s)
            else:
                h_scores.append(mean_s)
            del acts

    d_mean = np.mean(d_scores) if d_scores else 0
    h_mean = np.mean(h_scores) if h_scores else 0
    sep = d_mean - h_mean
    print("  %s: dishonest=%.4f (%d turns), honest=%.4f (%d turns), sep=%.4f" % (
        probe_name, d_mean, len(d_scores), h_mean, len(h_scores), sep))
