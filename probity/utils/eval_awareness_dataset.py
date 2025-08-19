"""
Dataset loaders for the Evaluation-Awareness experiments.

• load_eval_awareness_dataset …  token-level   (old behaviour)
• load_snippet_avg_dataset     …  snippet-level, averaged span (paper style)
"""

from __future__ import annotations
import json, random, re
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

# local imports
from probity.datasets.base import (
    ProbingExample, ProbingDataset,
    CharacterPositions, Position
)
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.utils.qwen3_support import apply_qwen3_chat_template

# --------------------------------------------------------------------- #
MAX_LENGTH      = 1024
NEGATIVE_RATIO  = 1.0
RNG             = random.Random(0)
# --------------------------------------------------------------------- #

# ═══════════════════════════════════════════════════════════════════════
#  Helper utilities shared by both loaders
# ═══════════════════════════════════════════════════════════════════════
def _insert_think_block(reasoning: str, answer: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n{answer}"

def _token_idxs_to_char_pos(ids: List[int], target_tok_idx: int,
                            offsets: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Turn token index back into (char_start, char_end) in original string."""
    start, end = offsets[target_tok_idx]
    return start, end


def _prepare_chat_text(
    tok,
    conv: List[dict],
    enable_thinking_flag: bool = False,
) -> str:
    system_prompt   = conv[0]["content"]
    user_prompt     = conv[1]["content"]
    assistant_ans   = conv[2]["content"]
    assistant_reas  = conv[2]["reasoning"]
    assistant_msg   = _insert_think_block(assistant_reas, assistant_ans)
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": assistant_msg},
    ]
    return apply_qwen3_chat_template(
        tok,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking_flag,
    )


# ═══════════════════════════════════════════════════════════════════════
#  1) token-level "ratio" loader (unchanged)
# ═══════════════════════════════════════════════════════════════════════
def load_eval_awareness_dataset(json_path: str,
                                model_name: str = "Qwen/Qwen3-32B",
                                enable_thinking_flag: bool = False,
                                max_len: int = MAX_LENGTH) -> TokenizedProbingDataset:
    print(f"\n### Loading evaluation-awareness dataset from {json_path}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Tokenizer BOS={tok.bos_token_id}  EOS/PAD={tok.eos_token_id}")

    data = json.load(open(json_path, "r"))
    total_entries = len(data["results"])
    print(f"Entries in file: {total_entries}")

    examples: List[ProbingExample] = []
    skipped_long, skipped_noev = 0, 0

    for idx, entry in enumerate(data["results"]):
        ev_list: List[str] = entry["reasoning_traces_results"]["parsed_judge_response"].get("evidence", [])
        if not ev_list:
            skipped_noev += 1
            continue

        conv = entry["reasoning_traces_results"]["full_conversations"][0]
        system_prompt = conv[0]["content"]
        user_prompt   = conv[1]["content"]
        assistant_answer = conv[2]["content"]
        assistant_reason = conv[2]["reasoning"]

        assistant_msg = _insert_think_block(assistant_reason, assistant_answer)

        messages = [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_msg},
        ]

        # canonical Qwen chat template
        chat_text = apply_qwen3_chat_template(
            tok, messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking_flag   # we already inserted <think>
        )

        # ------------------------------------------------------------------ #
        # Tokenise WITH OFFSETS so we can map back chars → token indices
        encoding = tok(chat_text,
                       add_special_tokens=False,
                       return_offsets_mapping=True)
        token_ids   = encoding["input_ids"]
        offsets     = encoding["offset_mapping"]         # list[(start,end)]

        if len(token_ids) > max_len:
            skipped_long += 1
            continue

        reasoning_block_pattern = re.compile(r"<think>(.*?)</think>", re.S)
        m = reasoning_block_pattern.search(chat_text)
        if not m:
            print(f"!!!  No <think> block in conversation {idx}, skipping.")
            continue
        think_start_char, think_end_char = m.span(1)

        # token indices belonging to reasoning block
        think_token_idxs = [i for i,(s,e) in enumerate(offsets)
                            if think_start_char <= s < think_end_char]

        # ------------------------------------------------------------------ #
        # mark positive tokens (evidence) & negatives (other reasoning)
        positive_token_idxs: List[int] = []
        for evid in ev_list:
            char_pos = chat_text.find(evid, think_start_char, think_end_char)
            if char_pos == -1:
                continue
            evid_tok_ids = tok(evid, add_special_tokens=False)["input_ids"]
            # naive exact match inside token seq
            for i in think_token_idxs:
                if token_ids[i:i+len(evid_tok_ids)] == evid_tok_ids:
                    positive_token_idxs.extend(range(i, i+len(evid_tok_ids)))

        positive_token_idxs = sorted(set(positive_token_idxs))
        negative_token_pool = sorted(set(think_token_idxs) - set(positive_token_idxs))
        RNG.shuffle(negative_token_pool)
        negative_token_idxs = negative_token_pool[:int(len(positive_token_idxs)*NEGATIVE_RATIO)]

        # ------------------------------------------------------------------ #
        # build ProbingExamples   (one per token position)
        def _make_example(tok_idx: int, label: int):
            char_s, char_e = _token_idxs_to_char_pos(token_ids, tok_idx, offsets)
            return ProbingExample(
                text = chat_text,
                label = label,
                label_text = "evidence" if label==1 else "non-evidence",
                character_positions = CharacterPositions(
                    {"TARGET": Position(char_s, char_e)}
                ),
                group_id = entry["id"]+f"_tok{tok_idx}",
                attributes = {
                    "conversation_idx": idx,
                    "category": entry["metadata"]["category"],
                    "token_idx": tok_idx,
                    "token_text": tok.decode(token_ids[tok_idx]),
                    "is_positive": bool(label),
                    "evidence_texts": ev_list if label else None,
                }
            )

        examples.extend(_make_example(t,1) for t in positive_token_idxs)
        examples.extend(_make_example(t,0) for t in negative_token_idxs)

        # ---------------- debug prints for the first few ------------------ #
        if idx < 3:
            print(f"\n--- CONVERSATION {idx} ----------------------------------")
            print(chat_text[:400] + (" …" if len(chat_text)>400 else ""))
            print(f"#tokens={len(token_ids)}   pos={len(positive_token_idxs)}   neg={len(negative_token_idxs)}")
            for t in positive_token_idxs[:5]:
                print(f"   POS tok#{t}:  «{tok.decode([token_ids[t]])}»")
            for t in negative_token_idxs[:5]:
                print(f"   NEG tok#{t}:  «{tok.decode([token_ids[t]])}»")

    print(f"\nFinished: built {len(examples)} token-level examples"
          f"  (skipped {skipped_noev} w/o evidence, {skipped_long} > max_len)")
    probing_ds = ProbingDataset(examples,
        dataset_attributes={"task": "eval_awareness", "source_file": json_path})
    tokenised_ds = TokenizedProbingDataset.from_probing_dataset(
        probing_ds, tok,
        padding="max_length", max_length=max_len, truncation=True,
        add_special_tokens=False
    )
    # final sanity print
    print(f"Tokenised dataset: {len(tokenised_ds.examples)} examples,"
          f"  position_types={tokenised_ds.position_types}\n")
    return tokenised_ds


# ═══════════════════════════════════════════════════════════════════════
#  2) snippet-level / averaged-span loader  (no extra judge fields)
# ═══════════════════════════════════════════════════════════════════════
def load_snippet_avg_dataset(
    json_path: str,
    model_name: str = "Qwen/Qwen3-32B",
    enable_thinking_flag: bool = False,
    max_len: int = MAX_LENGTH,
    balance: bool = True,
) -> TokenizedProbingDataset:
    """
    One example per *snippet*.

    • positive  = any span listed under parsed_judge_response["evidence"]  
    • negative  = the remaining text inside the same <think> … </think> block,
                  split into coarse sentence-level chunks.

    Hidden-state averaging is done later by passing `average_span=True` to
    trainer.prepare_supervised_data (already wired in the script file).
    """
    print(f"\n### Loading SNIPPET-AVG dataset from {json_path}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    raw = json.load(open(json_path))
    examples: List[ProbingExample] = []
    total_pos, total_neg = 0, 0

    for entry_id, entry in enumerate(raw["results"]):
        conv  = entry["reasoning_traces_results"]["full_conversations"][0]
        chat_text = _prepare_chat_text(tok, conv, enable_thinking_flag)

        # --- locate <think> block -------------------------------------
        m = re.search(r"<think>(.*?)</think>", chat_text, re.S)
        if not m:
            continue
        think_s, think_e = m.span(1)
        think_text = chat_text[think_s:think_e]

        # positive snippets (exact strings)
        pos_snips: List[str] = entry["reasoning_traces_results"]["parsed_judge_response"]["evidence"]

        # mark occupied character ranges so we can carve out the rest
        occupied: List[Tuple[int, int]] = []
        for snip in pos_snips:
            idx = think_text.find(snip)
            if idx != -1:
                occupied.append((idx, idx + len(snip)))

        # negative snippets = remaining text, split on simple separators
        neg_candidates = re.split(r"[.\n]", think_text)
        neg_snips: List[str] = []
        for chunk in neg_candidates:
            chunk = chunk.strip()
            if len(chunk) < 3:
                continue
            # skip if this chunk overlaps any positive span
            chunk_start = think_text.find(chunk)
            chunk_end   = chunk_start + len(chunk)
            overlap = any(
                not (chunk_end <= s or chunk_start >= e) for s, e in occupied
            )
            if not overlap:
                neg_snips.append(chunk)

        # optionally balance dataset (1-to-1)
        if balance:
            RNG.shuffle(neg_snips)
            neg_snips = neg_snips[: len(pos_snips)]

        # ---- helper to create a ProbingExample -----------------------

        def add_example(snip: str, lbl: int):
            global_start = chat_text.find(snip, think_s, think_e)
            if global_start == -1:
                return
            examples.append(
                ProbingExample(
                    text=chat_text,
                    label=lbl,
                    label_text="evidence" if lbl == 1 else "non-evidence",
                    character_positions=CharacterPositions(
                        {"TARGET": Position(global_start,
                                            global_start + len(snip))}
                    ),
                    attributes={
                        "snippet_text": snip,
                        "conversation_idx": entry_id,
                        "category": entry["metadata"]["category"],
                    },
                )
            )
        for s in pos_snips:
            add_example(s, 1)
        for s in neg_snips:
            add_example(s, 0)

        total_pos += len(pos_snips)
        total_neg += len(neg_snips)

    print(f"Built {len(examples)} snippet-level examples "
          f"({total_pos} pos / {total_neg} neg)")
    return TokenizedProbingDataset.from_probing_dataset(
        ProbingDataset(examples), tok,
        padding="max_length", max_length=max_len,
        truncation=True, add_special_tokens=False,
    )