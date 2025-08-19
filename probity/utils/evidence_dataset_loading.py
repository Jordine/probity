# probity/utils/evidence_dataset_loading.py
from typing import List, Dict
import json
from transformers import AutoTokenizer
from pathlib import Path

import sys
probity_root = Path(__file__).parent.parent
sys.path.insert(0, str(probity_root))

from probity.utils.qwen3_support import ensure_qwen3_transformerlens_support
ensure_qwen3_transformerlens_support()

from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions, Position
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.utils.qwen3_support import apply_qwen3_chat_template


def load_evidence_dataset(json_path: str,
                          model_name: str = "Qwen/Qwen3-32B",
                          enable_thinking: bool = True,
                          max_len: int = 2048) -> TokenizedProbingDataset:
    """
    Build a statement-level dataset where every evidence span is a *positive* example.
    Label = 1  for evidence tokens, any further negatives can be produced later.
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(json_path) as f:
        data = json.load(f)

    examples: List[ProbingExample] = []

    for entry in data["results"]:
        evs: List[str] = entry["reasoning_traces_results"]["parsed_judge_response"].get("evidence", [])
        if not evs:
            continue

        convo = entry["reasoning_traces_results"]["full_conversations"][0]

        # build canonical Qwen3 template text
        messages: List[Dict[str, str]] = []
        for m in convo:
            # reasoning is inserted as an assistant message *before* answer
            if m["role"] == "assistant" and "reasoning" in m:
                messages.append({"role": "assistant", "content": m["reasoning"]})
            messages.append({"role": m["role"], "content": m["content"]})

        chat_text: str = apply_qwen3_chat_template(
            tok, messages,
            enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=False
        )

        for ev in evs:
            start = chat_text.find(ev)
            if start == -1:
                continue
            end = start + len(ev)

            examples.append(
                ProbingExample(
                    text=chat_text,
                    label=1,
                    label_text="evidence",
                    character_positions=CharacterPositions(
                        {"TARGET": Position(start, end)}
                    ),
                    attributes={"evidence_text": ev,
                                "category": entry["metadata"]["category"]}
                )
            )

    probing_ds = ProbingDataset(
        examples,
        dataset_attributes={"task": "evidence_detection",
                            "source_file": json_path}
    )

    tok_ds = TokenizedProbingDataset.from_probing_dataset(
        probing_ds,
        tok,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=False         # chat template already added them
    )
    return tok_ds