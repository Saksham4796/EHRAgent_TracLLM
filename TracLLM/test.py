#!/usr/bin/env python3
"""Compute attribution scores for each memory record given a query and response.

This version has no CLI. Adjust the variables `QUERY`, `RESPONSE`, and `MEMORY`
below. `MEMORY` defaults to the contents of tasks.json, with each element
treated as an independent memory record for attribution.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

from src.attribution import PerturbationBasedAttribution
from src.models import create_model


# -----------------------------
# User-editable inputs
# -----------------------------
# Replace these with your own query/response if desired.
QUERY = (
    "How should I validate text data, avoid leakage when splitting, and set up "
    "tokenization plus transformer fine-tuning with logging and deployment in mind?"
)

RESPONSE = """[autogen]
assistant: Loading training pipeline steps.
assistant: Validating raw_df with pandera to catch nulls and invalid labels, then dropping duplicate texts.
assistant: Performing GroupShuffleSplit with a 20% holdout keyed by user_id to avoid leakage.
assistant: Normalizing text (lowercase, replace URLs with <URL>, mentions with <USER>) and tokenizing with bert-base-uncased at max_length=160.
assistant: Building a PyTorch Dataset that injects labels and uses dynamic padding in the collate_fn to keep batches tight.
assistant: Finetuning bert-base-uncased with AdamW (lr=2e-5, weight_decay=0.01), mixed precision, grad accumulation of 4, and gradient clipping.
assistant: Running 5-fold CV; bootstrapping fold F1s to estimate a 95% CI.
assistant: Sampling hyperparameters within bounded ranges and applying early stopping patience=2 to cut bad trials.
assistant: Saving weights, tokenizer, metrics, and git commit under runs/<timestamp>/ with lineage metadata.
assistant: Logging forward pass latency and GPU mem usage per batch.
assistant: Exposing FastAPI endpoints /predict (micro-batched) and /healthz for readiness."""

# Load memory records. By default, we read tasks.json where each item is a dict
# with task/reasoning/code keys. You can replace MEMORY with any list of dicts.
MEMORY_PATH = Path("tasks.json")
if MEMORY_PATH.exists():
    MEMORY = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
else:
    MEMORY = []


# -----------------------------
# Attribution configuration
# -----------------------------
MODEL_CONFIG_PATH = Path("model_configs/gpt4o_config.json")
DEVICE = "cpu"
EXPLANATION_LEVEL = "sentence"
TOP_K = 10000
SCORE_FUNCS = ["stc", "loo", "denoised_shapley"]
SH_N = 5
BETA = 0.2
LOO_WEIGHT = 2
OUTPUT_JSON_PATH = Path("memory_attribution.json")


def build_memory_contexts(memory_items: Sequence[dict]) -> list[str]:
    """Convert memory records into plain text spans for attribution."""
    contexts: list[str] = []
    for item in memory_items:
        task = item.get("task", "").strip()
        reasoning = item.get("reasoning", "").strip()
        code = item.get("code", "").strip()
        span = "\n".join(
            part for part in [f"Task: {task}", f"Reasoning: {reasoning}", f"Code:\n{code}"] if part
        ).strip()
        if span:
            contexts.append(span)
    return contexts


def build_model(config_path: Path, device: str):
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find model config at {config_path}")
    return create_model(config_path=config_path, device=device)


def run_attribution(
    question: str,
    contexts: Sequence[str],
    answer: str,
    attr: PerturbationBasedAttribution,
) -> Tuple[List[str], List[int], List[float], float, dict]:
    return attr.attribute(question, list(contexts), answer)


def print_ranked_results(
    texts: Sequence[str],
    important_ids: Sequence[int],
    importance_scores: Sequence[float],
    top_k: int,
) -> None:
    ranked = sorted(zip(important_ids, importance_scores), key=lambda x: x[1], reverse=True)
    if not ranked:
        print("No attribution scores were produced.")
        return

    print("\nTop attributed memory records:")
    displayed = 0
    for rank, (idx, score) in enumerate(ranked, start=1):
        paragraph = " ".join(texts[idx].split())
        wrapped = textwrap.fill(paragraph, width=90)
        print(f"\n#{rank} (score={score:.4f}, memory_id={idx})")
        print(wrapped)
        displayed += 1
        if displayed >= top_k:
            break


def print_sentence_scores(
    texts: Sequence[str],
    important_ids: Sequence[int],
    importance_scores: Sequence[float],
    span_label: str,
) -> None:
    if not texts:
        print("\nNo memory spans were generated for scoring.")
        return
    print(f"\n{span_label.capitalize()}-level attribution scores:")
    score_map = {idx: score for idx, score in zip(important_ids, importance_scores)}
    for idx, sentence in enumerate(texts, start=1):
        score = score_map.get(idx - 1, 0.0)
        normalized = " ".join(sentence.split())
        wrapped = textwrap.fill(normalized, width=90)
        print(f"\n{span_label.capitalize()} {idx} (score={score:.4f})")
        print(wrapped)


def summarize_score_funcs(ensemble_list: dict) -> None:
    if not ensemble_list:
        return
    contribution_counts = {name: 0 for name in ensemble_list}
    for name, entries in ensemble_list.items():
        contribution_counts[name] += len(entries)
    print("\nScore function contributions:")
    for name, count in contribution_counts.items():
        print(f"  - {name}: {count} spans scored")


def dump_memory_scores(
    path: Path,
    texts: Sequence[str],
    important_ids: Sequence[int],
    importance_scores: Sequence[float],
) -> None:
    """Persist final attribution score for each memory span to JSON."""
    if not path:
        return
    score_map = {int(idx): float(score) for idx, score in zip(important_ids, importance_scores)}
    payload = [
        {"memory_id": idx, "score": score_map.get(idx, 0.0)}
        for idx in range(len(texts))
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved memory-level attribution scores to {path}")


def main() -> None:
    contexts = build_memory_contexts(MEMORY)
    if not contexts:
        raise ValueError("MEMORY is empty. Populate MEMORY with a list of dicts or ensure tasks.json exists.")

    llm = build_model(MODEL_CONFIG_PATH, DEVICE)
    attr = PerturbationBasedAttribution(
        llm,
        explanation_level=EXPLANATION_LEVEL,
        K=TOP_K,
        attr_type="tracllm",
        score_funcs=SCORE_FUNCS,
        sh_N=SH_N,
        w=LOO_WEIGHT,
        beta=BETA,
        verbose=1,
    )

    print(f"\nQuery:\n{QUERY}\n")
    print(f"Response snippet:\n{textwrap.shorten(RESPONSE, width=240, placeholder=' ...')}\n")

    texts, important_ids, importance_scores, runtime, ensemble_list = run_attribution(
        QUERY, contexts, RESPONSE, attr
    )
    print_sentence_scores(texts, important_ids, importance_scores, "memory item")
    print_ranked_results(texts, important_ids, importance_scores, top_k=min(5, len(texts)))
    summarize_score_funcs(ensemble_list)
    dump_memory_scores(OUTPUT_JSON_PATH, texts, important_ids, importance_scores)
    print(f"\nRuntime: {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
