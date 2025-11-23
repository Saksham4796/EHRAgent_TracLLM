import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TRACLLM_ROOT = Path(__file__).resolve().parents[2] / "TracLLM"
if str(TRACLLM_ROOT) not in sys.path:
    sys.path.append(str(TRACLLM_ROOT))

from src.attribution import PerturbationBasedAttribution
from src.models import create_model


class TracLLMWrapper:
    """Thin wrapper that scores memory records with TracLLM."""

    def __init__(
        self,
        config_path: Optional[str],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        explanation_level: str = "sentence",
        top_k: int = 5,
        score_funcs: Sequence[str] = ("stc", "loo", "denoised_shapley"),
        shapley_samples: int = 5,
        loo_weight: float = 2.0,
        beta: float = 0.2,
        verbose: int = 1,
    ):
        model_config_path: Optional[Path] = None

        if config_path:
            config_path_obj = Path(config_path)
            if not config_path_obj.is_absolute():
                # Treat relative paths as repo-root relative, not nested under TracLLM twice.
                repo_root = Path(__file__).resolve().parents[2]
                config_path_obj = (repo_root / config_path_obj).resolve()

            if not config_path_obj.exists():
                raise FileNotFoundError(f"TracLLM model config not found at: {config_path_obj}")

            model_config_path = config_path_obj

        resolved_api_key = api_key or os.getenv("TRACLLM_API_KEY") or os.getenv("OPENAI_API_KEY")

        if model_config_path:
            self.llm = create_model(config_path=model_config_path, device=device)
        elif model:
            if not resolved_api_key:
                raise ValueError("TracLLM model requested but no API key found. Set TRACLLM_API_KEY or OPENAI_API_KEY.")
            self.llm = create_model(model_path=model, api_key=resolved_api_key, device=device)
        else:
            raise ValueError("TracLLM config path or model name must be provided.")
        self.attr = PerturbationBasedAttribution(
            self.llm,
            explanation_level=explanation_level,
            K=top_k,
            attr_type="tracllm",
            score_funcs=list(score_funcs),
            sh_N=shapley_samples,
            w=loo_weight,
            beta=beta,
            verbose=verbose,
        )

    @staticmethod
    def _memory_to_context(recs: List[Dict]) -> List[str]:
        contexts: List[str] = []
        for rec in recs:
            question = rec.get("question", "").strip()
            knowledge = rec.get("knowledge", "").strip()
            code = rec.get("code", "").strip()
            parts = [
                f"Question: {question}" if question else "",
                f"Knowledge: {knowledge}" if knowledge else "",
                f"Solution:\n{code}" if code else "",
            ]
            context = "\n".join([p for p in parts if p]).strip()
            contexts.append(context)
        return contexts

    @staticmethod
    def _memory_to_partitioned_contexts(recs: List[Dict]) -> Tuple[List[str], List[int]]:
        """Flatten each memory record into Question/Knowledge/Solution parts."""
        contexts: List[str] = []
        rec_index_for_part: List[int] = []
        for idx, rec in enumerate(recs):
            question = rec.get("question", "").strip()
            knowledge = rec.get("knowledge", "").strip()
            code = rec.get("code", "").strip()
            parts = [
                ("Question", question),
                ("Knowledge", knowledge),
                ("Solution", code),
            ]
            for label, text in parts:
                if not text:
                    continue
                context = f"{label}: {text}"
                contexts.append(context)
                rec_index_for_part.append(idx)
        return contexts, rec_index_for_part

    def score_memory_overall(
        self,
        query: str,
        response: str,
        recs: List[Dict],
    ) -> List[Tuple[float, Dict, int]]:
        """Return attribution scores ordered by importance (whole-record attribution)."""
        contexts = self._memory_to_context(recs)
        try:
            texts, important_ids, importance_scores, _, _ = self.attr.attribute(
                query,
                contexts,
                response,
            )
            score_map = {idx: float(score) for idx, score in zip(important_ids, importance_scores)}
        except Exception as exc:
            print(f"[TracLLM] Attribution failed, using uniform scores. Error: {exc}")
            score_map = {idx: 0.0 for idx in range(len(recs))}

        ranked = [(score_map.get(idx, 0.0), recs[idx], idx) for idx in range(len(recs))]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def score_memory_partition(
        self,
        query: str,
        response: str,
        recs: List[Dict],
    ) -> List[Tuple[float, Dict, int]]:
        """
        Return attribution scores ordered by importance where each Question/Knowledge/Solution
        part is scored independently and the max part score is used for the record.
        """
        contexts, rec_index_for_part = self._memory_to_partitioned_contexts(recs)
        if not contexts:
            return [(0.0, recs[idx], idx) for idx in range(len(recs))]

        try:
            texts, important_ids, importance_scores, _, _ = self.attr.attribute(
                query,
                contexts,
                response,
            )
            score_map = {idx: float(score) for idx, score in zip(important_ids, importance_scores)}
        except Exception as exc:
            print(f"[TracLLM] Partitioned attribution failed, using uniform scores. Error: {exc}")
            score_map = {idx: 0.0 for idx in range(len(contexts))}

        record_scores: Dict[int, float] = {idx: 0.0 for idx in range(len(recs))}
        for local_idx, rec_idx in enumerate(rec_index_for_part):
            part_score = score_map.get(local_idx, 0.0)
            record_scores[rec_idx] = max(record_scores.get(rec_idx, 0.0), part_score)

        ranked = [(record_scores.get(idx, 0.0), recs[idx], idx) for idx in range(len(recs))]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked
