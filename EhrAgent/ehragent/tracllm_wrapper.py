import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

TRACLLM_ROOT = Path(__file__).resolve().parents[2] / "TracLLM"
if str(TRACLLM_ROOT) not in sys.path:
    sys.path.append(str(TRACLLM_ROOT))

from src.attribution import PerturbationBasedAttribution
from src.models import create_model


class TracLLMWrapper:
    """Thin wrapper that scores memory records with TracLLM."""

    def __init__(
        self,
        config_path: str,
        device: str = "cpu",
        explanation_level: str = "sentence",
        top_k: int = 5,
        score_funcs: Sequence[str] = ("stc", "loo", "denoised_shapley"),
        shapley_samples: int = 5,
        loo_weight: float = 2.0,
        beta: float = 0.2,
        verbose: int = 1,
    ):
        if not config_path:
            raise ValueError("TracLLM config path must be provided.")

        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Treat relative paths as repo-root relative, not nested under TracLLM twice.
            repo_root = Path(__file__).resolve().parents[2]
            config_path = (repo_root / config_path).resolve()

        if not config_path.exists():
            raise FileNotFoundError(f"TracLLM model config not found at: {config_path}")

        self.llm = create_model(config_path=config_path, device=device)
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

    def score_memory(
        self,
        query: str,
        response: str,
        recs: List[Dict],
    ) -> List[Tuple[float, Dict, int]]:
        """Return attribution scores ordered by importance."""
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
