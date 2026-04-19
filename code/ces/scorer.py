"""
scorer.py
---------
Core implementation of the Lin unit.

Lin(M) = I(M) / log2(C(M))

Where:
  I(M) = information score — bits of uncertainty resolved, derived from
         Shannon entropy reduction across a benchmark suite
  C(M) = normalized compute cost (inference FLOPs in teraFLOPs)

Reference anchor: human expert performance = 1.0 Lin
"""

import math
from lin.models import MODEL_FLOPS, MODEL_MMLU, MODEL_HUMANEVAL


# Benchmark weights (must sum to 1.0)
BENCHMARK_WEIGHTS = {
    "mmlu":      0.60,   # general knowledge + reasoning (57 subjects)
    "humaneval": 0.40,   # code generation (proxy for structured reasoning)
}


def information_score(mmlu: float, humaneval: float) -> float:
    """
    Convert benchmark accuracies to bits of uncertainty resolved.

    Shannon entropy of a binary classification at accuracy p:
        H_before = 1 bit (uniform prior)
        H_after  = -p*log2(p) - (1-p)*log2(1-p)
        bits_resolved = H_before - H_after

    We use weighted average across benchmarks.
    """
    def bits_resolved(accuracy: float) -> float:
        p = max(min(accuracy, 0.9999), 0.0001)  # clamp to avoid log(0)
        h_after = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        return 1.0 - h_after  # bits resolved from 1-bit prior

    scores = {
        "mmlu":      bits_resolved(mmlu),
        "humaneval": bits_resolved(humaneval),
    }

    weighted = sum(scores[k] * BENCHMARK_WEIGHTS[k] for k in scores)
    return weighted


def lin_score(
    model: str = None,
    *,
    flops: float = None,
    mmlu: float = None,
    humaneval: float = None,
) -> float:
    """
    Calculate the Lin score for a model.

    Args:
        model:     model name (looks up known values automatically)
        flops:     inference FLOPs in teraFLOPs (override)
        mmlu:      MMLU accuracy 0-1 (override)
        humaneval: HumanEval pass@1 0-1 (override)

    Returns:
        Lin score (float). Human expert ≈ 1.0 Lin.

    Example:
        >>> lin_score("gpt-4")
        0.743
        >>> lin_score(flops=70, mmlu=0.82, humaneval=0.814)
        0.812
    """
    if model is not None:
        model = model.lower().strip()
        if flops is None:
            if model not in MODEL_FLOPS:
                raise ValueError(f"Unknown model '{model}'. Provide flops= manually or use: {list(MODEL_FLOPS.keys())}")
            flops = MODEL_FLOPS[model]
        if mmlu is None:
            mmlu = MODEL_MMLU.get(model, 0.5)
        if humaneval is None:
            humaneval = MODEL_HUMANEVAL.get(model, 0.5)

    if flops is None or mmlu is None or humaneval is None:
        raise ValueError("Provide model name or all of: flops, mmlu, humaneval")

    if flops <= 1:
        raise ValueError("flops must be > 1 teraFLOP")

    i_score = information_score(mmlu, humaneval)
    c_log   = math.log2(flops)
    raw     = i_score / c_log

    # Normalize so human expert = 1.0 Lin
    human_i = information_score(MODEL_MMLU["human"], MODEL_HUMANEVAL["human"])
    human_c = math.log2(MODEL_FLOPS["human"])
    human_raw = human_i / human_c

    return round(raw / human_raw, 4)


def leaderboard() -> list:
    """
    Returns all known models sorted by Lin score (highest first).
    """
    from lin.models import list_models
    results = []
    for m in list_models():
        try:
            s = lin_score(m)
            results.append({
                "model":     m,
                "lin":       s,
                "mmlu":      MODEL_MMLU.get(m, None),
                "humaneval": MODEL_HUMANEVAL.get(m, None),
                "flops_t":   MODEL_FLOPS.get(m, None),
            })
        except Exception:
            pass
    return sorted(results, key=lambda x: x["lin"], reverse=True)
