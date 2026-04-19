"""
compare_ces.py
--------------
Computes Spearman ρ(TEB, CES) to validate that TEB and CES are related
but distinct metrics — a core claim of the unified framework.

CES is computed from scratch using the paper's formula for all 4 TEB models.
Two models (Llama-3-8B, Llama-3-70B) have CES values directly from the paper.
Two models (Qwen2.5-7B, Qwen2.5-72B, Llama-3.3-70B) are computed from
published MMLU/HumanEval scores in their respective technical reports.

Expected result for a valid contribution:
  ρ ≈ 0.5–0.8  →  related but distinct (TEB captures something CES doesn't)
  ρ ≈ 0.95+    →  TEB is redundant with CES (problem)
  ρ ≈ 0–0.3    →  measuring completely different things (also interesting)

Usage:
    python compare_ces.py teb_results_*_scored.json
"""

from __future__ import annotations
import json, math, sys, argparse
from itertools import combinations


# ── CES formula (from lin_framework_unified.tex) ─────────────────────────────

W_MMLU = 0.47
W_HE   = 0.53

# Human expert anchor: MMLU=89.8%, HumanEval=100%, FLOPs=1000T
HUMAN_MMLU = 0.898
HUMAN_HE   = 1.000
HUMAN_FLOPS = 1000.0


def bits_resolved(p: float) -> float:
    """
    Binary entropy reduction: 1 + p*log2(p) + (1-p)*log2(1-p)
    This is the k=2 form used in pilot_experiment.py.
    At p=0.5 (chance): returns 0. At p=1.0 (perfect): returns 1.
    Capped at p=0.9999 to avoid log(0).
    """
    if p <= 0.5:
        return 0.0
    p = min(p, 0.9999)
    return 1.0 + p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)


def compute_ces(mmlu: float, he: float, flops_T: float,
                human_raw_ces: float) -> float:
    """
    Normalized CES = raw_CES(model) / raw_CES(human_expert)
    raw_CES = I(M) / log2(FLOPs_T)
    I(M) = W_MMLU * bits_resolved(p_MMLU) + W_HE * bits_resolved(p_HE)
    """
    I_model = W_MMLU * bits_resolved(mmlu) + W_HE * bits_resolved(he)
    C_model = math.log2(1.0 + flops_T)
    raw = I_model / C_model
    return raw / human_raw_ces


# Pre-compute human anchor raw CES once
_I_human = W_MMLU * bits_resolved(HUMAN_MMLU) + W_HE * bits_resolved(HUMAN_HE)
_C_human = math.log2(1.0 + HUMAN_FLOPS)
HUMAN_RAW_CES = _I_human / _C_human


# ── Model benchmark scores ────────────────────────────────────────────────────
# Sources:
#   Llama-3-8B / Llama-3-70B : lin_framework_unified.tex Table 2 (paper values)
#   Llama-3.3-70B             : Meta release blog, Nov 2024
#   Qwen2.5-7B                : Qwen2.5 Technical Report, Sep 2024
#   Qwen2.5-72B               : Qwen2.5 Technical Report, Sep 2024

MODEL_BENCHMARKS = {
    # name: (mmlu_frac, humaneval_frac, flops_T, source)
    "Llama-3-8B": (
        0.684, 0.620, 8.0,
        "paper Table 2 (validated)"
    ),
    "Llama-3-70B": (
        0.820, 0.805, 70.0,
        "paper Table 2 (validated)"
    ),
    "Llama-3.3-70B": (
        0.860, 0.884, 70.0,
        "Meta Llama 3.3 release blog, Nov 2024"
    ),
    "Qwen2.5-7B": (
        0.742, 0.884, 7.0,
        "Qwen2.5 Technical Report, Sep 2024"
    ),
    "Qwen2.5-72B": (
        0.866, 0.905, 72.0,
        "Qwen2.5 Technical Report, Sep 2024"
    ),
}


def spearman_rho(xs: list, ys: list) -> float:
    """Spearman rank correlation for two equal-length lists."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def rank(seq):
        sorted_idx = sorted(range(n), key=lambda i: seq[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and seq[sorted_idx[j]] == seq[sorted_idx[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[sorted_idx[k]] = avg_rank
            i = j + 1
        return r

    rx = rank(xs)
    ry = rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n ** 2 - 1))


def main():
    parser = argparse.ArgumentParser(description="Compare TEB vs CES rankings")
    parser.add_argument("scored_file", help="Path to teb_results_*_scored.json")
    parser.add_argument("--teb-metric", choices=["teb_score", "teb_l2l3_score"],
                        default="teb_l2l3_score",
                        help="Which TEB metric to use (default: teb_l2l3_score)")
    args = parser.parse_args()

    with open(args.scored_file) as f:
        scored = json.load(f)

    teb_metric = args.teb_metric
    models_data = scored.get("models", {})

    print(f"\nCES vs TEB Comparison")
    print(f"TEB metric used: {teb_metric}")
    print(f"{'─'*65}\n")

    # Compute CES for all models in the scored file
    rows = []
    skipped = []
    for model_name, model_data in models_data.items():
        if model_name not in MODEL_BENCHMARKS:
            skipped.append(model_name)
            continue
        mmlu, he, flops, source = MODEL_BENCHMARKS[model_name]
        ces = compute_ces(mmlu, he, flops, HUMAN_RAW_CES)
        teb = model_data.get(teb_metric)
        if teb is None:
            skipped.append(f"{model_name} (no TEB score)")
            continue
        rows.append({
            "model": model_name,
            "ces": ces,
            "teb": teb,
            "flops": flops,
            "mmlu": mmlu,
            "he": he,
            "source": source,
        })

    if skipped:
        print(f"Skipped (no benchmark data): {', '.join(skipped)}")
        print()

    if len(rows) < 2:
        print("Need at least 2 models with both CES and TEB scores to compute ρ.")
        sys.exit(1)

    # Sort by CES for display
    rows.sort(key=lambda r: -r["ces"])

    # Assign ranks
    for metric in ("ces", "teb"):
        sorted_by = sorted(rows, key=lambda r: -r[metric])
        for rank_i, row in enumerate(sorted_by, 1):
            row[f"{metric}_rank"] = rank_i

    # Print table
    header = (f"{'Model':<18} {'FLOPs':>6} {'CES':>7} {'CES_rank':>9} "
              f"{'TEB':>8} {'TEB_rank':>9} {'Δrank':>6}")
    print(header)
    print("─" * len(header))
    for row in sorted(rows, key=lambda r: r["ces_rank"]):
        delta = row["ces_rank"] - row["teb_rank"]
        print(f"{row['model']:<18} {row['flops']:>5.0f}T "
              f"{row['ces']:>7.4f} {row['ces_rank']:>9} "
              f"{row['teb']:>8.4f} {row['teb_rank']:>9} "
              f"{delta:>+6}")

    # Spearman ρ
    ces_vals = [r["ces"] for r in rows]
    teb_vals = [r["teb"] for r in rows]
    rho = spearman_rho(ces_vals, teb_vals)

    print(f"\nSpearman ρ(CES, {teb_metric}) = {rho:.4f}  (n={len(rows)})")

    if math.isnan(rho):
        interpretation = "Cannot compute — insufficient data."
    elif rho >= 0.90:
        interpretation = "HIGH correlation: TEB rankings closely mirror CES. Risk of redundancy."
    elif rho >= 0.60:
        interpretation = "MODERATE correlation: related but distinct. Supports unified framework claim."
    elif rho >= 0.30:
        interpretation = "LOW-MODERATE: TEB captures meaningfully different information than CES."
    else:
        interpretation = "LOW/NEGATIVE: TEB and CES measure largely orthogonal properties."

    print(f"Interpretation: {interpretation}")

    # Show where rankings diverge
    divergences = [(r["model"], r["ces_rank"], r["teb_rank"])
                   for r in rows if abs(r["ces_rank"] - r["teb_rank"]) >= 2]
    if divergences:
        print(f"\nLargest divergences (|Δrank| ≥ 2):")
        for model, cr, tr in sorted(divergences, key=lambda x: -abs(x[1]-x[2])):
            direction = "CES underestimates" if tr < cr else "CES overestimates"
            print(f"  {model:<20} CES_rank={cr}, TEB_rank={tr}  → {direction} deployment efficiency")
    else:
        print(f"\nNo large rank divergences found (all |Δrank| < 2).")

    # Data sources note
    print(f"\nBenchmark sources:")
    for row in rows:
        marker = "✓ paper" if "validated" in row["source"] else "  external"
        print(f"  {marker}  {row['model']}: MMLU={row['mmlu']:.1%}, HE={row['he']:.1%} — {row['source']}")

    # Save
    out = {
        "teb_metric": teb_metric,
        "spearman_rho": rho,
        "n_models": len(rows),
        "rows": rows,
        "interpretation": interpretation,
    }
    out_file = args.scored_file.replace("_scored.json", "_ces_comparison.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
