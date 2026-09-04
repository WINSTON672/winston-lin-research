#!/usr/bin/env python3
"""Cross-check every quantitative manuscript claim against saved results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TEX = (ROOT / "main.tex").read_text(encoding="utf-8")
FRAME = pd.read_csv(ROOT / "results" / "summary_by_seed.csv")
HEADLINE = json.loads((ROOT / "results" / "headline_stats.json").read_text())


def assert_contains(text: str) -> None:
    assert text in TEX, f"missing or stale manuscript value: {text}"


def percent(value: float) -> str:
    return f"{100 * value:.2f}"


def ci_percent(entry: dict[str, object], metric: str) -> str:
    lo, hi = entry[f"{metric}_ci95"]
    return f"[{percent(float(lo))}, {percent(float(hi))}]"


def main() -> None:
    policies = HEADLINE["policies"]
    expected_primary = {
        "direct_redaction": "Direct identifiers removed",
        "date_mask_immediate": "Date masked; immediate",
        "date_mask_term": "Date masked; term-level",
        "cohortshield_k10": "CohortShield $k=10$",
        "cohortshield_k20": "CohortShield $k=20$",
    }
    for policy, label in expected_primary.items():
        entry = policies[policy]
        row = (
            f"{label} & {percent(entry['unique_rate'])} "
            f"{ci_percent(entry, 'unique_rate')} & "
            f"{percent(entry['below_10_rate'])} "
            f"{ci_percent(entry, 'below_10_rate')} & "
            f"{percent(entry['mean_guess_success'])} "
            f"{ci_percent(entry, 'mean_guess_success')} & "
            f"{float(entry['median_candidate_count']):.2f} "
            f"[{float(entry['median_candidate_count_ci95'][0]):.2f}, "
            f"{float(entry['median_candidate_count_ci95'][1]):.2f}] & "
            f"{percent(entry['mean_utility'])}"
        )
        assert_contains(row)

    primary = FRAME[
        (FRAME.school_size == 500)
        & (FRAME.hidden_clue_probability == 0.25)
    ]
    unique = primary.pivot(index="replicate", columns="policy", values="unique_rate")
    direct_mean = float(unique.direct_redaction.mean())
    for policy in ("cohortshield_k10", "cohortshield_k20"):
        reduction = (direct_mean - float(unique[policy].mean())) / direct_mean
        assert_contains(percent(reduction) + r"\%")

    mismatch = (
        FRAME[FRAME.school_size == 500]
        .groupby(["hidden_clue_probability", "policy"], sort=True)
        .mean(numeric_only=True)
    )
    checks = [
        (0.50, "cohortshield_k10", "target_met_rate"),
        (0.50, "cohortshield_k20", "target_met_rate"),
        (0.50, "cohortshield_k10", "unique_rate"),
        (0.50, "cohortshield_k20", "unique_rate"),
    ]
    for hidden_probability, policy, metric in checks:
        value = float(mismatch.loc[(hidden_probability, policy), metric])
        assert_contains(percent(value) + r"\%")

    size = (
        FRAME[FRAME.hidden_clue_probability == 0.25]
        .groupby(["school_size", "policy"], sort=True)
        .mean(numeric_only=True)
    )
    for policy in ("direct_redaction", "cohortshield_k10", "cohortshield_k20"):
        values = [percent(float(size.loc[(n, policy), "unique_rate"])) for n in (200, 500, 1000)]
        rendered = r"\%, ".join(values[:-1]) + r"\%, and " + values[-1] + r"\%"
        assert_contains(rendered)

    summary_hash = hashlib.sha256((ROOT / "results" / "summary_by_seed.csv").read_bytes()).hexdigest()
    headline_hash = hashlib.sha256((ROOT / "results" / "headline_stats.json").read_bytes()).hexdigest()
    assert_contains(summary_hash)
    assert_contains(headline_hash)
    assert int(HEADLINE["total_synthetic_narratives"]) == 135_000
    assert_contains("135,000 synthetic narratives")
    assert_contains("1,350 rows")

    forbidden = ("TODO", "TBD", "PLACEHOLDER", "??")
    for token in forbidden:
        assert token not in TEX, token

    print("MANUSCRIPT VERIFICATION PASSED")
    print(f"quantitative_claim_groups={5 + 2 + len(checks) + 3 + 4}")
    print(f"main_tex_sha256={hashlib.sha256(TEX.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
