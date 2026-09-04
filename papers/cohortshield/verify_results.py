#!/usr/bin/env python3
"""Independent integrity checks for the CohortShield result artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "results" / "summary_by_seed.csv"
HEADLINE_PATH = ROOT / "results" / "headline_stats.json"

POLICIES = (
    "direct_redaction",
    "date_mask_immediate",
    "date_mask_term",
    "cohortshield_k10",
    "cohortshield_k20",
)
METRICS = (
    "unique_rate",
    "below_5_rate",
    "below_10_rate",
    "mean_guess_success",
    "median_candidate_count",
    "mean_utility",
    "median_release_window",
    "target_met_rate",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frame = pd.read_csv(SUMMARY_PATH)
    headline = json.loads(HEADLINE_PATH.read_text(encoding="utf-8"))

    expected_rows = 3 * 3 * 30 * len(POLICIES)
    assert len(frame) == expected_rows, (len(frame), expected_rows)
    assert not frame.isna().any().any()
    key = ["hidden_clue_probability", "school_size", "replicate", "policy"]
    assert not frame.duplicated(key).any()
    assert set(frame.policy) == set(POLICIES)
    assert set(frame.school_size) == {200, 500, 1000}
    assert set(frame.hidden_clue_probability) == {0.0, 0.25, 0.5}
    assert set(frame.narratives) == {500}

    rate_columns = [
        "unique_rate",
        "below_5_rate",
        "below_10_rate",
        "mean_guess_success",
        "mean_utility",
        "target_met_rate",
    ]
    for column in rate_columns:
        assert frame[column].between(0, 1).all(), column
    assert (frame.below_5_rate <= frame.below_10_rate).all()

    # Text-date masking with same-day release cannot change the candidate set in
    # this declared model; the equality is an intentional falsifiable check.
    left = frame[frame.policy == "direct_redaction"].sort_values(key[:-1])
    right = frame[frame.policy == "date_mask_immediate"].sort_values(key[:-1])
    for metric in METRICS:
        assert np.allclose(left[metric], right[metric]), metric

    # With no missed clue, the policy must meet its nominal k constraint in every
    # replicate.  A failure indicates a bug in either optimization or reporting.
    no_mismatch = frame[frame.hidden_clue_probability == 0.0]
    for policy in ("cohortshield_k10", "cohortshield_k20"):
        block = no_mismatch[no_mismatch.policy == policy]
        assert np.allclose(block.target_met_rate, 1.0), policy
        assert np.allclose(block.unique_rate, 0.0), policy

    primary = frame[
        (frame.school_size == 500)
        & (frame.hidden_clue_probability == 0.25)
    ]
    for policy in POLICIES:
        block = primary[primary.policy == policy]
        assert len(block) == 30
        for metric in METRICS:
            observed = float(block[metric].mean())
            reported = float(headline["policies"][policy][metric])
            assert np.isclose(observed, reported, rtol=0, atol=1e-14), (
                policy,
                metric,
                observed,
                reported,
            )
            lo, hi = headline["policies"][policy][f"{metric}_ci95"]
            assert lo <= reported <= hi, (policy, metric, lo, reported, hi)

    primary_unique = {
        policy: float(primary[primary.policy == policy].unique_rate.mean())
        for policy in POLICIES
    }
    assert primary_unique["direct_redaction"] > primary_unique["date_mask_term"]
    assert primary_unique["date_mask_term"] > primary_unique["cohortshield_k10"]
    assert primary_unique["cohortshield_k10"] > primary_unique["cohortshield_k20"]

    print("VERIFICATION PASSED")
    print(f"summary_rows={len(frame)}")
    print(f"summary_sha256={sha256(SUMMARY_PATH)}")
    print(f"headline_sha256={sha256(HEADLINE_PATH)}")
    print("primary_unique_rates=" + json.dumps(primary_unique, sort_keys=True))


if __name__ == "__main__":
    main()
