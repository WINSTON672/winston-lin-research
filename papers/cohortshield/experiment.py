#!/usr/bin/env python3
"""Reproducible simulation for CohortShield.

The simulation studies identity disclosure from quasi-identifiers and release
timing in a small institutional community.  It intentionally uses no real
student records.  Every reported result is derived from deterministic,
synthetic populations generated from the configuration below.

Outputs are deliberately compact to limit disk writes:
  - results/summary_by_seed.csv
  - results/headline_stats.json
  - figures/risk_by_policy.pdf
  - figures/privacy_utility.pdf
  - figures/school_size_sensitivity.pdf
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MASTER_SEED = 20260904
SCHOOL_SIZES = (200, 500, 1000)
REPLICATES = 30
NARRATIVES_PER_REPLICATE = 500
TERM_DAYS = 120
WINDOWS = (1, 3, 7, 14, 30, 60, TERM_DAYS)
K_TARGETS = (10, 20)
HIDDEN_CLUE_PROBABILITIES = (0.0, 0.25, 0.50)

# Relative narrative utility of preserving each quasi-identifier.  These values
# are declared before analysis and are used only to choose among privacy-feasible
# transformations; privacy outcomes do not depend on the utility weights.
FIELD_WEIGHTS = {
    "activity": 3.0,
    "grade": 2.0,
    "residence": 2.0,
    "role": 1.0,
}
TIME_WEIGHT = {
    1: 4.0,
    3: 3.5,
    7: 3.0,
    14: 2.5,
    30: 2.0,
    60: 1.5,
    TERM_DAYS: 1.0,
}


@dataclass(frozen=True)
class Narrative:
    author: int
    event: int
    event_day: int
    exposed: tuple[str, ...]
    delay_quantile: float
    hidden_group_exposed: bool


@dataclass
class Community:
    people: pd.DataFrame
    event_activity: np.ndarray
    event_day: np.ndarray
    attendance: np.ndarray


def seed_for(size: int, replicate: int) -> int:
    payload = f"{MASTER_SEED}:{size}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_community(size: int, rng: np.random.Generator) -> Community:
    """Create a synthetic school roster, calendar, and attendance matrix."""
    n_activities = max(12, round(size / 25))
    n_residences = max(6, math.ceil(size / 28))

    grades = np.resize(np.arange(9, 13), size)
    rng.shuffle(grades)

    ranks = np.arange(1, n_activities + 1, dtype=float)
    activity_prob = ranks ** -0.8
    activity_prob /= activity_prob.sum()
    activities = rng.choice(n_activities, size=size, p=activity_prob)
    residences = rng.integers(0, n_residences, size=size)

    # A deliberately unmodelled auxiliary attribute.  It represents a small
    # advisory/course/peer cohort that an institutional attacker may infer from
    # a contextual clue missed by the release filter.  Assignment is balanced
    # within grade and independent of the public activity roster.
    hidden_group = np.empty(size, dtype=int)
    next_group = 0
    for grade in range(9, 13):
        members = np.flatnonzero(grades == grade)
        rng.shuffle(members)
        for start in range(0, len(members), 12):
            hidden_group[members[start : start + 12]] = next_group
            next_group += 1

    # Exactly one publicly identifiable leader per nonempty activity.
    leader = np.zeros(size, dtype=bool)
    for activity in range(n_activities):
        members = np.flatnonzero(activities == activity)
        if len(members):
            leader[rng.choice(members)] = True

    people = pd.DataFrame(
        {
            "person": np.arange(size),
            "grade": grades,
            "residence": residences,
            "activity": activities,
            "role": leader,
            "hidden_group": hidden_group,
        }
    )

    event_activities: list[int] = []
    event_days: list[int] = []
    attendee_rows: list[np.ndarray] = []
    for activity in range(n_activities):
        members = np.flatnonzero(activities == activity)
        if not len(members):
            continue
        first_day = int(rng.integers(0, 8))
        cadence = int(rng.integers(7, 15))
        days = list(range(first_day, TERM_DAYS, cadence))
        member_seen = np.zeros(len(members), dtype=bool)
        local_rows: list[np.ndarray] = []
        for _day in days:
            row = np.zeros(size, dtype=bool)
            attends = rng.random(len(members)) < 0.68
            if not attends.any():
                attends[rng.integers(0, len(members))] = True
            row[members[attends]] = True
            member_seen |= attends
            local_rows.append(row)
        # Ensure every person can author at least one event-linked narrative.
        for local_index in np.flatnonzero(~member_seen):
            local_rows[int(rng.integers(0, len(local_rows)))][members[local_index]] = True
        for day, row in zip(days, local_rows):
            event_activities.append(activity)
            event_days.append(day)
            attendee_rows.append(row)

    return Community(
        people=people,
        event_activity=np.asarray(event_activities, dtype=int),
        event_day=np.asarray(event_days, dtype=int),
        attendance=np.vstack(attendee_rows),
    )


def sample_narratives(
    community: Community,
    count: int,
    rng: np.random.Generator,
    hidden_clue_probability: float,
) -> list[Narrative]:
    valid_pairs = np.argwhere(community.attendance)
    chosen = valid_pairs[rng.integers(0, len(valid_pairs), size=count)]
    narratives: list[Narrative] = []
    for event, author in chosen:
        row = community.people.iloc[int(author)]
        exposed: list[str] = ["activity"] if rng.random() < 0.80 else []
        if rng.random() < 0.45:
            exposed.append("grade")
        if rng.random() < 0.22:
            exposed.append("residence")
        if bool(row.role) and rng.random() < 0.65:
            exposed.append("role")
        narratives.append(
            Narrative(
                author=int(author),
                event=int(event),
                event_day=int(community.event_day[event]),
                exposed=tuple(exposed),
                delay_quantile=float(rng.random()),
                hidden_group_exposed=bool(rng.random() < hidden_clue_probability),
            )
        )
    return narratives


def candidate_count(
    community: Community,
    narrative: Narrative,
    retained: frozenset[str],
    window: int,
    exact_text_date: bool,
    include_hidden_clue: bool = False,
) -> int:
    """Return candidates consistent with released fields and timing.

    When an exact event date remains in text, the attacker knows the source
    event day.  Otherwise a record is delayed uniformly within the declared
    release window, and the attacker considers events that could have preceded
    that public release.  At TERM_DAYS, the public timestamp is treated as term
    level and therefore reveals no within-term timing.
    """
    people_mask = np.ones(len(community.people), dtype=bool)
    author_row = community.people.iloc[narrative.author]
    for field in retained:
        people_mask &= community.people[field].to_numpy() == author_row[field]
    if include_hidden_clue and narrative.hidden_group_exposed:
        people_mask &= (
            community.people.hidden_group.to_numpy() == author_row.hidden_group
        )

    if exact_text_date:
        event_mask = community.event_day == narrative.event_day
    elif window == TERM_DAYS:
        event_mask = np.ones(len(community.event_day), dtype=bool)
    else:
        delay = min(window - 1, int(narrative.delay_quantile * window))
        release_day = narrative.event_day + delay
        event_mask = (community.event_day >= release_day - (window - 1)) & (
            community.event_day <= release_day
        )

    if "activity" in retained:
        event_mask &= community.event_activity == int(author_row.activity)

    time_candidates = community.attendance[event_mask].any(axis=0)
    count = int(np.count_nonzero(people_mask & time_candidates))
    # The construction guarantees the author is feasible.  Treat a violation as
    # a hard simulation error rather than silently repairing the number.
    if count < 1 or not (people_mask & time_candidates)[narrative.author]:
        raise AssertionError("author excluded from their own candidate set")
    return count


def utility(exposed: tuple[str, ...], retained: frozenset[str], window: int) -> float:
    field_total = sum(FIELD_WEIGHTS[f] for f in exposed)
    maximum = field_total + TIME_WEIGHT[1]
    kept = sum(FIELD_WEIGHTS[f] for f in retained) + TIME_WEIGHT[window]
    return kept / maximum


def adaptive_policy(
    community: Community, narrative: Narrative, k_target: int
) -> tuple[int, float, int, frozenset[str]]:
    """Maximize declared utility subject to candidate-set size >= k_target."""
    exposed = tuple(narrative.exposed)
    options: list[tuple[float, int, int, int]] = []
    for window in WINDOWS:
        for take in range(len(exposed) + 1):
            for fields in itertools.combinations(exposed, take):
                retained = frozenset(fields)
                count = candidate_count(
                    community, narrative, retained, window, exact_text_date=False
                )
                score = utility(exposed, retained, window)
                if count >= k_target:
                    options.append((score, -window, len(retained), count))
    if not options:
        # This should only occur if fewer than k people participated in any
        # event during the term.  It is reported as a policy failure.
        count = candidate_count(
            community, narrative, frozenset(), TERM_DAYS, exact_text_date=False
        )
        return count, utility(exposed, frozenset(), TERM_DAYS), TERM_DAYS, frozenset()
    best = max(options)
    score, neg_window, retained_count, count = best
    chosen_window = -neg_window
    # Recover the exact retained set using the same deterministic ordering and
    # tie-breaking criteria used above.
    selected: frozenset[str] | None = None
    for take in range(len(exposed) + 1):
        for fields in itertools.combinations(exposed, take):
            retained = frozenset(fields)
            if len(retained) != retained_count:
                continue
            this_count = candidate_count(
                community,
                narrative,
                retained,
                chosen_window,
                exact_text_date=False,
            )
            if this_count == count and utility(exposed, retained, chosen_window) == score:
                selected = retained
                break
        if selected is not None:
            break
    if selected is None:
        raise AssertionError("failed to recover adaptive policy decision")
    return count, score, chosen_window, selected


def evaluate_policy(
    community: Community, narratives: list[Narrative], policy: str
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for narrative in narratives:
        exposed = frozenset(narrative.exposed)
        if policy == "direct_redaction":
            nominal_count = candidate_count(
                community, narrative, exposed, 1, exact_text_date=True
            )
            count = candidate_count(
                community,
                narrative,
                exposed,
                1,
                exact_text_date=True,
                include_hidden_clue=True,
            )
            score, window, retained_count, target_met = 1.0, 1, len(exposed), 1
        elif policy == "date_mask_immediate":
            nominal_count = candidate_count(
                community, narrative, exposed, 1, exact_text_date=False
            )
            count = candidate_count(
                community,
                narrative,
                exposed,
                1,
                exact_text_date=False,
                include_hidden_clue=True,
            )
            score, window, retained_count, target_met = 1.0, 1, len(exposed), 1
        elif policy == "date_mask_term":
            nominal_count = candidate_count(
                community, narrative, exposed, TERM_DAYS, exact_text_date=False
            )
            count = candidate_count(
                community,
                narrative,
                exposed,
                TERM_DAYS,
                exact_text_date=False,
                include_hidden_clue=True,
            )
            score = utility(narrative.exposed, exposed, TERM_DAYS)
            window, retained_count, target_met = TERM_DAYS, len(exposed), 1
        elif policy.startswith("cohortshield_k"):
            target = int(policy.rsplit("k", 1)[1])
            nominal_count, score, window, retained = adaptive_policy(
                community, narrative, target
            )
            count = candidate_count(
                community,
                narrative,
                retained,
                window,
                exact_text_date=False,
                include_hidden_clue=True,
            )
            retained_count = len(retained)
            target_met = int(count >= target)
        else:
            raise ValueError(policy)
        rows.append(
            {
                "candidate_count": count,
                "nominal_candidate_count": nominal_count,
                "unique": int(count == 1),
                "below_5": int(count < 5),
                "below_10": int(count < 10),
                "guess_success": 1.0 / count,
                "utility": score,
                "window": window,
                "retained_fields": retained_count,
                "target_met": target_met,
            }
        )
    return pd.DataFrame(rows)


def summarize_frame(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "unique_rate": float(frame.unique.mean()),
        "below_5_rate": float(frame.below_5.mean()),
        "below_10_rate": float(frame.below_10.mean()),
        "mean_guess_success": float(frame.guess_success.mean()),
        "median_candidate_count": float(frame.candidate_count.median()),
        "mean_utility": float(frame.utility.mean()),
        "median_release_window": float(frame.window.median()),
        "mean_retained_fields": float(frame.retained_fields.mean()),
        "target_met_rate": float(frame.target_met.mean()),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    policies = (
        "direct_redaction",
        "date_mask_immediate",
        "date_mask_term",
        "cohortshield_k10",
        "cohortshield_k20",
    )
    summaries: list[dict[str, float | int | str]] = []
    for hidden_probability in HIDDEN_CLUE_PROBABILITIES:
        for size in SCHOOL_SIZES:
            for replicate in range(REPLICATES):
                seed = seed_for(size, replicate)
                rng = np.random.default_rng(seed)
                community = build_community(size, rng)
                narratives = sample_narratives(
                    community,
                    NARRATIVES_PER_REPLICATE,
                    rng,
                    hidden_clue_probability=hidden_probability,
                )
                for policy in policies:
                    result = evaluate_policy(community, narratives, policy)
                    summaries.append(
                        {
                            "school_size": size,
                            "hidden_clue_probability": hidden_probability,
                            "replicate": replicate,
                            "seed": seed,
                            "policy": policy,
                            "narratives": len(result),
                            **summarize_frame(result),
                        }
                    )

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS / "summary_by_seed.csv", index=False)

    primary = summary[
        (summary.school_size == 500)
        & (summary.hidden_clue_probability == 0.25)
    ]
    headline: dict[str, object] = {
        "master_seed": MASTER_SEED,
        "school_sizes": list(SCHOOL_SIZES),
        "replicates_per_size": REPLICATES,
        "narratives_per_replicate": NARRATIVES_PER_REPLICATE,
        "hidden_clue_probabilities": list(HIDDEN_CLUE_PROBABILITIES),
        "total_synthetic_narratives": len(HIDDEN_CLUE_PROBABILITIES)
        * len(SCHOOL_SIZES)
        * REPLICATES
        * NARRATIVES_PER_REPLICATE,
        "primary_school_size": 500,
        "primary_hidden_clue_probability": 0.25,
        "primary_narratives": REPLICATES * NARRATIVES_PER_REPLICATE,
        "policies": {},
    }
    for index, policy in enumerate(policies):
        block = primary[primary.policy == policy]
        entry: dict[str, object] = {}
        for metric in (
            "unique_rate",
            "below_5_rate",
            "below_10_rate",
            "mean_guess_success",
            "median_candidate_count",
            "mean_utility",
            "median_release_window",
            "target_met_rate",
        ):
            values = block[metric].to_numpy(float)
            entry[metric] = float(values.mean())
            entry[f"{metric}_ci95"] = bootstrap_mean_ci(
                values, MASTER_SEED + 100 + index * 10 + len(metric)
            )
        headline["policies"][policy] = entry
    (RESULTS / "headline_stats.json").write_text(
        json.dumps(headline, indent=2) + "\n", encoding="utf-8"
    )

    labels = {
        "direct_redaction": "Direct identifiers\nremoved",
        "date_mask_immediate": "Date masked,\nimmediate release",
        "date_mask_term": "Date masked,\nterm-level release",
        "cohortshield_k10": "CohortShield\n$k=10$",
        "cohortshield_k20": "CohortShield\n$k=20$",
    }
    colors = ["#8C2F39", "#B35C44", "#D4A259", "#477998", "#2E6F57"]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    means = [
        primary[primary.policy == p].unique_rate.mean() * 100 for p in policies
    ]
    cis = [
        bootstrap_mean_ci(
            primary[primary.policy == p].unique_rate.to_numpy(float),
            MASTER_SEED + 300 + i,
        )
        for i, p in enumerate(policies)
    ]
    errors = np.array(
        [[m - lo * 100, hi * 100 - m] for m, (lo, hi) in zip(means, cis)]
    ).T
    x = np.arange(len(policies))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors, width=0.72)
    ax.set_xticks(x, [labels[p] for p in policies])
    ax.set_ylabel("Uniquely re-identifiable narratives (%)")
    ax.set_title("Primary simulation: 500-person community, 15,000 narratives")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_by_policy.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for p, c in zip(policies, colors):
        block = primary[primary.policy == p]
        ax.scatter(
            block.mean_utility.mean() * 100,
            block.mean_guess_success.mean() * 100,
            s=75,
            color=c,
            label=labels[p].replace("\n", " "),
        )
    ax.set_xlabel("Mean retained utility (%)")
    ax.set_ylabel("Uniform-attacker success probability (%)")
    ax.set_title("Privacy-utility operating points")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "privacy_utility.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for p, c in zip(policies, colors):
        grouped = summary[
            (summary.policy == p) & (summary.hidden_clue_probability == 0.25)
        ].groupby("school_size").unique_rate
        means = grouped.mean() * 100
        ax.plot(means.index, means.values, marker="o", color=c, label=labels[p].replace("\n", " "))
    ax.set_xlabel("Community size")
    ax.set_ylabel("Uniquely re-identifiable narratives (%)")
    ax.set_title("Sensitivity to community size")
    ax.set_xticks(SCHOOL_SIZES)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "school_size_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for p, c in zip(policies, colors):
        grouped = summary[
            (summary.policy == p) & (summary.school_size == 500)
        ].groupby("hidden_clue_probability").below_10_rate
        means = grouped.mean() * 100
        ax.plot(
            means.index * 100,
            means.values,
            marker="o",
            color=c,
            label=labels[p].replace("\n", " "),
        )
    ax.set_xlabel("Narratives containing an unmodelled group clue (%)")
    ax.set_ylabel("Narratives with fewer than 10 candidates (%)")
    ax.set_title("Sensitivity to auxiliary-information mismatch")
    ax.set_xticks(np.asarray(HIDDEN_CLUE_PROBABILITIES) * 100)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_mismatch_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    run()
