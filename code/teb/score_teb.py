"""
score_teb.py
------------
Task Efficiency Benchmark (TEB) — Scorer

Loads raw results from run_teb.py, scores each response, and computes:
  - quality_ratio:  task_score / max_score  per model per task
  - token_efficiency:  quality_ratio / (output_tokens / 500)
  - speedup:  human_baseline_minutes * 60 / response_duration_s
  - TEB score:  mean_quality_ratio / log2(FLOPs)  (analogous to CES)

Automated scoring for T1 tasks; interactive rubric scoring for T2/T3.
Use --auto-rubric to have scores prompted per criterion interactively.
Use --llm-judge to pass rubric to a judge model (requires JUDGE_TOKEN env var).

Usage:
    python score_teb.py teb_results_20260329_120000.json
    python score_teb.py teb_results_20260329_120000.json --auto-rubric
"""

from __future__ import annotations
import os, sys, re, json, math, ast, tempfile, subprocess, textwrap, argparse
from datetime import datetime


TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.json")


# ── Automated scorers ─────────────────────────────────────────────────────────

def score_T1_01(response_text: str, task: dict) -> tuple[float, str]:
    """Parse JSON array, check each of 15 records."""
    ground_truth = task["scoring"]["ground_truth"]
    max_score = task["scoring"]["max_score"]

    if not response_text:
        return 0.0, "No response"

    # Extract JSON array from response (strip markdown fences if present)
    text = response_text.strip()
    json_match = re.search(r'\[[\s\S]*\]', text)
    if not json_match:
        return 0.0, "No JSON array found in response"

    try:
        records = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return 0.0, f"Invalid JSON: {e}"

    if not isinstance(records, list):
        return 0.0, "Response is not a JSON array"

    score = 0
    notes = []

    for gt in ground_truth:
        # Find matching record by id
        match = next((r for r in records if isinstance(r, dict) and r.get("id") == gt["id"]), None)
        if not match:
            notes.append(f"  Missing record id={gt['id']}")
            continue

        record_ok = True

        # Check name
        name = match.get("name", {})
        if not isinstance(name, dict):
            notes.append(f"  id={gt['id']}: name is not an object")
            record_ok = False
        else:
            if name.get("first", "").strip() != gt["name"]["first"]:
                notes.append(f"  id={gt['id']}: name.first wrong (got '{name.get('first')}', expected '{gt['name']['first']}')")
                record_ok = False
            if name.get("last", "").strip() != gt["name"]["last"]:
                notes.append(f"  id={gt['id']}: name.last wrong (got '{name.get('last')}', expected '{gt['name']['last']}')")
                record_ok = False

        # Check email lowercase
        email = match.get("email", "")
        if email.lower() != gt["email"]:
            notes.append(f"  id={gt['id']}: email wrong (got '{email}', expected '{gt['email']}')")
            record_ok = False

        # Check ISO 8601 date
        joined = match.get("joined", "")
        if joined != gt["joined"]:
            notes.append(f"  id={gt['id']}: joined wrong (got '{joined}', expected '{gt['joined']}')")
            record_ok = False

        if record_ok:
            score += 1

    detail = f"{score}/{max_score} records correct"
    if notes:
        detail += "\nIssues:\n" + "\n".join(notes[:5])
        if len(notes) > 5:
            detail += f"\n  ... and {len(notes)-5} more"

    return score / max_score, detail


def score_T2_01_automated(response_text: str, task: dict) -> tuple[float, str]:
    """
    Extract the corrected function from the response and run 3 test cases.
    Each bug gets 3 points: 1 for identifying it (keyword check) + 2 for passing the test.
    """
    if not response_text:
        return 0.0, "No response"

    bugs = task["scoring"]["bugs"]
    max_score = task["scoring"]["max_score"]

    # Try to extract a Python function from the response
    # Look for ```python ... ``` blocks first, then bare def blocks
    code_block = re.search(r'```python\s*(.*?)```', response_text, re.DOTALL)
    if code_block:
        candidate_code = code_block.group(1)
    else:
        # Try to find def process_orders(...) in the response
        func_match = re.search(r'(def process_orders\(.*?)\Z', response_text, re.DOTALL)
        if func_match:
            candidate_code = func_match.group(1)
        else:
            # Fall back: use the whole response
            candidate_code = response_text

    # Run each test case in a subprocess for safety
    test_code = textwrap.dedent("""
    import sys
    exec(open('/tmp/teb_func.py').read())

    results = {}

    # Test bug 1: premium surcharge (should be +15%, not -15%)
    orders1 = [{'product': 'A', 'quantity': 10, 'unit_price': 100, 'is_premium': True}]
    try:
        val = process_orders(orders1)
        results['bug1_test'] = abs(val - 1150.0) < 0.01
        results['bug1_val'] = val
    except Exception as e:
        results['bug1_test'] = False
        results['bug1_val'] = str(e)

    # Test bug 2: off-by-one (>= threshold, not >)
    orders2 = [{'product': 'A', 'quantity': 1, 'unit_price': 100, 'is_premium': False}] * 10
    try:
        val = process_orders(orders2)
        results['bug2_test'] = abs(val - 950.0) < 0.01
        results['bug2_val'] = val
    except Exception as e:
        results['bug2_test'] = False
        results['bug2_val'] = str(e)

    # Test bug 3: return total not average
    orders3 = [
        {'product': 'A', 'quantity': 2, 'unit_price': 50, 'is_premium': False},
        {'product': 'B', 'quantity': 3, 'unit_price': 30, 'is_premium': False},
    ]
    try:
        val = process_orders(orders3)
        results['bug3_test'] = abs(val - 190.0) < 0.01
        results['bug3_val'] = val
    except Exception as e:
        results['bug3_test'] = False
        results['bug3_val'] = str(e)

    import json
    print(json.dumps(results))
    """)

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(candidate_code)
            tmp_path = f.name
        test_code = test_code.replace('/tmp/teb_func.py', tmp_path)
        proc = subprocess.run(
            [sys.executable, '-c', test_code],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            exec_results = {"bug1_test": False, "bug2_test": False, "bug3_test": False}
            exec_note = f"Execution error: {proc.stderr[:200]}"
        else:
            exec_results = json.loads(proc.stdout.strip())
            exec_note = ""
    except Exception as e:
        exec_results = {"bug1_test": False, "bug2_test": False, "bug3_test": False}
        exec_note = f"Test runner error: {e}"

    # Also check if bugs are identified in the text (keyword check for 1 bonus point each)
    text_lower = response_text.lower()
    bug_identified = {
        "bug1": any(kw in text_lower for kw in
                    ["surcharge", "premium", "1 - ", "1-", "should be +", "1 + ", "1+premium"]),
        "bug2": any(kw in text_lower for kw in
                    [">= bulk", ">=bulk", ">= 10", "off-by-one", "off by one", "should be >=", "greater than or equal"]),
        "bug3": any(kw in text_lower for kw in
                    ["return total", "average", "divide", "/ len", "should not divide", "should return total"]),
    }

    score = 0
    notes = []
    for i, (bug_key, bug) in enumerate([("bug1", bugs[0]), ("bug2", bugs[1]), ("bug3", bugs[2])]):
        test_passed = exec_results.get(f"{bug_key}_test", False)
        identified = bug_identified[bug_key]
        bug_score = 0
        if identified:
            bug_score += 1
        if test_passed:
            bug_score += 2
        score += bug_score
        notes.append(f"  {bug_key}: identified={identified}, test_passed={test_passed} → {bug_score}/3 pts")

    detail = f"{score}/{max_score} points\n" + "\n".join(notes)
    if exec_note:
        detail += f"\n  {exec_note}"
    if exec_results.get("bug1_val"):
        detail += f"\n  Test values: bug1={exec_results.get('bug1_val')}, bug2={exec_results.get('bug2_val')}, bug3={exec_results.get('bug3_val')}"

    return score / max_score, detail


def score_T2_03_automated(response_text: str, task: dict) -> tuple[float, str]:
    """Check if each of the 3 anomalies is identified and explained."""
    if not response_text:
        return 0.0, "No response"

    anomalies = task["scoring"]["anomalies"]
    max_score = task["scoring"]["max_score"]
    text_lower = response_text.lower()

    score = 0
    notes = []

    # Anomaly 1: 2024-02 East Widget-A total mismatch (10057.50 vs 1057.50)
    # Require BOTH: the specific wrong value AND evidence it's a calculation/mismatch issue
    a1_value_mentioned = "10057" in response_text or ("1057" in response_text and "10057" not in response_text)
    a1_is_math_error = any(kw in text_lower for kw in
                           ["mismatch", "doesn't match", "does not match", "should be", "calculation",
                            "45 ×", "45 *", "45×", "× 23", "* 23"])
    a1_found = a1_value_mentioned and a1_is_math_error
    a1_correct_val = ("1057.50" in response_text or "1,057.50" in response_text or
                      "1057.5" in response_text)
    a1_score = (2 if a1_found else 0) + (1 if a1_correct_val else 0)
    score += a1_score
    notes.append(f"  Anomaly 1 (total mismatch): found={a1_found}, correct_val={a1_correct_val} → {a1_score}/3")

    # Anomaly 2: negative units 2024-04 South Widget-B
    # Require: mentions negative/-120 AND connects it to units (not just revenue)
    a2_negative = "-120" in response_text or "negative" in text_lower
    a2_units_context = any(kw in text_lower for kw in
                           ["units_sold", "units sold", "negative units", "quantity"])
    a2_found = a2_negative and a2_units_context
    a2_correct_val = any(kw in response_text for kw in ["120", "+120"]) and any(
        kw in text_lower for kw in ["positive", "should be 120", "return", "refund", "correction"])
    a2_score = (2 if a2_found else 0) + (1 if a2_correct_val else 0)
    score += a2_score
    notes.append(f"  Anomaly 2 (negative units): found={a2_found}, correct_val={a2_correct_val} → {a2_score}/3")

    # Anomaly 3: unit price 18500 for Widget-A 2024-05 West
    a3_found = any(kw in text_lower for kw in
                   ["18500", "18,500", "2024-05", "west", "unit price", "unit_price",
                    "price", "1,628,000", "1628000", "outlier", "unusual"])
    a3_correct_val = any(kw in response_text for kw in ["23.50", "23.5", "18.50"])
    a3_score = (2 if a3_found else 0) + (1 if a3_correct_val else 0)
    score += a3_score
    notes.append(f"  Anomaly 3 (wrong unit price): found={a3_found}, correct_val={a3_correct_val} → {a3_score}/3")

    detail = f"{score}/{max_score} points\n" + "\n".join(notes)
    return score / max_score, detail


# ── Rubric-based interactive scoring ─────────────────────────────────────────

def score_rubric_interactive(response_text: str, task: dict) -> tuple[float, str]:
    """Print response and criteria; ask human to score each criterion."""
    task_id = task["id"]
    max_score = task["scoring"]["max_score"]
    criteria = task["scoring"]["criteria"]

    print(f"\n{'='*70}")
    print(f"SCORING: {task_id} — {task['title']}")
    print(f"{'='*70}")
    print("\n--- MODEL RESPONSE ---")
    print(response_text or "(no response)")
    print("\n--- SCORING CRITERIA ---")

    total_score = 0
    criterion_scores = []

    for c in criteria:
        print(f"\n[{c['id'].upper()}] {c['points']} points max")
        print(f"  {c['description']}")
        while True:
            try:
                raw = input(f"  Score (0-{c['points']}): ").strip()
                pts = float(raw)
                if 0 <= pts <= c["points"]:
                    total_score += pts
                    criterion_scores.append(f"{c['id']}: {pts}/{c['points']}")
                    break
                else:
                    print(f"  Enter a value between 0 and {c['points']}")
            except (ValueError, KeyboardInterrupt):
                print("  Skipping (0 points)")
                criterion_scores.append(f"{c['id']}: 0/{c['points']}")
                break

    # Revision burden: how much would a human need to edit this output?
    print(f"\n[REVISION BURDEN]")
    print(f"  0 = use as-is, 1 = minor edits (wording, formatting), 2 = major rework needed")
    revision_burden = 0
    while True:
        try:
            raw = input(f"  Revision burden (0/1/2): ").strip()
            rb = int(raw)
            if rb in (0, 1, 2):
                revision_burden = rb
                break
            else:
                print("  Enter 0, 1, or 2")
        except (ValueError, KeyboardInterrupt):
            break

    # Apply revision penalty: 0→×1.0, 1→×0.80, 2→×0.50
    revision_multipliers = {0: 1.0, 1: 0.80, 2: 0.50}
    raw_ratio = total_score / max_score
    effective_ratio = raw_ratio * revision_multipliers[revision_burden]

    detail = (f"{total_score}/{max_score} (revision_burden={revision_burden}, "
              f"effective_quality={effective_ratio:.3f})\n" + ", ".join(criterion_scores))
    return effective_ratio, detail


def score_rubric_auto_zero(response_text: str, task: dict) -> tuple[float, str]:
    """For non-interactive runs: return None to indicate needs manual scoring."""
    return None, "NEEDS_MANUAL_SCORING"


# ── Dispatch ──────────────────────────────────────────────────────────────────

AUTOMATED_SCORERS = {
    "T1-01": score_T1_01,
    "T2-01": score_T2_01_automated,
    "T2-03": score_T2_03_automated,
}

RUBRIC_TASKS = {"T1-02", "T1-03", "T2-02", "T2-04", "T3-01", "T3-02", "T3-03"}


def score_response(task_id: str, response_text: str, task: dict,
                   interactive: bool = True) -> tuple[float | None, str]:
    if task_id in AUTOMATED_SCORERS:
        return AUTOMATED_SCORERS[task_id](response_text, task)
    elif task_id in RUBRIC_TASKS:
        if interactive:
            return score_rubric_interactive(response_text, task)
        else:
            return score_rubric_auto_zero(response_text, task)
    else:
        return None, f"No scorer implemented for {task_id}"


# ── TEB score computation ─────────────────────────────────────────────────────

def compute_teb(quality_ratios: dict, response_times: dict) -> float:
    """
    TEB(M) = mean_quality_ratio / log2(1 + mean_response_time_s)

    Denominator is measured response time, not estimated FLOPs.
    Rationale: TEB asks "quality per unit of human wait time" — the cost
    a user actually experiences. CES covers quality per compute unit.
    Together they are complementary: CES is intrinsic, TEB is deployment-facing.

    Higher TEB = faster and/or higher quality.
    """
    task_ids = [tid for tid, v in quality_ratios.items() if v is not None]
    if not task_ids:
        return 0.0
    mean_quality = sum(quality_ratios[tid] for tid in task_ids) / len(task_ids)
    mean_time = sum(response_times.get(tid, 1.0) for tid in task_ids) / len(task_ids)
    return mean_quality / math.log2(1 + max(mean_time, 0.1))


def compute_token_efficiency(quality_ratio: float | None, output_tokens: int) -> float | None:
    """quality per 500-token chunk of output. Higher = more concise."""
    if quality_ratio is None or output_tokens == 0:
        return None
    return quality_ratio / max(output_tokens / 500, 0.1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score TEB results")
    parser.add_argument("results_file", help="Path to teb_results_*.json from run_teb.py")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive rubric scoring (mark as NEEDS_MANUAL_SCORING)")
    args = parser.parse_args()

    interactive = not args.no_interactive

    with open(args.results_file) as f:
        results = json.load(f)

    with open(TASKS_FILE) as f:
        task_data = json.load(f)

    tasks_by_id = {t["id"]: t for t in task_data["tasks"]}
    task_ids_in_results = results.get("tasks_run", [])

    print(f"\nTEB Scorer — loading {args.results_file}")
    print(f"Tasks: {task_ids_in_results}")
    print(f"Models: {list(results['models'].keys())}")

    scored = {}  # model_name → {task_id → {quality_ratio, token_efficiency, speedup, detail}}

    for model_name, model_data in results["models"].items():
        print(f"\n{'─'*60}")
        print(f"Model: {model_name}  (FLOPs: {model_data['flops_T']}T)")
        print(f"{'─'*60}")

        scored[model_name] = {
            "flops_T": model_data["flops_T"],
            "tasks": {}
        }

        for task_id, resp in model_data["responses"].items():
            task = tasks_by_id.get(task_id)
            if not task:
                continue

            if resp.get("error"):
                print(f"  {task_id}: ERROR — {resp['error']}")
                scored[model_name]["tasks"][task_id] = {
                    "quality_ratio": None,
                    "token_efficiency": None,
                    "speedup": None,
                    "detail": f"ERROR: {resp['error']}",
                }
                continue

            print(f"\n  {task_id} — {task['title']}")
            print(f"  output_tokens={resp.get('output_tokens', 0)}, "
                  f"duration={resp.get('duration_s', 0):.1f}s, "
                  f"speedup={resp.get('speedup_vs_human', 0):.0f}x")

            quality_ratio, detail = score_response(
                task_id, resp.get("response_text", ""), task, interactive=interactive
            )

            token_eff = compute_token_efficiency(quality_ratio, resp.get("output_tokens", 0))
            speedup = resp.get("speedup_vs_human")

            if quality_ratio is not None:
                print(f"  Quality ratio: {quality_ratio:.3f}  Token efficiency: {token_eff:.4f}")

            scored[model_name]["tasks"][task_id] = {
                "quality_ratio": quality_ratio,
                "token_efficiency": token_eff,
                "speedup_vs_human": speedup,
                "output_tokens": resp.get("output_tokens", 0),
                "duration_s": resp.get("duration_s", 0),
                "scoring_detail": detail,
            }

    # Determine which task IDs belong to which tier
    tier_map = {t["id"]: t["tier"] for t in task_data["tasks"]}
    human_baselines = {t["id"]: t["human_baseline_minutes"] * 60
                       for t in task_data["tasks"]}  # seconds

    # Human normalization baseline:
    # A human scores quality_ratio=1.0 on every task (they define the ground truth)
    # and takes human_baseline_s seconds. TEB_human per task = 1.0 / log2(1 + baseline_s).
    # TEB_normalized = TEB(M) / mean(TEB_human across tasks in this run).
    def human_teb_for_tasks(task_ids: list) -> float:
        vals = [1.0 / math.log2(1 + human_baselines[tid])
                for tid in task_ids if tid in human_baselines]
        return sum(vals) / len(vals) if vals else 1.0

    # Compute TEB (all tasks) and TEB_L2L3 (tier 2+3 only)
    print(f"\n\n{'='*60}")
    print("TASK EFFICIENCY BENCHMARK — RESULTS")
    print(f"{'='*60}\n")

    teb_scores = {}
    teb_l2l3_scores = {}
    for model_name, model_scored in scored.items():
        all_ratios = {
            tid: v["quality_ratio"]
            for tid, v in model_scored["tasks"].items()
        }
        all_times = {
            tid: v["duration_s"]
            for tid, v in model_scored["tasks"].items()
        }
        l2l3_ratios = {
            tid: qr for tid, qr in all_ratios.items()
            if tier_map.get(tid, 1) >= 2
        }
        l2l3_times = {
            tid: t for tid, t in all_times.items()
            if tier_map.get(tid, 1) >= 2
        }
        teb_raw = compute_teb(all_ratios, all_times)
        teb_l2l3_raw = compute_teb(l2l3_ratios, l2l3_times)

        # Normalize to human baseline (1.0 = human expert speed/quality tradeoff)
        human_anchor_all = human_teb_for_tasks(list(all_ratios.keys()))
        human_anchor_l2l3 = human_teb_for_tasks(list(l2l3_ratios.keys()))
        teb = teb_raw / human_anchor_all if human_anchor_all else teb_raw
        teb_l2l3 = teb_l2l3_raw / human_anchor_l2l3 if human_anchor_l2l3 else teb_l2l3_raw

        teb_scores[model_name] = teb
        teb_l2l3_scores[model_name] = teb_l2l3
        model_scored["teb_score"] = teb
        model_scored["teb_l2l3_score"] = teb_l2l3
        model_scored["human_anchor_all"] = human_anchor_all
        model_scored["human_anchor_l2l3"] = human_anchor_l2l3

    # Print summary table (TEB > 1.0 = faster+better than human baseline)
    header = f"{'Model':<16} {'FLOPs':>6} {'TEB(all)':>9} {'TEB(L2+L3)':>11} {'AvgQual':>7} {'AvgSpeed':>9}"
    print(header)
    print("─" * len(header))

    for model_name in sorted(teb_scores, key=lambda m: -teb_scores[m]):
        ms = scored[model_name]
        teb = ms["teb_score"]
        teb_l2l3 = ms["teb_l2l3_score"]
        flops = ms["flops_T"]
        tasks_data = ms["tasks"]

        valid_quality = [v["quality_ratio"] for v in tasks_data.values()
                         if v["quality_ratio"] is not None]
        avg_quality = sum(valid_quality) / len(valid_quality) if valid_quality else 0

        speedups = [v["speedup_vs_human"] for v in tasks_data.values()
                    if v["speedup_vs_human"] is not None]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0

        print(f"{model_name:<16} {flops:>5.0f}T {teb:>9.4f} {teb_l2l3:>11.4f} {avg_quality:>7.3f} {avg_speedup:>8.0f}x")

    print(f"\nAnchor: TEB = 1.0 means equivalent to human expert speed/quality tradeoff.")
    print(f"        TEB > 1.0 = AI is more efficient than human; TEB < 1.0 = not worth using.")
    print(f"(TEB_L2+L3 excludes Tier-1 tasks where model size doesn't differentiate)")

    # Per-task breakdown
    print(f"\n\nPer-Task Quality Ratios:")
    task_header = f"{'Task':<8}" + "".join(f"{m[:10]:>12}" for m in scored)
    print(task_header)
    print("─" * len(task_header))

    all_task_ids = sorted({tid for ms in scored.values() for tid in ms["tasks"]})
    for tid in all_task_ids:
        row = f"{tid:<8}"
        for model_name in scored:
            val = scored[model_name]["tasks"].get(tid, {}).get("quality_ratio")
            row += f"{'N/A':>12}" if val is None else f"{val:>12.3f}"
        print(row)

    # Save scored results
    out_file = args.results_file.replace(".json", "_scored.json")
    with open(out_file, "w") as f:
        json.dump({
            "source": args.results_file,
            "scored_at": datetime.now().isoformat(),
            "teb_scores": teb_scores,
            "teb_l2l3_scores": teb_l2l3_scores,
            "models": scored,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nScored results saved to: {out_file}")


if __name__ == "__main__":
    main()
