"""
Run all three FLUID stability experiments and print a final summary.

Usage:
    python run_all.py              # run all experiments
    python run_all.py --exp 1      # run only experiment 1
    python run_all.py --exp 2 3    # run experiments 2 and 3
    python run_all.py --workers 4  # limit CPU workers for experiment 2
"""

import argparse
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_exp1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Small-World Convergence")
    print("=" * 60)
    from experiment1 import run_all
    return run_all()


def run_exp2(n_workers=None):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Phase Diagram")
    print("=" * 60)
    from experiment2 import run_phase_diagram, print_2d_slice
    results = run_phase_diagram(n_workers=n_workers)
    print_2d_slice(results, T_prune_fixed=500)
    return results


def run_exp3():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Lyapunov Candidate")
    print("=" * 60)
    from experiment3 import run_lyapunov_test
    return run_lyapunov_test()


def print_final_summary():
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Experiment 1
    exp1_path = os.path.join(RESULTS_DIR, 'exp1_summary.json')
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            exp1 = json.load(f)
        print("\nExperiment 1 — Convergence:")
        for task, r in exp1.items():
            if isinstance(r, dict):
                print(f"  {task}: converged={r.get('converged')} "
                      f"edges={r.get('final_n_edges')} "
                      f"clustering={r.get('final_clustering', 0):.4f}")

    # Experiment 2
    exp2_path = os.path.join(RESULTS_DIR, 'exp2_phase_diagram.json')
    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2 = json.load(f)
        counts = exp2.get('counts', {})
        total = exp2.get('total', 1)
        print("\nExperiment 2 — Phase Diagram:")
        for outcome, cnt in counts.items():
            print(f"  {outcome}: {cnt}/{total} ({100*cnt/total:.1f}%)")

    # Experiment 3
    exp3_path = os.path.join(RESULTS_DIR, 'exp3_lyapunov.json')
    if os.path.exists(exp3_path):
        with open(exp3_path) as f:
            exp3 = json.load(f)
        frac = exp3.get('fraction_lyapunov_holds', 0.0)
        n_dec = exp3.get('n_decreasing', 0)
        total = exp3.get('total', 0)
        print(f"\nExperiment 3 — Lyapunov: "
              f"V non-increasing in {n_dec}/{total} ({100*frac:.1f}%) stable configs")

    # Verdict
    exp1_all_converged = False
    stable_region_exists = False
    lyapunov_holds = False

    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            exp1 = json.load(f)
        exp1_all_converged = all(
            r.get('converged', False)
            for r in exp1.values()
            if isinstance(r, dict)
        )

    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2 = json.load(f)
        stable_region_exists = exp2.get('counts', {}).get('stable', 0) > 0

    if os.path.exists(exp3_path):
        with open(exp3_path) as f:
            exp3 = json.load(f)
        lyapunov_holds = exp3.get('fraction_lyapunov_holds', 0.0) > 0.5

    print("\n--- Conjecture Verdict ---")
    if exp1_all_converged and stable_region_exists:
        print("CONJECTURE: SUPPORTED")
        if lyapunov_holds:
            print("LYAPUNOV: V = H_t/(|E|+1) satisfies non-increasing condition (>50% configs)")
        else:
            print("LYAPUNOV: Candidate V does not consistently hold")
    elif stable_region_exists:
        print("CONJECTURE: CONDITIONALLY SUPPORTED (stable region exists, not all tasks converge)")
    else:
        print("CONJECTURE: NOT SUPPORTED (no stable parameter region found)")


def main():
    parser = argparse.ArgumentParser(description='FLUID Stability Experiments')
    parser.add_argument('--exp', nargs='+', type=int, choices=[1, 2, 3],
                        help='Which experiments to run (default: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='CPU workers for experiment 2')
    args = parser.parse_args()

    to_run = set(args.exp) if args.exp else {1, 2, 3}

    if 1 in to_run:
        run_exp1()
    if 2 in to_run:
        run_exp2(n_workers=args.workers)
    if 3 in to_run:
        run_exp3()

    print_final_summary()


if __name__ == '__main__':
    main()
