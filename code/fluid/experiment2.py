"""
Experiment 2: Phase Diagram

Grid search over (theta_create, theta_prune, T_prune).
Classifies each configuration as: stable, fully_connected, disconnected, or growing.
Runs in parallel via multiprocessing.
"""

import json
import os
import itertools
from multiprocessing import Pool, cpu_count

import numpy as np

from network import FLUIDNetwork
from tasks import get_task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameter grid
THETA_CREATE_VALS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
THETA_PRUNE_VALS  = [0.001, 0.005, 0.01, 0.05, 0.1]
T_PRUNE_VALS      = [100, 300, 500, 1000]

N_STEPS = 200_000
N_HIDDEN = 100
TASK = 'xor'
CONVERGENCE_WINDOW = 50_000
RECORD_EVERY = 1_000


def classify_outcome(net):
    max_e = net.max_edges
    n_e = net.n_edges
    if n_e > 0.9 * max_e:
        return 'fully_connected'
    if n_e == 0:
        return 'disconnected'
    if net.is_converged(CONVERGENCE_WINDOW, record_every=RECORD_EVERY):
        return 'stable'
    return 'growing'


def run_config(args):
    theta_create, theta_prune, T_prune, seed = args
    gen, n_input, n_output = get_task(TASK, seed=seed)
    net = FLUIDNetwork(
        n_input=n_input,
        n_output=n_output,
        n_hidden=N_HIDDEN,
        theta_create=theta_create,
        theta_prune=theta_prune,
        T_prune=T_prune,
        T_window=50,
        lr=0.01,
        w_init_std=0.1,
        hebbian_freq=10,
        seed=seed,
    )

    for _ in range(N_STEPS):
        x, y = next(gen)
        net.step(x, y, record_every=RECORD_EVERY)

    outcome = classify_outcome(net)
    return {
        'theta_create': theta_create,
        'theta_prune': theta_prune,
        'T_prune': T_prune,
        'final_n_edges': net.n_edges,
        'outcome': outcome,
        'final_clustering': net.history['clustering'][-1] if net.history['clustering'] else 0.0,
        'final_loss': net.history['loss'][-1] if net.history['loss'] else None,
    }


def run_phase_diagram(n_workers=None):
    configs = list(itertools.product(THETA_CREATE_VALS, THETA_PRUNE_VALS, T_PRUNE_VALS))
    args = [(tc, tp, tprune, 42) for tc, tp, tprune in configs]

    n_workers = n_workers or min(cpu_count(), len(args))
    print(f"Running {len(args)} configurations with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(run_config, args)

    # Summary counts
    counts = {}
    for r in results:
        counts[r['outcome']] = counts.get(r['outcome'], 0) + 1
    print(f"\nOutcome counts: {counts}")
    total = len(results)
    for outcome, cnt in counts.items():
        print(f"  {outcome}: {cnt}/{total} ({100*cnt/total:.1f}%)")

    out_path = os.path.join(RESULTS_DIR, 'exp2_phase_diagram.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'counts': counts, 'total': total}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


def print_2d_slice(results, T_prune_fixed=500):
    """Print ASCII phase diagram for a fixed T_prune slice."""
    subset = [r for r in results if r['T_prune'] == T_prune_fixed]
    if not subset:
        print(f"No results for T_prune={T_prune_fixed}")
        return

    symbols = {'stable': 'S', 'fully_connected': 'F', 'disconnected': 'D', 'growing': 'G'}
    tc_vals = sorted(set(r['theta_create'] for r in subset))
    tp_vals = sorted(set(r['theta_prune'] for r in subset))

    print(f"\nPhase diagram (T_prune={T_prune_fixed}): S=stable F=full D=disconnected G=growing")
    print(f"theta_prune → {tp_vals}")
    print(f"theta_create ↓")
    grid = {(r['theta_create'], r['theta_prune']): r['outcome'] for r in subset}
    for tc in tc_vals:
        row = f"  {tc:.3f}  |"
        for tp in tp_vals:
            row += f" {symbols.get(grid.get((tc, tp), '?'), '?')}  "
        print(row)


if __name__ == '__main__':
    results = run_phase_diagram()
    print_2d_slice(results, T_prune_fixed=500)
