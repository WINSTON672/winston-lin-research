"""
Experiment 1: Small-World Convergence Test

For each task, train a 100-node FLUID network for 1e6 steps with default parameters.
Record |E_t|, clustering coefficient, and loss every 1000 steps.
Check whether topology converges and characterize the emergent graph structure.
"""

import json
import os
import numpy as np
import networkx as nx

from network import FLUIDNetwork
from tasks import get_task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEFAULT_PARAMS = dict(
    n_hidden=100,
    theta_create=0.5,
    theta_prune=0.01,
    T_prune=500,
    T_window=50,
    lr=0.01,
    w_init_std=0.1,
    hebbian_freq=10,
    seed=42,
)

N_STEPS = 1_000_000
RECORD_EVERY = 1_000
CONVERGENCE_WINDOW = 10_000  # steps


def compute_small_world_stats(hh_mask):
    """Return clustering coeff and avg path length on undirected hidden subgraph."""
    G = nx.from_numpy_array(hh_mask.astype(np.uint8))
    cc = nx.average_clustering(G)
    # Average path length only on largest connected component
    components = list(nx.connected_components(G))
    if not components:
        return cc, float('inf')
    lcc = G.subgraph(max(components, key=len)).copy()
    if lcc.number_of_nodes() < 2:
        return cc, float('inf')
    apl = nx.average_shortest_path_length(lcc)
    return cc, apl


def run_task(task_name):
    print(f"\n=== Experiment 1: {task_name} ===")
    gen, n_input, n_output = get_task(task_name, seed=0)
    net = FLUIDNetwork(n_input=n_input, n_output=n_output, **DEFAULT_PARAMS)

    for i in range(N_STEPS):
        x, y = next(gen)
        net.step(x, y, record_every=RECORD_EVERY)

        if i > 0 and i % 100_000 == 0:
            print(f"  step {i:>8d} | edges={net.n_edges:>5d} | "
                  f"loss={net.history['loss'][-1]:.4f} | "
                  f"converged={net.is_converged(CONVERGENCE_WINDOW, record_every=RECORD_EVERY)}")

    converged = net.is_converged(CONVERGENCE_WINDOW, record_every=RECORD_EVERY)
    steps_to_conv = None
    if converged:
        # Find first step where window was stable
        n_pts = CONVERGENCE_WINDOW // RECORD_EVERY
        h = net.history['n_edges']
        for k in range(n_pts, len(h)):
            window = h[k - n_pts:k]
            if max(window) - min(window) <= 1:
                steps_to_conv = net.history['t'][k]
                break

    cc, apl = compute_small_world_stats(net.hh_mask)

    result = {
        'task': task_name,
        'converged': converged,
        'steps_to_convergence': steps_to_conv,
        'final_n_edges': net.n_edges,
        'final_clustering': cc,
        'final_avg_path_length': apl if apl != float('inf') else None,
        'final_loss': net.history['loss'][-1] if net.history['loss'] else None,
        'history': net.history,
    }

    out_path = os.path.join(RESULTS_DIR, f'exp1_{task_name}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Converged: {converged} | Final edges: {net.n_edges} | "
          f"Clustering: {cc:.4f} | APL: {apl:.2f}")
    return result


def run_all():
    results = {}
    for task in ['xor', 'parity4', 'symreg']:
        results[task] = run_task(task)

    print("\n=== Summary ===")
    print(f"{'Task':<12} {'Converged':<12} {'Steps':<12} "
          f"{'|E|':<8} {'C(G)':<8} {'APL':<8} {'Loss'}")
    for task, r in results.items():
        print(f"{task:<12} {str(r['converged']):<12} "
              f"{str(r['steps_to_convergence']):<12} "
              f"{r['final_n_edges']:<8} "
              f"{r['final_clustering']:<8.4f} "
              f"{str(round(r['final_avg_path_length'], 2) if r['final_avg_path_length'] else 'inf'):<8} "
              f"{r['final_loss']:.6f}")

    out_path = os.path.join(RESULTS_DIR, 'exp1_summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull summary saved: {out_path}")
    return results


if __name__ == '__main__':
    run_all()
