"""
Experiment 3: Lyapunov Candidate Test

For each stable configuration found in Experiment 2, test whether
V(G_t) = H_t / (|E_t| + 1) is monotonically non-increasing over time.

H_t: entropy of the discretized hidden activation distribution (proxy for C_t).
If V is non-increasing, it satisfies the Lyapunov condition, providing
analytic evidence (not proof) of stability.
"""

import json
import os

import numpy as np

from network import FLUIDNetwork
from tasks import get_task

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_STEPS = 500_000
N_HIDDEN = 100
TASK = 'xor'
RECORD_EVERY = 500  # finer granularity for Lyapunov analysis


def is_lyapunov_decreasing(lyapunov_series, smoothing=50):
    """
    Test if the Lyapunov series is broadly non-increasing.

    Uses a smoothed version to ignore noise. Returns:
      - is_decreasing: bool (True if smoothed series is generally non-increasing)
      - fraction_decreasing: fraction of consecutive pairs where V[t+1] <= V[t]
      - trend_slope: linear regression slope on the smoothed series
    """
    arr = np.array(lyapunov_series, dtype=float)
    if len(arr) < smoothing * 2:
        return False, 0.0, 0.0

    # Smooth with moving average
    kernel = np.ones(smoothing) / smoothing
    smoothed = np.convolve(arr, kernel, mode='valid')

    # Fraction of steps where V is non-increasing
    diffs = np.diff(smoothed)
    frac = float(np.mean(diffs <= 0))

    # Linear trend
    t = np.arange(len(smoothed))
    slope = float(np.polyfit(t, smoothed, 1)[0])

    is_decreasing = frac > 0.6 and slope <= 0

    return is_decreasing, frac, slope


def run_config(theta_create, theta_prune, T_prune, seed=42):
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

    return net


def run_lyapunov_test(stable_configs=None):
    """
    Run Lyapunov test on stable configs from Experiment 2.
    If stable_configs is None, uses a set of default 'expected stable' params.
    """
    if stable_configs is None:
        # Load from experiment 2 if available
        exp2_path = os.path.join(RESULTS_DIR, 'exp2_phase_diagram.json')
        if os.path.exists(exp2_path):
            with open(exp2_path) as f:
                exp2 = json.load(f)
            stable_configs = [
                (r['theta_create'], r['theta_prune'], r['T_prune'])
                for r in exp2['results']
                if r['outcome'] == 'stable'
            ]
            print(f"Loaded {len(stable_configs)} stable configs from Experiment 2.")
        else:
            # Fallback: test a small set of plausible stable configs
            stable_configs = [
                (0.5, 0.01, 500),
                (0.3, 0.01, 500),
                (0.5, 0.05, 500),
                (0.5, 0.01, 300),
            ]
            print(f"Experiment 2 results not found. Testing {len(stable_configs)} default configs.")

    results = []
    n_decreasing = 0

    for tc, tp, tprune in stable_configs[:20]:  # cap at 20 to limit runtime
        print(f"  theta_create={tc} theta_prune={tp} T_prune={tprune} ...", end=' ')
        net = run_config(tc, tp, tprune)

        lyap = net.history['lyapunov']
        is_dec, frac, slope = is_lyapunov_decreasing(lyap)

        if is_dec:
            n_decreasing += 1

        result = {
            'theta_create': tc,
            'theta_prune': tp,
            'T_prune': tprune,
            'final_n_edges': net.n_edges,
            'is_lyapunov_decreasing': is_dec,
            'fraction_decreasing': frac,
            'trend_slope': slope,
            'lyapunov_initial': lyap[0] if lyap else None,
            'lyapunov_final': lyap[-1] if lyap else None,
        }
        results.append(result)
        print(f"{'DECREASING' if is_dec else 'NOT DECREASING'} "
              f"(frac={frac:.2f}, slope={slope:.2e})")

    total = len(results)
    print(f"\nLyapunov V non-increasing: {n_decreasing}/{total} "
          f"({100*n_decreasing/total:.1f}%)")

    out_path = os.path.join(RESULTS_DIR, 'exp3_lyapunov.json')
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'n_decreasing': n_decreasing,
            'total': total,
            'fraction_lyapunov_holds': n_decreasing / total if total > 0 else 0.0,
        }, f, indent=2)
    print(f"Saved: {out_path}")
    return results


if __name__ == '__main__':
    run_lyapunov_test()
