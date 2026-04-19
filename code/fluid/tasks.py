"""
Synthetic task generators for FLUID stability experiments.

Each generator yields (x, y) tuples indefinitely.
  x: numpy array, shape (n_input,)
  y: numpy array, shape (n_output,)
"""

import numpy as np


def xor_generator(rng=None):
    """XOR: y = x1 XOR x2. Binary inputs/output."""
    rng = rng or np.random.default_rng()
    data = [
        (np.array([0.0, 0.0]), np.array([0.0])),
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([1.0, 0.0]), np.array([1.0])),
        (np.array([1.0, 1.0]), np.array([0.0])),
    ]
    while True:
        idx = rng.integers(0, 4)
        yield data[idx]


def parity_generator(n_bits=4, rng=None):
    """Parity: y = XOR of all n_bits inputs. Binary inputs/output."""
    rng = rng or np.random.default_rng()
    while True:
        x = rng.integers(0, 2, n_bits).astype(float)
        y = np.array([float(int(x.sum()) % 2)])
        yield x, y


def symbolic_regression_generator(rng=None):
    """Symbolic regression: y = (x1^2 + x2^2) / 8. Continuous inputs/output."""
    rng = rng or np.random.default_rng()
    while True:
        x = rng.uniform(-2.0, 2.0, 2)
        y = np.array([(x[0] ** 2 + x[1] ** 2) / 8.0])
        yield x, y


TASKS = {
    'xor': (xor_generator, 2, 1),
    'parity4': (parity_generator, 4, 1),
    'symreg': (symbolic_regression_generator, 2, 1),
}


def get_task(name, seed=None):
    """Return (generator, n_input, n_output) for a named task."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Options: {list(TASKS)}")
    factory, n_in, n_out = TASKS[name]
    rng = np.random.default_rng(seed)
    if name == 'parity4':
        gen = factory(n_bits=4, rng=rng)
    else:
        gen = factory(rng=rng)
    return gen, n_in, n_out
