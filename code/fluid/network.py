"""
FLUID network: minimal implementation for Fluid Stability Conjecture testing.

Implements Mechanism 1 (Hebbian topology growth) and Mechanism 4 (edge pruning).
Deliberately excludes causal memory and temporal hierarchy to isolate stability.
Node count is fixed; only the hidden->hidden edge set is fluid.
"""

import numpy as np
import networkx as nx


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FLUIDNetwork:
    """
    Three-layer network with fluid hidden->hidden topology.

    Architecture:
      - Input layer (fixed): n_input nodes
      - Hidden layer (fluid edges): n_hidden nodes
      - Output layer (fixed): n_output nodes

    Input->hidden and hidden->output edges are static (gradient-trained).
    Hidden->hidden edges are created/deleted by the Hebbian rule.
    """

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden=100,
        theta_create=0.5,
        theta_prune=0.01,
        T_prune=500,
        T_window=50,
        lr=0.01,
        w_init_std=0.1,
        hebbian_freq=10,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.theta_create = theta_create
        self.theta_prune = theta_prune
        self.T_prune = T_prune
        self.T_window = T_window
        self.lr = lr
        self.hebbian_freq = hebbian_freq
        self.t = 0

        # Static weight matrices
        self.W_ih = rng.normal(0, w_init_std, (n_hidden, n_input))   # input->hidden
        self.W_ho = rng.normal(0, w_init_std, (n_output, n_hidden))  # hidden->output

        # Fluid weight matrix: hidden->hidden (zero = no edge)
        self.W_hh = np.zeros((n_hidden, n_hidden))
        self.hh_mask = np.zeros((n_hidden, n_hidden), dtype=bool)    # edge existence

        # Prune counter per h->h pair (steps edge has been below theta_prune)
        self.prune_cnt = np.zeros((n_hidden, n_hidden), dtype=np.int32)

        # Activation history: circular buffer (T_window, n_hidden)
        self.H_buf = np.zeros((T_window, n_hidden))
        self.buf_ptr = 0
        self.buf_count = 0

        # Previous hidden state for recurrent h->h pass
        self.prev_h = np.zeros(n_hidden)

        # Saved activations from last forward pass
        self._h = np.zeros(n_hidden)
        self._out = np.zeros(n_output)

        # Metrics history (recorded every record_every steps)
        self.history = {
            't': [],
            'n_edges': [],
            'clustering': [],
            'loss': [],
            'lyapunov': [],
        }

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def forward(self, x):
        """Forward pass. x: (n_input,) array. Returns (n_output,) array."""
        h = sigmoid(self.W_ih @ x + self.W_hh @ self.prev_h)
        out = sigmoid(self.W_ho @ h)

        # Store in circular buffer
        self.H_buf[self.buf_ptr] = h
        self.buf_ptr = (self.buf_ptr + 1) % self.T_window
        self.buf_count = min(self.buf_count + 1, self.T_window)

        self.prev_h = h
        self._h = h
        self._out = out
        return out

    def backward(self, x, target):
        """Backprop on static edges. Returns scalar MSE loss."""
        h, out = self._h, self._out

        d_out = (target - out) * out * (1.0 - out)          # (n_output,)
        self.W_ho += self.lr * np.outer(d_out, h)

        d_h = (self.W_ho.T @ d_out) * h * (1.0 - h)        # (n_hidden,)
        self.W_ih += self.lr * np.outer(d_h, x)

        return float(np.mean((target - out) ** 2))

    # ------------------------------------------------------------------
    # Hebbian topology update
    # ------------------------------------------------------------------

    def _get_recent_activations(self):
        """Return (T_obs, n_hidden) matrix from circular buffer."""
        n = self.buf_count
        if n < self.T_window:
            return self.H_buf[:n]
        # Unroll circular buffer to chronological order
        return np.roll(self.H_buf, -self.buf_ptr, axis=0)

    def hebbian_update(self):
        """Apply Hebbian creation/deletion rule to hidden->hidden edges."""
        if self.t % self.hebbian_freq != 0:
            return
        if self.buf_count < 2:
            return

        H = self._get_recent_activations()          # (T_obs, n_hidden)
        T_obs = H.shape[0]

        # Co-activation matrix rho: (n_hidden, n_hidden)
        rho = H.T @ H / T_obs
        np.fill_diagonal(rho, 0.0)

        # --- Create edges where rho > theta_create and no edge exists ---
        create = (rho > self.theta_create) & (~self.hh_mask)
        if create.any():
            self.W_hh[create] = 0.01
            self.hh_mask[create] = True
            self.prune_cnt[create] = 0

        # --- Update prune counters for existing edges ---
        weak = self.hh_mask & (np.abs(self.W_hh) < self.theta_prune)
        strong = self.hh_mask & ~weak
        self.prune_cnt[weak] += self.hebbian_freq
        self.prune_cnt[strong] = 0

        # --- Delete edges whose prune counter exceeds T_prune ---
        to_prune = self.hh_mask & (self.prune_cnt > self.T_prune)
        if to_prune.any():
            self.W_hh[to_prune] = 0.0
            self.hh_mask[to_prune] = False
            self.prune_cnt[to_prune] = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, x, y, record_every=1000):
        """One training step: forward + backward + Hebbian update."""
        pred = self.forward(x)
        loss = self.backward(x, y)
        self.hebbian_update()
        self.t += 1

        if self.t % record_every == 0:
            self._record(loss)

        return pred, loss

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _activation_entropy(self):
        """Entropy of discretized hidden activation distribution (proxy for C_t)."""
        H = self._get_recent_activations()
        if H.shape[0] == 0:
            return 0.0
        acts = H.flatten()
        hist, _ = np.histogram(acts, bins=10, range=(0.0, 1.0))
        p = hist.astype(float) / (hist.sum() + 1e-12)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _record(self, loss):
        n_edges = int(self.hh_mask.sum())

        # Clustering coefficient on undirected hidden subgraph
        if n_edges > 0:
            G = nx.from_numpy_array(self.hh_mask.astype(np.uint8))
            cc = nx.average_clustering(G)
        else:
            cc = 0.0

        # Lyapunov candidate V = H_t / (|E_t| + 1)
        lyapunov = self._activation_entropy() / (n_edges + 1)

        self.history['t'].append(self.t)
        self.history['n_edges'].append(n_edges)
        self.history['clustering'].append(cc)
        self.history['loss'].append(loss)
        self.history['lyapunov'].append(lyapunov)

    def is_converged(self, window_steps=10000, tol=1, record_every=1000):
        """True if edge count has not changed by more than tol in last window_steps."""
        n_pts = window_steps // record_every
        h = self.history['n_edges']
        if len(h) < n_pts:
            return False
        recent = h[-n_pts:]
        return (max(recent) - min(recent)) <= tol

    @property
    def n_edges(self):
        return int(self.hh_mask.sum())

    @property
    def max_edges(self):
        return self.n_hidden * (self.n_hidden - 1)
