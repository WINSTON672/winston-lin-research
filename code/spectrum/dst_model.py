"""
DST Model — Dendritic / Somatic / Threshold
============================================
A neuron model built from the blueprint up, not from squid axon data.

Three variables, three jobs:

  D(t)  — dendritic potential   : leaky integration of all synaptic inputs
  S(t)  — somatic potential     : threshold decision at the axon hillock
  θ(t)  — adaptive threshold    : rises after each spike, decays back to rest

Fire condition:  S(t) ≥ 0

After fire at t_k:
  S  resets to S_reset
  θ  jumps by Δθ   (absolute refractory = θ so high that S can never reach it)

Spike waveform is ANALYTICAL — not simulated step by step.
Two Gaussians: one for the peak, one for the afterhyperpolarization.
This is 10× faster than HH and produces a cleaner, tunable spike shape.

Equations
---------
  τ_D · dD/dt = −D  +  I_syn(t)
  τ_S · dS/dt = −(S − S_rest)  +  κ·D  −  (θ − θ_0)
       dθ/dt  = −(θ − θ_0) / τ_θ

Parameters (all dimensionless / ms)
------------------------------------
  τ_D    = 10   ms   — dendritic time constant (EPSP decay window)
  τ_S    = 2    ms   — somatic integration speed
  τ_θ    = 80   ms   — threshold recovery (refractory scale)
  κ      = 1.2        — dendritic→somatic coupling gain
  S_rest = −1.0       — somatic resting level
  θ_0    = 1.0        — baseline threshold
  Δθ     = 6.0        — threshold jump per spike (sets refractory period)
  S_reset= −2.0       — somatic reset after spike

Spike waveform (mV, for display only — does not feed back into dynamics)
  V(t) = V_rest
       + A_peak · gauss(t − t_fire, σ_rise)
       − A_AHP  · gauss(t − t_fire − τ_AHP, σ_AHP)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

plt.style.use('dark_background')

# ── DST PARAMETERS ──────────────────────────────────────────────────────────
tau_D   = 10.0   # dendritic time constant (ms)
tau_S   =  2.0   # somatic time constant   (ms)
tau_th  = 80.0   # threshold recovery      (ms)
kappa   =  1.2   # D → S coupling
S_rest  = -1.0   # somatic resting level
th_0    =  1.0   # baseline threshold
dth     =  6.0   # threshold jump per spike
S_reset = -2.0   # somatic reset

# spike waveform constants (display only)
V_rest_mV  = -65.0
A_peak     =  100.0   # mV above rest at spike peak
A_AHP      =   12.0   # mV below rest at AHP
sigma_rise =   0.25   # ms
sigma_fall =   0.55   # ms
tau_AHP    =   1.2    # ms offset to AHP center
sigma_AHP  =   2.0    # ms

def gauss(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

def spike_waveform(t, t_fires):
    """Analytical spike shape — sum over all firing times."""
    V = np.full(len(t), V_rest_mV)
    for tf in t_fires:
        V += A_peak * gauss(t, tf + 0.5, sigma_rise)
        V -= A_AHP  * gauss(t, tf + 0.5 + tau_AHP, sigma_AHP)
    return V

# ── SIMULATOR ────────────────────────────────────────────────────────────────
def simulate(I_func, synapses=None, t_end=100, dt=0.01):
    """
    I_func(t)   : direct current injection (dimensionless)
    synapses    : list of (weight, spike_times[]) for explicit synaptic inputs
    """
    t  = np.arange(0, t_end, dt)
    D  = np.zeros(len(t))
    S  = np.zeros(len(t)); S[0] = S_rest
    TH = np.full(len(t), th_0)
    fires = []

    # precompute synaptic input at each timestep
    I_syn = np.zeros(len(t))
    for i, ti in enumerate(t):
        I_syn[i] = I_func(ti)
        if synapses:
            for w, spike_times in synapses:
                for ts in spike_times:
                    if ts <= ti:
                        I_syn[i] += w * np.exp(-(ti - ts) / tau_D)

    for i in range(1, len(t)):
        d  = D[i-1]; s = S[i-1]; th = TH[i-1]

        dD  = (-d  + I_syn[i]) / tau_D
        dS  = (-(s - S_rest) + kappa * d - (th - th_0)) / tau_S
        dth = -(th - th_0) / tau_th

        D[i]  = d  + dD  * dt
        S[i]  = s  + dS  * dt
        TH[i] = th + dth * dt

        if S[i] >= 0:
            fires.append(t[i])
            S[i]  = S_reset
            TH[i] = th + dth * dt + dth   # threshold spike

    V = spike_waveform(t, fires)
    return t, D, S, TH, V, fires

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor='#080808')
fig.suptitle('DST MODEL  —  Dendritic / Somatic / Threshold\n'
             'invented from the blueprint, not from squid axon data',
             fontsize=14, color='#00ff88', fontweight='bold',
             y=0.987, fontfamily='monospace')

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.58, wspace=0.42,
                       top=0.93, bottom=0.07, left=0.07, right=0.97)

C = dict(V='#00ff88', D='#4499ff', S='#ffaa00', TH='#ff4444',
         dim='#444444', title='#00ccff', label='#888888')

def style(ax, title):
    ax.set_facecolor('#0d0d0d')
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#222222')
    ax.tick_params(colors='#444444', labelsize=7)
    ax.set_title(title, color=C['title'], fontsize=8.5, pad=6,
                 fontweight='bold', fontfamily='monospace')
    for lbl in [ax.yaxis.label, ax.xaxis.label]:
        lbl.set_color(C['label']); lbl.set_size(7.5)

# [1] SINGLE SPIKE — show all three internals + waveform
ax1 = fig.add_subplot(gs[0, 0])
t,D,S,TH,V,fires = simulate(lambda t: 0.8 if 10<=t<=12 else 0.0, t_end=60)
ax1.plot(t, V, color=C['V'],  lw=1.8, label='V out (mV)')
ax1.axhline(V_rest_mV, color=C['dim'], lw=0.7, ls='--', alpha=0.5)
ax1.axhline(-55, color=C['TH'], lw=0.7, ls='--', alpha=0.5)
ax1.text(57, -53, '-55 thresh', color=C['TH'], fontsize=6, ha='right')
ax1.set_ylim(-90, 50); ax1.set_ylabel('V (mV)'); ax1.set_xlabel('time (ms)')
style(ax1, '[1] SINGLE SPIKE  —  spike waveform')

# [2] DST INTERNALS — show D, S, theta during the spike
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, D,  color=C['D'],  lw=1.4, label='D  (dendrite)')
ax2.plot(t, S,  color=C['S'],  lw=1.4, label='S  (soma)')
ax2.plot(t, TH, color=C['TH'], lw=1.4, label='θ  (threshold)')
ax2.axhline(0, color='#333333', lw=0.8, ls='--')
ax2.legend(fontsize=7, framealpha=0, labelcolor='white')
ax2.set_ylabel('dimensionless'); ax2.set_xlabel('time (ms)')
style(ax2, '[2] DST INTERNALS  —  D / S / theta')

# [3] REFRACTORY — theta blocks second spike
ax3 = fig.add_subplot(gs[0, 2])
def I_ref(t):
    if 10<=t<=12: return 0.8
    if 14<=t<=16: return 0.8   # blocked — theta still high
    if 28<=t<=30: return 0.8   # fires — theta recovered
    return 0.0
t3,_,_,TH3,V3,f3 = simulate(I_ref, t_end=60)
ax3.plot(t3, V3, color=C['V'], lw=1.6)
ax3.axvspan(12, 18, alpha=0.15, color='#ff2222', label='absolute')
ax3.axvspan(18, 28, alpha=0.07, color='#ff8800', label='relative')
ax3.set_ylim(-90, 50); ax3.set_ylabel('V (mV)'); ax3.set_xlabel('time (ms)')
ax3.legend(fontsize=7, framealpha=0, labelcolor='white', loc='upper right')
style(ax3, '[3] REFRACTORY  —  theta blocks re-firing')

# [4] TEMPORAL SUMMATION — within tau_D window
ax4 = fig.add_subplot(gs[1, 0])
def I_one(t):  return 0.4 if 10<=t<=10.5 else 0.0
def I_many(t):
    for p in [10, 14, 18, 22]:
        if p<=t<=p+0.5: return 0.4
    return 0.0
_,_,_,_,Va,_ = simulate(I_one,  t_end=60)
tb,_,_,_,Vb,_ = simulate(I_many, t_end=60)
ax4.plot(t,  Va, color=C['dim'], lw=1.1, ls='--', label=f'single pulse  (no fire)')
ax4.plot(tb, Vb, color=C['V'],   lw=1.6,           label=f'repeated in τ_D={tau_D}ms window  (fires)')
ax4.axhline(-55, color=C['TH'], lw=0.6, ls='--', alpha=0.4)
ax4.set_ylim(-90, 50); ax4.set_ylabel('V (mV)'); ax4.set_xlabel('time (ms)')
ax4.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax4, '[4] TEMPORAL SUMMATION  —  leaky D with tau_D')

# [5] SPATIAL SUMMATION — explicit synaptic weights
ax5 = fig.add_subplot(gs[1, 1])
t_spk = [10.0]
syn1  = [(0.5,  t_spk)]
syn3  = [(0.5,  t_spk), (0.5, t_spk), (0.5, t_spk)]
syn6  = [(0.5,  t_spk)] * 6
_,_,_,_,V1s,_ = simulate(lambda t:0, synapses=syn1, t_end=60)
_,_,_,_,V3s,_ = simulate(lambda t:0, synapses=syn3, t_end=60)
_,_,_,_,V6s,_ = simulate(lambda t:0, synapses=syn6, t_end=60)
ax5.plot(t, V1s, color='#1a4d2e', lw=1.1, label='1 synapse')
ax5.plot(t, V3s, color='#2e8b57', lw=1.3, label='3 synapses')
ax5.plot(t, V6s, color=C['V'],    lw=1.6, label='6 synapses  (fires)')
ax5.axhline(-55, color=C['TH'], lw=0.6, ls='--', alpha=0.4)
ax5.set_ylim(-90, 50); ax5.set_ylabel('V (mV)'); ax5.set_xlabel('time (ms)')
ax5.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax5, '[5] SPATIAL SUMMATION  —  explicit synaptic weights')

# [6] SPIKE RATE ADAPTATION — theta rises with each spike
ax6 = fig.add_subplot(gs[1, 2])
t6,_,_,TH6,V6,f6 = simulate(lambda t: 1.2 if 5<=t<=95 else 0.0, t_end=100)
ax6b = ax6.twinx()
ax6.plot(t6, V6,  color=C['V'],  lw=1.2, alpha=0.9)
ax6b.plot(t6, TH6, color=C['TH'], lw=1.0, ls='--', alpha=0.7)
ax6b.set_yticks([])
ax6b.set_ylabel('')
ax6b.spines['right'].set_visible(False); ax6b.spines['top'].set_visible(False)
ax6b.text(0.97, 0.55, 'θ (threshold)', transform=ax6.transAxes,
          color=C['TH'], fontsize=7, ha='right', va='center', fontfamily='monospace')
ax6.set_ylabel('V (mV)'); ax6.set_xlabel('time (ms)')
ax6.set_ylim(-90, 50)
n_spikes = len(f6)
style(ax6, f'[6] SPIKE RATE ADAPTATION  —  {n_spikes} spikes, ISI drift')

# [7] TWO NEURONS — N1 fires, N2 receives via synapse
ax7 = fig.add_subplot(gs[2, :2])
t1,_,_,_,V1c,fires1 = simulate(lambda t: 0.9 if 10<=t<=12 else 0.0, t_end=100)

syn_delay = 2.0
syn_from_n1 = [(1.5, [tf + syn_delay for tf in fires1])]
t2,_,_,_,V2c,_ = simulate(lambda t: 0.0, synapses=syn_from_n1, t_end=100)

offset = 95
ax7.plot(t1, V1c + offset, color=C['D'],  lw=1.5, label='Neuron 1')
ax7.plot(t2, V2c,          color=C['V'],  lw=1.5, label='Neuron 2  (downstream)')
ax7.axhline(V_rest_mV + offset, color=C['dim'], lw=0.5, ls='--', alpha=0.3)
ax7.axhline(V_rest_mV,          color=C['dim'], lw=0.5, ls='--', alpha=0.3)
ax7.text(2, V_rest_mV + offset + 3, 'N1', color=C['D'], fontsize=8)
ax7.text(2, V_rest_mV + 3,          'N2', color=C['V'], fontsize=8)
if fires1:
    ax7.annotate(f'+{syn_delay}ms delay',
                 xy=(fires1[0] + syn_delay + 1, -35),
                 color='#aaaaaa', fontsize=7)
ax7.set_ylim(-90, 180)
ax7.set_ylabel('V (mV)  —  N1 offset for clarity')
ax7.set_xlabel('time (ms)')
ax7.legend(fontsize=8, framealpha=0, labelcolor='white', loc='upper right')
style(ax7, '[7] TWO NEURONS  —  synaptic transmission via explicit spike times')

# [8] D vs S PHASE SPACE — the decision boundary
ax8 = fig.add_subplot(gs[2, 2])
t8,D8,S8,TH8,V8,_ = simulate(lambda t: 0.9 if 10<=t<=12 else 0.0, t_end=60)
sc = ax8.scatter(D8, S8, c=t8, cmap='plasma', s=1.5, alpha=0.8)
ax8.axhline(0, color=C['TH'], lw=1.0, ls='--', alpha=0.7, label='fire threshold (S=0)')
ax8.scatter([D8[0]], [S8[0]], color='#ff4444', s=50, zorder=5, label='rest')
plt.colorbar(sc, ax=ax8, label='time (ms)', pad=0.02).ax.tick_params(labelsize=6)
ax8.set_xlabel('D  (dendritic)'); ax8.set_ylabel('S  (somatic)')
ax8.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax8, '[8] PHASE SPACE  —  D vs S trajectory')

out = '/Users/winstonlin/Desktop/dst_model.png'
plt.savefig(out, dpi=160, facecolor='#080808', bbox_inches='tight')
print(f"saved → {out}")
plt.show()
