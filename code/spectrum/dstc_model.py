"""
DSTC Model — Dendritic / Somatic / Threshold / Calcium
=======================================================
Extended DST with four additions that make the chip simulate neurons more:

  1. Two-compartment dendrite  — D_d (distal, τ=15ms) and D_p (proximal, τ=5ms)
  2. NMDA-like gate on D_d    — voltage-gated coincidence detection
  3. Inhibitory synapses      — shunting, pull toward E_inh below rest
  4. Calcium compartment      — Ca(t) rises per spike, decays slowly (τ=200ms)

Variables
---------
  D_d(t) — distal dendritic potential   (τ=15ms, NMDA + AMPA inputs)
  D_p(t) — proximal dendritic potential (τ=5ms,  local inputs + D_d feed-in)
  S(t)   — somatic potential            (as in DST)
  θ(t)   — adaptive threshold           (as in DST)
  Ca(t)  — calcium concentration        (spike history, plasticity signal)

NMDA gate
---------
  g_NMDA(V) = 1 / (1 + γ·exp(−β·V))   γ=3.0, β=2.5
  At rest (V=−1):  g≈0.027  — nearly Mg²⁺-blocked
  At spike (V=2):  g≈0.98   — fully open
  Hardware: MOSFET whose gate voltage is V_{D_d}

Inhibitory synapses
-------------------
  I_inh(t) = −g_inh·(V − E_inh)   E_inh = −2.5 (below S_rest)
  Hardware: NPN transistor current sink

Calcium
-------
  dCa/dt = −Ca / τ_Ca   [+δ_Ca per spike]
  Hardware: slow RC (200µF × 1kΩ) charged by S compartment spike output

Equations
---------
  τ_Dd·dD_d/dt = −D_d + I_AMPA(t) + g_NMDA(D_d)·I_NMDA(t) + I_inh_d(t)
  τ_Dp·dD_p/dt = −D_p + I_prox(t) + κ_dd·D_d + I_inh_p(t)
  τ_S ·dS/dt   = −(S−S_rest) + κ·D_p − (θ−θ_0)
       dθ/dt   = −(θ−θ_0) / τ_θ
       dCa/dt  = −Ca / τ_Ca

Hardware build path
-------------------
  D_d: 15µF × 1kΩ RC + MOSFET (NMDA gate)
  D_p:  5µF × 1kΩ RC
  S:   Schmitt trigger (bistable, self-resetting)
  θ:   80µF × 1kΩ slow RC
  Ca:  200µF × 1kΩ slow RC charged by spike output
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# ── PARAMETERS ───────────────────────────────────────────────────────────────
tau_Dd  = 15.0   # distal dendrite time constant (ms)
tau_Dp  =  5.0   # proximal dendrite time constant (ms)
tau_S   =  2.0   # somatic time constant (ms)
tau_th  = 80.0   # threshold recovery (ms)
tau_Ca  = 200.0  # calcium decay (ms)

kappa    = 2.0   # D_p → S coupling (higher than DST to compensate cascade attenuation)
kappa_dd = 0.8   # D_d → D_p coupling
S_rest   = -1.0
th_0     =  1.0
dth      =  6.0
S_reset  = -2.0
delta_Ca =  1.0  # Ca jump per spike
E_inh    = -2.5  # inhibitory reversal potential

# NMDA gate parameters (Mg2+ block relief)
gamma_N  = 3.0
beta_N   = 2.5

# spike waveform (display only, same as DST)
V_rest_mV = -65.0
A_peak    = 100.0
A_AHP     =  12.0
sigma_rise =  0.25
tau_AHP    =  1.2
sigma_AHP  =  2.0

def g_NMDA(V):
    """Voltage-gated NMDA conductance. Nearly 0 at rest, ~1 during depolarization."""
    return 1.0 / (1.0 + gamma_N * np.exp(-beta_N * V))

def gauss(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

def spike_waveform(t, t_fires):
    V = np.full(len(t), V_rest_mV)
    for tf in t_fires:
        V += A_peak * gauss(t, tf + 0.5, sigma_rise)
        V -= A_AHP  * gauss(t, tf + 0.5 + tau_AHP, sigma_AHP)
    return V

# ── SIMULATOR ─────────────────────────────────────────────────────────────────
def simulate(I_ampa_d=None, I_nmda_d=None, I_prox=None,
             I_inh_d=None, I_inh_p=None,
             synapses_d=None, synapses_p=None,
             t_end=100, dt=0.01):
    """
    All I_* are callables: f(t) → scalar.
    synapses_d / synapses_p: list of (weight, spike_times[]) — explicit synaptic inputs.
    """
    if I_ampa_d is None: I_ampa_d = lambda t: 0.0
    if I_nmda_d is None: I_nmda_d = lambda t: 0.0
    if I_prox   is None: I_prox   = lambda t: 0.0
    if I_inh_d  is None: I_inh_d  = lambda t: 0.0
    if I_inh_p  is None: I_inh_p  = lambda t: 0.0

    t   = np.arange(0, t_end, dt)
    Dd  = np.zeros(len(t))
    Dp  = np.zeros(len(t))
    S   = np.zeros(len(t)); S[0] = S_rest
    TH  = np.full(len(t), th_0)
    Ca  = np.zeros(len(t))
    fires = []

    # precompute synaptic inputs
    I_syn_d = np.zeros(len(t))
    I_syn_p = np.zeros(len(t))
    for i, ti in enumerate(t):
        I_syn_d[i] = I_ampa_d(ti)
        I_syn_p[i] = I_prox(ti)
        if synapses_d:
            for w, spt in synapses_d:
                for ts in spt:
                    if ts <= ti:
                        I_syn_d[i] += w * np.exp(-(ti - ts) / tau_Dd)
        if synapses_p:
            for w, spt in synapses_p:
                for ts in spt:
                    if ts <= ti:
                        I_syn_p[i] += w * np.exp(-(ti - ts) / tau_Dp)

    for i in range(1, len(t)):
        dd = Dd[i-1]; dp = Dp[i-1]; s = S[i-1]; th = TH[i-1]; ca = Ca[i-1]

        nmda_in = g_NMDA(dd) * I_nmda_d(t[i])
        inh_d   = I_inh_d(t[i]) * (dd - E_inh)
        inh_p   = I_inh_p(t[i]) * (dp - E_inh)

        dDd  = (-dd + I_syn_d[i] + nmda_in - inh_d) / tau_Dd
        dDp  = (-dp + I_syn_p[i] + kappa_dd * dd - inh_p) / tau_Dp
        dS   = (-(s - S_rest) + kappa * dp - (th - th_0)) / tau_S
        dth  = -(th - th_0) / tau_th
        dCa  = -ca / tau_Ca

        Dd[i]  = dd + dDd * dt
        Dp[i]  = dp + dDp * dt
        S[i]   = s  + dS  * dt
        TH[i]  = th + dth * dt
        Ca[i]  = ca + dCa * dt

        if S[i] >= 0:
            fires.append(t[i])
            S[i]  = S_reset
            TH[i] = th + dth * dt + dth
            Ca[i] = ca + dCa * dt + delta_Ca

    V = spike_waveform(t, fires)
    return t, Dd, Dp, S, TH, Ca, V, fires

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor='#080808')
fig.suptitle(
    'DSTC  —  Dendritic(×2) / Somatic / Threshold / Calcium',
    fontsize=13, color='#00ff88', fontweight='bold',
    y=0.992, fontfamily='monospace')
fig.text(0.5, 0.974, 'Extended DST: NMDA gate  ·  two-compartment dendrite  ·  inhibition  ·  Ca²⁺',
         ha='center', color='#44cc88', fontsize=9, fontfamily='monospace')

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.65, wspace=0.42,
                       top=0.955, bottom=0.06, left=0.07, right=0.97)

C = dict(V='#00ff88', Dd='#cc66ff', Dp='#4499ff', S='#ffaa00',
         TH='#ff4444', Ca='#ff9933', inh='#ff3366',
         dim='#444444', title='#00ccff', label='#888888')

def style(ax, title):
    ax.set_facecolor('#0d0d0d')
    for sp in ['top', 'right']:  ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color('#222222')
    ax.tick_params(colors='#444444', labelsize=7)
    ax.set_title(title, color=C['title'], fontsize=8.5, pad=6,
                 fontweight='bold', fontfamily='monospace')
    for lbl in [ax.yaxis.label, ax.xaxis.label]:
        lbl.set_color(C['label']); lbl.set_size(7.5)

# ── [1] ALL FIVE VARIABLES DURING A SINGLE SPIKE ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
t, Dd, Dp, S, TH, Ca, V, fires = simulate(
    I_prox=lambda t: 1.4 if 10 <= t <= 16 else 0.0,  # proximal input — direct, clean spike
    t_end=70)
ax1.plot(t, V,  color=C['V'],  lw=1.8, label='V out (mV)', zorder=5)
ax1.axhline(V_rest_mV, color=C['dim'], lw=0.6, ls='--', alpha=0.4)
ax1.set_ylim(-90, 50)
ax1.set_ylabel('V (mV)'); ax1.set_xlabel('time (ms)')

# internals on a normalized secondary axis — no twinx to avoid label collision
ax1_r = ax1.twinx()
for arr, color, label in [(Dd, C['Dd'], 'D_d'), (Dp, C['Dp'], 'D_p'),
                           (S,  C['S'],  'S'),   (TH, C['TH'], 'θ')]:
    ax1_r.plot(t, arr, color=color, lw=0.9, ls='--', alpha=0.65, label=label)
ax1_r.set_yticks([])
ax1_r.set_ylabel('')
ax1_r.spines['right'].set_visible(False); ax1_r.spines['top'].set_visible(False)
ax1_r.legend(fontsize=6, framealpha=0, labelcolor='white',
             loc='upper right', bbox_to_anchor=(0.99, 0.78))
style(ax1, '[1] SINGLE SPIKE  —  spike + internals')

# ── [2] NMDA GATE FUNCTION ────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
V_sweep = np.linspace(-3, 4, 500)
ax2.plot(V_sweep, g_NMDA(V_sweep), color=C['Dd'], lw=2.2)
ax2.axvline(-1.0, color=C['S'],  lw=1.0, ls='--', alpha=0.7, label='S_rest')
ax2.axvline( 0.0, color=C['TH'], lw=1.0, ls='--', alpha=0.7, label='fire threshold')
ax2.axhline(g_NMDA(-1.0), color=C['S'],  lw=0.8, ls=':', alpha=0.5)
ax2.axhline(g_NMDA( 0.0), color=C['TH'], lw=0.8, ls=':', alpha=0.5)
ax2.text(-2.8, g_NMDA(-1.0)+0.06, f'g={g_NMDA(-1.0):.3f}\nat rest', color=C['S'], fontsize=7, va='bottom')
ax2.text( 0.1, g_NMDA( 0.0)+0.06, f'g={g_NMDA(0.0):.2f}\nat threshold', color=C['TH'], fontsize=7, va='bottom')
ax2.fill_between(V_sweep, g_NMDA(V_sweep), alpha=0.08, color=C['Dd'])
ax2.set_xlabel('V_D_d (dimensionless)'); ax2.set_ylabel('g_NMDA(V)')
ax2.legend(fontsize=7, framealpha=0, labelcolor='white')
ax2.set_ylim(0, 1.05)
style(ax2, '[2] NMDA GATE  —  nearly blocked at rest, open when depolarized')

# ── [3] NMDA COINCIDENCE DETECTION ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

# Case A: NMDA input alone (nearly blocked at rest — no fire)
_,_,_,_,_,_,Va,_ = simulate(
    I_nmda_d=lambda t: 3.0 if 10 <= t <= 20 else 0.0,
    t_end=60)

# Case B: AMPA background depolarizes D_d first → NMDA gate opens → fires
_,_,_,_,_,_,Vb,fb = simulate(
    I_ampa_d=lambda t: 1.5 if  8 <= t <= 20 else 0.0,
    I_nmda_d=lambda t: 2.0 if 10 <= t <= 18 else 0.0,
    t_end=60)

ax3.plot(t[:6000], Va[:6000], color=C['dim'], lw=1.2, ls='--',
         label='NMDA alone  (blocked at rest)')
ax3.plot(t[:6000], Vb[:6000], color=C['V'],   lw=1.6,
         label='AMPA + NMDA  (coincidence → fires)')
ax3.axhline(V_rest_mV, color=C['dim'], lw=0.5, ls='--', alpha=0.4)
ax3.set_ylim(-90, 50)
ax3.set_ylabel('V (mV)'); ax3.set_xlabel('time (ms)')
ax3.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax3, '[3] COINCIDENCE DETECTION  —  NMDA needs background depolarization')

# ── [4] DISTAL vs PROXIMAL — different tau, different timing ──────────────────
ax4 = fig.add_subplot(gs[1, 0])
t4, Dd4, Dp4, _, _, _, V4, _ = simulate(
    I_ampa_d=lambda t: 2.5 if 10 <= t <= 14 else 0.0,  # distal input
    t_end=70)

ax4.plot(t4, Dd4, color=C['Dd'], lw=1.8, label=f'D_d  (distal, τ={tau_Dd}ms)')
ax4.plot(t4, Dp4, color=C['Dp'], lw=1.8, label=f'D_p  (proximal, τ={tau_Dp}ms)')
ax4.axhline(0, color='#333333', lw=0.7, ls='--')

# annotate the delay between peaks
peak_d = t4[np.argmax(Dd4)]
peak_p = t4[np.argmax(Dp4)]
ax4.annotate('', xy=(peak_p, np.max(Dp4)*0.6),
             xytext=(peak_d, np.max(Dp4)*0.6),
             arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.2))
ax4.text((peak_d + peak_p)/2, np.max(Dp4)*0.62,
         f'+{peak_p-peak_d:.1f}ms delay', color='#aaaaaa', fontsize=7, ha='center')

ax4.legend(fontsize=7, framealpha=0, labelcolor='white')
ax4.set_ylabel('dimensionless'); ax4.set_xlabel('time (ms)')
style(ax4, '[4] DISTAL → PROXIMAL  —  signal delay through compartments')

# ── [5] INHIBITORY SHUNTING ───────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])

# Without inhibition — fires
_,_,_,_,_,_,V_exc,_ = simulate(
    I_prox=lambda t: 1.4 if 10 <= t <= 16 else 0.0,
    t_end=60)

# With distal inhibition — weakened but may still fire
_,_,_,_,_,_,V_inh,_ = simulate(
    I_prox=  lambda t: 1.4 if 10 <= t <= 16 else 0.0,
    I_inh_d= lambda t: 0.5 if 10 <= t <= 16 else 0.0,
    t_end=60)

# With proximal inhibition — more effective, blocks firing
_,_,_,_,_,_,V_pinh,_ = simulate(
    I_prox=  lambda t: 1.4 if 10 <= t <= 16 else 0.0,
    I_inh_p= lambda t: 0.6 if 10 <= t <= 16 else 0.0,
    t_end=60)

ax5.plot(t[:6000], V_exc[:6000],  color=C['V'],   lw=1.6, label='excitation only  (fires)')
ax5.plot(t[:6000], V_inh[:6000],  color=C['Dd'],  lw=1.3, label='+ distal inh  (weakened)')
ax5.plot(t[:6000], V_pinh[:6000], color=C['inh'], lw=1.3, label='+ proximal inh  (blocked)')
ax5.axhline(V_rest_mV, color=C['dim'], lw=0.5, ls='--', alpha=0.4)
ax5.set_ylim(-90, 50)
ax5.set_ylabel('V (mV)'); ax5.set_xlabel('time (ms)')
ax5.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax5, '[5] INHIBITORY SHUNTING  —  proximal inh more effective than distal')

# ── [6] CALCIUM ACCUMULATION ──────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
t6, _, _, _, _, Ca6, V6, fires6 = simulate(
    I_prox=lambda t: 1.6 if 5 <= t <= 90 else 0.0,
    t_end=100)

ax6b = ax6.twinx()
ax6.plot(t6, V6,  color=C['V'],  lw=1.2, alpha=0.85)
ax6b.plot(t6, Ca6, color=C['Ca'], lw=1.6)
ax6b.set_ylabel('Ca²⁺ (a.u.)', color=C['Ca'], fontsize=7, labelpad=2)
ax6b.tick_params(colors=C['Ca'], labelsize=6, pad=2)
ax6b.spines['right'].set_color('#333333'); ax6b.spines['top'].set_visible(False)
ax6b.set_ylim(bottom=0)
# Ca label as text inside plot (avoids right-axis crowding)
ax6b.text(0.97, 0.92, f'Ca²⁺  τ={tau_Ca}ms', transform=ax6.transAxes,
          color=C['Ca'], fontsize=7, ha='right', va='top', fontfamily='monospace')
n_spikes = len(fires6)
style(ax6, f'[6] CALCIUM  —  {n_spikes} spikes, slow accumulation')
ax6.set_ylabel('V (mV)'); ax6.set_xlabel('time (ms)')
ax6.set_ylim(-90, 50)

# ── [7] STDP WINDOW — Ca²⁺ as plasticity signal ──────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])

# Simplified STDP: Δw ∝ exp(−|Δt|/τ_+) for pre-before-post (LTP)
#                  Δw ∝ −exp(−|Δt|/τ_−) for post-before-pre (LTD)
tau_plus  = 20.0
tau_minus = 20.0
A_plus    =  0.5
A_minus   =  0.5

dt_range = np.linspace(-80, 80, 400)
dw = np.where(dt_range > 0,
               A_plus  * np.exp(-dt_range / tau_plus),
              -A_minus * np.exp( dt_range / tau_minus))

ax7.fill_between(dt_range[dt_range > 0], dw[dt_range > 0],
                 alpha=0.15, color=C['V'])
ax7.fill_between(dt_range[dt_range < 0], dw[dt_range < 0],
                 alpha=0.15, color=C['inh'])
ax7.plot(dt_range, dw, color='white', lw=1.8)
ax7.axhline(0, color='#333333', lw=1.0)
ax7.axvline(0, color='#333333', lw=0.8, ls='--')
ax7.text( 20,  0.28, 'LTP\n(pre before post)\nCa²⁺ → weight+',
          color=C['V'],   fontsize=8, fontfamily='monospace', va='top')
ax7.text(-78, -0.15, 'LTD\n(post before pre)\nweight−',
          color=C['inh'], fontsize=8, fontfamily='monospace', va='top')
ax7.set_xlabel('Δt = t_pre − t_post  (ms)')
ax7.set_ylabel('Δw  (weight change)')
ax7.set_xlim(-80, 80)
style(ax7, '[7] STDP LEARNING WINDOW  —  Ca²⁺ enables online Hebbian plasticity')

# ── [8] BUILD STATUS — DSTC vs DST ───────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor('#0d0d0d'); ax8.axis('off')
ax8.set_title('[8] DSTC BUILD STATUS', color=C['title'], fontsize=9, pad=6,
              fontweight='bold', fontfamily='monospace')

checks = [
    ('DONE', 'D compartment  (10µF + 1kΩ RC)',    C['V']),
    ('----', '',                                    '#333333'),
    ('ADD',  'D_d: 15µF × 1kΩ  (τ=15ms)',         C['Dd']),
    ('ADD',  '  + MOSFET NMDA gate',               C['Dd']),
    ('ADD',  'D_p:  5µF × 1kΩ  (τ=5ms)',          C['Dp']),
    ('----', '',                                    '#333333'),
    ('NEXT', 'S:  Schmitt trigger',                C['S']),
    ('NEXT', '    bistable, self-resetting',        C['S']),
    ('----', '',                                    '#333333'),
    ('TODO', 'θ:  80µF × 1kΩ slow RC',            '#666666'),
    ('TODO', 'Ca: 200µF × 1kΩ slow RC',           C['Ca']),
    ('TODO', '    charged by spike output',         C['Ca']),
    ('----', '',                                    '#333333'),
    ('TODO', 'Wire D_d → D_p → S → θ',           '#666666'),
    ('TODO', 'Ca STDP readout circuit',            '#666666'),
    ('TODO', 'Full DSTC chip test',                '#666666'),
]

for i, (status, text, color) in enumerate(checks):
    y = 0.97 - i * 0.063
    if status == '----':
        ax8.plot([0.0, 1.0], [y + 0.02, y + 0.02], color='#222222', lw=0.7,
                 transform=ax8.transAxes)
        continue
    ax8.text(0.03, y, f'[{status}]', transform=ax8.transAxes,
             color=color, fontsize=7, fontfamily='monospace', va='top')
    ax8.text(0.30, y, text, transform=ax8.transAxes,
             color=color, fontsize=7, fontfamily='monospace', va='top')

out = '/Users/winstonlin/Desktop/dstc_model.png'
plt.savefig(out, dpi=160, facecolor='#080808', bbox_inches='tight')
print(f"saved → {out}")
plt.show()
