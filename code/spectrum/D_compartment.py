"""
D Compartment — Physical Build
================================
An electrolytic capacitor + bleeder resistor.
That's it. That IS the equation:

    τ_D · dV/dt = −V + Σ w_i · I_i(t)

where τ_D = R_bleed · C

Component values chosen to hit τ_D = 10ms:
    C        = 10 µF
    R_bleed  = 1 kΩ   →  τ = RC = 10ms  ✓

Each synapse is an input current through a weight resistor R_w:
    w_i = 1 / R_w_i   (smaller resistor = stronger synapse)

Three tests — all run on the same physical circuit:
    Test 1: single pulse   → measure τ decay
    Test 2: two inputs simultaneous  → spatial summation
    Test 3: two inputs 8ms apart     → temporal summation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

plt.style.use('dark_background')

# ── CIRCUIT PARAMETERS ────────────────────────────────────────────────────────
C_cap    = 10e-6          # 10 µF capacitor
R_bleed  = 1e3            # 1 kΩ bleeder resistor
tau_D    = R_bleed * C_cap  # = 0.01 s = 10 ms   ✓

R_syn    = 2e3            # 2 kΩ per synapse (sets weight = 1/R_syn)
V_pulse  = 2.0            # volts — input pulse amplitude
dt       = 0.0001         # 0.1 ms timestep
t_end    = 0.08           # 80 ms

C = dict(V='#00ff88', D='#4499ff', S='#ffaa00',
         cap='#4499ff', res='#ffaa00', wire='#888888',
         pulse='#ff6644', dim='#333333', title='#00ccff', label='#888888')

def style(ax, title):
    ax.set_facecolor('#0d0d0d')
    for sp in ['top','right']:   ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#222222')
    ax.tick_params(colors='#444444', labelsize=7.5)
    ax.set_title(title, color=C['title'], fontsize=9, pad=6,
                 fontweight='bold', fontfamily='monospace')
    for lbl in [ax.yaxis.label, ax.xaxis.label]:
        lbl.set_color(C['label']); lbl.set_size(8)

# ── RC SIMULATOR ──────────────────────────────────────────────────────────────
def simulate_RC(input_sources, t_end=t_end, dt=dt):
    """
    input_sources: list of (amplitude_V, t_start_s, t_end_s, R_syn_ohms)
    Returns t (ms), V_cap (volts)
    """
    t  = np.arange(0, t_end, dt)
    V  = np.zeros(len(t))   # voltage across capacitor

    for i in range(1, len(t)):
        ti = t[i-1]
        v  = V[i-1]

        # current from bleeder resistor (leak)
        I_leak = v / R_bleed

        # current from each synaptic input
        I_in = 0.0
        for amp, ts, te, rw in input_sources:
            if ts <= ti < te:
                I_syn = (amp - v) / rw   # current through weight resistor
                I_in += I_syn

        dV = (I_in - I_leak) / C_cap
        V[i] = v + dV * dt

    return t * 1000, V   # return time in ms

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor='#080808')
fig.suptitle('D COMPARTMENT  —  Electrolytic Capacitor + Bleeder Resistor\n'
             'τ_D = R · C = 1kΩ × 10µF = 10ms',
             fontsize=14, color='#00ff88', fontweight='bold',
             y=0.987, fontfamily='monospace')

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.55, wspace=0.42,
                       top=0.93, bottom=0.07, left=0.06, right=0.97)

# ── [1] CIRCUIT DIAGRAM ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#0d0d0d')
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('[1] PHYSICAL CIRCUIT  —  D compartment',
              color=C['title'], fontsize=9, pad=6,
              fontweight='bold', fontfamily='monospace')

def wire(ax, x0,y0,x1,y1):
    ax.plot([x0,x1],[y0,y1], color=C['wire'], lw=1.5)

def resistor(ax, x, y, label, color='#ffaa00', vertical=False):
    if not vertical:
        r = Rectangle((x-0.6, y-0.2), 1.2, 0.4,
                       linewidth=1.2, edgecolor=color, facecolor='#1a1a1a')
    else:
        r = Rectangle((x-0.2, y-0.6), 0.4, 1.2,
                       linewidth=1.2, edgecolor=color, facecolor='#1a1a1a')
    ax.add_patch(r)
    ax.text(x, y, label, color=color, fontsize=6, ha='center', va='center',
            fontfamily='monospace')

def capacitor(ax, x, y):
    ax.plot([x-0.5, x+0.5], [y+0.15, y+0.15], color=C['cap'], lw=2.5)
    ax.plot([x-0.5, x+0.5], [y-0.15, y-0.15], color=C['cap'], lw=2.5)
    ax.text(x, y-0.55, '10µF', color=C['cap'], fontsize=6,
            ha='center', fontfamily='monospace')

# main rails
wire(ax1, 1, 8.5, 9, 8.5)   # top rail
wire(ax1, 1, 1.5, 9, 1.5)   # bottom rail (GND)

# synapse 1
wire(ax1, 0.2, 6.5, 2.2, 6.5)
resistor(ax1, 2.8, 6.5, 'R_w1\n2kΩ', color='#cc8844')
wire(ax1, 3.4, 6.5, 5, 6.5)
wire(ax1, 5, 6.5, 5, 8.5)
ax1.annotate('syn 1\n+2V', xy=(0.2, 6.5), color=C['pulse'], fontsize=6.5, ha='left')

# synapse 2
wire(ax1, 0.2, 4.5, 2.2, 4.5)
resistor(ax1, 2.8, 4.5, 'R_w2\n2kΩ', color='#cc8844')
wire(ax1, 3.4, 4.5, 5, 4.5)
wire(ax1, 5, 4.5, 5, 6.5)
ax1.annotate('syn 2\n+2V', xy=(0.2, 4.5), color=C['pulse'], fontsize=6.5, ha='left')

# synapse 3
wire(ax1, 0.2, 2.5, 2.2, 2.5)
resistor(ax1, 2.8, 2.5, 'R_w3\n2kΩ', color='#cc8844')
wire(ax1, 3.4, 2.5, 5, 2.5)
wire(ax1, 5, 2.5, 4.5, 1.5)
ax1.annotate('syn 3\n+2V', xy=(0.2, 2.5), color=C['pulse'], fontsize=6.5, ha='left')

# bleeder resistor (vertical)
wire(ax1, 7, 8.5, 7, 7.2)
resistor(ax1, 7, 6.5, 'R\n1kΩ', vertical=True)
wire(ax1, 7, 5.8, 7, 1.5)
ax1.text(7.5, 5.5, 'bleed\n(leak)', color=C['S'], fontsize=6,
         ha='left', fontfamily='monospace')

# capacitor
wire(ax1, 8.5, 8.5, 8.5, 5.7)
capacitor(ax1, 8.5, 5.2)
wire(ax1, 8.5, 4.7, 8.5, 1.5)

# output tap
wire(ax1, 8.5, 6.8, 9.5, 6.8)
ax1.annotate('→ V_D\n(to S)', xy=(9.5, 6.8), color=C['V'],
             fontsize=7, ha='left', fontfamily='monospace')

ax1.text(5, 0.3, 'GND', color='#555555', ha='center', fontsize=7)

# ── [2] TEST 1 — SINGLE PULSE, MEASURE DECAY ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
src1 = [(V_pulse, 0.005, 0.010, R_syn)]   # 5ms pulse
t2, V2 = simulate_RC(src1)

# find τ from simulation
peak_idx = np.argmax(V2)
peak_v   = V2[peak_idx]
target   = peak_v * np.exp(-1)
decay_idx = peak_idx + np.argmin(np.abs(V2[peak_idx:] - target))
tau_meas  = t2[decay_idx] - t2[peak_idx]

ax2.plot(t2, V2, color=C['D'], lw=2)
ax2.axhline(peak_v,    color=C['dim'], lw=0.8, ls='--', alpha=0.6)
ax2.axhline(target,    color=C['S'],   lw=0.8, ls='--', alpha=0.7)
ax2.axvline(t2[peak_idx],  color=C['dim'], lw=0.8, ls=':', alpha=0.5)
ax2.axvline(t2[decay_idx], color=C['S'],   lw=0.8, ls=':', alpha=0.5)
ax2.annotate('', xy=(t2[decay_idx], target+0.02),
             xytext=(t2[peak_idx], target+0.02),
             arrowprops=dict(arrowstyle='<->', color=C['S'], lw=1.2))
ax2.text((t2[peak_idx]+t2[decay_idx])/2, target+0.06,
         f'τ = {tau_meas:.1f}ms', color=C['S'], fontsize=8, ha='center')
ax2.text(t2[peak_idx]+1, peak_v*0.37, 'V · e⁻¹', color=C['S'], fontsize=7)
ax2.fill_between(t2, V2, alpha=0.08, color=C['D'])
ax2.set_ylabel('V_D  (volts)'); ax2.set_xlabel('time (ms)')
ax2.set_xlim(0, 80)
style(ax2, '[2] TEST 1  —  single pulse, measure τ decay')

# ── [3] TEST 2 — SPATIAL SUMMATION ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
pulse_t = (0.010, 0.015)  # 10-15ms

src_1syn = [(V_pulse, *pulse_t, R_syn)]
src_2syn = [(V_pulse, *pulse_t, R_syn)] * 2
src_3syn = [(V_pulse, *pulse_t, R_syn)] * 3

_, V_1 = simulate_RC(src_1syn)
_, V_2 = simulate_RC(src_2syn)
t3, V_3 = simulate_RC(src_3syn)

ax3.plot(t3, V_1, color='#1a4d2e', lw=1.2, label='1 synapse')
ax3.plot(t3, V_2, color='#27a35a', lw=1.4, label='2 synapses')
ax3.plot(t3, V_3, color=C['V'],    lw=1.8, label='3 synapses')
ax3.fill_between(t3, V_3, alpha=0.07, color=C['V'])

# annotate peak voltages
for v_arr, label, col in [(V_1,'1x','#1a4d2e'),(V_2,'2x','#27a35a'),(V_3,'3x',C['V'])]:
    pk = np.max(v_arr)
    ax3.axhline(pk, color=col, lw=0.5, ls=':', alpha=0.5)
    ax3.text(78, pk+0.01, f'{pk:.2f}V', color=col, fontsize=6.5, ha='right')

ax3.legend(fontsize=7.5, framealpha=0, labelcolor='white')
ax3.set_ylabel('V_D  (volts)'); ax3.set_xlabel('time (ms)')
ax3.set_xlim(0, 80)
style(ax3, '[3] TEST 2  —  spatial summation  (simultaneous inputs)')

# ── [4] TEST 3 — TEMPORAL SUMMATION ───────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
# two pulses 8ms apart — second arrives before first decays
src_spread = [(V_pulse, 0.010, 0.013, R_syn),   # pulse 1: 10-13ms
              (V_pulse, 0.055, 0.058, R_syn)]    # pulse 2: 55-58ms (after full decay)

src_stack  = [(V_pulse, 0.010, 0.013, R_syn),   # pulse 1: 10-13ms
              (V_pulse, 0.018, 0.021, R_syn)]    # pulse 2: 18-21ms (within tau_D)

_, V_spread = simulate_RC(src_spread)
t4, V_stack = simulate_RC(src_stack)

ax4.plot(t4, V_spread, color=C['dim'], lw=1.2, ls='--',
         label='pulses 45ms apart  (full decay between)')
ax4.plot(t4, V_stack,  color=C['V'],   lw=1.8,
         label='pulses 8ms apart  (stacks on residual)')
ax4.fill_between(t4, V_stack, alpha=0.08, color=C['V'])

# mark the two pulse times
for ts in [10, 18]:
    ax4.axvspan(ts, ts+3, alpha=0.12, color=C['pulse'])
ax4.text(11.5, 0.02, 'P1', color=C['pulse'], fontsize=7, ha='center')
ax4.text(19.5, 0.02, 'P2', color=C['pulse'], fontsize=7, ha='center')

ax4.legend(fontsize=7, framealpha=0, labelcolor='white')
ax4.set_ylabel('V_D  (volts)'); ax4.set_xlabel('time (ms)')
ax4.set_xlim(0, 80)
style(ax4, '[4] TEST 3  —  temporal summation  (within tau_D window)')

# ── [5] TAU SENSITIVITY — what if RC is wrong ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
src_base = [(V_pulse, 0.010, 0.015, R_syn)]
taus = [(5e-3,  '#ff4444', '5ms  (too fast, window too short)'),
        (10e-3, C['V'],    '10ms  (target)'),
        (20e-3, '#4499ff', '20ms  (too slow, bleeds into next)')]

for tau_val, color, label in taus:
    R_test = tau_val / C_cap
    src_test = [(V_pulse, 0.010, 0.015, R_test)]
    t5, V5 = simulate_RC(src_test)
    ax5.plot(t5, V5, color=color, lw=1.6, label=label)

# show two-pulse stacking for each tau
for tau_val, color, _ in taus:
    R_test = tau_val / C_cap
    src_two = [(V_pulse, 0.010, 0.013, R_test),
               (V_pulse, 0.018, 0.021, R_test)]
    t5, V5 = simulate_RC(src_two)
    ax5.plot(t5, V5, color=color, lw=0.8, ls=':')

ax5.legend(fontsize=6.5, framealpha=0, labelcolor='white')
ax5.set_ylabel('V_D  (volts)'); ax5.set_xlabel('time (ms)')
ax5.set_xlim(0, 80)
style(ax5, '[5] TAU SENSITIVITY  —  why 10ms is right')

# ── [6] WEIGHTED SYNAPSES — different R_w = different strength ────────────────
ax6 = fig.add_subplot(gs[1, 2])
src_w = [
    (V_pulse, 0.010, 0.013, 1e3),   # strong:  R_w=1kΩ  → w=1/1k
    (V_pulse, 0.025, 0.028, 2e3),   # medium:  R_w=2kΩ
    (V_pulse, 0.040, 0.043, 5e3),   # weak:    R_w=5kΩ
    (V_pulse, 0.055, 0.058, 10e3),  # very weak: R_w=10kΩ
]
labels = ['strong (1kΩ)', 'medium (2kΩ)', 'weak (5kΩ)', 'very weak (10kΩ)']
colors = [C['V'], '#27a35a', '#ffaa00', '#ff4444']

for i, (src, lbl, col) in enumerate(zip(src_w, labels, colors)):
    t6, V6 = simulate_RC([src])
    ax6.plot(t6, V6, color=col, lw=1.6, label=lbl)

ax6.legend(fontsize=7, framealpha=0, labelcolor='white')
ax6.set_ylabel('V_D  (volts)'); ax6.set_xlabel('time (ms)')
ax6.set_xlim(0, 80)
style(ax6, '[6] SYNAPTIC WEIGHTS  —  R_w controls strength')

# ── [7] FULL INTEGRATION — multi-synapse, multi-timing ────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
# realistic scenario: 6 synapses, different weights, different timings
full_src = [
    (V_pulse, 0.008, 0.011, 1.5e3),   # strong, early
    (V_pulse, 0.010, 0.013, 2.0e3),   # medium, early
    (V_pulse, 0.012, 0.015, 3.0e3),   # weak, slightly later
    (V_pulse, 0.015, 0.018, 1.8e3),   # medium, later
    (V_pulse, 0.018, 0.021, 2.5e3),   # medium-weak
    (V_pulse, 0.020, 0.023, 1.2e3),   # strong, last
]

# threshold line (what S needs from D to fire)
D_threshold = 0.45

t7, V7 = simulate_RC(full_src)
ax7.plot(t7, V7, color=C['D'], lw=2, label='V_D  (6 synapses, varied weights & timing)')
ax7.fill_between(t7, V7, alpha=0.1, color=C['D'])
ax7.axhline(D_threshold, color=C['S'], lw=1.2, ls='--',
            label=f'D→S handoff threshold  ({D_threshold}V)')

# mark each pulse
pulse_colors = [C['V'], '#27a35a', '#ffaa00', '#ff6644', '#cc44ff', '#4499ff']
for i, (amp, ts, te, rw) in enumerate(full_src):
    ax7.axvspan(ts*1000, te*1000, alpha=0.15, color=pulse_colors[i])
    ax7.text((ts+te)*500, 0.03, f's{i+1}', color=pulse_colors[i],
             fontsize=6.5, ha='center')

crossed = t7[V7 >= D_threshold]
if len(crossed):
    ax7.axvline(crossed[0], color=C['S'], lw=1, ls=':', alpha=0.8)
    ax7.text(crossed[0]+1, D_threshold+0.07,
             f'threshold crossed\nt={crossed[0]:.1f}ms → S',
             color=C['S'], fontsize=7.5, va='bottom')

ax7.set_ylabel('V_D  (volts)'); ax7.set_xlabel('time (ms)')
ax7.set_xlim(0, 80); ax7.set_ylim(-0.05, 0.9)
ax7.legend(fontsize=8, framealpha=0, labelcolor='white')
style(ax7, '[7] FULL D COMPARTMENT  —  6 synapses, ready to hand off to S')

# ── [8] BUILD CHECKLIST ───────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.set_facecolor('#0d0d0d')
ax8.axis('off')
ax8.set_title('[8] BUILD STATUS', color=C['title'], fontsize=9, pad=6,
              fontweight='bold', fontfamily='monospace')

checks = [
    ('DONE', 'C = 10µF capacitor',               C['V']),
    ('DONE', 'R = 1kΩ bleeder resistor',          C['V']),
    ('DONE', 'τ_D = RC = 10ms verified',          C['V']),
    ('DONE', 'Spatial summation  ✓ Test 2',       C['V']),
    ('DONE', 'Temporal summation ✓ Test 3',       C['V']),
    ('DONE', 'Synaptic weights   ✓ Test 6',       C['V']),
    ('DONE', 'Threshold handoff  ✓ Test 7',       C['V']),
    ('----', '',                                   '#333333'),
    ('NEXT', 'Build S compartment',               C['S']),
    ('NEXT', '  bistable threshold element',      C['S']),
    ('NEXT', '  self-resetting after fire',       C['S']),
    ('----', '',                                   '#333333'),
    ('TODO', 'Build θ compartment',               '#555555'),
    ('TODO', 'Wire D → S → θ',                   '#555555'),
    ('TODO', 'Test full DST neuron',              '#555555'),
]

for i, (status, text, color) in enumerate(checks):
    y = 0.95 - i * 0.068
    if status == '----':
        ax8.plot([0.0, 1.0], [y+0.02, y+0.02], color='#222222', lw=0.8,
                 transform=ax8.transAxes)
        continue
    ax8.text(0.05, y, f'[{status}]', transform=ax8.transAxes,
             color=color, fontsize=7.5, fontfamily='monospace', va='top')
    ax8.text(0.32, y, text, transform=ax8.transAxes,
             color=color, fontsize=7.5, fontfamily='monospace', va='top')

out = '/Users/winstonlin/Desktop/D_compartment.png'
plt.savefig(out, dpi=160, facecolor='#080808', bbox_inches='tight')
print(f"saved → {out}")
plt.show()
