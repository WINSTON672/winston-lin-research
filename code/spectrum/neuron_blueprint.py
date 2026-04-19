import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# ── HODGKIN-HUXLEY PARAMETERS ────────────────────────────────────────────────
C_m   = 1.0
g_Na  = 120.0;  E_Na =  50.0
g_K   =  36.0;  E_K  = -77.0
g_L   =   0.3;  E_L  = -54.387
V_rest   = -65.0
V_thresh = -55.0

# ── GATE KINETICS (scalar) ───────────────────────────────────────────────────
def rates(v):
    dvm = v + 40
    am  = 1.0 if abs(dvm) < 1e-7 else 0.1 * dvm / (1 - np.exp(-dvm / 10))
    bm  = 4.0 * np.exp(-(v + 65) / 18)
    ah  = 0.07 * np.exp(-(v + 65) / 20)
    bh  = 1.0 / (1 + np.exp(-(v + 35) / 10))
    dvn = v + 55
    an  = 0.1 if abs(dvn) < 1e-7 else 0.01 * dvn / (1 - np.exp(-dvn / 10))
    bn  = 0.125 * np.exp(-(v + 65) / 80)
    return am, bm, ah, bh, an, bn

def init_gates(v0):
    am,bm,ah,bh,an,bn = rates(v0)
    return am/(am+bm), ah/(ah+bh), an/(an+bn)

# ── SIMULATOR ────────────────────────────────────────────────────────────────
def simulate(I_func, t_end=100, dt=0.01):
    t  = np.arange(0, t_end, dt)
    V  = np.zeros(len(t))
    m  = np.zeros(len(t))
    h  = np.zeros(len(t))
    n  = np.zeros(len(t))
    V[0] = V_rest
    m[0], h[0], n[0] = init_gates(V_rest)

    for i in range(1, len(t)):
        v, mi, hi, ni = V[i-1], m[i-1], h[i-1], n[i-1]
        am,bm,ah,bh,an,bn = rates(v)

        I_Na = g_Na * mi**3 * hi * (v - E_Na)
        I_K  = g_K  * ni**4      * (v - E_K)
        I_L  = g_L               * (v - E_L)

        V[i] = v  + ((I_func(t[i-1]) - I_Na - I_K - I_L) / C_m) * dt
        m[i] = mi + (am*(1-mi) - bm*mi) * dt
        h[i] = hi + (ah*(1-hi) - bh*hi) * dt
        n[i] = ni + (an*(1-ni) - bn*ni) * dt

    return t, V, g_Na*m**3*h, g_K*n**4

# ── FIGURE SETUP ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor='#080808')
fig.suptitle('NEURON BLUEPRINT  —  Hodgkin-Huxley Model',
             fontsize=17, color='#00ff88', fontweight='bold', y=0.985,
             fontfamily='monospace')

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.55, wspace=0.42,
                       top=0.94, bottom=0.07, left=0.07, right=0.97)

C = dict(V='#00ff88', thresh='#ff4444', Na='#4499ff',
         K='#ffaa00', dim='#666666', title='#00ccff', label='#aaaaaa')

def style(ax, title):
    ax.set_facecolor('#0f0f0f')
    for sp in ['top','right']:          ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']:        ax.spines[sp].set_color('#2a2a2a')
    ax.tick_params(colors='#555555', labelsize=7)
    ax.set_title(title, color=C['title'], fontsize=8.5, pad=6, fontweight='bold',
                 fontfamily='monospace')
    for lbl in [ax.yaxis.label, ax.xaxis.label]:
        lbl.set_color(C['label']); lbl.set_size(7.5)

def hlines(ax, t_end=None):
    ax.axhline(V_thresh, color=C['thresh'], lw=0.7, ls='--', alpha=0.6)
    ax.axhline(V_rest,   color=C['dim'],    lw=0.7, ls='--', alpha=0.4)

# ── [1] SINGLE ACTION POTENTIAL ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
t,V,gNa,gK = simulate(lambda t: 10.0 if 10<=t<=11 else 0.0, t_end=60)
ax1.plot(t, V, color=C['V'], lw=1.6)
hlines(ax1)
ax1.text(57, V_thresh+1.5, '−55 mV', color=C['thresh'], fontsize=6, ha='right')
ax1.text(57, V_rest+1.5,   '−65 mV', color=C['dim'],    fontsize=6, ha='right')
ax1.set_ylim(-90, 60); ax1.set_ylabel('V (mV)'); ax1.set_xlabel('time (ms)')
style(ax1, '[1] SINGLE ACTION POTENTIAL')

# ── [2] Na⁺ / K⁺ CONDUCTANCES ─────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t, gNa, color=C['Na'], lw=1.5, label='g_Na  (Na⁺ rushes IN)')
ax2.plot(t, gK,  color=C['K'],  lw=1.5, label='g_K   (K⁺ rushes OUT)')
ax2.fill_between(t, gNa, alpha=0.08, color=C['Na'])
ax2.fill_between(t, gK,  alpha=0.08, color=C['K'])
ax2.set_ylabel('conductance (mS/cm²)'); ax2.set_xlabel('time (ms)')
ax2.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax2, '[2] Na⁺ / K⁺ CONDUCTANCES')

# ── [3] REFRACTORY PERIOD ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
def I_ref(t):
    if  10<=t<=11: return 10.0   # fires
    if  13<=t<=14: return 10.0   # absolute refractory — blocked
    if  19<=t<=20: return 10.0   # relative refractory — fires but weaker
    return 0.0
t3,V3,_,_ = simulate(I_ref, t_end=60)
ax3.plot(t3, V3, color=C['V'], lw=1.6)
ax3.axvspan(11, 13, alpha=0.18, color='#ff2222', label='absolute  (1–2 ms)')
ax3.axvspan(13, 19, alpha=0.09, color='#ff8800', label='relative  (2–5 ms)')
hlines(ax3)
ax3.set_ylim(-90, 60); ax3.set_ylabel('V (mV)'); ax3.set_xlabel('time (ms)')
ax3.legend(fontsize=6.5, framealpha=0, labelcolor='white', loc='upper right')
style(ax3, '[3] REFRACTORY PERIOD')

# ── [4] TEMPORAL SUMMATION ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
def I_single(t): return 5.5 if 10<=t<=10.5 else 0.0
def I_temporal(t):
    for p in [10, 13, 16, 19]:
        if p <= t <= p+0.5: return 5.5
    return 0.0
_,Va,_,_ = simulate(I_single,  t_end=60)
tb,Vb,_,_ = simulate(I_temporal, t_end=60)
ax4.plot(t,  Va, color=C['dim'], lw=1.1, ls='--', label='single pulse  (fails)')
ax4.plot(tb, Vb, color=C['V'],   lw=1.6,           label='repeated <15 ms  (fires)')
hlines(ax4)
ax4.set_ylim(-90, 60); ax4.set_ylabel('V (mV)'); ax4.set_xlabel('time (ms)')
ax4.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax4, '[4] TEMPORAL SUMMATION')

# ── [5] SPATIAL SUMMATION ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
def pulse(amp): return lambda t: amp if 10<=t<=10.5 else 0.0
_,V1s,_,_ = simulate(pulse(3.0),  t_end=60)
_,V3s,_,_ = simulate(pulse(9.0),  t_end=60)
_,V6s,_,_ = simulate(pulse(15.0), t_end=60)
ax5.plot(t, V1s, color='#1a5c33', lw=1.1, label='1 synapse')
ax5.plot(t, V3s, color='#27a35a', lw=1.3, label='3 synapses')
ax5.plot(t, V6s, color=C['V'],    lw=1.6, label='6 synapses  (fires)')
hlines(ax5)
ax5.set_ylim(-90, 60); ax5.set_ylabel('V (mV)'); ax5.set_xlabel('time (ms)')
ax5.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax5, '[5] SPATIAL SUMMATION')

# ── [6] LEAKY INTEGRATION WINDOW ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
def I_stacked(t):
    for p in [10, 13, 16, 19, 22]:
        if p <= t <= p+0.5: return 5.5
    return 0.0
def I_spread(t):
    for p in [10, 30, 50]:
        if p <= t <= p+0.5: return 5.5
    return 0.0
_,Vs,_,_ = simulate(I_stacked, t_end=60)
_,Vsp,_,_ = simulate(I_spread,  t_end=60)
ax6.plot(t, Vsp, color=C['dim'], lw=1.1, ls='--', label='spread >15 ms  (each decays)')
ax6.plot(t, Vs,  color=C['V'],   lw=1.6,           label='stacked <15 ms  (accumulates)')
hlines(ax6)
ax6.set_ylim(-90, 60); ax6.set_ylabel('V (mV)'); ax6.set_xlabel('time (ms)')
ax6.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax6, '[6] LEAKY INTEGRATION WINDOW')

# ── [7] TWO NEURONS CONNECTED ──────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, :2])
t1, V1, _, _ = simulate(lambda t: 10.0 if 10<=t<=11 else 0.0, t_end=100)
spike_idx = np.where((V1[:-1] < 0) & (V1[1:] >= 0))[0]
delay = 2.0; dt = 0.01

def I_n2(t):
    for si in spike_idx:
        t0 = si*dt + delay
        if t0 <= t <= t0+0.5: return 14.0
    return 0.0

t2, V2, _, _ = simulate(I_n2, t_end=100)

offset = 110
ax7.plot(t1, V1 + offset, color=C['Na'], lw=1.5, label='Neuron 1  (stimulus)')
ax7.plot(t2, V2,          color=C['V'],  lw=1.5, label='Neuron 2  (downstream)')
ax7.axhline(V_thresh + offset, color=C['Na'],    lw=0.6, ls='--', alpha=0.4)
ax7.axhline(V_thresh,          color=C['thresh'],lw=0.6, ls='--', alpha=0.4)

ax7.text(2, offset+V_rest+2, 'N1', color=C['Na'],    fontsize=7)
ax7.text(2,        V_rest+2, 'N2', color=C['V'],     fontsize=7)
if len(spike_idx):
    t_fire = spike_idx[0]*dt
    ax7.text(t_fire + 2, offset + V_rest + 8, 'N1 fires',
             color=C['Na'], fontsize=7, fontfamily='monospace')
    t_n2 = t_fire + delay + 2
    ax7.text(t_n2, V_rest + 8, f'+{delay}ms delay',
             color=C['V'], fontsize=7, fontfamily='monospace')

ax7.set_ylim(-90, 180); ax7.set_ylabel('V (mV)  —  N1 offset +110 for clarity')
ax7.set_xlabel('time (ms)')
ax7.legend(fontsize=8, framealpha=0, labelcolor='white', loc='upper right')
style(ax7, '[7] TWO NEURONS CONNECTED  —  synaptic transmission')

# ── [8] PHASE PORTRAIT ─────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
tp,Vp,_,_ = simulate(lambda t: 10.0 if 10<=t<=11 else 0.0, t_end=80)
dVp = np.gradient(Vp, dt)
ax8.plot(Vp, dVp, color=C['V'], lw=1.4, alpha=0.9)
ax8.scatter([V_rest],   [0], color='#ff4444', s=40, zorder=5, label='rest point')
ax8.scatter([V_thresh], [dVp[np.argmin(np.abs(Vp[:300]-V_thresh))]],
            color='#ffaa00', s=40, zorder=5, label='threshold')
ax8.axhline(0, color='#2a2a2a', lw=0.8)
ax8.axvline(V_thresh, color=C['thresh'], lw=0.5, ls='--', alpha=0.4)
ax8.set_xlabel('V (mV)'); ax8.set_ylabel('dV/dt  (mV/ms)')
ax8.legend(fontsize=7, framealpha=0, labelcolor='white')
style(ax8, '[8] PHASE PORTRAIT  —  V vs dV/dt')

# ── SAVE ─────────────────────────────────────────────────────────────────────
out = '/Users/winstonlin/Desktop/neuron_blueprint.png'
plt.savefig(out, dpi=160, facecolor='#080808', bbox_inches='tight')
print(f"saved → {out}")
plt.show()
