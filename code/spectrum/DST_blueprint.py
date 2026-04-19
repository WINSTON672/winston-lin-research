"""
DST NEURON — MANUFACTURING BLUEPRINT
=====================================
Full technical drawings for physical construction of the DST neuron.
Three compartments: D (dendrite), S (soma), θ (threshold)
Suitable for handoff to electronics manufacturer or PCB fabricator.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

# ── BLUEPRINT STYLE ──────────────────────────────────────────────────────────
BG      = '#0a1628'    # blueprint blue
GRID    = '#0d1f3c'
LINE    = '#c8deff'    # white-blue lines
DIM     = '#7aadff'    # dimension lines
RED     = '#ff6b6b'    # warning / critical
GREEN   = '#6bff9e'    # verified / ok
YELLOW  = '#ffe66b'    # notes
ORANGE  = '#ffb347'    # component labels

plt.rcParams['font.family'] = 'monospace'

fig = plt.figure(figsize=(24, 17), facecolor=BG)

# title block
fig.text(0.5, 0.987,
         'DST NEURON — MANUFACTURING BLUEPRINT',
         ha='center', va='top', color=LINE,
         fontsize=18, fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.968,
         'Dendritic / Somatic / Threshold  ·  Rev 1.0  ·  Winston Lin  ·  2026',
         ha='center', va='top', color=DIM, fontsize=10, fontfamily='monospace')

# border
border = plt.axes([0.005, 0.005, 0.990, 0.990], facecolor='none')
border.set_xlim(0,1); border.set_ylim(0,1); border.axis('off')
for lw, alpha in [(3, 0.6), (1, 0.3)]:
    border.add_patch(Rectangle((0,0),1,1, fill=False,
                                edgecolor=LINE, linewidth=lw, alpha=alpha))

gs = gridspec.GridSpec(3, 4, figure=fig,
                       hspace=0.45, wspace=0.35,
                       top=0.955, bottom=0.04,
                       left=0.03, right=0.97)

def bp_ax(ax, title, subtitle=''):
    ax.set_facecolor(GRID)
    # grid
    for x in np.linspace(0,1,21):
        ax.plot([x,x],[0,1], color='#0f2444', lw=0.3, alpha=0.6,
                transform=ax.transAxes)
    for y in np.linspace(0,1,21):
        ax.plot([0,1],[y,y], color='#0f2444', lw=0.3, alpha=0.6,
                transform=ax.transAxes)
    for sp in ax.spines.values():
        sp.set_color(DIM); sp.set_linewidth(1.2)
    ax.tick_params(colors=DIM, labelsize=7.5, labelcolor=DIM)
    ax.set_title(f'  {title}', color=LINE, fontsize=9.5, pad=5,
                 fontweight='bold', loc='left', fontfamily='monospace')
    if subtitle:
        ax.text(0.98, 1.02, subtitle, transform=ax.transAxes,
                color=YELLOW, fontsize=7, ha='right', va='bottom',
                fontfamily='monospace')
    ax.xaxis.label.set_color(DIM); ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_color(DIM); ax.yaxis.label.set_size(8)

def draw_resistor(ax, x, y, w=0.8, h=0.25, label='', color=ORANGE, angle=0):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h,
                        boxstyle='square,pad=0.02',
                        linewidth=1.5, edgecolor=color,
                        facecolor='#1a2a4a')
    ax.add_patch(r)
    ax.text(x, y, label, color=color, fontsize=6.5,
            ha='center', va='center', fontfamily='monospace')

def draw_capacitor(ax, x, y, label='', color='#6baaff'):
    ax.plot([x-0.4, x+0.4], [y+0.12, y+0.12], color=color, lw=3)
    ax.plot([x-0.4, x+0.4], [y-0.12, y-0.12], color=color, lw=3)
    ax.text(x, y-0.45, label, color=color, fontsize=6.5,
            ha='center', fontfamily='monospace')

def draw_opamp(ax, x, y, color=LINE):
    tri = plt.Polygon([[x-0.5,y+0.5],[x-0.5,y-0.5],[x+0.5,y]],
                       closed=True, fill=True, facecolor='#1a2a4a',
                       edgecolor=color, lw=1.5)
    ax.add_patch(tri)
    ax.text(x+0.02, y, 'A', color=color, fontsize=8,
            ha='center', va='center', fontfamily='monospace')

def draw_gnd(ax, x, y, color=DIM):
    ax.plot([x,x],[y,y-0.2], color=color, lw=1.5)
    for i,w in enumerate([0.3,0.2,0.1]):
        ax.plot([x-w,x+w],[y-0.2-i*0.1,y-0.2-i*0.1], color=color, lw=1.5)

def wire(ax, pts, color=LINE, lw=1.5, ls='-'):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle='round')

def node(ax, x, y, color=LINE):
    ax.plot(x, y, 'o', color=color, markersize=4, zorder=5)

def label(ax, x, y, text, color=LINE, size=7.5, ha='left', va='center'):
    ax.text(x, y, text, color=color, fontsize=size,
            ha=ha, va=va, fontfamily='monospace')

def dim_line(ax, x0,y0,x1,y1, text, offset=0.15):
    mx, my = (x0+x1)/2, (y0+y1)/2
    ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle='<->', color=DIM, lw=1))
    ax.text(mx, my+offset, text, color=DIM, fontsize=6.5,
            ha='center', fontfamily='monospace')


# ════════════════════════════════════════════════════════════════════════════════
# [A] D COMPARTMENT SCHEMATIC
# ════════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs[0, :2])
bp_ax(ax_d, 'D COMPARTMENT — DENDRITIC INTEGRATOR', 'τ_D = R·C = 10ms')
ax_d.set_xlim(0, 14); ax_d.set_ylim(0, 8)
ax_d.axis('off')

# rails
wire(ax_d, [(1,7),(13,7)], color=DIM, lw=0.8, ls='--')   # VCC rail (dashed ref)
wire(ax_d, [(1,0.8),(13,0.8)], color=DIM, lw=1.2)         # GND rail
label(ax_d, 0.3, 0.8, 'GND', DIM, 7)

# three synaptic input lines
syn_data = [
    (1, 6.2, 'SYN_1', '2kΩ', 'w=0.5'),
    (1, 4.5, 'SYN_2', '2kΩ', 'w=0.5'),
    (1, 2.8, 'SYN_3', '2kΩ', 'w=0.5'),
]
junction_x = 6.5

for sx, sy, name, rval, wval in syn_data:
    # input terminal
    ax_d.plot(sx, sy, 's', color=GREEN, markersize=7, zorder=5)
    label(ax_d, sx-0.7, sy, name, GREEN, 7, ha='right')
    # wire to resistor
    wire(ax_d, [(sx,sy),(sx+1.3,sy)])
    # resistor
    draw_resistor(ax_d, sx+2.2, sy, label=rval)
    # wire from resistor to junction
    wire(ax_d, [(sx+2.7,sy),(junction_x,sy)])
    # wire up to main node
    wire(ax_d, [(junction_x,sy),(junction_x,4.5 if sy!=4.5 else sy)])
    node(ax_d, junction_x, sy)
    # weight annotation
    label(ax_d, sx+2.2, sy+0.42, wval, YELLOW, 6, ha='center')

# main vertical junction wire
wire(ax_d, [(junction_x,2.8),(junction_x,6.2)])
node(ax_d, junction_x, 4.5)

# bleeder resistor (vertical)
wire(ax_d, [(7.8,6.2),(7.8,5.5)])
draw_resistor(ax_d, 7.8, 4.8, w=0.5, h=1.1, label='R_bleed\n1kΩ', color=RED)
wire(ax_d, [(7.8,4.2),(7.8,0.8)])
# connect to junction
wire(ax_d, [(junction_x,4.5),(7.8,4.5)])
node(ax_d, 7.8, 4.5)
label(ax_d, 8.15, 5.5, '← LEAK', RED, 7)

# capacitor
wire(ax_d, [(9.5,6.2),(9.5,5.3)])
draw_capacitor(ax_d, 9.5, 4.8, label='C\n10µF')
wire(ax_d, [(9.5,4.3),(9.5,0.8)])
# connect to junction
wire(ax_d, [(junction_x,6.2),(9.5,6.2)])
node(ax_d, junction_x, 6.2)
label(ax_d, 9.5, 6.5, 'V_D\nmeasure\nhere', GREEN, 6.5, ha='center')

# op-amp buffer
wire(ax_d, [(9.5,4.9),(10.5,4.9)])
draw_opamp(ax_d, 11.2, 4.9)
wire(ax_d, [(11.7,4.9),(12.5,4.9)])
ax_d.plot(12.5, 4.9, 's', color=YELLOW, markersize=8, zorder=5)
label(ax_d, 12.7, 4.9, '→ TO S', YELLOW, 8)
label(ax_d, 11.2, 3.9, 'BUFFER\n(unity gain)', DIM, 6.5, ha='center')

# dimension annotations
dim_line(ax_d, 1.3, 1.4, 6.3, 1.4, 'SYNAPTIC INPUT SECTION', offset=0.2)
dim_line(ax_d, 7.3, 1.4, 10.0, 1.4, 'INTEGRATION\nNODE', offset=0.2)
dim_line(ax_d, 10.5, 1.4, 12.8, 1.4, 'OUTPUT\nSTAGE', offset=0.2)

# spec box
spec = ax_d.inset_axes([0.73, 0.55, 0.26, 0.42])
spec.set_facecolor('#0a1628'); spec.axis('off')
for sp in spec.spines.values(): sp.set_color(DIM); sp.set_linewidth(0.8)
specs = [
    ('PARAMETER', 'VALUE', 'UNIT'),
    ('─'*10,    '─'*7,   '─'*6),
    ('C',       '10',    'µF'),
    ('R_bleed', '1',     'kΩ'),
    ('τ_D',     '10',    'ms'),
    ('R_syn',   '2',     'kΩ / syn'),
    ('V_rail',  '5',     'V'),
    ('V_thresh','0.45',  'V'),
    ('Op-amp',  'TL071', '—'),
]
for i,(p,v,u) in enumerate(specs):
    y = 0.92 - i*0.11
    spec.text(0.02,y, p, color=LINE if i>1 else YELLOW, fontsize=6.5,
              fontfamily='monospace', va='top')
    spec.text(0.52,y, v, color=GREEN if i>1 else YELLOW, fontsize=6.5,
              fontfamily='monospace', va='top')
    spec.text(0.78,y, u, color=DIM, fontsize=6.5,
              fontfamily='monospace', va='top')


# ════════════════════════════════════════════════════════════════════════════════
# [B] S COMPARTMENT SCHEMATIC
# ════════════════════════════════════════════════════════════════════════════════
ax_s = fig.add_subplot(gs[0, 2:])
bp_ax(ax_s, 'S COMPARTMENT — SOMATIC THRESHOLD', 'COMPARATOR + SELF-RESET')
ax_s.set_xlim(0, 14); ax_s.set_ylim(0, 8)
ax_s.axis('off')

# input from D
ax_s.plot(0.5, 5.5, 's', color=YELLOW, markersize=8, zorder=5)
label(ax_s, 0.1, 5.5, 'FROM D', YELLOW, 7.5, ha='left')
wire(ax_s, [(0.5,5.5),(2.5,5.5)])

# comparator (drawn as triangle)
comp = plt.Polygon([[2.5,6.5],[2.5,4.0],[4.5,5.5]],
                    closed=True, fill=True, facecolor='#1a2a4a',
                    edgecolor=LINE, lw=1.5)
ax_s.add_patch(comp)
label(ax_s, 3.2, 5.5, 'LM393\nCOMP', LINE, 7, ha='center')
label(ax_s, 2.3, 6.2, '+', GREEN, 9)
label(ax_s, 2.3, 4.8, '−', RED,   9)

# reference voltage (theta input)
wire(ax_s, [(0.5,4.0),(2.5,4.0)])
ax_s.plot(0.5, 4.0, 's', color=RED, markersize=8, zorder=5)
label(ax_s, 0.1, 4.0, 'FROM θ\n(ref V)', RED, 7, ha='left')

# output wire
wire(ax_s, [(4.5,5.5),(5.5,5.5)])

# output node + feedback
node(ax_s, 5.5, 5.5)

# self-reset RC (feedback to discharge input)
wire(ax_s, [(5.5,5.5),(5.5,7.0),(3.5,7.0)])
draw_resistor(ax_s, 3.5, 7.0, w=0.9, h=0.35, label='R_fb\n10kΩ', color=RED)
wire(ax_s, [(3.0,7.0),(2.5,7.0),(2.5,6.5)])
label(ax_s, 3.5, 7.6, 'SELF-RESET FEEDBACK', RED, 7, ha='center')

# output spike shaper
wire(ax_s, [(5.5,5.5),(6.5,5.5)])
draw_resistor(ax_s, 7.3, 5.5, w=0.9, h=0.35, label='R_out\n1kΩ')
wire(ax_s, [(7.8,5.5),(8.5,5.5)])
node(ax_s, 8.5, 5.5)

# spike output terminal
wire(ax_s, [(8.5,5.5),(9.5,5.5)])
ax_s.plot(9.5, 5.5, 's', color=GREEN, markersize=8, zorder=5)
label(ax_s, 9.7, 5.5, '→ SPIKE\nOUT', GREEN, 7.5)

# feedback to theta
wire(ax_s, [(8.5,5.5),(8.5,3.5),(9.5,3.5)])
ax_s.plot(9.5, 3.5, 's', color=ORANGE, markersize=8, zorder=5)
label(ax_s, 9.7, 3.5, '→ TO θ\n(notify)', ORANGE, 7.5)

# GND
draw_gnd(ax_s, 3.5, 4.0)
wire(ax_s, [(2.5,4.0),(3.5,4.0),(3.5,4.0)])

# hysteresis band annotation
ax_s.annotate('', xy=(2.5, 6.0), xytext=(2.5, 5.0),
              arrowprops=dict(arrowstyle='<->', color=YELLOW, lw=1.2))
label(ax_s, 1.5, 5.5, 'hysteresis\nband\n50mV', YELLOW, 6.5, ha='center')

# threshold line on comparator
wire(ax_s, [(1.5,4.0),(1.5,3.2)])
draw_resistor(ax_s, 1.5, 2.7, w=0.7, h=0.35, label='R1\n10kΩ', color=DIM)
wire(ax_s, [(1.5,2.3),(1.5,0.8)])
draw_resistor(ax_s, 1.5, 1.2, w=0.7, h=0.35, label='R2\n10kΩ', color=DIM)
wire(ax_s, [(0.5,0.8),(13,0.8)], color=DIM, lw=1.2)
label(ax_s, 0.3, 0.8, 'GND', DIM, 7)
label(ax_s, 1.1, 3.5, 'V_ref\ndivider', DIM, 6.5, ha='right')

# spec box
spec2 = ax_s.inset_axes([0.73, 0.50, 0.26, 0.48])
spec2.set_facecolor('#0a1628'); spec2.axis('off')
for sp in spec2.spines.values(): sp.set_color(DIM); sp.set_linewidth(0.8)
specs2 = [
    ('PARAMETER', 'VALUE', 'UNIT'),
    ('─'*10,     '─'*7,   '─'*6),
    ('IC',       'LM393', '—'),
    ('V_thresh', '0.45',  'V'),
    ('hysteresis','50',   'mV'),
    ('R_fb',     '10',    'kΩ'),
    ('t_reset',  '~1',    'ms'),
    ('V_supply', '5',     'V'),
    ('f_max',    '1',     'kHz'),
]
for i,(p,v,u) in enumerate(specs2):
    y = 0.92 - i*0.11
    spec2.text(0.02,y,p,color=LINE if i>1 else YELLOW,fontsize=6.5,
               fontfamily='monospace',va='top')
    spec2.text(0.52,y,v,color=GREEN if i>1 else YELLOW,fontsize=6.5,
               fontfamily='monospace',va='top')
    spec2.text(0.78,y,u,color=DIM,fontsize=6.5,
               fontfamily='monospace',va='top')


# ════════════════════════════════════════════════════════════════════════════════
# [C] THETA COMPARTMENT
# ════════════════════════════════════════════════════════════════════════════════
ax_th = fig.add_subplot(gs[1, :2])
bp_ax(ax_th, 'θ COMPARTMENT — ADAPTIVE THRESHOLD', 'τ_θ = 80ms  |  Δθ per spike')
ax_th.set_xlim(0, 14); ax_th.set_ylim(0, 8)
ax_th.axis('off')

# spike input from S
ax_th.plot(0.5, 5.5, 's', color=ORANGE, markersize=8, zorder=5)
label(ax_th, 0.1, 5.5, 'SPIKE\nFROM S', ORANGE, 7, ha='left')
wire(ax_th, [(0.5,5.5),(1.5,5.5)])

# diode (only lets spike charge cap, not reverse)
ax_th.annotate('', xy=(2.5,5.5), xytext=(1.5,5.5),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax_th.plot([2.0,2.0],[5.1,5.9], color=GREEN, lw=2)
label(ax_th, 2.0, 6.2, '1N4148\n(diode)', GREEN, 6.5, ha='center')

# charge resistor
wire(ax_th, [(2.5,5.5),(3.0,5.5)])
draw_resistor(ax_th, 3.7, 5.5, label='R_chg\n100Ω')
wire(ax_th, [(4.2,5.5),(5.2,5.5)])

# storage capacitor (holds theta voltage)
node(ax_th, 5.2, 5.5)
wire(ax_th, [(5.2,5.5),(5.2,6.5)])
draw_capacitor(ax_th, 5.2, 4.8, label='C_θ\n1µF')
wire(ax_th, [(5.2,4.3),(5.2,0.8)])
wire(ax_th, [(1,0.8),(13,0.8)], color=DIM, lw=1.2)
label(ax_th, 0.3, 0.8, 'GND', DIM, 7)

# bleed resistor (slow decay = tau_theta = 80ms)
wire(ax_th, [(5.2,5.5),(6.5,5.5)])
draw_resistor(ax_th, 7.2, 5.5, label='R_θ\n80kΩ', color=RED)
wire(ax_th, [(7.7,5.5),(8.5,5.5),(8.5,0.8)])
label(ax_th, 7.2, 6.2, 'τ_θ = R·C\n= 80ms', YELLOW, 6.5, ha='center')

# output buffer to S
wire(ax_th, [(5.2,5.5),(5.2,7.0),(9.5,7.0)])
draw_opamp(ax_th, 10.2, 7.0)
wire(ax_th, [(10.7,7.0),(11.5,7.0)])
ax_th.plot(11.5, 7.0, 's', color=RED, markersize=8, zorder=5)
label(ax_th, 11.7, 7.0, '→ TO S\n(V_ref)', RED, 7.5)
label(ax_th, 10.2, 6.1, 'BUFFER', DIM, 6.5, ha='center')

# timing diagram inset
tins = ax_th.inset_axes([0.55, 0.05, 0.43, 0.55])
tins.set_facecolor('#0a1628')
for sp in tins.spines.values(): sp.set_color(DIM); sp.set_linewidth(0.8)
tins.tick_params(colors=DIM, labelsize=6.5)

t_sim = np.linspace(0, 300, 3000)
tau_theta = 80
theta = np.zeros(len(t_sim))
spike_times_th = [20, 50, 90, 140, 200]
delta_th = 1.2
for i in range(1, len(t_sim)):
    theta[i] = theta[i-1] + (-theta[i-1] / tau_theta) * 0.1
    for ts in spike_times_th:
        if abs(t_sim[i] - ts) < 0.15:
            theta[i] += delta_th

tins.plot(t_sim, theta + 1.0, color=RED, lw=1.5, label='θ(t)')
tins.axhline(1.0, color=DIM, lw=0.8, ls='--', label='θ_0 baseline')
for ts in spike_times_th:
    tins.axvline(ts, color=ORANGE, lw=0.8, ls=':', alpha=0.7)
tins.set_xlabel('time (ms)', fontfamily='monospace', fontsize=6.5, color=DIM)
tins.set_ylabel('θ (V)', fontfamily='monospace', fontsize=6.5, color=DIM)
tins.set_title('θ(t) — adapts with each spike', color=YELLOW,
               fontsize=7, fontfamily='monospace')
tins.legend(fontsize=6.5, framealpha=0, labelcolor='white', loc='upper right')


# ════════════════════════════════════════════════════════════════════════════════
# [D] FULL DST WIRING DIAGRAM
# ════════════════════════════════════════════════════════════════════════════════
ax_w = fig.add_subplot(gs[1, 2:])
bp_ax(ax_w, 'FULL DST — WIRING DIAGRAM', 'connect all three compartments')
ax_w.set_xlim(0, 14); ax_w.set_ylim(0, 10)
ax_w.axis('off')

def block(ax, x, y, w, h, label, color, sublabel=''):
    r = FancyBboxPatch((x-w/2,y-h/2), w, h,
                        boxstyle='round,pad=0.15',
                        linewidth=2, edgecolor=color, facecolor='#0a1628')
    ax.add_patch(r)
    ax.text(x, y+(0.2 if sublabel else 0), label,
            color=color, fontsize=11, ha='center', va='center',
            fontweight='bold', fontfamily='monospace')
    if sublabel:
        ax.text(x, y-0.35, sublabel, color=DIM, fontsize=7,
                ha='center', va='center', fontfamily='monospace')

# three main blocks
block(ax_w, 2.5, 6.5, 3.0, 2.5,  'D',  '#4499ff', 'DENDRITE\n10µF + 1kΩ')
block(ax_w, 7.0, 6.5, 3.0, 2.5,  'S',  GREEN,     'SOMA\nLM393 comp')
block(ax_w, 7.0, 2.5, 3.0, 2.5,  'θ',  RED,       'THRESHOLD\n1µF + 80kΩ')

# D → S (main signal)
ax_w.annotate('', xy=(5.35,6.5), xytext=(4.0,6.5),
              arrowprops=dict(arrowstyle='->', color=YELLOW, lw=2.5))
label(ax_w, 4.7, 7.0, 'V_D\n(analog)', YELLOW, 7.5, ha='center')

# S → output
ax_w.annotate('', xy=(11.5,6.5), xytext=(8.5,6.5),
              arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.5))
label(ax_w, 10.0, 7.1, 'SPIKE OUT\n(digital)', GREEN, 7.5, ha='center')
ax_w.plot(11.5, 6.5, 's', color=GREEN, markersize=10, zorder=5)

# S → θ (spike notification)
ax_w.annotate('', xy=(7.0,3.75), xytext=(7.0,5.25),
              arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))
label(ax_w, 7.6, 4.5, 'fire\nevent', ORANGE, 7.5, ha='left')

# θ → S (adaptive reference)
from matplotlib.patches import FancyArrowPatch
arrow = FancyArrowPatch((5.5, 2.5), (5.5, 5.25),
                         arrowstyle='->', color=RED, lw=2,
                         connectionstyle='arc3,rad=0.4')
ax_w.add_patch(arrow)
label(ax_w, 4.2, 4.0, 'V_θ\n(ref voltage)', RED, 7.5, ha='center')

# synapse inputs to D
for i, (sy, sname) in enumerate([(8.0,'SYN_1'),(6.5,'SYN_2'),(5.0,'SYN_3')]):
    ax_w.plot(0.3, sy, 's', color='#4499ff', markersize=7, zorder=5)
    ax_w.annotate('', xy=(1.0,sy), xytext=(0.3,sy),
                  arrowprops=dict(arrowstyle='->', color='#4499ff', lw=1.5))
    label(ax_w, 0.1, sy, sname, '#4499ff', 7, ha='right')

# GND and VCC
label(ax_w, 13.5, 0.8, 'GND ─────', DIM, 8, ha='right')
label(ax_w, 13.5, 9.5, 'VCC (5V) ─', ORANGE, 8, ha='right')
wire(ax_w, [(0,0.8),(14,0.8)], color=DIM, lw=1.0)
wire(ax_w, [(0,9.5),(14,9.5)], color=ORANGE, lw=0.8, ls='--')

# timing constraints box
tc = ax_w.inset_axes([0.01, 0.01, 0.35, 0.35])
tc.set_facecolor('#0a1628'); tc.axis('off')
for sp in tc.spines.values(): sp.set_color(DIM); sp.set_linewidth(0.8)
timing = [
    'TIMING CONSTRAINTS',
    '──────────────────',
    'τ_D  = 10 ms  (D decay)',
    'τ_S  =  2 ms  (S response)',
    'τ_θ  = 80 ms  (θ recovery)',
    't_spike ≈ 1 ms  (pulse width)',
    't_abs_ref = 1–2 ms',
    't_rel_ref = 2–5 ms',
]
for i, line in enumerate(timing):
    col = YELLOW if i < 2 else LINE
    tc.text(0.05, 0.95-i*0.125, line, color=col, fontsize=6.5,
            fontfamily='monospace', va='top', transform=tc.transAxes)


# ════════════════════════════════════════════════════════════════════════════════
# [E] SIGNAL TIMING DIAGRAM
# ════════════════════════════════════════════════════════════════════════════════
ax_t = fig.add_subplot(gs[2, :2])
bp_ax(ax_t, 'SIGNAL TIMING DIAGRAM', 'all three compartments over time')

t_ms = np.linspace(0, 150, 1500)
dt   = t_ms[1]-t_ms[0]

# simulate D
D_sig  = np.zeros(len(t_ms))
S_sig  = np.zeros(len(t_ms))
TH_sig = np.ones(len(t_ms)) * 0.45
spikes = []

tau_D_t = 10.0; tau_S_t = 2.0; tau_th_t = 80.0
kappa_t = 1.2; S_rest_t = -0.3; dth_t = 0.4; S_reset_t = -0.6

input_events = [(15,18,1.2),(22,25,1.2),(29,32,1.2),(70,73,1.2),(75,78,1.2),(80,83,1.2),(85,88,1.2)]

for i in range(1, len(t_ms)):
    ti = t_ms[i-1]
    I_syn = sum(amp for ts,te,amp in input_events if ts<=ti<te)
    D_sig[i]  = D_sig[i-1]  + ((-D_sig[i-1] + I_syn) / tau_D_t) * dt
    S_sig[i]  = S_sig[i-1]  + ((-(S_sig[i-1]-S_rest_t) + kappa_t*D_sig[i-1]
                                 - (TH_sig[i-1]-0.45)) / tau_S_t) * dt
    TH_sig[i] = TH_sig[i-1] + (-(TH_sig[i-1]-0.45) / tau_th_t) * dt
    if S_sig[i] >= 0:
        spikes.append(t_ms[i])
        S_sig[i]   = S_reset_t
        TH_sig[i] += dth_t

# normalize for display
D_n  = D_sig / (D_sig.max()+1e-9)
S_n  = (S_sig - S_sig.min()) / (S_sig.max()-S_sig.min()+1e-9)
TH_n = (TH_sig - 0.4) / 0.6

ax_t.plot(t_ms, D_n  + 2.2, color='#4499ff', lw=1.8, label='D  (dendritic)')
ax_t.plot(t_ms, S_n  + 1.0, color=GREEN,     lw=1.8, label='S  (somatic)')
ax_t.plot(t_ms, TH_n + 0.0, color=RED,       lw=1.5, label='θ  (threshold)', ls='--')

# spike markers
for sp in spikes:
    ax_t.axvline(sp, color=YELLOW, lw=1.0, ls=':', alpha=0.8)
    ax_t.plot(sp, 2.05, 'v', color=YELLOW, markersize=6, zorder=5)

ax_t.set_yticks([0.5, 1.5, 2.5])
ax_t.set_yticklabels(['θ', 'S', 'D'], color=LINE, fontsize=9)
ax_t.set_xlabel('time (ms)')
ax_t.set_xlim(0, 150)
ax_t.axhline(1.0, color=DIM, lw=0.5, alpha=0.4)
ax_t.axhline(2.2, color=DIM, lw=0.5, alpha=0.4)
ax_t.legend(fontsize=8, framealpha=0.2, labelcolor='white', loc='upper right',
            facecolor='#0a1628')
for ts,te,_ in input_events:
    ax_t.axvspan(ts, te, alpha=0.08, color='#4499ff')


# ════════════════════════════════════════════════════════════════════════════════
# [F] BILL OF MATERIALS
# ════════════════════════════════════════════════════════════════════════════════
ax_bom = fig.add_subplot(gs[2, 2:])
bp_ax(ax_bom, 'BILL OF MATERIALS  —  1× DST NEURON')
ax_bom.axis('off')

bom = [
    ('REF',    'COMPONENT',         'VALUE',   'QTY', 'PACKAGE',    'NOTE'),
    ('─'*5,    '─'*20,              '─'*8,     '─'*4, '─'*12,       '─'*20),
    ('C1',     'Electrolytic cap',  '10µF 10V','1',   'Radial',     'D compartment'),
    ('C2',     'Film cap',          '1µF  16V','1',   'Radial',     'θ storage'),
    ('C3',     'Ceramic cap',       '100nF',   '2',   '0805',       'Decoupling'),
    ('R1',     'Metal film res',    '1kΩ',     '1',   'Axial 1/4W', 'Bleeder (leak)'),
    ('R2–R4',  'Metal film res',    '2kΩ',     '3',   'Axial 1/4W', 'Synaptic weight'),
    ('R5',     'Metal film res',    '80kΩ',    '1',   'Axial 1/4W', 'θ decay (τ=80ms)'),
    ('R6',     'Metal film res',    '100Ω',    '1',   'Axial 1/4W', 'θ charge'),
    ('R7',     'Metal film res',    '10kΩ',    '1',   'Axial 1/4W', 'Feedback'),
    ('R8–R9',  'Metal film res',    '10kΩ',    '2',   'Axial 1/4W', 'V_ref divider'),
    ('R10',    'Metal film res',    '1kΩ',     '1',   'Axial 1/4W', 'Output'),
    ('U1',     'Op-amp',            'TL071',   '1',   'DIP-8',      'D buffer'),
    ('U2',     'Op-amp',            'TL071',   '1',   'DIP-8',      'θ buffer'),
    ('U3',     'Comparator',        'LM393',   '1',   'DIP-8',      'S threshold'),
    ('D1',     'Signal diode',      '1N4148',  '1',   'DO-35',      'θ charge gate'),
    ('J1–J3',  'Input terminal',    '—',       '3',   '2-pin 2.54', 'Synaptic inputs'),
    ('J4',     'Output terminal',   '—',       '1',   '2-pin 2.54', 'Spike output'),
    ('J5',     'Power connector',   '5V / GND','1',   '2-pin 2.54', 'Power supply'),
    ('PCB',    'Perfboard / PCB',   '10×8cm',  '1',   '—',          'Or breadboard'),
]

col_x = [0.01, 0.10, 0.35, 0.55, 0.63, 0.79]
col_w = ['REF','COMPONENT','VALUE','QTY','PACKAGE','NOTE']

for i, row in enumerate(bom):
    y = 0.97 - i * 0.052
    for j, (val, cx) in enumerate(zip(row, col_x)):
        if i == 0:
            color = YELLOW
        elif i == 1:
            color = DIM
        elif j == 3:
            color = ORANGE
        elif j == 0:
            color = GREEN
        else:
            color = LINE
        ax_bom.text(cx, y, val, transform=ax_bom.transAxes,
                    color=color, fontsize=6.8, va='top',
                    fontfamily='monospace')


# ── SAVE ─────────────────────────────────────────────────────────────────────
out = '/Users/winstonlin/Desktop/DST_blueprint.png'
plt.savefig(out, dpi=160, facecolor=BG, bbox_inches='tight')
print(f"saved → {out}")
plt.show()
