"""
overlay.py -- measured D compartment against the simulation.
============================================================

    python3 overlay.py [--csv measured.csv] [--drive 3.3]

Produces overlay_figure.png and overlay_report.md.

This is the figure the paper is missing. Section 8.1 of the draft is titled
"D Compartment (Built)" but contains a component table and a test protocol --
instructions for an experiment, not its result -- while the voltages quoted in
Section 7.3 come out of D_compartment.py, a simulator. Running this script
against real hardware is what makes that heading true.

The RC integrator below is a re-implementation of simulate_RC() from
D_compartment.py rather than an import, because that file builds its figure at
module level and importing it would run the whole thing. The equations are
identical; if you change one, change both.

Expect the measured time constant to miss 10 ms. Electrolytic capacitors are
commonly +/-20% and some are -20/+80%, they have real leakage, and the Pico's
ADC has its own error. A measured tau of 8.4 or 12.1 ms is not a failed
experiment -- it is the experiment, and the implied capacitance is a more
interesting number than the one printed on the part.
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('dark_background')
CLR = dict(meas='#00ff88', sim='#ff6644', fit='#ffaa00',
           title='#00ccff', label='#888888')

# ── circuit constants, from D_compartment.py ────────────────────────────────
C_CAP = 10e-6         # 10 uF nominal
R_BLEED = 1e3         # 1 kOhm nominal
R_SYN = 2e3           # 2 kOhm per synapse
TAU_NOMINAL = R_BLEED * C_CAP     # 10 ms

# Event schedules, mirroring pico_dcompartment.py. (amp_scale, t_on, t_off) in
# ms; amp_scale counts how many synapses are driven.
SCHEDULES = {
    'tau':       [(1, 10.0, 15.0)],
    'spatial1':  [(1, 10.0, 15.0)],
    'spatial2':  [(2, 10.0, 15.0)],
    'spatial3':  [(3, 10.0, 15.0)],
    'temporal8': [(1, 10.0, 15.0), (1, 23.0, 28.0)],
    'temporal45': [(1, 10.0, 15.0), (1, 60.0, 65.0)],
}


def simulate_rc(schedule, drive_v, t_end_ms, dt=1e-5):
    """dV/dt = (I_in - I_leak) / C, I_in = n*(drive - V)/R_syn, I_leak = V/R.

    Same integrator as D_compartment.py: n parallel synapses each source
    current through their own weight resistor, so the drive is loaded by the
    node voltage rather than being an ideal current source.
    """
    t = np.arange(0.0, t_end_ms * 1e-3, dt)
    V = np.zeros(len(t))
    for i in range(1, len(t)):
        ti, v = t[i - 1], V[i - 1]
        i_in = 0.0
        for n, t_on, t_off in schedule:
            if t_on * 1e-3 <= ti < t_off * 1e-3:
                i_in += n * (drive_v - v) / R_SYN
        V[i] = v + ((i_in - v / R_BLEED) / C_CAP) * dt
    return t * 1e3, V


def fit_tau(t_ms, v):
    """Least-squares tau on the decay after the peak.

    Fitted on log V over the window from just after the peak down to 20% of
    it: earlier and the pulse is still on, later and the ADC quantisation
    dominates the log.
    """
    k = int(np.argmax(v))
    peak = v[k]
    if peak <= 0:
        return None, None
    tail_t, tail_v = t_ms[k + 1:], v[k + 1:]
    keep = (tail_v > 0.20 * peak) & (tail_v > 1e-4)
    if keep.sum() < 10:
        return None, None
    x, y = tail_t[keep], np.log(tail_v[keep])
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None, None
    tau = -1.0 / slope
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float('nan')
    return tau, r2


def load(path):
    try:
        raw = np.genfromtxt(path, delimiter=',', names=True,
                            dtype=['U16', 'f8', 'f8'], encoding='utf-8')
    except OSError:
        sys.exit(f'{path} not found -- run capture.py first.')
    if raw.size == 0:
        sys.exit(f'{path} is empty.')
    out = {}
    for name in np.unique(raw['test']):
        sel = raw[raw['test'] == name]
        order = np.argsort(sel['t_us'])
        out[str(name)] = (sel['t_us'][order] / 1e3, sel['volts'][order])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='measured.csv')
    ap.add_argument('--drive', type=float, default=3.3,
                    help='pin drive voltage; Pico is 3.3 V, paper says 2.0 V')
    args = ap.parse_args()

    data = load(args.csv)
    names = [n for n in SCHEDULES if n in data]
    if not names:
        sys.exit(f'no recognised tests in {args.csv}; found '
                 f'{sorted(data)}')

    rows = []
    ncol = 2
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.1 * nrow),
                             facecolor='#080808', squeeze=False)
    fig.suptitle('D compartment  --  measured vs simulated  '
                 f'(drive {args.drive:.1f} V)',
                 color=CLR['meas'], fontsize=13, fontweight='bold',
                 fontfamily='monospace', y=0.995)

    for ax, name in zip(axes.flat, names):
        tm, vm = data[name]
        ts, vs = simulate_rc(SCHEDULES[name], args.drive, tm[-1])

        ax.set_facecolor('#0d0d0d')
        ax.plot(ts, vs, color=CLR['sim'], lw=1.6, alpha=0.85,
                label='simulated')
        ax.plot(tm, vm, color=CLR['meas'], lw=0.9, label='measured')
        ax.set_title(name, color=CLR['title'], fontsize=9, loc='left',
                     fontfamily='monospace', pad=4)
        ax.set_xlabel('t (ms)', color=CLR['label'], fontsize=8)
        ax.set_ylabel('V_D (V)', color=CLR['label'], fontsize=8)
        ax.tick_params(colors='#444444', labelsize=7)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=7, framealpha=0.0, labelcolor='#aaaaaa')

        tau_m, r2 = fit_tau(tm, vm)
        tau_s, _ = fit_tau(ts, vs)
        peak_m, peak_s = float(vm.max()), float(vs.max())
        rows.append(dict(test=name, tau_m=tau_m, r2=r2, tau_s=tau_s,
                         peak_m=peak_m, peak_s=peak_s))
        if tau_m:
            ax.annotate(f'tau_meas = {tau_m:.2f} ms',
                        xy=(0.97, 0.9), xycoords='axes fraction',
                        ha='right', color=CLR['fit'], fontsize=8,
                        fontfamily='monospace')

    for ax in axes.flat[len(names):]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig('overlay_figure.png', dpi=150, facecolor='#080808')

    # ── report ──────────────────────────────────────────────────────────────
    out = ['# D compartment: measured vs simulated', '',
           f'Drive {args.drive:.2f} V. Nominal R = {R_BLEED:.0f} ohm, '
           f'C = {C_CAP * 1e6:.0f} uF, tau = R*C = {TAU_NOMINAL * 1e3:.1f} ms.',
           '',
           '| Test | tau measured (ms) | tau simulated (ms) | fit R2 | '
           'peak measured (V) | peak simulated (V) |',
           '|---|---|---|---|---|---|']
    for r in rows:
        f = lambda x, p=2: f'{x:.{p}f}' if x is not None else '--'
        out.append(f"| {r['test']} | {f(r['tau_m'])} | {f(r['tau_s'])} "
                   f"| {f(r['r2'], 4)} | {f(r['peak_m'], 3)} "
                   f"| {f(r['peak_s'], 3)} |")

    taus = [r['tau_m'] for r in rows if r['tau_m']]
    if taus:
        tau_mean = float(np.mean(taus))
        c_implied = tau_mean * 1e-3 / R_BLEED
        err = 100.0 * (tau_mean * 1e-3 - TAU_NOMINAL) / TAU_NOMINAL
        out += ['', '## Implied component values', '',
                f'- Mean measured tau: **{tau_mean:.2f} ms** '
                f'({err:+.1f}% vs the {TAU_NOMINAL * 1e3:.0f} ms nominal)',
                f'- Implied capacitance at R = {R_BLEED:.0f} ohm: '
                f'**{c_implied * 1e6:.2f} uF**',
                '',
                'The deviation is the measurement, not an error in it. An '
                'electrolytic part at +/-20% covers 8-12 ms on its own; '
                'dielectric leakage pulls tau down further, and the Pico ADC '
                'contributes its own offset. Quote the measured value and the '
                'implied capacitance, and say which instrument produced them.']

    with open('overlay_report.md', 'w') as fh:
        fh.write('\n'.join(out) + '\n')

    print('\n'.join(out))
    print('\nwrote overlay_figure.png, overlay_report.md')


if __name__ == '__main__':
    main()
