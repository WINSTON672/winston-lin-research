# SPECTRUM bench

Two things the SPECTRUM draft needs before it can be submitted anywhere. One
runs today with no hardware; the other needs about £6 of parts and an
afternoon.

## 1. `baselines.py` — runs now

Compares DST against LIF, Izhikevich (2003) and AdEx (2005) under identical
stimuli, each driven at a multiple of its own rheobase so the comparison is
about the models rather than their units.

```bash
python3 baselines.py     # -> baselines_table.md, baselines_figure.png
```

**Why this exists.** The draft cites Hodgkin-Huxley and LIF and concludes that
three variables buy a repertoire one variable cannot reach. That is true and it
is not the comparison a reviewer will make. Izhikevich reaches the same
repertoire with two variables and four parameters, and was written for the same
stated reason — HH is too expensive for hardware. It is not cited anywhere in
`spectrum_build.tex`. Until it is, the paper's central claim reads as one that
was answered in 2003 with one variable fewer.

**What the run actually found**, which is better than expected:

- The variable-count framing does not survive. Drop it.
- DST is the only model tested that holds an **absolute refractory period**
  under a pulse pair at 20× rheobase. Izhikevich and AdEx recover through a
  slow current and can be forced to fire again immediately.
- DST is the **fastest of the three adapting models** in wall clock despite
  carrying the extra variable, because it is entirely linear — Izhikevich pays
  for a quadratic term that needs sub-stepping, AdEx for an exponential.

That last point is the paper. Not "fewer variables" but "no nonlinearity to
fabricate", which is also exactly why the D compartment is a capacitor and a
resistor rather than a transistor network.

## 2. The D compartment — needs parts

`spectrum_build.tex` §8.1 is titled **"D Compartment (Built)"** but contains a
component table and a test protocol — instructions for an experiment, not its
result. The voltages quoted in §7.3 under the heading "Physical D Compartment"
come from `D_compartment.py`, a simulator. Running this bench is what makes
that heading true. Until then the heading should read *(Proposed)*.

### Parts

| Part | Value | Notes |
|---|---|---|
| Raspberry Pi Pico | any | drives the pulses and samples; no oscilloscope needed |
| Electrolytic capacitor | 10 µF | **polarised** — + to node, striped leg to GND |
| Resistor | 1 kΩ | bleeder, sets τ = RC = 10 ms |
| Resistors ×3 | 2 kΩ | synaptic weights, w = 1/R |
| Breadboard + jumpers | | |

### Wiring

```
    GP2 ---[ 2k ]---+
    GP3 ---[ 2k ]---+--- NODE ---> GP26 (ADC0)
    GP4 ---[ 2k ]---+
                    |
                    +---[ 10uF ]--- GND     (+ leg to NODE)
                    |
                    +---[  1k  ]--- GND     (bleeder)
```

### Run

```bash
pip3 install pyserial
# copy pico_dcompartment.py onto the Pico as main.py, then:
python3 capture.py            # -> measured.csv
python3 overlay.py            # -> overlay_figure.png, overlay_report.md
```

`capture.py --list` shows candidate ports if auto-detection picks wrong.

### Two things that will ruin the measurement

1. **Idle synapse pins must be high-Z, not driven low.** A pin held low puts
   its 2 kΩ in parallel with the 1 kΩ bleeder. Three of them idling low gives
   R_eff = 500 Ω and τ = 5 ms — a 2× error that looks exactly like a bad
   capacitor. The firmware releases pins to `Pin.IN` for this reason.

2. **The Pico drives 3.3 V, the paper says 2.0 V.** The circuit is linear so
   amplitude simply scales, but the simulation has to be re-run at the same
   voltage before the traces are compared. `overlay.py --drive 3.3` is the
   default and handles it.

### Expect τ ≠ 10 ms

Electrolytics are commonly ±20%, some are −20/+80%, they leak, and the Pico's
ADC has its own offset. A measured 8.4 ms or 12.1 ms is not a failed
experiment — it *is* the experiment. `overlay.py` reports the fitted τ and the
capacitance it implies, which is a more honest number than the one printed on
the part.

## Verification status

`baselines.py` has been run; its table and figure in this directory are real
output. The capture/overlay path has been exercised end to end against a
synthetic trace only — the τ fitter recovered 8.60 ms from a synthetic 8.6 µF
part with R² = 1.0000 — and that synthetic data was deleted rather than left
here. **No measured hardware data exists yet.** Nothing in this directory
should be cited as a hardware result until `measured.csv` comes off a real
board.
