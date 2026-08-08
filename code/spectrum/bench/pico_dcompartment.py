"""
pico_dcompartment.py -- MicroPython firmware for the D compartment bench.
=========================================================================
Copy this onto a Raspberry Pi Pico as main.py (or run it from Thonny).

It drives the physical D compartment and samples the capacitor voltage,
emitting CSV over USB serial. Three tests, matching D_compartment.py:

    tau      single pulse, measure the decay time constant
    spatial  1, 2, 3 synapses simultaneously
    temporal two pulses 8 ms apart, then 45 ms apart

WIRING
------
    GP2 ---[ 2k ]---+
    GP3 ---[ 2k ]---+--- NODE ---> GP26 (ADC0)
    GP4 ---[ 2k ]---+
                    |
                    +---[ 10uF ]--- GND      <-- electrolytic: + to NODE
                    |
                    +---[ 1k  ]--- GND       <-- bleeder, sets tau = RC

    tau_D = R_bleed * C = 1k * 10uF = 10 ms

    The electrolytic capacitor is POLARISED. Positive leg to NODE, negative
    (the striped side) to GND. Backwards it will heat and can vent.

TWO THINGS THAT WILL RUIN THE MEASUREMENT IF YOU GET THEM WRONG
---------------------------------------------------------------
1.  Idle synapse pins are set to INPUT, not driven low. A pin driven low
    puts its 2k resistor in parallel with the 1k bleeder, so three pins
    idling low would give R_eff = 1k||2k||2k = 500R and tau = 5 ms, not
    10 ms -- a 2x error that looks exactly like a bad capacitor. High-Z is
    what the simulation assumes, so high-Z is what the hardware must do.

2.  The Pico drives 3.3 V, not the 2.0 V in the paper. Nothing is wrong
    with that -- the circuit is linear, so amplitude just scales -- but the
    simulation must be re-run at 3.3 V before the traces are compared.
    overlay.py does this automatically.

Sample timing is recorded, never assumed: MicroPython's loop jitter is real,
so every sample carries its own microsecond timestamp and the host uses
those rather than a nominal rate.
"""

from machine import Pin, ADC
import time
import sys

# ── hardware map ────────────────────────────────────────────────────────────
SYNAPSE_PINS = (2, 3, 4)      # each through its own 2k weight resistor
ADC_PIN = 26                  # ADC0, reads the capacitor node
ADC_FULL_SCALE = 3.3          # Pico ADC reference (volts)
ADC_MAX = 65535               # read_u16() range

adc = ADC(Pin(ADC_PIN))


def hi_z(gp):
    """Release a synapse: no current path at all. See note 1 in the header."""
    Pin(gp, Pin.IN)


def drive(gp, level):
    p = Pin(gp, Pin.OUT)
    p.value(level)
    return p


def release_all():
    for gp in SYNAPSE_PINS:
        hi_z(gp)


def settle(ms=200):
    """Let the capacitor fully discharge through the bleeder before a trial.
    200 ms is 20 tau -- residual charge below one ADC count."""
    release_all()
    time.sleep_ms(ms)


def run_trial(events, duration_ms=120, label=''):
    """Drive `events` and sample the node throughout.

    events: list of (t_ms, gpio, level). Applied in order, by wall clock from
            trial start. level 1 drives the pin high; level 0 releases it to
            high-Z (NOT driven low -- see header).

    Emits one CSV row per sample: label,t_us,volts
    """
    settle()
    plan = sorted(events, key=lambda e: e[0])
    idx = 0
    t0 = time.ticks_us()
    deadline = duration_ms * 1000

    while True:
        now = time.ticks_diff(time.ticks_us(), t0)
        if now >= deadline:
            break

        # apply any events whose time has arrived
        while idx < len(plan) and now >= plan[idx][0] * 1000:
            _, gp, level = plan[idx]
            if level:
                drive(gp, 1)
            else:
                hi_z(gp)
            idx += 1

        raw = adc.read_u16()
        volts = raw * ADC_FULL_SCALE / ADC_MAX
        print('%s,%d,%.5f' % (label, now, volts))

    release_all()


# ── the three tests ─────────────────────────────────────────────────────────

def test_tau():
    """Single 5 ms pulse on one synapse. The decay that follows is the
    measurement: time from peak to peak/e is tau_D."""
    a = SYNAPSE_PINS[0]
    run_trial([(10, a, 1), (15, a, 0)], duration_ms=120, label='tau')


def test_spatial():
    """1, 2, then 3 synapses pulsed together. Peak voltage should grow
    sub-linearly -- each added synapse also loads the node."""
    for n in (1, 2, 3):
        events = []
        for gp in SYNAPSE_PINS[:n]:
            events += [(10, gp, 1), (15, gp, 0)]
        run_trial(events, duration_ms=90, label='spatial%d' % n)


def test_temporal():
    """Two pulses on one synapse. At 8 ms apart the second lands inside the
    10 ms window and stacks on residual charge; at 45 ms it does not."""
    a = SYNAPSE_PINS[0]
    for gap in (8, 45):
        run_trial([(10, a, 1), (15, a, 0),
                   (15 + gap, a, 1), (20 + gap, a, 0)],
                  duration_ms=140, label='temporal%d' % gap)


def main():
    release_all()
    # Header lets the host detect the start of a run and ignore REPL noise.
    print('# SPECTRUM D-compartment bench')
    print('# columns: test,t_us,volts')
    print('# drive_volts=%.2f' % ADC_FULL_SCALE)
    print('BEGIN')
    test_tau()
    test_spatial()
    test_temporal()
    print('END')


if __name__ == '__main__':
    main()
