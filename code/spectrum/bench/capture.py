"""
capture.py -- read the bench run off the Pico and save it.
==========================================================

    python3 capture.py                  # auto-detect the port
    python3 capture.py --port /dev/cu.usbmodem1101
    python3 capture.py --list           # show candidate ports and exit

Writes measured.csv (test,t_us,volts). Needs pyserial:  pip3 install pyserial
"""

import argparse
import sys
import time

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit('pyserial missing.  pip3 install pyserial')

BAUD = 115200          # ignored over USB CDC, but pyserial wants a number
DEFAULT_OUT = 'measured.csv'


def candidates():
    """Ports that could plausibly be a Pico, best guess first."""
    ports = list(list_ports.comports())
    def score(p):
        blob = f'{p.description} {p.manufacturer} {p.product}'.lower()
        if p.vid == 0x2E8A:                       # Raspberry Pi vendor ID
            return 0
        if 'pico' in blob or 'micropython' in blob:
            return 1
        if 'usbmodem' in p.device or 'ttyACM' in p.device:
            return 2
        return 9
    return sorted(ports, key=score)


def pick_port(explicit):
    if explicit:
        return explicit
    found = candidates()
    if not found:
        sys.exit('No serial ports found. Is the Pico plugged in and running '
                 'MicroPython (not in BOOTSEL mass-storage mode)?')
    best = found[0]
    print(f'using {best.device}  ({best.description})')
    return best.device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--timeout', type=float, default=60.0,
                    help='seconds to wait for END before giving up')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    if args.list:
        for p in candidates():
            print(f'{p.device:24} vid={p.vid} {p.description}')
        return

    port = pick_port(args.port)
    rows, started, t0 = [], False, time.time()

    with serial.Serial(port, BAUD, timeout=1.0) as ser:
        ser.reset_input_buffer()
        print('waiting for BEGIN … (reset the Pico if nothing arrives)')
        while time.time() - t0 < args.timeout:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', 'replace').strip()
            if not line:
                continue
            if line.startswith('#'):
                print(line)
                continue
            if line == 'BEGIN':
                started, t0 = True, time.time()
                print('capturing …')
                continue
            if line == 'END':
                break
            if not started:
                continue                      # REPL noise before the run
            parts = line.split(',')
            if len(parts) != 3:
                continue                      # a truncated line, skip it
            rows.append(line)
        else:
            print(f'timed out after {args.timeout}s', file=sys.stderr)

    if not rows:
        sys.exit('No samples captured. Check the Pico is running '
                 'pico_dcompartment.py, not sitting at the REPL.')

    with open(args.out, 'w') as fh:
        fh.write('test,t_us,volts\n')
        fh.write('\n'.join(rows) + '\n')

    tests = sorted({r.split(',')[0] for r in rows})
    print(f'wrote {args.out}: {len(rows)} samples across {len(tests)} tests')
    print('  ' + ', '.join(tests))
    print('\nnext:  python3 overlay.py')


if __name__ == '__main__':
    main()
