#!/usr/bin/env python3
"""
pressure_logger.py — the pressure logger app.

Opens an app window where you enter the folder, mouse ID and details for each recording,
watch the live pressure signal, and see the sessions saved today. The Arduino stays
connected the whole time. A session starts at the first TTL pulse from the Inscopix DAQ
(or with the Start button) and is saved when the pulses stop:
  data/<folder>/<mouse>/<YYYY-MM-DD>/<mouse>_<YYYY-MM-DD>_S01_pressure.csv   every sample
  data/<folder>/<mouse>/<YYYY-MM-DD>/<mouse>_<YYYY-MM-DD>_S01_info.json      summary + details
  data/sessions.csv                                                          every session

Examples
  python3 pressure_logger.py                 # or double-click run_pressure.command
  python3 pressure_logger.py --simulate      # test without hardware (saves to test_data/)
  python3 pressure_logger.py --list-ports

IMPORTANT: close the Arduino IDE Serial Monitor first — only one program can open the port.
"""

import argparse
import os

from pressure_rig.app import run_app
from pressure_rig.sources import list_serial_ports

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="auto", help="serial port (default: find the Arduino)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--data-dir", help="where sessions are saved (default: data/ next to this script)")
    ap.add_argument("--ttl-timeout", type=float, default=2.0,
                    help="save a TTL session after this many seconds without pulses")
    ap.add_argument("--ttl-mode", choices=["pulses", "level"], default="pulses",
                    help="pulses: one TTL pulse per frame (default); level: TTL stays HIGH while recording")
    ap.add_argument("--http-port", type=int, default=8790,
                    help="local port for the app window (the next free one is used if busy)")
    ap.add_argument("--no-window", dest="window", action="store_false",
                    help="don't open the app window (open the printed address yourself)")
    ap.add_argument("--simulate", action="store_true", help="fake data incl. TTL bursts, no Arduino needed")
    ap.add_argument("--list-ports", action="store_true")
    args = ap.parse_args()

    if args.list_ports:
        for p in list_serial_ports():
            print(f"{p.device:30s} {p.description}")
        return
    if args.data_dir is None:
        args.data_dir = os.path.join(HERE, "test_data" if args.simulate else "data")
    run_app(args)


if __name__ == "__main__":
    main()
