# Pressure logger

An app window for the Arduino and the SparkFun NAU7802 pressure board. You keep it open all day. For each
recording you enter the **folder**, **mouse ID** and **details**, and the Arduino stays
connected between mice.

A recording **starts automatically at the first TTL pulse from the Inscopix DAQ** (the same
TTL that triggers the Imaging Source cameras) and is saved when the pulses stop. Every sample
also stores the TTL pulse count, so each pressure sample can be matched to a camera /
Inscopix frame.

## Folder layout

```
pressure/
├── run_pressure.command        double-click to start the app (macOS)
├── pressure_logger.py          same thing from the terminal
├── pressure_rig/               code: app.py (app + window server), recorder.py, sources.py, web/
├── arduino/pressure_ttl/       Arduino sketch: NAU7802 + TTL input
├── data/                       recordings (created automatically)
│   ├── sessions.csv            one row per session: folder, mouse, date, details, ...
│   ├── pain_cohort1/           the "Folder" you type in the app
│   │   └── M01/2026-10-02/     one folder per mouse per day
│   │       ├── M01_2026-10-02_S01_pressure.csv
│   │       └── M01_2026-10-02_S01_info.json
│   └── _old_pressure_logs/     logs from the earlier version
├── requirements.txt
└── README.md
```

## One-time setup

1. **Wire the TTL.** The DAQ trigger already goes through a 2-way BNC splitter to the two
   cameras. Add a third branch for the Arduino and leave the camera wiring as it is:
   ```
   Inscopix DAQ ──BNC── [BNC T adapter] ── existing 2-way splitter ──┬── camera 1
                              │                                      └── camera 2
                              └── [BNC-to-screw-terminal adapter]
                                     + (BNC centre) ──── Mega pin 18 (labelled "TX1 18")
                                     − (BNC shield) ──── Mega GND (any GND pin)
   ```
   Parts:
   - a BNC T adapter (1 male → 2 female), or a 1-to-3 BNC splitter
   - a BNC (male) to 2-pin screw-terminal adapter, like the green ones on the camera cables
   - two jumper wires

   If the terminals aren't marked, the + terminal is the one connected to the BNC centre pin
   (check with a multimeter). The signal must be 0–5 V; TTL is normally 3.3–5 V.
2. **Upload the sketch.** The red board is a SparkFun NAU7802 (Qwiic Scale), on I2C.
   In Library Manager install **SparkFun Qwiic Scale NAU7802 Arduino Library**, then open
   `arduino/pressure_ttl/pressure_ttl.ino`. Select *Arduino Mega 2560* and click Upload.
   Wire the header like this (Mega I2C pins are fixed):
   ```
   NAU7802 VIN -> Mega 5V     (VIN, not the 3V3 pin)
   NAU7802 GND -> Mega GND
   NAU7802 SDA -> Mega pin 20
   NAU7802 SCL -> Mega pin 21
   screw terminal -> syringe bridge (E+, E-, A+, A-; match the silkscreen)
   ```
   Leave INT and the Qwiic connector unused. Gain is 128 and the rate is 80 samples/s;
   both are constants at the top of the sketch. This replaces the sketch now on the board.
   To keep a copy of the current firmware first
   (takes about 1 minute; close the Serial Monitor and the app first):
   ```bash
   AVR=~/Library/Arduino15/packages/arduino/tools/avrdude/6.3.0-arduino17
   $AVR/bin/avrdude -C $AVR/etc/avrdude.conf -p atmega2560 -c wiring -P /dev/cu.usbmodem21101 -b 115200 -U flash:r:arduino/old_firmware_backup.hex:i
   ```
   To restore it later, run the same command with `-D -U flash:w:arduino/old_firmware_backup.hex:i`.
3. **Python.** The only package needed is pyserial, which is already installed on this Mac.
   On another computer, run `python3 -m pip install -r requirements.txt`.

## Using the app

1. Plug in the Arduino. Close the Arduino IDE Serial Monitor.
2. Double-click **`run_pressure.command`**. A Terminal window opens (leave it open) and the
   app opens in its own window. It connects to the Arduino by itself, and the baseline is set
   during the first second, so don't touch the sensor then.
3. Fill in **Folder** (e.g. `pain_cohort1`), **Mouse ID** (e.g. `M01`) and **Details**
   (condition, drug, notes…). **Next file** shows exactly where the recording will be saved.
4. Start IC Capture and the Inscopix recording as usual. At the first TTL pulse the
   Recording panel turns red (`● REC M01 S01`). When Inscopix stops, the session is saved
   2 s after the last pulse and appears under **Sessions saved today**.
5. Next mouse: change Mouse ID and Details, and record again. The session number counts up
   per mouse per day (S01, S02, …).
6. At the end of the day click **Quit app**.

More controls:

- **Details** are saved when the session ends, so you can add notes during a recording.
  Folder and Mouse ID are locked while recording.
- **Start / Stop recording** records manually, for example without TTL. Untick
  *Start automatically at the first TTL pulse* to use only the button.
- **Set baseline** re-zeros the signal.
- **Window**, **Lower** and **Upper** (the Schmitt-trigger thresholds) take effect on Enter.
- **Open folder** shows the mouse's folder in Finder.
- The app remembers Folder, Mouse ID, thresholds and the TTL setting for next time.
- If the USB cable comes out, the current session is saved. Plug it back in and press **Connect**.

## Saved data

`*_pressure.csv` — one row per sample:

| column | meaning |
|---|---|
| `t_s` | seconds since the session started (0 = first sample after the first TTL pulse) |
| `arduino_ms` | Arduino clock (ms) |
| `pc_time` | computer clock |
| `raw` | NAU7802 raw count (channel 1) |
| `baseline`, `corrected` | baseline used, and `raw - baseline` |
| `indicator` | Schmitt trigger output (1 from `corrected ≥ Upper` until `corrected ≤ Lower`) |
| `ttl` | TTL level at that sample |
| `ttl_count` | TTL pulses since the session started = number of the camera frame in progress |

`*_info.json` holds the folder, mouse, details, start and end times, duration, sample rate,
number of TTL pulses, baseline and thresholds. `data/sessions.csv` lists every session.

## Options

```bash
python3 pressure_logger.py --simulate         # test without hardware (saves to test_data/)
python3 pressure_logger.py --list-ports
python3 pressure_logger.py --help             # all options
```

Other useful options:

- `--port /dev/cu.usbmodemXXXX` — the Arduino is found automatically; use this to pick a port.
- `--ttl-timeout 2` — how many seconds without pulses before a session is saved.
- `--ttl-mode level` — use this if the TTL stays HIGH for the whole recording instead of one
  pulse per frame.
- `--no-window` — don't open the window; open the address printed in the Terminal yourself.

## Troubleshooting

- **The window didn't open** — open the address printed in the Terminal (e.g.
  `http://127.0.0.1:8790/`) in a browser. Double-clicking `run_pressure.command` again just
  reopens the window of the running app.
- **"Not connected … Resource busy"** — the Serial Monitor or another logger is using the port.
  Close it and press **Connect**.
- **"TTL not reported by the Arduino sketch"** — the board still runs the old sketch. Upload
  `arduino/pressure_ttl` (setup step 2).
- **TTL pulses stay at 0 while Inscopix records** — check the BNC centre → pin 18, the shield
  → GND, and that the T adapter is on the camera trigger cable.
- **A session never ends** — the TTL may stay HIGH. Use `--ttl-mode level`, or press Stop.
- **Extra short sessions** — the DAQ may also send TTL during live preview. Delete those, or
  untick the TTL checkbox and use the button.
