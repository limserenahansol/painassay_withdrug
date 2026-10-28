"""Where samples come from: the Arduino over USB serial, or a simulated stream."""

import random
import re
import time
from collections import namedtuple

READING_RE = re.compile(r"Reading:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
FIELD_RE = re.compile(r"\b(ttl|pulses|ms)=(-?\d+)")

# ttl / pulses / ms are None with an older sketch that only prints "Reading: <n>"
Sample = namedtuple("Sample", "value ttl pulses ms")


def parse_line(line):
    """'Reading: -29550 ttl=0 pulses=12 ms=34567' -> Sample, or None for any other line."""
    m = READING_RE.search(line)
    if not m:
        return None
    f = {k: int(v) for k, v in FIELD_RE.findall(line)}
    return Sample(float(m.group(1)), f.get("ttl"), f.get("pulses"), f.get("ms"))


def list_serial_ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


def find_arduino_port():
    """First port that looks like an Arduino, or None."""
    keys = ("arduino", "usbmodem", "usbserial", "wchusbserial", "ch340")
    for p in list_serial_ports():
        text = f"{p.device} {p.description} {p.manufacturer or ''}".lower()
        if any(k in text for k in keys):
            return p.device
    return None


def open_serial(port, baud):
    """Open the port now (errors raise RuntimeError) and return a line iterator.
    It yields "" every 0.5 s when nothing arrives, so the reader can still react to Stop.

    Opening the port resets the Mega. On macOS that first open sometimes returns
    only the boot line and then no samples; closing and opening again gets the
    stream. reset_input_buffer() is tcflush and stops this board's USB serial,
    so the boot chatter is discarded by reading it instead.
    """
    import serial

    def _open():
        try:
            return serial.Serial(port, baud, timeout=0.5)
        except serial.SerialException as e:
            raise RuntimeError(f"could not open {port} ({e}) - is the Arduino IDE Serial Monitor "
                               "or another logger still open?") from None

    ser = None
    prelude = []
    for _ in range(4):
        ser = _open()
        prelude = []
        deadline = time.time() + 2.5
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            prelude.append(raw)
            if raw.startswith(b"Reading:"):
                break
        else:
            ser.close()
            ser = None
            time.sleep(0.4)
            continue
        break
    if ser is None:
        raise RuntimeError(f"no samples from {port} - check the NAU7802 wiring (VIN, GND, SDA 20, SCL 21)")

    def lines():
        try:
            for raw in prelude:
                yield raw.decode("utf-8", errors="ignore").strip()
            while True:
                yield ser.readline().decode("utf-8", errors="ignore").strip()
        finally:
            ser.close()
    return lines()


def simulated_lines(rate_hz=80, frame_hz=20, ttl_start_s=4.0, ttl_on_s=12.0, ttl_off_s=8.0):
    """Fake stream in the pressure_ttl.ino format: 24-bit ADC noise with downward presses,
    plus a frame_hz TTL pulse train that runs ttl_on_s out of every ttl_on_s + ttl_off_s."""
    t0 = time.time()
    press_until, next_press, depth = 0.0, t0 + 2, 0.0
    per_burst = int(ttl_on_s * frame_hz)
    while True:
        now = time.time()
        el = now - t0
        if now >= next_press and now >= press_until:
            press_until = now + random.uniform(0.3, 2.0)
            depth = random.uniform(1000, 5000)
            next_press = press_until + random.uniform(1, 4)
        v = -29550 + random.gauss(0, 60) - (depth if now < press_until else 0)

        pulses, ttl = 0, 0
        if el >= ttl_start_s:
            cycles, phase = divmod(el - ttl_start_s, ttl_on_s + ttl_off_s)
            pulses = int(cycles) * per_burst
            if phase < ttl_on_s:
                pulses += min(int(phase * frame_hz) + 1, per_burst)
                ttl = int((phase * frame_hz) % 1 < 0.2)        # 20 % duty cycle
            else:
                pulses += per_burst
        yield f"Reading: {int(v)} ttl={ttl} pulses={pulses} ms={int(el * 1000)}"
        time.sleep(1 / rate_hz)
