"""Reader thread: baseline correction, Schmitt-trigger indicator, and per-session saving.

A session starts at the first TTL pulse (the Inscopix DAQ -> camera trigger, split to the
Arduino) or with the Start button, and is saved as
  <data>/<folder>/<mouse>/<YYYY-MM-DD>/<mouse>_<YYYY-MM-DD>_S##_pressure.csv   every sample
  <data>/<folder>/<mouse>/<YYYY-MM-DD>/<mouse>_<YYYY-MM-DD>_S##_info.json      summary + details
  <data>/sessions.csv                                                          one row per session
"""

import csv
import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime

from .sources import parse_line

BASELINE_S = 1.0             # startup / button baseline = mean of this many seconds
MAX_BUFFER_S = 300.0         # history kept for the plot
DEFAULT_HI = 100000.0
DEFAULT_LO = 250.0

COLUMNS = {
    "t_s": "seconds since the first sample of the session",
    "arduino_ms": "Arduino clock at the sample (ms since reset)",
    "pc_time": "computer clock when the sample arrived",
    "raw": "NAU7802 raw count (channel 1, gain 128, 80 samples/s)",
    "baseline": "baseline subtracted from raw",
    "corrected": "raw - baseline",
    "indicator": "Schmitt trigger: 1 from corrected >= upper until corrected <= lower",
    "ttl": "TTL input level at the sample",
    "ttl_count": "TTL pulses since the session started (= camera frame number, 1 pulse/frame)",
}
INDEX_COLUMNS = ["folder", "mouse", "date", "session", "trigger", "start_time", "end_time",
                 "duration_s", "n_samples", "rate_hz", "ttl_pulses", "details", "file"]


def safe_name(text, empty="unknown"):
    """Name usable in a file path: letters, digits, - and _ only."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text).strip()).strip("_") or empty


def blank(x):
    return "" if x is None else x


def today():
    return datetime.now().strftime("%Y-%m-%d")


def session_dir(data_dir, folder, mouse, date):
    return os.path.join(data_dir, *([folder] if folder else []), mouse, date)


def next_session_number(path, mouse, date):
    if not os.path.isdir(path):
        return 1
    pat = re.compile(rf"{re.escape(mouse)}_{date}_S(\d+)_")
    used = [int(m.group(1)) for f in os.listdir(path) if (m := pat.match(f))]
    return max(used, default=0) + 1


def todays_sessions(data_dir):
    """Rows of sessions.csv recorded today, oldest first."""
    try:
        with open(os.path.join(data_dir, "sessions.csv"), newline="") as f:
            return [r for r in csv.DictReader(f) if r.get("date") == today()]
    except OSError:
        return []


class State:
    """Shared by the reader thread and the app; guard with .lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = deque()          # (seq, t, raw, corrected, indicator)
        self.seq = 0
        self.baseline = 0.0
        self.calibrated = False
        self.hi, self.lo = DEFAULT_HI, DEFAULT_LO
        self.ind = 0
        self.rate_hz = 0.0
        self.folder, self.mouse, self.details = "", "", ""
        self.auto_ttl = True         # start sessions at the first TTL pulse
        self.request = None          # "start" / "stop" from the Record button
        self.status, self.port, self.conn_id = "disconnected", None, 0
        self.has_ttl = False         # the Arduino sketch reports ttl / pulses
        self.pulses = 0              # TTL pulses since the Arduino reset
        self.rec = None              # while recording: label, file, trigger, elapsed, pulses
        self.saved = []              # summaries of the sessions saved today, oldest first

    def new_connection(self, port):
        with self.lock:
            self.data.clear()
            self.calibrated, self.ind, self.rate_hz = False, 0, 0.0
            self.has_ttl, self.pulses, self.request = False, 0, None
            self.status, self.port = "connected", port
            self.conn_id += 1


class Session:
    """One recording: CSV of every sample + info JSON + a row in sessions.csv."""

    def __init__(self, data_dir, folder, mouse, trigger, pulses0, info):
        start = datetime.now()
        date = start.strftime("%Y-%m-%d")
        path = session_dir(data_dir, folder, mouse, date)
        os.makedirs(path, exist_ok=True)
        self.number = next_session_number(path, mouse, date)
        stem = os.path.join(path, f"{mouse}_{date}_S{self.number:02d}")
        self.csv_path, self.info_path = stem + "_pressure.csv", stem + "_info.json"
        self.data_dir, self.trigger = data_dir, trigger
        self.label = f"{mouse} S{self.number:02d}"
        self.info = dict(folder=folder, mouse=mouse, date=date, session=self.number,
                         trigger=trigger, start_time=start.isoformat(timespec="milliseconds"), **info)
        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLUMNS)
        self.n = 0
        self.t0 = self.t_last = None
        self.pulses0 = self.pulses_last = pulses0

    def write(self, t, s, base, corrected, ind):
        if self.t0 is None:
            self.t0 = t
        self.t_last = t
        if s.pulses is not None:
            if self.pulses0 is None:
                self.pulses0 = s.pulses
            self.pulses_last = s.pulses
        raw = int(s.value) if s.value.is_integer() else s.value
        count = self.pulse_count() if s.pulses is not None else None
        self.w.writerow([f"{t - self.t0:.4f}", blank(s.ms),
                         datetime.now().isoformat(timespec="milliseconds"), raw,
                         round(base, 2), round(corrected, 2), ind, blank(s.ttl), blank(count)])
        self.n += 1
        if self.n % 40 == 0:
            self.f.flush()

    def elapsed(self):
        return 0.0 if self.t0 is None else self.t_last - self.t0

    def pulse_count(self):
        if self.pulses0 is None or self.pulses_last is None:
            return None
        return self.pulses_last - self.pulses0

    def close(self, details=""):
        """Finish the files. Returns the sessions.csv row, or None if nothing was recorded."""
        self.f.close()
        if self.n == 0:
            os.remove(self.csv_path)
            return None
        dur = self.elapsed()
        info = {k: self.info[k] for k in ("folder", "mouse", "date", "session", "trigger", "start_time")}
        info.update(end_time=datetime.now().isoformat(timespec="milliseconds"),
                    duration_s=round(dur, 3), n_samples=self.n,
                    rate_hz=round((self.n - 1) / dur, 2) if dur > 0 else None,
                    ttl_pulses=self.pulse_count(), details=details,
                    file=os.path.relpath(self.csv_path, self.data_dir))
        info.update({k: v for k, v in self.info.items() if k not in info})
        info["columns"] = COLUMNS
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=2)
        row = {k: info.get(k) for k in INDEX_COLUMNS}
        self._add_to_index(row)
        return row

    def _add_to_index(self, row):
        path = os.path.join(self.data_dir, "sessions.csv")
        try:
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:          # e.g. sessions.csv open in Excel on Windows
            print(f"  (could not update {path}: {e})")


def reader(lines, cfg, stop, st):
    """Consume Arduino lines until `stop` is set, saving sessions as they happen.
    cfg: data_dir, port, ttl_mode ("pulses" / "level"), ttl_timeout (s)."""
    session = None
    rearm = True             # after a manual Stop, wait until the TTL goes quiet
    ttl_active_t = None      # time of the last TTL pulse (or HIGH level in level mode)
    prev_pulses = None
    warm, t_first = [], None
    times = deque()
    pc0 = time.perf_counter()
    err = None

    def start(trigger, pulses0):
        with st.lock:
            folder, mouse = st.folder, st.mouse or "unknown"
            info = dict(baseline_at_start=round(st.baseline, 2), schmitt_lower=st.lo,
                        schmitt_upper=st.hi, ttl_input=st.has_ttl)
        info.update(port=cfg.port, ttl_mode=cfg.ttl_mode, ttl_timeout_s=cfg.ttl_timeout)
        s = Session(cfg.data_dir, folder, mouse, trigger, pulses0, info)
        rel = os.path.relpath(s.csv_path, cfg.data_dir)
        with st.lock:
            st.rec = dict(label=s.label, file=rel, trigger=trigger, elapsed=0.0, pulses=0)
        print(f"[REC]   {s.label} started ({trigger}) -> {rel}")
        return s

    def finish(s):
        with st.lock:
            details = st.details      # saved at the end, so notes typed while recording count
        row = s.close(details)
        with st.lock:
            st.rec = None
            if row:
                st.saved.append(row)
        if row is None:
            print(f"[EMPTY] {s.label}: no samples, nothing saved")
            return
        pulses = "" if row["ttl_pulses"] is None else f", {row['ttl_pulses']} TTL pulses"
        print(f"[SAVED] {s.label}: {row['duration_s']:.1f} s, {row['n_samples']} samples{pulses}")

    try:
        for line in lines:
            if stop.is_set():
                break
            with st.lock:
                req, st.request = st.request, None
            if req == "start" and session is None:
                session = start("manual", prev_pulses)
            elif req == "stop" and session is not None:
                finish(session)
                session, rearm = None, False

            s = parse_line(line) if line else None
            if s is None:
                if line.startswith("NAU7802"):    # wiring / startup line from the sketch
                    print(f"[arduino] {line}")
                continue
            t = s.ms / 1000 if s.ms is not None else time.perf_counter() - pc0
            if t_first is None:
                t_first = t
                if s.pulses is None:
                    print("Note: this Arduino sketch doesn't report TTL -> sessions start only "
                          "with the Start button (upload arduino/pressure_ttl to fix).")
            times.append(t)
            while t - times[0] > 2.0:                     # rate over the last ~2 s
                times.popleft()

            rows = []
            with st.lock:
                st.rate_hz = (len(times) - 1) / (t - times[0]) if t > times[0] else 0.0
                st.has_ttl = s.pulses is not None
                st.pulses = s.pulses or 0
                auto = st.auto_ttl
                if st.calibrated:
                    batch = [(t, s)]
                else:                                     # startup baseline: first BASELINE_S
                    warm.append((t, s))
                    batch = []
                    if t - t_first >= BASELINE_S:
                        st.baseline = sum(w[1].value for w in warm) / len(warm)
                        st.calibrated = True
                        batch, warm = warm, []
                for bt, bs in batch:
                    v = bs.value - st.baseline
                    if st.ind == 0 and v >= st.hi:        # Schmitt trigger
                        st.ind = 1
                    elif st.ind == 1 and v <= st.lo:
                        st.ind = 0
                    st.seq += 1
                    st.data.append((st.seq, bt, bs.value, v, st.ind))
                    rows.append((bt, bs, st.baseline, v, st.ind))
                while st.data and t - st.data[0][1] > MAX_BUFFER_S:
                    st.data.popleft()

            if s.pulses is not None:                      # TTL-triggered sessions
                new_pulse = prev_pulses is not None and s.pulses > prev_pulses
                if new_pulse or (cfg.ttl_mode == "level" and s.ttl == 1):
                    ttl_active_t = t
                quiet = ttl_active_t is None or t - ttl_active_t > cfg.ttl_timeout
                rearm = rearm or quiet
                if session is None and new_pulse and rearm and auto and rows:
                    session = start("ttl", prev_pulses)
                elif session is not None and session.trigger == "ttl" and quiet:
                    finish(session)
                    session = None
                prev_pulses = s.pulses

            if session is not None:
                for row in rows:
                    session.write(*row)
                with st.lock:
                    if st.rec:
                        st.rec["elapsed"] = round(session.elapsed(), 2)
                        st.rec["pulses"] = session.pulse_count() or 0
    except Exception as e:                                # e.g. USB cable unplugged
        err = f"{type(e).__name__}: {e}"
        print(f"\nStopped reading: {err}")
    finally:
        if session is not None:
            finish(session)
        lines.close()
        with st.lock:
            st.status = f"disconnected ({err})" if err else "disconnected"
        stop.set()
