"""Per-recording count check: do the Inscopix frames, the SYNC pulses counted by the pressure
logger, the two camera videos and (when exported) the DAQ GPIO SYNC edges all agree?

The rules (CONTRACTS.md §1a, revised 2026-10-19):
    SYNC pulse n (1-based) ends isx_frame n-1+K and triggers cam1 frame n and cam2 frame n;
    the logger's ttl_count becomes n. Every device counts the same pulses, so a clean
    recording has  n_ttl == n_cam1 == n_cam2 (== GPIO edges) == n_frames_isx + K,
and count_check FAILs (never repairs) any other count. The expected count comes from
master_timeline.expected_n_ttl, the one place the count rule lives. Caveat: the index rule
alone would give n_frames_isx - K; the two agree only for K = 0 (the 2024 data: first SYNC
edge 0.0-0.6 ms after the movie start). The K hint and k_from_gpio below say so when K != 0.

Counts:
  n_frames_isx, n_dropped   isx_meta.read_meta(isxd)  (footer only, includes dropped frames)
  n_ttl                     last ttl_count of the logger *_pressure.csv
  n_cam1/2_frames           OpenCV CAP_PROP_FRAME_COUNT; with --exact every frame is decoded,
                            because container counts can be wrong
  n_gpio_sync_edges         rising edges of 'BNC Sync Output' in an IDPS GPIO CSV export
                            (columns 'Time (s), Channel Name, Value')

Writes / updates recordings.csv (one row per recording_id; other modules' columns and rows are
kept). count_check = PASS or FAIL; the reasons and warnings go into `notes` as one segment that
starts with 'count_check:' (other modules' segments, separated by ' | ', are kept).

Use:
  python count_check.py --manifest manifest.csv [--recording-id M01_d1_pre ...] [--exact]
  python count_check.py --recording-id TEST --isxd X.isxd --pressure-csv P.csv
                        --cam1 A.avi --cam2 B.avi [--gpio X.gpio] [--gpio-csv G.csv]
  python count_check.py --selftest
Options: --pulse-offset K, --cam-lead-frames, --recordings PATH (default out/recordings.csv in
this folder, never next to the manifest, which sits with the data), --sync-channel NAME.
Each FAIL reason carries a code (count_check.Fail.code, e.g. cam2_missing, gpio_tail) so that
run_pipeline can tell which reason an explicit option covers.
Exit code: 0 all PASS, 1 any FAIL, 2 bad input (missing manifest columns, a K that is not an
integer, etc.).
"""

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

import isx_meta
import master_timeline as mt

HERE = os.path.dirname(os.path.abspath(__file__))

RECORDINGS_COLUMNS = [
    "recording_id", "n_frames_isx", "n_dropped", "n_trimmed_early", "period_s", "fps", "n_ttl",
    "n_gpio_sync_edges", "n_cam1_frames", "n_cam2_frames", "pulse_to_frame_offset",
    "cam_lead_frames", "arduino_fit_a", "arduino_fit_b", "arduino_fit_ppm",
    "arduino_fit_resid_ms", "count_check", "notes"]
# Columns this module adds (CONTRACTS.md: a module may add columns, never rename one).
ADDED_COLUMNS = ["n_cropped", "period_source", "ttl_trigger", "cam_count_method",
                 "gpio_first_edge_lag_ms", "k_from_gpio"]
# Columns this module (re)writes; everything else in recordings.csv belongs to other modules.
OWNED_COLUMNS = ["n_frames_isx", "n_dropped", "period_s", "fps", "n_ttl", "n_gpio_sync_edges",
                 "n_cam1_frames", "n_cam2_frames", "pulse_to_frame_offset", "cam_lead_frames",
                 "count_check"] + ADDED_COLUMNS
MANIFEST_REQUIRED = ["recording_id", "isxd", "gpio", "pressure_csv", "pressure_json",
                     "cam1_video", "cam2_video"]
NOTES_SEP = " | "
DEFAULT_SYNC_CHANNEL = isx_meta.SYNC_CHANNEL
DEFAULT_CAM_LEAD = 1          # documentation only: superseded by count_to_isx_frame (§1a), kept = 1
DEFAULT_RECORDINGS = os.path.join(HERE, "out", "recordings.csv")


class InputError(Exception):
    """Bad input that is not a per-recording FAIL (e.g. a manifest without required columns)."""


class Fail(str):
    """A FAIL reason: the text, plus a short code (e.g. 'cam2_missing', 'gpio_tail') so that
    run_pipeline can tell which reasons an explicit option covers. It is a str everywhere else."""

    def __new__(cls, code, text):
        s = super().__new__(cls, text)
        s.code = code
        return s


def parse_k(text, where, strict=True):
    """K (pulse_to_frame_offset) as an int, or None if empty. A K that is not a whole number is
    an input error, never truncated. strict (the manifest): an integer literal only, as
    manifest.py demands; otherwise (recordings.csv, which pandas may have written as '0.0')
    any whole number."""
    s = "" if text is None else str(text).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = int(s) if strict else float(s)
    except ValueError:
        v = np.nan
    if not np.isfinite(v) or v != int(v):
        raise InputError(f"{where}: pulse_to_frame_offset '{s}' must be an integer")
    return int(v)


# ----------------------------------------------------------------------------- readers

def logger_counts(csv_path):
    """Pulse statistics of one pressure-logger session CSV."""
    df = pd.read_csv(csv_path, usecols=lambda c: c in ("t_s", "ttl_count"))
    if "ttl_count" not in df.columns or "t_s" not in df.columns:
        raise ValueError("no t_s / ttl_count columns (not a pressure_logger CSV?)")
    count = pd.to_numeric(df["ttl_count"], errors="coerce")
    ok = count.notna().to_numpy()
    if not ok.any():
        raise ValueError("ttl_count is empty in every row (Arduino sketch without TTL input?)")
    c = count[ok].to_numpy().astype(np.int64)
    t = pd.to_numeric(df["t_s"], errors="coerce")[ok].to_numpy(float)
    step = np.diff(c)
    rows_with_pulse = np.nonzero(step > 0)[0] + 1
    pulse_t = t[rows_with_pulse]
    gaps = np.diff(pulse_t)
    sample_dt = float(np.median(np.diff(t))) if len(t) > 1 else None
    return {
        "n_ttl": int(c[-1]),
        "first_count": int(c[0]),
        "max_count": int(c.max()),
        "n_rows": int(len(c)),
        "n_decreasing": int((step < 0).sum()),
        "n_multi": int((step >= 2).sum()),
        "sample_dt_s": sample_dt,
        # Mean, not median: increments land on 12-13 ms sample boundaries, so at 30 Hz the
        # intervals are 25 or 37 ms and their median (37) is not the pulse period (33).
        "pulse_dt_s": float((pulse_t[-1] - pulse_t[0]) / (c[rows_with_pulse[-1]] - c[rows_with_pulse[0]]))
        if len(rows_with_pulse) > 1 else None,
        "gaps_s": gaps,
        "gap_t_s": pulse_t[1:] if len(gaps) else np.array([]),
    }


def logger_info(json_path):
    """The logger's *_info.json (trigger 'ttl'/'manual', ttl_pulses, ttl_mode), or {}."""
    if not json_path or not os.path.isfile(json_path):
        return {}
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def sibling_sessions(csv_path):
    """{'S01': (ttl_pulses, csv file name), ...} for every logger session in the same folder."""
    out = {}
    for p in sorted(glob.glob(os.path.join(os.path.dirname(csv_path), "*_info.json"))):
        try:
            info = logger_info(p)
        except (OSError, ValueError):
            continue
        session = info.get("session")
        label = f"S{int(session):02d}" if isinstance(session, int) else os.path.basename(p)
        out[label] = (info.get("ttl_pulses"), os.path.basename(info.get("file", p)))
    return out


def video_frames(path, exact):
    """(container_count, decoded_count or None). Decodes when asked or when the container
    gives no usable count."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("OpenCV cannot open it")
    try:
        container = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded = None
        if exact or container <= 0:
            decoded = 0
            while cap.grab():          # grab() skips the colour conversion: faster than read()
                decoded += 1
        return container, decoded
    finally:
        cap.release()


def gpio_sync_edges(csv_path, channel=DEFAULT_SYNC_CHANNEL):
    """Rising edges of the SYNC channel in an IDPS GPIO CSV export.

    IDPS stores change events, so each pulse is a (t, 1) row and a (t + 10 ms, 0) row; the
    last row of the file (any channel) is the end of the GPIO recording.
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    need = {"Time (s)", "Channel Name", "Value"}
    if not need <= set(df.columns):
        raise ValueError(f"columns {list(df.columns)} are not an IDPS GPIO export {sorted(need)}")
    names = df["Channel Name"].astype(str).str.strip()
    sync = df[names == channel]
    if sync.empty:
        raise ValueError(f"no '{channel}' channel (channels: {sorted(names.unique())[:8]} ...)")
    sync = sync.sort_values("Time (s)", kind="stable")
    t = sync["Time (s)"].to_numpy(float)
    v = sync["Value"].to_numpy(float)
    lo, hi = float(v.min()), float(v.max())
    high = v > ((lo + hi) / 2 if hi > lo else 0.5)    # 0/1 in IDPS 1.9.6; volts would work too
    edges = t[np.nonzero(high[1:] & ~high[:-1])[0] + 1]
    n = len(edges)
    return {
        "n_edges": n,
        "first_edge_s": float(edges[0]) if n else None,
        "last_edge_s": float(edges[-1]) if n else None,
        "end_s": float(pd.to_numeric(df["Time (s)"], errors="coerce").max()),
        "mean_period_s": float((edges[-1] - edges[0]) / (n - 1)) if n > 1 else None,
        "starts_high": bool(high[0]),
    }


# ----------------------------------------------------------------------------- paths

def _first_existing(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def find_gpio_files(isxd, gpio, gpio_csv):
    """(gpio_file, gpio_csv): the binary GPIO file (for its start time) and the CSV export.

    The manifest has one `gpio` column; it may hold the raw .gpio, the IDPS *_gpio.isxd or the
    CSV export. Whatever is missing is looked for next to the given files, using IDAS/IDPS
    naming (X_video.isxd, X_video.gpio, X_video_gpio.isxd).
    """
    gpio_file = None
    if gpio and gpio.lower().endswith(".csv"):
        gpio_csv = gpio_csv or gpio
    elif gpio:
        gpio_file = gpio
    stem = isxd[:-len(".isxd")] if isxd and isxd.lower().endswith(".isxd") else None
    if gpio_file is None and stem:
        gpio_file = _first_existing([stem + ".gpio", stem + "_gpio.isxd"])
    if gpio_csv is None and gpio_file:
        gstem = os.path.splitext(gpio_file)[0]
        gpio_csv = _first_existing([gstem + ".csv", gstem + "_gpio.csv"])
    return gpio_file, gpio_csv


def find_logger_json(csv_path):
    """<stem>_info.json next to <stem>_pressure.csv."""
    if csv_path and csv_path.endswith("_pressure.csv"):
        p = csv_path[:-len("_pressure.csv")] + "_info.json"
        return p if os.path.isfile(p) else None
    return None


# ----------------------------------------------------------------------------- the check

def _diff_text(n, ref, unit):
    """('3 fewer frames', -3) / ('1 more pulse', 1); `unit` is given in the plural."""
    d = n - ref
    return f"{abs(d)} {'fewer' if d < 0 else 'more'} {unit if abs(d) != 1 else unit[:-1]}", d


class _Check:
    """Collects the counts (row), the FAIL reasons and the warnings of one recording."""

    def __init__(self, rec, k, cam_lead):
        self.rec = rec
        self.k = k
        self.row = {"recording_id": rec["recording_id"], "pulse_to_frame_offset": k,
                    "cam_lead_frames": cam_lead}
        self.fails, self.warns = [], []
        self.meta = None
        self.n_frames = None
        self.expected = None          # SYNC pulses the index rule predicts for this movie
        self.ref_text = None
        self.gpio = None
        self.k_gpio = None

    def fail(self, code, text):
        self.fails.append(Fail(code, text))


def _check_isx(c):
    path = c.rec.get("isxd")
    if not path:
        c.fail("isxd_missing", "isxd not available")
        return
    if not os.path.isfile(path):
        c.fail("isxd_missing", f"isxd not found: {path}")
        return
    try:
        meta = isx_meta.read_meta(path)
    except (OSError, ValueError) as e:
        c.fail("isxd_unreadable", f"isxd metadata unreadable: {e}")
        return
    if meta.get("kind") != "movie" or meta.get("n_frames") is None:
        c.fail("isxd_unreadable", f"isxd is not a movie (kind {meta.get('kind')})")
        return
    c.meta = meta
    c.n_frames = int(meta["n_frames"])
    c.row.update(n_frames_isx=c.n_frames, n_dropped=meta["n_dropped"], n_cropped=meta["n_cropped"],
                 period_s=meta["period_s"], period_source="isx_footer")
    if meta.get("size_consistent") is False:
        c.fail("isxd_truncated", f"isxd holds only {meta.get('frames_in_file')} of {c.n_frames} frames (truncated copy?)")
    if meta["n_dropped"]:
        c.warns.append(f"IDAS dropped {meta['n_dropped']} frames (they keep their isx_frame numbers)")
    if meta["n_cropped"]:
        c.warns.append(f"footer marks {meta['n_cropped']} cropped frames")
    # CONTRACTS.md count rule (binding): every pulse-driven count must equal n_frames_isx + K.
    c.expected = mt.expected_n_ttl(c.n_frames, c.k)
    c.ref_text = f"{c.expected}" if c.k == 0 else f"{c.expected} = n_frames_isx {c.n_frames} + K {c.k}"


def _check_gpio(c, sync_channel):
    for key in ("gpio", "gpio_csv"):
        if c.rec.get(key) and not os.path.isfile(c.rec[key]):
            c.warns.append(f"{key} not found: {c.rec[key]}")
    gpio_file, gpio_csv = find_gpio_files(c.rec.get("isxd"), c.rec.get("gpio"), c.rec.get("gpio_csv"))
    if gpio_csv and not os.path.isfile(gpio_csv):
        gpio_csv = None
    if not gpio_csv:
        c.warns.append("no GPIO CSV export found - n_gpio_sync_edges not checked "
                       "(import the .gpio in IDPS and export it to CSV)")
        return
    try:
        gp = gpio_sync_edges(gpio_csv, sync_channel)
    except (OSError, ValueError) as e:
        c.fail("gpio_unreadable", f"GPIO export unreadable ({e}): {gpio_csv}")
        return
    c.gpio = gp
    n_gpio = gp["n_edges"]
    c.row["n_gpio_sync_edges"] = n_gpio
    if gp["mean_period_s"]:
        # The DAQ clock measures the real frame period; a raw footer only has the nominal one.
        c.row.update(period_s=gp["mean_period_s"], period_source="gpio_sync_mean")
    if gp["starts_high"]:
        c.warns.append("GPIO SYNC is already high at the first sample (that pulse is not counted)")
    period = c.row.get("period_s")

    # Where does pulse 1 fall? On the 2024 files the first SYNC edge is 0.0-0.6 ms after the
    # movie start time, i.e. pulse 1 = isx_frame 0 (K = 0).
    gpio_t0 = None
    if gpio_file and os.path.isfile(gpio_file):
        try:
            gpio_t0 = isx_meta.read_meta(gpio_file).get("gpio_start_epoch_s")
        except (OSError, ValueError) as e:
            c.warns.append(f"GPIO file unreadable for its start time ({e})")
    movie_t0 = c.meta.get("start_epoch_s") if c.meta else None
    if None not in (movie_t0, gpio_t0, gp["first_edge_s"]) and period:
        lag = gpio_t0 + gp["first_edge_s"] - movie_t0
        c.k_gpio = int(round(lag / period))
        c.row.update(gpio_first_edge_lag_ms=round(lag * 1000, 2), k_from_gpio=c.k_gpio)
        if c.k_gpio != c.k:
            c.warns.append(f"first SYNC edge is {lag * 1000:.1f} ms after the movie start, i.e. "
                           f"pulse 1 ends isx_frame {c.k_gpio} (index-rule K = {c.k_gpio}), but "
                           f"K={c.k} is used")

    if c.expected is None or n_gpio == c.expected:
        return
    what, d = _diff_text(n_gpio, c.expected, "SYNC edges")
    tail = None if gp["last_edge_s"] is None else gp["end_s"] - gp["last_edge_s"]
    start_ok = c.k_gpio is None or c.k_gpio == c.k
    if d == -1 and start_ok and period and tail is not None and tail < period:
        c.fail("gpio_tail", f"GPIO export has 1 fewer SYNC edge than isx frames ({n_gpio} vs {c.ref_text}), but "
                       f"the GPIO file ends {tail * 1000:.0f} ms after its last edge (less than one period): "
                       "the GPIO log stopped before the final frame's pulse - benign (seen in 2 of 4 2024 "
                       "recordings); the logger and camera counts are the ones that matter")
    else:
        why = ("the DAQ saved frames without a SYNC pulse, or the GPIO log started late / stopped early"
               if d < 0 else "the DAQ sent pulses for frames it did not save")
        c.fail("gpio_count", f"GPIO export has {what} than isx frames ({n_gpio} vs {c.ref_text}) - {why}")


def _check_logger(c):
    csv_path = c.rec.get("pressure_csv")
    if not csv_path:
        c.fail("logger_missing", "pressure_csv not available")
        return
    if not os.path.isfile(csv_path):
        c.fail("logger_missing", f"pressure_csv not found: {csv_path}")
        return
    try:
        lg = logger_counts(csv_path)
    except (OSError, ValueError) as e:
        c.fail("logger_unreadable", f"pressure_csv unreadable: {e}")
        return
    n_ttl = lg["n_ttl"]
    c.row["n_ttl"] = n_ttl
    try:
        info = logger_info(c.rec.get("pressure_json") or find_logger_json(csv_path))
    except (OSError, ValueError) as e:
        info = {}
        c.warns.append(f"logger info.json unreadable: {e}")
    trigger = info.get("trigger", "")
    c.row["ttl_trigger"] = trigger
    if trigger == "manual":
        c.warns.append("logger session was started manually: ttl_count counts from the button press, "
                       "not from the first frame, so ttl_count may not equal the frame number")
    elif trigger == "ttl" and lg["first_count"] != 1:
        c.warns.append(f"auto-TTL session but the first row has ttl_count={lg['first_count']} (expected 1)")
    if info.get("ttl_mode") and info["ttl_mode"] != "pulses":
        c.warns.append(f"logger ttl_mode is '{info['ttl_mode']}', not 'pulses'")
    if info.get("ttl_pulses") is not None and info["ttl_pulses"] != n_ttl:
        c.warns.append(f"info.json says {info['ttl_pulses']} pulses but the CSV ends at ttl_count {n_ttl}")
    if lg["n_decreasing"]:
        c.fail("logger_decreasing", f"ttl_count decreases {lg['n_decreasing']} times - not one logger session "
                       "(Arduino reset or files joined?)")
    pulse_dt, sample_dt = lg["pulse_dt_s"], lg["sample_dt_s"]
    if lg["n_multi"] and pulse_dt and sample_dt and pulse_dt > 2 * sample_dt:
        c.warns.append(f"{lg['n_multi']} logger samples gained 2+ pulses at once although pulses are "
                       f"~{pulse_dt * 1000:.0f} ms apart - TTL bounce/noise may inflate ttl_count")
    if pulse_dt and sample_dt:
        # Sampling jitter moves an increment by at most one sample on either side.
        big = np.nonzero(lg["gaps_s"] > max(1.5 * pulse_dt, pulse_dt + 2 * sample_dt))[0]
        if len(big):
            j = big[np.argmax(lg["gaps_s"][big])]
            c.warns.append(f"{len(big)} gaps in the logger pulse train (longest {lg['gaps_s'][j]:.3f} s "
                           f"before t_s={lg['gap_t_s'][j]:.3f}) - pulses missing or the DAQ paused")

    if c.expected is None or n_ttl == c.expected:
        return
    what, d = _diff_text(n_ttl, c.expected, "pulses")
    why = ("logger started late (manual Start after Inscopix, or Inscopix started while the logger was "
           "still taking its baseline) or the TTL branch lost pulses") if d < 0 else \
          ("the session also counted pulses outside this recording (live preview less than 2 s before "
           "Record, or a manual-start session) or the TTL input bounces")
    msg = f"logger has {what} than isx frames ({n_ttl} vs {c.ref_text}) - {why}"
    if c.gpio is not None and c.gpio["n_edges"] == c.expected:
        msg += " (the GPIO export has the expected count, so the DAQ sent the right pulses)"
    if c.meta and c.meta["n_dropped"] and abs(d) == c.meta["n_dropped"]:
        msg += " (the difference equals the number of dropped frames)"
    others = {s: v for s, v in sibling_sessions(csv_path).items() if v[1] != os.path.basename(csv_path)}
    match = [s for s, (n, _) in others.items() if n == c.expected]
    if match:
        msg += (f"; logger session {', '.join(match)} in the same folder has exactly {c.expected} pulses "
                "- wrong pressure_csv in the manifest? (live preview makes extra short sessions)")
    elif others and abs(d) > 0.5 * c.expected:
        msg += "; other sessions in that folder: " + ", ".join(f"{s}={n}" for s, (n, _) in others.items())
    c.fail("logger_count", msg)


def _check_cameras(c, exact):
    # Camera frame n = pulse n, so each video must hold exactly the expected number of pulses.
    ref = c.expected if c.expected is not None else c.row.get("n_ttl")
    ref_label = c.ref_text if c.expected is not None else f"{ref} logger pulses"
    methods = set()
    for cam in ("cam1", "cam2"):
        path = c.rec.get(f"{cam}_video")
        if not path:
            c.fail(f"{cam}_missing", f"{cam} video not available")
            continue
        if not os.path.isfile(path):
            c.fail(f"{cam}_missing", f"{cam} video not found: {path}")
            continue
        try:
            container, decoded = video_frames(path, exact)
        except ValueError as e:
            c.fail(f"{cam}_missing", f"{cam} video unreadable ({e}): {path}")
            continue
        n = decoded if decoded is not None else container
        methods.add("decoded" if decoded is not None else "container")
        if decoded is not None and container > 0 and decoded != container:
            c.warns.append(f"{cam}: container says {container} frames, decoding found {decoded} (used {decoded})")
        c.row[f"n_{cam}_frames"] = n
        if ref is not None and n != ref:
            what, d = _diff_text(n, ref, "frames")
            why = "camera dropped frames or IC Capture started late / stopped early" if d < 0 else \
                  "camera not in external-trigger mode (free-running) or it also recorded preview pulses"
            c.fail(f"{cam}_count", f"{cam} has {what} than pulses ({n} vs {ref_label}) - {why}")
    c.row["cam_count_method"] = "/".join(sorted(methods))


def _hint_k(c):
    """One hint when every pulse-driven count agrees but differs from the isx frame count."""
    counts = [c.row.get(k) for k in ("n_ttl", "n_cam1_frames", "n_cam2_frames")]
    counts = [n for n in counts if n is not None]
    if c.expected is None or len(counts) < 2 or len(set(counts)) != 1 or counts[0] == c.expected:
        return
    p = counts[0]
    k_count = p - c.n_frames          # the K that makes n_ttl == n_frames_isx + K hold
    who = "logger and cameras" if c.row.get("n_ttl") is not None else "both cameras"
    hint = (f"{who} agree on {p} pulses but isx has {c.n_frames} frames: the count rule "
            f"n_ttl = n_frames_isx + K would need K = {k_count:+d}")
    if c.k_gpio is not None:
        hint += f"; the GPIO first edge says pulse 1 ends isx_frame {c.k_gpio} (index rule K = {c.k_gpio})"
        if c.k_gpio == 0:
            hint += " - so K is 0 and the difference is at the end of the recording, not a K offset"
        elif c.k_gpio == -k_count:
            hint += (" - a real K offset, but CONTRACTS.md's count rule (+K) and index rule (n-1+K) "
                     "disagree in sign for K != 0: settle the contract before analysing this rig")
    hint += "; do not change --pulse-offset until the rig test recording confirms it"
    c.fail("k_hint", hint)


def check_recording(rec, k=None, k_source=None, cam_lead=DEFAULT_CAM_LEAD, exact=False,
                    sync_channel=DEFAULT_SYNC_CHANNEL):
    """Count everything for one recording.

    rec: dict with recording_id, isxd, gpio, gpio_csv, pressure_csv, pressure_json,
         cam1_video, cam2_video (absolute paths or None).
    k:   pulse_to_frame_offset K, or None (then 0 is used and a warning is written).
    Returns (row dict for recordings.csv, list of FAIL reasons, list of warnings).
    """
    if k is None:
        c = _Check(rec, 0, cam_lead)
        c.warns.append("K not given (--pulse-offset) and not in recordings.csv - K=0 assumed; "
                       "fix K with the rig test recording")
    else:
        c = _Check(rec, k, cam_lead)
        if k_source:
            c.warns.append(k_source)
    _check_isx(c)
    _check_gpio(c, sync_channel)      # before the logger: its count helps explain a logger mismatch
    _check_logger(c)
    _check_cameras(c, exact)
    _hint_k(c)
    if c.row.get("period_s"):
        c.row["fps"] = 1.0 / c.row["period_s"]
    c.row["count_check"] = "PASS" if not c.fails else "FAIL"
    return c.row, c.fails, c.warns


def notes_segment(row, fails, warns):
    """The 'count_check: ...' part of the notes column."""
    if not fails:
        counts = row.get("n_ttl")
        body = [f"all counts agree ({counts} pulses)"]
    else:
        body = list(fails)
    body += ["WARN " + w for w in warns]
    return "count_check: " + "; ".join(body)


# ----------------------------------------------------------------------------- recordings.csv

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.9g}"
    return str(v)


def merge_notes(old, segment):
    keep = [p for p in str(old or "").split(NOTES_SEP) if p.strip() and not p.startswith("count_check:")]
    return NOTES_SEP.join([segment] + keep)


def read_recordings(path):
    if not os.path.isfile(path):
        return pd.DataFrame(columns=RECORDINGS_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "recording_id" not in df.columns:
        raise InputError(f"{path} has no recording_id column")
    return df


def update_recordings(path, results):
    """Insert or update one row per result; keep every other row and column."""
    df = read_recordings(path)
    for col in RECORDINGS_COLUMNS + ADDED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for row, fails, warns in results:
        rid = row["recording_id"]
        hit = df.index[df["recording_id"] == rid]
        if len(hit) == 0:
            df.loc[len(df)] = {c: "" for c in df.columns}
            hit = [len(df) - 1]
            df.at[hit[0], "recording_id"] = rid
        i = hit[0]
        for col in OWNED_COLUMNS:
            df.at[i, col] = _fmt(row.get(col))
        df.at[i, "notes"] = merge_notes(df.at[i, "notes"], notes_segment(row, fails, warns))
    extra = [c for c in df.columns if c not in RECORDINGS_COLUMNS + ADDED_COLUMNS]
    df = df[RECORDINGS_COLUMNS + ADDED_COLUMNS + extra]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)          # all-or-nothing: a crash never leaves half a table
    except PermissionError:
        os.remove(tmp)
        raise InputError(f"cannot write {path} - is it open in Excel?")


def existing_value(df, rid, col):
    if col not in df.columns:
        return None
    vals = df.loc[df["recording_id"] == rid, col]
    return vals.iloc[0] if len(vals) and str(vals.iloc[0]).strip() != "" else None


# ----------------------------------------------------------------------------- manifest / CLI

def manifest_records(manifest_path, only_ids=None):
    """Recording dicts from manifest.csv, with paths resolved against the manifest's folder."""
    df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = [c for c in MANIFEST_REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"{manifest_path} lacks contract columns {missing}")
    if df["recording_id"].duplicated().any():
        dup = sorted(df.loc[df["recording_id"].duplicated(), "recording_id"].unique())
        raise InputError(f"{manifest_path}: duplicate recording_id {dup}")
    base = os.path.dirname(os.path.abspath(manifest_path))

    def resolve(p):
        p = str(p).strip()
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    records = []
    for _, r in df.iterrows():
        if only_ids and r["recording_id"] not in only_ids:
            continue
        records.append({
            "recording_id": r["recording_id"],
            "isxd": resolve(r["isxd"]),
            "gpio": resolve(r["gpio"]),
            "gpio_csv": resolve(r["gpio_csv"]) if "gpio_csv" in df.columns else None,  # optional extra
            "pressure_csv": resolve(r["pressure_csv"]),
            "pressure_json": resolve(r["pressure_json"]),
            "cam1_video": resolve(r["cam1_video"]),
            "cam2_video": resolve(r["cam2_video"]),
            # Optional manifest column (manifest.py): K for the rig, fixed by the test recording.
            "manifest_k": r.get("pulse_to_frame_offset", "").strip() or None,
        })
    if only_ids:
        unknown = sorted(set(only_ids) - {r["recording_id"] for r in records})
        if unknown:
            raise InputError(f"recording_id not in the manifest: {unknown}")
    return records


def print_result(row, fails, warns):
    counts = ", ".join(f"{k}={row.get(k, '')}" for k in
                       ("n_frames_isx", "n_ttl", "n_cam1_frames", "n_cam2_frames", "n_gpio_sync_edges"))
    print(f"{row['recording_id']}: {row['count_check']}  (K={row['pulse_to_frame_offset']}; {counts})")
    for f in fails:
        print(f"   FAIL  {f}")
    for w in warns:
        print(f"   warn  {w}")


def run(records, recordings_path, k_cli=None, cam_lead_cli=None, exact=False,
        sync_channel=DEFAULT_SYNC_CHANNEL):
    existing = read_recordings(recordings_path)
    results = []
    for rec in records:
        rid = rec["recording_id"]
        # K: --pulse-offset, else the manifest column, else what recordings.csv already holds.
        k_old = parse_k(existing_value(existing, rid, "pulse_to_frame_offset"),
                        f"{recordings_path} ({rid})", strict=False)
        k_new = k_cli
        if k_new is None:
            k_new = parse_k(rec.get("manifest_k"), f"manifest ({rid})")
        k = k_new if k_new is not None else k_old
        note = None
        if k_new is not None and k_old is not None and k_new != k_old:
            note = f"pulse_to_frame_offset changed from {k_old} to {k_new}"
        lead_old = existing_value(existing, rid, "cam_lead_frames")
        lead = cam_lead_cli if cam_lead_cli is not None else \
            (int(float(lead_old)) if lead_old is not None else DEFAULT_CAM_LEAD)
        res = check_recording(rec, k=k, k_source=note, cam_lead=lead, exact=exact,
                              sync_channel=sync_channel)
        print_result(*res)
        results.append(res)
    update_recordings(recordings_path, results)
    print(f"wrote {recordings_path}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="manifest.csv (paths relative to its folder)")
    ap.add_argument("--recording-id", nargs="+", help="recording(s) to check; required without --manifest")
    ap.add_argument("--isxd")
    ap.add_argument("--gpio", help="raw .gpio, *_gpio.isxd or the GPIO CSV export")
    ap.add_argument("--gpio-csv", help="IDPS GPIO CSV export (Time (s), Channel Name, Value)")
    ap.add_argument("--pressure-csv")
    ap.add_argument("--pressure-json")
    ap.add_argument("--cam1")
    ap.add_argument("--cam2")
    ap.add_argument("--recordings", help="recordings.csv to update (default: out/recordings.csv "
                                         "in the pipeline folder)")
    ap.add_argument("--pulse-offset", type=int, default=None,
                    help="K in 'SYNC pulse n ends isx_frame n-1+K' (default: the manifest's optional "
                         "pulse_to_frame_offset column, else the value in recordings.csv, else 0 "
                         "with a warning)")
    ap.add_argument("--cam-lead-frames", type=int, default=None,
                    help=f"cam_lead_frames to record, documentation only since CONTRACTS.md 1a "
                         f"(default: existing value, else {DEFAULT_CAM_LEAD})")
    ap.add_argument("--exact", action="store_true", help="decode every video frame instead of trusting "
                                                         "the container frame count")
    ap.add_argument("--sync-channel", default=DEFAULT_SYNC_CHANNEL, help="GPIO channel holding SYNC")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data in out/ and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    try:
        if args.manifest:
            records = manifest_records(args.manifest, args.recording_id)
            # never next to the manifest: that is the data folder (often a data drive)
            out = args.recordings or DEFAULT_RECORDINGS
        else:
            if not args.recording_id or len(args.recording_id) != 1:
                ap.error("without --manifest give exactly one --recording-id plus the file paths")
            ab = lambda p: os.path.abspath(p) if p else None
            records = [{"recording_id": args.recording_id[0], "isxd": ab(args.isxd), "gpio": ab(args.gpio),
                        "gpio_csv": ab(args.gpio_csv), "pressure_csv": ab(args.pressure_csv),
                        "pressure_json": ab(args.pressure_json), "cam1_video": ab(args.cam1),
                        "cam2_video": ab(args.cam2)}]
            out = args.recordings or DEFAULT_RECORDINGS
        results = run(records, out, args.pulse_offset, args.cam_lead_frames, args.exact, args.sync_channel)
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0 if all(r[0]["count_check"] == "PASS" for r in results) else 1


# ----------------------------------------------------------------------------- self-test

def _write_video(path, n, size=(32, 24)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
    if not w.isOpened():
        raise RuntimeError("cv2.VideoWriter cannot write MJPG .avi here")
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), (i * 7) % 255, np.uint8))
    w.release()


def _write_logger(folder, stem, pulse_t, trigger="ttl", session=1, pre_s=0.0, tail_s=2.1, dt=0.0125):
    """A pressure_logger-style session: CSV of samples + info.json (as recorder.py writes them)."""
    t0 = pulse_t[0] - pre_s if trigger == "manual" else pulse_t[0] + 0.004
    times = np.arange(t0, pulse_t[-1] + tail_s, dt)
    counts = np.searchsorted(pulse_t, times, side="right")
    csv_path = os.path.join(folder, f"{stem}_S{session:02d}_pressure.csv")
    pd.DataFrame({"t_s": np.round(times - times[0], 4), "arduino_ms": np.round(times * 1000).astype(int),
                  "pc_time": "", "raw": 0, "baseline": 0.0, "corrected": 0.0, "indicator": 0, "ttl": 0,
                  "ttl_count": counts}).to_csv(csv_path, index=False)
    with open(csv_path.replace("_pressure.csv", "_info.json"), "w") as f:
        json.dump({"session": session, "trigger": trigger, "ttl_pulses": int(counts[-1]),
                   "ttl_mode": "pulses", "file": os.path.basename(csv_path)}, f)
    return csv_path


def _write_gpio_csv(path, edge_t, end_t):
    rows = [(0.0, "GPIO-1", 0), (0.0, DEFAULT_SYNC_CHANNEL, 0)]
    for t in edge_t:
        rows += [(round(t, 4), DEFAULT_SYNC_CHANNEL, 1), (round(t + 0.010, 4), DEFAULT_SYNC_CHANNEL, 0)]
    rows += [(round(end_t, 4), DEFAULT_SYNC_CHANNEL, 0), (round(end_t, 4), "GPIO-1", 0)]
    with open(path, "w") as f:
        f.write("Time (s), Channel Name, Value\n")
        for t, ch, v in rows:
            f.write(f"{t}, {ch}, {v}\n")


def _make_case(root, rid, n_frames=40, n_pulses=None, first_pulse_frame=0, cam2=None, gpio_edges=None,
               gpio_tail_s=0.030, trigger="ttl", period=0.05):
    """One synthetic recording. Pulses are emitted for frames first_pulse_frame .. ; the movie
    starts at the first-frame SYNC time, the GPIO 79 ms earlier (as on the 2024 rig)."""
    folder = os.path.join(root, rid)
    os.makedirs(folder)
    movie_ms = 1733930697233
    footer, nbytes = isx_meta.fake_movie_footer(n_frames, start_ms=movie_ms, period_us=int(period * 1e6))
    isxd = os.path.join(folder, f"{rid}_video.isxd")
    isx_meta.write_fake_isxd(isxd, footer, bytes(nbytes))
    n_pulses = n_frames - first_pulse_frame if n_pulses is None else n_pulses
    gpio_lead = 0.079
    pulse_t = gpio_lead + 0.0006 + (first_pulse_frame + np.arange(n_pulses)) * period   # GPIO clock
    n_gpio = n_pulses if gpio_edges is None else gpio_edges
    end_t = pulse_t[n_gpio - 1] + gpio_tail_s
    isx_meta.write_fake_raw_gpio(os.path.join(folder, f"{rid}_video.gpio"), movie_ms - round(gpio_lead * 1000),
                                 int(end_t * 5000))
    _write_gpio_csv(os.path.join(folder, f"{rid}_video.csv"), pulse_t[:n_gpio], end_t)
    pressure = _write_logger(folder, rid, pulse_t, trigger=trigger, pre_s=1.0)
    _write_video(os.path.join(folder, "cam1.avi"), n_pulses)
    _write_video(os.path.join(folder, "cam2.avi"), n_pulses if cam2 is None else cam2)
    rel = lambda p: os.path.relpath(p, root)
    return {"recording_id": rid, "isxd": rel(isxd), "gpio": rel(os.path.join(folder, f"{rid}_video.gpio")),
            "pressure_csv": rel(pressure), "pressure_json": "", "cam1_video": rel(os.path.join(folder, "cam1.avi")),
            "cam2_video": rel(os.path.join(folder, "cam2.avi"))}


def selftest():
    root = os.path.join(HERE, "out", "selftest_count_check")
    if os.path.isdir(root):
        shutil.rmtree(root)                 # only our own scratch folder inside out/
    os.makedirs(root)
    failures = []

    def check(cond, what):
        print(("  ok    " if cond else "  FAIL  ") + what)
        if not cond:
            failures.append(what)

    rows = [
        _make_case(root, "T_ok"),
        _make_case(root, "T_cam2short", cam2=37),
        _make_case(root, "T_gpiotail", gpio_edges=39, gpio_tail_s=0.030),
        _make_case(root, "T_k1", first_pulse_frame=1),
        # one pulse more than frames (pulse 1 one period before the movie start): the only
        # data the binding count rule n_ttl = n_frames_isx + K accepts with K = 1
        _make_case(root, "T_kplus", first_pulse_frame=-1, n_pulses=41),
        _make_case(root, "T_manual", trigger="manual"),
        _make_case(root, "T_wrongsession"),
        _make_case(root, "T_nocam"),
    ]
    # T_wrongsession: the manifest points at a 1-pulse preview session; the real one is S02.
    ws = os.path.join(root, "T_wrongsession")
    real = os.path.join(ws, "T_wrongsession_S01_pressure.csv")
    os.rename(real, real.replace("_S01_", "_S02_"))
    os.rename(real.replace("_pressure.csv", "_info.json"), real.replace("_S01_pressure.csv", "_S02_info.json"))
    with open(real.replace("_S01_pressure.csv", "_S02_info.json")) as f:
        info = json.load(f)
    info.update(session=2, file="T_wrongsession_S02_pressure.csv")
    with open(real.replace("_S01_pressure.csv", "_S02_info.json"), "w") as f:
        json.dump(info, f)
    _write_logger(ws, "T_wrongsession", np.array([1.0]), session=1)
    rows[-1]["cam2_video"] = ""
    manifest = os.path.join(root, "manifest.csv")
    cols = ["recording_id", "mouse_id", "sex", "day", "assay", "arm", "stim_order", "mild_g",
            "injection_time", "assay_t0_s", "isxd", "gpio", "imu", "pressure_csv", "pressure_json",
            "cam1_video", "cam2_video", "scoring_mat", "notes"]
    pd.DataFrame(rows).reindex(columns=cols).fillna("").to_csv(manifest, index=False)

    # Another module's values must survive count_check.
    rec_path = os.path.join(root, "recordings.csv")
    pd.DataFrame([{"recording_id": "T_ok", "arduino_fit_a": "1.23", "notes": "master_timeline: fit ok",
                   "foo_extra": "keep me"}]).to_csv(rec_path, index=False)

    code = main(["--manifest", manifest, "--recordings", rec_path, "--exact"])
    check(code == 1, "exit code 1 when any recording FAILs")
    df = pd.read_csv(rec_path, dtype=str, keep_default_na=False).set_index("recording_id")
    get = lambda rid, col: df.at[rid, col]

    check(get("T_ok", "count_check") == "PASS", "consistent recording -> PASS")
    check(all(get("T_ok", c) == "40" for c in ("n_frames_isx", "n_ttl", "n_cam1_frames", "n_cam2_frames",
                                               "n_gpio_sync_edges")), "all five counts = 40")
    check(get("T_ok", "k_from_gpio") == "0" and abs(float(get("T_ok", "gpio_first_edge_lag_ms")) - 0.6) < 0.05,
          "GPIO first edge 0.6 ms after movie start -> K 0")
    check(get("T_ok", "period_source") == "gpio_sync_mean" and abs(float(get("T_ok", "period_s")) - 0.05) < 1e-6,
          "period from the GPIO SYNC edges")
    check(get("T_ok", "cam_count_method") == "decoded" and get("T_ok", "cam_lead_frames") == "1",
          "--exact decodes; cam_lead_frames defaults to 1")
    check(get("T_ok", "arduino_fit_a") == "1.23" and get("T_ok", "foo_extra") == "keep me",
          "other modules' columns are kept")
    check("master_timeline: fit ok" in get("T_ok", "notes") and "K=0 assumed" in get("T_ok", "notes"),
          "notes keep other segments and say that K was assumed")

    check(get("T_cam2short", "count_check") == "FAIL"
          and "cam2 has 3 fewer frames than pulses" in get("T_cam2short", "notes"),
          "cam2 short by 3 -> FAIL with a readable reason")
    check(get("T_gpiotail", "count_check") == "FAIL"
          and "GPIO log stopped before the final frame's pulse" in get("T_gpiotail", "notes"),
          "GPIO one edge short at the end -> FAIL, explained as tail truncation")
    check(get("T_k1", "count_check") == "FAIL" and "would need K = -1" in get("T_k1", "notes")
          and "disagree in sign" in get("T_k1", "notes") and get("T_k1", "k_from_gpio") == "1",
          "frame 0 without pulse -> FAIL; count rule says K=-1, GPIO (index rule) says 1, sign conflict named")
    check(get("T_kplus", "count_check") == "FAIL" and "would need K = +1" in get("T_kplus", "notes")
          and get("T_kplus", "k_from_gpio") == "-1", "one extra pulse, K unset -> FAIL with the K hint")
    check(get("T_manual", "count_check") == "PASS" and "started manually" in get("T_manual", "notes")
          and get("T_manual", "ttl_trigger") == "manual", "manual logger start -> warning")
    check(get("T_wrongsession", "count_check") == "FAIL"
          and "S02 in the same folder has exactly 40 pulses" in get("T_wrongsession", "notes"),
          "1-pulse preview session in the manifest -> points to S02")
    check(get("T_nocam", "count_check") == "FAIL" and "cam2 video not available" in get("T_nocam", "notes"),
          "missing camera -> FAIL")

    # The count rule is never repaired: 39 pulses for 40 frames FAIL for every K but -1.
    code = main(["--manifest", manifest, "--recordings", rec_path, "--recording-id", "T_k1", "--pulse-offset", "1"])
    df2 = pd.read_csv(rec_path, dtype=str, keep_default_na=False).set_index("recording_id")
    check(code == 1 and df2.at["T_k1", "count_check"] == "FAIL"
          and "41 = n_frames_isx 40 + K 1" in df2.at["T_k1", "notes"],
          "--pulse-offset 1 with 39 pulses for 40 frames -> FAIL (n_ttl must be n_frames_isx + K)")
    # Fixing K for one recording: PASS, others untouched, one count_check segment only.
    code = main(["--manifest", manifest, "--recordings", rec_path, "--recording-id", "T_kplus", "--pulse-offset", "1"])
    df2 = pd.read_csv(rec_path, dtype=str, keep_default_na=False).set_index("recording_id")
    check(code == 0 and df2.at["T_kplus", "count_check"] == "PASS"
          and df2.at["T_kplus", "pulse_to_frame_offset"] == "1"
          and "index-rule K = -1" in df2.at["T_kplus", "notes"],
          "--pulse-offset 1 with 41 pulses for 40 frames -> PASS, GPIO sign warning kept")
    check(df2.at["T_ok", "count_check"] == "PASS" and len(df2) == len(df), "other rows untouched")
    main(["--manifest", manifest, "--recordings", rec_path, "--recording-id", "T_ok"])
    df3 = pd.read_csv(rec_path, dtype=str, keep_default_na=False).set_index("recording_id")
    check(df3.at["T_ok", "notes"].count("count_check:") == 1
          and "master_timeline: fit ok" in df3.at["T_ok", "notes"], "rerun replaces its own notes segment")
    code = main(["--manifest", manifest, "--recordings", rec_path, "--recording-id", "T_kplus"])
    df3 = pd.read_csv(rec_path, dtype=str, keep_default_na=False).set_index("recording_id")
    check(code == 0 and df3.at["T_kplus", "pulse_to_frame_offset"] == "1"
          and df3.at["T_kplus", "count_check"] == "PASS", "K stored in recordings.csv is reused")

    # Optional manifest column pulse_to_frame_offset (manifest.py) sets K without --pulse-offset.
    mk = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    mk["pulse_to_frame_offset"] = ""
    mk.loc[mk["recording_id"] == "T_kplus", "pulse_to_frame_offset"] = "1"
    mk_path = os.path.join(root, "manifest_k.csv")
    mk.to_csv(mk_path, index=False)
    code = main(["--manifest", mk_path, "--recording-id", "T_kplus", "--recordings",
                 os.path.join(root, "recordings_k.csv")])
    dk = pd.read_csv(os.path.join(root, "recordings_k.csv"), dtype=str, keep_default_na=False)
    check(code == 0 and dk.at[0, "pulse_to_frame_offset"] == "1" and "K=0 assumed" not in dk.at[0, "notes"],
          "manifest pulse_to_frame_offset column -> K=1, PASS")

    # FAIL reasons carry codes, so run_pipeline can tell which explicit option covers which one
    codes = {r["recording_id"]: [f.code for f in fails] for r, fails, _ in
             run(manifest_records(manifest, ["T_gpiotail", "T_nocam", "T_cam2short"]),
                 os.path.join(root, "recordings_codes.csv"), k_cli=0)}
    check(codes == {"T_gpiotail": ["gpio_tail"], "T_nocam": ["cam2_missing"],
                    "T_cam2short": ["cam2_count"]}, f"FAIL codes {codes}")
    # A K that is not a whole number is bad input (exit 2), never truncated to an integer
    mk2 = pd.read_csv(manifest, dtype=str, keep_default_na=False).assign(pulse_to_frame_offset="0.9")
    mk2_path = os.path.join(root, "manifest_kfloat.csv")
    mk2.to_csv(mk2_path, index=False)
    check(main(["--manifest", mk2_path, "--recording-id", "T_ok", "--recordings",
                os.path.join(root, "recordings_kfloat.csv")]) == 2, "K '0.9' in the manifest -> exit 2")
    check(os.path.dirname(DEFAULT_RECORDINGS) == os.path.join(HERE, "out"),
          "default recordings.csv is in the pipeline's out/, never next to the manifest")

    # Explicit-argument mode and a manifest without contract columns.
    ok = os.path.join(root, "T_ok")
    code = main(["--recording-id", "X", "--isxd", os.path.join(ok, "T_ok_video.isxd"),
                 "--pressure-csv", os.path.join(ok, "T_ok_S01_pressure.csv"),
                 "--cam1", os.path.join(ok, "cam1.avi"), "--cam2", os.path.join(ok, "cam2.avi"),
                 "--gpio-csv", os.path.join(ok, "T_ok_video.csv"), "--pulse-offset", "0",
                 "--recordings", os.path.join(root, "explicit.csv")])
    check(code == 0, "explicit-argument mode -> PASS")
    bad = os.path.join(root, "bad_manifest.csv")
    pd.DataFrame([{"recording_id": "A", "isxd": ""}]).to_csv(bad, index=False)
    check(main(["--manifest", bad]) == 2, "manifest without contract columns -> exit 2")

    print("count_check selftest:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
