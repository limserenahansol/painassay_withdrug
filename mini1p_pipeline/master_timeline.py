"""Master timeline: join every stream of one recording onto the Inscopix frame index.

Reads (CONTRACTS.md section 3):
  recordings.csv          counts, K (pulse_to_frame_offset), cam_lead_frames, Arduino fit
  Inscopix timestamps     optional IDPS frame-timestamp CSV (else the frame table's tsc_us,
                          else isx_frame * period_s)
  pressure_frames.csv     one row per SYNC pulse (pressure_sync)
  stim_events.csv         contact onsets from the logger (pressure_sync)
  camera_frames.csv       cam frame -> pulse -> isx_frame (camera_sync)
  behaviour_events.csv    scored events in cam1 frames (scoring_import)
  video_deliveries.csv    optional: scored delivery taps (scoring_import), used for heat
  EXTRACT frame table     <mouse>_d<day>_frame_table.csv (recording_id, isx_frame; extract_1p.m
                          adds valid, invalid_reason, extract_row, tsc_us)
  pressure_summary.json   optional: pressure_sync's clock map, reused so tables agree

Writes frames.csv (the single input to analysis) and fills in stim_events.csv
(t_expected/onset_t_master/onset_isx_frame/onset_cam_frame, video rows for trials the logger
cannot see, 'expected' rows for missing trials), behaviour_events.csv (master-clock columns,
linked event_id, latency) and the Arduino fit / t_source columns of recordings.csv.

Index rules used here, all from CONTRACTS.md section 1a (revised 2026-10-19):
  Pulse n ENDS isx_frame n-1+K, and the IDAS TSC timestamp of that frame marks the same
  moment, so  t_master_s(isx_frame n-1+K) = time of pulse n  (pulse_end_frame).
  Anything stamped with a pulse COUNT n happened after pulse n, i.e. during the NEXT frame:
      logger rows with ttl_count == n  ->  isx_frame n + K
      camera frame n (pulse n)         ->  isx_frame n + K
  count_to_isx_frame(n, K) = n + K is the one function that does this, for both streams.
  The old camera-only cam_lead_frames is superseded; the column stays 1 for documentation.
  Camera frame times are the trigger (snapshot) time, i.e. the time of its pulse.
  A logger onset at Arduino time t gets count n = the pulses before t (pressure_frames
  t_pulse_arduino_s, the same numbers pressure_sync used), isx_frame n + K and onset_cam_frame
  = the camera frame of pulse n (via camera_frames.csv, which knows about lost camera frames).
  The join refuses to run unless n_ttl == expected_n_ttl(n_frames_isx, K) = n_frames_isx + K
  (CONTRACTS.md count rule; see expected_n_ttl for why that only makes sense for K = 0), the
  pressure_frames it is given hold exactly pulses 1..n_ttl, every camera frame is one distinct
  pulse of 1..n_ttl, the timestamps cover the movie, and count_check is not FAIL (unless the
  caller accepts the FAIL explicitly). A wrong stim_order (most typed delivery keys of a slot
  disagree) also stops it.
  extract_row is the 1-based (MATLAB) row of the day's EXTRACT trace matrix, as written by
  extract_1p.m; subtract 1 for a numpy index.

Epoch / slot / block / trial come from the fixed schedule in CONTRACTS.md (baseline 0-300 s,
4 stimulus slots of 300 s with 60-s rests, 5 blocks of 60 s, trials at +0/15/30 s), anchored
at assay_t0_s. For per-frame labels a trial lasts until the next trial (trial 3: to the end
of its 60-s block).

Usage
  python master_timeline.py --recording-id M01_d1_pre --manifest manifest.csv \
      --recordings out/M01_d1_pre/recordings.csv --pressure-frames ... --stim-events ... \
      --camera-frames ... --behaviour-events ... [--video-deliveries ...] \
      [--timestamps ts.csv] [--frame-table M01_d1_frame_table.csv] --out-dir out/M01_d1_pre
  python master_timeline.py --selftest
"""

import argparse
import os
import re
import sys
import tempfile
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- contract tables
RECORDINGS_COLUMNS = [
    "recording_id", "n_frames_isx", "n_dropped", "n_trimmed_early", "period_s", "fps", "n_ttl",
    "n_gpio_sync_edges", "n_cam1_frames", "n_cam2_frames", "pulse_to_frame_offset",
    "cam_lead_frames", "arduino_fit_a", "arduino_fit_b", "arduino_fit_ppm",
    "arduino_fit_resid_ms", "count_check", "notes",
]
PRESSURE_FRAMES_COLUMNS = ["recording_id", "pulse", "isx_frame", "t_arduino_s", "force_mean",
                           "force_max", "n_samples"]
STIM_EVENTS_COLUMNS = [
    "event_id", "recording_id", "stim_type", "slot", "block", "trial", "source",
    "t_expected_master_s", "onset_t_arduino_s", "onset_t_master_s", "onset_isx_frame",
    "onset_cam_frame", "threshold_t_master_s", "peak_t_master_s", "peak_force",
    "offset_t_master_s", "duration_s", "match_residual_s", "flags",
]
CAMERA_FRAMES_COLUMNS = ["recording_id", "camera", "cam_frame", "pulse", "isx_frame",
                         "t_master_s", "method", "gap_before"]
BEHAVIOUR_EVENTS_COLUMNS = [
    "recording_id", "camera", "type", "on_cam_frame", "off_cam_frame", "on_t_master_s",
    "off_t_master_s", "on_isx_frame", "off_isx_frame", "event_id", "latency_s", "scorer",
]
BEHAVIOUR_TYPES = ("withdrawal", "flinch", "attending", "licking_biting", "guarding", "escape",
                   "uncertain")
AFFECTIVE_TYPES = ("attending", "licking_biting", "guarding", "escape")
REFLEX_TYPES = ("withdrawal", "flinch")
FRAMES_COLUMNS = [
    "recording_id", "isx_frame", "t_master_s", "t_assay_s", "is_dropped", "extract_row",
    "epoch", "slot", "stim_type", "block", "trial", "stim_on", "event_id", "force_max",
    "cam1_frame", "cam2_frame", "affective_state", "reflex", "motion_cam1", "motion_cam2",
]

# ---------------------------------------------------------------- schedule (CONTRACTS.md)
BASELINE_S = 300.0
SLOT_S = 300.0
REST_S = 60.0
N_SLOTS = 4
BLOCK_S = 60.0
N_BLOCKS = 5
TRIAL_OFFSETS_S = (0.0, 15.0, 30.0)
ASSAY_END_S = BASELINE_S + N_SLOTS * SLOT_S + (N_SLOTS - 1) * REST_S       # 1680 s
# A scored or detected delivery is matched to the scheduled trial nearest in time, but
# never further than half the 15-s trial spacing, so it cannot be given to a neighbour.
MATCH_TOLERANCE_S = 7.5
# A behaviour event is linked to the latest stimulus onset at most this long before it
# (the longest per-trial window, trial 3 of a block).
LINK_WINDOW_S = 30.0


class TimelineError(RuntimeError):
    """A contract violation that makes the join meaningless. Never caught silently."""


# ---------------------------------------------------------------- the index rule (1a)
def count_to_isx_frame(n, K):
    """isx_frame during which something stamped with pulse count n happened (CONTRACTS.md 1a).

    Pulse n ends isx_frame n-1+K; a logger row with ttl_count == n was taken after pulse n
    and before pulse n+1, and camera frame n is exposed right after pulse n, so both fall
    in the exposure of isx_frame n + K. Every module places logger samples and camera
    frames with this function, so the rule lives in exactly one place. Works on scalars
    and numpy arrays; the result can lie outside 0..n_frames-1 (count 0 before imaging,
    the last pulse's count after the movie ends): callers must range-check it."""
    return n + K


def pulse_end_frame(n, K):
    """isx_frame whose exposure pulse n ends: n - 1 + K. Its TSC timestamp IS the pulse
    time, so this is used for the Arduino clock fit and for camera trigger times, never to
    place a count-stamped sample (that is count_to_isx_frame)."""
    return n - 1 + K


def expected_n_ttl(n_frames_isx, K):
    """Pulse count a complete recording must have: n_frames_isx + K (CONTRACTS.md 1a/3,
    binding). count_check, camera_sync, run_pipeline and build_timeline all use this.

    Caveat, kept here so it is fixed in one place: the index rule (pulse n ends isx_frame
    n-1+K) implies n_frames_isx - K pulses for frames K..N-1. The two agree only for K = 0,
    which is what the 2024 data show (first SYNC edge 0.0-0.6 ms after the movie start).
    A rig with K != 0 must not be analysed until the contract settles the sign."""
    return n_frames_isx + K


def slot_start_s(slot):
    """Assay time (s) at which stimulus slot 1..4 starts: 300, 660, 1020, 1380."""
    return BASELINE_S + (slot - 1) * (SLOT_S + REST_S)


def schedule_trials(assay_t0_s, stim_order):
    """The 60 scheduled trials of an assay: slot, block, trial, stim_type, expected times."""
    rows = []
    for slot in range(1, N_SLOTS + 1):
        for block in range(1, N_BLOCKS + 1):
            for trial, off in enumerate(TRIAL_OFFSETS_S, start=1):
                t_assay = slot_start_s(slot) + (block - 1) * BLOCK_S + off
                rows.append(dict(slot=slot, block=block, trial=trial,
                                 stim_type=stim_order[slot - 1],
                                 t_expected_assay_s=t_assay,
                                 t_expected_master_s=t_assay + assay_t0_s))
    return pd.DataFrame(rows)


def classify_assay_times(t_assay):
    """Per-time epoch / slot / block / trial labels (arrays) for times since assay_t0_s."""
    t = np.asarray(t_assay, dtype=float)
    n = t.size
    epoch = np.full(n, "", dtype=object)
    slot = np.zeros(n, dtype=int)
    block = np.zeros(n, dtype=int)
    trial = np.zeros(n, dtype=int)
    epoch[t < 0] = "pre_baseline"
    epoch[(t >= 0) & (t < BASELINE_S)] = "baseline"
    epoch[t >= ASSAY_END_S] = "post_end"
    for s in range(1, N_SLOTS + 1):
        a = slot_start_s(s)
        in_slot = (t >= a) & (t < a + SLOT_S)
        epoch[in_slot] = "stim"
        slot[in_slot] = s
        u = t[in_slot] - a
        b = np.floor(u / BLOCK_S).astype(int) + 1
        block[in_slot] = b
        w = u - (b - 1) * BLOCK_S
        trial[in_slot] = np.where(w < TRIAL_OFFSETS_S[1], 1, np.where(w < TRIAL_OFFSETS_S[2], 2, 3))
        if s < N_SLOTS:
            epoch[(t >= a + SLOT_S) & (t < slot_start_s(s + 1))] = "rest"
    epoch[~np.isfinite(t)] = ""
    return dict(epoch=epoch, slot=slot, block=block, trial=trial)


# ---------------------------------------------------------------- small table helpers
def read_csv(path, required=None, what="table"):
    """Read a contract CSV; fail loudly if it is missing or lacks required columns."""
    if not path or not os.path.exists(path):
        raise TimelineError(f"{what} not found: {path}")
    df = pd.read_csv(path, dtype={"recording_id": str, "event_id": str, "flags": str,
                                  "camera": str, "method": str, "notes": str})
    missing = [c for c in (required or []) if c not in df.columns]
    if missing:
        raise TimelineError(f"{what} {path} lacks contract column(s): {', '.join(missing)}")
    return df


TEXT_COLUMNS = {"recording_id", "event_id", "stim_type", "source", "flags", "camera", "type",
                "scorer", "method", "notes", "count_check", "epoch"}


def ensure_columns(df, columns):
    """Add any missing contract columns (empty) and put the contract ones first. Text
    columns are object-typed so that filling them never fights a float NaN column."""
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    for c in TEXT_COLUMNS & set(df.columns):
        df[c] = df[c].astype(object)
    extra = [c for c in df.columns if c not in columns]
    return df[list(columns) + extra]


def replace_rows(path, new_rows, recording_id, columns, key="recording_id"):
    """Write new_rows into the table at path, replacing the rows of this recording only."""
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str)
        old = old[old[key] != recording_id]
        out = pd.concat([old, new_rows.astype(object)], ignore_index=True)
    else:
        out = new_rows
    out = ensure_columns(out, columns)
    out.to_csv(path, index=False)
    return path


def other_recordings(path, recording_id):
    """Recording ids other than recording_id found in the table at path (sorted list; empty
    if the file does not exist). pressure_frames.csv, stim_events.csv and frames.csv hold one
    recording each (CONTRACTS.md 3), so a writer refuses a file that belongs to another one."""
    if not os.path.exists(path):
        return []
    try:
        ids = pd.read_csv(path, usecols=["recording_id"], dtype=str)["recording_id"]
    except (ValueError, pd.errors.EmptyDataError):
        return []          # no recording_id column (or empty): nothing to protect
    return sorted(set(ids.dropna()) - {str(recording_id)})


def add_flag(flags, flag):
    """Append a flag to a semicolon list without duplicates."""
    parts = [p for p in str(flags if isinstance(flags, str) else "").split(";") if p]
    if flag not in parts:
        parts.append(flag)
    return ";".join(parts)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def _int_or_none(x):
    v = _num(x)
    return None if not np.isfinite(v) else int(round(v))


# ---------------------------------------------------------------- frame times
def read_isx_timestamps(path):
    """Return (isx_frame array, seconds array) from an IDPS timestamp CSV export or a
    pipeline CSV with isx_frame + a time column. Dropped frames may be absent or NaN."""
    df = pd.read_csv(path)
    names = {c.strip().lower(): c for c in df.columns}
    frame_col = next((names[k] for k in ("isx_frame", "global frame number", "frame",
                                         "frame_number") if k in names), None)
    time_col = next((c for k, c in names.items() if "timestamp" in k), None)
    if time_col is None:
        time_col = next((names[k] for k in ("t_master_s", "t_s", "time_s", "time (s)")
                         if k in names), None)
    if time_col is None:
        raise TimelineError(f"{path}: no timestamp column (looked for 'Frame Timestamp')")
    frames = (df[frame_col].to_numpy(dtype=float) if frame_col else
              np.arange(len(df), dtype=float))
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(frames)
    frames, t = frames[ok].astype(int), t[ok]
    return frames, _to_seconds(t, time_col)


def _to_seconds(t, name=""):
    """Timestamps in seconds. TSC exports are microseconds; frame periods are 10-170 ms,
    so a median step above 1000 can only be microseconds. A zero or negative TSC marks a
    frame without a timestamp (IDPS writes 0 for dropped frames); a 'start'-referenced
    export legitimately has 0 in its first row."""
    t = np.asarray(t, dtype=float).copy()
    low = str(name).lower()
    steps = np.diff(t[np.isfinite(t) & (t > 0)])
    micro = ("(us)" in low or "tsc" in low or "usec" in low or "(Âµs)" in low or
             (steps.size > 0 and np.median(steps) > 1000))
    bad = t <= 0
    if t.size and not micro and np.isfinite(t[0]) and t[0] == 0:
        bad[0] = False
    t[bad] = np.nan
    return t * 1e-6 if micro else t


def frame_times(n_frames, period_s=None, timestamps=None, n_dropped=None):
    """t_master_s for every isx_frame 0..n_frames-1, the dropped mask and t_source.

    timestamps: a timestamp CSV path, or an (isx_frame, seconds) pair (e.g. tsc_us of the
    EXTRACT frame table converted to seconds). With timestamps, t_master is seconds since
    isx_frame 0 (dropped frames interpolated, so the time axis stays continuous). Without,
    it is isx_frame * period_s.

    A frame without a timestamp (row absent, 0 or empty) is a dropped frame, so the table must
    leave exactly n_dropped frames (the footer's count, when given) without one, and never the
    last frame: a cut copy or the preprocessed (_pp, trimmed and renumbered) timestamps would
    otherwise turn thousands of real frames into 'dropped' ones with extrapolated times."""
    if timestamps is not None and not (isinstance(timestamps, str) and timestamps == ""):
        if isinstance(timestamps, str):
            fr, t = read_isx_timestamps(timestamps)
            timestamps_path = timestamps
        else:
            fr, t = (np.asarray(timestamps[0], int), np.asarray(timestamps[1], float))
            timestamps_path = "frame-table timestamps"
        if n_frames is None:
            n_frames = int(fr.max()) + 1
        if fr.size and (fr.min() < 0 or fr.max() >= n_frames):
            raise TimelineError(f"{timestamps_path}: frame numbers outside 0..{n_frames - 1}")
        if np.unique(fr).size != fr.size:
            raise TimelineError(f"{timestamps_path}: a frame number appears twice")
        full = np.full(n_frames, np.nan)
        full[fr] = t
        valid = np.isfinite(full)
        if valid.sum() < 2:
            raise TimelineError(f"{timestamps_path}: fewer than 2 valid timestamps")
        n_missing = int((~valid).sum())
        if not valid[-1] or (n_dropped is not None and n_missing != n_dropped):
            raise TimelineError(
                f"{timestamps_path}: {n_missing} of {n_frames} frames have no timestamp "
                f"(last frame {'without' if not valid[-1] else 'with'} one), but the movie has "
                f"{'an unknown number of' if n_dropped is None else n_dropped} dropped frames. "
                "The table does not cover this movie: a cut copy, another recording, or the "
                "preprocessed (_pp) timestamps instead of the raw ones?")
        idx = np.arange(n_frames)
        if np.any(np.diff(full[valid]) <= 0):
            raise TimelineError(f"{timestamps_path}: timestamps do not increase; refusing "
                                "to guess which frames are bad")
        first, last = idx[valid][0], idx[valid][-1]
        period = (full[last] - full[first]) / (last - first)
        t_rel = np.interp(idx, idx[valid], full[valid])
        # np.interp is flat outside the valid range; extend with the mean period instead
        t_rel[idx < first] = full[first] - (first - idx[idx < first]) * period
        t_rel[idx > last] = full[last] + (idx[idx > last] - last) * period
        t_master = t_rel - t_rel[0]
        return t_master, ~valid, "tsc"
    if n_frames is None or period_s is None or not np.isfinite(period_s):
        raise TimelineError("no timestamps and no n_frames/period_s: cannot build frame times")
    return np.arange(n_frames) * float(period_s), np.zeros(n_frames, bool), "period"


def mean_period(t_master, dropped):
    """Mean frame period (s) between the first and last frame that has a real timestamp."""
    ok = np.flatnonzero(~np.asarray(dropped, bool) & np.isfinite(t_master))
    if ok.size < 2:
        return np.nan
    return float((t_master[ok[-1]] - t_master[ok[0]]) / (ok[-1] - ok[0]))


def recording_frame_times(recording_id, rec, timestamps_path=None, frame_table=None):
    """The one choice of time base for a recording, shared by every module:
    the timestamp CSV if given, else the tsc_us column of the EXTRACT frame table, else
    isx_frame * period_s. Returns (t_master, dropped mask, t_source)."""
    N = _int_or_none(rec.get("n_frames_isx"))
    ts = timestamps_path or None
    if ts is None and frame_table is not None and "tsc_us" in frame_table:
        ft = frame_table[(frame_table["recording_id"] == recording_id).to_numpy()]
        tsc = pd.to_numeric(ft["tsc_us"], errors="coerce").to_numpy(float)
        if np.isfinite(tsc).sum() >= 2:
            ts = (ft["isx_frame"].to_numpy(int), _to_seconds(tsc, "tsc_us"))
    return frame_times(N, _num(rec.get("period_s")), ts, _int_or_none(rec.get("n_dropped")))


def fit_arduino_to_master(t_arduino, t_master):
    """Robust linear fit t_master = a + b * t_arduino. Returns a, b, ppm, resid_ms."""
    x = np.asarray(t_arduino, float)
    y = np.asarray(t_master, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        raise TimelineError("fewer than 3 pulses to fit the Arduino clock")
    keep = np.ones(x.size, bool)
    for _ in range(4):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= max(5 * mad, 0.002)
        if np.array_equal(new, keep):
            break
        keep = new
    b, a = np.polyfit(x[keep], y[keep], 1)
    r = y[keep] - (a + b * x[keep])
    return float(a), float(b), float((b - 1.0) * 1e6), float(np.std(r) * 1000.0)


# ---------------------------------------------------------------- the join
def pulse_times_master(pulses, K, t_master):
    """Master time of each pulse n = t_master_s of isx_frame n-1+K (CONTRACTS.md 1a), with
    the mean period beyond the ends of the movie (a pulse before frame 0 when K < 0, or the
    pulse of a frame that does not exist). NaN for a missing pulse number."""
    p = np.asarray(pulses, dtype=float)
    out = np.full(p.shape, np.nan)
    if t_master.size == 0:
        return out
    period = (t_master[-1] - t_master[0]) / max(t_master.size - 1, 1)
    f = pulse_end_frame(p, K)
    ok = np.isfinite(f)
    fi = np.where(ok, f, 0).astype(np.int64)
    inside = ok & (fi >= 0) & (fi < t_master.size)
    out[inside] = t_master[fi[inside]]
    before = ok & (fi < 0)
    out[before] = t_master[0] + fi[before] * period
    after = ok & (fi >= t_master.size)
    out[after] = t_master[-1] + (fi[after] - (t_master.size - 1)) * period
    return out


def _camera_maps(cam, camera, n_frames, K, t_master):
    """For one camera: dict cam_frame -> (isx_frame or None, uncertain, trigger time) and the
    inverse array isx_frame -> cam_frame (0 = none).

    isx_frame = count_to_isx_frame(pulse): cam frame n is exposed right after pulse n, during
    isx_frame n + K. The trigger (snapshot) time is the time of pulse n. Placement uses the
    pulse column, so a 'fit' table with lost frames is honoured; camera_sync's isx_frame
    column must agree, which catches a table written under the superseded n-1+K rule."""
    c = cam[cam["camera"] == camera]
    fwd = {}
    inv = np.zeros(n_frames, dtype=np.int64)
    if c.empty:
        return fwd, inv
    unc = c["uncertain"].fillna(0).astype(int).to_numpy() if "uncertain" in c else np.zeros(len(c), int)
    pulse = pd.to_numeric(c["pulse"], errors="coerce").to_numpy(float)
    isx = count_to_isx_frame(pulse, K)
    written = pd.to_numeric(c["isx_frame"], errors="coerce").to_numpy(float)
    both = np.isfinite(isx) & np.isfinite(written)
    if np.any(written[both] != isx[both]):
        bad = int((written[both] != isx[both]).sum())
        raise TimelineError(f"camera_frames {camera}: isx_frame != pulse + K in {bad} rows (K={K}); "
                            "the table was written under the superseded pulse-1+K rule - rerun "
                            "camera_sync")
    t_trig = pulse_times_master(pulse, K, t_master)
    for cf, i, u, tt in zip(c["cam_frame"].to_numpy(), isx, unc, t_trig):
        if not np.isfinite(i):
            fwd[int(cf)] = (None, int(u), np.nan)
            continue
        i = int(i)
        inside = 0 <= i < n_frames
        fwd[int(cf)] = (i if inside else None, int(u), float(tt))
        if inside:
            inv[i] = int(cf)
    return fwd, inv


def _pulse_lookup(pf):
    """Sorted pulse numbers and their Arduino times, for time -> pulse lookups. pressure_sync's
    t_pulse_arduino_s (fitted pulse grid clipped to the sample bracket) is the time it used to
    give each contact its pulse, so it is used here too; the bracket midpoint t_arduino_s is
    the fallback for a table without it."""
    p = pf.sort_values("pulse")
    col = "t_pulse_arduino_s" if "t_pulse_arduino_s" in p and p["t_pulse_arduino_s"].notna().all() \
        else "t_arduino_s"
    return p["pulse"].to_numpy(dtype=int), p[col].to_numpy(dtype=float)


def arduino_time_to_pulse(t, pulses, pulse_t):
    """Pulse number whose interval [pulse n, pulse n+1) contains Arduino time t, i.e. the
    ttl_count of a logger sample taken at t. 0 = before the first pulse."""
    if not np.isfinite(t):
        return None
    k = int(np.searchsorted(pulse_t, t, side="right"))
    return int(pulses[k - 1]) if k > 0 else 0


def master_time_to_frame(t, t_master):
    """isx_frame whose exposure contains master time t (None if outside the movie).

    t_master_s marks the END of a frame's exposure (CONTRACTS.md 1a), so frame k spans
    (t_master[k-1], t_master[k]]; frame 0 starts one mean period before t_master[0]. This
    agrees with count_to_isx_frame: a logger sample between pulse n and pulse n+1 lands in
    frame n + K either way."""
    if not np.isfinite(t) or t_master.size == 0:
        return None
    i = int(np.searchsorted(t_master, t, side="left"))
    if i >= t_master.size:
        return None
    if i == 0:
        period = (t_master[-1] - t_master[0]) / max(t_master.size - 1, 1)
        return 0 if t > t_master[0] - period else None
    return i


def _next_event_number(ids):
    nums = [int(m.group(1)) for x in ids if isinstance(x, str) and (m := re.search(r"_e(\d+)$", x))]
    return max(nums, default=0) + 1


def build_timeline(recording_id, rec, stim_order, assay_t0_s, pressure_frames=None,
                   stim_events=None, camera_frames=None, behaviour_events=None,
                   video_deliveries=None, timestamps_path=None, frame_table=None,
                   motion=None, arduino_fit=None, accept_count_fail=False,
                   trust_stim_order=False):
    """Join everything for one recording. All table arguments are DataFrames (or None).

    arduino_fit: optional dict(a, b, ppm, resid_ms) from pressure_sync; preferred over a
    fresh fit so that stim_events and recordings.csv share one clock map.
    accept_count_fail: join a recording whose count_check is FAIL (the caller has checked
    that an explicit option covers every reason). trust_stim_order: keep the manifest
    stim_order although most scored delivery keys of a slot disagree with it.
    Returns dict(frames, stim_events, behaviour_events, recording, notes)."""
    notes = []
    N = _int_or_none(rec.get("n_frames_isx"))
    K = _int_or_none(rec.get("pulse_to_frame_offset"))
    if N is None or N <= 0:
        raise TimelineError(f"{recording_id}: recordings.csv has no n_frames_isx")
    if K is None:
        raise TimelineError(f"{recording_id}: pulse_to_frame_offset (K) is empty; it must be "
                            "fixed with a test recording, never assumed")
    lead = _int_or_none(rec.get("cam_lead_frames"))
    if lead is not None and lead != 1:
        # superseded by count_to_isx_frame (CONTRACTS.md 1a); never applied as a number
        notes.append(f"cam_lead_frames={lead} ignored: the camera lead is part of "
                     "count_to_isx_frame now (column kept = 1)")
    rec = dict(rec, cam_lead_frames=1)
    n_ttl = _int_or_none(rec.get("n_ttl"))
    if n_ttl is None:
        # Without the pulse count nothing proves that pulse n is the n-th frame's pulse
        raise TimelineError(f"{recording_id}: recordings.csv has no n_ttl (logger pulse "
                            "count); run count_check first - the join needs n_ttl == "
                            "n_frames_isx + K")
    if n_ttl != expected_n_ttl(N, K):
        raise TimelineError(
            f"{recording_id}: n_ttl={n_ttl} but CONTRACTS.md needs n_frames_isx + K = "
            f"{N} + {K} = {expected_n_ttl(N, K)}. Logger pulses and Inscopix frames do not "
            "match, so no stream can be placed on the frame index. Fix the recording (or the "
            "logger session) first.")
    cc = str(rec.get("count_check") if isinstance(rec.get("count_check"), str) else "").strip()
    if cc == "FAIL":
        why = [p for p in str(rec.get("notes") or "").split(" | ") if p.startswith("count_check:")]
        if not accept_count_fail:
            raise TimelineError(
                f"{recording_id}: count_check is FAIL ({(why or ['no reason in notes'])[0][:300]}). "
                "The join refuses a failed recording; run_pipeline accepts one only with the "
                "option that covers the reason (--allow-mismatch with anchors, --no-camera, "
                "--allow-gpio-tail), master_timeline only with --accept-count-fail")
        notes.append("count_check FAIL accepted by the caller")
    elif cc != "PASS":
        notes.append("count_check has not been run for this recording")

    # ---- EXTRACT frame table (one row per raw frame from extract_1p.m, or the minimal
    # contract table of valid frames only)
    ft = None
    if frame_table is not None:
        ft = frame_table[(frame_table["recording_id"] == recording_id).to_numpy()]
        if ft.empty:
            notes.append("frame table has no rows for this recording")
            ft = None
    t_master, dropped_ts, t_source = recording_frame_times(recording_id, rec, timestamps_path,
                                                           frame_table)
    period_update = {}
    if t_source == "period":
        notes.append("t_master from isx_frame * period_s (no timestamps)")
        if str(rec.get("period_source", "")) == "isx_footer":
            # A raw IDAS footer stores the NOMINAL period (50 ms on the 2024 rig, true 49.96 ms):
            # times drift by ~0.08 %, i.e. ~1.3 s over a 28-min assay, so epoch labels near
            # boundaries and the Arduino fit are off. Frame indices are unaffected.
            notes.append("WARNING t_master uses the NOMINAL footer period; export the TSC "
                         "timestamps (isx_timestamps) or the GPIO CSV to get the true period")
    else:
        # The per-frame timestamps measure the true period; a raw footer only has the nominal one.
        p = mean_period(t_master, dropped_ts)
        if np.isfinite(p):
            period_update = dict(period_s=p, fps=1.0 / p, period_source="tsc")
    t_assay = t_master - assay_t0_s

    extract_row = np.full(N, -1, dtype=np.int64)
    is_dropped = dropped_ts.copy()
    if ft is not None:
        fr = ft["isx_frame"].to_numpy().astype(np.int64)
        if fr.min() < 0 or fr.max() >= N or np.unique(fr).size != fr.size:
            raise TimelineError(f"{recording_id}: frame table isx_frame values are outside "
                                "0..n_frames-1 or repeated")
        if "extract_row" in ft:
            # extract_1p.m writes it (1-based row of temporal_*, empty when not imaged)
            rows = pd.to_numeric(ft["extract_row"], errors="coerce").to_numpy(float)
        else:
            # minimal table: row k (1-based, MATLAB order) of the day's trace matrix
            pos = np.flatnonzero((frame_table["recording_id"] == recording_id).to_numpy())
            rows = (pos + 1).astype(float)
        if "valid" in ft:
            valid_ft = pd.to_numeric(ft["valid"], errors="coerce").to_numpy() == 1
            rows = np.where(valid_ft, rows, np.nan)
            if t_source == "tsc":
                # The frame table and the timestamps must agree on which frames exist: a
                # mismatch means one of them belongs to another recording (or the _pp table).
                reason = ft["invalid_reason"].fillna("").astype(str).to_numpy() \
                    if "invalid_reason" in ft else np.full(len(ft), "")
                bad = (valid_ft & dropped_ts[fr]) | ((reason == "dropped") & ~dropped_ts[fr])
                if bad.any():
                    raise TimelineError(
                        f"{recording_id}: frame table and timestamps disagree on {int(bad.sum())} "
                        f"frames (first isx_frame {int(fr[bad][0])}): valid in one, dropped in "
                        "the other")
        ok = np.isfinite(rows)
        extract_row[fr[ok]] = rows[ok].astype(np.int64)
        if "invalid_reason" in ft:
            reason = ft["invalid_reason"].fillna("").astype(str).to_numpy()
            # every invalid frame except an early-trimmed one counts as dropped (not imaged)
            is_dropped[fr[(reason != "") & (reason != "trimmed")]] = True
            n_trim_ft = int((reason == "trimmed").sum())
        else:
            present = np.zeros(N, bool)
            present[fr] = True
            interior = (np.arange(N) > fr.min()) & (np.arange(N) < fr.max())
            is_dropped |= interior & ~present
            n_trim_ft = int(fr.min())
        n_trim = _int_or_none(rec.get("n_trimmed_early"))
        if n_trim is None:
            rec = dict(rec, n_trimmed_early=n_trim_ft)
        elif n_trim != n_trim_ft:
            notes.append(f"frame table has {n_trim_ft} trimmed early frames, "
                         f"recordings.csv says {n_trim}")
    n_drop = _int_or_none(rec.get("n_dropped"))
    if n_drop is not None and int(is_dropped.sum()) != n_drop:
        notes.append(f"{int(is_dropped.sum())} dropped frames found, recordings.csv says {n_drop}")

    # ---- pressure: pulse table, Arduino clock fit, force per frame
    force_max = np.full(N, np.nan)
    pulses = pulse_t = None
    # Clock map priority: the fit pressure_sync used for stim_events (so both agree), else a
    # fresh fit on pressure_frames, else what recordings.csv already holds. A value left in
    # recordings.csv by an earlier run is never preferred over the current logger data.
    fit = {k: _num(rec.get(k)) for k in ("arduino_fit_a", "arduino_fit_b", "arduino_fit_ppm",
                                         "arduino_fit_resid_ms")}
    fit_source = "recordings.csv"
    if arduino_fit and np.isfinite(_num(arduino_fit.get("b"))):
        fit = dict(arduino_fit_a=_num(arduino_fit.get("a")), arduino_fit_b=_num(arduino_fit.get("b")),
                   arduino_fit_ppm=_num(arduino_fit.get("ppm")),
                   arduino_fit_resid_ms=_num(arduino_fit.get("resid_ms")))
        fit_source = "pressure_sync"
    if pressure_frames is not None and len(pressure_frames):
        pf = pressure_frames[pressure_frames["recording_id"] == recording_id]
        pulse_no = pf["pulse"].to_numpy(dtype=np.int64)
        # The table actually joined must hold pulses 1..n_ttl: a logger that started late or
        # belongs to another session would shift every logger time, and the Arduino fit hides
        # that shift in its intercept (the residual stays perfect).
        if len(pf) != n_ttl or not np.array_equal(np.sort(pulse_no), np.arange(1, n_ttl + 1)):
            raise TimelineError(
                f"{recording_id}: pressure_frames has {len(pf)} pulses "
                f"({int(pulse_no.min()) if len(pf) else '-'}..{int(pulse_no.max()) if len(pf) else '-'}) "
                f"but recordings.csv n_ttl = {n_ttl}: the logger table is from another session "
                "or the logger missed pulses, so every logger time would be shifted")
        bad = pf["isx_frame"].to_numpy() != count_to_isx_frame(pulse_no, K)
        if bad.any():
            raise TimelineError(f"{recording_id}: pressure_frames isx_frame != pulse + K in "
                                f"{int(bad.sum())} rows (K={K}); the table was written under the "
                                "superseded pulse-1+K rule - rerun pressure_sync")
        pulses, pulse_t = _pulse_lookup(pf)
        # force of the samples with ttl_count == n belongs to the frame they fall in
        isx = count_to_isx_frame(pulse_no, K)
        ok = (isx >= 0) & (isx < N)
        force_max[isx[ok]] = pf["force_max"].to_numpy(dtype=float)[ok]
        if fit_source != "pressure_sync":
            # t_arduino_s is the time of pulse n, which is the TSC time of the frame it ENDS
            end = pulse_end_frame(pulse_no, K)
            use = (end >= 0) & (end < N)
            # dropped frames have interpolated times, so they do not enter the fit
            use[use] &= ~is_dropped[end[use]]
            a, b, ppm, res = fit_arduino_to_master(pf["t_arduino_s"].to_numpy(float)[use],
                                                   t_master[end[use]])
            fit = dict(arduino_fit_a=a, arduino_fit_b=b, arduino_fit_ppm=ppm,
                       arduino_fit_resid_ms=res)
            fit_source = "pressure_frames"
    if not np.isfinite(fit["arduino_fit_b"]):
        fit_source = "none"
    notes.append(f"Arduino fit from {fit_source}")

    def ard_to_master(t):
        return fit["arduino_fit_a"] + fit["arduino_fit_b"] * t if np.isfinite(_num(t)) else np.nan

    # ---- camera maps (count_to_isx_frame, as for the logger)
    cam = camera_frames if camera_frames is not None else pd.DataFrame(columns=CAMERA_FRAMES_COLUMNS)
    cam = cam[cam["recording_id"] == recording_id]
    bad_method = set(cam["method"].dropna()) - {"hw_trigger", "fit"}
    if bad_method:
        raise TimelineError(f"{recording_id}: unknown camera_frames method(s) {bad_method}")
    for camera in ("cam1", "cam2"):
        p = pd.to_numeric(cam.loc[cam["camera"] == camera, "pulse"], errors="coerce").dropna()
        if len(p) and (p.min() < 1 or p.max() > n_ttl or p.duplicated().any()):
            raise TimelineError(f"{recording_id}: camera_frames {camera} pulses run "
                                f"{int(p.min())}..{int(p.max())} with "
                                f"{int(p.duplicated().sum())} repeats, but a camera frame must be "
                                f"one distinct pulse of 1..n_ttl = {n_ttl}")
    try:
        maps = {c: _camera_maps(cam, c, N, K, t_master) for c in ("cam1", "cam2")}
    except TimelineError as e:
        raise TimelineError(f"{recording_id}: {e}")
    # pulse -> cam frame for the stimulus view (identity rule; uses the table so a 'fit'
    # camera table is honoured)
    c1 = cam[cam["camera"] == "cam1"]
    pulse_to_cam1 = dict(zip(c1["pulse"].dropna().astype(int), c1.loc[c1["pulse"].notna(), "cam_frame"].astype(int)))
    c2 = cam[cam["camera"] == "cam2"]
    pulse_to_cam2 = dict(zip(c2["pulse"].dropna().astype(int), c2.loc[c2["pulse"].notna(), "cam_frame"].astype(int)))

    def cam_to_isx(camera, cf):
        """(isx_frame or None, uncertain, trigger time) of one camera frame."""
        v = maps.get(camera, ({}, None))[0].get(int(cf)) if np.isfinite(_num(cf)) else None
        return (None, 0, np.nan) if v is None else v

    # ---- stimulus events
    sched = schedule_trials(assay_t0_s, stim_order)
    se = stim_events if stim_events is not None else pd.DataFrame(columns=STIM_EVENTS_COLUMNS)
    se = ensure_columns(se[se["recording_id"] == recording_id].copy(), STIM_EVENTS_COLUMNS)
    for c in ("onset_cam2_frame", "video_cam_frame", "offset_isx_frame"):
        if c not in se:
            se[c] = np.nan
    se["flags"] = se["flags"].fillna("").astype(str)
    se["source"] = se["source"].fillna("").astype(str)

    def fill_pressure_row(i):
        r = se.loc[i]
        # Master times are always re-derived from the Arduino times with the ONE clock map
        # chosen above, so stim_events and recordings.csv agree even when pressure_sync ran
        # without frame times (its own map is then the Arduino period, a few hundred ppm off).
        if np.isfinite(fit["arduino_fit_b"]):
            for what in ("onset", "threshold", "peak", "offset"):
                ca, cm = f"{what}_t_arduino_s", f"{what}_t_master_s"
                if ca in se and np.isfinite(_num(r[ca])):
                    se.at[i, cm] = ard_to_master(_num(r[ca]))
            on, off = _num(se.at[i, "onset_t_master_s"]), _num(se.at[i, "offset_t_master_s"])
            if np.isfinite(on) and np.isfinite(off):
                se.at[i, "duration_s"] = off - on
            se.at[i, "match_residual_s"] = np.nan          # recomputed from the schedule below
        # `count` = ttl_count of the onset sample; it places the onset (count_to_isx_frame)
        # and names its camera frame (the camera frame of the same pulse count)
        count = None
        if pulses is not None and np.isfinite(_num(r["onset_t_arduino_s"])):
            count = arduino_time_to_pulse(_num(r["onset_t_arduino_s"]), pulses, pulse_t)
        isx = None
        if count:
            isx = count_to_isx_frame(count, K)
        elif count == 0:
            se.at[i, "flags"] = add_flag(se.at[i, "flags"], "before_imaging")
        else:
            isx = master_time_to_frame(_num(se.at[i, "onset_t_master_s"]), t_master)
            count = None if isx is None else isx - K
        given = _num(r["onset_isx_frame"])
        if np.isfinite(given) and isx is not None and int(given) != isx:
            # both sides apply count_to_isx_frame to the same sample, so a difference means a
            # stim_events.csv from another run or from the superseded count-1+K rule
            raise TimelineError(f"{recording_id}: {r['event_id']} has onset_isx_frame "
                                f"{int(given)} but its onset sample (ttl_count {count}) is in "
                                f"isx_frame {isx}; stim_events.csv does not belong to these "
                                "pressure_frames or was written under the superseded rule - "
                                "rerun pressure_sync")
        if isx is not None and not 0 <= isx < N:
            se.at[i, "flags"] = add_flag(se.at[i, "flags"],
                                         "before_imaging" if isx < 0 else "after_imaging")
            se.at[i, "onset_isx_frame"] = np.nan
        pulse = count
        if isx is not None and 0 <= isx < N:
            se.at[i, "onset_isx_frame"] = isx
            # camera_frames.csv is the authority on cam frame <-> pulse: pressure_sync can
            # only assume the identity, which a camera with lost frames breaks
            if pulse_to_cam1:
                cam1 = pulse_to_cam1.get(pulse, np.nan)
                if np.isfinite(_num(r["onset_cam_frame"])) and _num(r["onset_cam_frame"]) != _num(cam1):
                    changed_cam[0] += 1
                se.at[i, "onset_cam_frame"] = cam1
            se.at[i, "onset_cam2_frame"] = pulse_to_cam2.get(pulse, np.nan)
        # offset frame for stim_on
        off = None
        if pulses is not None and "offset_t_arduino_s" in se and np.isfinite(_num(r.get("offset_t_arduino_s"))):
            p = arduino_time_to_pulse(_num(r["offset_t_arduino_s"]), pulses, pulse_t)
            off = min(count_to_isx_frame(p, K), N - 1) if p else None
        elif np.isfinite(_num(se.at[i, "offset_t_master_s"])):
            off = master_time_to_frame(_num(se.at[i, "offset_t_master_s"]), t_master)
        elif np.isfinite(_num(r["duration_s"])) and np.isfinite(_num(se.at[i, "onset_t_master_s"])):
            off = master_time_to_frame(_num(se.at[i, "onset_t_master_s"]) + _num(r["duration_s"]), t_master)
        if off is not None:
            se.at[i, "offset_isx_frame"] = off
        if not np.isfinite(_num(r["duration_s"])) and np.isfinite(_num(se.at[i, "offset_t_master_s"])):
            se.at[i, "duration_s"] = _num(se.at[i, "offset_t_master_s"]) - _num(se.at[i, "onset_t_master_s"])

    changed_cam = [0]
    for i in se.index:
        if se.at[i, "source"] != "video" and (np.isfinite(_num(se.at[i, "onset_t_arduino_s"]))
                                              or np.isfinite(_num(se.at[i, "onset_t_master_s"]))):
            fill_pressure_row(i)
    if changed_cam[0]:
        notes.append(f"{changed_cam[0]} pressure onset_cam_frame values replaced from "
                     "camera_frames.csv (camera not 1:1 with pulses)")

    def nearest_trial(t):
        if not np.isfinite(t):
            return None
        d = np.abs(sched["t_expected_master_s"].to_numpy() - t)
        j = int(np.argmin(d))
        return j if d[j] <= MATCH_TOLERANCE_S else None

    def slot_block_at(t):
        """(slot, block) of a master time inside a stimulus slot, else (None, None): an
        unmatched delivery still belongs to the slot frames.csv puts it in."""
        if not np.isfinite(t):
            return None, None
        c = classify_assay_times([t - assay_t0_s])
        return (int(c["slot"][0]), int(c["block"][0])) if c["slot"][0] > 0 else (None, None)

    def trial_key(r):
        """(slot, block, trial), or None for a row not assigned to one scheduled trial
        ('extra' rows have a slot and block but no trial)."""
        k = [_num(r[c]) for c in ("slot", "block", "trial")]
        return tuple(int(x) for x in k) if all(np.isfinite(x) for x in k) else None

    def as_extra(t, flags):
        """Flags + slot/block/stim_type for a delivery that gets no scheduled trial."""
        slot, block = slot_block_at(t)
        flags = add_flag(flags, "extra")
        if slot is None:
            return add_flag(flags, "outside_slot"), {}
        return flags, dict(slot=slot, block=block, stim_type=stim_order[slot - 1])

    # Event ids (CONTRACTS.md 1a): a scheduled trial's id is its schedule position 001-060
    # whatever the inputs were; extras are numbered from 061 up.
    used_ids = set(se["event_id"].dropna().astype(str))
    extra_no = [max(len(sched) + 1, _next_event_number(list(used_ids)))]

    def new_id(j=None):
        eid = f"{recording_id}_e{j + 1:03d}" if j is not None else None
        if eid is None or eid in used_ids:
            eid = f"{recording_id}_e{extra_no[0]:03d}"
            extra_no[0] += 1
        used_ids.add(eid)
        return eid

    # schedule assignment for detected rows that pressure_sync left unassigned
    held = {trial_key(r) for _, r in se.iterrows()} - {None}
    for i in se.index:
        # pressure_sync already decided for its rows: 'extra' means no trial (an outside_slot
        # extra has no slot either); giving it the nearest trial would put two rows on one trial
        decided = "extra" in str(se.at[i, "flags"]).split(";")
        t_i = _num(se.at[i, "onset_t_master_s"])
        if decided or np.isfinite(_num(se.at[i, "slot"])) or not np.isfinite(t_i):
            continue
        j = nearest_trial(t_i)
        key = None if j is None else tuple(int(sched.at[j, c]) for c in ("slot", "block", "trial"))
        if key is None or key in held:
            se.at[i, "flags"], upd = as_extra(t_i, se.at[i, "flags"])
            for c, val in upd.items():
                se.at[i, c] = val
        else:
            for c in ("slot", "block", "trial", "stim_type"):
                se.at[i, c] = sched.at[j, c]
            held.add(key)

    # video deliveries: heat (and anything the logger missed) comes from the scored taps
    new_rows = []
    typed_by_slot = {}           # slot -> Counter of the typed stimulus names scored there
    if video_deliveries is not None and len(video_deliveries):
        vd = video_deliveries[video_deliveries["recording_id"] == recording_id]
        taken = set()
        for _, v in vd.sort_values("cam_frame").iterrows():
            camera = v["camera"] if isinstance(v.get("camera"), str) else "cam1"
            # the scored frame's snapshot time (its trigger pulse), placed by count_to_isx_frame
            isx, unc, t_trig = cam_to_isx(camera, v["cam_frame"])
            t_on = t_trig if isx is not None else np.nan
            j = nearest_trial(t_on)
            typed = v["stim_type"] if isinstance(v.get("stim_type"), str) else ""
            flags = v["flags"] if isinstance(v.get("flags"), str) else ""
            if unc:
                flags = add_flag(flags, "cam_uncertain")
            if j is None:
                key, same = None, se.iloc[0:0]
            else:
                key = (int(sched.at[j, "slot"]), int(sched.at[j, "block"]), int(sched.at[j, "trial"]))
                same = se[(se["slot"] == key[0]) & (se["block"] == key[1]) & (se["trial"] == key[2])]
            slot_here = key[0] if key else slot_block_at(t_on)[0]
            if typed and slot_here:
                typed_by_slot.setdefault(slot_here, Counter())[typed] += 1
            hit = same[same["source"] == "pressure"]
            if not hit.empty:
                # the logger already timed this trial; keep the scored frame for QC only
                k = hit.index[0]
                se.at[k, "video_cam_frame"] = v["cam_frame"]
                if typed and typed != se.at[k, "stim_type"]:
                    se.at[k, "flags"] = add_flag(se.at[k, "flags"], "type_mismatch")
                continue
            row = dict(recording_id=recording_id, source="video",
                       onset_cam_frame=v["cam_frame"] if camera == "cam1" else np.nan,
                       onset_cam2_frame=(maps["cam2"][1][isx] or np.nan) if isx is not None else np.nan,
                       onset_isx_frame=isx if isx is not None else np.nan,
                       onset_t_master_s=t_on)
            if key is None or key in taken:
                # Not one of the 60 trials: too far from every trial time, or a second tap for
                # a trial that already has one (a double key press?). Like pressure_sync's
                # extras it keeps slot and block but no trial, so one trial never has two rows.
                flags, upd = as_extra(t_on, flags)
                if key is not None:
                    flags = add_flag(flags, "duplicate")
                row.update(upd, trial=np.nan, t_expected_master_s=np.nan, match_residual_s=np.nan)
                row.setdefault("stim_type", typed or np.nan)
                if key is None and isx is None:
                    flags = add_flag(flags, "outside_imaging")
            else:
                row.update(slot=key[0], block=key[1], trial=key[2],
                           stim_type=sched.at[j, "stim_type"])
            if typed and isinstance(row.get("stim_type"), str) and typed != row["stim_type"]:
                flags = add_flag(flags, "type_mismatch")
            row["flags"] = flags
            # Re-running on an already merged stim_events.csv refreshes the video row of the
            # same tap; otherwise the trial's 'expected' row from pressure_sync (or its video
            # row from an earlier run) becomes this video row. Either keeps its event_id.
            prior = se[(se["source"] == "video") &
                       (pd.to_numeric(se["onset_cam_frame"], errors="coerce") == float(v["cam_frame"]))]
            if prior.empty and key is not None and key not in taken:
                prior = same[same["source"].isin(["expected", "video"])]
            if not prior.empty:
                for c, val in row.items():
                    se.at[prior.index[0], c] = val
            else:
                row["event_id"] = new_id(j if key is not None and key not in taken else None)
                new_rows.append(row)
            if key is not None:
                taken.add(key)
    if new_rows:
        se = _append_rows(se, new_rows)

    # A wrong stim_order relabels the whole stimulus axis, so it must not pass as per-row
    # type_mismatch flags: most typed taps of a slot disagreeing with it stops the join.
    disagree = []
    for s, cnt in sorted(typed_by_slot.items()):
        n_typed, want = sum(cnt.values()), stim_order[s - 1]
        n_bad = n_typed - cnt.get(want, 0)
        if n_typed >= 3 and n_bad > n_typed / 2:
            top, n_top = cnt.most_common(1)[0]
            disagree.append(f"slot {s} is '{want}' but {n_bad} of {n_typed} scored taps there "
                            f"are typed otherwise (most: '{top}' x{n_top})")
    if disagree:
        msg = (f"stim_order {','.join(stim_order)} disagrees with the scored delivery keys: "
               + "; ".join(disagree))
        if not trust_stim_order:
            raise TimelineError(f"{recording_id}: {msg}. Fix the manifest stim_order, or pass "
                                "--trust-stim-order if the typed key names are what is wrong")
        notes.append(f"WARNING {msg} (stim_order kept: --trust-stim-order)")

    # every scheduled trial gets a row; trials nobody saw become 'expected' + missing
    have = {trial_key(r) for _, r in se.iterrows()} - {None}
    add = []
    for j, s in sched.iterrows():
        key = (int(s["slot"]), int(s["block"]), int(s["trial"]))
        if key not in have:
            add.append(dict(event_id=new_id(j), recording_id=recording_id,
                            stim_type=s["stim_type"], slot=key[0], block=key[1], trial=key[2],
                            source="expected", flags="missing"))
    if add:
        se = _append_rows(se, add)
    se["source"] = se["source"].fillna("")
    se["flags"] = se["flags"].fillna("")
    # expected times from the schedule for every assigned row
    st = sched.set_index(["slot", "block", "trial"])["t_expected_master_s"]
    for i in se.index:
        key = trial_key(se.loc[i])
        if key is not None:
            se.at[i, "t_expected_master_s"] = st.loc[key]
            if se.at[i, "source"] in ("pressure", "video") and not np.isfinite(_num(se.at[i, "match_residual_s"])):
                se.at[i, "match_residual_s"] = _num(se.at[i, "onset_t_master_s"]) - st.loc[key]
    for c in ("t_expected_master_s", "onset_t_master_s", "match_residual_s"):
        se[c] = pd.to_numeric(se[c], errors="coerce").round(6)
    se["_t"] = se["onset_t_master_s"].astype(float).fillna(se["t_expected_master_s"].astype(float))
    se = se.sort_values(["_t", "event_id"]).drop(columns="_t").reset_index(drop=True)

    # ---- behaviour events -> master clock, linked to the latest stimulus onset
    be = behaviour_events if behaviour_events is not None else pd.DataFrame(columns=BEHAVIOUR_EVENTS_COLUMNS)
    be = ensure_columns(be[be["recording_id"] == recording_id].copy(), BEHAVIOUR_EVENTS_COLUMNS)
    if "flags" not in be:
        be["flags"] = ""
    be["flags"] = be["flags"].fillna("").astype(str)
    unknown = set(be["type"].dropna()) - set(BEHAVIOUR_TYPES)
    if unknown:
        raise TimelineError(f"{recording_id}: unknown behaviour type(s) {unknown}")
    # A 'duplicate' video tap (second key press for a trial) is not a stimulus: linking to it
    # would shorten the latency of the response to the real delivery.
    is_dup = se["flags"].apply(lambda f: "duplicate" in str(f).split(";"))
    onsets = se[se["source"].isin(["pressure", "video"]) & se["onset_t_master_s"].notna() & ~is_dup]
    on_t = onsets["onset_t_master_s"].to_numpy(dtype=float)
    order = np.argsort(on_t)
    on_t, on_id = on_t[order], onsets["event_id"].to_numpy()[order]
    for i in be.index:
        camera = be.at[i, "camera"] if isinstance(be.at[i, "camera"], str) else "cam1"
        for side in ("on", "off"):
            cf = be.at[i, f"{side}_cam_frame"]
            if camera not in maps or not maps[camera][0]:
                be.at[i, "flags"] = add_flag(be.at[i, "flags"], "no_camera_table")
                continue
            if np.isfinite(_num(cf)) and int(cf) not in maps[camera][0]:
                be.at[i, "flags"] = add_flag(be.at[i, "flags"], "outside_video")
                continue
            isx, unc, t_trig = cam_to_isx(camera, cf)
            if unc:
                be.at[i, "flags"] = add_flag(be.at[i, "flags"], "cam_uncertain")
            if isx is None:
                if np.isfinite(_num(cf)):
                    be.at[i, "flags"] = add_flag(be.at[i, "flags"], "outside_imaging")
                continue
            # isx_frame = count_to_isx_frame(pulse); the time is the frame's snapshot (trigger
            # pulse) time, so a latency is measured from the contact to the image showing it
            be.at[i, f"{side}_isx_frame"] = isx
            be.at[i, f"{side}_t_master_s"] = t_trig
        if be.at[i, "type"] != "uncertain" and on_t.size and np.isfinite(_num(be.at[i, "on_t_master_s"])):
            t = _num(be.at[i, "on_t_master_s"])
            k = int(np.searchsorted(on_t, t, side="right")) - 1
            if k >= 0 and t - on_t[k] <= LINK_WINDOW_S:
                be.at[i, "event_id"] = on_id[k]
                be.at[i, "latency_s"] = t - on_t[k]

    for c in ("on_t_master_s", "off_t_master_s", "latency_s"):
        be[c] = pd.to_numeric(be[c], errors="coerce").round(6)

    # ---- frames.csv
    cls = classify_assay_times(t_assay)
    stim_type = np.array([stim_order[s - 1] if s > 0 else "" for s in cls["slot"]], dtype=object)
    stim_on = np.zeros(N, dtype=np.int8)
    ev_id = np.full(N, "", dtype=object)
    for _, r in se.iterrows():
        a = _num(r["onset_isx_frame"])
        if not np.isfinite(a) or "duplicate" in str(r["flags"]).split(";"):
            continue
        b = _num(r.get("offset_isx_frame"))
        a = int(a)
        b = int(b) if np.isfinite(b) and int(b) >= a else a      # no offset: onset frame only
        stim_on[a:b + 1] = 1
        ev_id[a:b + 1] = r["event_id"]
    affect = np.full(N, "", dtype=object)
    reflex = np.full(N, "", dtype=object)
    for _, r in be.iterrows():
        a, b = _num(r["on_isx_frame"]), _num(r["off_isx_frame"])
        if not np.isfinite(a):
            continue
        a = int(a)
        b = int(b) if np.isfinite(b) else a
        if r["type"] in AFFECTIVE_TYPES:
            affect[a:b + 1] = r["type"]
        elif r["type"] in REFLEX_TYPES:
            reflex[a] = r["type"] if not reflex[a] else ";".join(sorted({*reflex[a].split(";"), r["type"]}))
    motion_on = {"cam1": np.full(N, np.nan), "cam2": np.full(N, np.nan)}
    if motion is not None and len(motion):
        # motion energy is per camera frame; put it on the isx frame that frame overlaps
        mo = motion[motion["recording_id"] == recording_id]
        for camera in ("cam1", "cam2"):
            m = mo[mo["camera"] == camera]
            if m.empty:
                continue
            lut = pd.Series(m["motion"].to_numpy(float), index=m["cam_frame"].astype(int))
            inv = maps[camera][1]
            has = inv > 0
            motion_on[camera][has] = lut.reindex(inv[has]).to_numpy()
    frames = pd.DataFrame({
        "recording_id": recording_id,
        "isx_frame": np.arange(N),
        "t_master_s": np.round(t_master, 6),
        "t_assay_s": np.round(t_assay, 6),
        "is_dropped": is_dropped.astype(np.int8),
        "extract_row": pd.array(np.where(extract_row >= 0, extract_row, 0), dtype="Int64"),
        "epoch": cls["epoch"],
        "slot": pd.array(cls["slot"], dtype="Int64"),
        "stim_type": stim_type,
        "block": pd.array(cls["block"], dtype="Int64"),
        "trial": pd.array(cls["trial"], dtype="Int64"),
        "stim_on": stim_on,
        "event_id": ev_id,
        "force_max": force_max,
        "cam1_frame": pd.array(maps["cam1"][1], dtype="Int64"),
        "cam2_frame": pd.array(maps["cam2"][1], dtype="Int64"),
        "affective_state": affect,
        "reflex": reflex,
        "motion_cam1": motion_on["cam1"],
        "motion_cam2": motion_on["cam2"],
    })
    frames.loc[extract_row < 0, "extract_row"] = pd.NA
    for c in ("slot", "block", "trial"):
        frames.loc[frames[c] == 0, c] = pd.NA
    for c in ("cam1_frame", "cam2_frame"):
        frames.loc[frames[c] == 0, c] = pd.NA

    rec_out = dict(rec)
    rec_out.update({k: v for k, v in fit.items() if np.isfinite(_num(v))})
    rec_out.update(period_update)
    rec_out["t_source"] = t_source
    se = _int_columns(se, ("slot", "block", "trial", "onset_isx_frame", "onset_cam_frame",
                           "onset_cam2_frame", "video_cam_frame", "offset_isx_frame"))
    be = _int_columns(be, ("on_cam_frame", "off_cam_frame", "on_isx_frame", "off_isx_frame"))
    return dict(frames=frames, stim_events=se, behaviour_events=be, recording=rec_out, notes=notes)


def _append_rows(df, rows):
    """df plus a list of row dicts (an empty df is replaced, which pandas' concat warns about)."""
    new = ensure_columns(pd.DataFrame(rows), list(df.columns))
    return new.reset_index(drop=True) if df.empty else pd.concat([df, new], ignore_index=True)


def _int_columns(df, cols):
    """Frame numbers and trial indices as nullable integers, so the CSV says 6226, not 6226.0."""
    df = df.copy()
    for c in cols:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce")
            df[c] = v.round().astype("Int64")
    return df


def _summary_fit(path, recording_id):
    """The clock map pressure_sync used, if it was fitted on frame times for this recording."""
    if not path or not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    if s.get("recording_id") != recording_id:
        return None          # a summary left by another recording in a shared folder
    if s.get("t_source") != "frame_times":
        return None          # a fit on the Arduino period alone is not the master clock
    return dict(a=s.get("arduino_fit_a"), b=s.get("arduino_fit_b"), ppm=s.get("arduino_fit_ppm"),
                resid_ms=s.get("arduino_fit_resid_ms"))


def run(recording_id, recordings_csv, out_dir, stim_order, assay_t0_s, timestamps=None,
        pressure_frames=None, stim_events=None, camera_frames=None, behaviour_events=None,
        video_deliveries=None, frame_table=None, pressure_summary=None, motion=None,
        accept_count_fail=False, trust_stim_order=False):
    """File-level wrapper: read the tables, build, write outputs into out_dir.

    pressure_summary: pressure_sync's pressure_summary.json (its Arduino fit is used when
    recordings.csv has none). motion: optional CSV recording_id, camera, cam_frame, motion.
    out_dir holds one recording: frames.csv and stim_events.csv of another recording there
    are never overwritten (CONTRACTS.md 3)."""
    for name in ("frames.csv", "stim_events.csv"):
        others = other_recordings(os.path.join(out_dir, name), recording_id)
        if others:
            raise TimelineError(f"{os.path.join(out_dir, name)} belongs to {others}, not "
                                f"{recording_id}: use one output folder per recording")
    recs = read_csv(recordings_csv, ["recording_id"], "recordings.csv")
    sel = recs[recs["recording_id"] == recording_id]
    if sel.empty:
        raise TimelineError(f"{recording_id} not in {recordings_csv}")
    rec = sel.iloc[0].to_dict()

    def opt(path, cols, what):
        return read_csv(path, cols, what) if path else None

    res = build_timeline(
        recording_id, rec, stim_order, assay_t0_s,
        pressure_frames=opt(pressure_frames, PRESSURE_FRAMES_COLUMNS, "pressure_frames.csv"),
        stim_events=opt(stim_events, ["event_id", "recording_id", "source"], "stim_events.csv"),
        camera_frames=opt(camera_frames, CAMERA_FRAMES_COLUMNS, "camera_frames.csv"),
        behaviour_events=opt(behaviour_events, ["recording_id", "camera", "type", "on_cam_frame",
                                                "off_cam_frame"], "behaviour_events.csv"),
        video_deliveries=opt(video_deliveries, ["recording_id", "cam_frame"], "video_deliveries.csv"),
        timestamps_path=timestamps,
        frame_table=opt(frame_table, ["recording_id", "isx_frame"], "EXTRACT frame table"),
        motion=opt(motion, ["recording_id", "camera", "cam_frame", "motion"], "motion table"),
        arduino_fit=_summary_fit(pressure_summary, recording_id),
        accept_count_fail=accept_count_fail, trust_stim_order=trust_stim_order)
    os.makedirs(out_dir, exist_ok=True)
    frames_path = os.path.join(out_dir, "frames.csv")
    res["frames"].to_csv(frames_path, index=False)
    replace_rows(os.path.join(out_dir, "stim_events.csv"), res["stim_events"], recording_id,
                 STIM_EVENTS_COLUMNS)
    replace_rows(os.path.join(out_dir, "behaviour_events.csv"), res["behaviour_events"],
                 recording_id, BEHAVIOUR_EVENTS_COLUMNS)
    rec_out = res["recording"]
    # notes hold one ' | '-separated segment per module; replace only this module's segment
    prev = rec_out.get("notes") if isinstance(rec_out.get("notes"), str) else ""
    keep = [p for p in prev.split(" | ") if p.strip() and not p.startswith("master_timeline:")]
    rec_out["notes"] = " | ".join(keep + ["master_timeline: " + "; ".join(res["notes"] or ["ok"])])
    replace_rows(os.path.join(out_dir, "recordings.csv"), pd.DataFrame([rec_out]), recording_id,
                 RECORDINGS_COLUMNS)
    res["paths"] = dict(frames=frames_path)
    return res


# ---------------------------------------------------------------- selftest
def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    # schedule
    s = schedule_trials(10.0, ["mild", "heat", "light", "pin"])
    check(len(s) == 60, "60 scheduled trials")
    check(s.iloc[0]["t_expected_master_s"] == 310.0, "first trial at 300 s + t0")
    r = s[(s.slot == 2) & (s.block == 1) & (s.trial == 2)].iloc[0]
    check(r.t_expected_assay_s == 675.0 and r.stim_type == "heat", "slot 2 block 1 trial 2")
    check(s.iloc[-1]["t_expected_assay_s"] == 1380 + 240 + 30, "last trial at 1650 s")
    c = classify_assay_times([-1, 0, 299.99, 300, 314.99, 315, 345, 359.99, 360, 600, 659.9,
                              660, 1679.9, 1680])
    check(list(c["epoch"]) == ["pre_baseline", "baseline", "baseline", "stim", "stim", "stim",
                               "stim", "stim", "stim", "rest", "rest", "stim", "stim", "post_end"],
          f"epochs {list(c['epoch'])}")
    check(list(c["trial"][3:9]) == [1, 1, 2, 3, 3, 1], f"trials {list(c['trial'][3:9])}")
    check(list(c["block"][3:9]) == [1, 1, 1, 1, 1, 2], "blocks")
    check(c["slot"][11] == 2 and c["slot"][12] == 4, "slots")

    with tempfile.TemporaryDirectory() as d:
        # timestamps in TSC microseconds with one dropped frame (row absent)
        n, per = 1720, 1.0
        idx = np.arange(n)
        ts = 5e9 + idx * per * 1e6
        keep = idx != 700
        pd.DataFrame({"Global Frame Number": idx[keep], "Movie Number": 0,
                      "Local Frame Number": idx[keep],
                      "Frame Timestamp (us)": ts[keep]}).to_csv(os.path.join(d, "ts.csv"), index=False)
        t, drop, src = frame_times(n, None, os.path.join(d, "ts.csv"))
        check(src == "tsc" and drop.sum() == 1 and drop[700], "dropped frame from timestamps")
        check(abs(t[700] - 700.0) < 1e-9 and t[0] == 0.0, "dropped frame time interpolated")

        a, b, ppm, res = fit_arduino_to_master(np.arange(100) * 1.0003 + 3.0, np.arange(100) * 1.0)
        check(abs(ppm - ((1 / 1.0003) - 1) * 1e6) < 1 and res < 1e-6, "Arduino fit")

        K, t0 = 0, 10.0
        rid = "T01_d1_pre"
        rec = dict(recording_id=rid, n_frames_isx=n, n_dropped=1, n_trimmed_early=1,
                   period_s=per, fps=1 / per, n_ttl=expected_n_ttl(n, K), pulse_to_frame_offset=K,
                   cam_lead_frames=1)
        pulse = np.arange(1, n + 1)
        # Arduino clock runs fast and starts at 50 s; pulse n at the TSC time of the frame it
        # ends (n-1+K), as CONTRACTS.md 1a says
        pt = 50.0 + (pulse - 1 + K) * per * 1.0002 + 0.01
        pf = pd.DataFrame(dict(recording_id=rid, pulse=pulse, isx_frame=count_to_isx_frame(pulse, K),
                               t_arduino_s=pt, force_mean=0.0, force_max=0.0, n_samples=1))
        pf.loc[pf.pulse == 311, "force_max"] = 500.0
        # contact 0.3 s after pulse 311 (master 310.3 s, slot 1 block 1 trial 1 = 300 s + t0):
        # its samples have ttl_count 311, so it lies in isx_frame 311 + K
        on_ard = pt[310] + 0.3
        se = pd.DataFrame([dict(event_id=f"{rid}_e001", recording_id=rid, stim_type="mild",
                                source="pressure", onset_t_arduino_s=on_ard,
                                offset_t_arduino_s=pt[312] + 0.2, flags=""),
                           # an unmatched contact: pressure_sync gives it a slot and block only
                           dict(event_id=f"{rid}_e061", recording_id=rid, stim_type="mild",
                                source="pressure", slot=1, block=2, onset_t_arduino_s=pt[380] + 0.3,
                                flags="extra")])
        cam = pd.concat([pd.DataFrame(dict(recording_id=rid, camera=c, cam_frame=pulse,
                                           pulse=pulse, isx_frame=count_to_isx_frame(pulse, K),
                                           t_master_s=np.nan, method="hw_trigger", gap_before=0))
                         for c in ("cam1", "cam2")])
        be = pd.DataFrame([dict(recording_id=rid, camera="cam1", type="withdrawal",
                                on_cam_frame=314, off_cam_frame=314, scorer="x"),
                           dict(recording_id=rid, camera="cam1", type="licking_biting",
                                on_cam_frame=320, off_cam_frame=325, scorer="x")])
        # heat is slot 2 (stim_order below): video tap at the frame of 660 s + t0
        vd = pd.DataFrame([dict(recording_id=rid, camera="cam1", cam_frame=670, key=2,
                                stim_type="heat", flags="")])
        ft = pd.DataFrame(dict(recording_id=["X_d1_pre"] * 5 + [rid] * (n - 2),
                               isx_frame=list(range(5)) + [i for i in range(1, n) if i != 700]))
        out = build_timeline(rid, rec, ["mild", "heat", "light", "pin"], t0, pf, se, cam, be,
                             vd, os.path.join(d, "ts.csv"), ft)
        st = out["stim_events"]
        e1 = st[st.event_id == f"{rid}_e001"].iloc[0]
        check(int(e1.onset_isx_frame) == 311 + K, f"pressure onset isx {e1.onset_isx_frame} != count 311 + K")
        check(int(e1.onset_cam_frame) == 311, f"pressure onset cam {e1.onset_cam_frame} != count 311")
        check(int(e1.slot) == 1 and int(e1.block) == 1 and int(e1.trial) == 1, "slot assignment")
        check(abs(e1.onset_t_master_s - (310 + 0.3 / 1.0002)) < 1e-6, "onset_t_master_s from fit")
        check(master_time_to_frame(e1.onset_t_master_s, out["frames"]["t_master_s"].to_numpy()) ==
              int(e1.onset_isx_frame), "master_time_to_frame agrees with count_to_isx_frame")
        hv = st[st.source == "video"]
        check(len(hv) == 1 and int(hv.iloc[0].onset_isx_frame) == count_to_isx_frame(670, K) and
              int(hv.iloc[0].slot) == 2 and abs(hv.iloc[0].onset_t_master_s - 669.0) < 1e-6,
              "heat video delivery: isx = cam + K, time = its trigger pulse, slot 2")
        check(len(st) == 61 and (st.source == "expected").sum() == 58, "missing trials are expected rows")
        ex = st[st.event_id == f"{rid}_e061"].iloc[0]
        check(int(ex.onset_isx_frame) == 381 + K and pd.isna(ex.trial) and "extra" in ex["flags"],
              "extra contact kept, placed, not given a trial")
        bev = out["behaviour_events"]
        w = bev[bev.type == "withdrawal"].iloc[0]
        check(int(w.on_isx_frame) == count_to_isx_frame(314, K) and abs(w.on_t_master_s - 313.0) < 1e-6,
              "withdrawal on_isx = cam + K, on_t = trigger pulse time")
        check(w.event_id == f"{rid}_e001" and 0 < w.latency_s < 5, "withdrawal linked to e001")
        fr = out["frames"]
        check(len(fr) == n and fr.loc[311, "stim_on"] == 1 and fr.loc[313, "stim_on"] == 1 and
              fr.loc[314, "stim_on"] == 0 and fr.loc[310, "stim_on"] == 0, "stim_on span")
        check(fr.loc[311, "event_id"] == f"{rid}_e001", "event_id on stim frames")
        check(fr.loc[311, "force_max"] == 500.0, "force_max of count 311 on isx_frame 311 + K")
        check(int(fr.loc[314, "cam1_frame"]) == 314 - K, "cam1_frame = isx_frame - K")
        check(pd.isna(fr.loc[0, "cam1_frame"]), "isx frame 0 has no cam frame (K=0)")
        # tables written under the superseded pulse-1+K rule must be refused, not shifted
        for what, args in (("pressure_frames", (pf.assign(isx_frame=pulse - 1 + K), se, cam)),
                           ("camera_frames", (pf, se, cam.assign(isx_frame=cam.pulse - 1 + K))),
                           ("stim_events", (pf, se.assign(onset_isx_frame=[310 + K, np.nan]), cam))):
            try:
                build_timeline(rid, rec, ["mild", "heat", "light", "pin"], t0, *args)
                check(False, f"old-rule {what} must raise")
            except TimelineError:
                pass
        check(master_time_to_frame(-0.5, np.arange(5.0)) == 0 and
              master_time_to_frame(-1.5, np.arange(5.0)) is None and
              master_time_to_frame(2.0, np.arange(5.0)) == 2 and
              master_time_to_frame(2.01, np.arange(5.0)) == 3 and
              master_time_to_frame(4.5, np.arange(5.0)) is None,
              "master_time_to_frame: frame k spans (t[k-1], t[k]]")
        check(fr.loc[314, "reflex"] == "withdrawal" and fr.loc[322, "affective_state"] == "licking_biting",
              "behaviour on frames")
        check(fr.loc[700, "is_dropped"] == 1 and pd.isna(fr.loc[700, "extract_row"]), "dropped frame")
        check(pd.isna(fr.loc[0, "extract_row"]) and int(fr.loc[1, "extract_row"]) == 6,
              "extract_row is the 1-based (MATLAB) row of the day table")
        check(fr.loc[int(t0 + 300), "epoch"] == "stim" and fr.loc[5, "epoch"] == "pre_baseline",
              "epochs on frames")
        bad = dict(rec, n_ttl=n + 3)
        try:
            build_timeline(rid, bad, ["mild", "heat", "light", "pin"], t0, pf, se, cam, be)
            check(False, "n_ttl mismatch must raise")
        except TimelineError:
            pass
        try:
            build_timeline(rid, dict(rec, pulse_to_frame_offset=np.nan),
                           ["mild", "heat", "light", "pin"], t0, pf, se, cam, be)
            check(False, "empty K must raise")
        except TimelineError:
            pass

        order = ["mild", "heat", "light", "pin"]
        ts_path = os.path.join(d, "ts.csv")

        def raises(what, **kw):
            args = dict(recording_id=rid, rec=rec, stim_order=order, assay_t0_s=t0,
                        pressure_frames=pf, stim_events=se, camera_frames=cam)
            args.update(kw)
            try:
                build_timeline(**args)
                check(False, f"{what} must raise")
            except TimelineError:
                pass

        # the join must check the tables it is actually given, and the count rule itself
        raises("pressure_frames of a logger that started 3 pulses late",
               pressure_frames=pf[pf.pulse > 3].assign(pulse=lambda x: x.pulse - 3,
                                                       isx_frame=lambda x: x.pulse - 3 + K))
        raises("empty n_ttl in recordings.csv", rec=dict(rec, n_ttl=np.nan))
        raises("count_check FAIL", rec=dict(rec, count_check="FAIL",
                                            notes="count_check: cam2 video not available"))
        ok = build_timeline(rid, dict(rec, count_check="FAIL"), order, t0, pf, se, cam,
                            accept_count_fail=True)
        check(any("FAIL accepted" in x for x in ok["notes"]), "accepted count_check FAIL is noted")
        raises("camera pulses beyond n_ttl", camera_frames=cam.assign(pulse=cam.pulse + 2,
                                                                      isx_frame=cam.pulse + 2 + K))
        # a timestamp table that stops early must not turn the rest of the movie into
        # 'dropped' frames with extrapolated times
        pd.read_csv(ts_path).iloc[:1000].to_csv(os.path.join(d, "ts_cut.csv"), index=False)
        raises("a cut timestamp table", timestamps_path=os.path.join(d, "ts_cut.csv"))
        ft_bad = pd.DataFrame(dict(recording_id=rid, isx_frame=np.arange(n), valid=1,
                                   invalid_reason=""))
        raises("frame table valid where timestamps say dropped", timestamps_path=ts_path,
               frame_table=ft_bad)

        # video deliveries: a double key press, a tap 45 s into a block, and no stim_events
        vd2 = pd.DataFrame([dict(recording_id=rid, camera="cam1", cam_frame=c, key=2,
                                 stim_type="heat", flags="") for c in (670, 672, 716)])
        be2 = pd.DataFrame([dict(recording_id=rid, camera="cam1", type="withdrawal",
                                 on_cam_frame=676, off_cam_frame=676, scorer="x")])
        o2 = build_timeline(rid, rec, order, t0, pf, se, cam, be2, vd2, ts_path)
        s2 = o2["stim_events"]
        first = s2[(s2.source == "video") & (s2.onset_cam_frame == 670)].iloc[0]
        dup = s2[(s2.source == "video") & (s2.onset_cam_frame == 672)].iloc[0]
        late = s2[(s2.source == "video") & (s2.onset_cam_frame == 716)].iloc[0]
        check(first.event_id == f"{rid}_e016" and int(first.trial) == 1, "first tap = trial e016")
        check(pd.isna(dup.trial) and int(dup.slot) == 2 and int(dup.block) == 1 and
              "duplicate" in dup["flags"] and "extra" in dup["flags"] and pd.isna(dup.t_expected_master_s)
              and dup.event_id == f"{rid}_e062", f"double tap: extra;duplicate, no trial ({dup['flags']})")
        check(((s2.slot == 2) & (s2.block == 1) & (s2.trial == 1)).sum() == 1, "one row per trial")
        check(int(late.slot) == 2 and int(late.block) == 1 and pd.isna(late.trial) and
              "extra" in late["flags"] and "outside_slot" not in late["flags"],
              f"tap inside a slot but far from a trial: extra with slot/block ({late['flags']})")
        w2 = o2["behaviour_events"].iloc[0]
        check(w2.event_id == f"{rid}_e016" and abs(w2.latency_s - 6.0) < 1e-6,
              f"withdrawal linked to the first tap, not the duplicate ({w2.event_id}, {w2.latency_s})")
        check(o2["frames"].loc[672, "event_id"] == "", "a duplicate tap does not mark stim_on")
        o3 = build_timeline(rid, rec, order, t0, pf, None, cam, None, vd2.iloc[:1], ts_path)
        s3 = o3["stim_events"].set_index("event_id")
        check(s3.loc[f"{rid}_e016", "source"] == "video" and s3.loc[f"{rid}_e001", "trial"] == 1
              and s3.loc[f"{rid}_e002", "trial"] == 2 and len(s3) == 60,
              "without stim_events: ids 001-060 follow the schedule")

        # a wrong stim_order: the taps typed 'mild' in the slot stim_order calls heat
        vd4 = pd.DataFrame([dict(recording_id=rid, camera="cam1", cam_frame=c, key=1,
                                 stim_type="mild", flags="") for c in (671, 686, 701, 731)])
        raises("taps typed 'mild' in the 'heat' slot", video_deliveries=vd4,
               timestamps_path=ts_path)
        o4 = build_timeline(rid, rec, order, t0, pf, se, cam, None, vd4, ts_path,
                            trust_stim_order=True)
        check(any("disagrees" in x for x in o4["notes"]), "--trust-stim-order keeps a warning")

        # one folder per recording: another recording's frames.csv is never overwritten
        out_d = os.path.join(d, "out")
        os.makedirs(out_d)
        pd.DataFrame(dict(recording_id=["X_d1_pre"], isx_frame=[0])).to_csv(
            os.path.join(out_d, "frames.csv"), index=False)
        pd.DataFrame([rec]).to_csv(os.path.join(d, "recordings.csv"), index=False)
        try:
            run(rid, os.path.join(d, "recordings.csv"), out_d, order, t0)
            check(False, "frames.csv of another recording must not be overwritten")
        except TimelineError:
            pass

    if failures:
        print("master_timeline selftest FAILED:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("master_timeline selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recording-id")
    ap.add_argument("--manifest", help="manifest.csv (for stim_order and assay_t0_s)")
    ap.add_argument("--stim-order", help="override, e.g. light,mild,heat,pin")
    ap.add_argument("--assay-t0", type=float, help="override assay_t0_s")
    ap.add_argument("--recordings", help="recordings.csv")
    ap.add_argument("--timestamps", help="Inscopix frame timestamp CSV (optional)")
    ap.add_argument("--pressure-frames")
    ap.add_argument("--stim-events")
    ap.add_argument("--camera-frames")
    ap.add_argument("--behaviour-events")
    ap.add_argument("--video-deliveries")
    ap.add_argument("--frame-table", help="EXTRACT <mouse>_d<day>_frame_table.csv")
    ap.add_argument("--pressure-summary", help="pressure_sync's pressure_summary.json")
    ap.add_argument("--motion", help="optional CSV recording_id,camera,cam_frame,motion")
    ap.add_argument("--accept-count-fail", action="store_true",
                    help="join although recordings.csv says count_check FAIL (you have checked "
                         "that the reason is harmless, e.g. a camera table rebuilt from anchors)")
    ap.add_argument("--trust-stim-order", action="store_true",
                    help="keep the manifest stim_order although most scored delivery keys of a "
                         "slot disagree with it (the typed key names are wrong)")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.recording_id and args.recordings and args.out_dir):
        ap.error("--recording-id, --recordings and --out-dir are required")
    stim_order, t0 = None, args.assay_t0
    if args.manifest:
        import manifest as mf
        row = mf.load_manifest_strict(args.manifest).row(args.recording_id)
        stim_order, t0 = row["stim_order"], (row["assay_t0_s"] if t0 is None else t0)
    if args.stim_order:
        import manifest as mf
        stim_order = mf.parse_stim_order(args.stim_order)
    if stim_order is None or t0 is None:
        ap.error("give --manifest, or --stim-order and --assay-t0")
    try:
        res = run(args.recording_id, args.recordings, args.out_dir, stim_order, t0,
                  args.timestamps, args.pressure_frames, args.stim_events, args.camera_frames,
                  args.behaviour_events, args.video_deliveries, args.frame_table,
                  args.pressure_summary, args.motion, args.accept_count_fail,
                  args.trust_stim_order)
    except TimelineError as e:
        print(f"MASTER TIMELINE FAILED: {e}", file=sys.stderr)
        return 1
    print(f"wrote {res['paths']['frames']} ({len(res['frames'])} frames)")
    for n_ in res["notes"]:
        print("  note:", n_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
