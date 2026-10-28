"""pressure_sync.py - one pressure-logger session -> SYNC pulse table + stimulus contact events.

Reads one Arduino pressure-logger session (<stem>_pressure.csv + <stem>_info.json, written by
vendor/pressure_logger) and writes, as defined in CONTRACTS.md:

  pressure_frames.csv    one row per SYNC pulse n: pulse, isx_frame = n + K (the frame the
                         samples with ttl_count == n fall in), t_arduino_s = the time of pulse n
                         on the Arduino clock (it ENDS isx_frame n-1+K), and the force of the
                         samples with ttl_count == n
  stim_events.csv        one row per detected contact (source=pressure) and one row per scheduled
                         trial with no detection (source=expected, e.g. heat, scored on video later)
  pressure_qc.png        whole-session force trace, detected onsets, schedule ticks, pulse residuals
  pressure_report.txt    short human-readable QC report
  pressure_summary.json  numbers for count_check / master_timeline (n_ttl, fps, Arduino fit, flags)

The sync rule (CONTRACTS.md section 1a, revised 2026-10-19):
  SYNC pulse n (1-based) ENDS isx_frame n-1+K, and that frame's TSC timestamp is the pulse time.
  Logger rows with ttl_count == n were taken after pulse n, during isx_frame n + K
  (master_timeline.count_to_isx_frame, the one function for logger samples AND camera frames).
  A contact onset is timed by the knee of its force rise (sub-sample, Arduino clock). Its count
  c = the pulses before that time, from each pulse's estimated time (the fitted pulse grid
  clipped to its sample bracket: pressure_frames.t_pulse_arduino_s); it gets onset_isx_frame =
  c + K and onset_cam_frame = c, and its seconds come from the Arduino fit, pulse n <->
  t_master(n-1+K). The last baseline sample before the contact is NOT used: it is up to a
  sample early, which put ~1 onset in 4 a frame early. The offset (end of stim_on) is the
  release knee. An onset whose uncertainty interval (fit statistics, never less than half a
  sample either side) holds a pulse is flagged pulse_ambiguous: its frame may be one off.
  The pulse grid assumes the logger's sample clock is not locked to the DAQ (bench sessions:
  residual SD 3.6 ms = uniform 12.5-ms quantisation).
ttl_count is the only sync signal used. `ttl` (one level read per sample) and `pc_time` (another
computer's clock) are never used, and neither is the online `indicator` (its thresholds were never
calibrated). Contact onsets are re-detected offline from `corrected`.

Usage:
  python pressure_sync.py --csv <stem>_pressure.csv --recording-id M01_d1_pre \
      --stim-order light,mild,heat,pin [--assay-t0 0] [--pulse-offset 0] [--frames-time frames.csv]
  python pressure_sync.py --manifest manifest.csv --recording-id M01_d1_pre
  python pressure_sync.py --csv testdata/pressure          (every *_pressure.csv in a folder)
  python pressure_sync.py --selftest                       (synthetic data only; exit 1 on failure)
Exit code: 0 ok; 1 a LOUD sync issue in any session (missed pulses, no pulses, manual start,
a stim_order the logger contradicts, ...; the tables and report are still written for QC);
2 a contract violation (unreadable or empty CSV, bad K, ...). An --out-dir whose
pressure_frames.csv / stim_events.csv belong to another recording is refused.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from master_timeline import arduino_time_to_pulse, count_to_isx_frame, other_recordings

HERE = Path(__file__).resolve().parent

LOGGER_COLUMNS = ["t_s", "arduino_ms", "pc_time", "raw", "baseline", "corrected", "indicator",
                  "ttl", "ttl_count"]
FRAMES_COLUMNS = ["recording_id", "pulse", "isx_frame", "t_arduino_s", "force_mean", "force_max",
                  "n_samples"]
# t_pulse_arduino_s: the pulse-time estimate (fitted grid clipped to the sample bracket) that
# decides which pulse interval a contact onset or offset falls in; master_timeline uses the
# same column, so both modules give an event the same pulse. t_arduino_s stays the bracket
# midpoint that the Arduino clock fit uses (CONTRACTS.md 1).
FRAMES_EXTRA = ["t_pulse_arduino_s", "missing_before", "pulse_flags"]
EVENTS_COLUMNS = ["event_id", "recording_id", "stim_type", "slot", "block", "trial", "source",
                  "t_expected_master_s", "onset_t_arduino_s", "onset_t_master_s",
                  "onset_isx_frame", "onset_cam_frame", "threshold_t_master_s",
                  "peak_t_master_s", "peak_force", "offset_t_master_s", "duration_s",
                  "match_residual_s", "flags"]
# Arduino-clock times are kept so master_timeline can re-map with its own frame-time fit.
EVENTS_EXTRA = ["t_source", "contact_force", "noise_sigma", "threshold_force", "impulse",
                "threshold_t_arduino_s", "peak_t_arduino_s", "offset_t_arduino_s",
                "onset_backtrack_t_arduino_s"]
STIM_KEYS = ("light", "mild", "heat", "pin")

# Assay schedule (CONTRACTS.md "schedule"), seconds after assay_t0_s on the master clock.
# These are the protocol, not acquisition rates.
SLOT_STARTS_S = (300.0, 660.0, 1020.0, 1380.0)
SLOT_LEN_S = 300.0
BLOCK_LEN_S = 60.0
N_BLOCKS = 5
TRIAL_OFFSETS_S = (0.0, 15.0, 30.0)
TRIALS_PER_SLOT = N_BLOCKS * len(TRIAL_OFFSETS_S)

DEFAULTS = dict(
    k_sigma=4.0,            # threshold = max(k * local sigma, min_force)
    min_force=0.0,          # counts; no counts-per-gram calibration exists yet
    min_hold=2,             # samples the threshold crossing must be held
    baseline_window_s=10.0, # running-median window for the slow drift of the stale tare
    weak_factor=2.0,        # weak = contact force below weak_factor * threshold
    tolerance_s=4.0,        # max |onset - scheduled trial time| for a match
    merge_gap_s=0.1,        # a dip to baseline shorter than this does not split a contact
    pre_fit_s=0.15,         # baseline span included in the onset knee fit
    # onset interval = knees within this many sigma^2 of the best fit's squared error: it held
    # the true contact for 359 of 360 synthetic taps (20 and 30 Hz), median width 8 ms
    onset_ci_sigma2=9.0,
    pause_s=1.0,            # a pulse gap longer than this is a TTL pause, not a missed pulse
    polarity=1,             # +1: a press raises `corrected`; -1 if A+/A- are swapped
)


class ContractError(RuntimeError):
    """Input that breaks CONTRACTS.md or the logger format; processing must stop."""


# ----------------------------------------------------------------------------- reading

def read_logger_csv(src) -> pd.DataFrame:
    """Read *_pressure.csv (path or text buffer) and enforce the logger format."""
    try:
        df = pd.read_csv(src)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        # a 0-byte or garbled file is a broken session, not a crash
        raise ContractError(f"logger CSV is empty or not a CSV ({type(e).__name__}: {e})")
    missing = [c for c in LOGGER_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"logger CSV lacks columns {missing}")
    if len(df) < 2:
        raise ContractError("logger CSV has fewer than 2 samples")
    for col in ("t_s", "corrected", "ttl_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            # An empty ttl_count means the Arduino sketch did not report pulses: no frame sync.
            raise ContractError(f"column {col} has {int(df[col].isna().sum())} empty/non-numeric "
                                f"values (empty ttl_count = old sketch without TTL)")
    if np.any(np.diff(df["t_s"].to_numpy()) <= 0):
        raise ContractError("t_s is not strictly increasing")
    count = df["ttl_count"].to_numpy()
    if np.any(count != np.round(count)) or np.any(count < 0):
        raise ContractError("ttl_count has non-integer or negative values")
    if np.any(np.diff(count) < 0):
        i = int(np.argmax(np.diff(count) < 0)) + 1
        raise ContractError(f"ttl_count decreases at row {i} (counter reset?)")
    df["ttl_count"] = count.astype(np.int64)
    df["arduino_ms"] = pd.to_numeric(df["arduino_ms"], errors="coerce")
    return df


def read_info(path) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def info_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name.replace("_pressure.csv", "_info.json"))


def check_info(info: dict | None, df: pd.DataFrame, pause_s: float = DEFAULTS["pause_s"]) -> list:
    """Validate info.json against the CSV. Returns [(level, code, message)];
    level LOUD = the frame mapping may be wrong, WARN = inconsistency, INFO = note."""
    issues = []
    t = df["t_s"].to_numpy()
    n_ttl = int(df["ttl_count"].iloc[-1])
    ms = df["arduino_ms"].to_numpy()
    if np.isnan(ms).any():
        issues.append(("LOUD", "no_arduino_ms", "arduino_ms is empty: t_s is then the PC clock, "
                       "not the Arduino clock"))
    elif np.max(np.abs((ms - ms[0]) / 1000.0 - t)) > 0.0011:
        issues.append(("WARN", "t_s_not_millis", "t_s differs from (arduino_ms - arduino_ms[0])/1000"))
    if info is None:
        issues.append(("LOUD", "no_info_json", "no *_info.json: the trigger (ttl/manual) is "
                       "unknown, so ttl_count == frame number cannot be checked"))
        return issues
    trigger = info.get("trigger")
    if trigger != "ttl":
        lead = _lead_in_s(df)
        if n_ttl == 0:
            why = "and no pulse was counted at all"
        elif int(df["ttl_count"].iloc[0]) == 0 and lead > pause_s:
            # Only a quiet stretch longer than any frame interval says the train had not
            # started yet; a Start pressed into a running train also shows a lead of 0-1 period.
            why = (f"the logger ran {lead:.2f} s without pulses before the first one, so pulse 1 "
                   f"is probably the first miniscope frame's pulse; confirm with count_check")
        else:
            why = (f"the first pulse came only {lead:.3f} s after the start, so pulses were "
                   f"probably already running and ttl_count is offset by an unknown number of frames")
        issues.append(("LOUD", "manual_start", f"trigger = {trigger!r}, not 'ttl': ttl_count may "
                       f"not equal the frame number ({why})"))
    elif int(df["ttl_count"].iloc[0]) == 0:
        issues.append(("WARN", "ttl_start_without_pulse", "trigger 'ttl' but the first row has ttl_count 0"))
    if info.get("ttl_mode", "pulses") != "pulses":
        issues.append(("LOUD", "ttl_mode", f"ttl_mode = {info.get('ttl_mode')!r}: sessions are "
                       f"started/stopped by TTL level, not pulses"))
    if info.get("ttl_input") is False:
        issues.append(("LOUD", "no_ttl_input", "info.json says the sketch reports no TTL"))
    if info.get("ttl_pulses") is not None and int(info["ttl_pulses"]) != n_ttl:
        issues.append(("LOUD", "pulse_count_mismatch", f"info ttl_pulses = {info['ttl_pulses']} "
                       f"but the CSV ends at ttl_count = {n_ttl} (truncated CSV?)"))
    if info.get("n_samples") is not None and int(info["n_samples"]) != len(df):
        issues.append(("WARN", "sample_count_mismatch",
                       f"info n_samples = {info['n_samples']} but the CSV has {len(df)} rows"))
    if info.get("duration_s") is not None and abs(float(info["duration_s"]) - (t[-1] - t[0])) > 0.05:
        issues.append(("WARN", "duration_mismatch", f"info duration_s = {info['duration_s']} but "
                       f"t_s spans {t[-1] - t[0]:.3f} s"))
    cols = info.get("columns")
    if isinstance(cols, dict) and list(cols) != LOGGER_COLUMNS:
        issues.append(("WARN", "column_list", f"info.json columns {list(cols)} differ from {LOGGER_COLUMNS}"))
    return issues


def _lead_in_s(df: pd.DataFrame) -> float:
    """Seconds from the first sample to the first ttl_count increment (0 if none)."""
    count = df["ttl_count"].to_numpy()
    rows = np.flatnonzero(count > count[0])
    return float(df["t_s"].iloc[rows[0]] - df["t_s"].iloc[0]) if len(rows) else 0.0


# ----------------------------------------------------------------------------- pulses

def robust_line(x, y, cut=4.0, n_iter=10):
    """Least-squares line y = a + b*x, refitted without points beyond `cut` robust SDs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    valid = keep.copy()
    a = b = np.nan
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        b, a = np.polyfit(x[keep], y[keep], 1)
        resid = y - (a + b * x)
        scale = 1.4826 * np.median(np.abs(resid[keep]))
        new = valid & (np.abs(resid) <= max(cut * scale, 1e-9))
        if np.array_equal(new, keep):
            break
        keep = new
    resid = y - (a + b * x)
    return a, b, resid, keep


def pulse_table(t: np.ndarray, count: np.ndarray, pause_s: float):
    """One row per SYNC pulse with its time bracket on the Arduino clock, plus QC numbers.

    Pulse n happened between the last sample with ttl_count < n and the first with >= n, so its
    time is the bracket midpoint. Missed pulses and glitches are judged from the bracket bounds,
    not the midpoints: at 80 Hz sampling a 30 Hz train shows midpoint intervals of 25 or 37 ms,
    so a midpoint median is not the period."""
    n_ttl = int(count[-1])
    qc = dict(n_ttl=n_ttl, period_s=np.nan, fps_arduino=np.nan, fit_c0=np.nan, fit_p=np.nan,
              fit_resid_ms=np.nan, fit_resid_max_ms=np.nan, n_fit=0, n_multi_increment=0,
              n_glitch=0, n_missed=0, n_gaps=0, n_pauses=0, longest_gap_s=np.nan, n_off_grid=0,
              n_open_bracket=0, first_pulse_t=np.nan, last_pulse_t=np.nan,
              lead_in_s=np.nan, tail_s=np.nan)
    cols = ["pulse", "row", "t_lo", "t_hi", "t_mid", "t_est", "step", "open", "multi", "glitch",
            "missing_before", "seq", "resid_ms", "used"]
    if n_ttl == 0:
        return pd.DataFrame(columns=cols), qc

    dt_med = float(np.median(np.diff(t)))
    pulse = np.arange(1, n_ttl + 1)
    row = np.searchsorted(count, pulse, side="left")          # first row with ttl_count >= n
    prev = row - 1
    has_prev = prev >= 0
    t_hi = t[row]
    # A TTL-started session begins at the sample after pulse 1, so that bracket has no left edge.
    t_lo = np.where(has_prev, t[np.maximum(prev, 0)], t_hi - dt_med)
    step = count[row] - np.where(has_prev, count[np.maximum(prev, 0)], 0)
    t_mid = 0.5 * (t_lo + t_hi)
    is_open = ~has_prev
    multi = step > 1

    if n_ttl > 1:
        m = min(10, n_ttl - 1)
        period = float(np.median((t_mid[m:] - t_mid[:-m]) / m))
    else:
        period = np.nan
    glitch = np.zeros(n_ttl, bool)
    missing_before = np.zeros(n_ttl, np.int64)
    if np.isfinite(period) and period > 0:
        lo = np.maximum(0.0, t_lo[1:] - t_hi[:-1])      # shortest possible interval
        hi = t_hi[1:] - t_lo[:-1]                        # longest possible interval
        glitch[1:] = hi < 0.5 * period                   # too close to be a real frame
        gap = lo > 1.5 * period                          # certainly at least one pulse missing
        mid_int = t_mid[1:] - t_mid[:-1]
        missing_before[1:] = np.where(gap, np.maximum(1, np.round(mid_int / period) - 1), 0)
        qc["n_gaps"] = int(gap.sum())
        qc["n_pauses"] = int((gap & (mid_int > pause_s)).sum())
        qc["longest_gap_s"] = float(mid_int.max())
    # seq = position in the true pulse train, so one missed pulse does not bend the fit
    seq = pulse + np.cumsum(missing_before) - np.cumsum(glitch)
    used = ~is_open & ~multi & ~glitch
    resid_ms = np.full(n_ttl, np.nan)
    # t_est = best estimate of each pulse time, used to decide on which side of a pulse a
    # contact falls. The DAQ pulse train is regular, so the fitted grid is good to well under a
    # millisecond (bench sessions: residual SD 3.6 ms = pure 12.5-ms sample quantisation),
    # while a bracket midpoint is only good to +-6 ms. Clipping to the bracket keeps every
    # estimate consistent with the samples, even where the grid is off.
    t_est = t_mid.copy()
    if used.sum() >= 3:
        c0, p, resid, keep = robust_line(seq[used], t_mid[used])
        resid_ms = (t_mid - (c0 + p * seq)) * 1000.0
        t_est = np.clip(c0 + p * seq, t_lo, t_hi)
        qc.update(fit_c0=c0, fit_p=p, fps_arduino=1.0 / p, n_fit=int(keep.sum()),
                  fit_resid_ms=float(np.std(resid[keep]) * 1000.0),
                  fit_resid_max_ms=float(np.nanmax(np.abs(resid_ms[used]))))
        # Half a period off the fitted grid = a miscount the bracket test could not see.
        qc["n_off_grid"] = int((np.abs(resid_ms[used]) > 500.0 * p).sum())
    qc.update(period_s=period, n_multi_increment=int(multi.sum()), n_glitch=int(glitch.sum()),
              n_missed=int(missing_before.sum()), n_open_bracket=int(is_open.sum()),
              first_pulse_t=float(t_mid[0]), last_pulse_t=float(t_mid[-1]),
              lead_in_s=float(t_hi[0] - t[0]), tail_s=float(t[-1] - t_hi[-1]))
    pulses = pd.DataFrame(dict(pulse=pulse, row=row, t_lo=t_lo, t_hi=t_hi, t_mid=t_mid,
                               t_est=np.round(t_est, 6), step=step,
                               open=is_open, multi=multi, glitch=glitch,
                               missing_before=missing_before, seq=seq, resid_ms=resid_ms, used=used))
    return pulses, qc


def pulse_flags(pulses: pd.DataFrame, period_s: float) -> list:
    out = []
    for r in pulses.itertuples():
        f = []
        if r.open:
            f.append("open_bracket")
        if r.multi:
            f.append("multi_increment")
        if r.glitch:
            f.append("glitch")
        if r.missing_before:
            f.append(f"missing_before={r.missing_before}")
        if np.isfinite(r.resid_ms) and np.isfinite(period_s) and abs(r.resid_ms) > 500.0 * period_s:
            f.append("off_grid")
        out.append(";".join(f))
    return out


def build_pressure_frames(rid, t, count, force, pulses, qc, K) -> pd.DataFrame:
    """pressure_frames.csv: one row per pulse; force over the samples with ttl_count == pulse."""
    n_ttl = qc["n_ttl"]
    if n_ttl == 0:
        return pd.DataFrame(columns=FRAMES_COLUMNS + FRAMES_EXTRA)
    period = qc["period_s"] if np.isfinite(qc["period_s"]) else np.inf
    dt = (t[-1] - t[0]) / (len(t) - 1)
    # The last pulse's count lasts through the whole tail after imaging stops (and a count
    # stalls through a missed pulse); keep only the samples within one period (+ one sample of
    # quantisation) of the pulse, so the tail is not averaged into the last frame.
    has = count >= 1
    idx = count[has] - 1
    in_window = t[has] <= pulses["t_hi"].to_numpy()[idx] + period + dt
    s = pd.DataFrame(dict(pulse=count[has][in_window], force=force[has][in_window]))
    g = s.groupby("pulse")["force"]
    agg = pd.DataFrame(dict(force_mean=g.mean(), force_max=g.max(), n_samples=g.size()))
    agg = agg.reindex(np.arange(1, n_ttl + 1))
    out = pd.DataFrame(dict(
        recording_id=rid,
        pulse=pulses["pulse"].to_numpy(),
        # the samples with ttl_count == n fall in isx_frame n + K; the pulse time itself is
        # the end of frame n-1+K (master_timeline fits it against that frame's timestamp)
        isx_frame=count_to_isx_frame(pulses["pulse"].to_numpy(), K),
        t_arduino_s=pulses["t_mid"].round(5).to_numpy(),
        force_mean=agg["force_mean"].round(2).to_numpy(),
        force_max=agg["force_max"].round(2).to_numpy(),
        n_samples=agg["n_samples"].fillna(0).astype(int).to_numpy(),
        t_pulse_arduino_s=pulses["t_est"].to_numpy(),
        missing_before=pulses["missing_before"].to_numpy(),
        pulse_flags=pulse_flags(pulses, qc["fit_p"]),
    ))
    return out


# ----------------------------------------------------------------------------- force + contacts

def remove_drift(x, win, short, polarity, mask=None):
    """Force = polarity * (corrected - running median over `win` samples).
    Local sigma = 1.4826 * running median of |corrected - running median over `short` samples|:
    the MAD of the high-passed trace, so a tare step or a long hold does not count as noise.
    Samples in `mask` (detected contacts) are left out, so long holds do not lift the baseline."""
    minp = max(3, win // 5)
    xs = pd.Series(np.where(mask, np.nan, x) if mask is not None else x)
    base = xs.rolling(win, center=True, min_periods=minp).median()
    base = base.interpolate(limit_direction="both").to_numpy()
    if np.isnan(base).all():
        base = np.full(len(x), np.nanmedian(x))
    force = polarity * (x - base)
    hp = np.abs(x - pd.Series(x).rolling(short, center=True, min_periods=1).median().to_numpy())
    fa = pd.Series(np.where(mask, np.nan, hp) if mask is not None else hp)
    sigma = 1.4826 * fa.rolling(win, center=True, min_periods=minp).median()
    sigma = sigma.interpolate(limit_direction="both").to_numpy()
    if np.isnan(sigma).all():
        sigma = np.full(len(x), 1.4826 * np.median(hp))
    return force, base, np.maximum(sigma, 1.0)   # NAU7802 counts are integers


def runs_of(mask):
    """(start, end) inclusive index pairs of True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def refine_onset(t, f, lo, hi, d_sse=0.0):
    """Knee of a 'flat, then straight rise' least-squares fit to f[lo..hi]: returns (the last
    sample at or before the knee, the knee time, ci_lo, ci_hi) or None if the fit fails. The
    knee time is sub-sample, and it is what times the onset: the last baseline sample before it
    can be up to one sample (12-14 ms) early, which put ~1 onset in 4 in the previous frame
    whenever a pulse fell between that sample and the contact. The plain 1-sigma back-track
    walks into correlated noise (lag-1 r ~ 0.5) in ~5% of taps; the knee fit uses every sample
    of the edge. [ci_lo, ci_hi] = the knees whose squared error is within d_sse of the best:
    a faint tap with two samples on its rise gets a wide interval, a firm one a narrow one."""
    tt = t[lo:hi + 1]
    y = f[lo:hi + 1]
    if len(tt) < 4:
        return None
    knees = np.arange(tt[1], tt[-1], 0.0005)
    if len(knees) == 0:
        return None
    h = np.maximum(0.0, tt[None, :] - knees[:, None])
    n = len(tt)
    sh, shh, sy, shy = h.sum(1), (h * h).sum(1), y.sum(), h @ y
    det = n * shh - sh * sh
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * shy - sh * sy) / det
        level = (sy - slope * sh) / n
        sse = ((y[None, :] - level[:, None] - slope[:, None] * h) ** 2).sum(1)
    sse[~(det > 0) | ~(slope > 0)] = np.inf
    if not np.isfinite(sse).any():
        return None
    knee = float(knees[int(np.argmin(sse))])
    near = knees[sse <= np.min(sse) + d_sse]
    return (lo + int(np.searchsorted(tt, knee, side="right")) - 1, knee,
            float(near.min()), float(near.max()))


def refine_offset(t, f, lo, hi):
    """Release knee: the end of a 'straight fall, then flat' fit to f[lo..hi], found as the
    onset knee of the time-reversed trace. Returns the knee time or None. The first sample
    back at baseline is up to a sample late, which ended stim_on a frame late."""
    if hi - lo < 3:
        return None
    r = refine_onset(-t[lo:hi + 1][::-1], f[lo:hi + 1][::-1], 0, hi - lo)
    return None if r is None else -r[1]          # [1] = the knee time


def _between(t, i):
    """Midpoint of samples i and i+1: the unbiased guess for an edge known only to lie
    between them."""
    return 0.5 * (t[i] + t[min(i + 1, len(t) - 1)])


def detect_pass(t, f, sigma, fs, prm):
    """Contacts = runs above max(k*sigma, min_force) held >= min_hold samples. Onset: back-track
    to the last sample within 1 sigma, then refine by the knee fit; offset: first sample back
    within 1 sigma after the last run of the contact. Each event also gets sub-sample times:
    t_onset = the knee (weak taps: between the back-track sample and the next one) and
    t_offset = between the last loaded sample and the first one back at baseline."""
    n = len(f)
    thr = np.maximum(prm["k_sigma"] * sigma, prm["min_force"])
    runs = [r for r in runs_of(f > thr) if r[1] - r[0] + 1 >= prm["min_hold"]]
    merge_gap = int(round(prm["merge_gap_s"] * fs))
    n_pre = max(2, int(round(prm["pre_fit_s"] * fs)))
    half_dt = 0.5 * float(np.median(np.diff(t)))

    def down_from(k):
        while k < n - 1 and f[k] > sigma[k]:
            k += 1
        return k

    events = []
    for start, end in runs:
        if events and start <= events[-1]["i_offset"] + merge_gap:
            events[-1]["i_offset"] = down_from(end)        # same contact, force came back up
            events[-1]["i_last_run"] = end
            continue
        floor = events[-1]["i_offset"] if events else 0
        j = start
        while j > floor and f[j] > sigma[j]:
            j -= 1
        knee = refine_onset(t, f, max(floor, j - n_pre), min(start + 1, n - 1),
                            prm["onset_ci_sigma2"] * sigma[start] ** 2)
        # no fit: the contact is only known to lie between the back-track sample and the next
        onset, t_knee, ci_lo, ci_hi = knee if knee is not None else \
            (j, _between(t, j), t[j], t[min(j + 1, n - 1)])
        events.append(dict(i_onset=max(onset, floor), t_knee=max(t_knee, t[floor]),
                           ci=(max(ci_lo, t[floor]), ci_hi),
                           i_backtrack=j, i_threshold=start, i_offset=down_from(end),
                           i_last_run=end))
    # Peak on a ~60 ms running mean: the raw maximum of a 1-s hold sits ~2.5 sigma above the
    # true force because it picks the largest noise sample, which would hide weak taps.
    smooth = pd.Series(f).rolling(max(1, int(round(0.06 * fs))) | 1, center=True,
                                  min_periods=1).mean().to_numpy()
    for k, ev in enumerate(events):
        seg = slice(ev["i_threshold"], ev["i_offset"] + 1)
        ev["i_peak"] = ev["i_threshold"] + int(np.argmax(smooth[seg]))
        ev["peak_force"] = float(smooth[ev["i_peak"]])
        ev["threshold_force"] = float(thr[ev["i_threshold"]])
        ev["noise_sigma"] = float(sigma[ev["i_threshold"]])
        # Contact force = median force while above half the peak: how hard the paw was pressed,
        # without the noise maximum. Weak = a contact force too close to the threshold.
        top = seg.start + np.flatnonzero(smooth[seg] >= 0.5 * ev["peak_force"])
        ev["contact_force"] = float(np.median(f[top]))
        ev["weak"] = ev["contact_force"] < prm["weak_factor"] * ev["threshold_force"]
        ev["t_onset"] = ev["t_knee"]
        # Never narrower than half a sample either side: the knee model's shape error (1-5 ms
        # on NAU7802-like averaged rises) and the uncorrected sensor response are not in the
        # fit's statistics.
        ev["ci"] = (min(ev["ci"][0], ev["t_knee"] - half_dt), max(ev["ci"][1], ev["t_knee"] + half_dt))
        if ev["weak"]:
            # A rise of only a few sigma gives the knee fit too little signal (it slides onto
            # pre-contact noise); the plain back-track is the better estimate there, good to
            # about a sample either side.
            jb = ev["i_backtrack"]
            ev["i_onset"] = jb
            ev["t_onset"] = _between(t, jb)
            ev["ci"] = (t[max(jb - 1, 0)], t[min(jb + 2, n - 1)])
        span = slice(ev["i_onset"], ev["i_offset"] + 1)
        ev["impulse"] = float(np.trapezoid(np.clip(f[span], 0, None), t[span]))
        ev["open_offset"] = ev["i_offset"] >= n - 1 and f[n - 1] > sigma[n - 1]
        # The release lies between the last loaded sample and the first one back at baseline;
        # that first baseline sample alone would end stim_on up to a sample (often a frame)
        # late. Strong contacts are timed by the release knee (fit window: the end of the last
        # threshold run to 0.15 s of baseline, never into the next contact), weak ones by the
        # midpoint of the two samples.
        if ev["open_offset"]:
            ev["t_offset"] = t[n - 1]
        else:
            nxt = events[k + 1]["i_backtrack"] if k + 1 < len(events) else n - 1
            hi = min(ev["i_offset"] + n_pre, nxt)
            knee = None if ev["weak"] else refine_offset(t, f, max(0, ev["i_last_run"] - 1), hi)
            t_off = _between(t, ev["i_offset"] - 1) if knee is None else knee
            ev["t_offset"] = float(np.clip(t_off, t[ev["i_last_run"]], t[ev["i_offset"]]))
    return events


def find_contacts(t, x, prm):
    """Two passes: the second recomputes drift and sigma with the first pass's contacts masked."""
    fs = (len(t) - 1) / (t[-1] - t[0])      # mean rate: ms stamps make the median interval 12 ms
    win = max(3, int(round(prm["baseline_window_s"] * fs)) | 1)
    short = max(3, int(round(1.0 * fs)) | 1)            # 1 s: longer than the noise correlation
    pad_pre, pad_post = int(round(0.25 * fs)), int(round(0.5 * fs))
    force, base, sigma = remove_drift(x, win, short, prm["polarity"])
    events = detect_pass(t, force, sigma, fs, prm)
    if events:
        mask = np.zeros(len(x), bool)
        for ev in events:
            mask[max(0, ev["i_onset"] - pad_pre): ev["i_offset"] + pad_post + 1] = True
        force, base, sigma = remove_drift(x, win, short, prm["polarity"], mask)
        events = detect_pass(t, force, sigma, fs, prm)
    for ev in events:
        # A contact longer than half the drift window is partly absorbed into the baseline,
        # so its force and offset are unreliable; it is also longer than any stimulus.
        ev["long"] = (t[ev["i_offset"]] - t[ev["i_onset"]]) > 0.5 * prm["baseline_window_s"]
    return events, force, base, sigma


# ----------------------------------------------------------------------------- schedule + clock

def parse_stim_order(text):
    if text is None or str(text).strip() == "" or str(text) == "nan":
        return None
    keys = [k.strip().lower() for k in str(text).split(",")]
    if sorted(keys) != sorted(STIM_KEYS):
        raise ContractError(f"stim_order {text!r} must list each of {STIM_KEYS} once")
    return keys


def build_schedule(stim_order, assay_t0):
    rows = []
    for s, slot_start in enumerate(SLOT_STARTS_S, start=1):
        for b in range(1, N_BLOCKS + 1):
            for k, off in enumerate(TRIAL_OFFSETS_S, start=1):
                rows.append(dict(sched_index=(s - 1) * TRIALS_PER_SLOT + (b - 1) * len(TRIAL_OFFSETS_S) + k,
                                 slot=s, block=b, trial=k,
                                 stim_type=stim_order[s - 1] if stim_order else "",
                                 t_expected=assay_t0 + slot_start + BLOCK_LEN_S * (b - 1) + off))
    return pd.DataFrame(rows)


def slot_of(t_master, assay_t0):
    """(slot, block) of a master time inside a stimulus slot, else (None, None)."""
    for s, start in enumerate(SLOT_STARTS_S, start=1):
        rel = t_master - (assay_t0 + start)
        if 0.0 <= rel < SLOT_LEN_S:
            return s, int(rel // BLOCK_LEN_S) + 1
    return None, None


def read_frames_time(path, rid):
    ft = pd.read_csv(path)
    for col in ("isx_frame", "t_master_s"):
        if col not in ft.columns:
            raise ContractError(f"{path}: frames-time table needs columns isx_frame, t_master_s")
    if "recording_id" in ft.columns:
        ft = ft[ft["recording_id"].astype(str) == str(rid)]
    ft = ft[np.isfinite(pd.to_numeric(ft["t_master_s"], errors="coerce"))]
    if ft.empty:
        raise ContractError(f"{path}: no finite t_master_s rows for {rid}")
    return ft.set_index(ft["isx_frame"].astype(np.int64))["t_master_s"].astype(float)


def clock_map(pulses, qc, K, frames_time=None, period_s=None):
    """t_master = a + b * t_arduino.
    With frame times: robust fit of pulse midpoints against t_master of isx_frame pulse-1+K.
    Without: t_master = isx_frame * period_s along the pulse fit, with period_s from --period-s
    or else the Arduino-clock period (then the Arduino/master rate difference is not removed)."""
    out = dict(a=np.nan, b=np.nan, ppm=np.nan, resid_ms=np.nan, resid_max_ms=np.nan,
               n_used=0, t_source="none")
    if frames_time is not None and len(pulses):
        frame = pulses["pulse"].to_numpy() - 1 + K
        y = frames_time.reindex(frame).to_numpy()
        use = pulses["used"].to_numpy() & np.isfinite(y)
        if use.sum() >= 3:
            a, b, resid, keep = robust_line(pulses["t_mid"].to_numpy()[use], y[use])
            out.update(a=a, b=b, ppm=(b - 1.0) * 1e6, resid_ms=float(np.std(resid[keep]) * 1e3),
                       resid_max_ms=float(np.max(np.abs(resid)) * 1e3), n_used=int(keep.sum()),
                       t_source="frame_times")
            return out
    if np.isfinite(qc["fit_p"]):
        c0, p = qc["fit_c0"], qc["fit_p"]
        period = period_s if period_s else p
        b = period / p
        out.update(a=(K - 1) * period - c0 * b, b=b,
                   ppm=(b - 1.0) * 1e6 if period_s else np.nan,
                   resid_ms=qc["fit_resid_ms"], resid_max_ms=qc["fit_resid_max_ms"],
                   n_used=qc["n_fit"], t_source="period_arg" if period_s else "arduino_period")
    return out


def check_stim_order(ev_df, stim_order):
    """[(LOUD, 'stim_order', msg)] when the logger contradicts stim_order. A hot water drop
    gives no force, so the heat slot must not look like a touch slot. A wrong stim_order would
    otherwise relabel the whole stimulus axis with no more than per-row flags."""
    if not stim_order or not len(ev_df):
        return []
    det = ev_df[(ev_df["source"] == "pressure") & ev_df["slot"].notna()]
    n = {s: int((det["slot"] == s).sum()) for s in range(1, len(SLOT_STARTS_S) + 1)}
    heat = stim_order.index("heat") + 1
    touch = [n[s] for s in n if s != heat]
    # a few stray touches during heat are allowed; a touch-like count is not
    if n[heat] >= 5 and n[heat] >= 0.5 * float(np.median(touch)):
        fewest = min(n, key=n.get)
        per_slot = ", ".join(f"slot {s} {stim_order[s - 1]}: {v}" for s, v in n.items())
        return [("LOUD", "stim_order",
                 f"stim_order puts heat in slot {heat}, but the logger found {n[heat]} contacts "
                 f"there ({per_slot}); heat gives no force, and slot {fewest} looks like the heat "
                 f"slot. Check the manifest stim_order")]
    return []


def assign(onset_master, sched_t, tol):
    """Each detection -> its nearest scheduled trial if within tol; a trial takes only its
    closest detection (with 15-s trials and a few-s tolerance no detection has two candidates)."""
    match = np.full(len(onset_master), -1)
    if len(onset_master) == 0 or not np.isfinite(onset_master).any():
        return match
    dist = np.abs(onset_master[:, None] - sched_t[None, :])
    nearest = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    d = dist[np.arange(len(onset_master)), nearest]
    taken = set()
    for i in np.argsort(d):
        if np.isfinite(d[i]) and d[i] <= tol and nearest[i] not in taken:
            match[i] = nearest[i]
            taken.add(nearest[i])
    return match


# ----------------------------------------------------------------------------- events table

def onset_count(t_on, pulses):
    """ttl_count a logger sample taken exactly at Arduino time t_on would carry, from the
    pulse-time estimates (pressure_frames.t_pulse_arduino_s). master_timeline repeats this with
    the same function on the same written values, so both give an event the same pulse."""
    if not len(pulses):
        return 0
    return arduino_time_to_pulse(t_on, pulses["pulse"].to_numpy(), pulses["t_est"].to_numpy())


def build_stim_events(rid, t, count, events, sched, cmap, qc, K, assay_t0, prm, session_flags,
                      pulses=None):
    a, b = cmap["a"], cmap["b"]
    pulses = pulses if pulses is not None else pd.DataFrame(columns=["pulse", "t_est"])
    pulse_t = pulses["t_est"].to_numpy(float)

    def to_master(ta):
        return a + b * ta if np.isfinite(b) else np.nan

    # Times are rounded once, here, to what the CSV holds: the pulse decision below and the
    # one master_timeline makes from the CSV then use identical numbers.
    t_on_all = np.array([round(float(ev["t_onset"]), 6) for ev in events], float)
    onset_master = np.array([to_master(x) for x in t_on_all], float)
    match = assign(onset_master, sched["t_expected"].to_numpy(), prm["tolerance_s"])
    n_ttl = qc["n_ttl"]
    rows, extras = [], []
    for ev, m, tm, t_on in zip(events, match, onset_master, t_on_all):
        t_off = round(float(ev["t_offset"]), 6)
        c = onset_count(t_on, pulses)
        flags = []
        frame = cam = None
        if c == 0:
            flags.append("before_imaging")
        elif c >= n_ttl:
            # after the last pulse: that pulse ended the last frame, so this is after the movie
            flags.append("after_imaging")
        else:
            # CONTRACTS.md 1a: count c falls in isx_frame c + K, seen by camera frame c
            frame, cam = count_to_isx_frame(c, K), c
        # A pulse inside the onset's interval: the data cannot say on which side of that pulse
        # the contact was, so the frame may be one off either way.
        if np.any((pulse_t >= ev["ci"][0]) & (pulse_t <= ev["ci"][1])):
            flags.append("pulse_ambiguous")
        if not np.isfinite(tm):
            flags.append("unsynced")
        row = dict(recording_id=rid, source="pressure",
                   onset_t_arduino_s=t_on, onset_t_master_s=tm,
                   onset_isx_frame=frame, onset_cam_frame=cam,
                   threshold_t_master_s=to_master(t[ev["i_threshold"]]),
                   peak_t_master_s=to_master(t[ev["i_peak"]]), peak_force=ev["peak_force"],
                   offset_t_master_s=to_master(t_off),
                   duration_s=(t_off - t_on) * (b if np.isfinite(b) else 1.0),
                   contact_force=ev["contact_force"], noise_sigma=ev["noise_sigma"],
                   threshold_force=ev["threshold_force"],
                   impulse=ev["impulse"], threshold_t_arduino_s=t[ev["i_threshold"]],
                   peak_t_arduino_s=t[ev["i_peak"]], offset_t_arduino_s=t_off,
                   onset_backtrack_t_arduino_s=t[ev["i_backtrack"]],
                   t_source=cmap["t_source"])
        if m >= 0:
            s = sched.iloc[m]
            row.update(slot=int(s.slot), block=int(s.block), trial=int(s.trial),
                       stim_type=s.stim_type, t_expected_master_s=s.t_expected,
                       match_residual_s=tm - s.t_expected, num=int(s.sched_index))
        else:
            flags.append("extra")
            slot, block = slot_of(tm, assay_t0) if np.isfinite(tm) else (None, None)
            if slot is None:
                flags.append("outside_slot")
            else:
                row.update(slot=slot, block=block,
                           stim_type=sched.loc[sched.slot == slot, "stim_type"].iloc[0])
            extras.append(row)
        if ev["weak"]:
            flags.append("weak")
        if ev["open_offset"]:
            flags.append("open_offset")
        if ev["long"]:
            flags.append("long")
        row["flags"] = flags + session_flags
        rows.append(row)
    for k, row in enumerate(sorted(extras, key=lambda r: r["onset_t_arduino_s"])):
        # extras numbered after the 60 scheduled trials (CONTRACTS.md 1a gap-fill rule)
        row["num"] = len(sched) + 1 + k
    t_lo_master, t_hi_master = to_master(t[0]), to_master(t[-1])
    for m in sorted(set(range(len(sched))) - set(match[match >= 0].tolist())):
        s = sched.iloc[m]
        flags = ["missing"]
        if not (t_lo_master <= s.t_expected <= t_hi_master):
            flags.append("not_recorded")          # also when there is no clock map at all
        rows.append(dict(recording_id=rid, source="expected", slot=int(s.slot), block=int(s.block),
                         trial=int(s.trial), stim_type=s.stim_type, t_expected_master_s=s.t_expected,
                         t_source=cmap["t_source"], flags=flags + session_flags,
                         num=int(s.sched_index)))
    df = pd.DataFrame(rows, columns=EVENTS_COLUMNS + EVENTS_EXTRA + ["num"])
    if df.empty:
        return df.drop(columns="num")
    df["event_id"] = [f"{rid}_e{n:03d}" for n in df["num"]]
    df["flags"] = df["flags"].apply(lambda f: ";".join(f))
    sort_t = df["onset_t_master_s"].fillna(df["t_expected_master_s"]).fillna(df["onset_t_arduino_s"])
    df = df.assign(_t=sort_t.to_numpy()).sort_values(["_t", "num"]).drop(columns=["_t", "num"])
    for col in ("slot", "block", "trial", "onset_isx_frame", "onset_cam_frame"):
        df[col] = df[col].astype("Int64")
    for col in ("t_expected_master_s", "onset_t_master_s",
                "threshold_t_master_s", "peak_t_master_s", "offset_t_master_s", "duration_s",
                "match_residual_s", "threshold_t_arduino_s", "peak_t_arduino_s",
                "onset_backtrack_t_arduino_s"):
        df[col] = df[col].astype(float).round(4)
    # already rounded to 6 decimals above; rounding again to 4 would move them across a pulse
    for col in ("onset_t_arduino_s", "offset_t_arduino_s"):
        df[col] = df[col].astype(float)
    for col in ("peak_force", "contact_force", "noise_sigma", "threshold_force", "impulse"):
        df[col] = df[col].astype(float).round(1)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- outputs

def plot_qc(path, rid, t, x, base, force, sigma, prm, ev_df, sched, pulses, cmap, assay_t0, qc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available: no QC PNG)", file=sys.stderr)
        return False
    synced = np.isfinite(cmap["b"])
    if synced:
        tx = (cmap["a"] + cmap["b"] * t - assay_t0) / 60.0
        xlabel = "assay time (min, master clock, 0 = assay_t0)"
    else:
        tx = t / 60.0
        xlabel = "logger time t_s (min, Arduino clock; no frame sync)"

    def x_of_master(tm):
        return (np.asarray(tm, float) - assay_t0) / 60.0

    # fixed hue per stimulus (follows the entity, not the slot position)
    hue = dict(light="#2a78d6", mild="#eb6834", heat="#1baf7a", pin="#eda100")
    ink, muted, red = "#0b0b0b", "#8a8984", "#e34948"
    fig, axs = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 2.2, 1]))
    ax = axs[0]
    ax.plot(tx, x, lw=0.4, color=muted, label="corrected (logger)")
    ax.plot(tx, base, lw=1.2, color=ink, label=f"running median ({prm['baseline_window_s']:g} s, contacts masked)")
    ax.set_ylabel("counts")
    ax.set_title("Logger 'corrected' and the slow baseline removed", loc="left", fontsize=10)
    legend_kw = dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False,
                     borderaxespad=0.2)
    ax.legend(ncol=2, **legend_kw)

    ax = axs[1]
    on_x = ax.get_xaxis_transform()          # x in data units, y in axes units (0 = bottom)
    if synced:
        for s, start in enumerate(SLOT_STARTS_S, start=1):
            st = sched.loc[sched.slot == s, "stim_type"].iloc[0]
            x0 = x_of_master(assay_t0 + start)
            ax.axvspan(x0, x0 + SLOT_LEN_S / 60.0, color=hue.get(st, "#cccccc"), alpha=0.12, lw=0)
            ax.text(x0 + 0.05, 0.98, f"slot {s}: {st or '?'}", transform=on_x,
                    va="top", fontsize=9, color=ink, clip_on=True)
        ax.plot(x_of_master(sched["t_expected"]), np.full(len(sched), 0.03), "|", color=ink,
                ms=10, mew=1, transform=on_x, label="scheduled trial")
    ax.plot(tx, force, lw=0.4, color=ink)
    ax.plot(tx, np.maximum(prm["k_sigma"] * sigma, prm["min_force"]), lw=0.8, ls="--",
            color=muted, label=f"threshold max({prm['k_sigma']:g} sigma, {prm['min_force']:g})")
    det = ev_df[ev_df["source"] == "pressure"] if len(ev_df) else ev_df
    if len(det):
        ex = det["onset_t_master_s"].to_numpy() if synced else det["onset_t_arduino_s"].to_numpy()
        ex = x_of_master(ex) if synced else ex / 60.0
        fl = det["flags"].fillna("")
        kinds = [("matched", ~fl.str.contains("extra") & ~fl.str.contains("weak"), "o", ink, ink),
                 ("weak", fl.str.contains("weak") & ~fl.str.contains("extra"), "o", ink, "none"),
                 ("extra", fl.str.contains("extra"), "^", red, red)]
        for name, sel, mk, ec, fc in kinds:
            sel = sel.to_numpy()
            if sel.any():
                ax.plot(ex[sel], det["peak_force"].to_numpy()[sel], mk, ms=7, mec=ec, mfc=fc,
                        ls="none", label=f"onset: {name} ({int(sel.sum())})")
    if synced and len(ev_df):
        miss = ev_df[(ev_df["source"] == "expected") & ~ev_df["flags"].str.contains("not_recorded")]
        if len(miss):
            ax.plot(x_of_master(miss["t_expected_master_s"]), np.full(len(miss), 0.03), "x",
                    color=red, ms=8, mew=1.5, transform=on_x,
                    label=f"scheduled, no detection ({len(miss)})")
    ax.set_ylabel("force (counts above baseline)")
    ax.set_title("Force; markers = peak force at contact onset", loc="left", fontsize=10)
    ax.legend(ncol=6, **legend_kw)

    ax = axs[2]
    if len(pulses) and np.isfinite(pulses["resid_ms"]).any():
        px = cmap["a"] + cmap["b"] * pulses["t_mid"].to_numpy() if synced else pulses["t_mid"].to_numpy()
        px = x_of_master(px) if synced else px / 60.0
        ax.plot(px, pulses["resid_ms"], ".", ms=1.5, color=ink, label="pulse midpoint - fit")
        bad = (pulses["glitch"] | (pulses["missing_before"] > 0)).to_numpy()
        if bad.any():
            ax.plot(px[bad], pulses["resid_ms"].to_numpy()[bad], "v", color=red, ms=7,
                    label=f"missed/glitch ({int(bad.sum())})")
        ax.legend(ncol=2, **legend_kw)
    else:
        ax.text(0.5, 0.5, "too few SYNC pulses for a fit", transform=ax.transAxes, ha="center")
    ax.set_ylabel("ms")
    ax.set_xlabel(xlabel)
    span = tx[-1] - tx[0]
    ax.set_xlim(tx[0] - 0.01 * span, tx[-1] + 0.01 * span)   # the recording, not the whole schedule
    ax.set_title(f"SYNC pulses: {qc['n_ttl']} counted, {qc['fps_arduino']:.4f} Hz on the Arduino "
                 f"clock, fit residual SD {qc['fit_resid_ms']:.1f} ms", loc="left", fontsize=10)
    n_det = int((ev_df["source"] == "pressure").sum()) if len(ev_df) else 0
    n_match = int(((ev_df["source"] == "pressure") & ev_df["trial"].notna()).sum()) if len(ev_df) else 0
    fig.suptitle(f"{rid}: {n_det} contacts detected, {n_match}/{len(sched)} scheduled trials matched "
                 f"(n = {len(t)} samples)", x=0.01, ha="left", fontsize=12)
    for a_ in axs:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="y", color="#e6e5e0", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def report_text(s, issues, ev_df, prm) -> str:
    L = [f"pressure_sync report: {s['recording_id']}",
         f"csv  : {s['csv']}", f"info : {s['info']}", ""]
    loud = [i for i in issues if i[0] == "LOUD"]
    if loud:
        L.append("!!! SYNC WARNINGS !!!")
        L += [f"  !!! {code}: {msg}" for _, code, msg in loud]
        L.append("")
    other = [i for i in issues if i[0] != "LOUD"]
    if other:
        L += [f"  {lvl} {code}: {msg}" for lvl, code, msg in other] + [""]
    L += ["SYNC pulses (ttl_count)",
          f"  pulses counted        : {s['n_ttl']}",
          f"  samples               : {s['n_samples']} at {s['sample_hz']:.2f} Hz (Arduino clock), "
          f"{s['n_sample_gaps']} sample gaps > 2.5x median",
          f"  first / last pulse    : {s['first_pulse_t_arduino_s']:.3f} / {s['last_pulse_t_arduino_s']:.3f} s "
          f"(lead-in {s['lead_in_s']:.3f} s, tail {s['tail_s']:.3f} s)",
          f"  fps (Arduino clock)   : {s['fps_arduino']:.5f} Hz from a linear fit over the pulses "
          f"(total pulses / total duration would give {s['fps_naive']:.4f})",
          f"  fit residual          : SD {s['pulse_fit_resid_ms']:.2f} ms, max {s['pulse_fit_resid_max_ms']:.2f} ms",
          f"  increments != +1      : {s['n_multi_increment']} pulses shared a sample with another pulse",
          f"  missed pulses         : {s['n_missed']} in {s['n_gaps']} gaps (>1.5 periods), "
          f"{s['n_pauses']} pauses > {prm['pause_s']:g} s, longest gap {s['longest_gap_s']:.4f} s",
          f"  glitches              : {s['n_glitch']} (pulse < 0.5 period after the previous one)",
          f"  off the fitted grid   : {s['n_off_grid']}",
          "",
          "Clock map  t_master = a + b * t_arduino",
          f"  source                : {s['t_source']}   (K = pulse_to_frame_offset = {s['pulse_to_frame_offset']})",
          f"  a, b                  : {s['arduino_fit_a']:.6f}, {s['arduino_fit_b']:.9f}",
          f"  ppm, residual SD      : {s['arduino_fit_ppm']:.1f} ppm, {s['arduino_fit_resid_ms']:.2f} ms",
          "",
          "Contacts (offline, from 'corrected'; 'indicator' ignored)",
          f"  detector              : max({prm['k_sigma']:g} sigma, {prm['min_force']:g}) held >= "
          f"{prm['min_hold']} samples, drift = {prm['baseline_window_s']:g} s running median, "
          f"polarity {prm['polarity']:+d}",
          f"  noise sigma (median)  : {s['noise_sigma_median']:.1f} counts",
          f"  detected              : {s['n_detected']} ({s['n_opposite_polarity']} would be found "
          f"with the opposite polarity)",
          f"  matched / extra       : {s['n_matched']} / {s['n_extra']} "
          f"({s['n_outside_slot']} outside any slot), weak {s['n_weak']}",
          f"  scheduled, no detection: {s['n_missing']} ({s['n_not_recorded']} outside the recording)",
          f"  median match residual : {s['median_match_residual_s']:.3f} s (tolerance {prm['tolerance_s']:g} s)"]
    if len(ev_df):
        L += ["", "Per stimulus (detected matched / 15, missing, extra)"]
        for st in sorted(set(ev_df["stim_type"].dropna()) - {""}):
            e = ev_df[ev_df["stim_type"] == st]
            fl = e["flags"].fillna("")
            matched = int(((e["source"] == "pressure") & e["trial"].notna()).sum())
            L.append(f"  {st:6s}: {matched:2d} / {TRIALS_PER_SLOT}, missing {int(fl.str.contains('missing').sum())}, "
                     f"extra {int(fl.str.contains('extra').sum())}")
        det = ev_df[ev_df["source"] == "pressure"]
        if 0 < len(det) <= 20:
            L += ["", "Detections"]
            for r in det.itertuples():
                L.append(f"  {r.event_id}: onset t_s {r.onset_t_arduino_s:.3f}, peak {r.peak_force:.0f} "
                         f"counts, duration {r.duration_s:.3f} s, flags {r.flags}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- driver

def process_session(csv_src, info, rid, stim_order=None, assay_t0=0.0, K=0, frames_time=None,
                    period_s=None, prm=None, out_dir=None, make_png=True, csv_name="", info_name=""):
    """Run the whole module on one session. Returns dict(frames, events, summary, issues, ...)."""
    prm = {**DEFAULTS, **(prm or {})}
    if out_dir is not None:
        # pressure_frames.csv and stim_events.csv hold ONE recording (CONTRACTS.md 3);
        # overwriting another recording's copy would silently delete its logger data
        for name in ("pressure_frames.csv", "stim_events.csv"):
            others = other_recordings(Path(out_dir) / name, rid)
            if others:
                raise ContractError(f"{Path(out_dir) / name} belongs to {others}, not {rid}: "
                                    "use one output folder per recording")
    df = read_logger_csv(csv_src)
    issues = check_info(info, df, prm["pause_s"])
    t = df["t_s"].to_numpy(float)
    x = df["corrected"].to_numpy(float)
    count = df["ttl_count"].to_numpy()
    dts = np.diff(t)
    pulses, qc = pulse_table(t, count, prm["pause_s"])
    if qc["n_ttl"] < 3:
        issues.append(("LOUD", "too_few_pulses", f"only {qc['n_ttl']} SYNC pulses: no fps, no "
                       f"clock map (preview or test session?)"))
    for code, n, what in (("missed_pulses", qc["n_missed"], "missed pulses"),
                          ("glitches", qc["n_glitch"], "glitch pulses"),
                          ("off_grid", qc["n_off_grid"], "pulses off the fitted grid"),
                          ("ttl_pause", qc["n_pauses"], "TTL pauses (more than one pulse train?)")):
        if n:
            issues.append(("LOUD", code, f"{n} {what}: ttl_count and frame number disagree after them"))
    if qc["n_multi_increment"]:
        issues.append(("WARN", "multi_increment", f"{qc['n_multi_increment']} pulses arrived in the "
                       f"same sample as another (increment != +1)"))
    if frames_time is None and not period_s and np.isfinite(qc["fit_p"]):
        issues.append(("INFO", "arduino_period", "no frame times or --period-s: t_master uses the "
                       "Arduino-clock period, so a few hundred ppm of clock-rate error remain"))

    events, force, base, sigma = find_contacts(t, x, prm)
    opposite, *_ = find_contacts(t, x, {**prm, "polarity": -prm["polarity"]})
    if len(opposite) > max(2, 2 * len(events)):
        issues.append(("WARN", "polarity", f"{len(opposite)} contacts with the opposite polarity vs "
                       f"{len(events)}: are A+/A- swapped? (--polarity -1)"))
    cmap = clock_map(pulses, qc, K, frames_time, period_s)
    sched = build_schedule(stim_order, assay_t0)
    session_flags = ["manual_start"] if any(i[1] == "manual_start" for i in issues) else []
    ev_df = build_stim_events(rid, t, count, events, sched, cmap, qc, K, assay_t0, prm,
                              session_flags, pulses)
    issues += check_stim_order(ev_df, stim_order)
    frames = build_pressure_frames(rid, t, count, force, pulses, qc, K)

    det = ev_df[ev_df["source"] == "pressure"] if len(ev_df) else ev_df
    fl = ev_df["flags"].fillna("") if len(ev_df) else pd.Series(dtype=str)
    n_ttl_t = (t[-1] - t[0])
    summary = dict(
        recording_id=rid, csv=csv_name, info=info_name,
        trigger=(info or {}).get("trigger"), n_samples=len(df),
        sample_hz=(len(t) - 1) / (t[-1] - t[0]),
        n_sample_gaps=int((dts > 2.5 * np.median(dts)).sum()),
        n_ttl=qc["n_ttl"], first_pulse_t_arduino_s=qc["first_pulse_t"],
        last_pulse_t_arduino_s=qc["last_pulse_t"], lead_in_s=qc["lead_in_s"], tail_s=qc["tail_s"],
        fps_arduino=qc["fps_arduino"], period_arduino_s=qc["fit_p"],
        fps_naive=qc["n_ttl"] / n_ttl_t if n_ttl_t > 0 else np.nan,
        pulse_fit_resid_ms=qc["fit_resid_ms"], pulse_fit_resid_max_ms=qc["fit_resid_max_ms"],
        n_multi_increment=qc["n_multi_increment"], n_glitch=qc["n_glitch"],
        n_missed=qc["n_missed"], n_gaps=qc["n_gaps"], n_pauses=qc["n_pauses"],
        longest_gap_s=qc["longest_gap_s"], n_off_grid=qc["n_off_grid"],
        n_open_bracket=qc["n_open_bracket"],
        pulse_to_frame_offset=K, arduino_fit_a=cmap["a"], arduino_fit_b=cmap["b"],
        arduino_fit_ppm=cmap["ppm"], arduino_fit_resid_ms=cmap["resid_ms"], t_source=cmap["t_source"],
        assay_t0_s=assay_t0, stim_order=",".join(stim_order) if stim_order else "",
        noise_sigma_median=float(np.median(sigma)),
        n_detected=len(det), n_opposite_polarity=len(opposite),
        n_matched=int(det["trial"].notna().sum()) if len(det) else 0,
        n_extra=int(fl.str.contains("extra").sum()),
        n_outside_slot=int(fl.str.contains("outside_slot").sum()),
        n_weak=int(fl.str.contains("weak").sum()),
        n_missing=int(fl.str.contains("missing").sum()),
        n_not_recorded=int(fl.str.contains("not_recorded").sum()),
        median_match_residual_s=float(det["match_residual_s"].median()) if len(det) and det["match_residual_s"].notna().any() else np.nan,
        params=prm, issues=[list(i) for i in issues])

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames.to_csv(out_dir / "pressure_frames.csv", index=False)
        ev_df.to_csv(out_dir / "stim_events.csv", index=False)
        (out_dir / "pressure_report.txt").write_text(report_text(summary, issues, ev_df, prm),
                                                     encoding="utf-8")
        with open(out_dir / "pressure_summary.json", "w", encoding="utf-8") as fh:
            json.dump(_json_clean(summary), fh, indent=2)
        if make_png:
            plot_qc(out_dir / "pressure_qc.png", rid, t, x, base, force, sigma, prm, ev_df, sched,
                    pulses, cmap, assay_t0, qc)
    return dict(frames=frames, events=ev_df, summary=summary, issues=issues, pulses=pulses,
                detections=events, force=force, sigma=sigma, t=t)


def _json_clean(o):
    """numpy scalars -> Python, NaN -> null (MATLAB jsondecode and strict parsers reject NaN)."""
    if isinstance(o, dict):
        return {k: _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def print_issues(rid, issues):
    for lvl, code, msg in issues:
        if lvl == "LOUD":
            print(f"!!! {rid} {code}: {msg}", file=sys.stderr)
        elif lvl == "WARN":
            print(f"  WARN {rid} {code}: {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- selftest

def synth_session(seed=0, lead_in_s=20.3, imaging_s=1700.0, assay_t0=5.0, K=0,
                  stim_order=("mild", "pin", "heat", "light"), arduino_ppm=354.0,
                  pulse_edits=None, trigger="manual"):
    """Synthetic logger session in the real format. Returns (csv_text, info, truth).

    Clocks: master frame f at f * Pm; pulse n at master time (n-1+K) * Pm; the Arduino clock runs
    arduino_ppm fast. Samples ~79.9 Hz with integer-ms stamps; each sample is the mean force over
    its conversion period (as the NAU7802 integrates) + drift + AR(1) noise (sd 100, lag-1 0.55)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    Pm = 1.0 / 30.0
    rate = 1.0 + arduino_ppm * 1e-6
    t_start = 100.0                                    # Arduino seconds since reset
    T1 = t_start + lead_in_s                           # Arduino time of pulse 1
    dur = lead_in_s + imaging_s + 2.5
    dt = 1.0 / 79.89
    ts_true = t_start + np.arange(int(dur / dt)) * dt + rng.uniform(0, 0.0004, int(dur / dt))
    ms = np.floor(ts_true * 1000.0).astype(np.int64)

    def ard(tm):                                       # master time -> Arduino time
        return T1 + (np.asarray(tm) - K * Pm) * rate

    n_pulses = int(imaging_s / Pm)
    pulse_t = ard((np.arange(1, n_pulses + 1) - 1 + K) * Pm)
    if pulse_edits:
        if "drop" in pulse_edits:
            pulse_t = np.delete(pulse_t, pulse_edits["drop"])
        if "glitch_after" in pulse_edits:
            pulse_t = np.sort(np.append(pulse_t, pulse_t[pulse_edits["glitch_after"]] + 0.004))
    count = np.searchsorted(pulse_t, ts_true, side="right")
    ttl = (np.searchsorted(pulse_t, ts_true, side="right") -
           np.searchsorted(pulse_t, ts_true - 0.010, side="right") > 0).astype(int)

    # taps on the schedule (master clock), then onto the Arduino clock
    fine_dt = 0.001
    fine_t = np.arange(t_start - 0.05, ts_true[-1] + 0.01, fine_dt)
    force = np.zeros(len(fine_t))
    sched = build_schedule(list(stim_order), assay_t0) if stim_order else None
    taps = []
    shape = dict(light=(2000, 3000, 0.010, 0.025, 0.6, 1.2), mild=(4000, 6000, 0.015, 0.030, 0.8, 1.5),
                 pin=(8000, 14000, 0.004, 0.008, 0.08, 0.2))
    if sched is not None:
        light_trials = sched.index[sched.stim_type == "light"]
        missing_idx = light_trials[4]                  # light block 2 trial 2: not delivered
        weak_idx = light_trials[9]                     # light block 4 trial 1: barely touched
        for i, s in sched.iterrows():
            if s.stim_type == "heat" or i == missing_idx:
                continue                               # a hot water drop gives no force
            taps.append(dict(kind=s.stim_type, t_master=s.t_expected + float(np.clip(rng.normal(0, 0.7), -2, 2)),
                             sched_index=int(s.sched_index), weak=(i == weak_idx)))
        pin_slot = 1 + list(stim_order).index("pin")
        pin_b3 = sched[(sched.slot == pin_slot) & (sched.block == 3) & (sched.trial == 2)].t_expected.iloc[0]
        taps.append(dict(kind="pin", t_master=pin_b3 + 7.5, sched_index=None, weak=False, extra=True))
        taps.append(dict(kind="mild", t_master=assay_t0 + SLOT_STARTS_S[0] + SLOT_LEN_S + 30.0,
                         sched_index=None, weak=False, extra=True, outside=True))
    for tap in taps:
        lo_a, hi_a, lo_r, hi_r, lo_h, hi_h = shape[tap["kind"]]
        A = 500.0 if tap["weak"] else rng.uniform(lo_a, hi_a)     # weak: 5 sigma
        tau_r = 0.015 if tap["weak"] else rng.uniform(lo_r, hi_r)
        hold = rng.uniform(lo_h, hi_h)
        t0 = float(ard(tap["t_master"]))
        tap.update(t0_arduino=t0, amplitude=A)
        i0 = int(np.searchsorted(fine_t, t0))
        tt = fine_t[i0:i0 + int((hold + 1.0) / fine_dt)] - t0
        rise = A * (1 - np.exp(-tt / tau_r)) * (1 + 0.04 * np.sin(2 * np.pi * 7 * tt) * (1 - np.exp(-tt / 0.1)))
        level = A * (1 - np.exp(-hold / tau_r))
        f = np.where(tt <= hold, rise, level * np.exp(-(tt - hold) / rng.uniform(0.03, 0.06)))
        force[i0:i0 + len(tt)] += f
    cs = np.concatenate([[0.0], np.cumsum(force)])
    hi = np.searchsorted(fine_t, ts_true, side="right")
    lo = np.searchsorted(fine_t, ts_true - dt, side="right")
    sampled = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    rel = ts_true - t_start
    drift = 3200 + 1.8 * rel + 150 * np.sin(2 * np.pi * rel / 300.0)
    rho = 0.55
    noise = lfilter([1.0], [1.0, -rho], rng.normal(0, 100 * np.sqrt(1 - rho ** 2), len(ts_true)))
    baseline = -17549.94
    raw = np.round(baseline + drift + sampled + noise).astype(np.int64)
    pc = (np.datetime64("2026-10-16T12:35:34.324") +
          ((ts_true - t_start) * (1 - 742e-6) * 1000).astype("timedelta64[ms]"))
    if trigger == "ttl":                               # session opens at the sample after pulse 1
        keep = count >= 1
        keep &= ts_true <= pulse_t[-1] + 2.0
    else:
        keep = np.ones(len(ts_true), bool)
    ms_k = ms[keep]
    out = pd.DataFrame(dict(
        t_s=[f"{v:.4f}" for v in (ms_k - ms_k[0]) / 1000.0], arduino_ms=ms_k,
        pc_time=np.datetime_as_string(pc[keep], unit="ms"), raw=raw[keep], baseline=baseline,
        corrected=np.round(raw[keep] - baseline, 2), indicator=0, ttl=ttl[keep],
        ttl_count=count[keep]))
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    info = dict(folder="synthetic", mouse="SYN", date="2026-10-16", session=1, trigger=trigger,
                n_samples=len(out), ttl_pulses=int(count[keep][-1]), ttl_mode="pulses",
                ttl_input=True, columns={c: "" for c in LOGGER_COLUMNS},
                duration_s=round(float((ms_k[-1] - ms_k[0]) / 1000.0), 3))
    for tap in taps:
        # truth: the contact time on the CSV's t_s axis, the ttl_count a sample taken at that
        # instant would carry, and how close the nearest pulse is
        tap["t0_ts"] = tap["t0_arduino"] - ms_k[0] / 1000.0
        tap["true_count"] = int(np.searchsorted(pulse_t, tap["t0_arduino"], side="right"))
        tap["pulse_margin_s"] = float(np.min(np.abs(pulse_t - tap["t0_arduino"])))
    truth = dict(taps=taps, Pm=Pm, K=K, rate=rate, n_pulses=len(pulse_t),
                 fps_arduino=1.0 / (Pm * rate), ppm=(1.0 / rate - 1.0) * 1e6,
                 t0_ms=ms_k[0], T1=T1, t_start=t_start, assay_t0=assay_t0,
                 stim_order=list(stim_order) if stim_order else None,
                 frames_time=pd.Series(np.arange(int(imaging_s / Pm) + 10) * Pm,
                                       index=np.arange(int(imaging_s / Pm) + 10)))
    return buf.getvalue(), info, truth


def selftest() -> int:
    failures = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            failures.append(msg)

    out_root = HERE / "out" / "selftest" / "pressure_sync"
    print("1. synthetic 28-min assay, manual start after a 20.3 s lead-in")
    csv_text, info, truth = synth_session(seed=7)
    res = process_session(io.StringIO(csv_text), info, "SYN_d1_pre", truth["stim_order"],
                          truth["assay_t0"], K=0, out_dir=out_root / "no_frame_times")
    s, ev, fr = res["summary"], res["events"], res["frames"]
    check(s["n_ttl"] == truth["n_pulses"], f"pulses counted {s['n_ttl']} == {truth['n_pulses']}")
    check(s["n_missed"] == 0 and s["n_glitch"] == 0 and s["n_multi_increment"] == 0,
          f"clean pulse train: missed {s['n_missed']}, glitch {s['n_glitch']}, multi {s['n_multi_increment']}")
    rel = abs(s["fps_arduino"] / truth["fps_arduino"] - 1)
    check(rel < 2e-6, f"fps on Arduino clock {s['fps_arduino']:.6f} vs truth {truth['fps_arduino']:.6f} ({rel * 1e6:.2f} ppm)")
    check(abs(s["fps_naive"] / truth["fps_arduino"] - 1) > 0.01,
          f"pulses / duration ({s['fps_naive']:.4f}) is wrong because of the lead-in: fit needed")
    check(any(i[1] == "manual_start" for i in res["issues"]) and ev["flags"].str.contains("manual_start").all(),
          "manual start is a LOUD issue and is on every stim_events row")
    check(list(fr.columns[:len(FRAMES_COLUMNS)]) == FRAMES_COLUMNS and len(fr) == truth["n_pulses"]
          and (fr["isx_frame"] == fr["pulse"]).all(), "pressure_frames: contract columns, one row per pulse, isx_frame = pulse + K (K=0)")
    check(list(ev.columns[:len(EVENTS_COLUMNS)]) == EVENTS_COLUMNS, "stim_events: contract columns in order")

    det = ev[ev["source"] == "pressure"].reset_index(drop=True)
    taps = truth["taps"]
    check(len(det) == len(taps), f"detections {len(det)} == taps delivered {len(taps)}")
    on_t = det["onset_t_arduino_s"].to_numpy()
    half = 0.5 * float(np.median(np.diff(res["t"])))
    t_err, t_err_weak, wrong_frame, wrong_assign = [], [], [], []
    for tap in taps:
        k = int(np.argmin(np.abs(on_t - tap["t0_ts"])))
        r = det.iloc[k]
        (t_err_weak if tap["weak"] else t_err).append(r["onset_t_arduino_s"] - tap["t0_ts"])
        # The frame must be the true contact's (count + K). Only a contact close to a pulse can
        # land one frame off, and then the row must say so: its onset interval holds that pulse
        # (pulse_ambiguous). A weak tap's onset is only good to ~2 samples.
        c_err = int(r["onset_isx_frame"]) - tap["true_count"]
        near = tap["pulse_margin_s"] < 2 * half or tap["weak"]
        if abs(c_err) > 1 or (c_err and not near) or (c_err and "pulse_ambiguous" not in r["flags"]):
            wrong_frame.append(f"contact {tap['pulse_margin_s'] * 1000:.1f} ms from a pulse: "
                               f"frame {int(r['onset_isx_frame'])}, true {tap['true_count']} ({r['flags']})")
        if tap.get("extra"):
            fl = r["flags"].split(";")
            if "extra" not in fl or ("outside_slot" in fl) != bool(tap.get("outside")) \
                    or pd.notna(r["trial"]):
                wrong_assign.append(f"extra tap at {tap['t_master']:.1f}: {r['flags']}")
        else:
            got = int(r["event_id"].split("_e")[-1])
            if got != tap["sched_index"] or "extra" in r["flags"]:
                wrong_assign.append(f"trial {tap['sched_index']} -> {r['event_id']} {r['flags']}")
            if ("weak" in r["flags"]) != tap["weak"]:
                wrong_assign.append(f"weak flag wrong for trial {tap['sched_index']}: {r['flags']}")
    t_err, t_err_weak = np.abs(t_err), np.abs(t_err_weak)
    check(t_err.max() < 2 * half and np.median(t_err) < 0.004,
          f"onset times of the {len(t_err)} normal taps: median |error| {np.median(t_err) * 1000:.1f} ms, "
          f"max {t_err.max() * 1000:.1f} ms (want < 4 ms / one sample); the knee is sub-sample")
    check(t_err_weak.max() < 5 * half, f"onset of the weak 500-count (5 sigma) tap within 2.5 samples "
          f"({t_err_weak.max() * 1000:.1f} ms); near threshold the rise is too shallow for the knee")
    n_amb = int(det["flags"].str.contains("pulse_ambiguous").sum())
    check(not wrong_frame, f"onset frame = the true contact's frame ({len(taps)} taps, {n_amb} "
          f"flagged pulse_ambiguous; a flagged one next to a pulse may be +-1): "
          + ("all right" if not wrong_frame else "; ".join(wrong_frame[:4])))
    check(not wrong_assign, "assignment to slot/block/trial, extra, outside_slot, weak: "
          + ("all right" if not wrong_assign else "; ".join(wrong_assign[:5])))
    exp = ev[ev["source"] == "expected"]
    heat = exp[exp["stim_type"] == "heat"]
    check(len(heat) == TRIALS_PER_SLOT and heat["flags"].str.startswith("missing").all(),
          f"15 heat trials emitted as source=expected/missing ({len(heat)})")
    lmiss = exp[exp["stim_type"] == "light"]
    check(len(lmiss) == 1 and int(lmiss["block"].iloc[0]) == 2 and int(lmiss["trial"].iloc[0]) == 2,
          "the undelivered light trial (block 2, trial 2) is source=expected/missing")
    check(len(ev) == 60 + 2, f"stim_events rows {len(ev)} == 60 scheduled + 2 extra")
    # CONTRACTS.md 1a: the onset's count c (pulses before it, from the written
    # t_pulse_arduino_s, as master_timeline recomputes it) gives isx_frame c + K and camera frame c
    c_on = np.array([arduino_time_to_pulse(x, fr["pulse"].to_numpy(), fr["t_pulse_arduino_s"].to_numpy())
                     for x in on_t])
    check(det["onset_isx_frame"].notna().all()
          and (det["onset_isx_frame"].to_numpy(np.int64) == c_on).all()
          and (det["onset_cam_frame"].to_numpy(np.int64) == c_on).all(),
          "onset_isx_frame = count before the onset + K (K=0), onset_cam_frame = that count")
    near_tap = [int(np.argmin(np.abs(on_t - tp["t0_ts"]))) for tp in taps]
    t_err = np.array([det["onset_t_master_s"].iloc[k] - tp["t_master"] for k, tp in zip(near_tap, taps)])
    bound = (truth["rate"] - 1) * 1700.0 + 0.03
    check(np.abs(t_err).max() < bound,
          f"fallback clock (Arduino period, 354 ppm fast): onset_t_master_s error max "
          f"{np.abs(t_err).max():.3f} s < {bound:.3f} s (clock-rate error only)")
    check((out_root / "no_frame_times" / "pressure_qc.png").exists(), "QC PNG written")

    print("2. same session with a frames-time table (master clock)")
    res2 = process_session(io.StringIO(csv_text), info, "SYN_d1_pre", truth["stim_order"],
                           truth["assay_t0"], K=0, frames_time=truth["frames_time"],
                           out_dir=out_root / "frame_times", make_png=False)
    s2, ev2 = res2["summary"], res2["events"]
    check(abs(s2["arduino_fit_ppm"] - truth["ppm"]) < 2.0,
          f"Arduino->master fit {s2['arduino_fit_ppm']:.1f} ppm vs truth {truth['ppm']:.1f}, residual {s2['arduino_fit_resid_ms']:.2f} ms")
    det2 = ev2[ev2["source"] == "pressure"]
    errs = []
    for tap in taps:
        k = int(np.argmin(np.abs(det2["onset_t_arduino_s"].to_numpy() - tap["t0_ts"])))
        errs.append(det2["onset_t_master_s"].iloc[k] - tap["t_master"])
    errs = np.abs(np.array(errs))
    normal = np.array([not tp["weak"] for tp in taps])
    check(errs[normal].max() < 0.012, f"onset_t_master_s within 12 ms of the true contact time "
          f"(max {errs[normal].max() * 1000:.1f} ms, median {np.median(errs[normal]) * 1000:.1f} ms)")
    res2k = process_session(io.StringIO(csv_text), info, "SYN_d1_pre", truth["stim_order"],
                            truth["assay_t0"], K=2, make_png=False)
    d2k = res2k["events"][res2k["events"]["source"] == "pressure"]
    check(((d2k["onset_isx_frame"] - d2k["onset_cam_frame"]) == 2).all()
          and (res2k["frames"]["isx_frame"] == res2k["frames"]["pulse"] + 2).all(),
          "K = 2: onset_isx_frame = count + K and pressure_frames isx_frame = pulse + K")

    print("3. pulse QC: TTL-started session, one dropped pulse, one 4-ms glitch, K = 3")
    csv3, info3, tr3 = synth_session(seed=3, lead_in_s=0.0, imaging_s=60.0, stim_order=None,
                                     K=3, pulse_edits=dict(drop=[900], glitch_after=400), trigger="ttl")
    res3 = process_session(io.StringIO(csv3), info3, "SYN_pulses", None, 0.0, K=3,
                           out_dir=out_root / "pulse_qc", make_png=True)
    s3, fr3 = res3["summary"], res3["frames"]
    check(s3["n_missed"] == 1, f"missed pulses found: {s3['n_missed']} (1 dropped)")
    check(s3["n_glitch"] == 1, f"glitches found: {s3['n_glitch']} (1 inserted)")
    check(s3["n_open_bracket"] == 1 and not any(i[1] == "manual_start" for i in res3["issues"]),
          "TTL start: pulse 1 bracket open, no manual_start flag")
    check(abs(s3["fps_arduino"] / tr3["fps_arduino"] - 1) < 5e-5,
          f"fps despite miss + glitch: {s3['fps_arduino']:.5f} vs {tr3['fps_arduino']:.5f}")
    check((fr3["isx_frame"] == fr3["pulse"] + 3).all(), "pressure_frames isx_frame = pulse + K with K = 3")
    check(any(i[1] == "missed_pulses" and i[0] == "LOUD" for i in res3["issues"]), "missed pulse is a LOUD issue")

    print("4. contract violations fail loudly")
    bad = pd.read_csv(io.StringIO(csv3)).drop(columns=["ttl_count"]).to_csv(index=False)
    try:
        read_logger_csv(io.StringIO(bad))
        check(False, "missing ttl_count raises ContractError")
    except ContractError:
        check(True, "missing ttl_count raises ContractError")
    dec = pd.read_csv(io.StringIO(csv3))
    dec.loc[500:, "ttl_count"] -= 5
    try:
        read_logger_csv(io.StringIO(dec.to_csv(index=False)))
        check(False, "decreasing ttl_count raises ContractError")
    except ContractError:
        check(True, "decreasing ttl_count raises ContractError")
    try:
        parse_stim_order("light,mild,heat")
        check(False, "incomplete stim_order raises ContractError")
    except ContractError:
        check(True, "incomplete stim_order raises ContractError")
    for what, text in (("a 0-byte CSV", ""), ("a CSV with a broken row", 't_s,ttl_count\n1,"2\n')):
        try:
            read_logger_csv(io.StringIO(text))
            check(False, f"{what} raises ContractError")
        except ContractError:
            check(True, f"{what} raises ContractError (exit 2), not a traceback")

    print("5. contacts and releases 2 ms from a pulse (30 Hz pulses, ~12.5-ms samples)")
    # A contact 2 ms after pulse n happened during frame n + K; the last baseline sample before
    # it still has count n-1, so stamping the onset with that sample put it a frame early.
    P, T1 = 1 / 30.0, 0.5
    pulse_t = T1 + np.arange(2000) * P
    # ~80 Hz samples whose clock is not locked to the DAQ (the NAU7802 runs on its own
    # oscillator; bench sessions show the uniform 3.6-ms quantisation this gives). A sample
    # clock locked at exactly 12.5 ms would repeat 3 pulse phases forever, and the pulse grid
    # fitted to bracket midpoints would then carry a fixed bias of up to ~2 ms.
    rng = np.random.default_rng(5)
    ts = pulse_t[0] + 0.004 + np.concatenate([[0], np.cumsum(rng.uniform(0.012, 0.013, 6000))])
    ts = ts[ts < pulse_t[-1] + 2.0]
    on_p, off_p = (752, 1203, 1654), (770, 1221, 1672)        # contact / release pulse numbers
    f = np.zeros(ts.size)
    for a_, b_ in zip(on_p, off_p):
        t_on, t_off = pulse_t[a_ - 1] + 0.002, pulse_t[b_ - 1] - 0.002   # 2 ms after / before
        # 30-ms rise and fall: more than two samples on each edge, as the NAU7802's averaging
        # over each conversion also gives a sharper press
        f += 3000 * np.clip((ts - t_on) / 0.030, 0, 1) * np.clip((t_off - ts) / 0.030, 0, 1)
    corr = np.round(f + rng.normal(0, 20, ts.size), 2)
    ms = np.round(ts * 1000).astype(np.int64)
    edge = pd.DataFrame(dict(t_s=(ms - ms[0]) / 1000.0, arduino_ms=ms, pc_time="", raw=0,
                             baseline=0.0, corrected=corr, indicator=0, ttl=0,
                             ttl_count=np.searchsorted(pulse_t, ts, side="right")))
    res5 = process_session(io.StringIO(edge.to_csv(index=False)),
                           dict(trigger="ttl", ttl_mode="pulses"), "SYN_edge", None, 0.0, K=0,
                           make_png=False)
    d5 = res5["events"][res5["events"]["source"] == "pressure"].reset_index(drop=True)
    p5 = res5["frames"]
    off_count = [arduino_time_to_pulse(x, p5["pulse"].to_numpy(), p5["t_pulse_arduino_s"].to_numpy())
                 for x in d5["offset_t_arduino_s"]]
    check(len(d5) == 3 and list(d5["onset_isx_frame"]) == list(on_p),
          f"onset 2 ms after pulse n -> isx_frame n + K: {list(d5['onset_isx_frame'])} (want {list(on_p)})")
    check(off_count == [p - 1 for p in off_p],
          f"release 2 ms before pulse n -> last stim frame n-1+K: {off_count} (want {[p - 1 for p in off_p]})")
    check(d5["flags"].str.contains("pulse_ambiguous").all(),
          "an onset 2 ms from a pulse (within half a sample) is flagged pulse_ambiguous")
    # a faint press is timed less well: its onset interval is wider than half a sample
    faint = edge.assign(corrected=np.round(f / 15.0 + rng.normal(0, 20, ts.size), 2))
    r6 = process_session(io.StringIO(faint.to_csv(index=False)), dict(trigger="ttl", ttl_mode="pulses"),
                         "SYN_edge", None, 0.0, K=0, make_png=False)
    ci_w = [ev["ci"][1] - ev["ci"][0] for ev in r6["detections"]]
    ci_w5 = [ev["ci"][1] - ev["ci"][0] for ev in res5["detections"]]
    check(len(ci_w) == 3 and min(ci_w) > max(ci_w5) and min(ci_w5) >= 0.012,
          f"onset interval: faint 10-sigma press {np.round(np.array(ci_w) * 1000, 1)} ms wider than "
          f"firm 150-sigma press {np.round(np.array(ci_w5) * 1000, 1)} ms (>= one sample)")

    print("6. stim_order, manual-start wording, exit codes, one recording per folder")
    swapped = [truth["stim_order"][i] for i in (0, 1, 3, 2)]       # heat and light swapped
    res6 = process_session(io.StringIO(csv_text), info, "SYN_d1_pre", swapped, truth["assay_t0"],
                           K=0, make_png=False)
    check(any(i[0] == "LOUD" and i[1] == "stim_order" for i in res6["issues"]),
          "heat slot with touch-like logger contacts -> LOUD stim_order issue")
    check(not any(i[1] == "stim_order" for i in res["issues"]), "the right stim_order raises nothing")
    msg = next(m for _, c, m in res["issues"] if c == "manual_start")
    check("ran 20.30 s without pulses" in msg, f"long quiet lead-in: pulse 1 probably frame 0 ({msg[:90]}...)")
    late = pd.read_csv(io.StringIO(csv_text))
    late = late[late["ttl_count"] >= 600].assign(ttl_count=lambda x: x["ttl_count"] - 599)
    res7 = process_session(io.StringIO(late.to_csv(index=False)), info, "SYN_late", None, 0.0, K=0,
                           make_png=False)
    msg7 = next(m for _, c, m in res7["issues"] if c == "manual_start")
    check("already running" in msg7, f"Start pressed into a running pulse train is not reassuring ({msg7[:90]}...)")
    folder = out_root / "cli"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "S_pressure.csv").write_text(csv3, encoding="utf-8")
    code = main(["--csv", str(folder / "S_pressure.csv"), "--recording-id", "SYN_cli", "--no-png",
                 "--pulse-offset", "0", "--out-dir", str(folder / "out")])
    check(code == 1, f"CLI exit code 1 when a LOUD issue is present (missed pulses): {code}")
    try:
        process_session(io.StringIO(csv3), info3, "SYN_other", None, 0.0, K=0,
                        out_dir=folder / "out", make_png=False)
        check(False, "a folder holding another recording's tables is refused")
    except ContractError:
        check(True, "a folder holding another recording's tables is refused")

    print(f"selftest outputs in {out_root}")
    print("SELFTEST " + ("PASSED" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


# ----------------------------------------------------------------------------- CLI

def manifest_row(path, rid):
    m = pd.read_csv(path, dtype=str).fillna("")
    rows = m[m["recording_id"] == rid]
    if len(rows) != 1:
        raise ContractError(f"{path}: {len(rows)} rows for recording_id {rid}")
    return rows.iloc[0], Path(path).resolve().parent


def expand_csvs(items):
    out = []
    for it in items:
        p = Path(it)
        out += sorted(p.glob("*_pressure.csv")) if p.is_dir() else [p]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--csv", nargs="+", help="*_pressure.csv file(s) or folder(s) of them")
    ap.add_argument("--info", help="*_info.json (default: next to the CSV)")
    ap.add_argument("--manifest", help="manifest.csv; fills csv/info/stim_order/assay_t0 for --recording-id")
    ap.add_argument("--recording-id", help="default: the CSV file stem")
    ap.add_argument("--stim-order", help="e.g. light,mild,heat,pin (slot 1..4)")
    ap.add_argument("--assay-t0", type=float, help="t_master_s of the baseline start (default 0)")
    ap.add_argument("--pulse-offset", type=int, default=None,
                    help="K: pulse n ends isx_frame n-1+K, ttl_count n falls in isx_frame n+K "
                         "(default: the manifest's pulse_to_frame_offset, else 0 with a warning)")
    ap.add_argument("--frames-time", help="CSV with isx_frame, t_master_s (optional recording_id)")
    ap.add_argument("--period-s", type=float, help="master-clock frame period if no frame-time table")
    ap.add_argument("--out-dir", help="default: <pipeline>/out/pressure_sync/<recording_id>")
    ap.add_argument("--k-sigma", type=float, default=DEFAULTS["k_sigma"])
    ap.add_argument("--min-force", type=float, default=DEFAULTS["min_force"], help="counts")
    ap.add_argument("--min-hold", type=int, default=DEFAULTS["min_hold"], help="samples")
    ap.add_argument("--baseline-window-s", type=float, default=DEFAULTS["baseline_window_s"])
    ap.add_argument("--weak-factor", type=float, default=DEFAULTS["weak_factor"])
    ap.add_argument("--tolerance-s", type=float, default=DEFAULTS["tolerance_s"])
    ap.add_argument("--polarity", type=int, choices=(1, -1), default=DEFAULTS["polarity"])
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    prm = dict(k_sigma=a.k_sigma, min_force=a.min_force, min_hold=a.min_hold,
               baseline_window_s=a.baseline_window_s, weak_factor=a.weak_factor,
               tolerance_s=a.tolerance_s, polarity=a.polarity)
    try:
        stim_order, assay_t0, csvs, info_arg = a.stim_order, a.assay_t0, a.csv, a.info
        K = a.pulse_offset
        if a.manifest:
            if not a.recording_id:
                ap.error("--manifest needs --recording-id")
            row, root = manifest_row(a.manifest, a.recording_id)
            stim_order = stim_order or row.get("stim_order")
            if assay_t0 is None and row.get("assay_t0_s", "") != "":
                assay_t0 = float(row["assay_t0_s"])
            if not csvs and row.get("pressure_csv"):
                csvs = [str(root / row["pressure_csv"])]
            if not info_arg and row.get("pressure_json"):
                info_arg = str(root / row["pressure_json"])
            k_text = str(row.get("pulse_to_frame_offset", "")).strip()
            if K is None and k_text != "":
                try:
                    K = int(k_text)        # as manifest.py: a K of '0.9' is an error, not 0
                except ValueError:
                    raise ContractError(f"manifest pulse_to_frame_offset '{k_text}' must be an integer")
        if K is None:
            # CONTRACTS.md: K is never assumed silently
            print("WARNING: K (pulse_to_frame_offset) not given: 0 used; fix it with the rig test "
                  "recording (--pulse-offset or the manifest column)", file=sys.stderr)
            K = 0
        if not csvs:
            ap.error("give --csv, --manifest or --selftest")
        stim_order = parse_stim_order(stim_order)
        files = expand_csvs(csvs)
        if len(files) > 1 and (a.recording_id or info_arg or a.out_dir):
            ap.error("--recording-id/--info/--out-dir apply to a single CSV")
        n_loud = 0
        for csv_path in files:
            rid = a.recording_id or csv_path.name.replace("_pressure.csv", "")
            info_p = Path(info_arg) if info_arg else info_path_for(csv_path)
            ft = read_frames_time(a.frames_time, rid) if a.frames_time else None
            out_dir = Path(a.out_dir) if a.out_dir else HERE / "out" / "pressure_sync" / rid
            res = process_session(csv_path, read_info(info_p), rid, stim_order,
                                  assay_t0 if assay_t0 is not None else 0.0, K, ft,
                                  a.period_s, prm, out_dir, not a.no_png, str(csv_path),
                                  str(info_p) if info_p.exists() else "")
            s = res["summary"]
            print(f"{rid}: {s['n_ttl']} pulses, {s['fps_arduino']:.4f} Hz (Arduino), "
                  f"missed {s['n_missed']}, glitch {s['n_glitch']}, {s['n_detected']} contacts "
                  f"({s['n_matched']} matched, {s['n_extra']} extra), {s['n_missing']} expected-only "
                  f"-> {out_dir}")
            print_issues(rid, res["issues"])
            n_loud += any(i[0] == "LOUD" for i in res["issues"])
    except ContractError as e:
        print(f"CONTRACT VIOLATION: {e}", file=sys.stderr)
        return 2
    if n_loud:
        # the tables and the report are written for QC, but a LOUD issue means the frame
        # mapping may be wrong, so a script must not carry on as if all were well
        print(f"{n_loud} of {len(files)} session(s) have LOUD sync issues (exit 1)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
