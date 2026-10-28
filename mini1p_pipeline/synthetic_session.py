"""Synthetic mini1p recording with KNOWN ground truth, in the real file formats.

Writes, per variant, everything the pipeline reads for one recording:
  inscopix/<rid>_video.isxd       tiny raw-IDAS-like movie (1x1 px) with a real footer
                                  (numTimes incl. dropped frames, dropped list, nominal period)
  inscopix/<rid>_raw_tsc.csv      IDPS frame-timestamp export, TSC in us (0 = dropped frame)
  inscopix/<rid>_gpio.csv         IDPS GPIO export with 'BNC Sync Output' change events
  pressure/<mouse>/<date>/<mouse>_<date>_S##_pressure.csv + _info.json
                                  logger session in the vendor/pressure_logger format: ~80 Hz
                                  Arduino-ms samples, ttl_count, contact taps for light / mild /
                                  pin at schedule times with jitter, none for heat
  cameras/<rid>_cam1.avi, _cam2.avi
                                  tiny uncompressed AVIs, one frame per SYNC pulse (pixel value =
                                  pulse number mod 250, so a frame can be identified)
  scoring/ScoringAB_<rid>_cam1.mat
                                  dual-view-scorer-like file: deliveries (taps ~0.3 s late),
                                  withdrawals / flinches, affective bouts, an uncertain stretch
  extract/<mouse>_d<day>_frame_table.csv
                                  extract_1p.m format (one row per raw frame of the day)
  truth.json                      the ground truth every pipeline output must reproduce
and out/synthetic/manifest.csv (one row per variant) plus out/synthetic/recordings_expected.csv
(the recordings.csv rows count_check / master_timeline must reproduce).

Physics simulated (CONTRACTS.md section 1a, revised 2026-10-19): SYNC pulse n is emitted at the
end of exposure of isx_frame n-1+K, at that frame's TSC timestamp (+0.4 ms, as the 2024 GPIO
first edges 0.0-0.6 ms after the movie start); the logger counts rising edges (ttl_count of a
sample = pulses so far); the cameras save one frame per pulse, exposed right after it. So logger
rows with ttl_count n and camera frame n fall DURING isx_frame n + K, and the ground truth is
written with master_timeline.count_to_isx_frame. The Arduino clock runs fast (ppm), its samples
are 12-14 ms apart, pc_time drifts; frames are ~20.015 Hz with 2 early (trimmed) and 3 dropped
frames. K = 0 (the 2024 value): the contract's count rule (n_frames_isx + K) and index rule
(n-1+K) disagree for K != 0, so no K != 0 session is simulated.

Contacts are placed uniformly across the pulse interval, so some fall within one logger sample
(<= 14 ms) of a pulse, where no detector sampling at ~80 Hz can always tell on which side of the
pulse they were. truth.json records each contact's distance to the nearest pulse
(pulse_margin_s) and its release time, so a test can demand the exact frame everywhere else
and allow +-1 frame only next to a pulse.

Variants
  good          everything consistent (must pass, every event exact)
  cam_drop      cam1 loses 3 frames (pulses 11000, 11001, 23000): must FAIL loudly unless
                --allow-mismatch with anchors (anchors_cam1.csv = verified contact frames)
  manual_start  logger started by hand 37 s into the recording: ttl_count misses the first
                pulses, so it must FAIL loudly

Usage
  python synthetic_session.py [--out out/synthetic] [--variant good cam_drop manual_start]
  python synthetic_session.py --selftest
"""

import argparse
import csv
import json
import os
import struct
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.io as sio

import manifest as mf
import master_timeline as mt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "out", "synthetic")

VARIANTS = {
    "good": dict(mouse="SYN01", sex="F", day=1, assay="post", arm="A", session=2,
                 stim_order="mild,heat,light,pin", seed=11, cam1_lost=(), manual_start_s=None,
                 day_prefix_frames=1000),
    "cam_drop": dict(mouse="SYN02", sex="M", day=1, assay="pre", arm="none", session=1,
                     stim_order="light,mild,heat,pin", seed=12, cam1_lost=(11000, 11001, 23000),
                     manual_start_s=None, day_prefix_frames=0),
    "manual_start": dict(mouse="SYN03", sex="F", day=2, assay="pre", arm="none", session=1,
                         stim_order="pin,light,mild,heat", seed=13, cam1_lost=(),
                         manual_start_s=37.0, day_prefix_frames=0),
}

DEFAULT_FPS = 20.015         # simulated DAQ frame rate (the 2024 nVista3 measured rate); --fps
SAMPLE_GAP_MAX_S = 0.0145    # longest logger sample interval (14 ms) plus margin: "one sample"
NOMINAL_PERIOD_US = 50000    # what a raw IDAS footer stores (nominal, not the true period)
N_TRIM = 2                   # early frames removed by IDPS preprocessing
DROPPED = (5003, 17011, 29017)
ASSAY_T0_S = 12.0
TAIL_S = 8.0
K = 0
CAM_LEAD = 1                 # documentation column only (superseded by count_to_isx_frame)
SYNC_DELAY_S = 0.0004        # SYNC edge after the frame's TSC timestamp (2024 GPIO: 0.0-0.6 ms)
ARDUINO_PPM = 354.0          # Arduino clock fast vs the DAQ (bench estimate)
PC_PPM = -742.0              # PC clock vs the Arduino (bench measurement)
BASELINE_TARE = -17549.94
STALE_OFFSET = 3240.0
NOISE_SD = 60.0
AMP = {"light": 900.0, "mild": 2000.0, "pin": 3500.0}
HOLD_S = {"light": (0.8, 1.2), "mild": (1.0, 1.5), "pin": (0.15, 0.35)}
P_WITHDRAW = {"light": 0.3, "mild": 0.6, "heat": 0.8, "pin": 0.9}
STIM_NAMES = ["Light touch", "Mild touch", "Heat", "Pin prick"]     # scorer default keys 1-4
KEY_OF = {"light": 1, "mild": 2, "heat": 3, "pin": 4}
DATE = "2026-10-20"


def rid_of(v):
    return mf.recording_id_for(v["mouse"], v["day"], v["assay"])


# ----------------------------------------------------------------------------- writers
def write_fake_isxd(path, n_frames, dropped, start_epoch_ms):
    """Raw-IDAS-like .isxd: N frames of 1x1 px (+2 header and 2 footer rows, uint16) then
    the JSON footer, a NUL byte and the footer length (layout of the 2024 recordings)."""
    footer = {
        "dataType": 0, "type": 0, "fileVersion": 1, "hasFrameHeaderFooter": True,
        "producer": {"name": "isxcore", "version": [1, 2, 1]},
        "extraProperties": {"adMode": "auto", "auto": {"clock": 5000, "mScope": {"intTime": 49.9}},
                            "microscope": {"type": "NVista3", "serial": 0, "fps": {"fps": 20},
                                           "exp": 50, "gain": 6.3},
                            "producer": {"name": "AcquisitionSW", "version": "synthetic"}},
        "spacingInfo": {"numPixels": {"x": 1, "y": 1},
                        "pixelSize": {"x": {"num": 3, "den": 1}, "y": {"num": 3, "den": 1}}},
        "timingInfo": {"numTimes": int(n_frames), "dropped": [int(d) for d in dropped],
                       "cropped": [], "blank": [],
                       "period": {"num": NOMINAL_PERIOD_US, "den": 1000000},
                       "start": {"secsSinceEpoch": {"num": int(start_epoch_ms), "den": 1000},
                                 "utcOffset": 0}},
    }
    blob = json.dumps(footer, indent=4).encode("utf-8")
    with open(path, "wb") as f:
        f.write(bytes(n_frames * 1 * (1 + 4) * 2))
        f.write(blob)
        f.write(b"\x00")
        f.write(struct.pack("<Q", len(blob)))


def write_tiny_video(path, pulses, fps):
    """One 4x4 uncompressed frame per saved pulse; pixel value = pulse mod 250."""
    import cv2
    w = cv2.VideoWriter(path, 0, fps, (4, 4), True)
    if not w.isOpened():
        raise RuntimeError(f"cv2 cannot write {path}")
    for p in pulses:
        w.write(np.full((4, 4, 3), int(p) % 250, np.uint8))
    w.release()


# ----------------------------------------------------------------------------- one recording
def generate(out_root, variant, seed=None, fps=DEFAULT_FPS):
    """Generate one variant under out_root/<variant>; returns the truth dict."""
    v = VARIANTS[variant]
    rng = np.random.default_rng(v["seed"] if seed is None else seed)
    rid = rid_of(v)
    root = os.path.join(out_root, variant)
    dirs = {k: os.path.join(root, k) for k in ("inscopix", "pressure", "cameras", "scoring", "extract")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    stim_order = mf.parse_stim_order(v["stim_order"])
    period = 1.0 / fps

    # ---- Inscopix frames and SYNC pulses (true DAQ time, s; frame 0 at 0)
    n = int(np.ceil((ASSAY_T0_S + mt.ASSAY_END_S + TAIL_S) * fps))
    ts = np.arange(n) * period + rng.normal(0, 20e-6, n)
    ts[0] = 0.0
    n_pulses = n - K              # pulses for frames K..n-1 (K = 0, so also n + K)
    # pulse n ends isx_frame n-1+K, at that frame's TSC timestamp (CONTRACTS.md 1a)
    pulse_t = ts[np.arange(n_pulses) + K] + SYNC_DELAY_S
    tsc0 = 912_345_678_901
    tsc_us = np.round(tsc0 + ts * 1e6).astype(np.int64)
    t_master = (tsc_us - tsc_us[0]) / 1e6                         # what the pipeline will see

    start_epoch_ms = int(datetime(2026, 10, 20, 17, 0, 0).timestamp() * 1000)
    isxd = os.path.join(dirs["inscopix"], f"{rid}_video.isxd")
    write_fake_isxd(isxd, n, DROPPED, start_epoch_ms)
    tsc_out = tsc_us.copy()
    tsc_out[list(DROPPED)] = 0                                    # IDPS: no timestamp
    ts_csv = os.path.join(dirs["inscopix"], f"{rid}_raw_tsc.csv")
    pd.DataFrame({"Global Frame Number": np.arange(n), "Movie Number": 0,
                  "Local Frame Number": np.arange(n),
                  "Frame Timestamp (us)": tsc_out}).to_csv(ts_csv, index=False)
    # GPIO export: change events relative to the GPIO start, which is 79 ms before the movie
    g0 = 0.079
    gt = np.concatenate([[0.0], g0 + pulse_t, g0 + pulse_t + 0.010])
    gv = np.concatenate([[0], np.ones(n_pulses, int), np.zeros(n_pulses, int)])
    order = np.argsort(gt, kind="stable")
    gpio = pd.DataFrame({"Time (s)": np.round(gt[order], 6), "Channel Name": "BNC Sync Output",
                         "Value": gv[order]})
    extra = pd.DataFrame({"Time (s)": [0.0, g0 + pulse_t[-1] + 0.2],
                          "Channel Name": ["EX-LED", "BNC Trigger Input"], "Value": [0.2, 0]})
    gpio_csv = os.path.join(dirs["inscopix"], f"{rid}_gpio.csv")
    pd.concat([extra.iloc[:1], gpio, extra.iloc[1:]]).to_csv(gpio_csv, index=False)

    # ---- schedule, contact times, cameras
    sched = mt.schedule_trials(ASSAY_T0_S, stim_order)
    cam1_pulses = np.array([p for p in range(1, n_pulses + 1) if p not in set(v["cam1_lost"])])
    cam2_pulses = np.arange(1, n_pulses + 1)
    cam_of_pulse = {int(p): i + 1 for i, p in enumerate(cam1_pulses)}

    def cam1_frame_of_pulse(p):
        while p not in cam_of_pulse:           # a lost frame: the contact shows in the next one
            p += 1
        return cam_of_pulse[p]

    trials = []
    for _, s in sched.iterrows():
        target = s["t_expected_master_s"] + rng.uniform(-1.0, 1.0)
        p = int(np.argmin(np.abs(pulse_t - target))) + 1           # pulse n near the target
        # anywhere between pulse p and p+1, so the contact's samples have ttl_count p
        t_c = pulse_t[p - 1] + rng.uniform(0.0, 1.0) * (pulse_t[p] - pulse_t[p - 1])
        trials.append(dict(slot=int(s["slot"]), block=int(s["block"]), trial=int(s["trial"]),
                           stim_type=s["stim_type"], t_contact=float(t_c), pulse=p,
                           pulse_margin_s=float(min(t_c - pulse_t[p - 1], pulse_t[p] - t_c)),
                           cam1=cam1_frame_of_pulse(p)))

    # ---- logger samples (Arduino clock), TTL count, force
    first_t = -3.0
    last_t = pulse_t[-1] + 4.0
    ms0 = 600_000
    steps = rng.choice([12, 13, 14], size=int((last_t - first_t) / 0.012) + 10, p=[0.55, 0.38, 0.07])
    ms = ms0 + np.concatenate([[0], np.cumsum(steps)])
    true_t = first_t + (ms - ms0) / 1000.0 / (1 + ARDUINO_PPM * 1e-6)
    keep = true_t <= last_t
    ms, true_t = ms[keep], true_t[keep]
    count = np.searchsorted(pulse_t, true_t, side="right")         # pulses so far
    last_pulse = np.where(count > 0, pulse_t[np.maximum(count - 1, 0)], -np.inf)
    ttl = ((true_t - last_pulse) < 0.010).astype(int)

    force = np.zeros(true_t.size)
    for tr in trials:
        if tr["stim_type"] == "heat":
            continue                                               # a water drop: no load
        a = AMP[tr["stim_type"]] * rng.uniform(0.85, 1.15)
        hold = rng.uniform(*HOLD_S[tr["stim_type"]])
        # linear rise over 30 ms from the contact instant, hold, release over 60 ms
        u = true_t - tr["t_contact"]
        on = (u >= 0) & (u < hold)
        force[on] += a * np.minimum(1.0, u[on] / 0.03) * (1 + rng.normal(0, 0.02, on.sum()))
        rel = (u >= hold) & (u < hold + 0.06)
        force[rel] += a * (1 - (u[rel] - hold) / 0.06)
        tr["hold_s"] = float(hold)
        tr["t_release"] = float(tr["t_contact"] + hold + 0.06)      # force back at zero
    noise = np.zeros(true_t.size)
    e = rng.normal(0, NOISE_SD * np.sqrt(1 - 0.25), true_t.size)
    for i in range(1, true_t.size):
        noise[i] = 0.5 * noise[i - 1] + e[i]
    drift = 0.5 * (true_t - first_t) + 50 * np.sin(2 * np.pi * true_t / 300.0)
    raw = np.round(BASELINE_TARE + STALE_OFFSET + drift + noise + force).astype(np.int64)

    # session bounds, following vendor/pressure_logger/pressure_rig/recorder.py
    if v["manual_start_s"] is None:
        start = int(np.flatnonzero(count >= 1)[0])                 # first sample after pulse 1
        pulses0 = int(count[start - 1]) if start > 0 else 0
        last_inc_ms = ms[np.flatnonzero(np.diff(count) > 0)[-1] + 1]
        stop = int(np.flatnonzero(ms - last_inc_ms > 2000)[0])     # 2 s TTL timeout
        trigger = "ttl"
    else:
        start = int(np.searchsorted(true_t, v["manual_start_s"]))
        pulses0 = int(count[start - 1])
        stop = int(np.searchsorted(true_t, pulse_t[-1] + 3.0))     # Stop pressed 3 s later
        trigger = "manual"
    sl = slice(start, stop)
    s_ms, s_t, s_count = ms[sl], true_t[sl], count[sl] - pulses0
    s_raw, s_ttl = raw[sl], ttl[sl]
    t_s = (s_ms - s_ms[0]) / 1000.0
    corrected = np.round(s_raw - BASELINE_TARE, 2)
    pc0 = datetime(2026, 10, 20, 10, 0, 0)
    pc = [pc0 + timedelta(seconds=float(x) * (1 + PC_PPM * 1e-6) + rng.uniform(0.003, 0.022))
          for x in s_t - s_t[0]]
    stem = f"{v['mouse']}_{DATE}_S{v['session']:02d}"
    pdir = os.path.join(dirs["pressure"], v["mouse"], DATE)
    os.makedirs(pdir, exist_ok=True)
    pcsv = os.path.join(pdir, stem + "_pressure.csv")
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "arduino_ms", "pc_time", "raw", "baseline", "corrected", "indicator",
                    "ttl", "ttl_count"])
        for i in range(len(s_ms)):
            w.writerow([f"{t_s[i]:.4f}", int(s_ms[i]), pc[i].isoformat(timespec="milliseconds"),
                        int(s_raw[i]), BASELINE_TARE, f"{corrected[i]:.2f}", 0, int(s_ttl[i]),
                        int(s_count[i])])
    dur = float(t_s[-1])
    info = dict(folder="synthetic", mouse=v["mouse"], date=DATE, session=v["session"],
                trigger=trigger, start_time=pc[0].isoformat(timespec="milliseconds"),
                end_time=pc[-1].isoformat(timespec="milliseconds"), duration_s=round(dur, 3),
                n_samples=len(s_ms), rate_hz=round((len(s_ms) - 1) / dur, 2),
                ttl_pulses=int(s_count[-1]), details=f"synthetic {variant}",
                file=os.path.relpath(pcsv, dirs["pressure"]).replace("\\", "/"),
                baseline_at_start=BASELINE_TARE, schmitt_lower=100000.0, schmitt_upper=200000.0,
                ttl_input=True, port="synthetic", ttl_mode="pulses", ttl_timeout_s=2.0,
                columns={c: "" for c in ["t_s", "arduino_ms", "pc_time", "raw", "baseline",
                                         "corrected", "indicator", "ttl", "ttl_count"]})
    pjson = os.path.join(pdir, stem + "_info.json")
    with open(pjson, "w") as f:
        json.dump(info, f, indent=2)

    # ---- cameras
    cam1 = os.path.join(dirs["cameras"], f"{rid}_cam1.avi")
    cam2 = os.path.join(dirs["cameras"], f"{rid}_cam2.avi")
    write_tiny_video(cam1, cam1_pulses, fps)
    write_tiny_video(cam2, cam2_pulses, fps)
    n_cam1 = len(cam1_pulses)

    # ---- manual scoring on cam1
    score = np.zeros(n_cam1, np.uint8)
    uncert = np.zeros(n_cam1, np.uint8)
    reflex, d_frames, d_types, beh = [], [], [], []
    lag = lambda: int(round(rng.uniform(0.25, 0.40) * fps))
    for tr in trials:
        c = tr["cam1"]
        df_ = c + lag()
        d_frames.append(df_)
        d_types.append(KEY_OF[tr["stim_type"]])
        tr["video_cam_frame"] = df_
        key = (tr["slot"], tr["block"], tr["trial"])
        if rng.random() < P_WITHDRAW[tr["stim_type"]]:
            d = int(rng.integers(12, 31)) if tr["stim_type"] == "heat" else int(rng.integers(2, 7))
            reflex.append((c + d, 1))
            beh.append(dict(type="withdrawal", on_cam_frame=c + d, off_cam_frame=c + d, link=key))
            if tr["stim_type"] == "pin" and rng.random() < 0.4:
                f2 = c + d + int(rng.integers(3, 9))
                reflex.append((f2, 2))
                beh.append(dict(type="flinch", on_cam_frame=f2, off_cam_frame=f2, link=key))
        bout = None
        if tr["stim_type"] == "pin" and rng.random() < 0.5:
            a0 = c + int(rng.integers(20, 41))
            bout = (2, a0, a0 + int(rng.integers(20, 121)))
        elif tr["stim_type"] == "heat" and rng.random() < 0.5:
            a0 = c + int(rng.integers(12, 31))
            bout = (1, a0, a0 + int(rng.integers(10, 61)))
        elif tr["stim_type"] == "mild" and tr["trial"] == 3 and rng.random() < 0.6:
            a0 = c + int(rng.integers(15, 31))
            bout = (3, a0, a0 + int(rng.integers(40, 151)))
        if bout:
            code, a0, b0 = bout
            score[a0 - 1:b0] = code
            beh.append(dict(type={1: "attending", 2: "licking_biting", 3: "guarding"}[code],
                            on_cam_frame=a0, off_cam_frame=b0, link=key))
    # one escape bout in the baseline (no stimulus yet: not linked) and one uncertain stretch
    e0 = int(np.searchsorted(t_master, ASSAY_T0_S + 100.0))
    score[e0 - 1:e0 + 29] = 4
    beh.append(dict(type="escape", on_cam_frame=e0, off_cam_frame=e0 + 29, link=None))
    u0 = int(np.searchsorted(t_master, ASSAY_T0_S + 620.0))
    uncert[u0 - 1:u0 + 39] = 1
    beh.append(dict(type="uncertain", on_cam_frame=u0, off_cam_frame=u0 + 39, link=None))
    reflex = sorted(reflex)
    obj = lambda xs: np.array(list(xs), dtype=object)
    mat = os.path.join(dirs["scoring"], f"ScoringAB_{rid}_cam1.mat")
    d_sorted = np.argsort(d_frames, kind="stable")
    sio.savemat(mat, dict(
        score=score.reshape(-1, 1), uncert=uncert.reshape(-1, 1),
        timeAxis=(np.arange(n_cam1) / fps).reshape(1, -1),
        affNames=obj(["None", "Paw attending", "Licking or biting",
                      "Sustained lifting / guarding", "Escape / rearing"]),
        affKeys=obj("asdf"), refNames=obj(["Paw withdrawal", "Flinch / flick"]),
        reflexEvents=np.array(reflex, dtype=np.uint16).reshape(-1, 2),
        dFrames=np.array(d_frames, np.uint16)[d_sorted].reshape(-1, 1),
        dSec=((np.array(d_frames)[d_sorted] - 1) / fps).reshape(-1, 1),
        dTypes=np.array(d_types, np.uint8)[d_sorted].reshape(-1, 1),
        stimNames=obj(STIM_NAMES), rep=np.zeros((len(d_frames), 1)), T=np.zeros((len(d_frames), 10)),
        winSec=np.zeros((len(d_frames), 1)), syncOff=0, offFrames=0, frameRate=fps,
        nUsed=n_cam1, obsWindow=30, guardMin=2, lagMs=250, mode=3, sessionNo="",
        mouseID=v["mouse"], sexID=v["sex"], dayNo=str(v["day"]), phase=v["assay"],
        botFile=os.path.basename(cam1), sidFile=os.path.basename(cam2)))

    # ---- EXTRACT frame table (extract_1p.m format: one row per raw frame of the day)
    reason = np.full(n, "", dtype=object)
    reason[:N_TRIM] = "trimmed"
    reason[list(DROPPED)] = "dropped"
    valid = (reason == "").astype(int)
    pre_n = v["day_prefix_frames"]
    rows = []
    other = mf.recording_id_for(v["mouse"], v["day"], "pre")
    for i in range(pre_n):                     # an earlier recording of the same day
        rows.append((other, i, "pre", 1, "", i + 1, i + 1, tsc0 - 10_000_000_000 + i * 49962))
    h5 = pre_n
    for i in range(n):
        tr_row = len(rows) + 1
        if valid[i]:
            h5 += 1
        rows.append((rid, i, v["assay"], int(valid[i]), reason[i], h5 if valid[i] else "",
                     tr_row if valid[i] else "", tsc_out[i] if tsc_out[i] > 0 else ""))
    ft = pd.DataFrame(rows, columns=["recording_id", "isx_frame", "assay", "valid",
                                     "invalid_reason", "h5_row", "extract_row", "tsc_us"])
    ft_csv = os.path.join(dirs["extract"], f"{v['mouse']}_d{v['day']}_frame_table.csv")
    ft.to_csv(ft_csv, index=False)

    # anchors = contact frames verified on cam1, paired with the logger contact pulses
    anchors = pd.DataFrame([(tr["cam1"], tr["pulse"]) for tr in trials if tr["stim_type"] != "heat"],
                           columns=["cam_frame", "pulse"])
    anchors_csv = os.path.join(root, "anchors_cam1.csv")
    anchors.to_csv(anchors_csv, index=False)

    # ---- the truth, from the simulated physics only. It must not call the pipeline's own
    # count_to_isx_frame: a wrong index rule would then move the truth with it and pass.
    pulse_of_cam1 = {i + 1: int(p) for i, p in enumerate(cam1_pulses)}

    def frame_of(t_true):
        # CONTRACTS.md 1a: the pulse that ends a frame's exposure marks its end, so frame k runs
        # from pulse k-K to pulse k+1-K. (The simulated SYNC edge comes 0.4 ms after the TSC
        # stamp; the contract does not resolve that, so frames are bounded by the pulses.)
        return int(np.searchsorted(pulse_t, t_true, side="right")) + K

    def isx_of_cam1(c):
        # a camera frame is the snapshot taken at its trigger pulse
        return frame_of(pulse_t[pulse_of_cam1[c] - 1])

    def trigger_t(c):
        # true snapshot time of a cam1 frame (its pulse) on the master clock
        return float(pulse_t[pulse_of_cam1[c] - 1] - ts[0])

    stim_truth, onset_t = [], {}
    for tr in trials:
        key = (tr["slot"], tr["block"], tr["trial"])
        if tr["stim_type"] == "heat":
            # the logger cannot see heat: the onset is the scored (late) tap's camera frame
            onset_t[key] = trigger_t(tr["video_cam_frame"])
            stim_truth.append(dict(slot=tr["slot"], block=tr["block"], trial=tr["trial"],
                                   stim_type="heat", source="video",
                                   onset_cam_frame=tr["video_cam_frame"],
                                   onset_isx_frame=isx_of_cam1(tr["video_cam_frame"]),
                                   onset_t_master_s=onset_t[key]))
        else:
            # the contact lies between pulse p and p+1 (its samples have ttl_count p), inside
            # the exposure of the frame that pulse p+1 ends
            onset_t[key] = float(tr["t_contact"] - ts[0])
            rel_margin = float(np.min(np.abs(pulse_t - tr["t_release"])))
            stim_truth.append(dict(slot=tr["slot"], block=tr["block"], trial=tr["trial"],
                                   stim_type=tr["stim_type"], source="pressure",
                                   contact_t_master_s=onset_t[key],
                                   pulse=tr["pulse"], pulse_margin_s=tr["pulse_margin_s"],
                                   onset_cam_frame=tr["cam1"],
                                   onset_isx_frame=frame_of(tr["t_contact"]),
                                   release_t_master_s=float(tr["t_release"] - ts[0]),
                                   release_pulse_margin_s=rel_margin,
                                   offset_isx_frame=frame_of(tr["t_release"])))
    beh_truth = []
    for b in beh:
        link = tuple(b["link"]) if b["link"] else None
        on_t = trigger_t(b["on_cam_frame"])
        beh_truth.append(dict(type=b["type"], on_cam_frame=int(b["on_cam_frame"]),
                              off_cam_frame=int(b["off_cam_frame"]),
                              on_isx_frame=isx_of_cam1(b["on_cam_frame"]),
                              off_isx_frame=isx_of_cam1(b["off_cam_frame"]),
                              on_t_master_s=on_t,
                              latency_s=on_t - onset_t[link] if link else None,
                              link=list(link) if link else None))
    first_valid = N_TRIM
    truth = dict(
        variant=variant, recording_id=rid, stim_order=stim_order, assay_t0_s=ASSAY_T0_S,
        pulse_to_frame_offset=K, cam_lead_frames=CAM_LEAD, n_frames_isx=n,
        n_ttl=int(s_count[-1]), n_ttl_true=n_pulses, n_cam1_frames=n_cam1,
        n_cam2_frames=len(cam2_pulses), n_dropped=len(DROPPED), dropped_frames=list(DROPPED),
        n_trimmed_early=N_TRIM, true_period_s=period, cam1_lost_pulses=list(v["cam1_lost"]),
        logger_trigger=trigger, logger_pulses_missed=pulses0,
        expect_fail=variant != "good",
        expect_fail_reason={"good": "", "cam_drop": "cam1 has 3 fewer frames than pulses",
                            "manual_start": f"logger missed the first {pulses0} pulses"}[variant],
        extract_row={str(first_valid): pre_n + first_valid + 1, "100": pre_n + 101,
                     str(DROPPED[0]): None},
        stim_events=stim_truth, behaviour_events=beh_truth,
        files=dict(isxd=isxd, isx_timestamps=ts_csv, gpio=gpio_csv, pressure_csv=pcsv,
                   pressure_json=pjson, cam1_video=cam1, cam2_video=cam2, scoring_mat=mat,
                   extract_frame_table=ft_csv, anchors_cam1=anchors_csv))
    with open(os.path.join(root, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    return truth


def manifest_row(truth, base_dir):
    v = VARIANTS[truth["variant"]]
    rel = lambda p: os.path.relpath(p, base_dir).replace("\\", "/")
    f = truth["files"]
    return dict(recording_id=truth["recording_id"], mouse_id=v["mouse"], sex=v["sex"],
                day=str(v["day"]), assay=v["assay"], arm=v["arm"], stim_order=v["stim_order"],
                mild_g="1.4", injection_time="10:41" if v["assay"] == "post" else "",
                assay_t0_s=str(ASSAY_T0_S), isxd=rel(f["isxd"]), gpio=rel(f["gpio"]), imu="",
                pressure_csv=rel(f["pressure_csv"]), pressure_json=rel(f["pressure_json"]),
                cam1_video=rel(f["cam1_video"]), cam2_video=rel(f["cam2_video"]),
                scoring_mat=rel(f["scoring_mat"]), notes=f"SYNTHETIC ({truth['variant']})",
                pulse_to_frame_offset=str(K), isx_timestamps=rel(f["isx_timestamps"]),
                extract_frame_table=rel(f["extract_frame_table"]))


def expected_recordings_row(truth):
    """The recordings.csv row count_check / master_timeline should produce (contract
    columns; the Arduino intercept depends on when the Arduino was reset, so it is left out)."""
    t = truth
    fails = []
    if t["n_ttl"] != mt.expected_n_ttl(t["n_frames_isx"], K):
        fails.append(t["expect_fail_reason"])
    for cam in ("cam1", "cam2"):
        if t[f"n_{cam}_frames"] != t["n_ttl_true"]:
            fails.append(f"{cam} has {t['n_ttl_true'] - t[f'n_{cam}_frames']} fewer frames than pulses")
    return dict(recording_id=t["recording_id"], n_frames_isx=t["n_frames_isx"],
                n_dropped=t["n_dropped"], n_trimmed_early=t["n_trimmed_early"],
                period_s=t["true_period_s"], fps=1.0 / t["true_period_s"], n_ttl=t["n_ttl"],
                n_gpio_sync_edges=t["n_ttl_true"], n_cam1_frames=t["n_cam1_frames"],
                n_cam2_frames=t["n_cam2_frames"], pulse_to_frame_offset=K,
                cam_lead_frames=CAM_LEAD, arduino_fit_a="",
                arduino_fit_b=1.0 / (1 + ARDUINO_PPM * 1e-6),
                arduino_fit_ppm=(1.0 / (1 + ARDUINO_PPM * 1e-6) - 1) * 1e6,
                arduino_fit_resid_ms="", count_check="FAIL" if fails else "PASS",
                notes="; ".join(fails) or "synthetic: all counts agree")


def generate_all(out_root=DEFAULT_OUT, variants=tuple(VARIANTS), fps=DEFAULT_FPS):
    """Generate the variants and write out_root/manifest.csv and
    out_root/recordings_expected.csv; returns ({variant: truth}, manifest path)."""
    os.makedirs(out_root, exist_ok=True)
    truths = {name: generate(out_root, name, fps=fps) for name in variants}
    mpath = os.path.join(out_root, "manifest.csv")
    mf.write_manifest(mpath, [manifest_row(t, out_root) for t in truths.values()])
    pd.DataFrame([expected_recordings_row(t) for t in truths.values()],
                 columns=mt.RECORDINGS_COLUMNS).to_csv(
        os.path.join(out_root, "recordings_expected.csv"), index=False)
    return truths, mpath


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as d:
        truths, mpath = generate_all(d, ("good", "cam_drop", "manual_start"))
        man = mf.load_manifest(mpath)
        check(not man.errors and not man.missing_files, f"synthetic manifest: {man.report()}")
        g = truths["good"]
        p = pd.read_csv(g["files"]["pressure_csv"])
        check(list(p.columns) == ["t_s", "arduino_ms", "pc_time", "raw", "baseline", "corrected",
                                  "indicator", "ttl", "ttl_count"], "logger columns")
        check(p["ttl_count"].iloc[0] == 1 and p["ttl_count"].iloc[-1] == mt.expected_n_ttl(g["n_frames_isx"], K),
              "TTL-start session counts 1..N")
        e = next(x for x in g["stim_events"] if x["source"] == "pressure")
        check(e["onset_isx_frame"] == e["pulse"] + K and
              0 < e["contact_t_master_s"] - (e["pulse"] - 1 + K) * g["true_period_s"] < g["true_period_s"] * 1.01,
              "truth: a contact after pulse p lies in isx_frame p + K (CONTRACTS.md 1a)")
        check(np.allclose(p["t_s"], (p["arduino_ms"] - p["arduino_ms"].iloc[0]) / 1000.0),
              "t_s is the Arduino millis clock")
        dt = np.diff(p["arduino_ms"])
        check(set(np.unique(dt)) <= {12, 13, 14}, "12-14 ms sample intervals")
        check(np.allclose(p["corrected"], p["raw"] - p["baseline"], atol=0.006), "corrected = raw - baseline")
        tsr = pd.read_csv(g["files"]["isx_timestamps"])
        check(len(tsr) == g["n_frames_isx"] and (tsr["Frame Timestamp (us)"] == 0).sum() == 3,
              "timestamp export: every frame, 0 at dropped frames")
        ft = pd.read_csv(g["files"]["extract_frame_table"])
        row = ft[(ft.recording_id == g["recording_id"]) & (ft.isx_frame == N_TRIM)].iloc[0]
        check(int(row.extract_row) == g["extract_row"][str(N_TRIM)], "frame table extract_row")
        import camera_sync
        check(camera_sync.count_video_frames(g["files"]["cam1_video"])[0] == g["n_cam1_frames"],
              "cam1 video frame count")
        check(truths["cam_drop"]["n_cam1_frames"] == truths["cam_drop"]["n_frames_isx"] - 3,
              "cam_drop loses 3 frames")
        m = truths["manual_start"]
        check(m["logger_trigger"] == "manual" and m["n_ttl"] == m["n_frames_isx"] - m["logger_pulses_missed"]
              and m["logger_pulses_missed"] > 0, "manual start misses pulses")
        s = sio.loadmat(g["files"]["scoring_mat"], squeeze_me=True)
        check(int(s["nUsed"]) == g["n_cam1_frames"] and len(s["dFrames"]) == 60, "ScoringAB fields")
        n_pressure = sum(e["source"] == "pressure" for e in g["stim_events"])
        check(n_pressure == 45 and len(g["stim_events"]) == 60, "45 pressure + 15 heat trials")
        ex = pd.read_csv(os.path.join(d, "recordings_expected.csv"))
        check(list(ex.count_check) == ["PASS", "FAIL", "FAIL"], "expected recordings rows")
        try:                       # the footer reader of the Inscopix part, if present
            import isx_meta
        except ImportError:
            isx_meta = None
        if isx_meta is not None:
            meta = isx_meta.read_meta(g["files"]["isxd"])
            check(meta.get("n_frames") == g["n_frames_isx"] and meta.get("n_dropped") == 3 and
                  meta.get("size_consistent") is True, f"fake isxd footer: {meta}")
    if failures:
        print("synthetic_session selftest FAILED:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("synthetic_session selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output folder (default out/synthetic)")
    ap.add_argument("--variant", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="simulated frame rate")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    truths, mpath = generate_all(args.out, tuple(args.variant), args.fps)
    for name, t in truths.items():
        print(f"{name:13s} {t['recording_id']}: {t['n_frames_isx']} frames, {t['n_ttl']} logger "
              f"pulses, cam1 {t['n_cam1_frames']}, cam2 {t['n_cam2_frames']}, "
              f"expect {'FAIL' if t['expect_fail'] else 'PASS'}")
    print("manifest:", mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
