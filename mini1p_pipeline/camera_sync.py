"""Camera sync: write camera_frames.csv (cam frame -> SYNC pulse -> isx_frame).

Both Imaging Source cameras are hardware-triggered by the Inscopix SYNC pulse, so when the
saved video has exactly as many frames as the logger counted pulses, camera frame n IS
pulse n (CONTRACTS.md section 1) and the table is written with method 'hw_trigger'.

When the counts differ, a camera dropped or added frames somewhere and the identity is
broken from that point on. This module does not guess: it stops with an error unless the
user passes --allow-mismatch AND an anchor table of (cam_frame, pulse) pairs, for example
stimulus contact frames verified on the video paired with the logger contact pulses. Then
it writes method 'fit': identity up to the last anchor with pulse == cam_frame, after that
the pulse offset interpolated between anchors (lost frames shift the offset in steps) and
held after the last anchor. Every frame after the first mismatch gets uncertain = 1,
because the anchors only bracket where the frames were lost.

Columns: recording_id, camera, cam_frame, pulse, isx_frame, t_master_s, method, gap_before
(+ uncertain). CONTRACTS.md 1a: camera frame n is exposed right after pulse n, i.e. DURING
isx_frame n + K, so isx_frame = master_timeline.count_to_isx_frame(pulse) = pulse + K (the same
function places logger samples). t_master_s is the trigger (snapshot) time = the time of the
pulse = t_master_s of isx_frame pulse-1+K, the frame that pulse ends. The old camera-only
cam_lead_frames is superseded by this rule (the column stays 1, for documentation).

Frame counts come from the video header (cv2), from a .json/.txt count file, or by decoding
every frame (--decode-count, slow but exact for files with an unreliable header).

Usage
  python camera_sync.py --recording-id M01_d1_pre --camera cam1 --video cam1.avi \
      --recordings out/M01_d1_pre/recordings.csv [--timestamps ts.csv | --frame-table ft.csv] \
      --out out/M01_d1_pre/camera_frames.csv [--allow-mismatch --anchors anchors.csv]
t_master_s uses the same time base as master_timeline (timestamps, else the frame table's
tsc_us, else a measured period); with only the nominal footer period it is left empty.
A 'fit' table is refused unless the anchors give one distinct pulse of 1..n_ttl per frame
(offsets rising from 0 to at most the number of lost frames); a camera with more frames than
pulses cannot be fitted.
  python camera_sync.py --selftest
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

import master_timeline as mt

CAMERAS = ("cam1", "cam2")


class CameraSyncError(RuntimeError):
    """Counts do not match and the user has not supplied a way to resolve it."""


def count_video_frames(path, decode=False):
    """Number of frames in a camera recording and how it was counted.

    A .json file ({"n_frames": N}) or .txt file (one integer) stands in for a video, which
    is how tests and other tools can pass a count without a movie."""
    if not path or not os.path.exists(path):
        raise CameraSyncError(f"camera file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            return int(json.load(f)["n_frames"]), "count_file"
    if ext == ".txt":
        with open(path) as f:
            return int(f.read().strip()), "count_file"
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise CameraSyncError(f"cv2 cannot open {path}")
    try:
        if decode:
            n = 0
            while cap.grab():
                n += 1
            return n, "decoded"
        n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if n <= 0:
            raise CameraSyncError(f"{path}: header has no frame count; use --decode-count")
        return n, "header"
    finally:
        cap.release()


def read_anchors(path):
    """Anchor table: cam_frame + pulse columns (one row per verified frame)."""
    a = pd.read_csv(path)
    if not {"cam_frame", "pulse"} <= set(a.columns):
        raise CameraSyncError(f"{path}: anchors need columns cam_frame and pulse")
    a = a[["cam_frame", "pulse"]].dropna().astype(int).sort_values("cam_frame")
    return a.reset_index(drop=True)


def anchors_from_events(video_cam_frames, pulses, max_lag_frames):
    """Pair each video-marked contact frame with the nearest logger contact pulse.

    Only use this with frame-verified contact frames: a live key press is ~0.3 s late,
    so it would shift the fit by several frames. Pairs further apart than max_lag_frames
    are dropped (they are not the same contact)."""
    v = np.sort(np.asarray(video_cam_frames, dtype=float))
    p = np.sort(np.asarray(pulses, dtype=float))
    rows = []
    if v.size == 0 or p.size == 0:
        return pd.DataFrame(columns=["cam_frame", "pulse"])
    for c in v:
        k = int(np.argmin(np.abs(p - c)))
        if abs(p[k] - c) <= max_lag_frames:
            rows.append((int(c), int(p[k])))
    return pd.DataFrame(rows, columns=["cam_frame", "pulse"]).drop_duplicates("pulse")


# Periods that were measured (GPIO SYNC edges, TSC), as opposed to the nominal footer period
MEASURED_PERIOD = ("gpio_sync_mean", "tsc", "caller")


def time_base(recording_id, rec, timestamps=None, frame_table=None):
    """(t_master per isx_frame or None, note): the time base master_timeline uses
    (mt.recording_frame_times: timestamp CSV, else the frame table's tsc_us, else n * period).
    A nominal IDAS footer period (50 ms against a true 49.96 ms on the 2024 rig) drifts by ~1 s
    over an assay, so camera times are then left empty rather than disagreeing with frames.csv
    (master_timeline recomputes camera times from the pulses in any case)."""
    if mt._int_or_none(rec.get("n_frames_isx")) is None:
        return None, "no n_frames_isx: t_master_s left empty"
    ft = pd.read_csv(frame_table, dtype={"recording_id": str}) if frame_table else None
    try:
        t, _, src = mt.recording_frame_times(recording_id, rec, timestamps, ft)
    except mt.TimelineError as e:
        return None, f"no time base ({e}): t_master_s left empty"
    if src == "period" and str(rec.get("period_source", "")) not in MEASURED_PERIOD:
        return None, ("only the nominal footer period is known: t_master_s left empty (give "
                      "--timestamps or --frame-table)")
    return t, None


def remove_camera_rows(out_path, recording_id, camera):
    """Drop one recording + camera from camera_frames.csv (a camera without a video this run
    must not keep the rows of an earlier run). Returns the number of rows removed."""
    if not os.path.exists(out_path):
        return 0
    old = pd.read_csv(out_path, dtype={"recording_id": str, "camera": str})
    gone = (old["recording_id"] == recording_id) & (old["camera"] == camera)
    if gone.any():
        old[~gone].to_csv(out_path, index=False)
    return int(gone.sum())


def build_camera_frames(recording_id, camera, n_cam, n_ttl, K, t_master=None,
                        allow_mismatch=False, anchors=None):
    """camera_frames rows for one camera, plus an info dict (method, first mismatch, ...)."""
    if camera not in CAMERAS:
        raise CameraSyncError(f"camera must be one of {CAMERAS}, got {camera}")
    if K is None:
        raise CameraSyncError("pulse_to_frame_offset (K) is unknown; it is never assumed")
    cam_frame = np.arange(1, n_cam + 1)
    info = dict(recording_id=recording_id, camera=camera, n_cam=n_cam, n_ttl=n_ttl)
    if n_cam == n_ttl:
        pulse = cam_frame.copy()
        method, uncertain_from = "hw_trigger", None
    else:
        msg = (f"{recording_id} {camera}: {n_cam} video frames but {n_ttl} SYNC pulses "
               f"(difference {n_cam - n_ttl:+d}). The camera dropped or added frames, so "
               "cam frame n is no longer pulse n after the first mismatch.")
        if not allow_mismatch:
            raise CameraSyncError(msg + " Re-run with --allow-mismatch and --anchors "
                                  "(verified contact frames) to write a 'fit' table, or fix "
                                  "the recording.")
        if anchors is None or len(anchors) < 2:
            raise CameraSyncError(msg + " --allow-mismatch needs at least 2 anchors "
                                  "(cam_frame, pulse).")
        if n_cam > n_ttl:
            # extra frames have no pulse at all; a frame -> pulse table cannot express that
            raise CameraSyncError(msg + " A camera with MORE frames than pulses was not "
                                  "(only) triggered by SYNC, and anchors cannot tell which "
                                  "frames have no pulse: fix the recording.")
        a = anchors.sort_values("cam_frame").drop_duplicates("cam_frame")
        x = a["cam_frame"].to_numpy(float)
        d = (a["pulse"] - a["cam_frame"]).to_numpy(float)     # pulse offset at each anchor
        lost = n_ttl - n_cam
        # Lost frames can only push the offset up, and never past the number lost; anything
        # else (a typo, a contact paired with the wrong pulse) would give pulses beyond n_ttl
        # or two frames on one pulse.
        if d.min() < 0 or d.max() > lost or np.any(np.diff(d) < 0):
            raise CameraSyncError(
                f"{recording_id} {camera}: anchor offsets (pulse - cam_frame) "
                f"{d.astype(int).tolist()} must start >= 0, never decrease and never exceed the "
                f"{lost} lost frames; check the anchors")
        # The identity holds up to the last anchor that still has offset 0; the lost/extra
        # frames lie somewhere after it, so every later frame is uncertain.
        bad = np.flatnonzero(d != 0)
        if bad.size == 0:
            uncertain_from = int(x[-1]) + 1
        elif bad[0] == 0:
            uncertain_from = 1
        else:
            uncertain_from = int(x[bad[0] - 1]) + 1
        # Lost frames shift the offset in steps, so the offset is interpolated between
        # anchors (exact where two neighbouring anchors agree) and held after the last one,
        # rather than forcing one straight line through every anchor.
        offset = np.rint(np.interp(cam_frame, x, d)).astype(np.int64)
        offset[cam_frame < uncertain_from] = 0
        pulse = cam_frame + offset
        if np.any(np.diff(pulse) <= 0) or pulse.min() < 1 or pulse.max() > n_ttl:
            raise CameraSyncError(f"{recording_id} {camera}: the anchor fit does not give one "
                                  f"distinct pulse of 1..{n_ttl} per frame")
        slope, intercept, _, _ = stats.theilslopes(a["pulse"].to_numpy(float), x)
        method = "fit"
        info.update(n_anchors=len(a), anchor_offsets=sorted(set(d.astype(int).tolist())),
                    theil_sen_slope=float(slope), theil_sen_intercept=float(intercept))
        tail = lost - int(d[-1])
        print(f"WARNING: {msg} Wrote a 'fit' table from {len(a)} anchors (offsets "
              f"{info['anchor_offsets']} pulses); frames >= {uncertain_from} are uncertain."
              + (f" The anchors account for {int(d[-1])} of the {lost} lost frames; the other "
                 f"{tail} are taken to be missing at the end (camera stopped early)."
                 if tail else ""), file=sys.stderr)
    isx = mt.count_to_isx_frame(pulse, K)
    t = np.full(n_cam, np.nan)
    if t_master is not None:
        # the trigger time: pulse n is the TSC time of the frame it ends (n-1+K)
        end = mt.pulse_end_frame(pulse, K)
        ok = (end >= 0) & (end < len(t_master))
        t[ok] = t_master[end[ok]]
    gap = np.diff(np.concatenate([[0], pulse])) - 1
    df = pd.DataFrame(dict(recording_id=recording_id, camera=camera, cam_frame=cam_frame,
                           pulse=pulse, isx_frame=isx, t_master_s=np.round(t, 6),
                           method=method, gap_before=gap,
                           uncertain=(cam_frame >= uncertain_from).astype(int)
                           if uncertain_from else 0))
    info.update(method=method, uncertain_from=uncertain_from)
    return df, info


def write_camera_frames(df, out_path):
    """Write rows into camera_frames.csv, replacing only this recording + camera."""
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, dtype={"recording_id": str, "camera": str})
        rid, cam = df["recording_id"].iloc[0], df["camera"].iloc[0]
        old = old[~((old["recording_id"] == rid) & (old["camera"] == cam))]
        df = pd.concat([old, df], ignore_index=True)
    df = mt.ensure_columns(df, mt.CAMERA_FRAMES_COLUMNS)
    df.to_csv(out_path, index=False)
    return out_path


def run(recording_id, camera, video, out_path, recordings=None, n_ttl=None, K=None,
        n_frames_isx=None, period_s=None, timestamps=None, allow_mismatch=False,
        anchors=None, decode=False, t_master=None, frame_table=None):
    """File-level wrapper used by the CLI and run_pipeline. t_master (array per isx_frame)
    is used as is when the caller already has the recording's time base (run_pipeline);
    otherwise it is built like master_timeline's (see time_base)."""
    rec = {}
    if recordings:
        recs = pd.read_csv(recordings, dtype={"recording_id": str})
        sel = recs[recs["recording_id"] == recording_id]
        if sel.empty:
            raise CameraSyncError(f"{recording_id} not in {recordings}")
        rec = sel.iloc[0].to_dict()
        n_ttl = n_ttl if n_ttl is not None else mt._int_or_none(rec.get("n_ttl"))
        K = K if K is not None else mt._int_or_none(rec.get("pulse_to_frame_offset"))
        n_frames_isx = n_frames_isx or mt._int_or_none(rec.get("n_frames_isx"))
    if period_s:
        rec.update(period_s=period_s, period_source="caller")
    rec["n_frames_isx"] = n_frames_isx
    if n_ttl is None:
        if n_frames_isx is None or K is None:
            raise CameraSyncError("need n_ttl (or n_frames_isx and K) to check the camera")
        n_ttl = mt.expected_n_ttl(n_frames_isx, K)          # CONTRACTS.md count rule
        print("note: no logger pulse count; using n_frames_isx + K as the pulse count",
              file=sys.stderr)
    n_cam, how = count_video_frames(video, decode)
    anc = read_anchors(anchors) if isinstance(anchors, str) else anchors
    t, note = (t_master, None) if t_master is not None else \
        time_base(recording_id, rec, timestamps, frame_table)
    if note:
        print(f"note: {recording_id} {camera}: {note}", file=sys.stderr)
    df, info = build_camera_frames(recording_id, camera, n_cam, n_ttl, K, t,
                                   allow_mismatch, anc)
    info.update(count_method=how, time_note=note)
    write_camera_frames(df, out_path)
    return df, info


def selftest():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    t = np.arange(1000) * 0.05
    df, info = build_camera_frames("T01_d1_pre", "cam1", 1000, 1000, 0, t)
    check(info["method"] == "hw_trigger" and (df.pulse == df.cam_frame).all(), "identity map")
    check((df.isx_frame == df.pulse).all() and df.t_master_s.iloc[9] == t[9],
          "isx_frame = count_to_isx_frame(pulse) = pulse + K; time = the trigger pulse "
          "(TSC time of isx_frame pulse-1+K)")
    df, _ = build_camera_frames("T01_d1_pre", "cam2", 1000, 1000, 2, t)
    check(df.isx_frame.iloc[0] == 3 and df.t_master_s.iloc[0] == t[2], "K shifts isx_frame and time")

    try:
        build_camera_frames("T01_d1_pre", "cam1", 997, 1000, 0, t)
        check(False, "a count mismatch must raise without --allow-mismatch")
    except CameraSyncError:
        pass
    try:
        build_camera_frames("T01_d1_pre", "cam1", 997, 1000, 0, t, allow_mismatch=True)
        check(False, "--allow-mismatch without anchors must raise")
    except CameraSyncError:
        pass

    # three frames lost at pulses 400-402: frames >= 400 show pulse frame+3
    true_pulse = np.concatenate([np.arange(1, 400), np.arange(403, 1001)])
    anc = pd.DataFrame(dict(cam_frame=[50, 200, 380, 450, 700, 950],
                            pulse=true_pulse[[49, 199, 379, 449, 699, 949]]))
    df, info = build_camera_frames("T01_d1_pre", "cam1", 997, 1000, 0, t, True, anc)
    check(info["method"] == "fit" and (df.method == "fit").all(), "method fit")
    check(info["uncertain_from"] == 381, f"uncertain from {info['uncertain_from']} (want 381)")
    check(df.uncertain.iloc[379] == 0 and df.uncertain.iloc[380] == 1 and df.uncertain.iloc[-1] == 1,
          "uncertain flags after the first mismatch")
    check(int(df.pulse.iloc[379]) == 380 and int(df.pulse.iloc[449]) == 453 and
          int(df.pulse.iloc[-1]) == 1000, "fit: identity before, anchor offset after the loss")

    # anchors that do not fit the counts must be refused, not written with a warning
    for what, n_cam, pairs in (
            ("offsets beyond the 3 lost frames (pulses past n_ttl)", 997,
             [(100, 100), (500, 505), (900, 910)]),
            ("offsets that go down (two frames on one pulse)", 997,
             [(100, 100), (500, 505), (900, 903)]),
            ("a camera with more frames than pulses", 1003, [(100, 100), (900, 897)])):
        try:
            build_camera_frames("T01_d1_pre", "cam1", n_cam, 1000, 0, t, True,
                                pd.DataFrame(pairs, columns=["cam_frame", "pulse"]))
            check(False, f"anchor fit must raise: {what}")
        except CameraSyncError:
            pass
    # anchors that explain only part of the loss: the rest is taken to be at the end
    df, info = build_camera_frames("T01_d1_pre", "cam1", 997, 1000, 0, t, True,
                                   pd.DataFrame(dict(cam_frame=[100, 500], pulse=[100, 501])))
    check(df.pulse.is_monotonic_increasing and df.pulse.max() == 998 and info["uncertain_from"] == 101,
          "partial anchors: pulses distinct and <= n_ttl, uncertain after the last exact anchor")

    a = anchors_from_events([10, 100, 205], [9, 98, 203, 500], max_lag_frames=3)
    check(list(a.pulse) == [9, 98, 203], "anchors pair nearest pulses")

    with tempfile.TemporaryDirectory() as d:
        import cv2
        p = os.path.join(d, "tiny.avi")
        w = cv2.VideoWriter(p, 0, 20.0, (4, 4), True)
        for i in range(123):
            w.write(np.full((4, 4, 3), i % 250, np.uint8))
        w.release()
        check(count_video_frames(p)[0] == 123, "header count of a tiny AVI")
        check(count_video_frames(p, decode=True)[0] == 123, "decoded count of a tiny AVI")
        with open(os.path.join(d, "n.json"), "w") as f:
            json.dump({"n_frames": 77}, f)
        check(count_video_frames(os.path.join(d, "n.json"))[0] == 77, "count file")
        rec = os.path.join(d, "recordings.csv")
        pd.DataFrame([dict(recording_id="T01_d1_pre", n_frames_isx=123, n_ttl=123, period_s=0.05,
                           pulse_to_frame_offset=0)]).to_csv(rec, index=False)
        out = os.path.join(d, "camera_frames.csv")
        run("T01_d1_pre", "cam1", p, out, recordings=rec)
        run("T01_d1_pre", "cam2", p, out, recordings=rec)
        run("T01_d1_pre", "cam1", p, out, recordings=rec)        # rerun replaces, not appends
        cf = pd.read_csv(out)
        check(len(cf) == 246 and list(cf.columns[:8]) == mt.CAMERA_FRAMES_COLUMNS,
              "camera_frames.csv holds both cameras once, contract columns first")
        # the recordings.csv period is the nominal footer one: no camera times rather than
        # times that drift away from frames.csv
        check(cf.t_master_s.isna().all(), "nominal footer period: t_master_s left empty")
        # with the frame table's TSC (true period 49.962 ms) the times are master_timeline's
        ftp = os.path.join(d, "ft.csv")
        pd.DataFrame(dict(recording_id="T01_d1_pre", isx_frame=np.arange(123),
                          tsc_us=7_000_000 + np.arange(123) * 49962)).to_csv(ftp, index=False)
        df, _ = run("T01_d1_pre", "cam1", p, out, recordings=rec, frame_table=ftp)
        tm, _, _ = mt.recording_frame_times("T01_d1_pre", dict(n_frames_isx=123, period_s=0.05),
                                            None, pd.read_csv(ftp, dtype={"recording_id": str}))
        check(abs(df.t_master_s.iloc[-1] - tm[122]) < 1e-6 and abs(df.t_master_s.iloc[-1] - 122 * 0.049962) < 1e-6,
              "frame-table TSC: cam frame n at t_master of isx_frame n-1+K")
        check(remove_camera_rows(out, "T01_d1_pre", "cam2") == 123 and
              set(pd.read_csv(out).camera) == {"cam1"}, "rows of a camera without video removed")

    if failures:
        print("camera_sync selftest FAILED:")
        for f_ in failures:
            print("  -", f_)
        return 1
    print("camera_sync selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recording-id")
    ap.add_argument("--camera", choices=CAMERAS)
    ap.add_argument("--video", help="camera video, or a .json/.txt frame-count file")
    ap.add_argument("--recordings", help="recordings.csv (n_ttl, K, n_frames_isx, period_s)")
    ap.add_argument("--n-ttl", type=int, help="override the logger pulse count")
    ap.add_argument("--pulse-to-frame-offset", type=int, help="override K")
    ap.add_argument("--timestamps", help="Inscopix frame timestamp CSV (for t_master_s)")
    ap.add_argument("--frame-table", help="EXTRACT <mouse>_d<day>_frame_table.csv; its tsc_us "
                                          "gives t_master_s when --timestamps is not given")
    ap.add_argument("--out", help="camera_frames.csv to write/update")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write a 'fit' table when counts differ (needs --anchors)")
    ap.add_argument("--anchors", help="CSV with cam_frame,pulse pairs (verified contacts)")
    ap.add_argument("--decode-count", action="store_true",
                    help="count frames by decoding the whole video")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.recording_id and args.camera and args.video and args.out):
        ap.error("--recording-id, --camera, --video and --out are required")
    try:
        df, info = run(args.recording_id, args.camera, args.video, args.out, args.recordings,
                       args.n_ttl, args.pulse_to_frame_offset, None, None, args.timestamps,
                       args.allow_mismatch, args.anchors, args.decode_count,
                       frame_table=args.frame_table)
    except CameraSyncError as e:
        print(f"CAMERA SYNC FAILED: {e}", file=sys.stderr)
        return 1
    print(f"{args.camera}: {info['n_cam']} frames ({info['count_method']}), "
          f"{info['n_ttl']} pulses, method {info['method']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
