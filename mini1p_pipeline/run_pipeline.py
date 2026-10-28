"""Run one manifest row end to end and print a one-page summary.

Steps (each writes its CONTRACTS.md table into --out-dir):
  1. count_check     recordings.csv: isx frames, logger pulses, camera frames, GPIO edges
  2. pressure_sync   pressure_frames.csv, stim_events.csv (+ pressure_summary.json, QC)
  3. camera_sync     camera_frames.csv for cam1 and cam2
  4. scoring_import  behaviour_events.csv, video_deliveries.csv
  5. master_timeline frames.csv, completes stim_events / behaviour_events / recordings

The run stops with a non-zero exit code and a loud message when:
  - --out-dir already holds an earlier run's outputs (one folder = one recording; --overwrite
    deletes them first), so no table of an earlier run is ever joined
  - the logger pulse count does not match the Inscopix frames (n_ttl != n_frames_isx + K,
    CONTRACTS.md; master_timeline.expected_n_ttl): nothing can be placed on the frame index
    (e.g. a logger started by hand mid-recording). No option overrides this.
  - count_check FAILs for any other reason, unless an explicit option names that reason:
    a camera count with --allow-mismatch plus an anchor table for that camera (camera_sync
    then writes a 'fit' table and flags every frame after the first mismatch as uncertain),
    a missing camera with --no-camera camX, the benign GPIO tail (export one edge short at the
    end) with --allow-gpio-tail. A GPIO gap or a truncated movie always stops the run.
  - pressure_sync reports a LOUD sync problem (missed pulses, no TTL, ...). A manual-start
    logger is accepted only when the counts prove it started before the first pulse.
    A stim_order the logger contradicts (touch-like contacts in the heat slot) is LOUD too.
  - master_timeline refuses the join (e.g. most scored delivery keys of a slot disagree with
    stim_order; --trust-stim-order keeps the manifest order for both checks)

K (pulse_to_frame_offset) comes from --pulse-offset or the manifest column; it is never
assumed. Every step places count-stamped data with master_timeline.count_to_isx_frame
(CONTRACTS.md 1a: logger rows with ttl_count n and camera frame n fall in isx_frame n + K).
The selftest builds the synthetic session (synthetic_session.py) in out/synthetic and asserts
that every ground-truth event lands on the right isx_frame and cam frame, at the right time.

Usage
  python run_pipeline.py --manifest manifest.csv --recording-id M01_d1_pre [--out-dir DIR]
      [--pulse-offset 0] [--scorer HL] [--overwrite] [--allow-mismatch --anchors-cam1 a.csv]
      [--no-camera cam2] [--allow-gpio-tail] [--trust-stim-order]
  python run_pipeline.py --selftest
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import camera_sync
import manifest as mf
import master_timeline as mt
import scoring_import

HERE = os.path.dirname(os.path.abspath(__file__))


class PipelineError(RuntimeError):
    """A step refused to continue; the message says which and why."""

    def __init__(self, step, msg):
        super().__init__(f"[{step}] {msg}")
        self.step = step


def _stage(name, attrs):
    """Import a pipeline module written by another part of the project and make sure it
    has the functions used here, so an unfinished module fails with a clear message."""
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise PipelineError(name, f"{name}.py cannot be imported ({e})")
    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        raise PipelineError(name, f"{name}.py has no {', '.join(missing)} (module unfinished?)")
    return mod


def _i(x):
    return mt._int_or_none(x)


# Every file a run writes into its out_dir (one folder = one recording, CONTRACTS.md 3)
OUTPUTS = ("recordings.csv", "isx_frame_times.csv", "pressure_frames.csv", "stim_events.csv",
           "pressure_summary.json", "pressure_report.txt", "pressure_qc.png",
           "camera_frames.csv", "behaviour_events.csv", "video_deliveries.csv", "frames.csv")


def prepare_out_dir(out_dir, overwrite):
    """Refuse a folder that holds an earlier run's outputs unless overwrite, which deletes
    exactly those files: tables that a step skips this time (a camera without video, a
    recording without scoring) would otherwise be joined from the earlier run."""
    found = [n for n in OUTPUTS if os.path.exists(os.path.join(out_dir, n))]
    if found and not overwrite:
        raise PipelineError("out_dir", f"{out_dir} already holds outputs of an earlier run "
                            f"({', '.join(found[:4])}{', ...' if len(found) > 4 else ''}); "
                            "pass --overwrite to replace them, or give another --out-dir")
    for n in found:
        os.remove(os.path.join(out_dir, n))
    os.makedirs(out_dir, exist_ok=True)


def allowed_fail_codes(allow_mismatch=False, anchors=None, no_camera=(), allow_gpio_tail=False):
    """count_check FAIL codes that an explicit option covers. Everything else stops the run."""
    allow = set()
    if allow_gpio_tail:
        allow.add("gpio_tail")             # GPIO log ended a few ms before the last pulse
    for cam in no_camera:
        allow |= {f"{cam}_missing", f"{cam}_count"}
    if allow_mismatch:
        allow |= {f"{cam}_count" for cam in (anchors or {})}   # camera_sync fits from anchors
    return allow


# ----------------------------------------------------------------------------- steps
def step_count_check(row, out_dir, K, allow):
    cc = _stage("count_check", ["run"])
    rec = {k: (row.get(k) or None) for k in ("isxd", "gpio", "gpio_csv", "pressure_csv",
                                              "pressure_json", "cam1_video", "cam2_video")}
    rec["recording_id"] = row["recording_id"]
    rec_path = os.path.join(out_dir, "recordings.csv")
    results = cc.run([rec], rec_path, k_cli=K)
    r, fails, warns = results[0]
    N, n_ttl = _i(r.get("n_frames_isx")), _i(r.get("n_ttl"))
    if N is None:
        raise PipelineError("count_check", "no Inscopix frame count: " + "; ".join(fails))
    if n_ttl is None:
        raise PipelineError("count_check", "no logger pulse count: " + "; ".join(fails))
    if n_ttl != mt.expected_n_ttl(N, K):
        raise PipelineError("count_check", f"logger pulses {n_ttl} != n_frames_isx + K = "
                            f"{mt.expected_n_ttl(N, K)}: " + "; ".join(fails))
    # Any other FAIL stops the run too, unless an option names that reason (CONTRACTS.md: PASS
    # needs every count, so a FAIL is never joined silently)
    blocking = [f for f in fails if getattr(f, "code", None) not in allow]
    if blocking:
        raise PipelineError("count_check", "; ".join(blocking) + " (options: --allow-mismatch "
                            "with --anchors-camX for a camera that lost frames, --no-camera camX "
                            "to run without a camera, --allow-gpio-tail for a GPIO export one "
                            "edge short at the end)")
    return rec_path, r, fails, warns


def step_frame_times(row, rec_row, out_dir):
    """One time base for every later step (see master_timeline.recording_frame_times)."""
    ft = None
    if row.get("extract_frame_table"):
        ft = pd.read_csv(row["extract_frame_table"], dtype={"recording_id": str})
    t, dropped, src = mt.recording_frame_times(row["recording_id"], rec_row,
                                               row.get("isx_timestamps") or None, ft)
    path = os.path.join(out_dir, "isx_frame_times.csv")
    pd.DataFrame(dict(recording_id=row["recording_id"], isx_frame=np.arange(t.size),
                      t_master_s=np.where(dropped, np.nan, np.round(t, 6)),
                      is_dropped=dropped.astype(int))).to_csv(path, index=False)
    return t, dropped, src, path


def step_pressure_sync(row, out_dir, K, frame_times_csv, logger_ok, trust_stim_order=False):
    ps = _stage("pressure_sync", ["process_session", "read_info", "read_frames_time", "ContractError"])
    csv_path = row["pressure_csv"]
    info_path = row.get("pressure_json") or str(ps.info_path_for(Path(csv_path)))
    try:
        res = ps.process_session(csv_path, ps.read_info(info_path), row["recording_id"],
                                 row["stim_order"], row["assay_t0_s"], K,
                                 frames_time=ps.read_frames_time(frame_times_csv, row["recording_id"]),
                                 out_dir=out_dir, csv_name=csv_path, info_name=info_path)
    except ps.ContractError as e:
        raise PipelineError("pressure_sync", str(e))
    loud = [i for i in res["issues"] if i[0] == "LOUD"]
    # A manual start is harmless only when the counts prove the logger saw pulse 1; a stim_order
    # the logger contradicts goes on only when the user says the manifest is right
    fatal = [i for i in loud if not (i[1] == "manual_start" and logger_ok)
             and not (i[1] == "stim_order" and trust_stim_order)]
    if fatal:
        raise PipelineError("pressure_sync", "; ".join(f"{c}: {m}" for _, c, m in fatal))
    return res


def step_camera_sync(row, out_dir, rec_path, t_master, allow_mismatch, anchors, no_camera=()):
    infos = {}
    path = os.path.join(out_dir, "camera_frames.csv")
    for cam in camera_sync.CAMERAS:
        video = row.get(f"{cam}_video")
        if not video or cam in no_camera:
            # no rows for this camera may survive from an earlier run into the join
            camera_sync.remove_camera_rows(path, row["recording_id"], cam)
            infos[cam] = dict(method="(no video)" if not video else "(--no-camera)")
            continue
        try:
            _, info = camera_sync.run(row["recording_id"], cam, video, path, recordings=rec_path,
                                      allow_mismatch=allow_mismatch, anchors=anchors.get(cam),
                                      t_master=t_master)
        except camera_sync.CameraSyncError as e:
            raise PipelineError("camera_sync", str(e))
        infos[cam] = info
    return path, infos


def run(row, out_dir, K, scorer="", allow_mismatch=False, anchors=None, no_camera=(),
        allow_gpio_tail=False, trust_stim_order=False, overwrite=False):
    """Run every step for one validated manifest row. Returns a dict for the summary."""
    # A non-zero K means one count-stamped stream started late (e.g. the logger missed the
    # first pulses). The cameras are triggered independently, so camera frame n and logger
    # count n would then no longer be the same pulse, which one constant cannot repair.
    # Only K = 0 is analysable (CONTRACTS.md 1a); every 2024 recording measured K = 0.
    if K not in (None, 0):
        raise PipelineError("pulse_offset", f"K = {K}: only K = 0 can be analysed - a stream "
                            "started late; re-record, or align it by hand from frame-verified "
                            "contact frames")
    prepare_out_dir(out_dir, overwrite)
    rid = row["recording_id"]
    anchors = anchors or {}
    allow = allowed_fail_codes(allow_mismatch, anchors, no_camera, allow_gpio_tail)
    rec_path, cc_row, cc_fails, cc_warns = step_count_check(row, out_dir, K, allow)
    recs = pd.read_csv(rec_path, dtype={"recording_id": str})
    rec_row = recs[recs["recording_id"] == rid].iloc[0].to_dict()
    t, dropped, t_src, ft_csv = step_frame_times(row, rec_row, out_dir)
    # step_count_check has already stopped the run if logger pulses != frames + K
    ps_res = step_pressure_sync(row, out_dir, K, ft_csv, logger_ok=True,
                                trust_stim_order=trust_stim_order)
    cam_path, cam_info = step_camera_sync(row, out_dir, rec_path, t, allow_mismatch, anchors,
                                          no_camera)
    have_scoring = bool(row.get("scoring_mat"))
    if have_scoring:
        try:
            scoring_import.run(row["scoring_mat"], rid, out_dir, scorer,
                               cam1_video=row.get("cam1_video") or None)
        except scoring_import.ScoringError as e:
            raise PipelineError("scoring_import", str(e))
    j = lambda name: os.path.join(out_dir, name)
    try:
        # every remaining count_check FAIL was covered by an option in step_count_check
        tl = mt.run(rid, rec_path, out_dir, row["stim_order"], row["assay_t0_s"],
                    timestamps=row.get("isx_timestamps") or None,
                    pressure_frames=j("pressure_frames.csv"), stim_events=j("stim_events.csv"),
                    camera_frames=cam_path,
                    behaviour_events=j("behaviour_events.csv") if have_scoring else None,
                    video_deliveries=j("video_deliveries.csv") if have_scoring else None,
                    frame_table=row.get("extract_frame_table") or None,
                    pressure_summary=j("pressure_summary.json"),
                    accept_count_fail=bool(cc_fails), trust_stim_order=trust_stim_order)
    except mt.TimelineError as e:
        raise PipelineError("master_timeline", str(e))
    return dict(row=row, out_dir=out_dir, cc_row=cc_row, cc_fails=cc_fails, cc_warns=cc_warns,
                t_source=t_src, n_dropped=int(dropped.sum()), ps=ps_res, cam=cam_info, tl=tl)


def summary(res):
    """One page of text."""
    row, tl = res["row"], res["tl"]
    rec = tl["recording"]
    fr, se, be = tl["frames"], tl["stim_events"], tl["behaviour_events"]
    L = [f"=== {row['recording_id']}  (mouse {row['mouse_id']} {row['sex']}, day {row['day']}, "
         f"{row['assay']}, arm {row['arm']}, order {','.join(row['stim_order'])}) ==="]
    c = res["cc_row"]
    L.append(f"counts     isx {c.get('n_frames_isx')} frames ({res['n_dropped']} dropped, "
             f"{rec.get('n_trimmed_early')} trimmed) | logger {c.get('n_ttl')} pulses | "
             f"cam1 {c.get('n_cam1_frames')} | cam2 {c.get('n_cam2_frames')} | GPIO "
             f"{c.get('n_gpio_sync_edges')} -> count_check {c.get('count_check')} "
             f"(K={c.get('pulse_to_frame_offset')}, cam_lead={c.get('cam_lead_frames')})")
    for f in res["cc_fails"]:
        L.append(f"           FAIL {f}")
    fps = c.get("fps")
    L.append(f"clock      t_master from {res['t_source']}; fps {float(fps):.4f}; Arduino fit "
             f"{mt._num(rec.get('arduino_fit_ppm')):+.1f} ppm, residual "
             f"{mt._num(rec.get('arduino_fit_resid_ms')):.2f} ms")
    cams = ", ".join(f"{k} {v.get('method')}" + (f" (uncertain from frame {v['uncertain_from']})"
                                                 if v.get("uncertain_from") else "")
                     for k, v in res["cam"].items())
    L.append(f"cameras    {cams}")
    sched = se[se["slot"].notna()]
    by = sched.groupby(["stim_type", "source"]).size().unstack(fill_value=0)
    parts = []
    for st in row["stim_order"]:
        if st in by.index:
            parts.append(f"{st}: " + ", ".join(f"{k} {int(v)}" for k, v in by.loc[st].items() if v))
    L.append(f"stimuli    {len(sched)} scheduled-trial rows | " + " | ".join(parts))
    flags = pd.Series([f for x in se["flags"].fillna("") for f in str(x).split(";") if f])
    if len(flags):
        L.append("           flags: " + ", ".join(f"{k} {v}" for k, v in flags.value_counts().items()))
    if len(be):
        cnt = be["type"].value_counts().to_dict()
        linked = int(be["event_id"].notna().sum())
        L.append(f"behaviour  {len(be)} events ({', '.join(f'{k} {v}' for k, v in cnt.items())}); "
                 f"{linked} linked to a stimulus")
        w = be[(be["type"] == "withdrawal") & be["latency_s"].notna()]
        if len(w):
            stim_of = se.set_index("event_id")["stim_type"]
            lat = w.assign(stim=w["event_id"].map(stim_of)).groupby("stim")["latency_s"].median()
            L.append("           median withdrawal latency (s): " +
                     ", ".join(f"{k} {v:.3f}" for k, v in lat.items()))
        bf = pd.Series([f for x in be["flags"].fillna("") for f in str(x).split(";") if f])
        if len(bf):
            L.append("           flags: " + ", ".join(f"{k} {v}" for k, v in bf.value_counts().items()))
    else:
        L.append("behaviour  (no scoring file)")
    ep = fr["epoch"].value_counts().to_dict()
    L.append(f"frames     {len(fr)} rows; " + ", ".join(f"{k} {v}" for k, v in ep.items()) +
             f"; stim_on {int(fr['stim_on'].sum())}; extract_row {int(fr['extract_row'].notna().sum())}")
    for n_ in tl["notes"]:
        L.append(f"note       {n_}")
    for w_ in res["cc_warns"]:
        L.append(f"warn       {w_}")
    for lvl, code, msg in res["ps"]["issues"]:
        if lvl != "INFO":
            L.append(f"{lvl.lower():10s} pressure_sync {code}: {msg}")
    L.append(f"outputs    {res['out_dir']}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- selftest
def _check_truth(truth, out_dir, check, exact_before=None):
    """Compare pipeline outputs with the synthetic ground truth. exact_before: cam1 frames
    >= this are expected to be flagged uncertain instead of exact.

    Pressure onsets are checked twice. The mapping (this pipeline's job) must be exact for
    the onset time pressure_sync reported: count = pulses before it (pressure_frames
    t_pulse_arduino_s), isx_frame = count_to_isx_frame(count) = count + K, cam frame = the camera
    frame of that pulse. The detection must give the TRUE contact's frame: contacts are spread
    evenly over the pulse interval, and only a contact flagged pulse_ambiguous and within two
    logger samples (29 ms; a faint light touch is timed to about a sample) of a pulse may land
    one frame off. Offsets: the last stim_on
    frame is the release's frame unless the release is within one sample of a pulse. Times are
    checked too: logger onsets against the true contact (Arduino fit), camera events against
    their true trigger time and latency.
    Returns (stimulus rows exact vs truth, behaviour events exact, onsets one frame off next to
    a pulse)."""
    import synthetic_session as syn
    one_sample = syn.SAMPLE_GAP_MAX_S
    fr = pd.read_csv(os.path.join(out_dir, "frames.csv"), dtype={"event_id": str}, low_memory=False)
    se = pd.read_csv(os.path.join(out_dir, "stim_events.csv"), dtype={"event_id": str, "flags": str})
    be = pd.read_csv(os.path.join(out_dir, "behaviour_events.csv"), dtype={"event_id": str, "flags": str})
    rec = pd.read_csv(os.path.join(out_dir, "recordings.csv")).iloc[0]
    p_no, p_t = mt._pulse_lookup(pd.read_csv(os.path.join(out_dir, "pressure_frames.csv")))
    lost = np.array(sorted(truth["cam1_lost_pulses"]), dtype=int)
    K = truth["pulse_to_frame_offset"]

    def cam1_of_pulse(p):
        return None if p in lost else int(p - np.searchsorted(lost, p))

    v = truth["variant"]
    check(int(rec.n_frames_isx) == truth["n_frames_isx"] and int(rec.n_ttl) == truth["n_ttl"]
          and int(rec.n_dropped) == truth["n_dropped"]
          and int(rec.n_trimmed_early) == truth["n_trimmed_early"], f"{v}: recordings counts")
    check(abs(float(rec.period_s) - truth["true_period_s"]) < 1e-6, f"{v}: true frame period")
    check(len(fr) == truth["n_frames_isx"] and fr.t_master_s.iloc[0] == 0.0, f"{v}: frames rows")
    check(set(fr.index[fr.is_dropped == 1]) == set(truth["dropped_frames"]), f"{v}: dropped frames")
    for isx, er in truth["extract_row"].items():
        got = fr.extract_row.iloc[int(isx)]
        check((pd.isna(got) and er is None) or (er is not None and got == er),
              f"{v}: extract_row at isx {isx} = {got}, want {er}")

    key = lambda d: (int(d["slot"]), int(d["block"]), int(d["trial"]))
    sched = {key(r): r for _, r in se[se.slot.notna()].iterrows()}
    n_ok = n_miss = 0
    t_err = []
    for t in truth["stim_events"]:
        r = sched.get(key(t))
        if r is None:
            check(False, f"{v}: no stim_events row for trial {key(t)}")
            continue
        want = (t["source"], t["stim_type"], t["onset_isx_frame"], t["onset_cam_frame"])
        if t["source"] == "pressure":
            # the pulse count of the onset time the detector reported
            det = mt.arduino_time_to_pulse(float(r.onset_t_arduino_s), p_no, p_t)
            t_err.append(float(r.onset_t_master_s) - t["contact_t_master_s"])
            check(int(r.onset_isx_frame) == mt.count_to_isx_frame(det, K)
                  and int(r.onset_cam_frame) == cam1_of_pulse(det),
                  f"{v}: trial {key(t)}: mapping of the detected onset (pulse {det}) is wrong: "
                  f"isx {r.onset_isx_frame} cam {r.onset_cam_frame}")
            if det != t["pulse"]:
                # a faint light touch is timed to ~1-1.5 samples, so 'next to a pulse' is two
                n_miss += 1
                check(t["pulse_margin_s"] < 2 * one_sample and abs(det - t["pulse"]) == 1
                      and "pulse_ambiguous" in str(r["flags"]),
                      f"{v}: trial {key(t)}: onset in pulse {det}, true {t['pulse']}, contact "
                      f"{t['pulse_margin_s'] * 1000:.1f} ms from a pulse, flags {r['flags']} (only "
                      "a pulse_ambiguous contact next to a pulse may be one frame off)")
                continue
            # the release's frame ends stim_on (exact unless it is next to a pulse)
            off = int(r.offset_isx_frame)
            if t["release_pulse_margin_s"] >= one_sample:
                check(off == t["offset_isx_frame"] and fr.stim_on.iloc[off] == 1
                      and fr.stim_on.iloc[off + 1] == 0,
                      f"{v}: trial {key(t)}: stim_on ends at isx {off}, release in {t['offset_isx_frame']}")
            else:
                check(abs(off - t["offset_isx_frame"]) <= 1, f"{v}: trial {key(t)}: offset {off}")
            check(fr.stim_on.iloc[int(r.onset_isx_frame) - 1] == 0,
                  f"{v}: trial {key(t)}: stim_on starts before the contact's frame")
        got = (r.source, r.stim_type, int(r.onset_isx_frame), int(r.onset_cam_frame))
        ok = got == want
        if t["source"] == "video" and exact_before is not None and t["onset_cam_frame"] >= exact_before:
            ok = r.source == "video" and "cam_uncertain" in str(r["flags"])
        elif t["source"] == "video":
            # the snapshot time of the scored frame = its trigger pulse (the TSC time of the
            # frame that pulse ends; the true edge is 0.4 ms later)
            ok = ok and abs(float(r.onset_t_master_s) - t["onset_t_master_s"]) < 0.002
        check(ok, f"{v}: trial {key(t)}: got {got} at {float(r.onset_t_master_s):.4f} s, want {want}"
                  + (f" at {t['onset_t_master_s']:.4f} s" if "onset_t_master_s" in t else ""))
        n_ok += ok
        i = int(r.onset_isx_frame)
        check(fr.stim_on.iloc[i] == 1 and fr.event_id.iloc[i] == r.event_id,
              f"{v}: frames.stim_on/event_id at onset of {r.event_id}")
    # The onset is the knee of the rise (sub-sample), mapped by the Arduino fit. The median
    # guards against a systematic shift such as a one-sample (~12 ms) or one-frame (~50 ms) lead.
    t_err = np.array(t_err)
    check(t_err.size > 0 and np.abs(t_err).max() < 0.020 and abs(np.median(t_err)) < 0.003,
          f"{v}: logger onset_t_master_s - true contact: median {np.median(t_err) * 1000:.1f} ms, "
          f"max |err| {np.abs(t_err).max() * 1000:.1f} ms (want < 3 / 20 ms; the faintest light "
          "touches, ~15 noise SD with two samples on the rise, are good to ~1 sample)")
    print(f"{v}: logger onset time error vs the true contact: median {np.median(t_err) * 1000:.2f} ms, "
          f"max |err| {np.abs(t_err).max() * 1000:.2f} ms")
    check(len(sched) == 60 and (se.source == "expected").sum() == 0, f"{v}: 60 trials, none missing")

    ev_key = {r.event_id: (int(r.slot), int(r.block), int(r.trial)) for _, r in se[se.slot.notna()].iterrows()}
    n_beh = 0
    lat_err = []
    for t in truth["behaviour_events"]:
        m = be[(be.type == t["type"]) & (be.on_cam_frame == t["on_cam_frame"])]
        if len(m) != 1:
            check(False, f"{v}: behaviour {t['type']} at cam {t['on_cam_frame']}: {len(m)} rows")
            continue
        r = m.iloc[0]
        if exact_before is not None and t["off_cam_frame"] >= exact_before:
            check("cam_uncertain" in str(r["flags"]), f"{v}: {t['type']} at cam {t['on_cam_frame']} "
                                                   "after the camera mismatch must be flagged")
            continue
        ok = int(r.on_isx_frame) == t["on_isx_frame"] and int(r.off_isx_frame) == t["off_isx_frame"]
        ok = ok and abs(float(r.on_t_master_s) - t["on_t_master_s"]) < 0.002
        link = ev_key.get(r.event_id) if isinstance(r.event_id, str) else None
        want = tuple(t["link"]) if t["link"] else None
        ok = ok and link == want
        if want is not None and link == want:
            # latency from the true contact (or the scored heat tap) to the first image
            lat_err.append(float(r.latency_s) - t["latency_s"])
        check(ok, f"{v}: {t['type']} cam {t['on_cam_frame']}: isx {r.on_isx_frame}-{r.off_isx_frame} "
                  f"link {link}; want {t['on_isx_frame']}-{t['off_isx_frame']} link {want}")
        n_beh += ok
        a, b = t["on_isx_frame"], t["off_isx_frame"]
        if t["type"] in mt.AFFECTIVE_TYPES:
            check((fr.affective_state.iloc[a:b + 1] == t["type"]).all(), f"{v}: affective_state frames")
        if t["type"] in mt.REFLEX_TYPES:
            check(t["type"] in str(fr.reflex.iloc[a]), f"{v}: reflex frame")
            check(int(fr.cam1_frame.iloc[a]) == t["on_cam_frame"], f"{v}: frames.cam1_frame")
    lat_err = np.array(lat_err)
    worst = np.abs(lat_err).max() * 1000 if lat_err.size else np.nan
    check(lat_err.size > 0 and worst < 30.0,
          f"{v}: behaviour latency_s vs truth: max |err| {worst:.1f} ms (want < 30 ms)")
    return n_ok, n_beh, n_miss


def _check_recordings(out_root, out_dir, rid, check):
    """count_check's row against synthetic_session's expected recordings row."""
    want = pd.read_csv(os.path.join(out_root, "recordings_expected.csv")).set_index("recording_id").loc[rid]
    got = pd.read_csv(os.path.join(out_dir, "recordings.csv")).set_index("recording_id").loc[rid]
    for c in ("n_frames_isx", "n_dropped", "n_ttl", "n_gpio_sync_edges", "n_cam1_frames",
              "n_cam2_frames", "pulse_to_frame_offset", "cam_lead_frames", "count_check"):
        check(str(got[c]) == str(want[c]), f"{rid}: recordings {c} = {got[c]}, want {want[c]}")
    check(abs(float(got["period_s"]) - float(want["period_s"])) < 1e-6, f"{rid}: period_s")
    if pd.notna(got.get("arduino_fit_ppm")):
        check(abs(float(got["arduino_fit_ppm"]) - float(want["arduino_fit_ppm"])) < 5,
              f"{rid}: Arduino clock rate {got['arduino_fit_ppm']} ppm, want {want['arduino_fit_ppm']:.1f}")


def selftest():
    import synthetic_session as syn
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    out_root = syn.DEFAULT_OUT
    truths, mpath = syn.generate_all(out_root)
    man = mf.load_manifest_strict(mpath, require_files=True)

    # 1. the consistent session: every event exact
    t = truths["good"]
    row = man.row(t["recording_id"])
    out = os.path.join(out_root, "good", "pipeline")
    res = run(row, out, t["pulse_to_frame_offset"], scorer="synthetic", overwrite=True)
    print(summary(res))
    n_ok, n_beh, n_miss = _check_truth(t, out, check)
    _check_recordings(out_root, out, t["recording_id"], check)
    print(f"good: {n_ok}/60 stimulus onsets and {n_beh}/{len(t['behaviour_events'])} behaviour "
          f"events exact vs ground truth; 45/45 pressure onsets mapped exactly from the detected "
          f"onset; {n_miss} contacts within one sample of a pulse one frame off (flagged)")

    # 2. cam1 lost 3 frames: must fail loudly, then work with --allow-mismatch + anchors
    t = truths["cam_drop"]
    row = man.row(t["recording_id"])
    out = os.path.join(out_root, "cam_drop", "pipeline")
    try:
        run(row, out, 0, overwrite=True)
        check(False, "cam_drop: must fail without --allow-mismatch")
    except PipelineError as e:
        check(e.step == "count_check" and "cam1 has 3 fewer frames" in str(e),
              f"cam_drop failed in the wrong step: {e}")
        print(f"cam_drop without --allow-mismatch failed as required: {str(e)[:120]}...")
    res = run(row, out, 0, allow_mismatch=True, anchors={"cam1": t["files"]["anchors_cam1"]},
              overwrite=True)
    info = res["cam"]["cam1"]
    anchors = pd.read_csv(t["files"]["anchors_cam1"])
    first_lost = min(t["cam1_lost_pulses"])
    want_from = int(anchors[anchors.pulse < first_lost].cam_frame.max()) + 1
    check(info["method"] == "fit" and info["uncertain_from"] == want_from,
          f"cam_drop: fit table uncertain from {info.get('uncertain_from')}, want {want_from}")
    cf = pd.read_csv(os.path.join(out, "camera_frames.csv"))
    c1 = cf[cf.camera == "cam1"]
    check((c1.method == "fit").all() and (c1[c1.cam_frame < want_from].pulse ==
                                          c1[c1.cam_frame < want_from].cam_frame).all() and
          (c1[c1.cam_frame >= want_from].uncertain == 1).all(), "cam_drop: camera_frames fit/uncertain")
    n_ok, n_beh, n_miss = _check_truth(t, out, check, exact_before=want_from)
    _check_recordings(out_root, out, t["recording_id"], check)
    print(f"cam_drop with anchors: {n_ok}/60 onsets as expected ({n_miss} pressure onset "
          f"detected one pulse off, mapping still exact), {n_beh} behaviour events exact before "
          f"cam frame {want_from}, later ones flagged")

    # 3. logger started by hand mid-recording: must fail loudly
    t = truths["manual_start"]
    out = os.path.join(out_root, "manual_start", "pipeline")
    try:
        run(man.row(t["recording_id"]), out, 0, overwrite=True)
        check(False, "manual_start: must fail")
    except PipelineError as e:
        check(e.step == "count_check" and "logger pulses" in str(e), f"manual_start: wrong failure {e}")
        print(f"manual_start failed as required: {str(e)[:120]}...")
    _check_recordings(out_root, out, t["recording_id"], check)
    check(not os.path.exists(os.path.join(out, "frames.csv")), "manual_start: no frames.csv written")

    # 4. every other refusal, on copies of the good recording's manifest row
    t = truths["good"]
    row = man.row(t["recording_id"])

    def must_fail(what, step, text, out_name, **kw):
        try:
            run(kw.pop("row", row), os.path.join(out_root, "good", out_name), 0, **kw)
            check(False, f"{what}: must fail")
        except PipelineError as e:
            check(e.step == step and text in str(e), f"{what}: wrong failure {e}")
            print(f"{what}: failed in {e.step} as required")

    must_fail("an out_dir with an earlier run", "out_dir", "--overwrite", "pipeline")
    gpio = pd.read_csv(row["gpio"])
    gap = os.path.join(out_root, "good", "inscopix", f"{t['recording_id']}_gpio_gap.csv")
    gpio[~gpio["Time (s)"].between(600.0, 650.0)].to_csv(gap, index=False)
    must_fail("1000 SYNC edges missing from the GPIO export", "count_check", "fewer SYNC edges",
              "pipeline_gpiogap", row=dict(row, gpio=gap), allow_gpio_tail=True, overwrite=True)
    must_fail("cam2 video missing", "count_check", "cam2 video not available", "pipeline_nocam2",
              row=dict(row, cam2_video=""), overwrite=True)
    res = run(dict(row, cam2_video=""), os.path.join(out_root, "good", "pipeline_nocam2"), 0,
              no_camera=("cam2",), overwrite=True)
    check(res["tl"]["frames"]["cam2_frame"].isna().all() and res["cam"]["cam2"]["method"] == "(no video)",
          "--no-camera cam2: the run completes and frames.csv has no cam2 frames")
    wrong = [row["stim_order"][i] for i in (1, 0, 2, 3)]           # mild and heat swapped
    must_fail("a wrong stim_order", "pressure_sync", "stim_order", "pipeline_wrongorder",
              row=dict(row, stim_order=wrong), overwrite=True)

    if failures:
        print(f"run_pipeline selftest FAILED ({len(failures)}):")
        for f_ in failures[:40]:
            print("  -", f_)
        return 1
    print("run_pipeline selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--recording-id")
    ap.add_argument("--out-dir", help="default: out/<recording_id> in the pipeline folder")
    ap.add_argument("--pulse-offset", type=int, help="K (default: manifest pulse_to_frame_offset)")
    ap.add_argument("--scorer", default="", help="scorer ID written to behaviour_events")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="let camera_sync write a 'fit' table when a camera count differs")
    ap.add_argument("--anchors-cam1", help="cam_frame,pulse anchors for cam1 (with --allow-mismatch)")
    ap.add_argument("--anchors-cam2", help="cam_frame,pulse anchors for cam2 (with --allow-mismatch)")
    ap.add_argument("--no-camera", action="append", choices=camera_sync.CAMERAS, default=[],
                    help="run without this camera (its count_check FAIL is accepted, no rows "
                         "are written for it); may be given twice")
    ap.add_argument("--allow-gpio-tail", action="store_true",
                    help="accept count_check's benign FAIL: a GPIO export one edge short because "
                         "the GPIO log stopped before the last pulse")
    ap.add_argument("--trust-stim-order", action="store_true",
                    help="keep the manifest stim_order although the logger or the scored delivery "
                         "keys contradict it")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete the outputs of an earlier run in --out-dir first (otherwise "
                         "such a folder is refused)")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.manifest and args.recording_id):
        ap.error("--manifest and --recording-id are required")
    try:
        man = mf.load_manifest_strict(args.manifest)
        row = man.row(args.recording_id)
    except mf.ManifestError as e:
        print(f"PIPELINE FAILED [manifest] {e}", file=sys.stderr)
        return 1
    K = args.pulse_offset if args.pulse_offset is not None else row["pulse_to_frame_offset"]
    if K is None:
        print("PIPELINE FAILED [manifest] K (pulse_to_frame_offset) is not set: give "
              "--pulse-offset or the manifest column (fixed with the rig test recording)",
              file=sys.stderr)
        return 1
    out_dir = args.out_dir or os.path.join(HERE, "out", args.recording_id)
    anchors = {k: p for k, p in (("cam1", args.anchors_cam1), ("cam2", args.anchors_cam2)) if p}
    try:
        res = run(row, out_dir, K, args.scorer, args.allow_mismatch, anchors, args.no_camera,
                  args.allow_gpio_tail, args.trust_stim_order, args.overwrite)
    except PipelineError as e:
        print(f"\nPIPELINE FAILED {e}", file=sys.stderr)
        return 1
    print(summary(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
