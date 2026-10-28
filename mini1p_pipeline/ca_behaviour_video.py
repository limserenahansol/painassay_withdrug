"""Representative behaviour + Ca video of one session (mini2p DS-1 style, extended).

  top     side camera | bottom camera (the scored views) | Ca movie: the real dF/F movie frame by frame
          (motion-corrected, F0 = the day mean image, 1-photon haze removed, one display range for
          the clip; the shown cells thinly outlined in their trace colour)
  bottom  the shown cells' local dF/F, the mean of ALL kept cells, and the behaviour under it:
          deliveries (colour = stimulus), withdrawal / flinch taps, movement, grooming;
          a red cursor moves with the video. Events are also written on the bottom view when
          they happen.
Frames are joined by number (K = 0): camera frame n = isx_frame n = the trace row of isx_frame n.

Window: --t0 / --dur seconds, or --auto: the dur-s stretch with the most withdrawals inside the
block of --stim (default Heat, else Pin prick); --stim "movement onset": the dur-s stretch of the
no-stimulus start (0-180 s) with the most movement onsets (spontaneous movement, no stimulus);
cells: the --n-cells with a time-locked UP response to that event in ca_d1_cells_event_tests.csv
(if given; largest first), then the cells with the most transients in the window (DOWN-only cells
are not picked).

  python ca_behaviour_video.py --day-root <mini1p_out/20261019/5941_d1> --rid 5941_d1_post
      --scoring-dir <final> --out-dir <analysis/ca_videos> --auto [--event-tests <csv>]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import winwords as WW  # noqa: E402  (10-23: the shade = the stimulus response window of the event analysis)
SHADE = WW.stim()[3] - WW.stim()[2]

FPS = 30.0
FF = r"C:\Users\hsollim\Downloads\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
BLOCKS = [("stim1", 180, 360), ("stim2", 390, 570), ("stim3", 600, 780), ("stim4", 810, 990)]
SCOL = {"Light touch": "#5dade2", "Mild touch": "#f5b041", "Pin prick": "#e74c3c", "Heat": "#8e44ad"}
TRC = ["#e6194b", "#3cb44b", "#c9a400", "#4363d8", "#f58231", "#911eb4", "#0f9b9b", "#f032e6", "#6b8e00", "#c97b7b"]
W_OUT, H_OUT = 1920, 1080


def hex2bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r if r.ndim == 3 else r[:, :, None]
    return out, s, x0, y0


class BlockReader:
    """rows of a chunked (gzip) h5 movie, read 100 frames at a time (two blocks cached)"""
    def __init__(self, ds, blk=100):
        self.ds, self.blk, self.cache = ds, blk, {}

    def get(self, r):
        b0 = (r // self.blk) * self.blk
        if b0 not in self.cache:
            if len(self.cache) >= 2:
                self.cache.pop(max(self.cache, key=lambda k: abs(k - b0)))      # the block farthest away
            self.cache[b0] = self.ds[b0:b0 + self.blk].astype(np.float32).transpose(0, 2, 1)
        return self.cache[b0][r - b0]


def scoring_for(scoring_dir, mouse, phase):
    for p in glob.glob(os.path.join(scoring_dir, "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True)
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        if str(S["mouseID"]).strip() == mouse and ph == phase:
            return S
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--rid", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--event-tests", default="")
    ap.add_argument("--stim", default="", help="Light touch | Mild touch | Pin prick | Heat | movement onset (default Heat, else Pin prick)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    tag = "_".join(a.rid.split("_")[:2])
    mouse, phase = tag.split("_")[0], a.rid.split("_")[-1]
    S = scoring_for(a.scoring_dir, mouse, phase)
    if S is None:
        sys.exit("no scoring file for " + a.rid)
    stim = [str(x) for x in np.atleast_1d(S["stimNames"])]
    dF = np.atleast_1d(S["dFrames"]).astype(int); dT = np.atleast_1d(S["dTypes"]).astype(int)
    re = np.atleast_2d(S["reflexEvents"]).astype(int).reshape(-1, 2)
    move = np.asarray(S["moveVec"]).astype(bool).ravel() if "moveVec" in S else None
    groom = np.asarray(S["groomVec"]).astype(bool).ravel() if "groomVec" in S else None
    target = a.stim or next((s for s in ("Heat", "Pin prick") if s in stim), stim[0])
    if target != "movement onset" and target not in stim:
        sys.exit(f"{a.rid}: {target!r} is not a stimulus of this session ({stim})")
    if target == "movement onset" and (a.auto or a.t0 is None):
        # spontaneous movement: the dur-s stretch of the no-stimulus start with the most movement onsets
        if move is None:
            sys.exit(f"{a.rid}: no movement vector in the scoring file")
        mon = np.flatnonzero(np.diff(np.r_[0, move.astype(int)]) == 1) / FPS
        starts = np.arange(0.0, max(BLOCKS[0][1] - a.dur, 0.0) + 0.1, 5.0)
        cnt = [np.sum((mon >= s0) & (mon < s0 + a.dur)) for s0 in starts]
        t0 = float(starts[int(np.argmax(cnt))])
    elif a.auto or a.t0 is None:
        # the dur-s stretch of the block with the most withdrawals (ties: the earliest), so the
        # video shows the behaviour as well as the deliveries
        k = stim.index(target)
        b0, b1 = BLOCKS[k][1], BLOCKS[k][2]
        wf = re[re[:, 1] == 1, 0] / FPS
        starts = np.arange(b0, max(b1 - a.dur, b0) + 0.1, 5.0)
        cnt = [np.sum((wf >= s0) & (wf < s0 + a.dur)) for s0 in starts]
        t0 = float(starts[int(np.argmax(cnt))])
    else:
        t0 = a.t0
    t1 = t0 + a.dur
    f0, f1 = int(round(t0 * FPS)), int(round(t1 * FPS))          # isx frames = camera frames
    # traces
    T = dict(np.load(os.path.join(a.day_root, "5_analysis", "traces", f"{a.rid}_traces.npz")))
    fr = T["frame"].astype(int)
    ids = T["cell_ids"].astype(int)
    D = T["dff"]
    rows = np.searchsorted(fr, np.arange(f0, f1))
    rows = np.clip(rows, 0, len(fr) - 1)
    Dw = D[rows]
    # which cells to show (2026-10-21, you: "more neurons fire during Heat than the 8 shown"): first
    # the cells with a time-locked UP response to this stimulus (time-shift rule; DOWN is mostly a
    # demixing artefact, see R4), then the cells that fire most (OASIS transients) in this window
    Ev = pd.read_csv(os.path.join(a.day_root, "5_analysis", "traces", f"{a.rid}_events.csv"))
    nev = Ev[(Ev.onset_s >= t0) & (Ev.onset_s < t1)].groupby("cell").size().reindex(ids, fill_value=0)
    up = []
    if a.event_tests and os.path.isfile(a.event_tests):
        R = pd.read_csv(a.event_tests)
        R = R[(R["mouse"].astype(str) == mouse) & (R.phase == phase) & (R.event == target)]
        R = R[R.cell.isin(ids) & (R.cls_shift == "UP")].sort_values("resp_mean", ascending=False)
        up = [int(c) for c in R.cell.values[:a.n_cells]]
    rest = [int(c) for c in nev.sort_values(ascending=False).index if int(c) not in up]
    pick = np.array((up + rest)[:a.n_cells], int)
    tagof = {c: (f"{target} UP" if c in up else f"{int(nev[c])} tr") for c in pick}
    why = (f"{len(up[:a.n_cells])} cells with a {target} UP response + the most active cells in this window "
           f"(label: '{target} UP' or the number of transients in these {a.dur:.0f} s)")
    cj = [int(np.where(ids == c)[0][0]) for c in pick]
    # footprints + day movie
    M = sio.loadmat(os.path.join(a.day_root, "3_extract", f"{tag}_extract.mat"), squeeze_me=True, struct_as_record=False,
                    variable_names=["spatial_weights", "mean_image", "frame_table"])
    Sp = np.asarray(M["spatial_weights"], np.float32)
    F0 = np.asarray(M["mean_image"], np.float32)
    ft = M["frame_table"]
    FT = pd.DataFrame({k: np.atleast_1d(getattr(ft, k)) for k in ("recording_id", "isx_frame", "valid", "h5_row")})
    FT = FT[FT.recording_id.astype(str) == a.rid]
    h5row = dict(zip(FT.isx_frame.astype(int), FT.h5_row.astype(float)))
    jf = os.path.join(a.day_root, "1_movie", f"{tag}_inscopix.json")
    if os.path.isfile(jf):                                         # whole day: the day view of pre + post
        hf = json.load(open(jf))["files"]["h5"]
        hf = hf if os.path.isfile(hf) else os.path.join(a.day_root, "1_movie", os.path.basename(hf))
    else:                                                          # a single-recording run (preview / test)
        hf = os.path.join(a.day_root, "1_movie", f"{a.rid}_mc.h5")
    contours = []
    for c in pick:
        fp = Sp[:, :, c - 1]
        m = (fp >= 0.4 * fp.max()).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.append(cs)
    # every OTHER selected ROI too (thin white outline), so each flashing cell can be checked
    others = []
    for c in ids:
        if c in set(int(x) for x in pick) or c - 1 >= Sp.shape[2]:
            continue
        fp = Sp[:, :, c - 1]
        if fp.max() <= 0:
            continue
        cs, _ = cv2.findContours((fp >= 0.3 * fp.max()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        others.append((int(c), cs))
    # static trace panel
    FS = float(os.environ.get("CA_VIDEO_FS", "1"))                 # font scale (the PI video: 2)
    CAMH = 470 if FS > 1 else 560                                  # height of the camera / Ca row (top band 40 px)
    PW, PH = W_OUT, H_OUT - 50 - CAMH
    fig = plt.figure(figsize=(PW / 100, PH / 100), dpi=100)
    if FS > 1:                                                     # room for the bigger labels and the time-axis label
        ax = fig.add_axes([0.12, 0.46, 0.86, 0.45])
        axb = fig.add_axes([0.12, 0.14, 0.86, 0.255], sharex=ax)
    else:
        ax = fig.add_axes([0.06, 0.30, 0.92, 0.66])
        axb = fig.add_axes([0.06, 0.05, 0.92, 0.20], sharex=ax)
    tt = np.arange(f0, f1) / FPS
    sc = np.nanpercentile(np.abs(D[:, cj]), 99.5, axis=0)
    sc[~(sc > 0)] = 1
    n_show = len(cj)
    for i, j in enumerate(cj):
        ax.plot(tt, np.nan_to_num(Dw[:, j] / sc[i]) * 0.9 + (n_show - i), color=TRC[i % len(TRC)], lw=1)
        if FS > 1:                                                  # one short line per cell (two lines overlapped at this size)
            lab_ = f"#{ids[j]}" + (" UP" if tagof.get(int(ids[j]), "").endswith("UP") else "")
            ax.text(t0 - 0.3, n_show - i, lab_, color=TRC[i % len(TRC)], ha="right", va="center", fontsize=11 * FS,
                    fontweight="bold")
        else:
            ax.text(t0 - 0.3, n_show - i + 0.3, f"#{ids[j]}\n{tagof.get(int(ids[j]), '')}", color=TRC[i % len(TRC)],
                    ha="right", fontsize=8)
    pm = np.nanmean(Dw, 1)
    ax.plot(tt, (pm - np.nanmean(pm)) / (np.nanstd(pm) + 1e-9) * 0.25 + 0, color="k", lw=1.2)
    ax.text(t0 - 0.3, 0.0 if FS > 1 else 0.3, f"mean ({len(ids)})" if FS > 1 else f"mean of\nall {len(ids)}", ha="right",
            va="center" if FS > 1 else "baseline", fontsize=11 * FS if FS > 1 else 8, fontweight="bold" if FS > 1 else "normal")
    if FS > 1:
        ax.tick_params(labelbottom=False)
    ax.set_yticks([]); ax.set_xlim(t0, t1); ax.set_ylim(-0.8, n_show + 1.2)
    where = f"{target} block" if target != "movement onset" else "no-stimulus start, spontaneous movement"
    if FS > 1:
        ax.set_title(f"{a.rid}, {where}: dF/F of {len(cj)} cells (each scaled to its max); black = mean of all {len(ids)}; "
                     f"shade = {WW.stim_span()}", fontsize=10.5 * FS, loc="left")
    else:
        ax.set_title(f"{a.rid}  {where}  |  local dF/F of {why} (each scaled to its own 99.5th pct), "
                     f"black = mean of all kept cells", fontsize=10)
    for f, k in zip(dF, dT):
        if f0 <= f < f1 and 1 <= k <= len(stim):
            ax.axvline(f / FPS, color=SCOL.get(stim[k - 1], "k"), lw=0.8, alpha=0.6)
            if FS > 1:
                ax.axvspan(f / FPS + WW.stim()[2], f / FPS + WW.stim()[2] + SHADE, color=SCOL.get(stim[k - 1], "k"), alpha=0.12, lw=0)
            axb.plot([f / FPS] * 2, [3.6, 4.4], color=SCOL.get(stim[k - 1], "k"), lw=1.5)
    for typ, y, col in ((1, 3, "#6c3483"), (2, 2, "#1e8449")):
        x = re[(re[:, 1] == typ) & (re[:, 0] >= f0) & (re[:, 0] < f1), 0] / FPS
        axb.plot(x, np.full(len(x), y), "|", color=col, ms=10, mew=2)
    for v, y, col in ((move, 1, "#17a2b8"), (groom, 0, "#d6338a")):
        if v is not None:
            seg = v[f0 - 1:f1 - 1] if f0 >= 1 else v[:f1 - 1]
            axb.fill_between(tt[:len(seg)], y - 0.35, y + 0.35, where=seg, color=col, lw=0, step="mid")
    axb.set_yticks([4, 3, 2, 1, 0], ["deliveries", "withdrawal", "flinch", "movement", "grooming"],
                   fontsize=10 * FS if FS > 1 else 8)
    axb.set_ylim(-0.6, 4.6)
    axb.tick_params(axis="x", labelsize=10 * FS if FS > 1 else 8)
    axb.set_xlabel("time in session (s)", fontsize=10 * FS if FS > 1 else 9)
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
    x_of = lambda t: int(round(ax.transData.transform((t, 0))[0]))
    plt.close(fig)
    # video
    cs_, cb_ = cv2.VideoCapture(str(S["sidFile"])), cv2.VideoCapture(str(S["botFile"]))
    for c in (cs_, cb_):
        c.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)                     # camera frame f0 (1-based)
    out = os.path.join(a.out_dir, f"{a.rid}_behaviour_ca_{target.replace(' ', '')}_{int(t0)}s.mp4")
    vis = []
    h = h5py.File(hf, "r")
    rd = BlockReader(h["mov"])                                     # h5py: (frames, W, H) of MATLAB [H W n]
    rr = [h5row.get(f, np.nan) for f in range(f0, f1)]
    okr = [int(r) - 1 for r in rr if np.isfinite(r)]
    if not okr:
        h.close()
        sys.exit(f"{a.rid}: the window {t0:.0f}-{t1:.0f} s is outside the movie")
    lo, hi = min(okr), max(okr)
    pp = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_OUT}x{H_OUT}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out],
                          stdin=subprocess.PIPE)
    # display (2026-10-21, you: "the Ca panel looked like a static mean picture"): the REAL movie,
    # frame by frame - dF/F = (F - F0) / F0 of the motion-corrected movie (F0 = the day mean image,
    # 5-frame (0.17 s) running mean), with the 1-photon haze removed (difference of Gaussians, 2 - 12 px),
    # the dark lens rim masked;
    # one fixed display range for the whole clip (median .. 99.9th percentile), so a cell that fires
    # flashes and a quiet one stays dark
    hp = lambda x: cv2.GaussianBlur(x, (0, 0), 2.0) - cv2.GaussianBlur(x, (0, 0), 12.0)
    F0s = np.maximum(F0, 1.0)
    act_at = lambda r: hp(((sum(rd.get(min(max(q, lo), hi)) for q in range(r - 2, r + 3)) / 5) - F0s) / F0s)
    # the lit lens: the smoothed mean image above 35 % of its range (dark rim = noise in dF/F)
    sm0 = cv2.GaussianBlur(F0, (0, 0), 15.0)
    q1, q99 = np.percentile(sm0, [1, 99])
    lens = sm0 > q1 + 0.35 * (q99 - q1)
    smp = np.stack([act_at(r) for r in okr[::max(len(okr) // 40, 1)]])
    vmin, vmax = float(np.percentile(smp[:, lens], 75)), float(np.percentile(smp[:, lens], 99.95))
    # colour bar of the Ca panel (rule: every heatmap has a colour bar with its unit), in the dark corner
    # below-left of the lens: inferno from vmin (dark) to vmax (bright), the unit = dF/F after haze removal
    CBX0, CBX1, CBY0, CBY1 = 1300, 1440, 40 + CAMH - 26, 40 + CAMH - 8
    cbar = cv2.applyColorMap(np.tile(np.linspace(0, 255, CBX1 - CBX0).astype(np.uint8), (CBY1 - CBY0, 1)),
                             cv2.COLORMAP_INFERNO)
    cb_lo, cb_hi = f"{100 * vmin:.1f}", f"{100 * vmax:.1f}"                # in % (2 decimals of dF/F read 0.00 - 0.02)
    (tw_hi, _), _ = cv2.getTextSize(cb_hi, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ev_text = []
    for f, k in zip(dF, dT):
        if 1 <= k <= len(stim):
            ev_text.append((f, stim[k - 1], hex2bgr(SCOL.get(stim[k - 1], "#ffffff"))))
    for f, typ in re:
        ev_text.append((f, "WITHDRAWAL" if typ == 1 else "FLINCH", (230, 230, 230)))
    # the shown cells' outlines and labels in panel coordinates (the Ca image is fitted into 640 x CAMH)
    _, sc_, px0, py0 = fit(np.zeros(F0.shape + (3,), np.uint8), 640, CAMH)
    cs_panel, lab_pos = [], []
    for q, cs in enumerate(contours):
        cs2 = [np.round(c.astype(np.float32) * sc_ + np.array([px0, py0], np.float32)).astype(np.int32) for c in cs]
        cs_panel.append(cs2)
        allp = np.vstack([c.reshape(-1, 2) for c in cs2]) if cs2 else np.array([[px0, py0]])
        lx, ly = int(allp[:, 0].max()) + 3, int(allp[:, 1].min()) + 6
        lab_pos.append(((min(lx, 640 - 40), max(ly, 14)), f"#{int(pick[q])}"))
    oth_panel = []
    for c, cs in others:
        cs2 = [np.round(cc_.astype(np.float32) * sc_ + np.array([px0, py0], np.float32)).astype(np.int32) for cc_ in cs]
        if not cs2:
            continue
        allp = np.vstack([cc_.reshape(-1, 2) for cc_ in cs2])
        oth_panel.append((cs2, (int(allp[:, 0].max()) + 2, int(allp[:, 1].min()) + 4), str(c)))
    for i, f in enumerate(range(f0, f1)):
        frame = np.zeros((H_OUT, W_OUT, 3), np.uint8)
        ok1, s_img = cs_.read(); ok2, b_img = cb_.read()
        if ok1:
            frame[40:40 + CAMH, 0:640], *_ = fit(s_img, 640, CAMH)
        if ok2:
            frame[40:40 + CAMH, 640:1280], *_ = fit(b_img, 640, CAMH)
        r = rr[i]
        if np.isfinite(r):
            im = np.clip((act_at(int(r) - 1) - vmin) / (vmax - vmin), 0, 1) * lens
            cimg = cv2.applyColorMap((im * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            fit_c, *_ = fit(cimg, 640, CAMH)
            for cs2, (lx, ly), lab in oth_panel:            # all other selected ROIs: outline only
                cv2.drawContours(fit_c, cs2, -1, (235, 235, 235), 1)
            # outlines drawn at the panel size (2 px, black rim for contrast) + the cell number in
            # the colour of its trace below (you: "the marking is too narrow to see which ROI is which")
            for q, cs in enumerate(cs_panel):
                col = hex2bgr(TRC[q % len(TRC)])
                cv2.drawContours(fit_c, cs, -1, (0, 0, 0), 4)
                cv2.drawContours(fit_c, cs, -1, col, 2)
                (lx, ly), lab = lab_pos[q]
                cv2.putText(fit_c, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(fit_c, lab, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            frame[40:40 + CAMH, 1280:1920] = fit_c
        frame[CBY0:CBY1, CBX0:CBX1] = cbar
        cv2.rectangle(frame, (CBX0 - 1, CBY0 - 1), (CBX1, CBY1), (255, 255, 255), 1)
        cv2.putText(frame, cb_lo, (CBX0, CBY0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, cb_hi, (CBX1 - tw_hi, CBY0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "dF/F (%)", (CBX0, CBY0 - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for txt, x in (("side", 10), ("bottom", 650), ("Ca: dF/F; white = all selected ROIs", 1290)):
            cv2.putText(frame, txt, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{a.rid}   t = {f / FPS:7.2f} s   (camera frame {f} = isx frame {f})", (650, 625),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        yy = 80
        for fe, txt, col in ev_text:
            if 0 <= f - fe < int(0.8 * FPS):
                cv2.putText(frame, txt, (660, yy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
                yy += 38
        if move is not None and 1 <= f <= len(move) and move[f - 1]:
            cv2.putText(frame, "moving", (1150, 40 + CAMH - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 180, 25), 2)
        if groom is not None and 1 <= f <= len(groom) and groom[f - 1]:
            cv2.putText(frame, "grooming", (1150, 40 + CAMH - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 80, 165), 2)
        pnl = panel.copy()
        x = x_of(f / FPS)
        cv2.line(pnl, (x, 0), (x, PH - 1), (0, 0, 255), 2)
        frame[H_OUT - PH:, :] = pnl[:PH, :W_OUT]
        pp.stdin.write(frame.tobytes())
        if i == len(rr) // 2:
            vis = frame.copy()
    pp.stdin.close(); pp.wait()
    h.close()
    if len(vis):
        cv2.imwrite(out.replace(".mp4", "_frame.png"), vis)
    print(f"{a.rid}: {a.dur:.0f} s video ({target} block, {len(pick)} cells) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
