"""Behaviour cameras of one mini1p session: video QC, scoring-quality videos, automatic
rearing and movement.

Both cameras are triggered by the Inscopix SYNC pulse, so camera frame n (1-based) was
exposed during isx_frame n (0-based) - every output row carries both numbers.

What it writes (out_dir, <tag>_...)
  video_qc.png / video_qc.json   per camera and frame: brightness, % saturated (>= 250) and
                                 % black (<= 5) pixels, sharpness (variance of the
                                 Laplacian), interlace comb ratio (> 1 = combing), motion
                                 energy, duplicated frames; plus plain recommendations
  bottom_scoring.mp4, side_scoring.mp4
                                 the same frames for scoring: bottom through a dark-expanding
                                 tone curve (paws ~20-40 grey become visible), side unchanged,
                                 30 fps = real time (the AVI header says 60). Frame count is
                                 checked equal to the input, so frame n is still frame n.
  dual_scoring.mp4               bottom | side side by side, with camera frame, isx frame
                                 and time on every frame
  features.csv                   cam_frame, isx_frame, t_s, bottom_motion, side_motion,
                                 side_top_y (top of the mouse + miniscope above the floor),
                                 rearing (0/1)
  rearing_bouts.csv              onset/offset frames and seconds of every rearing bout
  rearing_qc.png                 the side frame at every detected onset, and the top-y trace
                                 with the threshold: check the detector by eye

Rearing (side camera, quick and simple): pixels darker than dark_thr inside the arena box
(x0..x1, y0..floor_y) are the mouse and its miniscope; the cable is light and is ignored.
The top row of the largest dark blob is side_top_y. A rearing bout is side_top_y at least
rear_px above its session median AND an upright blob (height / width >= rear_aspect) for
>= 0.3 s (gaps < 0.3 s merged).

  python behavior_auto.py --bottom B.avi --side S.avi --out-dir D --tag 5941_d1_post \
      [--n-frames 9000] [--fps 30] [--box 190 590 25 360] [--no-video]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


def frame_stats(g, prev):
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    gf = g.astype(np.float32)
    a = np.abs(np.diff(gf, axis=0)).mean()
    b = np.abs(gf[2:] - gf[:-2]).mean()
    small = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    me = np.nan if prev is None else float(np.abs(small - prev).mean())
    dup = prev is not None and me == 0.0
    return dict(mean=float(g.mean()), sat=float((g >= 250).mean() * 100), black=float((g <= 5).mean() * 100),
                sharp=float(lap), comb=float(a / max(b, 1e-6)), motion=me, dup=bool(dup)), small


def top_of_mouse(g, box, dark_thr):
    x0, x1, y0, y1 = box
    roi = g[y0:y1, x0:x1]
    m = (roi < dark_thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.nan, 0, np.nan
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    aspect = st[k, cv2.CC_STAT_HEIGHT] / max(st[k, cv2.CC_STAT_WIDTH], 1)
    return float(st[k, cv2.CC_STAT_TOP] + y0), int(st[k, cv2.CC_STAT_AREA]), float(aspect)


def bouts(x, min_len, max_gap):
    x = np.asarray(x, bool).copy()
    # close gaps shorter than max_gap
    d = np.diff(np.r_[0, x.astype(int), 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    for i in range(len(on) - 1):
        if on[i + 1] - off[i] < max_gap:
            x[off[i]:on[i + 1]] = True
    d = np.diff(np.r_[0, x.astype(int), 0])
    on, off = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (off - on) >= min_len
    y = np.zeros_like(x)
    for a, b in zip(on[keep], off[keep]):
        y[a:b] = True
    return y, on[keep], off[keep]


def run(a):
    os.makedirs(a.out_dir, exist_ok=True)
    P = lambda s: os.path.join(a.out_dir, f"{a.tag}_{s}")
    cb, cs = cv2.VideoCapture(a.bottom), cv2.VideoCapture(a.side)
    nb, ns = int(cb.get(cv2.CAP_PROP_FRAME_COUNT)), int(cs.get(cv2.CAP_PROP_FRAME_COUNT))
    hdr = cb.get(cv2.CAP_PROP_FPS)
    if nb != ns:
        print(f"WARNING: bottom has {nb} frames, side {ns}: both are SYNC-triggered, they should be equal")
    n = min(nb, ns) if a.n_frames is None else min(a.n_frames, nb, ns)
    W, H = int(cb.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # bottom view: the mouse is backlit, fur ~5-20 and PAWS ~20-40 grey. A tone curve that
    # expands 0-50 to 0-190 and compresses 50-255 into 190-255 makes the paws visible while
    # the mesh stays visible (checked on 5941; CLAHE on top washed the paws out again, and a
    # temporal median would blur a fast withdrawal, so neither is used).
    xg = np.arange(256, dtype=np.float32)
    bottom_lut = np.where(xg <= 50, xg * 190 / 50, 190 + (xg - 50) * 65 / 205).clip(0, 255).astype(np.uint8)
    vw = {}
    if not a.no_video:
        four = cv2.VideoWriter_fourcc(*"mp4v")
        vw["bottom"] = cv2.VideoWriter(P("bottom_scoring.mp4"), four, a.fps, (W, H), False)
        vw["side"] = cv2.VideoWriter(P("side_scoring.mp4"), four, a.fps, (W, H), False)
        vw["dual"] = cv2.VideoWriter(P("dual_scoring.mp4"), four, a.fps, (2 * W + 6, H), True)
    S = {"bottom": [], "side": []}
    prev = {"bottom": None, "side": None}
    top = np.full(n, np.nan)
    area = np.zeros(n, int)
    aspect = np.full(n, np.nan)
    for i in range(n):
        okb, fb = cb.read()
        oks, fs = cs.read()
        if not (okb and oks):
            print(f"stopped at frame {i + 1}: read failed (bottom {okb}, side {oks})")
            n = i
            break
        gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
        for nm, g in (("bottom", gb), ("side", gs)):
            st, prev[nm] = frame_stats(g, prev[nm])
            S[nm].append(st)
        top[i], area[i], aspect[i] = top_of_mouse(gs, a.box, a.dark_thr)
        if vw:
            # side view unchanged: its white background is intended (contrast for the dark
            # mouse) and the mouse itself is not clipped
            eb, es = cv2.LUT(gb, bottom_lut), gs
            vw["bottom"].write(eb)
            vw["side"].write(es)
            d = np.hstack([cv2.cvtColor(eb, cv2.COLOR_GRAY2BGR), np.full((H, 6, 3), (0, 200, 0), np.uint8),
                           cv2.cvtColor(es, cv2.COLOR_GRAY2BGR)])
            txt = f"cam frame {i + 1}  isx frame {i + 1}  t {(i + 1) / a.fps:7.2f} s"
            cv2.putText(d, txt, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            vw["dual"].write(d)
        if (i + 1) % 1500 == 0:
            print(f"  {i + 1} / {n} frames")
    for v in vw.values():
        v.release()
    top, area, aspect = top[:n], area[:n], aspect[:n]

    # ---- rearing -----------------------------------------------------------------------
    base = np.nanmedian(top)
    thr = base - a.rear_px
    # upright AND high: a mouse sitting with its back to the camera, or close to it, also
    # looks taller, but it is not tall and narrow (checked on 5941, 2026-10-19)
    raw = (top < thr) & (aspect >= a.rear_aspect)
    rear, on, off = bouts(raw, int(round(0.3 * a.fps)), int(round(0.3 * a.fps)))
    cam = np.arange(1, n + 1)
    with open(P("features.csv"), "w") as f:
        f.write("cam_frame,isx_frame,t_s,bottom_motion,side_motion,side_top_y,side_aspect,rearing\n")
        for i in range(n):
            f.write(f"{cam[i]},{cam[i]},{cam[i] / a.fps:.4f},{S['bottom'][i]['motion']:.4f},"
                    f"{S['side'][i]['motion']:.4f},{top[i]:.1f},{aspect[i]:.3f},{int(rear[i])}\n")
    with open(P("rearing_bouts.csv"), "w") as f:
        f.write("bout,onset_cam_frame,offset_cam_frame,onset_isx_frame,onset_s,duration_s\n")
        for k, (s0, s1) in enumerate(zip(on, off), 1):
            f.write(f"{k},{s0 + 1},{s1},{s0 + 1},{(s0 + 1) / a.fps:.3f},{(s1 - s0) / a.fps:.3f}\n")

    # ---- QC figure + json --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = cam / a.fps
    fig, ax = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    for nm, c in (("bottom", "tab:blue"), ("side", "tab:orange")):
        arr = {k: np.array([s[k] for s in S[nm]]) for k in ("mean", "sat", "black", "sharp", "motion")}
        ax[0].plot(t, arr["mean"], c, lw=0.8, label=nm)
        ax[1].plot(t, arr["sat"], c, lw=0.8)
        ax[1].plot(t, arr["black"], c, lw=0.8, ls=":")
        ax[2].plot(t, arr["sharp"], c, lw=0.8)
        ax[3].plot(t, arr["motion"], c, lw=0.6)
    ax[0].set_ylabel("mean gray"); ax[0].legend(loc="upper right", frameon=False)
    ax[1].set_ylabel("% saturated (solid)\n% black (dotted)")
    ax[2].set_ylabel("sharpness\n(Laplacian var)")
    ax[3].set_ylabel("motion energy")
    ax[4].plot(t, top, "k", lw=0.6)
    ax[4].axhline(thr, color="r", lw=1, ls="--", label=f"rearing threshold (median - {a.rear_px} px)")
    for s0, s1 in zip(on, off):
        ax[4].axvspan((s0 + 1) / a.fps, s1 / a.fps, color="r", alpha=0.2, lw=0)
    ax[4].invert_yaxis()
    ax[4].set_ylabel("side: top of mouse (px)\n(up = higher)")
    ax[4].set_xlabel("time (s) = cam frame / 30")
    ax[4].legend(loc="upper right", frameon=False)
    fig.suptitle(f"{a.tag}: behaviour video QC, {n} frames, {len(on)} rearing bouts")
    fig.tight_layout()
    fig.savefig(P("video_qc.png"), dpi=110)
    plt.close(fig)

    # onset montage
    if len(on):
        k = min(len(on), 12)
        idx = on[np.linspace(0, len(on) - 1, k).astype(int)]
        cs2 = cv2.VideoCapture(a.side)
        tiles = []
        for j in idx:
            cs2.set(cv2.CAP_PROP_POS_FRAMES, int(j) + int(round(0.3 * a.fps)))
            ok, im = cs2.read()
            if not ok:
                continue
            x0, x1, y0, y1 = a.box
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 200, 0), 1)
            cv2.line(im, (x0, int(thr)), (x1, int(thr)), (0, 0, 255), 1)
            cv2.putText(im, f"rear onset {(j + 1) / a.fps:.1f} s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            tiles.append(cv2.resize(im, (W // 2, H // 2)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
        cv2.imwrite(P("rearing_qc.png"), np.vstack(rows))

    summ = {}
    for nm in ("bottom", "side"):
        arr = {k: np.array([s[k] for s in S[nm]], float) for k in ("mean", "sat", "black", "sharp", "comb", "motion")}
        rec = []
        if np.nanmedian(arr["sat"]) > 10:
            rec.append(f"{np.nanmedian(arr['sat']):.0f} % of pixels are saturated: lower exposure/gain or "
                       "light so the white background is not clipped (detail is lost there)")
        if np.nanmedian(arr["black"]) > 10:
            rec.append(f"{np.nanmedian(arr['black']):.0f} % of pixels are black: more light or gain")
        if nm == "bottom":
            rec.append("the mouse is backlit from below: paws are recorded only faintly (grey ~20-40); the "
                       "scoring video expands the dark range so they show. More light from the camera side "
                       "or a little more exposure would give them more grey levels")
        if np.nanmedian(arr["comb"]) > 1.0:
            rec.append("interlace combing: deinterlace before scoring")
        jumps = np.where(np.abs(np.diff(arr["mean"])) > 3)[0]
        if len(jumps):
            rec.append(f"{len(jumps)} sudden brightness jump(s), first at {(jumps[0] + 2) / a.fps:.1f} s "
                       "(light change, object or hand in view)")
        summ[nm] = dict(median_gray=float(np.nanmedian(arr["mean"])),
                        median_pct_saturated=float(np.nanmedian(arr["sat"])),
                        median_pct_black=float(np.nanmedian(arr["black"])),
                        median_sharpness=float(np.nanmedian(arr["sharp"])),
                        median_comb_ratio=float(np.nanmedian(arr["comb"])),
                        n_duplicate_frames=int(sum(s["dup"] for s in S[nm])),
                        recommendations=rec)
    if vw:
        for nm in ("bottom", "side", "dual"):
            fn = P(f"{nm}_scoring.mp4")
            c = cv2.VideoCapture(fn)
            m = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            summ.setdefault("videos", {})[nm] = dict(file=fn, frames=m, equal_to_input=(m == n))
            if m != n:
                print(f"WARNING: {fn} has {m} frames, input {n}")
    summ.update(dict(tag=a.tag, n_frames=n, header_fps=hdr, true_fps=a.fps, frames_bottom=nb, frames_side=ns,
                     rearing=dict(box=a.box, dark_thr=a.dark_thr, median_top_y=float(base),
                                  threshold_y=float(thr), aspect_min=a.rear_aspect, n_bouts=int(len(on)),
                                  time_rearing_s=float(rear.sum() / a.fps))))
    with open(P("video_qc.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: summ[k] for k in ("bottom", "side", "rearing")}, indent=1))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--box", type=int, nargs=4, default=[235, 585, 90, 360], help="x0 x1 y0 floor_y (side; y0 below the dark shadow band at the top of the cylinder)")
    ap.add_argument("--dark-thr", type=int, default=110)
    ap.add_argument("--rear-px", type=int, default=45)
    ap.add_argument("--rear-aspect", type=float, default=1.3, help="blob height / width for upright")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    run(a)


if __name__ == "__main__":
    sys.exit(main())
