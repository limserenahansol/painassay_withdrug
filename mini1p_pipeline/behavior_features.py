"""Per-frame behaviour features of one mini1p session from both cameras (silhouette "pose-lite"
features: the same idea as keypoint pipelines - pose first, behaviour second - without a
trained pose model).

BOTTOM camera (original AVI; the mouse is a dark silhouette on the bright mesh):
  b_area, b_cx, b_cy, b_theta (body axis, deg), b_major, b_minor   silhouette shape
  b_speed       centroid speed (px/frame, 3-frame smoothed)
  b_turn        |change of body axis| (deg/frame, axis is 180-periodic)
  b_body_me     mean |frame difference| inside the dilated silhouette (whole-body motion)
  b_edge_me     mean |frame difference| in a ring around the silhouette edge (paws / limbs)
  b_shape_chg   fraction of the silhouette that changed since the last frame (paw lift etc.)
  b_out_me      mean |frame difference| in the arena OUTSIDE the mouse (tool / hand)
  b_tool_px     pixels 15-80 px from the mouse that differ from the MOUSE-FREE background
                (per-pixel median of 200 frames with the mouse and its shadow masked) by > 25
                grey, in blobs >= 40 px: the stimulus tool (pin, filament holder, heat probe)
                or the hand. 15 px are skipped because the mouse's own shadow is there.
  b_tool_dist   distance (px) from those blobs to the mouse outline (NaN = none)
  b_near_me     mean |frame difference| 15-80 px from the mouse (tool tip, kicked-out paw)
GRID (<tag>_bgrid.npz): the frame cut into 8 x 6 cells of 90 x 80 px; per cell, with the mouse
  and its shadow masked: fg = mean |frame - mouse-free background|, me = mean |frame difference|.
  Where the hand and the tool are (they enter from the top and the right).
SIDE camera (original AVI; dark mouse on the intended white background):
  s_top, s_aspect, s_area, s_cx        top of the mouse, height/width, size, position
  s_me          mean |frame difference| inside the mouse bounding box (+ margin)

Camera frame n (1-based) = isx_frame n. Output: <out_dir>/<tag>_bfeatures.csv (+ a QC png of
the silhouette masks at 6 time points).

  python behavior_features.py --bottom B.avi --side S.avi --out-dir D --tag 5941_d1_post
"""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np

# rig geometry (5941, 2026-10-19; check the QC png for other days)
B_ARENA = (260, 60, 720, 420)        # x0 y0 x1 y1 of the bottom-view arena
B_DARK = 40                          # grey: mouse silhouette (after a 3-px blur)
B_BG_DROP = 30                       # ... AND this much darker than the session background
S_BOX = (235, 90, 585, 360)          # side view: below the shadow band, above the floor
S_DARK = 110


def background(path, n, k=200):
    # per-pixel 90th percentile of k frames spread over the session (the mouse moves, the
    # arena does not); frames before the room-light change are mostly outvoted
    c = cv2.VideoCapture(path)
    fr = []
    for j in np.linspace(0, n - 1, k).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = c.read()
        if ok:
            fr.append(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (0, 0), 3))
    st = np.stack(fr).astype(np.float32)
    bg = np.percentile(st, 90, axis=0).astype(np.float32)
    # mouse-free background: per-pixel median of the same frames with the mouse (+ 20 px for
    # its shadow) masked out; where the mouse covered a pixel in every frame, the 90th pct
    x0, y0, x1, y1 = B_ARENA
    arena = np.zeros(st.shape[1:], bool)
    arena[y0:y1, x0:x1] = True
    for g in st:
        m = ((g < B_DARK) & (bg - g > B_BG_DROP) & arena).astype(np.uint8)
        g[cv2.dilate(m, np.ones((41, 41), np.uint8)) > 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        free = np.nanmedian(st, axis=0)
    return bg, np.where(np.isfinite(free), free, bg).astype(np.float32)


def pick(mask, prev_c, min_area=1500):
    # the component nearest the previous mouse position (else the largest), >= min_area
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    ok = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area]
    if not ok:
        return None, None, None
    if prev_c is None:
        k = max(ok, key=lambda q: st[q, cv2.CC_STAT_AREA])
    else:
        k = min(ok, key=lambda q: np.hypot(*(cen[q] - prev_c)))
    return (lab == k), st[k], cen[k]


def largest(mask):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, None, None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == k), st[k], cen[k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-frames", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    cb, cs = cv2.VideoCapture(a.bottom), cv2.VideoCapture(a.side)
    n = min(int(cb.get(7)), int(cs.get(7)))
    if a.n_frames:
        n = min(n, a.n_frames)
    x0, y0, x1, y1 = B_ARENA
    arena = np.zeros((480, 720), bool)
    arena[y0:y1, x0:x1] = True
    k_dil = np.ones((31, 31), np.uint8)
    k_ring = np.ones((15, 15), np.uint8)
    cols = ["cam_frame", "isx_frame", "t_s", "b_area", "b_cx", "b_cy", "b_theta", "b_major", "b_minor",
            "b_speed", "b_turn", "b_body_me", "b_edge_me", "b_shape_chg", "b_out_me",
            "s_top", "s_aspect", "s_area", "s_cx", "s_me", "b_tool_px", "b_tool_dist", "b_near_me"]
    F = np.full((n, len(cols)), np.nan, np.float32)
    GW, GH = 8, 6                                    # grid cells (90 x 80 px)
    G_fg = np.full((n, GW * GH), np.nan, np.float16)
    G_me = np.full((n, GW * GH), np.nan, np.float16)
    k_excl = np.ones((31, 31), np.uint8)             # mouse + 15 px (its shadow)
    k_near = np.ones((161, 161), np.uint8)           # mouse + 80 px
    prev_b = prev_s = prev_m = None
    prev_theta = np.nan
    prev_c = None
    bg, bgfree = background(a.bottom, n)
    qc_idx = set(np.linspace(0, n - 1, 6).astype(int))
    qc = []
    for i in range(n):
        okb, fb = cb.read()
        oks, fs = cs.read()
        if not (okb and oks):
            n = i
            break
        gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
        gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
        F[i, 0] = i + 1
        F[i, 1] = i + 1
        F[i, 2] = (i + 1) / 30.0
        # ---- bottom ----
        sm = cv2.GaussianBlur(gb, (0, 0), 3)
        m = ((sm < B_DARK) & (bg - sm > B_BG_DROP) & arena).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mm, st, cen = pick(m, prev_c)
        if cen is not None:
            prev_c = cen
        if mm is not None:
            mu = cv2.moments(mm.astype(np.uint8), True)
            if mu["m00"] > 0:
                cx, cy = mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]
                a20, a02, a11 = mu["mu20"] / mu["m00"], mu["mu02"] / mu["m00"], mu["mu11"] / mu["m00"]
                th = 0.5 * np.degrees(np.arctan2(2 * a11, a20 - a02))
                lam = np.sqrt(((a20 - a02) / 2) ** 2 + a11 ** 2)
                F[i, 3:9] = [mu["m00"], cx, cy, th, 4 * np.sqrt((a20 + a02) / 2 + lam),
                             4 * np.sqrt(max((a20 + a02) / 2 - lam, 0))]
                if np.isfinite(prev_theta):
                    d = abs(th - prev_theta) % 180
                    F[i, 10] = min(d, 180 - d)
                prev_theta = th
            if prev_b is not None:
                diff = cv2.absdiff(gb, prev_b)
                body = cv2.dilate(mm.astype(np.uint8), k_dil) > 0
                ring = (cv2.dilate(mm.astype(np.uint8), k_ring) > 0) & ~(cv2.erode(mm.astype(np.uint8), k_ring) > 0)
                F[i, 11] = diff[body].mean()
                F[i, 12] = diff[ring].mean() if ring.any() else np.nan
                out = arena & ~(cv2.dilate(mm.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0)
                F[i, 14] = diff[out].mean() if out.any() else np.nan
                if prev_m is not None:
                    F[i, 13] = np.logical_xor(mm, prev_m).sum() / max(mm.sum(), 1)
            # foreign object (tool / hand): differs from the mouse-free background, outside
            # the mouse and its shadow
            mm8 = mm.astype(np.uint8)
            excl = cv2.dilate(mm8, k_excl) > 0
            ann = (cv2.dilate(mm8, k_near) > 0) & ~excl
            fgd = np.abs(sm.astype(np.float32) - bgfree)
            fg = cv2.morphologyEx(((fgd > 25) & ann).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            nc, lab, cst, _ = cv2.connectedComponentsWithStats(fg, 8)
            big = np.zeros(nc, bool)
            big[1:] = cst[1:, cv2.CC_STAT_AREA] >= 40
            fg = big[lab]
            F[i, 20] = fg.sum()
            if fg.any():
                dist = cv2.distanceTransform((~mm).astype(np.uint8), cv2.DIST_L2, 3)
                F[i, 21] = dist[fg].min()
            valid = (~excl).astype(np.float32)
            vcell = np.maximum(cv2.resize(valid, (GW, GH), interpolation=cv2.INTER_AREA), 1e-3)
            G_fg[i] = (cv2.resize(fgd * valid, (GW, GH), interpolation=cv2.INTER_AREA) / vcell).ravel()
            if prev_b is not None:
                F[i, 22] = diff[ann].mean() if ann.any() else np.nan
                G_me[i] = (cv2.resize(diff.astype(np.float32) * valid, (GW, GH), interpolation=cv2.INTER_AREA) / vcell).ravel()
            prev_m = mm
        prev_b = gb
        # ---- side ----
        sx0, sy0, sx1, sy1 = S_BOX
        roi = gs[sy0:sy1, sx0:sx1]
        ms = cv2.morphologyEx((roi < S_DARK).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        msk, sst, scen = largest(ms)
        if msk is not None:
            F[i, 15] = sst[cv2.CC_STAT_TOP] + sy0
            F[i, 16] = sst[cv2.CC_STAT_HEIGHT] / max(sst[cv2.CC_STAT_WIDTH], 1)
            F[i, 17] = sst[cv2.CC_STAT_AREA]
            F[i, 18] = scen[0] + sx0
            if prev_s is not None:
                bx, by, bw, bh = sst[0] + sx0, sst[1] + sy0, sst[2], sst[3]
                ya, yb = max(by - 15, 0), min(by + bh + 15, 480)
                xa, xb = max(bx - 15, 0), min(bx + bw + 15, 720)
                F[i, 19] = cv2.absdiff(gs[ya:yb, xa:xb], prev_s[ya:yb, xa:xb]).mean()
        prev_s = gs
        if i in qc_idx:
            vis = fb.copy()
            if mm is not None:
                vis[mm] = (0.5 * vis[mm] + np.array([0, 0, 120])).astype(np.uint8)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 1)
            vs = fs.copy()
            cv2.rectangle(vs, (sx0, sy0), (sx1, sy1), (0, 255, 0), 1)
            if msk is not None:
                cv2.line(vs, (sx0, int(F[i, 15])), (sx1, int(F[i, 15])), (255, 0, 0), 2)
            t = np.hstack([vis, vs])
            cv2.putText(t, f"frame {i + 1} ({(i + 1) / 30:.0f} s)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            qc.append(cv2.resize(t, (720, 240)))
        if (i + 1) % 5000 == 0:
            print(f"  {a.tag}: {i + 1} / {n}", flush=True)
    F = F[:n]
    # centroid speed, 3-frame smoothed
    sp = np.r_[np.nan, np.hypot(np.diff(F[:, 4]), np.diff(F[:, 5]))]
    F[:, 9] = np.convolve(np.nan_to_num(sp), np.ones(3) / 3, "same")
    out = os.path.join(a.out_dir, f"{a.tag}_bfeatures.csv")
    hdr = ",".join(cols)
    np.savetxt(out, F, delimiter=",", header=hdr, comments="", fmt="%.4f")
    np.savez_compressed(os.path.join(a.out_dir, f"{a.tag}_bgrid.npz"), fg=G_fg[:n], me=G_me[:n],
                        cells=np.array([GW, GH]))
    cv2.imwrite(os.path.join(a.out_dir, f"{a.tag}_bg_mousefree.png"), np.clip(bgfree, 0, 255).astype(np.uint8))
    if qc:
        cv2.imwrite(os.path.join(a.out_dir, f"{a.tag}_bfeatures_qc.png"), np.vstack(qc))
    print(f"behavior_features {a.tag}: {n} frames -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
