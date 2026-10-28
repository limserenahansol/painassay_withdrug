"""behavior_movement_audit.py - look at the automatic movement / grooming with your own eyes (contact sheets).

For one session (final scoring file), three kinds of samples, each one row of frames over the episode (bottom view
cropped around the mouse with its centroid path, plus the side view at the middle of the episode):
  MOVE      random automatic movement bouts                      -> is the mouse really turning / stepping?
  ESC-NO    your escape / rearing frames with NO automatic movement (longest runs) -> rearing in place, or missed?
  STILL-HI  no movement, none of your held behaviours, but the highest silhouette speed -> missed movement?
  GROOM     every automatic grooming bout (up to 8)              -> grooming?
Writes analysis\\behavior_quality\\audit_<rid>_<kind>.png

  python behavior_movement_audit.py --day 2 --mouse 5939 --phase post [--n 6]
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(BASE, "analysis", "behavior_quality")
FPS = 30.0
FINAL = {1: "20261019_final", 2: "20261021_final"}
AUTO = {1: "20261019", 2: "20261021"}


def runs(v, min_len):
    v = np.r_[0, np.asarray(v, int), 0]
    d = np.diff(v)
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = (off - on) >= min_len
    return on[k], off[k]


def find_session(day, mouse, phase):
    for p in sorted(glob.glob(os.path.join(BASE, "scoring_mini1p", FINAL[day], "ScoringAB_*.mat"))):
        S = sio.loadmat(p, squeeze_me=True)
        ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        if str(S["mouseID"]).strip() == mouse and ph == phase:
            return p, S
    raise SystemExit("session not found")


def grab(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, im = cap.read()
    return im if ok else None


def crop(im, cx, cy, h=150):
    H, W = im.shape[:2]
    cx, cy = int(np.clip(cx, h, W - h)), int(np.clip(cy, h, H - h))
    return im[cy - h:cy + h, cx - h:cx + h], cx - h, cy - h


def sheet(kind, items, cb, cs, F, rid, out, title):
    if not items:
        return None
    n = len(items)
    fig, axs = plt.subplots(n, 5, figsize=(15, 3.1 * n))
    axs = np.atleast_2d(axs)
    for r, (a, b, note) in enumerate(items):
        ts = np.linspace(a, b - 1, 4).astype(int)
        path = F.iloc[a:b][["b_cx", "b_cy"]].to_numpy(float)
        mid = int(np.nanmean(ts))
        c0 = np.nanmean(path, 0) if np.isfinite(path).any() else np.array([320, 240])
        for k, f in enumerate(ts):
            im = grab(cb, f)
            ax = axs[r, k]; ax.set_axis_off()
            if im is None:
                continue
            cr, x0, y0 = crop(im, c0[0], c0[1])
            ax.imshow(cv2.cvtColor(cr, cv2.COLOR_BGR2RGB))
            ax.plot(path[:, 0] - x0, path[:, 1] - y0, "-", color="#00e5ff", lw=1)
            if np.isfinite(path[min(f - a, len(path) - 1)]).all():
                ax.plot(*(path[min(f - a, len(path) - 1)] - [x0, y0]), "o", color="#ff3d00", ms=4)
            ax.set_title(f"{(f) / FPS:.1f} s", fontsize=9)
        ax = axs[r, 4]; ax.set_axis_off()
        if cs is not None:
            im = grab(cs, mid)
            if im is not None:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title("side, middle", fontsize=9)
        axs[r, 0].text(-0.05, 0.5, f"#{r + 1}\n{a / FPS:.1f}-{b / FPS:.1f} s\n{note}", transform=axs[r, 0].transAxes,
                       ha="right", va="center", fontsize=8)
    fig.suptitle(f"{rid}  {kind}: {title}", fontsize=12)
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    p = os.path.join(out, f"audit_{rid}_{kind}.png")
    fig.savefig(p, dpi=70); plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--mouse", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    rid = f"{a.mouse}_d{a.day}_{a.phase}"
    p, S = find_session(a.day, a.mouse, a.phase)
    n = int(S["nUsed"])
    mv = np.atleast_1d(S["moveVec"]).astype(bool)[:n]
    gr = np.atleast_1d(S["groomVec"]).astype(bool)[:n]
    sc = np.atleast_1d(S["score"]).astype(int)[:n]
    fe = glob.glob(os.path.join(BASE, "behavior_auto", AUTO[a.day], f"{rid}_bfeatures.csv"))
    if not fe:
        fe = glob.glob(os.path.join(BASE, "behavior_auto", AUTO[a.day], "**", f"*{a.mouse}*{a.phase}*bfeatures.csv"), recursive=True)
    F = pd.read_csv(fe[0]).iloc[:n].reset_index(drop=True)
    cb = cv2.VideoCapture(str(S["botFile"]))
    cs = cv2.VideoCapture(str(S["sidFile"])) if os.path.isfile(str(S["sidFile"])) else None
    rng = np.random.default_rng(0)
    files = []
    on, off = runs(mv, int(0.4 * FPS))
    pick = rng.choice(len(on), min(a.n, len(on)), replace=False) if len(on) else []
    files.append(sheet("MOVE", [(on[k], off[k], "auto movement") for k in sorted(pick)], cb, cs, F, rid, OUT,
                       "random automatic movement bouts - is the mouse turning / stepping?"))
    on, off = runs((sc == 4) & ~mv, int(0.5 * FPS))
    order = np.argsort(off - on)[::-1][:a.n]
    files.append(sheet("ESC-NO", [(on[k], off[k], "your escape,\nno auto movement") for k in sorted(order)], cb, cs, F, rid, OUT,
                       "you scored escape / rearing, automatic movement is off - rearing in place, or missed?"))
    spd = pd.Series(pd.to_numeric(F.b_speed, errors="coerce")).rolling(15, center=True, min_periods=1).mean().values
    quiet = ~mv & (sc == 0) & ~gr
    on, off = runs(quiet, int(0.7 * FPS))
    score = np.array([np.nanmax(spd[x:y]) if y > x else 0 for x, y in zip(on, off)])
    order = np.argsort(np.nan_to_num(score))[::-1][:a.n]
    items = []
    for k in sorted(order):
        pk = on[k] + int(np.nanargmax(spd[on[k]:off[k]]))
        items.append((max(on[k], pk - 15), min(off[k], pk + 15), f"no movement,\nspeed {score[k]:.2f}"))
    files.append(sheet("STILL-HI", items, cb, cs, F, rid, OUT,
                       "no automatic movement and none of your behaviours, highest silhouette speed - missed movement?"))
    on, off = runs(gr, int(0.5 * FPS))
    files.append(sheet("GROOM", [(on[k], off[k], "auto grooming") for k in range(min(8, len(on)))], cb, cs, F, rid, OUT,
                       "automatic grooming bouts - grooming?"))
    for f in files:
        if f:
            print(f)


if __name__ == "__main__":
    main()
