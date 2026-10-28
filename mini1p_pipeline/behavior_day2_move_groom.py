"""Day 2 (2026-10-21): automatic GENERAL MOVEMENT and GROOMING, exactly as on Day 1.

Same code, same parameters, same trained grooming model as Day 1 (behavior_final_move_groom.py):
  1. behavior_features.py   bottom silhouette + side features per frame  -> <tag>_bfeatures.csv
  2. behavior_side_grid.py  side-camera 3 x 3 grid motion                -> grooming\\<tag>_bsgrid.npz
  3. grooming probability   the Day-1 model (behavior_auto\\20261019\\grooming\\grooming_model.joblib,
                            trained on your 117 judged clips), features of behavior_grooming.features,
                            first 15 s and invalid frames 0, 1-s smoothing
  4. movement               behavior_auto_annotate.general_movement (silhouette artefacts bridged, first 15 s skipped)
  5. grooming bouts         behavior_final_move_groom.grooming_bouts (> 0.4 starts, > 0.2 continues, gaps < 2 s
                            closed, >= 1 s); no Day-2 judged clips yet
5941 post: the side camera lost 18 frames at ~170 s (camera_drop_locator.py), so the side features are shifted
back into bottom-camera frames (18 empty rows inserted there); the bottom camera = logger = imaging frames.

Writes behavior_auto\\20261021\\<tag>_move_groom.npz (move, groom, p, ps, valid), move_groom_summary_day2.csv
(% time and bouts per block, Day-1 format), move_groom_timeline_day2.png. Merging with your manual Day-2 scoring
is a separate step once you have scored the sessions (same as behavior_final_move_groom.py on Day 1).

  python behavior_day2_move_groom.py [--only 5941_d2_pre] [--parallel 2]
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
VID = BASE + r"\mini1p_SBI_day2"
OUT = BASE + r"\behavior_auto\20261021"
GD2 = OUT + r"\grooming"
MODEL = BASE + r"\behavior_auto\20261019\grooming\grooming_model.joblib"
TABLE = BASE + r"\scoring_mini1p\20261021_day2\day2_session_table.csv"
FPS = 30.0
SIDE_DROP = {"5941_d2_post": (5100, 18)}        # (bottom frame where the side camera lost frames, how many)


def video(folder, mouse, phase):
    for p in sorted(glob.glob(os.path.join(folder, "*.avi"))):
        nm = os.path.basename(p).lower()
        if re.match(r"\s*(\d+)", nm).group(1) == mouse and (("base" in nm) == (phase == "pre")):
            return p
    raise SystemExit(f"no video for {mouse} {phase} in {folder}")


def shift_side(arr, drop, n_bottom):
    """insert `k` empty rows at bottom frame `d` so side row i matches bottom frame i again"""
    d, k = drop
    pad = np.full((k,) + arr.shape[1:], np.nan, dtype=float)
    out = np.concatenate([arr[:d].astype(float), pad, arr[d:].astype(float)])
    return out[:n_bottom]


def extract(tag, bottom, side):
    if not os.path.isfile(os.path.join(OUT, f"{tag}_bfeatures.csv")):
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "behavior_features.py"), "--bottom", bottom, "--side", side,
                        "--out-dir", OUT, "--tag", tag], check=True)
    sg = os.path.join(GD2, f"{tag}_bsgrid.npz")
    if not os.path.isfile(sg):
        import cv2
        n = int(cv2.VideoCapture(side).get(7))
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "behavior_side_grid.py"), "--side", side, "--n", str(n),
                        "--out", sg, "--workers", "6"], check=True)
    return tag


def fix_drop(tag, n_bottom):
    """5941 post: realign the side-camera columns / grid to bottom frames (once)"""
    if tag not in SIDE_DROP:
        return
    flag = os.path.join(OUT, f"{tag}_side_shifted.txt")
    if os.path.isfile(flag):
        return
    F = pd.read_csv(os.path.join(OUT, f"{tag}_bfeatures.csv"))
    sc = [c for c in F.columns if c.startswith("s_")]
    n = len(F)
    F[sc] = shift_side(F[sc].values, SIDE_DROP[tag], n)
    F.to_csv(os.path.join(OUT, f"{tag}_bfeatures.csv"), index=False)
    sg = os.path.join(GD2, f"{tag}_bsgrid.npz")
    Z = dict(np.load(sg, allow_pickle=True))
    Z["X"] = shift_side(Z["X"], SIDE_DROP[tag], n_bottom).astype(np.float32)
    np.savez_compressed(sg, **Z)
    open(flag, "w").write(f"side columns and grid shifted by {SIDE_DROP[tag]} (frame, rows)\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args(argv)
    os.makedirs(GD2, exist_ok=True)
    T = pd.read_csv(TABLE)
    jobs = []
    for _, r in T.iterrows():
        tag = f"{r.mouse}_d2_{r.phase}"
        if a.only and a.only != tag:
            continue
        jobs.append((tag, video(VID, str(r.mouse), r.phase), video(os.path.join(VID, "side"), str(r.mouse), r.phase), r))
    with ThreadPoolExecutor(a.parallel) as ex:
        for t in ex.map(lambda j: extract(j[0], j[1], j[2]), jobs):
            print(f"{t}: features done", flush=True)
    import joblib
    import cv2
    import behavior_grooming as bg
    import behavior_auto_annotate as ba
    from behavior_final_move_groom import grooming_bouts
    from behavior_supervised import BLOCKS
    M = joblib.load(MODEL)
    bg.FD, bg.GD = OUT, GD2                     # Day-2 folders for behavior_grooming.features
    summ = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(jobs), 1, figsize=(16, 1.1 * len(jobs) + 1), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (tag, bottom, side, r) in zip(axs, jobs):
        n_bottom = int(cv2.VideoCapture(bottom).get(7))
        fix_drop(tag, n_bottom)
        X, valid = bg.features(tag)
        X = X[M["features"]]
        p = M["model"].predict_proba(X.values.astype(np.float32))[:, 1]
        p[~valid] = 0
        p[:450] = 0
        ps = bg.sm(p, 31)
        F = pd.read_csv(os.path.join(OUT, f"{tag}_bfeatures.csv"))
        n = min(len(F), len(ps))
        move, _, _, _, _ = ba.general_movement(F.iloc[:n].reset_index(drop=True))
        move[:int(15 * FPS)] = False
        groom = grooming_bouts(ps[:n], pd.DataFrame(columns=["t_s", "grooming"]), n)
        np.savez_compressed(os.path.join(OUT, f"{tag}_move_groom.npz"), move=move, groom=groom, p=p[:n].astype(np.float32),
                            ps=ps[:n].astype(np.float32), valid=valid[:n])
        stim = [r[f"block{k}"] for k in range(1, 5)]
        for b, s0, s1 in BLOCKS:
            sl = slice(int(s0 * FPS), min(n, int(s1 * FPS)))
            msp =np.diff(np.r_[0, move[sl].astype(int), 0]); gsp = np.diff(np.r_[0, groom[sl].astype(int), 0])
            summ.append(dict(session=tag, mouse=str(r.mouse), phase="Baseline" if r.phase == "pre" else "Post-treatment",
                             treatment=r.treatment, block=b, block_stim="baseline" if b == "baseline" else stim[int(b[-1]) - 1],
                             movement_pct=round(100 * move[sl].mean(), 2), movement_bouts=int((msp == 1).sum()),
                             grooming_pct=round(100 * groom[sl].mean(), 2), grooming_bouts=int((gsp == 1).sum()),
                             grooming_s=round(float(groom[sl].sum() / FPS), 1)))
        tt = np.arange(n) / FPS
        ax.fill_between(tt, 0, move[:n] * 0.45, step="mid", color="#17a2b8", lw=0)
        ax.fill_between(tt, 0.55, 0.55 + groom[:n] * 0.45, step="mid", color="#d6338a", lw=0)
        ax.plot(tt, 0.55 + ps[:n] * 0.45, color="k", lw=0.4)
        for _, s0, s1 in BLOCKS[1:]:
            ax.axvspan(s0, s1, color="#f5b041", alpha=0.12, lw=0)
        ax.set_yticks([0.22, 0.77], ["movement", "grooming"], fontsize=7)
        ax.set_ylabel(f"{r.mouse} {r.phase}", rotation=0, ha="right", va="center", fontsize=9)
        print(f"{tag}: movement {100 * move.mean():.1f}%, grooming {groom.sum() / FPS:.0f} s "
              f"({int((np.diff(np.r_[0, groom.astype(int)]) == 1).sum())} bouts), valid {100 * valid.mean():.0f}%", flush=True)
    axs[-1].set_xlabel("session time (s)")
    fig.suptitle("Day 2 automatic movement (cyan) and grooming (pink; black line = grooming probability) - same code and "
                 "model as Day 1; orange = stimulus blocks", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "move_groom_timeline_day2.png"), dpi=90); plt.close(fig)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "move_groom_summary_day2.csv"), index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
