"""make_sedation_figure.py  -  is the Day-2 reduction analgesia or sedation?

THE PROBLEM
    Every behaviour count fell on the drug day. Two explanations produce that
    same result and they are not the same finding:

        analgesia   the animal feels less, so it responds less to the stimulus
        sedation    the animal does everything less, responding included

    Neurotensin receptor agonists cause hypolocomotion and hypothermia, so for
    SBI-553 sedation is a live hypothesis rather than a technicality.

THE DISCRIMINATOR IS ALREADY IN THE SCORING
    Escape / rearing is exploration, not pain. Day 1 established that
    independently: its baseline rate is high and it DECREASES under
    stimulation, the opposite of a pain measure.

        a selective analgesic  -> affective down, exploration SPARED
        sedation               -> everything down, exploration included

    So the test is simply: how does escape/rearing compare with the rest?

WHAT THIS SCRIPT MAKES
    One figure. Left: escape/rearing against every other behaviour, as a
    Day2/Day1 ratio, per mouse. Right: matched video frames from the same
    timestamps on both days, so the postures can be compared without anyone
    choosing which moment to show.

    Frames are taken at IDENTICAL times on both days. Picking the most
    dramatic frame from each day would manufacture the conclusion.

USAGE
    python make_sedation_figure.py --day1 <corrected> --day2 <corrected> \\
                                   --mouse male1 --out <folder>

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.io import loadmat

VID = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\videos"
SIDE = {"Day1": os.path.join(VID, "cameraB"),
        "Day2": os.path.join(VID, "day2", "side")}
AFF = {1: "attending", 2: "lickbite", 3: "guarding", 4: "escape"}
REF = {1: "withdrawal", 2: "flinch"}
NICE = {"withdrawal": "Paw\nwithdrawal", "flinch": "Flinch",
        "attending": "Paw\nattending", "lickbite": "Licking /\nbiting",
        "guarding": "Guarding", "escape": "Escape /\nrearing"}
ORDER = ["withdrawal", "flinch", "attending", "lickbite", "guarding",
         "escape"]
WIN = {"withdrawal": 3.0, "flinch": 3.0, "attending": 10.0,
       "lickbite": 10.0, "guarding": 10.0, "escape": 10.0}
X0, X1, Y_TOP = 60, 680, 150
C1, C2 = "#444444", "#C0483B"
plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def s_(v, d=""):
    a = np.asarray(v).ravel()
    if a.size == 0:
        return d
    x = a[0]
    if isinstance(x, np.ndarray):
        x = x.ravel()[0] if x.size else d
    return str(x).strip()


def f_(v, d=np.nan):
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0]) if a.size else d


def rates(folder, use_window=False):
    """Events per delivery, per mouse per behaviour.

    use_window=False (the default) counts EVERY event in the session and
    divides by the total deliveries. That is the right numerator for "how much
    did the animal do", which is the sedation question.

    use_window=True counts only events starting within the response window of
    a delivery. An earlier version did that unconditionally and it distorted
    the headline: the window keeps only 29 % of escape/rearing events, because
    escape/rearing is spontaneous exploration that mostly happens away from
    the stimulus. Escape came out at x0.08 windowed against x0.22 counting
    everything - still the joint-largest fall, but not a tenfold one.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "ScoringAB_*.mat"))):
        M = loadmat(p)
        fps = f_(M["frameRate"], 30.0)
        n = int(f_(M.get("nUsed", 0), 0))
        sc = np.asarray(M["score"]).ravel().astype(int)
        sc = sc[:n] if n else sc
        n = len(sc)
        rx = np.asarray(M.get("reflexEvents", np.empty((0, 2)))).reshape(-1, 2)
        dF = np.asarray(M["dFrames"]).ravel().astype(int)
        mid = s_(M.get("mouseID", "")).upper()
        d = {}
        for b in ORDER:
            w = WIN[b]
            spans = ([(int(f), min(int(f + w * fps), n)) for f in dF]
                     if use_window else [(1, n)])
            tot = 0
            for f, hi in spans:
                if hi <= f:
                    continue
                if b in REF.values():
                    code = [k for k, v in REF.items() if v == b][0]
                    ff = rx[rx[:, 1] == code, 0] if rx.size else np.array([])
                    tot += int(np.sum((ff >= f) & (ff < hi)))
                else:
                    code = [k for k, v in AFF.items() if v == b][0]
                    seg = (sc[f - 1:hi - 1] == code).astype(int)
                    tot += int((np.diff(np.r_[0, seg]) == 1).sum())
            d[b] = tot / len(dF) if len(dF) else np.nan
        out[mid] = d
    return out


def floor_row(g):
    rm = g.mean(axis=1).astype(np.float32)
    return Y_TOP + int(np.argmin(np.diff(rm[Y_TOP:])))


def frame_at(folder, mouse, t):
    vids = [v for v in sorted(glob.glob(os.path.join(folder, "*.avi")))
            if os.path.basename(v).lower().startswith(mouse.lower())]
    if not vids:
        return None
    cap = cv2.VideoCapture(vids[0])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    yf = floor_row(g)
    crop = g[Y_TOP:max(yf + 8, Y_TOP + 40), X0:X1]
    lo, hi = np.percentile(crop, 1), np.percentile(crop, 99)
    s = np.clip((crop.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1),
                0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day1", required=True)
    ap.add_argument("--day2", required=True)
    ap.add_argument("--mouse", default="male1")
    ap.add_argument("--times", type=float, nargs="*",
                    default=[180, 480, 1140, 1500])
    ap.add_argument("--out", required=True)
    ap.add_argument("--response-window", action="store_true",
                    help="count only events inside the response window. "
                         "Default counts every event, which is the right "
                         "numerator for 'how much did the animal do'.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R1 = rates(a.day1, a.response_window)
    R2 = rates(a.day2, a.response_window)
    mice = sorted(set(R1) & set(R2))
    print(f"{len(mice)} mouse/mice in both days: {mice}")

    fig = plt.figure(figsize=(15.4, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35],
                          height_ratios=[1, 1], hspace=.30, wspace=.16)

    # ---- left top: ratio per behaviour, per mouse -----------------------
    ax = fig.add_subplot(gs[0, 0])
    for i, b in enumerate(ORDER):
        rr = [R2[m][b] / R1[m][b] for m in mice
              if R1[m][b] and np.isfinite(R1[m][b])]
        col = C2 if b == "escape" else "#8899A6"
        ax.plot(np.full(len(rr), i) + np.linspace(-.13, .13, len(rr)), rr,
                "o", ms=6, color=col, alpha=.85, zorder=4)
        if rr:
            ax.plot([i - .26, i + .26], [np.median(rr)] * 2, "-",
                    color=col, lw=3.2, zorder=5)
    ax.axhline(1, color="k", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_yticks([.02, .05, .1, .25, .5, 1, 2])
    ax.set_yticklabels(["0.02", "0.05", "0.1", "0.25", "0.5", "1", "2"],
                       fontsize=9)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([NICE[b] for b in ORDER], fontsize=8.5)
    ax.set_ylabel("Day 2 / Day 1\n(events per delivery)", fontsize=10.5)
    ax.set_title("Escape / rearing fell as hard as any pain behaviour",
                 fontsize=11.5, fontweight="bold")
    ax.grid(alpha=.2, axis="y")

    # ---- left bottom: the logic, in words -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    esc = [R2[m]["escape"] / R1[m]["escape"] for m in mice
           if R1[m]["escape"]]
    others = [R2[m][b] / R1[m][b] for m in mice for b in ORDER
              if b != "escape" and R1[m][b]]
    try:
        pp = stats.wilcoxon([R1[m]["escape"] for m in mice],
                            [R2[m]["escape"] for m in mice],
                            zero_method="wilcox", method="exact").pvalue
    except ValueError:
        pp = np.nan
    txt = (
        "Escape / rearing is EXPLORATION, not pain.\n"
        "Day 1 showed that on its own: high at baseline,\n"
        "and it goes DOWN under stimulation.\n\n"
        "  a selective analgesic   affective down,\n"
        "                          exploration SPARED\n"
        "  sedation                everything down,\n"
        "                          exploration included\n\n"
        f"Observed: escape/rearing x{np.median(esc):.2f} "
        f"(median over {len(mice)} mice),\n"
        f"paired p = {pp:.3f} - as large as any pain behaviour.\n"
        f"Reflexes fell least (withdrawal "
        f"x{np.median([R2[m]['withdrawal'] / R1[m]['withdrawal'] for m in mice if R1[m]['withdrawal']]):.2f}, "
        f"flinch "
        f"x{np.median([R2[m]['flinch'] / R1[m]['flinch'] for m in mice if R1[m]['flinch']]):.2f})\n"
        "- consistent with spinal reflexes surviving sedation.\n\n"
        "Every event is counted, not only those inside a\n"
        "response window: the window keeps just 29 % of\n"
        "escape/rearing, which is spontaneous.\n\n"
        "This does NOT look like selective analgesia."
    )
    ax.text(0, 1, txt, va="top", ha="left", fontsize=10.2, family="monospace",
            color="#1F2937")

    # ---- right: matched frames, same timestamps both days ---------------
    for r, t in enumerate(a.times[:4]):
        for c, (tag, folder, col) in enumerate(
                (("Day 1  no drug", SIDE["Day1"], C1),
                 ("Day 2  SBI-553", SIDE["Day2"], C2))):
            sub = gs[r // 2, 1].subgridspec(2, 2, hspace=.06, wspace=.04)
            axx = fig.add_subplot(sub[r % 2, c])
            im = frame_at(folder, a.mouse, t)
            if im is not None:
                axx.imshow(im, cmap="gray", vmin=0, vmax=255,
                           aspect="auto")
            axx.set_xticks([])
            axx.set_yticks([])
            for sp in axx.spines.values():
                sp.set_color(col)
                sp.set_linewidth(2.0)
            if r % 2 == 0 and r // 2 == 0:
                axx.set_title(tag, fontsize=10.5, color=col,
                              fontweight="bold")
            axx.text(.02, .96, f"{int(t)}s", transform=axx.transAxes,
                     fontsize=8.5, color="#FFD400", va="top")

    fig.suptitle(
        f"Analgesia or sedation?   {a.mouse}, side view, matched timestamps "
        f"on both days (not chosen for effect)\n"
        "All six mice received SBI-553. There was no vehicle group, so drug, "
        "day, order and habituation cannot be separated.",
        fontsize=12)
    p = os.path.join(a.out, "S1_sedation_evidence.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")

    print("\n  Day2/Day1 ratio, median over mice:")
    for b in ORDER:
        rr = [R2[m][b] / R1[m][b] for m in mice if R1[m][b]]
        print(f"    {b:11s} x{np.median(rr):.2f}"
              f"{'   <- exploration, not pain' if b == 'escape' else ''}")


if __name__ == "__main__":
    main()
