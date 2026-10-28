"""behavior_scoring_review.py - places in YOUR manual scoring worth a second look (suggestions only; nothing is changed).

Your hand scoring is the reference (it beat every automatic detector on Day 1). This only looks for patterns that are
more often a key slip than behaviour, so you can jump to them in the scorer (times in s and frames):
  ORPHAN     withdrawal / flinch tap > 5 s after the last delivery, inside a stimulus block (spontaneous, or a missed
             delivery)
  DOUBLE     two taps of the same reflex < 0.1 s apart (one press counted twice? taps 0.1-0.3 s apart are mostly
             separate flicks of one shake - 133 flinch, 30 withdrawal pairs over the 16 sessions - not listed)
  LATE       withdrawal tapped 1.5-2 s after its delivery (counted for that delivery; the median latency is ~0.3 s)
  NOSTIM     reflex tap in the first 3 min (no stimulus yet)
  SHORT      held behaviour (attending, lick/bite, guarding, escape) shorter than 0.2 s (a brushed key?)
  CLOSE      two deliveries < 0.5 s apart (one contact counted twice? the median spacing is ~2.7 s; Day 1 = your
             delivery taps, Day 2 = the pressure logger)
Writes analysis\\behavior_quality\\scoring_review_suggestions.csv (one row per place) + _summary.csv (counts per session)

  python behavior_scoring_review.py
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(BASE, "analysis", "behavior_quality")
FPS = 30.0
BLOCKS = [(180, 360), (390, 570), (600, 780), (810, 990)]
HELD = {1: "attending", 2: "lick/bite", 3: "guarding", 4: "escape/rearing"}
SRC = {1: "20261019_final", 2: "20261021_final"}


def runs(v):
    v = np.r_[0, np.asarray(v, int), 0]
    d = np.diff(v)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for day, fold in SRC.items():
        for p in sorted(glob.glob(os.path.join(BASE, "scoring_mini1p", fold, "ScoringAB_*.mat"))):
            S = sio.loadmat(p, squeeze_me=True)
            m = str(S["mouseID"]).strip(); ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
            man = str(S.get("manualSource", p))
            n = int(S["nUsed"])
            df = np.sort(np.atleast_1d(S["dFrames"]).astype(float))
            re = np.atleast_2d(S["reflexEvents"]).reshape(-1, 2).astype(float)
            sc = np.atleast_1d(S["score"]).astype(int)[:n]
            add = lambda kind, f, what: rows.append(dict(day=day, mouse=m, phase=ph, kind=kind, frame=int(f),
                                                         time_s=round((f - 1) / FPS, 2), what=what,
                                                         your_file=os.path.basename(man)))
            for ty, nm in ((1, "withdrawal"), (2, "flinch")):
                f = np.sort(re[re[:, 1] == ty, 0])
                for a, b in zip(f[:-1], f[1:]):
                    if b - a < 0.1 * FPS:
                        add("DOUBLE", b, f"{nm} tapped twice within {(b - a) / FPS:.2f} s")
                for x in f:
                    prev = df[df <= x]
                    lag = (x - prev[-1]) / FPS if len(prev) else np.inf
                    t = (x - 1) / FPS
                    if t < 180:
                        add("NOSTIM", x, f"{nm} before any stimulus ({t:.1f} s)")
                    elif lag > 5 and any(a0 <= t < b0 for a0, b0 in BLOCKS):
                        add("ORPHAN", x, f"{nm} {lag:.1f} s after the last delivery")
                    elif ty == 1 and 1.5 <= lag <= 2.0:
                        add("LATE", x, f"withdrawal {lag:.2f} s after its delivery")
            for code, nm in HELD.items():
                on, off = runs(sc == code)
                for a, b in zip(on, off):
                    if b - a < 0.2 * FPS:
                        add("SHORT", a + 1, f"{nm} held {1000 * (b - a) / FPS:.0f} ms")
            for a, b in zip(df[:-1], df[1:]):
                if b - a < 0.5 * FPS:
                    add("CLOSE", b, f"delivery {(b - a) / FPS:.2f} s after the previous one")
    R = pd.DataFrame(rows).sort_values(["day", "mouse", "phase", "frame"])
    R.to_csv(os.path.join(OUT, "scoring_review_suggestions.csv"), index=False)
    Sm = R.groupby(["day", "mouse", "phase", "kind"]).size().unstack(fill_value=0)
    Sm.to_csv(os.path.join(OUT, "scoring_review_summary.csv"))
    print(Sm.to_string())
    print(f"{len(R)} places -> {OUT}\\scoring_review_suggestions.csv")


if __name__ == "__main__":
    main()
