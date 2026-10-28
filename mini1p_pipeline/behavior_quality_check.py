"""behavior_quality_check.py - how trustworthy is each session's behaviour annotation? (Day 1 and Day 2)

Checks that need no new labels, using what is already in the final scoring files (your manual scoring + the
logger deliveries (Day 2) + the automatic general movement / grooming):
  reflex timing      every withdrawal / flinch tap vs the last delivery before it: evoked (0-2 s after a delivery),
                     late (2-5 s), ORPHAN (> 5 s after the last delivery, inside the stimulus blocks) - orphans are
                     either spontaneous or a delivery the logger (Day 2) / your delivery taps (Day 1) missed
  movement vs escape the frames you scored escape / rearing should mostly be automatic movement (sensitivity proxy)
  grooming vs yours  automatic grooming frames that you scored as lick / bite, guarding or attending (possible
                     confusion), and grooming during escape (impossible together)
  movement amount    % of time per period (no stimulus, blocks)
  Day-1 review       5941 SBI Day 1: your mode-4 review of the automatic movement (scoring_mini1p\\20261019_reviewed)
                     vs the automatic movement before your review = frame-wise precision / recall
Writes analysis\\behavior_quality\\behavior_quality.csv, orphan_reflexes.csv, behavior_quality.png

  python behavior_quality_check.py
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(BASE, "analysis", "behavior_quality")
FPS = 30.0
BLOCKS = [(180, 360), (390, 570), (600, 780), (810, 990)]


def sessions():
    for day, d in ((1, "20261019_final"), (2, "20261021_final")):
        for p in sorted(glob.glob(os.path.join(BASE, "scoring_mini1p", d, "ScoringAB_*.mat"))):
            S = sio.loadmat(p, squeeze_me=True)
            m = str(S["mouseID"]).strip()
            ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
            yield day, m, ph, p, S


def bouts(v):
    v = np.r_[0, np.asarray(v, int), 0]
    d = np.diff(v)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def in_blocks(f):
    t = (np.asarray(f) - 1) / FPS
    return np.any([(t >= a) & (t < b) for a, b in BLOCKS], axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, orph = [], []
    for day, m, ph, p, S in sessions():
        n = int(S["nUsed"])
        df = np.sort(np.atleast_1d(S["dFrames"]).astype(float))
        re = np.atleast_2d(S["reflexEvents"]).reshape(-1, 2).astype(float)
        sc = np.atleast_1d(S["score"]).astype(int)[:n]
        mv = np.atleast_1d(S["moveVec"]).astype(bool)[:n]
        gr = np.atleast_1d(S["groomVec"]).astype(bool)[:n]
        # reflex timing
        lag = []
        for f, ty in re:
            prev = df[df <= f]
            lag.append((f - prev[-1]) / FPS if len(prev) else np.inf)
        lag = np.array(lag)
        blk = in_blocks(re[:, 0]) if len(re) else np.zeros(0, bool)
        evoked = int(((lag >= 0) & (lag <= 2)).sum()); late = int(((lag > 2) & (lag <= 5)).sum())
        orphan = (lag > 5) & blk
        for k in np.flatnonzero(orphan):
            orph.append(dict(day=day, mouse=m, phase=ph, frame=int(re[k, 0]), time_s=round((re[k, 0] - 1) / FPS, 2),
                             type="withdrawal" if re[k, 1] == 1 else "flinch", s_after_last_delivery=round(lag[k], 1)))
        esc = sc == 4
        r = dict(day=day, mouse=m, phase=ph, frames=n, deliveries=len(df), reflex_taps=len(re),
                 reflex_evoked_0_2s=evoked, reflex_late_2_5s=late, reflex_orphan_in_blocks=int(orphan.sum()),
                 orphan_pct=round(100 * orphan.sum() / max(len(re), 1), 1),
                 escape_s=round(esc.sum() / FPS, 1),
                 escape_with_movement_pct=round(100 * (mv & esc).sum() / max(esc.sum(), 1), 1) if esc.any() else np.nan,
                 movement_pct_nostim=round(100 * mv[:int(180 * FPS)].mean(), 1),
                 movement_pct_blocks=round(100 * mv[np.r_[[np.arange(int(a * FPS), min(int(b * FPS), n)) for a, b in BLOCKS]].ravel()].mean(), 1),
                 grooming_s=round(gr.sum() / FPS, 1), grooming_bouts=len(bouts(gr)[0]),
                 grooming_on_lickbite_s=round((gr & (sc == 2)).sum() / FPS, 1),
                 grooming_on_guarding_s=round((gr & (sc == 3)).sum() / FPS, 1),
                 grooming_on_attending_s=round((gr & (sc == 1)).sum() / FPS, 1),
                 grooming_on_escape_s=round((gr & esc).sum() / FPS, 1),
                 grooming_with_movement_pct=round(100 * (gr & mv).sum() / max(gr.sum(), 1), 1) if gr.any() else np.nan,
                 lickbite_s=round((sc == 2).sum() / FPS, 1), attending_s=round((sc == 1).sum() / FPS, 1))
        rows.append(r)
    Q = pd.DataFrame(rows).sort_values(["day", "mouse", "phase"], ascending=[True, True, False])
    # Day-1 movement review (5941 SBI): your mode-4 corrections vs the automatic movement
    rev = []
    for p in glob.glob(os.path.join(BASE, "scoring_mini1p", "20261019_reviewed", "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True)
        m = str(S["mouseID"]).strip(); ph = "pre" if str(S["phase"]).lower().startswith("base") else "post"
        a = glob.glob(os.path.join(BASE, "scoring_mini1p", "20261019_auto", "auto_only", os.path.basename(p)))
        if not a:
            continue
        A = sio.loadmat(a[0], squeeze_me=True)
        u = np.atleast_1d(S["moveVec"]).astype(bool); x = np.atleast_1d(A["moveVec"]).astype(bool)
        k = min(len(u), len(x)); u, x = u[:k], x[:k]
        tp = (u & x).sum()
        rev.append(dict(mouse=m, phase=ph, user_movement_s=round(u.sum() / FPS, 1), auto_movement_s=round(x.sum() / FPS, 1),
                        precision=round(tp / max(x.sum(), 1), 3), recall=round(tp / max(u.sum(), 1), 3),
                        frame_agreement=round((u == x).mean(), 3)))
    Q.to_csv(os.path.join(OUT, "behavior_quality.csv"), index=False)
    pd.DataFrame(orph).to_csv(os.path.join(OUT, "orphan_reflexes.csv"), index=False)
    pd.DataFrame(rev).to_csv(os.path.join(OUT, "movement_review_day1.csv"), index=False)
    # figure
    lab = [f"D{r.day} {r.mouse} {r.phase}" for r in Q.itertuples()]
    fig, ax = plt.subplots(1, 4, figsize=(20, 6.5))
    y = np.arange(len(Q))
    ax[0].barh(y, Q.reflex_evoked_0_2s, color="#2e86c1", label="0-2 s after a delivery")
    ax[0].barh(y, Q.reflex_late_2_5s, left=Q.reflex_evoked_0_2s, color="#aed6f1", label="2-5 s")
    ax[0].barh(y, Q.reflex_orphan_in_blocks, left=Q.reflex_evoked_0_2s + Q.reflex_late_2_5s, color="#e74c3c",
               label="> 5 s (orphan, in blocks)")
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab, fontsize=8); ax[0].invert_yaxis()
    ax[0].set_xlabel("reflex taps (withdrawal + flinch)"); ax[0].legend(fontsize=8); ax[0].set_title("reflex taps vs the last delivery")
    ax[1].barh(y, Q.escape_with_movement_pct, color="#17a589")
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("% of your escape / rearing frames with automatic movement"); ax[1].set_title("movement sensitivity (proxy)")
    ax[2].barh(y - 0.2, Q.movement_pct_nostim, 0.4, color="#7f8c8d", label="no stimulus (0-180 s)")
    ax[2].barh(y + 0.2, Q.movement_pct_blocks, 0.4, color="#f39c12", label="stimulus blocks")
    ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].invert_yaxis(); ax[2].legend(fontsize=8)
    ax[2].set_xlabel("% time with automatic movement"); ax[2].set_title("movement amount")
    ax[3].barh(y, Q.grooming_s, color="#8e44ad", label="automatic grooming")
    ax[3].barh(y, Q.grooming_on_lickbite_s + Q.grooming_on_guarding_s + Q.grooming_on_attending_s, color="#e74c3c",
               label="... of it scored lick/bite, guarding or attending by you")
    ax[3].set_yticks(y); ax[3].set_yticklabels([]); ax[3].invert_yaxis(); ax[3].legend(fontsize=8)
    ax[3].set_xlabel("seconds"); ax[3].set_title("grooming vs your held behaviours")
    fig.suptitle("Behaviour annotation quality, all 16 sessions (Day 1 + Day 2): reflex timing, automatic movement vs your "
                 "escape scoring, grooming overlap with your scoring", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, "behavior_quality.png"), dpi=110); plt.close(fig)
    pd.set_option("display.width", 250)
    print(Q.to_string(index=False))
    print(pd.DataFrame(rev).to_string(index=False))
    print(f"orphan reflex taps: {len(orph)} -> {OUT}")


if __name__ == "__main__":
    main()
