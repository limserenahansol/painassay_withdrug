"""timing_examples.py - a few examples for the user (2026-10-23) of four timing questions:

  A  Day 1: the paw already moves before the stimulus (key) mark          ex_A_day1_stim_mark_<k>.png
  B  Day 2: the clicker press vs the paw movement and the probe contact   ex_B_day2_click_<k>.png
  C  Day 2: the clicker held > 1 s (a press inside gets no delivery)      ex_C_held_click_<k>.png
  D  Day 2: double presses < 0.5 s apart (the second one dropped)         ex_D_double_click_<k>.png

A / B: every frame (1/30 s) from 0.40 s before the movement start to 0.13 s after the withdrawal mark; each tile = side camera
(feet) over the bottom camera; green = movement start (two independent visual judges, analysis\\withdrawal_timing\\
visual_vs_auto.csv), blue = the stimulus mark (Day 1 key / Day 2 click), red = the withdrawal / flinch mark.
C / D: the clicker signal (corrected pressure, the indicator shaded) and the bottom camera at each press.

  python timing_examples.py
"""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "withdrawal_timing", "examples")
FPS = 30.0
SC = {1: os.path.join(B, "scoring_mini1p", "20261019_final"), 2: os.path.join(B, "scoring_mini1p", "20261021_final")}
TAB = os.path.join(B, "scoring_mini1p", "20261021_day2", "day2_session_table.csv")
LOGD = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_deliveries")


def scoring(d, mouse, ph):
    for p in glob.glob(os.path.join(SC[d], "ScoringAB_*.mat")):
        S = sio.loadmat(p, squeeze_me=True, variable_names=["mouseID", "phase", "sidFile", "botFile", "syncOff", "dFrames"])
        if str(S["mouseID"]).strip() == mouse and (("pre" if str(S["phase"]).lower().startswith("base") else "post") == ph):
            return S
    raise FileNotFoundError((d, mouse, ph))


def read(path, f0, f1):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0)); out = []
    for _ in range(f0, f1 + 1):
        ok, im = c.read(); out.append(im if ok else None)
    c.release()
    return out


def tile(side, bot, label, col, crop=None):
    if side is not None and crop is not None:
        x0, y0 = crop
        side = side[y0:y0 + 240, x0:x0 + 360]
    s = np.zeros((200, 300, 3), np.uint8) if side is None else cv2.resize(side, (300, 200))
    b = np.zeros((200, 300, 3), np.uint8) if bot is None else cv2.resize(bot, (300, 200))
    t = np.vstack([s, np.full((3, 300, 3), 255, np.uint8), b])
    if col is not None:
        cv2.rectangle(t, (0, 0), (299, t.shape[0] - 1), col, 6)
    cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(t, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return t


def sheet_AB(r, fn, title):
    d, mouse, ph = int(r.day), str(r.mouse), r.phase
    S = scoring(d, mouse, ph)
    off = int(round(float(S["syncOff"]) * FPS))
    w = int(r.mark_frame); dl = int(r.delivery_frame)
    mv = w - int(round(r.visual_delay * FPS))                  # movement start frame (judges)
    f0, f1 = min(mv, dl) - 12, w + 4
    sv = read(str(S["sidFile"]), f0 - 1 + off, f1 - 1 + off)
    bv = read(str(S["botFile"]), f0 - 1, f1 - 1)
    Fe = pd.read_csv(os.path.join(B, "behavior_auto", "20261019" if d == 1 else "20261021", f"{r.rid}_bfeatures.csv"))
    cx = np.nanmedian(Fe.s_cx.iloc[max(0, f0 - 1):f1]); top = np.nanmedian(Fe.s_top.iloc[max(0, f0 - 1):f1])
    crop = (int(np.clip((cx if np.isfinite(cx) else 400) - 180, 0, 720 - 360)),
            int(np.clip((top if np.isfinite(top) else 240) - 20, 0, 480 - 240)))
    tiles = []
    for i, f in enumerate(range(f0, f1 + 1)):
        col = (0, 0, 255) if f == w else (255, 140, 0) if f == dl else (0, 200, 0) if f == mv else None
        lab = f"{(f - dl) / FPS:+.2f} s" + (" STIM" if f == dl else " MOVE" if f == mv else " MARK" if f == w else "")
        tiles.append(tile(sv[i], bv[i], lab, col, crop))
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    head = np.full((70, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(head, f"{r.rid}  {r.kind}: times from YOUR STIMULUS MARK (blue). Green = the paw / leg starts to move "
                f"({(mv - dl) / FPS:+.2f} s), red = your {r.kind} mark ({(w - dl) / FPS:+.2f} s). Every frame (1/30 s); tile = side "
                f"over bottom camera.", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(fn, np.vstack([head, grid]))


def pressure(session):
    p = [q for q in glob.glob(rf"C:\Users\hsollim\Downloads\Rogger\{session}\**\*_pressure.csv", recursive=True) if "MACOSX" not in q][0]
    D = pd.read_csv(p)
    D = D[D.ttl_count >= 1].copy()
    # session time of each logger row: the camera frame count so far, interpolated between its changes
    ch = np.flatnonzero(np.diff(np.r_[-1, D.ttl_count.to_numpy()]) != 0)
    D["sess_s"] = np.interp(D.t_s.to_numpy(), D.t_s.to_numpy()[ch], D.ttl_count.to_numpy()[ch] / FPS)
    base = pd.Series(D.corrected.to_numpy()).rolling(799, center=True, min_periods=79).median().to_numpy()
    D["d"] = (D.corrected.to_numpy() - base) / 1e6
    return D


def sheet_CD(rid, session, t_a, t_b, used, dropped, fn, title):
    mouse, _, ph = rid.split("_")
    S = scoring(2, mouse, ph)
    D = pressure(session)
    Z = D[(D.sess_s >= t_a - 1.0) & (D.sess_s <= t_b + 1.0)]
    fig = plt.figure(figsize=(14, 7.6))
    a = fig.add_axes([0.06, 0.62, 0.9, 0.3])
    a.plot(Z.sess_s, Z.d, color="k", lw=1)
    on = Z.indicator.to_numpy() > 0
    a.fill_between(Z.sess_s, Z.d.min(), Z.d.max(), where=on, color="#f5b041", alpha=0.25, lw=0, label="clicker indicator ON")
    for t in used:
        a.axvline(t, color="#c0392b", lw=2, label="delivery used" if t == used[0] else None)
    for t in dropped:
        a.axvline(t, color="#2471a3", lw=2, ls="--", label="press dropped / not used" if t == dropped[0] else None)
    a.set_xlabel("session time (s)", fontsize=12); a.set_ylabel("clicker pressure\n(10^6 counts)", fontsize=12)
    a.legend(frameon=False, fontsize=10, loc="upper right"); a.set_title(title, loc="left", fontsize=13)
    # presses in the pressure trace (|d| > 0.3 x 10^6, the offline threshold): a press with no delivery within 0.5 s is marked
    dz = np.abs(Z.d.to_numpy()); up = np.flatnonzero((dz[1:] > 0.3) & (dz[:-1] <= 0.3)) + 1
    presses = [round(float(Z.sess_s.iloc[i]), 2) for i in up]
    lone = [t for t in presses if not any(abs(t - u) < 0.5 for u in list(used) + list(dropped))]
    for t in lone:
        a.axvline(t, color="#8e44ad", lw=2, ls=":", label="press with NO delivery" if t == lone[0] else None)
    a.legend(frameon=False, fontsize=10, loc="upper right")
    # bottom camera at each used / dropped time and at each press without a delivery
    times = sorted(set([round(t, 2) for t in list(used) + list(dropped) + lone]))[:6]
    for j, t in enumerate(times):
        f = int(round(t * FPS))
        im = read(str(S["botFile"]), f - 1, f - 1)[0]
        b = fig.add_axes([0.06 + j * 0.155, 0.05, 0.145, 0.48])
        if im is not None:
            b.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        b.set_axis_off()
        lab = ("used" if any(abs(t - u) < 0.02 for u in used) else "press, NO delivery" if t in lone else "dropped")
        b.set_title(f"{t:.2f} s ({lab})", fontsize=11, color={"used": "#c0392b", "dropped": "#2471a3"}.get(lab, "#8e44ad"))
    fig.savefig(fn, dpi=110); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    V = pd.read_csv(os.path.join(B, "analysis", "withdrawal_timing", "visual_vs_auto.csv"))
    V = V[V.typical & V.visual_delay.notna() & (V.judges_diff <= 0.07)]
    V["move_minus_stim"] = -V.visual_delay + V.mark_minus_delivery_s
    A = V[V.day == 1].sort_values("move_minus_stim").head(3)
    for k, r in enumerate(A.itertuples(), 1):
        sheet_AB(r, os.path.join(OUT, f"ex_A_day1_stim_mark_{k}.png"),
                 f"A{k}  Day 1: the paw moves {-r.move_minus_stim:.2f} s BEFORE your stimulus mark")
    Bd = V[V.day == 2].sort_values("move_minus_stim").iloc[[0, len(V[V.day == 2]) // 2, -1]]
    for k, r in enumerate(Bd.itertuples(), 1):
        sheet_AB(r, os.path.join(OUT, f"ex_B_day2_click_{k}.png"),
                 f"B{k}  Day 2: the paw moves {r.move_minus_stim:+.2f} s after your click")
    T = pd.read_csv(TAB).set_index(["mouse", "phase"])
    held = [("5941_d2_pre", 209.93, 4.83), ("1191_d2_pre", 331.93, 2.58), ("5939_d2_post", 957.13, 1.88)]
    for k, (rid, t0, dur) in enumerate(held, 1):
        m, _, ph = rid.split("_")
        L = pd.read_csv(os.path.join(LOGD, f"{rid}_logger_deliveries.csv"))
        used = sorted(L.onset_session_s[(L.onset_session_s >= t0 - 1) & (L.onset_session_s <= t0 + dur + 1)].round(2))
        sheet_CD(rid, T.loc[(int(m), ph), "logger_session"], t0, t0 + dur, used or [t0], [],
                 os.path.join(OUT, f"ex_C_held_click_{k}.png"), f"C{k}  {rid}: clicker held {dur:.2f} s from {t0:.2f} s")
    dbl = [("1191_d2_pre", 893.83, 893.90), ("5937_d2_pre", 213.17, 213.40), ("5939_d2_pre", 945.07, 945.50)]
    for k, (rid, t1, t2) in enumerate(dbl, 1):
        m, _, ph = rid.split("_")
        sheet_CD(rid, T.loc[(int(m), ph), "logger_session"], t1, t2, [t1], [t2],
                 os.path.join(OUT, f"ex_D_double_click_{k}.png"), f"D{k}  {rid}: second press {t2 - t1:.2f} s after the first (dropped)")
    print("->", OUT)


if __name__ == "__main__":
    main()
