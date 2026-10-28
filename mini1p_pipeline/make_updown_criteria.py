"""make_updown_criteria.py - 10-26 (user): "need to show before this slide how we select up and down cell ... what is criteria UP and
DOWN cell with representative neuron's traces".

One session (pin prick, 5939 Day 2 = after vehicle, folder-3 marks + window) and three of its neurons: the clearest UP, the clearest
DOWN and a typical not-responsive one. For each: every trial (heat map), the mean trace with the two windows, and the time-shift null
(the same response measured at 1000 shifted copies of the event train) with the observed value, p and BH q - the exact rule used in
cells_classes.csv (ca_event_analysis.shift_p + step_classify).

  SCORING_SUFFIX=_mix4  python make_updown_criteria.py  ->  <FIG>\\s3b_updown_criteria.png + .json
"""
import json
import os
import sys

os.environ.setdefault("SCORING_SUFFIX", "_mix4")
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import ca_day1_analysis as CA  # noqa: E402
import ca_event_analysis as EA  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
A = B + r"\analysis\_variants\mix4"
FIG = B + r"\lab_meeting\decks_Day1_Day2\v2_20261022\FINAL_20261026_PI5_mixmarks\figures"
M, D, PH, EV = "5939", 2, "post", "Pin prick"
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

cc = pd.read_csv(A + r"\event_analysis\cells_classes.csv")
cc["mouse"] = cc.mouse.astype(str)
X = cc[(cc.mouse == M) & (cc.day == D) & (cc.phase == PH) & (cc.event == EV)].copy()
pre, gap, a, b = (float(X.iloc[0][k]) for k in ("pre", "gap", "a", "b"))
up = X[X.cls == "activated"].sort_values("q").iloc[0]
dn = X[X.cls == "inhibited"].sort_values("q").iloc[0]

s = CA.load_session(EA.ROOT[D], M, PH, EA.SCORE[D], day=D)
Z = np.asarray(s["T"]["z"], float)
ok = np.isfinite(Z)
s["Zc"] = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.where(ok, Z, 0), 0)])
s["Nc"] = np.vstack([np.zeros((1, Z.shape[1]), np.int32), np.cumsum(ok, 0, dtype=np.int32)])
s["fr"] = s["T"]["frame"].astype(int)
s["ids"] = s["T"]["cell_ids"].astype(int)
frames = s["B"]["ev"][EV]
obs, p, nt = EA.shift_p(s, frames, pre, gap, a, b)                 # same call as step_classify -> must reproduce the table
q = CA.bh(p)
col = {int(c): j for j, c in enumerate(s["ids"])}


def flatness(cell):
    """largest |mean trace| from -1 to +2 s after removing the 0.5 s pre-event mean (small = a clearly flat, typical neuron)"""
    j = col[int(cell)]
    idx, _ = EA.ev_rows(s, frames)
    idx = idx[(idx > 4 * EA.FPS) & (idx < Z.shape[0] - 7 * EA.FPS)]
    L = np.arange(int(-1 * EA.FPS), int(2 * EA.FPS))
    P = np.arange(int(-pre * EA.FPS), 0)
    tr = np.nanmean(Z[idx[:, None] + L[None, :], j], 0) - np.nanmean(Z[idx[:, None] + P[None, :], j])
    return float(np.nanmax(np.abs(tr)))


cand = X[(X.cls == "non-responsive") & (X.p >= 0.4)]           # 10-26 (user: 'non responsive looks same as up'): flattest one
nr = cand.loc[min(cand.index, key=lambda i: flatness(cand.loc[i, "cell"]))]
EXS = [("UP", up, "#e74c3c"), ("DOWN", dn, "#2e86c1"), ("not responsive", nr, "#7f8c8d")]
for _, r, _ in EXS:
    j = col[int(r.cell)]
    assert abs(obs[j] - r.resp) < 1e-9 and abs(p[j] - r.p) < 1e-9, ("does not reproduce cells_classes.csv", r.cell)


def null_dist(j, n_shuf=1000, seed=0):
    """the 1000 shifted-train responses of cell j (same loop as EA.shift_p)"""
    fr, Zc, Nc = s["fr"], s["Zc"], s["Nc"]
    n = len(fr)
    p0, p1, k0, k1 = EA.win_rows(pre, gap, a, b)
    lo = int(np.searchsorted(fr, int(CA.SKIP_S * EA.FPS))) - p0
    idx, _ = EA.ev_rows(s, frames)
    idx = idx[(idx >= lo) & (idx + k1 < n)]

    def resp(ii):
        ii = ii[(ii >= lo) & (ii + k1 < n)]
        x = (Zc[ii + k1, j] - Zc[ii + k0, j]) / np.maximum(Nc[ii + k1, j] - Nc[ii + k0, j], 1)
        y = (Zc[ii + p1, j] - Zc[ii + p0, j]) / np.maximum(Nc[ii + p1, j] - Nc[ii + p0, j], 1)
        return np.nanmean(x - y)
    bnd = np.unique(np.searchsorted(fr, np.array([int(x * EA.FPS) for x in EA.SEG_S] + [int(fr[-1]) + 1])))
    seg = np.searchsorted(bnd, idx, side="right") - 1
    rng = np.random.default_rng(seed)
    w = max(k1, -p0)
    out = []
    for _ in range(n_shuf):
        ii = idx.copy()
        for k in np.unique(seg):
            if k < 0 or k + 1 >= len(bnd):
                continue
            A0, A1 = bnd[k], bnd[k + 1]
            L = A1 - A0
            if L <= 2 * w:
                continue
            msk = seg == k
            ii[msk] = A0 + (idx[msk] - A0 + int(rng.integers(w, L - w))) % L
        out.append(resp(np.sort(ii)))
    return np.array(out), idx


T0, T1 = -3.0, 6.0
lag = np.arange(int(T0 * EA.FPS), int(T1 * EA.FPS))
tt = lag / EA.FPS
fig = plt.figure(figsize=(12.3, 4.75))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.15, 0.95], hspace=0.62, wspace=0.28, left=0.07, right=0.965, top=0.90, bottom=0.10)
info = []
for c, (lab, r, colr) in enumerate(EXS):
    j = col[int(r.cell)]
    nd, idx = null_dist(j)
    rows = idx[:, None] + lag[None, :]
    ok = (rows >= 0) & (rows < Z.shape[0])
    tr = np.where(ok, Z[np.clip(rows, 0, Z.shape[0] - 1), j], np.nan)
    p0, p1, k0, k1 = EA.win_rows(pre, gap, a, b)
    base = np.nanmean(Z[np.clip(idx[:, None] + np.arange(p0, p1)[None, :], 0, Z.shape[0] - 1), j], 1)
    trb = tr - base[:, None]                                   # 10-26 (user): z minus the test's own 0.5 s pre-event window
    a0 = min(a, 0) - gap
    ax = fig.add_subplot(gs[0, c])
    vm = np.nanpercentile(np.abs(trb), 97)
    ax.imshow(trb, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[T0, T1, len(trb) + 0.5, 0.5], interpolation="nearest")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("trial", fontsize=10)
    ax.set_title(f"{lab}: neuron {int(r.cell)}  ({len(trb)} pin pricks)", fontsize=11.5, color=colr, fontweight="bold")
    ax.tick_params(labelsize=9)
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.04, pad=0.01)
    cb.set_label(f"z - {pre:g} s before", fontsize=8); cb.ax.tick_params(labelsize=8)
    ax2 = fig.add_subplot(gs[1, c], sharex=ax)
    mu = np.nanmean(trb, 0); se = np.nanstd(trb, 0, ddof=1) / np.sqrt(np.isfinite(trb).sum(0))
    ax2.axvspan(a0 - pre, a0, color="#d6eaf8", lw=0); ax2.axvspan(a, b, color="#d5d8dc", lw=0)
    ax2.fill_between(tt, mu - se, mu + se, color=colr, alpha=0.25, lw=0); ax2.plot(tt, mu, color=colr, lw=1.6)
    ax2.axhline(0, color="0.5", lw=0.7); ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel("time from pin prick (s)", fontsize=10); ax2.set_ylabel(f"z - mean of the\n{pre:g} s before", fontsize=10)
    ax2.tick_params(labelsize=9)
    ax3 = fig.add_subplot(gs[2, c])
    ax3.hist(nd, bins=40, color="0.75", lw=0)
    ax3.axvline(r.resp, color=colr, lw=2.2)
    ax3.set_xlabel("response R (z): observed (line) vs 1000 shifted trains", fontsize=9.5)
    ax3.set_ylabel("count", fontsize=10); ax3.tick_params(labelsize=9)
    right = r.resp < np.median(nd)                              # text on the side away from the observed line
    ax3.text(0.98 if right else 0.02, 0.95, f"R = {r.resp:+.3f}\np = {r.p:.3f}, q = {r.q:.3f}\n→ {lab}", transform=ax3.transAxes, ha="right" if right else "left", va="top",
             fontsize=9.5, color=colr, fontweight="bold")
    info.append(dict(label=lab, cell=int(r.cell), R=float(r.resp), p=float(r.p), q=float(r.q), n_trials=int(len(trb)),
                     null_lo=float(np.percentile(nd, 2.5)), null_hi=float(np.percentile(nd, 97.5))))
fig.text(0.07, 0.975, f"{M} (male), Day {D}, after vehicle; pin prick; response window {a:g}-{b:g} s vs the {pre:g} s before. "
                     "Blue band = pre-event window, grey band = response window.", fontsize=10.5, va="top")
fn = FIG + r"\s3b_updown_criteria.png"
fig.savefig(fn); plt.close(fig)
ns = dict(n_cells=int(len(X)), n_up=int((X.cls == "activated").sum()), n_down=int((X.cls == "inhibited").sum()))
json.dump(dict(mouse=M, day=D, phase=PH, event=EV, window=[pre, gap, a, b], examples=info, **ns),
          open(fn.replace(".png", ".json"), "w"), indent=1)
print("->", fn, info, ns)
