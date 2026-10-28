"""make_crossover_deck.py - Day 1 + Day 2 FULL deck (crossover, 4 mice), in the layout of your two decks
(11WT_SBI553_FINAL_hansol_checked_v3_summary: every mouse / every behaviour, per stimulus, same mouse both days;
Ntsr1 neurons+TRUFACT_MarkGreg_updated_somascore_HS: responsive cells per event on the FOV, UP / DOWN / rest averages,
population AUC + PSTH). Your rules (2026-10-22): few, intuitive slides; title on top, ONE key sentence + an arrow on the
panel to look at, one or two sentences at the bottom (what we did / found); legends, n, tests and definitions in the
NOTES; big fonts.
Units (as in the Ntsr1 deck): z = dF/F / robust SD (MAD) of the session; PSTH = mean +- SEM over neurons of the trial-
averaged z, each trial baselined on [-2, 0] s; AUC (z*s) = area of that trial-baselined z over [0, 2] s after the event.
Conditions per mouse: baseline = the session before the injection (both days), vehicle / SBI-553 = the injection session
of that day (crossover: 5941 + 5939 SBI-553 on Day 1, 5937 + 1191 on Day 2). Responsive = the time-shift test of
ca_day1_analysis (q <= 0.20, events shifted inside their own block: keeps the block state).
Day-2 selections are PROVISIONAL until you confirm them (red mark).

  python make_crossover_deck.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553 --out <deck.pptx> [--lang kr]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_day1_analysis as CA  # noqa: E402

MO = ["5941", "5939", "5937", "1191"]
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5941": "vehicle", "5939": "vehicle", "5937": "SBI-553", "1191": "SBI-553"}}
COND = ["baseline", "vehicle", "SBI-553"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
BEHEV = ["withdrawal", "flinch", "movement onset", "escape/rearing onset"]
SW, SH = 13.333, 7.5
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False, "axes.spines.right": False})


def ptxt(p):
    if not np.isfinite(p):
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def cond_of(m, day, phase):
    return "baseline" if phase == "pre" else ARMS[day][m]


def callout(fig, ax, xy, text, xytext=None, color="#c0392b", y=0.965):
    """the key-message arrow: a short bold label in the figure's top margin above the panel, pointing at xy (data
    coordinates of ax). Call after the layout (tight_layout with rect top <= 0.9)."""
    bb = ax.get_position()
    ax.annotate(text, xy=xy, xycoords="data", xytext=((bb.x0 + bb.x1) / 2, y), textcoords="figure fraction", ha="center",
                va="top", fontsize=13, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, shrinkA=2, shrinkB=4), annotation_clip=False)


# ------------------------------------------------------------------------------------------------ data
def load_tests(B):
    out = []
    for d in (1, 2):
        E = pd.read_csv(os.path.join(B, "analysis", f"ca_day{d}", f"ca_d{d}_cells_event_tests.csv"))
        E["mouse"] = E.mouse.astype(str); E["day"] = d
        E["condition"] = [cond_of(m, d, p) for m, p in zip(E.mouse, E.phase)]
        out.append(E)
    return pd.concat(out, ignore_index=True)


def load_rates(B):
    out = []
    for d in (1, 2):
        W = pd.read_csv(os.path.join(B, "analysis", f"ca_day{d}", f"ca_d{d}_cells_whole_session.csv"))
        W["mouse"] = W.mouse.astype(str); W["day"] = d
        num = sum(W[f"still rate | {s}"].fillna(0) * W[f"still min | {s}"].fillna(0) for s in STIM)
        den = sum(W[f"still min | {s}"].fillna(0) for s in STIM)
        W["rate_blocks"] = num / den.replace(0, np.nan)
        W["condition"] = [cond_of(m, d, p) for m, p in zip(W.mouse, W.phase)]
        out.append(W[["mouse", "day", "phase", "cell", "condition", "rate_blocks"]])
    return pd.concat(out, ignore_index=True)


def load_pairs(B):
    P = {}
    for m in MO:
        f = os.path.join(B, "analysis", "crossday", f"{m}_pairs.csv")
        P[m] = pd.read_csv(f) if os.path.isfile(f) else pd.DataFrame(columns=["day1_id", "day2_id"])
    return P


def psth_all(B, events, cache):
    """per session and event: time x cells trial-mean of the trial-baselined z (and the cell ids)"""
    if os.path.isfile(cache):
        Z = np.load(cache, allow_pickle=True)
        return Z["D"].item()
    D = {}
    score = {1: os.path.join(B, "scoring_mini1p", "20261019_final"), 2: os.path.join(B, "scoring_mini1p", "20261021_final")}
    root = {1: os.path.join(B, "mini1p_out", "20261019"), 2: os.path.join(B, "mini1p_out", "20261021")}
    for m in MO:
        for d in (1, 2):
            for ph in ("pre", "post"):
                s = CA.load_session(root[d], m, ph, score[d], day=d)
                if s is None or s["B"] is None:
                    continue
                ids = s["T"]["cell_ids"].astype(int)
                for ev in events:
                    X, tt, used = CA.peri(s, s["B"]["ev"].get(ev, []))
                    if X is None or len(used) < CA.MIN_TRIALS:
                        continue
                    pre = (tt >= CA.PRE_W[0]) & (tt < CA.PRE_W[1])
                    Xb = X - np.nanmean(X[:, pre, :], 1, keepdims=True)
                    D[(m, d, ph, ev)] = dict(mean=np.nanmean(Xb, 0).astype(np.float32), tt=tt, ids=ids, n=len(used))
    np.savez_compressed(cache, D=np.array(D, dtype=object))
    return D


# ------------------------------------------------------------------------------------------------ figures
def fig_rate(R, out):
    """every neuron of every session: transient rate in the stimulus blocks (still frames), pooled over mice"""
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    rng = np.random.default_rng(0)
    res = {}
    for k, c in enumerate(COND):
        v = R[R.condition == c].rate_blocks.dropna().values
        res[c] = v
        ax[0].boxplot(v, positions=[k], widths=0.55, showfliers=False, medianprops=dict(color="k", lw=2.5),
                      boxprops=dict(color=CC[c], lw=2), whiskerprops=dict(color=CC[c]), capprops=dict(color=CC[c]))
        for m in MO:
            w = R[(R.condition == c) & (R.mouse == m)].rate_blocks.dropna().values
            ax[0].scatter(k + rng.uniform(-0.22, 0.22, len(w)), w, s=9, color=MC[m], alpha=0.45, lw=0)
    ax[0].set_xticks(range(3), [f"{c}\nn = {len(res[c])}" for c in COND])
    ax[0].set_ylabel("Ca transients / min\n(stimulus blocks, still frames)")
    ax[0].set_title("A  all neurons pooled (dot = one neuron, colour = mouse)", loc="left")
    pv = mannwhitneyu(res["SBI-553"], res["vehicle"]).pvalue
    top = np.nanpercentile(np.r_[res["vehicle"], res["SBI-553"]], 97)
    ax[0].plot([1, 2], [top * 1.05] * 2, color="k", lw=1); ax[0].text(1.5, top * 1.07, star(pv), ha="center", fontsize=13)
    ax[0].set_ylim(0, top * 1.22)
    # per mouse medians, one line per mouse
    med = R.groupby(["mouse", "condition"]).rate_blocks.median().unstack()
    for m in MO:
        if m in med.index:
            ax[1].plot(range(3), med.loc[m, COND], marker=MK[m], color=MC[m], lw=2, ms=9, label=m)
    ax[1].set_xticks(range(3), COND); ax[1].set_ylabel("median transients / min (per mouse)")
    ax[1].set_title("B  one line per mouse", loc="left")
    ax[1].legend(title="mouse", frameon=False, fontsize=10)
    d_m = {m: med.loc[m, "SBI-553"] - med.loc[m, "vehicle"] for m in MO if m in med.index}
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    callout(fig, ax[0], (2, np.nanmedian(res["SBI-553"])), "SBI-553 vs vehicle")
    fig.savefig(out, dpi=150); plt.close(fig)
    return dict(p=pv, med={c: float(np.nanmedian(res[c])) for c in COND}, n={c: len(res[c]) for c in COND},
                n_lower=int(sum(v < 0 for v in d_m.values())), d_m=d_m)


def tracked_table(E, P, value="auc_post"):
    """one row per tracked neuron (Day-1 id + Day-2 id) and event: baseline (mean of the two days), vehicle, SBI-553 and
    the within-day changes"""
    rows = []
    for m in MO:
        X = E[E.mouse == m]
        for r in P[m].itertuples():
            a = X[(X.day == 1) & (X.cell == int(r.day1_id))].set_index(["phase", "event"])[value]
            b = X[(X.day == 2) & (X.cell == int(r.day2_id))].set_index(["phase", "event"])[value]
            for ev in set(a.index.get_level_values(1)) & set(b.index.get_level_values(1)):
                try:
                    v1p, v1i, v2p, v2i = a[("pre", ev)], a[("post", ev)], b[("pre", ev)], b[("post", ev)]
                except KeyError:
                    continue
                d1, d2 = ARMS[1][m], ARMS[2][m]
                inj = {d1: v1i, d2: v2i}; dch = {d1: v1i - v1p, d2: v2i - v2p}
                rows.append(dict(mouse=m, day1_id=int(r.day1_id), day2_id=int(r.day2_id), event=ev,
                                 baseline=(v1p + v2p) / 2, vehicle=inj["vehicle"], sbi=inj["SBI-553"],
                                 d_vehicle=dch["vehicle"], d_sbi=dch["SBI-553"]))
    return pd.DataFrame(rows)


def fig_population(E, D, events, out, label):
    """Ntsr1 layout: top = AUC per condition (bar = mean +- SEM over all neurons pooled, markers = mouse means);
    bottom = PSTH mean +- SEM over all neurons pooled"""
    n = len(events)
    fig, ax = plt.subplots(2, n, figsize=(12.8, 6.0), gridspec_kw=dict(height_ratios=[1, 1.05]))
    res = []
    for k, ev in enumerate(events):
        a = ax[0, k]
        X = E[E.event == ev]
        vals = {c: X[X.condition == c].auc_post.dropna().values for c in COND}
        for i, c in enumerate(COND):
            v = vals[c]
            a.bar(i, np.mean(v), yerr=np.std(v) / np.sqrt(max(len(v), 1)), color=CC[c], alpha=0.85, capsize=4, width=0.7)
            for j, m in enumerate(MO):
                w = X[(X.condition == c) & (X.mouse == m)].auc_post.dropna()
                if len(w):
                    a.scatter(i + (j - 1.5) * 0.12, w.mean(), marker=MK[m], color=MC[m], s=26, zorder=3, edgecolor="w", lw=0.5)
        a.axhline(0, color="0.3", lw=0.8)
        p = mannwhitneyu(vals["SBI-553"], vals["vehicle"]).pvalue if min(len(vals["SBI-553"]), len(vals["vehicle"])) > 3 else np.nan
        a.set_xticks(range(3), ["base", "veh", "SBI"])
        a.set_title(f"{ev}\nSBI vs veh: {star(p)}", fontsize=12.5)
        if k == 0:
            a.set_ylabel("AUC (z*s), 0-2 s")
        res.append(dict(event=ev, p=p, **{f"mean_{c}": float(np.mean(vals[c])) for c in COND},
                        **{f"n_{c}": len(vals[c]) for c in COND}))
        b = ax[1, k]
        for c in COND:
            M = [D[key]["mean"] for key in D if key[3] == ev and cond_of(key[0], key[1], key[2]) == c]
            if not M:
                continue
            tt = next(D[key]["tt"] for key in D if key[3] == ev)
            Y = np.concatenate(M, 1)
            mu = CA.sm(np.nanmean(Y, 1)[:, None], 6)[:, 0]
            se = CA.sm((np.nanstd(Y, 1) / np.sqrt(np.sum(np.isfinite(Y), 1)))[:, None], 6)[:, 0]
            b.plot(tt, mu, color=CC[c], lw=2, label=f"{c} (n={Y.shape[1]})")
            b.fill_between(tt, mu - se, mu + se, color=CC[c], alpha=0.2, lw=0)
        b.axvspan(0, 2, color="0.9", zorder=0); b.axvline(0, color="0.4", lw=0.8, ls="--"); b.axhline(0, color="0.6", lw=0.6)
        b.set_xlim(-3, 6); b.set_xlabel("time from event (s)")
        if k == 0:
            b.set_ylabel("z (trial-baselined)")
            b.legend(frameon=False, fontsize=9.5, loc="upper left")
    T = pd.DataFrame(res)
    T["diff"] = T["mean_SBI-553"] - T["mean_vehicle"]
    sig = T[T.p < 0.05]
    pick = (sig if len(sig) else T).assign(ad=lambda q: q["diff"].abs()).sort_values("ad").iloc[-1]
    k = events.index(pick.event)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    callout(fig, ax[0, k], (2, pick["mean_SBI-553"]), "look here", y=0.985)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, pick


def fig_heatmap(D, P, ev, out):
    """the SAME neurons (tracked on both days) in the three conditions; rows grouped by mouse (the mouse = the label on
    the left), sorted inside each mouse by the baseline response"""
    blocks, labels = [], []
    tt = None
    for m in MO:
        P_ = P[m]
        keys = {(d, ph): (m, d, ph, ev) for d in (1, 2) for ph in ("pre", "post")}
        if not all(k in D for k in keys.values()) or not len(P_):
            continue
        tt = D[keys[(1, "pre")]]["tt"]
        def col(d, ph, cid):
            K = D[keys[(d, ph)]]
            i = np.flatnonzero(K["ids"] == cid)
            return K["mean"][:, i[0]] if len(i) else np.full(len(tt), np.nan)
        rows = []
        for r in P_.itertuples():
            b1, b2 = col(1, "pre", int(r.day1_id)), col(2, "pre", int(r.day2_id))
            i1, i2 = col(1, "post", int(r.day1_id)), col(2, "post", int(r.day2_id))
            inj = {ARMS[1][m]: i1, ARMS[2][m]: i2}
            rows.append(((b1 + b2) / 2, inj["vehicle"], inj["SBI-553"]))
        base = np.array([r[0] for r in rows]); veh = np.array([r[1] for r in rows]); sbi = np.array([r[2] for r in rows])
        rw = (tt >= 0) & (tt < 2)
        o = np.argsort(-np.nanmean(base[:, rw], 1))
        blocks.append((m, base[o], veh[o], sbi[o]))
    if not blocks:
        return None
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 5.6), sharey=True)
    vmax = 1.5
    for k, (c, idx) in enumerate(zip(COND, (1, 2, 3))):
        Y = np.vstack([b[idx] for b in blocks])
        im = ax[k].imshow(CA.sm(Y.T, 6).T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          extent=[tt[0], tt[-1], len(Y), 0], interpolation="nearest")
        ax[k].axvline(0, color="k", lw=0.8, ls="--")
        ax[k].set_xlim(-3, 6); ax[k].set_xlabel("time from event (s)")
        ax[k].set_title(c, color=CC[c], fontsize=14, fontweight="bold")
        y0 = 0
        for m, b, *_ in blocks:
            ax[k].axhline(y0, color="k", lw=0.8)
            if k == 0:
                ax[k].text(-3.4, y0 + len(b) / 2, m, ha="right", va="center", fontsize=12, color=MC[m], fontweight="bold")
            y0 += len(b)
    for a_ in ax:
        a_.set_yticks([])
    ax[0].set_ylabel("tracked neurons, grouped by mouse", labelpad=46)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label("z (trial-baselined)")
    fig.suptitle(f"{ev}: the same neurons in the three conditions", fontsize=14, y=0.995)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    rw = (tt >= 0) & (tt < 2)
    m3 = [float(np.nanmean(np.vstack([b[i] for b in blocks])[:, rw])) for i in (1, 2, 3)]
    return dict(n=sum(len(b[1]) for b in blocks), mean=dict(zip(COND, m3)))


def fig_responsive(E, events, out):
    """% of neurons UP (above 0) and DOWN (below 0) per event and condition, pooled; markers = mice"""
    fig, ax = plt.subplots(1, len(events), figsize=(12.8, 4.8), sharey=True)
    res = []
    for k, ev in enumerate(events):
        a = ax[k]
        X = E[E.event == ev]
        for i, c in enumerate(COND):
            Y = X[X.condition == c]
            up, dn, n = (Y.cls_shift == "UP").sum(), (Y.cls_shift == "DOWN").sum(), len(Y)
            a.bar(i, 100 * up / max(n, 1), color=CC[c], width=0.7)
            a.bar(i, -100 * dn / max(n, 1), color=CC[c], width=0.7, alpha=0.45, hatch="//", edgecolor="w")
            for j, m in enumerate(MO):
                Z = Y[Y.mouse == m]
                if len(Z):
                    a.scatter(i + (j - 1.5) * 0.12, 100 * (Z.cls_shift == "UP").mean(), marker=MK[m], color=MC[m], s=22, zorder=3)
                    a.scatter(i + (j - 1.5) * 0.12, -100 * (Z.cls_shift == "DOWN").mean(), marker=MK[m], color=MC[m], s=22, zorder=3)
            res.append(dict(event=ev, condition=c, n=n, up=int(up), down=int(dn)))
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(range(3), ["base", "veh", "SBI"]); a.set_title(ev, fontsize=13)
        Rv = {c: (X[X.condition == c].cls_shift == "UP").sum() for c in COND}
        Nv = {c: (X.condition == c).sum() for c in COND}
        p = fisher_exact([[Rv["SBI-553"], Nv["SBI-553"] - Rv["SBI-553"]], [Rv["vehicle"], Nv["vehicle"] - Rv["vehicle"]]])[1]
        res[-1]["p_up"] = p
        a.text(1.5, a.get_ylim()[1] if k else 0, "", fontsize=1)
    ax[0].set_ylabel("% of neurons\nUP (solid) / DOWN (hatched)")
    T = pd.DataFrame(res)
    T["pct_up"] = 100 * T.up / T.n.clip(lower=1); T["pct_down"] = 100 * T.down / T.n.clip(lower=1)
    W = T.pivot(index="event", columns="condition", values="pct_up")
    W["d"] = W["SBI-553"] - W["vehicle"]
    ev_pick = W.d.abs().idxmax()
    k = events.index(ev_pick)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    callout(fig, ax[k], (2, float(W.loc[ev_pick, "SBI-553"])), "look here")
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, ev_pick, W


def footprints(B, m, d):
    t = f"{m}_d{d}"
    root = os.path.join(B, "mini1p_out", "20261019" if d == 1 else "20261021", t)
    S = sio.loadmat(os.path.join(root, "3_extract", f"{t}_extract.mat"), variable_names=["spatial_weights"])["spatial_weights"]
    mx = sio.loadmat(os.path.join(root, "3_extract", f"{t}_projections.mat"), variable_names=["max_dff"])["max_dff"]
    return np.asarray(S, np.float32), np.asarray(mx, np.float32)


def fig_maps(B, E, ev, out):
    """Ntsr1 'responsive cells per event': rows = mice, columns = the four sessions; red = UP, blue = DOWN, grey = the
    other neurons, on that day's max dF/F"""
    cols = [(1, "pre"), (1, "post"), (2, "pre"), (2, "post")]
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.9))
    for r_, m in enumerate(MO):
        r, c0 = r_ // 2, 4 * (r_ % 2)
        FP = {d: footprints(B, m, d) for d in (1, 2)}
        for c_, (d, ph) in enumerate(cols):
            c = c0 + c_
            a = ax[r, c]; a.set_axis_off()
            S, mx = FP[d]
            lo, hi = np.percentile(mx, [1, 99.8])
            a.imshow(np.clip((mx - lo) / (hi - lo), 0, 1) ** 0.7, cmap="gray")
            X = E[(E.mouse == m) & (E.day == d) & (E.phase == ph) & (E.event == ev)]
            nu = nd = 0
            for q in X.itertuples():
                j = int(q.cell) - 1
                if j >= S.shape[2]:
                    continue
                f = S[:, :, j]
                if f.max() <= 0:
                    continue
                col = {"UP": "#ff2d2d", "DOWN": "#2f8cff"}.get(q.cls_shift, "#9a9a9a")
                lw = 1.6 if q.cls_shift in ("UP", "DOWN") else 0.5
                a.contour(f >= 0.3 * f.max(), levels=[0.5], colors=[col], linewidths=lw)
                nu += q.cls_shift == "UP"; nd += q.cls_shift == "DOWN"
            cnd = cond_of(m, d, ph)
            a.set_title(f"{m}  D{d} {cnd}\n{nu} UP, {nd} DOWN / {len(X)}", fontsize=10.5, color=CC[cnd])
    fig.legend(handles=[Patch(color="#ff2d2d", label="UP"), Patch(color="#2f8cff", label="DOWN"),
                        Patch(color="#9a9a9a", label="no response")], loc="upper center", ncol=3, frameon=False, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=130); plt.close(fig)


def fig_resp_change(E, P, events, out):
    """tracked neurons responsive (UP or DOWN) in a day's baseline: % with the SAME response after the injection, on the
    vehicle day and on the SBI-553 day"""
    rows = []
    for m in MO:
        X = E[E.mouse == m]
        for r in P[m].itertuples():
            for d, cid in ((1, int(r.day1_id)), (2, int(r.day2_id))):
                Y = X[(X.day == d) & (X.cell == cid)]
                for ev in events:
                    a = Y[(Y.event == ev) & (Y.phase == "pre")]; b = Y[(Y.event == ev) & (Y.phase == "post")]
                    if not len(a) or not len(b):
                        continue
                    ca, cb = a.cls_shift.iloc[0], b.cls_shift.iloc[0]
                    rows.append(dict(mouse=m, event=ev, drug=ARMS[d][m], base=ca, inj=cb))
    T = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))
    out_rows = []
    for k, (title, sel) in enumerate((("A  responsive at baseline: still the same after the injection", "kept"),
                                       ("B  not responsive at baseline: responsive after the injection", "new"))):
        a = ax[k]
        for i, ev in enumerate(events):
            for j, drug in enumerate(("vehicle", "SBI-553")):
                Z = T[(T.event == ev) & (T.drug == drug)]
                if sel == "kept":
                    Z = Z[Z.base.isin(["UP", "DOWN"])]; v = (Z.base == Z.inj)
                else:
                    Z = Z[~Z.base.isin(["UP", "DOWN"])]; v = Z.inj.isin(["UP", "DOWN"])
                pct = 100 * v.mean() if len(Z) else np.nan
                a.bar(i + (j - 0.5) * 0.38, pct, width=0.36, color=CC[drug], label=drug if i == 0 else None)
                a.text(i + (j - 0.5) * 0.38, (pct if np.isfinite(pct) else 0) + 1, f"{int(v.sum())}/{len(Z)}", ha="center", fontsize=8.5)
                out_rows.append(dict(kind=sel, event=ev, drug=drug, n=len(Z), k=int(v.sum()), pct=pct))
        a.set_xticks(range(len(events)), [e.replace(" onset", "") for e in events], rotation=15)
        a.set_ylabel("% of tracked neurons"); a.set_title(title, loc="left", fontsize=12.5)
        a.legend(frameon=False)
    O = pd.DataFrame(out_rows)
    K = O[O.kind == "kept"].pivot(index="event", columns="drug", values="pct")
    N = O[O.kind == "kept"].pivot(index="event", columns="drug", values="n")
    K["d"] = K["SBI-553"] - K["vehicle"]
    ok = (N.min(1) >= 10) & K.d.notna()
    pick = K.d[ok].abs().idxmax() if ok.any() else (K.d.abs().idxmax() if K.d.notna().any() else events[0])
    i = events.index(pick)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    callout(fig, ax[0], (i + 0.19, float(K.loc[pick, "SBI-553"]) if np.isfinite(K.loc[pick, "SBI-553"]) else 0), "look here")
    fig.savefig(out, dpi=150); plt.close(fig)
    return O, pick, K


BEHS = ["withdrawal", "flinch", "attending", "lick/bite", "escape/rearing", "general movement"]


def behaviour_table(B):
    S = pd.read_csv(os.path.join(B, "analysis", "behavior_crossover", "crossover_per_session.csv"))
    S["mouse"] = S.mouse.astype(str)
    for b in BEHS:
        S[f"{b} /del"] = S[b] / S.deliveries.replace(0, np.nan)
    return S


def fig_behaviour(S, out):
    """WT-deck layout: events per delivery, thin line = one mouse, bold = mean +- SEM over the 4 mice"""
    fig, ax = plt.subplots(1, len(BEHS), figsize=(12.8, 4.6))
    res = []
    for k, b in enumerate(BEHS):
        a = ax[k]
        M = {}
        for m in MO:
            X = S[S.mouse == m]
            M[m] = [X[X.condition == "baseline"][f"{b} /del"].mean(), X[X.condition == "vehicle"][f"{b} /del"].mean(),
                    X[X.condition == "SBI-553"][f"{b} /del"].mean()]
            a.plot(range(3), M[m], color=MC[m], lw=1, alpha=0.7, marker=MK[m], ms=5)
        Y = np.array(list(M.values()))
        mu, se = np.nanmean(Y, 0), np.nanstd(Y, 0) / np.sqrt(len(Y))
        a.errorbar(range(3), mu, yerr=se, color="k", lw=2.6, capsize=4, marker="o", ms=7, zorder=3)
        p = wilcoxon(Y[:, 2], Y[:, 1]).pvalue if np.all(np.isfinite(Y[:, 1:])) and np.any(Y[:, 2] != Y[:, 1]) else np.nan
        a.set_xticks(range(3), ["base", "veh", "SBI"]); a.set_title(b, fontsize=12.5)
        a.set_ylim(bottom=0)
        if k == 0:
            a.set_ylabel("events per delivery")
        res.append(dict(behaviour=b, base=mu[0], vehicle=mu[1], sbi=mu[2], p=p,
                        n_lower=int(np.sum(Y[:, 2] < Y[:, 1]))))
    fig.legend(handles=[Line2D([], [], color=MC[m], marker=MK[m], label=m) for m in MO]
               + [Line2D([], [], color="k", lw=2.6, marker="o", label="mean +- SEM (4 mice)")],
               loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    T = pd.DataFrame(res)
    T["rel"] = (T.sbi - T.vehicle) / T.vehicle.replace(0, np.nan)
    C_ = T[(T.vehicle >= 0.05) & T.rel.notna()]
    pick = (C_ if len(C_) else T.dropna(subset=["rel"])).sort_values(["n_lower", "rel"], ascending=[False, True]).iloc[0]
    k = BEHS.index(pick.behaviour)
    fig.tight_layout(rect=(0, 0, 1, 0.8))
    callout(fig, ax[k], (2, pick.sbi), "look here", y=0.84)
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, pick


def fig_behaviour_stim(B, out):
    """WT-deck 'lower on the drug day, every behaviour': % of deliveries followed by the behaviour, per stimulus;
    mean +- SEM over the 4 mice per condition (baseline = mean of the two days)"""
    rows = []
    for d in (1, 2):
        T = pd.read_csv(os.path.join(B, "analysis", f"behavior_day{d}", f"behav_d{d}_per_session_stimulus.csv"))
        T["mouse"] = T.mouse.astype(str); T["day"] = d
        T["condition"] = [("baseline" if ph == "baseline" else ARMS[d][m]) for m, ph in zip(T.mouse, T.phase)]
        rows.append(T)
    T = pd.concat(rows, ignore_index=True)
    cols = [("withdrawal_pct", "withdrawal"), ("flinch_pct", "flinch"), ("attending_after_pct", "attending"),
            ("escape/rearing_after_pct", "escape / rearing"), ("general movement_after_pct", "general movement"),
            ("lick/bite_after_pct", "lick / bite")]
    fig, ax = plt.subplots(2, 3, figsize=(12.6, 6.4))
    best = None
    for k, (cname, lab) in enumerate(cols):
        a = ax.ravel()[k]
        mus = {}
        for c in COND:
            G = T[T.condition == c].groupby(["mouse", "stim"])[cname].mean().unstack()
            G = G.reindex(columns=STIM)
            mu, se = G.mean(0), G.std(0) / np.sqrt(G.notna().sum(0))
            mus[c] = mu
            a.errorbar(range(4), mu, yerr=se, color=CC[c], lw=2.2, marker="o", capsize=3, label=c)
        for i, st in enumerate(STIM):
            dv = mus["SBI-553"][st] - mus["vehicle"][st]
            if mus["vehicle"][st] >= 5 and np.isfinite(dv) and (best is None or abs(dv) > abs(best[3])):
                best = (k, i, lab, dv, st, mus["vehicle"][st], mus["SBI-553"][st])
        a.set_xticks(range(4), ["light", "mild", "pin", "heat"]); a.set_title(lab, fontsize=12.5)
        a.set_ylim(bottom=0)
        if k % 3 == 0:
            a.set_ylabel("% of deliveries")
    ax[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if best:
        a_ = ax.ravel()[best[0]]
        a_.annotate("look here", xy=(best[1], best[6]), xytext=(0.5, 0.93), textcoords="axes fraction", ha="center", va="top",
                    fontsize=13, fontweight="bold", color="#c0392b",
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.2, shrinkA=2, shrinkB=4))
    fig.savefig(out, dpi=150); plt.close(fig)
    return T, best


# ------------------------------------------------------------------------------------------------ slides
class Deck:
    def __init__(self, lang, prov):
        from pptx import Presentation
        from pptx.util import Inches
        self.prs = Presentation(); self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.lang = lang; self.prov = prov
        self.font = "Malgun Gothic" if lang == "kr" else "Arial"

    def box(self, s, x, y, w, h, text, pt, bold=False, rgb=(0x33, 0x33, 0x33), align=None):
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        for i, line in enumerate(text.split("\n")):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(pt); p.font.bold = bold; p.font.name = self.font
            p.font.color.rgb = RGBColor(*rgb)
            if align == "center":
                p.alignment = PP_ALIGN.CENTER
        return tb

    def slide(self, section, title, key, fig, bottom, notes):
        from pptx.util import Inches
        from PIL import Image
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.box(s, 0.45, 0.08, 6, 0.3, section, 11, True, (0x8A, 0x8F, 0x98))
        if self.prov:
            self.box(s, 8.6, 0.06, 4.4, 0.3, self.prov, 10.5, True, (0xC0, 0x39, 0x2B))
        self.box(s, 0.45, 0.32, 12.4, 0.6, title, 26, True, (0x1F, 0x2D, 0x3D))
        if key:
            self.box(s, 0.45, 0.92, 12.4, 0.45, key, 16, True, (0xC0, 0x39, 0x2B))
        top, boxh = 1.42, 4.75
        if fig:
            im = Image.open(fig)
            w = 12.4; h = w * im.size[1] / im.size[0]
            if h > boxh:
                h = boxh; w = h * im.size[0] / im.size[1]
            s.shapes.add_picture(fig, Inches(0.45 + (12.4 - w) / 2), Inches(top), Inches(w), Inches(h))
        if bottom:
            self.box(s, 0.45, 6.3, 12.4, 1.0, bottom, 15)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def cards(self, title, key, cards, bottom, notes, section="SUMMARY"):
        from pptx.util import Inches
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_SHAPE
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.box(s, 0.45, 0.08, 6, 0.3, section, 11, True, (0x8A, 0x8F, 0x98))
        if self.prov:
            self.box(s, 8.6, 0.06, 4.4, 0.3, self.prov, 10.5, True, (0xC0, 0x39, 0x2B))
        self.box(s, 0.45, 0.32, 12.4, 0.6, title, 26, True, (0x1F, 0x2D, 0x3D))
        if key:
            self.box(s, 0.45, 0.92, 12.4, 0.45, key, 16, True, (0xC0, 0x39, 0x2B))
        for i, (big, head, txt) in enumerate(cards):
            x = 0.45 + (i % 2) * 6.25; y = 1.55 + (i // 2) * 2.3
            r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(6.05), Inches(2.1))
            r.fill.solid(); r.fill.fore_color.rgb = RGBColor(0xF3, 0xF5, 0xF8); r.line.fill.background()
            self.box(s, x + 0.15, y + 0.15, 1.9, 1.0, big, 34, True, (0xC0, 0x39, 0x2B))
            self.box(s, x + 2.1, y + 0.15, 3.8, 0.8, head, 15, True, (0x1F, 0x2D, 0x3D))
            self.box(s, x + 2.1, y + 0.95, 3.8, 1.1, txt, 12.5)
        if bottom:
            self.box(s, 0.45, 6.3, 12.4, 1.0, bottom, 15)
        s.notes_slide.notes_text_frame.text = notes
        return s


def L(lang, en, kr):
    return kr if lang == "kr" else en


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    ap.add_argument("--fresh", action="store_true", help="recompute the PSTH cache")
    a = ap.parse_args(argv)
    B, lg = a.base, a.lang
    od = os.path.splitext(a.out)[0] + "_figures"
    os.makedirs(od, exist_ok=True)
    f = lambda nm: os.path.join(od, nm)
    prov = "" if os.path.isfile(os.path.join(B, "mini1p_out", "20261021", "ALL_CONFIRMED.flag")) else \
        L(lg, "PROVISIONAL - Day-2 neuron selection not yet confirmed", "잠정 - Day 2 뉴런 선택 미확정")
    E = load_tests(B); R = load_rates(B); P = load_pairs(B)
    cache = os.path.join(od, "psth_cache.npz")
    if a.fresh and os.path.isfile(cache):
        os.remove(cache)
    D = psth_all(B, STIM + BEHEV, cache)
    S = behaviour_table(B)
    dk = Deck(lg, prov)
    ntr = {m: len(P[m]) for m in MO}
    # ------------------------------------------------------------------ numbers + figures
    rr = fig_rate(R, f("f04_rate_pooled.png"))
    TP, pk = fig_population(E, D, STIM, f("f06_population_stimuli.png"), "stimuli")
    TB, pkb = fig_population(E, D, BEHEV, f("f07_population_behaviour.png"), "behaviour")
    key_ev = pk.event
    hm = fig_heatmap(D, P, key_ev, f("f08_heatmap_tracked.png"))
    TR, ev_r, WR = fig_responsive(E, STIM + ["withdrawal", "movement onset"], f("f09_responsive_pct.png"))
    fig_maps(B, E, ev_r, f("f10_maps.png"))
    TC, ev_c, KC = fig_resp_change(E, P, STIM + ["withdrawal", "movement onset"], f("f11_responsive_change.png"))
    TBh, pbh = fig_behaviour(S, f("f12_behaviour.png"))
    _, bst = fig_behaviour_stim(B, f("f13_behaviour_stimulus.png"))
    import make_longitudinal_deck as LD
    _, nd = LD.fig_design(B, f("f02_design.png"))
    TS = LD.tracking_summary(B)
    nt = LD.fig_tracking_counts(TS, f("f03_tracking.png")) if len(TS) else ""
    tt_ = tracked_table(E, P)
    # ------------------------------------------------------------------ 1 summary
    up = WR.loc[ev_r]
    cards = [
        (f"{sum(ntr.values())}", L(lg, "neurons followed on both days", "두 날 모두 추적한 뉴런"),
         L(lg, ", ".join(f"{m} {ntr[m]}" for m in MO) + ". Same field of view, the treatment swapped on Day 2.",
           ", ".join(f"{m} {ntr[m]}" for m in MO) + ". 같은 시야, Day 2에 처치를 바꿈.")),
        (f"{pk['mean_SBI-553']:+.2f}", L(lg, f"{key_ev}: population response after SBI-553 (AUC z*s)",
                                         f"{key_ev}: SBI-553 후 집단 반응 (AUC z*s)"),
         L(lg, f"vs {pk['mean_vehicle']:+.2f} after vehicle and {pk['mean_baseline']:+.2f} at baseline; all neurons pooled, {ptxt(pk.p)}.",
           f"vehicle 후 {pk['mean_vehicle']:+.2f}, 기준 {pk['mean_baseline']:+.2f}; 전체 뉴런, {ptxt(pk.p)}.")),
        (f"{up['SBI-553']:.0f}%", L(lg, f"{ev_r}: neurons UP after SBI-553", f"{ev_r}: SBI-553 후 UP 뉴런"),
         L(lg, f"vs {up['vehicle']:.0f}% after vehicle and {up['baseline']:.0f}% at baseline.",
           f"vehicle 후 {up['vehicle']:.0f}%, 기준 {up['baseline']:.0f}%.")),
        (f"{int(pbh.n_lower)}/4", L(lg, f"mice: {pbh.behaviour} lower after SBI-553 than after vehicle",
                                    f"마리: SBI-553 후 {pbh.behaviour}가 vehicle 후보다 낮음"),
         L(lg, f"{pbh.vehicle:.2f} -> {pbh.sbi:.2f} events per delivery (mean of 4 mice); 4 mice: min p = 0.125.",
           f"자극당 {pbh.vehicle:.2f} -> {pbh.sbi:.2f}회 (4마리 평균); 4마리: 최소 p = 0.125."))]
    dk.cards(L(lg, "Day 1 + Day 2: SBI-553 vs vehicle in the same mice", "Day 1 + Day 2: 같은 마우스에서 SBI-553 vs vehicle"),
             L(lg, "Crossover, 4 mice: each mouse got vehicle on one day and SBI-553 on the other",
               "Crossover 4마리: 각 마우스가 하루는 vehicle, 하루는 SBI-553"),
             cards, L(lg, "Ca imaging (mini1p, BLA) and pain behaviour in every session; the same neurons were followed across the two days.",
                      "모든 세션에서 Ca 이미징(mini1p, BLA)과 통증 행동; 두 날 같은 뉴런을 추적."),
             L(lg, "AUC = area of the trial-baselined z ([-2, 0] s) over [0, 2] s after the event (z*s). Stats in each slide's notes. "
                   "n = 4 mice (2 per order): no group test across mice (minimum possible p = 0.125).",
               "AUC = 사건 후 [0, 2] s의 trial 기준선([-2, 0] s) 보정 z 면적 (z*s). 통계는 각 슬라이드 노트. n = 4마리."),
             section=L(lg, "SUMMARY", "요약"))
    # ------------------------------------------------------------------ 2 design
    dk.slide(L(lg, "DESIGN", "설계"), L(lg, "Crossover: vehicle on one day, SBI-553 on the other", "Crossover: 하루 vehicle, 다른 날 SBI-553"),
             L(lg, "Each mouse is its own control", "각 마우스가 자기 대조군"), f("f02_design.png"),
             L(lg, "Every day: a baseline session, then the injection session; the stimulus order is randomised per session.",
               "매일: 기준 세션 후 주사 세션; 자극 순서는 세션마다 무작위."), nd)
    # ------------------------------------------------------------------ 3 tracking
    if nt:
        dk.slide(L(lg, "NEURONS", "뉴런"), L(lg, "The same neurons on Day 1 and Day 2", "Day 1과 Day 2의 같은 뉴런"),
                 L(lg, f"{sum(ntr.values())} neurons followed across both days (green + orange)",
                   f"두 날에 걸쳐 {sum(ntr.values())}개 뉴런 추적 (초록 + 주황)"), f("f03_tracking.png"),
                 L(lg, "Each Day-1 neuron was looked for on Day 2 and judged on Day-2 evidence (local pattern, movie proof, cell rules).",
                   "Day 1 뉴런을 Day 2에서 찾아 Day 2 증거(국소 패턴, 영상 증거, 세포 규칙)로 판정."),
                 nt + "\nMaps for your check: analysis\\crossday\\<mouse>_tracking_map.png and <mouse>_CONFIRM_p*.png.")
    # ------------------------------------------------------------------ 4 activity, all neurons
    dlow = rr["n_lower"]
    dk.slide(L(lg, "ALL NEURONS", "전체 뉴런"), L(lg, "Activity of all neurons: baseline vs vehicle vs SBI-553", "전체 뉴런 활동: 기준 vs vehicle vs SBI-553"),
             L(lg, f"Median transients/min: baseline {rr['med']['baseline']:.1f}, vehicle {rr['med']['vehicle']:.1f}, SBI-553 {rr['med']['SBI-553']:.1f} ({ptxt(rr['p'])})",
               f"분당 transient 중앙값: 기준 {rr['med']['baseline']:.1f}, vehicle {rr['med']['vehicle']:.1f}, SBI-553 {rr['med']['SBI-553']:.1f} ({ptxt(rr['p'])})"),
             f("f04_rate_pooled.png"),
             L(lg, f"All neurons of every session pooled (colour = mouse). SBI-553 below vehicle in {dlow}/4 mice (right).",
               f"모든 세션의 뉴런을 합침 (색 = 마우스). SBI-553이 vehicle보다 낮은 마우스 {dlow}/4 (오른쪽)."),
             L(lg, "Rate = Ca transients per minute of still frames inside the 4 stimulus blocks. A: box = median and quartiles over all "
                   f"neurons of that condition (baseline n = {rr['n']['baseline']}: both days' baseline sessions; vehicle n = {rr['n']['vehicle']}; "
                   f"SBI-553 n = {rr['n']['SBI-553']}); dots = neurons coloured by mouse. Mann-Whitney U SBI-553 vs vehicle over neurons "
                   f"(exploratory: neurons are nested in 4 mice). B: per-mouse medians.",
               "빈도 = 4개 자극 블록 안 움직임 없는 프레임의 분당 Ca transient. A: 상자 = 조건별 전체 뉴런의 중앙값·사분위; 점 = 뉴런(마우스 색). "
               "Mann-Whitney U SBI-553 vs vehicle (뉴런 단위, 탐색적: 뉴런은 4마리에 속함). B: 마우스별 중앙값."))
    # ------------------------------------------------------------------ 5 / 6 population responses
    for fn, T_, pick, evs, ttl, ttl_kr in (("f06_population_stimuli.png", TP, pk, STIM, "Population response to each stimulus",
                                            "자극별 집단 반응"),
                                           ("f07_population_behaviour.png", TB, pkb, BEHEV, "Population response to each behaviour",
                                            "행동별 집단 반응")):
        d = pick["mean_SBI-553"] - pick["mean_vehicle"]
        dk.slide(L(lg, "ALL NEURONS", "전체 뉴런"), L(lg, ttl, ttl_kr),
                 L(lg, f"{pick.event}: {'smaller' if d < 0 else 'larger'} after SBI-553 than after vehicle "
                       f"(AUC {pick['mean_SBI-553']:+.2f} vs {pick['mean_vehicle']:+.2f} z*s, {ptxt(pick.p)})",
                   f"{pick.event}: SBI-553 후가 vehicle 후보다 {'작음' if d < 0 else '큼'} "
                   f"(AUC {pick['mean_SBI-553']:+.2f} vs {pick['mean_vehicle']:+.2f} z*s, {ptxt(pick.p)})"),
                 f(fn),
                 L(lg, "Top: response size (AUC) of all neurons pooled; bottom: the average trace around the event.",
                   "위: 전체 뉴런의 반응 크기(AUC); 아래: 사건 주변 평균 trace."),
                 L(lg, "z = dF/F / robust SD (MAD) of the session; each trial baselined on [-2, 0] s. Top: bar = mean +- SEM over all "
                       "neurons of the condition (all sessions pooled), markers = mouse means (5941 circle, 5939 square, 5937 triangle, "
                       "1191 diamond). Stars = Mann-Whitney U SBI-553 vs vehicle over neurons (exploratory, not corrected). Bottom: mean "
                       "+- SEM over neurons; grey band = the [0, 2] s window of the AUC. n neurons: "
                       + "; ".join(f"{r.event} base {r.n_baseline} / veh {r.n_vehicle} / SBI {r['n_SBI-553']}" for _, r in T_.iterrows()),
                   "z = 세션의 dF/F / robust SD (MAD); 각 trial은 [-2, 0] s로 기준 보정. 위: 막대 = 조건별 전체 뉴런 평균 +- SEM, 표식 = 마우스 평균. "
                   "별 = Mann-Whitney U SBI-553 vs vehicle (뉴런 단위, 보정 없음). 아래: 뉴런 평균 +- SEM; 회색 = AUC 구간 [0, 2] s."))
    # ------------------------------------------------------------------ 7 heatmap
    if hm:
        dk.slide(L(lg, "SAME NEURONS", "같은 뉴런"), L(lg, f"{key_ev}: every tracked neuron in the three conditions", f"{key_ev}: 추적한 모든 뉴런, 세 조건"),
                 L(lg, f"Mean response 0-2 s: baseline {hm['mean']['baseline']:+.2f}, vehicle {hm['mean']['vehicle']:+.2f}, "
                       f"SBI-553 {hm['mean']['SBI-553']:+.2f} z",
                   f"0-2 s 평균 반응: 기준 {hm['mean']['baseline']:+.2f}, vehicle {hm['mean']['vehicle']:+.2f}, SBI-553 {hm['mean']['SBI-553']:+.2f} z"),
                 f("f08_heatmap_tracked.png"),
                 L(lg, f"One row = one neuron followed on both days ({hm['n']} neurons), grouped by mouse; red = activity rose after the event.",
                   f"한 줄 = 두 날 추적한 뉴런 하나 ({hm['n']}개), 마우스별 묶음; 빨강 = 사건 후 활동 증가."),
                 L(lg, "Rows sorted inside each mouse by the baseline response (mean of the two baseline sessions). Colour = trial-averaged "
                       "z, baselined on [-2, 0] s, 0.2-s smoothing. The event shown is the one with the largest SBI-553 vs vehicle "
                       "difference on the population slide.",
                   "마우스 안에서 기준 반응 순 정렬(두 기준 세션 평균). 색 = trial 평균 z ([-2, 0] s 보정, 0.2 s 평활)."))
    # ------------------------------------------------------------------ 8 responsive %
    dk.slide(L(lg, "RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "How many neurons respond to each event", "사건마다 반응하는 뉴런 수"),
             L(lg, f"{ev_r}: {WR.loc[ev_r, 'SBI-553']:.0f}% UP after SBI-553 vs {WR.loc[ev_r, 'vehicle']:.0f}% after vehicle "
                   f"({WR.loc[ev_r, 'baseline']:.0f}% at baseline)",
               f"{ev_r}: SBI-553 후 UP {WR.loc[ev_r, 'SBI-553']:.0f}% vs vehicle 후 {WR.loc[ev_r, 'vehicle']:.0f}% (기준 {WR.loc[ev_r, 'baseline']:.0f}%)"),
             f("f09_responsive_pct.png"),
             L(lg, "Solid = % of neurons that rose after the event (UP), hatched = % that fell (DOWN); markers = mice.",
               "실선 = 사건 후 상승한 뉴런 %(UP), 빗금 = 하강한 뉴런 %(DOWN); 표식 = 마우스."),
             L(lg, "Responsive = time-shift test per neuron (the same events shifted inside their own block, 1000 shuffles), BH q <= 0.20 "
                   "within the session (exploratory, as in the Ntsr1 deck). Fisher exact SBI-553 vs vehicle on the UP counts: "
                   + "; ".join(f"{r.event} p = {r.p_up:.3f}" for _, r in TR.dropna(subset=["p_up"]).iterrows()),
               "반응 = 뉴런별 시간 이동 검정 (같은 블록 안에서 사건을 이동, 1000회), 세션 안 BH q <= 0.20 (탐색적). UP 수 Fisher exact SBI-553 vs vehicle."))
    # ------------------------------------------------------------------ 9 maps
    dk.slide(L(lg, "RESPONSIVE NEURONS", "반응 뉴런"), L(lg, f"Where the {ev_r} neurons are, session by session", f"{ev_r} 반응 뉴런의 위치, 세션별"),
             L(lg, "Red = UP, blue = DOWN; each row is one mouse, Day 1 left, Day 2 right", "빨강 = UP, 파랑 = DOWN; 한 줄 = 한 마우스, 왼쪽 Day 1, 오른쪽 Day 2"),
             f("f10_maps.png"),
             L(lg, "The responsive neurons are scattered in the field of view; compare the two injection columns (vehicle vs SBI-553).",
               "반응 뉴런은 시야 전체에 흩어져 있음; 두 주사 열(vehicle vs SBI-553)을 비교."),
             L(lg, "Outlines on that day's max dF/F projection; thin grey = no response. Same test as the previous slide.",
               "그날의 max dF/F 위 윤곽; 얇은 회색 = 반응 없음. 앞 슬라이드와 같은 검정."))
    # ------------------------------------------------------------------ 10 responsive change
    kc = KC.loc[ev_c]
    dk.slide(L(lg, "RESPONSIVE NEURONS", "반응 뉴런"), L(lg, "Do the same neurons keep responding after the injection?", "같은 뉴런이 주사 후에도 계속 반응하는가?"),
             L(lg, f"{ev_c}: {kc['SBI-553']:.0f}% kept their response after SBI-553 vs {kc['vehicle']:.0f}% after vehicle",
               f"{ev_c}: SBI-553 후 {kc['SBI-553']:.0f}%가 반응 유지 vs vehicle 후 {kc['vehicle']:.0f}%"),
             f("f11_responsive_change.png"),
             L(lg, "Tracked neurons only, within each day: baseline session -> injection session of the same day.",
               "추적한 뉴런만, 같은 날 안에서: 기준 세션 -> 주사 세션."),
             L(lg, "A: neurons UP or DOWN at that day's baseline; bar = % with the SAME class after the injection (numbers = k/n). "
                   "B: neurons without a response at baseline; bar = % responsive after the injection. Vehicle day vs SBI-553 day of "
                   "the same mice. Descriptive (no test).",
               "A: 그날 기준 세션에서 UP/DOWN이던 뉴런 중 주사 후 같은 반응 %. B: 기준에 반응 없던 뉴런 중 주사 후 반응 %. 기술 통계."))
    # ------------------------------------------------------------------ 11 behaviour average
    dk.slide(L(lg, "BEHAVIOUR", "행동"), L(lg, "Behaviour per delivery: all mice", "자극당 행동: 모든 마우스"),
             L(lg, f"{pbh.behaviour}: {pbh.vehicle:.2f} after vehicle -> {pbh.sbi:.2f} after SBI-553 (lower in {int(pbh.n_lower)}/4 mice)",
               f"{pbh.behaviour}: vehicle 후 {pbh.vehicle:.2f} -> SBI-553 후 {pbh.sbi:.2f} ({int(pbh.n_lower)}/4 마리에서 낮음)"),
             f("f12_behaviour.png"),
             L(lg, "Every behaviour divided by the number of deliveries of its session; baseline = mean of the two baseline sessions.",
               "모든 행동은 그 세션의 자극 수로 나눔; 기준 = 두 기준 세션의 평균."),
             L(lg, "Thin line = one mouse, bold = mean +- SEM over the 4 mice. Wilcoxon signed-rank SBI-553 vs vehicle over mice: "
                   + "; ".join(f"{r.behaviour} p = {r.p:.3f}" for _, r in TBh.iterrows())
                   + " (4 mice: minimum possible p = 0.125, so no test can reach 0.05). Reflexes within 2 s of a delivery, other "
                     "behaviours: bout starts within 3 s.",
               "얇은 선 = 마우스, 굵은 선 = 4마리 평균 +- SEM. Wilcoxon SBI-553 vs vehicle (4마리: 최소 p = 0.125)."))
    # ------------------------------------------------------------------ 12 behaviour per stimulus
    dk.slide(L(lg, "BEHAVIOUR", "행동"), L(lg, "Behaviour per stimulus", "자극별 행동"),
             (L(lg, f"{bst[2]} after {bst[4]}: {bst[5]:.0f}% of deliveries after vehicle vs {bst[6]:.0f}% after SBI-553",
                  f"{bst[4]} 후 {bst[2]}: vehicle 후 자극의 {bst[5]:.0f}% vs SBI-553 후 {bst[6]:.0f}%") if bst else ""),
             f("f13_behaviour_stimulus.png"),
             L(lg, "% of deliveries followed by each behaviour, per stimulus type; mean +- SEM over the 4 mice.",
               "자극 종류별로 각 행동이 뒤따른 자극 %; 4마리 평균 +- SEM."),
             L(lg, "Baseline = mean of the two baseline sessions of a mouse. Reflexes within 2 s, other behaviours within 3 s. Descriptive.",
               "기준 = 마우스의 두 기준 세션 평균. 기술 통계."))
    # ------------------------------------------------------------------ 13 limits
    dk.cards(L(lg, "What the data can and cannot say yet", "지금 데이터로 말할 수 있는 것과 없는 것"), "",
             [("4", L(lg, "mice (2 per order)", "마리 (순서마다 2마리)"),
               L(lg, "No group test across mice can reach p < 0.05 (minimum 0.125); neuron-level tests are exploratory.",
                 "마우스 간 검정은 p < 0.05에 도달 불가 (최소 0.125); 뉴런 단위 검정은 탐색적.")),
              ("5937", L(lg, "Day-2 signal weak", "Day 2 신호 약함"),
               L(lg, "Few active neurons on Day 2 (also at baseline): its Day-2 numbers are near the floor.",
                 "Day 2에 활동 뉴런이 적음 (기준에서도): Day 2 수치는 바닥 근처.")),
              ("5939", L(lg, "tracking weakest", "추적이 가장 약함"),
               L(lg, "Activity of the same neurons agrees less between days; many Day-1 neurons recovered need your check.",
                 "같은 뉴런의 활동이 두 날 사이 덜 일치; 복구한 Day 1 뉴런 다수는 확인 필요.")),
              (L(lg, "Day 2", "Day 2"), L(lg, "selection provisional", "선택 잠정"),
               L(lg, "Your confirmation of the Day-2 neurons may change every number here.",
                 "Day 2 뉴런 확인 결과에 따라 모든 수치가 바뀔 수 있음."))],
             "", L(lg, "Next: your confirmation of the Day-2 selection, then this deck again with the final neurons.",
                   "다음: Day 2 선택 확인 후 최종 뉴런으로 이 덱을 다시 만듦."), section=L(lg, "LIMITS", "한계"))
    # ------------------------------------------------------------------ supplement: all heatmaps + maps
    for ev in STIM + ["withdrawal", "movement onset"]:
        if ev == key_ev:
            continue
        r_ = fig_heatmap(D, P, ev, f(f"s_heatmap_{ev.replace(' ', '_').replace('/', '_')}.png"))
        if r_:
            dk.slide(L(lg, "SUPPLEMENT", "부록"), L(lg, f"{ev}: every tracked neuron in the three conditions", f"{ev}: 추적한 모든 뉴런, 세 조건"), "",
                     f(f"s_heatmap_{ev.replace(' ', '_').replace('/', '_')}.png"),
                     L(lg, f"Mean 0-2 s: baseline {r_['mean']['baseline']:+.2f}, vehicle {r_['mean']['vehicle']:+.2f}, SBI-553 {r_['mean']['SBI-553']:+.2f} z.",
                       f"0-2 s 평균: 기준 {r_['mean']['baseline']:+.2f}, vehicle {r_['mean']['vehicle']:+.2f}, SBI-553 {r_['mean']['SBI-553']:+.2f} z."),
                     L(lg, "As the main heatmap slide.", "본문 히트맵 슬라이드와 같음."))
    for ev in STIM + ["withdrawal"]:
        if ev == ev_r:
            continue
        fn = f(f"s_maps_{ev.replace(' ', '_').replace('/', '_')}.png")
        fig_maps(B, E, ev, fn)
        dk.slide(L(lg, "SUPPLEMENT", "부록"), L(lg, f"Where the {ev} neurons are", f"{ev} 반응 뉴런의 위치"), "", fn, "",
                 L(lg, "As the main map slide.", "본문 지도 슬라이드와 같음."))
    dk.prs.save(a.out)
    TP.round(4).to_csv(f("t_population_stimuli.csv"), index=False); TB.round(4).to_csv(f("t_population_behaviour.csv"), index=False)
    TR.round(4).to_csv(f("t_responsive.csv"), index=False); TC.round(3).to_csv(f("t_responsive_change.csv"), index=False)
    TBh.round(4).to_csv(f("t_behaviour.csv"), index=False); tt_.round(4).to_csv(f("t_tracked_auc.csv"), index=False)
    print("->", a.out, "|", len(dk.prs.slides), "slides | figures in", od)


if __name__ == "__main__":
    main()
