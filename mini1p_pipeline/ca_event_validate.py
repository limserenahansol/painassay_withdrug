"""ca_event_validate.py - is the event analysis right, and can the neurons be categorised in better ways? (2026-10-22)

  null      FALSE POSITIVES: the classification run on fake event trains (shifted inside their blocks, nothing is locked to
            them): how many neurons does each rule call responsive? Rules: BH q <= 0.20 per session (used so far), max-T
            (Westfall-Young: family-wise 5 %, valid when neurons share a response), each on the raw activity and on the
            SELECTIVE activity (each neuron minus the mean of all neurons = what it does beyond the shared response)
  spikein   RECALL: GCaMP-shaped responses (rise 0.1 s, decay 1 s) of known size added to 10 % of the neurons at the real
            deliveries of a real session; how many does each rule / window find, and how many others does it call
  glm       SEPARATION: responses locked to the deliveries in some neurons and to the withdrawals in others, added to every
            real session (Day 1 taps: withdrawal 0.1 s after the mark; Day 2 logger: 0.43 s): does the encoding model
            put each in the right class?
  robust    the classes of every neuron x event x session with all four rules
  clusters  functional types: k-means on every neuron's mean response to the 6 main events (baseline sessions), then the
            type of the same neurons after the injection
  decode    can the population tell the 4 stimuli apart, or predict a withdrawal? (cross-validated per session, against
            the same decoder on a sham pre-event window that carries only drift)
Writes analysis\\event_analysis\\validation\\*.png / *.csv and validation.json.

  python ca_event_validate.py --step all
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_event_analysis as E  # noqa: E402
import pairstars as PS  # noqa: E402  (every star shows the pair it compares)

warnings.filterwarnings("ignore")
VD = os.path.join(E.OD, "validation")
FPS = E.FPS
RULES = ["BH q<=0.20", "max-T 5%", "selective BH", "selective max-T"]
RC = {"BH q<=0.20": "#7f8c8d", "max-T 5%": "#2c3e50", "selective BH": "#e67e22", "selective max-T": "#c0392b"}
MAIN6 = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "movement onset"]


def windows():
    J = json.load(open(os.path.join(E.OD, "summary.json"), encoding="utf-8"))
    return {ev: tuple(J["windows"][ev]["1"]) for ev in E.EVENTS}


def cums(Z):
    ok = np.isfinite(Z)
    Zc = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(np.where(ok, Z, 0), 0)])
    Nc = np.vstack([np.zeros((1, Z.shape[1]), np.int32), np.cumsum(ok, 0, dtype=np.int32)])
    return Zc, Nc


def add_selective(S):
    for s in S.values():
        Z = np.asarray(s["T"]["z"], float)
        s["Zc_sel"], s["Nc_sel"] = cums(Z - np.nanmean(Z, 1, keepdims=True))


def shift_test(s, frames, w, n_shuf=1000, seed=0, sel=False, Zc=None, Nc=None):
    """the within-segment shift test of ca_event_analysis.shift_p, also returning the max-T p (Westfall-Young single step:
    each neuron's |response| / its null SD against the largest such value over all neurons of each shuffle)"""
    fr = s["fr"]
    if Zc is None:
        Zc, Nc = (s["Zc_sel"], s["Nc_sel"]) if sel else (s["Zc"], s["Nc"])
    n = len(fr)
    p0, p1, k0, k1 = E.win_rows(*w)
    lo = int(np.searchsorted(fr, int(E.CA.SKIP_S * FPS))) - p0
    idx, _ = E.ev_rows(s, frames)
    idx = idx[(idx >= lo) & (idx + k1 < n)]
    if len(idx) < E.CA.MIN_TRIALS:
        return None

    def resp(ii):
        ii = ii[(ii >= lo) & (ii + k1 < n)]
        x = (Zc[ii + k1] - Zc[ii + k0]) / np.maximum(Nc[ii + k1] - Nc[ii + k0], 1)
        y = (Zc[ii + p1] - Zc[ii + p0]) / np.maximum(Nc[ii + p1] - Nc[ii + p0], 1)
        return np.nanmean(x - y, 0)
    obs = resp(idx)
    bnd = np.unique(np.searchsorted(fr, np.array([int(x * FPS) for x in E.SEG_S] + [int(fr[-1]) + 1])))
    seg = np.searchsorted(bnd, idx, side="right") - 1
    rng = np.random.default_rng(seed)
    null = np.zeros((n_shuf, Zc.shape[1]))
    ww = max(k1, -p0)
    for k_ in range(n_shuf):
        ii = idx.copy()
        for k in np.unique(seg):
            if k < 0 or k + 1 >= len(bnd):
                continue
            A0, A1 = bnd[k], bnd[k + 1]
            L = A1 - A0
            if L <= 2 * ww:
                continue
            msk = seg == k
            ii[msk] = A0 + (idx[msk] - A0 + int(rng.integers(ww, L - ww))) % L
        null[k_] = resp(np.sort(ii))
    p = (1 + (np.abs(null) >= np.abs(obs)[None, :]).sum(0)) / (1 + n_shuf)
    sd = np.nanstd(null, 0)
    sd = np.where(sd > 0, sd, np.nan)
    tn = np.abs(null / sd[None, :])
    mx = np.nanmax(tn, 1)
    to = np.abs(obs / sd)
    pm = (1 + (mx[:, None] >= to[None, :]).sum(0)) / (1 + n_shuf)
    pm[~np.isfinite(to)] = 1.0
    return dict(obs=obs, p=p, pmax=pm, n=len(idx))


def classes(r, rule):
    """'activated' / 'inhibited' / 'non-responsive' per neuron under one rule"""
    if rule.endswith("max-T") or rule == "max-T 5%":
        sig = r["pmax"] <= 0.05
    else:
        sig = E.bh(r["p"]) <= E.CA.Q_RESP
    return np.where(sig, np.where(r["obs"] > 0, "activated", "inhibited"), "non-responsive")


# ------------------------------------------------------------------------------------------------ null calibration
def step_null(S, W, K=10, n_shuf=500):
    rng = np.random.default_rng(11)
    rows = []
    t0 = time.time()
    for (m, d, ph), s in S.items():
        for ev in E.SEARCH_EV:
            fr = np.asarray(s["B"]["ev"].get(ev, []), int)
            if len(fr) < 10:
                continue
            for k in range(K):
                fake = E.shift_train(s, fr, rng)
                for sel in (False, True):
                    r = shift_test(s, fake, W[ev], n_shuf=n_shuf, seed=k, sel=sel)
                    if r is None:
                        continue
                    nc = len(r["obs"])
                    for rule in (["selective BH", "selective max-T"] if sel else ["BH q<=0.20", "max-T 5%"]):
                        c = classes(r, rule)
                        rows.append(dict(mouse=m, day=d, phase=ph, event=ev, fake=k, rule=rule, n_cells=nc,
                                         n_flag=int((c != "non-responsive").sum())))
        print(f"  null {m} d{d} {ph}: {time.time() - t0:.0f} s", flush=True)
    N = pd.DataFrame(rows)
    N["pct"] = 100 * N.n_flag / N.n_cells
    N.to_csv(os.path.join(VD, "V1_null_calibration.csv"), index=False)
    return N


def fig_null(N, R):
    """A: % of neurons called responsive on FAKE events, per rule; B: share of fake event trains with >= 10 % of the
    neurons called; C: the real % under each rule (all sessions x events)"""
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.9))                 # slide width (10-22)
    out = {}
    for i, rule in enumerate(RULES):
        X = N[N.rule == rule]
        v = X.pct.values
        out[rule] = dict(mean_pct=float(np.mean(v)), frac_any=float(np.mean(X.n_flag > 0)), frac_10=float(np.mean(v >= 10)),
                         q95=float(np.percentile(v, 95)))
        parts = ax[0].violinplot([v], positions=[i], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(RC[rule]); b.set_alpha(0.5)
        ax[0].scatter([i], [np.mean(v)], color=RC[rule], s=40, zorder=3)
        # 10-24 figure review: the annotation was ~7 pt on the slide and '95th' cryptic -> 11 pt, the header line says what
        # the two numbers are (mean and, in brackets, the 95th percentile)
        ax[0].text(i, 0.9, f"{np.mean(v):.1f}%\n({np.percentile(v, 95):.0f}%)", ha="center", va="top", fontsize=11,
                   transform=ax[0].get_xaxis_transform(), linespacing=1.1)
        ax[1].bar(i, 100 * np.mean(v >= 10), color=RC[rule])
        ax[1].text(i, 100 * np.mean(v >= 10), f" {100 * np.mean(v >= 10):.1f}%", ha="center", va="bottom", fontsize=11)
        if R is not None:
            rv = R[(R.rule == rule) & R.event.isin(E.SEARCH_EV)].groupby(["mouse", "day", "phase", "event"]).resp_flag.mean() * 100
            parts = ax[2].violinplot([rv.values], positions=[i], widths=0.8, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(RC[rule]); b.set_alpha(0.5)
            ax[2].scatter([i], [rv.mean()], color=RC[rule], s=40, zorder=3)
            out[rule]["real_mean_pct"] = float(rv.mean())
    for a in ax:
        a.set_xticks(range(4), [r.replace(" ", "\n", 1) for r in RULES], fontsize=11)
    ax[0].set_ylim(-2, 100)
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(0, hi * 1.25)
    ax[0].set_ylabel("% neurons called responsive")
    ax[0].set_title("A  FAKE events (nothing locked to them)", loc="left")
    ax[1].set_ylabel("% of fake event trains")
    ax[1].set_title("B  fake trains with >= 10 %\nof the neurons called", loc="left")
    ax[2].set_ylabel("% neurons responsive")
    if R is not None:
        ax[0].set_ylim(ax[2].get_ylim())                     # 10-22: fake (A) and real (C) on the same scale
    ax[0].text(0.5, 0.985, "mean (95th percentile) of the fake trains", ha="center", va="top", fontsize=11, color="0.25",
               transform=ax[0].transAxes)
    ax[2].set_title("C  REAL events (session x event)", loc="left")
    for a in ax:
        a.tick_params(axis="y", labelsize=11)
    # 10-25 figure review: the dot inside each violin was not defined -> a key under the panels (dot = mean, as plotted above)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(facecolor="0.6", alpha=0.5, label="violin = spread over fake trains (A) / session x event (C)"),
                        Line2D([], [], ls="", marker="o", ms=7, color="0.3", label="dot = mean")],
               loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(VD, "V1_null_calibration.png"), dpi=140)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ spike-in recall
KERNEL_T = np.arange(0, 5, 1 / FPS)


def kernel(amp):
    k = (1 - np.exp(-KERNEL_T / 0.1)) * np.exp(-KERNEL_T / 1.0)
    return amp * k / k.max()


def inject(s, frames, cells, amp, rng, lag_s=0.0):
    """z of the session with a GCaMP-shaped response (trial gain lognormal sd 0.5) added to the given cells at the frames"""
    Z = np.asarray(s["T"]["z"], float).copy()
    idx, _ = E.ev_rows(s, np.asarray(frames, int) + int(round(lag_s * FPS)))
    for c in cells:
        g = rng.lognormal(0, 0.5, len(idx))
        for i, gi in zip(idx, g):
            k = kernel(amp * gi)
            e = min(len(Z), i + len(k))
            Z[i:e, c] += k[:e - i]
    return Z


def step_spikein(S, W):
    rng = np.random.default_rng(5)
    rows = []
    t0 = time.time()
    amps = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0]
    for j, ((m, d, ph), s) in enumerate(S.items()):
        ev = E.STIM[j % 4]
        fr = np.asarray(s["B"]["ev"].get(ev, []), int)
        if len(fr) < 10:
            continue
        nc = s["Zc"].shape[1]
        cells = rng.choice(nc, max(3, nc // 10), replace=False)
        others = np.setdiff1d(np.arange(nc), cells)
        for amp in amps:
            Z = inject(s, fr, cells, amp, rng)
            Zc, Nc = cums(Z)
            Zs = Z - np.nanmean(Z, 1, keepdims=True)
            Zcs, Ncs = cums(Zs)
            for wname, w in (("chosen", W[ev]), ("old 2 s / 0-2 s", E.DEFAULT)):
                for sel, (zc, ncm) in ((False, (Zc, Nc)), (True, (Zcs, Ncs))):
                    r = shift_test(s, fr, w, n_shuf=500, seed=j, Zc=zc, Nc=ncm)
                    if r is None:
                        continue
                    for rule in (["selective BH", "selective max-T"] if sel else ["BH q<=0.20", "max-T 5%"]):
                        c = classes(r, rule)
                        want = "activated" if amp >= 0 else "inhibited"
                        rows.append(dict(mouse=m, day=d, phase=ph, event=ev, amp=amp, window=wname, rule=rule,
                                         recall=float(np.mean(c[cells] == want)), fp=float(np.mean(c[others] != "non-responsive"))))
        print(f"  spike-in {m} d{d} {ph}: {time.time() - t0:.0f} s", flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(VD, "V2_spikein.csv"), index=False)
    return P


def fig_spikein(P):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.6))                 # slide width (10-22)
    out = {}
    for k, (wname, ls) in enumerate((("chosen", "-"), ("old 2 s / 0-2 s", "--"))):
        for rule in RULES:
            X = P[(P.window == wname) & (P.rule == rule)]
            g = X.groupby("amp").recall.mean() * 100
            pos = g[g.index > 0]
            g = g[g.index != 0]
            ax[0].plot(pos.index, pos.values, ls, marker="o", color=RC[rule], lw=2, label=f"{rule}, {wname}" if True else None)
            neg = g[g.index < 0]
            ax[1].plot(-neg.index, neg.values, ls, marker="o", color=RC[rule], lw=2)
            out[f"{rule} | {wname}"] = {f"{a:+g}": float(v) for a, v in g.items()}
    for rule in RULES:
        X = P[(P.window == "chosen") & (P.rule == rule)]
        g = X.groupby("amp").fp.mean() * 100
        base = g.get(0.0, np.nan)
        ax[2].plot(g.index, g.values - base, "-o", color=RC[rule], lw=2, label=rule)
        out[f"extra calls | {rule}"] = {f"{a:+g}": float(v - base) for a, v in g.items()}
        out[f"calls without added | {rule}"] = float(base)
    ax[2].axhline(0, color="0.5", lw=0.8)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
    ax[1].set_xticks([0.5, 1], ["-0.5", "-1"])               # 10-25: drawn at |size|, labelled with the signed size
    ax[0].set_xlabel("added response, peak z")
    ax[1].set_xlabel("added DECREASE, peak z (negative)")
    ax[1].set_ylabel("% of the added neurons found")
    ax[0].set_ylim(0, 102); ax[1].set_ylim(0, 102)
    ax[0].set_ylabel("% of the added neurons found")
    ax[0].set_title("A  activation found (solid = chosen window)", loc="left")
    ax[1].set_title("B  decrease found", loc="left")
    ax[2].set_xlabel("added response, peak z\n(added to 10 % of the neurons)")
    ax[2].set_ylabel("extra % of the OTHER neurons\ncalled (vs nothing added)")
    ax[2].set_title("C  side effect on the other neurons\n(chosen window)", loc="left")
    ax[0].set_title("A  activation found", loc="left")
    # one legend for the three panels, outside them (10-22: the 8-entry legend inside A covered the lines)
    hl = [Line2D([], [], color=RC[r], lw=2.5, marker="o", label=r) for r in RULES] + \
         [Line2D([], [], color="0.3", lw=2, ls="-", label="chosen window"), Line2D([], [], color="0.3", lw=2, ls="--", label="old 2 s / 0-2 s window")]
    fig.legend(handles=hl, loc="lower center", ncol=6, frameon=False, fontsize=11, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(VD, "V2_spikein.png"), dpi=140)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ GLM separation
def step_glm(S, n_null=60):
    rng = np.random.default_rng(9)
    rows = []
    t0 = time.time()
    for (m, d, ph), s in S.items():
        ev = s["B"]["ev"]
        dl = E.deliveries(s)
        wd = np.asarray(ev["withdrawal"], int)
        nc = s["Zc"].shape[1]
        perm = rng.permutation(nc)
        k = max(3, int(0.15 * nc))
        cs, cw = perm[:k], perm[k:2 * k]
        Z = inject(s, dl, cs, 1.0, rng)
        s2 = dict(s)
        s2["T"] = dict(s["T"])
        s2["T"]["z"] = Z
        Z2 = inject(s2, wd, cw, 1.0, rng)
        s2["T"]["z"] = Z2
        g = E.glm_session(s2, n_null=n_null, seed=3)
        qs = {fam: E.bh(g[f"p_{fam}"]) for fam in E.FAMILIES}
        sig = {fam: (qs[fam] <= E.CA.Q_RESP) & (g[f"uv_{fam}"] > 0) for fam in E.FAMILIES}
        for j in range(nc):
            r = {f"sig_{fam}": bool(sig[fam][j]) for fam in E.FAMILIES}
            truth = "stimulus" if j in cs else ("withdrawal" if j in cw else "none added")
            rows.append(dict(mouse=m, day=d, phase=ph, cell=j, truth=truth, type=E.neuron_type(r)))
        print(f"  glm {m} d{d} {ph}: {time.time() - t0:.0f} s", flush=True)
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(VD, "V3_glm_separation.csv"), index=False)
    return G


def fig_glm(G):
    """10-24 figure review: labels, numbers and legend were ~7 pt on the slide, thin segments unlabelled, the lower 30 % of the
    slide empty -> drawn at the slide box shape (12.4 x 5.2 in) with 11.5-12 pt text; thin segments get their value outside
    the bar (E.stack_barh)"""
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    out = {}
    for k, d in enumerate((1, 2)):
        a = ax[k]
        X = G[G.day == d]
        rows_ = ["stimulus", "withdrawal", "none added"]
        cnt = X.groupby(["truth", "type"]).size().unstack(fill_value=0).reindex(rows_).fillna(0)
        E.stack_barh(a, cnt, E.TYPES, E.TYPC, fs=12, thin=7, h=0.6)
        a.set_yticks(range(len(cnt)), ["response added\nat the DELIVERIES", "response added\nat the WITHDRAWALS", "nothing added"],
                     fontsize=12)
        a.set_xlim(0, 100)
        a.set_xlabel("% of these neurons", fontsize=12)
        a.tick_params(axis="x", labelsize=11.5)
        _lat = json.load(open(os.path.join(E.OD, "summary.json"), encoding="utf-8")).get("timing", {}).get("wd_latency_median", {})
        _l = _lat.get(str(d), float("nan"))
        # 10-25 figure review: '+0.97 s' was unexplained -> it is the median lag from a delivery mark to the withdrawal
        a.set_title(f"Day {d}: {'key marks' if d == 1 else 'clicker'}\nmedian delivery-to-withdrawal lag +{_l:.2f} s", loc="left",
                    fontsize=13)
        out[f"day{d}"] = {r_: {t: float(100 * cnt.loc[r_, t] / cnt.loc[r_].sum()) for t in cnt.columns} for r_ in cnt.index}
    fig.legend(handles=[Patch(color=E.TYPC[t], label=t) for t in E.TYPES], loc="lower center", ncol=3, frameon=False, fontsize=12,
               columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.5)
    fig.savefig(os.path.join(VD, "V3_glm_separation.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ robust classes
def step_robust(S, W):
    rows = []
    t0 = time.time()
    for (m, d, ph), s in S.items():
        for ev in E.EVENTS:
            fr = s["B"]["ev"].get(ev, [])
            for sel in (False, True):
                r = shift_test(s, fr, W[ev], n_shuf=1000, seed=0, sel=sel)
                if r is None:
                    continue
                for rule in (["selective BH", "selective max-T"] if sel else ["BH q<=0.20", "max-T 5%"]):
                    c = classes(r, rule)
                    for j, cid in enumerate(s["ids"]):
                        rows.append(dict(mouse=m, day=d, phase=ph, condition=E.cond_of(m, d, ph), event=ev, cell=int(cid),
                                         rule=rule, cls=c[j], resp=float(r["obs"][j])))
        print(f"  robust {m} d{d} {ph}: {time.time() - t0:.0f} s", flush=True)
    R = pd.DataFrame(rows)
    R["resp_flag"] = R.cls != "non-responsive"
    R.to_csv(os.path.join(VD, "A1_classes_all_rules.csv"), index=False)
    return R


def fig_robust(R):
    """A: % responsive per session x event under each rule (rows = sessions in time order); B: per condition, the 6 main
    events, each rule (mean of the 4 mice), with the within-day McNemar change"""
    ses = [(m, d, ph) for m in E.MO for d in (1, 2) for ph in ("pre", "post")]
    evs = MAIN6
    fig = plt.figure(figsize=(18, 9.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)
    out = {}
    for k, rule in enumerate(RULES):
        a = fig.add_subplot(gs[0, k])
        X = R[R.rule == rule]
        P = X.groupby(["mouse", "day", "phase", "event"]).resp_flag.mean().unstack("event").reindex(columns=evs) * 100
        M = np.array([[P.loc[s_, e] if s_ in P.index else np.nan for e in evs] for s_ in ses])
        im = a.imshow(M, cmap="magma", vmin=0, vmax=70, aspect="auto")
        a.set_xticks(range(len(evs)), [E.SHORT[e] for e in evs], rotation=45, ha="right", fontsize=9.5)
        lab = [f"{m} D{d} {'base' if ph == 'pre' else ('SBI' if E.cond_of(m, d, ph) == 'SBI-553' else 'veh')}" for m, d, ph in ses]
        a.set_yticks(range(len(ses)), lab if k == 0 else [""] * len(ses), fontsize=8.5)
        a.set_title(rule, fontsize=13, color=RC[rule], fontweight="bold")
        for i in range(len(ses)):
            for j in range(len(evs)):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="w" if M[i, j] < 40 else "k")
        sd_ = np.nanstd(M, 0)
        out[rule] = dict(mean=float(np.nanmean(M)), sd_between_sessions=float(np.nanmean(sd_)))
    cax = fig.add_axes([0.92, 0.55, 0.008, 0.3])
    fig.colorbar(im, cax=cax).set_label("% responsive")
    b = fig.add_subplot(gs[1, :])
    xs = np.arange(len(evs))
    for k, rule in enumerate(RULES):
        X = R[R.rule == rule]
        for j, c in enumerate(E.COND):
            v = [100 * X[(X.event == e) & (X.condition == c)].resp_flag.mean() for e in evs]
            b.bar(xs + (k - 1.5) * 0.21 + (j - 1) * 0.065, v, width=0.062, color=E.CC[c], alpha=0.35 + 0.15 * k,
                  edgecolor=RC[rule], lw=0.8)
    b.set_xticks(xs, [E.SHORT[e] for e in evs])
    b.set_ylabel("% responsive (all neurons pooled)")
    b.set_title("per condition: four groups of 3 bars per event = the four rules (left to right as above); colours = baseline / vehicle / SBI-553",
                loc="left", fontsize=11.5)
    fig.savefig(os.path.join(VD, "A1_rules_compared.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ functional clusters
def psth_features(S, evs=MAIN6, t0=-1.0, t1=3.0, step=0.2):
    """per neuron-session: the trial-averaged z (-1..3 s, 0.2-s bins, baselined on -1..0 s) of every main event"""
    nb = int(round((t1 - t0) / step))
    rows, F = [], []
    for (m, d, ph), s in S.items():
        blocks = []
        for ev in evs:
            X, tt, used = E.CA.peri(s, s["B"]["ev"].get(ev, []))
            if X is None or len(used) < E.CA.MIN_TRIALS:
                blocks.append(np.full((nb, len(s["ids"])), np.nan))
                continue
            Xb = np.nanmean(X - np.nanmean(X[:, (tt >= -1) & (tt < 0), :], 1, keepdims=True), 0)
            B_ = np.full((nb, Xb.shape[1]), np.nan)
            for i in range(nb):
                r = (tt >= t0 + i * step) & (tt < t0 + (i + 1) * step)
                B_[i] = np.nanmean(Xb[r], 0)
            blocks.append(B_)
        A = np.vstack(blocks).T
        F.append(A)
        for c in s["ids"]:
            rows.append(dict(mouse=m, day=d, phase=ph, condition=E.cond_of(m, d, ph), cell=int(c)))
    return pd.DataFrame(rows), np.vstack(F), nb


def step_clusters(S):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    meta, F, nb = psth_features(S)
    Fz = np.nan_to_num(F, nan=0.0)
    base = (meta.phase == "pre").values
    pca = PCA(n_components=10, random_state=0).fit(Fz[base])
    Y = pca.transform(Fz)
    best = None
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Y[base])
        sc = silhouette_score(Y[base], km.labels_)
        if best is None or sc > best[0]:
            best = (sc, k, km)
    sc, k, km = best
    lab = km.predict(Y)
    # order the clusters by their mean response to movement onset (descending) so that the numbering means something
    mv = MAIN6.index("movement onset")
    resp_mv = [np.nanmean(F[lab == c][:, mv * nb + 5:mv * nb + 10]) for c in range(k)]
    order = np.argsort(resp_mv)[::-1]
    remap = {int(o): i for i, o in enumerate(order)}
    meta["cluster"] = [remap[int(x)] for x in lab]
    meta.to_csv(os.path.join(VD, "A2_clusters.csv"), index=False)
    return meta, F, nb, k, float(sc), pca.explained_variance_ratio_[:3].sum()


def fig_clusters(meta, F, nb, k):
    tt = -1 + 0.2 * (np.arange(nb) + 0.5)
    fig = plt.figure(figsize=(18, 2.0 * k + 3.2))
    gs = fig.add_gridspec(k + 1, len(MAIN6) + 2, height_ratios=[1] * k + [1.4], width_ratios=[1] * len(MAIN6) + [0.15, 1.6],
                          hspace=0.45, wspace=0.25)
    cols = plt.get_cmap("tab10")
    out = {}
    base = meta.phase.values == "pre"
    for c in range(k):
        sel = (meta.cluster.values == c) & base
        for j, ev in enumerate(MAIN6):
            a = fig.add_subplot(gs[c, j])
            Y = F[sel][:, j * nb:(j + 1) * nb]
            mu = np.nanmean(Y, 0)
            se = np.nanstd(Y, 0) / np.sqrt(max(1, np.isfinite(Y).any(1).sum()))
            a.plot(tt, mu, color=cols(c), lw=2)
            a.fill_between(tt, mu - se, mu + se, color=cols(c), alpha=0.25, lw=0)
            a.axvline(0, color="0.5", lw=0.7, ls="--")
            a.axhline(0, color="0.7", lw=0.6)
            a.set_ylim(-0.6, 1.0)
            if c == 0:
                a.set_title(E.SHORT[ev], fontsize=12)
            if j == 0:
                a.set_ylabel(f"type {c + 1}\n(n = {sel.sum()})", fontsize=10.5, color=cols(c))
            else:
                a.set_yticklabels([])
            if c < k - 1:
                a.set_xticklabels([])
    # fractions per condition
    a = fig.add_subplot(gs[:k, -1])
    fr = meta.groupby(["condition", "cluster"]).size().unstack(fill_value=0).reindex(E.COND)
    left = np.zeros(len(fr))
    for c in range(k):
        v = 100 * fr[c].values / fr.sum(1).values if c in fr else np.zeros(len(fr))
        a.barh(range(len(fr)), v, left=left, color=cols(c))
        for i, (l_, w_) in enumerate(zip(left, v)):
            if w_ >= 6:
                a.text(l_ + w_ / 2, i, f"{w_:.0f}", ha="center", va="center", fontsize=10, color="w")
        left += v
        out[f"type{c + 1}"] = {cn: float(100 * fr.loc[cn, c] / fr.loc[cn].sum()) for cn in fr.index}
    a.set_yticks(range(len(fr)), fr.index)
    a.invert_yaxis()
    a.set_xlabel("% of neuron-sessions")
    a.set_title("types per condition", loc="left", fontsize=12)
    # transitions of the same neurons baseline -> injection (same day), per drug
    b = fig.add_subplot(gs[k, :len(MAIN6)])
    M_ = meta.pivot_table(index=["mouse", "day", "cell"], columns="phase", values="cluster").dropna()
    M_ = M_.reset_index()
    M_["drug"] = [E.drug_of(m, d) for m, d in zip(M_.mouse, M_.day)]
    stay = {}
    xs = np.arange(k)
    for j_, drug in enumerate(("vehicle", "SBI-553")):
        Z = M_[M_.drug == drug]
        v = [100 * np.mean(Z[Z.pre == c].post == c) if (Z.pre == c).any() else np.nan for c in range(k)]
        stay[drug] = float(100 * np.mean(Z.pre == Z.post)) if len(Z) else np.nan
        b.bar(xs + (j_ - 0.5) * 0.38, v, width=0.36, color=E.CC[drug], label=f"{drug} (all: {stay[drug]:.0f}% stay)")
    b.set_xticks(xs, [f"type {c + 1}" for c in range(k)])
    b.set_ylabel("% staying in the type\nafter the injection")
    b.legend(frameon=False, fontsize=10)
    b.set_title("the same neurons, baseline -> injection of the same day", loc="left", fontsize=12)
    out["stay"] = stay
    fig.savefig(os.path.join(VD, "A2_functional_types.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ decoding
def step_decode(S, W):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    rows = []
    w = W["Heat"]
    sham = (0.5, 0.0, -1.0, -0.5)                              # -1..-0.5 s vs the 0.5 s before: drift only, no response
    for (m, d, ph), s in S.items():
        ev = s["B"]["ev"]
        X, y, Xs = [], [], []
        for k_, st in enumerate(E.STIM):
            f = np.asarray(ev.get(st, []), int)
            R, ff = E.responses(s, f, *w)
            Rs, _ = E.responses(s, ff, *sham)
            if len(R) != len(Rs):
                n_ = min(len(R), len(Rs))
                R, Rs = R[:n_], Rs[:n_]
            X.append(R); Xs.append(Rs); y += [k_] * len(R)
        X, Xs, y = np.nan_to_num(np.vstack(X)), np.nan_to_num(np.vstack(Xs)), np.array(y)
        if len(np.unique(y)) < 4 or np.bincount(y).min() < 5:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
        accs, accs_s = [], []
        for rep in range(5):
            cv = StratifiedKFold(5, shuffle=True, random_state=rep)
            accs.append(balanced_accuracy_score(y, cross_val_predict(clf, X, y, cv=cv)))
            accs_s.append(balanced_accuracy_score(y, cross_val_predict(clf, Xs, y, cv=cv)))
        # withdrawal yes / no inside each stimulus type (pooled over the types with enough of both)
        wd = np.asarray(ev["withdrawal"], int)
        Xw, yw = [], []
        dl = E.deliveries(s)
        for st in E.STIM:
            f = np.asarray(ev.get(st, []), int)
            R, ff = E.responses(s, f, *w)
            has = []
            for x in ff:
                nx = dl[dl > x + 3]
                hi = min(x + int(2 * FPS), nx[0] - 1) if len(nx) else x + int(2 * FPS)
                has.append(bool(np.any((wd >= x) & (wd <= hi))))
            has = np.array(has)
            if has.sum() >= 4 and (~has).sum() >= 4:
                Xw.append(R - R.mean(0)); yw += list(has.astype(int))
        aw = np.nan
        if Xw:
            Xw, yw = np.nan_to_num(np.vstack(Xw)), np.array(yw)
            if min(np.bincount(yw)) >= 5:
                aws = []
                for rep in range(5):
                    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
                    aws.append(balanced_accuracy_score(yw, cross_val_predict(clf, Xw, yw, cv=cv)))
                aw = float(np.mean(aws))
        rows.append(dict(mouse=m, day=d, phase=ph, condition=E.cond_of(m, d, ph), n_trials=len(y), n_cells=X.shape[1],
                         stim_acc=float(np.mean(accs)), stim_acc_sham=float(np.mean(accs_s)), wd_acc=aw))
        print(f"  decode {m} d{d} {ph}: stimulus {np.mean(accs):.2f} (sham {np.mean(accs_s):.2f}), withdrawal {aw:.2f}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(VD, "A3_decoding.csv"), index=False)
    return D


def fig_decode(D):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    out = {}
    a = ax[0]
    for i, c in enumerate(E.COND):
        X = D[D.condition == c]
        a.scatter(np.full(len(X), i - 0.12) + np.random.default_rng(i).uniform(-0.04, 0.04, len(X)), 100 * X.stim_acc, color=E.CC[c], s=40)
        a.scatter(np.full(len(X), i + 0.12) + np.random.default_rng(i + 9).uniform(-0.04, 0.04, len(X)), 100 * X.stim_acc_sham,
                  color=E.CC[c], s=40, marker="x")
        out[c] = dict(stim=float(100 * X.stim_acc.mean()), sham=float(100 * X.stim_acc_sham.mean()), wd=float(100 * X.wd_acc.mean()))
    a.axhline(25, color="0.5", ls="--", lw=1)
    a.set_xticks(range(3), E.COND)
    a.set_ylabel("balanced accuracy (%)")
    a.set_title("A  which of the 4 stimuli? (dot = real, x = sham window)", loc="left", fontsize=11.5)
    b = ax[1]
    for (m, d), X in D.groupby(["mouse", "day"]):
        if len(X) == 2:
            drug = E.drug_of(m, d)
            y = [100 * (X[X.phase == "pre"].stim_acc.iloc[0] - X[X.phase == "pre"].stim_acc_sham.iloc[0]),
                 100 * (X[X.phase == "post"].stim_acc.iloc[0] - X[X.phase == "post"].stim_acc_sham.iloc[0])]
            b.plot([0, 1], y, "-o", color=E.CC[drug], marker=E.MK[m], lw=1.8, ms=8)
    b.axhline(0, color="0.5", lw=0.8)
    b.set_xticks([0, 1], ["baseline", "injection"])
    b.set_ylabel("stimulus decoding above the sham (%)")
    b.set_title("B  same day, baseline -> injection (blue veh, red SBI)", loc="left", fontsize=11.5)
    c = ax[2]
    for (m, d), X in D.groupby(["mouse", "day"]):
        if len(X) == 2 and X.wd_acc.notna().all():
            drug = E.drug_of(m, d)
            c.plot([0, 1], [100 * X[X.phase == "pre"].wd_acc.iloc[0], 100 * X[X.phase == "post"].wd_acc.iloc[0]], "-o", color=E.CC[drug],
                   marker=E.MK[m], lw=1.8, ms=8)
    c.axhline(50, color="0.5", ls="--", lw=1)
    c.set_xticks([0, 1], ["baseline", "injection"])
    c.set_ylabel("balanced accuracy (%)")
    c.set_title("C  will a withdrawal follow? (same stimulus)", loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(VD, "A3_decoding.png"), dpi=140)
    plt.close(fig)
    return out


def step_clusters_responsive(S, R):
    """functional types among the RESPONSIVE neurons only (responsive to >= 1 of the 6 main events under max-T, any session),
    each neuron's profile scaled by its own largest |response| (shape, not size); k by silhouette 3..8"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    meta, F, nb = psth_features(S)
    X = R[(R.rule == "max-T 5%") & R.event.isin(MAIN6)]
    flag = X.groupby(["mouse", "day", "phase", "cell"]).resp_flag.any()
    key = list(zip(meta.mouse, meta.day, meta.phase, meta.cell))
    resp = np.array([bool(flag.get(k_, False)) for k_ in key])
    Fz = np.nan_to_num(F, nan=0.0)
    sc_ = np.maximum(np.abs(Fz).max(1, keepdims=True), 1e-6)
    Fn = Fz / sc_
    base = (meta.phase == "pre").values & resp
    if base.sum() < 20:
        return None
    pca = PCA(n_components=min(8, base.sum() - 1), random_state=0).fit(Fn[base])
    Y = pca.transform(Fn)
    best = None
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=30, random_state=0).fit(Y[base])
        sc = silhouette_score(Y[base], km.labels_)
        if best is None or sc > best[0]:
            best = (sc, k, km)
    sc, k, km = best
    lab = np.where(resp, km.predict(Y), -1)
    mv = MAIN6.index("movement onset")
    order = np.argsort([-np.nanmean(Fn[lab == c][:, mv * nb + 5:mv * nb + 10]) if (lab == c).any() else 0 for c in range(k)])
    remap = {int(o): i for i, o in enumerate(order)}
    meta["rtype"] = [remap[int(x)] if x >= 0 else -1 for x in lab]
    meta.to_csv(os.path.join(VD, "A2b_types_responsive.csv"), index=False)
    return meta, Fn, nb, k, float(sc)


def fig_clusters_responsive(meta, Fn, nb, k):
    tt = -1 + 0.2 * (np.arange(nb) + 0.5)
    cols = plt.get_cmap("tab10")
    fig = plt.figure(figsize=(18, 1.9 * k + 1.2))
    gs = fig.add_gridspec(k, len(MAIN6) + 2, width_ratios=[1] * len(MAIN6) + [0.12, 1.5], hspace=0.45, wspace=0.25)
    base = meta.phase.values == "pre"
    out = {}
    for c in range(k):
        sel = (meta.rtype.values == c) & base
        for j, ev in enumerate(MAIN6):
            a = fig.add_subplot(gs[c, j])
            Y = Fn[sel][:, j * nb:(j + 1) * nb]
            mu = np.nanmean(Y, 0)
            se = np.nanstd(Y, 0) / np.sqrt(max(1, len(Y)))
            a.plot(tt, mu, color=cols(c), lw=2)
            a.fill_between(tt, mu - se, mu + se, color=cols(c), alpha=0.25, lw=0)
            a.axvline(0, color="0.5", lw=0.7, ls="--")
            a.axhline(0, color="0.7", lw=0.6)
            a.set_ylim(-0.8, 0.8)
            if c == 0:
                a.set_title(E.SHORT[ev], fontsize=12)
            if j == 0:
                a.set_ylabel(f"type {c + 1}\n(n = {sel.sum()})", fontsize=10.5, color=cols(c))
            else:
                a.set_yticklabels([])
            if c < k - 1:
                a.set_xticklabels([])
    a = fig.add_subplot(gs[:, -1])
    M_ = meta[meta.rtype >= 0]
    fr = M_.groupby(["condition", "rtype"]).size().unstack(fill_value=0).reindex(E.COND).fillna(0)
    left = np.zeros(len(fr))
    for c in range(k):
        v = 100 * fr[c].values / np.maximum(fr.sum(1).values, 1) if c in fr else np.zeros(len(fr))
        a.barh(range(len(fr)), v, left=left, color=cols(c))
        for i, (l_, w_) in enumerate(zip(left, v)):
            if w_ >= 7:
                a.text(l_ + w_ / 2, i, f"{w_:.0f}", ha="center", va="center", fontsize=10, color="w")
        left += v
        out[f"type{c + 1}"] = {cn: float(100 * fr.loc[cn, c] / max(fr.loc[cn].sum(), 1)) for cn in fr.index}
    out["mean_resp"] = {f"type{c + 1}": float(np.nanmean(Fn[(meta.rtype.values == c) & base])) for c in range(k)}
    a.set_yticks(range(len(fr)), [f"{cn}\n(n = {int(fr.loc[cn].sum())})" for cn in fr.index])
    a.invert_yaxis()
    a.set_xlabel("% of the responsive neuron-sessions")
    a.set_title("types per condition", loc="left", fontsize=12)
    fig.savefig(os.path.join(VD, "A2b_types_responsive.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out


def step_pseudopop(S, W, n_neur=100, n_rep=100, n_train=8, n_test=3):
    """pseudo-population decoding of the 4 stimuli per condition: every neuron of every session of that condition is a
    column; a pseudo-trial takes, for each neuron, one random trial of the asked stimulus from its own session (trials split
    into train / test first); n_neur neurons drawn per repetition so the conditions are compared at the same size.
    The same with the sham window (drift only)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(21)
    w = W["Heat"]
    sham = (0.5, 0.0, -1.0, -0.5)
    res = []
    for cond in E.COND:
        for wname, ww in (("real", w), ("sham", sham)):
            cols = []                                         # per neuron: list of 4 arrays (trials) of its response
            for (m, d, ph), s in S.items():
                if E.cond_of(m, d, ph) != cond:
                    continue
                per = []
                for st in E.STIM:
                    Rr, _ = E.responses(s, s["B"]["ev"].get(st, []), *ww)
                    per.append(Rr)
                if min(len(x) for x in per) < n_train + n_test:
                    continue
                for j in range(per[0].shape[1]):
                    cols.append([np.nan_to_num(x[:, j]) for x in per])
            if len(cols) < n_neur:
                continue
            accs = []
            for rep in range(n_rep):
                pick = rng.choice(len(cols), n_neur, replace=False)
                Xtr, ytr, Xte, yte = [], [], [], []
                for k_ in range(4):
                    tr_rows, te_rows = [], []
                    for c in pick:
                        tri = cols[c][k_]
                        o = rng.permutation(len(tri))
                        tr_rows.append(tri[o[:n_train]])
                        te_rows.append(tri[o[n_train:n_train + n_test]])
                    # pseudo-trials: n_train (n_test) rows, each column from its own neuron's trials
                    Xtr.append(np.array(tr_rows).T); ytr += [k_] * n_train
                    Xte.append(np.array(te_rows).T); yte += [k_] * n_test
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
                clf.fit(np.vstack(Xtr), ytr)
                accs.append(float(np.mean(clf.predict(np.vstack(Xte)) == np.array(yte))))
            res.append(dict(condition=cond, window=wname, n_neurons_available=len(cols), acc=float(np.mean(accs)),
                            lo=float(np.percentile(accs, 2.5)), hi=float(np.percentile(accs, 97.5))))
            print(f"  pseudo-population {cond} {wname}: {np.mean(accs):.2f} ({len(cols)} neurons)", flush=True)
    D = pd.DataFrame(res)
    D.to_csv(os.path.join(VD, "A3b_pseudopop_decoding.csv"), index=False)
    return D


def fig_pseudopop(D):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    out = {}
    for i, c in enumerate(E.COND):
        for j, (wname, mk) in enumerate((("real", "o"), ("sham", "x"))):
            X = D[(D.condition == c) & (D.window == wname)]
            if not len(X):
                continue
            r = X.iloc[0]
            x = i + (j - 0.5) * 0.25
            ax.errorbar(x, 100 * r.acc, yerr=[[100 * (r.acc - r.lo)], [100 * (r.hi - r.acc)]], fmt=mk, color=E.CC[c], ms=9, capsize=4,
                        lw=1.8)
            out[f"{c} | {wname}"] = float(100 * r.acc)
    ax.axhline(25, color="0.5", ls="--", lw=1)
    ax.set_xticks(range(3), E.COND)
    ax.set_ylabel("accuracy, 4 stimuli (%)")
    ax.set_title("pseudo-population (100 neurons pooled over mice): o = response window, x = sham window", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(VD, "A3b_pseudopop_decoding.png"), dpi=140)
    plt.close(fig)
    return out


def step_key(R):
    """check 4: do the key findings hold in every mouse and with the strict test? Neural: % responsive change baseline ->
    injection (same day, same neurons) per mouse for pin prick and movement onset, BH vs max-T. Behaviour: % of deliveries
    per mouse, vehicle vs SBI-553 injection session, for the three behaviour differences of the deck."""
    from scipy.stats import binomtest as _bt
    rows = []
    for rule in ("BH q<=0.20", "max-T 5%"):
        for ev in ("Pin prick", "Heat", "movement onset"):
            X = R[(R.rule == rule) & (R.event == ev)]
            for drug in ("vehicle", "SBI-553"):
                pre = X[(X.phase == "pre") & np.array([E.drug_of(m, d) == drug for m, d in zip(X.mouse, X.day)], bool)]
                post = X[(X.phase == "post") & (X.condition == drug)]
                M = pre.merge(post, on=["mouse", "day", "cell"], suffixes=("_b", "_i"))
                b, i = M.resp_flag_b.values, M.resp_flag_i.values
                lost, gained = int((b & ~i).sum()), int((~b & i).sum())
                p_ = _bt(lost, lost + gained).pvalue if lost + gained else np.nan
                for m, Z in M.groupby("mouse"):
                    rows.append(dict(kind="neural", rule=rule, event=ev, drug=drug, mouse=m, base=100 * Z.resp_flag_b.mean(),
                                     inj=100 * Z.resp_flag_i.mean(), p_pooled=p_))
    D = pd.read_csv(os.path.join(E.OD, "P6_per_delivery.csv"))
    D["mouse"] = D.mouse.astype(str)
    for beh, st in (("flinch", "Pin prick"), ("movement", "Light touch"), ("escape/rearing", "Heat")):
        T = D[D.stim == st].groupby(["mouse", "condition"])[beh].mean().unstack() * 100
        for m in T.index:
            rows.append(dict(kind="behaviour", rule="", event=f"{beh} / {st}", drug="", mouse=m,
                             base=float(T.loc[m, "vehicle"]), inj=float(T.loc[m, "SBI-553"]), p_pooled=np.nan))
    K = pd.DataFrame(rows)
    K.to_csv(os.path.join(VD, "V4_key_findings_per_mouse.csv"), index=False)
    return K


def fig_key(K):
    """10-22 review: the per-mouse text labels were 7 pt and overlapped - each mouse is now a marker shape (legend under the
    first panel of each half), the star header is three short lines, every font >= 10 pt"""
    from matplotlib.lines import Line2D as _L2
    # 10-24 figure review: the stats labels collided with each other and with the '100' tick -> above the axes, one header
    # line ('baseline vs injection, same day') + BH / max-T per day; x ticks wrapped; y label in percentage points; the
    # filled / open key in the mouse legend; a spacer column between the halves (crop_split cuts there: 1/2 neurons, 2/2 behaviour)
    evs = ["Pin prick", "Heat", "movement onset"]
    behs = [e for e in K[K.kind == "behaviour"].event.unique()]
    mice = [m for m in ("5941", "5939", "5937", "1191") if m in set(K.mouse.astype(str))]
    # 10-25 figure review (slide 176): each half (crop_split cuts at the spacer) was taller than the slide box, so its panels
    # filled only the middle of the slide and the 4-line titles were ~8 pt -> a wider, lower figure (each half ~11 x 5.8 in,
    # close to the 12.4 x 5.2 in box), 2-line titles, 12-13 pt ticks / labels / titles (>= 10 pt on the slide)
    fig = plt.figure(figsize=(22, 5.8))
    # two grids: the neural half needs room above its axes for the star header; the behaviour half has none -> taller axes.
    # The white gap between them (0.48-0.55) is where crop_split cuts.
    gsl = fig.add_gridspec(1, len(evs), wspace=0.5, left=0.045, right=0.48, top=0.72, bottom=0.27)
    gsr = fig.add_gridspec(1, len(behs), wspace=0.5, left=0.585, right=0.99, top=0.84, bottom=0.27)
    ax = [fig.add_subplot(gsl[0, j]) for j in range(len(evs))] + [fig.add_subplot(gsr[0, j]) for j in range(len(behs))]
    out = {}
    for k, ev in enumerate(evs):
        a = ax[k]
        lab_ = {}
        for j, rule in enumerate(("BH q<=0.20", "max-T 5%")):
            X = K[(K.kind == "neural") & (K.event == ev) & (K.rule == rule)]
            for drug, x0 in (("vehicle", 0), ("SBI-553", 1)):
                Z = X[X.drug == drug]
                d = (Z.inj - Z.base).values
                xs = x0 + (j - 0.5) * 0.36 + np.linspace(-0.09, 0.09, len(d))
                for xx, dd, m in zip(xs, d, Z.mouse.astype(str)):
                    a.scatter([xx], [dd], s=64, marker=E.MK.get(m, "o"), color=E.CC[drug],
                              facecolors=E.CC[drug] if j == 0 else "none", lw=1.6, zorder=3, clip_on=False)
                pp = Z.p_pooled.iloc[0] if len(Z) else np.nan
                out[f"{ev} | {rule} | {drug}"] = dict(per_mouse={m: float(v) for m, v in zip(Z.mouse, d)}, p=float(pp),
                                                      n_up=int((d > 0).sum()), n=int(len(d)))
                lab_.setdefault(drug, []).append(PS.star(pp))
        a.set_xlim(-0.45, 1.45)
        for drug, x0 in (("vehicle", 0), ("SBI-553", 1)):       # the pair of every star: that day's baseline vs injection
            if drug in lab_:
                st_ = lab_[drug] + ["n.a."] * (2 - len(lab_[drug]))
                a.text(x0, 1.02, f"BH {st_[0]}\nmax-T {st_[1]}", ha="center", va="bottom", fontsize=11,
                       color=E.CC[drug], fontweight="bold", transform=a.get_xaxis_transform(), linespacing=1.1, clip_on=False)
        a.text(0.5, 1.2, "baseline vs injection, same day:", ha="center", va="bottom", fontsize=10.5, color="0.3",
               transform=a.transAxes)
        a.axhline(0, color="0.5", lw=0.8)
        a.set_xticks([0, 1], ["vehicle\nday", "SBI-553\nday"], fontsize=12)
        a.tick_params(axis="y", labelsize=12)
        a.set_title(f"{E.SHORT[ev]}: % responsive change", fontsize=13, pad=58)
        a.set_ylabel("change in % responsive\n(percentage points)", fontsize=12)
        if k == 1:                                             # under the middle panel of the half: mice + filled / open
            hs = [_L2([], [], ls="", marker=E.MK.get(m, "o"), color="0.3", ms=8, label=m) for m in mice]
            hs += [_L2([], [], ls="", marker="o", color="0.3", ms=8, label="filled = BH q <= 0.20"),
                   _L2([], [], ls="", marker="o", mfc="none", mec="0.3", mew=1.6, ms=8, label="open = max-T 5%")]
            a.legend(handles=hs, title="mouse (marker shape)", fontsize=11.5, title_fontsize=11.5, ncol=len(hs), loc="upper center",
                     bbox_to_anchor=(0.5, -0.3), frameon=False, handletextpad=0.25, columnspacing=1.0)
    for k, beh in enumerate(behs):
        a = ax[len(evs) + k]
        X = K[(K.kind == "behaviour") & (K.event == beh)]
        for _, r in X.iterrows():
            m = str(r.mouse)
            a.plot([0, 1], [r.base, r.inj], "-", color=E.MC.get(m, "k"), marker=E.MK.get(m, "o"), lw=1.8, ms=8, label=m)
        n_low = int((X.inj < X.base - 1).sum())                # lower by more than 1 point (a tie is not 'lower')
        out[f"behaviour | {beh}"] = dict(n_lower_sbi=n_low, n=int(len(X)))
        a.set_xticks([0, 1], ["vehicle", "SBI-553"], fontsize=12)
        a.tick_params(axis="y", labelsize=12)
        a.set_xlim(-0.25, 1.25)
        for ln in a.get_lines():
            ln.set_clip_on(False)
        a.set_title(f"{beh}\nlower after SBI-553: {n_low}/{len(X)} mice", fontsize=13)
        a.set_ylabel("% of deliveries\n(injection session)", fontsize=12)
        if k == 1:                                             # 'lower' is defined once, in the legend title
            hs = [_L2([], [], color=E.MC.get(m, "k"), marker=E.MK.get(m, "o"), lw=1.8, ms=8, label=m) for m in mice]
            a.legend(handles=hs, title="mouse   (lower = drop > 1 percentage point)", fontsize=11.5, title_fontsize=11.5,
                     ncol=len(hs), loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, handletextpad=0.3,
                     columnspacing=1.0)
    fig.savefig(os.path.join(VD, "V4_key_findings_per_mouse.png"), dpi=140)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "null", "spikein", "glm", "robust", "clusters", "decode", "figs", "more", "key"])
    a = ap.parse_args(argv)
    os.makedirs(VD, exist_ok=True)
    js = os.path.join(VD, "validation.json")
    J = json.load(open(js, encoding="utf-8")) if os.path.isfile(js) else {}
    steps = {"all": {"null", "spikein", "glm", "robust", "clusters", "decode", "more"}}.get(a.step, {a.step})
    t0 = time.time()
    S = E.load_all()
    add_selective(S)
    W = windows()
    print(f"{len(S)} sessions ({time.time() - t0:.0f} s)", flush=True)
    rp = os.path.join(VD, "A1_classes_all_rules.csv")
    if "robust" in steps:
        R = step_robust(S, W)
    else:
        R = pd.read_csv(rp) if os.path.isfile(rp) else None
        if R is not None:
            R["mouse"] = R.mouse.astype(str)
    if R is not None:
        J["robust"] = fig_robust(R)
    if "null" in steps:
        N = step_null(S, W)
    else:
        npth = os.path.join(VD, "V1_null_calibration.csv")
        N = pd.read_csv(npth) if os.path.isfile(npth) else None
    if N is not None:
        J["null"] = fig_null(N, R)
    if "spikein" in steps:
        P = step_spikein(S, W)
    else:
        pp = os.path.join(VD, "V2_spikein.csv")
        P = pd.read_csv(pp) if os.path.isfile(pp) else None
    if P is not None:
        J["spikein"] = fig_spikein(P)
    if "glm" in steps:
        G = step_glm(S)
    else:
        gp = os.path.join(VD, "V3_glm_separation.csv")
        G = pd.read_csv(gp) if os.path.isfile(gp) else None
    if G is not None:
        J["glm"] = fig_glm(G)
    if "clusters" in steps or a.step == "figs":
        meta, F, nb, k, sc, ev3 = step_clusters(S)
        J["clusters"] = dict(k=k, silhouette=sc, pca3=float(ev3), **fig_clusters(meta, F, nb, k))
    if "decode" in steps:
        D = step_decode(S, W)
    else:
        dp = os.path.join(VD, "A3_decoding.csv")
        D = pd.read_csv(dp) if os.path.isfile(dp) else None
    if D is not None:
        D["mouse"] = D.mouse.astype(str)
        J["decode"] = fig_decode(D)
    if R is not None and ("key" in steps or a.step in ("all", "figs")):
        J["key"] = fig_key(step_key(R))
    if ("more" in steps or a.step == "figs") and R is not None:
        r_ = step_clusters_responsive(S, R)
        if r_ is not None:
            meta, Fn, nb, k, sc = r_
            J["types_responsive"] = dict(k=k, silhouette=sc, **fig_clusters_responsive(meta, Fn, nb, k))
        pp = os.path.join(VD, "A3b_pseudopop_decoding.csv")
        Dp = step_pseudopop(S, W) if ("more" in steps or not os.path.isfile(pp)) else pd.read_csv(pp)
        J["pseudopop"] = fig_pseudopop(Dp)
    E.jdump(J, js)
    print(f"-> {VD} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
