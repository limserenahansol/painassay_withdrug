"""R  ROBUSTNESS of the main Day-1 result. Is the change of activity after the injection a property
of the data, or of one analysis choice?

Outcome (per cell, as C2 / E2b): log2((post + eps) / (pre + eps)) of the activity per minute of
still frames, in two windows:
  stimulus blocks   the four stimulus blocks pooled
  no-stimulus start the first 3 min (after the first 15 s)
Recomputed with one choice changed at a time:
  activity measure  OASIS transients (main) | transients >= 4 and >= 8 noise SD | an independent
                    detector (crossings of 4 noise SD by the 0.2-s smoothed dF/F) | OASIS
                    deconvolved activity | % of time above 3 noise SD
                    Noise SD = the cell's PSD noise pooled over both sessions: ONE threshold in
                    dF/F per cell for pre and post. (The z of the trace files divides by the MAD of
                    each session, which shrinks when a session is less active - 5941 post -25 % -
                    so z thresholds are lower in the quieter session and hide a decrease.)
  traces            background-corrected traces (traces_bgcorr) | footprint pixels minus the local
                    neuropil straight from the movie, no demixing (traces_roi, ca_roi_traces.py)
  frames            all frames (moving included) | still + good registration only (frame/template
                    r above the session threshold and the fastest 5 % of shift changes removed) |
                    the first 90 s of each block only (equal time into the block)
  cells             Ca-like shape | no shape flag at all | SNR top half | SNR bottom half | >= 100 px from the lens edge |
                    5941: only the cells you kept that the day run found (no rescued) | 5941: only
                    the cells added by the rule | without the rescued cells
  eps               0.02 and 0.5 /min instead of 0.1 (rates); other measures: 2 % of their median
Per mouse: median over cells, 95 % bootstrap CI over cells, Wilcoxon post vs pre.
Arms: SBI-553 - vehicle = mean of the 2 SBI-553 mouse medians - mean of the 2 vehicle ones, its CI
over cells only (mice fixed: with 2 mice per arm the mouse-level uncertainty cannot be estimated),
and whether BOTH SBI-553 mice lie below BOTH vehicle mice (the only ordering 2 vs 2 can show).
R2 leave one mouse out: the arm difference without each mouse in turn.

  python ca_robustness.py --out-root <mini1p_out/20261019> --scoring-dir <final> --out <analysis/ca_robustness>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)
import ca_day1_analysis as d1  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = d1.FPS
MICE = d1.MICE
ARMC = d1.ARMC
EPS = 0.1
RNG = np.random.default_rng(0)


def noise_sd(s, other):
    """per cell noise SD of dF/F pooled over the two sessions of the day (same cells)"""
    sn = np.asarray(s["T"]["sn"], float)
    if other is not None and np.array_equal(other["T"]["cell_ids"], s["T"]["cell_ids"]):
        sn = np.sqrt(np.nanmean(np.vstack([sn ** 2, np.asarray(other["T"]["sn"], float) ** 2]), 0))
    return np.where(sn > 1e-6, sn, np.nan)


def z_crossings(z, thr=4.0, min_f=3, gap_f=30):
    """independent detector: onsets where the 0.2-s smoothed trace (in noise SD) goes above thr
    for >= 0.1 s, >= 1 s after the previous onset; rows per cell"""
    zz = np.where(np.isfinite(z), z, 0.0)
    zs = uniform_filter1d(zz, 6, axis=0)
    out = []
    for j in range(z.shape[1]):
        ab = np.r_[False, zs[:, j] > thr, False]
        d = np.diff(ab.astype(int))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        keep, last = [], -10 ** 9
        for a, b in zip(st, en):
            if b - a >= min_f and a - last >= gap_f:
                keep.append(a); last = a
        out.append(np.array(keep, int))
    return out


def session_masks(s, first_s=None):
    """row masks: still / all, good registration, the two windows"""
    T, B = s["T"], s["B"]
    fr, t = T["frame"].astype(int), T["t_s"]
    ok = T["valid"].astype(bool) & (t >= d1.SKIP_S)
    mv = np.zeros(len(fr), bool)
    if B is not None:
        st = B["st"]["move"]
        inr = fr < len(st)
        mv[inr] = st[fr[inr]]
    good = np.ones(len(fr), bool)
    mot = s.get("motion")
    if mot is not None:
        c = mot["corr"]
        dsh = np.abs(np.diff(np.r_[mot["shift"][0], mot["shift"]]))
        lim = np.nanpercentile(dsh, 95)
        good = ~(c < mot["corr_thr"]) & ~(dsh > lim)
        good = uniform_filter1d((~good).astype(float), int(FPS), mode="nearest") == 0   # +-0.5 s around a bad frame
    nost, stim = np.zeros(len(fr), bool), np.zeros(len(fr), bool)
    for bn, s0, s1 in d1.BLOCKS:
        s0 = max(s0, d1.SKIP_S)
        e = s1 if first_s is None else min(s1, s0 + first_s)
        w = (t >= s0) & (t < e)
        if bn == "no stimulus":
            nost |= w
        else:
            stim |= w
    return dict(ok=ok, still=ok & ~mv, good=good, win={"stimulus blocks": stim, "no-stimulus start": nost})


def per_cell(s, measure, frames, first_s=None, zthr_cache=None, sn=None):
    """per cell activity per minute (or % time) in each window; sn = noise SD per cell (dF/F)"""
    T, E = s["T"], s["E"]
    fr = T["frame"].astype(int)
    ids = T["cell_ids"].astype(int)
    M = session_masks(s, first_s)
    base = M["ok"] if frames == "all" else M["still"]
    if frames == "still+good":
        base = base & M["good"]
    rows = {}
    if measure.startswith("oasis"):
        k = {"oasis": None, "oasis_n4": 4.0, "oasis_n8": 8.0}[measure]
        Ef = E
        if k is not None:
            thr = dict(zip(T["cell_ids"].astype(int), k * sn))
            Ef = E[E.amplitude_dff.values >= E.cell.map(thr).values]
        ev_rows = {}
        for c, g in Ef.groupby("cell"):
            on = g.onset_frame.values.astype(int)
            r = np.searchsorted(fr, on)
            inr = r < len(fr)
            r, on = r[inr], on[inr]
            ev_rows[int(c)] = r[fr[r] == on]
    elif measure == "zcross":
        zc = zthr_cache if zthr_cache is not None else z_crossings(np.asarray(T["z"], float))
    for wn, wm in M["win"].items():
        m = base & wm
        mins = m.sum() / FPS / 60
        v = np.full(len(ids), np.nan)
        if mins >= 0.5:
            if measure.startswith("oasis"):
                for j, c in enumerate(ids):
                    r = ev_rows.get(int(c), np.array([], int))
                    v[j] = m[r].sum() / mins
            elif measure == "zcross":
                for j in range(len(ids)):
                    v[j] = m[zc[j]].sum() / mins
            elif measure == "deconv":
                v = np.nansum(np.asarray(T["s"], float)[m], 0) / mins
            elif measure == "time_n3":
                Z = np.asarray(T["dff"], float)[m] / sn
                v = 100 * np.nanmean(Z > 3, 0)
        rows[wn] = v
    D = pd.DataFrame(rows)
    D.insert(0, "cell", ids)
    return D


def cell_table(out_root, m):
    R = pd.read_csv(os.path.join(out_root, f"{m}_d{d1.DAY}", "4_cell_review", f"{m}_d{d1.DAY}_cell_review.csv"))
    R = R[R.keep == 1].copy()
    R["cell"] = R["id"].astype(int)
    return R


VARIANTS = [
    # name, traces, measure, frames, first_s, cell filter, eps
    ("main: OASIS transients, still frames", "traces", "oasis", "still", None, None, EPS),
    ("transients >= 4 noise SD", "traces", "oasis_n4", "still", None, None, EPS),
    ("transients >= 8 noise SD", "traces", "oasis_n8", "still", None, None, EPS),
    ("other detector: 4 noise SD crossings", "traces", "zcross", "still", None, None, EPS),
    ("OASIS deconvolved activity", "traces", "deconv", "still", None, None, "rel"),
    ("% time above 3 noise SD", "traces", "time_n3", "still", None, None, "rel"),
    ("background-corrected traces", "traces_bgcorr", "oasis", "still", None, None, EPS),
    ("ROI - local neuropil traces (no demixing)", "traces_roi", "oasis", "still", None, None, EPS),
    ("all frames (moving included)", "traces", "oasis", "all", None, None, EPS),
    ("still + good registration", "traces", "oasis", "still+good", None, None, EPS),
    ("first 90 s of each block", "traces", "oasis", "still", 90.0, None, EPS),
    ("Ca-like cells only", "traces", "oasis", "still", None, "ca_like", EPS),
    ("no shape flag (Ca-like, 1 piece, 0.5-2x area, >= 60 px from edge)", "traces", "oasis", "still", None, "clean", EPS),
    ("SNR top half", "traces", "oasis", "still", None, "snr_hi", EPS),
    ("SNR bottom half", "traces", "oasis", "still", None, "snr_lo", EPS),
    (">= 100 px from the lens edge", "traces", "oasis", "still", None, "centre", EPS),
    ("5941: your kept cells found by the day run", "traces", "oasis", "still", None, "user", EPS),
    ("5941: cells added by the rule", "traces", "oasis", "still", None, "rule", EPS),
    ("without rescued cells", "traces", "oasis", "still", None, "norescue", EPS),
    ("only cells you decided (no attention mark)", "traces", "oasis", "still", None, "decided", EPS),
    ("eps 0.02 /min", "traces", "oasis", "still", None, None, 0.02),
    ("eps 0.5 /min", "traces", "oasis", "still", None, None, 0.5),
]


def cell_filter(R, how, m):
    if how is None:
        return set(R.cell)
    if how == "ca_like":
        return set(R.cell[R.ca_like == 1])
    if how == "clean":
        typ = R.area_px.median()
        ok = (R.ca_like == 1) & (R.n_pieces == 1) & R.area_px.between(0.5 * typ, 2 * typ) & (R.edge_px >= 60)
        return set(R.cell[ok])
    if how in ("snr_hi", "snr_lo"):
        med = R.snr.median()
        return set(R.cell[R.snr >= med]) if how == "snr_hi" else set(R.cell[R.snr < med])
    if how == "centre":
        return set(R.cell[R.edge_px >= 100])
    if how == "user":
        return set(R.cell[(R.decision_source == "user") & (R.verdict != "rescued")]) if m == "5941" else set(R.cell)
    if how == "rule":
        return set(R.cell[R.decision_source == "rule"])
    if how == "norescue":
        return set(R.cell[R.verdict != "rescued"])
    if how == "decided":
        att = R["attention"].fillna("").astype(str) if "attention" in R else pd.Series([""] * len(R), index=R.index)
        return set(R.cell[att == ""])
    raise ValueError(how)


def boot_median(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan, np.nan
    b = np.median(x[RNG.integers(0, len(x), (n, len(x)))], 1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--scoring-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sess = {}
    for tr in ("traces", "traces_bgcorr", "traces_roi"):
        d1.TRACES = tr
        for m, arm, sex in MICE:
            for ph in ("pre", "post"):
                s = d1.load_session(a.out_root, m, ph, a.scoring_dir)
                if s is not None:
                    sess[(tr, m, ph)] = s
    d1.TRACES = "traces"
    cells = {m: cell_table(a.out_root, m) for m, _, _ in MICE}
    zc = {}
    vals = {}                       # (variant, mouse) -> DataFrame cell, window, pre, post
    for name, tr, meas, frames, first_s, filt, eps in VARIANTS:
        for m, arm, sex in MICE:
            if (tr, m, "pre") not in sess or (tr, m, "post") not in sess:
                continue
            P = {}
            for ph in ("pre", "post"):
                s = sess[(tr, m, ph)]
                sn = noise_sd(s, sess[(tr, m, "post" if ph == "pre" else "pre")])
                key = (tr, m, ph)
                if meas == "zcross" and key not in zc:
                    zc[key] = z_crossings(np.asarray(s["T"]["dff"], float) / sn)
                P[ph] = per_cell(s, meas, frames, first_s, zc.get(key), sn).set_index("cell")
            keep = cell_filter(cells[m], filt, m)
            idx = [c for c in P["pre"].index if c in keep and c in P["post"].index]
            vals[(name, m)] = (P["pre"].loc[idx], P["post"].loc[idx], eps)
    # eps relative to each measure's median (non-rate measures)
    med = {}
    for (name, m), (pre, post, eps) in vals.items():
        if eps == "rel":
            med.setdefault(name, []).append(np.r_[pre.values.ravel(), post.values.ravel()])
    med = {k: 0.02 * np.nanmedian(np.concatenate(v)) for k, v in med.items()}
    rows = []
    ratios = {}
    from scipy.stats import wilcoxon
    for (name, m), (pre, post, eps) in vals.items():
        e = med[name] if eps == "rel" else eps
        arm = dict((mm, aa) for mm, aa, _ in MICE)[m]
        for wn in pre.columns:
            ok = pre[wn].notna() & post[wn].notna()
            y = np.log2((post[wn][ok] + e) / (pre[wn][ok] + e)).values
            ratios[(name, wn, m)] = y
            lo, hi = boot_median(y)
            try:
                p = float(wilcoxon(post[wn][ok], pre[wn][ok]).pvalue) if ok.sum() >= 5 and not np.allclose(y, 0) else np.nan
            except ValueError:
                p = np.nan
            sp, sq = float(pre[wn][ok].sum()), float(post[wn][ok].sum())
            rows.append(dict(variant=name, window=wn, mouse=m, arm=arm, n_cells=int(ok.sum()), eps=round(e, 4),
                             log2_population_ratio=np.log2(sq / sp) if sp > 0 and sq > 0 else np.nan,
                             pre_median=float(np.median(pre[wn][ok])) if ok.any() else np.nan,
                             post_median=float(np.median(post[wn][ok])) if ok.any() else np.nan,
                             log2_ratio_median=float(np.median(y)) if len(y) else np.nan, ci_lo=lo, ci_hi=hi, p_wilcoxon=p))
    R1 = pd.DataFrame(rows)
    R1.round(4).to_csv(os.path.join(a.out, "R1_robustness_per_mouse.csv"), index=False)
    # arm level
    sbi = [m for m, arm, _ in MICE if arm == "SBI-553"]
    veh = [m for m, arm, _ in MICE if arm == "vehicle"]
    arow = []
    for name, *_ in VARIANTS:
        for wn in ("stimulus blocks", "no-stimulus start"):
            ys = {m: ratios.get((name, wn, m)) for m in sbi + veh}
            if any(v is None or len(v) < 3 for v in ys.values()):
                continue
            md = {m: np.median(v) for m, v in ys.items()}
            diff = np.mean([md[m] for m in sbi]) - np.mean([md[m] for m in veh])
            bs = []
            for _ in range(2000):
                bm = {m: np.median(v[RNG.integers(0, len(v), len(v))]) for m, v in ys.items()}
                bs.append(np.mean([bm[m] for m in sbi]) - np.mean([bm[m] for m in veh]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            arow.append(dict(variant=name, window=wn, sbi_minus_vehicle=diff, ci_lo_cells=lo, ci_hi_cells=hi,
                             both_sbi_below_both_vehicle=bool(max(md[m] for m in sbi) < min(md[m] for m in veh)),
                             **{f"median {m}": md[m] for m in sbi + veh}))
    RA = pd.DataFrame(arow)
    RA.round(4).to_csv(os.path.join(a.out, "R1_robustness_arm.csv"), index=False)
    # leave one mouse out (main variant)
    lrow = []
    name = VARIANTS[0][0]
    for wn in ("stimulus blocks", "no-stimulus start"):
        md = {m: np.median(ratios[(name, wn, m)]) for m in sbi + veh if (name, wn, m) in ratios}
        for drop in [None] + sbi + veh:
            s_ = [m for m in sbi if m != drop]
            v_ = [m for m in veh if m != drop]
            lrow.append(dict(window=wn, left_out=drop or "none", sbi_mice="+".join(s_), vehicle_mice="+".join(v_),
                             sbi_minus_vehicle=np.mean([md[m] for m in s_]) - np.mean([md[m] for m in v_])))
    RL = pd.DataFrame(lrow)
    RL.round(4).to_csv(os.path.join(a.out, "R2_leave_one_mouse_out.csv"), index=False)
    fig_forest(R1, RA, a.out)
    print(RA[["variant", "window", "sbi_minus_vehicle", "ci_lo_cells", "ci_hi_cells", "both_sbi_below_both_vehicle"]]
          .round(2).to_string(index=False))
    print(RL.round(2).to_string(index=False))
    print("robustness ->", a.out)


def fig_forest(R1, RA, out):
    names = [v[0] for v in VARIANTS]
    wins = ["stimulus blocks", "no-stimulus start"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.42 * len(names) + 2.2), sharey=True)
    off = {"5941": -0.24, "5939": -0.08, "5937": 0.08, "1191": 0.24}
    mk = {"5941": "o", "5939": "s", "5937": "o", "1191": "s"}
    for k, wn in enumerate(wins):
        a = ax[k]
        for i, nm in enumerate(names):
            y0 = len(names) - 1 - i
            if i % 2 == 0:
                a.axhspan(y0 - 0.5, y0 + 0.5, color="0.95", lw=0, zorder=0)
            for m, arm, sex in MICE:
                r = R1[(R1.variant == nm) & (R1.window == wn) & (R1.mouse == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                a.errorbar(r.log2_ratio_median, y0 + off[m], xerr=[[r.log2_ratio_median - r.ci_lo], [r.ci_hi - r.log2_ratio_median]],
                           fmt=mk[m], color=ARMC[arm], ms=4.5, lw=1, capsize=0,
                           label=f"{m} ({arm})" if (i == 0 and k == 0) else None)
            q = RA[(RA.variant == nm) & (RA.window == wn)]
            if len(q):
                q = q.iloc[0]
                a.plot(q.sbi_minus_vehicle, y0, "D", color="k", ms=6, zorder=5,
                       label="SBI-553 - vehicle" if (i == 0 and k == 0) else None)
                a.text(1.02, y0, f"{q.sbi_minus_vehicle:+.2f}" + (" *" if q.both_sbi_below_both_vehicle else ""),
                       transform=a.get_yaxis_transform(), va="center", fontsize=7.5)
        a.axvline(0, color="0.4", lw=0.8)
        a.set_yticks(range(len(names)), names[::-1], fontsize=8)
        a.set_xlabel("log2(post / pre) activity, still frames (median over cells, 95% bootstrap CI)", fontsize=8)
        a.set_title(wn, fontsize=10)
        a.set_xlim(-3.2, 1.6)
        a.set_ylim(-0.6, len(names) - 0.4)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=5, frameon=False)
    n = R1[R1.variant == names[0]].groupby("mouse").n_cells.first()
    fig.suptitle("R1 robustness of the injection effect: one analysis choice changed per row.  Black diamond = mean of the "
                 "SBI-553 mouse medians - mean of the vehicle ones (right: value; * = both SBI-553 mice below both vehicle mice).\n"
                 f"Cells: {', '.join(f'{m} n={int(v)}' for m, v in n.items())}. Per-mouse Wilcoxon p in R1_robustness_per_mouse.csv; "
                 "2 mice per arm, so no arm-level test (minimum permutation p = 0.33).", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 0.97, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(os.path.join(out, "R1_robustness_forest.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
