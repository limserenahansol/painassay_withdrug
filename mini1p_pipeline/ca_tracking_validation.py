"""ca_tracking_validation.py - is a tracked Day-1 / Day-2 pair the same neuron? An independent check from activity.

A neuron keeps much of its character between days (how active it is at rest, its transient size and decay - set by the
cell and its GCaMP level), whereas two different neurons in the same region differ. So, over the tracked pairs of a
mouse, the Day-1 baseline value and the Day-2 baseline value of the same pair should correlate MORE than when the Day-2
partners are shuffled among neighbouring neurons (Day-2 kept neurons within 80 px of the true partner: controls for
region effects such as vignetting or depth). Measures (baseline sessions only, so the drugs do not enter):
  rate    Ca transients per minute (OASIS events, 5_analysis/traces)
  amp     median transient amplitude (dF/F)
  tau     decay time constant
  noise   noise SD of dF/F
Per mouse and measure: Spearman rho of the pairs, the shuffle null (2000), one-sided p. Also by tracking class (A vs B).
Writes analysis\\crossday\\tracking_validation.csv and tracking_validation.png

  python ca_tracking_validation.py [--mice 5941 5937]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401  (a figure save retries on a transient Windows write error)

BASE = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
DATE = {1: "20261019", 2: "20261021"}


def cell_table(m, day):
    tag = f"{m}_d{day}"
    d = os.path.join(BASE, "mini1p_out", DATE[day], tag, "5_analysis", "traces")
    T = np.load(os.path.join(d, f"{tag}_pre_traces.npz"))
    E = pd.read_csv(os.path.join(d, f"{tag}_pre_events.csv"))
    dur_min = float(T["valid"].sum()) / 30.0 / 60.0
    ids = T["cell_ids"].astype(int)
    g = E.groupby("cell")
    rate = g.size().reindex(ids, fill_value=0) / dur_min
    amp = g.amplitude_dff.median().reindex(ids)
    noise = pd.Series(T["sn"], index=ids)                               # noise SD of dF/F (OASIS)
    R = pd.read_csv(os.path.join(BASE, "mini1p_out", DATE[day], tag, "4_cell_review", f"{tag}_cell_review.csv")).set_index("id")
    return pd.DataFrame(dict(rate=rate.values, amp=amp.values, tau=T["tau_s"], noise=noise.values,
                             row=R.loc[ids, "row"].values, col=R.loc[ids, "col"].values), index=ids)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mice", nargs="*", default=["5941", "5939", "5937", "1191"])
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    rows = []
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    for m in a.mice:
        pf = os.path.join(BASE, "analysis", "crossday", f"{m}_pairs.csv")
        try:
            P = pd.read_csv(pf)
            C1, C2 = cell_table(m, 1), cell_table(m, 2)
        except FileNotFoundError as e:
            print(f"{m}: not ready ({e.filename})"); continue
        P = P[P.day1_id.isin(C1.index) & P.day2_id.isin(C2.index)].reset_index(drop=True)
        if len(P) < 8:
            print(f"{m}: only {len(P)} pairs"); continue
        xy2 = C2[["row", "col"]].to_numpy(float); ids2 = C2.index.values
        # neighbours of each true partner (candidates for the shuffle)
        nb = []
        for r in P.itertuples():
            p0 = C2.loc[r.day2_id, ["row", "col"]].to_numpy(float)
            d = np.hypot(*(xy2 - p0).T)
            nb.append(ids2[(d <= 80) & (ids2 != r.day2_id)])
        for k, meas in enumerate(("rate", "amp", "tau", "noise")):
            x = C1.loc[P.day1_id, meas].values; y = C2.loc[P.day2_id, meas].values
            ok = np.isfinite(x) & np.isfinite(y)
            rho = spearmanr(x[ok], y[ok]).correlation if ok.sum() >= 8 else np.nan
            null = []
            for _ in range(2000):
                yy = np.array([C2.at[rng.choice(n), meas] if len(n) else C2.at[r.day2_id, meas]
                               for n, r in zip(nb, P.itertuples())])
                o2 = np.isfinite(x) & np.isfinite(yy)
                null.append(spearmanr(x[o2], yy[o2]).correlation if o2.sum() >= 8 else np.nan)
            null = np.array(null, float)
            p = (np.sum(null >= rho) + 1) / (np.sum(np.isfinite(null)) + 1) if np.isfinite(rho) else np.nan
            rows.append(dict(mouse=m, measure=meas, n_pairs=int(ok.sum()), rho=rho, null_median=np.nanmedian(null),
                             null_95=np.nanpercentile(null, 95), p_one_sided=p))
            if "track_class" in P:
                for c in ("A", "B"):
                    q = (P.track_class == c).values & ok
                    if q.sum() >= 8:
                        rows[-1][f"rho_{c}"] = spearmanr(x[q], y[q]).correlation
                        rows[-1][f"n_{c}"] = int(q.sum())
            if meas == "rate":
                a_ = ax[a.mice.index(m)] if len(a.mice) == 4 else ax[0]
                a_.scatter(x, y, s=12, color="#1f4e79", alpha=0.75)
                a_.set_xscale("symlog", linthresh=1); a_.set_yscale("symlog", linthresh=1)
                a_.set_xlabel("Day 1 baseline: transients / min"); a_.set_ylabel("Day 2 baseline: transients / min")
                a_.set_title(f"{m}: rho {rho:.2f} (shuffled neighbours {np.nanmedian(null):.2f}, p {p:.3f}), n {ok.sum()}", fontsize=9.5)
    V = pd.DataFrame(rows)
    out = os.path.join(BASE, "analysis", "crossday")
    V.round(4).to_csv(os.path.join(out, "tracking_validation.csv"), index=False)
    fig.suptitle("Tracked pairs: the same neuron should keep its resting activity between days (each dot = one tracked pair; "
                 "null = Day-2 partner replaced by a neighbour within 80 px)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(out, "tracking_validation.png"), dpi=130); plt.close(fig)
    pd.set_option("display.width", 200)
    print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
