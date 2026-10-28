"""ca_event_motion_check.py - is the session-to-session jump in '% responsive' a brain-motion or a shared-signal effect?
For every session x event (16 x 9): the event-locked change of (1) the registration shift (|x, y| px of the motion
correction), (2) the frame-to-template correlation after registration, (3) the field brightness outside every footprint,
(4) the mean response of ALL neurons (the shared part), each in the same window as the neuron classes - against the % of
neurons classified responsive. Writes analysis\\event_analysis\\Q1_motion_check.png / .csv.

  python ca_event_motion_check.py [--dir <event_analysis folder>]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import ca_event_analysis as E  # noqa: E402


def locked(x, s, frames, w):
    """mean over events of x[response window] - x[baseline window] (x = one value per trace row)"""
    p0, p1, k0, k1 = E.win_rows(*w)
    idx, f = E.ev_rows(s, frames)
    idx = idx[(idx + p0 >= 0) & (idx + k1 < len(x))]
    if len(idx) < 5:
        return np.nan
    a = np.array([np.nanmean(x[i + k0:i + k1]) - np.nanmean(x[i + p0:i + p1]) for i in idx])
    return float(np.nanmean(a))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=E.OD)
    a = ap.parse_args(argv)
    J = json.load(open(os.path.join(a.dir, "summary.json"), encoding="utf-8"))
    W = {ev: tuple(J["windows"][ev]["1"]) for ev in E.EVENTS}
    T = pd.read_csv(os.path.join(a.dir, "cells_classes.csv"))
    T["mouse"] = T.mouse.astype(str)
    T["resp"] = T.cls.isin(["activated", "inhibited"])
    pr = T.groupby(["mouse", "day", "phase", "event"]).resp.mean()
    S = E.load_all()
    rows = []
    for (m, d, ph), s in S.items():
        mot = s.get("motion") or {}
        sh = np.asarray(mot.get("shift", np.full(len(s["fr"]), np.nan)), float)
        co = np.asarray(mot.get("corr", np.full(len(s["fr"]), np.nan)), float)
        t = f"{m}_d{d}"
        fb = os.path.join(E.ROOT[d], t, "5_analysis", E.CA.TRACES, f"{t}_{ph}_background.npz")
        if os.path.isfile(fb):
            Bg = np.load(fb)
            b = pd.Series(np.asarray(Bg["b"], float), index=np.asarray(Bg["frame"], int)).reindex(s["fr"]).values
            med = np.nanmedian(b)
            b = (b - med) / (1.4826 * np.nanmedian(np.abs(b - med)))
        else:
            b = np.full(len(s["fr"]), np.nan)
        pop = np.nanmean(np.asarray(s["T"]["z"], float), 1)
        for ev in E.EVENTS:
            fr = s["B"]["ev"].get(ev, [])
            if len(fr) < 5 or (m, d, ph, ev) not in pr:
                continue
            rows.append(dict(mouse=m, day=d, phase=ph, condition=E.cond_of(m, d, ph), event=ev, pct_resp=100 * pr[(m, d, ph, ev)],
                             shift=locked(sh, s, fr, W[ev]), corr=locked(co, s, fr, W[ev]), field=locked(b, s, fr, W[ev]),
                             population=locked(pop, s, fr, W[ev]), abs_population=abs(locked(pop, s, fr, W[ev]))))
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(a.dir, "Q1_motion_check.csv"), index=False)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    res = {}
    for k, (col, lab) in enumerate((("shift", "registration shift: after - before (px)"), ("corr", "frame-template correlation:\nafter - before"),
                                    ("field", "field brightness outside the cells:\nafter - before (SD)"),
                                    ("abs_population", "|mean response of ALL neurons| (z)"))):
        x = Q[col].values
        ok = np.isfinite(x)
        rho, p = spearmanr(x[ok], Q.pct_resp.values[ok]) if ok.sum() > 5 else (np.nan, np.nan)
        res[col] = dict(rho=float(rho), p=float(p), n=int(ok.sum()))
        for c in E.COND:
            sel = ok & (Q.condition == c).values
            ax[k].scatter(x[sel], Q.pct_resp.values[sel], s=26, color=E.CC[c], alpha=0.75, label=c)
        ax[k].set_xlabel(lab)
        ax[k].set_title(f"rho {rho:+.2f} {E.star(p) or 'ns'}", fontsize=12)
    ax[0].set_ylabel("% neurons responsive\n(one dot = session x event)")
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "Q1_motion_check.png"), dpi=140)
    plt.close(fig)
    J["motion_check"] = res
    E.jdump(J, os.path.join(a.dir, "summary.json"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
