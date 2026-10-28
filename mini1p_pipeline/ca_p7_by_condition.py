"""ca_p7_by_condition.py - basic plot 7 split by condition (2026-10-23, user: 'separate by baseline and SBI and veh - I want to
see whether a linear relation existing in baseline might be broken'). Added NEXT to P7_brain_behaviour.png (not replaced).

Per session (P7_sessions.csv of ca_event_analysis.py, timing-corrected marks): behaviour per delivery vs % of neurons activated
(or inhibited) by any stimulus. Baseline sessions (8: 4 mice x 2 days), after vehicle (4), after SBI-553 (4), each with its own
least-squares line and Spearman rho. Is the baseline relation kept after the injection? Each injection session is compared with
the baseline line: residual = observed % minus the % the baseline line predicts at that session's behaviour; mean residual per
drug. Descriptive: 8 / 4 / 4 sessions of 4 mice (Spearman with n = 4 cannot reach p < 0.05: its smallest two-sided p is 0.083).

Writes analysis\\event_analysis\\P7c_by_condition.png (+ _kr) and P7c_by_condition.csv.

  python ca_p7_by_condition.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_io  # noqa: E402,F401
import figfoot  # noqa: E402  (10-24: the footer goes into the slide notes, <png>.notes.txt)

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OD = os.path.join(B, "analysis", os.environ.get("CA_EVENT_OUT", "event_analysis"))
MO = ["5941", "5939", "5937", "1191"]
CC = {"baseline": "#7f8c8d", "vehicle": "#2471a3", "SBI-553": "#c0392b"}
MC = {"5941": "#e67e22", "5939": "#8e44ad", "5937": "#16a085", "1191": "#34495e"}
MK = {"5941": "o", "5939": "s", "5937": "^", "1191": "D"}
BEHS = ["withdrawal", "flinch", "attending", "movement", "escape/rearing"]
KR = {"baseline": "기준", "vehicle": "vehicle", "SBI-553": "SBI-553", "activated": "활성화", "inhibited": "억제"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    return float(r), float(p)


SHORT = {"baseline": "base", "vehicle": "veh", "SBI-553": "SBI"}
SHORT_KR = {"baseline": "기준", "vehicle": "veh", "SBI-553": "SBI"}


def fig(M, lg, fn):
    """10-24 figure review: the 2 x 5 grid was drawn at 15 x 7.4 in and shown at 70 % (rho text ~6.7 pt, ticks ~7.7 pt, footer
    ~7 pt) -> drawn at the slide box size (12.4 x 5.6 in) with every font >= 10.5 pt, fewer ticks, short in-panel rho lines;
    the behaviour names on top of the columns, x axis shared per column; the same y axis in every panel of a row; the legend
    shows the condition as the fill colour AND the fit line; the footer goes into the slide notes (figfoot)."""
    kr = lg == "kr"
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"] if kr else ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    f, ax = plt.subplots(2, len(BEHS), figsize=(12.4, 5.9), sharex="col", sharey="row")
    rows = []
    labs = {}
    for i, g in enumerate(("activated", "inhibited")):
        for k, b in enumerate(BEHS):
            a = ax[i, k]
            x0 = 100 * M[b].to_numpy(); y0 = 100 * M[g].to_numpy()
            xs = np.linspace(np.nanmin(x0), np.nanmax(x0), 20)
            lab = []
            base = M.condition == "baseline"
            fit_b = np.polyfit(100 * M[base][b], 100 * M[base][g], 1) if np.ptp(M[base][b]) > 0 else None
            for c in ("baseline", "vehicle", "SBI-553"):
                Z = M[M.condition == c]
                x, y = 100 * Z[b].to_numpy(), 100 * Z[g].to_numpy()
                for xv, yv, mm in zip(x, y, Z.mouse):
                    a.scatter(xv, yv, marker=MK[mm], s=46, color=CC[c], edgecolor=MC[mm], lw=1.6, zorder=3, clip_on=False)
                r, p = rho(x, y)
                if len(x) >= 3 and np.ptp(x) > 0:
                    cf = np.polyfit(x, y, 1)
                    xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)        # each line only over its own sessions
                    a.plot(xs, np.polyval(cf, xs), color=CC[c], lw=2.2 if c == "baseline" else 1.6,
                           ls="-" if c == "baseline" else "--", zorder=2)
                res = float(np.mean(y - np.polyval(fit_b, x))) if fit_b is not None and c != "baseline" else np.nan
                rows.append(dict(group=g, behaviour=b, condition=c, n=len(x), rho=r, p=p, slope=float(cf[0]) if len(x) >= 3 and np.ptp(x) > 0 else np.nan,
                                 mean_residual_from_baseline_line=res))
                nm = (SHORT_KR if kr else SHORT)[c]
                lab.append((c, f"{nm} rho {r:+.2f}" if np.isfinite(r) else f"{nm} n.a."))
            labs[(i, k)] = lab
            a.xaxis.set_major_locator(MaxNLocator(3))
            a.yaxis.set_major_locator(MaxNLocator(4))
            a.tick_params(labelsize=10.5)
            if i == 0:
                a.set_title(b, fontsize=12)
            else:
                a.set_xlabel("% of deliveries" if not kr else "자극 중 비율 (%)", fontsize=11)
            if k == 0:
                a.set_ylabel((f"% {g} neurons\n(any stimulus)") if not kr else f"% {KR[g]} 뉴런\n(어떤 자극이든)", fontsize=11)
    for i in range(2):                                       # one y range per row (shared), with room for the rho lines
        lo, hi = ax[i, 0].get_ylim()
        ax[i, 0].set_ylim(lo, hi + 0.62 * (hi - lo))
        for k in range(len(BEHS)):
            for j, (c, t_) in enumerate(labs[(i, k)]):
                ax[i, k].text(0.03, 0.98 - 0.125 * j, t_, color=CC[c], transform=ax[i, k].transAxes, fontsize=10.5, va="top",
                              fontweight="bold")
    hd = lambda t: Line2D([], [], ls="", marker="", label=t)                   # text-only legend entry (a header)
    h = [hd("condition (fill colour, fit line):" if not kr else "조건 (채움 색, 적합선):")]
    h += [Line2D([], [], color=CC[c], lw=2.2 if c == "baseline" else 1.6, ls="-" if c == "baseline" else "--", marker="o", ms=8,
                 mfc=CC[c], mec=CC[c], label=t_)
          for c, t_ in (("baseline", "baseline (8)" if not kr else "기준 (8)"), ("vehicle", "after vehicle (4)" if not kr else "vehicle 후 (4)"),
                        ("SBI-553", "after SBI-553 (4)" if not kr else "SBI-553 후 (4)"))]
    h += [hd("   mouse (shape, edge):" if not kr else "   마우스 (모양, 테두리):")]
    h += [Line2D([], [], marker=MK[m], ls="", mec=MC[m], mfc="w", mew=1.8, ms=8, label=m) for m in MO]
    f.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, fontsize=10.5, handletextpad=0.35, columnspacing=0.8,
             handlelength=1.8)
    figfoot.note(f, ("One point = one session; colour = condition, edge / marker = mouse; lines = least-squares fit per condition; "
                     "rho = Spearman per condition (descriptive: 8 / 4 / 4 sessions; n = 4 cannot reach p < 0.05).") if not kr else
                 ("점 하나 = 세션 하나; 색 = 조건, 테두리 / 표식 = 마우스; 선 = 조건별 최소제곱 직선; rho = 조건별 Spearman (기술 통계: 세션 8 / 4 / 4; "
                  "n = 4로는 p < 0.05 불가)."), lg)
    f.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8, w_pad=0.6)
    f.savefig(fn, dpi=150); plt.close(f)
    return pd.DataFrame(rows)


def main():
    M = pd.read_csv(os.path.join(OD, "P7_sessions.csv")); M["mouse"] = M.mouse.astype(str)
    R = fig(M, "en", os.path.join(OD, "P7c_by_condition.png"))
    fig(M, "kr", os.path.join(OD, "P7c_by_condition_kr.png"))
    R.to_csv(os.path.join(OD, "P7c_by_condition.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
