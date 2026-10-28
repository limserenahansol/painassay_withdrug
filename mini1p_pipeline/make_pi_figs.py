"""make_pi_figs.py - 10-26 (user: PI version of the compact deck; 'clear label, intuitive, readable font size'): two figures of
make_pi5_deck.py redrawn for the PI version, saved under NEW names (the old figures stay):
    sA1_behaviour_per_stimulus_stats_PI.png  - slide 'every behaviour, every stimulus' with readable fonts (ticks / stars / legend 11 pt)
    sF119_pin_only_PI.png                    - pin-prick UP / DOWN / rest traces alone, WITH the time axis (the user's crop of the 2-row
                                               figure removed the only x labels) and one shared y range
The data code is taken from make_pi5_deck.py itself (its module code up to the deck builder, plus the helpers these two figures use),
so the numbers are the same as in the deck.

  python make_pi_figs.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "make_pi5_deck.py"), encoding="utf-8").read().split("\n")


def block(first, last):
    """source lines first..last (1-based, inclusive) of make_pi5_deck.py"""
    return "\n".join(SRC[first - 1:last])


def find(prefix, start=1):
    for i in range(start - 1, len(SRC)):
        if SRC[i].startswith(prefix):
            return i + 1
    raise KeyError(prefix)


G = {"__name__": "make_pi_figs", "__file__": os.path.join(HERE, "make_pi5_deck.py")}
exec(compile(block(1, find("prs = Presentation()") - 1), "make_pi5_deck.py", "exec"), G)      # data + helpers (no figure calls)
i0 = find("BEH6 = ")
exec(compile(block(i0, find("def fig_beh_stim_stats") - 1), "make_pi5_deck.py", "exec"), G)    # BEH6, SHS, safe_did
exec(compile(block(find("CND4 = "), find("CND4 = ")), "make_pi5_deck.py", "exec"), G)
exec(compile(block(find("def group_traces"), find("def fig_f119") - 1), "make_pi5_deck.py", "exec"), G)

PI_CODE = r'''
def fig_beh_stim_stats_pi(fn):
    """as fig_beh_stim_stats, fonts readable when projected (user / PI 10-26)"""
    fig, axs = plt.subplots(2, 3, figsize=(12.3, 4.75), gridspec_kw=dict(wspace=0.24, hspace=0.62, left=0.075, right=0.995, top=0.80,
                                                                          bottom=0.075))
    R = {}
    for a, beh in zip(axs.ravel(), BEH6):
        vals = []
        for i, stim in enumerate(STIM):
            for dr, c_i, c_b, dx in (("vehicle", "vi", "vb", -0.19), ("SBI-553", "si", "sb", 0.19)):
                pm_i, _ = beh_pct(beh, stim, c_i)
                pm_b, _ = beh_pct(beh, stim, c_b)
                ch = {m: pm_i[m] - pm_b[m] for m in MICE if m in pm_i and m in pm_b}
                v = np.mean(list(ch.values()))
                a.bar(i + dx, v, width=0.36, color=CC[c_i], alpha=0.55, edgecolor=CC[c_i], zorder=2)
                for m, y in ch.items():
                    a.scatter(i + dx + OFF[m] * 0.35, y, marker=MK[m], s=22, color=MC[m], edgecolor="k", lw=0.4, zorder=4)
                vals += list(ch.values()) + [v]
                R[(beh, stim, dr)] = dict(v=v, p=cmh(beh, stim, f"{dr} vs its baseline"), ch=ch)
            R[(beh, stim, "did")] = safe_did(beh, stim)
        m_ = max(abs(min(vals)), abs(max(vals)), 1.0)
        a.set_ylim(-m_ * 1.15, m_ * 2.05)
        for i, stim in enumerate(STIM):
            for dr, dx in (("vehicle", -0.19), ("SBI-553", 0.19)):
                p = R[(beh, stim, dr)]["p"]
                a.text(i + dx, m_ * 1.10, st(p) if np.isfinite(p) else "", ha="center", va="bottom", fontsize=11,
                       color=CC["vi" if dr == "vehicle" else "si"], fontweight="bold" if np.isfinite(p) and p < 0.05 else "normal")
            pd_ = R[(beh, stim, "did")]
            if np.isfinite(pd_):
                bracket(a, i - 0.19, i + 0.19, m_ * 1.55, st(pd_), fs=11, h=0.04 * m_)
        a.axhline(0, color="0.3", lw=0.8)
        a.set_xticks(range(4), [SHS[s_] for s_ in STIM], fontsize=11.5)
        a.set_title(beh, fontsize=13)
        a.tick_params(axis="y", labelsize=11)
    for a in axs[:, 0]:
        a.set_ylabel("change in %\nof deliveries", fontsize=11.5)
    h = [Patch(color=CC["vi"], alpha=0.55, label="vehicle day"), Patch(color=CC["si"], alpha=0.55, label="SBI-553 day")]
    fig.legend(handles=h, ncol=2, loc="upper left", bbox_to_anchor=(0.005, 1.0), frameon=False, fontsize=11.5, handlelength=1.2,
               columnspacing=1.0)
    hm = [Line2D([], [], ls="", marker=MK[m], color=MC[m], markeredgecolor="k", markersize=7, label=f"{m} (SBI-553 Day {SBIDAY[m]})")
          for m in MICE]
    fig.legend(handles=hm, ncol=4, loc="upper right", bbox_to_anchor=(0.998, 1.0), frameon=False, fontsize=11, handlelength=0.9,
               columnspacing=0.9)
    fig.savefig(fn); plt.close(fig)
    return R


def fig_f119_pin_pi(fn, ev="Pin prick"):
    """pin-prick UP / DOWN / rest traces alone (as fig_f119 row 1), with the time axis and one shared y range"""
    fig, ax = plt.subplots(1, 4, figsize=(12.3, 3.9), sharex=True, sharey=True, squeeze=False,
                           gridspec_kw=dict(wspace=0.12, left=0.085, right=0.995, top=0.70, bottom=0.17))
    GC = {"UP": "#e74c3c", "DOWN": "#2e86c1", "rest": "#95a5a6"}
    Gt = group_traces(ev)
    pre, a_, b_ = win(ev)
    R, T = SG[ev]
    for k, c in enumerate(CND4):
        a = ax[0, k]
        tt = Gt[c]["tt"]; sel = (tt >= -2) & (tt <= 4)
        for g in ("rest", "DOWN", "UP"):
            Y = base_sub(Gt[c][g], tt, ev)[sel]
            if Y.shape[1] == 0:
                continue
            mu = Y.mean(1); se = Y.std(1, ddof=1) / np.sqrt(Y.shape[1]) if Y.shape[1] > 1 else 0 * mu
            a.fill_between(tt[sel], mu - se, mu + se, color=GC[g], alpha=0.2, lw=0); a.plot(tt[sel], mu, color=GC[g], lw=1.5)
        a.axvspan(min(a_, 0) - pre, min(a_, 0), color="#d6eaf8", lw=0, zorder=0); a.axvspan(a_, b_, color="#e5e7e9", lw=0, zorder=0)
        a.axvline(0, color="k", lw=0.7, ls="--"); a.axhline(0, color="0.6", lw=0.6)
        a.set_title(f"{CL[c]}\nUP {R[c]['kUP']}, DOWN {R[c]['kDOWN']}, rest {R[c]['n'] - R[c]['kUP'] - R[c]['kDOWN']}", fontsize=12,
                    color=CC[c] if c != "vb" else "0.4")
        a.tick_params(labelsize=11)
        a.set_xlabel("time from pin prick (s)", fontsize=11.5)
        a.set_xticks([-2, 0, 2, 4])
    ax[0, 0].set_ylabel("z - mean of the\n0.5 s before", fontsize=11.5)
    fig.text(0.54, 0.865, f"% UP:  vehicle day {st(T['UP_veh'])}, SBI-553 day {st(T['UP_sbi'])} (McNemar);  after vs after {st(T['UP_after'])}"
             f" (Fisher)", ha="center", fontsize=12, fontweight="bold", va="bottom")
    h = [Line2D([], [], color=GC[g], lw=2.5, label=g + (" (not responsive)" if g == "rest" else "")) for g in ("UP", "DOWN", "rest")] + \
        [Patch(color="#d6eaf8", label="0.5 s before"), Patch(color="#e5e7e9", label="response window")]
    fig.legend(handles=h, ncol=5, loc="upper center", bbox_to_anchor=(0.54, 1.0), frameon=False, fontsize=11.5)
    fig.savefig(fn); plt.close(fig)
'''
exec(compile(PI_CODE, "make_pi_figs.py", "exec"), G)
FIG = G["FIG"]
f1 = os.path.join(FIG, "sA1_behaviour_per_stimulus_stats_PI.png"); G["fig_beh_stim_stats_pi"](f1)
f2 = os.path.join(FIG, "sF119_pin_only_PI.png"); G["fig_f119_pin_pi"](f2)
print("->", f1); print("->", f2)
