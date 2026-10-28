"""deck_cfg.py - which imaging day the deck builders show (environment CA_DAY = 1 | 2; default 1), with that day's
paths, groups and the few sentences that differ between the days. The deck builders (make_day1_ca_slides,
make_day1_full_deck, make_day1_per_mouse_deck, make_day1_complete_deck) read everything day-specific from here, so
one code makes the Day-1 and the Day-2 decks (Day 2 = crossover: 5937 / 1191 SBI-553, 5941 / 5939 vehicle).

  CA_DAY=2 python make_day1_full_deck.py --base ... --out ...
"""
import os

DAY = int(os.environ.get("CA_DAY", "1"))
DATE = {1: "20261019", 2: "20261021"}[DAY]
LABEL = f"Day {DAY}"
T = f"_d{DAY}"                                   # recording tag suffix (5941_d1)
P = f"ca_d{DAY}"                                 # analysis file prefix (ca_d1_...)
SCORE = f"scoring_mini1p/{DATE}_final"
SEX = {"5941": "F", "5939": "M", "5937": "M", "1191": "F"}
ARMS = {1: {"5941": "SBI-553", "5939": "SBI-553", "5937": "vehicle", "1191": "vehicle"},
        2: {"5937": "SBI-553", "1191": "SBI-553", "5941": "vehicle", "5939": "vehicle"}}[DAY]
MICE = [(m, a, SEX[m]) for a0 in ("SBI-553", "vehicle") for m, a in ARMS.items() if a == a0]
GROUP = {a: [m for m, aa, _ in MICE if aa == a] for a in ("SBI-553", "vehicle")}
GROUP_LABEL = {a: f"{a} ({', '.join(GROUP[a])})" for a in GROUP}
ORDER_TXT = ", ".join(m for m, _, _ in MICE)
GROUPS_TXT = " and ".join(GROUP["SBI-553"]) + " (SBI-553), " + " and ".join(GROUP["vehicle"]) + " (vehicle)"
EX = "5941"                                      # the example field of view (the 5-min test mouse, followed on both days)
# Day 2: the neuron selection is waiting for your confirmation -> every slide carries a mark
PROVISIONAL = DAY == 2 and not os.path.isfile(os.path.join(r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553", "mini1p_out",
                                                            DATE, "ALL_CONFIRMED.flag"))
PROV_MARK = "PROVISIONAL - neuron selection not yet confirmed" if PROVISIONAL else ""
# your three categories (2026-10-22): green = the neuron exists on both days, magenta = Day 1 only, cyan = Day 2 only
C3 = {"both": "#2ecc40", "d1": "#f012be", "d2": "#00d0ff"}
ONE_COL = C3["d1"] if DAY == 1 else C3["d2"]
ONE_LAB = "Day 1 only" if DAY == 1 else "Day 2 only"
GREEN = "neuron on both days (Day 1 and Day 2)"
ORANGE = f"neuron on {ONE_LAB}"
GREEN_SHORT = "kept: on both days"
ORANGE_SHORT = f"kept: {ONE_LAB}"


def both_ids(B, m):
    """this day's neuron ids that exist on both days (analysis/crossday/<m>_pairs.csv, your confirmed tracking)"""
    import pandas as _pd
    f = os.path.join(B, "analysis", "crossday", f"{m}_pairs.csv")
    return set(_pd.read_csv(f)[f"day{DAY}_id"].astype(int)) if os.path.isfile(f) else set()


def one_day_flags(B, m, R):
    """'x' for the neurons of one day only, '' for the neurons on both days (the old 'attention' slot of the figures)"""
    import pandas as _pd
    b = both_ids(B, m)
    return _pd.Series(["" if int(i) in b else "x" for i in R.id], index=R.index)
NEXT = {1: "Day 2 (crossover, recorded 2026-10-21) will test the drug within each mouse.",
        2: "Day 2 is the crossover of Day 1: every mouse received the other treatment; Day 1 + Day 2 together compare "
           "the drug within each mouse."}[DAY]

# analysis folders: Day 1 names -> this day's
_AN2 = {"ca_day1": "ca_day2", "ca_day1_extended": "ca_day2_extended", "ca_day1_extended_znoise": "ca_day2_extended_znoise",
        "ca_day1_znoise": "ca_day2_znoise", "ca_day1_roi": "ca_day2_roi", "ca_day1_bgcorr": "ca_day2_bgcorr",
        "behavior_day1": "behavior_day2", "ca_robustness": "ca_robustness_day2",
        "ca_responsive_change": "ca_responsive_change_day2", "ca_response_shape": "ca_response_shape_day2",
        "ca_cell_event_matrix": "ca_cell_event_matrix_day2", "ca_videos": "ca_videos_day2",
        "behavior_examples": "behavior_examples"}


# the best-window re-run (2026-10-22, user: 'use the best time window for ALL neuronal analysis'): CA_WIN=best CA_AN_SUFFIX=_bw
# reads the window-dependent analyses from <folder>_bw (stimuli 0-1 s vs the 0.5 s before, behaviours 0-0.5 s vs the 1 s before)
WIN_BEST = os.environ.get("CA_WIN", "old") == "best"
AN_SUF = os.environ.get("CA_AN_SUFFIX", "")
_WINDEP = {"ca_day1", "ca_day1_extended", "ca_day1_roi", "ca_response_shape", "ca_responsive_change", "ca_robustness",
           "ca_longitudinal"}
WRULE = "z[0,1]-z[-0.5,0] (stimuli) / z[0,0.5]-z[-1,0] (behaviours)" if WIN_BEST else "z[0,2]-z[-2,0]"
WRESP = "best-window (stimuli 0-1 s, behaviours 0-0.5 s)" if WIN_BEST else "[0, 2] s"
WPRE = "the pre window (stimuli -0.5-0 s, behaviours -1-0 s)" if WIN_BEST else "[-2, 0] s"
WLIT = "0-1 s vs -0.5..0 s (stimuli); 0-0.5 s vs -1..0 s (behaviours)" if WIN_BEST else "0-2 s vs -2..0 s"


def an(name):
    """a Day-1 analysis folder name -> this day's folder (+ the best-window suffix for the window-dependent ones)"""
    base = name if DAY == 1 else _AN2.get(name, name)
    return base + AN_SUF if (AN_SUF and name in _WINDEP) else base


def fn(name):
    """a Day-1 analysis file name -> this day's"""
    if DAY == 1:
        return name
    for a, b in (("ca_d1_", P + "_"), ("behav_d1_", f"behav_d{DAY}_"), ("MAIN_day1_", f"MAIN_day{DAY}_")):
        if name.startswith(a):
            return b + name[len(a):]
    return name


def A(B, *parts):
    """os.path.join(B, 'analysis', folder, file...) with the Day-1 names mapped to this day"""
    if not parts:
        return os.path.join(B, "analysis")
    return os.path.join(B, "analysis", an(parts[0]), *[fn(p) for p in parts[1:]])
