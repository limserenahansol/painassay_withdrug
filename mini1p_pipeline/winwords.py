"""winwords.py - the response windows of the event analysis (summary.json of ca_event_analysis.py) as numbers and words, so
every deck, figure and video says the window the analysis actually used (10-23: after the timing fix the window search chose
0-0.5 s for the stimuli, before 0-1 s; 10-24 (user): WIN_MODE=mix = a window per event, 0-0.5 or 0-1 s, whichever showed the
clearer vehicle vs SBI-553 difference - exploratory, stated on the slides)."""
import json
import os

SUMMARY = os.path.join(r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis",
                       os.environ.get("CA_EVENT_OUT", "event_analysis"), "summary.json")
FALLBACK = {"stim": (0.5, 0.0, 0.0, 1.0), "beh": (1.0, 0.0, 0.0, 0.5)}
STIM = ("Light touch", "Mild touch", "Pin prick", "Heat")
BEH = ("withdrawal", "flinch", "movement onset", "escape/rearing onset", "grooming onset", "attending onset")
SHORT = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdrawal",
         "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
         "attending onset": "attending"}


def _all():
    try:
        return {ev: tuple(float(x) for x in w["1"]) for ev, w in json.load(open(SUMMARY, encoding="utf-8"))["windows"].items()}
    except (OSError, KeyError, ValueError):
        return {}


def of(ev):
    """(pre, gap, a, b) of one event"""
    return _all().get(ev, FALLBACK["stim" if ev in STIM else "beh"])


def stim():
    return of("Pin prick")


def beh():
    return of("withdrawal")


def _g(x):
    return f"{x:g}"


def mixed():
    return os.environ.get("WIN_MODE", "") == "mix"


def _family(evs, kr=False):
    """'0-0.5 s after vs the 0.5 s before' when the family shares one window, else per event"""
    ws = {ev: of(ev) for ev in evs}
    if len(set(ws.values())) == 1:
        w = next(iter(ws.values()))
        return (f"후 {_g(w[2])}-{_g(w[3])} s vs 직전 {_g(w[0])} s" if kr else
                f"{_g(w[2])}-{_g(w[3])} s after vs the {_g(w[0])} s before")
    groups = {}
    for ev, w in ws.items():
        groups.setdefault(w, []).append(SHORT[ev])
    parts = [(f"{', '.join(v)} 후 {_g(w[2])}-{_g(w[3])} s" if kr else f"{', '.join(v)} {_g(w[2])}-{_g(w[3])} s") for w, v in groups.items()]
    pre = next(iter(ws.values()))[0]
    return ("; ".join(parts) + (f" (직전 {_g(pre)} s 대비)" if kr else f" (vs the {_g(pre)} s before)"))


def en():
    return f"stimuli {_family(STIM)}; behaviours {_family(BEH if mixed() else BEH[:3])}"     # 10-24: mix = every behaviour


def kr():
    return f"자극 = {_family(STIM, True)}; 행동 = {_family(BEH if mixed() else BEH[:3], True)}"


def mix4():
    """10-25: marks AND window chosen per event (scoring_mix4.py)"""
    return os.environ.get("SCORING_SUFFIX", "_timefix") == "_mix4"


def prefix_en():
    if mix4():
        return ("marks and window per event (original or timing-corrected marks x 0-0.5 or 0-1 s, the combination with the most "
                "significant single-neuron + population tests; heat deliveries and withdrawal marks = original; exploratory)")
    return ("window per event (0-0.5 or 0-1 s, whichever showed the clearer vehicle vs SBI-553 difference; exploratory)"
            if mixed() else "best window (window search, checked on a held-out mouse)")


def prefix_kr():
    if mix4():
        return ("사건별 표시·창 (옛/보정 표시 x 0-0.5 또는 0-1 s 중 단일 뉴런 + 집단 검정에서 유의한 결과가 가장 많은 조합; heat 자극과 "
                "withdrawal 표시 = 옛 표시; 탐색적)")
    return ("사건별 창 (0-0.5 또는 0-1 s 중 vehicle vs SBI-553 차이가 더 뚜렷한 쪽; 탐색적)" if mixed()
            else "최적 창 (창 탐색, 빠진 마우스로 확인)")


def stim_span():
    s = stim()
    return f"{_g(s[2])}-{_g(s[3])} s"


def beh_span():
    b = beh()
    return f"{_g(b[2])}-{_g(b[3])} s"
