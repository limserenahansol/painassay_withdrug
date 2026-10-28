"""deck_event_slides.py - the slides of the event analysis (ca_event_analysis.py, 2026-10-22) for make_new_deck.py, Day 1 + Day 2
decks only: whole sessions (your basic plots 1-2), the response window (your 1/1 ... 10/10 s question + the exploration),
stimulus vs withdrawal vs flinch, responsive vs non-responsive neurons (plot 3), maps (plot 4), the fate of the baseline-
responsive neurons (plot 5), behaviour per stimulus (plot 6), behaviour vs neurons (plot 7). EN / KR; numbers in every
key sentence come from the analysis files (summary.json + the csv tables).
"""
import json
import os

import numpy as np
import pandas as pd

MO = ["5941", "5939", "5937", "1191"]
STIM = ["Light touch", "Mild touch", "Pin prick", "Heat"]
EVENTS = STIM + ["withdrawal", "flinch", "movement onset", "escape/rearing onset", "grooming onset"]
SHORT = {"Light touch": "light touch", "Mild touch": "mild touch", "Pin prick": "pin prick", "Heat": "heat", "withdrawal": "withdrawal",
         "flinch": "flinch", "movement onset": "movement onset", "escape/rearing onset": "escape / rearing", "grooming onset": "grooming", "attending onset": "attending"}
FILE = {"Light touch": "light", "Mild touch": "mild", "Pin prick": "pin", "Heat": "heat", "withdrawal": "withdrawal",
        "flinch": "flinch", "movement onset": "movement", "escape/rearing onset": "escape", "grooming onset": "grooming",
        "attending onset": "attending"}


def ptxt(p):
    if p is None or not np.isfinite(p):
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def star(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


class EA:
    """the event-analysis outputs of one folder"""

    def __init__(self, d, name="summary.json"):
        self.d = d
        js = os.path.join(d, name)
        self.J = json.load(open(js, encoding="utf-8")) if os.path.isfile(js) else {}

    def f(self, nm):
        return os.path.join(self.d, nm)

    def ok(self, nm):
        return os.path.isfile(self.f(nm))

    def csv(self, nm):
        p = self.f(nm)
        if not os.path.isfile(p):
            return pd.DataFrame()
        T = pd.read_csv(p)
        if "mouse" in T:
            T["mouse"] = T.mouse.astype(str)
        return T


def win_words(w, lg):
    pre, gap, a, b = w
    a0 = min(a, 0) - gap
    if lg == "kr":
        return f"사건 후 {a:g}-{b:g} s vs 직전 {pre:g} s" if a0 == 0 else f"사건 후 {a:g}-{b:g} s vs {a0 - pre:g}..{a0:g} s"
    return f"{a:g}-{b:g} s after vs the {pre:g} s before" if a0 == 0 else f"{a:g}-{b:g} s after vs {a0 - pre:g}..{a0:g} s"


# ------------------------------------------------------------------------------------------------ 1-2 whole sessions
def whole(dk, ea, L, short):
    J = ea.J
    st = J.get("plot1_stats", {})
    mice = [m for m in MO if ea.ok(f"P1_whole_{m}.png")] if not short else []     # PI: only the average over the mice
    for m in mice:
        S1 = st.get(m, {})
        ses = [k for k in ("d1_pre", "d1_post", "d2_pre", "d2_post") if k in S1]
        up = sum(S1[k]["blocks"] > S1[k]["no_stimulus"] for k in ses)
        dk.slide(L("WHOLE SESSIONS", "전체 세션"), L(f"{m}: every neuron through all four sessions", f"{m}: 네 세션 전체의 모든 뉴런"),
                 L(f"Activity is higher in the stimulus blocks than in the first 3 min in {up}/{len(ses)} sessions",
                   f"{len(ses)}개 세션 중 {up}개에서 자극 블록의 활동이 처음 3분보다 높음"),
                 ea.f(f"P1_whole_{m}.png"),
                 L(f"One row = one neuron ({S1.get('n_rows', '?')}; green strip = on both days, magenta = Day 1 only, cyan = Day 2 "
                   "only; grey = not there that day). Below: every scored event.",
                   f"한 줄 = 뉴런 하나 ({S1.get('n_rows', '?')}개; 초록 띠 = 두 날, 자홍 = Day 1만, 청록 = Day 2만; 회색 = 그날 없음). "
                   "아래: 채점한 모든 사건."),
                 L("Colour = z (dF/F / robust SD of the session), 1-s bins. Rows: neurons on both days first, then Day 1 only, then "
                   "Day 2 only; inside each group sorted by the mean activity of the two baselines. Event rows: deliveries (light, mild, "
                   "pin, heat), withdrawal, flinch (your marks), movement and grooming (automatic), attending, lick/bite, guarding, "
                   "escape (your scoring). Orange shading = the four stimulus blocks (180-360, 390-570, 600-780, 810-990 s). Mean "
                   "population z, first 3 min vs blocks: " + "; ".join(f"{k} {S1[k]['no_stimulus']:.2f} -> {S1[k]['blocks']:.2f}" for k in ses),
                   "색 = z (세션의 dF/F / robust SD), 1초 bin. 줄: 두 날 뉴런 먼저, 그다음 Day 1만, Day 2만; 그룹 안에서는 두 기준 세션의 "
                   "평균 활동 순. 사건 줄: 자극(light, mild, pin, heat), withdrawal, flinch (직접 표시), 움직임·그루밍 (자동), attending, "
                   "lick/bite, guarding, escape (직접 채점). 주황 음영 = 4개 자극 블록. 집단 평균 z, 처음 3분 -> 블록: "
                   + "; ".join(f"{k} {S1[k]['no_stimulus']:.2f} -> {S1[k]['blocks']:.2f}" for k in ses)))
    P2 = J.get("plot2", {})
    if ea.ok("P2_average_over_mice.png") and P2:
        rise = {c: P2[c]["blocks"] - P2[c]["no_stimulus"] for c in P2}
        dk.slide(L("WHOLE SESSIONS", "전체 세션"), L("The same, averaged over the 4 mice", "같은 그림, 4마리 평균"),
                 L("Population activity rises when the stimuli start: " + ", ".join(f"{c} {P2[c]['no_stimulus']:.2f} -> {P2[c]['blocks']:.2f} z"
                                                                                   for c in ("baseline", "vehicle", "SBI-553") if c in P2),
                   "자극이 시작되면 집단 활동이 오름: " + ", ".join(f"{ {'baseline': '기준'}.get(c, c)} {P2[c]['no_stimulus']:.2f} -> {P2[c]['blocks']:.2f} z"
                                                       for c in ("baseline", "vehicle", "SBI-553") if c in P2)),
                 ea.f("P2_average_over_mice_v2_colorbar.png" if (short and ea.ok("P2_average_over_mice_v2_colorbar.png"))
                      else "P2_average_over_mice.png"),          # 10-23: the PI deck has no colour-bar slide after it
                 L("Top: one row per mouse (mean of all its neurons, 5-s bins); bottom: mean +- SEM over the mice (thin = each mouse).",
                   "위: 마우스당 한 줄 (모든 뉴런 평균, 5초 bin); 아래: 마우스 평균 +- SEM (얇은 선 = 각 마우스)."),
                 L("Each mouse's neurons averaged first (so every mouse counts once), then the 4 mice. Baseline = mean of the "
                   "mouse's two baseline sessions; vehicle / SBI-553 = its injection session with that drug. Numbers in the key "
                   "sentence: first 3 min (no stimulus) -> the 4 stimulus blocks. Descriptive (4 mice). z is scaled per session "
                   "(robust SD), so absolute levels between sessions are only roughly comparable.",
                   "각 마우스의 뉴런을 먼저 평균(마우스마다 한 번씩), 그다음 4마리 평균. 기준 = 두 기준 세션 평균; vehicle / SBI-553 = 그 약물의 "
                   "주사 세션. 핵심 문장의 수치: 처음 3분(자극 없음) -> 4개 자극 블록. 기술 통계 (4마리). z는 세션마다 robust SD로 "
                   "나누므로 세션 간 절대 수준 비교는 대략적임."))
    V2 = ea.json("P2_versions.json") if hasattr(ea, "json") else {}
    if not V2:
        import json as _j, os as _o
        fp_ = _o.path.join(ea.d, "P2_versions.json")
        V2 = _j.load(open(fp_, encoding="utf-8")) if _o.path.isfile(fp_) else {}
    if ea.ok("P2_average_over_mice_v2_colorbar.png") and V2 and not short:      # 10-22: the same with its colour scale
        g = lambda k: V2.get(k, {})
        dk.slide(L("WHOLE SESSIONS", "전체 세션"), L("The same average, with its colour scale", "같은 평균, 색 척도 포함"),
                 L("Stimulus blocks: " + ", ".join(f"{lab} {g(k).get('blocks', float('nan')):.2f} z" for lab, k in
                                                   (("baseline", "baseline (both days, mean)"), ("vehicle", "after vehicle"), ("SBI-553", "after SBI-553"))),
                   "자극 블록: " + ", ".join(f"{lab} {g(k).get('blocks', float('nan')):.2f} z" for lab, k in
                                          (("기준", "baseline (both days, mean)"), ("vehicle", "after vehicle"), ("SBI-553", "after SBI-553")))),
                 ea.f("P2_average_over_mice_v2_colorbar.png"),
                 L("We averaged all neurons of each mouse (5-s bins; heatmap rows = mice, colour bar = population mean z), then the 4 "
                   "mice (bottom, mean +- SEM). n = 4 mice; descriptive.",
                   "마우스마다 모든 뉴런을 평균 (5초 bin; 히트맵 행 = 마우스, 색 막대 = 집단 평균 z), 그다음 4마리 (아래, 평균 +- SEM). 마우스 4; "
                   "기술 통계."),
                 L("Same data as the previous slide, drawn with a colour scale and units. z = dF/F / robust SD (MAD) of the session.",
                   "앞 슬라이드와 같은 데이터, 색 척도와 단위 포함. z = 세션의 dF/F / robust SD (MAD)."))
    if ea.ok("P2b_average_4panels_SBIday_vehicleday.png") and V2:                # 10-22: each injection next to its own day
        g = lambda k: V2.get(k, {}).get("blocks", float("nan"))
        dk.slide(L("WHOLE SESSIONS", "전체 세션"), L("Each injection next to the baseline of its own day", "각 주사를 같은 날 기준 옆에"),
                 L(f"Stimulus blocks: SBI-553 day {g('SBI-553 day: baseline'):.2f} -> {g('SBI-553 day: after SBI-553'):.2f} z; vehicle day "
                   f"{g('vehicle day: baseline'):.2f} -> {g('vehicle day: after vehicle'):.2f} z",
                   f"자극 블록: SBI-553 날 {g('SBI-553 day: baseline'):.2f} -> {g('SBI-553 day: after SBI-553'):.2f} z; vehicle 날 "
                   f"{g('vehicle day: baseline'):.2f} -> {g('vehicle day: after vehicle'):.2f} z"),
                 ea.f("P2b_average_4panels_SBIday_vehicleday.png"),
                 L("We put each injection session next to the baseline of the same day and mouse (crossover pairs). n = 4 mice per panel; "
                   "descriptive (z is scaled per session).",
                   "각 주사 세션을 같은 날·같은 마우스의 기준 세션 옆에 둠 (crossover 쌍). 패널당 마우스 4; 기술 통계 (z는 세션마다 척도)."),
                 L("5941 + 5939 got SBI-553 on Day 1 and vehicle on Day 2, 5937 + 1191 the reverse. Top: one row per mouse (mean of all "
                   "its neurons, 5-s bins); bottom: mean +- SEM over the 4 mice (thin = each mouse). Orange = stimulus blocks.",
                   "5941 + 5939는 Day 1에 SBI-553, Day 2에 vehicle; 5937 + 1191은 반대. 위: 마우스당 한 줄; 아래: 4마리 평균 +- SEM."))


# ------------------------------------------------------------------------------------------------ windows
def windows(dk, ea, L, short, lg):
    J = ea.J
    W = J.get("windows", {})
    if not W:
        return
    ws, wb = W["Heat"]["1"], W["withdrawal"]["1"]
    lomo = J.get("window_lomo", {})
    q_def = lomo.get("clean | default (2 s / 0-2 s)", np.nan)
    q_win = lomo.get(f"clean | {J.get('window_strategy', '')}", np.nan)
    q_long = lomo.get("all windows | one per event", np.nan)
    ratio = q_win / q_def if q_def and np.isfinite(q_def) and q_def > 0 else np.nan
    NW = J.get("named_windows", {})
    names = ["1/1", "2/2", "3/3", "5/5", "10/10"]
    mean_locked = [np.nanmean([NW[e]["locked"][i] for e in NW]) for i in range(5)] if NW else [np.nan] * 5
    best_named = names[int(np.nanargmax(mean_locked))] if NW else "?"
    c10 = np.nanmean([NW[e]["contamination"][4] for e in STIM if e in NW]) if NW else np.nan
    mv = NW.get("movement onset", {}).get("locked", [np.nan])[0]
    stim1 = np.nanmean([NW[e]["locked"][0] for e in STIM if e in NW]) if NW else np.nan
    if ea.ok("W2_named_windows.png"):
        dk.slide(L("RESPONSE WINDOW", "반응 시간 창"), L("Your windows: 1/1, 2/2, 3/3, 5/5 and 10/10 s", "말씀하신 창: 1/1, 2/2, 3/3, 5/5, 10/10 s"),
                 L(f"{best_named} s finds the most neurons locked to the event; at 10/10 s {c10:.0f}% of the trials already hold the next stimulus",
                   f"{best_named} s가 사건에 맞춰 반응하는 뉴런을 가장 많이 찾음; 10/10 s에서는 trial의 {c10:.0f}%에 이미 다음 자극이 들어 있음"),
                 ea.f("W2_named_windows.png"),
                 L(f"A delivery comes every ~3 s, so long windows mix several stimuli. Chosen: stimuli {win_words(ws, 'en')}, behaviours "
                   f"{win_words(wb, 'en')} ({ratio:.1f}x the old window in a held-out mouse). Clearest: movement onset ({mv:.0f}% of "
                   f"neurons beyond chance at 1/1 s; stimuli {stim1:.1f}%).",
                   f"자극이 약 3초마다 와서 긴 창은 여러 자극이 섞임. 선택: 자극 {win_words(ws, 'kr')}, 행동 {win_words(wb, 'kr')} "
                   f"(검증 마우스에서 기존 창의 {ratio:.1f}배). 가장 뚜렷한 것: 움직임 시작 (1/1 s에서 우연보다 많은 뉴런 {mv:.0f}%; 자극 {stim1:.1f}%)."),
                 L("For every window and session: the response of each neuron (mean z in the window after minus the window before) "
                   "on the ODD trials and on the EVEN trials (Wilcoxon over trials, p < 0.05 in both halves, same sign) = replicated "
                   "neurons. The same for fake events: A = every event moved 2-5 s earlier or later (keeps the block, the state and "
                   "the stimulus density; tests that the response is locked to the event itself); B = the event train shifted inside "
                   "its block (tests any change, slow ones included). Curves = mean over the 16 sessions of (real - fake), % of "
                   "neurons. C = % of trials with another delivery inside the window. Long windows look good in B because the "
                   "whole stimulation bout (several deliveries, the movement they cause) changes the activity for seconds - that "
                   "is not the response to one stimulus.",
                   "창과 세션마다: 각 뉴런의 반응(사건 후 창 평균 z - 사건 전 창)을 홀수 trial과 짝수 trial에서 따로 검정 (trial 간 "
                   "Wilcoxon, 두 절반 모두 p < 0.05, 같은 방향) = 재현된 뉴런. 가짜 사건으로 같은 계산: A = 사건마다 2-5 s 앞뒤로 옮김 "
                   "(블록·상태·자극 밀도는 유지, 사건 자체에 맞춘 반응인지 검사); B = 사건 열을 블록 안에서 이동 (느린 변화 포함 모든 "
                   "변화 검사). 곡선 = 16개 세션 평균 (실제 - 가짜), 뉴런 %. C = 창 안에 다른 자극이 있는 trial %. 긴 창이 B에서 "
                   "좋아 보이는 것은 자극 묶음 전체(여러 자극, 그에 따른 움직임)가 몇 초 동안 활동을 바꾸기 때문 - 자극 하나에 대한 "
                   "반응이 아님."))
    if not short and ea.ok("W1_window_search.png"):
        dk.slide(L("RESPONSE WINDOW", "반응 시간 창"), L("Exploring start, end, length and delay of the window", "창의 시작, 끝, 길이, 지연 탐색"),
                 L(f"Best: stimuli {win_words(ws, 'en')}; behaviours {win_words(wb, 'en')} ({ratio:.1f}x the old window)",
                   f"최적: 자극 {win_words(ws, 'kr')}; 행동 {win_words(wb, 'kr')} (기존 창의 {ratio:.1f}배)"),
                 ea.f("W1_window_search.png"),
                 L("Each cell = one window (rows: baseline length before the event; columns: response window). Faded = longer than "
                   "the ~3 s between deliveries (not used). Red box = the choice; x = the old window; checked on a mouse not used to choose.",
                   "칸 하나 = 창 하나 (행: 사건 전 기준 구간 길이; 열: 반응 창). 흐린 칸 = 자극 간격(~3 s)보다 긴 창 (사용 안 함). 빨간 상자 = "
                   "선택; x = 기존 창; 선택에 쓰지 않은 마우스에서 검증."),
                 L(f"Colour = replicated neurons, real - jitter sham (% of neurons, mean of 16 sessions), as on the previous slide. "
                   f"Bottom right: the window was chosen on 3 mice and scored on the 4th, for the four ways to choose (the old "
                   f"window; one window per family = stimuli / behaviours; one per event; allowing 10-s windows). Held-out "
                   f"quality: old {100 * q_def:.1f}%, chosen {100 * q_win:.1f}%, with 10-s windows {100 * q_long:.1f}% (those mix the next "
                   f"stimuli, see the slow-change slide). Final windows - stimuli: {win_words(ws, 'en')}; withdrawal, flinch, "
                   f"movement, escape, grooming: {win_words(wb, 'en')}. Same window on both days.",
                   f"색 = 재현 뉴런, 실제 - jitter 가짜 (뉴런 %, 16개 세션 평균). 오른쪽 아래: 3마리로 창을 고르고 나머지 1마리로 점수 "
                   f"(기존 창; 자극/행동 각각 한 창; 사건마다 한 창; 10초 창 허용). 검증 점수: 기존 {100 * q_def:.1f}%, 선택 {100 * q_win:.1f}%, "
                   f"10초 창 허용 {100 * q_long:.1f}% (다음 자극이 섞임). 최종 창 - 자극: {win_words(ws, 'kr')}; withdrawal, flinch, "
                   f"움직임, escape, 그루밍: {win_words(wb, 'kr')}. 두 날 같은 창."))
    if not short and ea.ok("W1b_window_search_slow.png"):
        bs = J.get("window_best_slow", {})
        pin = bs.get("Pin prick", {})
        dk.slide(L("RESPONSE WINDOW", "반응 시간 창"), L("Long windows: slow changes, not single responses", "긴 창: 하나의 반응이 아니라 느린 변화"),
                 L(f"Against the block sham, 10-s windows win (pin prick: {100 * pin.get('slow', np.nan):.1f}% of neurons) - the whole stimulation bout",
                   f"블록 가짜와 비교하면 10초 창이 이김 (pin prick: 뉴런 {100 * pin.get('slow', np.nan):.1f}%) - 자극 묶음 전체의 효과"),
                 ea.f("W1b_window_search_slow.png"),
                 L("Same grid, scored against the block sham (any change, slow ones included) instead of the 2-5 s jitter.",
                   "같은 격자, 2-5 s jitter 대신 블록 가짜(느린 변화 포함 모든 변화)와 비교."),
                 L("A 10-s window after a delivery contains ~3 more deliveries of the same stimulus and the movement / arousal they "
                   "cause, so it measures the bout, not the single stimulus; kept as a separate result, not used for the neuron classes.",
                   "자극 후 10초 창에는 같은 자극이 ~3번 더 있고 그에 따른 움직임·각성이 들어 있어, 자극 하나가 아니라 묶음을 잼; "
                   "뉴런 분류에는 쓰지 않고 따로 보고함."))
    tim = J.get("timing", {})
    if not short and ea.ok("W3_timing.png") and tim:
        l1, l2 = tim["wd_latency_median"]["1"], tim["wd_latency_median"]["2"]
        nf = tim.get("n_set_by_fix", {})
        nf1, nf2 = nf.get("1", 0), nf.get("2", 0)
        dk.slide(L("RESPONSE WINDOW", "반응 시간 창"), L("When do the responses happen?", "반응은 언제 일어나는가?"),
                 (L(f"Withdrawal follows the delivery by {l2:.2f} s on Day 2 (clicker, measured); on Day 1 {nf1} deliveries sit 0.07 s "
                    f"before their withdrawal by the 10-23 fix",
                    f"withdrawal은 Day 2 (클리커, 측정) 자극 {l2:.2f} s 뒤; Day 1은 자극 {nf1}개가 10-23 보정으로 withdrawal 0.07 s 전")
                  if nf1 else                                  # 10-24: the old-marks variant (no deliveries moved)
                  L(f"Withdrawal follows the delivery mark by {l1:.2f} s on Day 1 (key taps) and {l2:.2f} s on Day 2 (clicker); marks "
                    f"before the 10-23 timing fix",
                    f"withdrawal은 자극 표시 뒤 Day 1 (키) {l1:.2f} s, Day 2 (클리커) {l2:.2f} s; 10-23 타이밍 보정 전 표시")),
                 ea.f("W3_timing.png"),
                 L(f"Day 1: the stimulus and the withdrawal cannot be split by time (the remaining Day-1 deliveries: {l1:.2f} s). n = 16 "
                   f"sessions, 4 mice; descriptive (no test).",
                   f"Day 1: 자극과 withdrawal을 시간으로 나눌 수 없음 (나머지 Day 1 자극: {l1:.2f} s). n = 세션 16개, 4마리; 기술 통계 (검정 없음)."),
                 L(f"A: latency from each delivery mark to the first withdrawal within 2 s, deliveries the timing fix left where they "
                   f"were (Day 1 {nf1}, Day 2 {nf2} were set to 0.07 s before their reflex and are not shown). B-D: mean trace of the "
                   "neurons activated on the ODD trials (chosen window, p < 0.01), drawn on the EVEN trials so the selection cannot "
                   "make the curve; mean +- SEM over neurons. Movement-onset neurons peak "
                   f"{tim.get('peak_move_d1', np.nan):.2f} s (Day 1) and {tim.get('peak_move_d2', np.nan):.2f} s (Day 2) after the onset.",
                   f"A: 자극 표시부터 2 s 안 첫 withdrawal까지 지연, 보정이 옮기지 않은 자극만 (Day 1 {nf1}개, Day 2 {nf2}개는 반사 0.07 s 전으로 "
                   f"옮겨져 제외). B-D: 홀수 trial에서 활성화된 뉴런(선택 창, p < 0.01)의 "
                   "평균 trace를 짝수 trial로 그림 (선택이 곡선을 만들지 못하게); 뉴런 평균 +- SEM. 움직임 시작 뉴런의 최고점: "
                   f"Day 1 {tim.get('peak_move_d1', np.nan):.2f} s, Day 2 {tim.get('peak_move_d2', np.nan):.2f} s."))


# ------------------------------------------------------------------------------------------------ stimulus vs withdrawal
def overlap(dk, ea, L, short):
    O = ea.J.get("overlap", {})
    if not O or not ea.ok("O1_near_simultaneous.png"):
        return
    sp = O.get("split", {})
    td = O.get("types_by_day", {})
    keep = 100 * sp.get("frac_keep_half", np.nan)
    d1, d2 = td.get("1", {}), td.get("2", {})
    cs = "stimulus or reflex (cannot separate)"
    tim = ea.J.get("timing", {}).get("wd_latency_median", {})
    dk.slide(L("STIMULUS vs WITHDRAWAL", "자극 vs WITHDRAWAL"), L("Stimulus, withdrawal and flinch come almost together", "자극, withdrawal, flinch가 거의 동시에 옴"),
             L(f"{keep:.0f}% of the stimulus-activated neurons keep at least half of their response when no withdrawal follows",
               f"자극에 활성화된 뉴런의 {keep:.0f}%는 withdrawal이 없어도 반응의 절반 이상을 유지"),
             ea.f("O1_near_simultaneous.png"),
             L(f"So most are not just the withdrawal movement. The model splits them better on Day 2 (clicker, {tim.get('2', np.nan):.2f} s "
               f"gap): stimulus-only {d2.get('stimulus', 0):.0f}% vs {d1.get('stimulus', 0):.0f}% on Day 1; movement alone explains {d2.get('movement only', 0):.0f}-{d1.get('movement only', 0):.0f}%.",
               f"즉 대부분은 withdrawal 움직임만이 아님. 모델은 Day 2 (클리커, 간격 {tim.get('2', np.nan):.2f} s)에서 더 잘 나눔: 자극만 "
               f"{d2.get('stimulus', 0):.0f}% vs Day 1 {d1.get('stimulus', 0):.0f}%; 움직임만으로 설명되는 뉴런 {d2.get('movement only', 0):.0f}-{d1.get('movement only', 0):.0f}%."),
             L(f"A: neuron x stimulus pairs classified activated by that stimulus; mean response on the deliveries followed by a "
               f"withdrawal (within 2 s or before the next delivery) vs those without (n = {sp.get('n', 0)}; with > without, "
               f"Wilcoxon {ptxt(sp.get('p', np.nan))}; median {sp.get('median_with', np.nan):.2f} vs {sp.get('median_without', np.nan):.2f} z). "
               "B-D: encoding model per session (ridge regression at 10 Hz, 5-fold blocks of 20 s, 3-fold CV): 0-3 s kernels "
               "(0.5-s steps) for each stimulus type, withdrawal, flinch, movement onset, escape onset + movement and grooming "
               "states + the schedule blocks. Unique variance of a family = cross-validated R2 of the full model minus the model "
               "without it; tested against the same family shifted inside its blocks (100 shifts), BH q <= 0.20 per session. "
               "'Cannot separate' = stimulus + reflex together explain the neuron but neither alone (they are too close in time). "
               f"Day 1 cannot separate: {d1.get(cs, 0):.0f}%, Day 2: {d2.get(cs, 0):.0f}%.",
               f"A: 그 자극에 활성화로 분류된 뉴런 x 자극 쌍; withdrawal이 뒤따른 자극(2 s 안 또는 다음 자극 전)과 아닌 자극의 평균 반응 "
               f"(n = {sp.get('n', 0)}; with > without, Wilcoxon {ptxt(sp.get('p', np.nan))}; 중앙값 {sp.get('median_with', np.nan):.2f} vs "
               f"{sp.get('median_without', np.nan):.2f} z). B-D: 세션별 encoding 모델 (10 Hz ridge 회귀, 20초 블록 3-fold 교차검증): 자극 종류별, "
               "withdrawal, flinch, 움직임 시작, escape 시작의 0-3 s 커널 (0.5 s 단위) + 움직임·그루밍 상태 + 일정 블록. 고유 분산 = "
               "전체 모델의 교차검증 R2 - 그 묶음을 뺀 모델; 그 묶음을 블록 안에서 이동시킨 가짜(100번)와 비교, 세션 안 BH q <= 0.20. "
               "'나눌 수 없음' = 자극 + 반사를 합치면 설명되지만 하나만으로는 안 됨 (시간이 너무 가까움). "
               f"Day 1 나눌 수 없음 {d1.get(cs, 0):.0f}%, Day 2 {d2.get(cs, 0):.0f}%."))


# ------------------------------------------------------------------------------------------------ plot 3-5
def categories(dk, ea, L, short):
    J = ea.J
    R3 = ea.csv("P3a_responsive_pct.csv")
    if len(R3) and ea.ok("P3a_responsive_pct.png"):
        pc = R3[R3.pct_resp.notna()]
        base = pc[pc.condition == "baseline"].set_index("event")
        top = base.pct_resp.idxmax()
        fs = R3[R3.p_fisher.notna()] if "p_fisher" in R3 else pd.DataFrame()
        sig = fs[fs.p_fisher < 0.05].event.tolist() if len(fs) else []
        P = pc.pivot(index="event", columns="condition", values="pct_resp")
        txt = ", ".join(f"{SHORT[e]} (vehicle {P.loc[e, 'vehicle']:.0f}%, SBI-553 {P.loc[e, 'SBI-553']:.0f}%)" for e in sig if e in P.index)
        mc = R3[R3.p_mcnemar.notna()] if "p_mcnemar" in R3 else pd.DataFrame()
        up_inj = int(((mc.p_mcnemar < 0.05) & (mc.inj_pct > mc.base_pct)).sum()) if len(mc) else 0
        n_inj = int(len(mc))
        # the key: the event whose within-day change (same neurons, baseline -> injection) differs most between the drugs,
        # preferring opposite directions with at least one significant change
        best = None
        for ev in EVENTS:
            v = mc[(mc.condition == "vehicle: baseline -> injection") & (mc.event == ev)] if len(mc) else pd.DataFrame()
            s_ = mc[(mc.condition == "SBI-553: baseline -> injection") & (mc.event == ev)] if len(mc) else pd.DataFrame()
            if not len(v) or not len(s_):
                continue
            v, s_ = v.iloc[0], s_.iloc[0]
            dv, ds = v.inj_pct - v.base_pct, s_.inj_pct - s_.base_pct
            sc = abs(dv - ds) + (100 if (np.sign(dv) != np.sign(ds) and min(v.p_mcnemar, s_.p_mcnemar) < 0.05) else 0)
            if best is None or sc > best[0]:
                best = (sc, ev, v, s_)
        if best:
            _, ev, v, s_ = best
            key = L(f"{SHORT[ev]}: responsive neurons {v.base_pct:.0f} -> {v.inj_pct:.0f}% after vehicle {star(v.p_mcnemar)}, "
                    f"{s_.base_pct:.0f} -> {s_.inj_pct:.0f}% after SBI-553 {star(s_.p_mcnemar)} (same neurons, same day)",
                    f"{SHORT[ev]}: 반응 뉴런 vehicle 후 {v.base_pct:.0f} -> {v.inj_pct:.0f}% {star(v.p_mcnemar)}, SBI-553 후 "
                    f"{s_.base_pct:.0f} -> {s_.inj_pct:.0f}% {star(s_.p_mcnemar)} (같은 날 같은 뉴런)")
        else:
            key = L(f"Most at baseline: {SHORT[top]} ({base.loc[top, 'pct_resp']:.0f}%)", f"기준에서 가장 많음: {SHORT[top]} ({base.loc[top, 'pct_resp']:.0f}%)")
        dk.slide(L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L("Responsive vs non-responsive, with the chosen window",
                                                                      "반응 vs 무반응, 선택한 창으로"), key,
                 ea.f("P3a_responsive_pct.png"),
                 L(f"Red = activated, blue = inhibited, the rest = non-responsive. Heat, withdrawal and movement onset gain responsive neurons after BOTH "
                   f"drugs ({up_inj}/{n_inj} tests up from the same day's baseline); most responsive overall: {SHORT[top]}. "
                   "Per mouse and with the strict test: see check 4.",
                   f"빨강 = 활성화, 파랑 = 억제, 나머지 = 무반응. heat, withdrawal, 움직임 시작에서 두 약물 모두 주사 후 반응 뉴런이 늘어남 (같은 날 기준보다 "
                   f"증가 {up_inj}/{n_inj}); 가장 많이 반응하는 사건: {SHORT[top]}. 마우스별·엄격한 검정: 검증 4 참고."),
                 L("Responsive = within-block time-shift test per neuron (1000 shuffles) with the chosen window, BH q <= 0.20 in the "
                   "session (as before); activated / inhibited by the sign. Bars = all neurons of that condition pooled (baseline = "
                   "both baseline sessions), markers = mice. Each bracket shows the pair tested: blue (base-veh) / red (base-SBI) = that "
                   "injection vs the same neurons' baseline of the same day (McNemar); black (veh-SBI) = SBI-553 vs vehicle injection "
                   "sessions (Fisher exact, neurons pooled, exploratory; the baselines of the two drug days differ too, so read the "
                   "coloured brackets first). ns = not significant. SBI-553 vs vehicle (Fisher): "
                   + (txt or "none") + ". * p < 0.05, ** p < 0.01, "
                   "*** p < 0.001. Caution: 'inhibited' responses that drop faster than calcium can fall come with a brightening "
                   "of the field around the cells (10-21 shape analysis) and may be background artefacts; the session-to-session "
                   "jumps (0% vs 50% in the same mouse) come from responses shared by most neurons of a session.",
                   "반응 = 선택 창으로 뉴런별 블록 내 시간 이동 검정 (1000번), 세션 안 BH q <= 0.20 (이전과 같음); 방향으로 활성화/억제. "
                   "막대 = 그 조건의 모든 뉴런 (기준 = 두 기준 세션), 표식 = 마우스. 괄호 = 비교한 두 조건: 파랑(base-veh)·빨강(base-SBI) "
                   "= 같은 날 같은 뉴런의 기준 대비 (McNemar); 검정(veh-SBI) = SBI-553 vs vehicle (Fisher exact, 뉴런 합침, 탐색적). * p < 0.05, ** p < 0.01, *** p < 0.001. 주의: 칼슘보다 "
                   "빨리 떨어지는 '억제'는 주변 시야가 밝아지는 순간과 같이 나타나 (10-21 형태 분석) 배경 artefact일 수 있음; 같은 마우스에서 "
                   "세션마다 0% vs 50%로 튀는 것은 세션의 거의 모든 뉴런이 함께 반응하기 때문."))
    if not short and ea.ok("P3e_sessions.png"):
        T = ea.csv("cells_classes.csv")
        txt = ("", "")
        if len(T):
            T["r"] = T.cls.isin(["activated", "inhibited"])
            Pm = T.groupby(["event", "mouse", "day", "phase"]).r.mean().unstack(["day", "phase"]) * 100
            bb = (Pm[(2, "pre")] - Pm[(1, "pre")]).abs().mean()
            bi = pd.concat([(Pm[(1, "post")] - Pm[(1, "pre")]).abs(), (Pm[(2, "post")] - Pm[(2, "pre")]).abs()]).mean()
            txt = (f"Baseline -> baseline (Day 1 -> Day 2) changes {bb:.0f} points on average, baseline -> injection {bi:.0f} points",
                   f"기준 -> 기준 (Day 1 -> Day 2) 변화 평균 {bb:.0f}%p, 기준 -> 주사 {bi:.0f}%p")
        dk.slide(L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L("Responsive neurons session by session", "세션별 반응 뉴런"),
                 L(*txt), ea.f("P3e_sessions.png"),
                 L("One line per mouse through its four sessions in the real order; dot colour = condition (grey baseline, blue "
                   "vehicle, red SBI-553).", "마우스당 한 선, 실제 순서의 네 세션; 점 색 = 조건 (회색 기준, 파랑 vehicle, 빨강 SBI-553)."),
                 L("% of the mouse's neurons responsive (activated + inhibited) in each session. Points = mean absolute change over "
                   "mice and events. If the two baselines differ as much as baseline vs injection, a drug effect cannot be told "
                   "apart from day-to-day changes with these mice.",
                   "각 세션에서 마우스 뉴런 중 반응(활성화 + 억제) %. 수치 = 마우스와 사건에 대한 평균 절대 변화. 두 기준 세션의 차이가 "
                   "기준 vs 주사만큼 크면 이 마우스들로는 약물 효과와 날짜 간 변화를 구분할 수 없음."))
    tr = J.get("plot3_transitions", {})
    if not short and ea.ok("P3b_transitions.png") and tr:
        sv, ss = tr["stay"]["vehicle"], tr["stay"]["SBI-553"]
        pv, ps = 100 * sv[0] / max(sv[1], 1), 100 * ss[0] / max(ss[1], 1)
        nv, ns = tr["stay"]["vehicle_new"], tr["stay"]["SBI-553_new"]
        dk.slide(L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L("The same neurons from baseline to injection", "같은 뉴런, 기준에서 주사까지"),
                 L(f"Responsive at baseline -> still responsive: {pv:.0f}% after vehicle vs {ps:.0f}% after SBI-553 ({ptxt(tr.get('p_stay', np.nan))})",
                   f"기준에서 반응 -> 주사 후에도 반응: vehicle {pv:.0f}% vs SBI-553 {ps:.0f}% ({ptxt(tr.get('p_stay', np.nan))})"),
                 ea.f("P3b_transitions.png"),
                 L(f"Newly responsive after the injection: {100 * nv[0] / max(nv[1], 1):.0f}% (vehicle) vs {100 * ns[0] / max(ns[1], 1):.0f}% "
                   f"(SBI-553) of the non-responsive ({ptxt(tr.get('p_new', np.nan))}).",
                   f"주사 후 새로 반응: 무반응이던 뉴런 중 vehicle {100 * nv[0] / max(nv[1], 1):.0f}% vs SBI-553 {100 * ns[0] / max(ns[1], 1):.0f}% "
                   f"({ptxt(tr.get('p_new', np.nan))})."),
                 L("Every neuron x event pair of a day: its class at the baseline session (rows) and at the injection session of the "
                   "same day (colours). All events pooled. Fisher exact SBI-553 vs vehicle on the stay / new counts (pairs pooled, "
                   "exploratory).",
                   "하루의 모든 뉴런 x 사건 쌍: 기준 세션의 분류(행)와 같은 날 주사 세션의 분류(색). 모든 사건 합침. 유지/신규 수에 대한 "
                   "Fisher exact SBI-553 vs vehicle (쌍 합침, 탐색적)."))
    Sb = ea.csv("P3c_stability.csv")
    if not short and ea.ok("P3c_stability_overlap.png") and len(Sb):
        k, n = int(Sb.same_on_d2.sum()), int(Sb.d1_resp.sum())
        ch = float((Sb.chance * Sb.d1_resp).sum() / max(n, 1))
        Jm = pd.DataFrame(J.get("plot3_overlap", {}))
        pairs_ = []
        if len(Jm):
            for i, e1 in enumerate(Jm.index):
                for e2 in Jm.columns[i + 1:]:
                    if e1 != e2 and np.isfinite(Jm.loc[e1, e2]):
                        pairs_.append((Jm.loc[e1, e2], e1, e2))
        pairs_.sort(reverse=True)
        bp = pairs_[0] if pairs_ else (np.nan, "", "")
        dk.slide(L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L("Stable across days, shared between events", "날짜를 넘어 안정적, 사건 사이 공유"),
                 L(f"{k}/{n} Day-1 responsive neurons are the same class again on Day 2 ({100 * k / max(n, 1):.0f}% vs {ch:.0f}% by chance)",
                   f"Day 1 반응 뉴런 {k}/{n}개가 Day 2에도 같은 분류 ({100 * k / max(n, 1):.0f}% vs 우연 {ch:.0f}%)"),
                 ea.f("P3c_stability_overlap.png"),
                 L(f"B: the largest overlap is {SHORT.get(bp[1], bp[1])} with {SHORT.get(bp[2], bp[2])} ({bp[0]:.0f}% of the neurons "
                   "responsive to either respond to both).",
                   f"B: 가장 많이 겹치는 것은 {SHORT.get(bp[1], bp[1])}와 {SHORT.get(bp[2], bp[2])} (둘 중 하나에 반응하는 뉴런의 "
                   f"{bp[0]:.0f}%가 둘 다에 반응)."),
                 L("A: neurons on both days; of those responsive at the Day-1 baseline, % with the same class (activated / inhibited) "
                   "at the Day-2 baseline (no drug difference between the two); grey = expected by chance from the Day-2 class "
                   "frequencies; bracket = green vs grey (binomial test above chance). B: baseline sessions; overlap = neurons responsive to both events "
                   "/ neurons responsive to either (Jaccard, %).",
                   "A: 두 날 모두 있는 뉴런; Day 1 기준에서 반응한 뉴런 중 Day 2 기준에서도 같은 분류(활성화/억제)인 % (둘 다 약 없음); "
                   "회색 = Day 2 분류 빈도로 본 우연 기댓값; 괄호 = 초록 vs 회색 (이항 검정, 우연보다 높음). B: 기준 세션; 겹침 = 두 사건 모두 반응 / "
                   "어느 하나라도 반응 (Jaccard, %)."))
    if not short:
        for ev in ["Heat", "Pin prick", "withdrawal", "movement onset"]:
            event_heatmap(dk, ea, L, ev)
        Tp = J.get("plot4", {})
        if ea.ok("P4_topography.png") and Tp:
            dk.slide(L("WHERE ARE THEY", "위치"), L("Are the responsive neurons clustered in the field of view?", "반응 뉴런이 시야에서 모여 있는가?"),
                     L(f"{Tp['n_sig']} of {Tp['n_tests']} session x event maps are clustered (p < 0.05); {Tp['expected']:.1f} expected by chance",
                       f"세션 x 사건 지도 {Tp['n_tests']}개 중 {Tp['n_sig']}개가 모여 있음 (p < 0.05); 우연 기댓값 {Tp['expected']:.1f}개"),
                     ea.f("P4_topography.png"),
                     L("Dot = one session x event; above 0 = responsive neurons closer together than random neurons of the same session.",
                       "점 = 세션 x 사건 하나; 0보다 위 = 반응 뉴런이 같은 세션의 무작위 뉴런보다 가까이 모여 있음."),
                     L("Mean nearest-neighbour distance of the responsive neurons vs the same number of neurons drawn at random from "
                       "the session (500 draws): z = (random - observed) / SD of random; black edge = p < 0.05 (one-sided). Colour = "
                       "condition. The maps per mouse are in the supplement.",
                       "반응 뉴런의 평균 최근접 거리 vs 같은 세션에서 같은 수를 무작위로 뽑은 뉴런 (500번): z = (무작위 - 관찰) / 무작위 SD; "
                       "검은 테두리 = p < 0.05 (단측). 색 = 조건. 마우스별 지도는 부록."))
    mc = J.get("motion_check", {})
    if not short and ea.ok("Q1_motion_check.png") and mc:
        dk.slide(L("DATA QUALITY", "데이터 품질"), L("Why does '% responsive' jump between sessions?", "왜 '% 반응'이 세션마다 튀는가?"),
                 L(f"It goes with the field brightening (rho {mc['field']['rho']:+.2f}) and the shared response of all neurons (rho "
                   f"{mc['abs_population']['rho']:+.2f}); more brain motion goes with fewer, not more (rho {mc['shift']['rho']:+.2f})",
                   f"시야 밝아짐(rho {mc['field']['rho']:+.2f})과 모든 뉴런의 공통 반응(rho {mc['abs_population']['rho']:+.2f})과 같이 가고, "
                   f"뇌 움직임이 클수록 오히려 적음 (rho {mc['shift']['rho']:+.2f})"),
                 ea.f("Q1_motion_check.png"),
                 L("When most neurons of a session move together at an event, the per-neuron tests flag many of them at once; "
                   "the registration shift changes by < 0.3 px in 122/131 cases (max 0.6 px).",
                   "한 세션의 뉴런 대부분이 사건에서 함께 움직이면 뉴런별 검정이 한꺼번에 많이 잡음; 정합 이동은 131개 중 122개에서 0.3 px 미만 (최대 0.6 px)."),
                 L("One dot = one session x event (16 sessions x up to 9 events), y = % neurons responsive (chosen window). x: event-"
                   "locked change (response window - baseline window) of the motion-correction shift, the frame-to-template "
                   "correlation, the field brightness outside every footprint (robust z), and |the mean response of all neurons|. "
                   "Spearman rho. With the field regressed out of every neuron (analysis\\event_analysis_bgreg) the jumps stay, so "
                   "they are not only optical; part may be real shared (arousal) activity.",
                   "점 하나 = 세션 x 사건, y = 반응 뉴런 % (선택 창). x: 사건에 맞춘 변화 (반응 창 - 기준 창): 움직임 보정 이동, 프레임-템플릿 "
                   "상관, 모든 footprint 밖 시야 밝기 (robust z), 모든 뉴런 평균 반응의 절댓값. Spearman rho. 모든 뉴런에서 시야 신호를 "
                   "회귀로 빼도 (analysis\\event_analysis_bgreg) 튀는 현상은 남아서 광학만의 문제는 아님; 일부는 실제 공통(각성) 활동일 수 있음."))
    F = ea.csv("P5_fate_tests.csv")
    if len(F) and ea.ok("P5_fate.png"):
        ch = F[F.drug == "SBI vs vehicle change"].dropna(subset=["p"])
        main_ = F[F.drug.isin(["vehicle", "SBI-553"])]
        if len(ch):
            r = ch.sort_values("p").iloc[0]
            v = main_[(main_.event == r.event) & (main_.group == r.group) & (main_.drug == "vehicle")]
            s_ = main_[(main_.event == r.event) & (main_.group == r.group) & (main_.drug == "SBI-553")]
            key = L(f"Neurons {r.group} by {SHORT[r.event]} at baseline: {v.base.iloc[0]:+.2f} -> {v.inj.iloc[0]:+.2f} z with vehicle, "
                    f"{s_.base.iloc[0]:+.2f} -> {s_.inj.iloc[0]:+.2f} z with SBI-553 (change differs: {ptxt(r.p)})",
                    f"기준에서 {SHORT[r.event]}에 {'활성화' if r.group == 'activated' else '억제'}된 뉴런: vehicle {v.base.iloc[0]:+.2f} -> "
                    f"{v.inj.iloc[0]:+.2f} z, SBI-553 {s_.base.iloc[0]:+.2f} -> {s_.inj.iloc[0]:+.2f} z (변화 차이 {ptxt(r.p)})") \
                if len(v) and len(s_) else ""
        else:
            key = ""
        dk.slide(L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L("What happens to the baseline-responsive neurons after the injection", "기준 반응 뉴런은 주사 후 어떻게 되는가"),
                 key, ea.f("P5_fate.png"),
                 L("Neurons chosen on half of the baseline trials and measured on the other half, so 'getting smaller' is not just "
                   "regression to the mean.", "기준 trial 절반으로 뉴런을 고르고 나머지 절반으로 잼 - '작아짐'이 평균 회귀 때문만이 아님."),
                 L("Per day: neurons activated (top) or inhibited (bottom) at baseline on the ODD trials (Wilcoxon p < 0.05, chosen "
                   "window); their mean response on the EVEN baseline trials vs all trials after the injection of the same day; mean "
                   "+- SEM over neurons; blue = vehicle days, red = SBI-553 days. Coloured bracket = baseline vs injection of the "
                   "same neurons (Wilcoxon signed-rank); black bracket = the change after vehicle vs after SBI-553 (Mann-Whitney). * p < 0.05, ** p < "
                   "0.01, *** p < 0.001; neurons nested in 4 mice (exploratory).",
                   "날마다: 기준의 홀수 trial에서 활성화(위) 또는 억제(아래)된 뉴런 (Wilcoxon p < 0.05, 선택 창); 짝수 기준 trial의 평균 반응 vs "
                   "같은 날 주사 후 모든 trial; 뉴런 평균 +- SEM; 파랑 = vehicle 날, 빨강 = SBI-553 날. 색 괄호 = 같은 뉴런의 기준 vs 주사 (Wilcoxon); "
                   "검은 괄호 = vehicle 후 변화 vs SBI-553 후 변화 (Mann-Whitney). * p < 0.05, ** p < 0.01, *** p < 0.001; 뉴런은 4마리에 속함 (탐색적)."))


def event_heatmap(dk, ea, L, ev, section=None):
    fn = f"P3d_{FILE[ev]}.png"
    if not ea.ok(fn):
        return
    T = ea.csv("cells_classes.csv")
    X = T[T.event == ev]

    def pc(c):
        Y = X[X.condition == c]
        return f"{100 * (Y.cls == 'activated').mean():.0f} / {100 * (Y.cls == 'inhibited').mean():.0f}%" if len(Y) else "n.a."
    dk.slide(section or L("RESPONSIVE vs NON-RESPONSIVE", "반응 vs 무반응"), L(f"{SHORT[ev]}: every neuron by its class", f"{SHORT[ev]}: 분류별 모든 뉴런"),
             L(f"Activated / inhibited: baseline {pc('baseline')}, vehicle {pc('vehicle')}, SBI-553 {pc('SBI-553')}",
               f"활성화 / 억제: 기준 {pc('baseline')}, vehicle {pc('vehicle')}, SBI-553 {pc('SBI-553')}"),
             ea.f(fn),
             L("Top: one row per neuron-session, sorted activated / inhibited / non-responsive; bottom: the mean trace of each class.",
               "위: 뉴런-세션당 한 줄, 활성화 / 억제 / 무반응 순; 아래: 분류별 평균 trace."),
             L("Colour = trial-averaged z, each trial baselined on [-3, 0) s for the display, 0.2-s smoothing; grey band = the "
               "response window. Classes from the time-shift test with the chosen window (q <= 0.20); the traces use the same "
               "trials as the classes, so a dip just before 0 in the activated mean (a bump in the inhibited mean) partly shows how "
               "they were selected (short baseline). Baseline pools both baseline sessions (a neuron on both days appears twice).",
               "색 = trial 평균 z, 표시용으로 trial마다 [-3, 0) s 보정, 0.2 s 평활; 회색 띠 = 반응 창. 분류는 선택한 창의 시간 이동 검정 "
               "(q <= 0.20); trace는 분류와 같은 trial을 쓰므로 활성화 평균에서 0 직전의 하강(억제 평균의 상승)은 일부 선택 방식(짧은 기준 "
               "구간) 때문. 기준은 두 기준 세션을 합침 (두 날 뉴런은 두 번 나옴)."))


# ------------------------------------------------------------------------------------------------ behaviour
def behaviour(dk, ea, L, short):
    Rb = ea.csv("P6_behaviour_tests.csv")
    if len(Rb) and ea.ok("P6_behaviour_per_stimulus.png"):
        sv = Rb[Rb.test.str.startswith("SBI-553 vs vehicle")].dropna(subset=["p"])
        sv = sv[(sv[["veh_pct", "sbi_pct"]].max(1) >= 5)]
        if len(sv):
            sv = sv.assign(d=(sv.sbi_pct - sv.veh_pct).abs())
            sg = sv[sv.p < 0.05]
            r = (sg if len(sg) else sv).sort_values("d").iloc[-1]
            key = L(f"{r.behaviour} after {SHORT[r.stim]}: {r.veh_pct:.0f}% of deliveries after vehicle vs {r.sbi_pct:.0f}% after SBI-553 ({ptxt(r.p)})",
                    f"{SHORT[r.stim]} 후 {r.behaviour}: vehicle 후 자극의 {r.veh_pct:.0f}% vs SBI-553 후 {r.sbi_pct:.0f}% ({ptxt(r.p)})")
            nsig = int((sv.p < 0.05).sum())
        else:
            key, nsig = "", 0
        dk.slide(L("BEHAVIOUR", "행동"), L("Behaviour per stimulus, all mice", "자극별 행동, 모든 마우스"), key,
                 ea.f("P6_behaviour_per_stimulus.png"),
                 L(f"% of the deliveries of each stimulus followed by each behaviour; {nsig} stimulus x behaviour pairs differ "
                   "between SBI-553 and vehicle.",
                   f"각 자극 후 각 행동이 나온 자극 %; SBI-553과 vehicle이 다른 자극 x 행동 쌍 {nsig}개."),
                 L("Your rule: withdrawal / flinch from the delivery to 2 s later or the next delivery; attending, lick/bite, "
                   "guarding, escape/rearing, movement, grooming = a bout starting 0-3 s after the delivery. Bars = mean of the 4 "
                   "mice, dots = mice; baseline = both baseline sessions. Brackets (only p < 0.05 drawn): blue (base-veh) / "
                   "red (base-SBI) = that injection vs the same mice's baseline of the same day; black (veh-SBI) = SBI-553 vs vehicle; both Cochran-Mantel-Haenszel over the deliveries with each "
                   "mouse as its own stratum (a mouse without any such behaviour is left out). * p < 0.05, ** p < 0.01, *** p < 0.001; "
                   "deliveries are treated as independent inside a mouse.",
                   "규칙: withdrawal / flinch = 자극부터 2 s 또는 다음 자극까지; attending, lick/bite, guarding, escape/rearing, 움직임, "
                   "그루밍 = 자극 후 0-3 s 안에 시작. 막대 = 4마리 평균, 점 = 마우스; 기준 = 두 기준 세션. 괄호 (p < 0.05만): 파랑(base-veh)·빨강(base-SBI) = 같은 날 같은 "
                   "마우스의 기준 대비; 검정(veh-SBI) = SBI-553 vs vehicle; 둘 다 마우스를 층으로 한 Cochran-Mantel-Haenszel (그 행동이 전혀 없는 "
                   "마우스는 제외). * p < 0.05, ** p < 0.01, *** p < 0.001; 마우스 안에서 자극을 독립으로 봄."))
    if not short and ea.ok("P6b_behaviour_seconds.png"):
        dk.slide(L("BEHAVIOUR", "행동"), L("Time spent in each behaviour after a delivery", "자극 후 각 행동에 쓴 시간"),
                 L("Seconds of each behaviour in the 3 s after a delivery (the 'time' version of the previous slide)",
                   "자극 후 3 s 동안 각 행동의 초 (앞 슬라이드의 '시간' 버전)"),
                 ea.f("P6b_behaviour_seconds.png"),
                 L("Bars = mean of the mice, dots = mice.", "막대 = 마우스 평균, 점 = 마우스."),
                 L("Frames of the behaviour in [0, 3) s after each delivery / 30 fps, mean over the deliveries of a mouse. Descriptive.",
                   "각 자극 후 [0, 3) s 안 그 행동의 프레임 / 30 fps, 마우스의 자극 평균. 기술 통계."))
    Rr = ea.csv("P7_correlations.csv")
    if len(Rr) and ea.ok("P7_brain_behaviour.png"):
        r = Rr.loc[Rr.rho_within_mouse.abs().idxmax()]
        sig = Rr[(Rr.p < 0.05) | (Rr.p_within < 0.05)]
        if len(sig):
            key = L(f"Sessions with more {r.behaviour} have {'more' if r.rho_within_mouse > 0 else 'fewer'} {r.group} neurons "
                    f"(within mouse rho {r.rho_within_mouse:+.2f}, {ptxt(r.p_within)})",
                    f"{r.behaviour}가 많은 세션일수록 {'활성화' if r.group == 'activated' else '억제'} 뉴런이 "
                    f"{'많음' if r.rho_within_mouse > 0 else '적음'} (마우스 안 rho {r.rho_within_mouse:+.2f}, {ptxt(r.p_within)})")
        else:
            key = L(f"No behaviour goes with more or fewer responsive neurons (strongest: {r.behaviour} vs {r.group}, within mouse "
                    f"rho {r.rho_within_mouse:+.2f}, {ptxt(r.p_within)})",
                    f"반응 뉴런 수와 같이 움직이는 행동은 없음 (가장 강한 것: {r.behaviour} vs {'활성화' if r.group == 'activated' else '억제'}, "
                    f"마우스 안 rho {r.rho_within_mouse:+.2f}, {ptxt(r.p_within)})")
        dk.slide(L("BRAIN AND BEHAVIOUR", "뇌와 행동"), L("Do sessions with more behaviour have more responsive neurons?", "행동이 많은 세션에 반응 뉴런도 많은가?"),
                 key, ea.f("P7_brain_behaviour.png"),
                 L("One point = one session (16): behaviour per delivery vs % of neurons activated (top) or inhibited (bottom) by "
                   "any stimulus.", "점 하나 = 세션 하나 (16개): 자극당 행동 vs 어떤 자극에든 활성화(위) / 억제(아래)된 뉴런 %."),
                 L("Spearman rho over the 16 sessions ('all') and after removing each mouse's mean ('within mouse': does a session "
                   "with more behaviour than that mouse's usual have more responsive neurons?). Marker = mouse, fill = condition. "
                   "Sessions of one mouse are not independent, so the 'all' p is optimistic.",
                   "16개 세션의 Spearman rho ('all')와 마우스별 평균을 뺀 후 ('within mouse': 그 마우스 평소보다 행동이 많은 세션에 반응 "
                   "뉴런이 많은가?). 표식 = 마우스, 채움 = 조건. 한 마우스의 세션들은 독립이 아니어서 'all' p는 낙관적."))


# ------------------------------------------------------------------------------------------------ supplement
def supplement(dk, ea, L):
    for ev in ["Light touch", "Mild touch", "flinch", "escape/rearing onset", "grooming onset", "attending onset"]:
        event_heatmap(dk, ea, L, ev, section=L("SUPPLEMENT", "부록"))
    for m in MO:
        fn = f"P4b_maps_{m}.png"
        if ea.ok(fn):
            dk.slide(L("SUPPLEMENT", "부록"), L(f"{m}: where the responsive neurons are", f"{m}: 반응 뉴런의 위치"), "", ea.f(fn),
                     L("Each dot = one neuron at its place in the field of view; red = activated, blue = inhibited, grey = non-responsive; "
                       "rows = that day's baseline (top) and injection session (bottom), columns = events. n = 1 mouse per slide; classes "
                       "from the time-shift test (best window, BH q <= 0.20).",
                       "점 하나 = 시야의 제자리에 있는 뉴런 하나; 빨강 = 활성화, 파랑 = 억제, 회색 = 무반응; 행 = 그날 기준(위)과 주사 세션(아래), "
                       "열 = 사건. 슬라이드당 1마리; 분류는 시간 이동 검정 (최적 창, BH q <= 0.20)."),
                     L("Centroids from the cell review table of each day (Day-2 positions are in the Day-2 field of view).",
                       "각 날 세포 검토 표의 중심점 (Day 2 위치는 Day 2 시야 기준)."))


# ------------------------------------------------------------------------------------------------ checks + other ways (added 10-22)
def checks(dk, va, L, short):
    """is the analysis right? fake events, responses of known size, the separation model (ca_event_validate.py)"""
    J = va.J
    nl = J.get("null", {})
    if nl and va.ok("V1_null_calibration.png"):
        b, m = nl["BH q<=0.20"], nl["max-T 5%"]
        dk.slide(L("CHECKS", "검증"), L("Check 1: what does the test call on FAKE events?", "검증 1: 가짜 사건에서 검정은 무엇을 잡는가?"),
                 L(f"Fake events: {b['mean_pct']:.1f}% of neurons called (BH), {m['mean_pct']:.1f}% (max-T); real events: "
                   f"{b['real_mean_pct']:.0f}% / {m['real_mean_pct']:.0f}% - the responses are real",
                   f"가짜 사건: 뉴런의 {b['mean_pct']:.1f}% (BH), {m['mean_pct']:.1f}% (max-T); 실제 사건: {b['real_mean_pct']:.0f}% / "
                   f"{m['real_mean_pct']:.0f}% - 반응은 실제임"),
                 va.f("V1_null_calibration.png"),
                 L(f"But on {100 * b['frac_10']:.1f}% of the fake runs BH calls >= 10 % of the neurons at once (when most neurons move "
                   f"together); max-T almost never ({100 * m['frac_10']:.1f}%).",
                   f"하지만 BH는 가짜 실행의 {100 * b['frac_10']:.1f}%에서 뉴런 10% 이상을 한꺼번에 잡음 (뉴런 대부분이 함께 움직일 때); max-T는 "
                   f"거의 없음 ({100 * m['frac_10']:.1f}%)."),
                 L("Fake events = the real event train of every session x event shifted inside its own blocks (10 per session x "
                   "event, 500 shuffles each), so nothing is locked to them. Rules: BH q <= 0.20 per session (used in the deck); max-T "
                   "= Westfall-Young single step (each neuron's |response| / its null SD against the largest such value over all "
                   "neurons in each shuffle; family-wise 5 %, valid when neurons are correlated); 'selective' = the same on each "
                   "neuron minus the mean of all neurons. C = the same rules on the real events (session x event).",
                   "가짜 사건 = 각 세션 x 사건의 실제 사건 열을 자기 블록 안에서 이동 (세션 x 사건마다 10번, 각 500 shuffle) - 아무것도 고정되지 "
                   "않음. 규칙: 세션별 BH q <= 0.20 (덱에서 사용); max-T = Westfall-Young (각 뉴런의 |반응| / null SD를 shuffle마다 전체 뉴런의 "
                   "최댓값과 비교; family-wise 5 %, 뉴런이 상관돼도 유효); 'selective' = 각 뉴런에서 전체 뉴런 평균을 뺀 뒤 같은 검정. C = 실제 사건."))
    sp = J.get("spikein", {})
    if sp and va.ok("V2_spikein.png"):
        g = lambda k, a: sp.get(k, {}).get(a, np.nan)
        dk.slide(L("CHECKS", "검증"), L("Check 2: responses of known size added to the real data", "검증 2: 크기를 아는 반응을 실제 데이터에 넣기"),
                 L(f"A +0.5 z response is found in {g('BH q<=0.20 | chosen', '+0.5'):.0f}% of neurons with the chosen window vs "
                   f"{g('BH q<=0.20 | old 2 s / 0-2 s', '+0.5'):.0f}% with the old 2 s window",
                   f"+0.5 z 반응을 선택한 창은 뉴런의 {g('BH q<=0.20 | chosen', '+0.5'):.0f}%에서, 기존 2 s 창은 "
                   f"{g('BH q<=0.20 | old 2 s / 0-2 s', '+0.5'):.0f}%에서 찾음"),
                 va.f("V2_spikein.png"),
                 L(f"Decreases (-0.5 z): {g('BH q<=0.20 | chosen', '-0.5'):.0f}% vs {g('BH q<=0.20 | old 2 s / 0-2 s', '-0.5'):.0f}%. "
                   "Removing the shared response ('selective') makes the other neurons look inhibited - not used.",
                   f"감소 (-0.5 z): {g('BH q<=0.20 | chosen', '-0.5'):.0f}% vs {g('BH q<=0.20 | old 2 s / 0-2 s', '-0.5'):.0f}%. 공통 반응을 "
                   "빼면('selective') 다른 뉴런이 억제된 것처럼 보임 - 쓰지 않음."),
                 L("In every session 10 % of the neurons got a GCaMP-shaped response (rise 0.1 s, decay 1 s; trial gain lognormal, "
                   "SD 0.5) at each delivery of one stimulus; A/B = % of those neurons classified correctly, C = extra % of the other "
                   "neurons called, compared with nothing added. Solid = the chosen window, dashed = the old 2 s / 0-2 s window. "
                   "max-T needs about twice the response size but almost never calls extra neurons (<= 0.15 %).",
                   "모든 세션에서 뉴런의 10 %에 한 자극의 매 전달마다 GCaMP 모양 반응 (상승 0.1 s, 감쇠 1 s; trial gain lognormal SD 0.5)을 "
                   "더함; A/B = 그 뉴런이 맞게 분류된 %, C = 아무것도 안 넣었을 때보다 추가로 잡힌 다른 뉴런 %. 실선 = 선택 창, 점선 = 기존 창. "
                   "max-T는 약 두 배 큰 반응이 필요하지만 다른 뉴런을 거의 추가로 잡지 않음 (<= 0.15 %)."))
    gm = J.get("glm", {})
    if not short and gm and va.ok("V3_glm_separation.png"):
        d1, d2 = gm.get("day1", {}), gm.get("day2", {})
        p_ = lambda d, t, c: d.get(t, {}).get(c, 0.0)
        dk.slide(L("CHECKS", "검증"), L("Check 3: can the model tell a stimulus response from a withdrawal response?",
                                       "검증 3: 모델이 자극 반응과 withdrawal 반응을 구분하는가?"),
                 L(f"Day 2 (logger): {p_(d2, 'stimulus', 'stimulus'):.0f}% of stimulus-locked and {p_(d2, 'withdrawal', 'reflex'):.0f}% of "
                   f"withdrawal-locked neurons get the right class; Day 1 (taps): {p_(d1, 'stimulus', 'stimulus'):.0f}% / "
                   f"{p_(d1, 'withdrawal', 'reflex'):.0f}%",
                   f"Day 2 (로거): 자극 고정 뉴런의 {p_(d2, 'stimulus', 'stimulus'):.0f}%, withdrawal 고정 뉴런의 {p_(d2, 'withdrawal', 'reflex'):.0f}%가 "
                   f"맞는 분류; Day 1 (키): {p_(d1, 'stimulus', 'stimulus'):.0f}% / {p_(d1, 'withdrawal', 'reflex'):.0f}%"),
                 va.f("V3_glm_separation.png"),
                 L(f"On Day 1 {p_(d1, 'stimulus', 'stimulus or reflex (cannot separate)'):.0f}% of the stimulus-locked neurons come out "
                   "'cannot separate': the tap marks sit almost on the withdrawal - the model says so instead of guessing.",
                   f"Day 1에서는 자극 고정 뉴런의 {p_(d1, 'stimulus', 'stimulus or reflex (cannot separate)'):.0f}%가 '나눌 수 없음': 키 표시가 "
                   "withdrawal과 거의 같은 순간 - 모델이 추측하지 않고 그렇다고 말함."),
                 L("In every session 15 % of the neurons got a response (1 z, GCaMP shape) at every delivery, another 15 % at every "
                   "withdrawal; then the same encoding model (ridge, unique variance vs within-block shifted families, 60 shifts, BH q "
                   "<= 0.20). 'nothing added' = the real classes of the other neurons.",
                   "모든 세션에서 뉴런 15 %에 모든 자극 전달마다, 다른 15 %에 모든 withdrawal마다 반응 (1 z, GCaMP 모양)을 넣고 같은 encoding "
                   "모델 (ridge, 블록 안 이동 대비 고유 분산, 60번, BH q <= 0.20). 'nothing added' = 나머지 뉴런의 실제 분류."))


def check_key(dk, va, L, short):
    """check 4: do the key findings hold in every mouse and with the strict test? (added 10-22)"""
    K = va.J.get("key", {})
    if not K or not va.ok("V4_key_findings_per_mouse.png"):
        return
    pv, ps = K.get("Pin prick | BH q<=0.20 | vehicle", {}), K.get("Pin prick | BH q<=0.20 | SBI-553", {})
    mv = K.get("movement onset | BH q<=0.20 | SBI-553", {})
    mvt = K.get("movement onset | max-T 5% | SBI-553", {})
    b = {k.split(" | ")[1]: v for k, v in K.items() if k.startswith("behaviour")}
    fl, mo, es = b.get("flinch / Pin prick", {}), b.get("movement / Light touch", {}), b.get("escape/rearing / Heat", {})
    big = lambda d, sgn: ", ".join(f"{m} {v:+.0f}" for m, v in d.get("per_mouse", {}).items() if sgn * v >= 10) or "none"
    dk.slide(L("CHECKS", "검증"), L("Check 4: do the key findings hold in every mouse and with the strict test?",
                                   "검증 4: 핵심 결과가 마우스마다, 엄격한 검정에서도 유지되는가?"),
             L(f"Behaviour holds: lower after SBI-553 in {fl.get('n_lower_sbi', 0)}/4 (flinch, pin), {mo.get('n_lower_sbi', 0)}/4 (movement, "
               f"light), {es.get('n_lower_sbi', 0)}/4 (escape, heat) mice; pin and movement % changes come from 1-2 mice, heat rises in 3-4",
               f"행동은 유지: SBI-553 후 낮음 {fl.get('n_lower_sbi', 0)}/4 (flinch, pin), {mo.get('n_lower_sbi', 0)}/4 (movement, light), "
               f"{es.get('n_lower_sbi', 0)}/4 (escape, heat) 마리; pin·movement % 변화는 1-2마리에서 나오고 heat는 3-4마리에서 증가"),
             va.f("V4_key_findings_per_mouse.png"),
             L(f"Pin prick % responsive: vehicle up only in {big(pv, 1)}, SBI-553 down only in {big(ps, -1)} (pooled ns); the vehicle rise holds with "
               f"max-T. Movement onset after SBI-553: up in {mv.get('n_up', 0)}/4 mice (BH), {mvt.get('n_up', 0)}/4 (max-T).",
               f"pin prick 반응 %: vehicle 증가는 {big(pv, 1)}에서만, SBI-553 감소는 {big(ps, -1)}에서만 (합쳐서 ns); vehicle 증가는 max-T에서도 유지. "
               f"SBI-553 후 움직임 시작: {mv.get('n_up', 0)}/4 마리 증가 (BH), {mvt.get('n_up', 0)}/4 (max-T)."),
             L("Neural panels: change of the % responsive neurons from the baseline to the injection session of the same day, per "
               "mouse (dot), filled = BH q <= 0.20, open = max-T 5 %; star over 'base vs inj.' = pooled McNemar-type binomial test over all neurons (baseline vs injection of the same day). "
               "Behaviour panels: % of the deliveries followed by the behaviour in the vehicle and the SBI-553 injection session of "
               "each mouse; 'lower' = by more than 1 point. With 4 mice no test across mice can reach p < 0.05 (minimum 0.125).",
               "뉴런 패널: 같은 날 기준 -> 주사 세션의 반응 뉴런 % 변화, 마우스별 (점), 채움 = BH q <= 0.20, 빈 = max-T 5 %; 'base vs inj.' 위 별 = 모든 뉴런 합친 "
               "McNemar형 이항 검정 (같은 날 기준 vs 주사). 행동 패널: 각 마우스의 vehicle, SBI-553 주사 세션에서 그 행동이 뒤따른 자극 %; '낮음' = 1%p 넘게. "
               "4마리로는 마우스 간 검정이 p < 0.05에 못 미침 (최소 0.125)."))


def other_ways(dk, va, L, short):
    """other ways to categorise the neurons and read the activity (ca_event_validate.py)"""
    J = va.J
    rb = J.get("robust", {})
    if not short and rb and va.ok("A1_rules_compared.png"):
        dk.slide(L("OTHER WAYS", "다른 방법"), L("Other way 1: a strict test (max-T) and each neuron's own response",
                                              "다른 방법 1: 엄격한 검정(max-T)과 뉴런 자신의 반응"),
                 L(f"With max-T {rb['max-T 5%']['mean']:.0f}% of neurons respond per session x event (BH {rb['BH q<=0.20']['mean']:.0f}%); "
                   "the order of the events stays (movement onset first)",
                   f"max-T에서는 세션 x 사건마다 뉴런의 {rb['max-T 5%']['mean']:.0f}%가 반응 (BH {rb['BH q<=0.20']['mean']:.0f}%); 사건의 순서는 "
                   "같음 (움직임 시작이 가장 많음)"),
                 va.f("A1_rules_compared.png"),
                 L("Top: % responsive per session (rows, in time order) and event under each rule; bottom: per condition.",
                   "위: 규칙마다 세션(행, 시간 순) x 사건의 반응 %; 아래: 조건별."),
                 L(f"Spread between sessions (SD of % per event): BH {rb['BH q<=0.20']['sd_between_sessions']:.0f}, max-T "
                   f"{rb['max-T 5%']['sd_between_sessions']:.0f} points. max-T is the safe choice when neurons share a response (check 1); "
                   "it needs larger responses (check 2). Same windows and shift null as the main classes.",
                   f"세션 간 퍼짐 (사건별 % SD): BH {rb['BH q<=0.20']['sd_between_sessions']:.0f}, max-T {rb['max-T 5%']['sd_between_sessions']:.0f}%p. "
                   "max-T는 뉴런이 반응을 공유할 때 안전 (검증 1); 더 큰 반응이 필요 (검증 2). 창과 이동 null은 본 분류와 같음."))
    cl = J.get("clusters", {})
    if not short and cl and va.ok("A2_functional_types.png"):
        dk.slide(L("OTHER WAYS", "다른 방법"), L("Other way 2: functional types of all neurons (clustering)", "다른 방법 2: 모든 뉴런의 기능 유형 (군집화)"),
                 L(f"All neurons form a continuum: {cl['k']} groups with weak separation (silhouette {cl['silhouette']:.2f}) - mostly "
                   "'slightly up to everything' vs 'flat'",
                   f"모든 뉴런은 연속체: 구분이 약한 {cl['k']}개 군집 (silhouette {cl['silhouette']:.2f}) - 대부분 '모든 사건에 조금 오름' vs '평평함'"),
                 va.f("A2_functional_types.png"),
                 L(f"Same neurons, baseline -> injection: {cl['stay']['vehicle']:.0f}% stay in their type with vehicle, "
                   f"{cl['stay']['SBI-553']:.0f}% with SBI-553.",
                   f"같은 뉴런, 기준 -> 주사: vehicle {cl['stay']['vehicle']:.0f}%, SBI-553 {cl['stay']['SBI-553']:.0f}%가 같은 유형에 머묾."),
                 L("Features = each neuron's trial-averaged z (-1..3 s, 0.2-s bins, baseline -1..0 s) to light, mild, pin, heat, "
                   "withdrawal and movement onset; PCA (10) -> k-means on the baseline sessions (k 3-8 by silhouette); injection "
                   "sessions assigned to the nearest type.",
                   "특징 = 각 뉴런의 light, mild, pin, heat, withdrawal, 움직임 시작에 대한 trial 평균 z (-1..3 s, 0.2 s bin, 기준 -1..0 s); "
                   "PCA (10) -> 기준 세션 k-means (silhouette로 k 3-8); 주사 세션은 가장 가까운 유형으로."))
    tr = J.get("types_responsive", {})
    if not short and tr and va.ok("A2b_types_responsive.png"):
        mr = tr.get("mean_resp", {})
        if mr:
            neg = min(mr, key=mr.get)
            nb_, nv, ns = tr[neg]["baseline"], tr[neg]["vehicle"], tr[neg]["SBI-553"]
            key = L(f"Responsive neurons have {tr['k']} response shapes; the 'down to everything' type ({neg.replace('type', 'type ')}) is "
                    f"{nb_:.0f}% at baseline, {nv:.0f}% after vehicle, {ns:.0f}% after SBI-553",
                    f"반응 뉴런은 {tr['k']}가지 반응 모양; '모든 사건에 내려감' 유형 ({neg.replace('type', '유형 ')})은 기준 {nb_:.0f}%, vehicle 후 "
                    f"{nv:.0f}%, SBI-553 후 {ns:.0f}%")
        else:
            key = ""
        dk.slide(L("OTHER WAYS", "다른 방법"), L("Other way 3: response shapes of the responsive neurons", "다른 방법 3: 반응 뉴런의 반응 모양"),
                 key, va.f("A2b_types_responsive.png"),
                 L("Only neurons responsive to >= 1 event (max-T); each profile scaled by its own largest response, so the types are "
                   "about shape (which events, up or down, fast or slow), not size.",
                   "1개 이상 사건에 반응한 뉴런만 (max-T); 각 뉴런의 가장 큰 반응으로 나눠서 크기가 아니라 모양(어떤 사건, 오름/내림, 빠름/느림)으로 나눔."),
                 L(f"k-means on the baseline sessions (k = {tr['k']} by silhouette {tr['silhouette']:.2f}: still overlapping), the injection "
                   "sessions assigned to the nearest type; n per condition on the right. Exploratory: types are not stable across k.",
                   f"기준 세션 k-means (silhouette {tr['silhouette']:.2f}로 k = {tr['k']}: 여전히 겹침), 주사 세션은 가장 가까운 유형; 오른쪽에 조건별 n. "
                   "탐색적: k에 따라 유형이 바뀜."))
    dc = J.get("decode", {})
    if not short and dc and va.ok("A3_decoding.png"):
        b = dc.get("baseline", {})
        dk.slide(L("OTHER WAYS", "다른 방법"), L("Other way 4: can one session's population tell the stimuli apart?",
                                              "다른 방법 4: 한 세션의 뉴런 집단이 자극을 구분하는가?"),
                 L(f"Hardly: the 4 stimuli are decoded at {b.get('stim', np.nan):.0f}% at baseline (chance 25%, sham window "
                   f"{b.get('sham', np.nan):.0f}%); a coming withdrawal at {b.get('wd', np.nan):.0f}% (chance 50%)",
                   f"거의 못 함: 기준에서 4개 자극 {b.get('stim', np.nan):.0f}% (우연 25%, 가짜 창 {b.get('sham', np.nan):.0f}%); withdrawal 여부 "
                   f"{b.get('wd', np.nan):.0f}% (우연 50%)"),
                 va.f("A3_decoding.png"),
                 L("Logistic regression on every trial's population response (5-fold CV x 5, balanced accuracy), per session.",
                   "각 trial의 집단 반응으로 logistic 회귀 (5-fold CV x 5, balanced accuracy), 세션별."),
                 L("Sham = the same decoder on a window before the event (-1..-0.5 s vs -1.5..-1 s), which carries the block / drift "
                   "information but no response: real - sham = what the response itself adds.",
                   "가짜 창 = 사건 전 창 (-1..-0.5 s vs -1.5..-1 s)으로 같은 decoder: 블록/drift 정보는 있지만 반응은 없음; 실제 - 가짜 = 반응 자체가 "
                   "더하는 정보."))
    pp = J.get("pseudopop", {})
    if not short and pp and va.ok("A3b_pseudopop_decoding.png"):
        dk.slide(L("OTHER WAYS", "다른 방법"), L("Other way 5: all mice's neurons pooled (pseudo-population decoding)",
                                              "다른 방법 5: 모든 마우스의 뉴런을 합쳐서 (pseudo-population decoding)"),
                 L(f"100 pooled neurons decode the 4 stimuli at {pp.get('baseline | real', np.nan):.0f}% at baseline (sham "
                   f"{pp.get('baseline | sham', np.nan):.0f}%), {pp.get('vehicle | real', np.nan):.0f}% after vehicle, "
                   f"{pp.get('SBI-553 | real', np.nan):.0f}% after SBI-553 (chance 25%)",
                   f"합친 뉴런 100개가 4개 자극을 기준 {pp.get('baseline | real', np.nan):.0f}% (가짜 {pp.get('baseline | sham', np.nan):.0f}%), vehicle 후 "
                   f"{pp.get('vehicle | real', np.nan):.0f}%, SBI-553 후 {pp.get('SBI-553 | real', np.nan):.0f}%로 구분 (우연 25%)"),
                 va.f("A3b_pseudopop_decoding.png"),
                 L("The stimuli are not told apart trial by trial: the activity follows movement and arousal more than stimulus identity.",
                   "자극은 trial 단위로 구분되지 않음: 활동은 자극 종류보다 움직임과 각성을 더 따름."),
                 L("Pseudo-trials: for each of 100 randomly drawn neurons (any session of the condition), one random trial of the asked "
                   "stimulus from its own session; 8 train + 3 test trials per stimulus per neuron, 100 repetitions (bar = 95 % of them). "
                   "Below 25 % = no information (small test sets bias it slightly down).",
                   "pseudo-trial: 무작위로 뽑은 뉴런 100개 각각에 대해 자기 세션에서 그 자극의 trial 하나를 무작위로; 뉴런·자극마다 학습 8 + 시험 3 "
                   "trial, 100번 반복 (막대 = 95 %). 25 % 아래 = 정보 없음 (작은 시험 세트가 약간 낮게 만듦)."))
