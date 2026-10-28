"""make_sex_deck.py - SEX (2026-10-22): is there a pattern by sex? (females 5941 + 1191, males 5939 + 5937; each sex has one
mouse per drug order). Figures from ca_sex_analysis.py (analysis\\sex_analysis); every sentence is computed from
sex_summary.json. Same layout as the new-style decks: title, ONE key sentence, the figure, and at the bottom what we did /
found with n and the test; details in the notes. Used alone (this script) and as a section of the Day 1 + Day 2 decks
(make_new_deck.py calls add_sex_slides).

  python make_sex_deck.py --lang en|kr --out <deck.pptx>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winwords as WW  # noqa: E402

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
SD = os.path.join(B, "analysis", "sex_analysis")
FEM, MAL = ("5941", "1191"), ("5939", "5937")


def add_sex_slides(dk, lg, short=False):
    """the sex section; short = the PI version (summary + the rate figure)"""
    L = lambda en, kr: kr if lg == "kr" else en
    js = os.path.join(SD, "sex_summary.json")
    if not os.path.isfile(js):
        return 0
    J = json.load(open(js, encoding="utf-8"))
    f = lambda nm: os.path.join(SD, nm)
    s1, s3, s4, s5, s7 = J["s1"], J["s3"], J["s4"], J["s5"], J["s7"]
    c3 = lambda drug, m: s3[f"change {drug} | {m}"]["median_log2"]
    b1 = lambda b, m, c: s1[f"{b} | {m}"][c]
    r4 = lambda ev, c, m: s4.get(f"{ev} | {c} | {m}", np.nan)
    sec = L("SEX", "성별")
    nline = L("n = 2 females (5941, 1191) + 2 males (5939, 5937); each sex has one SBI-553-first and one vehicle-first mouse.",
              "n = 암컷 2 (5941, 1191) + 수컷 2 (5939, 5937); 성별마다 SBI-553 먼저 1마리, vehicle 먼저 1마리.")
    n0 = len(dk.prs.slides)
    # ------------------------------------------------------------------ 1 summary
    lt_f, lt_m = [r4("Light touch", "SBI-553", m) for m in FEM], [r4("Light touch", "SBI-553", m) for m in MAL]
    ht_f, ht_m = [r4("Heat", "SBI-553", m) for m in FEM], [r4("Heat", "SBI-553", m) for m in MAL]
    wv_f, wv_m = [b1("withdrawal", m, "vehicle") for m in FEM], [b1("withdrawal", m, "vehicle") for m in MAL]
    wb = [b1("withdrawal", m, "baseline") for m in FEM + MAL]
    mv = {m: (b1("movement", m, "baseline"), b1("movement", m, "SBI-553")) for m in FEM + MAL}
    dk.cards(sec, L("Is there a pattern by sex?", "성별에 따른 패턴이 있는가?"),
             L("Two candidate sex patterns (both females differ from both males); the rest follows single mice or the drug order",
               "후보 패턴 2개 (두 암컷이 두 수컷과 다름); 나머지는 한 마리 또는 투약 순서를 따름"),
             [(f"{c3('SBI-553', FEM[0]):+.1f} / {c3('SBI-553', FEM[1]):+.1f}",
               L("log2 rate change on the SBI-553 day, females", "SBI-553 날 빈도 변화 (log2), 암컷"),
               L(f"Males {c3('SBI-553', MAL[0]):+.2f} / {c3('SBI-553', MAL[1]):+.2f}: both females dropped more. Vehicle day mixed "
                 f"(5941 {c3('vehicle', '5941'):+.2f}, 1191 {c3('vehicle', '1191'):+.2f}, 5939 {c3('vehicle', '5939'):+.2f}, "
                 f"5937 {c3('vehicle', '5937'):+.2f}).",
                 f"수컷 {c3('SBI-553', MAL[0]):+.2f} / {c3('SBI-553', MAL[1]):+.2f}: 두 암컷이 더 떨어짐. vehicle 날은 섞임 "
                 f"(5941 {c3('vehicle', '5941'):+.2f}, 1191 {c3('vehicle', '1191'):+.2f}, 5939 {c3('vehicle', '5939'):+.2f}, "
                 f"5937 {c3('vehicle', '5937'):+.2f}).")),
              (f"{lt_f[0]:.0f} / {lt_f[1]:.0f} %", L("light-touch neurons after SBI-553, females", "SBI-553 후 light touch 반응 뉴런, 암컷"),
               L(f"Males {lt_m[0]:.0f} / {lt_m[1]:.0f} %; no sex split, nor for heat (females {ht_f[0]:.0f} / {ht_f[1]:.0f} %, males "
                 f"{ht_m[0]:.0f} / {ht_m[1]:.0f} %). The reflex neurons do not split by sex.",
                 f"수컷 {lt_m[0]:.0f} / {lt_m[1]:.0f} %; 성별로 안 나뉨, heat도 마찬가지 (암컷 {ht_f[0]:.0f} / {ht_f[1]:.0f} %, 수컷 {ht_m[0]:.0f} / "
                 f"{ht_m[1]:.0f} %). 반사 뉴런은 성별로 나뉘지 않음.")),
              (f"{wv_f[0]:.0f} / {wv_f[1]:.0f} %", L("withdrawal per delivery after vehicle, females", "vehicle 후 자극당 withdrawal, 암컷"),
               L(f"Males {wv_m[0]:.0f} / {wv_m[1]:.0f} %; after SBI-553 both females higher too; at baseline all 4 mice "
                 f"{min(wb):.0f}-{max(wb):.0f} %. Not drug-specific.",
                 f"수컷 {wv_m[0]:.0f} / {wv_m[1]:.0f} %; SBI-553 후에도 두 암컷이 높음; 기준에서는 4마리 모두 {min(wb):.0f}-{max(wb):.0f} %. "
                 "약물 특이적이지 않음.")),
              (L("order", "순서"), L("not sex: the movement drop after SBI-553", "성별 아님: SBI-553 후 움직임 감소"),
               L(f"Only the two mice with SBI-553 on Day 1: 5941 (F) {mv['5941'][0]:.0f} -> {mv['5941'][1]:.0f} %, 5939 (M) "
                 f"{mv['5939'][0]:.0f} -> {mv['5939'][1]:.0f} %; not 1191 (F) {mv['1191'][0]:.0f} -> {mv['1191'][1]:.0f} % or "
                 f"5937 (M) {mv['5937'][0]:.0f} -> {mv['5937'][1]:.0f} %.",
                 f"Day 1에 SBI-553을 받은 두 마리만: 5941 (F) {mv['5941'][0]:.0f} -> {mv['5941'][1]:.0f} %, 5939 (M) "
                 f"{mv['5939'][0]:.0f} -> {mv['5939'][1]:.0f} %; 1191 (F) {mv['1191'][0]:.0f} -> {mv['1191'][1]:.0f} %, "
                 f"5937 (M) {mv['5937'][0]:.0f} -> {mv['5937'][1]:.0f} %는 아님."))],
             nline + L(" Descriptive: 2 vs 2 mice cannot be tested; these are patterns to follow up.",
                       " 기술 통계: 2 vs 2마리는 검정할 수 없음; 더 볼 패턴."),
             L("Sex from the session table (_SEX in ca_day1_analysis). 5941 F (SBI-553 Day 1), 1191 F (SBI-553 Day 2), 5939 M "
               "(SBI-553 Day 1), 5937 M (SBI-553 Day 2). 5937's Day-2 signal is weak (floor), which pulls the male SBI-553-day rate "
               "change toward 0. Rate change = median over the mouse's neurons of log2 ((injection + 0.1) / (baseline of the same "
               "day + 0.1)), stimulus blocks, still frames. Responsive = activated or inhibited (time-shift test, BH q <= 0.20 per "
               "session). Behaviour = % of deliveries followed by the behaviour (all 4 stimuli). Movement = 'movement' bout starting "
               "0-3 s after a delivery.",
               "성별은 세션 표 (ca_day1_analysis의 _SEX). 5937의 Day 2 신호가 약해(바닥) 수컷 SBI-553 날 빈도 변화가 0 쪽으로 끌림. "
               "빈도 변화 = 마우스 뉴런의 log2 ((주사 + 0.1) / (같은 날 기준 + 0.1)) 중앙값. 반응 = 활성화 또는 억제 (시간 이동 검정, "
               "세션별 BH q <= 0.20). 행동 = 그 행동이 뒤따른 자극 % (4개 자극). 움직임 = 자극 후 0-3 s 안에 시작한 움직임."))
    # ------------------------------------------------------------------ rates (FULL and PI)
    dk.slide(sec, L("Activity of every neuron by sex", "성별 모든 뉴런의 활동"),
             L(f"On the SBI-553 day both females dropped more than both males (median log2 {c3('SBI-553', FEM[0]):+.2f}, "
               f"{c3('SBI-553', FEM[1]):+.2f} vs {c3('SBI-553', MAL[0]):+.2f}, {c3('SBI-553', MAL[1]):+.2f})",
               f"SBI-553 날 두 암컷이 두 수컷보다 더 떨어짐 (log2 중앙값 {c3('SBI-553', FEM[0]):+.2f}, {c3('SBI-553', FEM[1]):+.2f} vs "
               f"{c3('SBI-553', MAL[0]):+.2f}, {c3('SBI-553', MAL[1]):+.2f})"),
             f("S3_rates_by_sex.png"),
             L(f"We compared the transient rate of every neuron between the sexes (A) and its change from the same day's baseline "
               f"(B; black = each mouse). n = {s3['SBI-553 | F']['n']} F / {s3['SBI-553 | M']['n']} M neurons on the SBI-553 day, "
               "2 + 2 mice; Mann-Whitney (A), Wilcoxon vs 0 (B).",
               f"모든 뉴런의 transient 빈도를 성별로 비교(A)하고 같은 날 기준 대비 변화(B; 검정 = 마우스)를 봄. SBI-553 날 뉴런 "
               f"{s3['SBI-553 | F']['n']} F / {s3['SBI-553 | M']['n']} M, 마우스 2 + 2; Mann-Whitney (A), 0 대비 Wilcoxon (B)."),
             L("A: transients/min in the stimulus blocks (still frames), box = median and quartiles over neurons; bracket = female vs "
               "male neurons (Mann-Whitney; exploratory, neurons are nested in 2 mice per sex). B: per neuron log2 ((injection + "
               "0.1) / (baseline + 0.1)), same day; star over 'vs 0' = the change differs from 0 (Wilcoxon signed-rank); black "
               "markers = each mouse's median. Per mouse (median log2): SBI-553 day " + ", ".join(f"{m} {c3('SBI-553', m):+.2f}" for m in FEM + MAL)
               + "; vehicle day " + ", ".join(f"{m} {c3('vehicle', m):+.2f}" for m in FEM + MAL) + ". 5937's SBI-553 day (Day 2) is at "
               "the floor.",
               "A: 자극 블록(움직임 없는 프레임)의 분당 transient, 상자 = 뉴런 중앙값·사분위; 괄호 = 암컷 vs 수컷 뉴런 (Mann-Whitney; "
               "탐색적). B: 뉴런별 log2 ((주사 + 0.1) / (기준 + 0.1)), 같은 날; 'vs 0' 위 별 = 0과 다름 (Wilcoxon); 검은 표식 = 마우스 "
               "중앙값. 마우스별: SBI-553 날 " + ", ".join(f"{m} {c3('SBI-553', m):+.2f}" for m in FEM + MAL) + "; vehicle 날 "
               + ", ".join(f"{m} {c3('vehicle', m):+.2f}" for m in FEM + MAL) + "."))
    if short:
        return len(dk.prs.slides) - n0
    # ------------------------------------------------------------------ behaviour
    dk.slide(sec, L("Behaviour per delivery by sex", "성별 자극당 행동"),
             L(f"Withdrawal after either injection: both females higher (vehicle {wv_f[0]:.0f}, {wv_f[1]:.0f} % vs {wv_m[0]:.0f}, "
               f"{wv_m[1]:.0f} %); the movement drop after SBI-553 is in the Day-1 SBI-553 mice (5941 F, 5939 M)",
               f"주사 후 withdrawal: 두 암컷이 높음 (vehicle {wv_f[0]:.0f}, {wv_f[1]:.0f} % vs {wv_m[0]:.0f}, {wv_m[1]:.0f} %); SBI-553 후 "
               "움직임 감소는 Day 1 SBI-553 마우스(5941 F, 5939 M)에서"),
             f("S1_behaviour_by_sex.png"),
             L("We plotted the % of deliveries followed by each behaviour for every mouse (pink = female, teal = male). n = 2 + 2 mice, "
               "4 stimuli pooled; descriptive (no test with 2 mice per sex).",
               "마우스마다 각 행동이 뒤따른 자극 %를 그림 (분홍 = 암컷, 청록 = 수컷). 마우스 2 + 2, 4개 자극 합침; 기술 통계 (성별당 2마리라 "
               "검정 없음)."),
             L("% of deliveries followed by the behaviour (reflexes: within 2 s or before the next delivery; held behaviours: a bout "
               "starting 0-3 s after). Baseline = mean of the mouse's two baseline sessions. Guarding never followed a delivery and "
               "grooming < 2 %, so not shown. Per mouse, withdrawal baseline / vehicle / SBI-553: "
               + "; ".join(f"{m} {b1('withdrawal', m, 'baseline'):.0f} / {b1('withdrawal', m, 'vehicle'):.0f} / {b1('withdrawal', m, 'SBI-553'):.0f}"
                           for m in FEM + MAL) + "; movement: "
               + "; ".join(f"{m} {b1('movement', m, 'baseline'):.0f} / {b1('movement', m, 'vehicle'):.0f} / {b1('movement', m, 'SBI-553'):.0f}"
                           for m in FEM + MAL) + ".",
               "그 행동이 뒤따른 자극 % (반사: 2 s 안 또는 다음 자극 전; 지속 행동: 0-3 s 안 시작). 기준 = 두 기준 세션 평균. guarding은 "
               "없고 grooming < 2 %라 제외. 마우스별 withdrawal 기준 / vehicle / SBI-553: "
               + "; ".join(f"{m} {b1('withdrawal', m, 'baseline'):.0f} / {b1('withdrawal', m, 'vehicle'):.0f} / {b1('withdrawal', m, 'SBI-553'):.0f}"
                           for m in FEM + MAL) + "; 움직임: "
               + "; ".join(f"{m} {b1('movement', m, 'baseline'):.0f} / {b1('movement', m, 'vehicle'):.0f} / {b1('movement', m, 'SBI-553'):.0f}"
                           for m in FEM + MAL) + "."))
    # ------------------------------------------------------------------ drug effect
    wh = J["s2"].get("withdrawal | Heat | F", [np.nan, np.nan])
    dk.slide(sec, L("The SBI-553 effect on behaviour, by sex", "성별 SBI-553 행동 효과"),
             L(f"No sex rule: the two females often go opposite ways (withdrawal to heat: 5941 {wh[0]:+.0f}, 1191 {wh[1]:+.0f} points)",
               f"성별 규칙 없음: 두 암컷이 자주 반대 방향 (heat에 대한 withdrawal: 5941 {wh[0]:+.0f}, 1191 {wh[1]:+.0f}%p)"),
             f("S2_drug_effect_by_sex.png"),
             L("We took the SBI-553 minus the vehicle injection session of the same mouse, per stimulus (bar = sex mean, marker = "
               "mouse). n = 2 + 2 mice; descriptive.",
               "같은 마우스의 SBI-553 주사 세션 - vehicle 주사 세션, 자극별 (막대 = 성별 평균, 표식 = 마우스). 마우스 2 + 2; 기술 통계."),
             L("Below 0 = less after SBI-553. With 2 mice per sex no test is meaningful; the overall SBI-553 effects (flinch / pin, "
               "movement / light; escape / heat p = 0.06, ns) are tested on the main slides (CMH by mouse).",
               "0 아래 = SBI-553 후 적음. 성별당 2마리라 검정은 의미 없음; 전체 SBI-553 효과는 본 슬라이드 (마우스를 층으로 한 CMH)."))
    # ------------------------------------------------------------------ responsive
    wd = {m: r4("withdrawal", "vehicle", m) for m in FEM + MAL}
    dk.slide(sec, L("Responsive neurons by sex", "성별 반응 뉴런"),
             L(f"After SBI-553 light-touch and heat neurons do not split by sex: females ({lt_f[0]:.0f}, {lt_f[1]:.0f} % and {ht_f[0]:.0f}, "
               f"{ht_f[1]:.0f} %) vs males ({lt_m[0]:.0f}, {lt_m[1]:.0f} % and {ht_m[0]:.0f}, {ht_m[1]:.0f} %)",
               f"SBI-553 후 light touch·heat 반응 뉴런은 성별로 나뉘지 않음: 암컷({lt_f[0]:.0f}, {lt_f[1]:.0f} %와 {ht_f[0]:.0f}, {ht_f[1]:.0f} %), "
               f"수컷({lt_m[0]:.0f}, {lt_m[1]:.0f} %와 {ht_m[0]:.0f}, {ht_m[1]:.0f} %)"),
             f("S4_responsive_by_sex.png"),
             L(f"Bars = all neurons of a sex, black markers = each mouse. The reflex neurons follow single mice, not sex (withdrawal "
               f"after vehicle: 5941 {wd['5941']:.0f} %, 1191 {wd['1191']:.0f} %, 5937 {wd['5937']:.0f} %, 5939 {wd['5939']:.0f} %). "
               "n = 2 + 2 mice; descriptive.",
               f"막대 = 그 성별 전체 뉴런, 검은 표식 = 마우스. 반사 뉴런은 성별이 아니라 마우스를 따름 (vehicle 후 withdrawal: 5941 "
               f"{wd['5941']:.0f} %, 1191 {wd['1191']:.0f} %, 5937 {wd['5937']:.0f} %, 5939 {wd['5939']:.0f} %). 마우스 2 + 2; 기술 통계."),
             L("Responsive = activated or inhibited, time-shift test with the chosen window, BH q <= 0.20 per session (as on the main "
               "slides). Per mouse % after SBI-553, light touch / heat: " + "; ".join(f"{m} {r4('Light touch', 'SBI-553', m):.0f} / "
                                                                                    f"{r4('Heat', 'SBI-553', m):.0f}" for m in FEM + MAL) + ".",
               "반응 = 활성화 또는 억제, 선택 창의 시간 이동 검정, 세션별 BH q <= 0.20 (본 슬라이드와 같음). SBI-553 후 마우스별 % "
               "(light touch / heat): " + "; ".join(f"{m} {r4('Light touch', 'SBI-553', m):.0f} / {r4('Heat', 'SBI-553', m):.0f}"
                                                    for m in FEM + MAL) + "."))
    # ------------------------------------------------------------------ traces
    g5 = lambda sx, c: s5[f"movement onset | {sx} | {c}"]
    dk.slide(sec, L("Event-locked activity by sex", "성별 사건 고정 활동"),
             L(f"Movement onset rises after both injections in both sexes (F {g5('F', 'baseline'):.3f} -> {g5('F', 'vehicle'):.3f} / "
               f"{g5('F', 'SBI-553'):.3f} z, M {g5('M', 'baseline'):.3f} -> {g5('M', 'vehicle'):.3f} / {g5('M', 'SBI-553'):.3f} z): "
               "no sex-specific drug effect",
               f"움직임 시작 반응은 두 성별 모두 두 주사 후 상승 (F {g5('F', 'baseline'):.3f} -> {g5('F', 'vehicle'):.3f} / "
               f"{g5('F', 'SBI-553'):.3f} z, M {g5('M', 'baseline'):.3f} -> {g5('M', 'vehicle'):.3f} / {g5('M', 'SBI-553'):.3f} z): "
               "성별 특이적 약물 효과 없음"),
             f("S5_traces_by_sex.png"),
             L("We averaged the event-locked z of all neurons per sex and condition (top = females, bottom = males; mean +- SEM, n "
               "neurons in the legend). 2 + 2 mice; descriptive.",
               "성별·조건별 모든 뉴런의 사건 고정 z 평균 (위 = 암컷, 아래 = 수컷; 평균 +- SEM, 범례에 뉴런 n). 마우스 2 + 2; 기술 통계."),
             L("Each trial baselined on [-3, 0) s, trial-averaged per neuron, 0.2-s smoothing; numbers in the key sentence = mean of "
               f"the trace over each event's response window ({WW.en()}) (baseline / after vehicle / after SBI-553). n = neuron-sessions (baseline pools both baseline "
               "sessions).",
               f"trial마다 [-3, 0) s 기준, 뉴런별 trial 평균, 0.2 s 평활; 핵심 문장 수치 = 사건별 반응 창 평균 ({WW.kr()}) (기준 / vehicle 후 / SBI-553 후). "
               "n = 뉴런-세션."))
    # ------------------------------------------------------------------ population
    dk.slide(sec, L("Whole-session population activity by sex", "성별 세션 전체 집단 활동"),
             L("No sex split over the session: females and males interleave; 5937 is the lowest",
               "세션 전체로는 성별로 나뉘지 않음: 암수가 섞여 있고 5937이 가장 낮음"),
             f("S6_population_by_sex.png"),
             L("We plotted the mean z of all neurons of each mouse through the session (5-s bins; orange = stimulus blocks). n = 2 + 2 "
               "mice; descriptive.",
               "마우스마다 모든 뉴런의 평균 z를 세션 전체로 그림 (5초 bin; 주황 = 자극 블록). 마우스 2 + 2; 기술 통계."),
             L("Baseline = mean of the two baseline sessions. z is scaled per session (robust SD), so levels between sessions are only "
               "roughly comparable.", "기준 = 두 기준 세션 평균. z는 세션마다 robust SD로 나눔."))
    # ------------------------------------------------------------------ movement neurons
    dk.slide(sec, L("Movement-onset neurons after the injection, by sex", "성별 주사 후 움직임 시작 뉴런"),
             L(f"Females: up after SBI-553 ({s7['SBI-553 | F']['median']:+.2f} z) and down after vehicle "
               f"({s7['vehicle | F']['median']:+.2f} z), almost all from 1191; males: no clear change",
               f"암컷: SBI-553 후 증가 ({s7['SBI-553 | F']['median']:+.2f} z), vehicle 후 감소 ({s7['vehicle | F']['median']:+.2f} z) - 거의 "
               "모두 1191; 수컷: 뚜렷한 변화 없음"),
             f("S7_movement_neurons_by_sex.png"),
             L(f"We followed the neurons activated by movement onset at baseline into the injection session. n = "
               f"{s7['SBI-553 | F']['n']} F / {s7['SBI-553 | M']['n']} M neurons after SBI-553 (1191: {s7['SBI-553 | 1191']['n']}); "
               "Wilcoxon vs 0 over neurons.",
               f"기준에서 움직임 시작에 활성화된 뉴런을 주사 세션까지 따라감. SBI-553 후 뉴런 {s7['SBI-553 | F']['n']} F / "
               f"{s7['SBI-553 | M']['n']} M (1191: {s7['SBI-553 | 1191']['n']}); 뉴런 0 대비 Wilcoxon."),
             L("Neurons activated by movement onset on the ODD baseline trials (Wilcoxon p < 0.05); change = mean response after the "
               "injection minus the EVEN baseline trials (so regression to the mean cannot make it). Black markers = each mouse's "
               "median (n neurons). Star over 'vs 0' = Wilcoxon signed-rank over that sex's neurons.",
               "홀수 기준 trial에서 움직임 시작에 활성화된 뉴런 (Wilcoxon p < 0.05); 변화 = 주사 후 평균 반응 - 짝수 기준 trial (평균 회귀가 "
               "만들 수 없음). 검은 표식 = 마우스별 중앙값. 'vs 0' 위 별 = 그 성별 뉴런의 Wilcoxon."))
    # ------------------------------------------------------------------ limits
    dk.cards(sec, L("What sex can and cannot tell here", "여기서 성별로 말할 수 있는 것과 없는 것"), "",
             [("2", L("mice per sex", "성별당 마리"), L("A 'sex' difference is often one mouse (5941, 1191 or 5939).",
                                                    "'성별' 차이는 자주 한 마리 (5941, 1191 또는 5939).")),
              ("5937", L("male, weak Day-2 signal", "수컷, Day 2 신호 약함"),
               L("Its SBI-553 day is Day 2 (floor): the male SBI-553 numbers lean toward no change.",
                 "SBI-553 날이 Day 2 (바닥): 수컷 SBI-553 수치가 '변화 없음' 쪽으로 기움.")),
              (L("order", "순서"), L("balanced inside each sex", "성별 안에서 균형"),
               L("Each sex has one SBI-553-first and one vehicle-first mouse; the movement drop follows the order, not the sex.",
                 "성별마다 SBI-553 먼저 1마리, vehicle 먼저 1마리; 움직임 감소는 성별이 아니라 순서를 따름.")),
              (L("next", "다음"), L("more mice per sex", "성별당 더 많은 마우스"),
               L("A sex effect needs >= 4-6 mice per sex; until then, follow the two female patterns in new mice.",
                 "성별 효과에는 성별당 4-6마리 이상 필요; 그때까지 새 마우스에서 암컷 패턴 2개를 확인."))],
             nline + L(" Descriptive (no test); use these as hypotheses for the next cohort.", " 기술 통계 (검정 없음); 다음 코호트의 가설로 사용."), "")
    return len(dk.prs.slides) - n0


def main(argv=None):
    import make_new_deck as ND
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "kr"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dk = ND.Deck(a.lang, "")
    dk.n_txt = "n = 2 females + 2 males" if a.lang != "kr" else "n = 암컷 2 + 수컷 2"
    n = add_sex_slides(dk, a.lang)
    dk.prs.save(a.out)
    print("->", a.out, n, "slides")


if __name__ == "__main__":
    main()
