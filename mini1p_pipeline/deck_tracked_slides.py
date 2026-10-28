"""deck_tracked_slides.py - the 'same neurons on both days' slides (ca_tracked_crossover.py, 2026-10-23) for make_new_deck.py,
Day 1 + Day 2 decks only. Added NEXT to the population / responsive slides (nothing replaced):
  summary      every event, the tracked neurons: after SBI-553 vs after vehicle, and baseline-normalized (TX0)
  responsive   % UP / DOWN, the change from the same-day baseline, every neuron of each day (TX3)
  per event    the same neurons in all four sessions (heatmaps, TX1), baseline-normalized heatmaps (TX4) and each neuron through
               the four sessions (TX2); pin prick first, the other events marked [BACKUP] (grey + hidden in the PI deck)
Every sentence is computed from analysis\\tracked_crossover\\tracked_summary.json. 10-23 independent review: the drug comparison
is also a day comparison (SBI-553 on Day 1 in 5941 / 5939, on Day 2 in 5937 / 1191) - the balanced estimate and the day estimate
are given; the mixed model was replaced by a mouse-level t-test; p with 3 decimals below 0.1.
"""
import json
import os

OUT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\tracked_crossover"
TAG = lambda ev: ev.split()[0].lower()
BACKUP = "[BACKUP] "                       # notes prefix: assemble_master greys and hides these slides in the PI deck
ORDER = "SBI-553 on Day 1 in 5941, 5939 and on Day 2 in 5937, 1191"
ORDER_KR = "SBI-553: 5941·5939는 Day 1, 5937·1191은 Day 2"


def _p(p):
    if p is None:
        return "n.a."
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def _st(p):
    return "n.a." if p is None else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def add_p7_condition_slide(dk, lg):
    """10-23 (user): the session-level brain-behaviour plot split by condition - does a relation of the baseline sessions break
    after the injection? (ca_p7_by_condition.py)"""
    import pandas as pd
    ea = os.path.join(os.path.dirname(OUT), "event_analysis")
    cf = os.path.join(ea, "P7c_by_condition.csv")
    fig = os.path.join(ea, "P7c_by_condition" + ("_kr" if lg == "kr" else "") + ".png")
    if not (os.path.isfile(cf) and os.path.isfile(fig)):
        return 0
    L = lambda en, kr: kr if lg == "kr" else en
    R = pd.read_csv(cf)
    A = R[R.group == "activated"]
    b = A[A.condition == "baseline"]
    rv = A[A.condition == "vehicle"].mean_residual_from_baseline_line.mean()
    rs = A[A.condition == "SBI-553"].mean_residual_from_baseline_line.mean()
    lo, hi = b.rho.min(), b.rho.max()
    ps = ", ".join(f"{r.behaviour} {r.rho:+.2f} ({_p(r.p)})" for r in b.itertuples())
    dk.slide(L("BRAIN AND BEHAVIOUR", "뇌와 행동"),
             L("Do sessions with more behaviour have more responsive neurons? Split by condition",
               "행동이 많은 세션에 반응 뉴런도 많은가? 조건별로 나눔"),
             L(f"Baseline sessions show no relation either (rho {lo:+.2f} to {hi:+.2f}, all ns), so there is no baseline line to break; "
               f"after both injections sessions sit above it (+{rv:.0f} vehicle, +{rs:.0f} points SBI-553)",
               f"기준 세션에도 관계 없음 (rho {lo:+.2f} ~ {hi:+.2f}, 모두 ns) - 깨질 기준 관계가 없음; 두 주사 후 세션은 그 선보다 위 "
               f"(vehicle +{rv:.0f}, SBI-553 +{rs:.0f} %p)"),
             fig,
             L("One point = one session (baseline 8, after vehicle 4, after SBI-553 4); each condition with its own line. n = 4 mice; "
               "Test: Spearman per condition (descriptive: with 4 sessions the smallest p is 0.083).",
               "점 하나 = 세션 하나 (기준 8, vehicle 후 4, SBI-553 후 4); 조건마다 자기 직선. n = 4마리; 검정: 조건별 Spearman (기술 통계: "
               "세션 4개로는 최소 p 0.083)."),
             L("Activated neurons, baseline sessions: " + ps + ". Points above the line = more activated neurons than the baseline "
               "sessions at the same amount of behaviour (mean residual over the 5 behaviours). Two baseline sessions per mouse, so the 8 "
               "baseline points are not independent.",
               "활성화 뉴런, 기준 세션: " + ps + ". 선 위 = 같은 행동량의 기준 세션보다 활성화 뉴런이 많음 (5개 행동 평균 잔차). 마우스마다 기준 세션 "
               "2개라 기준 점 8개는 독립이 아님."))
    return 1


def add_tracked_slides(dk, lg, short=False):
    js = os.path.join(OUT, "tracked_summary.json")
    if not os.path.isfile(js):
        return 0
    J = json.load(open(js, encoding="utf-8"))
    L = lambda en, kr: kr if lg == "kr" else en
    f = lambda nm: os.path.join(OUT, nm + ("_kr" if lg == "kr" else "") + ".png")
    S = {r["event"]: r for r in J["tests"]}
    SA = {r["event"]: r for r in J["tests_A"]}
    n, nA = J["n_tracked"], J["n_tracked_A"]
    pm = ", ".join(f"{m} {k}" for m, k in J["per_mouse"].items())
    sec = L("SAME NEURONS", "같은 뉴런")
    k0 = 0
    # ------------------------------------------------------------------ summary
    sig = [(ev, r) for ev, r in S.items() if r.get("n") and ((r["p_after"] or 1) < 0.05 or (r["p_norm"] or 1) < 0.05)]
    days = [r["day_after"] for r in S.values() if r.get("n")]
    dlo, dhi = min(days), max(days)
    pin = S.get("Pin prick", {})
    if sig:
        key = L("Same neurons: " + "; ".join(f"{ev} after {r['med_after']:+.3f} z ({_p(r['p_after'])}), normalized {r['med_norm']:+.3f} z "
                                            f"({_p(r['p_norm'])})" for ev, r in sig[:2]),
                "같은 뉴런: " + "; ".join(f"{ev} 주사 후 {r['med_after']:+.3f} z ({_p(r['p_after'])}), 정규화 {r['med_norm']:+.3f} z "
                                       f"({_p(r['p_norm'])})" for ev, r in sig[:2]))
    else:
        key = L(f"Same {n} neurons: no drug difference, raw or baseline-normalized (all ns); responses after the injection were larger on "
                f"Day 2 in both drug orders (+{dlo:.3f} to +{dhi:.3f} z)",
                f"같은 뉴런 {n}개: 약물 차이 없음, 그대로도 기준 정규화해도 (모두 ns); 주사 후 반응은 투여 순서와 상관없이 Day 2에서 더 큼 "
                f"(+{dlo:.3f} ~ +{dhi:.3f} z)")
    det = "; ".join(f"{ev}: after {r['med_after']:+.3f} ({_p(r['p_after'])}; mouse-level {_p(r['p_mouse_after'])}; balanced "
                    f"{r['bal_after']:+.3f}, day {r['day_after']:+.3f}) / normalized {r['med_norm']:+.3f} ({_p(r['p_norm'])}; mouse-level "
                    f"{_p(r['p_mouse_norm'])}; balanced {r['bal_norm']:+.3f})" for ev, r in S.items() if r.get("n"))
    detA = "; ".join(f"{ev} {r['med_after']:+.3f} ({_p(r['p_after'])}) / {r['med_norm']:+.3f} ({_p(r['p_norm'])})"
                     for ev, r in SA.items() if r.get("n"))
    dk.slide(sec, L("Same neurons on both days: SBI-553 vs vehicle, baseline-normalized",
                    "두 날 모두 있는 같은 뉴런: SBI-553 vs vehicle, 기준 정규화"), key, f("TX0_summary"),
             L(f"Each tracked neuron compared with itself: A after SBI-553 - after vehicle; B (SBI-553 - its baseline) - (vehicle - its "
               f"baseline). n = {n} neurons, 4 mice; {ORDER}. Test: Wilcoxon signed-rank paired by neuron.",
               f"추적 뉴런을 자기 자신과 비교: A SBI-553 후 - vehicle 후; B (SBI-553 - 그날 기준) - (vehicle - 그날 기준). n = 뉴런 {n}개, "
               f"4마리; {ORDER_KR}. 검정: 뉴런별 짝지은 Wilcoxon signed-rank."),
             L(f"Response = each neuron's event-analysis response in the event's best window. Medians (z) with the neuron-level Wilcoxon p, "
               f"the mouse-level t-test on the 4 mouse means (df = 3), the estimate balanced over the drug order (mean of the SBI-553-first "
               f"and SBI-553-second groups) and the day estimate (Day 2 - Day 1): {det}. Class-A pairs only (same neuron, n = {nA}): "
               f"{detA}. Per mouse ({pm}): markers. Rank / sign tests across 4 mice cannot reach p < 0.05 (smallest 0.125). Neurons of one "
               f"day only are not used here (they are in the % slides).",
               f"반응 = 각 뉴런의 최적 창 반응. 중앙값 (z), 뉴런 단위 Wilcoxon p, 마우스 평균 4개의 t 검정 (df = 3), 투여 순서 균형 추정 (SBI-553 "
               f"먼저 / 나중 그룹의 평균)과 날 추정 (Day 2 - Day 1): {det}. A 등급만 (같은 뉴런, n = {nA}): {detA}. 마우스별 ({pm}): 표식. "
               f"마우스 4마리로 순위 / 부호 검정은 p < 0.05 불가 (최소 0.125). 하루에만 있는 뉴런은 여기 쓰지 않음 (% 슬라이드에 포함)."))
    k0 += 1
    # ------------------------------------------------------------------ % responsive, baseline-normalized
    R = [r for r in J["responsive"] if r.get("p") is not None]
    sigr = sorted([r for r in R if r["p"] < 0.05], key=lambda r: r["p"])

    def dsplit(r):
        d1, d2 = r.get("p_d1"), r.get("p_d2")
        if d1 is not None and d2 is not None:
            if d2 < 0.05 <= d1:
                return L("Day 2 only", "Day 2만")
            if d1 < 0.05 <= d2:
                return L("Day 1 only", "Day 1만")
        return L("both days", "두 날")
    if sigr:
        key = L("Change from the same-day baseline (exploratory, 14 tests): " + "; ".join(
                    f"{r['event']} {r['kind'].upper()} {r['chg_veh']:+.0f} vs {r['chg_sbi']:+.0f} points ({dsplit(r)})" for r in sigr[:2]),
                "같은 날 기준 대비 변화 (탐색적, 검정 14개): " + "; ".join(
                    f"{r['event']} {r['kind'].upper()} {r['chg_veh']:+.0f} vs {r['chg_sbi']:+.0f} %p ({dsplit(r)})" for r in sigr[:2]))
    else:
        key = L("No % UP / DOWN change from the same-day baseline differs between the two drug days (all ns)",
                "같은 날 기준 대비 % UP / DOWN 변화는 두 약물 날 사이에 차이 없음 (모두 ns)")
    rows = "; ".join(f"{r['event']} {r['kind']}: baseline {r['base_veh']:.0f} / {r['base_sbi']:.0f}%, change {r['chg_veh']:+.1f} / "
                     f"{r['chg_sbi']:+.1f} points ({_p(r['p'])}; Day 1 {r.get('chg_veh_d1', float('nan')):+.1f} / "
                     f"{r.get('chg_sbi_d1', float('nan')):+.1f}, {_p(r.get('p_d1'))}; Day 2 {r.get('chg_veh_d2', float('nan')):+.1f} / "
                     f"{r.get('chg_sbi_d2', float('nan')):+.1f}, {_p(r.get('p_d2'))}; mice lower on the SBI-553 day {r['mice_lower']}/4)"
                     for r in R)
    dk.slide(sec, L("% UP and DOWN, baseline-normalized (every neuron of each day)", "% UP·DOWN, 기준 정규화 (각 날의 모든 뉴런)"), key,
             f("TX3_responsive_norm"),
             L("% UP / DOWN after the injection minus the same neurons' baseline of that day, vehicle day vs SBI-553 day (216 / 246 "
               "neurons). n = 4 mice (markers); Test: Mann-Whitney U on the per-neuron change (neurons pooled, uncorrected, exploratory).",
               "주사 후 UP / DOWN % - 그날 같은 뉴런의 기준, vehicle 날 vs SBI-553 날 (뉴런 216 / 246). n = 4마리 (표식); 검정: "
               "뉴런별 변화의 Mann-Whitney U (뉴런 합산, 보정 없음, 탐색적)."),
             L("Every neuron of each day is used (baseline and injection of the same day are the same neurons). Vehicle day / SBI-553 "
               "day, and the same within each imaging day (different mice per drug there): " + rows + f". {ORDER}.",
               "각 날의 모든 뉴런 사용 (같은 날 기준과 주사는 같은 뉴런). vehicle 날 / SBI-553 날, 그리고 촬영일별 (그날은 약물마다 다른 "
               "마우스): " + rows + f". {ORDER_KR}."))
    k0 += 1
    # ------------------------------------------------------------------ per event: pin prick first, the rest backup
    evs = [e for e in S if S[e].get("n")]
    evs = ["Pin prick"] + [e for e in evs if e != "Pin prick"]
    for ev in evs:
        r = S[ev]
        bk = BACKUP if ev != "Pin prick" else ""
        if not os.path.isfile(f(f"TX1_heat_{TAG(ev)}")):
            continue
        dk.slide(sec, L(f"{ev}: the same neurons in all four sessions", f"{ev}: 네 세션의 같은 뉴런"),
                 L(f"{ev}: median response of the {r['n']} tracked neurons {r['med_vb']:+.3f} / {r['med_vi']:+.3f} (vehicle day) and "
                   f"{r['med_sb']:+.3f} / {r['med_si']:+.3f} z (SBI-553 day), baseline / after",
                   f"{ev}: 추적 뉴런 {r['n']}개의 반응 중앙값 {r['med_vb']:+.3f} / {r['med_vi']:+.3f} (vehicle 날), "
                   f"{r['med_sb']:+.3f} / {r['med_si']:+.3f} z (SBI-553 날), 기준 / 후"),
                 f(f"TX1_heat_{TAG(ev)}"),
                 L(f"One row = one neuron followed on both days, the same row in all four panels; colour bar = z minus the 3 s before "
                   f"the event. n = {r['n']} neurons, 4 mice; descriptive (no test).",
                   f"한 줄 = 두 날 모두 추적한 뉴런 하나, 네 패널에서 같은 줄; 색 막대 = 사건 전 3 s 기준 z. n = 뉴런 {r['n']}개, 4마리; "
                   f"기술 통계 (검정 없음)."),
                 bk + L("Trial-mean z of each neuron minus its 3 s before the event (display); rows sorted within each mouse by the mean "
                        f"of the four responses (best window); black bar on top = the response window. {ORDER}.",
                        f"각 뉴런의 trial 평균 z - 사건 전 3 s (표시용); 행 = 마우스 안에서 네 반응(최적 창)의 평균 순; 위 검은 막대 = 반응 창. "
                        f"{ORDER_KR}."))
        k0 += 1
        if os.path.isfile(f(f"TX4_heatnorm_{TAG(ev)}")):
            dk.slide(sec, L(f"{ev}: the same neurons, baseline-normalized", f"{ev}: 같은 뉴런, 기준 정규화"),
                     # 10-25: only the median paired difference ('d' read like Cohen's d, and two medians next to a
                     # difference of the opposite sign confused); the two day medians stay in the notes
                     L(f"{ev}: median paired difference (SBI-553 change - vehicle change) {r['med_norm']:+.3f} z, {_p(r['p_norm'])} "
                       f"(neurons pooled)",
                       f"{ev}: 짝지은 차이 중앙값 (SBI-553 변화 - vehicle 변화) {r['med_norm']:+.3f} z, {_p(r['p_norm'])} (뉴런 합산)"),
                     f(f"TX4_heatnorm_{TAG(ev)}"),
                     L(f"Each tracked neuron's trace after the injection minus its trace at that day's baseline: vehicle day, SBI-553 day, "
                       f"and their difference; rows as in the four-session heatmap. n = {r['n']} tracked neurons, 4 mice; Wilcoxon "
                       f"signed-rank (neurons pooled, exploratory).",
                       f"추적 뉴런마다 주사 후 trace - 그날 기준 trace: vehicle 날, SBI-553 날, 그 차이; 행은 네 세션 히트맵과 같음. n = 뉴런 "
                       f"{r['n']}개 (추적), 4마리; Wilcoxon signed-rank (뉴런 합산, 탐색적)."),
                     bk + L(f"Median change from the same-day baseline: SBI-553 day {r['med_chg_sbi']:+.3f} z, vehicle day "
                            f"{r['med_chg_veh']:+.3f} z (medians of each day; the key gives the median of the per-neuron paired "
                            f"differences, which need not equal their difference). "
                            f"Red = the neuron rose more after the injection than at its baseline. Difference panel: red = larger change on "
                            f"the SBI-553 day. {ORDER}: this is also a Day 1 vs Day 2 difference (balanced estimate {r['bal_norm']:+.3f} z, "
                            f"day estimate {r['day_norm']:+.3f} z).",
                            f"같은 날 기준 대비 변화 중앙값: SBI-553 날 {r['med_chg_sbi']:+.3f} z, vehicle 날 {r['med_chg_veh']:+.3f} z (각 날의 "
                            f"중앙값; 핵심 문장은 뉴런별 짝지은 차이의 중앙값이라 두 중앙값의 차와 같지 않을 수 있음). "
                            f"빨강 = 주사 후 기준보다 더 증가. 차이 패널: 빨강 = SBI-553 날 변화가 더 큼. {ORDER_KR}: Day 1 vs Day 2 차이이기도 함 "
                            f"(균형 추정 {r['bal_norm']:+.3f} z, 날 추정 {r['day_norm']:+.3f} z)."))
            k0 += 1
        # 10-25: said positively from the per-mouse medians ('0/4 mice have a smaller change' was a double negative)
        mm = [v for v in (r.get("mice_norm") or {}).values() if v is not None]
        npos, nneg = sum(v > 0 for v in mm), sum(v < 0 for v in mm)
        if npos >= nneg:
            cnt = L(f"{npos}/{len(mm)} mice have a larger change on the SBI-553 day",
                    f"SBI-553 날 변화가 더 큰 마우스 {npos}/{len(mm)}")
        else:
            cnt = L(f"{nneg}/{len(mm)} mice have a smaller change on the SBI-553 day",
                    f"SBI-553 날 변화가 더 작은 마우스 {nneg}/{len(mm)}")
        dk.slide(sec, L(f"{ev}: each neuron through the four sessions", f"{ev}: 뉴런 하나하나, 네 세션"),
                 L(f"Median paired difference (SBI-553-day change - vehicle-day change) {r['med_norm']:+.3f} z, {_p(r['p_norm'])} "
                   f"(neurons pooled); {cnt} (mouse-level {_p(r['p_mouse_norm'])})",
                   f"짝지은 차이 중앙값 (SBI-553 날 변화 - vehicle 날 변화) {r['med_norm']:+.3f} z, {_p(r['p_norm'])} (뉴런 합산); "
                   f"{cnt} (마우스 단위 {_p(r['p_mouse_norm'])})"),
                 f(f"TX2_neurons_{TAG(ev)}"),
                 L(f"A: each tracked neuron through the four sessions; B: its change on the SBI-553 day vs on the vehicle day; C: per-mouse "
                   f"median paired difference. n = {r['n']} tracked neurons, 4 mice; Test: Wilcoxon signed-rank paired by neuron, neurons "
                   f"pooled, exploratory (uncorrected); mouse-level t-test on the 4 mouse means.",
                   f"A: 추적 뉴런 하나하나의 네 세션; B: SBI-553 날 변화 vs vehicle 날 변화; C: 마우스별 짝지은 차이 중앙값. n = 추적 뉴런 "
                   f"{r['n']}개, 4마리; 검정: 뉴런별 짝지은 Wilcoxon signed-rank, 뉴런 합산, 탐색적 (보정 없음); 마우스 평균 4개의 마우스 단위 "
                   f"t 검정."),
                 bk + L(f"After SBI-553 vs after vehicle (the same neurons): {r['med_after']:+.3f} z, {_p(r['p_after'])} (mouse-level "
                        f"{_p(r['p_mouse_after'])}); the two baselines: {_p(r['p_vb_sb'])}; paired difference: {r['med_norm']:+.3f} z, {_p(r['p_norm'])} "
                        f"(mouse-level {_p(r['p_mouse_norm'])}); balanced over drug order {r['bal_norm']:+.3f} z, day estimate "
                        f"{r['day_norm']:+.3f} z. Class-A pairs only: {SA[ev]['med_norm']:+.3f} z, {_p(SA[ev]['p_norm'])}. {ORDER}.",
                        f"SBI-553 후 vs vehicle 후 (같은 뉴런): {r['med_after']:+.3f} z, {_p(r['p_after'])} (마우스 단위 {_p(r['p_mouse_after'])}); "
                        f"두 기준: {_p(r['p_vb_sb'])}; 짝지은 차이: {r['med_norm']:+.3f} z, {_p(r['p_norm'])} (마우스 단위 {_p(r['p_mouse_norm'])}); 투여 "
                        f"순서 균형 {r['bal_norm']:+.3f} z, 날 추정 {r['day_norm']:+.3f} z. A 등급만: {SA[ev]['med_norm']:+.3f} z, "
                        f"{_p(SA[ev]['p_norm'])}. {ORDER_KR}."))
        k0 += 1
    return k0
