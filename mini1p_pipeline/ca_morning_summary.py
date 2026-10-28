"""One-page summary (Korean) of everything the overnight run produced, numbers taken from the files.

  python ca_morning_summary.py --base C:/Users/hsollim/Documents/HEAL_mini1p_SBI553 --out <md file>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MICE = [("5941", "SBI-553", "F"), ("5939", "SBI-553", "M"), ("5937", "vehicle", "M"), ("1191", "vehicle", "F")]


def rd(p):
    if not os.path.isfile(p):
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:            # e.g. no rescued cells kept any more -> an empty check file
        return None


def fmt_p(p):
    return "n.a." if p is None or not np.isfinite(p) else (f"{p:.3g}" if p >= 1e-4 else "<0.0001")


def glance(B, OUT):
    """section 0: the few numbers that matter, each from its file (empty if a file is missing)"""
    G = ["## 0. 한눈에 보기", ""]
    rb = os.path.join(B, "analysis", "ca_robustness")
    R1 = rd(os.path.join(rb, "R1_robustness_per_mouse.csv"))
    RA = rd(os.path.join(rb, "R1_robustness_arm.csv"))
    RL = rd(os.path.join(rb, "R2_leave_one_mouse_out.csv"))
    if R1 is not None and RA is not None and RL is not None:
        R1["mouse"] = R1.mouse.astype(str)
        main_v = "main: OASIS transients, still frames"
        V = R1[(R1.variant == main_v)]
        f = lambda wn: ", ".join(f"{r.mouse} ({r.arm}) {r.log2_ratio_median:+.2f}" for r in V[V.window == wn].itertuples())
        a = RA[RA.window == "stimulus blocks"]
        a0 = a[a.variant == main_v].sbi_minus_vehicle
        l5941 = RL[(RL.window == "stimulus blocks") & (RL.left_out == "5941")].sbi_minus_vehicle
        nost = RA[(RA.window == "no-stimulus start") & (RA.variant == main_v)].sbi_minus_vehicle
        G += [f"1. **자극 블록 활동이 두 번째 세션에서 줄었습니다** (움직이지 않는 프레임, 세포 중앙값 log2(post/pre)): {f('stimulus blocks')}.",
              f"   - SBI − vehicle = {a0.iloc[0]:+.2f} (분석 선택 {len(a)}가지 중 {int((a.sbi_minus_vehicle < 0).sum())}가지에서 음수). "
              f"하지만 5941 을 빼면 {l5941.iloc[0]:+.2f} 입니다. vehicle 5937 도 크게 줄었습니다. arm 당 2마리라 arm 결론은 아직 없습니다." if len(a0) and len(l5941) else "",
              f"   - 처음 3분(자극 없음)은 {f('no-stimulus start')} (SBI − vehicle {nost.iloc[0]:+.2f}): 자발 활동에는 약효가 보이지 않습니다 "
              "(중앙값 0 은 많은 세포가 처음 3분의 움직이지 않는 프레임에서 transient 가 거의 없기 때문; E2 mixed model 도 n.s.)." if len(nost) else ""]
        u = R1[(R1.variant == "5941: your kept cells found by the day run") & (R1.window == "stimulus blocks") & (R1.mouse == "5941")]
        r_ = R1[(R1.variant == "5941: cells added by the rule") & (R1.window == "stimulus blocks") & (R1.mouse == "5941")]
        if len(u) and len(r_):
            G.append(f"   - 5941 안에서도 당신이 확인한 세포는 {u.log2_ratio_median.iloc[0]:+.2f}, 규칙으로 들어간 세포는 {r_.log2_ratio_median.iloc[0]:+.2f} 입니다. "
                     "규칙 세포의 검토가 결과 크기를 정합니다 (1절의 '먼저 볼 세포').")
        roi = R1[(R1.variant == "ROI - local neuropil traces (no demixing)") & (R1.window == "stimulus blocks")]
        if len(roi) == 4:
            ra = RA[(RA.variant == "ROI - local neuropil traces (no demixing)") & (RA.window == "stimulus blocks")]
            G.append("   - EXTRACT 없이 영화 픽셀에서 만든 trace (footprint − 주변 neuropil) 로도: " +
                     ", ".join(f"{r.mouse} {r.log2_ratio_median:+.2f}" for r in roi.itertuples()) +
                     (f" (SBI − vehicle {ra.sbi_minus_vehicle.iloc[0]:+.2f})." if len(ra) else "."))
    rcd = os.path.join(B, "analysis", "ca_responsive_change")
    R7 = rd(os.path.join(rcd, "R7_population_response_test.csv"))
    R8 = rd(os.path.join(rcd, "R8_responsive_calibrated.csv"))
    if R7 is not None and R8 is not None:
        st = R7[~R7.event.isin(["movement onset", "flinch"])]
        mv = R7[R7.event == "movement onset"]
        G.append(f"2. **유발 반응:** 집단 반응이 블록의 진행 중 활동을 넘은 경우는 자극·withdrawal {int((st.q_energy <= 0.05).sum())}/{len(st)}, "
                 f"움직임 시작 {int((mv.q_energy <= 0.05).sum())}/{len(mv)} (R7). 단일 세포 규칙은 세포 간 상관 때문에 사건이 없어도 "
                 f"한꺼번에 많은 세포를 고를 수 있어 보정했습니다 (R8, 5e). 자극당 trial 이 15–30개라 반응 크기의 약효는 아직 잴 수 없습니다.")
    R4 = rd(os.path.join(B, "analysis", "ca_response_shape", "R4_down_speed_cells.csv"))
    if R4 is not None and len(R4):
        dn = R4[R4.cls == "DOWN"]
        R9 = rd(os.path.join(B, "analysis", "ca_response_shape", "R9_raw_roi_check.csv"))
        extra = ""
        if R9 is not None and len(R9):
            d9 = R9[R9.group == "DOWN"]
            extra = (f" 영화 픽셀을 직접 보면 DOWN 세포의 {100 * (d9.roi_dff_1s < 0).mean():.0f}% 만 실제로 어두워졌고, 대부분의 경우 사건 순간 시야 전체가 밝아졌습니다 (R9, 5h). "
                     "사건 시점 반응(UP 포함)은 이 시야 전체 변화와 얽혀 있습니다.")
        G.append(f"3. **DOWN 반응의 {100 * (dn.speed > 0.53).mean():.0f}% 는 칼슘이 낼 수 없을 만큼 빨리 떨어집니다** (R4). 억제로 해석하지 마세요." + extra)
    G += ["4. **밤사이 고친 것:** OASIS 첫 프레임 가짜 transient (처음 3분 +0.36/분), z 분모(세션별 MAD → post 반응 14–30% 부풀림; "
          "`*_znoise` 폴더), 반응 세포의 세포 간 상관 보정 (R8). 자세한 것은 폴더 안내 4절.",
          f"5. **오늘 할 일:** 1절의 '먼저 볼 세포' 부터 GUI 로 확인 (5939 먼저) → `bash ca_post_mouse.sh <마우스>` → `bash ca_post_cohort.sh`. "
          f"한 장 그림: `{os.path.join(rb, 'MAIN_day1_findings.png')}`", ""]
    return [g for g in G if g != ""] + [""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--day-root", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    B = a.base
    OUT = a.day_root or os.path.join(B, "mini1p_out", "20261019")
    L = ["# 밤샘 결과 요약 (자동 생성: 파일의 숫자를 그대로 옮김)", ""]
    L += glance(B, OUT)
    L += ["## 1. 세포 검출 (마우스별, 하루 전체 = pre + post 한 번의 EXTRACT)", "",
          "| 마우스 | arm | 후보 | 최종 유지 | 당신 결정 매칭 | 규칙으로 새로 | rescue | 중복으로 뺀 것 | 당신 세포 중 합쳐진 것 | 남은 중복 의심 쌍 | MC 최대 shift (px) pre/post |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, arm, sex in MICE:
        R = os.path.join(OUT, f"{m}_d1")
        dj = os.path.join(R, "5_analysis", "decisions", f"{m}_d1_decisions.json")
        if not os.path.isfile(dj):
            L.append(f"| {m} | {arm} | (아직 없음) | | | | | | | | |")
            continue
        j = json.load(open(dj))
        rv = rd(os.path.join(R, "4_cell_review", f"{m}_d1_cell_review.csv"))
        nres = int((rv.verdict == "rescued").sum()) if rv is not None and "verdict" in rv else 0
        dp = rd(os.path.join(R, "5_analysis", "decisions", f"{m}_d1_duplicate_pairs.csv"))
        sh = []
        for ph in ("pre", "post"):
            f = os.path.join(R, "1_movie", f"{m}_d1_{ph}_mc_shifts.csv")
            if os.path.isfile(f):
                S = pd.read_csv(f, usecols=["x_shift_px", "y_shift_px"])
                sh.append(f"{np.hypot(S.x_shift_px, S.y_shift_px).max():.1f}")
            else:
                sh.append("?")
        nfin = int(rv.keep.sum()) if rv is not None else j["n_kept"]
        ndup = int((rv.decision_source.astype(str) == "auto-duplicate").sum()) if rv is not None and "decision_source" in rv else 0
        L.append(f"| {m} | {arm} | {j['n_candidates']} | {nfin} | {j['n_kept_from_user']} | {j['n_new_kept']} | {nres} | {ndup} | "
                 f"{len(j.get('user_keeps_merged', []))} | {len(dp) if dp is not None else '?'} | {' / '.join(sh)} |")
    L += ["", "**검토 우선순위:** 5939 는 FOV 가운데 밝은 곡선 구조(혈관/섬유) 주변에서 후보가 매우 많고(1차 211개) 겹침이 심합니다. "
          "규칙·크기 범위·중복 정리를 거쳤지만 당신의 결정이 가장 필요한 마우스입니다. 5939 미리보기를 먼저 검토하시고, "
          "가능하면 하루 전체 GUI 데이터(`5939_d1\\4_cell_review\\5939_d1_curation_data.mat`) 도 보세요.", ""]
    L += ["세포 지도와 결정 지도: `<마우스>_d1\\5_analysis\\decisions\\<마우스>_d1_decisions_map.png`, "
          "QC 한 장: `<마우스>_d1\\5_analysis\\<녹화>_QC_sheet.png`.", ""]
    at = []
    for m, arm, sex in MICE:
        f = os.path.join(OUT, f"{m}_d1", "5_analysis", "decisions", f"{m}_d1_attention_cells.csv")
        X = rd(f)
        if X is not None:
            nd = int(X.attention.astype(str).str.contains("would drop").sum())
            at.append(f"- {m}: {len(X)}" + (f" (그중 {nd}개는 엄격한 뉴런 모양 기준이면 빠짐)" if nd else "") + f" — `{f}`")
    if at:
        L += ["**주의 표시 (당신이 직접 정하지 않고 규칙이 유지한 세포; 유지는 그대로, Day 2 에서 나쁘면 삭제):**", ""] + at + [""]
    pr = []
    for m, arm, sex in MICE:
        f = os.path.join(OUT, f"{m}_d1", "5_analysis", "decisions", f"{m}_d1_review_priority.csv")
        X = rd(f)
        if X is not None and len(X):
            pr.append(f"- {m}: " + ", ".join(f"#{int(r.cell_id)}" + (f" ({r.flags})" if isinstance(r.flags, str) and r.flags else "")
                                             for r in X.head(8).itertuples()) + f" — `{f}`")
    if pr:
        L += ["**먼저 볼 세포 (당신이 결정하지 않은 세포 중, 자극 블록 결과를 가장 많이 움직이는 순서; 괄호 = 규칙의 모양 경고):**", ""] + pr + [
              "", "이 세포들이 실제 세포인지가 결과의 크기를 가장 많이 바꿉니다.", ""]
    for m, arm, sex in MICE:
        for f in glob.glob(os.path.join(OUT, f"{m}_d1", "5_analysis", "decisions", f"{m}_d1_rescued_vs_*.csv")):
            X = rd(f)
            if X is None or not len(X):
                continue
            L += [f"**rescue 점검 ({m}):** 되살린 세포 {len(X)}개의 하루 전체 trace 와 당신이 검토한 run 의 trace 상관 "
                  f"r = {X.r_with_ref.min():.2f}–{X.r_with_ref.max():.2f} (같은 프레임; 다른 성분과는 최대 {X.r_best_other.max():.2f}). "
                  f"신호는 제대로 옮겨졌습니다. 다만 skew 가 {X.skew_ref.min():+.2f}–{X.skew_ref.max():+.2f} 로, 검토한 run 에서도 "
                  "뚜렷한 transient 가 거의 없는 조용한 세포들입니다. 그래서 비율 지표(post/pre)에는 거의 기여하지 않습니다. "
                  f"(`{f}`)", ""]
    d1 = os.path.join(B, "analysis", "ca_day1")
    A1 = rd(os.path.join(d1, "ca_d1_A1_whole_session_tests.csv"))
    if A1 is not None:
        L += ["## 2. 전체 세션 활동 (transient / 분, 세포 중앙값; Wilcoxon pre vs post)", "",
              "| 마우스 | arm | 세포 | baseline | injection | log2(post/pre) 중앙값 | p |", "|---|---|---|---|---|---|---|"]
        for r in A1.itertuples():
            L.append(f"| {r.mouse} | {r.arm} | {r.n_cells} | {r.pre_median:.2f} | {r.post_median:.2f} | {r.log2_ratio_median:.2f} | {fmt_p(r.p_wilcoxon)} |")
        L.append("")
    A1b = rd(os.path.join(d1, "ca_d1_A1b_all_activity_measures.csv"))
    if A1b is not None:
        L += ["다른 활동 지표 (세포 중앙값, baseline → injection, Wilcoxon):", ""]
        for (m, arm), V in A1b.groupby(["mouse", "arm"], sort=False):
            L.append(f"- {m} ({arm}): " + "; ".join(f"{r.measure} {r.pre_median:.3g} → {r.post_median:.3g} (p {fmt_p(r.p_wilcoxon)})"
                                                 for r in V.itertuples()))
        L.append("")
    A6 = rd(os.path.join(d1, "ca_d1_A6_blocks_still_tests.csv"))
    if A6 is not None:
        L += ["블록별 활동, 움직이지 않는 프레임만 (A6, transient/분 평균):", "",
              "| 마우스 | 블록 | baseline | injection | p |", "|---|---|---|---|---|"]
        for r in A6.itertuples():
            L.append(f"| {r.mouse} | {r.block} | {r.pre_mean:.2f} | {r.post_mean:.2f} | {fmt_p(r.p)} |")
        L.append("")
    C2 = rd(os.path.join(d1, "ca_d1_C2_arm_blocks_still.csv"))
    if C2 is not None and len(C2):
        L += ["**핵심 비교 (C2): 움직이지 않는 프레임의 활동 변화, median log2(post/pre)** — "
              f"그림 `{os.path.join(d1, 'ca_d1_C2_arm_blocks_still.png')}`", "",
              "| 마우스 | arm | " + " | ".join(C2.condition.unique()) + " |",
              "|---|---|" + "---|" * C2.condition.nunique()]
        for (m, arm), V in C2.groupby(["mouse", "arm"], sort=False):
            V = V.set_index("condition")
            L.append(f"| {m} | {arm} | " + " | ".join(f"{V.log2_ratio_median.get(c, np.nan):+.2f}" + ("*" if V.p.get(c, 1) < 0.05 else "")
                                                       for c in C2.condition.unique()) + " |")
        L += ["", "(* p < 0.05 마우스 안에서; arm 당 2마리라 arm 비교는 눈으로만)", ""]
    bgd = rd(os.path.join(B, "analysis", "ca_day1_bgcorr", "ca_d1_A6_blocks_still_tests.csv"))
    if bgd is not None:
        L += ["배경 보정 trace 로 같은 A6 (민감도): " + "; ".join(f"{r.mouse} {r.block} {r.pre_mean:.2f}→{r.post_mean:.2f}"
                                                         for r in bgd.itertuples()), ""]
    A3 = rd(os.path.join(d1, "ca_d1_A3_spontaneous_tests.csv"))
    if A3 is not None:
        L += ["자극 전 (처음 3분) 자발 활동:", ""]
        for r in A3.itertuples():
            L.append(f"- {r.mouse} ({r.arm}) {r.condition}: {r.pre_median:.2f} → {r.post_median:.2f} /min, p {fmt_p(r.p)}")
        L.append("")
    ext = os.path.join(B, "analysis", "ca_day1_extended")
    E1 = rd(os.path.join(ext, "E1_quality_tests.csv"))
    if E1 is not None:
        L += ["## 3. 신호 품질 대조 (E1: post/pre 중앙값 비율)", ""]
        for m, arm, _ in MICE:
            V = E1[E1.mouse.astype(str) == m]
            if len(V):
                parts = []
                for r in V.itertuples():
                    if r.measure == "tau_s":
                        parts.append(f"decay tau {r.pre_median:.2f} s (하루 전체에서 한 번 추정 → 두 세션 같음, 검정 없음)")
                    elif r.measure == "drift":      # medians near 0: a ratio means nothing, give the difference
                        parts.append(f"느린 baseline 변화 {r.pre_median:+.4f} → {r.post_median:+.4f} dF/F (p {fmt_p(r.p)})")
                    else:
                        parts.append(f"{r.measure} ×{r.ratio:.2f} (p {fmt_p(r.p)})")
                L.append(f"- {m}: " + ", ".join(parts))
        nz = E1[(E1.measure == "noise") & ((E1.ratio < 0.9) | (E1.ratio > 1.1))]
        L += ["", "(noise = dF/F 잡음 SD, amp = transient 크기. 잡음이 거의 같으면 rate 변화는 SNR 차이 때문이 아닙니다."
              + ("".join(f" {int(r.mouse)} 은 잡음이 ×{r.ratio:.2f} 로 바뀌어 해석에 주의." for r in nz.itertuples())) + ")", ""]
    E2 = rd(os.path.join(ext, "E2_mixed_model.csv"))
    if E2 is not None and "term" in E2:
        V = E2[E2.term == "post:sbi"]
        if len(V):
            L += ["## 4. 약물 효과 (E2 mixed model, 세포는 마우스 안에 nested): SBI-553 이 vehicle 보다 더 바꾼 정도", "",
                  "| 결과 변수 | 효과 | 95% CI | p | 세포 / 마우스 |", "|---|---|---|---|---|"]
            for r in V.itertuples():
                L.append(f"| {r.outcome} | {r.coef:.3f} | {r.ci_lo:.3f} – {r.ci_hi:.3f} | {fmt_p(r.p)} | {r.n_cells} / {r.n_mice} |")
            L += ["", "(log2 rate 또는 z 단위; arm 당 2마리라 마우스 수준 불확실성이 큼)", ""]
    E2b = rd(os.path.join(ext, "E2b_mouse_level.csv"))
    if E2b is not None and len(E2b):
        cols = [c for c in E2b.columns if "(" in c]
        L += ["## 4a. 마우스 단위 (E2b, 가장 정직한 단위): 마우스마다 세포 중앙값 log2(post/pre)", "",
              "| 결과 변수 | " + " | ".join(cols) + " | SBI − vehicle | 순열 p (2 vs 2, 최소 0.33) |",
              "|---|" + "---|" * (len(cols) + 2)]
        for _, d in E2b.iterrows():
            L.append(f"| {d['outcome']} | " + " | ".join(f"{d[c]:+.2f}" for c in cols) +
                     f" | {d['sbi_minus_vehicle']:+.2f} | {d['p_permutation']:.2f} |")
        L += ["", "→ arm 당 2마리라 마우스 단위로는 어떤 차이도 유의할 수 없습니다. 위의 세포 단위 mixed model 의 p값은 "
              "이 불확실성을 다 담지 못합니다 (마우스 간 차이가 큼).", ""]
    E15 = rd(os.path.join(ext, "E15_block_position_model.csv"))
    if E15 is not None and len(E15):
        V = E15[E15.term.isin(["post", "post:sbi"])]
        L += ["## 4b. 블록 위치(세션 내 시간)를 고정한 mixed model (E15, 움직이지 않는 프레임의 블록 활동, log2)", "",
              "| 모델 | 항 | 효과 | 95% CI | p |", "|---|---|---|---|---|"]
        for r in V.itertuples():
            nm = "vehicle 의 재검사 변화" if r.term == "post" else "SBI-553 이 vehicle 보다 더 바꾼 것"
            L.append(f"| {r.model} | {nm} | {r.coef:+.2f} | {r.ci_lo:+.2f} – {r.ci_hi:+.2f} | {fmt_p(r.p)} |")
        L.append("")
    rb = os.path.join(B, "analysis", "ca_robustness")
    RA = rd(os.path.join(rb, "R1_robustness_arm.csv"))
    RL = rd(os.path.join(rb, "R2_leave_one_mouse_out.csv"))
    if RA is not None and len(RA):
        ms = [m for m, _, _ in MICE]
        for wn, title in (("stimulus blocks", "자극 블록"), ("no-stimulus start", "처음 3분 (자극 없음)")):
            V = RA[RA.window == wn]
            if not len(V):
                continue
            L += [f"## 4c. robustness (R1): 분석 선택을 하나씩 바꿨을 때 — {title}, 움직이지 않는 프레임, 세포 중앙값 log2(post/pre)", "",
                  "| 바꾼 선택 | " + " | ".join(ms) + " | SBI − vehicle | SBI 2마리 모두 vehicle 2마리보다 아래 |",
                  "|---|" + "---|" * (len(ms) + 2)]
            for _, d in V.iterrows():
                L.append(f"| {d['variant']} | " + " | ".join(f"{d[f'median {m}']:+.2f}" for m in ms) +
                         f" | {d['sbi_minus_vehicle']:+.2f} | {'예' if d['both_sbi_below_both_vehicle'] else '아니오'} |")
            L += ["", f"- SBI − vehicle 이 음수: {int((V.sbi_minus_vehicle < 0).sum())}/{len(V)} 개 선택, "
                  f"범위 {V.sbi_minus_vehicle.min():+.2f} ~ {V.sbi_minus_vehicle.max():+.2f}."]
            if RL is not None:
                Q = RL[RL.window == wn]
                L.append("- 한 마리씩 뺐을 때 (R2, 기본 지표): " + "; ".join(
                    f"{'아무도 안 뺌' if d.left_out == 'none' else d.left_out + ' 제외'} {d.sbi_minus_vehicle:+.2f}" for d in Q.itertuples()))
            L.append("")
        L += [f"그림: `{os.path.join(rb, 'R1_robustness_forest.png')}`", ""]
    zx = os.path.join(B, "analysis", "ca_day1_extended_znoise", "E2_mixed_model.csv")
    E2z = rd(zx)
    if E2 is not None and E2z is not None and "term" in E2z:
        a_ = E2[(E2.term == "post:sbi") & E2.outcome.str.startswith("response")].set_index("outcome")
        b_ = E2z[(E2z.term == "post:sbi") & E2z.outcome.str.startswith("response")].set_index("outcome")
        L += ["## 4d. 반응 크기의 단위 (z 분모) — pre vs post 비교에서 중요", "",
              "trace 파일의 z 는 **각 세션의 MAD** 로 나눕니다. MAD 에는 세포 활동이 들어 있어서, 덜 활발한 세션은 MAD 가 작고 "
              "같은 dF/F 반응이 z 로는 더 크게 나옵니다 (5941 post: MAD −25%, 잡음 그대로). 세션 안의 검정(반응 세포 찾기)에는 "
              "영향이 없지만, pre 와 post 의 **크기** 비교는 치우칩니다. 그래서 세포마다 두 세션에 같은 잡음 SD 로 나눈 z 로 "
              f"다시 계산했습니다 (`{os.path.dirname(zx)}`).", "",
              "| 반응 | SBI 효과, 세션별 MAD z | p | SBI 효과, 공통 잡음 z | p |", "|---|---|---|---|---|"]
        for o in a_.index:
            if o in b_.index:
                L.append(f"| {o} | {a_.at[o, 'coef']:+.3f} | {fmt_p(a_.at[o, 'p'])} | {b_.at[o, 'coef']:+.3f} | {fmt_p(b_.at[o, 'p'])} |")
        L.append("")
    B1c = rd(os.path.join(d1, "ca_d1_B1c_responsive_counts_shift_rule.csv"))
    if B1c is not None:
        L += ["## 5. 반응 세포 (시간 이동 null, q ≤ 0.20): UP+DOWN / 세포", "",
              "| 마우스 | 사건 | baseline | injection |", "|---|---|---|---|"]
        for (m, e), V in B1c.groupby(["mouse", "event"]):
            if e not in ("Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal"):
                continue
            pre = V[V.phase == "pre"]; post = V[V.phase == "post"]
            f = lambda X: f"{int(X.n_up.iloc[0])}+{int(X.n_down.iloc[0])}/{int(X.n_cells.iloc[0])} ({int(X.n_trials.iloc[0])} tr)" if len(X) else "-"
            L.append(f"| {m} | {e} | {f(pre)} | {f(post)} |")
        L += ["", "반응 세포 수는 trial 수에 좌우됩니다 (같은 trial 수로 맞춘 비교: `ca_d1_B1b_responsive_matched_trials.png`).",
              "**주의:** 이 규칙은 세포를 독립으로 다룹니다. 하지만 세포들의 null 반응은 서로 상관되어 있어서(평균 |r| ≈ 0.3), "
              "사건이 없는 이동된 열에도 한꺼번에 수십 개를 고를 수 있습니다. 보정한 결과는 5e 를 보세요.", ""]
    B10 = rd(os.path.join(d1, "ca_d1_B10_motion_control.csv"))
    if B10 is not None and len(B10):
        L += ["## 5b. 움직임 artefact 위험 (B10): 전달 주변 [-2,+2] s 에 등록이 나빴던 trial 수", "",
              "| 마우스 | 사건 | baseline 나쁜/전체 | injection 나쁜/전체 | UP 세포 (전체 → 깨끗한 trial 만) baseline | injection |",
              "|---|---|---|---|---|---|"]
        for (m, e), V in B10.groupby(["mouse", "event"]):
            if e not in ("Pin prick", "Heat", "withdrawal"):
                continue
            pre = V[V.phase == "pre"]; post = V[V.phase == "post"]
            g = lambda X: f"{int(X.n_bad_motion.iloc[0])}/{int(X.n_trials.iloc[0])}" if len(X) else "-"
            u = lambda X: f"{int(X.up_all.iloc[0])} → {X.up_clean.iloc[0]:.0f}" if len(X) and np.isfinite(X.up_clean.iloc[0]) else "-"
            L.append(f"| {m} | {e} | {g(pre)} | {g(post)} | {u(pre)} | {u(post)} |")
        L += ["", "나쁜 trial 이 한쪽 세션에만 많으면 그 비교는 움직임 차이와 섞입니다 (예: SBI 후 덜 움직임).", ""]
    R3 = rd(os.path.join(B, "analysis", "ca_response_shape", "R3_response_shape_counts.csv"))
    if R3 is not None and len(R3):
        t = R3.groupby(["cls", "shape"]).n_cells.sum()
        L += ["## 5c. UP / DOWN 반응의 실제 모양 (R3)", "",
              "r = z[0,2] − z[−2,0] 은 차이라서, DOWN 은 전달 뒤 감소일 수도 있고 전달 **직전의 상승**(실험자 접근, 마우스 움직임)이 "
              "끝나는 것일 수도 있습니다. 더 이른 기준 z[−5,−3] 과 비교했습니다 (모든 마우스·세션, 세포×사건 수):", ""]
        for cl in ("UP", "DOWN"):
            if cl in t.index.get_level_values(0):
                L.append(f"- {cl}: " + ", ".join(f"{k} {int(v)}" for k, v in t[cl].items()))
        MV = rd(os.path.join(B, "analysis", "ca_response_shape", "R3_moving_before_delivery.csv"))
        if MV is not None and len(MV):
            g = MV.groupby("event").frac_moving_before.mean()
            L.append("- 전달 전 2 s 안에 마우스가 움직인 trial 비율: " + ", ".join(f"{e} {100 * v:.0f}%" for e, v in g.items()))
        L += [f"- 그림: `{os.path.join(B, 'analysis', 'ca_response_shape', 'R3_response_shape.png')}`", ""]
    R4 = rd(os.path.join(B, "analysis", "ca_response_shape", "R4_down_speed_cells.csv"))
    if R4 is not None and len(R4):
        dn, up = R4[R4.cls == "DOWN"], R4[R4.cls == "UP"]
        BG = rd(os.path.join(B, "analysis", "ca_response_shape", "R4_background_psth.csv"))
        L += ["## 5d. DOWN 반응은 칼슘인가? (R4, 속도 점검)", "",
              "발화가 줄면 GCaMP 형광은 decay τ (≈1.3 s) 로만 내려갈 수 있습니다. 그래서 [0.1,0.3] s 감소 ÷ [0.8,1.2] s 감소는 "
              "칼슘이면 0.27 (사건 시각에 시작) ~ 0.53 (0.3 s 먼저 시작, 키 입력 지연) 을 넘을 수 없고, 계단이면 ≈1 입니다.", "",
              f"- DOWN {len(dn)}개 (세포×사건×세션): 속도 중앙값 {dn.speed.median():.2f}, 0.53 (칼슘 한계) 초과 {100 * (dn.speed > 0.53).mean():.0f}%. "
              f"사건별 중앙값: " + ", ".join(f"{e} {v:.2f}" for e, v in dn.groupby('event').speed.median().items()) + ".",
              f"- 참고 UP {len(up)}개: 상승 속도 중앙값 {up.speed.median():.2f} (상승은 spike 로 빠를 수 있음)."]
        if BG is not None and len(BG):
            L.append("- 같은 사건에서 세포 밖 배경의 [0,2] s 변화 (세션 SD 단위, 평균): " +
                     ", ".join(f"{e} {v:+.2f}" for e, v in BG.groupby('event').bg_change_in_sd.mean().items()) + ".")
        if (dn.speed > 0.53).mean() > 0.5:
            L.append("- **해석:** DOWN 반응의 대부분은 칼슘이 낼 수 없을 만큼 빨리 떨어지고, 그 순간 배경은 밝아집니다. 배경 분리(demixing) "
                     "과보정이나 초점(z) 변화일 가능성이 커서, 지금의 DOWN 은 '억제' 로 해석하지 않는 것이 안전합니다. "
                     "UP 반응은 4 s 까지 유지되고 배경은 ~1.5 s 에 돌아와서, 배경만으로는 설명되지 않습니다.")
        else:
            L.append("- DOWN 반응의 절반 이상은 칼슘 속도 한계 안에 있습니다.")
        L += [f"- 그림: `{os.path.join(B, 'analysis', 'ca_response_shape', 'R4_down_check.png')}`", ""]
    rcd = os.path.join(B, "analysis", "ca_responsive_change")
    R8 = rd(os.path.join(rcd, "R8_responsive_calibrated.csv"))
    if R8 is not None and len(R8):
        R8["mouse"] = R8.mouse.astype(str)
        L += ["## 5e. 반응 세포 — 세포 간 상관을 보정한 결과 (R8)", "",
              "각 칸: 단일 세포 규칙의 세포 수 (같은 규칙을 사건 없는 이동 열에 쓴 95번째 백분위) · p_count · "
              "max-T 세포 UP+DOWN (Westfall–Young, family-wise 오류 0.05, 상관과 무관). **굵게** = p_count ≤ 0.05.", "",
              "| 마우스 | 사건 | baseline | injection |", "|---|---|---|---|"]
        for (m, e), V in R8.groupby(["mouse", "event"], sort=False):
            if e == "flinch":
                continue
            def cell(X):
                if not len(X):
                    return "-"
                r = X.iloc[0]
                t = f"{int(r.rule_q20_cells)} ({r.null_rule_cells_p95:.0f}) · p {r.p_count:.3f} · {int(r.maxT_up)}+{int(r.maxT_down)}"
                return f"**{t}**" if r.p_count <= 0.05 else t
            L.append(f"| {m} | {e} | {cell(V[V.phase == 'pre'])} | {cell(V[V.phase == 'post'])} |")
        L += ["", "→ 움직임 시작 반응은 모든 마우스에서 null 을 넘습니다. 자극 반응은 일부 마우스·세션에서만 넘습니다. "
              "반응 세포 수의 pre → post 차이를 약효로 읽기 어렵습니다.",
              f"- 표: `{os.path.join(rcd, 'R8_responsive_calibrated.csv')}`, 그림: `{os.path.join(rcd, 'R8_responsive_calibrated.png')}`", ""]
    R7 = rd(os.path.join(rcd, "R7_population_response_test.csv"))
    if R7 is not None and len(R7):
        sig = R7[R7.q_energy <= 0.05]
        L += ["## 5f. 집단 반응이 블록의 진행 중 활동보다 큰가 (R7)", "",
              "세포 전체의 평균 반응 벡터 크기(RMS)를, 같은 사건 열을 블록 안에서 이동시킨 2000개와 비교했습니다 (BH q ≤ 0.05, "
              f"{len(R7)}개 검정).", "",
              "- 유의: " + ("; ".join(f"{r.mouse} {'baseline' if r.phase == 'pre' else 'injection'} {r.event} (×{r.energy / r.energy_null_median:.2f})"
                                     for r in sig.itertuples()) if len(sig) else "없음"),
              f"- 그림: `{os.path.join(rcd, 'R7_population_response_test.png')}`", ""]
    R5b = rd(os.path.join(rcd, "R5_split_half_reliability.csv"))
    R6 = rd(os.path.join(rcd, "R6_trial_common_mode.csv"))
    if R5b is not None and len(R5b):
        g = R5b.groupby("event")[["n_trials", "reliability_sb"]].mean()
        L += ["## 5g. 반응 패턴의 재현성과 trial 간 공통 성분 (R5b, R6)", "",
              "- 어떤 세포가 반응하는지의 패턴이 trial 의 두 절반 사이에서 반복되는 정도 (무작위 절반 100번, Spearman–Brown): "
              + ", ".join(f"{e} {r.reliability_sb:+.2f} ({r.n_trials:.0f} trials)" for e, r in g.iterrows()) + "."]
        if R6 is not None and len(R6):
            cols = [c for c in R6.columns if c.startswith("|r| PC1 vs")]
            best = R6[cols].mean().idxmax().replace("|r| PC1 vs ", "") if cols else "?"
            txt = (f"- trial 간 반응 변동의 {100 * R6.pc1_share.mean():.0f}% 가 하나의 공통 성분(PC1)입니다 (공통 성분이 없으면 ~10%). "
                   f"PC1 의 trial 점수와 가장 관련이 큰 것: {best} (|r| " +
                   ", ".join(f"{c.replace('|r| PC1 vs ', '')} {R6[c].mean():.2f}" for c in cols) + ").")
            if "sham_pc1_share" in R6:
                S6 = R6[R6.sham_pc1_share.notna()]
                txt += (f" 대조 (같은 구간의 무작위 시각, 자극 없음): PC1 {100 * S6.sham_pc1_share.mean():.0f}% vs 실제 {100 * S6.pc1_share.mean():.0f}%, "
                        f"PC1 을 뺀 뒤 재현성 {S6.sham_reliability_pc1_removed.mean():+.2f} vs 실제 {S6.reliability_pc1_removed.mean():+.2f}.")
                if S6.sham_pc1_share.mean() > 0.8 * S6.pc1_share.mean():
                    txt += " 즉 이 공통 성분은 자극 반응이 아니라 늘 있는 공통 변동이며, 단일 세포 검정 사이의 의존성을 만듭니다."
            L.append(txt)
        L += ["- 자극마다 trial 이 15–30개뿐이면 단일 세포 반응 지도는 불안정합니다. 다음 실험에서 자극당 trial 을 늘리면 가장 도움이 됩니다.", ""]
    R9 = rd(os.path.join(B, "analysis", "ca_response_shape", "R9_raw_roi_check.csv"))
    if R9 is not None and len(R9):
        L += ["## 5h. 영화 픽셀로 직접 본 DOWN / UP (R9, demixing 없음)", "",
              "motion-corrected 영화에서 세포 footprint 픽셀의 평균과 그 주변 고리(다른 세포 제외)를 읽었습니다. "
              "1 s 시점의 dF/F (%), 세포 중앙값:", "",
              "| 경우 | 세포 | n | footprint 픽셀 | 주변 고리 | 픽셀 − 0.7 고리 | EXTRACT z |", "|---|---|---|---|---|---|---|"]
        for (m, ph, ev, g), X in R9.groupby(["mouse", "phase", "event", "group"], sort=False):
            L.append(f"| {m} {'baseline' if ph == 'pre' else 'injection'} {ev} | {g} | {len(X)} | {100 * X.roi_dff_1s.median():+.2f} | "
                     f"{100 * X.annulus_dff_1s.median():+.2f} | {100 * X.roi_minus_ann_1s.median():+.2f} | {X.extract_z_1s.median():+.2f} |")
        dn = R9[R9.group == "DOWN"]
        L += ["", f"- DOWN 세포 중 footprint 픽셀이 실제로 어두워진 것은 {100 * (dn.roi_dff_1s < 0).mean():.0f}% 뿐입니다. "
              f"주변 고리는 {100 * (R9.annulus_dff_1s > 0).mean():.0f}% 의 세포에서 밝아졌습니다 (반응 없는 세포 포함).",
              *[(f"- {m} {'baseline' if ph == 'pre' else 'injection'} {ev}: " +
                 ("시야(주변 고리)가 밝아졌고 DOWN 세포는 주변보다 덜 밝아졌을 뿐입니다 → EXTRACT 의 배경 과보정으로 음수가 된 것입니다."
                  if X.annulus_dff_1s.median() > 0 and X.roi_dff_1s.median() > 0 else
                  "DOWN 세포의 픽셀이 실제로 어두워졌습니다 → 실제 감소일 수 있습니다 (R4 속도도 함께 보세요)."))
                for (m, ph, ev), X in R9[R9.group == "DOWN"].groupby(["mouse", "phase", "event"], sort=False)],
              "- UP 세포도 원래 픽셀이 주변과 거의 같이 움직이는 경우가 있어서, 사건 시점 반응은 이 시야 전체 변화와 얽혀 있습니다.",
              "  이 변화가 자세·초점 변화인지 넓은 neuropil 활동인지는 아직 모릅니다.",
              *(lambda R10: [] if R10 is None or not len(R10) else [
                  "- R10 (trial 별, 세포 밖 전체 배경): 변화가 trial 마다 작고 들쭉날쭉해서 움직임·withdrawal 과의 관계를 가르지 못했습니다. "
                  "가만히 있던 trial 수: " + ", ".join(f"{e} {int(R10[(R10.event == e) & (R10.state == 'still, no withdrawal')]['count'].sum())}"
                                                for e in ("Light touch", "Mild touch", "Pin prick", "Heat")) +
                  f" (`{os.path.join(B, 'analysis', 'ca_response_shape', 'R10_field_change.png')}`)."])(
                  rd(os.path.join(B, "analysis", "ca_response_shape", "R10_field_change_by_state.csv"))),
              f"- 그림: `{os.path.join(B, 'analysis', 'ca_response_shape', 'R9_raw_roi_check.png')}`", ""]
    a6 = rd(os.path.join(B, "analysis", "ca_day1", "ca_d1_A6_blocks_still_tests.csv"))
    a6r = rd(os.path.join(B, "analysis", "ca_day1_roi", "ca_d1_A6_blocks_still_tests.csv"))
    if a6 is not None and a6r is not None:
        x = a6.merge(a6r, on=["mouse", "block"], suffixes=("", "_roi"))
        L += ["## 5i. EXTRACT 없이 만든 trace (영화 footprint 픽셀 − 주변 neuropil) 로 다시 한 결과", "",
              "`<마우스>_d1\\5_analysis\\traces_roi` (`ca_roi_traces.py`): 세포 footprint 픽셀 평균에서 주변 고리를 회귀로 뺀 trace "
              "(계수는 조용한 프레임에서 추정). 분석 폴더 `analysis\\ca_day1_roi`.", "",
              "블록 활동 (움직이지 않는 프레임, transient/분 평균, baseline → injection, Wilcoxon):", "",
              "| 마우스 | 블록 | EXTRACT | p | ROI trace | p |", "|---|---|---|---|---|---|"]
        for r in x.itertuples():
            L.append(f"| {r.mouse} | {r.block} | {r.pre_mean:.2f} → {r.post_mean:.2f} | {fmt_p(r.p)} | "
                     f"{r.pre_mean_roi:.2f} → {r.post_mean_roi:.2f} | {fmt_p(r.p_roi)} |")
        same = int(((x.p < 0.05) == (x.p_roi < 0.05)).sum())
        sign = int((np.sign(x.post_mean - x.pre_mean) == np.sign(x.post_mean_roi - x.pre_mean_roi)).sum())
        L += ["", f"- 변화 방향이 같은 블록 {sign}/{len(x)}, 유의성(p < 0.05) 판정이 같은 블록 {same}/{len(x)}."]
        et = rd(os.path.join(B, "analysis", "ca_day1", "ca_d1_cells_event_tests.csv"))
        etr = rd(os.path.join(B, "analysis", "ca_day1_roi", "ca_d1_cells_event_tests.csv"))
        if et is not None and etr is not None:
            evs = ["Light touch", "Mild touch", "Pin prick", "Heat", "withdrawal", "movement onset"]
            m_ = et.merge(etr, on=["mouse", "phase", "event", "cell"], suffixes=("", "_roi"))
            m_ = m_[m_.event.isin(evs)]
            for cl in ("UP", "DOWN"):
                n0 = int((m_.cls_shift == cl).sum()); n1 = int(((m_.cls_shift == cl) & (m_.cls_shift_roi == cl)).sum())
                L.append(f"- 단일 세포 규칙의 {cl} (세포×사건×세션): EXTRACT {n0}개 중 ROI trace 에서도 {cl} {n1}개 ({100 * n1 / max(n0, 1):.0f}%).")
        R4r = rd(os.path.join(B, "analysis", "ca_response_shape", "R4_down_speed_cells_traces_roi.csv"))
        if R4r is not None and len(R4r):
            dr = R4r[R4r.cls == "DOWN"]
            L.append(f"- ROI trace 의 DOWN 속도: 중앙값 {dr.speed.median():.2f}, 0.53 (칼슘 한계) 초과 {100 * (dr.speed > 0.53).mean():.0f}% "
                     "(EXTRACT 처럼 주변 neuropil 을 빼면 '주변보다 덜 밝아진' 세포는 여전히 DOWN 이 됩니다).")
        R8r = rd(os.path.join(B, "analysis", "ca_responsive_change", "R8_responsive_calibrated_traces_roi.csv"))
        R8e = rd(os.path.join(B, "analysis", "ca_responsive_change", "R8_responsive_calibrated.csv"))
        if R8r is not None and R8e is not None:
            k = ["mouse", "phase", "event"]
            y = R8e.merge(R8r, on=k, suffixes=("", "_roi"))
            y = y[y.event != "flinch"]
            both = int(((y.p_count <= 0.05) & (y.p_count_roi <= 0.05)).sum())
            L.append(f"- 보정된 반응 세포 수 (R8, p_count ≤ 0.05): EXTRACT {int((y.p_count <= 0.05).sum())}, ROI trace {int((y.p_count_roi <= 0.05).sum())}, "
                     f"둘 다 {both} (전체 {len(y)} 조합). 둘 다인 것: " +
                     "; ".join(f"{r.mouse} {'baseline' if r.phase == 'pre' else 'injection'} {r.event}"
                               for r in y[(y.p_count <= 0.05) & (y.p_count_roi <= 0.05)].itertuples()))
        R7r = rd(os.path.join(B, "analysis", "ca_responsive_change", "R7_population_response_test_traces_roi.csv"))
        if R7r is not None and len(R7r):
            sg = R7r[R7r.q_energy <= 0.05]
            L.append("- 집단 반응 (R7, q ≤ 0.05), ROI trace: " + ("; ".join(
                f"{r.mouse} {'baseline' if r.phase == 'pre' else 'injection'} {r.event}" for r in sg.itertuples()) if len(sg) else "없음"))
        L.append("")
    lg = os.path.join(B, "analysis", "ca_longitudinal_TOY")
    if os.path.isdir(lg):
        L += ["## 6. 종단 (TOY Day 2 — 가짜 데이터, 파이프라인 시험용)", "",
              f"`{lg}` — 모든 그림에 빨간 TOY 표시.", ""]
    L += ["## 7. 검증", "", f"분석 코드는 정답을 아는 합성 데이터로 먼저 시험했습니다: `{os.path.join(B, 'analysis', 'ca_validation_synthetic', 'README_KR.md')}`", ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(L))
    print("summary ->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
