"""make_pipeline_figs.py - how we did it, for the PI (2026-10-22): the pipeline at a glance, one 2-min session video
(behaviour + Ca), how the stimulus deliveries are detected (pressure logger), the behaviours scored by hand and the automatic
movement / rearing / grooming detection. Figures = big-font re-layouts of the QC images the pipeline already made (nothing
re-analysed); add_pipeline_slides() puts them into a deck (make_new_deck.Deck).

  python make_pipeline_figs.py            (writes analysis\\pipeline_figs\\pipe_*.png)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import winwords as WW  # noqa: E402  (10-23: the window the event analysis used, in words)
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

B = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(B, "analysis", "pipeline_figs")
SHEET = os.path.join(B, "analysis", "behavior_examples_v2", "behaviour_examples_sheet.png")    # 10-23: both days, pure taps
SHEET_ROW, SHEET_Y0, SHEET_Y1 = 330, 26, 202                 # sheet row height, camera band (the v1 sheet: 280, 24, 204)
LOGS = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_check", "5941_d2_pre_logger_check_sheet.png")
LOGQ = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_deliveries", "5941_d2_pre_logger_qc.png")
LOGN = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_check", "logger_check_summary.csv")
BFQ = os.path.join(B, "behavior_auto", "20261019", "5937_d1_pre_bfeatures_qc.png")
GRM = os.path.join(B, "analysis", "behavior_quality", "audit_5937_d2_pre_GROOM.png")
VID = os.path.join(B, "analysis", "ca_videos_pi")
DEFS = [("withdrawal", "paw lifted / pulled away right after the contact", "자극 직후 발을 들거나 뺌"),
        ("flinch", "fast shake or flick of the stimulated paw", "자극받은 발을 빠르게 흔들거나 튕김"),
        ("attending", "turns the head to look at / sniff the paw, no mouth contact", "머리를 돌려 발을 보거나 냄새 맡음 (입 닿지 않음)"),
        ("lick / bite", "mouth contact with the stimulated paw", "자극받은 발을 핥거나 묾"),
        ("escape / rearing", "stands up on the hind legs, tries to climb or jump out", "뒷발로 서거나 기어오르기·뛰어나가기 시도"),
        ("general movement", "AUTOMATIC: walks a few steps or turns the body", "자동: 몇 걸음 걷거나 몸을 돌림"),
        ("grooming", "AUTOMATIC: forepaws stroke the face or the head licks the flank", "자동: 앞발로 얼굴을 쓸거나 옆구리를 핥음")]


def crop(path, box):
    return np.asarray(Image.open(path).convert("RGB").crop(box))


def fig_flow(lg, out):
    L = lambda en, kr: kr if lg == "kr" else en
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])                 # 1 data unit = 1 inch (default margins shrank the boxes below their text)
    ax.set_axis_off(); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6)
    font = "Malgun Gothic" if lg == "kr" else None
    boxes = [(0.2, 3.3, L("RECORD", "기록"), L("mini1p Ca imaging of the BLA\n(GCaMP, 30 Hz)\n+ side and bottom cameras\n+ hand clicker at each delivery",
                                              "BLA mini1p Ca 이미징\n(GCaMP, 30 Hz)\n+ 옆·아래 카메라\n+ 자극마다 누른 손 클리커"), "#d6eaf8"),
             (3.4, 3.3, L("SYNC", "동기화"), L("one frame counter:\ncamera frame n =\nimaging frame n", "프레임 하나로 맞춤:\n카메라 프레임 n =\n이미징 프레임 n"), "#e8f6f3"),
             (6.6, 3.3, L("NEURONS", "뉴런"), L("motion correction\n-> EXTRACT footprints\n-> each neuron checked\nby eye on the max projection",
                                               "움직임 보정\n-> EXTRACT footprint\n-> max projection에서\n뉴런마다 직접 확인"), "#fdebd0"),
             (9.8, 3.3, L("SIGNALS", "신호"), L("dF/F per neuron\nz = dF/F / robust SD\nDay 1 <-> Day 2 tracking",
                                               "뉴런별 dF/F\nz = dF/F / robust SD\nDay 1 <-> Day 2 추적"), "#f9e79f"),
             (0.2, 0.3, L("EVENTS", "사건"), L("deliveries: clicker (Day 2)\nor hand marks (Day 1)\nreflexes, affective, escape/rearing: by hand\nmovement, grooming: automatic",
                                              "자극: 클리커 (Day 2) 또는\n직접 표시 (Day 1)\n반사·정서·escape/rearing: 직접 채점\n움직임, grooming: 자동"), "#fadbd8"),
             (4.6, 0.3, L("TEST EACH NEURON", "뉴런마다 검정"),
              (L("one window per event:\n0-0.5 or 0-1 s after\nvs the 0.5 / 1 s before\ntime-shift test, BH q <= 0.20",
                 "사건마다 창 하나:\n후 0-0.5 또는 0-1 s\nvs 직전 0.5 / 1 s\n시간 이동 검정, BH q <= 0.20") if WW.mixed() else
               L(f"one window per event type, from a\nwindow search (stimuli {WW.stim_span()} vs {WW.stim()[0]:g} s\nbefore; behaviours {WW.beh_span()} vs {WW.beh()[0]:g} s before)\ntime-shift test, BH q <= 0.20",
                 f"사건 종류마다 창 하나 (창 탐색으로 선택:\n자극 후 {WW.stim_span()} vs 직전 {WW.stim()[0]:g} s;\n행동 후 {WW.beh_span()} vs 직전 {WW.beh()[0]:g} s)\n시간 이동 검정, BH q <= 0.20")), "#e8daef"),
             (9.0, 0.3, L("COMPARE", "비교"), L("crossover: every mouse gets\nvehicle one day, SBI-553 the other\neach injection vs the baseline\nof the same day (n = 4 mice)",
                                               "crossover: 각 마우스가 하루 vehicle,\n다른 날 SBI-553\n각 주사 vs 같은 날 기준\n(n = 4마리)"), "#d5f5e3")]
    for x, y, h, t, c in boxes:
        w = 3.0 if y > 2 else (4.0 if x < 1 else 4.1 if x < 5 else 3.8)
        ax.add_patch(FancyBboxPatch((x, y), w, 2.0, boxstyle="round,pad=0.05", fc=c, ec="0.35", lw=1.2))
        ax.text(x + w / 2, y + 1.75, h, ha="center", va="center", fontsize=15, fontweight="bold", fontname=font)
        ax.text(x + w / 2, y + 0.85, t, ha="center", va="center", fontsize=11.5, linespacing=1.25, fontname=font)
    arr = [((3.2, 4.3), (3.4, 4.3)), ((6.4, 4.3), (6.6, 4.3)), ((9.6, 4.3), (9.8, 4.3)), ((10.6, 3.3), (8.0, 2.35)),
           ((4.2, 1.3), (4.6, 1.3)), ((8.7, 1.3), (9.0, 1.3))]
    for a, b in arr:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, lw=2, color="0.25"))
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


RAW = os.path.join(r"C:\Users\hsollim\Downloads\Rogger", "5941f_base_real", "5941f_base_real", "2026-10-21",
                   "5941f_base_real_2026-10-21_S01_pressure.csv")           # 5941 Day-2 baseline (logger_sessions.csv)
DELIV = os.path.join(B, "scoring_mini1p", "20261021_day2", "logger_deliveries", "5941_d2_pre_logger_deliveries.csv")
EX_S = 241.07                                                                # the delivery of the check sheet's BOTH row


def fig_logger(lg, out):
    """10-22 (user: label every graph): drawn from the raw logger file instead of the QC image - A one block with the delivery
    onsets used, B one delivery in close-up (threshold, onset), C the bottom camera around that onset"""
    L = lambda en, kr: kr if lg == "kr" else en
    font = "Malgun Gothic" if lg == "kr" else None
    D = pd.read_csv(RAW, usecols=["t_s", "corrected", "ttl_count"])
    x = D.corrected.to_numpy(float)
    fs = (len(x) - 1) / (D.t_s.iloc[-1] - D.t_s.iloc[0])
    n_b = int(round(10 * fs)) | 1
    base = pd.Series(x).rolling(n_b, center=True, min_periods=n_b // 10).median().to_numpy()
    y = (x - base) / 1e6                                                     # 10^6 counts above the 10-s running median
    ts = D.ttl_count.to_numpy() / 30.0                                       # session time = camera frame / 30 (the sync rule)
    import scipy.io as _sio
    import glob as _glob
    _p = [q for q in _glob.glob(os.path.join(B, "scoring_mini1p", "20261021_final_timefix", "ScoringAB_*.mat"))
          if str(_sio.loadmat(q, squeeze_me=True, variable_names=["mouseID"])["mouseID"]).strip() == "5941"
          and str(_sio.loadmat(q, squeeze_me=True, variable_names=["phase"])["phase"]).lower().startswith("base")][0]
    _M = _sio.loadmat(_p, squeeze_me=True, variable_names=["dSec", "dTypes"])
    _ds, _dt = np.atleast_1d(_M["dSec"]).astype(float), np.atleast_1d(_M["dTypes"]).astype(int)
    # 10-23: the light-touch deliveries of block 1 actually used (figure review: [180, 360) missed the one at 366.2 s)
    E1 = pd.DataFrame(dict(onset_session_s=np.sort(_ds[(_dt == 1) & (_ds >= 175) & (_ds < 390)])))
    fig = plt.figure(figsize=(13, 5.4))
    a = fig.add_axes([0.07, 0.58, 0.91, 0.33])
    m = (ts >= 175) & (ts <= 368)
    a.plot(ts[m], y[m], color="k", lw=0.6)
    for v in E1.onset_session_s:
        a.axvline(v, color="#e67e22", lw=1.0, alpha=0.85)
    a.axvline(EX_S, color="#c0392b", lw=2.4)
    a.set_xlim(175, 368); a.set_ylim(-3, 9)
    # 10-23 (pressure trace + the user's check): 209.9 s = an indicator onset with no press (removed); the press at 214.3 s
    # sat inside the held indicator and had no delivery (added; the user: 'this is the real stimulus')
    a.axvline(209.9, color="0.55", lw=1.4, ls="--")
    a.annotate(L("214.3 s: press inside a held click, added 10-23\n209.9 s (dashed): click without a press, removed",
                 "214.3 s: 눌린 채인 클릭 안의 누름, 10-23 추가\n209.9 s (점선): 누름 없는 클릭, 제거"), xy=(214.3, 1.0), xytext=(251, 3.2),
               fontsize=10, color="#7d3c98", fontname=font, zorder=5,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7d3c98", lw=0.8, alpha=0.95),
               arrowprops=dict(arrowstyle="-|>", color="#7d3c98", lw=1.4))
    a.set_xlabel(L("session time (s)", "세션 시간 (s)"), fontsize=12, fontname=font)
    a.set_ylabel(L("clicker pressure\n(10^6 counts)", "클리커 압력\n(10^6 counts)"), fontsize=12, fontname=font)
    a.tick_params(labelsize=11)
    a.set_title(L(f"A  one stimulus block (5941, Day 2 baseline, light touch): the {len(E1)} deliveries used (orange lines)",
                  f"A  한 자극 블록 (5941, Day 2 기준, light touch): 사용한 자극 {len(E1)}번 (주황 선)"),
                fontsize=13, loc="left", fontname=font)
    a.text(EX_S + 1.5, 8.2, L("shown in B and C", "B, C에 확대"), color="#c0392b", fontsize=11.5, fontweight="bold", fontname=font)
    b = fig.add_axes([0.07, 0.08, 0.27, 0.33])
    k = (ts >= EX_S - 1.0) & (ts <= EX_S + 1.5)
    b.plot(ts[k] - EX_S, y[k], color="k", lw=1.2)
    b.axhline(0.3, color="0.5", ls="--", lw=1)                 # logger_deliveries.py: --thr 3e5, --onset-level 1e5
    b.axhline(0.1, color="0.6", ls=":", lw=1)
    b.axvline(0, color="#c0392b", lw=2)
    b.set_xlim(-1.0, 1.5)
    b.set_xlabel(L("time from the onset (s)", "시작부터 시간 (s)"), fontsize=12, fontname=font)
    b.set_ylabel(L("clicker pressure\n(10^6 counts)", "클리커 압력\n(10^6 counts)"), fontsize=12, fontname=font)
    b.tick_params(labelsize=11)
    b.set_title(L("B  one click: onset = delivery time", "B  클릭 1번: 시작 = 자극 시점"), fontsize=13, loc="left", fontname=font)
    b.text(-0.97, 0.45, L("threshold 0.3 (--)\nonset 0.1 (...)", "역치 0.3 (--)\n시작 0.1 (...)"),
           ha="left", va="bottom", fontsize=10, color="0.35", fontname=font, linespacing=1.15)
    c = fig.add_axes([0.40, 0.02, 0.58, 0.42])
    c.imshow(crop(LOGS, (0, 184 * 2 + 24, 960, 184 * 3))); c.set_axis_off()
    for j, t_ in enumerate(("-0.3 s", "0 s", "+0.2 s", "+0.5 s")):
        c.text((j + 0.5) / 4, -0.03, t_, transform=c.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")
    c.set_title(L("C  bottom camera around that click (0 s = click onset)", "C  그 클릭 전후의 아래 카메라 (0 s = 클릭 시작)"),
                fontsize=13, loc="left", fontname=font)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    S = pd.read_csv(LOGN)
    return S


def fig_behaviours(lg, idx, out):
    L = lambda en, kr: kr if lg == "kr" else en
    font = "Malgun Gothic" if lg == "kr" else None
    n = len(idx)
    cols = 2
    rows_ = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_, cols, figsize=(13, 3.3 * rows_ + 0.2), squeeze=False)
    for k in range(rows_ * cols):
        a = ax.ravel()[k]; a.set_axis_off()
        if k >= n:
            continue
        i = idx[k]
        y0 = SHEET_ROW * i
        im = crop(SHEET, (0, y0 + SHEET_Y0, 540, y0 + SHEET_Y1))           # example 1: side view | bottom view
        a.imshow(im)
        nm, en, kr = DEFS[i]
        a.set_title(nm + "\n" + L(en, kr), fontsize=14, fontname=font, loc="left", linespacing=1.3)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


GRM_ROWS = [(68, 239), (266, 438)]                                 # tiles of the GROOM audit sheet (px; 2 of its 3 bouts)
GRM_COLS = [(138, 310), (320, 492), (503, 675), (685, 857), (868, 1040)]
GRM_T = [[285.3, 287.1, 289.0, 290.8], [663.0, 665.2, 667.4, 669.6]]   # the frame times printed (tiny) on those tiles


def fig_auto(lg, out):
    """10-24 (figure review): every label redrawn at a readable size (the tiny stamps of the QC images are cropped off) and a
    key for every mark: red outline / green search box / blue height line (behavior_features.py), cyan path / orange dot =
    the body centre during the bout and in that frame (behavior_movement_audit.py)"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    L = lambda en, kr: kr if lg == "kr" else en
    font = "Malgun Gothic" if lg == "kr" else None
    fig = plt.figure(figsize=(13, 5.6))
    for i, (k, ts) in enumerate(((2, 399), (3, 599))):            # 10-23: row 1 (200 s) = outline merged with the dark corner
        a = fig.add_axes([0.0, 0.53 - i * 0.42, 0.42, 0.37])
        a.imshow(crop(BFQ, (0, 240 * k + 16, 720, 240 * (k + 1))))  # top 16 px = the tiny yellow frame stamp, cut off
        a.set_axis_off()
        a.text(0.01, 0.97, L(f"{ts} s", f"{ts} s"), transform=a.transAxes, ha="left", va="top", fontsize=12, fontweight="bold",
               color="k", bbox=dict(boxstyle="round,pad=0.2", fc="#fff59d", ec="none"))
        if i == 0:
            a.set_title(L("A  movement: the mouse found in every frame (5937, Day 1 baseline)",
                          "A  움직임: 모든 프레임에서 쥐를 찾음 (5937, Day 1 기준)"), fontsize=12.5, fontname=font, loc="left")
            a.text(0.25, -0.02, L("bottom camera", "아래 카메라"), transform=a.transAxes, ha="center", va="top", fontsize=11,
                   fontname=font)
            a.text(0.75, -0.02, L("side camera", "옆 카메라"), transform=a.transAxes, ha="center", va="top", fontsize=11,
                   fontname=font)
    fig.legend(handles=[Patch(facecolor="#a01010", label=L("mouse outline", "쥐 윤곽")),
                        Patch(facecolor="none", edgecolor="#00c800", lw=1.5, label=L("area searched", "찾는 영역")),
                        Line2D([], [], color="#0000ff", lw=2.5, label=L("top of the mouse (height)", "쥐의 위쪽 끝 (높이)"))],
               loc="lower left", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.0, -0.01), handlelength=1.5,
               columnspacing=1.0, prop=dict(family=font, size=11) if font else None)
    G = np.asarray(Image.open(GRM).convert("RGB"))
    x0, w_, h_ = 0.47, 0.098, 0.3
    for r, (ya, yb) in enumerate(GRM_ROWS):
        for c, (xa, xb) in enumerate(GRM_COLS):
            a = fig.add_axes([x0 + c * (w_ + 0.008), 0.5 - r * 0.4, w_, h_])
            a.imshow(G[ya:yb, xa:xb] if c < 4 else G[ya + 28:ya + 143, xa:xb]); a.set_axis_off()   # side: its tiny label cut off
            a.set_title(f"{GRM_T[r][c]:.1f} s" if c < 4 else L("side camera", "옆 카메라"), fontsize=11, fontname=font)
        fig.text(x0 - 0.006, 0.5 - r * 0.4 + h_ / 2, L(f"bout {r + 1}", f"bout {r + 1}"), rotation=90, ha="right", va="center",
                 fontsize=11.5, fontname=font)
    fig.text(x0, 0.912, L("B  grooming (automatic): paw / head strokes while the body stays (2 bouts)",
                         "B  grooming (자동): 몸은 가만히, 앞발·머리만 움직임 (bout 2개)"), fontsize=12.5, fontname=font, ha="left",
             va="bottom")
    fig.legend(handles=[Line2D([], [], color="#00e5ff", lw=2, label=L("path of the body centre in the bout", "bout 동안 몸 중심의 경로")),
                        Line2D([], [], ls="none", marker="o", color="#ff3d00", ms=7, label=L("body centre in this frame", "이 프레임의 몸 중심"))],
               loc="lower left", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(x0 - 0.01, -0.01), handlelength=1.5,
               columnspacing=1.2, prop=dict(family=font, size=11) if font else None)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def make(lg):
    os.makedirs(OUT, exist_ok=True)
    sfx = "_kr" if lg == "kr" else ""
    fig_flow(lg, os.path.join(OUT, f"pipe_01_flow{sfx}.png"))
    S = fig_logger(lg, os.path.join(OUT, f"pipe_02_logger{sfx}.png"))
    fig_behaviours(lg, [0, 1, 2, 3], os.path.join(OUT, f"pipe_03_behaviours_a{sfx}.png"))
    fig_behaviours(lg, [4, 5, 6], os.path.join(OUT, f"pipe_03_behaviours_b{sfx}.png"))
    fig_auto(lg, os.path.join(OUT, f"pipe_04_auto{sfx}.png"))
    return S


def _n_del2():
    import pandas as _pd
    S_ = _pd.read_csv(r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\behavior_crossover\crossover_per_session.csv")
    return int(S_[S_.day == 2].deliveries.sum())


N_DEL2 = _n_del2()


OLDMARKS = ("mix4" if os.environ.get("SCORING_SUFFIX", "_timefix") == "_mix4"            # 10-25: marks chosen per event
            else "" if os.environ.get("SCORING_SUFFIX", "_timefix") else "old")         # 10-24: the old-marks deck variant


def add_pipeline_slides(dk, lg, video=None, poster=None, video2=None, poster2=None):
    """the HOW WE DID IT section (6-8 slides) into a make_new_deck.Deck; video2 = every scored behaviour (behaviour_examples);
    returns the keys of the slides made, in order"""
    keys = ["PIPE:glance"]
    from pptx.util import Inches
    L = lambda en, kr: kr if lg == "kr" else en
    sfx = "_kr" if lg == "kr" else ""
    f = lambda nm: os.path.join(OUT, f"{nm}{sfx}.png")
    S = pd.read_csv(LOGN)
    tot = S[["mine", "logger_indicator", "both"]].sum()
    sec = L("HOW WE DID IT", "방법")
    dk.slide(sec, L("The pipeline at a glance", "파이프라인 한눈에"),
             L("Ca imaging, two cameras and the stimulus clicker on one frame clock; one response window per event type",
               "Ca 이미징, 카메라 2대, 자극 클리커를 한 프레임 시계로; 사건 종류마다 반응 창 하나"),
             f("pipe_01_flow"),
             L("Each mouse: a baseline session, then an injection session (vehicle one day, SBI-553 the other). n = 4 mice, 16 sessions; descriptive (no test).",
               "마우스마다 기준 세션 후 주사 세션 (하루 vehicle, 다른 날 SBI-553). n = 4마리, 세션 16개; 기술 통계 (검정 없음)."),
             L("Frames are joined by number (camera frame n = imaging frame n, checked with the camera pulses the clicker box records). "
               "Neurons: EXTRACT candidates, each confirmed by eye on the max dF/F projection; Day-1 neurons found again on Day 2 "
               "(tracking checked by eye). z = dF/F / robust SD (MAD) per session. "
               + ("Withdrawal / flinch key marks are moved 0.40 s (Day 1) / 0.37 s (Day 2) earlier, to the paw-movement start measured "
                  "on video (10-23)." if OLDMARKS == "" else "Old marks: withdrawal / flinch as keyed, before the 10-23 timing fix."
                  if OLDMARKS == "old" else "Marks chosen per event: heat deliveries and withdrawal marks as keyed (original); flinch "
                  "marks moved 0.40 s (Day 1) / 0.37 s (Day 2) earlier and the other deliveries timing-corrected (10-23).")
               + (" " + WW.prefix_en()[0].upper() + WW.prefix_en()[1:] + ": " + WW.en() + "; no correction for the choice." if WW.mixed() else ""),
               "프레임은 번호로 맞춤 (카메라 n = 이미징 n, 클리커 상자가 기록한 카메라 펄스로 확인). 뉴런: EXTRACT 후보를 max dF/F에서 눈으로 "
               "확인; Day 1 뉴런을 Day 2에서 다시 찾음 (확인한 추적). z = 세션별 dF/F / robust SD (MAD). "
               + ("withdrawal / flinch 키 표시는 영상으로 잰 발 움직임 시작에 맞춰 0.40 s (Day 1) / 0.37 s (Day 2) 앞당김 (10-23)."
                  if OLDMARKS == "" else "옛 표시: withdrawal / flinch 키를 누른 그대로 (10-23 타이밍 보정 전)." if OLDMARKS == "old"
                  else "사건별 표시: heat 자극과 withdrawal 표시는 누른 그대로 (옛 표시); flinch 표시는 0.40 s (Day 1) / 0.37 s (Day 2) "
                  "앞당김, 나머지 자극은 보정 (10-23).")
               + (" " + WW.prefix_kr() + ": " + WW.kr() + "; 선택에 대한 보정 없음." if WW.mixed() else "")))
    if video and os.path.isfile(video):
        keys.append("PIPE:video")
        s = dk.head(sec, L("One session, behaviour and Ca together (2 min)", "한 세션, 행동과 Ca를 함께 (2분)"),
                    L("One pin-prick block of 5941: paw contacts (bottom camera), the Ca movie and 6 pin-prick UP neurons "
                      f"(shade = {WW.stim_span()} after each delivery = the response window)",
                      f"5941의 pin prick 블록: 발 접촉 (아래 카메라), Ca 영상, pin prick UP 뉴런 6개 (음영 = 자극 후 {WW.stim_span()} = 반응 창)"))
        s.shapes.add_movie(video, Inches(2.07), Inches(1.45), Inches(9.16), Inches(5.15), poster_frame_image=poster, mime_type="video/mp4")
        dk.box(s, 0.45, 6.68, 12.4, 0.8, L("5941, Day 1 baseline. Single trials are noisy: the averaged responses of all neurons are in Q3. "
                                          "Ca colour bar = dF/F (%), haze removed. n = 1 example session; descriptive (no test).",
                                          "5941, Day 1 기준. 한 번의 trial은 잡음이 큼: 모든 뉴런의 평균 반응은 Q3에. Ca 색 막대 = dF/F (%), "
                                          "배경 제거. n = 예시 세션 1개; 기술 통계 (검정 없음)."), 13.5)
        s.notes_slide.notes_text_frame.text = L("Neurons shown: those with a pin-prick UP response in the best window (event analysis), "
                                                "then the most active ones. Traces = local dF/F, each scaled to its own maximum (no common "
                                                "scale); black = mean of all 67 neurons. Rows under the traces: deliveries, withdrawal / flinch "
                                                "taps, movement and grooming (automatic). Click the video to play (2 min, re-encoded for the deck).",
                                                "보이는 뉴런: 최적 창에서 pin prick UP 반응 뉴런, 그다음 가장 활발한 뉴런. trace = 국소 dF/F, "
                                                "각자 최대값으로 맞춤 (공통 축 없음); 검정 = 뉴런 67개 평균. 아래 행: 자극, withdrawal / flinch, "
                                                "움직임·grooming (자동). 클릭하면 재생 (2분).")
    # 10-22 fix: 'mine' of logger_check_summary.csv is the OFFLINE press detection (logger_delivery_check.py), not your marks
    keys.append("PIPE:logger")
    dk.slide(sec, L("How the Day-2 stimulus times are found: a hand clicker", "Day 2 자극 시점 찾기: 손 클리커"),
             L(f"Day 2: the clicker's indicator and an offline press detection agree on {int(tot.both)} of {int(tot.logger_indicator)} "
               f"indicator onsets ({100 * tot.both / tot.logger_indicator:.0f}%); {N_DEL2} Day-2 deliveries used",
               f"Day 2: 클리커 표시와 오프라인 누름 검출이 {int(tot.logger_indicator)}번 중 {int(tot.both)}번 일치 "
               f"({100 * tot.both / tot.logger_indicator:.0f}%); Day 2 자극 {N_DEL2}개 사용"),
             f("pipe_02_logger"),
             L("A hand clicker pressed at each delivery (it gives the time only): each click is a pressure spike, its onset = the "
               "delivery time (C). Used: indicator onsets + presses only the offline detection sees (10-21), with 1 removed and "
               "3 added on 10-23. n = 8 Day-2 sessions (4 mice); descriptive (no test).",
               "자극마다 손으로 누른 클리커 (시점 정보만): 클릭마다 압력 신호, 그 시작 = 자극 시점 (C). 사용: 표시 시작 + 오프라인 검출만 "
               "보는 누름 (10-21), 10-23에 1개 제거·3개 추가. n = Day 2 세션 8개 (4마리); 기술 통계 (검정 없음)."),
             L("Per session (logger indicator / offline detection / both, within 0.5 s): " + "; ".join(
                 f"{r.recording_id} {r.logger_indicator}/{r.mine}/{r.both}" for r in S.itertuples()) +
               ". The clicker is pressed by hand at each delivery, so its indicator onsets are the delivery times (light clicks "
               "that only the indicator saw are kept). 10-23, the 12 clicks held > 1 s checked on the pressure trace: 1 indicator onset "
               "without a press removed (5941 Day 2 baseline 209.9 s), 3 presses inside a held click added (5941 Day 2 baseline 214.3 s, "
               "1191 Day 2 baseline 334.1 s, 1191 Day 2 injection 652.3 s), 8 kept. A second press < 0.5 s after the first is "
               "dropped (19; the first is used). Day 1 used the hand marks only (no clicker).",
               "세션별 (로거 표시 / 오프라인 검출 / 둘 다, 0.5 s 이내): " + "; ".join(
                   f"{r.recording_id} {r.logger_indicator}/{r.mine}/{r.both}" for r in S.itertuples()) +
               ". 클리커는 자극마다 손으로 눌렀으므로 표시 시작 = 자극 시점 (표시만 본 가벼운 클릭도 사용). 10-23, 1초 넘게 눌린 클릭 12번을 "
               "압력 신호로 확인: 누름 없는 표시 1개 제거 (5941 Day 2 기준 209.9 s), 눌린 클릭 안의 누름 3개 추가 (5941 Day 2 기준 214.3 s, "
               "1191 Day 2 기준 334.1 s, 1191 Day 2 주사 652.3 s), 8개 유지. 첫 누름 0.5 s 안의 두 번째 누름은 뺌 (19개; 첫 번째 사용). "
               "Day 1은 직접 표시만 사용 (클리커 없음)."))
    keys.append("PIPE:beh1")
    dk.slide(sec, L("The behaviours we score (1/2)", "채점하는 행동 (1/2)"),
             L("Reflexes (withdrawal, flinch) and affective behaviours (attending, lick / bite), scored by hand frame by frame",
               "반사 (withdrawal, flinch)와 정서 행동 (attending, lick / bite)을 프레임 단위로 직접 채점"),
             f("pipe_03_behaviours_a"),
             L("One example of each (left = side camera, right = bottom camera at the scored frame); scored in all 16 sessions, n = 4 mice; descriptive (no test).",
               "행동마다 예시 하나 (왼쪽 = 옆 카메라, 오른쪽 = 아래 카메라, 채점한 프레임); 세션 16개 모두 채점, n = 4마리; 기술 통계 (검정 없음)."),
             L("Withdrawal / flinch: from the delivery to 2 s later or the next delivery. Held behaviours: a bout starting 0-3 s after "
               "a delivery counts for that delivery. 10-23: the withdrawal / flinch key marks came 0.40 s (Day 1) / 0.37 s (Day 2) after "
               "the paw started to move (video, 4x slower, 2 independent judges + an automatic motion check on all 1,691 marks, 1,467 with a clear burst), so "
               "every mark was moved earlier by that much; a delivery that its own reflex would then precede was moved to 0.07 s "
               "before it (Day 1 383 of 698, Day 2 109 of 597 deliveries).",
               "withdrawal / flinch: 자극부터 2 s 또는 다음 자극까지. 지속 행동: 자극 후 0-3 s 안에 시작한 bout. 10-23: withdrawal / flinch 키 "
               "표시가 발 움직임 시작보다 0.40 s (Day 1) / 0.37 s (Day 2) 늦었음 (4배 느린 영상, 독립 판정 2명 + 표시 1,691개 전체(뚜렷한 1,467개) 자동 움직임 "
               "확인) - 모든 표시를 그만큼 앞당김; 그 반사가 자기 자극보다 앞서게 된 자극은 반사 0.07 s 전으로 옮김 (Day 1 698개 중 383개, "
               "Day 2 597개 중 109개)."))
    keys.append("PIPE:beh2")
    dk.slide(sec, L("The behaviours we score (2/2)", "채점하는 행동 (2/2)"),
             L("Escape / rearing scored by hand; general movement and grooming detected automatically (examples checked by eye)",
               "escape / rearing은 직접 채점; 일반 움직임과 grooming은 자동 검출 (예시는 직접 확인)"),
             f("pipe_03_behaviours_b"),
             L("One example of each (left = side camera, right = bottom camera); guarding is in the rule but occurred 0 times. Scored in "
               "all 16 sessions, n = 4 mice; descriptive (no test).",
               "행동마다 예시 하나 (왼쪽 = 옆, 오른쪽 = 아래 카메라); guarding은 규칙에 있지만 0번 나옴. 세션 16개 모두, n = 4마리; "
               "기술 통계 (검정 없음)."),
             L("Movement = centroid or body-axis change of the mouse outline; grooming = forepaw / head motion with the body still.",
               "움직임 = 쥐 윤곽의 중심·몸 축 변화; grooming = 몸은 가만히, 앞발·머리 움직임."))
    if video2 and os.path.isfile(video2):
        keys.append("PIPE:bvideo")
        s = dk.head(sec, L("Every scored behaviour on video (2 min)", "채점한 모든 행동을 영상으로 (2분)"),
                    L("Three examples of each scored behaviour (side and bottom cameras), reflexes replayed 4x slower",
                      "채점한 행동마다 예시 3개 (옆·아래 카메라), 반사는 4배 느리게 다시 재생"))
        import cv2 as _cv2                                     # the box follows the clip's aspect (1080 x 640 since 10-23)
        _c = _cv2.VideoCapture(video2); _ar = (_c.get(3) / _c.get(4)) if _c.get(4) else 1080 / 560; _c.release()
        _h = 5.15; _w = min(12.4, _h * _ar)
        s.shapes.add_movie(video2, Inches((13.333 - _w) / 2), Inches(1.45), Inches(_w), Inches(_h), poster_frame_image=poster2,
                           mime_type="video/mp4")
        dk.box(s, 0.45, 6.68, 12.4, 0.8, L("Examples from both days (Day 1, Day 2, Day 1), corrected marks (10-23); withdrawal / flinch from deliveries with "
                                          "only that reflex. Black line = delivery, dot = tap, red = the frame shown. n = 4 mice; descriptive (no test).",
                                          "두 날의 예시 (Day 1, Day 2, Day 1), 보정한 표시 (10-23); withdrawal·flinch는 그 반사만 있는 자극에서. "
                                          "검은 선 = 자극, 점 = 탭, 빨강 = 보이는 프레임. n = 4마리; 기술 통계 (검정 없음)."), 13.5)
        s.notes_slide.notes_text_frame.text = L("analysis\\behavior_examples_v2\\behaviour_examples_all.mp4 (10-23, both days), re-encoded "
                                                "for the deck; drawn from the timing-corrected scoring (withdrawal / flinch taps 0.40 s (Day 1) / "
                                                "0.37 s (Day 2) earlier; deliveries that their reflex came before sit 0.07 s before it (Day 1 383, Day 2 109); Day 2 = "
                                                "the clicker time). The single-behaviour clips are in the same folder; guarding has no clip (0 bouts). "
                                                "The 10-20 Day-1-only version is kept in analysis\\behavior_examples. Click the video to play.",
                                                "analysis\\behavior_examples_v2\\behaviour_examples_all.mp4 (10-23, 두 날), 덱용 재인코딩; 보정한 채점 사용 (withdrawal / flinch 탭 "
                                                "Day 1 0.40 s / Day 2 0.37 s 앞당김; 반사가 먼저였던 자극은 반사 0.07 s 전 (Day 1 383개, Day 2 109개); Day 2 = 클리커 시점). 행동별 "
                                                "클립은 같은 폴더; guarding은 0번이라 클립 없음. 10-20 Day 1 버전은 analysis\\behavior_examples에 "
                                                "그대로 둠. 클릭하면 재생.")
    keys.append("PIPE:auto")
    dk.slide(sec, L("Automatic detection: movement and grooming", "자동 검출: 움직임과 grooming"),
             L("The mouse outline and its height are tracked in every frame; grooming = paw / face motion without walking",
               "모든 프레임에서 쥐 윤곽과 높이를 추적; grooming = 걷지 않고 앞발·얼굴만 움직임"),
             f("pipe_04_auto"),
             L("Left: bottom-camera outline (red) and side-camera height (blue); right: detected grooming bouts. Checked by eye on example "
               "clips, not yet frame by frame against hand scoring. n = 16 sessions, 4 mice; descriptive (no test).",
               "왼쪽: 아래 카메라 윤곽 (빨강)과 옆 카메라 높이 (파랑); 오른쪽: 검출한 grooming bout. 예시 클립을 눈으로 확인, 아직 "
               "직접 채점과 프레임 단위 비교는 안 함. n = 세션 16개, 4마리; 기술 통계 (검정 없음)."),
             L("Movement and grooming are automatic; their example clips were checked by eye before use (behaviour_examples). "
               "Overlap (16 sessions, 10-23): 39% of the grooming frames are also 'movement', but grooming is only 3.1% of the "
               "movement frames, so it hardly changes the movement result. Examples from 5937 (399 s, 599 s). Known failure: at "
               "200 s of the same session the outline merged with the dark corner of the arena (shown in "
               "behavior_auto\\20261019\\5937_d1_pre_bfeatures_qc.png) - one reason a frame-by-frame check is still needed.",
               "움직임과 grooming은 자동; 예시 클립은 사용 전에 눈으로 확인 (behaviour_examples). 겹침 (세션 16개, 10-23): grooming "
               "프레임의 39%가 'movement'로도 잡히지만, grooming은 movement 프레임의 3.1%뿐이라 움직임 결과는 거의 안 바뀜. 예시 = 5937 "
               "(399 s, 599 s). 알려진 실패: 같은 세션 200 s에서 윤곽이 바닥의 어두운 구석과 합쳐짐 - 프레임 단위 확인이 아직 필요한 이유."))
    return keys


if __name__ == "__main__":
    for lg in ("en", "kr"):
        make(lg)
    print("->", OUT)
