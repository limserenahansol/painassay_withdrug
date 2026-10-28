"""make_step_figs.py  -  where exactly does each pipeline stop on our video?

WHY THIS SHAPE
    Hansol's note: the earlier deck was too wordy, and what a PI wants is
    step 1, step 2, step 3 ... and the step at which ours breaks. So each
    figure is a numbered chain with one line per step: what the published
    method needs, what we measured, and a stamp.

    No paper figures are reproduced - those are copyrighted. What is shown
    is the published SPECIFICATION, quoted in a few words, against OUR OWN
    frame at the same step.

TWO CHAINS
    PainFace, with the result of the real run on our 11-min mini2p video
    (7 frames scored, one every 100 s).
    Any facial MGS method on SBI-553, with the 20296-frame sweep.

    The useful part is WHERE they stop. For SBI-553 it is not step 1 - the
    video is fine for the body. It is step 4, the eye.

Written in English and in Korean from the same numbers, so the two decks
cannot drift apart.

OUTPUT  ->  qc/steps/
    S1_painface_steps_<lang>.png
    S2_sbi_steps_<lang>.png

USAGE
    python make_step_figs.py

Hansol Lim
"""
from __future__ import annotations

import glob
import os
import textwrap

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
OUT = os.path.join(ROOT, "qc", "steps")
M2P_VID = (r"I:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
           r"15-56-48\MiceVideo1\MiceVideo")
SBI_VID = os.path.join(ROOT, "videos", "cameraB")
M2P_CROP = (740, 250, 1420, 1000)
SBI_CROP = (180, 150, 660, 470)
PASS, FAIL, PART = "#2E7D5B", "#C1272D", "#D98A00"
INK, DIM, PAPER = "#1A1A1A", "#6E6E6E", "#F2EFEA"
KFONT = "Malgun Gothic"


def frame(video_dir, pattern, t, crop, fps=None, enhance=False):
    v = sorted(glob.glob(os.path.join(video_dir, pattern)))[0]
    cap = cv2.VideoCapture(v)
    f = fps or cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * f))
    ok, im = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = crop
    g = g[y0:y1, x0:x1]
    if enhance:
        g = np.clip((g.astype(np.float32) - 18) * (255.0 / 90), 0,
                    255).astype(np.uint8)
        g = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(g)
    return g


def chain(steps, head, sub, foot, thumbs, path, korean=False):
    """One numbered row per step: need, measurement, stamp, our frame."""
    if korean:
        plt.rcParams["font.family"] = KFONT
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    n = len(steps)
    H = 2.55 * n + 1.9
    fig = plt.figure(figsize=(17.5, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    top = 100 - 100 * 1.55 / H
    ax.text(50, 100 - 100 * .55 / H, head, ha="center", va="center",
            fontsize=20, fontweight="bold")
    ax.text(50, 100 - 100 * 1.12 / H, sub, ha="center", va="center",
            fontsize=12.5, color=DIM)
    rh = 100 * 2.05 / H
    for i, (num, need, got, verdict, tkey) in enumerate(steps):
        y = top - (i + 1) * rh
        ax.add_patch(mp.Rectangle((1.5, y), 97, rh * .93,
                                  fc=PAPER if i % 2 == 0 else "white",
                                  ec="none", zorder=0))
        col = {"PASS": PASS, "FAIL": FAIL, "PARTIAL": PART,
               "부분": PART, "통과": PASS, "실패": FAIL,
               "오염": FAIL, "WORSE": FAIL, "더 나빠짐": FAIL,
               "UNCHANGED": FAIL, "그대로": FAIL,
               "NEW": PART, "새 발견": PART,
               "I WAS WRONG": PART, "제가 틀렸음": PART,
               "CONFIRMED": FAIL, "확인됨": FAIL,
               "MEASURED": PART, "측정됨": PART,
               "NOT SHOWN": DIM, "입증 안 됨": DIM,
               "NO WEIGHTS": FAIL, "가중치 없음": FAIL,
               "NOT RUNNABLE": FAIL, "실행 불가": FAIL,
               "NO INFORMATION": DIM, "정보 없음": DIM}.get(verdict, PART)
        ax.add_patch(mp.Circle((4.6, y + rh * .47), 2.2, fc=col, ec=col,
                               zorder=2, transform=ax.transData))
        ax.text(4.6, y + rh * .47, num, ha="center", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=3)
        ax.text(8.6, y + rh * .74, need, fontsize=12.5, va="center",
                fontweight="bold")
        # wrap: the measurement line is long and was running under and
        # past the thumbnail column
        wrapped = textwrap.fill(got, 70 if korean else 80)
        nl = wrapped.count(chr(10)) + 1
        ax.text(8.6, y + rh * (.30 if nl < 3 else .26), wrapped,
                fontsize=10.6 if nl > 2 else 11.2, va="center",
                color="#333333", linespacing=1.45)
        ax.add_patch(mp.FancyBboxPatch((56.5, y + rh * .28), 11.5,
                                       rh * .40,
                                       boxstyle="round,pad=0.01", fc=col,
                                       ec=col, zorder=2))
        ax.text(62.25, y + rh * .48, verdict, ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold", zorder=3)
        im = thumbs.get(tkey)
        if im is not None:
            # keep the real aspect: aspect="auto" stretched the frames wide
            # and a distorted thumbnail is the wrong kind of evidence
            row_h_in = rh * .86 / 100 * H
            w_in = row_h_in * im.shape[1] / im.shape[0]
            if w_in > 5.2:
                w_in = 5.2
                row_h_in = w_in * im.shape[0] / im.shape[1]
            a2 = fig.add_axes([.695, (y + rh * .07) / 100,
                               w_in / 17.5, row_h_in / H])
            a2.imshow(im, cmap="gray")
            a2.set_xticks([])
            a2.set_yticks([])
            for sp in a2.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(2.2)
    ax.text(50, 100 * .50 / H, foot, ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK, linespacing=1.6)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    th = {
        "m2p_face": frame(M2P_VID, "*.avi", 160, M2P_CROP, 25),
        "m2p_ear": frame(M2P_VID, "*.avi", 0, M2P_CROP, 25),
        "m2p_away": frame(M2P_VID, "*.avi", 100, M2P_CROP, 25),
        "sbi_raw": frame(SBI_VID, "male2side*.avi", 320, SBI_CROP),
        "sbi_enh": frame(SBI_VID, "male2side*.avi", 320, SBI_CROP,
                         enhance=True),
        "sbi_prof": frame(SBI_VID, "male2side*.avi", 840, SBI_CROP,
                          enhance=True),
    }

    # ---------------------------------------------------------- PainFace EN
    pf_en = [
        ("1", "Accept the video",
         "it RAN - greyscale, converted mp4, imaged through plastic. Their "
         "FAQ says no to monochrome; the platform processed it anyway.",
         "PASS", "m2p_face"),
        ("2", "Find the mouse and its face",
         "orbital unit located in 7 of 7 frames, confidence 0.60-1.00",
         "PASS", "m2p_face"),
        ("3", "Locate all four action-unit boxes",
         "all 4 found in only 2 of 7 frames. Nose, ears and whiskers each "
         "missing in 3 of 7 - confidence exactly 0.000.",
         "PARTIAL", "m2p_away"),
        ("4", "Score each unit 0 / 1 / 2",
         "orbital 0.29   nose 0.00   whiskers 0.00   EARS 1.50 of 2 - and "
         "the miniscope sits where the ears are",
         "FAIL", "m2p_ear"),
        ("5", "Add up to a Total Grimace Score",
         "returned in 2 of 7 frames. Both totals = 2, carried by the EAR "
         "unit. Drop ears and the score is 0 and 1 out of 6.",
         "FAIL", "m2p_ear"),
    ]
    chain(pf_en, "PainFace on our mini2p video - which step stops it?",
          "it is not step 1. The platform ran. It breaks at step 4, on the "
          "one action unit our hardware occupies.",
          "VERDICT   PainFace returns a number, and the number is the "
          "miniscope. Not usable on an implanted mouse.",
          th, os.path.join(OUT, "S1_painface_steps_EN.png"))

    # ---------------------------------------------------------- PainFace KR
    pf_kr = [
        ("1", "영상을 받아들이는가",
         "돌았습니다 - 흑백, 변환 mp4, 플라스틱 통과 촬영. FAQ는 흑백 불가라고 "
         "하지만 플랫폼은 처리했습니다.", "통과", "m2p_face"),
        ("2", "마우스와 얼굴을 찾는가",
         "orbital unit을 7개 프레임 전부에서 찾음, 신뢰도 0.60-1.00",
         "통과", "m2p_face"),
        ("3", "4개 action unit 박스를 다 찾는가",
         "4개 다 찾은 건 7개 중 2개뿐. 코·귀·수염이 각각 3개 프레임에서 "
         "누락 - 신뢰도 정확히 0.000", "부분", "m2p_away"),
        ("4", "각 unit에 0 / 1 / 2 점수",
         "orbital 0.29   코 0.00   수염 0.00   귀 1.50/2 - 그런데 귀 자리에 "
         "미니스코프가 있습니다", "실패", "m2p_ear"),
        ("5", "합계 Total Grimace Score",
         "7개 중 2개만 반환. 두 총점 다 2점이고 전부 귀 unit이 만든 것. 귀를 "
         "빼면 6점 만점에 0점과 1점.", "실패", "m2p_ear"),
    ]
    chain(pf_kr, "PainFace를 우리 mini2p 영상에 - 어느 단계에서 막히나?",
          "1단계가 아닙니다. 플랫폼은 돌았습니다. 4단계에서 깨집니다 - "
          "우리 장비가 차지한 그 action unit에서.",
          "결론   PainFace는 숫자를 주고, 그 숫자는 미니스코프입니다. "
          "implant 장착 마우스에는 쓸 수 없습니다.",
          th, os.path.join(OUT, "S1_painface_steps_KR.png"), korean=True)

    # ------------------------------------------------------- SBI-553 EN
    sbi_en = [
        ("1", "Is the animal exposed well enough?",
         "YES, once the tone curve uses the animal's own band (18-108 DN) "
         "instead of the whole frame. Fur texture and whiskers come back.",
         "PASS", "sbi_enh"),
        ("2", "Can the head be located?",
         "YES in 13452 of 20296 frames = 66.3 %. The backlit pinna is the "
         "brightest thing on the animal, 31 px = 9.1 mm.",
         "PASS", "sbi_enh"),
        ("3", "Is the face presented to the camera?",
         "YES in 78 of 156 adjudicated frames = 50 %. Pose is NOT the "
         "bottleneck.", "PASS", "sbi_prof"),
        ("4", "Is the EYE resolved?",
         "NO. 1 confirmed eye in 168 frames adjudicated pixel by pixel. "
         "Face contrast 3.22 DN against a 1.5-3.0 DN noise floor.",
         "FAIL", "sbi_raw"),
        ("5", "Nose and cheek?",
         "NO. The light is BEHIND the animal, so the camera-facing face is "
         "the unlit side. Shape change is 2-3 px at 3.41 px/mm.",
         "FAIL", "sbi_raw"),
    ]
    chain(sbi_en, "Any facial MGS method on SBI-553 - which step stops it?",
          "steps 1 to 3 PASS. Sharpening and contrast genuinely help, and "
          "recover the ear. It stops at the eye.",
          "VERDICT   1 of 5 action units. MGS needs the set. The blocker is "
          "the LIGHT BEHIND the animal, not the sensor.",
          th, os.path.join(OUT, "S2_sbi_steps_EN.png"))

    # ------------------------------------------------------- SBI-553 KR
    sbi_kr = [
        ("1", "동물이 충분히 노출됐나?",
         "됐습니다. 톤 커브를 전체 프레임이 아니라 동물 자체 대역(18-108 DN)"
         "으로 잡으면 털 결과 수염이 돌아옵니다.", "통과", "sbi_enh"),
        ("2", "머리를 찾을 수 있나?",
         "20296 프레임 중 13452 = 66.3 %. 역광 받은 귀가 동물에서 가장 밝고 "
         "31 px = 9.1 mm.", "통과", "sbi_enh"),
        ("3", "얼굴이 카메라를 향하나?",
         "판정한 156 프레임 중 78 = 50 %. 자세는 병목이 아닙니다.",
         "통과", "sbi_prof"),
        ("4", "눈이 해상되나?",
         "안 됩니다. 픽셀 단위로 판정한 168 프레임 중 확인된 눈 1개. 얼굴 "
         "대비 3.22 DN, 노이즈 바닥 1.5-3.0 DN.", "실패", "sbi_raw"),
        ("5", "코와 볼은?",
         "안 됩니다. 빛이 동물 뒤에 있어 카메라를 향한 얼굴이 그늘 쪽입니다. "
         "3.41 px/mm에서 형태 변화가 2-3 px.", "실패", "sbi_raw"),
    ]
    chain(sbi_kr, "얼굴 기반 MGS를 SBI-553에 - 어느 단계에서 막히나?",
          "1-3단계는 통과합니다. 샤프닝과 대비 보정은 실제로 도움이 되고 "
          "귀를 살려냅니다. 눈에서 멈춥니다.",
          "결론   5개 중 1개 unit. MGS는 세트가 필요합니다. 막는 것은 "
          "센서가 아니라 동물 뒤의 조명입니다.",
          th, os.path.join(OUT, "S2_sbi_steps_KR.png"), korean=True)


    # ------------------------------------------------------ PainFace re-run
    # Hansol re-ran PainFace at 300-frame (10 s) instead of 3000-frame
    # (100 s) sampling: 66 scored frames instead of 7. NOT higher
    # resolution - the same file, the same 680x750 crop, only denser
    # sampling. These rows are what that changed.
    rr_en = [
        ("1", "Yield of a usable Total Grimace Score",
         "run 1  2 of 7 = 29 %, CI 4-71 %   ->   run 2  8 of 66 = 12.1 %, "
         "CI 5.4-22.5 %. Run 1 was not BIASED - across all 10 grid phases "
         "the mean is 11.4 % - it was imprecise and lucky in phase.",
         "MEASURED", "m2p_face"),
        ("2", "Frames where NO action unit was found",
         "17 of 66 = 26 %. Adjudicated: 12 head not presented, 5 face dark "
         "or occluded, and 0 of 17 with the mouse outside the crop - so the "
         "framing is fine.", "NEW", "m2p_away"),
        ("3", "Is the Total carried by the EAR unit?",
         "yes - 11 of 18 total points = 61 %. Drop the ear and 4 of the 8 "
         "Totals go to exactly 0 (mean 2.25 -> 0.875). But frame-level this "
         "is description, not significance: sign test p = 0.109.",
         "UNCHANGED", "m2p_ear"),
        ("4", "Is the ear score the MINISCOPE? - I overstated this",
         "WITHDRAWN as stated. Ears = 2 at confidence 0.9934 and 0.9204 on "
         "frames where the head is completely OUT OF VIEW - no ear to "
         "score. Only 3 of 13 sit over identifiable hardware. The channel "
         "is a false-positive detector keying on BODY CONTOUR, and its "
         "confidence is ANTI-informative.", "I WAS WRONG", "m2p_ear"),
        ("5", "Does ORBITAL carry signal? - also not what I said",
         "it DOES vary: 48 detections, mean 0.46, four 2s at confidence "
         "0.98-1.00. But of the 18 rows scored orbital >= 1, an eye is "
         "visible in only 4 and ABSENT in 13. Whether any visible eye is "
         "narrowed is NOT DETERMINED at 13-27 px orbit size.",
         "I WAS WRONG", "m2p_face"),
        ("6", "Nose, whiskers, cheek, and stimulus blocks",
         "whiskers 0 in all 33 detections - zero variance. Cheek NA in all "
         "66 because the black-mouse model is a FOUR-AU model by design, "
         "not a failure. 0 of 13 frames show BOTH ears, so the bilateral "
         "ear item is never satisfiable. Nine block tests, smallest "
         "p = 0.095, pointing the wrong way.", "CONFIRMED", "m2p_away"),
    ]
    chain(rr_en, "PainFace re-run at 10 s sampling - what actually changed?",
          "denser sampling, not higher resolution: same file, 66 scored "
          "frames instead of 7. The headline held; two of my supporting "
          "claims did not.",
          "HOW TO REPORT IT   do NOT quote a Total Grimace Score from this "
          "session. Report orbital + nose + whisker on 0-6 with the EAR "
          "EXCLUDED A PRIORI - that gives 23 of 66 = 34.8 % yield, of which "
          "70 % read zero.",
          th, os.path.join(OUT, "S3_painface_rerun_EN.png"))

    rr_kr = [
        ("1", "쓸 수 있는 Total 반환률",
         "1차  7개 중 2개 = 29 %, CI 4-71 %   ->   2차  66개 중 8개 = "
         "12.1 %, CI 5.4-22.5 %. 1차가 편향된 게 아니었습니다 - 10가지 "
         "격자 위상 전체 평균이 11.4 % - 부정확하고 위상이 운 좋았을 "
         "뿐입니다.", "측정됨", "m2p_face"),
        ("2", "action unit을 하나도 못 찾은 프레임",
         "66개 중 17개 = 26 %. 판정 결과: 12개는 머리를 안 보여줌, 5개는 "
         "얼굴이 어둡거나 가려짐, 그리고 마우스가 crop 밖인 것은 17개 중 "
         "0개 - crop 설정은 문제없습니다.", "새 발견", "m2p_away"),
        ("3", "총점을 귀 unit이 만드는가?",
         "그렇습니다 - 총점 18점 중 11점 = 61 %. 귀를 빼면 8개 총점 중 "
         "4개가 정확히 0이 됩니다(평균 2.25 -> 0.875). 다만 프레임 단위로는 "
         "기술일 뿐 유의성은 아닙니다: 부호검정 p = 0.109.",
         "그대로", "m2p_ear"),
        ("4", "귀 점수가 미니스코프인가? - 제가 과장했습니다",
         "그 표현은 철회합니다. 머리가 완전히 보이지 않는 프레임에서 귀 = 2, "
         "신뢰도 0.9934와 0.9204 - 점수 낼 귀가 없습니다. 식별 가능한 장비 "
         "위에 있는 것은 13개 중 3개뿐. 이 채널은 몸통 윤곽에 반응하는 "
         "오검출기이고, 신뢰도 열은 역정보입니다.", "제가 틀렸음", "m2p_ear"),
        ("5", "ORBITAL에 신호가 있나? - 이것도 제 말과 다릅니다",
         "변하기는 합니다: 48회 검출, 평균 0.46, 신뢰도 0.98-1.00의 2점이 "
         "4개. 그런데 orbital >= 1인 18개 행 중 눈이 보이는 것은 4개뿐이고 "
         "13개는 눈이 없습니다. 보이는 눈이 실제로 가늘어졌는지는 안구 크기 "
         "13-27 px에서 어느 쪽으로도 판정 불가입니다.", "제가 틀렸음",
         "m2p_face"),
        ("6", "코·수염·볼, 그리고 자극 블록",
         "수염은 33회 검출 전부 0 - 분산 0. 볼은 66개 전부 NA인데, 이는 "
         "흑모 모델이 설계상 4-AU 모델이기 때문이고 실패가 아닙니다. 양쪽 "
         "귀가 다 보이는 프레임은 13개 중 0개라 MGS의 양측 귀 항목은 애초에 "
         "충족 불가. 블록 검정 9개, 최소 p = 0.095에 방향도 반대.",
         "확인됨", "m2p_away"),
    ]
    chain(rr_kr, "PainFace 10초 간격 재실행 - 실제로 무엇이 바뀌었나?",
          "해상도가 아니라 샘플링 밀도입니다: 같은 파일, 채점 프레임 7개 "
          "-> 66개. 본 결론은 유지됐고, 제 뒷받침 주장 둘은 틀렸습니다.",
          "보고 방법   이 세션에서 Total Grimace Score는 인용하지 마십시오. "
          "귀를 사전 배제하고 orbital + 코 + 수염을 0-6으로 보고하십시오 - "
          "그러면 66개 중 23개 = 34.8 % 수율이고, 그 중 70 %가 0점입니다.",
          th, os.path.join(OUT, "S3_painface_rerun_KR.png"), korean=True)


    # ---------------------------------------------------------- GrimACE
    # Positive control first: on GrimACE's OWN 1280x1024 example video the
    # installation gives quality up to 0.879, face confidence up to 0.959,
    # a crop in 7 of 10 frames and MGS averages spanning 0.00-1.00. So any
    # failure on our video is a domain failure, not an install failure.
    gr_en = [
        ("1", "Get the code",
         "GitHub ETHZ-INS/GrimACE_manuscript, AGPL-3.0: 5.1 MB, training "
         "code + recorder. models/ holds one .gitkeep. I stopped here and "
         "called it unrunnable. Wrong.", "PARTIAL", "m2p_face"),
        ("2", "Get the trained weights",
         "published on Zenodo 10.5281/zenodo.15119195 - 8.35 GB of zips, "
         "three weights CRC32-verified, loaded with no key mismatch. Plus "
         "2970 human-scored MGS frames from 3 raters.", "PASS", "m2p_face"),
        ("3", "Prove the install on THEIR video",
         "7 of 7 face detections, max confidence 0.9594, five DISTINCT MGS "
         "vectors. The install is correct, so anything below is domain and "
         "not setup.", "PASS", "m2p_ear"),
        ("4", "Frame-quality gate - THIS is the wall",
         "GrimACE hard-codes a minimum of 4.25/6 = 0.708. Our frames top "
         "out at 0.4908 = 2.94/6, short by 1.31 points. 0 of 326 frames "
         "pass, under every crop, every photometric transform, and explicit "
         "mean/std matching to their normaliser.", "FAIL", "m2p_away"),
        ("5", "Face gate - NOT a wall, and that is worse",
         "I said 0 of 448 and nothing rescues it. WITHDRAWN: the 0.9 gate "
         "IS crossable - 0.9141 at t = 440 s with histogram equalisation "
         "and a 300x300 window. But that window holds the cylinder rim and "
         "the cable and NO MOUSE. Preprocess harder and GrimACE scores "
         "furniture.", "I WAS WRONG", "m2p_away"),
        ("6", "What it emits when forced is its PRIOR",
         "the vector [eye 0, nose 0, cheek 0, ear 1, whisker 3] also comes "
         "out of pure black, pure white, mid-grey, a checkerboard and 30 "
         "noise crops. EAR = 1 in 37 of 37 degenerate inputs. So "
         "'GrimACE also says the ear is up' is the prior talking.",
         "NO INFORMATION", "m2p_ear"),
    ]
    chain(gr_en, "GrimACE on our mini2p video - it ran, and it cannot score it",
          "the weights are public; my 'not runnable' was wrong. Positive "
          "control passed first, so this is a domain failure. The deciding "
          "gate is frame QUALITY, not face detection.",
          "VERDICT   NOT RUNNABLE on this video - quality gate 0.708 "
          "required, 0.4908 achieved, 0 of 326 frames.  CHEAPEST NEXT TEST: "
          "inverting the image alone moves face detection 0.64 -> 0.88, so "
          "the back-lighting polarity is most of the gap. Resolution and "
          "fps are irrelevant - the video path checks neither.",
          th, os.path.join(OUT, "S4_grimace_steps_EN.png"))

    gr_kr = [
        ("1", "코드 받기",
         "GitHub ETHZ-INS/GrimACE_manuscript, AGPL-3.0: 5.1 MB, 학습 코드 + "
         "녹화 소프트웨어. models/에는 .gitkeep 하나. 저는 여기서 멈추고 "
         "실행 불가라고 했습니다. 틀렸습니다.", "부분", "m2p_face"),
        ("2", "학습된 가중치 받기",
         "Zenodo 10.5281/zenodo.15119195에 공개 - zip 8.35 GB, 가중치 3개를 "
         "CRC32 검증하고 키 불일치 없이 로드. 채점자 3명의 MGS 프레임 "
         "2,970장도 포함.", "통과", "m2p_face"),
        ("3", "그들 영상으로 설치 검증",
         "얼굴 검출 7/7, 최대 신뢰도 0.9594, 서로 다른 MGS 벡터 5종. 설치가 "
         "정확하므로 이하의 실패는 설치가 아니라 도메인 문제입니다.",
         "통과", "m2p_ear"),
        ("4", "프레임 품질 게이트 - 여기가 벽입니다",
         "GrimACE는 최소 4.25/6 = 0.708을 코드에 박아 두었습니다. 우리 "
         "프레임은 최대 0.4908 = 2.94/6, 1.31점 부족. 어떤 crop, 어떤 "
         "광도 변환, 그들 정규화에 평균·표준편차를 맞춰도 326개 중 0개 "
         "통과.", "실패", "m2p_away"),
        ("5", "얼굴 게이트 - 벽이 아니고, 그게 더 나쁩니다",
         "제가 448개 중 0개, 어떤 방법도 안 된다고 했습니다. 철회합니다: "
         "0.9 게이트는 넘을 수 있습니다 - t = 440초에서 히스토그램 "
         "평활화 + 300x300 창으로 0.9141. 그런데 그 창에는 실린더 테두리와 "
         "케이블만 있고 마우스가 없습니다. 전처리를 더 하면 GrimACE는 "
         "배경 가구에 점수를 냅니다.", "제가 틀렸음", "m2p_away"),
        ("6", "강제했을 때 나오는 건 모델의 사전확률입니다",
         "벡터 [눈 0, 코 0, 볼 0, 귀 1, 수염 3]은 순수 검정, 순수 흰색, "
         "중간 회색, 체커보드, 노이즈 crop 30개에서도 똑같이 나옵니다. "
         "퇴화 입력 37개 중 37개에서 귀 = 1. 즉 'GrimACE도 귀가 높다고 "
         "한다'는 사전확률이 말하는 것입니다.", "정보 없음", "m2p_ear"),
    ]
    chain(gr_kr, "GrimACE를 우리 mini2p 영상에 - 돌았지만 점수를 낼 수 없습니다",
          "가중치는 공개돼 있고 제 '실행 불가'는 틀렸습니다. 대조 실행을 "
          "먼저 통과했으므로 도메인 문제입니다. 결정적 게이트는 얼굴 "
          "검출이 아니라 프레임 품질입니다.",
          "결론   이 영상에서는 실행 불가 - 품질 게이트 0.708 요구, 0.4908 "
          "달성, 326개 중 0개.  가장 싼 다음 시험: 영상을 반전만 해도 얼굴 "
          "검출이 0.64 -> 0.88로 올라갑니다. 역광 극성이 격차의 대부분입니다. "
          "해상도와 fps는 무관합니다 - 비디오 경로는 둘 다 검사하지 않습니다.",
          th, os.path.join(OUT, "S4_grimace_steps_KR.png"), korean=True)

    for f in sorted(glob.glob(os.path.join(OUT, "*.png"))):
        print("wrote", f)


if __name__ == "__main__":
    main()
