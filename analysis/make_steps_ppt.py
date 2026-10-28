"""make_steps_ppt.py  -  the short version, for a PI. English and Korean.

WHAT CHANGED FROM THE FIRST DECK
    Hansol: "too wordy - just explain simply why it failed, at which step,
    like grimace has step 1, 2, 3, 4". So this deck is one slide per
    question, a picture on every slide, and almost no prose. The long
    version with all the caveats stays as Grimace_feasibility_20260930.pptx.

    No paper figures are reproduced - those are copyrighted. What is
    compared is the published SPECIFICATION, quoted in a few words, against
    our own frame at the same step.

THE TWO ANSWERS
    PainFace   ran on our mini2p video and broke at STEP 4: the Total
               Grimace Score is carried by the EAR unit, and a head-mounted
               miniscope sits where the ears are.
    SBI-553    passes steps 1 to 3. Contrast and sharpening genuinely help
               and recover the ear. It breaks at STEP 4: the eye.

OUTPUT  ->  <project root>\\
    Grimace_steps_EN_<date>.pptx
    Grimace_steps_KR_<date>.pptx

USAGE
    python make_steps_ppt.py

Hansol Lim
"""
from __future__ import annotations

import os
import sys

from pptx.util import Inches

sys.path.insert(0, r"C:\Users\hsollim\OneDrive\Desktop\cursor"
                   r"\pain_2plane_split")
import deck as D  # noqa: E402
from deck import (ACC, blank, fig, new_deck, path_line, save,  # noqa: E402
                  text)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EX = os.path.join(ROOT, "qc", "explainer")
ST = os.path.join(ROOT, "qc", "steps")
SW = os.path.join(ROOT, "qc", "face_sweep")
EN_ = os.path.join(ROOT, "qc", "enhanced")
M2P = (r"I:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
       r"15-56-48\output_split\motion")
STAMP = "20260930"


def dim():
    from deck import DIM
    return DIM


SLIDES = {
    "EN": dict(
        font="Calibri",
        t1="Mouse grimace scoring on our video: which step fails, and why",
        t2="Both PainFace and GrimACE were run. One scored it; the other refused.",
        t3=("mini2p  CEA-Ntsr1, 1600x1200, side-lit, mouse wears a "
            "two-photon miniscope\n"
            "SBI-553  mini1p, 720x480, light BEHIND the animal, no implant"),
        items=[
            ("1  What the grimace scale scores",
             "five facial action units, each 0 / 1 / 2. This is what every "
             "tool is estimating.",
             os.path.join(EX, "E1_mgs_action_units.png"), 5.0,
             "Bottom line is PainFace's own rule: if a human cannot see the "
             "unit, the network will not either."),
            ("2  What each method needs to have been given",
             "left of the line: the processing chain. Right: the recording "
             "conditions it was trained on.",
             os.path.join(EX, "E2_pipeline_theory.png"), 5.1,
             "All four read reflectance texture on a face pointed at the "
             "camera."),
            ("3  PainFace on our mini2p video - step by step",
             "it is not step 1. The platform accepted greyscale, converted, "
             "through-plastic video and ran.",
             os.path.join(ST, "S1_painface_steps_EN.png"), 5.3,
             "It breaks at step 4, on the one action unit our hardware sits "
             "on."),
            ("4  PainFace's own output, frame by frame",
             "the 7 frames it scored, one every 100 s, with its verdict on "
             "each.",
             os.path.join(M2P, "fig31_painface_frames.png"), 5.2,
             "Total = NA in 5 of 7 frames: fewer than four units found. "
             "Cheek is always NA - PainFace dropped that unit."),
            ("5  Why the score is not a pain score",
             "every frame PainFace called the EAR unit above zero, at "
             "confidence 0.85 to 1.00.",
             os.path.join(M2P, "fig33_painface_ear_unit.png"), 4.6,
             "Ear = 100 % of the first total and 50 % of the second. Drop "
             "the ear unit and the score is 0 and 1 out of 6."),
            ("6  Per action unit, what PainFace returned",
             "detection rate, score, and where the total comes from.",
             os.path.join(M2P, "fig32_painface_summary.png"), 5.0,
             "Nose and whiskers were 0 every time they were found. Orbital "
             "0.29. Ears 1.50 of 2."),
            ("7  The PainFace re-run at 10 s sampling - what changed",
             "same file, same crop, 66 scored frames instead of 7. Denser "
             "sampling, NOT higher resolution.",
             os.path.join(ST, "S3_painface_rerun_EN.png"), 5.3,
             "Yield is now MEASURED rather than worse: 29 % -> 12.1 %, CI 4-71 % -> 5.4-22.5 %. Run 1 was not biased - across all 10 grid phases the mean is 11.4 %. Two of my claims were wrong: the ear channel is NOT reading the miniscope (it fires at 0.92-0.99 on frames with no head in view), and orbital does vary - but of 18 orbital-positive rows an eye is visible in only 4."),
            ("8  GrimACE - it ran, and it cannot score this video",
             "the weights are public on Zenodo after all; my earlier 'not runnable' was wrong.",
             os.path.join(ST, "S4_grimace_steps_EN.png"), 5.3,
             "Positive control on GrimACE's OWN video passed first - 7 of 7 detections, confidence 0.9594, five distinct MGS vectors - so the install is correct. The wall is the FRAME-QUALITY gate: hard-coded 4.25/6 = 0.708, ours tops out at 0.4908, 0 of 326 frames pass under any transform. The face gate IS crossable (0.9141) but only on a window containing no mouse. And the forced output is the model's PRIOR - ear = 1 on 37 of 37 degenerate inputs including pure black and noise. Cheapest next test: inverting the image moves face detection 0.64 -> 0.88."),
            ("9  SBI-553 and any facial method - step by step",
             "steps 1 to 3 pass. The video is fine for the body.",
             os.path.join(ST, "S2_sbi_steps_EN.png"), 5.3,
             "It breaks at step 4: 1 confirmed eye in 168 frames "
             "adjudicated pixel by pixel."),
            ("10  Can sharpness and contrast fix SBI-553?",
             "partly - and my first look was wrong about why.",
             os.path.join(EN_, "fig_enhance_before_after.png"), 5.0,
             "Fur, whiskers and the ear come back. The eye does not. The "
             "light is BEHIND the animal."),
            ("11  What the enhanced head actually shows",
             "16 head crops, animal-local stretch, nearest-neighbour "
             "magnification, 10-pixel bar on each.",
             os.path.join(SW, "MGS_EARHEAD_00.png"), 5.2,
             "A clear ear in most frames. No eye in any of them."),
            ("12  The eye, at the same scale bar, on three cameras",
             "15 native pixels marked in red on every panel.",
             os.path.join(SW, "19_EVIDENCE_eye_scale_3cameras.png"), 5.3,
             "Top: mini2p, the best eye in the session. Middle and bottom: "
             "the other two cameras see mesh and the implant."),
        ],
        close_h="Bottom line",
        close=("TWO TOOLS, THE SAME VIDEO, AND NEITHER GIVES A USABLE "
               "SCORE\n"
               "  PainFace  scored it: 8 of 66 = 12.1 % (CI 5.4-22.5 %). "
               "61 % of all score points come from the EAR channel, and "
               "that channel\n"
               "            fires at confidence 0.92-0.99 on frames where "
               "the head is completely out of view. Its confidence is "
               "ANTI-informative.\n"
               "  GrimACE   cannot score it: frame-quality gate requires "
               "0.708, we reach 0.4908, 0 of 326 frames under any "
               "transform.\n"
               "            Its positive control passed first, so this is "
               "domain, not install. Forced output is its prior - ear = 1 "
               "on 37 of 37 degenerate inputs.\n\n"
               "HOW TO REPORT THIS SESSION\n"
               "  do NOT quote a Total Grimace Score. Report orbital + "
               "nose + whisker on 0-6 with the EAR EXCLUDED A PRIORI, not "
               "imputed.\n"
               "  that gives 23 of 66 = 34.8 % yield, of which 70 % read "
               "zero. No stimulus-block effect: 9 tests, smallest "
               "p = 0.095, wrong direction.\n\n"
               "SBI-553  passes steps 1-3. Contrast and sharpening DO help "
               "and recover the ear (66 % of 20296 frames, 31 px).\n"
               "           Breaks at step 4: 1 confirmed eye in 168 "
               "adjudicated frames.\n\n"
               "WHAT I GOT WRONG, FOR THE RECORD\n"
               "  said PainFace would refuse greyscale - it ran\n"
               "  said GrimACE was not runnable - weights are on Zenodo; "
               "it ran, and the wall is the quality gate, not the face "
               "gate\n"
               "  said the ear score IS the miniscope - only 3 of 13 sit "
               "over hardware; the channel keys on body contour\n"
               "  said orbital was flat - it varies, but 13 of 18 "
               "orbital-positive frames have no eye in view\n"
               "  called SBI-553 a pure silhouette - animal-band contrast "
               "recovers fur, whiskers and the ear\n\n"
               "THE CHEAPEST NEXT TEST, AND THE ONE THAT SETTLES IT\n"
               "  cheap: invert the image. That alone moves GrimACE face "
               "detection 0.64 -> 0.88 - back-lighting polarity is most of "
               "the gap.\n"
               "  decisive: the same rig and lighting with a black mouse "
               "and NO headcap. If the ear channel still fires at 0.92-0.99 "
               "on headless frames,\n"
               "            the miniscope is exonerated and the fix is the "
               "CAMERA VIEW, not the implant."),
    ),
    "KR": dict(
        font="Malgun Gothic",
        t1="우리 영상에서 grimace 점수화: 어느 단계에서 실패하고, 왜인가",
        t2="PainFace와 GrimACE 둘 다 돌렸습니다. 하나는 점수를 냈고, 하나는 거부했습니다.",
        t3=("mini2p  CEA-Ntsr1, 1600x1200, 측면 조명, 2광자 미니스코프 장착\n"
            "SBI-553  mini1p, 720x480, 빛이 동물 뒤, implant 없음"),
        items=[
            ("1  Grimace scale이 점수화하는 것",
             "5개 얼굴 action unit, 각각 0 / 1 / 2. 모든 도구가 추정하려는 "
             "값입니다.",
             os.path.join(EX, "E1_mgs_action_units.png"), 5.0,
             "맨 아래가 PainFace 자체 규칙입니다 - 사람이 못 보는 unit은 "
             "모델도 못 봅니다."),
            ("2  각 방법이 전제하는 녹화 조건",
             "선 왼쪽은 처리 사슬, 오른쪽은 학습된 녹화 조건입니다.",
             os.path.join(EX, "E2_pipeline_theory.png"), 5.1,
             "네 방법 모두 카메라를 향한 얼굴의 반사 텍스처를 읽습니다."),
            ("3  PainFace를 mini2p 영상에 - 단계별",
             "1단계가 아닙니다. 플랫폼은 흑백·변환·플라스틱 통과 영상을 "
             "받아서 돌렸습니다.",
             os.path.join(ST, "S1_painface_steps_KR.png"), 5.3,
             "4단계에서 깨집니다 - 우리 장비가 올라앉은 그 action unit에서."),
            ("4  PainFace가 실제로 낸 출력, 프레임별",
             "점수를 낸 7개 프레임(100초 간격)과 각 프레임의 판정입니다.",
             os.path.join(M2P, "fig31_painface_frames.png"), 5.2,
             "7개 중 5개가 Total = NA - unit 4개를 다 못 찾았기 때문입니다. "
             "Cheek은 항상 NA (PainFace가 버린 unit)."),
            ("5  이 점수가 통증 점수가 아닌 이유",
             "PainFace가 귀 unit을 0보다 크게 준 모든 프레임. 신뢰도 "
             "0.85-1.00.",
             os.path.join(M2P, "fig33_painface_ear_unit.png"), 4.6,
             "귀가 첫 총점의 100 %, 두 번째의 50 %. 귀를 빼면 6점 만점에 "
             "0점과 1점."),
            ("6  action unit별로 PainFace가 반환한 값",
             "검출률, 점수, 그리고 총점이 어디서 오는지.",
             os.path.join(M2P, "fig32_painface_summary.png"), 5.0,
             "코와 수염은 찾을 때마다 0점. orbital 0.29. 귀 1.50/2."),
            ("7  PainFace 10초 간격 재실행 - 무엇이 바뀌었나",
             "같은 파일, 같은 crop, 채점 프레임만 7개에서 66개로. 해상도가 "
             "아니라 샘플링 밀도입니다.",
             os.path.join(ST, "S3_painface_rerun_KR.png"), 5.3,
             "수율이 나빠진 게 아니라 이제 측정됐습니다: 29 % -> 12.1 %, CI 4-71 % -> 5.4-22.5 %. 1차는 편향이 아니었습니다 - 격자 위상 10가지 평균이 11.4 %. 제 주장 둘이 틀렸습니다: 귀 채널은 미니스코프를 읽는 게 아니고(머리가 안 보이는 프레임에서 0.92-0.99로 발화), orbital은 변하지만 orbital 양성 18개 행 중 눈이 보이는 것은 4개뿐입니다."),
            ("8  GrimACE - 돌았지만 이 영상은 채점 불가",
             "가중치는 Zenodo에 공개돼 있었습니다. 제 앞선 '실행 불가' 판단이 틀렸습니다.",
             os.path.join(ST, "S4_grimace_steps_KR.png"), 5.3,
             "GrimACE 자기 영상으로 대조 실행을 먼저 통과 - 검출 7/7, 신뢰도 0.9594, 서로 다른 MGS 벡터 5종 - 즉 설치는 정확합니다. 벽은 프레임 품질 게이트입니다: 코드에 박힌 4.25/6 = 0.708, 우리는 최대 0.4908, 어떤 변환으로도 326개 중 0개 통과. 얼굴 게이트는 넘을 수 있지만(0.9141) 마우스가 없는 창에서만 넘습니다. 그리고 강제 출력은 모델의 사전확률입니다 - 순수 검정과 노이즈를 포함한 퇴화 입력 37개 중 37개에서 귀 = 1. 가장 싼 다음 시험: 영상 반전만으로 얼굴 검출 0.64 -> 0.88."),
            ("9  SBI-553에 얼굴 기반 방법 - 단계별",
             "1-3단계는 통과합니다. 몸을 보는 데는 영상이 충분합니다.",
             os.path.join(ST, "S2_sbi_steps_KR.png"), 5.3,
             "4단계에서 깨집니다 - 픽셀 단위로 판정한 168 프레임 중 확인된 "
             "눈 1개."),
            ("10  샤프닝과 대비 보정으로 SBI-553을 살릴 수 있나?",
             "부분적으로 됩니다 - 그리고 제 첫 판단은 이유가 틀렸습니다.",
             os.path.join(EN_, "fig_enhance_before_after.png"), 5.0,
             "털·수염·귀는 돌아옵니다. 눈은 안 돌아옵니다. 빛이 동물 "
             "뒤에 있습니다."),
            ("11  개선된 머리 영상이 실제로 보여주는 것",
             "머리 crop 16장, 동물 자체 대역 스트레치, NEAREST 확대, 각 "
             "장에 10픽셀 바.",
             os.path.join(SW, "MGS_EARHEAD_00.png"), 5.2,
             "대부분 프레임에서 귀는 선명합니다. 눈은 어디에도 없습니다."),
            ("12  같은 스케일바로 본 눈 - 카메라 3대",
             "모든 패널에 15 native 픽셀을 빨간 바로 표시했습니다.",
             os.path.join(SW, "19_EVIDENCE_eye_scale_3cameras.png"), 5.3,
             "위: mini2p, 세션 최고의 눈. 가운데와 아래: 나머지 두 대는 "
             "mesh와 implant만 봅니다."),
        ],
        close_h="결론",
        close=("도구 둘, 같은 영상, 그리고 둘 다 쓸 수 있는 점수를 주지 "
               "않습니다\n"
               "  PainFace  점수를 냈습니다: 66개 중 8개 = 12.1 % "
               "(CI 5.4-22.5 %). 전체 점수의 61 %가 귀 채널에서 나오고, 그 "
               "채널은\n"
               "            머리가 완전히 보이지 않는 프레임에서 신뢰도 "
               "0.92-0.99로 발화합니다. 신뢰도 열이 역정보입니다.\n"
               "  GrimACE   채점 불가: 프레임 품질 게이트가 0.708을 "
               "요구하는데 우리는 0.4908, 어떤 변환으로도 326개 중 0개.\n"
               "            대조 실행을 먼저 통과했으니 설치가 아니라 "
               "도메인 문제입니다. 강제 출력은 사전확률 - 퇴화 입력 37/37에서 "
               "귀 = 1.\n\n"
               "이 세션을 어떻게 보고할 것인가\n"
               "  Total Grimace Score는 인용하지 마십시오. 귀를 사전 "
               "배제하고(대체하지 말고) orbital + 코 + 수염을 0-6으로 "
               "보고하십시오.\n"
               "  그러면 66개 중 23개 = 34.8 % 수율이고 그 중 70 %가 0점. "
               "자극 블록 효과 없음: 검정 9개, 최소 p = 0.095, 방향도 "
               "반대.\n\n"
               "SBI-553  1-3단계 통과. 대비 보정과 샤프닝은 실제로 도움이 "
               "되고 귀를 살려냅니다(20296 프레임 중 66 %, 31 px).\n"
               "           4단계에서 깨집니다 - 판정 168 프레임 중 확인된 "
               "눈 1개.\n\n"
               "제가 틀린 것, 기록을 위해\n"
               "  PainFace가 흑백을 거부할 거라고 했습니다 - 돌았습니다\n"
               "  GrimACE가 실행 불가라고 했습니다 - 가중치는 Zenodo에 "
               "있고 돌았으며, 벽은 얼굴 게이트가 아니라 품질 게이트였습니다\n"
               "  귀 점수가 미니스코프라고 했습니다 - 장비 위에 있는 것은 "
               "13개 중 3개뿐이고, 채널은 몸통 윤곽에 반응합니다\n"
               "  orbital이 평평하다고 했습니다 - 변하지만, orbital 양성 "
               "18개 중 13개는 눈이 보이지 않습니다\n"
               "  SBI-553을 순수 실루엣이라고 했습니다 - 동물 대역 대비 "
               "보정으로 털·수염·귀가 나옵니다\n\n"
               "가장 싼 다음 시험, 그리고 결판낼 시험\n"
               "  싼 것: 영상을 반전하십시오. 그것만으로 GrimACE 얼굴 검출이 "
               "0.64 -> 0.88 - 역광 극성이 격차의 대부분입니다.\n"
               "  결정적: 같은 리그·같은 조명에 headcap 없는 검은 마우스. "
               "머리가 안 보이는 프레임에서 귀 채널이 여전히 0.92-0.99로 "
               "발화하면,\n"
               "            미니스코프는 무죄이고 고쳐야 할 것은 implant가 "
               "아니라 카메라 뷰입니다."),
    ),
}


def build(lang):
    c = SLIDES[lang]
    D.FONT = c["font"]
    p = new_deck()

    s = blank(p)
    text(s, c["t1"], Inches(.8), Inches(1.9), Inches(11.7), Inches(1.1),
         size=30, bold=True)
    text(s, c["t2"], Inches(.8), Inches(3.05), Inches(11.7), Inches(.6),
         size=18, color=ACC)
    text(s, c["t3"], Inches(.8), Inches(3.9), Inches(11.7), Inches(1.4),
         size=15, color=dim())
    path_line(s, STAMP + "   |   Hansol Lim")

    for head, look, path, hbox, note in c["items"]:
        fig(p, head, look, path, None, hbox=hbox, slice_tall=False,
            note=note)

    s = blank(p)
    text(s, c["close_h"], Inches(.6), Inches(.4), Inches(12.2),
         Inches(.7), size=26, bold=True, color=ACC)
    text(s, c["close"], Inches(.6), Inches(1.35), Inches(12.2),
         Inches(5.7), size=13.5)
    return save(p, "Grimace_steps_" + lang + "_" + STAMP, ROOT)


if __name__ == "__main__":
    for lg in ("EN", "KR"):
        print(build(lg))
