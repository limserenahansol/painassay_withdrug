# 08 — HEAL mini1p + SBI-553 · Workflow & Suggestions

> Companion to `GUIDE.md`. **한국어는 아래.**

---
## ENGLISH

### 1. Workflow in one picture

```
Stimulus_randomisation_mini1p.xlsx   (24 sessions, pre-specified orders)
        │
        ▼
session recorded  →  2-camera video  →  videos\
        │
        │  manual_scoring_pain_assay.m   (BLIND: no treatment shown)
        ▼
videos\output\  RawScores_<vid>.csv  +  .mat  +  time-series / raster PNG
        │
        │  paste CSV into Raw_scores, then join Treatment + Stimulus
        ▼
Behavioural_scoring_book.xlsx  →  Summary_per_session (auto SUMIF)
        │
        ▼
crossover analysis: baseline vs vehicle vs SBI-553, within mouse
```

Question: does SBI-553 reduce **affective-motivational** pain behaviour without simply
flattening **reflexive** responses — and does mini1p activity track that split?

### 2. What the design already gets right

Within-mouse crossover, so each animal is its own control. Position-balanced stimulus
randomisation, so no stimulus is systematically first. Blind scoring by construction — the
script never reads treatment. Reflexive and affective kept apart, which is the whole point
of the Corder/Biafra framework.

### 3. Suggestions

**S1 — Score a subset twice and report reliability.**
Re-score ~20% of videos (ideally a second scorer) and report Cohen's κ for the binary
withdrawal call and ICC for the duration measures. Reviewers ask for this in every manual
scoring paper, and it is far cheaper to collect now than to retrofit.

**S2 — Lock the observation window before scoring, not after.**
The window is currently `(TBC)` and the script takes it as a dialog input. Once Greg confirms
it, hard-code the default so it cannot drift between videos. Different windows across
sessions would silently bias duration measures.

**S3 — Analyse durations as rates, not raw seconds.**
If a window ever gets truncated (next stimulus arrives early, video ends), raw seconds are
not comparable. The script already clips the window at the next stimulus onset — report
duration ÷ actual window length so those sessions stay usable.

**S4 — Mixed model, not paired t-tests.**
`behaviour ~ treatment * stimulus + (1|mouse)` with mouse as random effect. n = 6 with 4
stimuli and 3 conditions each is 72 observations, but only 6 independent animals — treating
stimuli as independent would be pseudoreplication. Same argument as project 02.

**S5 — Pre-register the primary outcome.**
Pick one affective measure (licking/biting duration is the usual choice) as primary before
unblinding, and treat the rest as secondary. With 6 mice and 6 behaviours, an unspecified
primary invites a false positive.

**S6 — Link scoring to imaging on the same clock.**
`stimulusOnsetFrames` in the `.mat` is the natural sync point to the mini1p trace. Log the
camera TTL alongside the miniscope frame clock so peri-stimulus dF/F can be aligned to the
scored behaviour without post-hoc guessing.

**Quick wins:** keep `videos\` flat, one session per file with the session number in the
filename; back up `videos\output\` before re-scoring anything, since re-running a video
overwrites its CSV and MAT.

---
## 한국어

### 1. 워크플로우 한눈에

```
Stimulus_randomisation_mini1p.xlsx  (24 세션, 순서 사전 지정)
        │
        ▼
세션 촬영 → 2대 카메라 영상 → videos\
        │
        │  manual_scoring_pain_assay.m  (BLIND: 처치 미표시)
        ▼
videos\output\  RawScores_<vid>.csv + .mat + 시계열/래스터 PNG
        │
        │  CSV를 Raw_scores에 붙여넣고 Treatment·Stimulus 결합
        ▼
Behavioural_scoring_book.xlsx → Summary_per_session (SUMIF 자동)
        │
        ▼
교차 분석: baseline vs vehicle vs SBI-553, 동물 내 비교
```

질문: SBI-553이 **reflexive** 반응을 단순히 둔화시키는 게 아니라 **affective-motivational**
통증 행동을 선택적으로 줄이는가, 그리고 mini1p 활동이 그 분리를 따라가는가.

### 2. 이미 잘 되어 있는 부분

동물 내 교차 설계라 각 마우스가 자기 대조군입니다. 자극 순서가 위치 균형을 이루어 특정
자극이 항상 먼저 오지 않습니다. 스크립트가 처치를 읽지 않으므로 **구조적으로 blind**입니다.
reflexive와 affective를 분리한 것이 Corder/Biafra 틀의 핵심입니다.

### 3. 제안

**S1 — 일부를 두 번 채점하고 신뢰도를 보고하세요.**
영상의 약 20%를 재채점(가능하면 두 번째 채점자)하고, 이분형 withdrawal은 Cohen's κ,
지속시간은 ICC로 보고. 수기 채점 논문에서 항상 요구되며, 나중에 소급하는 것보다 지금
모으는 게 훨씬 쌉니다.

**S2 — 관찰 창을 채점 전에 확정하세요.**
현재 `(TBC)`이고 스크립트가 대화상자로 받습니다. Greg가 확정하면 기본값을 하드코딩해서
영상마다 달라지지 않게 하세요. 세션마다 창이 다르면 지속시간 지표가 조용히 편향됩니다.

**S3 — 지속시간을 초가 아니라 비율로 분석하세요.**
창이 잘리는 경우(다음 자극이 일찍 옴, 영상 종료) raw 초는 비교 불가입니다. 스크립트가 이미
다음 자극 시점에서 창을 자르므로, **지속시간 ÷ 실제 창 길이**로 보고하면 그 세션도 살릴 수
있습니다.

**S4 — 대응 t-검정이 아니라 혼합모형.**
`behaviour ~ treatment * stimulus + (1|mouse)`. n=6에 자극 4개 × 조건 3개면 관측치는 72개지만
**독립 동물은 6마리**입니다. 자극을 독립으로 취급하면 pseudoreplication입니다 — 프로젝트 02와
같은 논리입니다.

**S5 — 주 결과 지표를 미리 정하세요.**
unblinding 전에 affective 지표 하나(보통 licking/biting 지속시간)를 primary로 지정하고 나머지는
secondary로. 6마리에 행동 6종인데 primary가 없으면 위양성이 나오기 쉽습니다.

**S6 — 채점과 영상을 같은 시계에 묶으세요.**
`.mat`의 `stimulusOnsetFrames`가 mini1p 트레이스와의 자연스러운 동기 지점입니다. 카메라 TTL을
miniscope 프레임 클럭과 함께 기록해 두면 peri-stimulus dF/F를 추측 없이 정렬할 수 있습니다.

**빠른 개선:** `videos\`는 평평하게 유지하고 파일명에 세션 번호를 넣으세요. 재채점 전에는
`videos\output\`을 백업하세요 — 같은 영상을 다시 돌리면 CSV와 MAT을 덮어씁니다.
