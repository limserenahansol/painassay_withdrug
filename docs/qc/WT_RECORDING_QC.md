# WT recording QC — bottom + side, one mouse

Source: `08_HEAL_mini1p_SBI553/mousevideo_WT/{bottom,side}/`
Files: `female1bottom0001 26-09-02 12-00-19.avi`, `female1side0001 26-09-02 12-00-20.avi`
Figure: [WT_recording_QC.png](WT_recording_QC.png)

---

## 1. Recording specs

| | bottom | side |
|---|---|---|
| Resolution | 720 × 480 | 720 × 480 |
| Frame rate | 30.00 fps | 30.00 fps |
| Frames | 50,427 | 50,471 |
| Duration | 1681 s (28.0 min) | 1682 s (28.0 min) |
| File size | 6.07 GB | 6.07 GB |
| Codec | `dvsd` (DV) | `dvsd` (DV) |
| Interlace comb (moving pixels) | 0.55 → none | 0.65 → none |
| Mean brightness | 97.4 / 255 | 142.6 / 255 |
| Clipped pixels | 8.4 % | 11.1 % |

**The optics advice was acted on and it worked.** The bottom view is now
backlit transillumination — the mouse is a clean high-contrast silhouette
instead of the unrecoverable dark blob on the old rig. The side view has a
bright cloth backdrop and the experimenter is out of frame. The specular
streak on the cylinder now sits at x ≈ 420–600, i.e. **off the animal**.

---

## 2. Segmentation quality

Measured on 60 frames evenly spaced across each session.

| Metric | bottom (I < 45, full frame) | side (I < 75, ROI x 110–600, y 90–370) |
|---|---|---|
| Mouse detected | 60/60 (100 %) | 60/60 (100 %) |
| Foreground/background contrast | 1.2 SD | **2.8 SD** |
| Body area | 21,942 px | 20,812 px |
| Area CV | 0.22 | 0.23 |
| Body bounding box | 228 × 195 px | 298 × 124 px |
| Solidity | 0.70 | **0.82** |
| Within-animal SD | 12.5 grey levels | 16.3 grey levels |
| Within-animal p5–p95 | 37 grey levels | 51 grey levels |
| Motion-energy bouts (z > 5, 10 min) | 3 | **73** |

**Side view is the better analysis view on every measure.**

### Two things that had to be fixed to get there

1. **Background subtraction does not work on these recordings.** The mouse is
   almost stationary — centroid scatter is only SD 19 × 14 px (side) and
   15 × 9 px (bottom), median offset 11–14 px, max 58–73 px, against a body
   ~200–300 px long. So a temporal-median background plate *already contains
   the mouse*, and `|frame − background|` recovers only the sliver where the
   animal differs from its own median position (drops to 0.67 SD contrast,
   area CV 0.88). Use an absolute threshold inside a fixed ROI, or a learned
   tracker.
2. **The side view needs an x-ROI.** A dark cloth fold occupies x < 110 at
   92–117 grey; the mouse body is 55 grey. Above a threshold of ~90 the two
   merge into one frame-spanning blob and detection collapses to 0 %.
   Cropping to x 110–600 and thresholding at 75 gives 100 %.

---

## 3. Automatic labelling — per behaviour verdict

| # | Behaviour | Auto? | View | Limiting factor |
|---|---|---|---|---|
| — | **Stimulus delivery** (camera A, 1 event) | **YES — high confidence** | bottom | none; robust z ≈ 21, ~32–36 discrete events per 10 min |
| 1 | Paw withdrawal (0/1) | Partial | side + bottom | 30 fps: a withdrawal is 50–150 ms = **1.5–4.5 frames** |
| 2 | Flinch / flick (count) | **NO** | — | too fast for 30 fps |
| 3 | Paw attending | YES | side | needs DLC keypoints (snout↔paw distance) |
| 4 | Licking | Marginal | side | tongue ≈ 5 px — **not resolvable** |
| 5 | Biting | Marginal | side | separable from licking only via head-jerk kinematics |
| 6 | Sustained lifting / guarding (> 2 s) | **YES** | side | slow event, well matched to 30 fps |
| 7 | Escape / rearing | **YES** | side | body area + eccentricity |

### Why the bottom view cannot do paw-level scoring

The mouse images as a **fully saturated black silhouette with no internal
detail** (within-animal p5–p95 = 37 grey levels, and the interior is at floor
value). You get centroid, area, orientation and tail — nothing else. Worse, the
honeycomb mesh has a pitch of ~20 px while a paw is ~25 px, so a paw sitting
over a dark hexagon wall is **invisible**, and paw visibility flickers with the
mesh phase rather than with behaviour. This is a geometry problem, not a
processing problem — no filter recovers it.

The bottom view's real value is the **stimulus-delivery channel**, and there it
is excellent: the hand plus von Frey filament is a large unmistakable dark
intruder (see figure, bottom-right panel).

---

## 4. Problems to fix before the real 24 sessions

1. **Cameras are not synchronised.** File timestamps are 12:00:19 vs 12:00:20
   and the two files differ by 44 frames (1.5 s). The `syncOffset` field in
   `score_B_mouse_behavior.m` currently carries this by hand.
   → *Fix:* put an LED in both fields of view and flash it once at session
   start. One frame of ground truth removes the whole problem.
2. **Frame rate.** 30 fps cannot support withdrawal latency or flinch counts.
   → *Fix:* if reflexive measures matter, record ≥ 120 fps (drop resolution if
   needed). Otherwise accept that behaviours 1–2 stay manual and are scored as
   present/absent only — and say so in the methods.
3. **Side-view field of view is too wide.** The mouse spans 298 of 720 px
   (41 %). → *Fix:* zoom so it fills ~70 %. This roughly doubles paw and
   snout pixel counts and is the cheapest single improvement for licking vs
   biting.
4. **Exposure.** 8–11 % of pixels are clipped. → *Fix:* stop down ~1 EV. The
   silhouette does not need a blown-out backdrop.
5. **Left cloth fold on the side view.** → *Fix:* tension the cloth so the ROI
   crop is not needed.
6. **Hand merges with the mouse silhouette during delivery** (visible in the
   figure, middle panel of row 2 — the blob exceeds the area gate and tracking
   jumps). → *Fix:* mark delivery frames from the bottom view and exclude them
   from body tracking, or interpolate across them.
7. **Storage.** DV at 6.07 GB / 28 min × 2 cameras × 24 sessions ≈ **290 GB**.
   → *Fix:* transcode to H.264 CRF 20 for analysis (~15–20× smaller, no
   meaningful loss for behaviour scoring). Keep the DV originals on the drive
   they came from.

---

## 5. Recommended pipeline

```
camera A (bottom)  --> classical hand-intrusion detector  --> stimulus onset frames
                                                               (replaces score_A entirely)
camera B (side)    --> DeepLabCut / SLEAP keypoints       --> feature time series
                       (snout, L/R fore-paw, L/R hind-paw,      |
                        tail base, body centre, both ears)      v
                                                          behaviour classifier
                                                          (SimBA or gradient boosting)
                                                               |
                                                               v
                                            behaviours 3, 6, 7 automatic
                                            behaviours 1, 2, 4, 5 manual
```

Training data needed:
- **DLC keypoints:** 200–400 hand-labelled frames drawn across all 6 mice and
  both treatments. This is a one-off cost.
- **Behaviour classifier:** the `TrainingLabels_<vid>.csv` that
  `score_B_mouse_behavior.m` already writes is exactly the right input.
  Budget ~4–6 fully manually scored sessions before the classifier is worth
  trusting.

**Do not skip manual scoring for this study.** Score all 24 sessions by hand as
planned, and build the classifier in parallel from those labels. The automatic
pipeline is for the *next* cohort, where it can be validated against the manual
labels you will already have.

---

## 6. Bottom line

- **QC verdict: the recordings are usable.** Both views segment at 100 % with
  the right settings, and the rig is much improved over the old setup.
- **Automatic labelling: partially viable now.** Stimulus delivery (camera A)
  can be fully automated today. Guarding, escape/rearing and paw attending are
  reachable with DLC on the side view. Withdrawal, flinch, licking and biting
  are blocked by frame rate and pixel size, not by algorithm choice.
- **The two cheapest fixes with the largest payoff:** tighten the side-view
  zoom, and add a sync LED.

---

## 요약 (한국어)

**1. 녹화 자체는 문제 없습니다.** 720×480, 30 fps, 28분, 두 대 모두 6.07 GB.
인터레이스 문제 없고, 예전 리그보다 훨씬 좋아졌습니다. 조명 조언이 실제로
반영되었고 효과가 있었습니다 — bottom은 이제 backlit 실루엣이고, side는 흰
배경막에 실험자가 화면에서 빠졌습니다. 실린더 반사 줄무늬도 마우스에서
벗어났습니다.

**2. Side view가 모든 지표에서 더 좋습니다.**
- 대비 2.8 SD (bottom 1.2), solidity 0.82 (bottom 0.70)
- 행동 신호(motion energy z>5): side 73회 vs bottom 3회 — **24배 차이**
- 두 뷰 모두 올바른 설정에서 60/60 (100%) 검출

**3. 두 가지를 고쳐야 분할이 됩니다.**
- **배경 차분은 이 영상에 쓸 수 없습니다.** 마우스가 28분 동안 거의 안
  움직여서(중심 이동 median 11–14 px, 몸길이 200–300 px) 시간축 median
  배경에 이미 마우스가 들어갑니다. 절대 임계값 + 고정 ROI를 써야 합니다.
- **Side는 x-ROI가 필요합니다.** 왼쪽 천 주름(92–117)이 마우스(55)와
  임계값 90 이상에서 병합됩니다. x 110–600, 임계값 75로 100% 나옵니다.

**4. 자동 라벨링 — 행동별 판정**

| 행동 | 자동? | 뷰 | 병목 |
|---|---|---|---|
| **자극 전달** (카메라 A) | **가능 — 확실함** | bottom | 없음. z≈21, 10분당 32–36회 |
| 1 Paw withdrawal | 부분적 | side+bottom | 30 fps로 50–150 ms = **1.5–4.5 프레임** |
| 2 Flinch/flick | **불가** | — | 30 fps로 너무 빠름 |
| 3 Paw attending | 가능 | side | DLC 키포인트 필요 |
| 4 Licking | 한계 | side | 혀가 ~5 px — **해상 불가** |
| 5 Biting | 한계 | side | 머리 움직임 패턴으로만 구분 |
| 6 Guarding (>2 s) | **가능** | side | 느린 사건, 30 fps로 충분 |
| 7 Escape/rearing | **가능** | side | 면적 + 편심률 |

**5. Bottom view로는 발 단위 채점이 불가능합니다.** 마우스가 내부 대비가
전혀 없는 완전 포화 검은 실루엣입니다. 게다가 벌집 메시 간격(~20 px)이
발 크기(~25 px)와 비슷해서, 발이 어두운 육각형 벽 위에 놓이면 아예 안
보입니다. 발이 보이는지가 행동이 아니라 **메시 위치에 따라 깜빡입니다.**
이건 광학·기하 문제라서 어떤 필터로도 복구 안 됩니다. Bottom view의 진짜
가치는 **자극 전달 검출**이고, 거기서는 아주 좋습니다.

**6. 본 실험(24 세션) 전에 고칠 것 — 중요한 순서대로**
1. **동기화.** 파일 타임스탬프가 12:00:19 vs 12:00:20이고 프레임 수가 44
   프레임(1.5 s) 다릅니다. → 두 카메라에 다 보이는 LED를 세션 시작 때 한 번
   깜빡이세요. 이거 하나로 해결됩니다.
2. **Side view를 더 당겨 찍으세요.** 마우스가 720 px 중 298 px(41%)만
   차지합니다. 70%까지 채우면 발·코 픽셀이 2배가 되고, licking vs biting
   구분에 가장 효과적인 단일 개선입니다.
3. **프레임률.** 반사 반응(1, 2번)을 정량화하려면 120 fps 이상 필요합니다.
   아니면 present/absent만 보고하고 논문에 그렇게 쓰세요.
4. **노출을 1 EV 낮추세요.** 픽셀 8–11%가 포화입니다.
5. **왼쪽 천 주름을 당겨 펴세요.** 그러면 ROI crop이 필요 없습니다.
6. **자극 전달 중에는 손과 마우스 실루엣이 붙습니다** → bottom에서 전달
   프레임을 표시해 body tracking에서 제외하세요.
7. **저장 용량.** 6.07 GB × 2대 × 24세션 ≈ **290 GB**. 분석용으로는 H.264
   CRF 20으로 변환하세요 (15–20배 작아지고 채점에는 손실 없음).

**7. 결론 — 이번 연구는 수동 채점을 그대로 하세요.** 24 세션 전부 손으로
채점하고, 그 라벨(`score_B_mouse_behavior.m`이 이미 만드는
`TrainingLabels_<vid>.csv`)로 분류기를 **병행해서** 만드세요. 자동 파이프라인은
*다음* 코호트용이고, 그때는 이미 확보한 수동 라벨로 검증할 수 있습니다.
지금 당장 자동화할 수 있는 건 **카메라 A의 자극 전달 시각**이고, 이것만으로도
수동 작업의 절반이 없어집니다.
