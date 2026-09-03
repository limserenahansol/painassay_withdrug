# Automatic labelling — what has actually been tested

Date: 2026-09-02 · one WT session (`mousevideo_WT`, female 1, bottom + side)

**Read this before trusting any automatic output.** It separates what is
measured from what is not.

---

## The short answer

| Detector | Evidence | Verdict |
|---|---|---|
| Stimulus delivery (bottom) | 12/12 random detections real, 4/4 controls clean | **precision good, count NOT reliable** |
| Rearing / escape (side) | d′ = 6.0, 12/12 and 4/4 visually correct | **detection good, duration NOT calibrated** |
| Guarding, attending | none | **untested — needs manual labels** |
| Licking, biting, withdrawal, flinch | none | **not attempted; blocked by rig, not code** |

There is **no supervised performance number yet**, because there are no manual
labels yet. The classifier pipeline (`auto_B_train.py`) has only been run on
synthetic labels derived from the features themselves — that is a circular test
which proves the code executes and nothing about accuracy. Those F1 values must
not be quoted.

---

## 1. Stimulus delivery detector

`auto/auto_A_delivery_detect.py` on the bottom view, whole 28 min session.

Figure: [auto_check_delivery.png](auto_check_delivery.png) — a **random** sample
of 12 of the 40 candidates, plus 4 control frames drawn from times with no
detection. Random, not top-scoring, so it is an unbiased look at precision.

**What is established**

- **Precision: 12/12** of the random sample show a visible hand or applicator.
  Clopper–Pearson 95 % CI [0.74, 1.00] → precision is **at least 74 %** with
  95 % confidence.
- **4/4 controls** show the mouse alone, no hand, no applicator.
- The habituation period is clean: first detection at 319 s, and the intrusion
  trace is flat and event-free before that. It is not firing on mouse movement.

**What is NOT established**

- **The count is not reliable.** 40 raw candidates, but:
  - 5 inter-event gaps are **< 5 s** — almost certainly one delivery split into
    two detections. Merging those gives **35**, not 40.
  - 9 gaps are **> 60 s** (max 117 s) — a delivery could be hidden in there.
  - 1 event is 28.7 s long (flagged) — the hand was probably left in frame.
- **Recall is unmeasured.** Without the true delivery count for this session
  there is no way to say how many were missed.

An earlier note in `GUIDE.md` read the 40 as matching "4 stimuli × 10 repeats".
That was over-reading a coincidence and has been corrected.

**So use it for what it is:** a list of timestamps to check your manual count
against, not a count in its own right.

---

## 2. Rearing / escape detector

`auto/auto_B_features.py` on the side view, then an **unsupervised** rule — no
labels used at all: `height_norm` (body height ÷ width), 0.5 s rolling mean,
above the 98th percentile for ≥ 0.3 s.

Figure: [auto_check_rearing.png](auto_check_rearing.png) — random 12 of 48
detections, plus 4 controls at median posture.

**What is established**

- Segmentation held on **16,824 / 16,824 frames (100 %)** across the full
  session, using the QC-validated ROI and threshold.
- The posture feature separates cleanly:

  | | height ÷ width |
  |---|---|
  | detected (upright) | **1.14 ± 0.09** |
  | rest of session (crouched) | **0.46 ± 0.13** |
  | separation | **d′ = 6.0** |

- **12/12** random detections show the mouse fully upright against the cylinder
  wall — elongated body, tail hanging, forepaws up in two of them.
- **4/4** controls show a crouched mouse on all fours (h/w 0.33–0.42).
- Percentiles: p50 = 0.43, p95 = 0.85, p99 = 1.12. The upright tail is real and
  well separated from the bulk.

**What is NOT established**

- **The bout count and total duration are threshold-dependent**, and the
  threshold has no principled value yet:

  | threshold | h/w | bouts | % of session |
  |---|---|---|---|
  | p95 | 0.85 | 72 | 5.0 % |
  | p97 | 1.00 | 61 | 3.0 % |
  | **p98** | **1.07** | **48** | **2.0 %** |
  | p99 | 1.12 | 26 | 1.0 % |

  That is a **2.8× swing** in bout count over a narrow threshold range.
- **The boundary cases are the problem, not the extremes.** 9.6 % of the
  session (≈ 2.7 min) sits in the ambiguous zone h/w 0.6–1.0 — partial rears,
  stretching, wall-sniffing. The montage only demonstrates that the *clear*
  cases are trivially separable.

**So:** reliable for "did the mouse rear here", not yet calibrated for "how many
seconds of rearing". Your manual labels are what will fix the threshold.

---

## 3. Everything else

- **Guarding (behaviour 6)** and **paw attending (behaviour 3)**: the features
  exist (`low_contour_rough`, convexity defects, motion windows) and
  `auto_B_train.py` is wired to model them, but **nothing has been validated**.
  Neither has an unsupervised proxy as clean as posture height, so both need
  real labels.
- **Licking, biting, paw withdrawal, flinch**: not attempted, and not a coding
  problem. At 30 fps a withdrawal is 1.5–4.5 frames, and the tongue is ~5 px at
  this magnification. See `WT_RECORDING_QC.md` §4 — zooming the side view in and
  raising the frame rate are the fixes, not a better model.

---

## What to do next

1. Score 4–6 sessions in `score_AB_dual_view.m`.
2. Run `auto_B_features.py --step 1` on the matching side videos.
3. Run `auto_B_train.py train` and read the **bout** columns, not frame F1.
4. Use the labels to pick the rearing threshold instead of a percentile guess.
5. Validate on sessions you scored but did not train on, before using any
   automatic output in an analysis.

Never mix predicted and manually scored sessions inside one statistical
comparison.
