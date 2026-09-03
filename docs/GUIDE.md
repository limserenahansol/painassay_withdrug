# 08 — HEAL · mini1p imaging + SBI-553 pain assay

Acute pain battery under **mini1p** single-photon imaging, with **SBI-553** (biased NTSR1
allosteric modulator) tested against a matched vehicle in a **within-mouse crossover**.
Reflexive and affective-motivational behaviours are scored **separately** — there is no
combined pain score.

| | |
|---|---|
| Canonical folder | `C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\` |
| Scoring code | `HEAL_mini1p_SBI553\scoring\` — `RUN_scoring.m` → `score_AB_dual_view.m` |
| Analysis code | `HEAL_mini1p_SBI553nalysis\` — QC → blocks → stats → day comparison |
| **Full pipeline** | ★ **[PIPELINE.md](PIPELINE.md)** — start here to run or share this |
| Open with | `GOTO_HEAL_mini1p_SBI553.lnk` |
| Language | MATLAB (scoring) + Excel (schedule, scoring book) |
| Grant | UNC HEAL (SPO 322136) |
| Collaborators | Greg, David, Mark |
| Status | protocol drafted; awaiting Greg/David sign-off on the (TBC) fields |

**📁 Files in the canonical folder**

| File | Purpose |
|------|---------|
| `mini1p_SBI553_protocol_1page.docx` | ★ one-page protocol; **(TBC)** fields are what Greg/David must fill |
| `Stimulus_randomisation_mini1p.xlsx` | 24-session schedule + QC + open questions |
| `Behavioural_scoring_book.xlsx` | definitions, `Raw_scores` (one row per delivery), `Normalized_per_stim` (96 rows), `QC_normalisation` |
| `scoring\score_AB_dual_view.m` | ★ **both cameras + live label track** — delivery + type + the 6 behaviours, 3 modes |
| `scoring\RUN_scoring.m` | ★ **open this and press F5** — launcher for the scorer |
| `scoring\auto\` | `auto_A_delivery_detect.py`, `auto_B_features.py`, `auto_B_train.py` |
| `qc\` | `WT_RECORDING_QC.md`, `AUTO_LABELLING_RESULTS.md` + figures |
| `scoring\score_A_stimulus_delivery.m` | superseded (and cannot open a 28 min DV file) |
| `scoring\score_B_mouse_behavior.m` | superseded; re-scores behaviour on existing delivery times |
| `scoring\manual_scoring_pain_assay.m` | earlier single-camera version, reference only |
| `videos\cameraA\` , `videos\cameraB\` | drop videos here; `output\` auto-created in each |

---

## Design

| | |
|---|---|
| Animals | 6 mini1p mice — 3 male (M1–M3), 3 female (F1–F3) |
| Days | 2. Each day: no-injection baseline → injection → repeat assay |
| Crossover | Day 1 vehicle → Day 2 SBI-553, and the reverse. 3/3 per day, sex-balanced |
| Comparisons | no-injection baseline · vehicle · SBI-553, all within the same mouse |
| Stimuli | Light touch · Mild touch · Heat · Pin prick — 4 per session |
| Randomisation | all 24 permutations used once; each stimulus appears 6× in each position |
| Earliest start | ≥6 weeks post-surgery → mid-to-late October, per mouse from its own surgery date |

**Sessions** = 6 mice × (Day1 baseline, Day1 post, Day2 baseline, Day2 post) = **24**.
Scoring rows = 24 × 4 stimuli = **96**.

---

## Still to be confirmed by Greg / David

These are flagged in orange in the protocol and listed on the `To_confirm` sheet.

| Item | Who |
|------|-----|
| SBI-553 effect window (injection → assay start; ~20 min mentioned, not fixed) | Greg |
| Rest/acclimation, baseline→injection interval, assay duration, inter-stimulus interval | Greg |
| SBI-553 dose / route / vehicle formulation | Greg |
| Vehicle control composition — **matched vehicle, not plain saline** | Greg |
| Morphine dose — 10 mg/kg likely too high; 3–5 mg/kg provisional only | Greg |
| Morphine vs SBI-553 sequence and washout | Greg / David |
| Re-randomise stimulus order within a mouse (baseline vs post)? | Greg / Mark |
| Exact scoring items and observation window (Corder / Biafra) | Greg |

---

## Manual scoring — one pass, both cameras at once

### How to run it

1. Put the session's two videos in these folders, **one video each**:
   - `Documents\HEAL_mini1p_SBI553\videos\cameraA\` ← **bottom** view
   - `Documents\HEAL_mini1p_SBI553\videos\cameraB\` ← **side** view
2. Open `scoring\RUN_scoring.m` in MATLAB and press **F5**.
3. Fill in the dialog, press **ENTER** on the video window, score.
4. Results appear in `Documents\HEAL_mini1p_SBI553\videos\output\`.

`RUN_scoring.m` is a launcher: it checks the folders, prints a clear message if
a video is missing, then calls `score_AB_dual_view.m`. To practise on the WT
recordings first, uncomment the two marked lines inside it.

Equivalent one-liner in the MATLAB Command Window:

```matlab
cd('C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\scoring');
score_AB_dual_view('..\videos\cameraA', '..\videos\cameraB');
```

**Run it from the MATLAB desktop, not `matlab -batch`** — it needs a window and
your keyboard.

Stopped half way? The script prints the time it reached; run again and put that
number in the *Start at time (s)* field.

**`score_AB_dual_view.m` is the script to use.** Both videos are shown side by side in one
window — bottom view left, side view right — so a single keystroke can be based on both.
This matters most for licking vs biting: the side view shows the head-to-paw geometry, the
bottom view shows the mouth actually on the paw. Stimulus delivery and the six behaviours
can be scored together in one pass, or split into two passes - see the modes below.

Frames are **streamed**, never buffered. Measured on the WT DV files: dual-stream playback
runs at ~135 fps (4.5× real time), the first seek takes ~2 s, and frame stepping after that
is 1–20 ms in either direction.

### Three modes

You pick one when the script starts.

| Mode | What you score | Output |
|---|---|---|
| **1 STIMULUS ONLY** | just the deliveries and their type | `DeliveryTimes_*.mat` / `.csv`, `DeliveryCounts_*.csv` |
| **2 BEHAVIOUR ONLY** | behaviours, importing a mode-1 delivery file | everything else |
| **3 BOTH AT ONCE** | one pass over the session | everything |

Doing **1 then 2** is slower but easier — marking contact frames is a different
kind of attention from watching behaviour. Mode 3 is one 28 min pass instead of
two. Mode 2 shows the imported deliveries as lines on the live track, so you
always see where the epochs are.

### Keys

Same muscle memory as `manual_scoring_video_multibehavior_batch2.m` — **SPACE**
pauses, **`q`** stops, affective behaviours are held on the home row from `a`.

| Key | Action | Class |
|-----|--------|-------|
| **hold** `a` | Paw attending — no mouth contact | affective |
| **hold** `s` | **Licking or biting** — any mouth contact with the paw | affective |
| **hold** `d` | Sustained lifting / guarding | affective |
| **hold** `f` | Escape / rearing | affective |
| **tap** `w` | Paw withdrawal | reflexive |
| **tap** `e` | Flinch / flick | reflexive |
| **tap** `1` `2` `3` `4` | **stimulus delivery, by type** | epoch |
| **tap** `0` | delivery of unknown type — avoid, it has no denominator | epoch |
| **hold** `u` | uncertain → excluded from classifier training | — |
| `SPACE` · `←` `→` | pause · step one frame while paused | |
| `.` · `,` | faster · slower (1.5× per press, 0.1–8×) | |
| `z` · `q` | undo last delivery or reflex mark · stop and save | |
| **mouse drag** | jump anywhere in the session (seek bar) or the last 30 s (track) | |
| `BACKSPACE` | delete the one mark nearest the cursor | |

**Six behaviours: 2 reflexive + 4 affective.**

### Going back to fix a mistake

**Drag the seek bar** at the bottom of the window to jump anywhere in the
session, exactly like a video player. Clicking the label track above it seeks
finely within the last 30 s. Seeking auto-pauses.

Then, depending on what is wrong:

| Problem | Fix |
|---|---|
| a held behaviour is wrong | play forward through it again with the correct key held (or nothing held) — the frames you pass through are overwritten |
| a delivery or reflex mark is in the wrong place | park the cursor on it and press `BACKSPACE`, then re-mark it |
| you just mis-tapped a moment ago | `z` |

`BACKSPACE` removes the **single nearest** mark within ±0.4 s and prints its
name, so it never quietly takes out the neighbour. Press again for the next
one. This matters because real marks come as close as 0.9 s apart — measured
in the `female1` session — so a delete-everything-in-window rule would be
destructive.

Use `BACKSPACE` rather than `z` after seeking: `z` undoes the most *recent*
mark chronologically, which is not the one you went back for.

### Live label track

A scrolling raster sits under the two videos showing the last 30 s (adjustable)
of everything you have marked: one coloured row per affective behaviour, a row
each for withdrawal and flinch, and a black vertical line at every delivery.
The right edge is the current frame.

It exists so you can see your own label history while scoring — a stuck key, a
missed delivery, or a bout you forgot to release shows up immediately instead of
at the end of the session. It is redrawn every 4th frame with a single
`set(CData)` call, so it does not slow playback.

### Licking and biting are ONE category

An earlier version separated them. They are pooled here because the two cannot
be told apart reliably at this magnification — the tongue is ~5 px in the side
view (see [qc/WT_RECORDING_QC.md](qc/WT_RECORDING_QC.md)). Scoring them
separately would have manufactured a distinction the video cannot support.

Note what this does and does not cost. The distinction that was impossible was
*licking vs biting*. "Is the mouth on the paw or not" is a head-to-paw posture
question, which the side view carries well — so pooling actually makes this
category **more** automatable, not less.

**Guarding** = the paw is held up with no weight-bearing, *beyond* the brief
withdrawal reflex. Threshold is **> 2 s** — shorter episodes are discarded
automatically (dialog field, default 2). Verified: a 3.4 s bout is kept, a 0.7 s
bout is dropped.

### Why the stimulus type has to be entered

**Mice do not get the same number of each stimulus** — 10 pin pricks for one animal, 11 for
another. Raw totals are therefore not comparable between animals, so pressing `1`–`4` rather
than one generic "delivery" key is what makes normalisation possible.
`DeliveryCounts_<vid>.csv` is the denominator.

Two denominators are written, because an observation window can be cut short when the next
stimulus arrives early:

| Column suffix | Divided by | Use for |
|---|---|---|
| `_per_stim` | n delivered | **counts** — withdrawals, flinches, episodes |
| `_pct_time` | time actually observed | **durations** — licking, biting, guarding |

When the two disagree, the windows were truncated — trust `_pct_time` for durations.

*Worked example, verified in Excel:* Heat n=10 → withdrawal rate 0.500; Pin prick n=11 →
0.545. The raw counts (5 vs 6) are not comparable; the rates are.

### Camera sync — must be fixed in hardware

The dialog takes a *side minus bottom* offset in seconds. The WT files show why this cannot be
left at 0: timestamps 12:00:19 vs 12:00:20, and the two files differ by 44 frames. Worse,
**the experimenter's hand is visible only in the bottom view**, so there is no common event to
align on by eye. Put an LED in both fields of view and flash it once at session start.

### Outputs

| File | Contents |
|------|----------|
| `DeliveryTimes_<vid>.csv` · `.mat` | delivery frame, time, stimulus code and name |
| `DeliveryCounts_<vid>.csv` | **n delivered per stimulus — the denominator** |
| `RawScores_<vid>.csv` | one row per delivery, exact `Raw_scores` column order |
| `Normalized_<vid>.csv` | one row per stimulus type, every measure per stimulus |
| `TrainingLabels_<vid>.csv` | frame-level labels for a classifier (see below) |
| `ScoringAB_<vid>.mat` | everything, including the frame-wise `score` |
| `BehaviorTimeSeries` · `RasterPlot` · `NormalizedPerStim` `.png` | figures |

`Treatment` is written as **BLIND** on purpose. Fill it afterwards by joining on
Session / Mouse ID against `Stimulus_randomisation_mini1p.xlsx`.

### Built for training your own classifier later

`TrainingLabels_<vid>.csv` carries two label columns on purpose:

- `affective_code` — exactly as scored. **Use this for the behavioural statistics.**
- `affective_code_ml` — shifted earlier by the key-press lag (dialog field, default 250 ms).
  **Use this to train a classifier.** A human presses the key a few hundred ms after the
  behaviour starts and releases a few hundred ms after it ends; left uncorrected that lag
  teaches the model that the behaviour begins later than it does, which caps achievable
  accuracy. Correcting only the ML copy leaves the scored measures untouched.

`exclude_from_training = 1` marks frames you held `u` on, plus the tail where the shifted
label ran off the end. Those frames never teach the model something you were unsure of.

### Superseded scripts (kept, not deleted)

| File | Status |
|---|---|
| `score_A_stimulus_delivery.m` | superseded. Also **cannot open a 28 min DV file** — it buffers every frame, ~52 GB. |
| `score_B_mouse_behavior.m` | **obsolete - do not use.** Scores 7 behaviours with licking/biting separate and writes columns the scoring book no longer has. Use **mode 2** instead. |
| `manual_scoring_pain_assay.m` | earlier single-camera version, reference only. |

### Automatic labelling — `scoring\auto\`

Three scripts, in the order you will use them. **None of them replaces manual
scoring for this study** — they exist so that your 24 manually scored sessions
become a classifier you can use on the *next* cohort.

**What has actually been tested is in
[qc/AUTO_LABELLING_RESULTS.md](qc/AUTO_LABELLING_RESULTS.md).** There is no
supervised accuracy number yet, because there are no manual labels yet.

DeepLabCut / SLEAP / torch are **not installed** on this machine, so these use
side-view silhouette features and scikit-learn, which are. If you install DLC
later, append its keypoint columns to the features CSV and `auto_B_train.py`
picks them up with no code change.

#### 1. `auto_A_delivery_detect.py` — cross-check your delivery count

```bash
python auto/auto_A_delivery_detect.py <bottom_video_or_folder> --session 7
```

Detects the hand and applicator entering the bottom view. Tested on the WT
session — see **[qc/AUTO_LABELLING_RESULTS.md](qc/AUTO_LABELLING_RESULTS.md)**:
precision is at least 74 % (12/12 random detections were real, 95 % CI), the
habituation period is clean, but **the count is not reliable** — 5 gaps under
5 s look like one delivery split in two, and 9 gaps over 60 s could hide a miss.

It also cannot tell the four stimuli apart, so it does **not** give you the
normalisation denominator. Use it as a list of timestamps to check your manual
count against, not as a count in its own right.

#### 2. `auto_B_features.py` — side-view features, one row per frame

```bash
python auto/auto_B_features.py <side_video_or_folder>
```

Verified on the full WT side session: the mouse was segmented in
**16,824 / 16,824 frames (100 %)**, producing 92 feature columns — shape,
posture, convexity defects, lower-outline roughness, motion, and centred
rolling mean / SD / range over ±0.5 s and ±2 s.

Uses the QC-validated settings: ROI `y 90–370, x 110–600`, threshold `I < 75`.
It does **not** use background subtraction, for the reason in the QC report —
the mouse barely moves, so a temporal-median plate contains the animal.

**Use `--step 1` for real work.** `--step N` analyses every Nth frame; it is
only for quick checks.

#### 3. `auto_B_train.py` — train on your own scoring, then apply

```bash
python auto/auto_B_train.py train --features F1.csv F2.csv --labels L1.csv L2.csv
python auto/auto_B_train.py apply --model behaviour_model.joblib --features Fnew.csv
```

Joins the features to `TrainingLabels_<vid>.csv` on `frame`. Models only the
three behaviours the QC found reachable:

| Code | Behaviour | Key |
|---|---|---|
| 1 | Paw attending | `a` |
| 2 | Licking or biting | `s` |
| 3 | Sustained lifting / guarding | `d` |
| 4 | Escape / rearing | `f` |

Paw withdrawal and flinch are **deliberately not modelled** — at 30 fps a
withdrawal is 1.5–4.5 frames. They stay manual until the frame rate goes up.

Four things it does that matter for getting an honest answer:

- **Trains on `affective_code_ml`**, the lag-corrected label, and drops every
  frame with `exclude_from_training = 1`.
- **Cross-validates grouped by session**, never splitting one. With a single
  session it falls back to a chronological split and says so.
- **Reports bout-level recall and false-positive rate**, not just frame F1.
  Frame F1 is inflated because neighbouring frames are nearly identical; a bout
  you miss entirely is what actually costs you.
- **Smooths predictions before applying the > 2 s guarding rule.** Raw
  classifier output is fragmented, and applying a duration rule to it directly
  deletes whole real bouts. Median filter → bridge short gaps → then enforce
  the duration.

**Watch the row rate.** All duration rules are in seconds, and the script infers
rows-per-second from the `time_s` column rather than assuming 30. If you extract
features with `--step 3` the rows are at 10 Hz, and a hard-coded 30 would scale
the 2 s guarding threshold to 6 s of real time and silently delete every bout.
Train and apply must use the same `--step`.

#### How to actually get there

1. Score all 24 sessions by hand in `score_AB_dual_view.m`, as planned.
2. Run `auto_B_features.py --step 1` on the matching side videos.
3. After about 4–6 scored sessions, run `train` and read the **bout** columns.
   A behaviour is worth automating when bout recall is high and bout FP is low.
4. Validate on held-out sessions you scored but did not train on.
5. Only then consider using it on a new cohort — and never mix predicted and
   manually scored sessions inside one statistical comparison.

---

## Before the implanted cohort — behaviour test run

Naïve WT mice in the BFM room, to validate the rig before any mini1p animal is used:

1. install setup, set the 2 camera angles (stimulus contact + whole-body escape both visible)
2. apply the 4 stimuli, record
3. score reflexive and affective separately with the script above (6 behaviours)
4. confirm morphine produces a detectable analgesic shift — **positive control**, dose still (TBC)
5. then check SBI-553 reproduces its behavioural effect in the same setup

---

## Reuse from project 04 (mini2p)

Project 04 runs the same two-camera acute-pain battery, so most of its downstream analysis
transfers. **Do not copy the .m files** — edit the canonical copy in 04 and call it from here,
or the two will drift (the hub warns about exactly this).

| From project 04 | Use here for | Change needed |
|---|---|---|
| `pain_2plane_step4_behavior_scoring.m` | ancestor of `score_AB_dual_view.m` | already reimplemented — 04 uses Cam1 = stimulus keys, Cam2 = reaction on/off; here both cameras are shown together with a live label track, the stimulus type is recorded, and there are 6 behaviours |
| `pain_2plane_step5_merge.m` | build a `final_neuron_behavior.mat` equivalent | mini1p is single-plane → one dF/F matrix instead of two |
| `downstream_step2_peri_event_traces.m` | ★ peri-stimulus dF/F, F0 = mean of [-2,0] s, outlier-trial rejection | swap the event vector for `deliveryFrameB` |
| `downstream_step3_auc_stats.m` | ★ pre-vs-post AUC, ±2 s and ±1 s windows | none |
| `downstream_step5_full_session_trace.m` | full-session trace with event lines | none |
| `downstream_step6_5min_bins.m` | binned mean dF/F, transient rate | sessions here are short — bin per stimulus instead |
| `downstream_grouped_analysis.m` | split neurons into Increase / Decrease responders | none |
| `downstream_step1_sync_video.m` | 3-view synced clip for figures | slow; optional |

**Key difference to keep in mind:** in 04 the cameras are Cam1 = stimulus, Cam2 = reaction, both
30 Hz, neurons ~4.6 Hz. Here camera A is a **bottom** view and camera B a **side** view — see the
automation note below, because the bottom view changes what can be measured automatically.

**Canonical 04 folder:** `OneDrive\Desktop\cursor\pain_2plane_pipeline\`

---

## Rig QC and what can be automated

Full report: **[qc/WT_RECORDING_QC.md](qc/WT_RECORDING_QC.md)** · figure: `qc/WT_recording_QC.png`
Measured on the two WT recordings in `mousevideo_WT/` (one mouse, bottom + side,
720 × 480, 30 fps, 28 min, 6.07 GB each, DV codec, no interlace combing).

**Verdict: the recordings are usable.** Both views segment the mouse in 60/60
sampled frames. The side view wins on every metric — contrast 2.8 vs 1.2 SD,
solidity 0.82 vs 0.70, and 73 vs 3 motion-energy bouts per 10 min.

| Behaviour | Automatic? | View | Blocked by |
|---|---|---|---|
| **Stimulus delivery** (pass A) | **yes, today** | bottom | nothing — hand + filament at robust z ≈ 21 |
| 1 Paw withdrawal | partial | side + bottom | 30 fps → a withdrawal is 1.5–4.5 frames |
| 2 Flinch / flick | no | — | 30 fps |
| 3 Paw attending | yes, with DLC | side | needs keypoints |
| 4 Licking | marginal | side | tongue ≈ 5 px |
| 5 Biting | marginal | side | separable only by head-jerk kinematics |
| 6 Sustained lifting / guarding (> 2 s) | yes, with DLC | side | — |
| 7 Escape / rearing | yes | side | — |

Two findings that change how the scripts must be written:

- **The bottom view cannot do paw-level scoring.** The mouse images as a fully
  saturated silhouette with no internal detail, and the honeycomb mesh pitch
  (~20 px) is the same size as a paw (~25 px), so paw visibility flickers with
  mesh position rather than with behaviour. Geometry, not processing — no filter
  fixes it. Use the bottom view for delivery timing and body position only.
- **Background subtraction does not work here.** The mouse barely moves over
  28 min (centroid scatter SD 19 × 14 px against a 200–300 px body), so a
  temporal-median plate already contains the animal. Use an absolute threshold
  inside a fixed ROI: bottom `I < 45` full frame; side `I < 75` inside
  x 110–600, y 90–370 (the ROI is required — a dark cloth fold at x < 110
  merges with the mouse and detection collapses to 0 %).

**Fix before the 24 real sessions**, in order of payoff:

1. **Sync LED visible in both cameras**, flashed once at session start. The WT
   files start 1 s apart and differ by 44 frames; the sync-offset dialog field in
   `score_AB_dual_view.m` is carrying this by hand right now.
2. **Zoom the side view in** — the mouse fills only 41 % of frame width. Getting
   to ~70 % doubles paw and snout pixels and is the single best lever for
   telling licking from biting.
3. Raise frame rate to ≥ 120 fps *if* withdrawal latency and flinch counts are
   wanted; otherwise report behaviours 1–2 as present/absent and say so.
4. Stop down ~1 EV (8–11 % of pixels are clipped) and tension the backdrop cloth.
5. Transcode to H.264 for analysis — DV at 6.07 GB × 2 cameras × 24 sessions is
   ≈ 290 GB.

**Plan for this study: score all 24 sessions manually as designed.** Build the
classifier in parallel from the `TrainingLabels_<vid>.csv` that pass B already
writes; the automatic pipeline is for the *next* cohort, validated against these
manual labels. The one thing worth automating now is pass A.

---

## Related

- Project **04 Pain_2plane_imaging** — mini2p version of the same battery; downstream analysis is reusable (table above)
- Project **01 Opioid_Behavior** — `manual_scoring_video_multibehavior_batch2.m` is the ancestor
  of this scoring script (hold-key paradigm, VideoReader loop, raster + .mat output)
