# Delivery detector, validated against real manual scoring

Date: 2026-09-02 · session `female1` (mode-1 stimulus pass, 87 manual marks)

This supersedes the delivery-detector section of
[AUTO_LABELLING_RESULTS.md](AUTO_LABELLING_RESULTS.md), which was written
before any manual ground truth existed.

---

## Verdict: not usable as a substitute for manual marking

| | |
|---|---|
| Manual marks | **87** |
| Auto episodes | **40** |
| Manual marks inside an auto episode (±1 s) | **56 / 87 = 64 %** |
| Auto episodes with no manual mark | **10 / 40** |

---

## 1. The misses are real, not a timing offset

Widening the tolerance recovers marks only slowly:

| tolerance | coverage |
|---|---|
| ±0 s | 47 % |
| ±1 s | 64 % |
| ±5 s | 82 % |
| ±20 s | 97 % |

Median distance from a mark to the nearest episode is 0.2 s, but p90 is
**10.4 s** and the max is 28.3 s. So most marks sit right on an episode and a
long tail sits nowhere near one. Reaching high coverage needs a ±20 s window,
which covers almost any time point and therefore means nothing. **These are
genuine detection failures.**

## 2. The failure is stimulus-specific — and that is the killer

| Stimulus | n marks | covered | rate |
|---|---|---|---|
| Mild touch | 26 | 22 | **85 %** |
| Pin prick | 16 | 11 | 69 % |
| heat | 23 | 13 | 57 % |
| Light | 22 | 10 | **45 %** |

Coverage ranges from 85 % down to 45 % depending on which applicator is used.
This is an **optics problem, not a threshold-tuning problem**: the detector
works by measuring how much light the intruding object blocks, and a thin
light-touch filament blocks very little. No parameter choice fixes that, and
a stimulus-dependent detection rate would bias any per-stimulus comparison in
exactly the direction that matters.

## 3. Correcting an earlier claim

An earlier note said the detector had "precision ≥ 74 % (12/12 random
detections were real)". That visual check asked **whether a hand was visible**,
not whether a delivery happened. The hand is also in frame while repositioning
between touches, so "hand present" is a *superset* of "delivery". Both results
are consistent — 12/12 did show a hand — but the wording overstated what had
been established. As a **delivery** detector, 10 of 40 episodes contain no
manual mark at all.

The earlier reading of "40 candidates ≈ 4 stimuli × 10 repeats" was also wrong:
the real count is 87 individual touches.

---

## 4. Good news: the design is BLOCKED

| Stimulus | marks | time span | n |
|---|---|---|---|
| Mild touch | 1–26 | 319.8 – 566.6 s | 26 |
| Pin prick | 27–42 | 640.6 – 939.1 s | 16 |
| Light | 43–64 | 1030.5 – 1307.0 s | 22 |
| heat | 65–87 | 1389.5 – 1646.8 s | 23 |

Exactly four runs — all of one stimulus, then all of the next.

**This makes post-hoc labelling far safer than previously warned.** The earlier
caution — "one missed delivery shifts every subsequent label" — applies to an
*interleaved* design. Here you only have to get **3 block boundaries** right. A
missed or duplicated detection inside a block changes that stimulus's count
(which is the denominator, so it still matters) but **mislabels nothing**.

### But do not detect the blocks by gap size

| boundary | gap |
|---|---|
| Mild touch → Pin prick | 74.0 s |
| Pin prick → Light | 91.4 s |
| Light → heat | 82.5 s |
| **largest gap WITHIN a block** | **82.1 s** |

The largest within-block gap (82.1 s) is bigger than the smallest between-block
gap (74.0 s). A gap-threshold rule would split blocks in the wrong place. The
boundaries have to come from the written record or from looking at the video.

---

## 5. What to do

**Short term — keep marking deliveries by hand.** At 45–85 % stimulus-dependent
coverage the detector cannot replace it, and it is too noisy to be a useful QC
flag either. The blocked design plus your written record is more reliable than
the detector.

**The real fix is hardware, not software.** Put a TTL pulse or an LED on the
stimulus applicator so delivery time is recorded electronically. That gives:

- frame-accurate onset with no human reaction time,
- 100 % detection regardless of how much light the applicator blocks,
- and, with one channel or LED colour per applicator, the **stimulus identity
  for free** — which is the one thing image analysis fundamentally cannot
  recover, since the four applicators look the same from below.

This is the same LED that the sync recommendation in
[WT_RECORDING_QC.md](WT_RECORDING_QC.md) already calls for. One piece of
hardware solves camera sync, delivery timing, and stimulus identity together.
