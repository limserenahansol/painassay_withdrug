# mini1p / SBI-553 pain assay — full pipeline

Everything from a raw recording to a Day 1 vs Day 2 figure. Written so someone
else can pick it up.

**Code lives in** `C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\`
(`scoring\` and `analysis\`). This folder is documentation only — the hub
convention.

---

## The session

```
5 min baseline (no stimulus)  |  1 min rest
5 min stimulus block          |  1 min rest      x 4, order from the sheet
```

Confirmed against the real recordings: baseline ≈ 5.3 min, four blocks,
rests 1.2–1.5 min, total ≈ 28 min. Two cameras: **bottom** (sees the
experimenter's hand and the applicator) and **side** (sees the behaviour).

Six mice — 3 male, 3 female. Day 1 no drug, Day 2 drug, same animals.

---

## Step by step

### 1. Score — `scoring\RUN_scoring.m`

Open in MATLAB, press **F5**. Put the two videos in
`videos\cameraA\` (bottom) and `videos\cameraB\` (side) first.

Pick a mode: **1** stimulus only · **2** behaviour only (imports a mode-1
delivery file) · **3** both in one pass.

| Key | Meaning |
|---|---|
| **hold** `a` `s` `d` `f` | attending · licking-or-biting · guarding · escape/rearing |
| **tap** `w` `e` | paw withdrawal · flinch |
| **tap** `1` `2` `3` `4` | stimulus delivery, by type (mode 1 and 3) |
| **tap** `1` `2` | withdrawal · flinch (mode 2 only — the number row is free there) |
| **hold** `u` | uncertain → excluded from classifier training |
| `SPACE` · `←` `→` · `.` `,` · `z` · `q` | pause · frame step · faster/slower · undo · save |
| **mouse drag** on the seek bar | jump anywhere in the session |
| `BACKSPACE` | delete the one mark nearest the cursor |

Both videos are shown side by side with a live label track underneath, so a
stuck key or a missed delivery is visible immediately.

**Fill in Session / Mouse ID / Sex / Day.** Left blank, the treatment cannot be
joined later. All six Day-1 sessions were scored with these empty and had to be
back-filled from the filename plus the randomisation sheet.

**Set "Guarding minimum hold" to 0** if you are tapping rather than holding.

Stopped half way? Run again — it offers to resume and keeps what you scored.

### 2. QC and correct — `analysis\qc_and_correct_all.py`

```bash
python qc_and_correct_all.py
```

Reads `videos\output`, **never modifies it**, writes `videos\output_corrected`.
Read `QC_REPORT.md` in that folder before anything else.

Fixes, all logged line by line in `QC_changes.csv`:

| # | Problem | Action |
|---|---|---|
| 1 | stimulus names mistyped | **per-mouse source.** F1 F2 M2 typed = sheet, no issue. **M1 M3 keep the TYPED names** — the order on the day did not follow the sheet. **F3 uses the sheet** — its typed names had "Mild touch" twice and no "Light touch", so they cannot be right. |
| 2 | two stimuli inside one 5-min block | **the clock decides.** Blocks run 5–10, 11–16, 17–22, 23–28 min, so a mark's block is fixed by its time. Any mark whose code disagrees is a mis-key → re-assigned. |
| 3 | fast repeated clicks | bouts of the same behaviour ≤ 0.35 s apart merged into one event |
| 4 | guarding tap duration | each guarding event given a **nominal 1 s** — a tap records the keypress, not the behaviour. Event **counts are unchanged**. |
| 5 | missing metadata | filled from the filename + the sheet |
| 6 | guarding duration filter | set to 0 — tap scoring, counts are the measure |

**Why the clock and not neighbour voting.** A first attempt decided a mark's
block by the majority code among its neighbours. That cannot resolve a block
boundary: there the marks alternate, no majority exists, and it left 7 cases
unresolved. The clock is unambiguous and fixed all of them — F2 9 marks,
M2 3, M3 6 — leaving every session with exactly 4 clean blocks.

Left alone and **flagged** in `QC_flags.csv`. After the clock rule only 2
remain, and both are informational: M1 and M3 kept their typed names while the
sheet says something different, which is the chosen behaviour.

**A retracted conclusion worth recording.** M2 had 6 marks in 3 s during a rest
period, and an earlier pass called it an unrecoverable key-mash. Checking the
behaviour channel settled it: that window contains **withdrawal ×6 and
flinch ×14**. The mouse responded, so those were real deliveries — three of
them simply carried the wrong key. Always check the behaviour before writing a
delivery off.

### 3. Block measures — `analysis\step1_block_measures.py`

```bash
python step1_block_measures.py "..\videos\output_corrected"
```

Baseline = 0–300 s. Each block = its first delivery + 300 s. Fixed windows, so
every block has the same denominator.

Two denominators, because both vary between blocks:

| suffix | divided by | use for |
|---|---|---|
| `_per_min` | minutes in the window | comparing to baseline, which has no deliveries |
| `_per_delivery` | number of that stimulus given | comparing one stimulus to another |

### 4. Per-stimulus statistics — `analysis\step2_block_stats.py`

```bash
python step2_block_stats.py "..\videos\output_corrected" --ref-stimulus Heat
```

Baseline-subtracted deltas, the SBI-vs-Vehicle difference-in-differences, an
injection control (Vehicle vs None), and a mixed model
`delta ~ treatment*stimulus + block_pos + (1|mouse)`.

The plain `trt:=X` coefficient is the effect of X **at the reference stimulus**,
not an average. Both references are forced and printed.

### 5. Day 1 vs Day 2 — `analysis\step3_day_comparison.py`

```bash
# today, one day only
python step3_day_comparison.py --day1 "..\videos\output_corrected"

# tomorrow
python step3_day_comparison.py --day1 "..\videos\output_corrected" \
                               --day2 "..\videos\output_day2_corrected"
```

Session-level, one number per mouse per session over the whole recording:
`n_events`, `events_per_min`, `pct_of_session`, `total_dur_s`, `median_iei_s`,
plus delivery count and stimulus ITI. This is the *"rearing 10× (20 s/1000 s)
today, 1× (1 s/1000 s) tomorrow"* comparison.

Outputs box plots with a line per mouse joining its two days, paired Wilcoxon
tests, FDR, and a per-stimulus baseline-subtracted panel.

### 6. Slide clip — `analysis\make_ppt_clip.py`

```bash
python make_ppt_clip.py --cut --mouse female3 --stimulus "Pin prick"
# score the cut clip:  scoring\RUN_ppt_scoring.m
python make_ppt_clip.py --render-scored --zoom
```

Read-only with respect to your data. Produces H.264 MP4 (what PowerPoint plays)
plus stills. Marks shorter than 0.25 s are drawn as **dots**, not widened bars,
so nobody reads a bar width as a duration.

### 7. Lab-meeting deck — `analysis\make_lab_meeting_figs.py` + `make_lab_meeting_ppt.py`

```bash
python make_lab_meeting_figs.py           # slide-ready figures
python make_lab_meeting_ppt.py            # 12-slide deck, video embedded
```

Add `--day2 <folder>` to both once Day 2 is scored and the figures/deck rebuild
with two days side by side.

Figures are purpose-built for projection (large fonts, two colours, no
suptitles — the slide carries the message). The deck embeds the 30 s clip
you scored yourself and ends on an explicit limitations slide.

```bash
python make_extra_figs.py                 # the comparisons below
```

| figure | shows |
|---|---|
| `F1_design` | the session timeline |
| `F2_dose_response` | response by stimulus, per minute, baseline-subtracted |
| `F2b_event_counts` | ★ **raw event count in each 5 min block** — blocks are a fixed 300 s so no normalisation is needed |
| `F3_per_delivery` | events per stimulus delivered |
| `F4_baseline_vs_block` | why escape/rearing runs the other way |
| `F5_per_mouse` | all six mice individually |
| `F6_day1_vs_day2_counts` | ★ whole-session counts, Day 1 vs Day 2 |
| `F6b_day1_vs_day2_rate` | the same as events/min |
| `E1_psth` | ★ **peri-stimulus time histogram** — every delivery aligned at t = 0 |
| `E2_response_probability` | ★ **P(response within 3 s) per delivery** |
| `E3_within_block_timecourse` | habituation within the 5 min block, 1 min bins |
| `E4_block_position` | control: response by block position, not stimulus |
| `E5_reflex_vs_affective` | do the two classes dissociate? one point per mouse |

### What the extra figures add

**`E1` peri-stimulus histogram.** Withdrawal and flinch spike exactly at
contact; licking and guarding build over 1—8 s. The two classes separate in
time, in this data. It is also the same alignment as peri-event dF/F, so
behaviour and imaging can be put side by side.

*One artefact to know about:* the affective traces drop to exactly zero at
t = 0. That is the scorer — tapping a delivery key means letting go of the
held behaviour key. The figure says so on its face.

**`E2` response probability per delivery.** One Bernoulli trial per delivery
instead of one rate per block: **3,204 trials against 24 block means.** That is
where the statistical power is at n = 6.

| P(response within 3 s) | Light touch | Mild touch | Heat | Pin prick |
|---|---|---|---|---|
| paw withdrawal | 0.744 | 0.905 | **1.000** | 0.931 |
| flinch | 0.230 | 0.494 | **0.773** | 0.537 |
| licking / biting | 0.023 | 0.150 | **0.264** | 0.222 |
| guarding | 0.017 | 0.137 | 0.160 | **0.207** |

Heat evokes a withdrawal on essentially every delivery.

---

## Day 1 QC outcome (2026-09-02, six sessions)

| mouse | session | deliveries | runs before → after | mis-keys fixed | name source |
|---|---|---|---|---|---|
| F1 | 1 | 87 | 4 → 4 | 0 | sheet (= typed) |
| F2 | 2 | 88 | 16 → **4** | 9 | sheet (= typed) |
| F3 | 3 | 85 | 4 → 4 | 0 | **sheet** |
| M1 | 4 | 100 | 4 → 4 | 0 | **typed** |
| M2 | 5 | 83 | 10 → **4** | 3 | sheet (= typed) |
| M3 | 6 | 91 | 10 → **4** | 6 | **typed** |

93 changes, **2 flags** (both informational). **Every session now has exactly
four clean blocks.** Block order per mouse:

| mouse | block 1 | block 2 | block 3 | block 4 |
|---|---|---|---|---|
| F1 | Mild touch | Pin prick | Light touch | Heat |
| F2 | Heat | Mild touch | Pin prick | Light touch |
| F3 | Heat | Light touch | Pin prick | Mild touch |
| M1 | Mild touch | Light touch | Heat | Pin prick |
| M2 | Pin prick | Mild touch | Heat | Light touch |
| M3 | Pin prick | Mild touch | Light touch | Heat |

### The Day 1 result

Per-stimulus event rate, baseline-subtracted, median over the six mice:

| behaviour | Light touch | Mild touch | Heat | Pin prick |
|---|---|---|---|---|
| paw withdrawal | +3.35 | +5.10 | **+5.70** | +5.00 |
| flinch | +1.58 | +2.80 | **+4.57** | +2.80 |
| paw attending | +0.80 | +1.00 | +1.50 | **+1.60** |
| licking / biting | +0.52 | +0.70 | **+2.64** | +1.10 |
| guarding | +0.31 | +0.80 | **+1.90** | +1.20 |
| escape / rearing | −2.20 | −2.10 | **+0.67** | −0.20 |

**The assay separates the stimuli.** Light touch gives the smallest response
and Heat the largest, for every behaviour. Pin prick sits between Mild touch
and Heat — so the ordering is not strictly monotonic with nominal intensity,
and a "dose-response" claim would overstate it.

**Escape / rearing is not a pain measure here.** It is the only behaviour with
a substantial baseline (3.8 events/min) and the only one that goes DOWN during
stimulation. That reads as exploration being suppressed. Report it separately
and do not pool it with the affective measures.

Session medians, Day 1:

| behaviour | events | events/min | median interval |
|---|---|---|---|
| withdrawal | 97.5 | 3.46 | 2.1 s |
| flinch | 59.0 | 2.09 | 1.4 s |
| escape | 78.0 | 2.79 | 9.5 s |
| attending | 25.5 | 0.91 | 13.7 s |
| guarding | 24.0 | 0.85 | 4.8 s |
| licking/biting | 21.0 | 0.74 | 13.6 s |

`guarding` is 1.03 % of the session after the nominal 1 s per event.

Deliveries 83–100 per session (mean 89), stimulus ITI median 2.0–4.8 s,
sessions 1680–1722 s.

---

## Read this before reporting anything

**Counts and rates are the measures.** The sessions were scored by tapping, so
a bout's recorded length is the length of the keypress, not of the behaviour.
`step3` plots durations in grey and labels them UNRELIABLE.

**The one exception is guarding**, which now carries a **nominal 1 s per
event** because a guarding tap means guarding was seen and it lasts about a
second. Note what that does and does not buy you: `% of time` for guarding is
now `count × 1 s ÷ session length`, i.e. a rescaled count. It is fine to plot
on a "% of time" axis, but it carries no information the count did not, and it
should be described as nominal in the methods.

**The `> 2 s` guarding criterion is now the scorer's judgement**, not a rule the
code applies, because a duration filter would discard almost every tapped mark.
State that in the methods.

**Paired is primary, but the group test has more resolution at n = 6.** The
same six mice are used on both days, so the paired test is the correct design.
Note though that the exact paired test is floored at p = 0.031 while
Mann-Whitney on 6 vs 6 can reach 0.0022 — so a large, consistent effect will
sit at the paired floor while the group test reports something smaller. Both
are printed side by side in `BlockCounts_stats.csv` and
`SessionComparison_paired_and_group.csv`. Report the paired test; do not switch
to the group test because its p is smaller.

**Statistical floor.** Exact Wilcoxon at n = 6 cannot return a two-tailed p
below **2/2⁶ = 0.031**. One pre-specified comparison can reach 0.05; a table of
6 behaviours × 5 measures cannot. Pick the primary outcome before looking.

**Validation.** The two-day path was tested against a synthetic Day 2 with a
known effect (escape events cut 70 %). It recovered exactly that — escape
events 78 → 25.5, p = 0.031, q = 0.031 — and reported no change in the other
five behaviours. `step1`/`step2` were validated the same way on 24 synthetic
sessions; see `Documents\HEAL_mini1p_SBI553\analysis\README.md`.

---

## Bugs found and fixed along the way

Worth knowing about, because two of them silently corrupted data.

| Bug | Effect | Fix |
|---|---|---|
| `KeyRelease` fired during Windows key auto-repeat | one held key became a train of 2–3 frame fragments; durations destroyed, counts inflated up to 2.5× | wall-clock debounce, 0.25 s |
| resume overwrote earlier work with zeros | restarting mid-session lost everything before the restart point | offers to resume, and `nUsed` can never shrink |
| lost mouse button-up left the motion callback attached | every stray mouse move triggered a ~2 s cold seek — the window froze permanently | `gDragging` flag, 3 s watchdog, and the seek is only committed on release |
| `stimNames` saved as a column cell | mode 2 crashed on `horzcat` | shape forced to a row at all four assignment points |
| ffmpeg given an odd frame width | 0-byte MP4, silently | pad to even, and ffmpeg's stderr is now shown |
| pandas read `"None"` as NaN | the no-injection sessions vanished from the treatment join | `keep_default_na=False` |

A save-time check now warns if the affective keys look tapped rather than held.

---

## Rig changes worth making

From `qc\WT_RECORDING_QC.md` and `qc\DELIVERY_DETECTOR_VALIDATED.md`:

1. **A TTL or LED on the applicator.** One piece of hardware solves camera
   sync, delivery timing and stimulus identity together. Image analysis cannot
   recover stimulus identity — the four applicators look identical from below,
   and detection rate ranged 45–85 % depending on which one was used.
2. **Zoom the side view in.** The mouse fills 41 % of frame width; ~70 % would
   roughly double paw and snout pixels.
3. **A second 5 min baseline at the end of the session.** Without it, drift over
   28 min is confounded with the baseline-vs-stimulus comparison, and
   randomising stimulus order does not help because baseline is always first.
4. **Draw the ITI from a pre-generated list.** Delivery counts varied 83–100. If
   the interval is paced by eye it correlates with the animal's behaviour, which
   is a confound rather than noise.

---

## Automatic labelling — where it stands

`analysis\` has `auto_A_delivery_detect.py`, `auto_B_features.py`,
`auto_B_train.py`. Current honest status is in
`qc\AUTO_LABELLING_RESULTS.md` and `qc\DELIVERY_DETECTOR_VALIDATED.md`:

- **delivery detection: not usable.** 64 % of manual marks matched, and coverage
  ranged 45–85 % by stimulus, which would bias per-stimulus comparisons.
- **rearing: detection good, duration not calibrated.** d′ = 6.0 between
  upright and crouched, 12/12 random detections correct, but the bout count
  swings 2.8× across a narrow threshold range.
- **the rest: not enough labelled events** to train on (attending 28,
  guarding 22, lick/bite 63 across two mice).

Score manually; treat the automatic pipeline as work for the next cohort.

---

## File map

| Path | What |
|---|---|
| `scoring\RUN_scoring.m` | ★ open, F5 — the scorer |
| `scoring\score_AB_dual_view.m` | the scoring engine, both cameras + live track |
| `scoring\RUN_ppt_scoring.m` | score a short clip for a slide |
| `analysis\qc_and_correct_all.py` | ★ audit + corrected copy |
| `analysis\step1_block_measures.py` | per-block measures |
| `analysis\step2_block_stats.py` | per-stimulus stats, DiD, mixed model |
| `analysis\step3_day_comparison.py` | ★ Day 1 vs Day 2 box plots, now incl. `n_per_delivery` |
| `analysis\step4_individual_stats.py` | ★ population **and** per-mouse statistics, forest plot |
| `analysis\make_day2_figs.py` | ★ the Day 1 vs Day 2 figure set (D1–D7); `YLAB` at the top is the single place the axis label is defined |
| `analysis\make_synthetic_day2.py` | a fake Day 2, for deck planning and for validating the analysis |
| `analysis\make_full_ppt.py` | ★ the complete deck: method + Day 1 + Day 2 plan |
| `analysis\make_ppt_clip.py` | slide clip |
| `analysis\repair_merge_fast_clicks.py` | earlier, narrower repair — superseded by `qc_and_correct_all.py` |
| `videos\output\` | your scoring, never modified |
| `videos\output_corrected\` | corrected copy + `QC_REPORT.md` — use this downstream |
| `qc\` (this folder) | recording QC and automatic-labelling assessments |

---

## Normalisation: total event number / total stimulus delivery

That phrase is the axis label used throughout, spelled out as a ratio on
purpose. Two earlier wordings were rejected:

- **"events per stimulus"** — ambiguous. The x-axis of most panels *is* the
  four stimulus **types**, so "per stimulus" reads as "per stimulus type"
  rather than per individual delivery.
- **"normalized events"** — hides the denominator. The pipeline contains three
  different normalisations (per minute, delta against baseline, per delivery)
  and a label that does not name the divisor cannot distinguish them.


The experimenter delivers the stimulus by hand, so **the number of taps is not
fixed**. On Day 1 alone, across six mice:

| Stimulus | fewest taps | most taps | spread |
|---|---|---|---|
| Light touch | 17 | 30 | 1.76x |
| Mild touch | 19 | 26 | 1.37x |
| Heat | 15 | 23 | 1.53x |
| Pin prick | 16 | 31 | 1.94x |

A raw event count is therefore **not comparable between animals, and will not
be comparable between days**. Ten flinches from ten taps and ten flinches from
twenty taps are the same raw count and a different animal:

    mouse A   10 taps, 10 flinches  ->  1.00 per delivery
    mouse B   20 taps, 10 flinches  ->  0.50 per delivery

Note that `events_per_min` does **not** fix this. It divides by time, and the
session length is fixed by the protocol while the tap count is not. The block
being a fixed 300 s is irrelevant to this confound - the denominator that
varies is the number of stimuli inside the block, not its duration.

So the measures are:

| Measure | Axis label | Denominator | Use it for |
|---|---|---|---|
| `n_events` | total event number | none | never on its own; shown only beside `n_del` |
| `events_per_min` | events / min | session or block minutes | comparing a block with baseline, where there are no deliveries at all |
| **`n_per_delivery`** | **total event number / total stimulus delivery** | that mouse's own delivery count | **everything else - between mice, between stimuli, between days** |

`step1_block_measures.py` and `step3_day_comparison.py` both emit
`n_per_delivery`, and `analysis\make_day2_figs.py` plots nothing else.
`D6_why_normalise.png` shows the confound and its removal side by side.

### Count, not yes/no

Within a response window there are two ways to score a behaviour:

- **hit** - did anything happen, 0 or 1
- **rate** - how many bouts happened

The rate is the primary. The hit rate **saturates**: if a window usually holds
two or three bouts, removing half of them barely moves the probability. This
was measured, not assumed. Injecting a known 40 % reduction into synthetic
data:

| Readout | True effect | Recovered |
|---|---|---|
| total event number / total stimulus delivery | 0.60x | **0.64-0.68x** |
| any response, yes/no | 0.60x | 0.82x (badly underestimated) |

Per-mouse significance followed the same pattern: 9 of 36 mouse x behaviour
cells by rate, only 4 by hit rate.

---

## How long is a stimulus block? Two definitions, both computed

Behaviour does not stop when the stimulus does, so the 1 min rest that follows
each block was tested both ways:

| `--block-s` | Window | Meaning |
|---|---|---|
| 300 | first delivery + 5 min | the stimulus period proper |
| 360 | first delivery + 6 min | the stimulus period **plus** the rest that follows it |

Attributing the rest minute to the block before it is the defensible reading —
nothing else caused it, and the next block has not started. Windows are clipped
so they never reach the next block's first delivery.

**What is and is not comparable between the two:**

| Measure | Comparable? | Why |
|---|---|---|
| `n_per_delivery` | **yes** | the rest minute contains no deliveries, so the denominator is identical and any rise is real extra behaviour |
| `n_bouts` | yes, within a version | the difference between versions *is* the rest-minute contribution |
| `rate_per_min` | **no** | the divisor changes 5 → 6 min, so it falls ~17 % mechanically |

### Result on Day 1: the rest minute adds almost nothing

| Behaviour | 5 min | 6 min | change | extra events per block |
|---|---|---|---|---|
| Paw withdrawal | 1.133 | 1.135 | +0.1 % | 0.04 |
| Flinch | 0.830 | 0.830 | +0.0 % | 0.00 |
| Paw attending | 0.306 | 0.308 | +0.6 % | 0.04 |
| Licking / biting | 0.377 | 0.386 | +2.2 % | 0.12 |
| Guarding | 0.238 | 0.241 | +1.1 % | 0.08 |
| **Escape / rearing** | 0.710 | 0.746 | **+5.0 %** | 0.79 |

Pain behaviours are flat. Escape/rearing rises in **every** mouse — consistent
with rearing being exploration that resumes once the stimulus stops, which is
the same conclusion the baseline-versus-block comparison reached independently.

**Why so little?** Because the 5 min window *already* contains the quiet tail.
The window starts at the block's first delivery, but the experimenter finishes
delivering before the 300 s mark: the first-to-last delivery span is 139–299 s,
median 256 s. So the 5 min version already includes a mean of **45 s** of
post-stimulus time, and the extra minute lands further out where behaviour has
already subsided.

This is a Day 1 result. If ongoing behaviour appears during rest on the drug
day, the 6 min version is where it will show — which is why both are kept.

Scripts: `step1_block_measures.py --block-s 300|360`, then
`compare_block_windows.py` for `W1`/`W2` and the statistics.

---

## Statistics at two levels

With six animals the population test has a **hard arithmetic floor**. The
exact paired Wilcoxon cannot return a two-tailed p below `2/2^n`:

| Usable pairs | Smallest possible p |
|---|---|
| 6 | 0.031 |
| 5 | 0.062 |
| 4 | 0.125 |

Reaching 0.031 requires **all six** animals to move the same way. One
non-responder is dropped by the test, n falls to 5, and `p < 0.05` becomes
*arithmetically impossible* regardless of effect size. This is not
hypothetical - it is what the synthetic Day 2 produced, and it is why
`D7_power_planning.png` exists.

**The per-delivery data rescues this.** Each delivery is a Bernoulli trial and
each mouse receives roughly 90 of them per day, so a **within-mouse** test is
possible and properly powered:

| Level | Unit | n | Test | Limit |
|---|---|---|---|---|
| Population | the mouse | 6 | paired Wilcoxon (**primary**) | floors at 0.031, or 0.062 with one non-responder |
| Population | the mouse | 6 vs 6 | Mann-Whitney | reaches 0.0022, but discards the pairing the design bought |
| **Individual** | one delivery | ~90 per mouse per day | Fisher exact, rate ratio | well powered for reflexes; only large shifts for affective |
| Direction | the mouse | 6 | how many moved the same way | not a p-value, but survives the floor |

Report the paired test as primary even when it floors, state the floor beside
it, and use the direction count and the per-mouse forest plot to carry the
information the floor cannot express.

`step4_individual_stats.py` produces all of this, including a
`min_detectable_delta` column per mouse - the smallest change that animal's
trial count could have detected at 80 % power. **A non-significant per-mouse
result with `|delta| < min_detectable_delta` means underpowered, not no
effect.** The column exists so that distinction cannot be skipped.

### Power, from the real Day-1 spread

Simulated with the measured between-mouse variability (median CV 0.49),
paired Wilcoxon, alpha 0.05:

| True effect | n = 6, all respond | n = 6, one non-responder |
|---|---|---|
| 20 % | 0.37 | 0.21 |
| 30 % | 0.78 | 0.49 |
| 40 % | 0.98 | 0.49 |
| 60 % | 1.00 | 0.49 |

**n = 8-10 is the robust range.** At n = 6 the design only works if every
animal responds.

---

## Day 2: run order

Once Day 2 is scored into `videos\output_day2\`:

    cd analysis

    python qc_and_correct_all.py --src ..\videos\output_day2 ^
                                 --out ..\videos\output_day2_corrected

    python step1_block_measures.py ..\videos\output_day2_corrected ^
                                   --out ..\videos\output_day2_corrected

    python step3_day_comparison.py --day1 ..\videos\output_corrected ^
                                   --day2 ..\videos\output_day2_corrected ^
                                   --out ..\lab_meeting\figs_day2

    python step4_individual_stats.py --day1 ..\videos\output_corrected ^
                                     --day2 ..\videos\output_day2_corrected ^
                                     --no-isolation ^
                                     --out ..\lab_meeting\figs_day2

    python make_day2_figs.py --day1 ..\videos\output_corrected ^
                             --day2 ..\videos\output_day2_corrected ^
                             --out ..\lab_meeting\figs_day2

    python make_full_ppt.py --day2-figs ..\lab_meeting\figs_day2 --real

Note `--no-isolation` on step4. The isolation filter keeps only deliveries with
10 s of clearance, which is 122 of 535 - fine for the population test, too thin
for a per-mouse one. Dropping it is valid **for the day comparison
specifically**, because the mis-attribution inflates both days equally (the
pacing is set by the protocol, not by the drug). It is *not* valid for
comparing one stimulus against another.

### The synthetic Day 2

`make_synthetic_day2.py` builds a fake Day 2 for two purposes: settling the
deck layout before the experiment, and validating that the analysis recovers a
known effect. It injects a 40 % affective reduction, leaves escape/rearing
alone, drops 15 % of reflex events, makes one mouse a deliberate
non-responder, and jitters the delivery count so the two days differ in tap
number.

**Every downstream script must then be run with `--mockup`**, which stamps
MOCKUP across each panel and appends `_MOCKUP` to each filename. Do not remove
that flag. Nothing synthetic is ever merged with real output; the real Day 2
replaces the folder wholesale.

Validation outcome: escape correctly unchanged (1.00x, no false positive), the
non-responder correctly flat in every behaviour, and the affective reduction
recovered at 0.64-0.68x against a true 0.60x.

---

## Bug log, additions

| Symptom | Cause | Fix |
|---|---|---|
| `block_counts()` docstring claimed a fixed 300 s block made raw counts comparable | confused block *duration* with stimulus *count* - the duration is fixed, the tap count is not | rewrote it; `block_counts()` now returns and tests `n_per_del` alongside `n_bouts` and `n_del` |
| `step3` had no stimulus-normalised measure at all | `n_per_delivery` existed in `step1` but was never carried into the day comparison | added to both the affective and reflexive branches |
| binary response probability underestimated a known 40 % effect as 18 % | a yes/no hit saturates once a window holds several bouts | `n_responses` counts bouts; the rate is now primary and the hit rate is the sanity check |
| forest-plot x-axis was an unreadable smear of `6x10^-1` labels | default matplotlib log ticks over a narrow range | fixed halving/doubling ladder with plain labels |
| significance marks collided with panel titles | placed at data height | headroom set first, then marks drawn in axes coordinates |
| `step4` non-ASCII corrupted mid-edit (a middle dot became two bytes) | a `Get-Content` / `Set-Content` round trip in PowerShell 5.1 | repaired; edit source files with the Edit tool, never through a PowerShell text round trip |
| Markdown files *looked* corrupted in the terminal | PowerShell 5.1 `Get-Content` decodes UTF-8 as ANSI on display only - the files were always fine | verify with Python before believing the terminal |
