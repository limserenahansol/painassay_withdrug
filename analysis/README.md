# Downstream analysis — block design

Two scripts, run in order, on the `ScoringAB_*.mat` files that
`score_AB_dual_view.m` writes.

```bash
python step1_block_measures.py "..\videos\output"
python step2_block_stats.py    "..\videos\output" --ref-stimulus Heat
```

---

## The design these assume

```
5 min baseline (no stimulus)   |  1 min rest
5 min stimulus block           |  1 min rest      x 4, order randomised
```

Confirmed against the real `female1` session: baseline 5.33 min, four blocks,
rests 1.23–1.52 min.

**Period definition** (chosen 2026-09-02): baseline = 0–300 s; each block =
that block's first delivery + 300 s. Fixed windows so every block has the same
denominator.

---

## Why the block, not the delivery, is the unit

In `female1`, **86 % of inter-stimulus intervals were under 30 s** (median
4.7 s, minimum 0.9 s). A per-delivery observation window is therefore truncated
almost every time and its length varies wildly. `Normalized_<vid>.csv` from the
scorer is kept as a secondary view only.

## Two denominators, because both vary

| measure | divided by | answers |
|---|---|---|
| `rate_per_min` | minutes in the window | "how much more than baseline" — the only thing comparable to baseline, which has no deliveries |
| `n_per_delivery` | number of that stimulus delivered | "which stimulus evokes more" — needed because delivery counts differed by 46 % in `female1` (16 vs 26) |

---

## Validation against known ground truth

24 synthetic sessions were generated from the real randomisation sheet with one
effect injected: **SBI-553 halves lick/bite during the Heat block only.**

**The pipeline recovers it accurately.**

| | value |
|---|---|
| realised DiD, mean over 6 mice | **−0.567** /min |
| realised DiD, median | **−0.700** /min |
| step2 per-cell median (lickbite / Heat) | **−0.700** ✓ |
| step2 mixed-model coefficient | **−0.575** ✓ |

**But nothing reaches significance, and that is correct.**

| | |
|---|---|
| Cohen's dz | 0.46 |
| power at n = 6 | **0.15** |
| n for 80 % power | ~40 mice per crossover |
| per-cell Wilcoxon p (lickbite / Heat) | 0.438 |
| mixed-model p (SBI vs Vehicle at Heat) | 0.236 |
| every FDR q | > 0.64 |

The top raw-p hits — `flinch / Light touch` p = 0.033, `escape / Mild touch`
p = 0.040 — are **false positives**; nothing was injected there. FDR correctly
kills them.

So: the machinery estimates the right effect size and correctly declines to
call it significant. The limit is 6 mice, not the code.

*(Caveat: the between-mouse variance above comes from a synthetic noise model.
The real variance is unknown until the sessions are scored. Treat dz = 0.46 and
"n ≈ 40" as illustrative of the machinery, not as a power calculation for this
study.)*

---

## Reading the mixed model — the easy mistake

`delta ~ C(treatment)*C(stimulus) + pos + (1|mouse)`

A plain `trt:=SBI-553` term is the effect of SBI-553 **at the reference
stimulus**, not an average over stimuli. Both references are forced and printed
so this cannot be misread:

```
reference levels: stimulus = 'Heat', treatment = 'Vehicle'
```

With `--ref-stimulus Heat --ref-treatment Vehicle`, `trt:Vehicle=SBI-553` is
exactly the SBI-vs-Vehicle contrast inside the Heat block. Change the reference
and the same coefficient means something else. (During development this was
mis-read once: a p = 0.024 that looked like the drug effect was in fact the
SBI-vs-**None** contrast, a different comparison.)

---

## Statistical floors to state in the paper

- Exact Wilcoxon signed-rank at n = 6 cannot return a two-tailed p below
  **2/2⁶ = 0.031**. A single pre-specified test can reach 0.05; **24 stimulus ×
  behaviour cells cannot all survive FDR by construction.**
- **Pick the primary outcome before looking at the table.** Everything else is
  exploratory and should be labelled as such.

## One way to gain power without more mice

For the two **reflexive** measures the block mean throws information away. Each
block contains 16–26 deliveries, each a Bernoulli trial ("did the paw
withdraw?"). Modelling at the delivery level —

```
withdrawal ~ treatment * stimulus + (1 | mouse) + (1 | session)   [binomial]
```

— uses ~20 trials per block instead of one rate, which is a large gain in
information for the same animals. `RawScores_<vid>.csv` already has one row per
delivery with `Withdrawal (0/1)`, so the data is there. Not yet implemented —
say the word.

Duration measures do not decompose this cleanly, and note that behaviours
compete for the same timeline, so rate and %-time measures are compositional
and saturate when the animal is busy.

---

## Design issues worth fixing before running all 6 mice

1. **The baseline is only at the start of the session.** Any drift over 28 min
   (settling, warming, habituation) is confounded with the baseline-vs-stimulus
   comparison, and randomising stimulus order does not help because baseline is
   always position 0. **Adding a second 5 min baseline at the end** costs 5 min
   and makes drift estimable. Without it you cannot separate "Heat increased
   licking" from "licking increased over the session".
2. **Only 1 min rest between blocks.** Guarding and licking can outlast a
   minute, so block *N* is contaminated by block *N−1*. `pos` (block position)
   is in the model as a partial control, but a longer rest would be better.
3. **Delivery counts varied 46 %** in `female1` (16–26). If the ITI is paced by
   eye — waiting for the mouse to settle — then delivery count correlates with
   the animal's behaviour, which is a feedback confound rather than noise.
   Drawing ITIs from a pre-generated list removes it.
