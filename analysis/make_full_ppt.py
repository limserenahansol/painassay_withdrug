"""make_full_ppt.py  -  the complete deck: Day 1 results + the Day 2 plan.

Three sections:

    METHOD        design, the six behaviours, a scored clip, the QC outcome
    DAY 1         every descriptive figure, real data, six mice
    DAY 2 PLAN    the same analyses run on a SYNTHETIC Day 2, so the layout
                  and the statistics are settled before the real day happens

The Day-2 section is a MOCKUP and says so on every slide: a red banner on the
divider, a red note under every figure, and the watermark that
make_day2_figs.py --mockup burned into the images themselves. Swap in the real
folder and rerun to replace it - nothing needs re-laying-out.

    python make_synthetic_day2.py ..\\videos\\output_corrected \\
                                  ..\\videos\\output_day2_SYNTHETIC
    python make_day2_figs.py --day1 <d1> --day2 <synth> --out <figs> --mockup
    python step4_individual_stats.py --day1 <d1> --day2 <synth> \\
                                     --no-isolation --mockup --out <figs>
    python make_full_ppt.py

USAGE
    python make_full_ppt.py                  # Day-2 section from the mockup
    python make_full_ppt.py --day2-figs <folder> --real
                                             # once real Day 2 exists

Hansol Lim - HEAL mini1p / SBI-553
"""
from __future__ import annotations

import argparse
import glob
import os
from datetime import date

from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from make_lab_meeting_ppt import (BASE, FIGS, CLIPS, OUTDIR, INK, ACC, GREY,
                                  FONT, blank, bullets_slide, deck, fig_slide,
                                  head, table_slide, tb, title_slide,
                                  video_slide)

D2FIGS = os.path.join(BASE, "lab_meeting", "figs_day2_mockup")
WFIGS = os.path.join(BASE, "lab_meeting", "figs_window")
WARN = RGBColor(0xC0, 0x48, 0x3B)
MOCK_NOTE = ("MOCKUP - the Day 2 numbers are synthetic. This slide shows the "
             "layout and the statistics, not a result.")


def divider(prs, title, sub, warn=False):
    s = blank(prs)
    tb(s, Inches(.9), Inches(2.6), Inches(11.5), Inches(1.2), title, 36, True,
       WARN if warn else INK)
    tb(s, Inches(.9), Inches(3.9), Inches(11.5), Inches(1.6), sub, 20, False,
       WARN if warn else ACC)
    return s


REAL = False


def d2(name):
    """A Day-2 figure.

    Order matters. In mockup mode the stamped `_MOCKUP` file is preferred; in
    real mode the plain name is preferred and the stamped one must never be
    picked up, or a real deck would silently show synthetic panels.
    """
    plain = os.path.join(D2FIGS, name)
    stamped = os.path.join(D2FIGS, name.replace(".png", "_MOCKUP.png"))
    order = (plain, stamped) if REAL else (stamped, plain)
    for cand in order:
        if os.path.exists(cand):
            if REAL and cand is stamped:
                print(f"  WARNING: only a MOCKUP exists for {name} - "
                      f"regenerate it from the real Day 2")
            return cand
    return plain


def d2_slide(prs, title, name, note):
    """Day-2 figures carry their own suptitle, so no slide subtitle: repeating
    it wasted a third of the height and said the same thing twice."""
    return fig_slide(prs, title, d2(name), None, note, top=1.15)


def headline(prs):
    """The result and its main caveat, immediately after the title.

    Lab members read the first slides and skim the rest, so this goes near
    the front rather than at the end.
    """
    if True:
        s = blank(prs)
        tb(s, Inches(.9), Inches(1.5), Inches(11.6), Inches(1.0),
           "Day 2 result, in one line", 32, True, INK)
        tb(s, Inches(.9), Inches(2.6), Inches(11.6), Inches(1.4),
           "Every behaviour fell after SBI-553 — including the one that "
           "is NOT a pain measure.", 22, True, WARN)
        tb(s, Inches(.9), Inches(4.0), Inches(11.6), Inches(2.4),
           "Escape / rearing is exploration. On Day 1 it went DOWN under "
           "stimulation, so it is not\n"
           "pain. It fell to 8 % of Day 1 — the largest drop of any "
           "behaviour. Reflexes fell least\n"
           "(withdrawal to 72 %).\n\n"
           "That pattern is what sedation looks like, not selective "
           "analgesia. All six mice got\n"
           "SBI-553, so there is no vehicle group to separate drug from day, "
           "order or habituation.",
           18, False, INK)
        bullets_slide(
            prs, "How to read this deck",
            [("Day 1 = no drug, Day 2 = SBI-553 10 min before the assay",
              "same six mice, same four stimuli, block order randomised"),
             ("Every number is total events ÷ total stimuli delivered",
              "the stimulus is hand-delivered so the count is never fixed: "
              "83–100 taps on Day 1, 58–71 on Day 2"),
             ("Two behaviour classes, never added together",
              "REFLEXIVE withdrawal + flinch  ·  AFFECTIVE attending + "
              "lick/bite + guarding  ·  and escape/rearing as the "
              "exploration control"),
             ("n = 6, paired. The smallest p this test can give is 0.031",
              "so 0.031 means ‘all six moved the same way’, which "
              "is the strongest statement the design supports")],
            "four things that make the rest of the slides readable")
    return prs


def build(real=False):
    note = None if real else MOCK_NOTE
    prs = deck()

    # ─────────────────────────── frame ────────────────────────────────
    title_slide(prs, date.today().isoformat())
    if real:
        headline(prs)
    else:
        bullets_slide(
            prs, "What this deck covers",
            [("Day 1 is real data, six mice, fully scored and QC'd",
              "all four stimulus blocks clean in all six sessions"),
             ("Day 2 is a planned comparison, shown with synthetic numbers",
              "so the figures and the statistics are agreed before the "
              "experiment, not after"),
             ("Every measure is total event number / total stimulus delivery",
              "never a raw count - the number of taps is not fixed, 16 to 31 "
              "for pin prick on Day 1 alone"),
             ("Statistics at two levels",
              "population: six mice, paired.  individual: each mouse's own "
              "~90 deliveries")],
            "four things to take away")

    # ─────────────────────────── method ───────────────────────────────
    divider(prs, "Method", "design, behaviours, and what the scoring "
                           "actually captures")
    fig_slide(prs, "Session design",
              os.path.join(FIGS, "F1_design.png"),
              "5 min baseline, then four 5 min stimulus blocks on a fixed "
              "clock, 1 min rest between",
              "Block order is randomised per mouse, so stimulus identity is "
              "not confounded with time in the session.")
    bullets_slide(
        prs, "Six behaviours, two classes - kept separate on purpose",
        [("REFLEXIVE   paw withdrawal, flinch",
          "spinally mediated, locked to contact. Scored as taps, 3 s "
          "response window."),
         ("AFFECTIVE-MOTIVATIONAL   paw attending, licking / biting, guarding",
          "supraspinal, builds over 1-8 s after contact. 10 s window."),
         ("ESCAPE / REARING   scored, but read as exploration",
          "baseline rate is 3.8 per min and it DECREASES under stimulation. "
          "Not a pain measure."),
         ("Never summed into one pain score",
          "a biased agonist is expected to move the affective class without "
          "matching the reflexive one - summing would hide exactly that.")],
        "Corder / Scherrer framework")

    mp4 = sorted(glob.glob(os.path.join(CLIPS, "PPT_*.mp4")))
    if mp4:
        cands = sorted(glob.glob(mp4[0][:-4] + "_frame*.png"))
        video_slide(prs, "What the scoring looks like", mp4[0],
                    cands[len(cands) // 2] if cands else None,
                    "30 s, both cameras, labels burned in",
                    "click to play  ·  side view is the analysis view; the "
                    "bottom view cannot resolve individual paws")

    table_slide(
        prs, "Day 1 QC outcome", ["", "result"],
        [["Sessions scored", "6 of 6, all four stimulus blocks clean"],
         ["Stimulus mis-keys resolved", "18 of 18, using the fixed block clock"],
         ["Fast repeated taps merged", "corrected copy only, counts unchanged"],
         ["Original scoring", "never modified - corrections go to a separate "
                              "folder"],
         ["Deliveries per mouse", "83 to 100 across the session"],
         ["Remaining flags", "2, informational only (typed vs sheet stimulus "
                             "names)"]],
        "nothing was inferred or filled in; every correction is traceable to "
        "the block clock",
        colw=[3, 7])

    # ────────────── the measurement problem, up front ────────────────
    divider(prs, "Why  total event number / total stimulus delivery",
            "the one methodological point that everything else rests on")
    d2_slide(prs, "The denominator is not constant",
             "D6_why_normalise.png",
             "Ten flinches from ten taps and ten flinches from twenty taps "
             "are the same raw count and a different animal - 10/10 = 1.00 "
             "vs 10/20 = 0.50. Every figure from here on divides by the "
             "delivery count.")

    # ────────── the block window: 5 min or 6 min including rest ──────
    divider(prs, "How long is a stimulus block?",
            "behaviour does not stop when the stimulus does, so the 1 min "
            "rest was tested both ways")
    bullets_slide(
        prs, "Two block definitions, both computed",
        [("5 min  -  the stimulus period only",
          "the block's first delivery plus 300 s"),
         ("6 min  -  the stimulus period plus the 1 min rest that follows it",
          "the rest is attributed to the block before it, because nothing "
          "else caused it and the next block has not started"),
         ("The denominator is identical in both",
          "the rest minute contains no deliveries, so total events / total "
          "stimulus delivery is directly comparable between the two"),
         ("events per MINUTE is not comparable between them",
          "the divisor changes 5 to 6, so that rate drops about 17 % "
          "mechanically even if nothing changed")],
        "both versions are in the repository; the figures below use the 5 min "
        "version unless stated")
    fig_slide(prs, "The rest minute adds almost nothing - except rearing",
              os.path.join(WFIGS, "W1_block_window_comparison.png"), None,
              "Pain behaviours are flat (+0 to +2 %). Escape / rearing rises "
              "+5 % in every mouse - consistent with it being exploration "
              "that resumes once the stimulus stops, not pain.",
              top=1.15)
    fig_slide(prs, "Why the rest minute contributes so little",
              os.path.join(WFIGS, "W2_rest_minute_contribution.png"), None,
              "Because the 5 min window ALREADY contains a mean of 45 s of "
              "quiet time: deliveries finish before the 300 s mark "
              "(first-to-last span, median 256 s). The extra minute lands "
              "further out, where behaviour has already subsided.",
              top=1.15)

    # ─────────────────────────── Day 1 ───────────────────────────────
    divider(prs, "Day 1", "six mice, no drug - what the assay measures")
    fig_slide(prs, "Response by stimulus",
              os.path.join(FIGS, "F2_dose_response.png"),
              "rate per minute within each 5 min block",
              "Light touch smallest and Heat largest for every behaviour; "
              "pin prick sits between them. Not monotonic with nominal "
              "intensity.")
    fig_slide(prs, "Raw event counts per block",
              os.path.join(FIGS, "F2b_event_counts.png"),
              "the unnormalised view, shown for completeness",
              "Read with the previous slide in mind: the block is a fixed "
              "300 s but the number of taps inside it is not.")
    fig_slide(prs, "Total event number / total stimulus delivery",
              os.path.join(FIGS, "F3_per_delivery.png"),
              "each mouse divided by its own delivery count",
              "This is the comparable version of the previous slide.")
    fig_slide(prs, "Stimulus blocks against each animal's own baseline",
              os.path.join(FIGS, "F4_baseline_vs_block.png"),
              "5 min baseline vs the four stimulus blocks",
              "Escape / rearing goes DOWN under stimulation - the clearest "
              "evidence it is exploration rather than pain.")
    fig_slide(prs, "Every mouse individually",
              os.path.join(FIGS, "F5_per_mouse.png"),
              "no averaging - one panel per animal",
              "Between-animal spread is large. It is the reason the paired "
              "design matters more than the group size.")
    fig_slide(prs, "Response time course around each delivery",
              os.path.join(FIGS, "E1_psth.png"),
              "peri-stimulus histogram, aligned to stimulus contact",
              "Reflexes are locked to contact; affective behaviour builds "
              "over several seconds. This is what sets the two response "
              "windows.")
    fig_slide(prs, "Probability of responding to a single delivery",
              os.path.join(FIGS, "E2b_response_probability_10s.png"),
              "10 s window, restricted to deliveries with 10 s of clearance",
              "A 3 s window put Heat BELOW light touch for attending - it "
              "cut the affective response off before it started. 10 s "
              "reverses that (0.600 vs 0.267).")
    fig_slide(prs, "Does the response fade within a block?",
              os.path.join(FIGS, "E3_within_block_timecourse.png"),
              "1 min bins across each 5 min stimulus block",
              "Worth watching on Day 2: a drug can lower the level or "
              "steepen the fade, and those are different mechanisms.")
    fig_slide(prs, "Control: block position does not drive the result",
              os.path.join(FIGS, "E4_block_position.png"),
              "response by position in the session, ignoring which stimulus "
              "it was",
              "Flat, which is what validates the randomised block order.")
    fig_slide(prs, "Reflexive against affective, per mouse",
              os.path.join(FIGS, "E5_reflex_vs_affective.png"),
              "the two classes are not interchangeable",
              "The axis SBI-553 is supposed to move: affective down without "
              "a matching reflexive change.")
    bullets_slide(
        prs, "Day 1, what we can say",
        [("The assay separates stimulus intensities",
          "light touch smallest, heat largest, for all six behaviours"),
         ("Reflexive and affective responses have different time courses",
          "locked to contact vs building over 1-8 s - so different windows"),
         ("Escape / rearing is exploration, not pain",
          "3.8 per min at baseline and it falls under stimulation"),
         ("Block position is not a confound", "the randomisation works"),
         ("Between-animal spread is the limiting factor, not scoring noise",
          "median CV 0.49 across mice - which is why Day 2 must be paired")],
        "real data, six mice")

    # ────────────────────── Day 2 ────────────────────────────────────
    if real:
        divider(prs, "Day 2   SBI-553",
                "same six mice, dosed 10 min before the assay. "
                "No vehicle group.")
        d2_slide(prs, "Analgesia or sedation? The exploration control answers "
                      "it",
                 "S1_sedation_evidence.png",
                 "Escape / rearing is not a pain measure, and it fell "
                 "hardest (to 8 %). Reflexes fell least (to 72 %). A "
                 "selective analgesic would have spared exploration.")
    else:
        divider(prs, "Day 2 plan",
                "EVERY NUMBER IN THIS SECTION IS SYNTHETIC.\n\n"
                "The figures below were generated from a fake Day 2 with a "
                "known 40 % affective reduction injected, escape left "
                "unchanged, and one deliberate non-responder. They exist to "
                "settle the layout and the statistics in advance - and, as "
                "it turns out, to show that n = 6 is fragile.", warn=True)

    d2_slide(prs, "Both days on the same axes, one line per mouse",
             "D8_dose_response_both_days.png",
             note or "the size of the drop and the change in curve shape are "
                     "both only visible when the two days share an axis")
    d2_slide(prs, "The same data, paired mouse by mouse",
             "D2_per_delivery.png",
             note or "grey line = one mouse measured twice")
    # The dot-and-stick per-mouse figure (D5) was dropped: it duplicated the
    # forest plot below without the confidence intervals.
    d2_slide(prs, "Individual level: each mouse's own change index",
             "Fig_forest_per_mouse_allDeliveries.png",
             note or "a per-mouse test is possible only because each "
                     "delivery is a trial")
    d2_slide(prs, "Does the drug change the response shape?",
             "D1_psth_day_compare.png",
             note or "amplitude, latency and duration are separable here")
    d2_slide(prs, "Does the drug change the within-block fade?",
             "D3_within_block.png",
             note or "a lower level and a steeper fade are different "
                     "mechanisms")
    d2_slide(prs, "The dissociation the drug is meant to produce",
             "D4_reflex_vs_affective.png",
             note or "a biased agonist moves points DOWN more than LEFT")

    # ─────────────────── the power problem the mockup found ──────────
    divider(prs, "How much can n = 6 actually show?" if real
            else "What the mockup revealed",
            "from the measured between-mouse variability and from the "
            "arithmetic of the exact test")
    d2_slide(prs, "n = 6 may not be enough",
             "D7_power_planning.png",
             "Left: with one non-responder, power for a 40 % effect at "
             "n = 6 falls from 0.98 to about 0.49. Right: the exact paired "
             "test floors at 2/2^n, so at n = 5 usable pairs p < 0.05 is "
             "arithmetically impossible.")
    table_slide(
        prs, "Two levels of statistics, and what each can deliver",
        ["level", "unit", "n", "test", "what limits it"],
        [["Population", "the mouse", "6",
          "paired Wilcoxon (primary)", "floors at p = 0.031; one "
                                       "non-responder raises it to 0.062"],
         ["Population", "the mouse", "6 vs 6",
          "Mann-Whitney (secondary)", "reaches 0.0022, but ignores the "
                                      "pairing the design bought"],
         ["Individual", "one delivery", "~90 per mouse per day",
          "Fisher exact / change index", "well powered for reflexes, detects "
                                       "only large shifts for affective"],
         ["Direction", "the mouse", "6",
          "how many moved the same way", "not a p-value, but survives the "
                                         "floor"]],
        "the population test is the design; the per-mouse test is what "
        "rescues it at n = 6",
        colw=[1.6, 1.6, 2.0, 2.6, 4.4])
    if real:
        bullets_slide(
            prs, "What this experiment cannot tell us yet",
            [("A vehicle group. All six mice got SBI-553",
              "so drug, day, block order and habituation are confounded. "
              "The next cohort must have a vehicle arm."),
             ("Whether the drop is analgesia at all",
              "escape / rearing fell to 8 % - the exploration control fell "
              "hardest. A locomotion or body-temperature readout would "
              "settle it directly."),
             ("A dose that separates the two",
              "if 10 min post-dose is peak sedation, a lower dose or a "
              "later time point may show analgesia without hypoactivity"),
             ("Animal number",
              "n = 6 gives ~0.49 power for a 40 % effect if one animal does "
              "not respond. n = 8-10 is robust."),
             ("A hardware stimulus marker (TTL or LED on the applicator)",
              "solves timing, stimulus identity and camera sync at once, and "
              "takes the scorer out of the delivery mark")],
            "the honest list, so nobody over-reads the result")
        bullets_slide(
            prs, "What it does show",
            [("The assay works and the scoring is reliable",
              "four clean stimulus blocks in all twelve sessions, and the "
              "stimuli separate on Day 1"),
             ("SBI-553 at this dose and timing produces a large, "
              "consistent behavioural reduction",
              "6/6 mice for lick/bite and escape, paired p = 0.031 - the "
              "floor of the test"),
             ("The reduction is NOT selective for pain behaviours",
              "reflexes fell least (x0.72), exploration fell most (x0.08). "
              "That ordering is the opposite of a biased analgesic."),
             ("The per-delivery design makes per-mouse tests possible",
              "escape / rearing is individually significant in 5 of the 6 "
              "animals on their own ~60-100 deliveries")],
            "what the data supports as it stands")
    else:
        bullets_slide(
            prs, "Decisions needed before Day 2",
            [("Animal number",
              "n = 6 gives ~0.49 power for a 40 % effect if one animal does "
              "not respond. n = 8-10 is robust."),
             ("Keep the delivery count as even as possible between days",
              "normalisation handles unequal counts, but it cannot recover "
              "power that was never there"),
             ("Add a TTL or LED marker on the applicator",
              "solves stimulus timing, identity and camera sync at once"),
             ("Decide now: second baseline block at session end?",
              "it would separate drug effect from time-in-session drift"),
             ("Pick the primary outcome before Day 2 is scored",
              "one behaviour, one stimulus, one measure")],
            "open questions, not conclusions")

    return prs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real", action="store_true",
                    help="drop the MOCKUP notes; only use once the Day 2 "
                         "figures come from real scoring")
    ap.add_argument("--day2-figs", default=None,
                    help="folder holding the Day 2 figures. Defaults to the "
                         "mockup folder, so --real without this would show "
                         "synthetic figures with the warnings removed - the "
                         "worst possible combination.")
    a = ap.parse_args()
    global D2FIGS, REAL
    REAL = a.real
    if a.day2_figs:
        D2FIGS = a.day2_figs
    elif a.real:
        raise SystemExit(
            "--real needs --day2-figs pointing at the REAL Day 2 figures.\n"
            "Without it the deck would show synthetic numbers with the "
            "MOCKUP warnings stripped off.")
    prs = build(real=a.real)
    out = a.out or os.path.join(
        OUTDIR, f"mini1p_SBI553_full_{date.today().isoformat()}"
                f"{'' if a.real else '_DAY2MOCKUP'}.pptx")
    n = 1
    base = out
    while True:
        try:
            prs.save(out)
            break
        except PermissionError:
            # the file is open in PowerPoint; never force it
            n += 1
            r, e = os.path.splitext(base)
            out = f"{r}_v{n}{e}"
    print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} slides")
    print(f"wrote {out}")
    if not a.real:
        print("\nDay 2 section is a MOCKUP. Once the real Day 2 is scored:")
        print("  python make_day2_figs.py --day1 <d1> --day2 <real> "
              "--out lab_meeting\\figs_day2")
        print("  python make_full_ppt.py --real")


if __name__ == "__main__":
    main()
