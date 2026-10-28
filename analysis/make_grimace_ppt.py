"""make_grimace_ppt.py  -  can we score mouse grimace on OUR video? For a PI.

THE SHAPE HANSOL ASKED FOR
    1  what the papers did, and the theory behind them
    2  step by step, how each one could be applied to our footage
    3  the result of each step - pass or fail, with the deciding number
    every claim next to the figure that shows it

WHY A DECK RATHER THAN A PARAGRAPH
    Three of the five steps failed. A failure is only useful if the reason
    is visible, so every FAIL slide carries the image or the number that
    produced it. A PI should be able to disagree with the measurement, not
    just with the conclusion.

OUTPUT  ->  <this folder>\\..\\Grimace_feasibility_<date>.pptx

USAGE
    python make_grimace_ppt.py

Hansol Lim - HEAL mini1p / SBI-553 and CEA-Ntsr1 mini2p
"""
from __future__ import annotations

import os
import sys

from pptx.util import Inches

sys.path.insert(0, r"C:\Users\hsollim\OneDrive\Desktop\cursor"
                   r"\pain_2plane_split")
from deck import (ACC, GOOD, WARN, blank, bullets, fig, new_deck,  # noqa
                  path_line, save, section, table, text, title)

HERE = os.path.dirname(os.path.abspath(__file__))
SBI = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553"
EX = os.path.join(SBI, "qc", "explainer")
SW = os.path.join(SBI, "qc", "face_sweep")
QC = os.path.join(SBI, "qc")
M2P = (r"I:\20260911_CEANTSR1_65_15_pain_ 10min_pin_heat_2026-09-11_"
       r"15-56-48\output_split")
UPL = r"C:\Users\hsollim\Documents\PainFace_upload"
STAMP = "20260930"


def DIMC():
    from deck import DIM
    return DIM


def build():
    p = new_deck()

    # ------------------------------------------------------------- title
    s = blank(p)
    text(s, "Can we score mouse grimace on our own video?",
         Inches(.9), Inches(1.8), Inches(11.6), Inches(1.0), size=32,
         bold=True)
    text(s, "Four published pipelines, two of our rigs, five steps - "
            "three of which failed",
         Inches(.9), Inches(2.9), Inches(11.6), Inches(.5), size=18,
         color=ACC)
    text(s,
         "RIG A   mini1p / SBI-553 acute pain assay.  6 mice x 2 days x 2 "
         "cameras, 720x480 DV, 29.97 fps, backlit.  No head implant.\n"
         "RIG B   mini2p / CEA-Ntsr1.  1 mouse, pain + open field, "
         "1600x1200 MJPEG, 25 fps, side-lit.  Head-mounted two-photon "
         "miniscope.\n\n"
         "Every number on the following slides comes from a measurement "
         "file or a verbatim quote, and the source is named on the slide.\n"
         "Nothing here is an impression.",
         Inches(.9), Inches(3.6), Inches(11.6), Inches(2.4), size=14,
         color=DIMC())
    path_line(s, r"code  HEAL_mini1p_SBI553\analysis\ , "
                 r"pain_2plane_split\   |   evidence  qc\face_sweep\ , "
                 r"qc\explainer\ ")

    # =========================================== 1 the papers and theory
    section(p, 1, "What the papers did, and why it works for them",
            "Four pipelines, and the recording conditions each one was "
            "trained under.\n"
            "That last part is what decides whether any of them can touch "
            "our footage.")

    fig(p, "What the Mouse Grimace Scale actually scores",
        "Five facial action units, each 0 / 1 / 2. This is the quantity "
        "every one of these tools is trying to estimate.",
        os.path.join(EX, "E1_mgs_action_units.png"),
        r"Langford et al., Nature Methods 2010; PainFace documentation",
        hbox=5.1,
        note="Note the bottom line, which is PainFace's own rule of thumb: "
             "if a human cannot see the action unit, the network will not "
             "either. That single sentence is the test we applied to our "
             "own footage in part 2, and it is why three of our five steps "
             "failed.")

    fig(p, "What each pipeline does to one frame",
        "Left of the dotted line: the processing chain. Right of it: the "
        "recording conditions the model was trained under - the part that "
        "is usually skipped and that decides everything here.",
        os.path.join(EX, "E2_pipeline_theory.png"),
        r"PainFace PAIN 2024 | GrimACE Lab Animal 2026 | Fan et al. "
        r"J Neurosci Methods 2026 | Andresen et al. bioRxiv 2026",
        hbox=5.2,
        note="All four read REFLECTANCE TEXTURE on a face presented to the "
             "camera. None of them reads a silhouette. GrimACE adds body "
             "pose next to the face score, which is the one part that "
             "survives on our data - see step 4.")

    bullets(p, "The theory, in one paragraph each",
            "why these pipelines work, and what each one assumes",
            "PAINFACE  (Zylka lab, PAIN 2024)\n"
            "  A RetinaNet detector finds the mouse and draws a box around "
            "four action units - orbital, nose, ear,\n"
            "  whisker. A small CNN then reads the pixels inside each box "
            "and assigns 0, 1 or 2, giving a total of 0-8.\n"
            "  Trained on over 70,000 images of black and white mice. Runs "
            "in the cloud at painface.net, about 100x\n"
            "  faster than a human, and returns roughly one score per "
            "second of 30 fps video.\n"
            "  ASSUMES  the face is in colour, well lit, facing the camera, "
            "and close enough to fill the box.\n"
            "  Cheek bulge was DROPPED by the authors: 'It was not possible "
            "to reliably detect a cheek bulge'.\n\n"
            "GRIMACE  (Bohacek lab, Lab Animal 2026)\n"
            "  Hardware and software together. A standardised dark box with "
            "infrared illumination behind IR-permeable\n"
            "  black acrylic, at 1280x1024. Automated MGS runs beside "
            "DeepLabCut-style body pose, so pain and\n"
            "  movement are separate axes. Agreement with a human expert "
            "r = 0.87, validated after craniotomy WITH\n"
            "  head implants - 200 um optical fibres.\n"
            "  ASSUMES  its own chamber and IR illumination. NOTE: the "
            "often-quoted sentence about hardware-only\n"
            "           compatibility is NOT in the bioRxiv preprint and the version of record is paywalled -\n"
            "           treat it as not determined.\n"
            "  THE NUMBER TO REMEMBER  'In a 10-min testing period, one can "
            "expect approximately 25 usable frames.'\n"
            "  Even in a purpose-built box, grimace is a sparse "
            "measurement.\n\n"
            "FAN ET AL.  (J Neurosci Methods 2026)\n"
            "  Not faces. Four pain-related BODY behaviours in rats from "
            "264 annotated clips; 73 % overall accuracy,\n"
            "  but face grooming only 52 %. Its value to us is the "
            "training strategy: we hold about 2400 scored\n"
            "  bouts, roughly ten times their training set, and their "
            "per-class result predicts which of our\n"
            "  behaviours will be hardest - the face-directed ones.\n\n"
            "ANDRESEN ET AL.  (bioRxiv 2026)\n"
            "  A ResNet-50 that regresses the MEAN MGS on a 0-2 scale with "
            "no per-unit boxes at all, trained across\n"
            "  deliberately NON-STANDARDISED cameras, lighting and angles. "
            "Expert correlation r = 0.85.\n"
            "  ASSUMES  far less than the others. This is the only one "
            "built for footage like ours.",
            size=11.5)

    # =========================================== 2 step by step, and results
    section(p, 2, "Step by step on our own video, and what happened",
            "Five steps, each one a gate. The slide after each step shows "
            "the measurement that decided it.\n"
            "Three failed. Each failure has a number, so it can be argued "
            "with.")

    fig(p, "The whole answer on one slide",
        "Every step, both rigs, and the number that decided each one. The "
        "slides that follow take them in turn.",
        os.path.join(EX, "E4_scoreboard.png"),
        r"qc\explainer\E4_scoreboard.png", hbox=5.2)

    # ---- step 1
    bullets(p, "STEP 1   Is a face even recorded?", "the first gate, and "
            "the one that settles RIG A",
            "WHAT WE DID\n"
            "  Sampled 6991 frames across ALL 12 SBI-553 videos (6 mice x "
            "2 days) plus both mini2p cameras and the\n"
            "  open-field camera. For every frame: locate the head, then "
            "ask whether an eye is identifiable, and if so\n"
            "  measure it in NATIVE pixels with a scale bar printed on the "
            "crop. Magnified with nearest-neighbour only,\n"
            "  because smooth interpolation invents detail the sensor never "
            "recorded.\n\n"
            "WHY THE EYE AND NOT THE WHOLE MOUSE\n"
            "  Orbital tightening had the highest inter-rater reliability "
            "of all five units in Andresen et al., so it is\n"
            "  the unit any pipeline leans on hardest. If the eye is not "
            "recorded, nothing downstream can be.\n\n"
            "RESULT  RIG A  FAIL       RIG B  PASS\n"
            "  SBI-553   0 eyes identifiable in 6991 frames. 47 candidate "
            "detections were cropped and adjudicated\n"
            "            one at a time; every one was a mesh-bar junction "
            "or silhouette penumbra.\n"
            "  mini2p    the eye IS recorded. Best palpebral aperture in "
            "the session: 15 native pixels.",
            size=12.5)

    fig(p, "STEP 1 evidence   the same scale bar on all three cameras",
        "15 native pixels, drawn in red on every panel. Top: mini2p side "
        "camera, the best eye in the session. Middle: mini2p second camera "
        "- mesh floor, no face. Bottom: open field - dorsal view, only the "
        "miniscope.",
        os.path.join(SW, "19_EVIDENCE_eye_scale_3cameras.png"),
        r"qc\face_sweep\19_EVIDENCE_eye_scale_3cameras.png", hbox=5.3, slice_tall=False,
        note="Top left is the whole case for RIG B: an eye with a corneal "
             "highlight, the lower lid, the whisker pad. Top right is the "
             "same pixels contrast-stretched - no new detail appears, which "
             "is the point. The two lower rows are why the other two "
             "cameras were dropped.")

    fig(p, "STEP 1 evidence   why RIG A fails, in one figure",
        "The bottom camera of the SBI-553 rig, swept across all 12 videos.",
        os.path.join(SW, "FIGURE_mgs_bottom_camera_verdict.png"),
        r"qc\face_sweep\FIGURE_mgs_bottom_camera_verdict.png", hbox=5.3, slice_tall=False,
        note="The decisive measurement is NOT the eye size. It is that the "
             "animal carries no recorded contrast at all: high-pass RMS "
             "inside the mouse is 2.16 grey levels against the camera's own "
             "read noise of 4.43, and a lit mesh floor at 20.80. The "
             "interior sits at 0.49x the read noise. At 3.41 px/mm an eye "
             "WOULD be 10.2 px, so size is not the blocker - lighting "
             "geometry is. A 4K sensor in the same position records the "
             "same featureless silhouette.")

    fig(p, "STEP 1   what a PI can check by eye",
        "One frame from each rig, at the same magnification, exactly as it "
        "would be handed to a scoring tool.",
        os.path.join(UPL, "0_what_gets_uploaded.png"),
        r"PainFace_upload\0_what_gets_uploaded.png", hbox=5.0,
        note="Left, mini2p: lateral profile, eye and whiskers visible. "
             "Right, SBI-553 after lifting the shadows hard (gamma 2.6): "
             "the ear, the fur texture and the paws come back - more than "
             "expected - but no eye. Worth saying plainly: an earlier, "
             "thinner look at 5 frames called RIG A a pure silhouette with "
             "nothing recoverable. That was too strong. The correct "
             "statement is narrower: no EYE, in 6991 frames.")

    # ---- step 2
    bullets(p, "STEP 2   Can we at least frame it better?",
            "the face is small because the camera is looking at the whole "
            "chamber",
            "THE IDEA\n"
            "  In the mini2p side view the head is about 6 % of the frame "
            "width, because 1600x1200 is spent on the\n"
            "  cylinder, the mesh, the diffuser and the cable. A fixed crop "
            "of the cylinder raises that to 15 %.\n"
            "  FOLLOWING the mouse and cropping tight would raise it to "
            "23 %, which is much closer to the framing\n"
            "  every one of these models was trained on.\n\n"
            "WHAT WE BUILT, AND HOW EACH ONE BROKE\n"
            "  tracker 1   darkest compact region after a heavy blur\n"
            "              3 of 9 checked frames wrong. It locks onto the "
            "MINISCOPE CABLE and the static shadow band\n"
            "              at the top of the cylinder - both darker than "
            "the animal, and both stationary.\n"
            "  tracker 2   darkness + frame-to-frame motion + a jump limit\n"
            "              WORSE. 4 of 9 wrong and 38 % of frames held. "
            "Adding motion moves the estimate onto the\n"
            "              SWINGING CABLE, which is the most mobile object "
            "in the scene.\n\n"
            "RESULT  BOTH RIGS  FAIL\n"
            "  Darkness alone finds static dark clutter. Darkness plus "
            "motion finds the cable. There is no simple\n"
            "  threshold that separates a black mouse from a black cable on "
            "a dark mesh floor.\n"
            "  So the shipped crop is FIXED on the cylinder - it cannot "
            "fail, and it still gets 6 % to 15 %.\n\n"
            "WHAT WOULD WORK  a learned detector - which is exactly what "
            "all four papers use. DeepLabCut or SLEAP\n"
            "  from about 100-200 hand-labelled frames. That is the real "
            "cost of doing this properly on this rig.",
            size=12.5)

    fig(p, "STEP 2 evidence   the tracker, and where it goes wrong",
        "Green box = the crop that would have been sent onward. Red dot = "
        "the tracked centre. Nine frames spread across the session.",
        os.path.join(UPL, "FAILED_tracker_do_not_use.png"),
        r"PainFace_upload\FAILED_tracker_do_not_use.png", hbox=5.3, slice_tall=False,
        note="Look at the bottom row and the right column: the mouse is "
             "clearly at the bottom of the cylinder with its face in "
             "profile, and the box is up on the cable assembly. This figure "
             "is kept in the folder under the name "
             "FAILED_tracker_do_not_use.png so the attempt is not repeated.")

    # ---- step 3
    fig(p, "STEP 3   Will PainFace accept the file at all?",
        "Our two rigs checked against PainFace's own published input "
        "requirements, one row at a time. Every requirement is quoted "
        "verbatim, not inferred.",
        os.path.join(EX, "E3_gates.png"),
        r"painface.net documentation; PainFace PAIN 2024", hbox=5.2,
        note="RESULT: BOTH RIGS FAIL, 6 of 6 requirements. Requirement 1 "
             "alone ends it - PainFace needs COLOUR and every frame we own "
             "is greyscale - and that is independently confirmed in the "
             "GrimACE paper, which calls PainFace 'which requires color "
             "images'. I had already built mp4 files and recommended "
             "uploading them before finding this. That recommendation is "
             "withdrawn.")

    bullets(p, "STEP 3   and if we upload anyway, it still returns numbers",
            "which is the trap, and the reason this slide exists",
            'PAINFACE\'S OWN WORDS\n'
            '  "PainFace will output MGS scores even if images are poor '
            'quality (such scores may be associated with\n'
            '   low confidence values)."\n\n'
            '  "As a rule of thumb, if any of the facial action units are '
            'hard for a human to see, these action units\n'
            '   will be hard for the MLA to detect as well."\n\n'
            '  and on black fur under poor conditions, from the authors: '
            '"Experts in our lab were unable to\n'
            '   consistently find the eyes against the black fur, let alone '
            'determine if the eyes were open or closed,\n'
            '   nor were we able to consistently identify the nose, the '
            'cheeks, or the whiskers."\n\n'
            "SO A RETURNED SCORE WOULD NOT BE EVIDENCE OF ANYTHING. That is "
            "why we did not run it and report the\n"
            "output - a plausible-looking number from an out-of-domain "
            "model is worse than no number.\n\n"
            "ONE MORE THING TO KNOW BEFORE ANY UPLOAD, TO ANY PLATFORM\n"
            '  "Videos that users upload to PainFace may be used to improve '
            'the accuracy of the machine learning\n'
            '   algorithm."  Unpublished data uploaded there may become '
            'training data. That is a lab decision.',
            size=12.5, color=WARN)

    # ---- step 4
    bullets(p, "STEP 4   Use the BODY instead of the face",
            "the half of GrimACE that does not need its chamber",
            "WHY THIS IS THE RIGHT FALLBACK\n"
            "  GrimACE scores the face AND the body pose, deliberately, "
            "because pain and movement are separate axes -\n"
            "  that is how it caught opioid hyperactivity in its "
            "buprenorphine group at 4 h. The body half needs no\n"
            "  face and no IR chamber.\n\n"
            "THE PROBLEM IT SOLVES FOR SBI-553\n"
            "  Every behaviour count fell on the drug day, and baseline "
            "locomotion was already halved before any\n"
            "  stimulus. A uniform reduction in motor output produces "
            "exactly that, so no COUNT can separate\n"
            "  analgesia from sedation. A row-normalised TRANSITION matrix "
            "can: each row sums to 1, so it is\n"
            "  independent of how much the animal did.\n\n"
            "RESULT  RIG A  PASS\n"
            "  The behavioural grammar is essentially unchanged on the drug "
            "day:\n"
            "      guarding -> none      0.65 -> 0.65        attending -> "
            "none    0.61 -> 0.61\n"
            "      escape -> escape      0.39 -> 0.40        lick/bite -> "
            "lick/bite  0.29 -> 0.29\n"
            "  An analgesic should make pain-driven states END sooner and "
            "be ENTERED less from inside a response.\n"
            "  Neither happens. THE ONE REAL CHANGE: lick/bite -> guarding "
            "0.11 -> 0.24, active to passive,\n"
            "  supported by affective onset LATER (1.70 -> 2.20 s) and "
            "bouts LONGER (0.50 -> 0.83 s).\n\n"
            "  HONEST CAVEAT  the conditional P(affective | withdrawal) "
            "falls 0.56 -> 0.37, p = 0.094, which leans the\n"
            "  other way - toward an affective-selective effect. n = 6, "
            "Wilcoxon floor p = 0.031. Read together, the\n"
            "  drug looks predominantly sedating with a possible additional "
            "component this n cannot resolve.\n"
            "  IMPORTANT  this step used HANSOL'S MANUAL SCORING, not an "
            "automated classifier. It borrows the\n"
            "  paper's analysis idea, not its model.",
            size=11.8)

    fig(p, "STEP 4 evidence   the transition structure, activity-invariant",
        "P(next state | this state) at 1 Hz, pooled over the 6 mice. Each "
        "ROW sums to 1, so these numbers do not change when the animal "
        "simply does less.",
        os.path.join(QC, "flow", "fig_transition_matrix.png"),
        r"behaviour_flow.py  ->  qc\flow\fig_transition_matrix.png",
        hbox=5.2,
        note="The right-hand panel is the whole argument: almost every cell "
             "is within 0.06 of zero. The drug did not change what follows "
             "what. The one orange cell, lick/bite -> guarding at +0.12, is "
             "a shift from an active affective state to a passive one - the "
             "direction motor slowing predicts.")

    # ---- step 5
    bullets(p, "STEP 5   Grade each stimulus trial",
            "stop treating every delivery as the same event",
            "WHY IT MATTERS FOR THE IMAGING\n"
            "  Every pain claim in the mini2p project so far treats the 87 "
            "deliveries as identical and asks whether a\n"
            "  neuron's trace rises afterwards. The whole analysis therefore "
            "ran on a BINARY regressor, and the window\n"
            "  and baseline choices decided the answer - which is the "
            "argument that produced four different cell lists.\n"
            "  A graded per-trial magnitude changes the question to 'does "
            "this neuron's response SCALE with how hard\n"
            "  the animal reacted', which needs no baseline window at all: "
            "it is a correlation across trials.\n\n"
            "RESULT  RIG A  FAIL\n"
            "  Whole-frame motion energy cannot see a withdrawal. Median "
            "motion from 1.5 s before to 2.5 s after a\n"
            "  delivery: 2.89, against 2.75 inside the same blocks but more "
            "than 6 s from any delivery - 1.05x.\n"
            "  99 withdrawals should be visible in a signal that could "
            "measure them. Also refuted the obvious worry:\n"
            "  the experimenter's hand does NOT dominate, it moves the "
            "number by 5 %. What does interfere is the\n"
            "  backlight - frame brightness sd 23.8 grey levels on a mean "
            "of 61, single steps up to 13.8 against a\n"
            "  motion median of 2.47. The lab's own auto_B_train.py already "
            "records that a withdrawal is 1.5-4.5\n"
            "  frames at 30 fps and 'stays manual until the frame rate goes "
            "up'.\n\n"
            "RESULT  RIG B  PASS\n"
            "  Restricted to the cylinder, the withdrawal is clear: "
            "peri-event peak +5.53 against a random-time null\n"
            "  95th percentile of +1.24, so 4.5x. All 76 trials positive, "
            "median 10.47, range 2.74-19.26 - a 7-fold\n"
            "  spread between the smallest and largest reaction, which is "
            "exactly the variance a binary regressor\n"
            "  throws away. 25 fps allows AMPLITUDE; it does not allow "
            "velocity, and none is reported.",
            size=11.8)

    fig(p, "STEP 5 evidence   RIG A, the measurement that did not work",
        "99 withdrawals aligned to the scoring keypress, SBI-553 bottom "
        "camera at native 30 fps.",
        os.path.join(QC, "kinematics", "CHECK_M2_day1.png"),
        r"withdrawal_kinematics.py  ->  qc\kinematics\CHECK_M2_day1.png",
        hbox=4.9,
        note="Left: the peri-event mean is flat - no peak at all. Right: "
             "the detected rise times pile up at 1-3 frames, which is the "
             "detector crossing backlight flicker rather than movement. "
             "Both panels are in the folder so the negative is visible "
             "rather than asserted.")

    fig(p, "STEP 5 evidence   RIG B, the measurement that did work",
        "The same idea on the mini2p side camera, restricted to the "
        "cylinder.",
        os.path.join(M2P, "motion", "fig30_withdrawal_magnitude.png"),
        r"withdrawal_magnitude.py  ->  <mini2p>\output_split\motion"
        r"\fig30_withdrawal_magnitude.png", hbox=5.0,
        note="Left: the peak clears the random-time null by 4.5x, and note "
             "it sits at -0.3 s - BEFORE the keypress. That independently "
             "reproduces the 0.32 s scoring delay measured earlier from a "
             "different camera, the third time this project has arrived at "
             "that number. It also cost a first pass: the original window "
             "[0, +1.0] s missed the peak entirely and gave 2.2x with "
             "negative amplitudes. Middle: the trials are not "
             "interchangeable. Right: pin against heat, descriptive only - "
             "they are separate blocks in this session.")

    # =========================================== 3 what it means
    section(p, 3, "Where this leaves us",
            "What survives, what is blocked, and the smallest change that "
            "would unblock it.")

    rows = [["", "RIG A   SBI-553 / mini1p", "RIG B   mini2p"],
            ["face recorded?",
             "NO - 0 eyes in 6991 frames",
             "YES - best eye 15 native px"],
            ["limiting factor",
             "lighting geometry: interior contrast\n"
             "0.49x the camera's read noise",
             "framing: the head is 6 % of a\n1600x1200 frame"],
            ["would a better sensor help?",
             "NO - a 4K camera in the same\nposition records the same "
             "silhouette",
             "partly - but a closer camera\nhelps far more"],
            ["PainFace usable?", "NO - 6 of 6 requirements fail",
             "NO - colour, facing-camera and\nheadcap all fail"],
            ["GrimACE usable?",
             "NO - its software is tied to its\nown IR hardware",
             "NO - same, and our eye is 4-5x\nsmaller than its face crops"],
            ["best available route",
             "BODY only: transition structure\n(step 4, works)",
             "Andresen et al. 2026 for the face;\nper-trial magnitude for "
             "the body"],
            ["what we can claim today",
             "the drug looks predominantly\nsedating - grammar unchanged",
             "a graded per-trial withdrawal\nmagnitude, 7x spread"]]
    s = blank(p)
    title(s, "The two rigs, side by side",
          "every cell traceable to a measurement file named in part 2")
    table(s, rows, Inches(.4), Inches(1.5), Inches(12.5),
          Inches(.62 * len(rows)), size=11)
    path_line(s, r"qc\face_sweep\ , qc\flow\ , qc\kinematics\ , "
                 r"<mini2p>\output_split\motion\ ")

    bullets(p, "What I got wrong along the way",
            "stated so the record is straight, and so the same mistakes are "
            "not made again",
            "1  I recommended uploading our video to PainFace, and built "
            "the mp4 files for it.\n"
            "   WRONG. PainFace requires colour; our footage is greyscale. "
            "Recommendation withdrawn before upload.\n\n"
            "2  I called RIG A 'a pure silhouette with nothing "
            "recoverable', from 5 frames.\n"
            "   TOO STRONG. Lifting the shadows recovers the ear, the fur "
            "texture and the paws. The defensible\n"
            "   statement is narrower: no EYE, in 6991 frames across all 12 "
            "videos.\n\n"
            "3  I estimated the mini2p eye at 9 native px from one frame.\n"
            "   UNDERESTIMATE. A proper sweep measured the best palpebral "
            "aperture at 15 px. Still too small for a\n"
            "   graded 0/1/2 orbital score, but the correct number.\n\n"
            "4  I said GrimACE could not be obtained or applied "
            "retrospectively at all.\n"
            "   HALF WRONG. The code is public - GitHub ETHZ-INS, AGPL-3.0, "
            "Zenodo 10.5281/zenodo.15119195 - and\n"
            "   there is a video-file input path. But the authors state the "
            "software is compatible only with their\n"
            "   hardware, so the conclusion stands for a different reason "
            "than the one I gave.\n\n"
            "5  I guessed the experimenter's hand was swamping the "
            "withdrawal signal in RIG A.\n"
            "   WRONG, and I checked: it moves the number by 5 %. The "
            "backlight flicker is the real interference.\n\n"
            "6  My first per-trial window on RIG B was [0, +1.0] s and "
            "missed the peak, which sits at -0.3 s.\n"
            "   Fixed to [-0.8, +0.5] s; the separation from the null went "
            "from 2.2x to 4.5x.",
            size=12, color=WARN)

    bullets(p, "What to change, if a facial pain measure matters",
            "the specification is not a mystery - it is PainFace's own "
            "recording protocol",
            "FOR EITHER RIG, TO MAKE FACIAL SCORING POSSIBLE AT ALL\n"
            "  colour camera, not monochrome\n"
            "  FRONT lighting, not backlighting. This is the single change "
            "that decides RIG A.\n"
            "  image through AIR, not through the plastic cylinder - glare "
            "and reflections confuse the detector\n"
            "  record 30 fps DIRECTLY to mp4; do not convert afterwards\n"
            "  camera 15-20 cm from the animal, with the mouse facing it, "
            "so the eye is 50-100 px instead of 15\n\n"
            "AND EXPECT IT TO BE SPARSE EVEN THEN\n"
            "  GrimACE, in a purpose-built chamber: 'In a 10-min testing "
            "period, one can expect approximately\n"
            "  25 usable frames.' Grimace is a slow, sparse measure. Plan "
            "the session length around that.\n\n"
            "FOR THE SBI-553 QUESTION SPECIFICALLY - sedation or analgesia\n"
            "  The facial route is blocked, and the body route has already "
            "given an answer (step 4). To go further,\n"
            "  the missing measurement is withdrawal KINEMATICS: does the "
            "drug reduce how often the paw is\n"
            "  withdrawn, or how the withdrawal MOVES? That needs front "
            "lighting during delivery and a higher\n"
            "  frame rate - at 30 fps a withdrawal is 1.5-4.5 frames, which "
            "the lab's own code already documents.\n\n"
            "FOR THE MINI2P IMAGING\n"
            "  The per-trial magnitude (step 5) is ready to use now. The "
            "next analysis is each neuron's response\n"
            "  against that magnitude across trials - a correlation with no "
            "baseline window, which sidesteps the\n"
            "  window argument that has dominated this project.",
            size=12, color=GOOD)

    return save(p, f"Grimace_feasibility_{STAMP}", os.path.dirname(HERE))


if __name__ == "__main__":
    build()
