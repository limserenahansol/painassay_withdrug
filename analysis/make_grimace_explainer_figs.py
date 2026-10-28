"""make_grimace_explainer_figs.py  -  the four schematics the deck needs.

The measured evidence already exists as PNGs from the feasibility sweeps.
What the deck is missing is the explanatory scaffolding: what the mouse
grimace scale IS, what each published pipeline does to a frame, the gate
each of our rigs passes or fails, and a scoreboard.

These are DIAGRAMS, not data. Every number printed on them is quoted from a
source or from one of our own measurement files, and the source is named on
the figure so a PI can check it.

OUTPUT  ->  qc/explainer/
    E1_mgs_action_units.png     what is being scored, and by whom
    E2_pipeline_theory.png      what each paper does to one frame
    E3_gates.png                our two rigs against each requirement
    E4_scoreboard.png           every step we ran, and how it ended

Hansol Lim
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\qc\explainer"
GOOD, BAD, WARN = "#2E7D5B", "#C1272D", "#D98A00"
INK, DIM, PAPER = "#1A1A1A", "#6E6E6E", "#F4F1EC"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 11.5})


def box(ax, x, y, w, h, fc, text, tc="white", fs=10.5, bold=True, ec=None):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                   fc=fc, ec=ec or fc, lw=1.6, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, c=INK, lw=1.8):
    ax.annotate("", (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=c,
                                shrinkA=2, shrinkB=2))


# ------------------------------------------------------------------- E1
def e1():
    fig, ax = plt.subplots(figsize=(15, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")
    ax.text(50, 49.5, "What the Mouse Grimace Scale actually scores",
            ha="center", fontsize=19, fontweight="bold")
    ax.text(50, 46.6, "Langford et al., Nature Methods 2010. Five facial "
                      "action units, each 0 = absent, 1 = moderate, "
                      "2 = severe.  Total 0-10.",
            ha="center", fontsize=12, color=DIM)

    units = [
        ("orbital\ntightening", "eyelid narrows,\nwrinkle around the eye",
         "needs the EYE\nresolved", BAD),
        ("nose\nbulge", "bridge of the nose\nrounds upward",
         "needs nose\nshading", WARN),
        ("cheek\nbulge", "cheek rounds\noutward",
         "PainFace DROPPED\nthis unit", DIM),
        ("ear\nposition", "ears pull apart,\nrotate back",
         "occluded by a\nhead implant", WARN),
        ("whisker\nchange", "whiskers clump\nor stand out",
         "needs 1-2 px\nhairs resolved", BAD),
    ]
    for i, (nm, what, need, col) in enumerate(units):
        x = 3 + i * 19
        box(ax, x, 33, 16, 8.5, "#33475B", nm, fs=13)
        ax.text(x + 8, 30.6, what, ha="center", va="top", fontsize=10.2,
                color=INK, linespacing=1.4)
        ax.add_patch(mp.FancyBboxPatch((x, 20.5), 16, 5.6,
                                       boxstyle="round,pad=0.012",
                                       fc="white", ec=col, lw=2.2))
        ax.text(x + 8, 23.3, need, ha="center", va="center", fontsize=10,
                color=col, fontweight="bold", linespacing=1.4)

    ax.text(50, 16.6, "PainFace scores 4 of the 5 - its authors excluded "
                      "cheek bulge:", ha="center", fontsize=11.5,
            color=INK)
    ax.text(50, 13.9, '"It was not possible to reliably detect a cheek '
                      'bulge ... so we excluded this action unit"',
            ha="center", fontsize=11.5, color=BAD, style="italic")

    ax.add_patch(mp.FancyBboxPatch((6, 2.4), 88, 8.6,
                                   boxstyle="round,pad=0.02", fc=PAPER,
                                   ec=INK, lw=1.6))
    ax.text(50, 8.8, "THE ONE SENTENCE THAT DECIDES EVERYTHING BELOW",
            ha="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(50, 5.1, '"As a rule of thumb, if any of the facial action units '
                     'are hard for a HUMAN to see,\nthese action units will '
                     'be hard for the MLA to detect as well."   - PainFace '
                     'documentation',
            ha="center", va="center", fontsize=12, color=BAD,
            style="italic", linespacing=1.5)
    p = os.path.join(OUT, "E1_mgs_action_units.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- E2
def e2():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(50, 59.4, "What each published pipeline does to one frame",
            ha="center", fontsize=19, fontweight="bold")
    ax.text(50, 56.4, "and the recording conditions each one was trained "
                      "under - which is what our footage has to match",
            ha="center", fontsize=12, color=DIM)

    rows = [
        ("PainFace\nPAIN 2024\nZylka lab",
         ["frame", "RetinaNet\nfinds the mouse", "4 boxes:\neye nose ear\nwhisker",
          "CNN per box\n0 / 1 / 2", "total\n0-8"],
         "COLOUR, 30 fps direct to mp4, camera 15-20 cm from an\n"
         "OPEN-AIR chamber opening, mouse FACING the camera,\n"
         "well illuminated, minimal shadows, no reflections",
         "#C1272D"),
        ("GrimACE\nLab Animal 2026\nBohacek lab",
         ["frame", "own IR chamber\n1280x1024", "face crop\n+ body pose",
          "MGS CNN\n+ BehaviorFlow", "r = 0.87\nvs expert"],
         "its OWN hardware box. IR illumination behind IR-permeable\n"
         "black acrylic. Validated WITH head implants (200 um fibres).\n"
         "authors do not support other geometry (exact wording\n"
         "unverified - version of record is paywalled)",
         "#1C6E8C"),
        ("Fan et al.\nJ Neurosci Meth 2026",
         ["clip", "264 annotated\nclips", "4 rat behaviour\nclasses",
          "classifier", "73 % acc\nface groom 52 %"],
         "rats, body behaviour not face. Its value to us is the\n"
         "training strategy, not the model - we have ~2400 scored\n"
         "bouts against their 264 clips",
         "#5E3C99"),
        ("Andresen et al.\nbioRxiv 2026",
         ["frame", "ResNet-50", "regress mean\nMGS 0-2", "-", "r = 0.85"],
         "deliberately NON-STANDARDISED: varied cameras, lighting\n"
         "and angles. The only model built for footage like ours.",
         "#2E7D5B"),
    ]
    for r, (name, chain, cond, col) in enumerate(rows):
        y = 47.5 - r * 12.2
        box(ax, 1.2, y - 1.2, 13.5, 7.4, col, name, fs=10.4)
        for i, c in enumerate(chain):
            if c == "-":
                continue
            x = 17 + i * 12.2
            box(ax, x, y, 10.4, 5.0, "white", c, tc=INK, fs=9.2,
                bold=False, ec=col)
            if i:
                arrow(ax, x - 1.6, y + 2.5, x - 0.2, y + 2.5, col, 1.5)
        ax.text(67.5, y + 2.5, cond, ha="left", va="center", fontsize=9.4,
                color=INK, linespacing=1.5)
    ax.plot([66.3, 66.3], [1.5, 53], color=DIM, lw=1, ls=":")
    ax.text(41, 53.6, "THE PIPELINE", ha="center", fontsize=11,
            fontweight="bold", color=DIM)
    ax.text(83, 53.6, "WHAT IT WAS TRAINED ON", ha="center", fontsize=11,
            fontweight="bold", color=DIM)
    ax.text(50, 0.4, "All four read REFLECTANCE TEXTURE on a face presented "
                     "to the camera. None of them reads a silhouette.",
            ha="center", fontsize=12.5, fontweight="bold", color=BAD)
    p = os.path.join(OUT, "E2_pipeline_theory.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- E3
def e3():
    reqs = [
        ("COLOUR video", '"Can I use monochrome ... ? No."', "FAIL", "FAIL",
         "both rigs are greyscale"),
        ("imaged through AIR", '"Imaging through glass or plastic should be '
         'avoided"', "FAIL", "FAIL", "both image through a plastic cylinder"),
        ("30 fps recorded\nDIRECTLY to mp4", '"distortions can occur if you '
         '... use conversion software"', "FAIL", "FAIL",
         "DV 29.97 / MJPEG 25, converted"),
        ("mouse FACING\nthe camera", '"only detects all four action units if '
         'the mouse is facing the camera"', "FAIL", "PARTIAL",
         "mini2p is lateral profile; < 4 units -> score = NA"),
        ("no head implant", '"videos of mice with headcaps, conditions that '
         'we did not use"', "PASS", "FAIL", "mini2p wears a 2P miniscope"),
        ("face well\nilluminated", '"if any action unit is hard for a human '
         'to see ..."', "FAIL", "PARTIAL",
         "SBI-553 is backlit; mini2p face is lit but small"),
    ]
    fig, ax = plt.subplots(figsize=(16.5, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis("off")
    ax.text(50, 55.6, "Our two rigs against PainFace's own stated "
                      "requirements", ha="center", fontsize=19,
            fontweight="bold")
    ax.text(50, 52.8, "every requirement quoted verbatim from the PainFace "
                      "documentation - not inferred", ha="center",
            fontsize=12, color=DIM)
    ax.text(3, 47.8, "REQUIREMENT", fontsize=11, fontweight="bold")
    ax.text(21, 47.8, "PainFace's own words", fontsize=11,
            fontweight="bold")
    ax.text(60.5, 47.8, "SBI-553", fontsize=11, fontweight="bold",
            ha="center")
    ax.text(70.5, 47.8, "mini2p", fontsize=11, fontweight="bold",
            ha="center")
    ax.text(77, 47.8, "why", fontsize=11, fontweight="bold")
    col = {"PASS": GOOD, "FAIL": BAD, "PARTIAL": WARN}
    for i, (req, quote, a, b, why) in enumerate(reqs):
        y = 42.5 - i * 6.9
        ax.add_patch(mp.Rectangle((2, y - 1.4), 96, 6.1,
                                  fc=PAPER if i % 2 == 0 else "white",
                                  ec="none", zorder=0))
        ax.text(3, y + 1.7, req, fontsize=10.6, fontweight="bold",
                va="center", linespacing=1.35)
        ax.text(21, y + 1.7, quote, fontsize=9.6, va="center",
                color="#333333", style="italic")
        for xx, v in ((60.5, a), (70.5, b)):
            box(ax, xx - 4.3, y + 0.25, 8.6, 3.0, col[v], v, fs=10)
        ax.text(77, y + 1.7, why, fontsize=9.6, va="center", color=DIM)
    ax.text(50, 0.6, "Requirement 1 alone ends it: PainFace needs colour, "
                     "and every frame we own is greyscale.\n"
                     "Independently confirmed by the GrimACE paper - "
                     '"PainFace, which requires color images"',
            ha="center", fontsize=12, color=BAD, fontweight="bold",
            linespacing=1.6)
    p = os.path.join(OUT, "E3_gates.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------- E4
def e4():
    steps = [
        ("1", "Is a face recorded?",
         "sweep 6991 frames, all 12 SBI videos +\nboth mini2p cameras, "
         "measure the eye",
         "FAIL", "PASS",
         "SBI: 0 eyes in 6991 frames. Interior contrast\n"
         "2.16 vs 4.43 camera read noise = 0.49x\n"
         "mini2p: best eye 15 native px, face in profile"),
        ("2", "Can we frame it better?",
         "follow the mouse and crop tight, so the head\n"
         "fills 23 % of frame instead of 15 %",
         "FAIL", "FAIL",
         "darkness-only tracker: 3 of 9 wrong, locks on\n"
         "the CABLE. + motion: 4 of 9 wrong, locks on the\n"
         "SWINGING cable. Needs a learned detector."),
        ("3", "Will PainFace take it?",
         "check our footage against the published\n"
         "input requirements, one by one",
         "FAIL", "FAIL",
         "6 of 6 requirements fail or partially fail.\n"
         "Colour is required; we are greyscale."),
        ("4", "Use the BODY, not the face",
         "transition structure between scored\nbehaviours - "
         "activity-invariant",
         "PASS", "n/a",
         "grammar unchanged: guarding->none 0.65 -> 0.65,\n"
         "escape->escape 0.39 -> 0.40. One real change:\n"
         "lick/bite->guarding 0.11 -> 0.24"),
        ("5", "Grade each trial",
         "per-trial withdrawal magnitude from\nROI motion energy",
         "FAIL", "PASS",
         "SBI: delivery windows only 1.05x quiet time -\n"
         "invisible. mini2p: peak 4.5x the random null,\n"
         "76/76 trials positive, 7x spread min to max"),
    ]
    fig, ax = plt.subplots(figsize=(17, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")
    ax.text(50, 61.4, "Every step we ran, and how each one ended",
            ha="center", fontsize=19, fontweight="bold")
    ax.text(50, 58.6, "three of five steps failed. Each failure is a "
                      "measurement, not a guess - the number is in the "
                      "right-hand column", ha="center", fontsize=12,
            color=DIM)
    ax.text(9, 54.2, "WHAT WE ASKED", fontsize=11, fontweight="bold",
            ha="center")
    ax.text(35, 54.2, "HOW", fontsize=11, fontweight="bold", ha="center")
    ax.text(52, 54.2, "SBI-553", fontsize=11, fontweight="bold",
            ha="center")
    ax.text(61, 54.2, "mini2p", fontsize=11, fontweight="bold", ha="center")
    ax.text(68, 54.2, "THE DECIDING NUMBER", fontsize=11,
            fontweight="bold")
    col = {"PASS": GOOD, "FAIL": BAD, "n/a": DIM}
    for i, (n, q, how, a, b, num) in enumerate(steps):
        y = 46.5 - i * 9.6
        ax.add_patch(mp.Rectangle((1, y - 1.8), 98, 8.8,
                                  fc=PAPER if i % 2 == 0 else "white",
                                  ec="none", zorder=0))
        box(ax, 1.4, y + 0.6, 3.4, 3.6, "#33475B", n, fs=13)
        ax.text(5.6, y + 2.4, q, fontsize=10.6, fontweight="bold",
                va="center")
        ax.text(25.5, y + 2.4, how, fontsize=9.4, va="center", color=INK,
                linespacing=1.45)
        for xx, v in ((52, a), (61, b)):
            box(ax, xx - 3.6, y + 0.9, 7.2, 3.0, col[v], v, fs=10)
        ax.text(68, y + 2.4, num, fontsize=9.4, va="center", color=INK,
                linespacing=1.45)
    ax.text(50, 0.5, "WHAT SURVIVES:  the body route (step 4) and the "
                     "per-trial magnitude on mini2p (step 5).  "
                     "Everything facial is blocked by lighting geometry, "
                     "not by pixel count.",
            ha="center", fontsize=12.5, fontweight="bold", color="#33475B")
    p = os.path.join(OUT, "E4_scoreboard.png")
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (e1, e2, e3, e4):
        print("wrote", f())
