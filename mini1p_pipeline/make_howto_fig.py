"""make_howto_fig.py - 10-26: one 'how to read every slide' figure for the PI deck: the colour code of the 4 sessions, what each
bracket compares, and all neurons vs the same tracked neurons.  ->  <PI5 figures>\\s_howto.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

FIG = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\lab_meeting\decks_Day1_Day2\v2_20261022\FINAL_20261026_PI5_mixmarks\figures"
CC = {"vb": "#c4c4c4", "vi": "#2471a3", "sb": "#7a7a7a", "si": "#c0392b"}
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

fig, ax = plt.subplots(1, 2, figsize=(12.3, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.08, left=0.01, right=0.99,
                                                                   top=0.97, bottom=0.03))
a = ax[0]
a.set_xlim(-0.7, 7.6); a.set_ylim(-1.4, 10); a.axis("off")
H = [3.6, 5.2, 3.0, 1.8]
for k, (c, lab) in enumerate(zip(("vb", "vi", "sb", "si"), ("baseline", "after\nvehicle", "baseline", "after\nSBI-553"))):
    a.bar(k, H[k], color=CC[c], width=0.7, bottom=1.6)
    a.text(k, 1.35, lab, ha="center", va="top", fontsize=11)
a.text(0.5, -0.05, "vehicle day", ha="center", va="top", fontsize=12, color=CC["vi"], fontweight="bold")
a.text(2.5, -0.05, "SBI-553 day", ha="center", va="top", fontsize=12, color=CC["si"], fontweight="bold")


def br(x1, x2, y, col, lw=2):
    a.plot([x1, x1, x2, x2], [y, y + 0.3, y + 0.3, y], color=col, lw=lw)


br(0, 1, 7.2, CC["vi"]); br(2, 3, 7.2, CC["si"]); br(1, 3, 8.2, "k"); br(0, 2, 9.2, "0.55")
rows = [(9.35, "0.45", "grey: baseline vs baseline (same start?)"),
        (8.35, "k", "black: after vehicle vs after SBI-553 (between days)"),
        (7.35, CC["si"], "red: SBI-553 vs its own baseline (same day, same neurons)"),
        (6.35, CC["vi"], "blue: vehicle vs its own baseline (same day, same neurons)")]
for y, col, t in rows:
    a.text(3.75, y, t, va="center", fontsize=11.5, color=col, fontweight="bold")
a.text(3.75, 4.6, "Stars on every bracket:\n* p < 0.05   ** p < 0.01   *** p < 0.001   ns", va="center", fontsize=11.5)
a.text(3.75, 2.6, "Markers on bars = the 4 mice\n(F = 5941, 1191;  M = 5939, 5937)", va="center", fontsize=11.5, color="0.3")

b = ax[1]
b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off"); b.set_aspect("equal")
b.add_patch(Ellipse((3.6, 5.0), 6.0, 6.6, fc=CC["vi"], alpha=0.18, ec=CC["vi"], lw=2))
b.add_patch(Ellipse((6.4, 5.0), 6.0, 6.6, fc=CC["si"], alpha=0.18, ec=CC["si"], lw=2))
b.text(1.9, 5.0, "vehicle\nday\n216", ha="center", va="center", fontsize=12.5, color=CC["vi"], fontweight="bold")
b.text(8.1, 5.0, "SBI-553\nday\n246", ha="center", va="center", fontsize=12.5, color=CC["si"], fontweight="bold")
b.text(5.0, 5.0, "tracked\n185\nsame cell\nboth days", ha="center", va="center", fontsize=12, fontweight="bold")
b.text(5.0, 9.3, "all neurons of each day  vs  the same tracked neurons", ha="center", fontsize=11.5, fontweight="bold")
b.text(5.0, 0.6, "all neurons: unpaired between days  ·  tracked: each cell vs itself", ha="center", fontsize=10.5, color="0.3")
fig.savefig(FIG + r"\s_howto.png"); plt.close(fig)
print("ok")
