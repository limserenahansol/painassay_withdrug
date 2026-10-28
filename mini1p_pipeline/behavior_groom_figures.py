"""Figures of the final automatic grooming (and movement) decision, all 8 sessions of 2026-10-19:
  grooming_decision_timeline.png   one row per session: grooming bouts (pink) and movement (teal)
                                   over the 16.5-min schedule
  grooming_decision_bouts_p<k>.png every grooming bout: bout number, session, time, and 5 side-view
                                   frames spread over the bout (for the user to say right / wrong)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from behavior_final_move_groom import S, valid_name, spans, FINAL_DIR, AUTO_DIR, FPS
from behavior_supervised import BLOCKS

OUT = r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\behavior_auto\20261019\grooming"
fig, ax = plt.subplots(figsize=(16, 6.5))
rows = []
for i, (tag, stem, side, sno, sex, phase, stim, manual) in enumerate(S):
    A = sio.loadmat(rf"{AUTO_DIR}\ScoringAB_{valid_name(stem)}.mat", squeeze_me=True)
    g, m = np.asarray(A["groomVec"]).astype(bool), np.asarray(A["moveVec"]).astype(bool)
    y = len(S) - 1 - i
    for s, e in spans(m):
        ax.add_patch(plt.Rectangle((s / FPS, y - 0.12), (e - s) / FPS, 0.24, color="#17a2b8", alpha=0.35, lw=0))
    for s, e in spans(g):
        ax.add_patch(plt.Rectangle((s / FPS, y - 0.38), max((e - s) / FPS, 2), 0.76, color="#d6338a", lw=0))
        rows.append((tag, side, s, e))
    ax.text(1015, y, f"groom {g.sum() / FPS:.0f} s, move {100 * m.mean():.0f}%", va="center", fontsize=9)
for k, (b, s0, s1) in enumerate(BLOCKS[1:], 1):
    ax.axvspan(s0, s1, color="0.92", zorder=0)
    ax.text((s0 + s1) / 2, len(S) - 0.35, f"block {k}", ha="center", fontsize=9, color="0.4")
ax.set_yticks(range(len(S)), [f"{t[0]} ({t[5][:4]})" for t in S][::-1])
ax.set_xlim(0, 1130); ax.set_ylim(-0.6, len(S) - 0.1)
ax.set_xlabel("time in session (s)")
ax.set_title("Automatic grooming (pink, numbered on the next pages) and general movement (teal), 8 sessions of 2026-10-19")
fig.text(0.5, -0.01, "Grooming model trained on your 117 judged clips (28 grooming); a clip you judged always keeps your answer.",
         ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT + r"\grooming_decision_timeline.png", dpi=110, bbox_inches="tight")
plt.close(fig)

tiles = []
for k, (tag, side, s, e) in enumerate(rows, 1):
    cap = cv2.VideoCapture(rf"K:\mini1p_pain_behavior\{side}")
    im_row = []
    for f in np.linspace(s, e - 1, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
        c = cv2.resize(im[40:455, 150:630], (230, 199))
        cv2.putText(c, "%.1f s" % (f / FPS), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        im_row.append(c)
    lb = np.full((199, 150, 3), 255, np.uint8)
    cv2.putText(lb, "G%d" % k, (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (140, 30, 200), 2)
    cv2.putText(lb, tag, (6, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(lb, "%.1f-%.1f s" % (s / FPS, e / FPS), (6, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    tiles.append(np.hstack([lb] + im_row))
for p in range(0, len(tiles), 8):
    cv2.imwrite(OUT + r"\grooming_decision_bouts_p%d.png" % (p // 8 + 1), np.vstack(tiles[p:p + 8]))
print(len(rows), "bouts;", (len(tiles) + 7) // 8, "pages")
