#!/bin/bash
# RUN_pi_video_20261022.sh - the 2-min representative session video for the PI deck (behaviour + Ca), the neurons picked with
# the BEST-window classes (event analysis), re-encoded small enough for PowerPoint. 5941, Day 1 baseline, pin-prick block
# (19 pin-prick + 21 withdrawal UP neurons, no drug).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/analysis/ca_videos_pi
FF="/c/Users/hsollim/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
w() { cygpath -w "$1"; }
mkdir -p $OUT
python -I - <<'EOF'
import pandas as pd
T = pd.read_csv(r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\event_analysis\cells_classes.csv")
T = T[T.day == 1].copy()
T["cls_shift"] = T.cls.map({"activated": "UP", "inhibited": "DOWN"}).fillna("no response")
T["resp_mean"] = T.resp
T[["mouse", "phase", "event", "cell", "cls_shift", "resp_mean"]].to_csv(
    r"C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\analysis\ca_videos_pi\event_tests_bestwindow_d1.csv", index=False)
print("best-window classes written")
EOF
CA_VIDEO_FS=2 python -I $PIPE/ca_behaviour_video.py --day-root "$(w $BASE/mini1p_out/20261019/5941_d1)" --rid 5941_d1_pre \
    --scoring-dir "$(w $BASE/scoring_mini1p/20261019_final)" --out-dir "$(w $OUT)" --auto --dur 120 --stim "Pin prick" --n-cells 6 \
    --event-tests "$(w $OUT/event_tests_bestwindow_d1.csv)" || { echo "*** video failed"; exit 1; }
src=$(ls -t $OUT/5941_d1_pre_behaviour_ca_Pinprick_*s.mp4 | grep -v _small | head -1)
"$FF" -y -loglevel error -i "$(w $src)" -vf "scale=1600:-2" -c:v libx264 -crf 29 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w ${src%.mp4}_small.mp4)" && ls -la "${src%.mp4}_small.mp4"
echo "video done $(date '+%T')"
