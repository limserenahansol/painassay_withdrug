#!/bin/bash
# RUN_timefix_videos_20261023.sh - the two deck videos again from the timing-corrected scoring (10-23): the 2-min PI video
# (neurons picked from the NEW best-window classes) and the behaviour examples (both days; also the 10-23 syncOff fix).
# The current files are kept as *_v5pre_timefix.*; run after RUN_timefix_analyses_20261023.sh (needs cells_classes.csv).
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
VID=$BASE/analysis/ca_videos_pi
EX=$BASE/analysis/behavior_examples_v2
FF="/c/Users/hsollim/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
for f in 5941_d1_pre_behaviour_ca_Pinprick_390s_small.mp4 5941_d1_pre_behaviour_ca_Pinprick_390s_frame.png \
         5941_d1_pre_behaviour_ca_Pinprick_390s.mp4 behaviour_examples_all_small.mp4 behaviour_examples_all_frame.png event_tests_bestwindow_d1.csv; do
  [ -f $VID/$f ] && [ ! -f $VID/${f%.*}_v5pre_timefix.${f##*.} ] && cp $VID/$f $VID/${f%.*}_v5pre_timefix.${f##*.}
done
[ -d ${EX}_v5pre_timefix ] || cp -r $EX ${EX}_v5pre_timefix
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
    --scoring-dir "$(w $BASE/scoring_mini1p/20261019_final_timefix)" --out-dir "$(w $VID)" --auto --dur 120 --stim "Pin prick" --n-cells 6 \
    --event-tests "$(w $VID/event_tests_bestwindow_d1.csv)" || echo "*** PI video failed"
src=$(ls -t $VID/5941_d1_pre_behaviour_ca_Pinprick_*s.mp4 | grep -v "_small\|_v[0-9]" | head -1)
"$FF" -y -loglevel error -i "$(w $src)" -vf "scale=1600:-2" -c:v libx264 -crf 29 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w ${src%.mp4}_small.mp4)" && ls -la "${src%.mp4}_small.mp4"
echo "[$(date '+%T')] PI video done"
python -I -X utf8 $PIPE/behavior_example_clips.py --final-dir "$(w $BASE/scoring_mini1p/20261019_final_timefix)" \
    "$(w $BASE/scoring_mini1p/20261021_final_timefix)" --out-dir "$(w $EX)" --n 3 || { echo "*** clips failed"; exit 1; }
"$FF" -y -loglevel error -i "$(w $EX/behaviour_examples_all.mp4)" -c:v libx264 -crf 27 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w $VID/behaviour_examples_all_small.mp4)" && ls -la $VID/behaviour_examples_all_small.mp4
"$FF" -y -loglevel error -ss 6 -i "$(w $EX/behaviour_examples_all.mp4)" -frames:v 1 "$(w $VID/behaviour_examples_all_frame.png)"
python -I -c "import cv2; c=cv2.VideoCapture(r'$(w $VID/behaviour_examples_all_small.mp4)'); print('frames', c.get(7), 'size', c.get(3), c.get(4), 'sec', c.get(7)/30)"
echo "all done $(date '+%F %T')"
