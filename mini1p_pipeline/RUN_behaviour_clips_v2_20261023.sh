#!/bin/bash
# RUN_behaviour_clips_v2_20261023.sh - the behaviour example video again from BOTH days (16 sessions; the 10-20 one used Day 1
# only), readable timeline, withdrawal / flinch from different deliveries, 4x slower tap replays; then the small deck copy.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
OUT=$BASE/analysis/behavior_examples_v2
VID=$BASE/analysis/ca_videos_pi
FF="/c/Users/hsollim/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
w() { cygpath -w "$1"; }
echo "start $(date '+%T')"
python -I -X utf8 $PIPE/behavior_example_clips.py --final-dir "$(w $BASE/scoring_mini1p/20261019_final)" "$(w $BASE/scoring_mini1p/20261021_final)" \
    --out-dir "$(w $OUT)" --n 3 || { echo "*** clips failed"; exit 1; }
for f in behaviour_examples_all_small.mp4 behaviour_examples_all_frame.png; do
  [ -f $VID/$f ] && [ ! -f $VID/${f%.*}_v1day1.${f##*.} ] && cp $VID/$f $VID/${f%.*}_v1day1.${f##*.}
done
"$FF" -y -loglevel error -i "$(w $OUT/behaviour_examples_all.mp4)" -c:v libx264 -crf 27 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w $VID/behaviour_examples_all_small.mp4)" && ls -la $VID/behaviour_examples_all_small.mp4
"$FF" -y -loglevel error -ss 6 -i "$(w $OUT/behaviour_examples_all.mp4)" -frames:v 1 "$(w $VID/behaviour_examples_all_frame.png)"
python -I -c "import cv2; c=cv2.VideoCapture(r'$(w $VID/behaviour_examples_all_small.mp4)'); print('frames', c.get(7), 'size', c.get(3), c.get(4), 'sec', c.get(7)/30)"
echo "done $(date '+%T')"
