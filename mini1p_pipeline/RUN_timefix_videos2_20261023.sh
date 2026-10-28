#!/bin/bash
# RUN_timefix_videos2_20261023.sh - both deck videos once more (10-23 afternoon): the behaviour examples now alternate Day 1 /
# Day 2 (user: 'a new video for this slide with the new stimuli and withdrawal marks'; the 10:46 version picked Day 1 for every
# reflex), and the PI video with the shorter title. The 10:46 versions are kept as *_v6_day1picks.*
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
VID=$BASE/analysis/ca_videos_pi
EX=$BASE/analysis/behavior_examples_v2
FF="/c/Users/hsollim/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
w() { cygpath -w "$1"; }
echo "start $(date '+%F %T')"
for f in behaviour_examples_all_small.mp4 behaviour_examples_all_frame.png 5941_d1_pre_behaviour_ca_Pinprick_390s_small.mp4 \
         5941_d1_pre_behaviour_ca_Pinprick_390s_frame.png; do
  [ -f $VID/$f ] && [ ! -f $VID/${f%.*}_v6_day1picks.${f##*.} ] && cp $VID/$f $VID/${f%.*}_v6_day1picks.${f##*.}
done
[ -f $EX/behaviour_examples_all.mp4 ] && [ ! -f $EX/behaviour_examples_all_v6_day1picks.mp4 ] && cp $EX/behaviour_examples_all.mp4 $EX/behaviour_examples_all_v6_day1picks.mp4
python -I -X utf8 $PIPE/behavior_example_clips.py --final-dir "$(w $BASE/scoring_mini1p/20261019_final_timefix)" \
    "$(w $BASE/scoring_mini1p/20261021_final_timefix)" --out-dir "$(w $EX)" --n 3 || { echo "*** clips failed"; exit 1; }
"$FF" -y -loglevel error -i "$(w $EX/behaviour_examples_all.mp4)" -c:v libx264 -crf 27 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w $VID/behaviour_examples_all_small.mp4)" && ls -la $VID/behaviour_examples_all_small.mp4
"$FF" -y -loglevel error -ss 6 -i "$(w $EX/behaviour_examples_all.mp4)" -frames:v 1 "$(w $VID/behaviour_examples_all_frame.png)"
echo "[$(date '+%T')] behaviour examples done"
bash $PIPE/RUN_timefix_pi_video_20261023.sh
echo "all done $(date '+%F %T')"
