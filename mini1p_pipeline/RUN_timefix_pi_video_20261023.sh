#!/bin/bash
# RUN_timefix_pi_video_20261023.sh - the 2-min PI video again (10-23: shorter title so it is not cut off; shade = the 0-0.5 s
# stimulus response window). Uses the classes already written to event_tests_bestwindow_d1.csv by RUN_timefix_videos_20261023.sh.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
BASE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553
VID=$BASE/analysis/ca_videos_pi
FF="/c/Users/hsollim/Downloads/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared/bin/ffmpeg.exe"
w() { cygpath -w "$1"; }
CA_VIDEO_FS=2 python -I $PIPE/ca_behaviour_video.py --day-root "$(w $BASE/mini1p_out/20261019/5941_d1)" --rid 5941_d1_pre \
    --scoring-dir "$(w $BASE/scoring_mini1p/20261019_final_timefix)" --out-dir "$(w $VID)" --auto --dur 120 --stim "Pin prick" --n-cells 6 \
    --event-tests "$(w $VID/event_tests_bestwindow_d1.csv)" || { echo "*** PI video failed"; exit 1; }
src=$VID/5941_d1_pre_behaviour_ca_Pinprick_390s.mp4
"$FF" -y -loglevel error -i "$(w $src)" -vf "scale=1600:-2" -c:v libx264 -crf 29 -preset medium -pix_fmt yuv420p -movflags +faststart \
    "$(w ${src%.mp4}_small.mp4)" && ls -la "${src%.mp4}_small.mp4"
echo "PI video done $(date '+%T')"
