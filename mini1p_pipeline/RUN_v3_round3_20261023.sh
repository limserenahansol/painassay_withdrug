#!/bin/bash
# RUN_v3_round3_20261023.sh - after the round-3 fixes: every new-style deck again into NEW_STYLE_v3 (P2 colour bar on the PI
# slide, P5 brackets / labels, V4 y labels), the pipeline figures, then the final assembly + audit + previews + review set.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
echo "start $(date '+%F %T')"
python -I -X utf8 $PIPE/make_pipeline_figs.py 2>&1 | tail -1
bash $PIPE/RUN_v3_newstyle_20261023.sh 2>&1 | grep -v "^drwx\|^-rw\|^total\|^$" | tail -8
bash $PIPE/RUN_v3_finish_20261023.sh
echo "round3 all done $(date '+%F %T')"
