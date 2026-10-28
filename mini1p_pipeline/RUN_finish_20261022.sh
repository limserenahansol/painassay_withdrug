#!/bin/bash
# RUN_finish_20261022.sh - after RUN_fix3 (old decks) and RUN_new_decks (first pass) are done: the Korean old decks again
# (phrase glossary) and the new-style decks once more (map layout fix), then 'all done'
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
L=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021/logs
until grep -q "all done" $L/fix3_20261022.log 2> /dev/null && grep -q "all done" $L/new_decks.log 2> /dev/null; do sleep 30; done
bash $PIPE/RUN_kr_retranslate_20261022.sh > $L/kr_retranslate.log 2>&1
bash $PIPE/RUN_new_decks_20261022.sh > $L/new_decks_pass2.log 2>&1
python -I $PIPE/ca_confirm_list.py > /dev/null 2>&1
echo "all done $(date '+%F %T')"
