#!/bin/bash
# after the complete deck: rebuild the PI main deck and the per-mouse supplement with the plain-result sentences
cd /c/Users/hsollim/Documents/HEAL_mini1p_SBI553
until grep -q "slides | figures\|Traceback" complete_deck.log 2>/dev/null; do sleep 20; done
python -I mini1p_pipeline/make_day1_full_deck.py --base "C:/Users/hsollim/Documents/HEAL_mini1p_SBI553" --out "C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\lab_meeting\Day1_Ca_behaviour_SBI553_PI_summary_20261021.pptx" > pi_deck.log 2>&1
grep -E "slides \||Traceback|Error" pi_deck.log | tail -3
python -I mini1p_pipeline/make_day1_per_mouse_deck.py --base "C:/Users/hsollim/Documents/HEAL_mini1p_SBI553" --out "C:\Users\hsollim\Documents\HEAL_mini1p_SBI553\lab_meeting\Day1_Ca_per_mouse_supplement_v2_20261021.pptx" > per_mouse_v2.log 2>&1
grep -E "slides \||Traceback|Error" per_mouse_v2.log | tail -3
echo "rebuild done $(date '+%T')"
