#!/bin/bash
# RUN_crossover_deck_20261022.sh - the new Day 1 + Day 2 FULL deck (EN + KR, make_crossover_deck.py) once the cohort
# analyses of RUN_fix3 are done (D done), with previews
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
B="C:/Users/hsollim/Documents/HEAL_mini1p_SBI553"
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
L=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_out/20261021/logs
w() { cygpath -w "$1"; }
until grep -q "D done" $L/fix3_20261022.log; do sleep 30; done
echo "[$(date '+%T')] EN"
CA_DAY=2 python -I $PIPE/make_crossover_deck.py --base "$B" --out "$(w $D/EN/Day1+2_FULL.pptx)" --fresh > $D/logs/Day1+2_FULL.log 2>&1 || echo "*** EN failed"
mkdir -p "$D/EN/Day1+2_FULL_figures/previews"; rm -f "$D/EN/Day1+2_FULL_figures/previews/"*.png
python -I $PIPE/pptx_preview.py "$(w $D/EN/Day1+2_FULL.pptx)" "$(w "$D/EN/Day1+2_FULL_figures/previews")" --dpi 60 > /dev/null 2>&1
echo "[$(date '+%T')] KR"
mkdir -p "$D/KR/Day1+2_FULL_KR_figures"; cp "$D/EN/Day1+2_FULL_figures/psth_cache.npz" "$D/KR/Day1+2_FULL_KR_figures/" 2>/dev/null
CA_DAY=2 python -I $PIPE/make_crossover_deck.py --base "$B" --out "$(w $D/KR/Day1+2_FULL_KR.pptx)" --lang kr > $D/logs/Day1+2_FULL_KR.log 2>&1 || echo "*** KR failed"
rm -f "$D/KR/previews/Day1+2_FULL_KR_slide"*.png
python -I $PIPE/pptx_preview.py "$(w $D/KR/Day1+2_FULL_KR.pptx)" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
echo "all done $(date '+%F %T')"
