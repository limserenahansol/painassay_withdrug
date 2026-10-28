#!/bin/bash
# RUN_kr_retranslate_20261022.sh - the Korean versions of the old-style decks again (the phrase glossary of 10-22 now
# translates the data-dependent sentences too); a KR deck you have open in PowerPoint is not touched: <deck>_KR_new.pptx
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
w() { cygpath -w "$1"; }
for name in Day1_PI_summary Day1_per_mouse Day1_FULL Day2_PI_summary Day2_per_mouse Day2_FULL Day1+2_longitudinal; do
  [ -f "$D/EN/$name.pptx" ] || continue
  f="$D/KR/${name}_KR.pptx"
  if [ -f "$f" ] && ! ( : >> "$f" ) 2> /dev/null; then f="$D/KR/${name}_KR_new.pptx"; fi
  echo "[$(date '+%T')] $(basename "$f")"
  python -I $PIPE/pptx_translate.py --in "$(w "$D/EN/$name.pptx")" --out "$(w "$f")" --dump "$(w $D/logs/missing_kr_$name.json)" | head -1
  stem=$(basename "$f" .pptx)
  rm -f "$D/KR/previews/${stem}_slide"*.png
  python -I $PIPE/pptx_preview.py "$(w "$f")" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
done
echo "all done $(date '+%F %T')"
