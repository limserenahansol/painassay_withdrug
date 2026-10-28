#!/bin/bash
# RUN_kr_only_20261022.sh <deck> ... - the Korean versions of the old-style decks again (after new translations were added
# to translations\kr_part*.json): merge the parts, translate EN/<deck>.pptx (or EN/<deck>_new.pptx if that is newer), write
# KR/<deck>_KR.pptx - or KR/<deck>_KR_new.pptx when the KR file is open in PowerPoint - and re-render its previews.
PIPE=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/mini1p_pipeline
D=/c/Users/hsollim/Documents/HEAL_mini1p_SBI553/lab_meeting/decks_Day1_Day2
w() { cygpath -w "$1"; }
target() {
  local f=$1
  if [ -f "$f" ] && ! ( : >> "$f" ) 2> /dev/null; then echo "${f%.pptx}_new.pptx"; else echo "$f"; fi
}
python -I $PIPE/translations/merge.py
for name in "$@"; do
  en=$D/EN/$name.pptx
  [ -f $D/EN/${name}_new.pptx ] && [ $D/EN/${name}_new.pptx -nt $en ] && en=$D/EN/${name}_new.pptx
  out=$(target $D/KR/${name}_KR.pptx)
  echo "[$(date '+%T')] $name -> $(basename $out)"
  python -I $PIPE/pptx_translate.py --in "$(w $en)" --out "$(w $out)" --dump "$(w $D/logs/missing_kr_$name.json)" | head -1
  rm -f $D/KR/previews/$(basename ${out%.pptx})_slide*.png
  python -I $PIPE/pptx_preview.py "$(w $out)" "$(w $D/KR/previews)" --dpi 60 > /dev/null 2>&1
done
echo "all done $(date '+%F %T')"
