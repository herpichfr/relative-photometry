#!/bin/bash
# redo_all.sh [TEL:NIGHT ...] -- in-place PSF redo (psf_redo.sh) of all 14 nights, strictly one at a time; the two re-fits first, then the re-finalizes.
# With arguments: only the listed TEL:NIGHT, still in this order and with the mode listed below.
# Stops at the first failing night.  Per-night logs/markers: /ssdsto1/data/<TEL>_reduced/psf_redo_logs/<NIGHT>.{log,state}
# (a rerun of this script resumes a failed night; finished nights are skipped step by step).  Log: /ssdsto1/data/psf_redo_all.log
# Does NOT run the multinight tie: do that by hand afterwards.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LOG=/ssdsto1/data/psf_redo_all.log
ALL="T80S:20251107:refit T80S:20251207:refit ROBO43:20250911:refinalize T80S:20251104:refinalize T80S:20251105:refinalize T80S:20251106:refinalize T80S:20251112:refinalize T80S:20251118:refinalize T80S:20251130:refinalize T80S:20251201:refinalize T80S:20251204:refinalize T80S:20251206:refinalize T80S:20251208:refinalize T80S:20251209:refinalize"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

KEYS=
for j in $ALL; do KEYS="$KEYS ${j%:*}"; done          # TEL:NIGHT without the mode

JOBS=
if [ $# -gt 0 ]; then
  for a in "$@"; do
    case "$KEYS " in *" $a "*) ;; *) echo "unknown TEL:NIGHT '$a' (known:$KEYS)" >&2; exit 2 ;; esac
  done
  for j in $ALL; do case " $* " in *" ${j%:*} "*) JOBS="$JOBS $j" ;; esac; done
else
  JOBS=$ALL
fi

say "REDO_START:$JOBS"
for j in $JOBS; do
  MODE=${j##*:}; key=${j%:*}; TEL=${key%%:*}; NIGHT=${key##*:}
  say "NIGHT_START $TEL $NIGHT $MODE"
  if bash "$HERE/psf_redo.sh" "$TEL" "$NIGHT" "$MODE" >> "$LOG" 2>&1; then
    say "NIGHT_OK $TEL $NIGHT"
  else
    say "REDO_FAIL $TEL $NIGHT"; exit 1
  fi
done
say "REDO_NIGHTS_DONE"
