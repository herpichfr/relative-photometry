#!/bin/bash
# rerun_all.sh [TEL:NIGHT ...] -- PSF re-processing of all reduced nights, strictly one at a time, chronologically
# (so the DB night_ids follow date order).  With arguments: only the listed TEL:NIGHT, still in this order.
# Stops at the first failing night.  Per-night logs/markers: /ssdsto1/data/<TEL>_reduced/psf_rerun_logs/<NIGHT>.{log,state}
# (a rerun of this script resumes a failed night; finished nights are skipped step by step).  Log: /ssdsto1/data/psf_rerun_all.log
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LOG=/ssdsto1/data/psf_rerun_all.log
ALL="ROBO43:20250911 T80S:20251104 T80S:20251105 T80S:20251106 T80S:20251107 T80S:20251112 T80S:20251118 T80S:20251130 T80S:20251201 T80S:20251204 T80S:20251206 T80S:20251207 T80S:20251208 T80S:20251209"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

JOBS=
if [ $# -gt 0 ]; then
  for a in "$@"; do
    case " $ALL " in *" $a "*) ;; *) echo "unknown TEL:NIGHT '$a' (known: $ALL)" >&2; exit 2 ;; esac
  done
  for j in $ALL; do case " $* " in *" $j "*) JOBS="$JOBS $j" ;; esac; done
else
  JOBS=$ALL
fi

say "RERUN_START:$JOBS"
for j in $JOBS; do
  TEL=${j%%:*}; NIGHT=${j##*:}
  say "NIGHT_START $TEL $NIGHT"
  if bash "$HERE/psf_night.sh" "$TEL" "$NIGHT" >> "$LOG" 2>&1; then
    say "NIGHT_OK $TEL $NIGHT"
  else
    say "RERUN_FAIL $TEL $NIGHT"; exit 1
  fi
done
say "RERUN_NIGHTS_DONE"
