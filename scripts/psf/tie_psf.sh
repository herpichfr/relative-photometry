#!/bin/bash
# tie_psf.sh -- after all 13 T80S nights are re-processed (rerun_all.sh): multi-night tie on the PSF relphot dirs, multisearch,
# DB load, then analyze.  Refuses to run if the output stem exists.  Log: $M/${S}_pipeline.log (markers TIE_DONE / TIE_FAIL).
# ANALYZE_ARGS (default "--all --keep-vetted") are the arguments of `relphot db analyze`.
set -uo pipefail
export OMP_NUM_THREADS=2 TMPDIR=/ssdsto1/data/mnt
T=/mnt/sto01/T80S/reduced; M=$T/multinight; S=mn_1104_1209_loose_psf
NIGHTS="20251104 20251105 20251106 20251107 20251112 20251118 20251130 20251201 20251204 20251206 20251207 20251208 20251209"
LOOSE=20251112,20251118,20251130,20251201,20251204,20251206,20251207,20251208,20251209
ANALYZE_ARGS=${ANALYZE_ARGS:---all --keep-vetted}
LOG=$M/${S}_pipeline.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() { say "TIE_FAIL $*"; exit 1; }

[ ! -e "$M/$S.npz" ] || { echo "$M/$S.npz exists; refusing to overwrite" >&2; exit 1; }
for N in $NIGHTS; do
  [ -f "$T/$N/psf/PROVENANCE.txt" ] && [ -f "$T/$N/relphot/lc/night_lc.npz" ] || fail "night $N is not PSF-processed ($T/$N/psf/PROVENANCE.txt or relphot/lc/night_lc.npz missing)"
done
cd "$M" || fail "cd $M"

step() { say "== $*"; local t0=$SECONDS; "$@" >> "$LOG" 2>&1 || fail "$*"; say "   [$((SECONDS - t0)) s]"; }

step relphot multinight $(for N in $NIGHTS; do echo -n "$T/$N/relphot "; done) --labels "$(echo $NIGHTS | tr ' ' ',')" --loose $LOOSE --out "$M/$S"
step relphot multisearch "$M/$S.npz" --out-dir "$M/${S}_search"
step relphot db load-multinight "$M/$S" --search-dir "$M/${S}_search"
# analyze can deadlock against concurrent web review writes: retry up to 5 times, only on DeadlockDetected
for i in 1 2 3 4 5; do
  say "== relphot db analyze $ANALYZE_ARGS (attempt $i)"; t0=$SECONDS
  if relphot db analyze $ANALYZE_ARGS > "$M/${S}_analyze.log" 2>&1; then say "   [$((SECONDS - t0)) s]"; break; fi
  grep -q DeadlockDetected "$M/${S}_analyze.log" || fail "analyze (see ${S}_analyze.log)"
  [ $i -eq 5 ] && fail "analyze, 5 deadlocks"
  say "   deadlock, retry in 60 s"; sleep 60
done
say "TIE_DONE"
