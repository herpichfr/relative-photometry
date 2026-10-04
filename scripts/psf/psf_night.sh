#!/bin/bash
# psf_night.sh TEL NIGHT -- re-process one reduced night with PSF photometry, replacing its aperture-photometry relphot products.
#
# Steps (a marker word per finished step goes to $STATE; a rerun skips finished steps, so a failed night can be resumed):
#   pre -> FIT -> FINALIZE -> MOVE_APERTURE -> SYNC_PSF -> RELPHOT -> SYNC_RELPHOT -> DB_LOAD -> CLEAN
# Fit/finalize/relphot run on the SSD staging dir $STG; the permanent night dir $PERM gets <night>/psf and <night>/relphot.
# The old aperture products of $PERM are MOVED (not deleted) to $SCR.  Exit 0 only after NIGHT_DONE; any failure logs
# "NIGHT_FAIL TEL NIGHT step" and exits 1.
set -uo pipefail

[ $# -eq 2 ] || { echo "usage: $0 TEL NIGHT" >&2; exit 2; }
TEL=$1; NIGHT=$2
REPO=/home/herpich/Dropbox/relative-photometry
CODE=$(dirname "$(readlink -f "$0")")          # = $REPO/scripts/psf
case $TEL in
  T80S)   NP=13; STEP1=7;  STEM=night_lc;    REFARGS="--no-variables"; LCARGS="--no-variables"; SEARCHARGS="" ;;  # STEM must be night_lc: relphot multinight hard-codes lc/night_lc.npz
  ROBO43) NP=12; STEP1=50; STEM=wasp145_psf; REFARGS="--aper 0";       LCARGS="";               SEARCHARGS="--snr-threshold 5.5" ;;
  *) echo "unknown telescope $TEL (T80S or ROBO43)" >&2; exit 2 ;;
esac
PERM=/mnt/sto01/$TEL/reduced/$NIGHT
STG=/ssdsto1/data/${TEL}_reduced/$NIGHT
SCR=/mnt/sto01/scratch/aperture/$TEL/reduced/$NIGHT
LOGD=/ssdsto1/data/${TEL}_reduced/psf_rerun_logs      # outside the staged tree (logging inside it broke the size verification of earlier pipelines)
LOG=$LOGD/$NIGHT.log; STATE=$LOGD/$NIGHT.state; MEMLOG=$LOGD/$NIGHT.mem.log; NFF=$LOGD/$NIGHT.nf
export TMPDIR=/ssdsto1/data/mnt OMP_NUM_THREADS=2      # relphot steps; the fit/finalize commands set OMP_NUM_THREADS=1 explicitly
mkdir -p "$LOGD" || exit 1
exec 9> "$LOGD/$NIGHT.lock"
flock -n 9 || { echo "another psf_night.sh is running for $TEL $NIGHT" >&2; exit 3; }

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
has_marker() { grep -qx "$1" "$STATE" 2>/dev/null; }

# step NAME FUNCTION: skip when the marker is present; run FUNCTION; mark on success, NIGHT_FAIL + exit 1 on failure
step() {
  local name=$1 fn=$2
  if has_marker "$name"; then log "== skip $name (marker present)"; return 0; fi
  log "== $(date +%T) $name"; local t0=$SECONDS
  if "$fn"; then
    echo "$name" >> "$STATE"; log "   [$((SECONDS - t0)) s] $name ok"
  else
    log "   [$((SECONDS - t0)) s] $name FAILED"; log "NIGHT_FAIL $TEL $NIGHT $name"; exit 1
  fi
}

mem_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }

# ---------------------------------------------------------------- RAM gate and guard (every invocation, also on resume)
GUARD_PID=
start_guard() {
  local waited=0
  while [ "$(mem_kb)" -le $((30 * 1024 * 1024)) ]; do
    [ $waited -eq 0 ] && log "waiting for MemAvailable > 30 GB (now $(mem_kb) kB)"
    waited=1; sleep 30
  done
  [ $waited -eq 1 ] && log "MemAvailable > 30 GB, starting"
  (
    exec 9>&-                                   # do not hold the night lock
    while true; do
      sleep 20
      a=$(mem_kb)
      echo "$(date '+%F %T') $a" >> "$MEMLOG"
      if [ "$a" -lt $((4 * 1024 * 1024)) ]; then
        echo "$(date '+%F %T') GUARD_KILL MemAvailable=${a} kB" >> "$LOG"
        pkill -TERM -f "$CODE/run_night.py"
        pkill -TERM -f "relphot (ingest|reference|lightcurves|search)"
      fi
    done
  ) &
  GUARD_PID=$!
}
stop_guard() { [ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null; }
trap stop_guard EXIT
trap 'exit 143' TERM INT HUP

# ---------------------------------------------------------------- steps
step_pre() {
  [ -d "$PERM" ] || { log "$PERM does not exist"; return 1; }
  if [ -d "$PERM/forced" ]; then
    local n; n=$(ls "$PERM"/forced/*_proc_forced_catalog.csv 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { log "no forced catalogues in $PERM/forced"; return 1; }
    if [ -f "$NFF" ] && [ "$(cat "$NFF")" != "$n" ]; then log "frame count changed: $(cat "$NFF") -> $n"; return 1; fi
    echo "$n" > "$NFF"
  else
    [ -f "$NFF" ] || { log "$PERM/forced absent and no stored frame count"; return 1; }
  fi
  log "NF=$(cat "$NFF") forced catalogues"
}

step_fit() {
  mkdir -p "$STG/psf/psf_work/log" || return 1
  OMP_NUM_THREADS=1 PSF_PROD=$PERM PSF_OUT=$STG/psf python "$CODE/run_night.py" "$TEL" "$NIGHT" $NP $STEP1 \
    > "$STG/psf/psf_work/log/run_fit.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { log "run_night.py exit $rc"; tail -n 5 "$STG/psf/psf_work/log/run_fit.log" >> "$LOG"; return 1; }
  local n; n=$(ls "$STG"/psf/psf_work/raw/*.pkl 2>/dev/null | wc -l)
  [ "$n" -eq "$NF" ] || { log "raw pickles $n != NF $NF"; return 1; }
  log "run_night: $(tail -n 1 "$STG/psf/psf_work/log/run_fit.log")"
}

step_finalize() {
  OMP_NUM_THREADS=1 PSF_PROD=$PERM PSF_OUT=$STG/psf python "$CODE/finalize.py" "$TEL" "$NIGHT" \
    > "$STG/psf/psf_work/log/finalize.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { log "finalize.py exit $rc"; tail -n 5 "$STG/psf/psf_work/log/finalize.log" >> "$LOG"; return 1; }
  local nc nl
  nc=$(ls "$STG"/psf/*_proc_forced_catalog.csv 2>/dev/null | wc -l)
  nl=$(find "$STG/psf" -maxdepth 1 -type l -name '*_proc.fits' | wc -l)
  [ "$nc" -eq "$NF" ] && [ "$nl" -eq "$NF" ] || { log "catalogues $nc / links $nl != NF $NF"; return 1; }
  {
    echo "PSF photometry re-processing"
    echo "date:        $(date '+%F %T')"
    echo "git HEAD:    $(git -C "$REPO" rev-parse HEAD)"
    echo "git status:  $(git -C "$REPO" status --porcelain scripts/psf | wc -l) modified/untracked entries under scripts/psf"
    echo "code dir:    $CODE"
    echo "telescope:   $TEL   night: $NIGHT   nproc: $NP   pass1_step: $STEP1"
    echo "input night dir:       $PERM"
    echo "input forced dir:      $PERM/forced (moved to $SCR after FINALIZE)"
    echo "input catalogue dir:   $PERM (*_proc_catalog.csv, moved to $SCR after FINALIZE)"
    echo "static_extras.csv rows: $(tail -n +2 "$STG/psf/psf_work/static_extras.csv" | wc -l)"
  } > "$STG/psf/PROVENANCE.txt"
  log "finalize: $(grep FINALIZE_OK "$STG/psf/psf_work/log/finalize.log")"
}

MOVE_ITEMS="forced relphot relphot_standard_20261002 superseded photometry.log photometry.stdout photometry_status.csv forced.log load_night.log relphot_time.txt relphot_peak_rss_kb run_forced_reanalysis.sh run_forced_reanalysis.log"
step_move_aperture() {
  mkdir -p "$SCR" || return 1
  local item moved="" ncat nscr0 nscr1
  # refuse to overwrite anything already in scratch (items still present in PERM only; items moved by an interrupted earlier run are gone from PERM)
  for item in $MOVE_ITEMS; do
    if [ -e "$PERM/$item" ] || [ -L "$PERM/$item" ]; then
      if [ -e "$SCR/$item" ] || [ -L "$SCR/$item" ]; then log "$SCR/$item already exists"; return 1; fi
    fi
  done
  for item in $MOVE_ITEMS; do
    if [ -e "$PERM/$item" ] || [ -L "$PERM/$item" ]; then
      mv "$PERM/$item" "$SCR/" || { log "mv $item failed"; return 1; }
      moved="$moved $item"
    fi
  done
  ncat=$(find "$PERM" -maxdepth 1 -name '*_proc_catalog.csv' | wc -l)
  nscr0=$(find "$SCR" -maxdepth 1 -name '*_proc_catalog.csv' | wc -l)
  if [ "$ncat" -gt 0 ]; then
    find "$PERM" -maxdepth 1 -name '*_proc_catalog.csv' -exec mv -n -t "$SCR/" {} + || { log "catalogue mv failed"; return 1; }
  fi
  nscr1=$(find "$SCR" -maxdepth 1 -name '*_proc_catalog.csv' | wc -l)
  # verify
  for item in $moved; do
    { [ -e "$SCR/$item" ] || [ -L "$SCR/$item" ]; } || { log "verify: $item missing in $SCR"; return 1; }
  done
  for item in $MOVE_ITEMS; do
    { [ -e "$PERM/$item" ] || [ -L "$PERM/$item" ]; } && { log "verify: $item still in $PERM"; return 1; }
  done
  [ "$(find "$PERM" -maxdepth 1 -name '*_proc_catalog.csv' | wc -l)" -eq 0 ] || { log "verify: aperture catalogues left in $PERM"; return 1; }
  [ $((nscr1 - nscr0)) -eq "$ncat" ] || { log "verify: moved $((nscr1 - nscr0)) catalogues, expected $ncat"; return 1; }
  log "moved items:${moved:- none}; moved $ncat *_proc_catalog.csv; $SCR now holds $nscr1 catalogues"
}

step_sync_psf() {
  [ ! -e "$PERM/psf" ] || { log "$PERM/psf already exists"; return 1; }
  bash "$REPO/deploy/relocate_reduced.sh" --sync "$STG/psf" "$PERM/psf" >> "$LOG" 2>&1 || { log "relocate_reduced.sh (psf) failed"; return 1; }
  local nl=0 l
  for l in "$PERM"/psf/*_proc.fits; do [ -e "$l" ] && nl=$((nl + 1)); done
  [ "$nl" -eq "$NF" ] || { log "resolving proc.fits links in $PERM/psf: $nl != NF $NF"; return 1; }
  log "$nl proc.fits links in $PERM/psf resolve"
}

step_relphot() {
  [ ! -e "$PERM/relphot" ] || { log "$PERM/relphot already exists"; return 1; }
  rm -rf "$STG/relphot"                                   # always start fresh: clip_lc.py would reuse an old lc/preclip/
  mkdir -p "$STG/relphot/lc" && cd "$STG/relphot" || return 1
  ls "$PERM"/psf/*_proc.fits > files_psf.txt              # permanent paths: night.npz and the DB frame.file_path hold them
  [ "$(wc -l < files_psf.txt)" -eq "$NF" ] || { log "files_psf.txt has $(wc -l < files_psf.txt) lines != NF $NF"; return 1; }
  local rc
  rp() { echo "== $(date +%T) $*"; local t0=$SECONDS; "$@"; local r=$?; echo "   [$((SECONDS - t0)) s] exit $r"; [ $r -eq 0 ] || echo "RELPHOT_FAIL: $1 $2"; return $r; }
  {
    rp relphot ingest --photometry forced --out night.npz $(cat files_psf.txt) &&
    rp relphot reference night.npz --out ref.npz $REFARGS &&
    rp relphot lightcurves night.npz ref.npz --out lc/$STEM $LCARGS &&
    rp python "$CODE/clip_lc.py" "$STG/relphot" $STEM 4.0 15 &&
    rp relphot search night.npz ref.npz lc/$STEM.npz --transits-dir lc/susp_transits --variables-dir lc/susp_var --no-plot $SEARCHARGS &&
    echo "RELPHOT_DONE $TEL $NIGHT"
  } > relphot.log 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "relphot chain failed"; tail -n 5 relphot.log >> "$LOG"; return 1; }
  grep -q "ingest: forced photometry" relphot.log || { log "relphot.log lacks 'ingest: forced photometry'"; return 1; }
  local f
  for f in members.npz starstats.parquet lightcurves.parquet search_metrics.parquet; do
    [ -s "lc/${STEM}_$f" ] || { log "missing lc/${STEM}_$f"; return 1; }
  done
}

step_sync_relphot() {
  bash "$REPO/deploy/relocate_reduced.sh" --sync "$STG/relphot" "$PERM/relphot" >> "$LOG" 2>&1 || { log "relocate_reduced.sh (relphot) failed"; return 1; }
}

step_db_load() {
  relphot db load-night "$PERM/relphot" --telescope "$TEL" --label "$NIGHT" --lc-stem "$STEM" > "$PERM/relphot/load_night.log" 2>&1
  local rc=$?
  tail -n 5 "$PERM/relphot/load_night.log" >> "$LOG"
  [ $rc -eq 0 ] || { log "db load-night exit $rc"; return 1; }
}

step_clean() {
  local m
  cd / || return 1
  for m in pre FIT FINALIZE MOVE_APERTURE SYNC_PSF RELPHOT SYNC_RELPHOT DB_LOAD; do
    has_marker "$m" || { log "refusing to clean: marker $m missing"; return 1; }
  done
  case $STG in /ssdsto1/data/*_reduced/[0-9]*) rm -rf "$STG" ;; *) log "refusing to rm -rf odd path $STG"; return 1 ;; esac
}

# ---------------------------------------------------------------- run
log "##### psf_night.sh $TEL $NIGHT (CODE=$CODE NP=$NP STEP1=$STEP1 STEM=$STEM)"
start_guard
step pre step_pre
NF=$(cat "$NFF" 2>/dev/null) || { log "no frame count"; log "NIGHT_FAIL $TEL $NIGHT pre"; exit 1; }
step FIT           step_fit
step FINALIZE      step_finalize
step MOVE_APERTURE step_move_aperture
step SYNC_PSF      step_sync_psf
step RELPHOT       step_relphot
step SYNC_RELPHOT  step_sync_relphot
step DB_LOAD       step_db_load
step CLEAN         step_clean
log "NIGHT_DONE $TEL $NIGHT"
cp "$LOG" "$PERM/psf/psf_rerun.log"
