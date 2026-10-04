#!/bin/bash
# psf_redo.sh TEL NIGHT MODE -- regenerate the PSF products of one night IN PLACE after a code fix, rerun relphot and reload the night into the DB.
# MODE = refit        re-run the PSF fit (pass 1 + static extras + pass 2; the PSFEx models are reused) and finalize  (registration fix)
#      = refinalize   reuse psf_work/ of the current PSF products, re-run finalize only                                (faint-star zero-point fix)
#
# Steps (a marker word per finished step goes to $STATE; a rerun skips finished steps, so a failed night can be resumed):
#   pre -> FIT -> STAGE_WORK -> FINALIZE -> SWAP_PSF -> RELPHOT -> SWAP_RELPHOT -> DB_LOAD -> CLEAN
# FIT runs in refit mode only, STAGE_WORK in refinalize mode only; the step that does not apply to the MODE logs that and still writes its marker.
# Fit inputs: images in $PERM, forced + detection catalogues in the scratch tree $SCRA (forced/*_proc_forced_catalog.csv, forced_reference.csv, *_proc_catalog.csv).
# Fit/finalize/relphot run on the SSD staging dir $STG.  SWAP_PSF / SWAP_RELPHOT MOVE (never delete) the superseded PSF v1 products
# $PERM/psf and $PERM/relphot to $OLD/psf and $OLD/relphot, then rsync the new ones from $STG into place with deploy/relocate_reduced.sh --sync
# (never --db: the DB paths do not change).  The relative <stem>_proc.fits links of $OLD/psf dangle by design (they point to ../<stem>_proc.fits).
# DB_LOAD upserts on source_dir = $PERM/relphot and keeps the night_id, which is recorded in pre ($NIGHT.nid) and re-checked after the load.
# Resume after a failure between SWAP_PSF and SWAP_RELPHOT: each swap writes an extra marker (SWAP_PSF_MV / SWAP_RELPHOT_MV) right after its mv,
# so a rerun does not move again, repeats only the (idempotent) rsync, and pre accepts the missing $PERM/psf resp. $PERM/relphot and the present $OLD dir.
# Exit 0 only after NIGHT_DONE; any failure logs "NIGHT_FAIL TEL NIGHT step" and exits 1.  The only thing ever deleted is $STG (and $STG/relphot) on the SSD.
set -uo pipefail

[ $# -eq 3 ] || { echo "usage: $0 TEL NIGHT MODE   (MODE = refit | refinalize)" >&2; exit 2; }
TEL=$1; NIGHT=$2; MODE=$3
REPO=/home/herpich/Dropbox/relative-photometry
CODE=$(dirname "$(readlink -f "$0")")          # = $REPO/scripts/psf
case $TEL in
  T80S)   NP=13; STEP1=7;  STEM=night_lc;    REFARGS="--no-variables"; LCARGS="--no-variables"; SEARCHARGS="" ;;  # STEM must be night_lc: relphot multinight hard-codes lc/night_lc.npz
  ROBO43) NP=12; STEP1=50; STEM=wasp145_psf; REFARGS="--aper 0";       LCARGS="";               SEARCHARGS="--snr-threshold 5.5" ;;
  *) echo "unknown telescope $TEL (T80S or ROBO43)" >&2; exit 2 ;;
esac
case $MODE in
  refit|refinalize) ;;
  *) echo "unknown mode $MODE (refit or refinalize)" >&2; exit 2 ;;
esac
PERM=/mnt/sto01/$TEL/reduced/$NIGHT
STG=/ssdsto1/data/${TEL}_reduced/$NIGHT
SCRA=/mnt/sto01/scratch/aperture/$TEL/reduced/$NIGHT
OLD=${PSF_REDO_OLD:-/mnt/sto01/scratch/psf_v1}/$TEL/reduced/$NIGHT      # superseded products are moved here (PSF_REDO_OLD overrides the root for a later redo pass)
LOGD=/ssdsto1/data/${TEL}_reduced/psf_redo_logs        # outside the staged tree (logging inside it broke the size verification of earlier pipelines)
LOG=$LOGD/$NIGHT.log; STATE=$LOGD/$NIGHT.state; MEMLOG=$LOGD/$NIGHT.mem.log; NFF=$LOGD/$NIGHT.nf; NIDF=$LOGD/$NIGHT.nid
export TMPDIR=/ssdsto1/data/mnt OMP_NUM_THREADS=2      # relphot steps; the fit/finalize commands set OMP_NUM_THREADS=1 explicitly
mkdir -p "$LOGD" || exit 1
exec 9> "$LOGD/$NIGHT.lock"
flock -n 9 || { echo "another psf_redo.sh is running for $TEL $NIGHT" >&2; exit 3; }

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
has_marker() { grep -qx "$1" "$STATE" 2>/dev/null; }
exists() { [ -e "$1" ] || [ -L "$1" ]; }

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
db_nid() {
  podman exec -i relphotdb-db psql -U postgres -At relphot -c "SELECT night_id FROM relphot.night WHERE source_dir = '$PERM/relphot'"
}

step_pre() {
  [ -d "$PERM" ] || { log "$PERM does not exist"; return 1; }
  local nscra nf nraw nid
  nscra=$(ls "$SCRA"/forced/*_proc_forced_catalog.csv 2>/dev/null | wc -l)
  [ "$nscra" -gt 0 ] || { log "no forced catalogues in $SCRA/forced"; return 1; }
  [ -s "$SCRA/forced/forced_reference.csv" ] || { log "$SCRA/forced/forced_reference.csv missing"; return 1; }
  if has_marker SWAP_PSF_MV; then
    # resume after the psf swap: $PERM/psf may be absent or half-synced, the frame count is the stored one
    [ -f "$NFF" ] || { log "psf swap done but no stored frame count"; return 1; }
    [ "$(cat "$NFF")" = "$nscra" ] || { log "frame count changed: $(cat "$NFF") -> $nscra"; return 1; }
  else
    [ -f "$PERM/psf/PROVENANCE.txt" ] || { log "$PERM/psf/PROVENANCE.txt missing (no PSF products to redo)"; return 1; }
    nf=$(ls "$PERM"/psf/*_proc_forced_catalog.csv 2>/dev/null | wc -l)
    nraw=$(ls "$PERM"/psf/psf_work/raw/*.pkl 2>/dev/null | wc -l)
    { [ "$nf" -eq "$nscra" ] && [ "$nraw" -eq "$nscra" ]; } || { log "frame counts differ: PSF catalogues $nf, raw pickles $nraw, forced in $SCRA $nscra"; return 1; }
    if [ -f "$NFF" ] && [ "$(cat "$NFF")" != "$nf" ]; then log "frame count changed: $(cat "$NFF") -> $nf"; return 1; fi
    echo "$nf" > "$NFF"
  fi
  if ! has_marker SWAP_RELPHOT_MV; then
    [ -f "$PERM/relphot/lc/$STEM.npz" ] || { log "$PERM/relphot/lc/$STEM.npz missing (night not loaded?)"; return 1; }
  fi
  if exists "$OLD/psf" && ! has_marker SWAP_PSF_MV; then log "$OLD/psf already exists"; return 1; fi
  if exists "$OLD/relphot" && ! has_marker SWAP_RELPHOT_MV; then log "$OLD/relphot already exists"; return 1; fi
  if [ -s "$NIDF" ]; then
    nid=$(cat "$NIDF")
  else
    nid=$(db_nid 2>> "$LOG")
  fi
  case $nid in ''|*[!0-9]*) log "no valid DB night_id for $PERM/relphot: '$nid'"; return 1 ;; esac
  echo "$nid" > "$NIDF"
  log "NF=$(cat "$NFF") forced catalogues, mode $MODE, DB night_id=$nid"
}

step_fit() {
  if [ "$MODE" != refit ]; then log "FIT not needed in $MODE mode (psf_work of $PERM/psf is reused)"; return 0; fi
  mkdir -p "$STG/psf/psf_work/psfex" "$STG/psf/psf_work/log" || return 1
  # reuse the PSFEx models; pass 1, static extras and pass 2 are redone (no pass1/ or static_extras.csv is copied)
  local n0 n1
  n0=$(ls "$PERM"/psf/psf_work/psfex/*.psf 2>/dev/null | wc -l)
  [ "$n0" -gt 0 ] || { log "no PSFEx models in $PERM/psf/psf_work/psfex"; return 1; }
  rsync -a --include='*.psf' --exclude='*' "$PERM/psf/psf_work/psfex/" "$STG/psf/psf_work/psfex/" >> "$LOG" 2>&1 || { log "rsync of PSFEx models failed"; return 1; }
  n1=$(ls "$STG"/psf/psf_work/psfex/*.psf 2>/dev/null | wc -l)
  [ "$n1" -eq "$n0" ] || { log "PSFEx models copied: $n1 != $n0 in $PERM"; return 1; }
  # a previous interrupted FIT left $STG/psf: run_night.py resumes by itself (skips existing pickles)
  OMP_NUM_THREADS=1 PSF_PROD=$PERM PSF_OUT=$STG/psf PSF_FORCED=$SCRA/forced PSF_CATDIR=$SCRA python "$CODE/run_night.py" "$TEL" "$NIGHT" $NP $STEP1 \
    > "$STG/psf/psf_work/log/run_fit.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { log "run_night.py exit $rc"; tail -n 5 "$STG/psf/psf_work/log/run_fit.log" >> "$LOG"; return 1; }
  local n; n=$(ls "$STG"/psf/psf_work/raw/*.pkl 2>/dev/null | wc -l)
  [ "$n" -eq "$NF" ] || { log "raw pickles $n != NF $NF"; return 1; }
  log "run_night: $(tail -n 1 "$STG/psf/psf_work/log/run_fit.log")"
}

step_stage_work() {
  if [ "$MODE" != refinalize ]; then log "STAGE_WORK not needed in $MODE mode"; return 0; fi
  mkdir -p "$STG/psf/psf_work" || return 1
  rsync -a "$PERM/psf/psf_work/" "$STG/psf/psf_work/" >> "$LOG" 2>&1 || { log "rsync of psf_work failed"; return 1; }
  mkdir -p "$STG/psf/psf_work/log" || return 1
  local n; n=$(ls "$STG"/psf/psf_work/raw/*.pkl 2>/dev/null | wc -l)
  [ "$n" -eq "$NF" ] || { log "staged raw pickles $n != NF $NF"; return 1; }
  [ -f "$STG/psf/psf_work/static_extras.csv" ] || { log "static_extras.csv missing in staged psf_work"; return 1; }
  log "staged psf_work: $n raw pickles"
}

step_finalize() {
  mkdir -p "$STG/psf/psf_work/log" || return 1
  OMP_NUM_THREADS=1 PSF_PROD=$PERM PSF_OUT=$STG/psf PSF_FORCED=$SCRA/forced python "$CODE/finalize.py" "$TEL" "$NIGHT" \
    > "$STG/psf/psf_work/log/finalize.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || { log "finalize.py exit $rc"; tail -n 5 "$STG/psf/psf_work/log/finalize.log" >> "$LOG"; return 1; }
  local nc nl
  nc=$(ls "$STG"/psf/*_proc_forced_catalog.csv 2>/dev/null | wc -l)
  nl=$(find "$STG/psf" -maxdepth 1 -type l -name '*_proc.fits' | wc -l)
  [ "$nc" -eq "$NF" ] && [ "$nl" -eq "$NF" ] || { log "catalogues $nc / links $nl != NF $NF"; return 1; }
  grep -q FINALIZE_OK "$STG/psf/psf_work/log/finalize.log" || { log "finalize.log lacks FINALIZE_OK"; return 1; }
  {
    echo "PSF photometry re-processing"
    echo "date:        $(date '+%F %T')"
    echo "git HEAD:    $(git -C "$REPO" rev-parse HEAD)"
    echo "git status:  $(git -C "$REPO" status --porcelain scripts/psf | wc -l) modified/untracked entries under scripts/psf"
    echo "code dir:    $CODE"
    echo "telescope:   $TEL   night: $NIGHT   nproc: $NP   pass1_step: $STEP1"
    echo "mode: $MODE"
    echo "supersedes: $OLD/psf"
    echo "input night dir:       $PERM"
    echo "input forced dir: $SCRA/forced"
    echo "input catalogue dir: $SCRA"
    echo "static_extras.csv rows: $(tail -n +2 "$STG/psf/psf_work/static_extras.csv" | wc -l)"
  } > "$STG/psf/PROVENANCE.txt"
  log "finalize: $(grep FINALIZE_OK "$STG/psf/psf_work/log/finalize.log")"
}

step_swap_psf() {
  if ! has_marker SWAP_PSF_MV; then
    [ -d "$PERM/psf" ] || { log "$PERM/psf does not exist"; return 1; }
    exists "$OLD/psf" && { log "$OLD/psf already exists"; return 1; }
    mkdir -p "$OLD" || return 1
    mv "$PERM/psf" "$OLD/psf" || { log "mv of $PERM/psf to $OLD/psf failed"; return 1; }
    echo SWAP_PSF_MV >> "$STATE"
    log "moved $PERM/psf -> $OLD/psf"
  else
    log "SWAP_PSF_MV present: $PERM/psf already moved to $OLD/psf, repeating the sync"
  fi
  [ -d "$OLD/psf" ] || { log "$OLD/psf missing"; return 1; }
  bash "$REPO/deploy/relocate_reduced.sh" --sync "$STG/psf" "$PERM/psf" >> "$LOG" 2>&1 || { log "relocate_reduced.sh (psf) failed"; return 1; }
  local nl=0 l
  for l in "$PERM"/psf/*_proc.fits; do [ -e "$l" ] && nl=$((nl + 1)); done
  [ "$nl" -eq "$NF" ] || { log "resolving proc.fits links in $PERM/psf: $nl != NF $NF"; return 1; }
  grep -qx "mode: $MODE" "$PERM/psf/PROVENANCE.txt" || { log "$PERM/psf/PROVENANCE.txt lacks 'mode: $MODE'"; return 1; }
  log "$nl proc.fits links in $PERM/psf resolve; PROVENANCE mode: $MODE"
}

step_relphot() {
  rm -rf "$STG/relphot"                                   # always start fresh: clip_lc.py would reuse an old lc/preclip/
  mkdir -p "$STG/relphot/lc" && cd "$STG/relphot" || return 1
  ls "$PERM"/psf/*_proc.fits > files_psf.txt              # permanent paths (the NEW catalogues are already in place): night.npz and the DB frame.file_path hold them
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

step_swap_relphot() {
  if ! has_marker SWAP_RELPHOT_MV; then
    [ -d "$PERM/relphot" ] || { log "$PERM/relphot does not exist"; return 1; }
    exists "$OLD/relphot" && { log "$OLD/relphot already exists"; return 1; }
    mkdir -p "$OLD" || return 1
    mv "$PERM/relphot" "$OLD/relphot" || { log "mv of $PERM/relphot to $OLD/relphot failed"; return 1; }
    echo SWAP_RELPHOT_MV >> "$STATE"
    log "moved $PERM/relphot -> $OLD/relphot"
  else
    log "SWAP_RELPHOT_MV present: $PERM/relphot already moved to $OLD/relphot, repeating the sync"
  fi
  [ -d "$OLD/relphot" ] || { log "$OLD/relphot missing"; return 1; }
  bash "$REPO/deploy/relocate_reduced.sh" --sync "$STG/relphot" "$PERM/relphot" >> "$LOG" 2>&1 || { log "relocate_reduced.sh (relphot) failed"; return 1; }
  [ -f "$PERM/relphot/lc/$STEM.npz" ] || { log "$PERM/relphot/lc/$STEM.npz missing after the sync"; return 1; }
}

step_db_load() {
  relphot db load-night "$PERM/relphot" --telescope "$TEL" --label "$NIGHT" --lc-stem "$STEM" > "$PERM/relphot/load_night.log" 2>&1
  local rc=$?
  tail -n 5 "$PERM/relphot/load_night.log" >> "$LOG"
  [ $rc -eq 0 ] || { log "db load-night exit $rc"; return 1; }
  local nid; nid=$(db_nid 2>> "$LOG")
  [ "$nid" = "$(cat "$NIDF")" ] || { log "DB night_id changed by the load: before $(cat "$NIDF"), now '$nid'"; return 1; }
  log "DB night_id $nid unchanged"
}

step_clean() {
  local m
  cd / || return 1
  for m in pre FIT STAGE_WORK FINALIZE SWAP_PSF RELPHOT SWAP_RELPHOT DB_LOAD; do
    has_marker "$m" || { log "refusing to clean: marker $m missing"; return 1; }
  done
  case $STG in /ssdsto1/data/*_reduced/[0-9]*) rm -rf "$STG" ;; *) log "refusing to rm -rf odd path $STG"; return 1 ;; esac
}

# ---------------------------------------------------------------- run
log "##### psf_redo.sh $TEL $NIGHT $MODE (CODE=$CODE NP=$NP STEP1=$STEP1 STEM=$STEM)"
start_guard
step pre step_pre
NF=$(cat "$NFF" 2>/dev/null) || { log "no frame count"; log "NIGHT_FAIL $TEL $NIGHT pre"; exit 1; }
step FIT           step_fit
step STAGE_WORK    step_stage_work
step FINALIZE      step_finalize
step SWAP_PSF      step_swap_psf
step RELPHOT       step_relphot
step SWAP_RELPHOT  step_swap_relphot
step DB_LOAD       step_db_load
step CLEAN         step_clean
log "NIGHT_DONE $TEL $NIGHT"
cp "$LOG" "$PERM/psf/psf_redo.log"
