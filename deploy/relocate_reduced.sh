#!/usr/bin/env bash
# Move a reduced-data tree from the SSD staging area to permanent storage and point the results DB at it.
#
#   deploy/relocate_reduced.sh [--sync] [--db] FROM TO
#
# FROM, TO: a whole tree   /ssdsto1/data/T80S_reduced           /mnt/sto01/T80S/reduced
#           or one night   /ssdsto1/data/T80S_reduced/20251208  /mnt/sto01/T80S/reduced/20251208
#
# --sync  rsync -a FROM/ into TO/ first (omit it when the tree was already copied).
# Always: verifies by size that every file and link of FROM exists in TO (aborts otherwise);
#         retargets symlinks in TO that point into FROM to relative links inside TO;
#         rewrites FROM -> TO in TO's */relphot/files*.txt lists.
# --db    backs up the DB (deploy/backup.sh), then in one transaction rewrites the FROM prefix to TO in
#         relphot.night.source_dir, relphot.frame.file_path and relphot.mn_run.stem.
# FROM is never modified or deleted: remove it yourself once the result is checked.
set -euo pipefail
SYNC=0; DB=0
while [[ $# -gt 0 && $1 == --* ]]; do
  case $1 in
    --sync) SYNC=1 ;;
    --db) DB=1 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
[[ $# -eq 2 ]] || { echo "usage: $0 [--sync] [--db] FROM TO" >&2; exit 2; }
FROM=$(realpath -e "$1"); TO=$(realpath -m "$2")
[[ $FROM != "$TO" ]] || { echo "FROM and TO are the same directory" >&2; exit 2; }

if (( SYNC )); then
  mkdir -p "$TO"
  rsync -a --info=stats1 "$FROM/" "$TO/"
fi
[[ -d $TO ]] || { echo "$TO does not exist (use --sync)" >&2; exit 1; }

TMP=$(mktemp -d -p "${TMPDIR:-/ssdsto1/data/mnt}" relocate.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

# 1. verify: type, relative path and size of every file and link (link size and file-list size ignored:
#    those are rewritten below)
list() {
  (cd "$1" && find . \( -type f -o -type l \) -printf '%y\t%P\t%s\n') \
    | awk -F'\t' -v OFS='\t' '$1 == "l" || $2 ~ /(^|\/)relphot\/files[^\/]*\.txt$/ { $3 = "" } { print }' \
    | LC_ALL=C sort
}
list "$FROM" > "$TMP/from"
list "$TO" > "$TMP/to"
LC_ALL=C comm -23 "$TMP/from" "$TMP/to" > "$TMP/missing"
if [[ -s $TMP/missing ]]; then
  echo "ABORT: $(wc -l < "$TMP/missing") files/links of $FROM are missing from $TO or differ in size, e.g.:" >&2
  head -20 "$TMP/missing" >&2
  exit 1
fi
echo "verified: all $(wc -l < "$TMP/from") files/links of $FROM are in $TO"

# 2. symlinks into FROM -> relative links inside TO
n=0
while IFS= read -r -d '' link; do
  tgt=$(readlink "$link")
  case $tgt in
    "$FROM"/*) ln -sfnr "$TO/${tgt#"$FROM"/}" "$link"; n=$((n + 1)) ;;
  esac
done < <(find "$TO" -type l -print0)
echo "retargeted $n symlinks"

# 3. file lists written by the processing scripts
find "$TO" -path '*/relphot/files*.txt' -type f -print0 | xargs -0 -r sed -i "s#$FROM/#$TO/#g"

# 4. results DB
if (( DB )); then
  bash "$(dirname "$(realpath "$0")")/backup.sh"
  podman exec -i relphotdb-db psql -U postgres -v ON_ERROR_STOP=1 -v from="$FROM/" -v to="$TO/" relphot <<'SQL'
BEGIN;
UPDATE relphot.night  SET source_dir = :'to' || substr(source_dir, length(:'from') + 1) WHERE starts_with(source_dir, :'from');
UPDATE relphot.frame  SET file_path  = :'to' || substr(file_path,  length(:'from') + 1) WHERE starts_with(file_path,  :'from');
UPDATE relphot.mn_run SET stem       = :'to' || substr(stem,       length(:'from') + 1) WHERE starts_with(stem,       :'from');
COMMIT;
SQL
fi
echo "done; $FROM is untouched -- remove it yourself once $TO is checked"
