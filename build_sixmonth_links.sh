#!/usr/bin/env bash
# Build a combined symlink dir for a 6-month run (parent + extension), overlap
# deduped (parent wins, matching extension_validation/ext_utils.daily_index).
# xmitgcm then reads the result as a single dt60 run referenced to 2012-10-01,
# so every validation notebook/script only needs its run_dir repointed here.
#
# Usage: build_sixmonth_links.sh <parent_dir> <extension_dir> <out_dir>
set -eu
par=$1; ext=$2; out=$3
mkdir -p "$out"
# parent first (grid + all its diag files)
for f in "$par"/*.data "$par"/*.meta; do ln -sf "$f" "$out/$(basename "$f")"; done
# extension: only files the parent did not already provide (later iters)
for f in "$ext"/*.data "$ext"/*.meta; do
  b=$(basename "$f"); [ -e "$out/$b" ] || ln -s "$f" "$out/$b"
done
echo "$out:"
for p in diag_state diag_surf diag_kpp; do
  echo "  $p: $(ls "$out/$p".*.data 2>/dev/null | wc -l) files"
done
