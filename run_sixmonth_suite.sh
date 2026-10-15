#!/usr/bin/env bash
# Run the direct-read quantitative suite against one 6-month farm.
# Usage: run_sixmonth_suite.sh <run_dir> <out_sub>
#   run_dir : combined symlink farm (sixmonth_hydro | sixmonth_QH)
#   out_sub : figure subfolder name (sixmonth | sixmonth_QH)
set -u
REPO=/home/edavenport/analysis/tpose24-validation
cd "$REPO"
RUN_DIR=$1; OUT_SUB=$2
export TPOSE_RUN_DIR=$RUN_DIR TPOSE_OUT_SUB=$OUT_SUB TPOSE_DELTAT=60
KERNEL="--ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=tpose"

run_nb () { echo "### nbconvert $1"; conda run -n tpose jupyter nbconvert --to notebook \
  --execute --inplace $KERNEL "$1" || echo "FAILED: $1"; }

echo "===== OSCAR ($OUT_SUB) ====="
run_nb OSCAR_comparison/notebook_obs_oscar_validation.ipynb

echo "===== EDJ caches ($OUT_SUB) ====="
conda run -n tpose python EDJ_interaction/build_edj_cache.py "$OUT_SUB"          || echo "FAILED edj"
conda run -n tpose python EDJ_interaction/build_edj_fulldepth_cache.py "$OUT_SUB" || echo "FAILED edj_full"

echo "===== DONE $OUT_SUB direct-read batch ====="
