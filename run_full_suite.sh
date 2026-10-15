#!/usr/bin/env bash
# Full quantitative suite against one 6-month farm. All notebooks/scripts are
# env-driven (TPOSE_RUN_DIR / TPOSE_OUT_SUB / TPOSE_DELTAT).  Figures land in
# <check>/<out_sub>/.  Videos are NOT included here (run separately, hydro only).
#
# Usage: run_full_suite.sh <run_dir> <out_sub> <wvel_model>
#   e.g. run_full_suite.sh /data/SO3/edavenport/tpose24/sixmonth_QH sixmonth_QH QH6
set -u
REPO=/home/edavenport/analysis/tpose24-validation
cd "$REPO"
RUN_DIR=$1; OUT_SUB=$2; WMODEL=$3
export TPOSE_RUN_DIR=$RUN_DIR TPOSE_OUT_SUB=$OUT_SUB TPOSE_DELTAT=60 TPOSE_WVEL_MODEL=$WMODEL
K="--ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=tpose"
nb () { echo ">>> nbconvert $1"; conda run -n tpose jupyter nbconvert --to notebook \
  --execute --inplace $K "$1" 2>&1 | grep -iE "error|traceback|Writing" | head -3 || echo "FAIL $1"; }
py () { echo ">>> python $*"; conda run -n tpose python "$@" 2>&1 | grep -ivE "warn|Couldn't" | tail -3; }

echo "############ SUITE: $OUT_SUB ($RUN_DIR) ############"

echo "=== offline TAO/wave profiles ==="
py TAO_comparison/sample_mitprof.py "$RUN_DIR" "$RUN_DIR/PROF_eq"

echo "=== obs/reanalysis ==="
nb GLORYS_comparison/notebook_glorys_validation.ipynb
nb OISST_comparison/notebook_obs_oisst_validation.ipynb
nb AVISO_comparison/notebook_obs_aviso_validation.ipynb
nb OSCAR_comparison/notebook_obs_oscar_validation.ipynb

echo "=== TAO 140W ==="
nb TAO_comparison/notebook_1_tao_validation.ipynb
nb TAO_comparison/notebook_1c_tao_validation_spectra.ipynb

echo "=== EDJ ==="
py EDJ_interaction/build_edj_cache.py "$OUT_SUB"
py EDJ_interaction/build_edj_fulldepth_cache.py "$OUT_SUB"
nb EDJ_interaction/notebook_edj_interaction.ipynb

echo "=== wave_stacking ==="
py wave_stacking/build_model_cache.py "$OUT_SUB"
py wave_stacking/build_v_cache.py "$OUT_SUB"
nb wave_stacking/notebook_wave_stacking.ipynb

echo "=== vertical_velocity ==="
py vertical_velocity/build_wvel_cache.py "$WMODEL"
nb vertical_velocity/notebook_wvel_resolution.ipynb

echo "############ DONE: $OUT_SUB ############"
