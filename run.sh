#!/usr/bin/env bash
# Stages of the study. Every stage is resumable: finished seeds and runs are skipped.
set -euo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-$(nproc)}"
stage="${1:-help}"
shift || true

figures_es1() {  # $1 = run name
  cd figures
  local lags
  lags=$(python -c "import json,os,sys;r=os.environ.get('FLOWSMOOTHER_RESULTS','../results');q='/quick' if os.environ.get('QUICK')=='1' else '';print(*json.load(open(f'{r}{q}/es1/$1/config.json'))['lags'])")
  for lag in $lags; do
    python f00_particle_row.py --run "$1" --lag "$lag"
    python f00_particle_row.py --run "$1" --lag "$lag" --kind quiet
    python f00_particle_row.py --run "$1" --lag "$lag" --struct square
    python f03_sparsity.py --run "$1" --lag "$lag"
    python f04_crosstime_correlation.py --run "$1" --lag "$lag"
    python f05_error_maps.py --run "$1" --lag "$lag"
    python f11_prior_sensitivity.py --run "$1" --lag "$lag" || true
    python f12_heatmap.py --run "$1" --lag "$lag"
  done
  for f in f01_wakes_on_flow f02_wake_length f06_paired_gain_square f07_paired_gain_patch \
           f08_methods_boxes f09_density_curves f10_gain_by_wake_length; do
    python "$f.py" --run "$1"
  done
  cd ..
}

figures_es2() {
  cd figures
  python f13_cycled_rmse.py --run "$1"
  python f13_cycled_rmse.py --run "$1" --metric an_h
  python f14_spread_error.py --run "$1"
  python f15_wake_animation.py --run "$1" || true
  cd ..
}

case "$stage" in
  test)     python -m pytest -q "$@" ;;
  clim)     cd experiments && python es0_climatology.py "$@" ;;
  spinup)   cd experiments && python es0_spinup_check.py --workers "$W" "$@" ;;
  es1)      # ./run.sh es1 pilot   |   ./run.sh es1 full
    preset="${1:-pilot}"; shift || true
    cd experiments
    python es1_first_analysis.py --preset "$preset" --seed-workers "$W" --model-workers 1 "$@"
    python summarize_es1.py --run "$preset"
    python verify_predecessors.py --run "$preset" ;;
  es2)      # ./run.sh es2 pilot   |   ./run.sh es2 full
    preset="${1:-pilot}"; shift || true
    cd experiments && python es2_cycled.py --preset "$preset" --workers "$W" "$@" ;;
  figures)  # ./run.sh figures es1 pilot
    kind="${1:-es1}"; name="${2:-pilot}"
    if [ "$kind" = es1 ]; then figures_es1 "$name"; else figures_es2 "$name"; fi ;;
  pilot)    # the first reading of the hypothesis, end to end
    [ -f results/clim/record.npy ] || "$0" clim
    "$0" es1 pilot && "$0" figures es1 pilot ;;
  full)
    [ -f results/clim/record.npy ] || "$0" clim
    "$0" es1 full && "$0" figures es1 full ;;
  cycled)
    [ -f results/clim/record.npy ] || "$0" clim
    "$0" es2 "${1:-pilot}" && "$0" figures es2 "${1:-pilot}" ;;
  quick)    # everything at toy size, to check the installation (results/quick)
    export QUICK=1
    python -m pytest -q
    (cd experiments && python es0_climatology.py && python es0_spinup_check.py --workers "$W" \
      && python es1_first_analysis.py --seed-workers "$W" --model-workers 1 \
      && python summarize_es1.py && python verify_predecessors.py --run quick \
      && python es2_cycled.py --workers "$W")
    figures_es1 quick && figures_es2 quick ;;
  shell)    exec bash ;;
  *)
    cat <<'TXT'
usage: run.sh <stage> [args]      (WORKERS=<n> sets the parallelism; default all cores)

  test             unit tests (predecessors, wakes, rows, analyses)
  quick            everything at toy size, about 9 minutes on 2 cores, into results/quick
  clim             climatological record with pairs (60 + 800 model days)
  spinup           how long the adjustment runs must be
  pilot            E-S1 pilot (5 seeds, N = 20, 10 %, lags 6/12/24 h) + tables + figures
  full             E-S1 complete (20 seeds, N = 20/40, four densities, r = 2/3) + figures
  cycled [preset]  E-S2 (pilot | full) + figures
  es1 <preset> [args]   es2 <preset> [args]   figures <es1|es2> <run>   shell
TXT
    ;;
esac
