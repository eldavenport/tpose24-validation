# 6-month quantitative suite — TODO / deferred

Run against the two 6-month TPOSE24 records via combined symlink farms:
- hydrostatic : `/data/SO3/edavenport/tpose24/sixmonth_hydro`  → figures in `<check>/sixmonth/`
- quasi-hydro : `/data/SO3/edavenport/tpose24/sixmonth_QH`     → figures in `<check>/sixmonth_QH/`

## Deferred

- **KPP / mixing analysis (TAO notebooks 1b, 1d)** — skipped for now.
  `diag_kpp` output exists only for the Oct–Dec parent of the hydrostatic run
  (`oct2012_dt60_10072026`) and is **absent entirely for QH**
  (`quasi_hydrostatic/*` wrote no `diag_kpp`). A full-window mixing validation
  needs KPP diagnostics re-output (or the mixing notebooks limited to Oct–Dec
  hydro, with nothing for QH). Also `EquatorMix_comparison` (fast-CTD mixing)
  is KPP/ε-dependent and was not run.

## Known coverage caveats (not blockers)

- **OSCAR**: obs granules only cover 2012-10-01 … 2012-12-30, so OSCAR figures
  auto-intersect to the first ~3 months for both runs.
- **Cosmetic**: some figure titles still read "Oct–Dec 2012" / resolution-run
  labels; axes/data are the full 6-month window.
- **vertical_velocity**: run single-model (`HYD6`/`QH6`); the TP6-vs-Ri7
  resolution-comparison cell was dropped (no meaning for a single run). No
  low-res baseline cache was built.
- **Single-model obs notebooks**: the TPOSE6 baseline was dropped from
  GLORYS/OISST/AVISO/OSCAR so the full 6-month window isn't truncated.

## Infrastructure added this round
- `build_sixmonth_links.sh` — builds the combined parent+extension farms.
- `TAO_comparison/sample_mitprof.py` — offline MITprof model-equiv profiles
  (full Oct–Apr, 2012+2013 obs) feeding TAO 1/1c and wave_stacking.
- `run_full_suite.sh <run_dir> <out_sub> <wvel_model>` — env-driven full suite.
- Notebooks/cache scripts parameterized by env: `TPOSE_RUN_DIR`,
  `TPOSE_OUT_SUB`, `TPOSE_DELTAT`, `TPOSE_WVEL_MODEL`.
- Subsampling optimizations: vertical_velocity depth-cap (`READ_DEPTH`),
  EDJ daily subsample (`--full` to override).
