"""
run_qh_drift.py
---------------
Run the exact Oct 2012 - Apr 2013 drift validation that was done for the
standard (hydrostatic) 6-month TPOSE24 record, but on the quasi-hydrostatic
(QH) pair:

    parent     oct2012_QH               (Oct-Dec 2012)
    extension  jan2013_QH_from_pickup   (Jan-Apr 2013)

It reuses extension_validation/{sample_profiles,build_ext_cache,plot_ext}.py
unchanged -- those modules key everything off a few module-level path globals,
so we just repoint them at the QH runs and at this folder for output.  Same
obs/reanalysis references (GLORYS12, OISST, TAO 140 degW) as the standard run,
so the figures here drop straight next to extension_validation/ for a
QH-vs-hydrostatic comparison.

Outputs (this folder):
    cache/ext_valid_cache.nc           daily GLORYS/OISST misfit time series
    ext_*.png                          same figure set as extension_validation/
    <QH_JAN>/PROF_eq/*_model.nc        offline TAO model-equivalent profiles

Usage:  python run_qh_drift.py [--test N]   (N limits the misfit loop for a smoke test)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), 'extension_validation')
sys.path.insert(0, EXT)

import ext_utils as e
import sample_profiles
import build_ext_cache
import plot_ext

QH_OCT = '/data/SO3/edavenport/tpose24/quasi_hydrostatic/oct2012_QH'
QH_JAN = '/data/SO3/edavenport/tpose24/quasi_hydrostatic/jan2013_QH_from_pickup'
QH_PROF_EQ = os.path.join(QH_JAN, 'PROF_eq')
QH_CACHE = os.path.join(HERE, 'cache')

# repoint the run dirs (functions resolve these ext_utils globals at call time)
e.PAR_DIR = QH_OCT
e.NEW_DIR = QH_JAN
e.CACHE_DIR = QH_CACHE

# offline-profile output goes in the QH extension run dir
sample_profiles.OUT_DIR = QH_PROF_EQ

# figure + cache locations for the plotter
plot_ext.HERE = HERE
plot_ext.CACHE = os.path.join(QH_CACHE, 'ext_valid_cache.nc')
plot_ext.PROF_EQ = QH_PROF_EQ


def main(test=None):
    print('=== sampling QH TAO model-equivalent profiles ===')
    sample_profiles.main()
    print('=== building QH GLORYS/OISST misfit cache ===')
    build_ext_cache.main(test=test)
    print('=== plotting QH drift figures ===')
    plot_ext.main()


if __name__ == '__main__':
    t = int(sys.argv[sys.argv.index('--test') + 1]) if '--test' in sys.argv else None
    main(test=t)
