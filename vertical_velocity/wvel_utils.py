"""Shared helpers for the vertical-velocity resolution comparison.

Compares WVEL between the low-resolution TPOSE6 (1/6 deg) assimilation and the
three high-resolution TPOSE24 (1/24 deg) nested runs over Oct-Dec 2012.

Assumptions documented inline:
- diag_state records are packed THETA,SALT,UVEL,VVEL,WVEL,... (float32, big-endian).
  WVEL is record index 4 in both TPOSE6 (6 flds: +DRHODR) and TPOSE24
  (7 flds: +PHIHYD,DRHODR), verified against xmitgcm (exact match).
- WVEL(k) is the flux across the TOP face of cell k, carried on the Zl coordinate.
- Land / missing cells are written as -999 and masked to NaN.
- Coordinate arrays (XC, YC, Zl) are taken from a 2-iteration xmitgcm load so we
  do not reconstruct the spherical-polar grid by hand; the bulk field reads are
  done with direct big-endian memmap offsets for speed and low memory.
"""

import os
import re
import glob
import numpy as np

DT_DTYPE = '>f4'
MISSING = -999.0
REC_WVEL = 4  # record index of WVEL in diag_state for both resolutions

# 0N, 140W
TARGET_LON = 220.0
TARGET_LAT = 0.0
# depth (m, negative down) of the horizontal slice used for the animations
ANIM_DEPTH = -100.0

CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'

# label -> config. 'lowres' flags the single 1/6 deg baseline.
MODELS = {
    'TP6': {
        'label': 'TPOSE6 (1/6°)',
        'dir': '/data/SO3/edavenport/tpose6/diags/sep2012/run_iter14',
        'grid_dir': '/data/SO6/TPOSE_diags/tpose6/grid_6',
        'NX': 1128, 'NY': 336, 'NZ': 66,
        'ref_date': '2012-09-01', 'deltaT': 1200.0, 'iter_step': 72,  # daily
        'lowres': True,
    },
    'Ri3': {
        'label': 'TPOSE24 Ri3 (1/24°)',
        'dir': '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri3',
        'grid_dir': '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri3',
        'NX': 512, 'NY': 384, 'NZ': 138,
        'ref_date': '2012-10-01', 'deltaT': 300.0, 'iter_step': 36,  # 3-hourly
        'lowres': False,
    },
    'Ri5': {
        'label': 'TPOSE24 Ri5 (1/24°)',
        'dir': '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri5',
        'grid_dir': '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month_Ri5',
        'NX': 512, 'NY': 384, 'NZ': 138,
        'ref_date': '2012-10-01', 'deltaT': 300.0, 'iter_step': 36,
        'lowres': False,
    },
    'Ri7': {
        'label': 'TPOSE24 Ri7 (1/24°)',
        'dir': '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons',
        'grid_dir': '/data/SO3/edavenport/tpose24/oct2012_3month_transp_cons',
        'NX': 512, 'NY': 384, 'NZ': 138,
        'ref_date': '2012-10-01', 'deltaT': 300.0, 'iter_step': 36,
        'lowres': False,
    },
}

# order used everywhere: low-res first, then the three high-res runs
MODEL_ORDER = ['TP6', 'Ri3', 'Ri5', 'Ri7']

# drop the first 10 days of every run (initialization shock), matching notebook_6
SPINUP_DAYS = 10.0


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def diag_iters(model):
    """Sorted iteration numbers with a diag_state file present."""
    d = MODELS[model]['dir']
    its = [int(re.search(r'\.(\d+)\.data$', f).group(1))
           for f in glob.glob(os.path.join(d, 'diag_state.*.data'))]
    return np.array(sorted(its))


def iter_times(model, iters):
    """datetime64[s] for each iteration."""
    m = MODELS[model]
    ref = np.datetime64(m['ref_date'])
    secs = (np.asarray(iters) * m['deltaT']).astype('timedelta64[s]')
    return ref + secs


def get_coords(model):
    """1D XC, YC (deg), Zl (m, negative) and the nearest (i, j) to 0N,140W.

    Uses a light 2-iteration xmitgcm load purely for the coordinate arrays.
    """
    from xmitgcm import open_mdsdataset
    m = MODELS[model]
    its = diag_iters(model)[:2].tolist()
    ds = open_mdsdataset(m['dir'], grid_dir=m['grid_dir'], iters=its,
                         prefix=['diag_state'], ref_date=m['ref_date'],
                         delta_t=m['deltaT'])
    xc = np.asarray(ds.XC.astype(float).values)
    yc = np.asarray(ds.YC.astype(float).values)
    xc1d = xc[0, :] if xc.ndim == 2 else xc
    yc1d = yc[:, 0] if yc.ndim == 2 else yc
    zl = np.asarray(ds.Zl.astype(float).values)
    i = int(np.abs(xc1d - TARGET_LON).argmin())
    j = int(np.abs(yc1d - TARGET_LAT).argmin())
    return dict(xc1d=xc1d, yc1d=yc1d, zl=zl, i=i, j=j)


def _base_offset(m):
    return REC_WVEL * m['NZ'] * m['NY'] * m['NX']


def read_field(model, it):
    """Full WVEL field (NZ, NY, NX) for one iteration, NaN-masked.

    Single sequential read of the WVEL record (NFS-friendly); use when both a
    column and a horizontal level are needed from the same file.
    """
    m = MODELS[model]
    path = os.path.join(m['dir'], f'diag_state.{it:010d}.data')
    fld = np.fromfile(path, dtype=DT_DTYPE, count=m['NZ'] * m['NY'] * m['NX'],
                      offset=_base_offset(m) * 4).astype('f8')
    fld = fld.reshape(m['NZ'], m['NY'], m['NX'])
    fld[fld == MISSING] = np.nan
    return fld


def read_column(model, it, i, j):
    """WVEL column (NZ,) at grid point (i, j) for one iteration, NaN-masked."""
    m = MODELS[model]
    path = os.path.join(m['dir'], f'diag_state.{it:010d}.data')
    mm = np.memmap(path, dtype=DT_DTYPE, mode='r')
    base = _base_offset(m)
    idx = base + (np.arange(m['NZ']) * m['NY'] + j) * m['NX'] + i
    col = np.array(mm[idx]).astype('f8')
    col[col == MISSING] = np.nan
    return col


def read_level(model, it, k):
    """WVEL horizontal slice (NY, NX) at level k for one iteration, NaN-masked."""
    m = MODELS[model]
    path = os.path.join(m['dir'], f'diag_state.{it:010d}.data')
    mm = np.memmap(path, dtype=DT_DTYPE, mode='r')
    base = _base_offset(m)
    n = m['NY'] * m['NX']
    lev = np.array(mm[base + k * n: base + (k + 1) * n]).astype('f8').reshape(m['NY'], m['NX'])
    lev[lev == MISSING] = np.nan
    return lev


def nearest_k(zl, depth):
    """Index of the Zl level nearest to `depth` (m, negative)."""
    return int(np.abs(np.asarray(zl) - depth).argmin())


def mean_ci(a, axis=0, conf=1.96):
    """Mean and 95% CI half-width from the standard error along `axis`.

    CI = conf * std / sqrt(N), N = count of finite samples. Returns
    (mean, half_width, std, n). NaNs are ignored per-cell.
    """
    a = np.asarray(a, dtype='f8')
    n = np.sum(np.isfinite(a), axis=axis)
    mean = np.nanmean(a, axis=axis)
    std = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        half = conf * std / np.sqrt(n)
        half = np.where(n > 1, half, np.nan)
    return mean, half, std, n
