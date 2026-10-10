"""
ext_utils.py
------------
Helpers for validating the Jan-Apr 2013 TPOSE24 extension (`jan2013_from_pickup`)
concatenated with its Oct-Dec 2012 parent (`oct2012_3mo_dt60_AB3`).

Both runs are dt=60 s, 3-hourly diag output, ref 2012-10-01.  The parent
(oct2012_dt60_10072026) is the full 0..92-day run and covers iters 180..132480
(day 0.12..92.0 = Jan 1 2013); the extension restarted from its iter-129600
pickup and is bit-identical to it in the day-90..92 overlap, continuing to iter
262080 (day 182.0 = Apr 1 2013).  A calendar-day axis is built by taking the
first output of each day across both directories (overlap days dedup to the
parent).  NB: oct2012_3mo_dt60_AB3 is a *different* dt60 run that stopped early
at iter 129240 (day 89.75) -- NOT the extension's parent.

Field reads use direct big-endian memmap offsets (as in drift_test/drift_utils.py):
diag_state packs THETA,SALT,UVEL,VVEL,WVEL,PHIHYD,DRHODR (float32); diag_surf
packs ETAN,KPPhbl.  Land/missing = -999 -> NaN.
"""
import os, re, glob
import numpy as np

DT = '>f4'; MISS = -999.0
NZ, NY, NX = 138, 384, 512
NFLDS_STATE = 7
REC = {'THETA': 0, 'SALT': 1, 'UVEL': 2, 'VVEL': 3, 'WVEL': 4}

PAR_DIR = '/data/SO3/edavenport/tpose24/oct2012_dt60_10072026'  # full 0-92d parent
NEW_DIR = '/data/SO3/edavenport/tpose24/jan2013_from_pickup'
REF_DATE = np.datetime64('2012-10-01')
DELTA_T = 60.0
CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'


def _iters(d, prefix='diag_state'):
    its = [int(re.search(r'\.(\d+)\.data$', f).group(1))
           for f in glob.glob(os.path.join(d, f'{prefix}.*.data'))]
    return np.array(sorted(its))


def daily_index():
    """(dirs, iters, dates) with one entry per calendar day, parent then extension.

    Returns parallel arrays: dir path, iteration number, and datetime64[D] for
    the first output of each calendar day over the full Oct 2012 - Apr 2013 record.
    """
    rows = []
    for d in (PAR_DIR, NEW_DIR):
        its = _iters(d)
        secs = (its * DELTA_T).astype('timedelta64[s]')
        dates = (REF_DATE + secs).astype('datetime64[D]')
        _, first = np.unique(dates, return_index=True)
        for i in first:
            rows.append((d, int(its[i]), dates[i]))
    rows.sort(key=lambda r: r[2])
    # dedup calendar day (extension day 90 may repeat parent's last partial day)
    seen, out = set(), []
    for d, it, dt in rows:
        if dt in seen:
            continue
        seen.add(dt); out.append((d, it, dt))
    dirs = np.array([r[0] for r in out])
    iters = np.array([r[1] for r in out])
    dates = np.array([r[2] for r in out], dtype='datetime64[D]')
    return dirs, iters, dates


def read_levels(run_dir, it, var, nk):
    """Top ``nk`` levels (nk, NY, NX) of ``var`` from diag_state at iteration it."""
    path = os.path.join(run_dir, f'diag_state.{it:010d}.data')
    off = REC[var] * NZ * NY * NX * 4
    a = np.fromfile(path, dtype=DT, count=nk * NY * NX, offset=off).astype('f8')
    a = a.reshape(nk, NY, NX)
    a[a == MISS] = np.nan
    return a


def read_surf(run_dir, it, var):
    """Surface level (NY, NX) of a diag_state ``var``."""
    return read_levels(run_dir, it, var, 1)[0]


def read_etan(run_dir, it):
    """ETAN (NY, NX) from diag_surf record 0."""
    path = os.path.join(run_dir, f'diag_surf.{it:010d}.data')
    a = np.fromfile(path, dtype=DT, count=NY * NX, offset=0).astype('f8').reshape(NY, NX)
    a[a == MISS] = np.nan
    return a


def load_grid():
    """1-D lon/lat/Z, drF, and 3-D hFacC (float64) from a 1-iteration xmitgcm load."""
    from xmitgcm import open_mdsdataset
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = open_mdsdataset(NEW_DIR, iters=[_iters(NEW_DIR)[0]],
                             prefix=['diag_state'], ref_date='2012-10-01',
                             delta_t=DELTA_T)
    xc = np.asarray(ds.XC.astype(float).values)
    yc = np.asarray(ds.YC.astype(float).values)
    return dict(
        lon=xc[0, :] if xc.ndim == 2 else xc,
        lat=yc[:, 0] if yc.ndim == 2 else yc,
        z=np.asarray(ds.Z.astype(float).values),
        drF=np.asarray(ds.drF.astype(float).values),
        hFacC=np.asarray(ds.hFacC.astype(float).values),
    )


def upper_mean(field3d, grid, depth=100.0):
    """drF*hFacC-weighted average of a (nk,NY,NX) field over the top ``depth`` m."""
    kz = np.where(grid['z'] >= -depth)[0]
    nk = field3d.shape[0]
    kz = kz[kz < nk]
    dz = grid['drF'][kz][:, None, None] * grid['hFacC'][kz]
    sub = field3d[kz]
    num = np.nansum(np.where(np.isfinite(sub), sub * dz, 0.0), axis=0)
    den = np.nansum(np.where(np.isfinite(sub), dz, 0.0), axis=0)
    out = np.full(num.shape, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out
