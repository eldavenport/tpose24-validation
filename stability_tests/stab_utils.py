"""Shared helpers for the dt=300 vs dt=60 stability test.

Assumptions documented inline:
- diag_state records are packed as THETA,SALT,UVEL,VVEL,WVEL,PHIHYD,DRHODR (float32,
  big-endian), so UVEL/VVEL/WVEL are records 2/3/4.
- WVEL(k) is the flux across the top face of cell k; for area/volume averages and for
  CFL it is weighted/scaled with the cell-k geometry (RAC, hFacC, DRF).
- Vertical advective CFL uses cell thickness DRF: CFL_w = |W| * deltaT / DRF(k)
  (matches MITgcm advcfl_wvel, which uses recip_drF).
"""

import os
import re
import glob
import numpy as np

NX, NY, NZ = 512, 384, 138
DT_DTYPE = '>f4'
REF_DATE = np.datetime64('2012-10-01')

# diag_state record indices
REC_UVEL, REC_VVEL, REC_WVEL = 2, 3, 4

CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'

RUNS = {
    'dt300': {
        'dir': '/data/SO3/edavenport/tpose24/oct2012_3mo_dt300_AB3/',
        'deltaT': 300.0,
        'label': 'dt=300s (3mo)',
    },
    'dt60': {
        'dir': '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3/',
        'deltaT': 60.0,
        'label': 'dt=60s (3mo)',
    },
}

# subregion 0N,140W +/- 5deg
BOX_LON = (215.0, 225.0)
BOX_LAT = (-5.0, 5.0)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def _read_field(path, shape):
    return np.fromfile(path, dtype=DT_DTYPE).reshape(shape)


def load_grid(run):
    """Return grid arrays and box index slices for a run."""
    d = RUNS[run]['dir']
    XC = _read_field(d + 'XC.data', (NY, NX))
    YC = _read_field(d + 'YC.data', (NY, NX))
    RC = np.fromfile(d + 'RC.data', dtype=DT_DTYPE)
    DRF = np.fromfile(d + 'DRF.data', dtype=DT_DTYPE)
    RAC = _read_field(d + 'RAC.data', (NY, NX))
    hFacC = _read_field(d + 'hFacC.data', (NZ, NY, NX))
    hFacW = _read_field(d + 'hFacW.data', (NZ, NY, NX))
    hFacS = _read_field(d + 'hFacS.data', (NZ, NY, NX))
    xc1d, yc1d = XC[0, :], YC[:, 0]
    ix = np.where((xc1d >= BOX_LON[0]) & (xc1d <= BOX_LON[1]))[0]
    iy = np.where((yc1d >= BOX_LAT[0]) & (yc1d <= BOX_LAT[1]))[0]
    box = (slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1))
    depth = -np.cumsum(DRF) + DRF / 2.0
    return dict(XC=XC, YC=YC, xc1d=xc1d, yc1d=yc1d, RC=RC, DRF=DRF, depth=depth,
                RAC=RAC, hFacC=hFacC, hFacW=hFacW, hFacS=hFacS, box=box)


def diag_iters(run, prefix='diag_state'):
    """Sorted iteration numbers available for a diagnostic."""
    d = RUNS[run]['dir']
    its = [int(re.search(r'\.(\d+)\.data$', f).group(1))
           for f in glob.glob(d + prefix + '.*.data')]
    return np.array(sorted(its))


def iter_to_days(run, iters):
    return np.asarray(iters) * RUNS[run]['deltaT'] / 86400.0


def iter_to_date(run, iters):
    secs = (np.asarray(iters) * RUNS[run]['deltaT']).astype('timedelta64[s]')
    return REF_DATE + secs


def read_record(run, prefix, it, rec):
    """Read one 3D record from a diagnostic file."""
    path = RUNS[run]['dir'] + f'{prefix}.{it:010d}.data'
    off = rec * NX * NY * NZ * 4
    a = np.fromfile(path, dtype=DT_DTYPE, count=NX * NY * NZ, offset=off)
    return a.reshape(NZ, NY, NX)


def read_uvw(run, it):
    return (read_record(run, 'diag_state', it, REC_UVEL),
            read_record(run, 'diag_state', it, REC_VVEL),
            read_record(run, 'diag_state', it, REC_WVEL))


def read_etan(run, it):
    path = RUNS[run]['dir'] + f'diag_surf.{it:010d}.data'
    return np.fromfile(path, dtype=DT_DTYPE, count=NX * NY).reshape(NY, NX)


def area_weighted_profile(field3d, hFac, RAC):
    """Horizontal area+hFac weighted mean at each level -> profile (NZ,)."""
    w = hFac * RAC
    num = (field3d * w).sum(axis=(1, 2))
    den = w.sum(axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def column_transport(vel3d, hFac, DRF):
    """Depth-integrated velocity per column -> (NY,NX), units m^2/s."""
    return (vel3d * hFac * DRF[:, None, None]).sum(axis=0)


def cfl_w_field(W, deltaT, DRF):
    """Per-cell vertical advective CFL number."""
    return np.abs(W) * deltaT / DRF[:, None, None]


# STDOUT %MON monitor parsing
_MON_RE = re.compile(r'%MON\s+(\S+)\s+=\s+([-\d.ED+]+)')


def parse_stdout_monitor(path, keys):
    """Parse %MON monitor blocks from a MITgcm STDOUT file.

    Returns dict of key -> np.array, aligned by monitor block. 'time_tsnumber'
    (timestep number) delimits blocks and is always returned.
    """
    want = set(keys) | {'time_tsnumber'}
    out = {k: [] for k in want}
    cur = {}

    def flush():
        if 'time_tsnumber' in cur:
            for k in want:
                out[k].append(cur.get(k, np.nan))

    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            if '%MON' not in line:
                continue
            m = _MON_RE.search(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key == 'time_tsnumber':
                flush()
                cur = {}
            if key in want:
                cur[key] = float(val.replace('D', 'E'))
        flush()
    return {k: np.array(v) for k, v in out.items()}
