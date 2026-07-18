"""drift_utils.py
----------------
Helpers for quantifying how far the nested TPOSE24 (1/24 deg) run drifts from
its boundary conditions over the Oct-Dec 2012 simulation.

The TPOSE24 open boundaries are prescribed from TPOSE6 (1/6 deg) daily averages,
so TPOSE24 is pinned to the parent at the domain edges and free to evolve in the
interior.  "Drift" is therefore measured as the growth in time of the difference
between the TPOSE24 interior and the TPOSE6 parent, over the TPOSE24 footprint.

Design (mirrors vertical_velocity/wvel_utils.py):
- diag_state records are packed THETA,SALT,UVEL,VVEL,WVEL,... (float32, big-endian).
  THETA is record 0, SALT is record 1 in both resolutions (verified vs xmitgcm).
- ETAN is record 0 of diag_surf in both resolutions.
- Land / missing cells are written as -999 and masked to NaN.
- Coordinate and grid arrays (XC, YC, Z, drF, rA, hFacC, Depth) come from a short
  xmitgcm load; bulk field reads use direct big-endian memmap offsets for speed.

All domain reductions are computed on each model's native grid over the TPOSE24
footprint.  Horizontal averages weight by wet cell volume (rA * hFacC * drF), so
partial cells and topography are handled consistently across the two grids.
"""

import os
import re
import glob

import numpy as np

DT_DTYPE = '>f4'
MISSING = -999.0

# diag_state record indices (packed THETA,SALT,UVEL,VVEL,WVEL,...)
REC = {'THETA': 0, 'SALT': 1, 'UVEL': 2, 'VVEL': 3, 'WVEL': 4}

CACHE_DIR = '/data/SO3/edavenport/tpose24/cache'

# reference constants for heat / salt content and steric height
RHO0 = 1027.5      # kg m-3, Boussinesq reference density
CP = 3994.0        # J kg-1 K-1, seawater heat capacity

# label -> config.  'parent' flags the TPOSE6 baseline that supplies the BCs.
MODELS = {
    'TP6': {
        'label': 'TPOSE6 (1/6°, parent)',
        'dir': '/data/SO3/edavenport/tpose6/diags/sep2012/run_iter14',
        'grid_dir': '/data/SO6/TPOSE_diags/tpose6/grid_6',
        'NX': 1128, 'NY': 336, 'NZ': 66,
        'nFlds': 6,   # THETA,SALT,UVEL,VVEL,WVEL,DRHODR
        'ref_date': '2012-09-01', 'deltaT': 1200.0,
        'parent': True,
    },
    'TP24': {
        'label': 'TPOSE24 (1/24°, nest)',
        'dir': '/data/SO3/edavenport/tpose24/oct2012_3mo_dt300_AB3',
        'grid_dir': '/data/SO3/edavenport/tpose24/oct2012_3mo_dt300_AB3',
        'NX': 512, 'NY': 384, 'NZ': 138,
        'nFlds': 7,   # THETA,SALT,UVEL,VVEL,WVEL,PHIHYD,DRHODR
        'ref_date': '2012-10-01', 'deltaT': 300.0,
        'parent': False,
    },
}

PARENT = 'TP6'
NEST = 'TP24'

# common daily comparison window (TPOSE24 covers Oct-Dec 2012)
T0, T1 = np.datetime64('2012-10-01'), np.datetime64('2012-12-31')

# top layer (m) averaged for the upper-ocean 2-D drift maps
UPPER_DEPTH = 100.0


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


# ---------------------------------------------------------------------------
# iterations / time
# ---------------------------------------------------------------------------

def diag_iters(model, prefix='diag_state'):
    """Sorted iteration numbers with a ``prefix`` file present."""
    d = MODELS[model]['dir']
    its = [int(re.search(r'\.(\d+)\.data$', f).group(1))
           for f in glob.glob(os.path.join(d, f'{prefix}.*.data'))]
    return np.array(sorted(its))


def iter_times(model, iters):
    """datetime64[s] for each iteration."""
    m = MODELS[model]
    ref = np.datetime64(m['ref_date'])
    secs = (np.asarray(iters) * m['deltaT']).astype('timedelta64[s]')
    return ref + secs


def daily_iters(model):
    """(iters, times) sampled to one step per calendar day within [T0, T1].

    TPOSE6 output is daily already; TPOSE24 is 3-hourly, so the first step of
    each day is kept.  Both models therefore share a daily calendar axis.
    """
    its = diag_iters(model)
    ts = iter_times(model, its)
    keep = (ts >= T0) & (ts < (T1 + np.timedelta64(1, 'D')))
    its, ts = its[keep], ts[keep]
    days = ts.astype('datetime64[D]')
    _, first = np.unique(days, return_index=True)
    return its[first], ts[first].astype('datetime64[D]')


# ---------------------------------------------------------------------------
# field reads (direct big-endian memmap)
# ---------------------------------------------------------------------------

def _read_rec(path, rec, nz, ny, nx):
    fld = np.fromfile(path, dtype=DT_DTYPE, count=nz * ny * nx,
                      offset=rec * nz * ny * nx * 4).astype('f8')
    fld = fld.reshape(nz, ny, nx)
    fld[fld == MISSING] = np.nan
    return fld


def read_state(model, it, var):
    """Full (NZ, NY, NX) field of ``var`` from diag_state for one iteration."""
    m = MODELS[model]
    path = os.path.join(m['dir'], f'diag_state.{it:010d}.data')
    return _read_rec(path, REC[var], m['NZ'], m['NY'], m['NX'])


def read_etan(model, it):
    """Surface ETAN (NY, NX) from diag_surf (record 0) for one iteration."""
    m = MODELS[model]
    path = os.path.join(m['dir'], f'diag_surf.{it:010d}.data')
    e = np.fromfile(path, dtype=DT_DTYPE, count=m['NY'] * m['NX'],
                    offset=0).astype('f8').reshape(m['NY'], m['NX'])
    e[e == MISSING] = np.nan
    return e


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def load_grid(model):
    """Native grid arrays for ``model``.

    Returns a dict with 1-D XC/YC/Z/drF and 2-D rA/Depth and 3-D hFacC, all
    float64.  A light 2-iteration xmitgcm load supplies the (partial-cell)
    grid geometry so it is not reconstructed by hand.
    """
    from xmitgcm import open_mdsdataset
    import warnings
    m = MODELS[model]
    its = diag_iters(model)[:2].tolist()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds = open_mdsdataset(m['dir'], grid_dir=m['grid_dir'], iters=its,
                             prefix=['diag_state'], ref_date=m['ref_date'],
                             delta_t=m['deltaT'])
    xc = np.asarray(ds.XC.astype(float).values)
    yc = np.asarray(ds.YC.astype(float).values)
    return dict(
        xc1d=xc[0, :] if xc.ndim == 2 else xc,
        yc1d=yc[:, 0] if yc.ndim == 2 else yc,
        z=np.asarray(ds.Z.astype(float).values),
        drf=np.asarray(ds.drF.astype(float).values),
        rA=np.asarray(ds.rA.astype(float).values),
        hFacC=np.asarray(ds.hFacC.astype(float).values),
        depth=np.asarray(ds.Depth.astype(float).values),
    )


def footprint_bounds():
    """(lon0, lon1, lat0, lat1) of the TPOSE24 nest, used to crop the parent."""
    g = load_grid(NEST)
    return (float(g['xc1d'].min()), float(g['xc1d'].max()),
            float(g['yc1d'].min()), float(g['yc1d'].max()))


def footprint_mask(xc1d, yc1d, bounds):
    """2-D boolean mask of columns inside ``bounds`` for a lon/lat grid."""
    lon0, lon1, lat0, lat1 = bounds
    mx = (xc1d >= lon0) & (xc1d <= lon1)
    my = (yc1d >= lat0) & (yc1d <= lat1)
    return my[:, None] & mx[None, :]


def dist_to_boundary_km(xc1d, yc1d, bounds):
    """2-D distance (km) from each column to the nearest footprint edge.

    Only meaningful inside the footprint; outside it is set to NaN.
    """
    lon0, lon1, lat0, lat1 = bounds
    lon = xc1d[None, :] * np.ones((yc1d.size, 1))
    lat = yc1d[:, None] * np.ones((1, xc1d.size))
    dlon = np.minimum(lon - lon0, lon1 - lon) * np.cos(np.deg2rad(lat))
    dlat = np.minimum(lat - lat0, lat1 - lat)
    d = np.minimum(dlon, dlat) * 111.0
    d[(d < 0)] = np.nan
    return d


# ---------------------------------------------------------------------------
# volume-weighted reductions over the footprint (native grid)
# ---------------------------------------------------------------------------

def column_weights(grid, mask):
    """Per-level wet horizontal weight w_k(y,x) = rA*hFacC and layer volume.

    Returns (w3d, layer_vol, area_surf) where
      w3d        : (NZ, NY, NX)  rA*hFacC inside the footprint, 0 elsewhere
      layer_vol  : (NZ,)         wet volume of each level = sum(w3d)*drF
      area_surf  : scalar        wet surface area of the footprint (sum rA*hFacC[0])
    """
    rA = grid['rA'][None, :, :]
    hf = grid['hFacC']
    m = mask[None, :, :]
    w3d = np.where(m, rA * hf, 0.0)
    layer_vol = w3d.sum(axis=(1, 2)) * grid['drf']
    area_surf = w3d[0].sum()
    return w3d, layer_vol, area_surf


def profile_mean(field3d, w3d):
    """Horizontal wet-volume-weighted mean profile of ``field3d`` (NZ,)."""
    num = np.nansum(np.where(np.isfinite(field3d), field3d * w3d, 0.0), axis=(1, 2))
    den = np.nansum(np.where(np.isfinite(field3d), w3d, 0.0), axis=(1, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def upper_layer_map(field3d, grid, depth=UPPER_DEPTH):
    """Thickness-weighted 0..depth average of ``field3d`` -> 2-D (NY, NX).

    Weights are drF*hFacC so partial cells and topography are respected.
    """
    kz = np.where(grid['z'] >= -depth)[0]
    dz = (grid['drf'][kz][:, None, None] * grid['hFacC'][kz])
    sub = field3d[kz]
    num = np.nansum(np.where(np.isfinite(sub), sub * dz, 0.0), axis=0)
    den = np.nansum(np.where(np.isfinite(sub), dz, 0.0), axis=0)
    out = np.full(num.shape, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


# ---------------------------------------------------------------------------
# coarse-graining TPOSE24 -> TPOSE6 footprint grid (conservative binning)
# ---------------------------------------------------------------------------

def _cell_edges(centers):
    c = np.asarray(centers, float)
    e = np.empty(c.size + 1)
    e[1:-1] = 0.5 * (c[:-1] + c[1:])
    e[0] = c[0] - 0.5 * (c[1] - c[0])
    e[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return e


def coarsen_to_grid(field2d, src_grid, dst_xc, dst_yc):
    """Area-weighted average of a fine 2-D field onto a coarse lon/lat grid.

    Each fine (TPOSE24) cell is binned into the coarse (TPOSE6) cell containing
    its centre and accumulated with its cell area rA.  This block-averages away
    the mesoscale structure the parent cannot represent, so the comparison
    isolates the large-scale state the boundaries actually constrain.
    """
    xe, ye = _cell_edges(dst_xc), _cell_edges(dst_yc)
    xc, yc, rA = src_grid['xc1d'], src_grid['yc1d'], src_grid['rA']
    ix = np.digitize(xc, xe) - 1
    iy = np.digitize(yc, ye) - 1
    ny, nx = dst_yc.size, dst_xc.size
    IX, IY = np.meshgrid(ix, iy)
    val = field2d.ravel()
    a = rA.ravel()
    ixr, iyr = IX.ravel(), IY.ravel()
    good = np.isfinite(val) & (ixr >= 0) & (ixr < nx) & (iyr >= 0) & (iyr < ny)
    flat = iyr[good] * nx + ixr[good]
    num = np.bincount(flat, weights=(val * a)[good], minlength=ny * nx)
    den = np.bincount(flat, weights=a[good], minlength=ny * nx)
    out = np.full(ny * nx, np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.reshape(ny, nx)


# ---------------------------------------------------------------------------
# steric height from a mean profile (gsw)
# ---------------------------------------------------------------------------

def steric_components(theta_prof, salt_prof, theta_ref, salt_ref,
                      z, dz_col, lon_ref, lat_ref):
    """Total, thermosteric and halosteric height (m) of a profile vs a reference.

    theta_prof, salt_prof : (NZ,) potential temperature / practical salinity now
    theta_ref, salt_ref   : (NZ,) reference (t=0) profiles
    z                      : (NZ,) depth (m, negative down)
    dz_col                 : (NZ,) representative column thickness per level (m)
    lon_ref, lat_ref       : footprint-centre position for the EOS conversions

    Steric height anomaly is  -1/rho0 * integral( rho(now) - rho(ref) ) dz, so a
    lighter (warmer / fresher) column gives a positive height.  Thermosteric
    varies only theta (salt held at ref); halosteric varies only salt.
    """
    import gsw
    p = gsw.p_from_z(z, lat_ref)

    def rho(pt, sp):
        SA = gsw.SA_from_SP(sp, p, lon_ref, lat_ref)
        CT = gsw.CT_from_pt(SA, pt)
        return gsw.rho(SA, CT, p)

    r_now = rho(theta_prof, salt_prof)
    r_ref = rho(theta_ref, salt_ref)
    r_thermo = rho(theta_prof, salt_ref)
    r_halo = rho(theta_ref, salt_prof)

    def integ(r):
        good = np.isfinite(r) & np.isfinite(r_ref)
        return -np.nansum(((r - r_ref) * dz_col)[good]) / RHO0

    return dict(total=integ(r_now), thermo=integ(r_thermo), halo=integ(r_halo))
