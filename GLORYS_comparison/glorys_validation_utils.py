"""
glorys_validation_utils.py
--------------------------
Helpers for validating the TPOSE6 (1/6°) and TPOSE24 (1/24°) runs against
the GLORYS12 reanalysis (1/12°) over the Oct–Dec 2012 overlap window.

GLORYS variables live in three sibling directories, one variable group each:

    glorys_data     zos (SSH), uo, vo (currents)
    glorys_T_data   thetao (potential temperature)
    glorys_S_data   so (salinity)

All files share the same lon/lat/depth grid (0.083°, 31 depth levels to ~454 m).
Longitudes are stored -180:180 and converted to 0–360 °E to match the model.

This module holds the data-side helpers (loading, regridding onto GLORYS,
depth handling, weighted domain means); the notebook does the plotting.
Generic obs helpers (subset_domain, regrid_model_to_obs, align_daily,
daily_mean, weighted statistics, model colours) are reused from
``obs_validation_utils``.
"""

import glob
import os

import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset

import obs_validation_utils as u

# ---------------------------------------------------------------------------
# GLORYS data locations
# ---------------------------------------------------------------------------

GLORYS_UV_DIR = '/data/SO3/edavenport/tpose6/glorys_data'    # zos, uo, vo
GLORYS_T_DIR = '/data/SO3/edavenport/tpose6/glorys_T_data'   # thetao
GLORYS_S_DIR = '/data/SO3/edavenport/tpose6/glorys_S_data'   # so


def load_glorys(months):
    """Open GLORYS SSH/currents/temperature/salinity for the given months.

    Parameters
    ----------
    months : sequence of str
        Month tags ``'YYYY_MM'`` naming the ``glorys_{tag}.nc`` files.

    Returns
    -------
    xarray.Dataset
        Merged dataset (zos, uo, vo, thetao, so) with longitude on the
        0–360 °E convention and coordinates sorted ascending.
    """
    def _open(directory):
        files = [os.path.join(directory, f'glorys_{m}.nc') for m in months]
        return xr.open_mfdataset(files, combine='by_coords')

    ds = xr.merge([_open(GLORYS_UV_DIR),
                   _open(GLORYS_T_DIR),
                   _open(GLORYS_S_DIR)])
    ds = u.to_0360(ds).sortby('latitude')
    return ds


# ---------------------------------------------------------------------------
# Model loading (grid_dir may differ from data_dir; e.g. TPOSE6)
# ---------------------------------------------------------------------------

def load_model(run_dir, grid_dir=None, prefix=('diag_state',),
               ref_date='2012-10-01', delta_t=300, iters=None):
    """Open a TPOSE run, allowing grid_dir != run_dir (needed for TPOSE6)."""
    if grid_dir is None:
        grid_dir = run_dir
    prefix = list(prefix)
    if iters is None:
        iters = u.discover_iters(run_dir, prefix[0])
    ds = open_mdsdataset(data_dir=run_dir, grid_dir=grid_dir,
                         iters=iters, prefix=prefix,
                         ref_date=ref_date, delta_t=delta_t)
    for c in ('XC', 'YC', 'XG', 'YG', 'Z', 'Zl'):
        if c in ds.coords:
            ds[c] = ds[c].astype(float)
    return ds


def domain_intersection(domains):
    """Intersection ``(lon0, lon1, lat0, lat1)`` of several model domains."""
    lon0 = max(d[0] for d in domains)
    lon1 = min(d[1] for d in domains)
    lat0 = max(d[2] for d in domains)
    lat1 = min(d[3] for d in domains)
    return (lon0, lon1, lat0, lat1)


# ---------------------------------------------------------------------------
# Spatial reductions
# ---------------------------------------------------------------------------

def weighted_domain_mean(field, lat):
    """cos(lat)-weighted mean over the last two axes (lat, lon).

    ``field`` has shape (..., n_lat, n_lon); ``lat`` has length n_lat.
    NaNs (land) are ignored.
    """
    w = np.cos(np.deg2rad(np.asarray(lat, float)))[:, None]
    w = np.broadcast_to(w, field.shape[-2:])
    valid = np.isfinite(field)
    num = np.nansum(np.where(valid, field * w, 0.0), axis=(-2, -1))
    den = np.nansum(np.where(valid, w, 0.0), axis=(-2, -1))
    return num / den


def remove_domain_mean(field, lat):
    """Subtract the cos(lat)-weighted domain mean from each (lat, lon) slice."""
    dm = weighted_domain_mean(field, lat)
    return field - dm[..., None, None]


def band_mean(da, lat_name='latitude', half_width=1.0):
    """cos(lat)-weighted mean over the equatorial band |lat| <= half_width."""
    lat = da[lat_name]
    sel = da.where(np.abs(lat) <= half_width, drop=True)
    w = np.cos(np.deg2rad(sel[lat_name]))
    return sel.weighted(w).mean(lat_name)


def profile_domain_mean(da, lat_name='latitude', lon_name='longitude'):
    """cos(lat)-weighted domain mean, returning a profile over the depth dim."""
    w = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(w).mean((lat_name, lon_name))
