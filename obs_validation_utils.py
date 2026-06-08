"""
obs_validation_utils.py
-----------------------
Shared helpers for validating TPOSE24 surface fields against gridded
observational products (OISST, AVISO/DUACS, OSCAR).

The three validation notebooks

    notebook_obs_oisst_validation.ipynb   (SST   vs OISST)
    notebook_obs_aviso_validation.ipynb   (SSH   vs AVISO/DUACS)
    notebook_obs_oscar_validation.ipynb   (U,V   vs OSCAR)

all share the same workflow: a list of ``(label, run_dir)`` model
specifications is supplied at the top of the notebook, every model is
loaded + daily-averaged + regridded onto the observation grid, and the
number of plotted lines / panels grows automatically with the number of
models.  This module holds the pieces common to all three notebooks so
that the notebooks themselves stay focused on the product-specific
loading and the figures.

Conventions
-----------
* Model longitude is 0–360 °E.  Observation longitudes are converted to
  the same convention with ``lon % 360`` before any selection.
* The model is regridded *onto the (coarser) observation grid* for a fair
  comparison — the observation resolution is the limiting factor.
* Spatial averages / RMSE use cos(latitude) area weighting.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from xmitgcm import open_mdsdataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def discover_iters(run_dir, prefix='diag_state'):
    """Return the sorted list of output iterations present in ``run_dir``.

    Iterations are parsed from the ``{prefix}.{iter}.data`` files so the
    loader works for any run without hard-coding the number of diagnostics.
    """
    files = sorted(glob.glob(os.path.join(run_dir, f'{prefix}.*.data')))
    iters = []
    for f in files:
        m = re.search(rf'{re.escape(prefix)}\.0*(\d+)\.data$', os.path.basename(f))
        if m:
            iters.append(int(m.group(1)))
    return sorted(set(iters))


def load_tpose24(run_dir, prefix=('diag_state', 'diag_surf'),
                 ref_date='2012-10-01', delta_t=300, iters=None):
    """Open a TPOSE24 run as an xarray Dataset.

    Parameters
    ----------
    run_dir : str
        Run directory.  Holds both the MITgcm grid files and the
        ``diag_*`` diagnostic output (grid_dir == data_dir for these runs).
    prefix : sequence of str
        Diagnostic file prefixes to load (``diag_state`` holds
        THETA/SALT/UVEL/VVEL/WVEL, ``diag_surf`` holds ETAN).
    ref_date, delta_t : str, float
        Passed to ``open_mdsdataset``; the oct2012 runs all start
        2012-10-01 with a 300 s timestep (3-hourly output).
    iters : list of int, optional
        Iterations to load.  Defaults to every iteration on disk for the
        first prefix.

    Returns
    -------
    xarray.Dataset
        With float64 horizontal/vertical coordinates.
    """
    prefix = list(prefix)
    if iters is None:
        iters = discover_iters(run_dir, prefix[0])
    ds = open_mdsdataset(
        data_dir=run_dir, grid_dir=run_dir,
        iters=iters, prefix=prefix,
        ref_date=ref_date, delta_t=delta_t,
    )
    for c in ('XC', 'YC', 'Z', 'Zl', 'XG', 'YG'):
        if c in ds.coords:
            ds[c] = ds[c].astype(float)
    return ds


def model_domain(ds):
    """Return ``(lon_min, lon_max, lat_min, lat_max)`` of the model grid."""
    return (float(ds.XC.min()), float(ds.XC.max()),
            float(ds.YC.min()), float(ds.YC.max()))


def daily_mean(da, time_dim='time'):
    """Resample a model DataArray to daily means (model output is sub-daily)."""
    return da.resample({time_dim: '1D'}).mean()


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def to_0360(ds, lon_name='longitude'):
    """Convert a dataset's longitude coordinate to the 0–360 convention."""
    return ds.assign_coords({lon_name: (ds[lon_name] % 360)}).sortby(lon_name)


def subset_domain(da, lon_min, lon_max, lat_min, lat_max,
                  lon_name='longitude', lat_name='latitude'):
    """Select the lon/lat box (0–360 lon).

    The coordinates are sorted ascending first so the box selection works
    regardless of the product's native coordinate ordering (OSCAR stores
    latitude north-to-south, OISST/AVISO south-to-north).
    """
    da = da.sortby(lon_name).sortby(lat_name)
    return da.sel({lon_name: slice(lon_min, lon_max),
                   lat_name: slice(lat_min, lat_max)})


def align_daily(model_da, obs_da, time_dim='time'):
    """Match a model and an observation DataArray on common calendar days.

    Both inputs are floored to day resolution (model output is labelled at
    midnight, many obs products at noon) and intersected, so the returned
    pair share an identical daily ``time`` axis.
    """
    m = model_da.assign_coords({time_dim: model_da[time_dim].dt.floor('D')})
    o = obs_da.assign_coords({time_dim: obs_da[time_dim].dt.floor('D')})
    common = np.intersect1d(m[time_dim].values, o[time_dim].values)
    return m.sel({time_dim: common}), o.sel({time_dim: common})


def regrid_model_to_obs(model_da, obs_lon, obs_lat,
                        x='XC', y='YC', method='linear'):
    """Interpolate a model field onto the observation lon/lat grid.

    ``obs_lon`` must already be in the 0–360 convention.  Returns a
    DataArray on dims (..., latitude, longitude).  Passing plain numpy
    arrays as the targets keeps the interpolated dims named after the
    model coords (``x``/``y``) so they can be renamed cleanly.
    """
    out = model_da.interp({x: np.asarray(obs_lon), y: np.asarray(obs_lat)},
                          method=method)
    out = out.rename({x: 'longitude', y: 'latitude'})
    out = out.assign_coords(longitude=np.asarray(obs_lon),
                            latitude=np.asarray(obs_lat))
    # interp can drop the pandas index on unrelated dims (e.g. time);
    # re-assigning the coordinate values rebuilds it so .sel works.
    for d in out.dims:
        if d in out.coords and d not in out.indexes:
            out = out.assign_coords({d: out[d].values})
    return out


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------

def lat_weights(lat):
    """cos(latitude) weights, normalised to sum to 1 over a (lat, lon) field."""
    lat = np.asarray(lat, dtype=float)
    w1d = np.cos(np.deg2rad(lat))
    return w1d


def weighted_spatial_rmse(model_vals, obs_vals, lat):
    """Area-weighted spatial RMSE.  Inputs shape (n_time, n_lat, n_lon)."""
    w = lat_weights(lat)[:, None] * np.ones(model_vals.shape[-1])
    diff = model_vals - obs_vals
    valid = np.isfinite(diff)
    wsum = np.nansum(np.where(valid, w, np.nan), axis=(-2, -1))
    sse = np.nansum(np.where(valid, diff ** 2 * w, np.nan), axis=(-2, -1))
    return np.sqrt(sse / wsum)


def weighted_spatial_bias(model_vals, obs_vals, lat):
    """Area-weighted spatial mean of (model - obs)."""
    w = lat_weights(lat)[:, None] * np.ones(model_vals.shape[-1])
    diff = model_vals - obs_vals
    valid = np.isfinite(diff)
    wsum = np.nansum(np.where(valid, w, np.nan), axis=(-2, -1))
    wsd = np.nansum(np.where(valid, diff * w, np.nan), axis=(-2, -1))
    return wsd / wsum


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

# A qualitative palette indexed by model number; obs is always plotted black.
MODEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#17becf']


def model_color(i):
    return MODEL_COLORS[i % len(MODEL_COLORS)]
