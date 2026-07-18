"""Cache the drift reductions for TPOSE24 vs its TPOSE6 boundary conditions.

For each model this writes one NetCDF to CACHE_DIR (drift_cache_{model}.nc):

  scalar / profile time series over the TPOSE24 footprint (native-grid,
  wet-volume-weighted):
    theta_prof (time, z)      : mean potential temperature profile
    salt_prof  (time, z)      : mean salinity profile
    volmean_theta (time)      : footprint volume-mean theta
    volmean_salt  (time)      : footprint volume-mean salinity
    heat_content  (time)      : RHO0*CP * integral(theta dV)          [J]
    salt_content  (time)      : integral(salt dV)                     [g/kg m3]
    steric_total/thermo/halo (time) : steric height vs day 0          [m]
    etan_mean (time)          : footprint-mean sea-surface height     [m]

  2-D daily maps on the SHARED TPOSE6 footprint grid (TPOSE6 cropped,
  TPOSE24 conservatively coarse-grained onto it):
    t100 (time, y, x)         : 0..UPPER_DEPTH thickness-weighted mean theta
    s100 (time, y, x)         : 0..UPPER_DEPTH thickness-weighted mean salinity
    etan_map (time, y, x)     : sea-surface height

  plus static: z, drf, layer_vol, dz_col, dist_km(y,x), lon/lat of the grid.

Usage:
    python build_drift_cache.py [TP6 TP24]     # default: both
"""

import os
import sys
import time

import numpy as np
import xarray as xr

import drift_utils as d


def target_grid():
    """Shared 2-D map grid: the TPOSE6 columns inside the TPOSE24 footprint."""
    bounds = d.footprint_bounds()
    g6 = d.load_grid(d.PARENT)
    lon0, lon1, lat0, lat1 = bounds
    ix = np.where((g6['xc1d'] >= lon0) & (g6['xc1d'] <= lon1))[0]
    iy = np.where((g6['yc1d'] >= lat0) & (g6['yc1d'] <= lat1))[0]
    tx, ty = g6['xc1d'][ix], g6['yc1d'][iy]
    dist = d.dist_to_boundary_km(tx, ty, bounds)
    return bounds, tx, ty, dist, ix, iy


def build_model(model, bounds, tx, ty, dist, ix6, iy6):
    m = d.MODELS[model]
    out = d.cache_path(f'drift_cache_{model}.nc')
    print(f'\n=== {model}  ({m["label"]}) ===', flush=True)

    grid = d.load_grid(model)
    z, drf = grid['z'], grid['drf']
    mask = d.footprint_mask(grid['xc1d'], grid['yc1d'], bounds)
    w3d, layer_vol, area_surf = d.column_weights(grid, mask)
    dz_col = layer_vol / area_surf                 # representative column thickness
    lon_ref = 0.5 * (bounds[0] + bounds[1])
    lat_ref = 0.5 * (bounds[2] + bounds[3])
    wsurf = w3d[0]                                  # surface wet weight for ETAN

    its, days = d.daily_iters(model)
    nt, nz = len(its), m['NZ']
    ny, nx = ty.size, tx.size
    print(f'footprint columns={int((mask).sum())}, wet surf area={area_surf:.3e} m2, '
          f'{nt} days {str(days[0])}..{str(days[-1])}', flush=True)

    theta_prof = np.full((nt, nz), np.nan)
    salt_prof = np.full((nt, nz), np.nan)
    etan_mean = np.full(nt, np.nan)
    t100 = np.full((nt, ny, nx), np.nan, np.float32)
    s100 = np.full((nt, ny, nx), np.nan, np.float32)
    etan_map = np.full((nt, ny, nx), np.nan, np.float32)

    is_parent = m['parent']
    t0 = time.time()
    for t, it in enumerate(its):
        th = d.read_state(model, int(it), 'THETA')
        sa = d.read_state(model, int(it), 'SALT')
        theta_prof[t] = d.profile_mean(th, w3d)
        salt_prof[t] = d.profile_mean(sa, w3d)

        th100 = d.upper_layer_map(th, grid)
        sa100 = d.upper_layer_map(sa, grid)
        eta = d.read_etan(model, int(it))
        etan_mean[t] = np.nansum(eta * wsurf) / np.nansum(wsurf)

        if is_parent:
            t100[t] = th100[np.ix_(iy6, ix6)]
            s100[t] = sa100[np.ix_(iy6, ix6)]
            etan_map[t] = eta[np.ix_(iy6, ix6)]
        else:
            t100[t] = d.coarsen_to_grid(th100, grid, tx, ty)
            s100[t] = d.coarsen_to_grid(sa100, grid, tx, ty)
            etan_map[t] = d.coarsen_to_grid(eta, grid, tx, ty)

        if t % 10 == 0 or t == nt - 1:
            print(f'  [{t+1}/{nt}] {str(days[t])}  ({time.time()-t0:.0f}s)', flush=True)

    # derived integrated quantities
    volmean_theta = np.nansum(theta_prof * layer_vol, axis=1) / np.nansum(layer_vol)
    volmean_salt = np.nansum(salt_prof * layer_vol, axis=1) / np.nansum(layer_vol)
    heat_content = d.RHO0 * d.CP * np.nansum(theta_prof * layer_vol, axis=1)
    salt_content = np.nansum(salt_prof * layer_vol, axis=1)

    # steric height vs the day-0 profile
    st = {'total': np.full(nt, np.nan), 'thermo': np.full(nt, np.nan),
          'halo': np.full(nt, np.nan)}
    th_ref, sa_ref = theta_prof[0], salt_prof[0]
    for t in range(nt):
        c = d.steric_components(theta_prof[t], salt_prof[t], th_ref, sa_ref,
                                z, dz_col, lon_ref, lat_ref)
        for k in st:
            st[k][t] = c[k]

    ds = xr.Dataset(
        data_vars=dict(
            theta_prof=(('time', 'z'), theta_prof),
            salt_prof=(('time', 'z'), salt_prof),
            volmean_theta=('time', volmean_theta),
            volmean_salt=('time', volmean_salt),
            heat_content=('time', heat_content),
            salt_content=('time', salt_content),
            steric_total=('time', st['total']),
            steric_thermo=('time', st['thermo']),
            steric_halo=('time', st['halo']),
            etan_mean=('time', etan_mean),
            layer_vol=('z', layer_vol),
            dz_col=('z', dz_col),
            drf=('z', drf),
            t100=(('time', 'y', 'x'), t100),
            s100=(('time', 'y', 'x'), s100),
            etan_map=(('time', 'y', 'x'), etan_map),
            dist_km=(('y', 'x'), dist.astype(np.float32)),
        ),
        coords=dict(time=days, z=z, y=ty, x=tx),
        attrs=dict(model=model, label=m['label'],
                   area_surf_m2=float(area_surf),
                   lon_ref=lon_ref, lat_ref=lat_ref,
                   upper_depth_m=d.UPPER_DEPTH,
                   footprint=str(bounds)),
    )
    os.makedirs(d.CACHE_DIR, exist_ok=True)
    ds.to_netcdf(out)
    print(f'  wrote {out}', flush=True)


def main():
    models = sys.argv[1:] or [d.PARENT, d.NEST]
    bounds, tx, ty, dist, ix6, iy6 = target_grid()
    print(f'footprint bounds (lon0,lon1,lat0,lat1) = {tuple(round(b,3) for b in bounds)}')
    print(f'shared map grid: {ty.size} x {tx.size} (TPOSE6 columns)')
    for model in models:
        build_model(model, bounds, tx, ty, dist, ix6, iy6)


if __name__ == '__main__':
    main()
