"""
build_ext_cache.py
------------------
Daily misfit of the concatenated 6-month TPOSE24 record (Oct 2012 - Apr 2013)
against GLORYS12 and OISST, over the TPOSE24 domain.  The point is to see whether
misfit *grows* through the Jan-Mar extension (drift) or stays bounded.

For each calendar day the model field is regridded onto the (coarser) reference
grid and reduced to a cos(lat)-weighted domain RMSE and bias.  SSH has its
instantaneous domain mean removed from both model and GLORYS first (the known
barotropic offset; see project_drift_test), so the SSH misfit measures the
*pattern/tilt*, not the uniform offset.

Output -> cache/ext_valid_cache.nc  (small: a handful of daily time series).

Usage:  python build_ext_cache.py [--test N]
"""
import os, sys
import numpy as np
import xarray as xr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'GLORYS_comparison'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_validation_utils as u
import glorys_validation_utils as gu
import ext_utils as e

# model domain (fixed)
LON0, LON1, LAT0, LAT1 = 209.35, 230.65, -5.48, 10.48
MONTHS = ['2012_10', '2012_11', '2012_12',
          '2013_01', '2013_02', '2013_03', '2013_04']


def wmean(a, lat2d):
    w = np.cos(np.deg2rad(lat2d))
    v = np.isfinite(a)
    return np.nansum(np.where(v, a * w, 0.0)) / np.nansum(np.where(v, w, 0.0))


def wrmse(diff, lat2d):
    return np.sqrt(wmean(diff ** 2, lat2d))


def regrid(model2d, lon1d, lat1d, tgt_lon, tgt_lat):
    """Linear-interp a native model 2-D field onto a target lon/lat grid."""
    da = xr.DataArray(model2d, dims=['lat', 'lon'],
                      coords={'lat': lat1d, 'lon': lon1d})
    out = da.interp(lon=tgt_lon, lat=tgt_lat, method='linear')
    return out.values  # (n_tgt_lat, n_tgt_lon)


def glorys_upper_mean(da3d, depth_name='depth', dmax=100.0):
    """Thickness-weighted mean of a GLORYS (depth, lat, lon) field over top dmax m."""
    z = da3d[depth_name].values
    kz = np.where(z <= dmax)[0]
    zc = z[kz]
    edges = np.empty(len(zc) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (zc[:-1] + zc[1:])
    edges[-1] = dmax
    th = np.diff(edges)
    sub = da3d.isel({depth_name: kz}).values  # (nk, lat, lon)
    w = th[:, None, None]
    v = np.isfinite(sub)
    num = np.nansum(np.where(v, sub * w, 0.0), axis=0)
    den = np.nansum(np.where(v, w, 0.0), axis=0)
    out = np.full(num.shape, np.nan)
    g = den > 0
    out[g] = num[g] / den[g]
    return out


def main(test=None):
    grid = e.load_grid()
    lon1d, lat1d = grid['lon'], grid['lat']
    dirs, iters, dates = e.daily_index()
    if test:
        dirs, iters, dates = dirs[:test], iters[:test], dates[:test]
    print(f"{len(dates)} days: {dates[0]} .. {dates[-1]}")

    # --- references ---
    gl = gu.load_glorys(MONTHS).sel(longitude=slice(LON0, LON1),
                                    latitude=slice(LAT0, LAT1)).load()
    g_lon, g_lat = gl.longitude.values, gl.latitude.values
    g_lat2d = np.broadcast_to(g_lat[:, None], (g_lat.size, g_lon.size))

    oi = u.to_0360(xr.open_dataset(
        '/data/SO3/edavenport/tpose6/oisst_data/'
        'oisst_equatorial_pacific_2012to2013.nc')).sortby('latitude')
    oi = oi.sel(longitude=slice(LON0, LON1), latitude=slice(LAT0, LAT1))
    o_lon, o_lat = oi.longitude.values, oi.latitude.values
    o_lat2d = np.broadcast_to(o_lat[:, None], (o_lat.size, o_lon.size))

    nk = int(np.sum(grid['z'] >= -100.0))

    cols = {k: np.full(len(dates), np.nan) for k in (
        'sst_rmse_gl', 'sst_bias_gl', 'sst_rmse_oi', 'sst_bias_oi',
        'ssh_rmse_gl', 'spd_rmse_gl', 't100_rmse_gl', 't100_bias_gl',
        's100_rmse_gl', 's100_bias_gl', 'sst_model', 'ssh_model_anom')}

    for i, (d, it, dt) in enumerate(zip(dirs, iters, dates)):
        day = np.datetime64(dt)
        # model fields
        sst_m = e.read_surf(d, it, 'THETA')
        eta_m = e.read_etan(d, it)
        th_m = e.read_levels(d, it, 'THETA', nk)
        sa_m = e.read_levels(d, it, 'SALT', nk)
        u_m = e.read_surf(d, it, 'UVEL')
        v_m = e.read_surf(d, it, 'VVEL')
        t100_m = e.upper_mean(th_m, grid)
        s100_m = e.upper_mean(sa_m, grid)

        # GLORYS this day
        try:
            gd = gl.sel(time=day, method='nearest')
            if abs((gd.time.values - day).astype('timedelta64[D]').astype(int)) > 1:
                gd = None
        except Exception:
            gd = None
        # OISST this day
        try:
            od = oi.sel(time=day, method='nearest')
            if abs((od.time.values - day).astype('timedelta64[D]').astype(int)) > 1:
                od = None
        except Exception:
            od = None

        if gd is not None:
            m_sst = regrid(sst_m, lon1d, lat1d, g_lon, g_lat)
            cols['sst_rmse_gl'][i] = wrmse(m_sst - gd.thetao.isel(depth=0).values, g_lat2d)
            cols['sst_bias_gl'][i] = wmean(m_sst - gd.thetao.isel(depth=0).values, g_lat2d)
            # SSH: remove instantaneous domain mean from both (barotropic offset)
            m_eta = regrid(eta_m, lon1d, lat1d, g_lon, g_lat)
            g_eta = gd.zos.values
            m_eta = m_eta - wmean(m_eta, g_lat2d)
            g_eta = g_eta - wmean(g_eta, g_lat2d)
            cols['ssh_rmse_gl'][i] = wrmse(m_eta - g_eta, g_lat2d)
            # surface speed
            m_spd = regrid(np.hypot(u_m, v_m), lon1d, lat1d, g_lon, g_lat)
            g_spd = np.hypot(gd.uo.isel(depth=0).values, gd.vo.isel(depth=0).values)
            cols['spd_rmse_gl'][i] = wrmse(m_spd - g_spd, g_lat2d)
            # upper-100m T, S
            m_t = regrid(t100_m, lon1d, lat1d, g_lon, g_lat)
            g_t = glorys_upper_mean(gd.thetao)
            cols['t100_rmse_gl'][i] = wrmse(m_t - g_t, g_lat2d)
            cols['t100_bias_gl'][i] = wmean(m_t - g_t, g_lat2d)
            m_s = regrid(s100_m, lon1d, lat1d, g_lon, g_lat)
            g_s = glorys_upper_mean(gd.so)
            cols['s100_rmse_gl'][i] = wrmse(m_s - g_s, g_lat2d)
            cols['s100_bias_gl'][i] = wmean(m_s - g_s, g_lat2d)

        if od is not None:
            m_sst_o = regrid(sst_m, lon1d, lat1d, o_lon, o_lat)
            cols['sst_rmse_oi'][i] = wrmse(m_sst_o - od.sst.values, o_lat2d)
            cols['sst_bias_oi'][i] = wmean(m_sst_o - od.sst.values, o_lat2d)

        # model context
        m_lat2d = np.broadcast_to(lat1d[:, None], sst_m.shape)
        cols['sst_model'][i] = wmean(sst_m, m_lat2d)
        cols['ssh_model_anom'][i] = wmean(eta_m, m_lat2d)

        if i % 20 == 0:
            print(f"  {dt}  SSTrmse_gl={cols['sst_rmse_gl'][i]:.3f}  "
                  f"T100rmse={cols['t100_rmse_gl'][i]:.3f}")

    ds = xr.Dataset({k: ('time', v) for k, v in cols.items()},
                    coords={'time': dates})
    ds.attrs['note'] = ('Daily cos(lat)-weighted domain misfit of concatenated '
                        'TPOSE24 (oct2012_dt60_10072026 + jan2013_from_pickup) vs '
                        'GLORYS12 and OISST over 209.35-230.65E, -5.48-10.48N. '
                        'SSH misfit is after removing each days domain-mean eta.')
    os.makedirs(e.CACHE_DIR, exist_ok=True)
    out = os.path.join(e.CACHE_DIR, 'ext_valid_cache.nc')
    ds.to_netcdf(out)
    print("wrote", out)


if __name__ == '__main__':
    t = None
    if '--test' in sys.argv:
        t = int(sys.argv[sys.argv.index('--test') + 1])
    main(test=t)
