"""
plot_ext.py
-----------
Figures for the Jan-Apr 2013 TPOSE24 extension validation.  Reads the daily
misfit cache (build_ext_cache.py) and the offline TAO model-equivalent profiles
(sample_profiles.py).  The question each figure answers: does the model's
distance from the reanalysis / obs grow through the Jan-Mar extension, or stay
bounded?

Output PNGs -> extension_validation/.
"""
import os, sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = '/data/SO3/edavenport/tpose24/cache/ext_valid_cache.nc'
PROF_EQ = '/data/SO3/edavenport/tpose24/jan2013_from_pickup/PROF_eq'
JUNCTION = np.datetime64('2012-12-30')  # pickup / start of extension


def _vline(ax, tmax):
    ax.set_xlim(np.datetime64('2012-10-01'), tmax)
    ax.axvline(JUNCTION, color='0.5', ls='--', lw=1)
    ax.axvspan(JUNCTION, tmax, color='0.92', zorder=0)


def fig_misfit_timeseries():
    ds = xr.open_dataset(CACHE)
    t = ds.time.values
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(t, ds.sst_rmse_gl, label='RMSE vs GLORYS', color='#1f77b4')
    ax[0].plot(t, ds.sst_rmse_oi, label='RMSE vs OISST', color='#2ca02c')
    ax[0].plot(t, ds.sst_bias_gl, label='bias vs GLORYS', color='#1f77b4', ls=':')
    ax[0].plot(t, ds.sst_bias_oi, label='bias vs OISST', color='#2ca02c', ls=':')
    ax[0].axhline(0, color='k', lw=0.5)
    ax[0].set_ylabel('SST (°C)')
    ax[0].legend(ncol=2, fontsize=8)

    ax[1].plot(t, ds.t100_rmse_gl, label='T RMSE', color='#d62728')
    ax[1].plot(t, ds.t100_bias_gl, label='T bias', color='#d62728', ls=':')
    ax[1].plot(t, ds.s100_rmse_gl, label='S RMSE', color='#9467bd')
    ax[1].plot(t, ds.s100_bias_gl, label='S bias', color='#9467bd', ls=':')
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set_ylabel('upper-100 m vs GLORYS\n(°C  /  g kg⁻¹)')
    ax[1].legend(ncol=2, fontsize=8)

    ax[2].plot(t, ds.ssh_rmse_gl, label='SSH pattern RMSE (demeaned)', color='#ff7f0e')
    ax[2].plot(t, ds.spd_rmse_gl, label='surface speed RMSE', color='#17becf')
    ax[2].set_ylabel('SSH (m)  /  speed (m s⁻¹)')
    ax[2].legend(ncol=2, fontsize=8)

    for a in ax:
        _vline(a, t.max())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax[0].set_title('TPOSE24 domain misfit vs GLORYS12 / OISST  '
                    '(shaded = Jan–Apr 2013 extension)')
    fig.tight_layout()
    p = os.path.join(HERE, 'ext_misfit_timeseries.png')
    fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)


def fig_ssh_mean():
    ds = xr.open_dataset(CACHE)
    t = ds.time.values
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, ds.ssh_model_anom * 100, color='#ff7f0e')
    ax.set_ylabel('domain-mean η (cm)')
    _vline(ax, t.max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_title('TPOSE24 domain-mean sea surface height (barotropic offset)')
    fig.tight_layout()
    p = os.path.join(HERE, 'ext_ssh_domain_mean.png')
    fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)


def _prof_misfit(name, ovar, label, units):
    f = os.path.join(PROF_EQ, name + '_model.nc')
    if not os.path.exists(f):
        print('skip', f); return
    ds = xr.open_dataset(f)
    mod = ds[ovar + '_model'].values
    obs = ds[ovar + '_obs'].values
    obs = np.where(np.abs(obs) > 900, np.nan, obs)  # MITprof missing-data fill
    dep = ds.prof_depth.values
    lat = ds.prof_lat.values
    t = ds.prof_time.values

    # mean profile (all in-domain, extension period) + RMSE-vs-time near-surface
    diff = mod - obs
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))
    # mean over profiles, paired (model only where obs exists), populated levels
    modp = np.where(np.isfinite(obs), mod, np.nan)
    mo, oo, do = np.nanmean(modp, 0), np.nanmean(obs, 0), np.nanmean(diff, 0)
    k = np.isfinite(oo)
    # (a) mean obs vs model profile
    ax[0].plot(oo[k], dep[k], 'ko-', ms=4, label='TAO obs')
    ax[0].plot(mo[k], dep[k], 'ro-', ms=4, label='TPOSE24')
    ax[0].invert_yaxis(); ax[0].set_ylabel('depth (m)')
    ax[0].set_xlabel(f'{label} ({units})'); ax[0].legend(); ax[0].set_title('mean profile')
    # (b) mean misfit profile
    ax[1].plot(do[k], dep[k], 'bo-', ms=4)
    ax[1].axvline(0, color='k', lw=0.5); ax[1].invert_yaxis()
    ax[1].set_xlabel(f'model − obs ({units})'); ax[1].set_title('mean misfit')
    # (c) column-RMSE vs time
    rmse_t = np.sqrt(np.nanmean(diff ** 2, axis=1))
    order = np.argsort(t)
    ax[2].plot(t[order], rmse_t[order], '.', ms=3, color='#1f77b4')
    ax[2].set_ylabel(f'profile RMSE ({units})'); ax[2].set_title('misfit vs time')
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.suptitle(f'{name}  {label}  (140°W line, {np.isfinite(diff).any(1).sum()} profiles)')
    fig.tight_layout()
    p = os.path.join(HERE, f'ext_prof_{name}_{ovar}.png')
    fig.savefig(p, dpi=130); plt.close(fig); print('wrote', p)


def main():
    fig_misfit_timeseries()
    fig_ssh_mean()
    _prof_misfit('TAO_WO_2013_CTD_daily_ED', 'prof_T', 'temperature', '°C')
    _prof_misfit('TAO_WO_2013_CTD_daily_ED', 'prof_S', 'salinity', 'g kg⁻¹')
    _prof_misfit('TAO_WO_2013_ADCP', 'prof_U', 'zonal velocity', 'm s⁻¹')


if __name__ == '__main__':
    main()
