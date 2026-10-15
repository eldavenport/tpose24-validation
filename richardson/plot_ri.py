"""
plot_ri.py
----------
140W (220E) Richardson-number figures, bottom 500 m in height-above-bottom.
Transects are interpolated in 2-D (lat, HAB) with griddata so the few thick
deep model levels render smoothly instead of per-column jagged.

Outputs (richardson/):
  fig_ri_mean_bottom500_{log,lin}.png   time-mean Ri transect (hydro|QH|diff)
  fig_ri_q25_bottom500_{log,lin}.png    25th-pctile Ri transect
  fig_ri_timeseries_1p5N_{log,lin}.png  Ri(time, HAB) Hovmoller at 140W, 1.5N
"""
import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm, Normalize

CACHE = '/data/SO3/edavenport/tpose24/cache'
HERE = os.path.dirname(os.path.abspath(__file__))
HABg = np.linspace(0, 500, 26)
R025 = 0.25

# colour setups keyed by scale
def scales():
    return {
        'log': dict(
            field=lambda G: G,                       # already log10
            norm=TwoSlopeNorm(vmin=-1.0, vcenter=np.log10(R025), vmax=3.0),
            cmap='Spectral_r', clabel='log$_{10}$ Ri', clevel=np.log10(R025),
            dfield=lambda a, b: b - a,               # log10 ratio
            dnorm=TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5),
            dcmap='RdBu_r', dlabel='log$_{10}$(Ri$_{QH}$/Ri$_{hydro}$)'),
        'lin': dict(
            field=lambda G: 10.0 ** G,
            norm=TwoSlopeNorm(vmin=0.0, vcenter=R025, vmax=1.0),
            cmap='RdBu', clabel='Ri', clevel=R025,
            dfield=lambda a, b: np.clip(10.0 ** b - 10.0 ** a, -1, 1),
            dnorm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
            dcmap='RdBu_r', dlabel='Ri$_{QH}$ − Ri$_{hydro}$ (clipped)'),
    }


def hab_transect(ds, var, latg):
    """griddata log10(Ri) onto (lat, HAB) from the scattered (lat, HAB) samples."""
    z = ds.depth.values; lat = ds.lat.values; bot = ds.bottom.values
    a = ds[var].values                                # (depth, lat)
    P, V = [], []
    for j in range(lat.size):
        hab = bot[j] + z                              # z negative -> HAB above bottom
        for k in range(z.size):
            if np.isfinite(a[k, j]) and 0 <= hab[k] <= 500:
                P.append((lat[j], hab[k])); V.append(np.log10(a[k, j]))
    LAT, HAB = np.meshgrid(latg, HABg)
    return griddata(np.array(P), np.array(V), (LAT, HAB), method='linear')


def transect_fig(stat, title, scale):
    s = scales()[scale]
    h = xr.open_dataset(f'{CACHE}/cache_ri_hydro.nc')
    q = xr.open_dataset(f'{CACHE}/cache_ri_QH.nc')
    latg = np.linspace(-3, 3, 97)
    Gh, Gq = hab_transect(h, stat, latg), hab_transect(q, stat, latg)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for a, G, lab in [(ax[0], Gh, 'hydrostatic'), (ax[1], Gq, 'quasi-hydrostatic')]:
        im = a.pcolormesh(latg, HABg, s['field'](G), cmap=s['cmap'], norm=s['norm'], shading='auto')
        a.contour(latg, HABg, s['field'](G), levels=[s['clevel']], colors='k', linewidths=1.5)
        a.set_title(f'{title} — {lab}'); a.set_xlabel('latitude (°N)')
        plt.colorbar(im, ax=a, shrink=0.85, label=s['clabel'])
    im = ax[2].pcolormesh(latg, HABg, s['dfield'](Gh, Gq), cmap=s['dcmap'], norm=s['dnorm'], shading='auto')
    ax[2].set_title('QH vs hydrostatic'); ax[2].set_xlabel('latitude (°N)')
    plt.colorbar(im, ax=ax[2], shrink=0.85, label=s['dlabel'])
    ax[0].set_ylabel('height above bottom (m)')
    fig.suptitle(f'{title} Ri — 140°W (220°E), bottom 500 m, black contour Ri = 0.25', y=1.04)
    p = f'{HERE}/fig_ri_{stat.split("_")[1]}_bottom500_{scale}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote', p)


def ts_to_hab(ds):
    """Ri(time, HAB) at 1.5N: map the single-column depths to HAB, interp to HABg."""
    z = ds.depth.values; bot = ds.attrs['bottom_ts']
    hab = bot + z
    ri = ds.Ri_ts.values                              # (time, depth)
    good = (hab >= -20) & (hab <= 520)
    order = np.argsort(hab[good])
    out = np.full((ri.shape[0], HABg.size), np.nan)
    lh = np.log10(ri[:, good][:, order])
    hh = hab[good][order]
    for t in range(ri.shape[0]):
        out[t] = np.interp(HABg, hh, lh[t], left=np.nan, right=np.nan)
    return ds.time.values, out                        # log10 Ri (time, HAB)


def timeseries_fig(scale):
    s = scales()[scale]
    h = xr.open_dataset(f'{CACHE}/cache_ri_hydro.nc')
    q = xr.open_dataset(f'{CACHE}/cache_ri_QH.nc')
    t, Gh = ts_to_hab(h); _, Gq = ts_to_hab(q)
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for a, G, lab in [(ax[0], Gh, 'hydrostatic'), (ax[1], Gq, 'quasi-hydrostatic')]:
        im = a.pcolormesh(t, HABg, s['field'](G).T, cmap=s['cmap'], norm=s['norm'], shading='auto')
        a.contour(t, HABg, s['field'](G).T, levels=[s['clevel']], colors='k', linewidths=1.0)
        a.set_title(f'Ri at 140°W, 1.5°N — {lab}'); a.set_ylabel('height above bottom (m)')
        plt.colorbar(im, ax=a, shrink=0.9, label=s['clabel'])
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    p = f'{HERE}/fig_ri_timeseries_1p5N_{scale}.png'
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); print('wrote', p)


if __name__ == '__main__':
    for scale in ('log', 'lin'):
        transect_fig('Ri_mean', 'time-mean', scale)
        transect_fig('Ri_q25', 'lower-quartile', scale)
        timeseries_fig(scale)
