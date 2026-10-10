"""Quick qualitative check of the 3-month quasi-hydrostatic (QH) tpose24 run.

Compares the QH run against the standard hydrostatic tpose24 (dt60), which
shares the same grid, timestep and 3-hourly output cadence, so iterations line
up one-to-one. Produces two figures in this directory:

  qh_timeseries.png   domain-mean SST / SSH / surface speed over the run,
                      plus daily global max speed and max |WVEL| (blow-up check)
  qh_surface_maps.png final-day SST, SSH and surface speed: QH, hydrostatic, diff

diag_state records are packed THETA,SALT,UVEL,VVEL,WVEL,PHIHYD,DRHODR (float32,
big-endian). Surface fields (k=0) and SST are the leading ny*nx blocks of their
records, so every step is read cheaply; full-volume maxima are subsampled daily.
Nothing is written to disk except the PNGs.
"""

import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT = '>f4'
MISSING = -999.0
NX, NY, NZ = 512, 384, 138
REC = {'THETA': 0, 'SALT': 1, 'UVEL': 2, 'VVEL': 3, 'WVEL': 4}
SLAB = NY * NX                       # one horizontal level (elements)
VOL = NZ * NY * NX                   # one 3-D record (elements)

RUNS = {
    'QH': '/data/SO3/edavenport/tpose24/oct2012_QH',
    'HS': '/data/SO3/edavenport/tpose24/oct2012_3mo_dt60_AB3',
}
LABEL = {'QH': 'quasi-hydrostatic', 'HS': 'hydrostatic'}
OUT = os.path.dirname(os.path.abspath(__file__))
DELTAT = 60.0                        # s, both runs


def iters(run):
    fs = glob.glob(os.path.join(RUNS[run], 'diag_state.*.data'))
    return np.array(sorted(int(re.search(r'\.(\d+)\.data$', f).group(1)) for f in fs))


def surf_slab(run, it, rec, prefix='diag_state'):
    """k=0 level of a diag_state record, or record of diag_surf, as (NY,NX)."""
    path = os.path.join(RUNS[run], f'{prefix}.{it:010d}.data')
    a = np.fromfile(path, dtype=DT, count=SLAB, offset=rec * VOL * 4).astype('f8')
    a = a.reshape(NY, NX)
    a[a == MISSING] = np.nan
    return a


def full_rec(run, it, rec):
    """Full (NZ,NY,NX) diag_state record."""
    path = os.path.join(RUNS[run], f'diag_state.{it:010d}.data')
    a = np.fromfile(path, dtype=DT, count=VOL, offset=rec * VOL * 4).astype('f8')
    a = a.reshape(NZ, NY, NX)
    a[a == MISSING] = np.nan
    return a


def load_lonlat():
    xc = np.fromfile(os.path.join(RUNS['QH'], 'XC.data'), dtype=DT).reshape(NY, NX)
    yc = np.fromfile(os.path.join(RUNS['QH'], 'YC.data'), dtype=DT).reshape(NY, NX)
    return xc[0, :].astype('f8'), yc[:, 0].astype('f8')


# iterations common to both runs (the 20-day QH window)
its = {r: iters(r) for r in RUNS}
common = np.intersect1d(its['QH'], its['HS'])
days = common * DELTAT / 86400.0
daily = common[::8]                  # 3-hourly -> ~daily subsample for volume maxima
days_d = daily * DELTAT / 86400.0
print(f'common steps: {common.size}, span {days[0]:.2f}-{days[-1]:.2f} days')

# ---- time series (surface fields every step, volume maxima daily) ----
ts = {r: {k: [] for k in ('sst', 'ssh', 'spd')} for r in RUNS}
for r in RUNS:
    for it in common:
        sst = surf_slab(r, it, REC['THETA'])
        ssh = surf_slab(r, it, 0, prefix='diag_surf')
        u = surf_slab(r, it, REC['UVEL'])
        v = surf_slab(r, it, REC['VVEL'])
        spd = np.sqrt(u ** 2 + v ** 2)
        ts[r]['sst'].append(np.nanmean(sst))
        ts[r]['ssh'].append(np.nanmean(ssh))
        ts[r]['spd'].append(np.nanmean(spd))
    for k in ts[r]:
        ts[r][k] = np.array(ts[r][k])

vmax = {r: {'spd': [], 'w': []} for r in RUNS}
for r in RUNS:
    for it in daily:
        u = full_rec(r, it, REC['UVEL'])
        v = full_rec(r, it, REC['VVEL'])
        w = full_rec(r, it, REC['WVEL'])
        vmax[r]['spd'].append(np.nanmax(np.sqrt(u ** 2 + v ** 2)))
        vmax[r]['w'].append(np.nanmax(np.abs(w)))
    for k in vmax[r]:
        vmax[r][k] = np.array(vmax[r][k])

fig, ax = plt.subplots(2, 3, figsize=(15, 7))
C = {'QH': 'C3', 'HS': 'k'}
for r in RUNS:
    ax[0, 0].plot(days, ts[r]['sst'], C[r], label=LABEL[r])
    ax[0, 1].plot(days, ts[r]['ssh'], C[r])
    ax[0, 2].plot(days, ts[r]['spd'], C[r])
    ax[1, 0].plot(days_d, vmax[r]['spd'], C[r] + '-o', ms=3)
    ax[1, 1].plot(days_d, vmax[r]['w'], C[r] + '-o', ms=3)
ax[0, 0].set_title('domain-mean SST (°C)')
ax[0, 1].set_title('domain-mean SSH ETAN (m)')
ax[0, 2].set_title('mean surface speed (m s$^{-1}$)')
ax[1, 0].set_title('max surface+interior speed (m s$^{-1}$)')
ax[1, 1].set_title('max |WVEL| (m s$^{-1}$)')
ax[1, 2].axis('off')
ax[0, 0].legend(loc='best')
for a in ax.flat:
    if a.has_data():
        a.set_xlabel('day')
        a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'qh_timeseries.png'), dpi=120)
print('wrote qh_timeseries.png')

# ---- final-day surface maps: QH, hydrostatic, difference ----
lon, lat = load_lonlat()
it = common[-1]
rows = [
    ('SST (°C)', lambda r: surf_slab(r, it, REC['THETA']), 'viridis'),
    ('SSH ETAN (m)', lambda r: surf_slab(r, it, 0, prefix='diag_surf'), 'viridis'),
    ('surface speed (m s$^{-1}$)',
     lambda r: np.sqrt(surf_slab(r, it, REC['UVEL']) ** 2
                       + surf_slab(r, it, REC['VVEL']) ** 2), 'turbo'),
]
fig, ax = plt.subplots(len(rows), 3, figsize=(16, 11))
ext = [lon[0], lon[-1], lat[0], lat[-1]]
for i, (name, getter, cmap) in enumerate(rows):
    qh, hs = getter('QH'), getter('HS')
    vmin = np.nanpercentile(np.concatenate([qh.ravel(), hs.ravel()]), 1)
    vmax_ = np.nanpercentile(np.concatenate([qh.ravel(), hs.ravel()]), 99)
    for j, (fld, ttl) in enumerate([(qh, f'{LABEL["QH"]}'), (hs, f'{LABEL["HS"]}')]):
        im = ax[i, j].imshow(fld, origin='lower', extent=ext, aspect='auto',
                             cmap=cmap, vmin=vmin, vmax=vmax_)
        plt.colorbar(im, ax=ax[i, j], shrink=0.8)
        ax[i, j].set_title(f'{name} — {ttl}')
    d = qh - hs
    dlim = np.nanpercentile(np.abs(d), 99)
    im = ax[i, 2].imshow(d, origin='lower', extent=ext, aspect='auto',
                         cmap='RdBu_r', vmin=-dlim, vmax=dlim)
    plt.colorbar(im, ax=ax[i, 2], shrink=0.8)
    ax[i, 2].set_title(f'{name} — QH − hydrostatic')
fig.suptitle(f'day {days[-1]:.1f}')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'qh_surface_maps.png'), dpi=120)
print('wrote qh_surface_maps.png')
