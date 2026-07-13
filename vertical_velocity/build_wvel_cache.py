"""Cache WVEL reductions for the low- vs high-resolution comparison.

For each model in wvel_utils.MODELS this writes one NetCDF to CACHE_DIR:
  - w_col   (time, depth)      : WVEL column at 0N, 140W  (all output times)
  - w_slice (time_anim, y, x)  : WVEL on the ANIM_DEPTH horizontal level,
                                 subsampled in time for the animation frames.

Field reads use direct big-endian memmap offsets (wvel_utils.read_*); only the
coordinate arrays come from a short xmitgcm load. Matches the direct-read
convention in wave_stacking/build_model_cache.py (>f4 -> float on store).

Usage:
    python build_wvel_cache.py [MODEL ...]   # default: all of MODEL_ORDER
"""

import os
import sys
import time

import numpy as np
import xarray as xr

import wvel_utils as wu

# animation frame cadence (in output steps): 3-hourly high-res -> 12-hourly,
# daily low-res -> daily. Keeps ~160-180 frames over the 3 months either way.
ANIM_STEP = {False: 4, True: 1}  # keyed by MODELS[..]['lowres']


def build_model(model):
    m = wu.MODELS[model]
    out = wu.cache_path(f'wvel_cache_{model}.nc')
    print(f'\n=== {model}  ({m["label"]}) ===', flush=True)

    coords = wu.get_coords(model)
    xc1d, yc1d, zl = coords['xc1d'], coords['yc1d'], coords['zl']
    i, j = coords['i'], coords['j']
    kz = wu.nearest_k(zl, wu.ANIM_DEPTH)
    print(f'point 0N,140W -> i={i}, j={j} '
          f'(XC={xc1d[i]:.3f}, YC={yc1d[j]:.3f});  '
          f'anim level k={kz} (Zl={zl[kz]:.1f} m)', flush=True)

    iters = wu.diag_iters(model)
    times = wu.iter_times(model, iters)
    nt, nz = len(iters), m['NZ']

    # animation slice cropped to the TPOSE24 footprint + 5 deg buffer, so the
    # low-res field is stored over a comparable extent to the high-res domain.
    lon0, lon1 = wu.TARGET_LON - 16.0, wu.TARGET_LON + 11.0
    lat0, lat1 = -11.0, 16.0
    ix = np.where((xc1d >= lon0) & (xc1d <= lon1))[0]
    iy = np.where((yc1d >= lat0) & (yc1d <= lat1))[0]
    sx, sy = slice(ix.min(), ix.max() + 1), slice(iy.min(), iy.max() + 1)

    step = ANIM_STEP[m['lowres']]
    anim_idx = np.arange(0, nt, step)

    w_col = np.full((nt, nz), np.nan, np.float32)
    w_slice = np.full((len(anim_idx), iy.size, ix.size), np.nan, np.float32)

    t0 = time.time()
    a = 0
    for t, it in enumerate(iters):
        # one sequential read of the WVEL field yields both the column and level
        fld = wu.read_field(model, int(it))
        w_col[t] = fld[:, j, i]
        if a < len(anim_idx) and t == anim_idx[a]:
            w_slice[a] = fld[kz, sy, sx]
            a += 1
        if t % 50 == 0 or t == nt - 1:
            el = time.time() - t0
            print(f'  {t+1}/{nt}  ({el:.0f}s)', flush=True)

    ds = xr.Dataset(
        data_vars=dict(
            w_col=(['time', 'depth'], w_col),
            w_slice=(['time_anim', 'YC', 'XC'], w_slice),
        ),
        coords=dict(
            time=times,
            depth=zl.astype(float),
            time_anim=times[anim_idx],
            YC=yc1d[sy].astype(float),
            XC=xc1d[sx].astype(float),
        ),
        attrs=dict(
            model=model, label=m['label'], lowres=int(m['lowres']),
            deltaT=m['deltaT'], anim_depth_m=float(zl[kz]),
            point_lon=float(xc1d[i]), point_lat=float(yc1d[j]),
            spinup_days=wu.SPINUP_DAYS,
        ),
    )
    ds.to_netcdf(out)
    print(f'saved -> {out}  ({time.time()-t0:.0f}s, '
          f'{nt} cols, {len(anim_idx)} frames)', flush=True)


if __name__ == '__main__':
    models = sys.argv[1:] or wu.MODEL_ORDER
    os.makedirs(wu.CACHE_DIR, exist_ok=True)
    for mdl in models:
        build_model(mdl)
