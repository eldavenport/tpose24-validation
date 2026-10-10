"""
sample_profiles.py
------------------
Offline model-equivalent profiles for the Jan-Apr 2013 extension.

The run's online PROFILES package used the *2012* obs files (which end Dec 31
2012), so no model-equivalent profiles exist for the extension.  Here we sample
the concatenated TPOSE24 record at the Jan-Apr 2013 TAO profile locations/times
directly from diag_state (nearest 3-hourly output, nearest grid column, linear
in depth) -- the equivalent of what the online package would have produced.

Only the 140 degW (220 degE) TAO line falls inside the TPOSE24 domain
(209.35-230.65 degE): CTD T/S at several latitudes and the 0degN ADCP (EUC).

Writes  PROF_eq/<name>_model.nc  in the extension run dir, with the model values
on the obs (iPROF, iDEPTH) layout alongside the obs for direct comparison.

Usage:  python sample_profiles.py
"""
import os, sys
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_utils as e

PROF_DIR = '/data/SO3/edavenport/tpose24/profiles'
OUT_DIR = '/data/SO3/edavenport/tpose24/jan2013_from_pickup/PROF_eq'
LON0, LON1, LAT0, LAT1 = 209.35, 230.65, -5.48, 10.48
T0, T1 = np.datetime64('2013-01-01'), np.datetime64('2013-04-01')

# obs file -> (model vars to sample) keyed to obs var names
FILES = {
    'TAO_WO_2013_CTD_daily_ED': {'prof_T': 'THETA', 'prof_S': 'SALT'},
    'TAO_WO_2013_ADCP':         {'prof_U': 'UVEL', 'prof_V': 'VVEL'},
}


def model_timeline():
    """All (dir, iter, datetime64[m]) model outputs across parent+extension."""
    rows = []
    for d in (e.PAR_DIR, e.NEW_DIR):
        its = e._iters(d)
        secs = (its * e.DELTA_T).astype('timedelta64[s]')
        dt = e.REF_DATE.astype('datetime64[s]') + secs
        for it, t in zip(its, dt):
            rows.append((d, int(it), t))
    rows.sort(key=lambda r: r[2])
    dirs = np.array([r[0] for r in rows])
    its = np.array([r[1] for r in rows])
    ts = np.array([r[2] for r in rows], dtype='datetime64[s]')
    return dirs, its, ts


def column(run_dir, it, var, nz=e.NZ):
    """Model column (NZ,) of ``var`` at grid index via memmap (j, i set by caller)."""
    # returns a function bound to the memmap for repeated (j,i) picks
    path = os.path.join(run_dir, f'diag_state.{it:010d}.data')
    mm = np.memmap(path, dtype=e.DT, mode='r', shape=(e.NFLDS_STATE, e.NZ, e.NY, e.NX))
    return mm[e.REC[var]]


def main():
    grid = e.load_grid()
    lon1d, lat1d, z = grid['lon'], grid['lat'], grid['z']
    dirs, its, ts = model_timeline()
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, varmap in FILES.items():
        ds = xr.open_dataset(os.path.join(PROF_DIR, name + '.nc'), decode_times=False)
        plon = ds['prof_lon'].values % 360
        plat = ds['prof_lat'].values
        pdep = ds['prof_depth'].values  # (iDEPTH,) positive down
        ymd = ds['prof_YYYYMMDD'].values.astype('int64')
        hms = ds['prof_HHMMSS'].values.astype('int64')
        # obs datetime
        yy, mm_, dd = ymd // 10000, (ymd // 100) % 100, ymd % 100
        HH, MM = hms // 10000, (hms // 100) % 100
        ptime = np.array([np.datetime64(f'{y:04d}-{m:02d}-{d:02d}T{h:02d}:{mi:02d}')
                          for y, m, d, h, mi in zip(yy, mm_, dd, HH, MM)],
                         dtype='datetime64[m]').astype('datetime64[s]')

        sel = ((plon >= LON0) & (plon <= LON1) & (plat >= LAT0) & (plat <= LAT1)
               & (ptime >= T0) & (ptime < T1))
        idx = np.where(sel)[0]
        print(f"{name}: {len(idx)} in-domain profiles in {T0}..{T1}")
        if len(idx) == 0:
            continue

        ndep = len(pdep)
        out = {ov: np.full((len(idx), ndep), np.nan) for ov in varmap}
        for n, ip in enumerate(idx):
            k = np.argmin(np.abs(ts - ptime[ip]))          # nearest model time
            j = int(np.argmin(np.abs(lat1d - plat[ip])))    # nearest grid column
            i = int(np.argmin(np.abs(lon1d - plon[ip])))
            for ov, mv in varmap.items():
                fld = column(dirs[k], its[k], mv)
                col = np.asarray(fld[:, j, i], dtype='f8')
                col[col == e.MISS] = np.nan
                good = np.isfinite(col)
                if good.sum() < 2:
                    continue
                # interp model (at z, negative-down) to -obs_depth
                out[ov][n] = np.interp(-pdep, z[good][::-1], col[good][::-1],
                                       left=np.nan, right=np.nan)

        co = {
            'prof_lon': ('iPROF', plon[idx]),
            'prof_lat': ('iPROF', plat[idx]),
            'prof_time': ('iPROF', ptime[idx]),
            'prof_depth': ('iDEPTH', pdep),
        }
        data = {}
        for ov, mv in varmap.items():
            obs = ds[ov].values[idx].astype('f8')
            obs[np.abs(obs) > 900] = np.nan  # MITprof missing-data fill -> NaN
            data[ov + '_model'] = (('iPROF', 'iDEPTH'), out[ov])
            data[ov + '_obs'] = (('iPROF', 'iDEPTH'), obs)
        xr.Dataset(data, coords=co).to_netcdf(os.path.join(OUT_DIR, name + '_model.nc'))
        print(f"  wrote {name}_model.nc")


if __name__ == '__main__':
    main()
