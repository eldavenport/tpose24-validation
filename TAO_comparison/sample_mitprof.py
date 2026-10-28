"""
sample_mitprof.py
-----------------
Offline model-equivalent TAO profiles for a 6-month combined farm, in the same
MITprof layout the online PROFILES package produces (raw obs vars + prof_*estim
filled from the model).  This is the product TAO notebooks 1/1c and
wave_stacking read; the 6-month farm has no online profiles, so we build it here.

For each obs profile inside the TPOSE24 domain and [T0,T1], the model field is
sampled at the nearest 3-hourly output, nearest grid column, linearly in depth.
Profiles are grouped by model-output index so each diag_state file is opened
once (not once per profile).  The 2012 and 2013 obs files are concatenated so a
single output spans Oct 2012 - Apr 2013.  prof_lon is written in -180:180 to
match the notebooks' `lons == -140.` selection.

Usage:
    conda run -n tpose python sample_mitprof.py <run_dir> <out_dir>
      run_dir : combined farm (sixmonth_hydro | sixmonth_QH)
      out_dir : where to write <name>_model.nc (the farm's PROF_eq)
"""
import os, sys, re, glob
import numpy as np
import xarray as xr

PROF_DIR = '/data/SO3/edavenport/tpose24/profiles'
LON0, LON1, LAT0, LAT1 = 209.35, 230.65, -5.48, 10.48
REF = np.datetime64(os.environ.get('TPOSE_REF_DATE', '2012-10-01'))
DELTA_T = int(float(os.environ.get('TPOSE_DELTAT', 60)))
DT, MISS = '>f4', -999.0
NZ, NY, NX = 138, 384, 512
NFLDS = 7
REC = {'THETA': 0, 'SALT': 1, 'UVEL': 2, 'VVEL': 3}

# merged output name -> (2012 obs, 2013 obs, {obs_var: model_rec})
JOBS = {
    'TAO_WO_ALL_ADCP_v2': ('TAO_WO_2012_ADCP_v2.nc', 'TAO_WO_2013_ADCP.nc',
                           {'prof_U': 'UVEL', 'prof_V': 'VVEL'}),
    'TAO_WO_ALL_CTD_daily_ED': ('TAO_WO_2012_CTD_daily_ED.nc', 'TAO_WO_2013_CTD_daily_ED.nc',
                                {'prof_T': 'THETA', 'prof_S': 'SALT'}),
}


def load_grid(run_dir):
    xc = np.fromfile(os.path.join(run_dir, 'XC.data'), dtype=DT).reshape(NY, NX)
    yc = np.fromfile(os.path.join(run_dir, 'YC.data'), dtype=DT).reshape(NY, NX)
    rc = np.fromfile(os.path.join(run_dir, 'RC.data'), dtype=DT)  # (NZ,) centre depths, negative
    return xc[0].astype('f8'), yc[:, 0].astype('f8'), rc.astype('f8')


def model_timeline(run_dir):
    its = sorted(int(re.search(r'\.(\d+)\.data$', f).group(1))
                 for f in glob.glob(os.path.join(run_dir, 'diag_state.*.data')))
    its = np.array(its)
    t = REF.astype('datetime64[s]') + (its * DELTA_T).astype('timedelta64[s]')
    return its, t


def obs_times(ds):
    ymd = ds['prof_YYYYMMDD'].values.astype('int64')
    hms = ds['prof_HHMMSS'].values.astype('int64')
    y, mo, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    hh, mm = hms // 10000, (hms // 100) % 100
    return np.array([np.datetime64(f'{a:04d}-{b:02d}-{c:02d}T{e:02d}:{f:02d}')
                     for a, b, c, e, f in zip(y, mo, d, hh, mm)],
                    dtype='datetime64[m]').astype('datetime64[s]')


def main(run_dir, out_dir):
    lon1d, lat1d, z = load_grid(run_dir)
    its, mt = model_timeline(run_dir)
    # validation window = model output span (obs outside it are dropped)
    T0 = mt[0].astype('datetime64[D]')
    T1 = mt[-1].astype('datetime64[D]') + np.timedelta64(1, 'D')
    os.makedirs(out_dir, exist_ok=True)
    print(f'run {run_dir}: {len(its)} outputs {str(mt[0])[:10]}..{str(mt[-1])[:10]} '
          f'window {str(T0)}..{str(T1)}')

    for name, (f12, f13, varmap) in JOBS.items():
        d12 = xr.open_dataset(os.path.join(PROF_DIR, f12), decode_times=False)
        d13 = xr.open_dataset(os.path.join(PROF_DIR, f13), decode_times=False)
        ds = xr.concat([d12, d13], dim='iPROF')
        # prof_depth has no iPROF dim; concat broadcasts it to 2D -> collapse back
        if ds['prof_depth'].ndim == 2:
            ds['prof_depth'] = ds['prof_depth'].isel(iPROF=0)
        plon = ds['prof_lon'].values % 360
        plat = ds['prof_lat'].values
        pdep = ds['prof_depth'].values
        pt = obs_times(ds)

        sel = ((plon >= LON0) & (plon <= LON1) & (plat >= LAT0) & (plat <= LAT1)
               & (pt >= T0) & (pt < T1))
        idx = np.where(sel)[0]
        print(f'{name}: {len(idx)} in-domain profiles ({len(ds.iPROF)} total)')

        ndep = len(pdep)
        estim = {ov: np.full((len(ds.iPROF), ndep), np.nan) for ov in varmap}
        # nearest model-output index per selected profile, then group by it
        k_of = np.abs(mt[None, :] - pt[idx][:, None]).argmin(axis=1)
        j_of = np.abs(lat1d[None, :] - plat[idx][:, None]).argmin(axis=1)
        i_of = np.abs(lon1d[None, :] - plon[idx][:, None]).argmin(axis=1)

        order = np.argsort(k_of)
        for grp in _group_bounds(k_of[order]):
            g = order[grp]            # positions into idx that share one model output
            k = k_of[g[0]]
            path = os.path.join(run_dir, f'diag_state.{int(its[k]):010d}.data')
            mm = np.memmap(path, dtype=DT, mode='r', shape=(NFLDS, NZ, NY, NX))
            for ov, mv in varmap.items():
                rec = mm[REC[mv]]
                for p in g:
                    col = np.asarray(rec[:, j_of[p], i_of[p]], dtype='f8')
                    col[col == MISS] = np.nan
                    good = np.isfinite(col)
                    if good.sum() < 2:
                        continue
                    estim[ov][idx[p]] = np.interp(-pdep, z[good][::-1], col[good][::-1],
                                                  left=np.nan, right=np.nan)
            del mm

        out = ds.copy()
        out['prof_lon'] = ('iPROF', np.where(plon > 180, plon - 360, plon))
        for ov in varmap:
            est = 'prof_' + ov.split('_')[1] + 'estim'  # prof_U -> prof_Uestim
            out[est] = (('iPROF', 'iDEPTH'), estim[ov])
        p = os.path.join(out_dir, name + '_model.nc')
        out.to_netcdf(p)
        nfin = np.isfinite(estim[list(varmap)[0]]).any(1).sum()
        print(f'  wrote {p}  ({nfin} profiles with model estimate)')


def _group_bounds(sorted_k):
    """Yield arrays of positions (into the sorted order) sharing each k value."""
    boundaries = np.where(np.diff(sorted_k) != 0)[0] + 1
    groups = np.split(np.arange(len(sorted_k)), boundaries)
    return groups


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
