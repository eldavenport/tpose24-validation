"""
build_ri_cache.py <run_dir> <tag>
---------------------------------
Gradient Richardson number Ri = N2/S2 on the 140W (220E) meridional transect,
3S-3N, full depth, every 3-hourly output.  Reduces over time to the mean and
lower-quartile Ri at each (lat, interface depth), saved to cache/ for plotting.

  N2 = -(g/rho0) * DRHODR            (DRHODR = diag_state record 6, on W-levels)
  S2 = (dU/dz)^2 + (dV/dz)^2         (centered difference of UVEL/VVEL)
Both evaluated at cell interfaces RF[1..NZ-1] (between centres k-1 and k).

Ri is clipped to [1e-3, 1e3] before reducing: the floor lumps static-instability
(N2<0) events as "very low Ri"; the ceiling keeps weak-shear points finite.

Output -> cache/cache_ri_<tag>.nc : Ri_mean, Ri_q25 (depth, lat) + bottom depth.
"""
import sys, os, re, glob
import numpy as np
import xarray as xr

DT = '>f4'; MISS = -999.0
NZ, NY, NX = 138, 384, 512
REC = {'UVEL': 2, 'VVEL': 3, 'DRHODR': 6}
G, RHO0 = 9.81, 1027.0
LON_T = 220.0                      # 140 W
LAT0, LAT1 = -3.0, 3.0
RI_LO, RI_HI = 1e-3, 1e3
CACHE = '/data/SO3/edavenport/tpose24/cache'


def grid(run_dir):
    xc = np.fromfile(f'{run_dir}/XC.data', dtype=DT).reshape(NY, NX)[0]
    yc = np.fromfile(f'{run_dir}/YC.data', dtype=DT).reshape(NY, NX)[:, 0]
    rc = np.fromfile(f'{run_dir}/RC.data', dtype=DT).astype('f8')          # (NZ,) centres, neg
    rf = np.fromfile(f'{run_dir}/RF.data', dtype=DT).astype('f8')          # (NZ+1,) faces, neg
    dep = np.fromfile(f'{run_dir}/Depth.data', dtype=DT).reshape(NY, NX)   # positive
    return xc, yc, rc, rf, dep


def main(run_dir, tag):
    xc, yc, rc, rf, dep = grid(run_dir)
    i0 = int(np.argmin(np.abs(xc - LON_T)))
    jb = np.where((yc >= LAT0) & (yc <= LAT1))[0]
    lat = yc[jb]
    bottom = dep[jb, i0]
    dz_c = rc[:-1] - rc[1:]                     # (NZ-1,) centre spacing >0, at interfaces k=1..NZ-1
    zl = rf[1:NZ]                               # (NZ-1,) interface depths (neg) for k=1..NZ-1
    its = np.array(sorted(int(re.search(r'\.(\d+)\.data$', f).group(1))
                          for f in glob.glob(f'{run_dir}/diag_state.*.data')))
    times = (np.datetime64('2012-10-01') + (its * 60).astype('timedelta64[s]'))
    j15 = int(np.argmin(np.abs(lat - 1.5)))        # 1.5N column for the time series
    nt, nk, nj = len(its), NZ - 1, len(jb)
    # dry cells below the seafloor hold 0.0 (not -999); hFacC==0 -> mask to NaN
    hf = np.fromfile(f'{run_dir}/hFacC.data', dtype=DT).reshape(NZ, NY, NX)
    dry = (hf[:, jb, i0] == 0.0)
    if dry.shape != (NZ, nj):
        dry = dry.T
    print(f'{tag}: i={i0} (XC={xc[i0]:.2f}E), {nj} lats {lat[0]:.2f}..{lat[-1]:.2f}, {nt} times')

    fsz = NZ * NY * NX * 4
    def read_slice(fh, rec):
        """Contiguous read of one field (sequential/NFS-friendly), sliced to (NZ, nj)."""
        fh.seek(REC[rec] * fsz)
        blk = np.frombuffer(fh.read(fsz), dtype=DT).reshape(NZ, NY, NX)
        sl = blk[:, jb, i0].astype('f8')
        if sl.shape != (NZ, nj):
            sl = sl.T
        return sl

    ri = np.full((nt, nk, nj), np.nan)
    for t, it in enumerate(its):
        with open(f'{run_dir}/diag_state.{it:010d}.data', 'rb') as fh:
            u = read_slice(fh, 'UVEL')
            v = read_slice(fh, 'VVEL')
            dr = read_slice(fh, 'DRHODR')
        for a in (u, v, dr):
            a[a == MISS] = np.nan
            a[dry] = np.nan          # exclude below-seafloor (hFacC==0) cells
        dudz = (u[:-1] - u[1:]) / dz_c[:, None]                   # (NZ-1, nj) at interfaces
        dvdz = (v[:-1] - v[1:]) / dz_c[:, None]
        s2 = dudz ** 2 + dvdz ** 2
        n2 = -(G / RHO0) * dr[1:NZ]                               # DRHODR at interface k
        with np.errstate(divide='ignore', invalid='ignore'):
            r = n2 / np.where(s2 > 1e-12, s2, np.nan)
        ri[t] = np.clip(r, RI_LO, RI_HI)
        if t % 200 == 0:
            print(f'  {t+1}/{nt}', flush=True)

    with np.errstate(invalid='ignore'):
        ri_mean = np.nanmean(ri, axis=0)
        ri_q25 = np.nanpercentile(ri, 25, axis=0)
        frac_lo = np.nanmean((ri < 0.25).astype('f8'), axis=0)   # fraction of time Ri<0.25

    ds = xr.Dataset(
        dict(Ri_mean=(['depth', 'lat'], ri_mean),
             Ri_q25=(['depth', 'lat'], ri_q25),
             frac_Ri_lt_025=(['depth', 'lat'], frac_lo),
             bottom=(['lat'], bottom),
             Ri_ts=(['time', 'depth'], ri[:, :, j15])),      # 1.5N time series
        coords=dict(depth=zl, lat=lat, time=times),
        attrs=dict(run_dir=run_dir, lon=float(xc[i0]), ntime=nt,
                   lat_ts=float(lat[j15]), bottom_ts=float(bottom[j15]),
                   note='Ri clipped [1e-3,1e3]; N2 from DRHODR, S2 centered diff, at interfaces'))
    os.makedirs(CACHE, exist_ok=True)
    out = f'{CACHE}/cache_ri_{tag}.nc'
    ds.to_netcdf(out)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
