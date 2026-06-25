"""
Extract UVEL at 0N/1N/1S, 140W from diag_state files and save to cache NetCDF.
Uses sequential Z-slice reads (NFS-friendly).

Usage:
    python build_model_cache.py <SUBFOLDER>
    SUBFOLDER in {3month, 3month_Ri3, 3month_Ri5}
"""
import sys, os, re, time, glob
import numpy as np
import xarray as xr
import pandas as pd
import xmitgcm

SUBFOLDER  = sys.argv[1] if len(sys.argv) > 1 else '3month_Ri5'
RUN_DIR    = f'/data/SO3/edavenport/tpose24/oct2012_TP6Vel_{SUBFOLDER}'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, SUBFOLDER)
CACHE_FILE = os.path.join(SAVE_DIR, 'cache_model_points.nc')

os.makedirs(SAVE_DIR, exist_ok=True)

NZ, NY, NX = 138, 384, 512
I_UVEL     = 2      # THETA=0, SALT=1, UVEL=2
nz_use     = 77     # levels above ~315 m

IX    = 256   # XC ≈ 220.02°E (≈ 140°W)
IY_0N = 131   # YC ≈ -0.02°N
IY_1N = 155   # YC ≈  0.98°N
IY_1S = 108   # YC ≈ -0.98°N

FIELD_OFFSET = I_UVEL * NZ * NY * NX * 4

data_files = sorted(glob.glob(f'{RUN_DIR}/diag_state.*.data'))
iters      = np.array([int(re.search(r'\.(\d+)\.data', f).group(1)) for f in data_files])
t_mod      = pd.DatetimeIndex(
    [pd.Timestamp('2012-10-01') + pd.Timedelta(seconds=int(i) * 300) for i in iters])
n_t = len(data_files)

print(f'SUBFOLDER : {SUBFOLDER}')
print(f'RUN_DIR   : {RUN_DIR}')
print(f'CACHE_FILE: {CACHE_FILE}')
print(f'Extracting {n_t} files …  ({nz_use} Z-slices per file, sequential read)')

U_0N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
U_1N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
U_1S = np.full((n_t, nz_use), np.nan, dtype=np.float32)

t0 = time.time()
for i, fp in enumerate(data_files):
    with open(fp, 'rb') as f:
        f.seek(FIELD_OFFSET)
        for iz in range(nz_use):
            zslice = np.frombuffer(f.read(NY * NX * 4), dtype='>f4').reshape(NY, NX)
            U_0N[i, iz] = zslice[IY_0N, IX]
            U_1N[i, iz] = zslice[IY_1N, IX]
            U_1S[i, iz] = zslice[IY_1S, IX]
    if i % 50 == 0 or i == n_t - 1:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed if elapsed > 0 else 0
        eta     = (n_t - i - 1) / rate if rate > 0 else 0
        print(f'  {i+1}/{n_t}  ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

print(f'Extraction done in {time.time()-t0:.1f}s')

ds_g  = xmitgcm.open_mdsdataset(RUN_DIR, prefix=['diag_state'],
                                  iters=[int(iters[0])], read_grid=True)
Z_mod = ds_g.Z.values[:nz_use]

ds_out = xr.Dataset(
    {'U_0N': (['time', 'depth'], U_0N.astype(float)),
     'U_1N': (['time', 'depth'], U_1N.astype(float)),
     'U_1S': (['time', 'depth'], U_1S.astype(float))},
    coords={'time': t_mod, 'depth': Z_mod})
ds_out.to_netcdf(CACHE_FILE)
print('Saved →', CACHE_FILE)
