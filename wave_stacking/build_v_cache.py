"""
Extract VVEL at 0N/1N/1S, 140W from diag_state files and save to V cache NetCDF.
Reads only the VVEL field (field index 3) sequentially per file.

Usage:
    python build_v_cache.py <SUBFOLDER>
    SUBFOLDER in {3month, 3month_Ri3, 3month_Ri5}
"""
import sys, os, re, time, glob
import numpy as np
import xarray as xr
import pandas as pd

SUBFOLDER  = sys.argv[1] if len(sys.argv) > 1 else '3month_Ri5'
RUN_DIR    = f'/data/SO3/edavenport/tpose24/oct2012_TP6Vel_{SUBFOLDER}'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.join(SCRIPT_DIR, SUBFOLDER)
CACHE_FILE = os.path.join(SAVE_DIR, 'cache_model_points_V.nc')

os.makedirs(SAVE_DIR, exist_ok=True)

NZ, NY, NX = 138, 384, 512
I_VVEL     = 3      # THETA=0, SALT=1, UVEL=2, VVEL=3
nz_use     = 77

# VVEL is on YG grid; use same indices as YC for negligible ~0.04° offset
IX    = 256
IY_0N = 131
IY_1N = 155
IY_1S = 108

FIELD_OFFSET = I_VVEL * NZ * NY * NX * 4

data_files = sorted(glob.glob(f'{RUN_DIR}/diag_state.*.data'))
iters      = np.array([int(re.search(r'\.(\d+)\.data', f).group(1)) for f in data_files])
t_mod      = pd.DatetimeIndex(
    [pd.Timestamp('2012-10-01') + pd.Timedelta(seconds=int(i) * 300) for i in iters])
n_t = len(data_files)

print(f'SUBFOLDER : {SUBFOLDER}')
print(f'CACHE_FILE: {CACHE_FILE}')
print(f'Extracting VVEL from {n_t} files …')

V_0N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
V_1N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
V_1S = np.full((n_t, nz_use), np.nan, dtype=np.float32)

t0 = time.time()
for i, fp in enumerate(data_files):
    with open(fp, 'rb') as f:
        f.seek(FIELD_OFFSET)
        for iz in range(nz_use):
            zslice = np.frombuffer(f.read(NY * NX * 4), dtype='>f4').reshape(NY, NX)
            V_0N[i, iz] = zslice[IY_0N, IX]
            V_1N[i, iz] = zslice[IY_1N, IX]
            V_1S[i, iz] = zslice[IY_1S, IX]
    if i % 50 == 0 or i == n_t - 1:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed if elapsed > 0 else 0
        eta     = (n_t - i - 1) / rate if rate > 0 else 0
        print(f'  {i+1}/{n_t}  ({elapsed:.0f}s, ETA {eta:.0f}s)', flush=True)

print(f'Extraction done in {time.time()-t0:.1f}s')

# Load depth coordinate from existing U cache
ds_u  = xr.open_dataset(os.path.join(SAVE_DIR, 'cache_model_points.nc'))
Z_mod = ds_u['depth'].values

ds_out = xr.Dataset(
    {'V_0N': (['time', 'depth'], V_0N.astype(float)),
     'V_1N': (['time', 'depth'], V_1N.astype(float)),
     'V_1S': (['time', 'depth'], V_1S.astype(float))},
    coords={'time': t_mod, 'depth': Z_mod})
ds_out.to_netcdf(CACHE_FILE)
print('Saved →', CACHE_FILE)
