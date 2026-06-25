"""
Extract UVEL, VVEL, DRHODR at 0N/1N/1S, 140W from diag_state files.
Extends to 1200 m depth (nz_use=96 levels, deepest ≈ -1250 m).

diag_state field order: THETA=0, SALT=1, UVEL=2, VVEL=3, WVEL=4, PHIHYD=5, DRHODR=6

Usage:
    conda run -n tpose python build_edj_cache.py <SUBFOLDER>
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
CACHE_FILE = os.path.join(SAVE_DIR, 'cache_edj.nc')

os.makedirs(SAVE_DIR, exist_ok=True)

NZ, NY, NX = 138, 384, 512
nz_use     = 96   # covers to ≈ -1250 m; display clipped to -1200 m in plots

I_UVEL   = 2
I_VVEL   = 3
I_DRHODR = 6

SLICE_BYTES = NY * NX * 4
FIELD_SIZE  = NZ * SLICE_BYTES  # bytes for one complete 3-D field

IX    = 256   # XC ≈ 220.02°E  ≈ 140°W
IY_0N = 131   # YC ≈ -0.02°N
IY_1N = 155   # YC ≈  0.98°N
IY_1S = 108   # YC ≈ -0.98°N

data_files = sorted(glob.glob(f'{RUN_DIR}/diag_state.*.data'))
iters      = np.array([int(re.search(r'\.(\d+)\.data', f).group(1)) for f in data_files])
t_mod      = pd.DatetimeIndex(
    [pd.Timestamp('2012-10-01') + pd.Timedelta(seconds=int(i) * 300) for i in iters])
n_t = len(data_files)

print(f'SUBFOLDER : {SUBFOLDER}')
print(f'RUN_DIR   : {RUN_DIR}')
print(f'CACHE_FILE: {CACHE_FILE}')
print(f'Files     : {n_t}')
print(f'nz_use    : {nz_use}  (reading {nz_use * SLICE_BYTES / 1e6:.0f} MB per field per file)')

U_0N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
U_1N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
U_1S = np.full((n_t, nz_use), np.nan, dtype=np.float32)
V_0N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
V_1N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
V_1S = np.full((n_t, nz_use), np.nan, dtype=np.float32)
D_0N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
D_1N = np.full((n_t, nz_use), np.nan, dtype=np.float32)
D_1S = np.full((n_t, nz_use), np.nan, dtype=np.float32)

t0 = time.time()
for i, fp in enumerate(data_files):
    with open(fp, 'rb') as fh:
        # Read all nz_use levels as one contiguous block per field (3 seeks per file).
        fh.seek(I_UVEL * FIELD_SIZE)
        blk = np.frombuffer(fh.read(nz_use * SLICE_BYTES), dtype='>f4').reshape(nz_use, NY, NX)
        U_0N[i] = blk[:, IY_0N, IX]
        U_1N[i] = blk[:, IY_1N, IX]
        U_1S[i] = blk[:, IY_1S, IX]

        fh.seek(I_VVEL * FIELD_SIZE)
        blk = np.frombuffer(fh.read(nz_use * SLICE_BYTES), dtype='>f4').reshape(nz_use, NY, NX)
        V_0N[i] = blk[:, IY_0N, IX]
        V_1N[i] = blk[:, IY_1N, IX]
        V_1S[i] = blk[:, IY_1S, IX]

        fh.seek(I_DRHODR * FIELD_SIZE)
        blk = np.frombuffer(fh.read(nz_use * SLICE_BYTES), dtype='>f4').reshape(nz_use, NY, NX)
        D_0N[i] = blk[:, IY_0N, IX]
        D_1N[i] = blk[:, IY_1N, IX]
        D_1S[i] = blk[:, IY_1S, IX]

    if i % 50 == 0 or i == n_t - 1:
        elapsed = time.time() - t0
        rate    = (i + 1) / elapsed if elapsed > 0 else 0
        eta     = (n_t - i - 1) / rate if rate > 0 else 0
        print(f'  {i+1}/{n_t}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)', flush=True)

print(f'Extraction done in {time.time() - t0:.1f}s')

# Depth coordinate from grid
ds_g  = xmitgcm.open_mdsdataset(RUN_DIR, prefix=['diag_state'],
                                  iters=[int(iters[0])], read_grid=True)
Z_mod = ds_g.Z.values[:nz_use]

ds_out = xr.Dataset(
    {'U_0N':      (['time', 'depth'], U_0N.astype(float)),
     'U_1N':      (['time', 'depth'], U_1N.astype(float)),
     'U_1S':      (['time', 'depth'], U_1S.astype(float)),
     'V_0N':      (['time', 'depth'], V_0N.astype(float)),
     'V_1N':      (['time', 'depth'], V_1N.astype(float)),
     'V_1S':      (['time', 'depth'], V_1S.astype(float)),
     'DRHODR_0N': (['time', 'depth'], D_0N.astype(float)),
     'DRHODR_1N': (['time', 'depth'], D_1N.astype(float)),
     'DRHODR_1S': (['time', 'depth'], D_1S.astype(float))},
    coords={'time': t_mod, 'depth': Z_mod})
ds_out.attrs['subfolder'] = SUBFOLDER
ds_out.attrs['nz_use']    = nz_use
ds_out.attrs['IX']        = IX
ds_out.to_netcdf(CACHE_FILE)
print('Saved →', CACHE_FILE)
