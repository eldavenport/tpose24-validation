"""First-30-day depth-averaged horizontal velocity maps, per run.

Depth averages over 0-70 m and 0-250 m of U and V. Caches the time-mean map and
the temporal std map (eddy amplitude) for each band. Reads only the top levels
needed (partial-depth read of the U/V records).

Usage: python preprocess_uv_maps.py <run>

Assumptions:
- U averaged with hFacW*DRF, V with hFacS*DRF.
- First 30 days = first 240 outputs (8/day).
"""

import sys
import numpy as np
import stab_utils as su

N30 = 30 * 8


def depth_avg(vel, hFac, drf, band):
    w = (hFac * drf[:, None, None])[band]
    return (vel[band] * w).sum(0) / w.sum(0)


def main(run):
    g = su.load_grid(run)
    depth, DRF = g['depth'], g['DRF']
    b70 = depth >= -70
    b250 = depth >= -250
    kmax = int(np.where(b250)[0].max()) + 1          # levels to read
    b70k, b250k = b70[:kmax], b250[:kmax]
    drf = DRF[:kmax]
    hW, hS = g['hFacW'][:kmax], g['hFacS'][:kmax]

    iters = su.diag_iters(run, 'diag_state')[:N30]
    nt = len(iters)
    shp = (su.NY, su.NX)
    acc = {k: np.zeros(shp) for k in
           ('u70', 'v70', 'u250', 'v250', 'u70sq', 'v70sq', 'u250sq', 'v250sq')}

    cU = kmax * su.NX * su.NY
    for it in iters:
        path = su.RUNS[run]['dir'] + f'diag_state.{it:010d}.data'
        U = np.fromfile(path, dtype=su.DT_DTYPE, count=cU,
                        offset=su.REC_UVEL * su.NX * su.NY * su.NZ * 4).reshape(kmax, su.NY, su.NX)
        V = np.fromfile(path, dtype=su.DT_DTYPE, count=cU,
                        offset=su.REC_VVEL * su.NX * su.NY * su.NZ * 4).reshape(kmax, su.NY, su.NX)
        u70 = depth_avg(U, hW, drf, b70k); v70 = depth_avg(V, hS, drf, b70k)
        u250 = depth_avg(U, hW, drf, b250k); v250 = depth_avg(V, hS, drf, b250k)
        acc['u70'] += u70; acc['v70'] += v70
        acc['u250'] += u250; acc['v250'] += v250
        acc['u70sq'] += u70**2; acc['v70sq'] += v70**2
        acc['u250sq'] += u250**2; acc['v250sq'] += v250**2

    out = {}
    for b in ('70', '250'):
        um, vm = acc[f'u{b}'] / nt, acc[f'v{b}'] / nt
        us = np.sqrt(np.maximum(acc[f'u{b}sq'] / nt - um**2, 0))
        vs = np.sqrt(np.maximum(acc[f'v{b}sq'] / nt - vm**2, 0))
        out[f'umean{b}'] = um; out[f'vmean{b}'] = vm
        out[f'ustd{b}'] = us; out[f'vstd{b}'] = vs
    out['ndays'] = nt / 8.0
    np.savez(su.cache_path(f'stab_uv30_{run}.npz'), **out)
    print(f'[{run}] kmax={kmax} ({depth[kmax-1]:.0f} m) nt={nt} -> stab_uv30_{run}.npz')


if __name__ == '__main__':
    main(sys.argv[1])
