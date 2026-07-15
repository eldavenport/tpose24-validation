"""Near-surface time-mean W maps for several time windows, per run.

Reads only the near-surface level slab (ksurf) from each diag_state file, so it is
fast. Windows: full, first30, last30, last14 (days). Caches one npz per run.

Usage: python wsurf_windows.py <run>
"""

import sys
import numpy as np
import stab_utils as su

SPD = 8


def main(run):
    g = su.load_grid(run)
    depth = g['depth']
    ksurf = int(np.argmin(np.abs(depth - (-50.0))))
    iters = su.diag_iters(run, 'diag_state')
    nt = len(iters)
    slab_off = su.REC_WVEL * su.NX * su.NY * su.NZ * 4 + ksurf * su.NX * su.NY * 4

    acc = np.zeros((nt, su.NY, su.NX), dtype=np.float32)
    for i, it in enumerate(iters):
        path = su.RUNS[run]['dir'] + f'diag_state.{it:010d}.data'
        acc[i] = np.fromfile(path, dtype=su.DT_DTYPE, count=su.NX * su.NY,
                             offset=slab_off).reshape(su.NY, su.NX)

    n30, n14 = 30 * SPD, 14 * SPD
    windows = {
        'full': slice(0, nt),
        'first30': slice(0, min(n30, nt)),
        'last30': slice(max(0, nt - n30), nt),
        'last14': slice(max(0, nt - n14), nt),
    }
    out = {f'wsurf_{k}': acc[s].mean(axis=0) for k, s in windows.items()}
    out['ksurf'] = ksurf
    out['depth_ksurf'] = float(depth[ksurf])
    np.savez(su.cache_path(f'stab_wsurf_win_{run}.npz'), **out)
    print(f'[{run}] ksurf={ksurf} ({depth[ksurf]:.0f} m) -> stab_wsurf_win_{run}.npz')


if __name__ == '__main__':
    main(sys.argv[1])
