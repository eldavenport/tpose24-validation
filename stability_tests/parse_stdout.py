"""Parse dt300 STDOUT %MON monitor series and cache to npz.

Usage: python parse_stdout.py <run>   (only dt300 has STDOUT)
"""

import sys
import glob
import numpy as np
import stab_utils as su

KEYS = ['advcfl_wvel_max', 'advcfl_W_hf_max', 'advcfl_uvel_max', 'advcfl_vvel_max',
        'dynstat_wvel_max', 'dynstat_wvel_min', 'dynstat_wvel_mean', 'dynstat_wvel_sd']


def main(run):
    files = sorted(glob.glob(su.RUNS[run]['dir'] + 'STDOUT*'))
    if not files:
        print(f'[{run}] no STDOUT files found'); return
    print(f'[{run}] parsing {files}')
    agg = {k: [] for k in KEYS + ['time_tsnumber']}
    for f in files:
        d = su.parse_stdout_monitor(f, KEYS)
        for k in agg:
            agg[k].append(d[k])
    data = {k: np.concatenate(v) for k, v in agg.items()}
    iters = data.pop('time_tsnumber').astype(int)
    order = np.argsort(iters)
    iters = iters[order]
    days = su.iter_to_days(run, iters)
    save = {k: v[order] for k, v in data.items()}
    out = su.cache_path(f'stab_stdout_{run}.npz')
    np.savez(out, iters=iters, days=days, **save)
    print(f'[{run}] {len(iters)} monitor records -> {out}')


if __name__ == '__main__':
    main(sys.argv[1])
