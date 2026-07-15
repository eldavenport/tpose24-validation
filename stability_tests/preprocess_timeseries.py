"""Single-pass reduction of diag_state/diag_surf into cached time series and maps.

Usage: python preprocess_timeseries.py <run>   where run in {dt300, dt60}

Caches one npz per run to CACHE_DIR. Reads UVEL/VVEL/WVEL once per diag_state file.

Assumptions:
- Near-surface W map level = level nearest 50 m depth (k=30, ~51 m).
- Depth-avg W map = DRF-weighted mean over top 100 m.
- Equilibrated windows are the last 14 and last 30 simulated days (8 steps/day).
- CFL_w column map = max over depth of |W|*deltaT/DRF.
"""

import sys
import time
import numpy as np
import stab_utils as su

STEPS_PER_DAY = 8  # 3-hourly output


def main(run):
    g = su.load_grid(run)
    box = g['box']
    DRF = g['DRF']
    depth = g['depth']
    deltaT = su.RUNS[run]['deltaT']
    RAC, hFacC, hFacW, hFacS = g['RAC'], g['hFacC'], g['hFacW'], g['hFacS']

    iters = su.diag_iters(run, 'diag_state')
    nt = len(iters)
    days = su.iter_to_days(run, iters)

    ksurf = int(np.argmin(np.abs(depth - (-50.0))))  # near-surface level
    ktop100 = np.where(depth >= -100.0)[0]           # levels within top 100 m
    w100 = DRF[ktop100] / DRF[ktop100].sum()

    win14 = set(range(max(0, nt - 14 * STEPS_PER_DAY), nt))
    win30 = set(range(max(0, nt - 30 * STEPS_PER_DAY), nt))

    # per-time series
    wprof_dom = np.full((nt, su.NZ), np.nan)
    wprof_box = np.full((nt, su.NZ), np.nan)
    utrans_box = np.full(nt, np.nan); vtrans_box = np.full(nt, np.nan)
    utrans_dom = np.full(nt, np.nan); vtrans_dom = np.full(nt, np.nan)
    cflw_dom_max = np.full(nt, np.nan)
    cflw_box_mean = np.full(nt, np.nan)
    wabs_dom_max = np.full(nt, np.nan)
    wstd_dom = np.full(nt, np.nan)
    eta_dom = np.full(nt, np.nan); eta_box = np.full(nt, np.nan)

    surf_iters = set(su.diag_iters(run, 'diag_surf').tolist())

    # spatial accumulators (NY,NX)
    shp = (su.NY, su.NX)
    cfl_sum = np.zeros(shp); cfl_max = np.zeros(shp)
    wsurf_sum = {'full': np.zeros(shp), 'w14': np.zeros(shp), 'w30': np.zeros(shp)}
    w100_sum = {'full': np.zeros(shp), 'w14': np.zeros(shp), 'w30': np.zeros(shp)}
    cnt = {'full': 0, 'w14': 0, 'w30': 0}

    RACbox = RAC[box]
    t0 = time.time()
    for i, it in enumerate(iters):
        U, V, W = su.read_uvw(run, it)

        wprof_dom[i] = su.area_weighted_profile(W, hFacC, RAC)
        wprof_box[i] = su.area_weighted_profile(W[:, box[0], box[1]],
                                                hFacC[:, box[0], box[1]], RACbox)

        uT = su.column_transport(U, hFacW, DRF)
        vT = su.column_transport(V, hFacS, DRF)
        utrans_dom[i] = (uT * RAC).sum() / RAC.sum()
        vtrans_dom[i] = (vT * RAC).sum() / RAC.sum()
        utrans_box[i] = (uT[box] * RACbox).sum() / RACbox.sum()
        vtrans_box[i] = (vT[box] * RACbox).sum() / RACbox.sum()

        cflw = su.cfl_w_field(W, deltaT, DRF)
        colmax = cflw.max(axis=0)
        cflw_dom_max[i] = colmax.max()
        cflw_box_mean[i] = colmax[box].mean()
        wabs_dom_max[i] = np.abs(W).max()
        wstd_dom[i] = W.std()

        if it in surf_iters:
            eta = su.read_etan(run, it)
            eta_dom[i] = (eta * RAC).sum() / RAC.sum()
            eta_box[i] = (eta[box] * RACbox).sum() / RACbox.sum()

        cfl_sum += colmax
        np.maximum(cfl_max, colmax, out=cfl_max)

        wsurf = W[ksurf]
        w100m = (W[ktop100] * w100[:, None, None]).sum(axis=0)
        for tag, sel in (('full', True), ('w14', i in win14), ('w30', i in win30)):
            if sel:
                wsurf_sum[tag] += wsurf
                w100_sum[tag] += w100m
                cnt[tag] += 1

        if i % 20 == 0 or i == nt - 1:
            print(f'[{run}] {i+1}/{nt} it={it} day={days[i]:.2f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    def meanmap(dct):
        return {tag: dct[tag] / cnt[tag] for tag in dct}

    out = su.cache_path(f'stab_ts_{run}.npz')
    np.savez(
        out,
        iters=iters, days=days,
        depth=depth, ksurf=ksurf,
        wprof_dom=wprof_dom, wprof_box=wprof_box,
        utrans_box=utrans_box, vtrans_box=vtrans_box,
        utrans_dom=utrans_dom, vtrans_dom=vtrans_dom,
        cflw_dom_max=cflw_dom_max, cflw_box_mean=cflw_box_mean,
        wabs_dom_max=wabs_dom_max, wstd_dom=wstd_dom,
        eta_dom=eta_dom, eta_box=eta_box,
        cfl_colmax_mean=cfl_sum / nt, cfl_colmax_max=cfl_max,
        wsurf_full=meanmap(wsurf_sum)['full'],
        wsurf_w14=meanmap(wsurf_sum)['w14'],
        wsurf_w30=meanmap(wsurf_sum)['w30'],
        w100_full=meanmap(w100_sum)['full'],
        w100_w14=meanmap(w100_sum)['w14'],
        w100_w30=meanmap(w100_sum)['w30'],
        cnt_full=cnt['full'], cnt_w14=cnt['w14'], cnt_w30=cnt['w30'],
    )
    print(f'[{run}] wrote {out}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
