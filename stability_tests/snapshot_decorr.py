"""Matched-time snapshot divergence of depth-averaged velocity, dt60 vs dt300.

For a set of elapsed days, reads the instantaneous U/V from both runs at the same
simulated time, depth-averages (0-70 m, 0-250 m), and measures the divergence
(RMS of dt60-dt300 and vector pattern correlation). Saves example day-30 fields.

Assumption: matched simulated day d -> dt300 iter d*288, dt60 iter d*1440.
"""

import numpy as np
import stab_utils as su

DAYS = [1, 5, 10, 20, 30, 45, 60, 75, 85]


def read_uv_topk(run, it, kmax):
    d = su.RUNS[run]['dir'] + f'diag_state.{it:010d}.data'
    c = kmax * su.NX * su.NY
    U = np.fromfile(d, dtype=su.DT_DTYPE, count=c,
                    offset=su.REC_UVEL*su.NX*su.NY*su.NZ*4).reshape(kmax, su.NY, su.NX)
    V = np.fromfile(d, dtype=su.DT_DTYPE, count=c,
                    offset=su.REC_VVEL*su.NX*su.NY*su.NZ*4).reshape(kmax, su.NY, su.NX)
    return U, V


def davg(vel, hFac, drf, band):
    w = (hFac*drf[:, None, None])[band]
    return (vel[band]*w).sum(0) / w.sum(0)


def main():
    g = su.load_grid('dt300')
    depth, DRF = g['depth'], g['DRF']
    b250 = depth >= -250
    kmax = int(np.where(b250)[0].max())+1
    b70k, b250k = (depth >= -70)[:kmax], b250[:kmax]
    drf, hW, hS = DRF[:kmax], g['hFacW'][:kmax], g['hFacS'][:kmax]

    def rms(*f):
        return float(np.sqrt(np.nanmean(sum(x**2 for x in f))))

    def corr(u1, v1, u2, v2):
        return float(np.nansum(u1*u2+v1*v2) /
                     np.sqrt(np.nansum(u1**2+v1**2)*np.nansum(u2**2+v2**2)))

    res = {b: {'rms_diff': [], 'corr': [], 'rms_field': []} for b in ('70', '250')}
    saved = {}
    for d in DAYS:
        U3, V3 = read_uv_topk('dt300', d*288, kmax)
        U6, V6 = read_uv_topk('dt60', d*1440, kmax)
        for b, bk in (('70', b70k), ('250', b250k)):
            u3, v3 = davg(U3, hW, drf, bk), davg(V3, hS, drf, bk)
            u6, v6 = davg(U6, hW, drf, bk), davg(V6, hS, drf, bk)
            res[b]['rms_diff'].append(rms(u6-u3, v6-v3))
            res[b]['corr'].append(corr(u3, v3, u6, v6))
            res[b]['rms_field'].append(0.5*(rms(u3, v3)+rms(u6, v6)))
            if d == 30:
                saved[f'u3_{b}'] = u3; saved[f'v3_{b}'] = v3
                saved[f'u6_{b}'] = u6; saved[f'v6_{b}'] = v6
        print(f'day {d:2d}: 0-70 rms_diff={res["70"]["rms_diff"][-1]:.3f} '
              f'corr={res["70"]["corr"][-1]:.3f} | 0-250 rms_diff={res["250"]["rms_diff"][-1]:.3f} '
              f'corr={res["250"]["corr"][-1]:.3f}', flush=True)

    out = {'days': np.array(DAYS)}
    for b in ('70', '250'):
        for k in res[b]:
            out[f'{k}_{b}'] = np.array(res[b][k])
    out.update(saved)
    np.savez(su.cache_path('stab_snapdecorr.npz'), **out)
    print('wrote stab_snapdecorr.npz')


if __name__ == '__main__':
    main()
