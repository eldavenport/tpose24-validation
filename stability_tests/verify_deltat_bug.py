import sys
import numpy as np
sys.path.insert(0, '/home/edavenport/analysis/tpose24-osse')
from osse_tools import load_model

D300 = '/data/SO3/edavenport/tpose24/oct2012_3mo_dt300_AB3/'
D60 = '/data/SO3/edavenport/tpose24/oct2012_1mo_dt60_AB3/'
SL = slice('2012-10-08', '2012-11-01')
MAXD = 250
# daily subsample for a quick field check
I300 = list(range(288, 26173, 288))
I60 = list(range(1440, 43200, 1440))

bug = load_model(D60, list(range(180, 43200, 180)))            # default delta_t=300
ok = load_model(D60, list(range(180, 43200, 180)), delta_t=60)  # correct
print('=== time-axis labelling of the dt60 run ===')
for name, ds in (('dt60 delta_t=300 (notebook default)', bug),
                 ('dt60 delta_t=60  (correct)', ok)):
    t = ds.time.values
    ts = ds.sel(time=SL).time.values
    print(f'{name}:\n  full axis {str(t[0])[:16]} .. {str(t[-1])[:16]}'
          f'\n  Oct08-Nov01 slice: n={len(ts)}  {str(ts[0])[:16]} .. {str(ts[-1])[:16]}')
tb = bug.sel(time=SL).time.values
day_lab = (tb - np.datetime64('2012-10-01')) / np.timedelta64(1, 'D')
print(f'  -> buggy slice REAL simulated days = labelled/5 = '
      f'{day_lab[0]/5:.1f} .. {day_lab[-1]/5:.1f}\n')


def tdmean(ds):
    return ds.UVEL.where(ds.Z >= -MAXD).sel(time=SL).mean(['time', 'Z']).compute()


print('=== time/depth-mean U (0-250 m), daily subsample ===')
u300 = tdmean(load_model(D300, I300))
u60_ok = tdmean(load_model(D60, I60, delta_t=60))
u60_bug = tdmean(load_model(D60, I60))


def rms(a, b):
    d = (a - b).values
    return float(np.sqrt(np.nanmean(d**2)))


print(f'RMS(dt60 correct  - dt300) = {rms(u60_ok, u300):.4f} m/s')
print(f'RMS(dt60 buggy dt - dt300) = {rms(u60_bug, u300):.4f} m/s')
print(f'field RMS dt300            = {float(np.sqrt(np.nanmean(u300.values**2))):.4f} m/s')
