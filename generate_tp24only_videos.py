"""Generate TPOSE24-only (no TPOSE6 background) versions of all boundary videos."""
import os
os.chdir('/home/edavenport/analysis/tpose24-validation')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import cmocean.cm as cmo
import imageio_ffmpeg
from xmitgcm import open_mdsdataset
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
_writer = animation.FFMpegWriter(fps=6, codec='h264', extra_args=['-pix_fmt', 'yuv420p'])

def make_levels(data_arr, symmetric=False, plo=1, phi=99):
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(data_arr), phi))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1e-10
        return np.linspace(-vmax, vmax, 51)
    fin = data_arr[np.isfinite(data_arr)]
    vmin = float(np.nanpercentile(fin, plo))
    vmax = float(np.nanpercentile(fin, phi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = -1e-10, 1e-10
    return np.linspace(vmin, vmax, 51)

def add_colorbar(fig, ax, levels, cmap, label):
    norm = mcolors.Normalize(vmin=levels[0], vmax=levels[-1])
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return fig.colorbar(sm, ax=ax, label=label, shrink=0.85)

def clear_ax(ax):
    while ax.collections:
        ax.collections[0].remove()

# ── Load TPOSE24 ──────────────────────────────────────────────────────────────
print('Loading TPOSE24...')
run24 = '/data/SO3/edavenport/tpose24/oct2012_TP6Vel_3month'
ds24 = open_mdsdataset(
    data_dir=run24, grid_dir=run24,
    iters=list(range(36, 26173, 36)),
    prefix=['diag_state', 'diag_surf'],
    ref_date='2012-10-01', delta_t=300,
)
for c in ('XC', 'YC', 'Z', 'Zl', 'XG', 'YG'):
    if c in ds24.coords:
        ds24[c] = ds24[c].astype(float)

lon_min24 = float(ds24.XC.min())
lon_max24 = float(ds24.XC.max())
lat_min24 = float(ds24.YC.min())
lat_max24 = float(ds24.YC.max())

ds24 = ds24.sel(time=slice('2012-10-11', None))
ds24_6h = ds24.isel(time=slice(None, None, 2))
t24_times = ds24_6h.time.values
print(f'6-hourly frames: {len(t24_times)}')

xc24 = ds24.XC.values
yc24 = ds24.YC.values
xg24 = ds24.XG.values
yg24 = ds24.YG.values

def setup_ax(ax):
    ax.set_xlim(lon_min24, lon_max24)
    ax.set_ylim(lat_min24, lat_max24)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

# ── SST ───────────────────────────────────────────────────────────────────────
print('\n--- SST ---')
z_sfc = float(ds24.Z.sel(Z=0, method='nearest'))
sst24 = ds24_6h.THETA.sel(Z=z_sfc, method='nearest').compute()
sst_levels = make_levels(sst24.values.ravel(), symmetric=False)

fig, ax = plt.subplots(figsize=(11, 6))
setup_ax(ax)
add_colorbar(fig, ax, sst_levels, cmo.thermal, 'SST (°C)')
title_txt = ax.set_title('')
ax.contourf(xc24, yc24, sst24.isel(time=0).values, levels=sst_levels, cmap=cmo.thermal, extend='both')
plt.tight_layout()

def draw_sst(i):
    clear_ax(ax)
    ax.contourf(xc24, yc24, sst24.isel(time=i).values, levels=sst_levels, cmap=cmo.thermal, extend='both')
    title_txt.set_text(f'SST (TPOSE24 only)  {str(t24_times[i])[:13]}')

anim = animation.FuncAnimation(fig, draw_sst, frames=len(t24_times), interval=150)
print('Rendering...')
anim.save('surface/3month/video_SST_tp24only.mp4', writer=_writer, dpi=100)
plt.close()
print('Saved video_SST_tp24only.mp4')
del sst24

# ── SSH ───────────────────────────────────────────────────────────────────────
print('\n--- SSH ---')
ssh24 = ds24_6h.ETAN.compute()
ssh_levels = make_levels(ssh24.values.ravel(), symmetric=True)

fig, ax = plt.subplots(figsize=(11, 6))
setup_ax(ax)
add_colorbar(fig, ax, ssh_levels, cmo.balance, 'SSH (m)')
title_txt = ax.set_title('')
ax.contourf(xc24, yc24, ssh24.isel(time=0).values, levels=ssh_levels, cmap=cmo.balance, extend='both')
plt.tight_layout()

def draw_ssh(i):
    clear_ax(ax)
    ax.contourf(xc24, yc24, ssh24.isel(time=i).values, levels=ssh_levels, cmap=cmo.balance, extend='both')
    title_txt.set_text(f'SSH (TPOSE24 only)  {str(t24_times[i])[:13]}')

anim = animation.FuncAnimation(fig, draw_ssh, frames=len(t24_times), interval=150)
print('Rendering...')
anim.save('surface/3month/video_SSH_tp24only.mp4', writer=_writer, dpi=100)
plt.close()
print('Saved video_SSH_tp24only.mp4')
del ssh24

# ── Surface velocity ──────────────────────────────────────────────────────────
print('\n--- Surface velocity ---')
zl_sfc = float(ds24.Zl.sel(Zl=0, method='nearest'))
u24 = ds24_6h.UVEL.sel(Z=z_sfc, method='nearest').compute()
v24 = ds24_6h.VVEL.sel(Z=z_sfc, method='nearest').compute()
w24 = ds24_6h.WVEL.sel(Zl=zl_sfc, method='nearest').compute()
ul = make_levels(u24.values.ravel(), symmetric=True)
vl = make_levels(v24.values.ravel(), symmetric=True)
wl = make_levels(w24.values.ravel(), symmetric=True)

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for ax in axes:
    setup_ax(ax)
add_colorbar(fig, axes[0], ul, cmo.balance, f'UVEL (m/s) sfc')
add_colorbar(fig, axes[1], vl, cmo.balance, f'VVEL (m/s) sfc')
add_colorbar(fig, axes[2], wl, cmo.balance, f'WVEL (m/s) sfc')
title_u = axes[0].set_title('', fontsize=16)
title_v = axes[1].set_title('', fontsize=16)
title_w = axes[2].set_title('', fontsize=16)
axes[0].contourf(xg24, yc24, u24.isel(time=0).values, levels=ul, cmap=cmo.balance, extend='both')
axes[1].contourf(xc24, yg24, v24.isel(time=0).values, levels=vl, cmap=cmo.balance, extend='both')
axes[2].contourf(xc24, yc24, w24.isel(time=0).values, levels=wl, cmap=cmo.balance, extend='both')
plt.tight_layout()

def draw_sfc(i):
    clear_ax(axes[0]); clear_ax(axes[1]); clear_ax(axes[2])
    axes[0].contourf(xg24, yc24, u24.isel(time=i).values, levels=ul, cmap=cmo.balance, extend='both')
    axes[1].contourf(xc24, yg24, v24.isel(time=i).values, levels=vl, cmap=cmo.balance, extend='both')
    axes[2].contourf(xc24, yc24, w24.isel(time=i).values, levels=wl, cmap=cmo.balance, extend='both')
    date_str = str(t24_times[i])[:13]
    title_u.set_text(f'UVEL  sfc  {date_str}')
    title_v.set_text(f'VVEL  sfc  {date_str}')
    title_w.set_text(f'WVEL  sfc  {date_str}')

anim = animation.FuncAnimation(fig, draw_sfc, frames=len(t24_times), interval=150)
print('Rendering...')
anim.save('velocity/3month/video_velocity_surface_tp24only.mp4', writer=_writer, dpi=100)
plt.close()
print('Saved video_velocity_surface_tp24only.mp4')
del u24, v24, w24

# ── 70 m velocity ─────────────────────────────────────────────────────────────
print('\n--- 70m velocity ---')
z_70 = float(ds24.Z.sel(Z=-70., method='nearest'))
zl_70 = float(ds24.Zl.sel(Zl=-70., method='nearest'))
u24 = ds24_6h.UVEL.sel(Z=z_70, method='nearest').compute()
v24 = ds24_6h.VVEL.sel(Z=z_70, method='nearest').compute()
w24 = ds24_6h.WVEL.sel(Zl=zl_70, method='nearest').compute()
ul = make_levels(u24.values.ravel(), symmetric=True)
vl = make_levels(v24.values.ravel(), symmetric=True)
wl = make_levels(w24.values.ravel(), symmetric=True)

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for ax in axes:
    setup_ax(ax)
add_colorbar(fig, axes[0], ul, cmo.balance, f'UVEL (m/s) {z_70:.0f}m')
add_colorbar(fig, axes[1], vl, cmo.balance, f'VVEL (m/s) {z_70:.0f}m')
add_colorbar(fig, axes[2], wl, cmo.balance, f'WVEL (m/s) {zl_70:.0f}m')
title_u = axes[0].set_title('', fontsize=16)
title_v = axes[1].set_title('', fontsize=16)
title_w = axes[2].set_title('', fontsize=16)
axes[0].contourf(xg24, yc24, u24.isel(time=0).values, levels=ul, cmap=cmo.balance, extend='both')
axes[1].contourf(xc24, yg24, v24.isel(time=0).values, levels=vl, cmap=cmo.balance, extend='both')
axes[2].contourf(xc24, yc24, w24.isel(time=0).values, levels=wl, cmap=cmo.balance, extend='both')
plt.tight_layout()

def draw_70(i):
    clear_ax(axes[0]); clear_ax(axes[1]); clear_ax(axes[2])
    axes[0].contourf(xg24, yc24, u24.isel(time=i).values, levels=ul, cmap=cmo.balance, extend='both')
    axes[1].contourf(xc24, yg24, v24.isel(time=i).values, levels=vl, cmap=cmo.balance, extend='both')
    axes[2].contourf(xc24, yc24, w24.isel(time=i).values, levels=wl, cmap=cmo.balance, extend='both')
    date_str = str(t24_times[i])[:13]
    title_u.set_text(f'UVEL  Z≈{z_70:.0f}m  {date_str}')
    title_v.set_text(f'VVEL  Z≈{z_70:.0f}m  {date_str}')
    title_w.set_text(f'WVEL  Zl≈{zl_70:.0f}m  {date_str}')

anim = animation.FuncAnimation(fig, draw_70, frames=len(t24_times), interval=150)
print('Rendering...')
anim.save('velocity/3month/video_velocity_70m_tp24only.mp4', writer=_writer, dpi=100)
plt.close()
print('Saved video_velocity_70m_tp24only.mp4')
del u24, v24, w24

# ── 500 m velocity ────────────────────────────────────────────────────────────
print('\n--- 500m velocity ---')
z_500 = float(ds24.Z.sel(Z=-500., method='nearest'))
zl_500 = float(ds24.Zl.sel(Zl=-500., method='nearest'))
u24 = ds24_6h.UVEL.sel(Z=z_500, method='nearest').compute()
v24 = ds24_6h.VVEL.sel(Z=z_500, method='nearest').compute()
w24 = ds24_6h.WVEL.sel(Zl=zl_500, method='nearest').compute()
ul = make_levels(u24.values.ravel(), symmetric=True)
vl = make_levels(v24.values.ravel(), symmetric=True)
wl = make_levels(w24.values.ravel(), symmetric=True)

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for ax in axes:
    setup_ax(ax)
add_colorbar(fig, axes[0], ul, cmo.balance, f'UVEL (m/s) {z_500:.0f}m')
add_colorbar(fig, axes[1], vl, cmo.balance, f'VVEL (m/s) {z_500:.0f}m')
add_colorbar(fig, axes[2], wl, cmo.balance, f'WVEL (m/s) {zl_500:.0f}m')
title_u = axes[0].set_title('', fontsize=16)
title_v = axes[1].set_title('', fontsize=16)
title_w = axes[2].set_title('', fontsize=16)
axes[0].contourf(xg24, yc24, u24.isel(time=0).values, levels=ul, cmap=cmo.balance, extend='both')
axes[1].contourf(xc24, yg24, v24.isel(time=0).values, levels=vl, cmap=cmo.balance, extend='both')
axes[2].contourf(xc24, yc24, w24.isel(time=0).values, levels=wl, cmap=cmo.balance, extend='both')
plt.tight_layout()

def draw_500(i):
    clear_ax(axes[0]); clear_ax(axes[1]); clear_ax(axes[2])
    axes[0].contourf(xg24, yc24, u24.isel(time=i).values, levels=ul, cmap=cmo.balance, extend='both')
    axes[1].contourf(xc24, yg24, v24.isel(time=i).values, levels=vl, cmap=cmo.balance, extend='both')
    axes[2].contourf(xc24, yc24, w24.isel(time=i).values, levels=wl, cmap=cmo.balance, extend='both')
    date_str = str(t24_times[i])[:13]
    title_u.set_text(f'UVEL  Z≈{z_500:.0f}m  {date_str}')
    title_v.set_text(f'VVEL  Z≈{z_500:.0f}m  {date_str}')
    title_w.set_text(f'WVEL  Zl≈{zl_500:.0f}m  {date_str}')

anim = animation.FuncAnimation(fig, draw_500, frames=len(t24_times), interval=150)
print('Rendering...')
anim.save('velocity/3month/video_velocity_500m_tp24only.mp4', writer=_writer, dpi=100)
plt.close()
print('Saved video_velocity_500m_tp24only.mp4')
del u24, v24, w24

print('\nAll tp24only videos done.')
