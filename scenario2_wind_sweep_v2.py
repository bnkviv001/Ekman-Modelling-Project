"""
scenario2_wind_sweep_v2.py

Wind stress sweep, v2: adds a hodograph panel (middle) so the actual
spiral shape is visible at every wind stress value, not just the
foreshortened 3-D view. Because this is a linear model, the hodograph
shape should look IDENTICAL at every tau (same curl, same rotation
angle at each depth) -- only its overall size changes. That invariance
is itself a nice thing to point out on the slide.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from ekman_model import run_ekman
from ekman_viz import add_3d_arrow, plot_hodograph, style_hodograph_axis

# ============================================================
# MODEL PARAMETERS
# ============================================================
lat = -45.
A = 5.e-2
tau_values = np.linspace(0.02, 0.2, 15)

results = [run_ekman(lat=lat, A=A, Tauwx=0., Tauwy=tau) for tau in tau_values]
z = results[0]['z']
nz = results[0]['nz']
Ekdepth = results[0]['Ekdepth']
surf_speed = [np.hypot(r['uavg'][0], r['vavg'][0]) for r in results]

lim = 1.20 * max(max(np.max(np.abs(r['uavg'])), np.max(np.abs(r['vavg'])))
                  for r in results)

# ============================================================
# FIGURE SETUP -- three panels: 3-D spiral, hodograph, response curve
# ============================================================
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1], wspace=0.32)
ax3 = fig.add_subplot(gs[0], projection='3d')
axh = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

fig.subplots_adjust(left=0.045, right=0.97, bottom=0.11, top=0.80)

fig.suptitle('Effect of Wind Stress on the Ekman Spiral',
             fontsize=16, fontweight='bold', y=0.97)
fig.text(0.5, 0.90,
         'Southern Hemisphere (45°S): increasing wind stress strengthens '
         'the current but does not change the Ekman depth or spiral shape',
         ha='center', fontsize=10.5)

# ---- right panel: surface speed vs wind stress (unchanged) ----
ax2.plot(tau_values, surf_speed, marker='o', markersize=5, linewidth=1.8,
         color='tab:blue')
ax2.set_xlabel(r'Wind stress, $\tau_y$ [N m$^{-2}$]', fontsize=10, labelpad=8)
ax2.set_ylabel(r'Surface current speed [m s$^{-1}$]', fontsize=10, labelpad=8)
ax2.set_title('Surface Current Response', fontsize=12, fontweight='bold', pad=12)
ax2.grid(True, alpha=0.35)
ax2.tick_params(axis='both', labelsize=9)
ax2.margins(x=0.06, y=0.08)
ax2.text(0.04, 0.96, f'Ekman depth = {Ekdepth:.1f} m\n'
         r'Constant because $A$ and $f$ are fixed',
         transform=ax2.transAxes, va='top', ha='left', fontsize=9.5,
         bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                    edgecolor='0.7', alpha=0.9))
marker, = ax2.plot([], [], 'o', color='tab:red', markersize=11, zorder=5)
readout = ax2.text(0.97, 0.05, '', transform=ax2.transAxes,
                    ha='right', va='bottom', fontsize=9.5)


def draw_static_3d(ax):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-100, 0)
    ax.set_xlabel(r'Eastward $u$ [m s$^{-1}$]', labelpad=8, fontsize=9)
    ax.set_ylabel(r'Northward $v$ [m s$^{-1}$]', labelpad=8, fontsize=9)
    ax.set_zlabel('Depth [m]', labelpad=6, fontsize=9)
    ax.tick_params(axis='both', labelsize=7.5)
    ax.view_init(elev=30, azim=-58)


def update(frame):
    r = results[frame]

    # -- 3D panel --
    ax3.cla()
    draw_static_3d(ax3)
    for i in range(nz):
        add_3d_arrow(ax3, z[i], r['uavg'][i], r['vavg'][i], color='tab:blue')
    ax3.set_title('3-D Ekman Profile', fontsize=11.5, fontweight='bold', pad=10)
    ax3.text2D(0.03, 0.93, fr'$\tau_y$ = {tau_values[frame]:.3f} N m$^{{-2}}$',
               transform=ax3.transAxes, fontsize=9.5,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.7', alpha=0.9))

    # -- hodograph panel --
    axh.cla()
    style_hodograph_axis(axh, lim, title='Hodograph (spiral shape)')
    plot_hodograph(axh, r['uavg'], r['vavg'], color='tab:blue')
    axh.text(0.03, 0.96, 'Shape is fixed --\nonly size changes',
             transform=axh.transAxes, fontsize=8.5, va='top',
             style='italic', color='0.4')

    # -- response panel marker --
    marker.set_data([tau_values[frame]], [surf_speed[frame]])
    readout.set_text(fr'$\tau_y$ = {tau_values[frame]:.3f} N m$^{{-2}}$'
                      '\n' fr'Surface speed = {surf_speed[frame]:.3f} m s$^{{-1}}$')

    return [marker, readout]


anim = FuncAnimation(fig, update, frames=len(tau_values), interval=500, blit=False)
anim.save('wind_speed_sweep_v2.gif', writer=PillowWriter(fps=2))
plt.close(fig)
print('Saved wind_speed_sweep_v2.gif')
