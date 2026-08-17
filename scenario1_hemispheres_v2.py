# -*- coding: utf-8 -*-
"""
scenario1_hemispheres_v2.py

Northern vs Southern Hemisphere Ekman spiral comparison.

v2 changes from the original:
  - Added a hodograph panel (u vs v, top-down) alongside the 3-D panel.
    This is the fix for the "looks like a flat line" issue: a single
    fixed 3-D camera angle unavoidably foreshortens part of a spiral
    that turns through 180+ degrees, since vectors pointing toward/away
    from the camera collapse toward the depth axis. The hodograph has
    no such blind spot -- it makes the clockwise (N) vs counter-clockwise
    (S) rotation direction immediately visible.
  - Slightly raised the 3-D elevation angle (23 -> 30) for a marginally
    clearer 3-D view; this is a secondary improvement, the hodograph is
    the real fix.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from ekman_model import run_ekman
from ekman_viz import add_3d_arrow, plot_hodograph, style_hodograph_axis

# ============================================================
# MODEL PARAMETERS
# ============================================================
A = 5.e-2
Tauwy = 0.1
Tauwx = 0.

north = run_ekman(lat=45., A=A, Tauwx=Tauwx, Tauwy=Tauwy)
south = run_ekman(lat=-45., A=A, Tauwx=Tauwx, Tauwy=Tauwy)
z = north['z']
nz = north['nz']

lim = 1.25 * max(np.max(np.abs(north['uavg'])), np.max(np.abs(north['vavg'])),
                  np.max(np.abs(south['uavg'])), np.max(np.abs(south['vavg'])))

# ============================================================
# FIGURE SETUP -- two panels: 3-D spiral, hodograph
# ============================================================
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.28)
ax3 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[1])

fig.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.84)

fig.suptitle('Ekman Spiral Response in the Northern and Southern Hemispheres',
             fontsize=15, fontweight='bold', y=0.97)
fig.text(0.5, 0.905,
         r'Same wind stress ($\tau_y$=0.1 N/m$^2$), lat = $\pm$45$^\circ$ -- '
         'Coriolis sign flip reverses the rotation sense',
         ha='center', fontsize=10.5)

legend_handles = [
    Line2D([0], [0], color='tab:blue', lw=2, label='Northern Hemisphere (+45°, f>0)'),
    Line2D([0], [0], color='tab:red', lw=2, label='Southern Hemisphere (-45°, f<0)'),
    Line2D([0], [0], color='black', lw=2, label='Wind stress direction'),
]

# ---- static hodograph background (full curves, low alpha) ----
style_hodograph_axis(ax2, lim, title='Hodograph -- rotation direction')
ax2.annotate('', xy=(0, 0.7 * lim), xytext=(0, 0),
             arrowprops=dict(arrowstyle='-|>', color='black', lw=2))
ax2.text(0.02 * lim, 0.72 * lim, 'Wind stress', fontsize=9.5, fontweight='bold')


def draw_static_3d(ax):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-100, 0)
    ax.set_xlabel('Eastward velocity, $u$ [m s$^{-1}$]', labelpad=12)
    ax.set_ylabel('Northward velocity, $v$ [m s$^{-1}$]', labelpad=12)
    ax.set_zlabel('Depth [m]', labelpad=10)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='z', pad=4)

    wind_length = 0.70 * lim
    add_3d_arrow(ax, 0, 0, wind_length, color='black', linewidth=2.4, head_size=16)
    ax.text(0.001, wind_length * 1.08, 0, 'Wind stress',
            color='black', fontsize=10, fontweight='bold')

    ax.view_init(elev=30, azim=-58)
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(-0.08, 1.00), frameon=True, fontsize=9,
              title='Hemisphere / forcing')


def update(frame):
    k = frame + 1

    # -- 3D panel --
    ax3.cla()
    draw_static_3d(ax3)
    for i in range(k):
        add_3d_arrow(ax3, z[i], north['uavg'][i], north['vavg'][i], color='tab:blue')
        add_3d_arrow(ax3, z[i], south['uavg'][i], south['vavg'][i], color='tab:red')
    ax3.set_title(f'Current animation depth: {z[frame]:.0f} m', fontsize=11, pad=12)

    # -- hodograph panel --
    ax2.cla()
    style_hodograph_axis(ax2, lim, title='Hodograph -- rotation direction')
    ax2.annotate('', xy=(0, 0.7 * lim), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=2))
    ax2.text(0.02 * lim, 0.72 * lim, 'Wind stress', fontsize=9.5, fontweight='bold')
    plot_hodograph(ax2, north['uavg'], north['vavg'], color='tab:blue',
                   label='N. Hemisphere', up_to=k)
    plot_hodograph(ax2, south['uavg'], south['vavg'], color='tab:red',
                   label='S. Hemisphere', up_to=k)
    ax2.legend(loc='lower right', fontsize=9)

    return []


anim = FuncAnimation(fig, update, frames=nz, interval=200, blit=False)
anim.save('hemisphere_comparison_v2.gif', writer=PillowWriter(fps=5))
plt.close(fig)
print('Saved hemisphere_comparison_v2.gif')
