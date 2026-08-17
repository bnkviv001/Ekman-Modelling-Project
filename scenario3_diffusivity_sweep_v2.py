"""
scenario3_diffusivity_sweep_v2.py

Eddy diffusivity sweep, v2: adds a hodograph panel (middle). Unlike the
wind stress sweep, the hodograph SHAPE genuinely changes here (not just
its size) -- smaller A gives a tightly wound curl completed within a
shallow part of the 100 m domain, larger A stretches the same amount of
turning over a much deeper range, so within the fixed 100 m column the
curve looks straighter / less wound.
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
Tauwy = 0.1
Tauwx = 0.
A_values = np.geomspace(1e-2, 2e-1, 12)

results = [run_ekman(lat=lat, A=A, Tauwx=Tauwx, Tauwy=Tauwy) for A in A_values]
z = results[0]['z']
nz = results[0]['nz']
Ekdepths = [r['Ekdepth'] for r in results]

lim = 1.20 * max(max(np.max(np.abs(r['uavg'])), np.max(np.abs(r['vavg'])))
                  for r in results)

# ============================================================
# FIGURE SETUP -- three panels: 3-D spiral, hodograph, Ekman depth curve
# ============================================================
fig = plt.figure(figsize=(16, 6.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.85, 1], wspace=0.32)
ax3 = fig.add_subplot(gs[0], projection='3d')
axh = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

fig.subplots_adjust(left=0.045, right=0.97, bottom=0.11, top=0.80)

fig.suptitle('Effect of Eddy Diffusivity on the Ekman Spiral',
             fontsize=16, fontweight='bold', y=0.97)
fig.text(0.5, 0.90,
         'Southern Hemisphere (45°S): increasing eddy diffusivity spreads the '
         'same rotation over a deeper column, so the spiral looks less tightly wound',
         ha='center', fontsize=10.5)

# ---- right panel: Ekman depth vs diffusivity (unchanged) ----
ax2.plot(A_values, Ekdepths, marker='o', markersize=5, linewidth=1.8, color='tab:green')
ax2.set_xscale('log')
ax2.set_xlabel(r'Eddy diffusivity, $A$ [m$^2$ s$^{-1}$]', fontsize=10, labelpad=8)
ax2.set_ylabel('Ekman depth [m]', fontsize=10, labelpad=8)
ax2.set_title('Ekman Depth Response', fontsize=12, fontweight='bold', pad=12)
ax2.grid(True, which='both', alpha=0.35)
ax2.tick_params(axis='both', labelsize=9)
ax2.margins(x=0.06, y=0.08)
ax2.text(0.04, 0.96, r'$D_E = \sqrt{\frac{2A}{|f|}}$' '\n'
         r'Latitude and wind stress are fixed',
         transform=ax2.transAxes, va='top', ha='left', fontsize=10,
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
        add_3d_arrow(ax3, z[i], r['uavg'][i], r['vavg'][i], color='tab:purple')
    ax3.set_title('3-D Ekman Profile', fontsize=11.5, fontweight='bold', pad=10)
    ax3.text2D(0.03, 0.93, fr'$A$ = {A_values[frame]:.3f} m$^2$ s$^{{-1}}$'
               '\n' fr'$D_E$ = {Ekdepths[frame]:.1f} m',
               transform=ax3.transAxes, fontsize=9.5,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.7', alpha=0.9))

    # -- hodograph panel --
    axh.cla()
    style_hodograph_axis(axh, lim, title='Hodograph (spiral shape)')
    plot_hodograph(axh, r['uavg'], r['vavg'], color='tab:purple')

    # -- Ekman depth panel marker --
    marker.set_data([A_values[frame]], [Ekdepths[frame]])
    readout.set_text(fr'$A$ = {A_values[frame]:.3f} m$^2$ s$^{{-1}}$'
                      '\n' fr'Ekman depth = {Ekdepths[frame]:.1f} m')

    return [marker, readout]


anim = FuncAnimation(fig, update, frames=len(A_values), interval=600, blit=False)
anim.save('diffusivity_sweep_v2.gif', writer=PillowWriter(fps=2))
plt.close(fig)
print('Saved diffusivity_sweep_v2.gif')
