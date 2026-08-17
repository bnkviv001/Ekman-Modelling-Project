"""
ekman_viz.py

Shared plotting helpers for the Ekman scenario animations.

Adds a 2-D hodograph helper alongside the existing 3-D arrow helper.
The hodograph (u vs v, traced across depth) is the standard way to show
a full Ekman spiral unambiguously -- a single fixed 3-D viewing angle
necessarily foreshortens part of any spiral that turns through more
than ~90-120 degrees, since vectors pointing toward/away from the
camera collapse toward the depth axis. The hodograph has no such
blind spot because it's a true plan-view projection.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import art3d


def add_3d_arrow(ax, z_level, u, v, color='tab:blue',
                  linewidth=1.5, head_size=9):
    """Draw a single horizontal velocity arrow at one depth level."""
    arrow = FancyArrowPatch(
        (0, 0), (u, v),
        arrowstyle='-|>', mutation_scale=head_size,
        linewidth=linewidth, color=color, shrinkA=0, shrinkB=0
    )
    ax.add_patch(arrow)
    art3d.pathpatch_2d_to_3d(arrow, z=z_level, zdir='z')


def plot_hodograph(ax, u, v, color='tab:blue', label=None,
                    up_to=None, surface_marker=True):
    """
    Plot a hodograph trace (u vs v) with fading opacity so the surface
    (start) end of the curve is visually distinct from the deep end.

    up_to: only plot the first `up_to` points (for progressive reveal
           animations); None means plot the full array.
    """
    n = len(u) if up_to is None else up_to
    if n < 1:
        return
    u_k, v_k = u[:n], v[:n]

    ax.plot(u_k, v_k, '-', color=color, linewidth=1.6, alpha=0.85,
             label=label, zorder=3)
    ax.scatter(u_k, v_k, c=color, s=14, zorder=4,
               alpha=np.linspace(1.0, 0.35, n))

    if surface_marker and n >= 1:
        ax.scatter([u_k[0]], [v_k[0]], c='black', s=45,
                   zorder=5, marker='o')


def style_hodograph_axis(ax, lim, title='Hodograph (top-down view)'):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.75', linewidth=0.8, zorder=1)
    ax.axvline(0, color='0.75', linewidth=0.8, zorder=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'Eastward velocity, $u$ [m s$^{-1}$]', fontsize=9.5)
    ax.set_ylabel(r'Northward velocity, $v$ [m s$^{-1}$]', fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
    ax.tick_params(labelsize=8)
