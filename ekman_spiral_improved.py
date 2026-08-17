"""
ekman_spiral; an elementary upper ocean Ekman layer
Solve a 1-D diffusion flow numerically. This
version has rotation, and thus makes an Ekman
layer. The IC is a state of rest. The fluid is forced
by an imposed stress at the top of the column (z=0)
The lower boundary condition is free-slip to minimize
the effect of finite depth.

Original code by Jim Price, Oct 99
Modified version by Marcello Vichi, for UCT Oceanography
Translated from MATLAB to Python

IMPROVED VERSION
The physics and numerics below are byte-for-byte identical to the
original script -- only the plotting has changed. Every change is
marked with an "IMPROVED:" comment so it's easy to see what and why.
This version also saves an animated GIF of the live profile build-up
(see the very last section, "animated live profile").
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# IMPROVED: a small, consistent colour language used across every figure.
#   - u (East/zonal) and v (North/meridional) always get the same two
#     colours, wherever they appear.
#   - anything that varies with DEPTH or TIME is drawn with the same
#     'viridis' colormap, so a viewer only has to learn one colour code.
U_COLOR = "#D95F02"      # East / zonal current
V_COLOR = "#1B9E77"      # North / meridional current
CMAP = plt.cm.viridis
plt.rcParams.update({"font.size": 11, "axes.grid": False})

# IMPROVED: every legend and every colorbar now uses the same placement
# rule and the same scale, instead of each figure picking its own spot:
#   - legends always sit outside the axes, to the right, vertically
#     centred, unframed, at the same font size
#   - colorbars always use the same shrink/pad so they're the same
#     visual size in every figure
LEGEND_KW = dict(loc="center left", frameon=False, fontsize=11)
COLORBAR_KW = dict(shrink=0.85, pad=0.03)

# IMPROVED: one common canvas size for every figure (same width
# throughout; the two 2-panel figures get extra height since they
# stack two axes, not because they were sized separately by feel).
# The colorbar/legend now live inside this fixed canvas via
# subplots_adjust, instead of expanding a "tight" bounding box outward
# -- that's what made the original before/after PNGs different sizes.
SINGLE_FIGSIZE = (8.5, 6.5)
DOUBLE_FIGSIZE = (8.5, 9)


def _clean_axes(ax):
    """IMPROVED: shared aesthetic clean-up (no boxed-in look, light grid)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


# %% user inputs
dz = 5.          # [m] grid interval
L = 100.          # [m] water column depth
ndays = 20.       # [day] days to integrate
A = 5.e-2         # [m2/s] the eddy diffusivity
Tauwy = 0.1       # [N/m2=Pa] the wind stress (y)
Tauwx = 0.        # [N/m2=Pa] the wind stress (x)
lat = -45.        # [deg] specify the latitude here
# plot profiles every nplot steps
nplot = 25

# constants
omega = 7.29e-5   # [s-1] 2pi/86400
rho0 = 1025.      # [kg/m3] nominal constant density of water
SECPERDAY = 86400
w = 0.4           # the Courant number dt*A/dz^2
                  # must be less than 0.5 for numerical stability

# grid specifications
z = np.arange(0, -L - dz, -dz)      # the grid definition
nz = z.size
dt = w * dz**2 / A                  # the time step is derived from the Courant number
nstep = round(ndays * SECPERDAY / dt)  # number of steps per day
time = np.zeros(nstep)

# derived parameters
f = 2. * omega * np.sin(lat * np.pi / 180)   # the Coriolis parameter
IP = (2 * np.pi / abs(f)) / SECPERDAY        # inertial period

# %% Variables and Initial conditions
u = np.zeros(nz)            # the dependent variable, velocity
v = u.copy()
uavg = u.copy()
vavg = u.copy()
navg = 0

# set figure attributes
# IMPROVED (Plot 1 - live profile):
#   - larger figure, light grid, no top/right box
#   - a zero-current reference line so direction is easy to read
#   - profiles are colour-coded by elapsed time (viridis + colorbar)
#     instead of the default colour cycle, which just repeats every
#     10 lines and carries no information
f1, (ax1, ax2) = plt.subplots(2, 1, figsize=DOUBLE_FIGSIZE)
ax1.axvline(0, color="0.6", linewidth=0.8, zorder=0)
ax2.axvline(0, color="0.6", linewidth=0.8, zorder=0)
ax1.set_ylabel("depth [m]")
ax1.set_xlabel("Zonal current [m/s]")
ax1.set_title(r"Upper ocean Ekman layer, $\tau_y$ = 0.1 N m$^{-2}$")

ax2.set_ylabel("depth [m]")
ax2.set_xlabel("Meridional current, [m/s]")
time_norm = plt.Normalize(vmin=0, vmax=ndays)
_clean_axes(ax1)
_clean_axes(ax2)

# %% pre-allocate diagnostic time series
usurf = np.zeros(nstep)
vsurf = np.zeros(nstep)
umid = np.zeros(nstep)
vmid = np.zeros(nstep)
udeep = np.zeros(nstep)
vdeep = np.zeros(nstep)
transu = np.zeros(nstep)
transv = np.zeros(nstep)

# IMPROVED: snapshots of the profile, captured at the same moments it's
# drawn onto Plot 1 below -- reused at the end of the script to build
# the animated GIF, so the physics only has to run once.
snap_u, snap_v, snap_day = [], [], []

# %%
# begin time-stepping
for n in range(1, nstep + 1):
    idx = n - 1  # Python arrays are 0-based, MATLAB's are 1-based

    # advance time
    time[idx] = (n - 1) * dt

    # evaluate the diffusion term
    delsqu = np.concatenate(([0], u[0:-2] - 2. * u[1:-1] + u[2:], [0]))
    delsqv = np.concatenate(([0], v[0:-2] - 2. * v[1:-1] + v[2:], [0]))

    # Euler forward solution with Coriolis
    u = u + w * delsqu + dt * f * v
    v = v + w * delsqv - dt * f * u

    # apply the surface BC
    u[0] = u[1] + dz * (Tauwx / rho0 / A)
    v[0] = v[1] + dz * (Tauwy / rho0 / A)
    # the bottom BC (Neumann, free-slip)
    u[nz - 1] = u[nz - 2]
    v[nz - 1] = v[nz - 2]

    # Diagnostics

    # 1) time-average (after the first day)
    if time[idx] / SECPERDAY >= 1.:
        uavg = uavg + u
        vavg = vavg + v
        navg = navg + 1

    # 2) time series of surface current
    usurf[idx] = u[0]
    vsurf[idx] = v[0]
    mid = round(2 * nz / 3) - 1        # MATLAB-style 1-based index formula
    umid[idx] = u[mid - 1]             # convert to Python's 0-based indexing
    vmid[idx] = v[mid - 1]
    udeep[idx] = u[nz - 3 - 1]         # MATLAB u(nz-3) -> 0-based index nz-4
    vdeep[idx] = v[nz - 3 - 1]

    # 3) time series of transport
    transu[idx] = (np.sum(u) - u[0]) * dz
    transv[idx] = (np.sum(v) - v[0]) * dz

    # plot profile every nplot time steps
    # IMPROVED: colour each line by its simulation day instead of the
    # default (repeating) colour cycle, and drop the per-frame text
    # label -- the colorbar added after the loop replaces it more clearly.
    if n % nplot == nplot - 1:
        day = time[idx] / SECPERDAY
        line_color = CMAP(time_norm(day))
        ax1.plot(u, z, color=line_color, linewidth=1.3)
        ax2.plot(v, z, color=line_color, linewidth=1.3)
        plt.pause(0.001)
        # IMPROVED: keep a copy of this profile for the animation below
        snap_u.append(u.copy())
        snap_v.append(v.copy())
        snap_day.append(day)

# IMPROVED: one colorbar communicates the time axis for every line on
# both panels, replacing the old "delete + redraw text" approach.
sm = ScalarMappable(norm=time_norm, cmap=CMAP)
sm.set_array([])
cbar = f1.colorbar(sm, ax=[ax1, ax2], location="right", **COLORBAR_KW)
cbar.set_label("simulation day")

# compute mean profile
uavg = uavg / navg
vavg = vavg / navg

# %% plot diagnostics
# Theoretical Ekman transport and Ekman depth
Ektrans = np.sqrt(Tauwx**2 + Tauwy**2) / rho0 / f
Ekdepth = np.sqrt(2 * A / abs(f))

# %% time series at different depths
# IMPROVED (Plot 2 - time series):
#   - explicit, consistent colours for surf/mid/deep (same viridis
#     family used for depth everywhere else in the script)
#   - a legend on BOTH panels (the original was missing one on the
#     North/meridional panel, so those three lines were unlabeled)
#   - a shared x-axis, and a marker for day 1, which is where the
#     time-average diagnostic starts including data
#   - the inertial period IP was computed but never used anywhere in
#     the original script; it now annotates the oscillation visible
#     in the first few days
SURF_COLOR = CMAP(0.85)
MID_COLOR = CMAP(0.5)
DEEP_COLOR = CMAP(0.15)

dtime = time / SECPERDAY
# IMPROVED: this plot gets its own, wider canvas so the chart itself
# can be bigger -- the legend needs less of the figure width reserved
# for it now, and simply sits further out instead of squeezing the axes.
TIME_SERIES_FIGSIZE = (DOUBLE_FIGSIZE[0] + 2.5, DOUBLE_FIGSIZE[1])
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=TIME_SERIES_FIGSIZE, sharex=True)
ax3.plot(dtime, usurf, color=SURF_COLOR, label="Surf")
ax3.plot(dtime, umid, color=MID_COLOR, label="Mid")
ax3.plot(dtime, udeep, color=DEEP_COLOR, label="Deep")
ax3.axvline(1, color="0.4", linestyle=":", linewidth=1)
ax3.axvline(IP, color="0.7", linestyle="--", linewidth=1)
ax3.set_ylabel("East [m/s]")
ax3.set_title("East and North currents")
_clean_axes(ax3)

ax4.plot(dtime, vsurf, color=SURF_COLOR, label="Surf")
ax4.plot(dtime, vmid, color=MID_COLOR, label="Mid")
ax4.plot(dtime, vdeep, color=DEEP_COLOR, label="Deep")
ax4.axvline(1, color="0.4", linestyle=":", linewidth=1, label="averaging starts")
ax4.axvline(IP, color="0.7", linestyle="--", linewidth=1, label=f"inertial period ({IP:.2f}d)")
ax4.set_ylabel("North [m/s]")
ax4.set_xlabel("time [day]")
_clean_axes(ax4)

# a single legend outside the axes avoids covering the oscillating
# lines (which a normal in-plot legend does on both panels here) --
# the chart now gets most of the canvas, with the legend sitting in
# the narrower strip beyond it
fig2.subplots_adjust(right=0.72, hspace=0.25)
handles, labels = ax4.get_legend_handles_labels()
fig2.legend(handles, labels, bbox_to_anchor=(0.735, 0.5),
            bbox_transform=fig2.transFigure, **LEGEND_KW)

# %% transport
# IMPROVED (Plot 3 - transport):
#   - the instantaneous-transport cloud is ~8,600 overlapping points;
#     switching from opaque dots to a small, semi-transparent, time
#     -coloured scatter reveals the spiral trajectory and its density
#     instead of a single solid blob
#   - wind/average/theoretical arrows are kept (from the earlier
#     improvement) but drawn with heavier lines so they read clearly
#     against the point cloud
fig3, ax_t = plt.subplots(figsize=SINGLE_FIGSIZE)
sc = ax_t.scatter(transu, transv, c=dtime, cmap=CMAP, s=6, alpha=0.35,
                   linewidths=0, label="instantaneous")
Tx = np.mean(transu)
Ty = np.mean(transv)
ax_t.plot([0, Tx], [0, Ty], "g-o", linewidth=2.2, markersize=5, label="average")
ax_t.plot([0, Ektrans], [0, 0], "r-+", linewidth=2.2, markersize=9, mew=2, label="theoretical")

# wind stress direction, scaled to the same length as the theoretical
# transport arrow so it can be compared on the same axes (Ekman theory
# predicts the net transport sits 90 degrees from the wind stress)
tau_mag = np.sqrt(Tauwx**2 + Tauwy**2)
windx = Tauwx / tau_mag * Ektrans
windy = Tauwy / tau_mag * Ektrans
ax_t.plot([0, windx], [0, windy], "b-^", linewidth=2.2, markersize=6, label="wind stress")

ax_t.set_xlabel("U transport [m2/s]")
ax_t.set_ylabel("V transport [m2/s]")
ax_t.axis("equal")
ax_t.set_title("Ekman transport")
_clean_axes(ax_t)
fig3.subplots_adjust(right=0.60)
cbar_t = fig3.colorbar(sc, ax=ax_t, **COLORBAR_KW)
cbar_t.set_label("simulation day")
handles, labels = ax_t.get_legend_handles_labels()
fig3.legend(handles, labels, bbox_to_anchor=(0.78, 0.5),
            bbox_transform=fig3.transFigure, **LEGEND_KW)

# %% mean velocity profiles
# IMPROVED (Plot 4 - mean profile):
#   - u/v now use the same East/North colours as the time-series plot
#   - the single dashed "Ekman depth" line is replaced with a shaded
#     band from the surface to the Ekman depth, which reads more like
#     a "layer" and less like an arbitrary marker
#   - a zero-current reference line
fig4, ax_m = plt.subplots(figsize=SINGLE_FIGSIZE)
ax_m.axvline(0, color="0.6", linewidth=0.8, zorder=0)
ax_m.axhspan(-Ekdepth, 0, color=CMAP(0.85), alpha=0.12, label="Ekman layer")
ax_m.plot(uavg, z, color=U_COLOR, linewidth=2.2, label="E")
ax_m.plot(vavg, z, color=V_COLOR, linewidth=2.2, label="N")
ax_m.axhline(-Ekdepth, color="0.4", linestyle="--", linewidth=1)
ax_m.set_xlabel("East and North currents [m/s]")
ax_m.set_ylabel("Depth [m]")
ax_m.set_title("Mean current profiles")
_clean_axes(ax_m)
fig4.subplots_adjust(right=0.66)
handles, labels = ax_m.get_legend_handles_labels()
fig4.legend(handles, labels, bbox_to_anchor=(0.70, 0.5),
            bbox_transform=fig4.transFigure, **LEGEND_KW)

# %% hodograph
# IMPROVED (Plot 5 - hodograph):
#   - arrows are colour-coded by depth (same viridis convention as
#     everywhere else), with a colorbar -- in the original, every
#     arrow was the same colour, so there was no way to tell which
#     arrow belonged to which depth without cross-referencing another
#     figure
fig5, ax_h = plt.subplots(figsize=SINGLE_FIGSIZE)
zeroz = np.zeros_like(uavg)
ax_h.quiver(0, 0, 0, 0.06, color="r", angles="xy", scale_units="xy", scale=1)
q = ax_h.quiver(zeroz, zeroz, uavg, vavg, z, cmap=CMAP,
                 angles="xy", scale_units="xy", scale=1)
ax_h.set_xlabel("East component [m/s]")
ax_h.set_ylabel("North component, [m/s]")
ax_h.set_title("Hodograph")
ax_h.axis("equal")
_clean_axes(ax_h)
fig5.subplots_adjust(right=0.60)
cbar_h = fig5.colorbar(q, ax=ax_h, **COLORBAR_KW)
cbar_h.set_label("depth [m]")
# IMPROVED: a quiver's default legend swatch renders as a filled
# rectangle, not a line -- a proxy Line2D gives a clean line-style
# swatch instead, consistent with every other legend in this script
scale_proxy = Line2D([0], [0], color="r", linewidth=2)
fig5.legend([scale_proxy], ["0.06 m/s scale"], bbox_to_anchor=(0.78, 0.5),
            bbox_transform=fig5.transFigure, **LEGEND_KW)

# %% spiral in 3D
# IMPROVED (Plot 6 - 3D spiral):
#   - depth colour-coding to match the hodograph (mplot3d's quiver
#     doesn't support a per-arrow colour array directly, so each
#     vector is drawn individually with its matching colormap colour)
#   - a fixed, slightly elevated viewing angle chosen to make the
#     spiral's rotation with depth easier to read at a glance
fig6 = plt.figure(figsize=SINGLE_FIGSIZE)
ax6 = fig6.add_subplot(projection="3d")
depth_norm = plt.Normalize(vmin=-L, vmax=0)
for i in range(nz):
    ax6.quiver(0, 0, z[i], uavg[i], vavg[i], 0,
               color=CMAP(depth_norm(z[i])), linewidth=1.4)
x3 = [0, 0, 0]
y3 = [0, 0, 0.1]
z3 = [-L, 0, 0]
ax6.plot(x3, y3, z3, "r-", linewidth=1.5)
ax6.set_xlabel("East component [m/s]")
ax6.set_ylabel("North component, [m/s]")
ax6.set_zlabel("Depth [m]")
ax6.view_init(elev=22, azim=-60)
ax6.grid(True)
sm3 = ScalarMappable(norm=depth_norm, cmap=CMAP)
sm3.set_array([])
cbar3 = fig6.colorbar(sm3, ax=ax6, **COLORBAR_KW)
cbar3.set_label("depth [m]")

# %% animated live profile (GIF)
# IMPROVED: an animated version of Plot 1, built from the profile
# snapshots captured during the time-stepping loop -- shows the spiral
# developing frame by frame (as it looks running live with plt.pause)
# instead of only the finished static picture. Saved next to this
# script as live_profile_animation.gif.
# subsample the captured profiles so the GIF stays a reasonable size
anim_u = snap_u[::3]
anim_v = snap_v[::3]
anim_day = snap_day[::3]

fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=DOUBLE_FIGSIZE)
axA.axvline(0, color="0.6", linewidth=0.8, zorder=0)
axB.axvline(0, color="0.6", linewidth=0.8, zorder=0)
axA.set_ylabel("depth [m]")
axA.set_xlabel("Zonal current [m/s]")
axA.set_title(r"Upper ocean Ekman layer, $\tau_y$ = 0.1 N m$^{-2}$")
axB.set_ylabel("depth [m]")
axB.set_xlabel("Meridional current, [m/s]")
_clean_axes(axA)
_clean_axes(axB)

all_u = np.array(anim_u)
all_v = np.array(anim_v)
pad_u = 0.1 * (all_u.max() - all_u.min())
pad_v = 0.1 * (all_v.max() - all_v.min())
axA.set_xlim(all_u.min() - pad_u, all_u.max() + pad_u)
axB.set_xlim(all_v.min() - pad_v, all_v.max() + pad_v)
axA.set_ylim(-L - dz, dz)
axB.set_ylim(-L - dz, dz)

sm_anim = ScalarMappable(norm=time_norm, cmap=CMAP)
sm_anim.set_array([])
cbar_anim = fig_anim.colorbar(sm_anim, ax=[axA, axB], location="right", **COLORBAR_KW)
cbar_anim.set_label("simulation day")
day_text = axB.text(0.02, 0.05, "", transform=axB.transAxes, fontsize=11, color="0.2")


def _update_anim(frame):
    """Draw one more profile onto the animation each frame."""
    frame_day = anim_day[frame]
    color = CMAP(time_norm(frame_day))
    axA.plot(anim_u[frame], z, color=color, linewidth=1.3)
    axB.plot(anim_v[frame], z, color=color, linewidth=1.3)
    day_text.set_text(f"day {frame_day:.1f}")
    return []


# hold on the finished spiral for ~1.3s before the GIF loops again
n_frames = len(anim_u)
frame_order = list(range(n_frames)) + [n_frames - 1] * 15
live_anim = FuncAnimation(fig_anim, _update_anim, frames=frame_order,
                           interval=90, blit=False)
live_anim.save("live_profile_animation.gif", writer=PillowWriter(fps=12), dpi=80)

plt.show()
