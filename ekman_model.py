"""
ekman_model.py

Reusable Ekman-layer solver, refactored out of ekman_spiral.m / Ekman_Model.ipynb
into a function so it can be run repeatedly for different parameter values
(latitude, wind stress, diffusivity) without copy-pasting the time loop.

The numerics (explicit Euler, centred diffusion, surface stress BC,
free-slip bottom BC) are unchanged from the original code.
"""

import numpy as np

OMEGA = 7.29e-5   # [s-1] Earth's rotation rate
RHO0 = 1025.      # [kg/m3]
SECPERDAY = 86400.


def run_ekman(lat=-45., A=5.e-2, Tauwx=0., Tauwy=0.1,
              dz=5., L=100., ndays=20., courant=0.4,
              nplot=25, spinup_days=1., store_frames=False):
    """
    Integrate the 1-D Ekman diffusion/rotation model to (near-)equilibrium.

    Returns a dict with:
      z        depth grid [m], z[0] = 0 (surface)
      u, v     final instantaneous profiles [m/s]
      uavg,vavg time-averaged profiles after spinup_days [m/s]
      f        Coriolis parameter [s-1]
      IP       inertial period [days]
      Ekdepth  theoretical Ekman depth sqrt(2A/|f|) [m]
      Ektrans  theoretical Ekman transport magnitude [m2/s]
      t_frames, u_frames, v_frames   (only if store_frames=True)
    """
    z = np.arange(0, -L - dz, -dz)
    nz = z.size
    dt = courant * dz**2 / A
    nstep = max(1, round(ndays * SECPERDAY / dt))

    f = 2. * OMEGA * np.sin(lat * np.pi / 180)
    IP = np.inf if f == 0 else (2 * np.pi / abs(f)) / SECPERDAY

    u = np.zeros(nz)
    v = np.zeros(nz)
    uavg = np.zeros(nz)
    vavg = np.zeros(nz)
    navg = 0

    u_frames, v_frames, t_frames = [], [], []

    for n in range(1, nstep + 1):
        t = (n - 1) * dt

        delsqu = np.concatenate(([0], u[:-2] - 2. * u[1:-1] + u[2:], [0]))
        delsqv = np.concatenate(([0], v[:-2] - 2. * v[1:-1] + v[2:], [0]))

        u = u + courant * delsqu + dt * f * v
        v = v + courant * delsqv - dt * f * u

        u[0] = u[1] + dz * (Tauwx / RHO0 / A)
        v[0] = v[1] + dz * (Tauwy / RHO0 / A)
        u[-1] = u[-2]
        v[-1] = v[-2]

        if t / SECPERDAY >= spinup_days:
            uavg += u
            vavg += v
            navg += 1

        if store_frames and (n % nplot == 0):
            u_frames.append(u.copy())
            v_frames.append(v.copy())
            t_frames.append(t)

    if navg > 0:
        uavg /= navg
        vavg /= navg
    else:
        # ndays too short relative to spinup_days: fall back to last state
        uavg, vavg = u.copy(), v.copy()

    Ekdepth = np.inf if f == 0 else np.sqrt(2 * A / abs(f))
    Ektrans = np.inf if f == 0 else np.sqrt(Tauwx**2 + Tauwy**2) / RHO0 / abs(f)

    out = dict(z=z, u=u, v=v, uavg=uavg, vavg=vavg, f=f, IP=IP,
               Ekdepth=Ekdepth, Ektrans=Ektrans, nz=nz, dt=dt, nstep=nstep)
    if store_frames:
        out.update(u_frames=u_frames, v_frames=v_frames, t_frames=t_frames)
    return out
