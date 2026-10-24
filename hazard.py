"""Simplified 2-D gas / smoke dispersion (advected Gaussian puffs).

Model
-----
The source emits one puff per tick.  Every puff is a 2-D Gaussian blob that

* drifts with the wind (displacement = integral of the wind vector),
* widens by turbulent diffusion (sigma^2 = sigma0^2 + 2 * integral K dt, where
  K grows with wind speed),
* loses strength by ventilation / dilution (exp(-age / tau)).

The hazard index at a point is the sum of all puffs, multiplied by
``HAZARD_SCALE``.  Because the wind history is stored as cumulative sums, a
changing wind (wind shift during the incident) is handled exactly, and the
cost of evaluating the field is O(puffs x points) with plain NumPy.

This is a teaching/demo model.  It ignores walls, buoyancy, terrain and
chemistry, and is NOT suitable for real industrial safety decisions.
"""
from __future__ import annotations

import math

import numpy as np

from config import (CELL_SIZE, DT, GRID_NX, GRID_NY, HAZARD_SCALE, K_BASE,
                    K_WIND, LEVEL_RANK, SIGMA0, VENT_TAU, level_from_h)

_COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass_label(deg: float) -> str:
    return _COMPASS[int((float(deg) % 360.0) / 22.5 + 0.5) % 16]


def wind_vector(wind_from_deg: float, speed: float) -> tuple[float, float]:
    """Velocity (east, north) of air for a wind blowing FROM ``wind_from_deg``.

    Meteorological convention: 0 = from north, 90 = from east.
    """
    r = math.radians(float(wind_from_deg))
    s = max(0.0, float(speed))
    return (-math.sin(r) * s, -math.cos(r) * s)


def diffusivity(speed: float, factor: float = 1.0) -> float:
    return K_BASE * max(0.0, float(factor)) + K_WIND * max(0.0, float(speed))


class PlumeModel:
    """Stateful puff plume.  Call :meth:`step` once per simulation tick."""

    def __init__(self, source_xy: tuple[float, float], dt: float = DT,
                 sigma0: float = SIGMA0, vent_tau: float = VENT_TAU,
                 scale: float = HAZARD_SCALE) -> None:
        self.source = (float(source_xy[0]), float(source_xy[1]))
        self.dt = float(dt)
        self.sigma0 = float(sigma0)
        self.vent_tau = float(vent_tau)
        self.scale = float(scale)
        self.emit: list[float] = []      # puff mass emitted at each tick
        self.dx: list[float] = [0.0]     # cumulative wind displacement
        self.dy: list[float] = [0.0]
        self.kc: list[float] = [0.0]     # cumulative diffusivity integral

    @property
    def n(self) -> int:
        return len(self.emit)

    def step(self, q: float, wind_from: float, wind_speed: float,
             diff_factor: float = 1.0) -> None:
        """Advance one tick with emission rate ``q`` (units per second)."""
        q = max(0.0, float(q))
        if not math.isfinite(q):
            q = 0.0
        ux, uy = wind_vector(wind_from, wind_speed)
        self.emit.append(q * self.dt)
        self.dx.append(self.dx[-1] + ux * self.dt)
        self.dy.append(self.dy[-1] + uy * self.dt)
        self.kc.append(self.kc[-1] + diffusivity(wind_speed, diff_factor) * self.dt)

    # ------------------------------------------------------------------
    def hazard(self, xs, ys, *, extra_steps: int = 0,
               wind: tuple[float, float] | None = None,
               q_future: float = 0.0, diff_factor: float = 1.0,
               emission_scale: float = 1.0, emit_from: int = 0) -> np.ndarray:
        """Hazard index at points ``(xs, ys)`` (any matching shape).

        extra_steps / wind / q_future  : forecast ``extra_steps`` ticks ahead
            assuming constant ``wind = (from_deg, speed)`` and emission
            ``q_future`` per second.
        emission_scale : multiplies every emitted mass (used by the fit).
        emit_from      : ignore puffs emitted before this tick index.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        shape = xs.shape
        fx, fy = xs.ravel(), ys.ravel()

        e = np.asarray(self.emit, dtype=float)
        dx = np.asarray(self.dx, dtype=float)
        dy = np.asarray(self.dy, dtype=float)
        kc = np.asarray(self.kc, dtype=float)
        if extra_steps > 0:
            if wind is None:
                raise ValueError("wind=(from_deg, speed) is required to forecast")
            ux, uy = wind_vector(*wind)
            k = diffusivity(wind[1], diff_factor)
            steps = np.arange(1, int(extra_steps) + 1, dtype=float)
            dx = np.concatenate([dx, dx[-1] + ux * self.dt * steps])
            dy = np.concatenate([dy, dy[-1] + uy * self.dt * steps])
            kc = np.concatenate([kc, kc[-1] + k * self.dt * steps])
            e = np.concatenate(
                [e, np.full(int(extra_steps), max(0.0, q_future) * self.dt)])
        n_total = len(e)
        if n_total == 0:
            return np.zeros(shape)
        e = e * float(emission_scale)
        if emit_from > 0:
            e[: min(int(emit_from), n_total)] = 0.0
        active = e > 0.0
        if not active.any():
            return np.zeros(shape)

        idx = np.nonzero(active)[0]
        cx = self.source[0] + (dx[n_total] - dx[idx])
        cy = self.source[1] + (dy[n_total] - dy[idx])
        s2 = self.sigma0 ** 2 + 2.0 * (kc[n_total] - kc[idx])
        age = (n_total - idx) * self.dt
        w = e[idx] * np.exp(-age / self.vent_tau) / (2.0 * math.pi * s2)
        d2 = (fx[None, :] - cx[:, None]) ** 2 + (fy[None, :] - cy[:, None]) ** 2
        conc = (w[:, None] * np.exp(-d2 / (2.0 * s2[:, None]))).sum(axis=0)
        out = np.nan_to_num(self.scale * conc, nan=0.0, posinf=1e6, neginf=0.0)
        return out.reshape(shape)


# --------------------------------------------------------------------------
# Grid / zone helpers
# --------------------------------------------------------------------------
def grid_points(facility) -> tuple[np.ndarray, np.ndarray]:
    return facility.grid_xx.ravel(), facility.grid_yy.ravel()


def field_on_grid(model: PlumeModel, facility, **kwargs) -> np.ndarray:
    """Evaluate the hazard index on the facility grid -> shape (NY, NX)."""
    xs, ys = grid_points(facility)
    return model.hazard(xs, ys, **kwargs).reshape(GRID_NY, GRID_NX)


def zone_hazard(facility, field: np.ndarray) -> dict[str, float]:
    """Worst (maximum) hazard index inside every zone."""
    out: dict[str, float] = {}
    for zid, mask in facility.zone_masks.items():
        out[zid] = float(field[mask].max()) if mask.any() else 0.0
    return out


def zone_levels(zone_h: dict[str, float]) -> dict[str, str]:
    return {z: level_from_h(h) for z, h in zone_h.items()}


def max_level(levels) -> str:
    best = "LOW"
    for lv in levels:
        if LEVEL_RANK[lv] > LEVEL_RANK[best]:
            best = lv
    return best


__all__ = [
    "PlumeModel", "wind_vector", "diffusivity", "compass_label",
    "field_on_grid", "zone_hazard", "zone_levels", "max_level", "grid_points",
    "CELL_SIZE",
]