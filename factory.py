"""Fictional industrial facility: zones, doors, exits, sensors, workers.

Top-down layout (x: west->east 0..100 m, y: south->north 0..60 m)::

    y 40-60  | PU-A      | PU-B      | CHEM      | PROC      |   (production row)
    y 30-40  | COR-N1      | COR-N2      | COR-N3          |   (north corridor)
    y 15-30  | CTRL   | ELEC   | WARE   | REST   |             (support row)
    y  8-15  | COR-S1             | COR-S2             |       (south corridor)
    y  0- 8  | UTIL        | DOCK        | MAINT           |   (service row)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from config import (CELL_SIZE, FACILITY_HEIGHT, FACILITY_WIDTH, GRID_NX,
                    GRID_NY)


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1


@dataclass(frozen=True)
class Exit:
    id: str
    name: str
    zone_id: str
    x: float
    y: float


@dataclass(frozen=True)
class SensorSpec:
    id: str
    zone_id: str
    x: float
    y: float


_ZONES = (
    Zone("PU-A", "Production Unit A", "production", 0, 40, 25, 60),
    Zone("PU-B", "Production Unit B", "production", 25, 40, 50, 60),
    Zone("CHEM", "Chemical Storage", "storage", 50, 40, 75, 60),
    Zone("PROC", "Processing Area", "processing", 75, 40, 100, 60),
    Zone("COR-N1", "North Corridor W", "corridor", 0, 30, 33, 40),
    Zone("COR-N2", "North Corridor C", "corridor", 33, 30, 66, 40),
    Zone("COR-N3", "North Corridor E", "corridor", 66, 30, 100, 40),
    Zone("CTRL", "Control Room", "control", 0, 15, 25, 30),
    Zone("ELEC", "Electrical Room", "electrical", 25, 15, 50, 30),
    Zone("WARE", "Warehouse", "warehouse", 50, 15, 75, 30),
    Zone("REST", "Worker Rest Area", "rest", 75, 15, 100, 30),
    Zone("COR-S1", "South Corridor W", "corridor", 0, 8, 50, 15),
    Zone("COR-S2", "South Corridor E", "corridor", 50, 8, 100, 15),
    Zone("UTIL", "Utility / Boiler Room", "utility", 0, 0, 33, 8),
    Zone("DOCK", "Loading Dock", "dock", 33, 0, 66, 8),
    Zone("MAINT", "Maintenance Workshop", "workshop", 66, 0, 100, 8),
)

_EXITS = (
    Exit("EX1", "West Gate", "COR-N1", 0.0, 35.0),
    Exit("EX2", "East Gate", "COR-N3", 100.0, 35.0),
    Exit("EX3", "South-West Gate", "COR-S1", 0.0, 11.5),
    Exit("EX4", "South-East Gate", "COR-S2", 100.0, 11.5),
    Exit("EX5", "Loading Dock Exit", "DOCK", 50.0, 0.0),
    Exit("EX6", "North-East Exit", "PROC", 87.5, 60.0),
    Exit("EX7", "North-West Exit", "PU-A", 12.5, 60.0),
)

# doors between adjacent zones (the door position is computed from geometry)
_DOORS = (
    ("PU-A", "COR-N1"), ("PU-B", "COR-N1"), ("PU-B", "COR-N2"),
    ("CHEM", "COR-N2"), ("CHEM", "COR-N3"), ("PROC", "COR-N3"),
    ("PU-A", "PU-B"), ("PU-B", "CHEM"), ("CHEM", "PROC"),
    ("COR-N1", "COR-N2"), ("COR-N2", "COR-N3"),
    ("CTRL", "COR-N1"), ("CTRL", "COR-S1"),
    ("ELEC", "COR-N1"), ("ELEC", "COR-N2"), ("ELEC", "COR-S1"),
    ("WARE", "COR-N2"), ("WARE", "COR-N3"), ("WARE", "COR-S2"),
    ("REST", "COR-N3"), ("REST", "COR-S2"),
    ("CTRL", "ELEC"), ("ELEC", "WARE"), ("WARE", "REST"),
    ("COR-S1", "COR-S2"),
    ("UTIL", "COR-S1"), ("DOCK", "COR-S1"), ("DOCK", "COR-S2"),
    ("MAINT", "COR-S2"),
)

# simulated workers on shift at the moment the scenario starts
DEFAULT_OCCUPANCY = {
    "PU-A": 5, "PU-B": 4, "CHEM": 2, "PROC": 4, "COR-N1": 1, "COR-N3": 1,
    "CTRL": 3, "ELEC": 1, "WARE": 3, "REST": 4, "UTIL": 1, "DOCK": 2,
    "MAINT": 2,
}

_EPS = 1e-9


def shared_door(a: Zone, b: Zone) -> tuple[float, float]:
    """Midpoint of the wall segment shared by two adjacent zones."""
    if abs(a.x1 - b.x0) < _EPS or abs(b.x1 - a.x0) < _EPS:
        x = a.x1 if abs(a.x1 - b.x0) < _EPS else a.x0
        lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
        if hi - lo > _EPS:
            return (x, (lo + hi) / 2.0)
    if abs(a.y1 - b.y0) < _EPS or abs(b.y1 - a.y0) < _EPS:
        y = a.y1 if abs(a.y1 - b.y0) < _EPS else a.y0
        lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
        if hi - lo > _EPS:
            return ((lo + hi) / 2.0, y)
    raise ValueError(f"zones {a.id} and {b.id} do not share a wall")


def dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class Facility:
    """Immutable description of the plant plus derived graph and grid masks."""

    def __init__(self) -> None:
        self.width = FACILITY_WIDTH
        self.height = FACILITY_HEIGHT
        self.zones: dict[str, Zone] = {z.id: z for z in _ZONES}
        self.exits: dict[str, Exit] = {e.id: e for e in _EXITS}
        self.sensors: list[SensorSpec] = [
            SensorSpec(f"S{i + 1:02d}", z.id, *z.center)
            for i, z in enumerate(_ZONES)
        ]
        self.sensor_by_id = {s.id: s for s in self.sensors}
        self.sensor_by_zone = {s.zone_id: s for s in self.sensors}
        self.graph: nx.Graph = self._build_graph()
        self.grid_x = (np.arange(GRID_NX) + 0.5) * CELL_SIZE
        self.grid_y = (np.arange(GRID_NY) + 0.5) * CELL_SIZE
        gx, gy = np.meshgrid(self.grid_x, self.grid_y)
        self.grid_xx, self.grid_yy = gx, gy
        self.zone_masks: dict[str, np.ndarray] = {
            z.id: (gx >= z.x0) & (gx < z.x1) & (gy >= z.y0) & (gy < z.y1)
            for z in _ZONES
        }

    # ------------------------------------------------------------------
    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        for z in self.zones.values():
            g.add_node(z.id, pos=z.center, kind="zone")
        for e in self.exits.values():
            g.add_node(e.id, pos=(e.x, e.y), kind="exit")
        for a_id, b_id in _DOORS:
            a, b = self.zones[a_id], self.zones[b_id]
            door = shared_door(a, b)
            length = dist(a.center, door) + dist(door, b.center)
            g.add_edge(a_id, b_id, length=length, door=door)
        for e in self.exits.values():
            z = self.zones[e.zone_id]
            g.add_edge(e.id, z.id, length=dist(z.center, (e.x, e.y)),
                       door=(e.x, e.y))
        return g

    # ------------------------------------------------------------------
    def zone_at(self, x: float, y: float) -> str:
        """Id of the zone containing a point (points are clamped inside)."""
        x = min(max(x, 0.0), self.width - 1e-6)
        y = min(max(y, 0.0), self.height - 1e-6)
        for z in self.zones.values():
            if z.contains(x, y):
                return z.id
        raise ValueError(f"point ({x}, {y}) is outside every zone")

    def neighbours(self, zone_id: str) -> list[str]:
        return [n for n in self.graph.neighbors(zone_id) if n in self.zones]

    def zone_ids(self) -> list[str]:
        return list(self.zones.keys())

    def exit_ids(self) -> list[str]:
        return list(self.exits.keys())

    def source_point(self, zone_id: str, seed: int) -> tuple[float, float]:
        """True leak position: zone centre plus a small seeded offset."""
        z = self.zones[zone_id]
        rng = np.random.default_rng(int(seed) + 7919)
        ox = float(rng.uniform(-0.18, 0.18)) * z.width
        oy = float(rng.uniform(-0.18, 0.18)) * z.height
        cx, cy = z.center
        return (cx + ox, cy + oy)

    def place_workers(self, seed: int,
                      occupancy: dict[str, int] | None = None) -> list[dict]:
        """Deterministically scatter workers inside their zones."""
        occ = DEFAULT_OCCUPANCY if occupancy is None else occupancy
        rng = np.random.default_rng(int(seed) + 104729)
        workers: list[dict] = []
        n = 0
        for zid in self.zones:          # fixed zone order -> reproducible
            z = self.zones[zid]
            for _ in range(int(occ.get(zid, 0))):
                n += 1
                mx = min(2.0, z.width / 4.0)
                my = min(2.0, z.height / 4.0)
                x = float(rng.uniform(z.x0 + mx, z.x1 - mx))
                y = float(rng.uniform(z.y0 + my, z.y1 - my))
                workers.append({"id": f"W{n:02d}", "x": x, "y": y})
        return workers


_FACILITY_SINGLETON: Facility | None = None


def get_facility() -> Facility:
    """Shared facility instance (it is immutable, so sharing is safe)."""
    global _FACILITY_SINGLETON
    if _FACILITY_SINGLETON is None:
        _FACILITY_SINGLETON = Facility()
    return _FACILITY_SINGLETON