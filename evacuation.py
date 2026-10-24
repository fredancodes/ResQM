"""Graph based evacuation / rescue routing with A*.

Graph: nodes are zones (at their centre) and emergency exits; edges are doors.
Edge cost = geometric length x average hazard multiplier of its two end
nodes.  Rules (all enforced here, never by the caller):

* a zone at CRITICAL level is never entered,
* a zone above ``MAX_TRANSIT_LEVEL`` (default MODERATE) is never transited,
  it may only be the *starting* zone of an evacuee,
* blocked zones / blocked exits are never used,
* if no exit is reachable the result is ``NO_SAFE_ROUTE`` - the planner never
  falls back to an unsafe route.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx

from config import (LEVEL_COST, LEVEL_RANK, MAX_TRANSIT_LEVEL, WORKER_SPEED)
from factory import Facility, dist

SAFE = "SAFE"
NO_SAFE_ROUTE = "NO_SAFE_ROUTE"
NO_EXIT_AVAILABLE = "NO_EXIT_AVAILABLE"


@dataclass
class RouteResult:
    zone_id: str
    occupants: int = 0
    status: str = NO_SAFE_ROUTE
    reason: str = ""
    exit_id: str | None = None
    path: list = field(default_factory=list)          # node ids incl. exit
    waypoints: list = field(default_factory=list)     # door points to walk
    length_m: float = 0.0
    cost: float = 0.0
    eta_s: float = 0.0
    alternatives: list = field(default_factory=list)  # other exits, sorted
    passes_through: list = field(default_factory=list)
    override: str = ""                                # master override note
    time_critical: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _mult(level: str) -> float:
    return LEVEL_COST.get(level, float("inf"))


class EvacuationPlanner:
    def __init__(self, facility: Facility) -> None:
        self.f = facility
        self.G = facility.graph

    # ------------------------------------------------------------------
    def exit_status(self, levels: dict, blocked_exits=(), blocked_zones=(),
                    max_transit: str = MAX_TRANSIT_LEVEL) -> dict:
        """Availability of every exit: AVAILABLE / BLOCKED / UNSAFE."""
        limit = LEVEL_RANK[max_transit]
        out = {}
        for eid, ex in self.f.exits.items():
            z = ex.zone_id
            if eid in blocked_exits:
                out[eid] = "BLOCKED"
            elif z in blocked_zones or LEVEL_RANK[levels.get(z, "LOW")] > limit:
                out[eid] = "UNSAFE"
            else:
                out[eid] = "AVAILABLE"
        return out

    def _allowed(self, start: str, levels, blocked_zones, blocked_exits,
                 max_transit: str):
        limit = LEVEL_RANK[max_transit]
        nodes = {z for z in self.f.zones
                 if z not in blocked_zones
                 and LEVEL_RANK[levels.get(z, "LOW")] <= limit
                 and levels.get(z, "LOW") != "CRITICAL"}
        if levels.get(start, "LOW") != "CRITICAL":
            nodes.add(start)                      # the start is always left
        exits = {e for e, ex in self.f.exits.items()
                 if e not in blocked_exits and ex.zone_id in nodes}
        return nodes, exits

    def _weighted_graph(self, nodes, exits, cost_levels) -> nx.Graph:
        sub = self.G.subgraph(set(nodes) | set(exits)).copy()
        for a, b, d in sub.edges(data=True):
            ma = _mult(cost_levels.get(a, "LOW")) if a in self.f.zones else 1.0
            mb = _mult(cost_levels.get(b, "LOW")) if b in self.f.zones else 1.0
            d["cost"] = d["length"] * (ma + mb) / 2.0
        return sub

    def _heuristic(self, a, b) -> float:
        return dist(self.G.nodes[a]["pos"], self.G.nodes[b]["pos"])

    # ------------------------------------------------------------------
    def waypoints(self, path: list) -> list:
        """Door points to walk through for a node path (exit point last)."""
        pts = []
        for a, b in zip(path[:-1], path[1:]):
            pts.append(tuple(self.G.edges[a, b]["door"]))
        return pts

    def route(self, start: str, levels: dict, *, cost_levels: dict | None = None,
              blocked_zones=(), blocked_exits=(), occupants: int = 0,
              max_transit: str = MAX_TRANSIT_LEVEL) -> RouteResult:
        """Best safe route from ``start`` zone to any available exit."""
        res = RouteResult(zone_id=start, occupants=int(occupants))
        if start not in self.f.zones:
            res.reason = f"unknown zone {start}"
            return res
        cost_levels = levels if cost_levels is None else cost_levels
        if levels.get(start, "LOW") == "CRITICAL":
            res.reason = "start zone is CRITICAL - occupants cannot leave safely"
            return res
        nodes, exits = self._allowed(start, levels, set(blocked_zones),
                                     set(blocked_exits), max_transit)
        if not exits:
            res.status = NO_EXIT_AVAILABLE
            res.reason = "no emergency exit is open and safe"
            return res
        g = self._weighted_graph(nodes, exits, cost_levels)
        cands = []
        for ex in sorted(exits):
            try:
                path = nx.astar_path(g, start, ex, heuristic=self._heuristic,
                                     weight="cost")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            cost = sum(g.edges[a, b]["cost"] for a, b in zip(path[:-1], path[1:]))
            length = sum(g.edges[a, b]["length"] for a, b in zip(path[:-1], path[1:]))
            cands.append((cost, ex, path, length))
        if not cands:
            res.reason = "every exit is cut off by hazardous or blocked zones"
            return res
        cands.sort(key=lambda c: (c[0], c[1]))
        cost, ex, path, length = cands[0]
        res.status = SAFE
        res.exit_id = ex
        res.path = path
        res.waypoints = self.waypoints(path)
        res.length_m = float(length)
        res.cost = float(cost)
        res.eta_s = float(length) / WORKER_SPEED
        res.passes_through = [p for p in path if p in self.f.zones]
        res.alternatives = [
            {"exit_id": e, "cost": float(c), "length_m": float(l),
             "path": p, "waypoints": self.waypoints(p)}
            for c, e, p, l in cands[1:]
        ]
        return res

    def plan(self, occupants: dict, levels: dict, *, cost_levels=None,
             blocked_zones=(), blocked_exits=(),
             max_transit: str = MAX_TRANSIT_LEVEL) -> dict:
        """Routes for every zone that currently has occupants."""
        return {
            z: self.route(z, levels, cost_levels=cost_levels,
                          blocked_zones=blocked_zones,
                          blocked_exits=blocked_exits, occupants=n,
                          max_transit=max_transit)
            for z, n in occupants.items() if n > 0
        }

    # ------------------------------------------------------------------
    def approach_path(self, src: str, dst: str, levels: dict, *,
                      blocked_zones=(), blocked_exits=(),
                      max_level: str = "MODERATE") -> list | None:
        """Safe approach path for a standard rescue team (A*, no unsafe zone).

        Every node on the path - including the destination - must be at or
        below ``max_level``.  Returns None when no safe approach exists.
        """
        limit = LEVEL_RANK[max_level]
        ok = lambda z: (z not in blocked_zones
                        and LEVEL_RANK[levels.get(z, "LOW")] <= limit)
        if dst not in self.f.zones or not ok(dst):
            return None
        nodes = {z for z in self.f.zones if ok(z)}
        keep = nodes | ({src} if src in self.f.exits else set())
        if src not in keep:
            return None
        g = self._weighted_graph(
            {n for n in keep if n in self.f.zones},
            {n for n in keep if n in self.f.exits}, levels)
        try:
            return nx.astar_path(g, src, dst, heuristic=self._heuristic,
                                 weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def path_text(path: list) -> str:
    return " > ".join(path) if path else "-"