"""Simulation engine: the simulated world plus the agent orchestration loop.

One call of :meth:`Simulation.tick` advances the world by ``DT`` seconds:

 1. physics   - the ground-truth plume advances (leak, wind),
 2. sensors   - MQ135 / PM10 / DHT11 readings are generated,
 3. exposure  - simulated workers accumulate dose from the true hazard,
 4. agents    - Detection -> Prediction -> Evacuation -> Rescue -> Master
                (+ one re-plan round if the master asks for it),
 5. workers   - follow the routes released by the Master agent.

The agents never see the ground truth, only sensor snapshots, a noisy wind
measurement and the occupancy report.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from agents import (EvacuationAgent, HazardDetectionAgent,
                    HazardPredictionAgent, MessageBus,
                    RescueCoordinationAgent)
from config import (DEMO, DOSE_FLOOR_H, DOSE_INCAPACITATED, DT, EVAC_PRESETS,
                    LEVEL_RANK, MAX_SIM_TIME, SCENARIOS, WORKER_SPEED)
from coordinator import MasterCoordinationAgent
from evacuation import SAFE
from factory import get_facility
from hazard import PlumeModel, field_on_grid

TERMINAL = ("EVACUATED", "INCAPACITATED", "TRAPPED")


@dataclass(frozen=True)
class SimConfig:
    scenario: str = "chem_storage"
    source_zone: str | None = None        # None -> scenario default
    intensity: int = 7                    # 1..10
    wind_from: float = 315.0
    wind_speed: float = 2.5
    seed: int = 42
    evac_preset: str = "all_open"
    hazard_type: str | None = None        # only used by the custom scenario
    growth: str | None = None             # only used by the custom scenario
    trigger_at: float | None = None       # auto-trigger time (demo mode)
    wind_events: tuple = ()               # ((t, from_deg, speed), ...)

    def world_signature(self) -> tuple:
        """Settings that define the world; changing them requires a reset."""
        return (self.scenario, self.source_zone, self.seed, self.hazard_type,
                self.growth)

    @staticmethod
    def for_scenario(key: str, **overrides) -> "SimConfig":
        sc = SCENARIOS[key]
        base = SimConfig(scenario=key, source_zone=None,
                         intensity=sc.default_intensity,
                         wind_from=sc.default_wind_from,
                         wind_speed=sc.default_wind_speed)
        return replace(base, **overrides)

    @staticmethod
    def demo() -> "SimConfig":
        d = DEMO
        return SimConfig(scenario=d["scenario"], source_zone=d["source_zone"],
                         intensity=d["intensity"], wind_from=d["wind_from"],
                         wind_speed=d["wind_speed"], seed=d["seed"],
                         evac_preset=d["evac_preset"], trigger_at=d["trigger_at"],
                         wind_events=tuple(d["wind_events"]))


@dataclass
class Worker:
    id: str
    x: float
    y: float
    zone: str
    status: str = "WORKING"      # WORKING EVACUATING TRAPPED INCAPACITATED EVACUATED
    dose: float = 0.0
    exit_id: str | None = None
    evacuated_t: float | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y, "zone": self.zone,
                "status": self.status, "dose": self.dose, "exit_id": self.exit_id}


class Simulation:
    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.f = get_facility()
        sc = SCENARIOS[self.cfg.scenario]
        self.scenario = sc
        self.source_zone = self.cfg.source_zone or sc.source_zone
        if self.source_zone not in self.f.zones:
            raise ValueError(f"unknown source zone {self.source_zone!r}")
        self.hazard_type = (self.cfg.hazard_type if sc.key == "custom" and self.cfg.hazard_type
                            else sc.hazard_type)
        self.growth = (self.cfg.growth if sc.key == "custom" and self.cfg.growth
                       else sc.growth)
        self.flammable = sc.flammable or self.hazard_type in ("flammable_gas", "fire_smoke")
        self.source_xy = self.f.source_point(self.source_zone, self.cfg.seed)
        self.plume = PlumeModel(self.source_xy)
        from sensors import SensorNetwork
        self.sensors = SensorNetwork(self.f, self.hazard_type, self.cfg.seed)

        self.workers: list[Worker] = [
            Worker(w["id"], w["x"], w["y"], self.f.zone_at(w["x"], w["y"]))
            for w in self.f.place_workers(self.cfg.seed)]

        self.bus = MessageBus()
        self.detection = HazardDetectionAgent(self.f)
        self.prediction = HazardPredictionAgent(self.f)
        self.evacuation = EvacuationAgent(self.f)
        self.rescue = RescueCoordinationAgent(self.f)
        self.master = MasterCoordinationAgent(self.f)
        self.agents = [self.detection, self.prediction, self.evacuation,
                       self.rescue, self.master]

        self.t = 0.0
        self.tick_idx = 0
        self.intensity = int(self.cfg.intensity)
        self.wind_from = float(self.cfg.wind_from)
        self.wind_speed = float(self.cfg.wind_speed)
        self.leak_active = False
        self.wind_version = 0
        self.t_leak: float | None = None
        self._wind_events = sorted(self.cfg.wind_events)
        self.set_evac_preset(self.cfg.evac_preset)

        self.events: list[dict] = []
        self.sensor_frames: deque = deque(maxlen=300)
        self.zone_h_history: deque = deque(maxlen=400)
        self.status_history: list[tuple[float, str]] = []
        self.out: dict = {}
        self.coord: dict = {}
        self.complete = False
        self._resolved_ticks = 0
        self._seen: set = set()
        self._truth_cache: tuple[int, np.ndarray] | None = None
        self.t_first_abnormal: float | None = None
        self.t_detect: float | None = None
        self.t_alarm: float | None = None
        self.t_complete: float | None = None
        self.log_event("SYSTEM", "Simulation",
                       f"Scenario '{sc.label}' loaded (seed {self.cfg.seed}); "
                       "facility in normal operation")
        self.tick(initial=True)      # one baseline reading so the UI has data

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------
    def set_wind(self, wind_from: float, wind_speed: float) -> None:
        self.wind_from = float(wind_from) % 360.0
        self.wind_speed = max(0.0, float(wind_speed))

    def set_intensity(self, intensity: int) -> None:
        self.intensity = int(min(10, max(1, intensity)))

    def set_evac_preset(self, key: str) -> None:
        preset = EVAC_PRESETS.get(key, EVAC_PRESETS["all_open"])
        self.evac_preset = key if key in EVAC_PRESETS else "all_open"
        self.blocked_exits = list(preset["blocked_exits"])
        self.blocked_zones = list(preset["blocked_zones"])

    def trigger_leak(self) -> bool:
        """Start the hazard at the current time.  Returns False if already on."""
        if self.leak_active:
            return False
        self.leak_active = True
        self.t_leak = self.t
        self.log_event("INCIDENT", "Simulation",
                       f"{self.scenario.label.split('. ', 1)[-1]} INITIATED in "
                       f"{self.f.zones[self.source_zone].name} "
                       f"(intensity {self.intensity}/10) [simulated]")
        return True

    # ------------------------------------------------------------------
    def log_event(self, category: str, source: str, text: str) -> None:
        self.events.append({"t": round(self.t, 1), "category": category,
                            "source": source, "text": text})

    def _q(self) -> float:
        """True emission rate at the current time."""
        if not self.leak_active:
            return 0.0
        q0 = self.intensity * self.scenario.q_factor
        dt = max(0.0, self.t - (self.t_leak or 0.0))
        tau = max(self.scenario.growth_tau, 1e-6)
        if self.growth == "t2":
            return q0 * min(1.0, (dt / tau) ** 2)
        if self.growth == "step":
            return q0
        return q0 * (1.0 - math.exp(-dt / tau))

    # ------------------------------------------------------------------
    def truth_grid(self) -> np.ndarray:
        """Ground-truth hazard field (for display / validation only)."""
        if self._truth_cache and self._truth_cache[0] == self.tick_idx:
            return self._truth_cache[1]
        g = field_on_grid(self.plume, self.f,
                          diff_factor=self.scenario.diffusivity)
        self._truth_cache = (self.tick_idx, g)
        return g

    def _site_state(self) -> dict:
        alive = [w for w in self.workers if w.status != "EVACUATED"]
        occ: dict[str, int] = {}
        for w in alive:
            occ[w.zone] = occ.get(w.zone, 0) + 1
        return {"occupancy": occ, "workers": [w.as_dict() for w in self.workers],
                "blocked_exits": list(self.blocked_exits),
                "blocked_zones": list(self.blocked_zones)}

    # ------------------------------------------------------------------
    def tick(self, initial: bool = False) -> None:
        if self.complete and not initial:
            return
        # scheduled events (demo script)
        if (not initial and not self.leak_active and self.cfg.trigger_at is not None
                and self.t >= self.cfg.trigger_at):
            self.trigger_leak()
        while self._wind_events and self.t >= self._wind_events[0][0] and not initial:
            _, wf, ws = self._wind_events.pop(0)
            self.set_wind(wf, ws)
            self.wind_version += 1       # tells the dashboard to sync its sliders
            self.log_event("ENVIRONMENT", "Simulation",
                           f"Wind shift: now from {wf:.0f} deg at {ws:.1f} m/s [scripted]")

        # 1. physics ----------------------------------------------------
        self.plume.step(self._q(), self.wind_from, self.wind_speed,
                        self.scenario.diffusivity)
        tick_index = self.tick_idx            # index of the interval just simulated
        self.tick_idx += 1
        self.t = self.tick_idx * DT

        # 2. sensors ----------------------------------------------------
        sx = np.array([s.x for s in self.f.sensors])
        sy = np.array([s.y for s in self.f.sensors])
        live = [w for w in self.workers if w.status != "EVACUATED"]
        wx = np.array([w.x for w in live])
        wy = np.array([w.y for w in live])
        h_all = self.plume.hazard(np.concatenate([sx, wx]), np.concatenate([sy, wy]),
                                  diff_factor=self.scenario.diffusivity) \
            if len(live) + len(sx) else np.zeros(0)
        # note: diff_factor only matters for forecasts; hazard() ignores it
        h_s, h_w = h_all[:len(sx)], h_all[len(sx):]
        df = self.sensors.read(tick_index, self.t, h_s)
        wind_meas = self.sensors.read_weather(self.wind_from, self.wind_speed)
        self.sensor_frames.append(df)

        # 3. exposure ---------------------------------------------------
        for w, h in zip(live, h_w):
            if h > DOSE_FLOOR_H:
                w.dose += (h - DOSE_FLOOR_H) * DT
            if w.dose >= DOSE_INCAPACITATED and w.status != "INCAPACITATED":
                w.status = "INCAPACITATED"
                self.log_event("CASUALTY", "Simulation",
                               f"Worker {w.id} INCAPACITATED in {w.zone} (simulated dose)")

        # 4. agents -----------------------------------------------------
        self._run_agents(tick_index, df, wind_meas)

        # 5. workers ----------------------------------------------------
        self._update_workers()
        self._check_complete()

    # ------------------------------------------------------------------
    def _run_agents(self, tick_index: int, df: pd.DataFrame, wind_meas: dict) -> None:
        t = self.t
        bus = self.bus
        bus.publish(t, "Sensor Network", "detection", "sensor_data",
                    {"snapshot": df, "wind": wind_meas},
                    f"sensor_data: {len(df)} nodes + wind {wind_meas['from']:.0f}/"
                    f"{wind_meas['speed']:.1f}")
        site = self._site_state()
        for rcv in ("evacuation", "rescue", "master"):
            bus.publish(t, "Worker Tracker", rcv, "site_state", site,
                        f"site_state: {sum(site['occupancy'].values())} people on site")
        ctx = {"t": t, "tick": tick_index, "pass": 1}
        det = self.detection.step(bus, ctx)
        pred = self.prediction.step(bus, ctx)
        evac = self.evacuation.step(bus, ctx)
        resc = self.rescue.step(bus, ctx)
        coord = self.master.step(bus, ctx)
        if self.master.pending_replan:
            ctx2 = dict(ctx, **{"pass": 2})
            evac = self.evacuation.step(bus, ctx2) or evac
            resc = self.rescue.step(bus, ctx2) or resc
            coord = self.master.step(bus, ctx2)
        bus.consume("simulation")
        self.coord = coord or self.coord
        # keep the latest payload of every agent for the dashboard
        self.out = {
            "detection": det,
            "prediction": pred or self.out.get("prediction"),
            "evacuation": evac or self.out.get("evacuation"),
            "rescue": resc or self.out.get("rescue"),
            "master": self.coord,
        }

        for ag in self.agents:
            for cat, text in ag.drain_events():
                self.log_event(cat, ag.name, text)

        # bookkeeping ---------------------------------------------------
        if det:
            if (self.leak_active and self.t_first_abnormal is None
                    and any(s["instant"] >= 1 for s in det["sensor_states"].values())):
                self.t_first_abnormal = t
                worst = max(det["sensor_states"].items(), key=lambda kv: kv[1]["anomaly"])
                self.log_event("SENSOR", "Sensor Network",
                               f"First abnormal reading: {worst[0]} in {worst[1]['zone']} "
                               f"(anomaly {worst[1]['anomaly']:.2f}x alarm level)")
            if det["confirmed"] and self.t_detect is None:
                self.t_detect = t
            if not det["confirmed"] and self.t_detect is not None and \
                    self.detection.incident is None:
                pass
        if self.coord:
            order = self.coord.get("evac_order", {})
            if order.get("active") and self.t_alarm is None:
                self.t_alarm = t
            self.status_history.append((t, self.coord.get("status", "NORMAL")))
        pm = self.out.get("prediction")
        if pm and pm.get("active"):
            self.zone_h_history.append((t, dict(pm["zone_H"])))

    # ------------------------------------------------------------------
    def _update_workers(self) -> None:
        order = (self.coord or {}).get("evac_order", {})
        routes = (self.coord or {}).get("final_routes", {})
        go = bool(order.get("active")) and self.t >= (order.get("start_t") or 1e18)
        for w in self.workers:
            if w.status in ("EVACUATED", "INCAPACITATED"):
                continue
            w.zone = self.f.zone_at(w.x, w.y)
            r = routes.get(w.zone)
            in_scope = bool(r and r.get("in_scope"))
            if not (go and in_scope):
                w.status = "WORKING" if w.status in ("WORKING", "TRAPPED") else w.status
                if w.status == "EVACUATING" and not (go and in_scope):
                    w.status = "WORKING"
                continue
            if r["status"] != SAFE or not r["waypoints"]:
                if w.status != "TRAPPED":
                    self.log_event("CASUALTY", "Simulation",
                                   f"Worker {w.id} in {w.zone} has NO safe route - "
                                   "shelter in place, rescue requested")
                w.status = "TRAPPED"
                continue
            if w.status == "TRAPPED":
                self.log_event("EVACUATION", "Simulation",
                               f"Worker {w.id}: safe route found again - evacuating")
            w.status = "EVACUATING"
            w.exit_id = r["exit_id"]
            remaining = WORKER_SPEED * DT
            for wp in r["waypoints"]:
                d = math.hypot(wp[0] - w.x, wp[1] - w.y)
                if d <= remaining + 1e-9:
                    w.x, w.y = float(wp[0]), float(wp[1])
                    remaining -= d
                else:
                    w.x += (wp[0] - w.x) * remaining / d
                    w.y += (wp[1] - w.y) * remaining / d
                    remaining = 0.0
                    break
            ex = self.f.exits[r["exit_id"]]
            if math.hypot(w.x - ex.x, w.y - ex.y) < 1e-6:
                w.status = "EVACUATED"
                w.evacuated_t = self.t
                self.log_event("EVACUATION", "Simulation",
                               f"Worker {w.id} reached exit {ex.id} ({ex.name})")
            else:
                w.zone = self.f.zone_at(w.x, w.y)

    def _check_complete(self) -> None:
        order = (self.coord or {}).get("evac_order", {})
        if self.t >= MAX_SIM_TIME and not self.complete:
            self.complete = True
            self.t_complete = self.t
            self.log_event("SYSTEM", "Simulation", "Maximum simulation time reached")
            return
        if not order.get("active") or self.t < (order.get("start_t") or 1e18):
            self._resolved_ticks = 0
            return
        scope_all = order.get("site_wide")
        pending = [w for w in self.workers
                   if w.status in ("WORKING", "EVACUATING")
                   and (scope_all or w.zone in order.get("scope", []))]
        if not pending:
            self._resolved_ticks += 1
        else:
            self._resolved_ticks = 0
        if self._resolved_ticks >= 5 and not self.complete:
            self.complete = True
            self.t_complete = self.t
            self.log_event("SYSTEM", "Simulation",
                           "Response complete - every ordered worker is evacuated, "
                           "trapped or incapacitated. Final summary available.")

    # ------------------------------------------------------------------
    # convenience for UI / tests
    # ------------------------------------------------------------------
    def sensor_history(self) -> pd.DataFrame:
        if not self.sensor_frames:
            return pd.DataFrame()
        return pd.concat(list(self.sensor_frames), ignore_index=True)

    def latest_sensors(self) -> pd.DataFrame:
        return self.sensor_frames[-1] if self.sensor_frames else pd.DataFrame()

    def worker_counts(self) -> dict:
        c = {"WORKING": 0, "EVACUATING": 0, "TRAPPED": 0, "INCAPACITATED": 0,
             "EVACUATED": 0}
        for w in self.workers:
            c[w.status] = c.get(w.status, 0) + 1
        return c

    def summary(self) -> dict:
        c = self.worker_counts()
        det = self.detection.incident
        m = (self.coord or {}).get("metrics", {})
        rs = self.out.get("rescue") or {}
        return {
            "scenario": self.scenario.label,
            "source_zone": self.source_zone,
            "leak_started_t": self.t_leak,
            "first_abnormal_t": self.t_first_abnormal,
            "detected_t": self.t_detect,
            "detection_delay_s": (None if (self.t_detect is None or self.t_leak is None)
                                  else self.t_detect - self.t_leak),
            "evac_order_t": (self.coord or {}).get("evac_order", {}).get("issued_t"),
            "peak_status": self.master.peak_status,
            "current_status": (self.coord or {}).get("status", "NORMAL"),
            "workers": c, "workers_total": len(self.workers),
            "route_recalculations": self.evacuation.recalc_count,
            "conflict_resolutions": self.master.route_overrides_total,
            "rescue_priorities": rs.get("ranking", [])[:5],
            "specialist_zones": rs.get("specialist_zones", []),
            "incident_open": det is not None,
            "complete": self.complete, "t": self.t,
        }