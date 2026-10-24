"""Agents 1-4 of ResQM plus the message bus they talk through.

* Agent 1 - HazardDetectionAgent   : sensors -> confirmed incident report
* Agent 2 - HazardPredictionAgent  : incident -> fitted plume, risk map, forecast
* Agent 3 - EvacuationAgent        : risk map + occupancy -> A* routes
* Agent 4 - RescueCoordinationAgent: risk + routes + casualties -> ranked rescue plan

Agent 5 (the master) lives in ``coordinator.py``.

Agents never call each other.  They only read their inbox on the
``MessageBus`` and publish messages, so the data flow is visible in the log.
All logic is rule based / classical algorithms - no trained ML model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config import (CONFIRM_TICKS, CLEAR_TICKS, DT, FIT_MAX_LAG_TICKS,
                    FORECAST_DISPLAY_S, FORECAST_HORIZONS_S, HAZARD_TYPES,
                    LEVEL_RANK, LEVELS, DOSE_EXPOSED, DOSE_INCAPACITATED,
                    RESCUE_SPECIALIST_TEAMS, RESCUE_STANDARD_TEAMS,
                    SENSOR_BASELINE, level_from_h)
from evacuation import EvacuationPlanner, NO_SAFE_ROUTE, SAFE, path_text
from hazard import PlumeModel, field_on_grid, wind_vector, zone_hazard
from sensors import anomaly_score, channel_levels

# --------------------------------------------------------------------------
# Message bus
# --------------------------------------------------------------------------


@dataclass
class Message:
    seq: int
    t: float
    sender: str
    receiver: str
    topic: str
    summary: str
    payload: dict = field(default_factory=dict, repr=False)


class MessageBus:
    """Very small publish / inbox system with a visible log."""

    def __init__(self, max_log: int = 800) -> None:
        self.inboxes: dict[str, list[Message]] = {}
        self.log: deque[Message] = deque(maxlen=max_log)
        self._seq = 0

    def publish(self, t: float, sender: str, receiver: str, topic: str,
                payload: dict, summary: str = "") -> Message:
        self._seq += 1
        msg = Message(self._seq, float(t), sender, receiver, topic,
                      summary or topic, payload)
        self.inboxes.setdefault(receiver, []).append(msg)
        self.log.append(msg)
        return msg

    def consume(self, receiver: str) -> list[Message]:
        msgs = self.inboxes.get(receiver, [])
        self.inboxes[receiver] = []
        return msgs

    def log_rows(self, last: int = 60, include_topics=None) -> list[dict]:
        rows = []
        for m in list(self.log)[-last:]:
            if include_topics and m.topic not in include_topics:
                continue
            rows.append({"seq": m.seq, "t": m.t, "from": m.sender, "to": m.receiver,
                         "topic": m.topic, "summary": m.summary})
        return rows


# --------------------------------------------------------------------------
# Base agent
# --------------------------------------------------------------------------


class BaseAgent:
    key = "agent"
    name = "Agent"
    role = ""

    def __init__(self) -> None:
        self.status = "STANDBY"
        self.last_input = "-"
        self.last_analysis = "-"
        self.last_decision = "-"
        self.last_output = "-"
        self.activations = 0
        self.last_active_t = None
        self.events: list[tuple[str, str]] = []     # (category, text)
        self._mem: dict[str, dict] = {}
        self._active_incident: int | None = None

    def emit_event(self, category: str, text: str) -> None:
        self.events.append((category, text))

    def drain_events(self) -> list[tuple[str, str]]:
        ev, self.events = self.events, []
        return ev

    def _read(self, bus: MessageBus) -> list[Message]:
        msgs = bus.consume(self.key)
        for m in msgs:
            self._mem[m.topic] = m.payload
        return msgs

    def _set(self, status, inp, analysis, decision, output, t) -> None:
        self.status = status
        self.last_input = inp
        self.last_analysis = analysis
        self.last_decision = decision
        self.last_output = output
        self.last_active_t = t

    def _activate_for(self, incident_id, t: float, why: str) -> None:
        if incident_id is not None and incident_id != self._active_incident:
            self._active_incident = incident_id
            self.activations += 1
            self.emit_event("AGENT", f"{self.name} activated - {why}")

    def snapshot(self) -> dict:
        return {"key": self.key, "name": self.name, "role": self.role,
                "status": self.status, "input": self.last_input,
                "analysis": self.last_analysis, "decision": self.last_decision,
                "output": self.last_output, "activations": self.activations,
                "last_active_t": self.last_active_t}


# --------------------------------------------------------------------------
# Agent 1 - Hazard detection
# --------------------------------------------------------------------------


def severity_label(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.5:
        return "HIGH"
    if score >= 0.25:
        return "MODERATE"
    return "LOW"


class HazardDetectionAgent(BaseAgent):
    key = "detection"
    name = "Hazard Detection Agent"
    role = "Monitors MQ135 / PM10 / DHT11 readings, confirms and localises incidents"

    def __init__(self, facility, confirm_ticks: int = CONFIRM_TICKS) -> None:
        super().__init__()
        self.f = facility
        self.confirm_ticks = confirm_ticks
        self.hist = {s.id: deque(maxlen=confirm_ticks) for s in facility.sensors}
        self.incident: dict | None = None
        self.counter = 0
        self.calm = 0
        self.suppressed_blips = 0

    # ------------------------------------------------------------------
    def _evaluate(self, df) -> dict:
        states = {}
        for r in df.itertuples(index=False):
            if r.health != "OK" or r.gas != r.gas:
                states[r.sensor_id] = {
                    "zone": r.zone_id, "levels": {c: 0 for c in
                                                   ("gas", "pm10", "temp", "hum")},
                    "instant": 0, "confirmed": 0, "anomaly": 0.0,
                    "status": "FAULT"}
                self.hist[r.sensor_id].append(0)
                continue
            lv = channel_levels(r.gas, r.pm10, r.temp, r.hum)
            inst = max(lv.values())
            h = self.hist[r.sensor_id]
            h.append(inst)
            conf = min(h) if len(h) == self.confirm_ticks else 0
            if inst >= 1 and conf == 0:
                pass                                   # waiting for confirmation
            states[r.sensor_id] = {
                "zone": r.zone_id, "levels": lv, "instant": inst,
                "confirmed": conf,
                "anomaly": anomaly_score(r.gas, r.pm10, r.temp, r.hum),
                "status": ("ALARM" if inst == 2 else "WARNING" if inst == 1
                           else "NORMAL")}
        return states

    @staticmethod
    def _incident_type(states: dict) -> str | None:
        fire = any(s["levels"]["temp"] >= 1 and s["levels"]["pm10"] >= 1
                   for s in states.values())
        smoke_only = any(s["levels"]["pm10"] == 2 for s in states.values())
        gas = any(s["levels"]["gas"] >= 1 for s in states.values())
        if fire:
            return "fire_smoke"
        if gas:
            return "gas_leak"
        if smoke_only:
            return "fire_smoke"
        return None

    # ------------------------------------------------------------------
    def step(self, bus: MessageBus, ctx: dict) -> dict:
        t, tick = ctx["t"], ctx["tick"]
        msgs = self._read(bus)
        data = self._mem.get("sensor_data")
        if not data:
            self.status = "STANDBY"
            return {}
        df = data["snapshot"]
        states = self._evaluate(df)
        n = len(states)
        n_warn = sum(1 for s in states.values() if s["status"] == "WARNING")
        n_alarm = sum(1 for s in states.values() if s["status"] == "ALARM")
        faulty = [sid for sid, s in states.items() if s["status"] == "FAULT"]
        conf_alarm = [sid for sid, s in states.items() if s["confirmed"] == 2]
        conf_warn = [sid for sid, s in states.items() if s["confirmed"] >= 1]
        inst_abn = [sid for sid, s in states.items() if s["instant"] >= 1]
        if inst_abn and not conf_warn and self.incident is None:
            self.suppressed_blips += 1

        declare = self.incident is None and (conf_alarm or len(conf_warn) >= 2)
        if declare:
            self.counter += 1
            pool = conf_alarm or conf_warn
            origin_sid = max(pool, key=lambda s: states[s]["anomaly"])
            itype = self._incident_type(states) or "gas_leak"
            self.incident = {
                "id": self.counter, "origin_zone": states[origin_sid]["zone"],
                "origin_sensor": origin_sid, "detect_tick": tick, "detect_t": t,
                "type": itype}
            self.calm = 0
            self.emit_event(
                "DETECTION",
                f"Incident #{self.counter} CONFIRMED by {len(conf_warn)} sensor(s) "
                f"near {self.f.zones[self.incident['origin_zone']].name} "
                f"({'fire / smoke' if itype == 'fire_smoke' else 'gas release'} "
                f"signature)")

        confirmed = self.incident is not None
        severity = 0.0
        if confirmed:
            t_now = self._incident_type(states)
            if t_now == "fire_smoke" and self.incident["type"] != "fire_smoke":
                self.incident["type"] = "fire_smoke"
                self.emit_event("DETECTION", "Incident type re-classified: fire / smoke "
                                "(temperature + PM10 rising together)")
            peak = max((s["anomaly"] for s in states.values()), default=0.0)
            zones_alarm = {s["zone"] for s in states.values() if s["status"] == "ALARM"}
            severity = float(np.clip(0.5 * min(peak, 3.0) / 3.0
                                     + 0.5 * min(len(zones_alarm) / 6.0, 1.0), 0, 1))
            if inst_abn:
                self.calm = 0
            else:
                self.calm += 1
                if self.calm >= CLEAR_TICKS:
                    self.emit_event("DETECTION", f"Incident #{self.incident['id']} "
                                    "cleared - all sensors back to baseline")
                    self.incident = None
                    confirmed = False
                    severity = 0.0
        sev_lbl = severity_label(severity) if confirmed else "NONE"
        zone_anom = {}
        for s in states.values():
            zone_anom[s["zone"]] = max(zone_anom.get(s["zone"], 0.0), s["anomaly"])
        channels = sorted({ch for s in states.values()
                           for ch, v in s["levels"].items() if v >= 1})

        report = {
            "confirmed": confirmed,
            "incident_id": self.incident["id"] if confirmed else None,
            "type": self.incident["type"] if confirmed else None,
            "origin_zone": self.incident["origin_zone"] if confirmed else None,
            "detect_tick": self.incident["detect_tick"] if confirmed else None,
            "detect_t": self.incident["detect_t"] if confirmed else None,
            "severity": severity, "severity_label": sev_lbl,
            "sensor_states": states, "zone_anomaly": zone_anom,
            "violating_zones": sorted({s["zone"] for s in states.values()
                                       if s["instant"] >= 1}),
            "faulty_sensors": faulty, "channels": channels,
            "snapshot": df, "wind": data.get("wind"),
            "tick": tick, "t": t,
        }
        for rcv in ("prediction", "evacuation", "rescue", "master"):
            bus.publish(t, self.key, rcv, "incident_report", report,
                        (f"Incident #{report['incident_id']} {sev_lbl} @ "
                         f"{report['origin_zone']}") if confirmed else
                        "No incident (sensors nominal)")

        if confirmed:
            self._activate_for(self.incident["id"], t, "incident confirmed")
            top = max(states, key=lambda s: states[s]["anomaly"])
            self._set("INCIDENT ACTIVE",
                      f"{n} sensor nodes ({n_warn} warn, {n_alarm} alarm, {len(faulty)} fault)",
                      f"Origin {self.incident['origin_zone']}; peak anomaly "
                      f"{states[top]['anomaly']:.2f}x alarm at {top}; channels "
                      f"{', '.join(channels) or '-'}",
                      f"Severity {sev_lbl} ({severity:.2f}); type "
                      f"{self.incident['type']}",
                      "incident_report -> Prediction, Evacuation, Rescue, Master", t)
        else:
            self._set("MONITORING" if not inst_abn else "WATCHING",
                      f"{n} sensor nodes ({n_warn} warn, {n_alarm} alarm)",
                      "All channels within limits" if not inst_abn else
                      f"{len(inst_abn)} transient abnormal reading(s) - awaiting "
                      f"{self.confirm_ticks}-tick confirmation",
                      "No incident declared",
                      "incident_report(confirmed=False) -> all agents", t)
        return report


# --------------------------------------------------------------------------
# Agent 2 - Hazard prediction
# --------------------------------------------------------------------------


class HazardPredictionAgent(BaseAgent):
    key = "prediction"
    name = "Hazard Prediction Agent"
    role = "Fits a Gaussian-puff plume to sensor data, forecasts spread, builds the risk map"

    def __init__(self, facility, diff_factor_by_type: dict | None = None) -> None:
        super().__init__()
        self.f = facility
        self.wind_log: list[tuple[float, float]] = []   # measured wind per tick
        self.fit: dict = {}
        self._last_levels: dict | None = None

    def reset_incident_state(self) -> None:
        self.fit = {}
        self._active_incident = None

    # ------------------------------------------------------------------
    @staticmethod
    def _diff_factor(itype: str) -> float:
        # the predictor does not know the true gas: it assumes a typical
        # light-to-medium factor per incident class.
        return 1.7 if itype == "fire_smoke" else 1.15

    def _observed_h(self, df, itype: str) -> dict:
        key = "toxic_gas" if itype != "fire_smoke" else "fire_smoke"
        prim = HAZARD_TYPES[key]["primary"]
        gain = HAZARD_TYPES[key][prim]
        base = SENSOR_BASELINE[prim]
        obs = {}
        for r in df.itertuples(index=False):
            v = getattr(r, prim)
            if r.health != "OK" or v != v:
                continue
            obs[r.sensor_id] = max(0.0, (float(v) - base) / gain)
        return obs

    def _fit(self, incident: dict, df, itype: str) -> dict:
        n = len(self.wind_log)
        diff = self._diff_factor(itype)
        src = self.f.zones[incident["origin_zone"]].center
        start0 = max(0, incident["detect_tick"] - FIT_MAX_LAG_TICKS)
        model = PlumeModel(src)
        for i in range(start0, n):
            wf, ws = self.wind_log[i]
            model.step(1.0, wf, ws, diff)       # unit emission, masked later
        obs = self._observed_h(df, itype)
        sids = [s for s in obs]
        pts_x = np.array([self.f.sensor_by_id[s].x for s in sids])
        pts_y = np.array([self.f.sensor_by_id[s].y for s in sids])
        o = np.array([obs[s] for s in sids])
        best = None
        for lag in range(0, FIT_MAX_LAG_TICKS + 1):
            start_tick = incident["detect_tick"] - lag
            if start_tick < start0:
                continue
            emit_from = start_tick - start0
            h = model.hazard(pts_x, pts_y, emit_from=emit_from)
            denom = float((h * h).sum())
            if denom < 1e-9:
                continue
            scale = max(0.0, float((h * o).sum()) / denom)
            resid = float(((o - scale * h) ** 2).sum())
            if best is None or resid < best["resid"] - 1e-9:
                best = {"lag": lag, "scale": scale, "resid": resid,
                        "emit_from": emit_from}
        if best is None or best["scale"] <= 0:
            best = {"lag": 0, "scale": 1.0, "resid": float("nan"),
                    "emit_from": incident["detect_tick"] - start0}
        best.update({"model": model, "diff": diff, "obs": obs, "source": src,
                     "rmse": (best["resid"] / max(len(o), 1)) ** 0.5
                     if best["resid"] == best["resid"] else float("nan")})
        return best

    # ------------------------------------------------------------------
    def step(self, bus: MessageBus, ctx: dict) -> dict:
        t = ctx["t"]
        self._read(bus)
        rep = self._mem.get("incident_report")
        if not rep:
            return {}
        wind = rep.get("wind") or {"from": 0.0, "speed": 0.0}
        self.wind_log.append((float(wind["from"]), float(wind["speed"])))
        levels0 = {z: "LOW" for z in self.f.zones}
        if not rep["confirmed"]:
            if self.fit:
                self.reset_incident_state()
            out = {"active": False, "t": t, "tick": ctx["tick"], "zone_H": {z: 0.0 for z in self.f.zones},
                   "zone_levels": levels0, "forecast_H": {z: 0.0 for z in self.f.zones},
                   "forecast_levels": levels0, "forecast30_levels": levels0,
                   "time_to_high": {}, "time_to_critical": {}, "grid": None,
                   "forecast_grid": None, "est": {}, "threatened": [],
                   "forecast_threatened": [], "incident_point": None}
            for rcv in ("evacuation", "rescue", "master"):
                bus.publish(t, self.key, rcv, "hazard_map", out, "No hazard (standby)")
            self._set("STANDBY", "incident_report: none", "Waiting for a confirmed incident",
                      "No prediction needed", "hazard_map(active=False)", t)
            return out

        self._activate_for(rep["incident_id"], t, "received confirmed incident report")
        itype = rep["type"]
        fit = self._fit({"origin_zone": rep["origin_zone"],
                         "detect_tick": rep["detect_tick"]}, rep["snapshot"], itype)
        self.fit = fit
        model, scale, emit_from = fit["model"], fit["scale"], fit["emit_from"]
        last_wind = self.wind_log[-1]

        grid_now = field_on_grid(model, self.f, emission_scale=scale,
                                 emit_from=emit_from)
        zone_now = zone_hazard(self.f, grid_now)
        # sensor-supremacy rule: never report less than the sensor actually sees
        for sid, h_obs in fit["obs"].items():
            z = self.f.sensor_by_id[sid].zone_id
            zone_now[z] = max(zone_now[z], h_obs)

        horizons = {}
        grids = {}
        for hs in FORECAST_HORIZONS_S:
            steps = int(round(hs / DT))
            g = field_on_grid(model, self.f, emission_scale=scale, emit_from=emit_from,
                              extra_steps=steps, wind=last_wind, q_future=1.0,
                              diff_factor=fit["diff"])
            grids[hs] = g
            horizons[hs] = zone_hazard(self.f, g)
        fh = FORECAST_DISPLAY_S if FORECAST_DISPLAY_S in horizons else max(horizons)
        forecast_h = {z: max(horizons[fh][z], zone_now[z]) for z in self.f.zones}
        f30 = {z: max(horizons[30][z], zone_now[z]) for z in self.f.zones} \
            if 30 in horizons else dict(zone_now)

        t_high, t_crit = {}, {}
        for z in self.f.zones:
            th = tc = None
            if zone_now[z] >= 40.0:
                th = 0.0
            if zone_now[z] >= 70.0:
                tc = 0.0
            for hs in FORECAST_HORIZONS_S:
                hv = max(horizons[hs][z], zone_now[z])
                if th is None and hv >= 40.0:
                    th = float(hs)
                if tc is None and hv >= 70.0:
                    tc = float(hs)
            t_high[z], t_crit[z] = th, tc

        lv_now = {z: level_from_h(h) for z, h in zone_now.items()}
        lv_fc = {z: level_from_h(h) for z, h in forecast_h.items()}
        lv_30 = {z: level_from_h(h) for z, h in f30.items()}
        threatened = [z for z, l in lv_now.items() if LEVEL_RANK[l] >= 1]
        fc_threat = [z for z, l in lv_fc.items()
                     if LEVEL_RANK[l] >= 1 and z not in threatened]
        est_point = self.f.zones[rep["origin_zone"]].center
        out = {
            "active": True, "t": t, "tick": ctx["tick"], "zone_H": zone_now,
            "zone_levels": lv_now, "forecast_H": forecast_h,
            "forecast_levels": lv_fc, "forecast30_levels": lv_30,
            "time_to_high": t_high, "time_to_critical": t_crit,
            "grid": grid_now, "forecast_grid": grids[fh],
            "est": {"source": est_point, "source_zone": rep["origin_zone"],
                    "detected_zone": rep["origin_zone"], "lag_ticks": fit["lag"],
                    "scale": scale, "rmse": fit["rmse"], "wind": last_wind,
                    "horizon_s": fh, "type": itype},
            "threatened": threatened, "forecast_threatened": fc_threat,
            "incident_point": est_point,
        }
        n_red = sum(1 for l in lv_now.values() if l == "CRITICAL")
        n_org = sum(1 for l in lv_now.values() if l == "HIGH")
        sig = tuple(sorted(lv_now.items()))
        if sig != self._last_levels:
            if self._last_levels is not None:
                changed = [z for z in lv_now if lv_now[z] != dict(self._last_levels).get(z)]
                self.emit_event(
                    "PREDICTION",
                    "Risk map updated: " + ", ".join(
                        f"{z}->{lv_now[z]}" for z in changed[:6])
                    + (" ..." if len(changed) > 6 else ""))
            else:
                self.emit_event("PREDICTION",
                                f"First hazard map generated ({len(threatened)} zone(s) "
                                f"above LOW, {len(fc_threat)} more forecast within "
                                f"{fh}s)")
            self._last_levels = sig
        for rcv in ("evacuation", "rescue", "master"):
            bus.publish(t, self.key, rcv, "hazard_map", out,
                        f"hazard_map: {len(threatened)} threatened, {n_org} HIGH, "
                        f"{n_red} CRITICAL, +{len(fc_threat)} forecast")
        ws = last_wind
        self._set("PREDICTING",
                  f"incident #{rep['incident_id']} origin {rep['origin_zone']} "
                  f"({itype}); wind {ws[0]:.0f} deg / {ws[1]:.1f} m/s (measured)",
                  f"Best fit: release started {fit['lag'] * DT:.0f}s before detection, "
                  f"strength x{scale:.2f}, fit RMSE {fit['rmse']:.1f} index pts",
                  f"{len(threatened)} zones threatened now, {len(fc_threat)} additional "
                  f"zones forecast within {fh}s",
                  "hazard_map -> Evacuation, Rescue, Master", t)
        return out


# --------------------------------------------------------------------------
# Agent 3 - Evacuation
# --------------------------------------------------------------------------


class EvacuationAgent(BaseAgent):
    key = "evacuation"
    name = "Evacuation Agent"
    role = "Builds the corridor graph and finds A* routes that avoid hazardous zones"

    def __init__(self, facility) -> None:
        super().__init__()
        self.f = facility
        self.planner = EvacuationPlanner(facility)
        self._sigs: dict[str, tuple] = {}
        self.recalc_count = 0
        self.directive_blocked: set[str] = set()
        self._first = True

    def step(self, bus: MessageBus, ctx: dict) -> dict:
        t = ctx["t"]
        for m in self._read(bus):
            if m.topic == "directive":
                self.directive_blocked = set(m.payload.get("extra_blocked_zones", []))
        hz = self._mem.get("hazard_map")
        site = self._mem.get("site_state")
        rep = self._mem.get("incident_report")
        if not hz or not site:
            return {}
        occ = site["occupancy"]
        blocked_exits = set(site.get("blocked_exits", []))
        blocked_manual = set(site.get("blocked_zones", []))
        levels = dict(hz["zone_levels"])
        active = bool(hz.get("active"))
        extra = set(self.directive_blocked)
        blocked_zones = blocked_manual | extra

        if active:
            self._activate_for(rep["incident_id"] if rep else 1, t,
                               "hazard map received - planning evacuation routes")
        # Soft cost: a zone is charged for the worse of "now" and "+30 s
        # forecast" (capped at HIGH, because only the CURRENT level may block).
        cost_levels = dict(levels)
        for z, lv in hz.get("forecast30_levels", {}).items():
            if LEVEL_RANK[lv] > LEVEL_RANK[cost_levels.get(z, "LOW")]:
                cost_levels[z] = "HIGH" if lv == "CRITICAL" else lv

        exit_status = self.planner.exit_status(levels, blocked_exits, blocked_zones)
        routes = {}
        if active:
            routes = self.planner.plan(occ, levels, cost_levels=cost_levels,
                                       blocked_zones=blocked_zones,
                                       blocked_exits=blocked_exits)
        changed = []
        for z, r in routes.items():
            sig = (r.status, r.exit_id, tuple(r.path))
            if self._sigs.get(z) != sig:
                if z in self._sigs or not self._first:
                    changed.append(z)
                self._sigs[z] = sig
        for z in list(self._sigs):
            if z not in routes:
                del self._sigs[z]
        if routes:
            if self._first:
                self._first = False
                self.emit_event("EVACUATION",
                                f"Initial A* evacuation plan computed for "
                                f"{len(routes)} occupied zone(s)")
                self.recalc_count += 1
            elif changed:
                self.recalc_count += 1
                self.emit_event("EVACUATION",
                                "Routes recalculated for " + ", ".join(changed[:5])
                                + (" ..." if len(changed) > 5 else "")
                                + " (hazard map changed)")
        trapped = [z for z, r in routes.items() if r.status != SAFE]
        out = {
            "active": active, "routes": {z: r.to_dict() for z, r in routes.items()},
            "exit_status": exit_status,
            "blocked_zones": sorted(blocked_zones
                                    | {z for z, l in levels.items() if l == "CRITICAL"}),
            "restricted_transit": sorted(z for z, l in levels.items()
                                         if LEVEL_RANK[l] >= LEVEL_RANK["HIGH"]),
            "blocked_exits": sorted(blocked_exits),
            "no_route_zones": trapped, "recalc_count": self.recalc_count,
            "changed_zones": changed, "t": t,
        }
        for rcv in ("rescue", "master"):
            bus.publish(t, self.key, rcv, "evacuation_plan", out,
                        f"evacuation_plan: {len(routes)} routes, {len(trapped)} without safe route")
        if not active:
            self._set("STANDBY", "hazard_map: inactive", "All zones LOW - no evacuation needed",
                      "Idle", "evacuation_plan(empty)", t)
        else:
            n_ok = len(routes) - len(trapped)
            avail = [e for e, s in exit_status.items() if s == "AVAILABLE"]
            self._set("PLANNING" if not trapped else "NO-ROUTE ALERT",
                      f"hazard_map + occupancy ({sum(occ.values())} people in "
                      f"{len(routes)} zones)",
                      f"{len(avail)}/{len(exit_status)} exits usable; "
                      f"{len(out['blocked_zones'])} zones blocked",
                      f"{n_ok} safe routes; {len(trapped)} zone(s) with NO safe route"
                      + (f" ({', '.join(trapped)})" if trapped else ""),
                      "evacuation_plan -> Rescue, Master", t)
        return out


# --------------------------------------------------------------------------
# Agent 4 - Rescue coordination
# --------------------------------------------------------------------------

_RISK_W = {"LOW": 0.2, "MODERATE": 1.0, "HIGH": 2.5, "CRITICAL": 5.0}
_STATUS_W = {"INCAPACITATED": 3.0, "TRAPPED": 2.0, "EXPOSED": 1.5,
             "WORKING": 1.0, "EVACUATING": 0.6}


class RescueCoordinationAgent(BaseAgent):
    key = "rescue"
    name = "Rescue Coordination Agent"
    role = "Ranks zones by occupants x risk and decides standard vs specialist response"

    def __init__(self, facility) -> None:
        super().__init__()
        self.f = facility
        self.planner = EvacuationPlanner(facility)
        self._last_top: tuple | None = None

    def _staging_exit(self, levels, blocked_exits, origin_xy) -> str | None:
        best, best_key = None, None
        for eid, ex in self.f.exits.items():
            if eid in blocked_exits:
                continue
            rank = LEVEL_RANK[levels.get(ex.zone_id, "LOW")]
            d = 0.0 if origin_xy is None else ((ex.x - origin_xy[0]) ** 2 + (ex.y - origin_xy[1]) ** 2) ** 0.5
            key = (rank, -d)
            if best_key is None or key < best_key:
                best, best_key = eid, key
        return best

    def step(self, bus: MessageBus, ctx: dict) -> dict:
        t = ctx["t"]
        self._read(bus)
        hz = self._mem.get("hazard_map")
        ev = self._mem.get("evacuation_plan")
        site = self._mem.get("site_state")
        rep = self._mem.get("incident_report")
        if not hz or not site:
            return {}
        active = bool(hz.get("active"))
        levels = hz["zone_levels"]
        fc = hz.get("forecast30_levels", levels)
        workers = [w for w in site["workers"] if w["status"] != "EVACUATED"]
        blocked_exits = set(site.get("blocked_exits", []))
        blocked_zones = set(site.get("blocked_zones", []))
        no_route = set((ev or {}).get("no_route_zones", []))

        ranking = []
        if active:
            self._activate_for(rep["incident_id"] if rep else 1, t,
                               "casualty / occupancy data received")
            by_zone: dict[str, list] = {}
            for w in workers:
                by_zone.setdefault(w["zone"], []).append(w)
            for z, ws in by_zone.items():
                lvl = levels.get(z, "LOW")
                score = 0.0
                n_inc = n_trap = n_exp = 0
                for w in ws:
                    st = w["status"]
                    if st == "INCAPACITATED":
                        n_inc += 1
                    elif st == "TRAPPED" or z in no_route:
                        n_trap += 1
                        st = "TRAPPED"
                    elif w.get("dose", 0.0) >= DOSE_EXPOSED:
                        n_exp += 1
                        st = "EXPOSED"
                    score += _STATUS_W.get(st, 1.0) * _RISK_W[lvl]
                if LEVEL_RANK[fc.get(z, lvl)] > LEVEL_RANK[lvl]:
                    score *= 1.3
                ranking.append({"zone": z, "occupants": len(ws), "level": lvl,
                                "forecast30": fc.get(z, lvl), "incapacitated": n_inc,
                                "trapped": n_trap, "exposed": n_exp,
                                "score": round(score, 2)})
            ranking.sort(key=lambda r: (-r["score"], r["zone"]))
        origin = hz.get("incident_point")
        staging = self._staging_exit(levels, blocked_exits, origin) if active else None
        std_left, spec_left = RESCUE_STANDARD_TEAMS, RESCUE_SPECIALIST_TEAMS
        specialist_zones, standard_zones, immediate = [], [], []
        for i, r in enumerate(ranking, 1):
            r["rank"] = i
            lvl = r["level"]
            need_help = r["incapacitated"] + r["trapped"] > 0
            unsafe = LEVEL_RANK[lvl] >= LEVEL_RANK["HIGH"]
            r["route"] = []
            if unsafe:
                r["team"] = "SPECIALIST"
                r["action"] = ("Specialist team with SCBA / HAZMAT gear only - zone is "
                               f"{lvl}; standard responders must NOT enter")
                specialist_zones.append(r["zone"])
            elif need_help:
                path = (self.planner.approach_path(staging, r["zone"], levels,
                                                   blocked_zones=blocked_zones,
                                                   blocked_exits=blocked_exits)
                        if staging else None)
                if path:
                    r["team"] = "STANDARD"
                    r["route"] = path
                    r["action"] = "Standard rescue team via safe approach " + path_text(path)
                    standard_zones.append(r["zone"])
                else:
                    r["team"] = "SPECIALIST"
                    r["action"] = "No safe approach for a standard team - specialist required"
                    specialist_zones.append(r["zone"])
            else:
                r["team"] = "NONE"
                r["action"] = "Self-evacuation in progress - monitor"
            if r["team"] != "NONE" and (r["incapacitated"] > 0 or unsafe or r["score"] >= 8):
                r["priority"] = "IMMEDIATE"
                immediate.append(r["zone"])
            elif r["team"] != "NONE":
                r["priority"] = "HIGH"
            else:
                r["priority"] = "MONITOR"
            # team assignment
            if r["team"] == "SPECIALIST":
                if spec_left > 0:
                    spec_left -= 1
                    r["assigned"] = f"Specialist team S{RESCUE_SPECIALIST_TEAMS - spec_left}"
                else:
                    r["assigned"] = "QUEUED - awaiting specialist team"
            elif r["team"] == "STANDARD":
                if std_left > 0:
                    std_left -= 1
                    r["assigned"] = f"Standard team R{RESCUE_STANDARD_TEAMS - std_left}"
                else:
                    r["assigned"] = "QUEUED - awaiting standard team"
            else:
                r["assigned"] = "-"
        out = {
            "active": active, "ranking": ranking, "staging_exit": staging,
            "immediate_attention": immediate, "specialist_zones": specialist_zones,
            "standard_zones": standard_zones,
            "priority_count": sum(1 for r in ranking if r["team"] != "NONE"),
            "t": t,
        }
        bus.publish(t, self.key, "master", "rescue_plan", out,
                    f"rescue_plan: {out['priority_count']} priority zone(s), "
                    f"{len(specialist_zones)} need specialist")
        top_sig = tuple((r["zone"], r["team"]) for r in ranking[:3])
        if active and top_sig != self._last_top and ranking:
            top = ranking[0]
            self.emit_event("RESCUE",
                            f"Rescue priority #1: {top['zone']} ({top['occupants']} person(s), "
                            f"{top['level']}) - {top['team']} response")
            self._last_top = top_sig
        if not active:
            self._set("STANDBY", "hazard_map: inactive", "No casualties / hazards",
                      "Idle", "rescue_plan(empty)", t)
        elif not ranking:
            self._set("MONITORING", "hazard_map + occupancy", "No people remain on site",
                      "No rescue needed", "rescue_plan(empty) -> Master", t)
        else:
            top = ranking[0]
            self._set("PRIORITISING",
                      f"hazard_map + evacuation_plan + {len(workers)} people still on site",
                      f"Top zone {top['zone']} score {top['score']} "
                      f"({top['incapacitated']} incapacitated, {top['trapped']} trapped)",
                      f"{len(specialist_zones)} specialist target(s), "
                      f"{len(standard_zones)} standard target(s)",
                      "rescue_plan -> Master", t)
        return out