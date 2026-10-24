"""Agent 5 - Master Coordination Agent.

The master receives the output of the four specialist agents, applies explicit
decision rules and produces

* the overall emergency status (NORMAL / ADVISORY / ALERT / EMERGENCY / CRITICAL),
* conflict resolutions between the agents,
* the final evacuation routes that the simulated workers follow,
* a list of coordinated (simulated) response actions and alerts.

Nothing here is a canned message: every decision is computed from the actual
reports of the other agents in the same tick.

Decision rules
--------------
Status
  R1  no confirmed incident                     -> NORMAL (ADVISORY if a sensor is confirmed abnormal)
  R2  incident confirmed                        -> at least ALERT
  R3  HIGH zone with people, OR a zone forecast CRITICAL within 30 s with
      people, OR detection severity HIGH/CRITICAL   -> at least EMERGENCY
  R4  CRITICAL zone with people, OR a worker is trapped / incapacitated /
      has no safe route                         -> CRITICAL
Conflicts
  C1  a selected route arrives in a zone after the predictor says it turns
      HIGH -> ask the Evacuation Agent to re-plan with that zone blocked;
      adopt the new route if it is safe in time, else keep the old one flagged
      TIME-CRITICAL
  C2  too many people assigned to one exit (> EXIT_CAPACITY) -> move groups to
      the next-best exit when it costs at most EXIT_REBALANCE_COST_FACTOR more
  C3  detection severity lower than the predicted worst zone -> precaution:
      the more severe assessment wins
  C4  a sensor failed in / next to an elevated zone -> assume worst case and
      block that zone for transit
"""
from __future__ import annotations

import copy

from agents import BaseAgent, MessageBus
from config import (EXIT_CAPACITY, EXIT_REBALANCE_COST_FACTOR, LEVEL_RANK,
                    RESPONSE_DELAY_S, STATUS_RANK, TIME_CONFLICT_MARGIN_S,
                    WORKER_SPEED)
from evacuation import SAFE, path_text
from factory import dist

_DET_RANK = {"NONE": -1, "LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


class MasterCoordinationAgent(BaseAgent):
    key = "master"
    name = "Master Coordination Agent"
    role = "Fuses all agent outputs, resolves conflicts, sets status, triggers simulated actions"

    def __init__(self, facility) -> None:
        super().__init__()
        self.f = facility
        self.pending_replan = False
        self.status = "NORMAL"
        self.peak_status = "NORMAL"
        self.order = self._empty_order()
        self.actions: dict[str, dict] = {}
        self._action_sigs: dict[str, tuple] = {}
        self._conflict_sigs: set = set()
        self._last_status = "NORMAL"
        self._p1: dict = {}
        self._conflict_zones: set = set()
        self.history: list[dict] = []
        self.rules_log: list[str] = []
        self.last_output_payload: dict = {}
        self.route_overrides_total = 0

    @staticmethod
    def _empty_order() -> dict:
        return {"active": False, "issued_t": None, "start_t": None,
                "site_wide": False, "scope": []}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _route_conflicts(self, route: dict, hz: dict, latency: float) -> list:
        """Zones on the route that turn HIGH before the walker has left them."""
        if route.get("status") != SAFE or not route.get("path"):
            return []
        path = route["path"]
        pts = [self.f.zones[path[0]].center] + [tuple(p) for p in route["waypoints"]]
        cum = [0.0]
        for a, b in zip(pts[:-1], pts[1:]):
            cum.append(cum[-1] + dist(a, b))
        t_high = hz.get("time_to_high", {})
        bad = []
        for k in range(1, len(path) - 1):          # skip start zone and exit node
            z = path[k]
            th = t_high.get(z)
            if th is None or th <= 0.0:
                continue
            leave_t = latency + cum[min(k + 1, len(cum) - 1)] / WORKER_SPEED
            if leave_t + TIME_CONFLICT_MARGIN_S >= th:
                bad.append(z)
        return bad

    def _standing_blocked(self, rep: dict, hz: dict) -> tuple[set, list]:
        """C4: zones whose sensor failed while the area is (or may become) hot."""
        zones, notes = set(), []
        lv = hz.get("zone_levels", {})
        f30 = hz.get("forecast30_levels", lv)
        for sid in rep.get("faulty_sensors", []):
            z = self.f.sensor_by_id[sid].zone_id
            near = [z] + self.f.neighbours(z)
            hot = any(LEVEL_RANK[lv.get(n, "LOW")] >= LEVEL_RANK["HIGH"] or
                      LEVEL_RANK[f30.get(n, "LOW")] >= LEVEL_RANK["MODERATE"] or
                      n == rep.get("origin_zone") for n in near)
            if hot:
                zones.add(z)
                notes.append({"rule": "C4", "zones": [z],
                              "text": f"Sensor {sid} in {z} FAILED inside an elevated-risk area",
                              "resolution": f"Worst case assumed: {z} blocked for transit"})
        return zones, notes

    def _rebalance_exits(self, routes: dict, hz: dict, latency: float) -> tuple[dict, list]:
        """C2: avoid piling everybody onto one exit."""
        notes = []
        load: dict[str, int] = {}
        for z, r in routes.items():
            if r["status"] == SAFE:
                load[r["exit_id"]] = load.get(r["exit_id"], 0) + r["occupants"]
        for ex in sorted(load, key=lambda e: -load[e]):
            if load[ex] <= EXIT_CAPACITY:
                continue
            movable = sorted((z for z, r in routes.items()
                              if r["status"] == SAFE and r["exit_id"] == ex),
                             key=lambda z: -routes[z]["length_m"])
            for z in movable:
                if load[ex] <= EXIT_CAPACITY:
                    break
                r = routes[z]
                for alt in r["alternatives"]:
                    if (load.get(alt["exit_id"], 0) + r["occupants"] <= EXIT_CAPACITY
                            and alt["cost"] <= r["cost"] * EXIT_REBALANCE_COST_FACTOR):
                        cand = copy.deepcopy(r)
                        cand.update({
                            "exit_id": alt["exit_id"], "path": alt["path"],
                            "waypoints": alt["waypoints"], "cost": alt["cost"],
                            "length_m": alt["length_m"],
                            "eta_s": alt["length_m"] / WORKER_SPEED,
                            "passes_through": [p for p in alt["path"] if p in self.f.zones],
                            "alternatives": [a for a in r["alternatives"] if a is not alt],
                        })
                        if self._route_conflicts(cand, hz, latency):
                            continue
                        cand["override"] = (f"C2: exit {ex} over capacity "
                                            f"({load[ex]}>{EXIT_CAPACITY}) - rerouted to {alt['exit_id']}")
                        load[ex] -= r["occupants"]
                        load[alt["exit_id"]] = load.get(alt["exit_id"], 0) + r["occupants"]
                        routes[z] = cand
                        notes.append({"rule": "C2", "zones": [z],
                                      "text": f"{ex} would receive {load[ex] + r['occupants']} people "
                                              f"(capacity {EXIT_CAPACITY})",
                                      "resolution": f"{z} ({r['occupants']} people) moved to {alt['exit_id']}"})
                        break
        return routes, notes

    # ------------------------------------------------------------------
    def step(self, bus: MessageBus, ctx: dict) -> dict:
        t = ctx["t"]
        self._read(bus)
        rep = self._mem.get("incident_report")
        hz = self._mem.get("hazard_map")
        ev = self._mem.get("evacuation_plan")
        rs = self._mem.get("rescue_plan")
        site = self._mem.get("site_state")
        if not (rep and hz and site):
            return {}
        pass_no = ctx.get("pass", 1)
        standing, standing_notes = (self._standing_blocked(rep, hz)
                                    if rep["confirmed"] else (set(), []))
        latency = max(0.0, (self.order["start_t"] or t) - t) if self.order["active"] else RESPONSE_DELAY_S

        if pass_no == 1:
            routes = copy.deepcopy((ev or {}).get("routes", {}))
            self._p1 = {"routes": copy.deepcopy(routes)}
            conflict_zones = set()
            if rep["confirmed"] and hz.get("active"):
                for z, r in routes.items():
                    conflict_zones.update(self._route_conflicts(r, hz, latency))
            self._conflict_zones = conflict_zones
            if conflict_zones:
                bus.publish(t, self.key, "evacuation", "directive",
                            {"extra_blocked_zones": sorted(standing | conflict_zones),
                             "reason": "C1 time-aware conflict"},
                            f"directive: re-plan with {', '.join(sorted(conflict_zones))} blocked (C1)")
                self.pending_replan = True
                self._set("RESOLVING CONFLICT",
                          "reports from Detection, Prediction, Evacuation, Rescue",
                          f"C1: routes reach {', '.join(sorted(conflict_zones))} after the "
                          "forecast says it turns HIGH",
                          "Requested re-plan from Evacuation Agent",
                          "directive -> Evacuation", t)
                return {}
            return self._finalize(bus, t, rep, hz, ev, rs, site, routes, standing,
                                  standing_notes, [], latency)

        # ---- second pass: a re-plan was requested -----------------------
        self.pending_replan = False
        base = self._p1.get("routes", {})
        new = copy.deepcopy((ev or {}).get("routes", {}))
        routes, c1_notes = {}, []
        for z, r in base.items():
            if z in self._conflict_zones or self._route_conflicts(r, hz, latency):
                alt = new.get(z)
                if alt and alt["status"] == SAFE and not self._route_conflicts(alt, hz, latency):
                    alt["override"] = ("C1: re-planned - old route would arrive after the "
                                       "forecast HIGH zone")
                    routes[z] = alt
                    c1_notes.append({"rule": "C1", "zones": [z],
                                     "text": f"Route of {z} ({r['exit_id']}) arrives too late",
                                     "resolution": f"Re-planned via {alt['exit_id']}: "
                                                   f"{path_text(alt['path'])}"})
                else:
                    r["time_critical"] = True
                    r["override"] = "C1: no safer alternative - leave immediately (TIME-CRITICAL)"
                    routes[z] = r
                    c1_notes.append({"rule": "C1", "zones": [z],
                                     "text": f"Route of {z} is predicted to be cut off",
                                     "resolution": "No better route exists - kept and flagged TIME-CRITICAL"})
            else:
                routes[z] = r
        return self._finalize(bus, t, rep, hz, ev, rs, site, routes, standing,
                              standing_notes, c1_notes, latency)

    # ------------------------------------------------------------------
    def _finalize(self, bus, t, rep, hz, ev, rs, site, routes, standing,
                  standing_notes, c1_notes, latency) -> dict:
        levels = hz["zone_levels"]
        f30 = hz.get("forecast30_levels", levels)
        workers = [w for w in site["workers"]]
        alive = [w for w in workers if w["status"] != "EVACUATED"]
        occ: dict[str, int] = {}
        for w in alive:
            occ[w["zone"]] = occ.get(w["zone"], 0) + 1
        n_total = len(workers)
        n_evac = sum(1 for w in workers if w["status"] == "EVACUATED")
        n_trapped = sum(1 for w in alive if w["status"] == "TRAPPED")
        n_incap = sum(1 for w in alive if w["status"] == "INCAPACITATED")
        no_route = [z for z, r in routes.items() if r["status"] != SAFE]
        inc = bool(rep["confirmed"])
        sev = rep.get("severity_label", "NONE")
        rules: list[str] = []

        # ---- C2 exit balancing -----------------------------------------
        routes, c2_notes = (self._rebalance_exits(routes, hz, latency)
                            if inc else (routes, []))

        # ---- status rules ------------------------------------------------
        if not inc:
            any_conf = any(s["confirmed"] >= 1 for s in rep["sensor_states"].values())
            status = "ADVISORY" if any_conf else "NORMAL"
            rules.append("R1: no confirmed incident -> " + status)
        else:
            status = "ALERT"
            rules.append("R2: incident confirmed -> ALERT")
            high_occ = [z for z, l in levels.items()
                        if LEVEL_RANK[l] >= LEVEL_RANK["HIGH"] and occ.get(z, 0) > 0]
            fc_crit = [z for z, l in f30.items() if l == "CRITICAL" and occ.get(z, 0) > 0]
            if high_occ or fc_crit or sev in ("HIGH", "CRITICAL"):
                status = "EMERGENCY"
                why = []
                if high_occ:
                    why.append(f"HIGH zone(s) occupied: {', '.join(high_occ[:3])}")
                if fc_crit:
                    why.append(f"forecast CRITICAL <=30s: {', '.join(fc_crit[:3])}")
                if sev in ("HIGH", "CRITICAL"):
                    why.append(f"detection severity {sev}")
                rules.append("R3: " + "; ".join(why) + " -> EMERGENCY")
            crit_occ = [z for z, l in levels.items() if l == "CRITICAL" and occ.get(z, 0) > 0]
            if crit_occ or n_trapped or n_incap or no_route:
                status = "CRITICAL"
                why = []
                if crit_occ:
                    why.append(f"CRITICAL zone(s) occupied: {', '.join(crit_occ[:3])}")
                if n_trapped or n_incap:
                    why.append(f"{n_trapped} trapped, {n_incap} incapacitated")
                if no_route:
                    why.append(f"no safe route from {', '.join(no_route[:3])}")
                rules.append("R4: " + "; ".join(why) + " -> CRITICAL")
            if all(occ.get(z, 0) == 0 for z in levels) and status == "ALERT":
                rules.append("R5: hazard persists but site is empty -> stay ALERT")

        # ---- C3 detection vs prediction ----------------------------------
        notes = list(standing_notes) + c1_notes + c2_notes
        pred_peak = max((LEVEL_RANK[l] for l in levels.values()), default=0)
        if inc and pred_peak >= _DET_RANK.get(sev, 0) + 2:
            notes.append({"rule": "C3", "zones": [],
                          "text": f"Detection severity {sev} but prediction shows a "
                                  f"{[k for k, v in LEVEL_RANK.items() if v == pred_peak][0]} zone",
                          "resolution": "Precautionary principle: the more severe assessment is used"})

        self.status = status
        if STATUS_RANK[status] > STATUS_RANK[self.peak_status]:
            self.peak_status = status
        if status != self._last_status:
            self.emit_event("STATUS", f"Overall status {self._last_status} -> {status}")
            self._last_status = status
        self.rules_log = rules

        # ---- evacuation order --------------------------------------------
        order = self.order
        if inc and STATUS_RANK[status] >= STATUS_RANK["ALERT"]:
            if not order["active"]:
                order.update(active=True, issued_t=t, start_t=t + RESPONSE_DELAY_S)
                self.emit_event("EVACUATION", f"Evacuation order issued (workers start moving "
                                f"in {RESPONSE_DELAY_S:.0f}s)")
            scope = set(order["scope"])
            scope |= {z for z, l in levels.items() if LEVEL_RANK[l] >= 1}
            scope |= {z for z, l in f30.items() if LEVEL_RANK[l] >= 1}
            scope.add(rep["origin_zone"])
            order["scope"] = sorted(scope)
            if STATUS_RANK[status] >= STATUS_RANK["EMERGENCY"] and not order["site_wide"]:
                order["site_wide"] = True
                self.emit_event("EVACUATION", "Order widened to SITE-WIDE evacuation")
        elif not inc and order["active"]:
            self.order = order = self._empty_order()

        # ---- final routes with scope --------------------------------------
        final_routes = {}
        for z, r in routes.items():
            r = dict(r)
            r["in_scope"] = bool(order["active"] and (order["site_wide"]
                                                       or z in order["scope"]))
            final_routes[z] = r
        self.route_overrides_total += len(c1_notes) + len(c2_notes)

        # ---- simulated actions ----------------------------------------------
        actions = self._build_actions(t, inc, status, rep, hz, final_routes, rs,
                                      standing, order)
        self._log_actions(actions)

        # ---- conflict log ----------------------------------------------------
        for n in notes:
            sig = (n["rule"], tuple(n["zones"]), n["resolution"])
            if sig not in self._conflict_sigs:
                self._conflict_sigs.add(sig)
                self.emit_event("CONFLICT", f"{n['rule']}: {n['text']} -> {n['resolution']}")

        alerts = self._build_alerts(status, rep, hz, final_routes, rs, order)
        elevated = [z for z, l in levels.items() if LEVEL_RANK[l] >= 1]
        metrics = {
            "active_incidents": 1 if inc else 0,
            "affected_zones": len(elevated),
            "occupants_threatened": sum(occ.get(z, 0) for z in elevated),
            "active_sensors": sum(1 for s in rep["sensor_states"].values()
                                  if s["status"] != "FAULT"),
            "total_sensors": len(rep["sensor_states"]),
            "rescue_priority_count": (rs or {}).get("priority_count", 0),
            "workers_total": n_total, "workers_evacuated": n_evac,
            "workers_trapped": n_trapped, "workers_incapacitated": n_incap,
            "workers_on_site": len(alive),
        }
        if not order["active"]:
            evac_status = "NOT REQUIRED"
        elif t < (order["start_t"] or 0):
            evac_status = f"ORDER ISSUED - moving in {order['start_t'] - t:.0f}s"
        elif len(alive) == 0:
            evac_status = "COMPLETE - site empty"
        else:
            evac_status = f"IN PROGRESS - {n_evac}/{n_total} out, {len(alive)} on site"
        metrics["evac_status"] = evac_status

        payload = {
            "status": status, "peak_status": self.peak_status, "rules": rules,
            "conflicts": notes, "final_routes": final_routes,
            "actions": actions, "alerts": alerts, "evac_order": copy.deepcopy(order),
            "restricted_zones": sorted({z for z, l in levels.items()
                                        if LEVEL_RANK[l] >= LEVEL_RANK["HIGH"]
                                        or f30.get(z) == "CRITICAL"} | standing),
            "metrics": metrics, "t": t,
        }
        self.last_output_payload = payload
        bus.publish(t, self.key, "simulation", "coordination", payload,
                    f"coordination: status {status}, {len(final_routes)} route(s), "
                    f"{len(actions)} action(s)")
        for rcv in ("evacuation", "rescue", "prediction", "detection"):
            bus.publish(t, self.key, rcv, "status_update", {"status": status},
                        f"status_update: {status}")
        bus.publish(t, self.key, "evacuation", "directive",
                    {"extra_blocked_zones": sorted(standing), "reason": "standing (C4)"},
                    "directive: standing blocked zones " + (", ".join(sorted(standing)) or "none"))
        if inc:
            self._activate_for(rep["incident_id"], t, "coordinating all agents")
        self._set(
            status,
            f"incident_report + hazard_map + evacuation_plan + rescue_plan "
            f"({len(alive)} people on site)",
            "; ".join(rules[-2:]) if rules else "-",
            f"{len(final_routes)} routes released, {len(notes)} conflict(s) resolved, "
            f"{len(actions)} simulated action(s)",
            "coordination -> Simulation; directive/status_update -> agents", t)
        return payload

    # ------------------------------------------------------------------
    def _build_actions(self, t, inc, status, rep, hz, routes, rs, standing, order) -> list:
        acts: list[dict] = []
        if not inc:
            return acts
        levels = hz["zone_levels"]
        scope = (list(self.f.zones) if order["site_wide"] else order["scope"])
        acts.append({"id": "alarm", "type": "ALARM_ACTIVATION",
                     "label": "Emergency alarm activated (SIMULATED)",
                     "detail": ("SITE-WIDE" if order["site_wide"] else
                                f"{len(scope)} zone(s): {', '.join(scope[:6])}"),
                     "sig": ("site" if order["site_wide"] else tuple(scope))})
        notif = []
        for z, r in routes.items():
            if not r.get("in_scope"):
                continue
            if r["status"] == SAFE:
                notif.append(f"{z}: go to {r['exit_id']} via {path_text(r['path'])}"
                             + (" [TIME-CRITICAL]" if r.get("time_critical") else ""))
            else:
                notif.append(f"{z}: NO SAFE ROUTE - shelter in place, rescue requested")
        if notif:
            acts.append({"id": "notify", "type": "EVACUATION_NOTIFICATION",
                         "label": f"Evacuation notices sent to {len(notif)} zone(s) "
                                  "(SIMULATED SMS / PA)",
                         "detail": " | ".join(notif[:8]) + (" ..." if len(notif) > 8 else ""),
                         "sig": tuple(notif)})
        restricted = sorted({z for z, l in levels.items() if LEVEL_RANK[l] >= LEVEL_RANK["HIGH"]}
                            | standing | {z for z, l in hz.get("forecast30_levels", {}).items()
                                          if l == "CRITICAL"})
        if restricted:
            acts.append({"id": "restrict", "type": "RESTRICTED_ZONES",
                         "label": f"{len(restricted)} zone(s) marked RESTRICTED (SIMULATED)",
                         "detail": ", ".join(restricted), "sig": tuple(restricted)})
        power = sorted(z for z, l in levels.items()
                       if LEVEL_RANK[l] >= 1 and z != "CTRL")
        if power:
            feeder = " + MAIN FEEDER in ELEC" if "ELEC" in power else ""
            acts.append({"id": "power", "type": "POWER_ISOLATION",
                         "label": "Power isolation RECOMMENDED (not executed - simulation)",
                         "detail": "Isolate non-intrinsically-safe supplies in: "
                                   + ", ".join(power) + feeder + ". Keep CTRL on UPS.",
                         "sig": tuple(power)})
        if rs and rs.get("ranking"):
            lines = [f"#{r['rank']} {r['zone']}: {r['assigned']} ({r['team']})"
                     for r in rs["ranking"] if r["team"] != "NONE"]
            if lines:
                acts.append({"id": "dispatch", "type": "RESCUE_DISPATCH",
                             "label": "Rescue team dispatch RECOMMENDED (SIMULATED)",
                             "detail": f"staging at {rs.get('staging_exit')}: " + " | ".join(lines[:6]),
                             "sig": tuple(lines)})
        return acts

    def _log_actions(self, actions: list) -> None:
        cur = {a["id"]: a for a in actions}
        for aid, a in cur.items():
            prev = self._action_sigs.get(aid)
            if prev != a["sig"]:
                verb = "issued" if prev is None else "updated"
                self.emit_event("ACTION", f"{a['label']} - {verb}: {a['detail'][:140]}")
                self._action_sigs[aid] = a["sig"]
        for aid in list(self._action_sigs):
            if aid not in cur:
                del self._action_sigs[aid]
        self.actions = cur

    def _build_alerts(self, status, rep, hz, routes, rs, order) -> list:
        alerts = []
        if not rep["confirmed"]:
            if status == "ADVISORY":
                alerts.append({"level": "ADVISORY",
                               "text": "A sensor reports an abnormal reading - verifying."})
            return alerts
        zname = self.f.zones[rep["origin_zone"]].name
        kind = "Fire / smoke" if rep["type"] == "fire_smoke" else "Gas release"
        alerts.append({"level": status,
                       "text": f"{kind} detected near {zname}. Status {status}. "
                               f"Severity {rep['severity_label']}."})
        no_safe = [z for z, r in routes.items() if r["status"] != SAFE]
        if no_safe:
            alerts.append({"level": "CRITICAL",
                           "text": "NO SAFE ROUTE from: " + ", ".join(no_safe)
                                   + ". Shelter in place; specialist rescue required."})
        if rs and rs.get("specialist_zones"):
            alerts.append({"level": "EMERGENCY",
                           "text": "Specialist (SCBA/HAZMAT) rescue needed in: "
                                   + ", ".join(rs["specialist_zones"][:5])})
        if hz.get("forecast_threatened"):
            alerts.append({"level": "ALERT",
                           "text": "Plume forecast to reach: "
                                   + ", ".join(hz["forecast_threatened"][:6])
                                   + f" within {hz['est'].get('horizon_s', 60)}s"})
        if order["active"]:
            alerts.append({"level": "ALERT",
                           "text": ("SITE-WIDE" if order["site_wide"] else "Zone-level")
                                   + " evacuation ordered."})
        return alerts