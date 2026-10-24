"""ResQM - Multi-Agent Industrial Disaster Response System (Streamlit dashboard).

Run with:  streamlit run app.py

Everything shown is SIMULATED.  See README.md for the assumptions.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from config import (DEMO, DISCLAIMER, EVAC_PRESET_ORDER, EVAC_PRESETS,
                    HAZARD_TYPES, LEVEL_RANK, RISK_COLORS, SCENARIO_ORDER,
                    SCENARIOS, STATUS_COLORS, STATUS_RANK)
from evacuation import path_text
from factory import get_facility
from hazard import compass_label
from simulation import SimConfig, Simulation
from viz import (facility_figure, sensor_figure, zone_hazard_figure)

REFRESH_S = 0.45
FAC = get_facility()
ZONE_IDS = list(FAC.zones)
SPEEDS = [1, 2, 5, 10]

st.set_page_config(page_title="ResQM - Emergency Operations Centre",
                   page_icon="🚨", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: #0b1220; color: #d6e0f5; }
[data-testid="stSidebar"] { background: #0e1730; border-right: 1px solid #1f2c47; }
h1, h2, h3, h4 { color: #e6eeff; letter-spacing: .2px; }
[data-testid="stMetric"] { background: #111a2e; border: 1px solid #1f2c47;
    border-left: 4px solid #3d6bff; border-radius: 6px; padding: 8px 12px; }
[data-testid="stMetricLabel"] p { color: #8fa3c7; font-size: .78rem; text-transform: uppercase;
    letter-spacing: .6px; }
.resqm-banner { border-radius: 6px; padding: 10px 16px; margin: 4px 0 10px 0;
    font-weight: 700; letter-spacing: .5px; border: 1px solid #1f2c47; }
.resqm-alert { border-radius: 6px; padding: 7px 12px; margin: 3px 0; font-size: .92rem;
    background: #131d33; border-left: 5px solid #f1c40f; }
.resqm-note { color: #8fa3c7; font-size: .8rem; }
.resqm-chip { display:inline-block; padding: 3px 10px; margin: 2px 4px 2px 0; border-radius: 12px;
    font-size: .78rem; font-weight: 600; border: 1px solid #2a3a5e; background:#111a2e; }
.resqm-card { background:#111a2e; border:1px solid #1f2c47; border-radius:6px;
    padding:8px 12px; margin-bottom:6px; font-size:.86rem; }
.resqm-card b { color:#9fc0ff; }
div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# small compatibility helpers (Streamlit changed width arguments over time)
# ---------------------------------------------------------------------------
def show_plot(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df: pd.DataFrame, height: int | None = None) -> None:
    kw = {} if height is None else {"height": height}
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


def zname(z: str) -> str:
    return f"{z} - {FAC.zones[z].name}" if z in FAC.zones else z


# ---------------------------------------------------------------------------
# session state / controls
# ---------------------------------------------------------------------------
def _scenario_defaults(key: str) -> None:
    sc = SCENARIOS[key]
    ss = st.session_state
    ss["source_zone"] = sc.source_zone
    ss["intensity"] = sc.default_intensity
    ss["wind_from"] = int(sc.default_wind_from)
    ss["wind_speed"] = float(sc.default_wind_speed)
    ss["custom_hazard"] = sc.hazard_type


def cfg_from_widgets() -> SimConfig:
    ss = st.session_state
    custom = ss["scenario"] == "custom"
    return SimConfig(
        scenario=ss["scenario"], source_zone=ss["source_zone"],
        intensity=int(ss["intensity"]), wind_from=float(ss["wind_from"]),
        wind_speed=float(ss["wind_speed"]), seed=int(ss["seed"]),
        evac_preset=ss["evac_preset"],
        hazard_type=ss["custom_hazard"] if custom else None,
        growth=ss["custom_growth"] if custom else None)


def new_sim(cfg: SimConfig, running: bool = False) -> None:
    st.session_state.sim = Simulation(cfg)
    st.session_state.running = running
    st.session_state.applied = {
        "wind": (cfg.wind_from, cfg.wind_speed), "intensity": cfg.intensity,
        "evac": cfg.evac_preset, "wind_version": 0}


def init_state() -> None:
    ss = st.session_state
    if "sim" in ss:
        return
    ss["scenario"] = "chem_storage"
    _scenario_defaults("chem_storage")
    ss["seed"] = 42
    ss["evac_preset"] = "all_open"
    ss["speed"] = 2
    ss["custom_growth"] = "ramp"
    ss["demo_mode"] = False
    ss["notice"] = ""
    new_sim(cfg_from_widgets())


# --- callbacks (run before the script re-executes) -------------------------
def cb_scenario() -> None:
    _scenario_defaults(st.session_state["scenario"])


def cb_start() -> None:
    if not st.session_state.sim.complete:
        st.session_state.running = True


def cb_pause() -> None:
    st.session_state.running = False


def cb_step() -> None:
    sim = st.session_state.sim
    st.session_state.running = False
    try:
        sim.tick()
    except Exception as exc:                       # pragma: no cover
        st.session_state.notice = f"Simulation error: {exc}"


def cb_reset() -> None:
    st.session_state.demo_mode = False
    st.session_state.notice = "Simulation reset to normal operation."
    new_sim(cfg_from_widgets())


def cb_trigger() -> None:
    sim = st.session_state.sim
    if sim.complete:
        st.session_state.notice = "Run finished - press Reset first."
        return
    if not sim.trigger_leak():
        st.session_state.notice = "The hazard is already active."
    st.session_state.running = True


def cb_demo() -> None:
    ss = st.session_state
    ss["scenario"] = DEMO["scenario"]
    ss["source_zone"] = DEMO["source_zone"]
    ss["intensity"] = DEMO["intensity"]
    ss["wind_from"] = int(DEMO["wind_from"])
    ss["wind_speed"] = float(DEMO["wind_speed"])
    ss["seed"] = DEMO["seed"]
    ss["speed"] = DEMO["speed"]
    ss["evac_preset"] = DEMO["evac_preset"]
    ss["custom_hazard"] = SCENARIOS[DEMO["scenario"]].hazard_type
    ss["demo_mode"] = True
    ss["notice"] = ("DEMO MODE started: normal operation, leak at t=20 s, "
                    "scripted wind shift at t=110 s.")
    new_sim(SimConfig.demo(), running=True)


init_state()
ss = st.session_state
sim: Simulation = ss.sim

# scripted wind shifts update the sidebar widgets (before they are drawn)
if ss.applied["wind_version"] != sim.wind_version:
    ss["wind_from"] = int(round(sim.wind_from)) % 360
    ss["wind_speed"] = float(sim.wind_speed)
    ss.applied.update(wind=(float(ss["wind_from"]), float(ss["wind_speed"])),
                      wind_version=sim.wind_version)

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚨 ResQM")
    st.caption("Multi-Agent Industrial Disaster Response System")
    st.markdown("**Run control**")
    c1, c2, c3 = st.columns(3)
    c1.button("▶ Start", on_click=cb_start, width="stretch") if False else \
        c1.button("▶ Start", on_click=cb_start)
    c2.button("⏸ Pause", on_click=cb_pause)
    c3.button("⏭ Step", on_click=cb_step)
    c4, c5 = st.columns(2)
    c4.button("🔄 Reset", on_click=cb_reset)
    c5.button("☣ Trigger leak", on_click=cb_trigger)
    st.button("🎬 DEMO MODE (one click)", on_click=cb_demo)
    st.selectbox("Simulation speed (ticks per refresh)", SPEEDS, key="speed",
                 help="One tick = 2 simulated seconds.")

    st.markdown("---\n**Scenario** (changing these resets the run)")
    st.selectbox("Disaster scenario", list(SCENARIO_ORDER), key="scenario",
                 format_func=lambda k: SCENARIOS[k].label, on_change=cb_scenario)
    st.selectbox("Incident location", ZONE_IDS, key="source_zone",
                 format_func=zname)
    if ss["scenario"] == "custom":
        st.selectbox("Hazard type", list(HAZARD_TYPES), key="custom_hazard",
                     format_func=lambda k: HAZARD_TYPES[k]["label"])
        st.selectbox("Release growth", ["ramp", "t2", "step"], key="custom_growth",
                     format_func={"ramp": "Ramps up (leak)", "t2": "Grows ~t² (fire)",
                                  "step": "Instant full rate"}.get)
    st.number_input("Random seed (reproducible runs)", min_value=0, max_value=99999,
                    step=1, key="seed")

    st.markdown("---\n**Live conditions** (apply immediately)")
    st.slider("Leak intensity (1-10)", 1, 10, key="intensity")
    st.slider("Wind direction - blowing FROM (deg)", 0, 355, step=5, key="wind_from")
    st.caption(f"Wind from {compass_label(ss['wind_from'])} "
               f"→ gas drifts toward {compass_label(ss['wind_from'] + 180)}")
    st.slider("Wind speed (m/s)", 0.0, 10.0, step=0.5, key="wind_speed")
    st.selectbox("Evacuation scenario", list(EVAC_PRESET_ORDER), key="evac_preset",
                 format_func=lambda k: EVAC_PRESETS[k]["label"])
    st.markdown("---")
    st.caption(DISCLAIMER)

# --- apply widget changes to the simulation ---------------------------------
wanted = cfg_from_widgets()
if wanted.world_signature() != sim.cfg.world_signature():
    new_sim(wanted)
    sim = ss.sim
    ss["demo_mode"] = False
    ss["notice"] = "Scenario settings changed - simulation reset."
else:
    ap = ss.applied
    if ap["wind"] != (wanted.wind_from, wanted.wind_speed):
        sim.set_wind(wanted.wind_from, wanted.wind_speed)
        ap["wind"] = (wanted.wind_from, wanted.wind_speed)
        sim.log_event("ENVIRONMENT", "Operator",
                      f"Wind set to {wanted.wind_speed:.1f} m/s from {wanted.wind_from:.0f}°")
    if ap["intensity"] != wanted.intensity:
        sim.set_intensity(wanted.intensity)
        ap["intensity"] = wanted.intensity
        sim.log_event("ENVIRONMENT", "Operator",
                      f"Leak intensity set to {wanted.intensity}/10")
    if ap["evac"] != wanted.evac_preset:
        sim.set_evac_preset(wanted.evac_preset)
        ap["evac"] = wanted.evac_preset
        sim.log_event("ENVIRONMENT", "Operator",
                      f"Evacuation scenario: {EVAC_PRESETS[wanted.evac_preset]['label']}")

# --- advance the simulation ----------------------------------------------------
if ss.running and not sim.complete:
    try:
        for _ in range(int(ss["speed"])):
            sim.tick()
            if sim.complete:
                break
    except Exception as exc:                        # keep the UI alive
        ss.running = False
        ss.notice = f"Simulation stopped by an error: {exc}"
if sim.complete:
    ss.running = False

# ---------------------------------------------------------------------------
# data used by every section
# ---------------------------------------------------------------------------
coord = sim.coord or {}
metrics = coord.get("metrics", {})
status = coord.get("status", "NORMAL")
det = sim.out.get("detection") or {}
pm = sim.out.get("prediction") or {}
evac = sim.out.get("evacuation") or {}
resc = sim.out.get("rescue") or {}
levels = pm.get("zone_levels") or {z: "LOW" for z in FAC.zones}
routes = coord.get("final_routes", {})
counts = sim.worker_counts()

# ---------------------------------------------------------------------------
# header + status banner
# ---------------------------------------------------------------------------
hc1, hc2 = st.columns([3, 2])
hc1.markdown("# 🚨 ResQM · Emergency Operations Centre")
hc1.caption("Multi-Agent Industrial Disaster Response System · SIMULATION ONLY")
hc2.markdown(
    f"<div class='resqm-note' style='text-align:right'>Simulated time "
    f"<b style='font-size:1.5rem;color:#e6eeff'>{int(sim.t // 60):02d}:{int(sim.t % 60):02d}</b>"
    f" &nbsp;·&nbsp; tick {sim.tick_idx} &nbsp;·&nbsp; "
    f"{'<span style=\"color:#2ecc71\">● RUNNING</span>' if ss.running else '<span style=\"color:#f1c40f\">● PAUSED</span>'}"
    f"{' &nbsp;·&nbsp; <span style=\"color:#80d8ff\">RUN COMPLETE</span>' if sim.complete else ''}"
    f"<br>Scenario: {sim.scenario.label} · source {sim.source_zone} · seed {sim.cfg.seed}</div>",
    unsafe_allow_html=True)

col = STATUS_COLORS[status]
reason = (coord.get("rules") or ["Facility in normal operation - all sensors nominal."])[-1]
st.markdown(
    f"<div class='resqm-banner' style='background:{col}22;border-left:8px solid {col};color:{col}'>"
    f"OVERALL STATUS: {status}"
    f"<span style='color:#d6e0f5;font-weight:400;margin-left:18px'>{reason}</span></div>",
    unsafe_allow_html=True)
if ss.notice:
    st.info(ss.notice)
for a in coord.get("alerts", [])[:4]:
    lc = STATUS_COLORS.get(a["level"], "#f1c40f")
    st.markdown(f"<div class='resqm-alert' style='border-left-color:{lc}'>"
                f"<b style='color:{lc}'>{a['level']}</b> &nbsp; {a['text']}</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# A. overview metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Emergency status", status, help=f"Peak so far: {coord.get('peak_status', 'NORMAL')}")
m2.metric("Active incidents", metrics.get("active_incidents", 0))
m3.metric("Affected zones", metrics.get("affected_zones", 0),
          help="Zones classified MODERATE or worse by the Prediction Agent")
m4.metric("Occupants in threatened areas", metrics.get("occupants_threatened", 0))
m5, m6, m7, m8 = st.columns(4)
m5.metric("Active sensors", f"{metrics.get('active_sensors', len(FAC.sensors))}/"
                            f"{metrics.get('total_sensors', len(FAC.sensors))}")
m6.metric("Evacuation status", metrics.get("evac_status", "NOT REQUIRED"))
m7.metric("Rescue priorities", metrics.get("rescue_priority_count", 0))
m8.metric("Evacuated / on site", f"{counts['EVACUATED']} / "
                                  f"{len(sim.workers) - counts['EVACUATED']}")

# demo narrator ---------------------------------------------------------------
if ss.get("demo_mode"):
    ev_text = " ".join(e["text"] for e in sim.events)
    steps = [
        ("Normal operation - sensors at baseline", sim.t >= 10),
        ("Abnormal gas readings appear", sim.t_first_abnormal is not None),
        ("Hazard Detection Agent confirms the incident", sim.t_detect is not None),
        ("Prediction Agent activated and fits the plume", bool(pm.get("active"))),
        ("Hazard spreads progressively", sum(1 for l in levels.values() if l != "LOW") >= 3),
        ("Zone risk classes change (green→yellow→orange→red)",
         "Risk map updated" in ev_text),
        ("A* evacuation routes recalculated", sim.evacuation.recalc_count >= 2),
        ("Rescue priorities ranked", bool(resc.get("ranking"))),
        ("Master agent coordinates + simulated actions", len(coord.get("actions", [])) >= 3),
        ("Final emergency response summary ready", sim.complete),
    ]
    done = sum(1 for _, ok in steps if ok)
    with st.expander(f"🎬 Demo narrator - {done}/{len(steps)} steps shown", expanded=True):
        st.progress(done / len(steps))
        st.markdown(" &nbsp; ".join(
            f"<span class='resqm-chip' style='color:{'#2ecc71' if ok else '#8fa3c7'}'>"
            f"{'✔' if ok else '○'} {i + 1}. {label}</span>"
            for i, (label, ok) in enumerate(steps)), unsafe_allow_html=True)

tab_map, tab_sens, tab_agents, tab_evac, tab_hist = st.tabs([
    "🗺 Live Facility Map", "📡 Sensor Monitoring", "🤖 Multi-Agent Control Centre",
    "🚪 Evacuation & Rescue", "🕒 Incident History"])

# ---------------------------------------------------------------------------
# B. live facility map
# ---------------------------------------------------------------------------
with tab_map:
    lc1, lc2, lc3, lc4, lc5 = st.columns([2.2, 1, 1, 1, 1])
    layer = lc1.radio("Heatmap layer", ["agents", "forecast", "truth"], horizontal=True,
                      format_func={"agents": "Agents' risk estimate",
                                   "forecast": "Forecast (+60 s)",
                                   "truth": "Ground truth (simulated)"}.get, key="layer")
    s_routes = lc2.checkbox("Routes", True, key="v_routes")
    s_workers = lc3.checkbox("Workers", True, key="v_workers")
    s_sensors = lc4.checkbox("Sensors", True, key="v_sensors")
    s_fc = lc5.checkbox("Forecast line", True, key="v_fc")
    show_plot(facility_figure(sim, layer, s_routes, s_workers, s_sensors, s_fc), "facility_map")
    leg = " ".join(f"<span class='resqm-chip' style='border-color:{c};color:{c}'>{n}</span>"
                   for n, c in (("LOW", RISK_COLORS["LOW"]), ("MODERATE", RISK_COLORS["MODERATE"]),
                                ("HIGH", RISK_COLORS["HIGH"]), ("CRITICAL", RISK_COLORS["CRITICAL"])))
    st.markdown(leg + "<span class='resqm-note'> &nbsp; Zone border = risk class · "
                "heatmap = hazard index (dimensionless, NOT a chemical concentration) · "
                "dotted red line = forecast HIGH contour · ▲ exits · ■ sensors · ★ estimated source · "
                "dashed route = TIME-CRITICAL</span>", unsafe_allow_html=True)
    if pm.get("active"):
        e = pm["est"]
        st.caption(f"Prediction Agent estimate: source in {e['source_zone']} "
                   f"(first alarm in {e['detected_zone']}), strength x{e['scale']:.1f}, "
                   f"release began ~{e['lag_ticks'] * 2:.0f}s before detection.")

# ---------------------------------------------------------------------------
# C. sensors
# ---------------------------------------------------------------------------
with tab_sens:
    st.caption("MQ135 = uncalibrated gas/air-quality INDICATOR (arbitrary units) · PM10 = particulate "
               "matter · DHT11 = temperature + humidity (reads only 0-50 °C / 20-90 %RH). "
               "All values are simulated.")
    latest = sim.latest_sensors()
    states = det.get("sensor_states", {})
    rows = []
    for r in latest.itertuples(index=False):
        s = states.get(r.sensor_id, {})
        stt = s.get("status", "NORMAL")
        icon = {"NORMAL": "🟢", "WARNING": "🟡", "ALARM": "🔴", "FAULT": "⚫"}[stt]
        viol = [ch for ch, v in s.get("levels", {}).items() if v >= 1]
        rows.append({"Sensor": r.sensor_id, "Zone": r.zone_id, "Status": f"{icon} {stt}",
                     "Health": r.health, "MQ135 (a.u.)": round(r.gas, 0) if r.gas == r.gas else None,
                     "PM10": round(r.pm10, 0) if r.pm10 == r.pm10 else None,
                     "Temp °C": round(r.temp, 1) if r.temp == r.temp else None,
                     "Hum %": round(r.hum, 0) if r.hum == r.hum else None,
                     "Anomaly (x alarm)": round(s.get("anomaly", 0.0), 2),
                     "Violations": ", ".join(viol) or "-"})
    sdf = pd.DataFrame(rows)
    n_abn = sum(1 for r in rows if not r["Status"].endswith("NORMAL"))
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Sensors reporting", sum(1 for r in rows if r["Health"] == "OK"))
    sc2.metric("Abnormal readings", n_abn)
    sc3.metric("Failed sensors", sum(1 for r in rows if r["Health"] == "FAULT"))
    sc4.metric("Transient blips ignored", sim.detection.suppressed_blips)
    show_df(sdf.sort_values("Anomaly (x alarm)", ascending=False), height=300)
    hist = sim.sensor_history()
    if not hist.empty:
        default = [r["Sensor"] for r in sorted(rows, key=lambda x: -x["Anomaly (x alarm)"])[:4]]
        chosen = st.multiselect("Sensors to plot", [s.id for s in FAC.sensors],
                                default=default, key="plot_sensors")
        if chosen:
            g1, g2 = st.columns(2)
            with g1:
                show_plot(sensor_figure(hist, chosen, "gas"), "ch_gas")
                show_plot(sensor_figure(hist, chosen, "temp"), "ch_temp")
            with g2:
                show_plot(sensor_figure(hist, chosen, "pm10"), "ch_pm")
                show_plot(sensor_figure(hist, chosen, "hum"), "ch_hum")

# ---------------------------------------------------------------------------
# D. multi-agent control centre
# ---------------------------------------------------------------------------
with tab_agents:
    flow = ["Sensor Network", "Detection", "Prediction", "Evacuation", "Rescue", "Master"]
    agent_status = {a.key: a.status for a in sim.agents}
    chips = []
    for label, key in zip(flow, [None, "detection", "prediction", "evacuation", "rescue", "master"]):
        stt = agent_status.get(key, "LIVE")
        active = stt not in ("STANDBY", "MONITORING", "IDLE", "NORMAL", "WATCHING")
        c = "#ff9800" if active else "#2ecc71"
        chips.append(f"<span class='resqm-chip' style='border-color:{c};color:{c}'>{label}"
                     f"<span class='resqm-note'> · {stt}</span></span>")
    st.markdown(" → ".join(chips), unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        for a in sim.agents:
            snap = a.snapshot()
            hot = snap["status"] not in ("STANDBY", "MONITORING", "IDLE", "NORMAL", "WATCHING")
            with st.expander(f"{'🟠' if hot else '🟢'} {snap['name']} — {snap['status']}",
                             expanded=hot):
                st.markdown(
                    f"<div class='resqm-note'>{snap['role']} · activations: {snap['activations']}</div>"
                    f"<div class='resqm-card'><b>Latest input</b><br>{snap['input']}</div>"
                    f"<div class='resqm-card'><b>Latest analysis</b><br>{snap['analysis']}</div>"
                    f"<div class='resqm-card'><b>Decision</b><br>{snap['decision']}</div>"
                    f"<div class='resqm-card'><b>Output sent</b><br>{snap['output']}</div>",
                    unsafe_allow_html=True)
    with right:
        st.markdown("**Master decision rules fired (this tick)**")
        for r in coord.get("rules", []):
            st.markdown(f"- {r}")
        conflicts = coord.get("conflicts", [])
        st.markdown(f"**Conflicts resolved ({len(conflicts)})**")
        if conflicts:
            show_df(pd.DataFrame([{"Rule": c["rule"], "Issue": c["text"],
                                   "Resolution": c["resolution"]} for c in conflicts]), 170)
        else:
            st.caption("No conflicts between agents right now.")
        acts = coord.get("actions", [])
        st.markdown(f"**Coordinated actions — SIMULATED ({len(acts)})**")
        for a in acts:
            st.markdown(f"<div class='resqm-card'><b>{a['label']}</b><br>{a['detail']}</div>",
                        unsafe_allow_html=True)
        if not acts:
            st.caption("No response actions needed.")
    st.markdown("**Agent communication log (message bus)**")
    only_sig = st.checkbox("Hide per-tick status updates", True, key="hide_status_msgs")
    rows = sim.bus.log_rows(400, None)
    if only_sig:
        rows = [r for r in rows if r["topic"] not in ("status_update", "site_state", "directive")]
    rows = rows[-40:][::-1]
    if rows:
        show_df(pd.DataFrame([{"t (s)": r["t"], "From": r["from"], "To": r["to"],
                               "Topic": r["topic"], "Content": r["summary"]} for r in rows]), 300)

# ---------------------------------------------------------------------------
# E. evacuation and rescue
# ---------------------------------------------------------------------------
with tab_evac:
    e1, e2 = st.columns([3, 2])
    with e1:
        st.markdown("**Evacuation routes (A*, hazard-weighted)**")
        if routes:
            rr = []
            for z, r in sorted(routes.items()):
                alt = ", ".join(f"{a['exit_id']} (+{a['cost'] - r['cost']:.0f})"
                                for a in r.get("alternatives", [])[:2]) or "-"
                rr.append({"Zone": z, "People": r["occupants"],
                           "Risk": levels.get(z, "LOW"),
                           "Result": ("✅ SAFE" if r["status"] == "SAFE" else "⛔ NO SAFE ROUTE")
                                     + (" ⏱" if r.get("time_critical") else ""),
                           "Exit": r.get("exit_id") or "-",
                           "Route": path_text(r["path"]) if r["status"] == "SAFE" else r.get("reason", ""),
                           "Length (m)": round(r["length_m"], 0), "ETA (s)": round(r["eta_s"], 0),
                           "Alternatives (extra cost)": alt,
                           "Master note": r.get("override", "")})
            show_df(pd.DataFrame(rr), 330)
        else:
            st.caption("No routes needed - no active incident or nobody on site.")
        st.markdown("**Occupancy and risk by zone**")
        occ = {}
        for w in sim.workers:
            if w.status != "EVACUATED":
                occ[w.zone] = occ.get(w.zone, 0) + 1
        orows = []
        for z in FAC.zones.values():
            orows.append({"Zone": z.id, "Name": z.name, "People": occ.get(z.id, 0),
                          "Risk now": levels.get(z.id, "LOW"),
                          "Forecast +30s": (pm.get("forecast30_levels") or {}).get(z.id, "LOW"),
                          "Hazard index": round((pm.get("zone_H") or {}).get(z.id, 0.0), 1),
                          "To HIGH in (s)": (pm.get("time_to_high") or {}).get(z.id)})
        odf = pd.DataFrame(orows).sort_values(["Hazard index"], ascending=False)
        show_df(odf, 330)
    with e2:
        st.markdown("**Exits**")
        xs = evac.get("exit_status", {})
        show_df(pd.DataFrame([{"Exit": e.id, "Name": e.name, "Zone": e.zone_id,
                               "Status": {"AVAILABLE": "🟢 AVAILABLE", "BLOCKED": "⛔ BLOCKED",
                                          "UNSAFE": "🟠 UNSAFE"}.get(xs.get(e.id, "AVAILABLE"))}
                              for e in FAC.exits.values()]), 280)
        bz = evac.get("blocked_zones", [])
        st.markdown(f"**Blocked / restricted routes** ({len(bz)} blocked, "
                    f"{len(coord.get('restricted_zones', []))} restricted)")
        st.write(", ".join(bz) if bz else "None")
        st.caption("Restricted: " + (", ".join(coord.get("restricted_zones", [])) or "none"))
        st.markdown("**Workers**")
        st.write(counts)
    st.markdown("### Rescue priority ranking")
    ranking = resc.get("ranking", [])
    if ranking:
        st.caption(f"Staging point: {resc.get('staging_exit')} · specialist zones: "
                   f"{', '.join(resc.get('specialist_zones', [])) or 'none'} · "
                   "standard responders are never sent into HIGH / CRITICAL zones.")
        show_df(pd.DataFrame([{
            "Rank": r["rank"], "Zone": r["zone"], "People": r["occupants"],
            "Incapacitated": r["incapacitated"], "Trapped": r["trapped"],
            "Risk": r["level"], "Score": r["score"], "Priority": r["priority"],
            "Response": r["team"], "Assigned": r["assigned"], "Recommendation": r["action"]}
            for r in ranking]), 300)
    else:
        st.caption("No rescue tasking - nobody at risk.")

# ---------------------------------------------------------------------------
# F. incident history
# ---------------------------------------------------------------------------
with tab_hist:
    sm = sim.summary()
    st.markdown("### Final emergency response summary" if sim.complete
                else "### Response summary (live)")
    h1, h2, h3, h4, h5 = st.columns(5)
    fmt = lambda v: "-" if v is None else f"{v:.0f} s"
    h1.metric("Leak started", fmt(sm["leak_started_t"]))
    h2.metric("First abnormal reading", fmt(sm["first_abnormal_t"]))
    h3.metric("Incident confirmed", fmt(sm["detected_t"]),
              None if sm["detection_delay_s"] is None else f"+{sm['detection_delay_s']:.0f}s after leak")
    h4.metric("Evacuation ordered", fmt(sm["evac_order_t"]))
    h5.metric("Peak status", sm["peak_status"])
    k1, k2, k3, k4, k5 = st.columns(5)
    w = sm["workers"]
    k1.metric("Evacuated", f"{w['EVACUATED']}/{sm['workers_total']}")
    k2.metric("Trapped (shelter)", w["TRAPPED"])
    k3.metric("Incapacitated (sim.)", w["INCAPACITATED"])
    k4.metric("Route recalculations", sm["route_recalculations"])
    k5.metric("Conflicts resolved", sm["conflict_resolutions"])
    if sim.complete and sm["rescue_priorities"]:
        st.markdown("**Outstanding rescue priorities:** " + "; ".join(
            f"#{r['rank']} {r['zone']} ({r['team']})" for r in sm["rescue_priorities"]))
    hist_z = list(sim.zone_h_history)
    if len(hist_z) > 2:
        topz = sorted(FAC.zones, key=lambda z: -max(h.get(z, 0) for _, h in hist_z))[:6]
        show_plot(zone_hazard_figure(hist_z, topz), "zone_hist")
    st.markdown("### Chronological incident timeline")
    cats = sorted({e["category"] for e in sim.events})
    pick = st.multiselect("Categories", cats, default=[c for c in cats if c != "CASUALTY"] + (
        ["CASUALTY"] if "CASUALTY" in cats else []), key="tl_cats")
    ev = [e for e in sim.events if e["category"] in pick]
    edf = pd.DataFrame(ev, columns=["t", "category", "source", "text"]) if ev else \
        pd.DataFrame(columns=["t", "category", "source", "text"])
    edf = edf.rename(columns={"t": "t (s)", "category": "Category", "source": "Source",
                              "text": "Event"})
    show_df(edf.iloc[::-1] if len(edf) else edf, 420)
    st.download_button("⬇ Download timeline (CSV)", edf.to_csv(index=False).encode("utf-8"),
                       file_name="resqm_incident_timeline.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# auto-advance loop
# ---------------------------------------------------------------------------
if ss.running and not sim.complete:
    time.sleep(REFRESH_S)
    st.rerun()