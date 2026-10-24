"""ResQM - Plotly figures for the dashboard (map, sensor traces, zone history).

All figures only *read* the simulation; nothing here changes its state.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from config import (CHANNEL_LABELS, FACILITY_HEIGHT, FACILITY_WIDTH,
                    RISK_COLORS, RISK_THRESHOLDS, SENSOR_BASELINE, THRESHOLDS)
from factory import get_facility

BG = "#0b1220"
PANEL = "#111a2e"
GRID = "#1f2c47"
TEXT = "#d6e0f5"

WORKER_COLORS = {"WORKING": "#9fc0ff", "EVACUATING": "#2ecc71",
                 "TRAPPED": "#e67e22", "INCAPACITATED": "#e74c3c",
                 "EVACUATED": "#5c6b8a"}
SENSOR_COLORS = {"NORMAL": "#2ecc71", "WARNING": "#f1c40f",
                 "ALARM": "#e74c3c", "FAULT": "#7f8c8d"}
EXIT_COLORS = {"AVAILABLE": "#2ecc71", "UNSAFE": "#e67e22", "BLOCKED": "#e74c3c"}

# heat colour scale tied to the risk thresholds (H axis 0..100)
_ZMAX = 100.0


def _scale():
    m, h, c = (RISK_THRESHOLDS[k] / _ZMAX for k in ("MODERATE", "HIGH", "CRITICAL"))
    col = RISK_COLORS
    return [
        [0.0, "rgba(11,18,32,0)"], [max(m - 1e-4, 0.0), "rgba(46,204,113,0.10)"],
        [m, "rgba(241,196,15,0.55)"], [h - 1e-4, "rgba(241,196,15,0.75)"],
        [h, "rgba(230,126,34,0.75)"], [c - 1e-4, "rgba(230,126,34,0.90)"],
        [c, "rgba(231,76,60,0.85)"], [1.0, "rgba(231,76,60,1.0)"],
    ]


def _layout(fig: go.Figure, title: str | None = None, height: int = 300) -> go.Figure:
    fig.update_layout(
        title=title, height=height, paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11), margin=dict(l=40, r=15, t=40 if title else 15, b=35),
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# --------------------------------------------------------------------------
def facility_figure(sim, layer: str = "agents", show_routes: bool = True,
                    show_workers: bool = True, show_sensors: bool = True,
                    show_forecast: bool = True) -> go.Figure:
    fac = get_facility()
    pm = sim.out.get("prediction") or {}
    evac = sim.out.get("evacuation") or {}
    det = sim.out.get("detection") or {}
    coord = sim.coord or {}
    levels = pm.get("zone_levels") or {z: "LOW" for z in fac.zones}

    if layer == "truth":
        grid = sim.truth_grid()
    elif layer == "forecast":
        grid = pm.get("forecast_grid")
    else:
        grid = pm.get("grid")
    if grid is None:
        grid = np.zeros((len(fac.grid_y), len(fac.grid_x)))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=fac.grid_x, y=fac.grid_y, z=np.clip(grid, 0, _ZMAX), zmin=0, zmax=_ZMAX,
        colorscale=_scale(), showscale=True, hoverinfo="x+y+z", name="hazard index",
        colorbar=dict(title="H", thickness=10, len=0.8, tickfont=dict(size=9)),
        zsmooth="best"))

    if show_forecast and pm.get("forecast_grid") is not None:
        fig.add_trace(go.Contour(
            x=fac.grid_x, y=fac.grid_y, z=pm["forecast_grid"], showscale=False,
            contours=dict(start=RISK_THRESHOLDS["HIGH"], end=RISK_THRESHOLDS["HIGH"],
                          size=1, coloring="lines"),
            line=dict(color="#ff5252", width=2, dash="dot"),
            hoverinfo="skip", name="forecast HIGH contour", showlegend=False))

    # zone borders + labels
    for z in fac.zones.values():
        lv = levels.get(z.id, "LOW")
        fig.add_shape(type="rect", x0=z.x0, y0=z.y0, x1=z.x1, y1=z.y1,
                      line=dict(color=RISK_COLORS[lv], width=3 if lv != "LOW" else 1.2),
                      fillcolor="rgba(0,0,0,0)", layer="above")
        fig.add_annotation(x=z.x0 + 0.8, y=z.y1 - 0.8, text=f"<b>{z.id}</b>",
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=9, color=RISK_COLORS[lv] if lv != "LOW" else "#8fa3c7"))

    # exits
    xs = evac.get("exit_status", {})
    fig.add_trace(go.Scatter(
        x=[e.x for e in fac.exits.values()], y=[e.y for e in fac.exits.values()],
        mode="markers+text", text=[e.id for e in fac.exits.values()],
        textposition="top center", textfont=dict(size=9),
        marker=dict(symbol="triangle-up", size=13, line=dict(width=1, color="#fff"),
                    color=[EXIT_COLORS[xs.get(e.id, "AVAILABLE")] for e in fac.exits.values()]),
        hovertext=[f"{e.id} {e.name}: {xs.get(e.id, 'AVAILABLE')}" for e in fac.exits.values()],
        hoverinfo="text", name="exits"))

    # sensors
    if show_sensors:
        states = det.get("sensor_states", {})
        sp = fac.sensors
        stt = [states.get(s.id, {}).get("status", "NORMAL") for s in sp]
        fig.add_trace(go.Scatter(
            x=[s.x for s in sp], y=[s.y for s in sp], mode="markers",
            marker=dict(symbol="square", size=8, color=[SENSOR_COLORS[s] for s in stt],
                        line=dict(width=1, color="#0b1220")),
            hovertext=[f"{s.id} ({s.zone_id}): {t}" for s, t in zip(sp, stt)],
            hoverinfo="text", name="sensors"))

    # routes
    if show_routes:
        for z, r in (coord.get("final_routes") or {}).items():
            if r.get("status") != "SAFE" or not r.get("in_scope") or not r.get("waypoints"):
                continue
            c = fac.zones[z].center
            pts = [c] + [tuple(p) for p in r["waypoints"]]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines",
                line=dict(color="#4fc3f7", width=2,
                          dash="dash" if r.get("time_critical") else "solid"),
                hovertext=f"{z} -> {r['exit_id']}", hoverinfo="text",
                showlegend=False))

    # workers
    if show_workers:
        ws = [w for w in sim.workers if w.status != "EVACUATED"]
        if ws:
            fig.add_trace(go.Scatter(
                x=[w.x for w in ws], y=[w.y for w in ws], mode="markers",
                marker=dict(size=7, color=[WORKER_COLORS.get(w.status, "#fff") for w in ws],
                            line=dict(width=0.5, color="#fff")),
                hovertext=[f"{w.id} {w.status} (dose {w.dose:.0f})" for w in ws],
                hoverinfo="text", name="workers"))

    # estimated source
    est = pm.get("est") or {}
    if est.get("source"):
        fig.add_trace(go.Scatter(
            x=[est["source"][0]], y=[est["source"][1]], mode="markers",
            marker=dict(symbol="star", size=16, color="#ffd54f",
                        line=dict(width=1, color="#000")),
            hovertext="Estimated source (zone centre)", hoverinfo="text",
            name="estimated source"))

    _layout(fig, height=520)
    fig.update_xaxes(range=[-3, FACILITY_WIDTH + 3], title="x (m)", constrain="domain")
    fig.update_yaxes(range=[-3, FACILITY_HEIGHT + 4], title="y (m)",
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(legend=dict(orientation="h", y=-0.12))
    return fig


# --------------------------------------------------------------------------
def sensor_figure(hist, sensor_ids, channel: str) -> go.Figure:
    fig = go.Figure()
    for sid in sensor_ids:
        d = hist[hist["sensor_id"] == sid]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(x=d["t"], y=d[channel], mode="lines",
                                 name=f"{sid} ({d['zone_id'].iloc[0]})",
                                 connectgaps=False))
    th = THRESHOLDS[channel]
    fig.add_hline(y=th["warn"], line=dict(color="#f1c40f", dash="dash", width=1),
                  annotation_text="warn", annotation_font_size=9)
    fig.add_hline(y=th["alarm"], line=dict(color="#e74c3c", dash="dash", width=1),
                  annotation_text="alarm", annotation_font_size=9)
    fig.add_hline(y=SENSOR_BASELINE[channel], line=dict(color="#5c6b8a", dash="dot", width=1))
    _layout(fig, CHANNEL_LABELS[channel], 260)
    fig.update_xaxes(title="simulated time (s)")
    return fig


def zone_hazard_figure(hist_z, zones) -> go.Figure:
    """``hist_z`` is a list of (t, {zone: H}) tuples."""
    fig = go.Figure()
    ts = [t for t, _ in hist_z]
    for z in zones:
        fig.add_trace(go.Scatter(x=ts, y=[h.get(z, 0.0) for _, h in hist_z],
                                 mode="lines", name=z))
    for lvl, col in (("MODERATE", RISK_COLORS["MODERATE"]), ("HIGH", RISK_COLORS["HIGH"]),
                     ("CRITICAL", RISK_COLORS["CRITICAL"])):
        fig.add_hline(y=RISK_THRESHOLDS[lvl], line=dict(color=col, dash="dash", width=1),
                      annotation_text=lvl, annotation_font_size=9)
    _layout(fig, "Zone hazard index over time (dimensionless)", 300)
    fig.update_xaxes(title="simulated time (s)")
    fig.update_yaxes(title="hazard index")
    return fig