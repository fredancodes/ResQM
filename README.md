# ResQM - Multi-Agent Industrial Disaster Response System

A **simulation-only** emergency operations centre for a fictional 100 m x 60 m
plant. Five cooperating rule-based agents detect a gas leak / fire, predict how
it spreads, plan evacuation routes with A*, prioritise rescue and coordinate the
overall response. No trained ML model is used.

> **SIMULATION ONLY.** Sensor values, dispersion, workers and response actions
> are all simulated. The hazard index is a dimensionless number, not a chemical
> concentration; MQ135 values are uncalibrated indicators. Never use this for
> real safety decisions.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py        # dashboard (click "DEMO MODE" for a scripted run)
python smoke_test.py        # headless test of every scenario + evacuation preset
```

`smoke_test.py` also builds every Plotly figure when plotly is installed; without
plotly it tests the simulation/agent engine only.

## Files

| File | Role |
|---|---|
| `config.py` | Every tunable constant, scenarios, evacuation presets, demo script |
| `factory.py` | Zones, doors, exits, sensor nodes, worker placement, routing graph |
| `hazard.py` | Advected Gaussian-puff dispersion model (ground truth and forecasts) |
| `sensors.py` | Simulated MQ135 / PM10 / DHT11 network with noise, bias, spikes, heat faults |
| `evacuation.py` | A* routing that never enters CRITICAL zones and never falls back to unsafe routes |
| `agents.py` | Message bus + Detection, Prediction, Evacuation, Rescue agents |
| `coordinator.py` | Master agent: status ladder, conflict rules C1-C4, simulated actions |
| `simulation.py` | World loop: physics, sensors, exposure, agents, worker movement |
| `viz.py` | Plotly figures (facility map, sensor traces, zone hazard history) |
| `app.py` | Streamlit dashboard |

## Agent data flow

Sensors -> **Detection** -> **Prediction** -> **Evacuation** -> **Rescue** -> **Master**.
Agents never call each other; they read their inbox on the `MessageBus` (visible
in the dashboard log). The agents never see the ground-truth plume - only sensor
snapshots, a noisy wind measurement and the occupancy report.

## Calibration note

`HAZARD_SCALE` (config.py) was calibrated so that a default-intensity release
reaches roughly the CRITICAL band (H >= 70) in the source zone and neighbouring
zones step through MODERATE/HIGH/CRITICAL as the plume spreads. Sensor thresholds
(`THRESHOLDS`) and risk thresholds (`RISK_THRESHOLDS`) are consistent with it:
a gas sensor alarms at about H = 33.

## Known limitations

* **Point sensors can miss a plume.** There is one sensor per zone, at the zone
  centre. If wind carries a release away from that sensor (e.g. a source at a
  zone edge with wind blowing out of the building) the system may not detect
  it even though people nearby are exposed. This mirrors real sensor-coverage
  gaps; add sensors or change the wind to see the difference.
* The dispersion model ignores walls, buoyancy, terrain and chemistry.
* The exposure/dose model is illustrative, not toxicology.
* The Prediction Agent assumes the source is the zone centre of the first alarm.