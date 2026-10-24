"""Headless smoke test: run every scenario, build every figure."""
from __future__ import annotations

import sys
import traceback

from config import EVAC_PRESET_ORDER, SCENARIO_ORDER
from simulation import SimConfig, Simulation
from viz import facility_figure, sensor_figure, zone_hazard_figure

fails = 0


def run(label, cfg, ticks=200, trigger=True):
    global fails
    try:
        sim = Simulation(cfg)
        if trigger and cfg.trigger_at is None:
            sim.trigger_leak()
        for _ in range(ticks):
            sim.tick()
            if sim.complete:
                break
        for layer in ("agents", "forecast", "truth"):
            facility_figure(sim, layer, True, True, True, True)
        h = sim.sensor_history()
        for ch in ("gas", "pm10", "temp", "hum"):
            sensor_figure(h, ["S01", "S03"], ch)
        if len(sim.zone_h_history) > 2:
            zone_hazard_figure(list(sim.zone_h_history), ["CHEM", "PROC"])
        sm = sim.summary()
        print(f"OK  {label:28s} t={sim.t:6.0f}s status={sm['peak_status']:9s} "
              f"detect={sm['detected_t']} workers={sm['workers']}")
        assert sim.out.get("prediction") is not None
    except Exception:
        fails += 1
        print(f"FAIL {label}")
        traceback.print_exc()


for key in SCENARIO_ORDER:
    run(f"scenario {key}", SimConfig.for_scenario(key))
for ev in EVAC_PRESET_ORDER:
    run(f"evac {ev}", SimConfig.for_scenario("chem_storage", evac_preset=ev))
run("DEMO", SimConfig.demo(), ticks=450, trigger=False)
sys.exit(1 if fails else 0)