"""Simulated sensor network (MQ135 indicator, PM10, DHT11 temperature/humidity).

Everything here is SIMULATED.  A reading is generated as

    value = baseline + per-sensor bias + gain * H(sensor position) + noise

where H is the ground-truth hazard index from the plume model.  The MQ135
number is an uncalibrated air-quality *indicator*; it is never converted to
a concentration of a specific chemical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (CHANNELS, HAZARD_TYPES, SENSOR_BASELINE, SENSOR_BIAS,
                    SENSOR_FAULT_TEMP, SENSOR_FAULT_TICKS, SENSOR_LIMITS,
                    SENSOR_NOISE, SPIKE_PM10, SPIKE_PROB, THRESHOLDS,
                    WIND_NOISE_DEG, WIND_NOISE_SPEED)

SNAPSHOT_COLUMNS = ["tick", "t", "sensor_id", "zone_id", "x", "y",
                    "gas", "pm10", "temp", "hum", "health"]


def channel_levels(gas: float, pm10: float, temp: float, hum: float) -> dict:
    """Threshold level per channel: 0 normal, 1 warning, 2 alarm.

    Humidity is abnormal on the LOW side (fire / dry gas), so it uses
    ``<=`` comparisons.  NaN (failed sensor) yields level 0 per channel;
    the caller must handle faults separately.
    """
    out = {}
    for ch, v in (("gas", gas), ("pm10", pm10), ("temp", temp)):
        th = THRESHOLDS[ch]
        if v != v:
            out[ch] = 0
        elif v >= th["alarm"]:
            out[ch] = 2
        elif v >= th["warn"]:
            out[ch] = 1
        else:
            out[ch] = 0
    th = THRESHOLDS["hum"]
    if hum != hum:
        out["hum"] = 0
    elif hum <= th["alarm"]:
        out["hum"] = 2
    elif hum <= th["warn"]:
        out["hum"] = 1
    else:
        out["hum"] = 0
    return out


def anomaly_score(gas: float, pm10: float, temp: float, hum: float) -> float:
    """Worst channel, normalised so that 1.0 == alarm threshold, 0 == baseline."""
    best = 0.0
    for ch, v in (("gas", gas), ("pm10", pm10), ("temp", temp)):
        if v != v:
            continue
        base, alarm = SENSOR_BASELINE[ch], THRESHOLDS[ch]["alarm"]
        best = max(best, (v - base) / (alarm - base))
    if hum == hum:
        base, alarm = SENSOR_BASELINE["hum"], THRESHOLDS["hum"]["alarm"]
        best = max(best, (base - hum) / (base - alarm))
    return max(0.0, float(best))


class SensorNetwork:
    """One sensor node per zone plus a simulated weather station."""

    def __init__(self, facility, hazard_type: str, seed: int) -> None:
        self.facility = facility
        self.gains = HAZARD_TYPES[hazard_type]
        self.rng = np.random.default_rng(int(seed) + 31337)
        self.specs = facility.sensors
        n = len(self.specs)
        self.bias = np.stack(
            [self.rng.normal(0.0, SENSOR_BIAS[ch], n) for ch in CHANNELS], axis=1)
        self.hot_ticks = np.zeros(n, dtype=int)
        self.faulted = np.zeros(n, dtype=bool)

    def read(self, tick: int, t: float, h_at_sensors: np.ndarray) -> pd.DataFrame:
        """Generate one snapshot.  ``h_at_sensors`` is ordered like ``specs``."""
        n = len(self.specs)
        h = np.nan_to_num(np.asarray(h_at_sensors, dtype=float), nan=0.0)
        noise = np.stack(
            [self.rng.normal(0.0, SENSOR_NOISE[ch], n) for ch in CHANNELS], axis=1)
        spike = self.rng.random(n) < SPIKE_PROB          # benign disturbances

        gas = SENSOR_BASELINE["gas"] + self.bias[:, 0] + self.gains["gas"] * h + noise[:, 0]
        pm = SENSOR_BASELINE["pm10"] + self.bias[:, 1] + self.gains["pm10"] * h + noise[:, 1]
        temp = SENSOR_BASELINE["temp"] + self.bias[:, 2] + self.gains["temp"] * h + noise[:, 2]
        hum = SENSOR_BASELINE["hum"] + self.bias[:, 3] - self.gains["hum"] * h + noise[:, 3]
        pm = np.where(spike, np.maximum(pm, SPIKE_PM10), pm)

        # heat damage: unclipped temperature far above the DHT11 range
        hot = temp > SENSOR_FAULT_TEMP
        self.hot_ticks = np.where(hot, self.hot_ticks + 1, 0)
        self.faulted |= self.hot_ticks >= SENSOR_FAULT_TICKS

        rows = []
        for i, sp in enumerate(self.specs):
            if self.faulted[i]:
                vals = (np.nan, np.nan, np.nan, np.nan)
                health = "FAULT"
            else:
                vals = tuple(float(np.clip(v, *SENSOR_LIMITS[ch])) for v, ch in
                             zip((gas[i], pm[i], temp[i], hum[i]), CHANNELS))
                health = "OK"
            rows.append((tick, t, sp.id, sp.zone_id, sp.x, sp.y, *vals, health))
        return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)

    def read_weather(self, true_from: float, true_speed: float) -> dict:
        """Noisy wind measurement (simulated weather mast)."""
        f = (float(true_from) + self.rng.normal(0.0, WIND_NOISE_DEG)) % 360.0
        s = max(0.0, float(true_speed) + self.rng.normal(0.0, WIND_NOISE_SPEED))
        return {"from": f, "speed": s}