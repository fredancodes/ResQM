"""ResQM - central configuration.

Every tunable number of the simulation lives here so that the agents, the
simulation engine and the dashboard always agree with each other.

IMPORTANT (honesty note): all quantities below describe a *fictional,
simulated* plant.  The "hazard index" is a dimensionless number, NOT a
calibrated chemical concentration, and the sensor values are simulated
indicators (an MQ135 reading is an uncalibrated air-quality indicator).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Geometry / time
# --------------------------------------------------------------------------
FACILITY_WIDTH = 100.0      # metres (x axis, west -> east)
FACILITY_HEIGHT = 60.0      # metres (y axis, south -> north)
CELL_SIZE = 2.0             # metres per dispersion grid cell
GRID_NX = int(FACILITY_WIDTH / CELL_SIZE)   # 50
GRID_NY = int(FACILITY_HEIGHT / CELL_SIZE)  # 30
DT = 2.0                    # simulated seconds per tick
MAX_SIM_TIME = 900.0        # hard stop of one run (simulated seconds)

# --------------------------------------------------------------------------
# Risk classification (on the dimensionless hazard index H)
# --------------------------------------------------------------------------
LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}
RISK_THRESHOLDS = {"MODERATE": 15.0, "HIGH": 40.0, "CRITICAL": 70.0}
RISK_COLORS = {
    "LOW": "#2ecc71",       # green
    "MODERATE": "#f1c40f",  # yellow
    "HIGH": "#e67e22",      # orange
    "CRITICAL": "#e74c3c",  # red
}
RISK_LABELS = {
    "LOW": "Green - low risk",
    "MODERATE": "Yellow - moderate risk",
    "HIGH": "Orange - high risk",
    "CRITICAL": "Red - critical risk",
}

# Overall emergency status ladder (decided by the Master Coordination Agent)
STATUS_LEVELS = ("NORMAL", "ADVISORY", "ALERT", "EMERGENCY", "CRITICAL")
STATUS_RANK = {s: i for i, s in enumerate(STATUS_LEVELS)}
STATUS_COLORS = {
    "NORMAL": "#2ecc71",
    "ADVISORY": "#3498db",
    "ALERT": "#f1c40f",
    "EMERGENCY": "#e67e22",
    "CRITICAL": "#e74c3c",
}


def level_from_h(h: float) -> str:
    """Map a hazard index value to a risk level name."""
    if h != h:  # NaN guard
        return "LOW"
    if h >= RISK_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    if h >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    if h >= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "LOW"


# --------------------------------------------------------------------------
# Simplified dispersion model (2-D Gaussian puffs, see hazard.py)
# --------------------------------------------------------------------------
SIGMA0 = 2.5          # initial puff radius (m)
K_BASE = 1.2          # base eddy diffusivity (m^2/s)
K_WIND = 0.9          # extra diffusivity per m/s of wind (m^2/s per m/s)
VENT_TAU = 240.0      # ventilation / dilution time-constant (s)
HAZARD_SCALE = 400.0   # converts model mass-density to the hazard index H
FORECAST_HORIZONS_S = (15, 30, 45, 60, 90)   # seconds ahead the predictor looks
FORECAST_DISPLAY_S = 60                      # horizon shown on the map

# --------------------------------------------------------------------------
# Simulated sensors (MQ135 indicator, PM10, DHT11)
# --------------------------------------------------------------------------
CHANNELS = ("gas", "pm10", "temp", "hum")
CHANNEL_LABELS = {
    "gas": "MQ135 gas indicator (a.u., uncalibrated)",
    "pm10": "PM10 (ug/m3, simulated)",
    "temp": "Temperature (deg C, DHT11 range 0-50)",
    "hum": "Humidity (% RH, DHT11 range 20-90)",
}
SENSOR_BASELINE = {"gas": 120.0, "pm10": 35.0, "temp": 28.0, "hum": 55.0}
SENSOR_NOISE = {"gas": 6.0, "pm10": 4.0, "temp": 0.4, "hum": 1.5}
SENSOR_BIAS = {"gas": 5.0, "pm10": 3.0, "temp": 0.3, "hum": 1.0}
SENSOR_LIMITS = {          # physical range of each simulated sensor
    "gas": (0.0, 1023.0),  # 10-bit ADC style
    "pm10": (0.0, 1000.0),
    "temp": (0.0, 50.0),   # DHT11 cannot read above 50 C (known limitation)
    "hum": (20.0, 90.0),   # DHT11 humidity range
}
THRESHOLDS = {
    "gas": {"warn": 220.0, "alarm": 400.0},
    "pm10": {"warn": 150.0, "alarm": 350.0},
    "temp": {"warn": 38.0, "alarm": 45.0},
    "hum": {"warn": 40.0, "alarm": 30.0},   # LOW humidity is the abnormal side
}
SENSOR_FAULT_TEMP = 70.0     # unclipped temperature that destroys a sensor
SENSOR_FAULT_TICKS = 3       # ... if sustained this many ticks
SPIKE_PROB = 0.002           # benign single-tick PM10 spike (tests debouncing)
SPIKE_PM10 = 165.0
WIND_NOISE_DEG = 4.0
WIND_NOISE_SPEED = 0.25

# How strongly each hazard type drives each sensor channel per unit of H.
HAZARD_TYPES = {
    "toxic_gas": {
        "label": "Toxic / corrosive gas release",
        "gas": 8.5, "pm10": 0.15, "temp": 0.02, "hum": 0.05,
        "primary": "gas",
    },
    "flammable_gas": {
        "label": "Flammable gas release",
        "gas": 8.5, "pm10": 0.10, "temp": 0.04, "hum": 0.0,
        "primary": "gas",
    },
    "fire_smoke": {
        "label": "Fire and smoke",
        "gas": 3.0, "pm10": 6.0, "temp": 0.6, "hum": 0.35,
        "primary": "pm10",
    },
}

# --------------------------------------------------------------------------
# Agents / coordination
# --------------------------------------------------------------------------
CONFIRM_TICKS = 2            # consecutive violating ticks needed to confirm
CLEAR_TICKS = 25             # calm ticks before an incident is cleared
FIT_MAX_LAG_TICKS = 15       # prediction agent searches release start lags
RESPONSE_DELAY_S = 6.0       # time between evacuation order and first movement
WORKER_SPEED = 1.4           # m/s walking speed while evacuating
EXIT_CAPACITY = 10           # persons an exit can absorb in the response window
EXIT_REBALANCE_COST_FACTOR = 1.6
TIME_CONFLICT_MARGIN_S = 3.0

# transit cost multipliers by risk level (CRITICAL is blocked, never routed)
LEVEL_COST = {"LOW": 1.0, "MODERATE": 3.0, "HIGH": 10.0}
# Highest level an evacuation route may pass THROUGH (strict policy).
MAX_TRANSIT_LEVEL = "MODERATE"

# Simulated exposure model (illustrative only, not toxicology)
DOSE_FLOOR_H = 15.0          # H above which exposure accumulates
DOSE_INCAPACITATED = 1500.0  # accumulated dose that incapacitates a worker
DOSE_EXPOSED = 600.0         # dose flagged as "exposed"

RESCUE_STANDARD_TEAMS = 2
RESCUE_SPECIALIST_TEAMS = 1


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    hazard_type: str
    source_zone: str
    q_factor: float            # relative release strength
    diffusivity: float         # relative spreading (light/buoyant gas > 1)
    growth: str                # "ramp" | "t2" | "step"
    growth_tau: float          # seconds
    default_intensity: int
    default_wind_from: float
    default_wind_speed: float
    flammable: bool
    description: str
    assumptions: tuple = field(default_factory=tuple)


SCENARIOS = {
    "chem_storage": Scenario(
        key="chem_storage",
        label="1. Chemical leak in storage unit",
        hazard_type="toxic_gas", source_zone="CHEM",
        q_factor=1.0, diffusivity=1.0, growth="ramp", growth_tau=12.0,
        default_intensity=7, default_wind_from=315.0, default_wind_speed=2.5,
        flammable=False,
        description="A pressurised tank in Chemical Storage starts venting a "
                    "toxic vapour. Heavy, fairly slow-spreading plume.",
        assumptions=(
            "Release rate ramps up over ~12 s then stays constant.",
            "Gas indicator (MQ135) rises strongly; PM10 and temperature barely move.",
            "Plume is advected by the wind and spreads by turbulent diffusion.",
        ),
    ),
    "gas_production": Scenario(
        key="gas_production",
        label="2. Gas leak in production unit",
        hazard_type="flammable_gas", source_zone="PU-B",
        q_factor=0.85, diffusivity=1.5, growth="ramp", growth_tau=6.0,
        default_intensity=6, default_wind_from=90.0, default_wind_speed=1.5,
        flammable=True,
        description="A flange fails in Production Unit B releasing a light, "
                    "flammable gas that spreads quickly in all directions.",
        assumptions=(
            "Light gas: higher diffusivity than the toxic-vapour scenario.",
            "Release reaches full rate within ~6 s (sudden flange failure).",
            "Treated as flammable: power isolation is recommended.",
        ),
    ),
    "fire_smoke": Scenario(
        key="fire_smoke",
        label="3. Fire and smoke emergency",
        hazard_type="fire_smoke", source_zone="WARE",
        q_factor=1.1, diffusivity=1.8, growth="t2", growth_tau=70.0,
        default_intensity=6, default_wind_from=200.0, default_wind_speed=2.0,
        flammable=True,
        description="A fire starts in the Warehouse and grows quadratically. "
                    "Smoke (PM10) and temperature dominate; sensors near the "
                    "fire may fail from heat.",
        assumptions=(
            "Fire grows ~t^2 and saturates after ~70 s.",
            "PM10 and temperature are the dominant indicators.",
            "DHT11 saturates at 50 C and fails above 70 C (simulated damage).",
        ),
    ),
    "custom": Scenario(
        key="custom",
        label="4. Custom scenario",
        hazard_type="toxic_gas", source_zone="PROC",
        q_factor=1.0, diffusivity=1.0, growth="ramp", growth_tau=10.0,
        default_intensity=5, default_wind_from=270.0, default_wind_speed=2.0,
        flammable=False,
        description="You choose the hazard type, source zone, wind and "
                    "intensity from the sidebar.",
        assumptions=("All parameters are user supplied.",),
    ),
}
SCENARIO_ORDER = ("chem_storage", "gas_production", "fire_smoke", "custom")

# Evacuation test scenarios (applied through the sidebar)
EVAC_PRESETS = {
    "all_open": {
        "label": "A. All exits open",
        "blocked_exits": (), "blocked_zones": (),
    },
    "west_blocked": {
        "label": "B. West gate (EX1) and NW exit (EX7) blocked",
        "blocked_exits": ("EX1", "EX7"), "blocked_zones": (),
    },
    "east_sealed": {
        "label": "C. East side sealed (EX2, EX4, EX6)",
        "blocked_exits": ("EX2", "EX4", "EX6"), "blocked_zones": (),
    },
    "lockdown": {
        "label": "D. Total lockdown - every exit blocked (no-route test)",
        "blocked_exits": ("EX1", "EX2", "EX3", "EX4", "EX5", "EX6", "EX7"),
        "blocked_zones": (),
    },
}
EVAC_PRESET_ORDER = ("all_open", "west_blocked", "east_sealed", "lockdown")

# Deterministic demo preset (one-click DEMO MODE)
DEMO = {
    "scenario": "chem_storage",
    "source_zone": "CHEM",
    "intensity": 7,
    "wind_from": 315.0,
    "wind_speed": 2.5,
    "seed": 42,
    "speed": 2,
    "evac_preset": "all_open",
    "trigger_at": 20.0,                       # leak starts at t = 20 s
    "wind_events": ((110.0, 250.0, 3.0),),    # (t, wind_from, wind_speed) shift
}

DISCLAIMER = (
    "SIMULATION ONLY - all sensor values, gas dispersion, workers and "
    "response actions are simulated. The dispersion model is a simplified "
    "Gaussian-puff approximation and must NOT be used for real safety "
    "decisions. MQ135 values are uncalibrated indicators, not chemical "
    "concentrations."
)