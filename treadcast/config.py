"""All the numbers the simulator and the models share, in one place.

Units: mm for tread depth, km for distance, bar for pressure, °C for temperature,
kg for wheel load, g (9.81 m/s²) for acceleration.
"""

from dataclasses import dataclass

SEED = 7
DAYS = 730                      # two years of fleet operation
START_DATE = "2024-01-01"

# --- tire: 315/80 R22.5, a common truck size ---
RIM_RADIUS_MM = 285.75          # 22.5 inch rim
SIDEWALL_MM = 252.0             # 80% of 315 mm
CARCASS_RADIUS_MM = RIM_RADIUS_MM + SIDEWALL_MM - 21.0   # outer radius minus the deepest new tread

LEGAL_LIMIT_MM = 1.6            # EU legal minimum
REPLACE_AT_MM = 3.0             # fleet policy: plan a new tire at this depth
SERVICE_REPLACE_MM = 4.0        # at a service, a tire at or below this is replaced

G = 9.81


@dataclass(frozen=True)
class Position:
    name: str
    axle: str                   # "steer" or "drive"
    new_depth_mm: float
    base_wear: float            # mm per 1000 km, before all factors
    full_load_kg: float
    empty_load_kg: float
    cold_pressure_bar: float


POSITIONS = [
    Position("FL", "steer", 16.0, 0.070, 3550, 3000, 8.5),
    Position("FR", "steer", 16.0, 0.070, 3550, 3000, 8.5),
    Position("RLO", "drive", 21.0, 0.085, 2875, 1300, 8.0),
    Position("RLI", "drive", 21.0, 0.085, 2875, 1300, 8.0),
    Position("RRI", "drive", 21.0, 0.085, 2875, 1300, 8.0),
    Position("RRO", "drive", 21.0, 0.085, 2875, 1300, 8.0),
]


@dataclass(frozen=True)
class Sku:
    name: str
    wear_factor: float          # >1 wears faster
    stiffness: float            # vertical stiffness factor
    ring_hz: float              # tread-block ringing frequency when new


SKUS = {
    "steer": [Sku("S-Long", 0.90, 1.00, 620.0), Sku("S-Value", 1.15, 0.94, 580.0)],
    "drive": [Sku("D-Long", 0.92, 1.02, 540.0), Sku("D-Value", 1.20, 0.96, 500.0)],
}


@dataclass(frozen=True)
class Route:
    name: str
    trucks: int
    work_prob: float            # chance a weekday is a driving day
    km_mean: float              # km per driving day
    speed_kmh: float            # typical steady speed
    load_mean: float            # average load fraction (0 = empty, 1 = full)
    roughness: float            # road roughness, scales signal noise
    wear_factor: float          # stop-and-go and turning wear


ROUTES = [
    Route("long-haul", 5, 0.85, 640.0, 80.0, 0.75, 0.8, 1.00),
    Route("regional", 4, 0.80, 330.0, 66.0, 0.60, 1.0, 1.12),
    Route("urban", 3, 0.80, 170.0, 38.0, 0.50, 1.4, 1.40),
]

# --- sensor and signal ---
WINDOW_DEG = 45.0               # template covers ±45° around the contact patch
TEMPLATE_POINTS = 256
SENSOR_NOISE_G = 0.6
GPS_DISTANCE_NOISE = 0.0015     # relative error of a day's GPS distance
GPS_MISSING_PROB = 0.15
LOAD_SENSOR_NOISE = 0.08        # relative error of the axle-load estimate

# --- services and labels ---
SERVICE_EVERY_KM = 30000.0
SERVICE_EVERY_DAYS = 75
GAUGE_NOISE_MM = 0.3
PSEUDO_LABEL_WEIGHT = 0.3       # interpolated labels count less than real readings

# --- evaluation ---
TEST_TRUCKS = ["T03", "T08", "T11"]      # one per route; their tires are never trained on
TRAIN_CUTOFF_DAY = 500                    # training rows come from before this day only
