from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CMAPSS_DATA_DIR = Path(__file__).resolve().parent / "CMAPSSData"

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]

DEFAULT_SUBSET = "FD001"
DEFAULT_TOP_K_SENSORS = 5

ACTIONS = (
    "CONTINUE",
    "INSPECT",
    "MINOR_REPAIR",
    "MAJOR_OVERHAUL",
    "REPLACE",
)

TERMINAL_MAINTENANCE_ACTIONS = {
    "MINOR_REPAIR",
    "MAJOR_OVERHAUL",
    "REPLACE",
}
