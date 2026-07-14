from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd

from .config import COLUMNS, CMAPSS_DATA_DIR, DEFAULT_TOP_K_SENSORS, SENSOR_COLUMNS


@dataclass(frozen=True)
class PreparedSubset:
    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame
    selected_sensors: list[str]
    sensor_directions: dict[str, float]
    sensor_stats: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class SplitSubset:
    """Prepared CMAPSS subset with a leakage-free train/eval split.

    Sensor selection and normalization stats are fit on training units only
    and applied to eval and test units — eval-unit distribution does not
    influence any preprocessing step.
    """
    subset: str
    train: pd.DataFrame
    eval: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.DataFrame
    selected_sensors: list[str]
    sensor_directions: dict[str, float]
    sensor_stats: dict[str, tuple[float, float]]


def load_subset(subset: str, data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = data_dir or CMAPSS_DATA_DIR
    train = pd.read_csv(base_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(base_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_rul = pd.read_csv(base_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])
    return train, test, test_rul


def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    max_cycles = prepared.groupby("unit")["cycle"].transform("max")
    prepared["rul"] = max_cycles - prepared["cycle"]
    return prepared


def add_test_rul(test_frame: pd.DataFrame, test_rul: pd.DataFrame) -> pd.DataFrame:
    """Attach RUL values to test data using the ground-truth RUL file.

    RUL_FD00x.txt gives the RUL at the *last* cycle of each test unit (in unit order).
    For cycle c of a unit: rul_c = final_rul + (last_cycle - c).
    """
    prepared = test_frame.copy()
    unit_ids = sorted(prepared["unit"].unique())
    final_ruls = test_rul["RUL"].values  # 0-indexed, one entry per unit

    chunks: list[pd.DataFrame] = []
    for i, unit_id in enumerate(unit_ids):
        chunk = prepared[prepared["unit"] == unit_id].copy()
        last_cycle = int(chunk["cycle"].max())
        chunk["rul"] = int(final_ruls[i]) + (last_cycle - chunk["cycle"])
        chunks.append(chunk)

    return pd.concat(chunks).sort_values(["unit", "cycle"]).reset_index(drop=True)


def select_informative_sensors(
    train_frame: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K_SENSORS,
    min_std: float = 1e-6,
) -> tuple[list[str], dict[str, float]]:
    correlations: list[tuple[str, float]] = []
    sensor_directions: dict[str, float] = {}
    for sensor in SENSOR_COLUMNS:
        std = float(train_frame[sensor].std())
        if std <= min_std:
            continue
        corr = float(train_frame[sensor].corr(train_frame["rul"]))
        if pd.isna(corr):
            continue
        correlations.append((sensor, abs(corr)))
        sensor_directions[sensor] = 1.0 if corr >= 0 else -1.0

    correlations.sort(key=lambda item: item[1], reverse=True)
    selected = [sensor for sensor, _ in correlations[:top_k]]
    return selected, {sensor: sensor_directions[sensor] for sensor in selected}


def add_health_features(
    frame: pd.DataFrame,
    selected_sensors: list[str],
    sensor_directions: dict[str, float],
    sensor_stats: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    prepared = frame.copy()
    contribution_columns: list[str] = []

    for sensor in selected_sensors:
        mean_value, std_value = sensor_stats[sensor]
        std_value = std_value or 1.0
        z_column = f"{sensor}_z"
        contribution_column = f"{sensor}_degradation"
        prepared[z_column] = (prepared[sensor] - mean_value) / std_value
        prepared[contribution_column] = -sensor_directions[sensor] * prepared[z_column]
        contribution_columns.append(contribution_column)

    prepared["health_index"] = prepared[contribution_columns].mean(axis=1)
    prepared["degradation_rate"] = (
        prepared.groupby("unit")["health_index"].diff().fillna(0.0)
    )
    prepared["anomaly_flag"] = (
        prepared[[f"{sensor}_z" for sensor in selected_sensors]].abs().max(axis=1) >= 2.0
    ).astype(int)
    return prepared


def split_train_units(
    frame: pd.DataFrame,
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a prepared train DataFrame into training and held-out eval sets by unit.

    Returns (train_frame, eval_frame) with no unit overlap.
    """
    rng = random.Random(seed)
    unit_ids = sorted(frame["unit"].unique())
    n_train = round(len(unit_ids) * train_frac)
    shuffled = list(unit_ids)
    rng.shuffle(shuffled)
    train_units = set(shuffled[:n_train])
    eval_units = set(shuffled[n_train:])
    train_frame = frame[frame["unit"].isin(train_units)].reset_index(drop=True)
    eval_frame = frame[frame["unit"].isin(eval_units)].reset_index(drop=True)
    return train_frame, eval_frame


def prepare_split(
    subset: str = "FD001",
    data_dir: Path | None = None,
    top_k: int = DEFAULT_TOP_K_SENSORS,
    train_frac: float = 0.8,
    seed: int = 42,
) -> SplitSubset:
    """Load and prepare a CMAPSS subset with a leakage-free train/eval split.

    Correct order of operations:
      1. Load raw data and compute RUL labels.
      2. Split unit IDs into train and eval sets (raw, before any feature engineering).
      3. Fit sensor selection and normalization stats on training units only.
      4. Apply the train-fit stats to both train and eval units.

    This prevents eval-unit sensor distributions from influencing preprocessing.
    Use this function for all experiment scripts. prepare_subset() still exists
    but fits stats on all units and should only be used for exploratory work.
    """
    train_raw, test_raw, test_rul = load_subset(subset, data_dir=data_dir)
    train_with_rul = add_train_rul(train_raw)

    # Step 2: split on raw data before any feature engineering
    train_raw_split, eval_raw_split = split_train_units(train_with_rul, train_frac=train_frac, seed=seed)

    # Step 3: fit sensor selection and normalization on training units only
    selected_sensors, sensor_directions = select_informative_sensors(train_raw_split, top_k=top_k)
    sensor_stats = {
        sensor: (float(train_raw_split[sensor].mean()), float(train_raw_split[sensor].std()) or 1.0)
        for sensor in selected_sensors
    }

    # Step 4: apply train-fit stats to all splits
    train_prepared = add_health_features(train_raw_split, selected_sensors, sensor_directions, sensor_stats)
    eval_prepared = add_health_features(eval_raw_split, selected_sensors, sensor_directions, sensor_stats)
    test_prepared = add_health_features(
        add_test_rul(test_raw, test_rul), selected_sensors, sensor_directions, sensor_stats
    )

    return SplitSubset(
        subset=subset,
        train=train_prepared,
        eval=eval_prepared,
        test=test_prepared,
        test_rul=test_rul,
        selected_sensors=selected_sensors,
        sensor_directions=sensor_directions,
        sensor_stats=sensor_stats,
    )


def prepare_subset(
    subset: str,
    data_dir: Path | None = None,
    top_k: int = DEFAULT_TOP_K_SENSORS,
) -> PreparedSubset:
    train_raw, test_raw, test_rul = load_subset(subset, data_dir=data_dir)
    train = add_train_rul(train_raw)
    selected_sensors, sensor_directions = select_informative_sensors(train, top_k=top_k)

    sensor_stats = {
        sensor: (float(train[sensor].mean()), float(train[sensor].std()) or 1.0)
        for sensor in selected_sensors
    }

    train_prepared = add_health_features(train, selected_sensors, sensor_directions, sensor_stats)
    test_prepared = add_health_features(
        add_test_rul(test_raw, test_rul), selected_sensors, sensor_directions, sensor_stats
    )

    return PreparedSubset(
        subset=subset,
        train=train_prepared,
        test=test_prepared,
        test_rul=test_rul,
        selected_sensors=selected_sensors,
        sensor_directions=sensor_directions,
        sensor_stats=sensor_stats,
    )
