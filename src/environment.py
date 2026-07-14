from __future__ import annotations

from dataclasses import dataclass
import random

import pandas as pd

from .config import ACTIONS, TERMINAL_MAINTENANCE_ACTIONS


@dataclass(frozen=True)
class RewardConfig:
    failure_penalty: float = -100.0
    inspect_cost: float = -1.0
    minor_repair_cost: float = -5.0
    major_overhaul_cost: float = -20.0
    replacement_cost: float = -50.0
    timely_bonus: float = 10.0          # bonus for acting at the optimal CRITICAL window
    late_maintenance_penalty: float = -5.0  # penalty for waiting until IMMINENT
    premature_heavy_penalty: float = -15.0
    premature_minor_penalty: float = -8.0
    watch_replace_penalty: float = -5.0


@dataclass(frozen=True)
class BucketConfig:
    """Thresholds for discretising continuous health and degradation features.

    Health thresholds (health_index):
        HEALTHY  < healthy_max
        WATCH    < watch_max
        CRITICAL < critical_max
        IMMINENT >= critical_max

    Degradation thresholds (degradation_rate):
        SLOW     <= slow_max
        MODERATE <= moderate_max
        FAST     > moderate_max
    """
    healthy_max: float = -0.10
    watch_max: float = 0.53
    critical_max: float = 1.35
    slow_max: float = 0.0
    moderate_max: float = 0.05


class MaintenanceEnv:
    def __init__(
        self,
        frame: pd.DataFrame,
        reward_config: RewardConfig | None = None,
        bucket_config: BucketConfig | None = None,
        rng_seed: int = 42,
    ) -> None:
        self.frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
        self.reward_config = reward_config or RewardConfig()
        self.bucket_config = bucket_config or BucketConfig()
        self.rng = random.Random(rng_seed)
        self.unit_ids = sorted(int(unit_id) for unit_id in self.frame["unit"].unique())
        self.episodes = {
            unit_id: group.reset_index(drop=True)
            for unit_id, group in self.frame.groupby("unit")
        }
        self.current_unit_id: int | None = None
        self.current_episode: pd.DataFrame | None = None
        self.current_index = 0

    def reset(self, unit_id: int | None = None) -> tuple[str, str, int]:
        if unit_id is None:
            unit_id = self.rng.choice(self.unit_ids)
        self.current_unit_id = int(unit_id)
        self.current_episode = self.episodes[self.current_unit_id]
        self.current_index = 0
        return self._current_state()

    def sample_unit_id(self) -> int:
        return self.rng.choice(self.unit_ids)

    def _current_row(self) -> pd.Series:
        if self.current_episode is None:
            raise RuntimeError("Call reset() before step().")
        return self.current_episode.iloc[self.current_index]

    def _current_state(self) -> tuple[str, str, int]:
        return self._row_to_state(self._current_row(), self.bucket_config)

    @staticmethod
    def _row_to_state(
        row: pd.Series,
        bucket_config: BucketConfig | None = None,
    ) -> tuple[str, str, int]:
        bc = bucket_config or BucketConfig()
        health_bucket = MaintenanceEnv._health_bucket(float(row["health_index"]), bc)
        degradation_bucket = MaintenanceEnv._degradation_bucket(float(row["degradation_rate"]), bc)
        anomaly_flag = int(row["anomaly_flag"])
        return (health_bucket, degradation_bucket, anomaly_flag)

    @staticmethod
    def _health_bucket(
        health_index: float,
        bucket_config: BucketConfig | None = None,
    ) -> str:
        """Bucket health_index using BucketConfig thresholds (default: FD001 p50/p75/p90).

        Higher health_index values indicate more degradation:
            HEALTHY  < healthy_max   (sensor readings near baseline)
            WATCH    < watch_max     (detectable drift)
            CRITICAL < critical_max  (clear degradation trend — optimal action window)
            IMMINENT >= critical_max (near-failure zone)
        """
        bc = bucket_config or BucketConfig()
        if health_index < bc.healthy_max:
            return "HEALTHY"
        if health_index < bc.watch_max:
            return "WATCH"
        if health_index < bc.critical_max:
            return "CRITICAL"
        return "IMMINENT"

    @staticmethod
    def _degradation_bucket(
        rate: float,
        bucket_config: BucketConfig | None = None,
    ) -> str:
        bc = bucket_config or BucketConfig()
        if rate > bc.moderate_max:
            return "FAST"
        if rate > bc.slow_max:
            return "MODERATE"
        return "SLOW"

    def step(self, action: str) -> tuple[tuple[str, str, int] | None, float, bool, dict]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        row = self._current_row()
        state = self._row_to_state(row, self.bucket_config)
        info = {
            "unit_id": self.current_unit_id,
            "cycle": int(row["cycle"]),
            "rul": float(row["rul"]),
            "state": state,
            "action": action,
            "event": "transition",
        }

        if action in TERMINAL_MAINTENANCE_ACTIONS:
            reward = self._maintenance_reward(state[0], action)
            info["event"] = "maintenance"
            info["terminal_reason"] = action
            return None, reward, True, info

        if state[0] == "IMMINENT" and action in {"CONTINUE", "INSPECT"}:
            reward = self.reward_config.failure_penalty + self._step_cost(action)
            info["event"] = "failure"
            info["terminal_reason"] = "unexpected_failure"
            return None, reward, True, info

        if self.current_episode is None:
            raise RuntimeError("Current episode is not initialized.")

        self.current_index += 1
        if self.current_index >= len(self.current_episode):
            reward = self._step_cost(action)
            info["event"] = "completed"
            info["terminal_reason"] = "end_of_trajectory"
            return None, reward, True, info

        reward = self._step_cost(action)
        next_state = self._current_state()
        return next_state, reward, False, info

    def _step_cost(self, action: str) -> float:
        if action == "INSPECT":
            return self.reward_config.inspect_cost
        return 0.0

    def _maintenance_reward(self, rul_bucket: str, action: str) -> float:
        reward = 0.0
        if action == "MINOR_REPAIR":
            reward += self.reward_config.minor_repair_cost
        elif action == "MAJOR_OVERHAUL":
            reward += self.reward_config.major_overhaul_cost
        elif action == "REPLACE":
            reward += self.reward_config.replacement_cost

        if rul_bucket == "CRITICAL":
            reward += self.reward_config.timely_bonus       # optimal maintenance window
        elif rul_bucket == "IMMINENT":
            reward += self.reward_config.late_maintenance_penalty  # too late — risky
        elif rul_bucket == "HEALTHY":
            if action in {"MAJOR_OVERHAUL", "REPLACE"}:
                reward += self.reward_config.premature_heavy_penalty
            else:
                reward += self.reward_config.premature_minor_penalty
        elif rul_bucket == "WATCH" and action == "REPLACE":
            reward += self.reward_config.watch_replace_penalty

        return reward
