from __future__ import annotations

from collections import defaultdict
import random

from .config import ACTIONS


class TabularAgent:
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        rng_seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = random.Random(rng_seed)
        self.q_table = defaultdict(lambda: {action: 0.0 for action in ACTIONS})

    def get_q_values(self, state: tuple[str, str, int]) -> dict[str, float]:
        return self.q_table[state]

    def greedy_action(self, state: tuple[str, str, int]) -> str:
        q_values = self.get_q_values(state)
        max_q = max(q_values.values())
        best_actions = [action for action, value in q_values.items() if value == max_q]
        # Alphabetical tie-break keeps evaluation deterministic regardless of RNG state.
        return min(best_actions)

    def select_action(self, state: tuple[str, str, int], explore: bool = True) -> str:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        return self.greedy_action(state)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class QLearningAgent(TabularAgent):
    def update(
        self,
        state: tuple[str, str, int],
        action: str,
        reward: float,
        next_state: tuple[str, str, int] | None,
        done: bool,
    ) -> None:
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done or next_state is None else max(self.q_table[next_state].values())
        target = reward + self.discount_factor * max_next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)


class SARSAgent(TabularAgent):
    def update(
        self,
        state: tuple[str, str, int],
        action: str,
        reward: float,
        next_state: tuple[str, str, int] | None,
        next_action: str | None,
        done: bool,
    ) -> None:
        current_q = self.q_table[state][action]
        next_q = 0.0 if done or next_state is None or next_action is None else self.q_table[next_state][next_action]
        target = reward + self.discount_factor * next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)


class DynaQAgent(QLearningAgent):
    def __init__(self, planning_steps: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self.planning_steps = planning_steps
        # Stores *all* observed transitions per (state, action) so planning samples
        # reflect the stochastic transition distribution rather than only the latest outcome.
        self.model: dict[
            tuple[tuple[str, str, int], str],
            list[tuple[float, tuple[str, str, int] | None, bool]],
        ] = {}
        self._model_keys: list[tuple[tuple[str, str, int], str]] = []

    def update(
        self,
        state: tuple[str, str, int],
        action: str,
        reward: float,
        next_state: tuple[str, str, int] | None,
        done: bool,
    ) -> None:
        super().update(state, action, reward, next_state, done)
        key = (state, action)
        if key not in self.model:
            self.model[key] = []
            self._model_keys.append(key)
        self.model[key].append((reward, next_state, done))

        if not self._model_keys or self.planning_steps <= 0:
            return

        for _ in range(self.planning_steps):
            sampled_state, sampled_action = self.rng.choice(self._model_keys)
            sampled_reward, sampled_next_state, sampled_done = self.rng.choice(
                self.model[(sampled_state, sampled_action)]
            )
            super().update(sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done)
