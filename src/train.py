from __future__ import annotations

import random
from typing import Callable

import numpy as np

from .agents import DynaQAgent, QLearningAgent, SARSAgent, TabularAgent
from .environment import MaintenanceEnv
from .policies import RuleBasedPolicy, apply_safety_override


def _maybe_override(action: str, state: tuple[str, str, int], use_safety_override: bool) -> str:
    if not use_safety_override:
        return action
    return apply_safety_override(action, state)


def train_agent(
    env: MaintenanceEnv,
    agent: TabularAgent,
    episodes: int = 500,
    rng_seed: int = 42,
    use_safety_override: bool = False,
    greedy_eval_callback: Callable[[int], None] | None = None,
    greedy_eval_interval: int = 50,
) -> list[float]:
    """Train agent for `episodes` episodes.

    greedy_eval_callback, if provided, is called with the current episode number
    every `greedy_eval_interval` episodes.  Use this to record greedy-policy
    performance during training without mixing it with exploratory reward.
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    rewards: list[float] = []

    for ep_idx in range(episodes):
        state = env.reset(unit_id=env.sample_unit_id())
        total_reward = 0.0

        if isinstance(agent, SARSAgent):
            action = _maybe_override(agent.select_action(state, explore=True), state, use_safety_override)
            while True:
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                next_action = None
                if not done and next_state is not None:
                    next_action = _maybe_override(
                        agent.select_action(next_state, explore=True),
                        next_state,
                        use_safety_override,
                    )
                agent.update(state, action, reward, next_state, next_action, done)
                if done:
                    break
                state, action = next_state, next_action
        else:
            while True:
                action = _maybe_override(agent.select_action(state, explore=True), state, use_safety_override)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                if isinstance(agent, QLearningAgent):  # covers DynaQAgent (subclass)
                    agent.update(state, action, reward, next_state, done)
                else:
                    raise TypeError(f"Unsupported agent type: {type(agent).__name__}")

                if done:
                    break
                state = next_state

        agent.decay_epsilon()
        rewards.append(total_reward)

        if greedy_eval_callback is not None and (ep_idx + 1) % greedy_eval_interval == 0:
            greedy_eval_callback(ep_idx + 1)

    return rewards


def evaluate_rule_policy(
    env: MaintenanceEnv,
    policy: RuleBasedPolicy | None = None,
    use_safety_override: bool = False,
) -> list[dict]:
    rule_policy = policy or RuleBasedPolicy()
    summaries: list[dict] = []
    for unit_id in env.unit_ids:
        state = env.reset(unit_id=unit_id)
        total_reward = 0.0
        while True:
            action = rule_policy.select_action(state)
            action = _maybe_override(action, state, use_safety_override)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                summaries.append(
                    {
                        "unit_id": unit_id,
                        "total_reward": total_reward,
                        "terminal_action": info["action"],
                        "terminal_event": info["event"],
                        "terminal_cycle": info["cycle"],
                        "rul_at_action": info["rul"],
                        "health_bucket_at_action": info["state"][0],
                    }
                )
                break
            state = next_state
    return summaries
