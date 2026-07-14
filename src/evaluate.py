from __future__ import annotations

import statistics

from .agents import TabularAgent
from .environment import MaintenanceEnv
from .policies import apply_safety_override


def evaluate_agent(
    env: MaintenanceEnv,
    agent: TabularAgent,
    use_safety_override: bool = False,
) -> list[dict]:
    episode_summaries: list[dict] = []
    for unit_id in env.unit_ids:
        state = env.reset(unit_id=unit_id)
        total_reward = 0.0
        while True:
            policy_action = agent.select_action(state, explore=False)
            executed_action = (
                apply_safety_override(policy_action, state) if use_safety_override else policy_action
            )
            next_state, reward, done, info = env.step(executed_action)
            total_reward += reward
            if done:
                episode_summaries.append(
                    {
                        "unit_id": unit_id,
                        "total_reward": total_reward,
                        "policy_action": policy_action,
                        "terminal_action": executed_action,
                        "override_applied": executed_action != policy_action,
                        "terminal_event": info["event"],
                        "terminal_cycle": info["cycle"],
                        "rul_at_action": info["rul"],
                        "health_bucket_at_action": info["state"][0],
                    }
                )
                break
            state = next_state
    return episode_summaries


def summarize_episodes(episode_summaries: list[dict]) -> dict[str, float]:
    if not episode_summaries:
        raise ValueError("Cannot summarize: no episode data provided.")
    rewards = [summary["total_reward"] for summary in episode_summaries]
    n = len(episode_summaries)
    failure_rate = sum(summary["terminal_event"] == "failure" for summary in episode_summaries) / n
    # MAJOR_OVERHAUL or REPLACE taken while still HEALTHY — the original metric.
    early_replacement_rate = sum(
        summary["terminal_action"] in {"MAJOR_OVERHAUL", "REPLACE"}
        and summary["health_bucket_at_action"] == "HEALTHY"
        for summary in episode_summaries
    ) / n
    # Any terminal maintenance action taken before reaching CRITICAL — catches
    # premature MINOR_REPAIR (e.g. SARSA's cycle-1 behaviour) that early_replacement_rate misses.
    premature_any_rate = sum(
        summary["terminal_event"] == "maintenance"
        and summary["health_bucket_at_action"] in {"HEALTHY", "WATCH"}
        for summary in episode_summaries
    ) / n
    mean_action_cycle = statistics.mean(summary["terminal_cycle"] for summary in episode_summaries)
    return {
        "average_reward": statistics.mean(rewards),
        "std_reward": statistics.stdev(rewards) if n > 1 else 0.0,
        "failure_rate": failure_rate,
        "early_replacement_rate": early_replacement_rate,
        "premature_any_rate": premature_any_rate,
        "mean_action_cycle": mean_action_cycle,
    }
