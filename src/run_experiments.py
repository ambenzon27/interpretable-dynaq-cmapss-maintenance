"""
run_experiments.py
==================
Runs all three Phase 1 experiments for the AI 322 mini-project.

Experiment 1  Planning-step ablation     DynaQ n in {0, 5, 10, 20}
Experiment 2  Algorithm comparison       rule-based, SARSA, Q-learning, DynaQ(best_n)
Experiment 3  Safety-override ablation   same trained agent, eval with/without override

Usage:
    python -m src.run_experiments
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.data import SplitSubset, prepare_split
from src.agents import DynaQAgent, QLearningAgent, SARSAgent, TabularAgent
from src.environment import BucketConfig, MaintenanceEnv, RewardConfig
from src.evaluate import evaluate_agent, summarize_episodes
from src.policies import apply_safety_override
from src.train import evaluate_rule_policy, train_agent

# ── fixed experiment settings ────────────────────────────────────────────────
SEEDS = [42, 7, 123]
EPISODES = 500
PLANNING_STEPS = [0, 5, 10, 20]
CONVERGENCE_WINDOW = 50       # rolling window for episodes-to-threshold
CONVERGENCE_THRESHOLD = 0.0   # break-even reward level
GREEDY_EVAL_INTERVAL = 50     # run greedy-policy eval every N training episodes

# ── Phase 2 sensitivity settings ─────────────────────────────────────────────
SENS_SEEDS = [42, 7, 123, 0, 99]   # 5 seeds for more robust sensitivity estimates
FAILURE_PENALTY_VARIANTS = [-50.0, -200.0]   # default = -100
COST_SCALE_VARIANTS = [0.5, 2.0]             # multiplier for action costs; default = 1.0
CRITICAL_MAX_VARIANTS = [1.15, 1.55]         # default = 1.35  (±0.20)
MODERATE_MAX_VARIANTS = [0.03, 0.08]         # default = 0.05


# ── helpers ──────────────────────────────────────────────────────────────────

def _aggregate(per_seed_metrics: list[dict]) -> dict:
    """Compute mean ± std across seeds for every metric.

    Note: std_reward_mean is the mean of per-seed within-episode reward stds,
    not the pooled episode-level std.  Use it only as a rough spread indicator.
    """
    agg = {}
    for key in per_seed_metrics[0]:
        vals = [m[key] for m in per_seed_metrics]
        agg[f"{key}_mean"] = statistics.mean(vals)
        agg[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return agg


def _print_agg(label: str, agg: dict) -> None:
    print(f"    {label:<24}  "
          f"reward {agg['average_reward_mean']:7.2f} ±{agg['average_reward_std']:5.2f}  |  "
          f"fail {agg['failure_rate_mean']:.3f} ±{agg['failure_rate_std']:.3f}  |  "
          f"early_heavy {agg['early_replacement_rate_mean']:.3f} ±{agg['early_replacement_rate_std']:.3f}  |  "
          f"prem_any {agg['premature_any_rate_mean']:.3f} ±{agg['premature_any_rate_std']:.3f}  |  "
          f"cycle {agg['mean_action_cycle_mean']:6.1f} ±{agg['mean_action_cycle_std']:5.1f}")


def _convergence_stats(
    curves: list[list[float]],
    window: int = CONVERGENCE_WINDOW,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> dict:
    """Compute convergence metrics from per-seed training reward curves.

    auc             mean per-episode reward over all training episodes
                    (higher = better overall sample efficiency)
    eps_to_threshold  first episode where the rolling-window mean >= threshold
                    (lower = faster convergence; reports n_episodes if never reached)
    """
    n_episodes = len(curves[0])
    aucs: list[float] = []
    eps_to_thresh: list[float] = []

    for curve in curves:
        aucs.append(statistics.mean(curve))
        hit: float = float(n_episodes)   # default: never reached
        for i in range(window, n_episodes + 1):
            if statistics.mean(curve[i - window : i]) >= threshold:
                hit = float(i)
                break
        eps_to_thresh.append(hit)

    return {
        "auc_mean": statistics.mean(aucs),
        "auc_std": statistics.stdev(aucs) if len(aucs) > 1 else 0.0,
        "eps_to_threshold_mean": statistics.mean(eps_to_thresh),
        "eps_to_threshold_std": statistics.stdev(eps_to_thresh) if len(eps_to_thresh) > 1 else 0.0,
    }


def _greedy_convergence_stats(
    greedy_curves: list[list[tuple[int, float]]],
    threshold: float = CONVERGENCE_THRESHOLD,
    n_episodes: int = EPISODES,
) -> dict:
    """Compute convergence metrics from periodic greedy-policy eval checkpoints.

    Uses the greedy (non-exploratory) reward signal, which is a cleaner convergence
    indicator than the exploratory training reward used by _convergence_stats.

    greedy_auc              mean greedy reward across all checkpoints
    greedy_eps_to_threshold first checkpoint episode where greedy reward >= threshold
                            (reports n_episodes if never reached)
    """
    aucs: list[float] = []
    eps_to_thresh: list[float] = []

    for curve in greedy_curves:
        if not curve:
            aucs.append(float("nan"))
            eps_to_thresh.append(float(n_episodes))
            continue
        aucs.append(statistics.mean(r for (_, r) in curve))
        hit = float(n_episodes)
        for ep, r in curve:
            if r >= threshold:
                hit = float(ep)
                break
        eps_to_thresh.append(hit)

    valid_aucs = [a for a in aucs if a == a]  # exclude nan
    return {
        "greedy_auc_mean": statistics.mean(valid_aucs) if valid_aucs else float("nan"),
        "greedy_auc_std": statistics.stdev(valid_aucs) if len(valid_aucs) > 1 else 0.0,
        "greedy_eps_to_threshold_mean": statistics.mean(eps_to_thresh),
        "greedy_eps_to_threshold_std": statistics.stdev(eps_to_thresh) if len(eps_to_thresh) > 1 else 0.0,
    }


def _count_override_triggers(
    agent: TabularAgent,
    eval_env: MaintenanceEnv,
) -> dict:
    """Count how many times each override rule fires during a greedy evaluation pass.

    Returns counts over all eval units (not per-unit averages).
    Rule 1 — IMMINENT + {CONTINUE, INSPECT} → MAJOR_OVERHAUL
    Rule 2 — anomaly_flag=1 + CONTINUE      → INSPECT
    """
    imminent_count = 0
    anomaly_count = 0

    for unit_id in eval_env.unit_ids:
        state = eval_env.reset(unit_id=unit_id)
        while True:
            action = agent.select_action(state, explore=False)
            rul_bucket, _, anomaly_flag = state
            if rul_bucket == "IMMINENT" and action in {"CONTINUE", "INSPECT"}:
                imminent_count += 1
            elif anomaly_flag == 1 and action == "CONTINUE":
                anomaly_count += 1
            next_state, _, done, _ = eval_env.step(apply_safety_override(action, state))
            if done:
                break
            state = next_state

    return {"imminent_overrides": imminent_count, "anomaly_overrides": anomaly_count}


def run_seeded(
    agent_factory: Callable[[int], TabularAgent],
    split: SplitSubset,
    seeds: list[int] = SEEDS,
    episodes: int = EPISODES,
    use_safety_override: bool = False,
    greedy_eval_interval: int = GREEDY_EVAL_INTERVAL,
    reward_config: RewardConfig | None = None,
    bucket_config: BucketConfig | None = None,
) -> dict:
    """Train and evaluate a fresh agent for each seed.

    The train/eval split is fixed (same for all seeds).
    Each seed re-instantiates agent, train_env, and eval_env for full reproducibility.

    Returns
    -------
    curves            per-seed list of per-episode exploratory training rewards
    greedy_curves     per-seed list of (episode, mean_greedy_reward) checkpoints
                      collected every greedy_eval_interval episodes during training
    summaries         per-seed list of episode evaluation dicts (final policy)
    per_seed_metrics  per-seed summarize_episodes output
    aggregated        mean ± std across seeds
    """
    curves: list[list[float]] = []
    greedy_curves: list[list[tuple[int, float]]] = []
    all_summaries: list[list[dict]] = []

    for seed in seeds:
        train_env = MaintenanceEnv(split.train, rng_seed=seed,
                                   reward_config=reward_config, bucket_config=bucket_config)
        eval_env = MaintenanceEnv(split.eval, rng_seed=seed,
                                  reward_config=reward_config, bucket_config=bucket_config)
        agent = agent_factory(seed)

        greedy_checkpoints: list[tuple[int, float]] = []

        def _greedy_callback(
            ep: int,
            _agent: TabularAgent = agent,
            _eval_env: MaintenanceEnv = eval_env,
            _use_override: bool = use_safety_override,
        ) -> None:
            sums = evaluate_agent(_eval_env, _agent, use_safety_override=_use_override)
            mean_r = statistics.mean(s["total_reward"] for s in sums)
            greedy_checkpoints.append((ep, mean_r))

        curve = train_agent(
            train_env, agent,
            episodes=episodes,
            rng_seed=seed,
            use_safety_override=use_safety_override,
            greedy_eval_callback=_greedy_callback,
            greedy_eval_interval=greedy_eval_interval,
        )
        summaries = evaluate_agent(eval_env, agent, use_safety_override=use_safety_override)
        curves.append(curve)
        greedy_curves.append(greedy_checkpoints)
        all_summaries.append(summaries)

    per_seed_metrics = [summarize_episodes(s) for s in all_summaries]
    return {
        "curves": curves,
        "greedy_curves": greedy_curves,
        "summaries": all_summaries,
        "per_seed_metrics": per_seed_metrics,
        "aggregated": _aggregate(per_seed_metrics),
    }


# ── experiments ──────────────────────────────────────────────────────────────

def experiment1_planning_ablation(split: SplitSubset) -> dict:
    """Experiment 1: DynaQ planning-step ablation over n in {0, 5, 10, 20}.

    n=0 is the Q-learning-equivalent control (no planning updates).
    Reports both final eval metrics and convergence metrics from training curves.
    Best n is selected by highest mean average reward; ties broken by parsimony (lowest n).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 — Planning-Step Ablation")
    print(f"  Convergence: rolling window={CONVERGENCE_WINDOW} eps, threshold={CONVERGENCE_THRESHOLD}")
    print(f"  Greedy eval: every {GREEDY_EVAL_INTERVAL} episodes (cleaner signal than exploratory AUC)")
    print("=" * 80)
    print(f"    {'label':<24}  {'reward':>13}        {'fail':>10}      "
          f"{'early_heavy':>13}      {'prem_any':>10}      {'cycle':>13}")
    print("    " + "-" * 86)

    by_n: dict[int, dict] = {}
    for n in PLANNING_STEPS:
        by_n[n] = run_seeded(
            lambda seed, n=n: DynaQAgent(planning_steps=n, rng_seed=seed),
            split,
        )
        _print_agg(f"DynaQ n={n}", by_n[n]["aggregated"])
        conv = _convergence_stats(by_n[n]["curves"])
        gconv = _greedy_convergence_stats(by_n[n]["greedy_curves"])
        by_n[n]["convergence"] = conv
        by_n[n]["greedy_convergence"] = gconv
        print(f"      {'':24}  exploratory AUC {conv['auc_mean']:6.1f} ±{conv['auc_std']:.1f}  |  "
              f"eps_to_threshold {conv['eps_to_threshold_mean']:.0f} ±{conv['eps_to_threshold_std']:.0f}")
        print(f"      {'':24}  greedy AUC      {gconv['greedy_auc_mean']:6.1f} ±{gconv['greedy_auc_std']:.1f}  |  "
              f"greedy_eps_to_threshold {gconv['greedy_eps_to_threshold_mean']:.0f} "
              f"±{gconv['greedy_eps_to_threshold_std']:.0f}")

    # Tie-aware best_n selection
    reward_vals = {n: by_n[n]["aggregated"]["average_reward_mean"] for n in PLANNING_STEPS}
    best_reward = max(reward_vals.values())
    tied = all(v == best_reward for v in reward_vals.values())

    if tied:
        best_n = 0   # parsimony: fewest planning steps
        print(f"\n  → All n values tied on final eval metrics (reward={best_reward:.2f}).")
        print(f"    n=0 selected by parsimony (Q-learning equivalent, fewest planning steps).")
        print(f"    Convergence metrics above show whether planning affected learning speed.")
        # Exp 1 and Exp 2 share the same eval set; no information leaked because all n tied.
        print(f"    (Note: Exp 1 and Exp 2 share the same eval set — no bias here because all n tied.)")
    else:
        best_n = min(n for n in PLANNING_STEPS if reward_vals[n] == best_reward)
        print(f"\n  → Best n = {best_n}  (highest mean avg reward={best_reward:.2f})")
        # WARNING: Exp 1 and Exp 2 share the same eval set, and n values are NOT tied.
        # DynaQ(best_n) results in Exp 2 may be inflated by selection bias.
        # Use separate validation/test splits for strictly fair reporting.
        print(f"    WARNING: Exp 1 and Exp 2 share the same eval set; n values are not tied.")
        print(f"    DynaQ(n={best_n}) results in Exp 2 may be inflated by selection bias.")

    return {"by_n": by_n, "best_n": best_n}


def experiment2_algorithm_comparison(split: SplitSubset, best_n: int) -> dict:
    """Experiment 2: Compare rule-based, SARSA, Q-learning, and DynaQ(best_n).

    Rule-based is deterministic and requires no training.
    All learning agents use the same SEEDS for a fair comparison.
    'early_heavy' = early_replacement_rate: MAJOR_OVERHAUL/REPLACE in HEALTHY state only.
    'prem_any'    = any terminal maintenance action in HEALTHY or WATCH.
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 2 — Algorithm Comparison  [DynaQ n={best_n}]")
    print("  early_heavy = MAJOR_OVERHAUL/REPLACE in HEALTHY only")
    print("  prem_any    = any terminal maintenance action in HEALTHY or WATCH")
    print("                (catches premature MINOR_REPAIR that early_heavy misses)")
    print("=" * 80)
    print(f"    {'label':<24}  {'reward':>13}        {'fail':>10}      "
          f"{'early_heavy':>13}      {'prem_any':>10}      {'cycle':>13}")
    print("    " + "-" * 86)

    results: dict[str, dict] = {}

    # Rule-based: deterministic — eval output is identical across seeds,
    # but we run once per seed for a consistent framework
    rule_per_seed = [
        summarize_episodes(evaluate_rule_policy(MaintenanceEnv(split.eval, rng_seed=s)))
        for s in SEEDS
    ]
    results["rule_based"] = {
        "per_seed_metrics": rule_per_seed,
        "aggregated": _aggregate(rule_per_seed),
    }
    _print_agg("rule_based", results["rule_based"]["aggregated"])

    results["sarsa"] = run_seeded(lambda seed: SARSAgent(rng_seed=seed), split)
    _print_agg("sarsa", results["sarsa"]["aggregated"])

    results["q_learning"] = run_seeded(lambda seed: QLearningAgent(rng_seed=seed), split)
    _print_agg("q_learning", results["q_learning"]["aggregated"])

    results["dyna_q"] = run_seeded(
        lambda seed, n=best_n: DynaQAgent(planning_steps=n, rng_seed=seed),
        split,
    )
    _print_agg(f"dyna_q (n={best_n})", results["dyna_q"]["aggregated"])

    print("\n  Greedy convergence (mean ± std across seeds):")
    for label, key in [("sarsa", "sarsa"), ("q_learning", "q_learning"),
                       (f"dyna_q (n={best_n})", "dyna_q")]:
        gconv = _greedy_convergence_stats(results[key]["greedy_curves"])
        results[key]["greedy_convergence"] = gconv
        print(f"    {label:<24}  greedy AUC {gconv['greedy_auc_mean']:6.1f} ±{gconv['greedy_auc_std']:.1f}  |  "
              f"greedy_eps_to_threshold {gconv['greedy_eps_to_threshold_mean']:.0f} "
              f"±{gconv['greedy_eps_to_threshold_std']:.0f}")

    # SARSA note: prem_any captures the cycle-1 early intervention that early_heavy misses.
    # The on-policy update mechanism is one plausible explanation, but the causal link
    # is not confirmed without further ablations.
    sarsa_prem = results["sarsa"]["aggregated"]["premature_any_rate_mean"]
    print(f"\n  SARSA prem_any={sarsa_prem:.3f}: agent intervenes before CRITICAL in most episodes.")
    print(f"  (mean_cycle~1.2; early_heavy=0 because it uses MINOR_REPAIR, not heavy actions)")

    return results


def experiment3_safety_ablation(split: SplitSubset, best_n: int) -> dict:
    """Experiment 3: Safety-override ablation on the best learned policy.

    Training is always done WITHOUT the override so the raw learned policy is
    inspectable. The override is applied only at evaluation time in the
    'with_override' condition. This separates what the agent learned from
    what the guardrail adds.

    Override trigger counts are measured directly to support attribution claims.
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 3 — Safety-Override Ablation  [DynaQ n={best_n}]")
    print("=" * 80)
    print(f"    {'label':<24}  {'reward':>13}        {'fail':>10}      "
          f"{'early_heavy':>13}      {'prem_any':>10}      {'cycle':>13}")
    print("    " + "-" * 86)

    base_summaries: list[list[dict]] = []
    override_summaries: list[list[dict]] = []
    curves: list[list[float]] = []
    greedy_curves: list[list[tuple[int, float]]] = []
    all_trigger_counts: list[dict] = []

    for seed in SEEDS:
        # Train without override — raw learned policy
        train_env = MaintenanceEnv(split.train, rng_seed=seed)
        eval_env_for_greedy = MaintenanceEnv(split.eval, rng_seed=seed)
        agent = DynaQAgent(planning_steps=best_n, rng_seed=seed)

        greedy_checkpoints: list[tuple[int, float]] = []

        def _greedy_callback(
            ep: int,
            _agent: TabularAgent = agent,
            _env: MaintenanceEnv = eval_env_for_greedy,
        ) -> None:
            sums = evaluate_agent(_env, _agent, use_safety_override=False)
            mean_r = statistics.mean(s["total_reward"] for s in sums)
            greedy_checkpoints.append((ep, mean_r))

        curve = train_agent(train_env, agent, episodes=EPISODES, rng_seed=seed,
                            use_safety_override=False,
                            greedy_eval_callback=_greedy_callback,
                            greedy_eval_interval=GREEDY_EVAL_INTERVAL)
        curves.append(curve)
        greedy_curves.append(greedy_checkpoints)

        # Evaluate the same trained agent without override (base)
        base_summaries.append(
            evaluate_agent(MaintenanceEnv(split.eval, rng_seed=seed), agent,
                           use_safety_override=False)
        )
        # Evaluate the same trained agent with override
        override_summaries.append(
            evaluate_agent(MaintenanceEnv(split.eval, rng_seed=seed), agent,
                           use_safety_override=True)
        )
        # Count which override rules fired and how many times
        all_trigger_counts.append(
            _count_override_triggers(agent, MaintenanceEnv(split.eval, rng_seed=seed))
        )

    base_per_seed = [summarize_episodes(s) for s in base_summaries]
    override_per_seed = [summarize_episodes(s) for s in override_summaries]
    base_agg = _aggregate(base_per_seed)
    override_agg = _aggregate(override_per_seed)

    _print_agg("base (no override)", base_agg)
    _print_agg("with override", override_agg)

    # Report measured trigger counts
    avg_imminent = statistics.mean(c["imminent_overrides"] for c in all_trigger_counts)
    avg_anomaly = statistics.mean(c["anomaly_overrides"] for c in all_trigger_counts)
    n_eval = split.eval["unit"].nunique()
    print(f"\n  Override trigger counts (mean across {len(SEEDS)} seeds, {n_eval} eval units):")
    print(f"    Rule 1 — IMMINENT+passive → MAJOR_OVERHAUL : {avg_imminent:.1f} total triggers")
    print(f"    Rule 2 — anomaly+CONTINUE → INSPECT        : {avg_anomaly:.1f} total triggers")

    bf = base_agg["failure_rate_mean"]
    of = override_agg["failure_rate_mean"]
    be = base_agg["early_replacement_rate_mean"]
    oe = override_agg["early_replacement_rate_mean"]
    reward_delta = override_agg["average_reward_mean"] - base_agg["average_reward_mean"]
    print(f"\n  Failure rate change:        {bf:.3f} → {of:.3f}  (Δ {of - bf:+.3f})")
    print(f"  Early heavy interv. change: {be:.3f} → {oe:.3f}  (Δ {oe - be:+.3f})")
    print(f"  Reward change:              {base_agg['average_reward_mean']:.2f} → "
          f"{override_agg['average_reward_mean']:.2f}  (Δ {reward_delta:+.2f})")

    return {
        "curves": curves,
        "greedy_curves": greedy_curves,
        "trigger_counts": all_trigger_counts,
        "base": {"summaries": base_summaries, "per_seed_metrics": base_per_seed,
                 "aggregated": base_agg},
        "with_override": {"summaries": override_summaries,
                          "per_seed_metrics": override_per_seed,
                          "aggregated": override_agg},
    }


# ── Phase 2 helpers ──────────────────────────────────────────────────────────

def _scale_action_costs(scale: float) -> RewardConfig:
    """Return a RewardConfig with the four direct action costs multiplied by `scale`.

    Scaled (×scale): inspect_cost, minor_repair_cost, major_overhaul_cost, replacement_cost.
    Not scaled: failure_penalty, timely_bonus, late_maintenance_penalty,
                premature_heavy_penalty, premature_minor_penalty, watch_replace_penalty.
    Leaving these six unchanged ensures the question is purely about whether Q-learning's
    timing survives a changed action-cost/bonus ratio.
    """
    d = RewardConfig()
    return RewardConfig(
        failure_penalty=d.failure_penalty,
        inspect_cost=d.inspect_cost * scale,
        minor_repair_cost=d.minor_repair_cost * scale,
        major_overhaul_cost=d.major_overhaul_cost * scale,
        replacement_cost=d.replacement_cost * scale,
        timely_bonus=d.timely_bonus,
        late_maintenance_penalty=d.late_maintenance_penalty,
        premature_heavy_penalty=d.premature_heavy_penalty,
        premature_minor_penalty=d.premature_minor_penalty,
        watch_replace_penalty=d.watch_replace_penalty,
    )


def _run_variant(
    split: SplitSubset,
    seeds: list[int],
    reward_config: RewardConfig | None = None,
    bucket_config: BucketConfig | None = None,
    best_n: int = 0,
) -> dict:
    """Run all four algorithms for one sensitivity variant."""
    result: dict = {}

    result["rule_based"] = {
        "per_seed_metrics": [
            summarize_episodes(evaluate_rule_policy(
                MaintenanceEnv(split.eval, rng_seed=s,
                               reward_config=reward_config, bucket_config=bucket_config)
            ))
            for s in seeds
        ],
    }
    result["rule_based"]["aggregated"] = _aggregate(result["rule_based"]["per_seed_metrics"])

    result["sarsa"] = run_seeded(
        lambda seed: SARSAgent(rng_seed=seed),
        split, seeds=seeds,
        reward_config=reward_config, bucket_config=bucket_config,
    )
    result["q_learning"] = run_seeded(
        lambda seed: QLearningAgent(rng_seed=seed),
        split, seeds=seeds,
        reward_config=reward_config, bucket_config=bucket_config,
    )
    result["dyna_q"] = run_seeded(
        lambda seed, n=best_n: DynaQAgent(planning_steps=n, rng_seed=seed),
        split, seeds=seeds,
        reward_config=reward_config, bucket_config=bucket_config,
    )

    return result


def _print_variant_summary(label: str, variant: dict) -> None:
    """Print a one-section comparison for a single sensitivity variant."""
    print(f"\n  ── {label}")
    for alg_key in ("rule_based", "sarsa", "q_learning", "dyna_q"):
        if alg_key not in variant:
            continue
        agg = variant[alg_key]["aggregated"]
        _print_agg(f"    {alg_key}", agg)
    ql_r = variant["q_learning"]["aggregated"]["average_reward_mean"]
    ql_fail = variant["q_learning"]["aggregated"]["failure_rate_mean"]
    rb_r = variant["rule_based"]["aggregated"]["average_reward_mean"]
    ranking_ok = ql_r > rb_r and ql_fail == 0.0
    print(f"    → Q-learning ({ql_r:.2f}, fail={ql_fail:.3f}) {'>' if ql_r > rb_r else '<='} "
          f"rule-based ({rb_r:.2f})  ranking {'MAINTAINED ✓' if ranking_ok else 'CHANGED ✗'}")


def experiment4_sensitivity(split: SplitSubset, best_n: int = 0) -> dict:
    """Experiment 4: Sensitivity analysis.

    P2.1 — Reward scaling
        (a) Failure penalty: {-50, -200}  (default = -100)
            Key question: does SARSA recover? does Q-learning ranking hold?
        (b) Action cost scale: ×0.5, ×2.0  (default = ×1.0)
            Key question: does Q-learning's CRITICAL-timing survive a changed cost/bonus ratio?

    P2.2 — Discretization sensitivity
        (a) critical_max shift: {1.15, 1.55}  (default = 1.35)
            Key question: does Q-learning adapt to a narrower/wider CRITICAL window?
        (b) moderate_max shift: {0.03, 0.08}  (default = 0.05)
            Key question: does changing the degradation boundary affect convergence?

    Uses SENS_SEEDS (5 seeds) for more robust estimates than the 3-seed Phase 1 runs.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 — Sensitivity Analysis")
    print(f"  Seeds: {SENS_SEEDS}   Episodes: {EPISODES}")
    print("=" * 80)
    print(f"    {'label':<28}  {'reward':>13}        {'fail':>10}      "
          f"{'early_heavy':>13}      {'prem_any':>10}      {'cycle':>13}")

    results: dict = {}

    # ── baseline ─────────────────────────────────────────────────────────────
    print("\n  BASELINE (default RewardConfig, default BucketConfig)")
    baseline = _run_variant(split, SENS_SEEDS, best_n=best_n)
    results["baseline"] = baseline
    _print_variant_summary("baseline", baseline)

    # ── P2.1a — failure penalty ───────────────────────────────────────────────
    print("\n  P2.1a — Failure penalty variants  (SARSA included to test recovery)")
    results["p2_1a_failure_penalty"] = {}
    for fp in FAILURE_PENALTY_VARIANTS:
        rc = RewardConfig(failure_penalty=fp)
        v = _run_variant(split, SENS_SEEDS, reward_config=rc, best_n=best_n)
        results["p2_1a_failure_penalty"][str(fp)] = v
        _print_variant_summary(f"failure_penalty={fp}", v)

    # ── P2.1b — action cost scale ─────────────────────────────────────────────
    print("\n  P2.1b — Action cost scale variants")
    results["p2_1b_cost_scale"] = {}
    for scale in COST_SCALE_VARIANTS:
        rc = _scale_action_costs(scale)
        v = _run_variant(split, SENS_SEEDS, reward_config=rc, best_n=best_n)
        results["p2_1b_cost_scale"][str(scale)] = v
        _print_variant_summary(f"cost_scale=×{scale}", v)

    # ── P2.2a — critical_max (CRITICAL/IMMINENT boundary) ────────────────────
    print(f"\n  P2.2a — CRITICAL/IMMINENT boundary variants  (default critical_max=1.35)")
    results["p2_2a_critical_max"] = {}
    for cm in CRITICAL_MAX_VARIANTS:
        bc = BucketConfig(critical_max=cm)
        v = _run_variant(split, SENS_SEEDS, bucket_config=bc, best_n=best_n)
        results["p2_2a_critical_max"][str(cm)] = v
        _print_variant_summary(f"critical_max={cm}", v)

    # ── P2.2b — moderate_max (MODERATE/FAST boundary) ────────────────────────
    print(f"\n  P2.2b — MODERATE/FAST degradation boundary variants  (default moderate_max=0.05)")
    results["p2_2b_moderate_max"] = {}
    for mm in MODERATE_MAX_VARIANTS:
        bc = BucketConfig(moderate_max=mm)
        v = _run_variant(split, SENS_SEEDS, bucket_config=bc, best_n=best_n)
        results["p2_2b_moderate_max"][str(mm)] = v
        _print_variant_summary(f"moderate_max={mm}", v)

    # ── summary verdict ───────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  SENSITIVITY SUMMARY")
    print("─" * 80)
    all_variants = (
        list(results["p2_1a_failure_penalty"].items())
        + list(results["p2_1b_cost_scale"].items())
        + list(results["p2_2a_critical_max"].items())
        + list(results["p2_2b_moderate_max"].items())
    )
    n_maintained = sum(
        v["q_learning"]["aggregated"]["average_reward_mean"]
        > v["rule_based"]["aggregated"]["average_reward_mean"]
        and v["q_learning"]["aggregated"]["failure_rate_mean"] == 0.0
        for _, v in all_variants
    )
    print(f"  Q-learning > rule-based in {n_maintained}/{len(all_variants)} variants "
          f"({'ROBUST' if n_maintained == len(all_variants) else 'NOT FULLY ROBUST'})")

    return results


# ── result persistence ────────────────────────────────────────────────────────

def _serialize(obj: object) -> object:
    """Recursively convert non-JSON-serializable types to serializable ones."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_serialize(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):   # nan
        return None
    return obj


def save_results(
    split_meta: dict,
    exp1: dict,
    exp2: dict,
    exp3: dict,
    exp4: dict | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Write per-seed metrics, aggregates, convergence stats, and curves to JSON.

    Both exploratory training curves and greedy checkpoint curves are included
    to support Phase 3 convergence plotting.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _exp1_payload(by_n: dict) -> dict:
        return {
            str(n): {
                "aggregated": data["aggregated"],
                "per_seed_metrics": data["per_seed_metrics"],
                "curves": data.get("curves", []),
                "greedy_curves": data.get("greedy_curves", []),
                "convergence": data.get("convergence", {}),
                "greedy_convergence": data.get("greedy_convergence", {}),
            }
            for n, data in by_n.items()
        }

    def _alg_payload(data: dict) -> dict:
        return {
            "aggregated": data["aggregated"],
            "per_seed_metrics": data["per_seed_metrics"],
            "curves": data.get("curves", []),
            "greedy_curves": data.get("greedy_curves", []),
            "greedy_convergence": data.get("greedy_convergence", {}),
        }

    def _variant_payload(variant: dict) -> dict:
        payload: dict = {}
        for alg in ("rule_based", "sarsa", "q_learning", "dyna_q"):
            if alg in variant:
                payload[alg] = {
                    "aggregated": variant[alg]["aggregated"],
                    "per_seed_metrics": variant[alg]["per_seed_metrics"],
                }
        return payload

    def _exp4_payload(exp4_results: dict) -> dict:
        out: dict = {}
        for group_key, group_val in exp4_results.items():
            if group_key == "baseline":
                out["baseline"] = _variant_payload(group_val)
            else:
                out[group_key] = {k: _variant_payload(v) for k, v in group_val.items()}
        return out

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "seeds": SEEDS,
            "sens_seeds": SENS_SEEDS,
            "episodes": EPISODES,
            "planning_steps": PLANNING_STEPS,
            "greedy_eval_interval": GREEDY_EVAL_INTERVAL,
            "convergence_window": CONVERGENCE_WINDOW,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "failure_penalty_variants": FAILURE_PENALTY_VARIANTS,
            "cost_scale_variants": COST_SCALE_VARIANTS,
            "critical_max_variants": CRITICAL_MAX_VARIANTS,
            "moderate_max_variants": MODERATE_MAX_VARIANTS,
        },
        "split": split_meta,
        "experiment1": {
            "best_n": exp1["best_n"],
            "by_n": _exp1_payload(exp1["by_n"]),
        },
        "experiment2": {
            alg: _alg_payload(exp2[alg])
            for alg in ("rule_based", "sarsa", "q_learning", "dyna_q")
        },
        "experiment3": {
            "base": {
                "aggregated": exp3["base"]["aggregated"],
                "per_seed_metrics": exp3["base"]["per_seed_metrics"],
            },
            "with_override": {
                "aggregated": exp3["with_override"]["aggregated"],
                "per_seed_metrics": exp3["with_override"]["per_seed_metrics"],
            },
            "trigger_counts": exp3["trigger_counts"],
            "greedy_curves": exp3.get("greedy_curves", []),
        },
    }
    if exp4 is not None:
        payload["experiment4"] = _exp4_payload(exp4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{timestamp}.json"
    out_path.write_text(json.dumps(_serialize(payload), indent=2))
    return out_path


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Preparing FD001 (leakage-free split, seed=42) ...")
    split = prepare_split("FD001", train_frac=0.8, seed=42)
    print(f"  Train units : {split.train['unit'].nunique()}")
    print(f"  Eval  units : {split.eval['unit'].nunique()}")
    print(f"  Sensors     : {split.selected_sensors}")
    print(f"  Seeds       : {SEEDS}   Episodes: {EPISODES}")

    exp1 = experiment1_planning_ablation(split)
    best_n = exp1["best_n"]

    exp2 = experiment2_algorithm_comparison(split, best_n)

    exp3 = experiment3_safety_ablation(split, best_n)

    exp4 = experiment4_sensitivity(split, best_n)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

    split_meta = {
        "subset": split.subset,
        "train_units": int(split.train["unit"].nunique()),
        "eval_units": int(split.eval["unit"].nunique()),
        "selected_sensors": list(split.selected_sensors),
    }
    out_path = save_results(split_meta, exp1, exp2, exp3, exp4)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
