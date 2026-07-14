from __future__ import annotations


def apply_safety_override(action: str, state: tuple[str, str, int]) -> str:
    rul_bucket, _, anomaly_flag = state
    if rul_bucket == "IMMINENT" and action in {"CONTINUE", "INSPECT"}:
        return "MAJOR_OVERHAUL"
    if anomaly_flag == 1 and action == "CONTINUE":
        return "INSPECT"
    return action


class RuleBasedPolicy:
    def select_action(self, state: tuple[str, str, int]) -> str:
        rul_bucket, degradation_bucket, anomaly_flag = state
        if rul_bucket == "IMMINENT":
            return "REPLACE"
        if rul_bucket == "CRITICAL" and degradation_bucket == "FAST":
            return "MAJOR_OVERHAUL"
        if rul_bucket in {"CRITICAL", "WATCH"} and degradation_bucket == "MODERATE":
            return "MINOR_REPAIR"
        if anomaly_flag == 1:
            return "INSPECT"
        return "CONTINUE"
