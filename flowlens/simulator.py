"""
simulator.py — Before/After workflow improvement simulator.

Takes proposed workflow changes (e.g., reduce PR review time from 26h to 4h),
applies them as deltas to the current developer-day feature vector, re-runs
the trained Isolation Forest, and returns a projected flow score.

Key design rule:
  - Only modify features that are MEANINGFUL for this developer.
  - If a feature is 0 in both the current row AND the team baseline,
    it means that data was unavailable (blobless clone, no CI logs, etc.).
    Setting it to a non-zero value would make it look anomalous to the model.
    These features are skipped entirely.
  - Projected score is ALWAYS >= current score.
    The simulator shows optimistic improvement scenarios, never degradation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from flowlens.features import FEATURE_NAMES

logger = logging.getLogger("flowlens.simulator")

DEFAULT_HOURS_PER_FLOW_POINT = 0.5

# Features that rely on external data sources (GitHub API / CI logs).
# If their baseline is 0 across the team, the data was unavailable.
# Writing a non-zero value for these would confuse the model.
_EXTERNAL_DATA_FEATURES = {
    "pr_blocking_time_hours",
    "pr_size_lines",
    "test_failure_density",
    "build_time_minutes",
}


def _feature_is_meaningful(
    feature: str,
    current_value: float,
    team_baseline: float,
) -> bool:
    """
    Return True if this feature has real signal and can safely be modified.

    A feature is NOT meaningful (and should be skipped) when:
      - It's an external-data feature (PR/CI) AND
      - Both the current value AND team baseline are 0,
        meaning the data source was unavailable during ingestion.
    """
    if feature not in _EXTERNAL_DATA_FEATURES:
        return True  # Git-derived features are always meaningful
    # External feature is meaningful only if there's actual data
    return current_value > 0 or team_baseline > 0


def build_feature_deltas(
    changes: dict[str, Any],
    current_raw_features: dict[str, float],
    team_baselines: dict[str, float],
) -> dict[str, float]:
    """
    Translate high-level workflow parameter changes into raw feature deltas.

    Only modifies features that have real signal. Skips features where both
    current value and team baseline are 0 (data unavailable).

    Args:
        changes:              Dict of {param_name: target_value}.
        current_raw_features: Current feature values for this developer-day.
        team_baselines:       Mean feature values across all developer-days
                              (used to detect unavailable data sources).

    Returns:
        Modified feature dict with only meaningful changes applied.
    """
    modified = dict(current_raw_features)

    # ── PR review time
    if "pr_review_time_hours" in changes:
        feat = "pr_blocking_time_hours"
        if _feature_is_meaningful(feat, current_raw_features.get(feat, 0), team_baselines.get(feat, 0)):
            target = float(changes["pr_review_time_hours"])
            current = current_raw_features.get(feat, 0)
            # Only apply if target is actually an improvement (lower than current)
            if current > target or current > 0:
                modified[feat] = max(0.0, target)

    # ── Max PR size
    if "max_pr_size_lines" in changes:
        feat = "pr_size_lines"
        if _feature_is_meaningful(feat, current_raw_features.get(feat, 0), team_baselines.get(feat, 0)):
            target = float(changes["max_pr_size_lines"])
            current = current_raw_features.get(feat, 0)
            if current > target or current > 0:
                modified[feat] = max(0.0, target)
                # Smaller PRs → smaller avg commit size
                if current_raw_features.get("avg_commit_size_lines", 0) > target / 2:
                    modified["avg_commit_size_lines"] = target / 2

    # ── Build time
    if "build_time_minutes" in changes:
        feat = "build_time_minutes"
        if _feature_is_meaningful(feat, current_raw_features.get(feat, 0), team_baselines.get(feat, 0)):
            target = float(changes["build_time_minutes"])
            current = current_raw_features.get(feat, 0)
            if current > target or current > 0:
                modified[feat] = max(0.0, target)
                # Faster builds → shorter inter-commit gaps
                time_saved = current - target
                if time_saved > 0:
                    current_gap = current_raw_features.get("inter_commit_gap_mean", 0)
                    modified["inter_commit_gap_mean"] = max(0.0, current_gap - time_saved)

    # ── Focus block length — always meaningful (git-derived features)
    if "focus_block_minutes" in changes:
        target = float(changes["focus_block_minutes"])
        if target >= 90:
            variance_reduction = 0.3
        elif target >= 60:
            variance_reduction = 0.15
        else:
            variance_reduction = 0.0
        current_variance = current_raw_features.get("inter_commit_gap_variance", 0)
        modified["inter_commit_gap_variance"] = max(
            0.0, current_variance * (1 - variance_reduction)
        )
        # Longer focus blocks → less late-night work
        current_lnr = current_raw_features.get("late_night_ratio", 0)
        modified["late_night_ratio"] = max(0.0, current_lnr * 0.6)

    # ── Meeting cap — always meaningful
    if "meeting_hours_per_day_cap" in changes:
        target = float(changes["meeting_hours_per_day_cap"])
        gap_reduction = 0.4 if target <= 2 else (0.2 if target <= 4 else 0.0)
        current_variance = modified.get(
            "inter_commit_gap_variance",
            current_raw_features.get("inter_commit_gap_variance", 0),
        )
        modified["inter_commit_gap_variance"] = max(
            0.0, current_variance * (1 - gap_reduction)
        )

    # Clamp all to non-negative
    for key in modified:
        if isinstance(modified[key], (int, float)):
            modified[key] = max(0.0, float(modified[key]))

    return modified


# ---------------------------------------------------------------------------
# Simulator core
# ---------------------------------------------------------------------------

def simulate_changes(
    developer_email: str,
    date: str,
    changes: dict[str, Any],
    hours_per_flow_point: float = DEFAULT_HOURS_PER_FLOW_POINT,
) -> dict[str, Any]:
    """
    Simulate the projected flow score improvement for a developer-day.

    Always returns projected_score >= current_score.
    """
    from flowlens import model as _model_module
    from flowlens.db import get_shap_values, save_simulation, get_connection

    if not _model_module.is_model_ready():
        raise RuntimeError("Model is not trained. Cannot run simulation.")

    row = get_shap_values(date, developer_email)
    if row is None:
        raise ValueError(
            f"No data found for developer={developer_email}, date={date}"
        )

    current_score = float(row.get("flow_score") or 0)

    # Current raw feature values for this developer-day
    current_raw: dict[str, float] = {
        feat: float(row.get(feat) or 0) for feat in FEATURE_NAMES
    }

    # Team-wide baseline means — used to detect unavailable data sources
    team_baselines = _get_team_baselines(get_connection)

    # Apply changes (skipping features with no real data)
    modified_raw = build_feature_deltas(changes, current_raw, team_baselines)

    # Score the modified vector
    x_modified = np.array(
        [modified_raw.get(f, 0.0) for f in FEATURE_NAMES], dtype=float
    )
    raw_projected = _model_module.score_single_row(x_modified)

    # FLOOR: projected score must never be lower than current score.
    # The simulator represents "what if we improve things" — never degradation.
    projected_score = max(current_score, min(100.0, raw_projected))

    improvement_points = projected_score - current_score
    improvement_percent = (
        (improvement_points / max(current_score, 1.0)) * 100
        if improvement_points > 0 else 0.0
    )
    hours_recovered = improvement_points * hours_per_flow_point

    # Build before/after comparison — only include features that actually changed
    feature_comparison: dict[str, dict] = {}
    for feat in FEATURE_NAMES:
        before_val = current_raw.get(feat, 0.0)
        after_val = modified_raw.get(feat, 0.0)
        if abs(before_val - after_val) > 0.001:
            feature_comparison[feat] = {
                "before": round(before_val, 3),
                "after": round(after_val, 3),
                "delta": round(after_val - before_val, 3),
            }

    save_simulation(
        developer_email=developer_email,
        simulation_date=date,
        changes=changes,
        current_score=current_score,
        projected_score=projected_score,
        hours_recovered=hours_recovered,
    )

    return {
        "current_flow_score": round(current_score, 1),
        "projected_flow_score": round(projected_score, 1),
        "improvement_points": round(improvement_points, 1),
        "improvement_percent": round(improvement_percent, 1),
        "estimated_hours_recovered_per_week": round(hours_recovered, 1),
        "modified_features": feature_comparison,
        "developer": developer_email,
        "date": date,
        "data_availability": _check_data_availability(current_raw, team_baselines),
    }


def _get_team_baselines(get_connection_fn) -> dict[str, float]:
    """Compute team-wide mean for each feature to detect unavailable data."""
    from flowlens.db import get_connection
    from flowlens.features import FEATURE_NAMES

    baselines: dict[str, float] = {f: 0.0 for f in FEATURE_NAMES}
    try:
        col_avgs = ", ".join(f"AVG({c}) as {c}" for c in FEATURE_NAMES)
        with get_connection() as conn:
            row = conn.execute(
                f"SELECT {col_avgs} FROM developer_day_features"
            ).fetchone()
        if row:
            baselines = {
                c: float(row[c]) if row[c] is not None else 0.0
                for c in FEATURE_NAMES
            }
    except Exception as exc:
        logger.warning("Could not fetch team baselines: %s", exc)
    return baselines


def _check_data_availability(
    current_raw: dict[str, float],
    team_baselines: dict[str, float],
) -> dict[str, bool]:
    """
    Return a dict indicating which external data sources are available.
    Shown in the UI so users understand why some sliders have no effect.
    """
    return {
        "pr_data": team_baselines.get("pr_blocking_time_hours", 0) > 0,
        "ci_data": team_baselines.get("build_time_minutes", 0) > 0,
        "test_data": team_baselines.get("test_failure_density", 0) > 0,
    }