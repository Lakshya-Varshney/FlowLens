"""
features.py — Developer-day feature engineering pipeline.

Transforms raw commit events (one row per commit) into a developer-day
feature matrix suitable for Isolation Forest anomaly detection.

Each output row represents ONE developer on ONE calendar day.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger("flowlens.features")

# Ordered list of feature column names (must match db.py FEATURE_COLS)
FEATURE_NAMES: list[str] = [
    "daily_commit_count",
    "session_duration_minutes",
    "inter_commit_gap_mean",
    "inter_commit_gap_variance",
    "late_night_ratio",
    "weekend_ratio",
    "avg_commit_size_lines",
    "merge_commit_ratio",
    "files_touched_diversity",
    "commit_message_length_mean",
    "pr_blocking_time_hours",
    "pr_size_lines",
    "test_failure_density",
    "build_time_minutes",
]

# Hours considered "late night" for late_night_ratio
_LATE_NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_developer_day_features(
    raw_df: pd.DataFrame,
    min_commits_per_day: int = 2,
    min_days_per_developer: int = 5,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Transform raw commits DataFrame into developer-day feature matrix.

    Args:
        raw_df:                 Raw commits DataFrame from ingest.py.
        min_commits_per_day:    Drop rows with fewer commits (insufficient signal).
        min_days_per_developer: Drop developers with fewer active days.

    Returns:
        X_scaled  — StandardScaler-scaled numpy array (n_rows × 14)
        X_raw     — Unscaled feature DataFrame (same rows, readable values)
        meta_df   — Metadata (developer_email, feature_date, etc.) aligned by index
    """
    logger.info("Starting feature engineering on %d raw commits …", len(raw_df))

    # Ensure timestamp is available for sorting
    raw_df = _prepare_raw_df(raw_df)

    # Aggregate per developer per day
    feature_df = _aggregate_developer_days(raw_df)

    # Apply minimum commit filter
    before = len(feature_df)
    feature_df = feature_df[feature_df["daily_commit_count"] >= min_commits_per_day]
    logger.debug("Dropped %d rows below min_commits_per_day=%d.", before - len(feature_df), min_commits_per_day)

    # Apply minimum days-per-developer filter
    dev_day_counts = feature_df.groupby("developer_email")["feature_date"].count()
    qualifying_devs = dev_day_counts[dev_day_counts >= min_days_per_developer].index
    before = len(feature_df)
    feature_df = feature_df[feature_df["developer_email"].isin(qualifying_devs)]
    logger.debug("Dropped %d rows from developers below min_days=%d.", before - len(feature_df), min_days_per_developer)

    if feature_df.empty:
        raise ValueError(
            "Feature matrix is empty after filtering. "
            "Try reducing min_commits_per_day or min_days_per_developer, "
            "or ingest more data."
        )

    feature_df = feature_df.reset_index(drop=True)

    # Split metadata from features
    meta_cols = ["developer_email", "developer_name", "feature_date", "repo_name"]
    meta_df = feature_df[meta_cols].copy()

    X_raw = feature_df[FEATURE_NAMES].copy()

    # Fill any residual NaNs with column medians (robust fallback)
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    X_raw = X_raw.fillna(0)  # final catch for all-NaN columns

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values)

    # Attach raw feature values back to meta_df for DB persistence
    for col in FEATURE_NAMES:
        meta_df[col] = X_raw[col].values

    logger.info(
        "Feature engineering complete: %d developer-day rows × %d features.",
        *X_scaled.shape,
    )
    # Return scaler so the simulator can correctly scale new feature vectors
    return X_scaled, X_raw, meta_df, scaler


# ---------------------------------------------------------------------------
# Internal aggregation logic
# ---------------------------------------------------------------------------

def _prepare_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure essential columns exist and types are correct.
    """
    df = raw_df.copy()
    df["timestamp_utc"] = pd.to_numeric(df["timestamp_utc"], errors="coerce")
    df["commit_date"] = df["commit_date"].astype(str)
    df["commit_hour"] = pd.to_numeric(df["commit_hour"], errors="coerce").fillna(12).astype(int)
    df["commit_weekday"] = pd.to_numeric(df["commit_weekday"], errors="coerce").fillna(0).astype(int)
    df["insertions"] = pd.to_numeric(df.get("insertions", 0), errors="coerce").fillna(0)
    df["deletions"] = pd.to_numeric(df.get("deletions", 0), errors="coerce").fillna(0)
    df["is_merge"] = pd.to_numeric(df.get("is_merge", 0), errors="coerce").fillna(0).astype(int)
    df["commit_message"] = df.get("commit_message", "").fillna("").astype(str)
    if "files_changed" not in df.columns:
        df["files_changed"] = [[] for _ in range(len(df))]
    if "repo_name" not in df.columns:
        df["repo_name"] = "unknown"
    return df


def _aggregate_developer_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group commits by (author_email, commit_date) and compute all 14 features.
    """
    groups = df.groupby(["author_email", "commit_date"])
    records: list[dict] = []

    with tqdm(
        total=len(groups),
        desc="  ⚙  Engineering features",
        unit=" dev-days",
        colour="magenta",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
    ) as pbar:
        for (email, date), group in groups:
            record = _compute_group_features(email, date, group, df)
            records.append(record)
            pbar.update(1)

    return pd.DataFrame(records)


def _compute_group_features(
    email: str,
    date: str,
    group: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict:
    """
    Compute all 14 features for a single (developer, date) group.
    """
    n = len(group)
    timestamps = group["timestamp_utc"].sort_values().values

    # --- 1. daily_commit_count ---
    daily_commit_count = float(n)

    # --- 2. session_duration_minutes ---
    if n >= 2:
        session_duration_minutes = float((timestamps[-1] - timestamps[0]) / 60)
    else:
        session_duration_minutes = 1.0

    # --- 3 & 4. inter_commit_gap_mean & inter_commit_gap_variance ---
    if n >= 2:
        gaps = np.diff(timestamps) / 60  # gap in minutes
        inter_commit_gap_mean = float(gaps.mean())
        inter_commit_gap_variance = float(gaps.var())
    else:
        inter_commit_gap_mean = 0.0
        inter_commit_gap_variance = 0.0

    # --- 5. late_night_ratio ---
    late_night_count = group["commit_hour"].apply(lambda h: h in _LATE_NIGHT_HOURS).sum()
    late_night_ratio = float(late_night_count / n)

    # --- 6. weekend_ratio ---
    weekend_count = group["commit_weekday"].apply(lambda d: d >= 5).sum()
    weekend_ratio = float(weekend_count / n)

    # --- 7. avg_commit_size_lines ---
    total_lines = (group["insertions"] + group["deletions"]).sum()
    avg_commit_size_lines = float(total_lines / n)

    # --- 8. merge_commit_ratio ---
    merge_commit_ratio = float(group["is_merge"].mean())

    # --- 9. files_touched_diversity ---
    all_files: list[str] = []
    for flist in group["files_changed"]:
        if isinstance(flist, list):
            all_files.extend(flist)
        elif isinstance(flist, str):
            try:
                import json
                parsed = json.loads(flist)
                all_files.extend(parsed)
            except Exception:
                pass
    files_touched_diversity = float(len(set(all_files)))

    # --- 10. commit_message_length_mean ---
    commit_message_length_mean = float(group["commit_message"].str.len().mean())

    # --- 11 & 12. PR features (default 0 — set externally if GitHub API data available) ---
    pr_blocking_time_hours = float(group.get("pr_blocking_time_hours", pd.Series([0])).fillna(0).mean())
    pr_size_lines = float(group.get("pr_size_lines", pd.Series([0])).fillna(0).mean())

    # --- 13 & 14. CI features (default 0 — set externally from CI log parser) ---
    test_failure_density = float(group.get("test_failure_density", pd.Series([0])).fillna(0).mean())
    build_time_minutes = float(group.get("build_time_minutes", pd.Series([0])).fillna(0).mean())

    return {
        "developer_email": email,
        "developer_name": group["author_name"].iloc[0],
        "feature_date": date,
        "repo_name": group["repo_name"].iloc[0] if "repo_name" in group.columns else "unknown",
        # Features
        "daily_commit_count": daily_commit_count,
        "session_duration_minutes": session_duration_minutes,
        "inter_commit_gap_mean": inter_commit_gap_mean,
        "inter_commit_gap_variance": inter_commit_gap_variance,
        "late_night_ratio": late_night_ratio,
        "weekend_ratio": weekend_ratio,
        "avg_commit_size_lines": avg_commit_size_lines,
        "merge_commit_ratio": merge_commit_ratio,
        "files_touched_diversity": files_touched_diversity,
        "commit_message_length_mean": commit_message_length_mean,
        "pr_blocking_time_hours": pr_blocking_time_hours,
        "pr_size_lines": pr_size_lines,
        "test_failure_density": test_failure_density,
        "build_time_minutes": build_time_minutes,
    }