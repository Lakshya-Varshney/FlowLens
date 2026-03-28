"""
db.py — SQLite persistence layer for FlowLens.

Responsibilities:
  - Initialize all tables on first run
  - Persist raw commit events
  - Persist engineered features + ML results
  - Store simulation results and ingestion run metadata
  - Provide query helpers consumed by api.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("flowlens.db")

# Database file path — created automatically under ./data/
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "flowlens.db"


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """Yield a SQLite connection with row_factory set for dict-like access."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrent read/write
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_RAW_COMMITS = """
CREATE TABLE IF NOT EXISTS raw_commits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash     TEXT NOT NULL,
    author_email    TEXT NOT NULL,
    author_name     TEXT NOT NULL,
    timestamp_utc   INTEGER NOT NULL,
    commit_date     TEXT NOT NULL,
    commit_hour     INTEGER NOT NULL,
    commit_weekday  INTEGER NOT NULL,
    insertions      INTEGER DEFAULT 0,
    deletions       INTEGER DEFAULT 0,
    files_changed   TEXT,
    commit_message  TEXT,
    is_merge        INTEGER DEFAULT 0,
    repo_name       TEXT NOT NULL DEFAULT 'unknown'
);
"""

_CREATE_IDX_RAW_COMMITS = [
    "CREATE INDEX IF NOT EXISTS idx_rc_author_date ON raw_commits(author_email, commit_date);",
    "CREATE INDEX IF NOT EXISTS idx_rc_date ON raw_commits(commit_date);",
]

_CREATE_DEV_DAY_FEATURES = """
CREATE TABLE IF NOT EXISTS developer_day_features (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    developer_email             TEXT NOT NULL,
    developer_name              TEXT NOT NULL,
    feature_date                TEXT NOT NULL,
    repo_name                   TEXT NOT NULL DEFAULT 'unknown',

    daily_commit_count          REAL,
    session_duration_minutes    REAL,
    inter_commit_gap_mean       REAL,
    inter_commit_gap_variance   REAL,
    late_night_ratio            REAL,
    weekend_ratio               REAL,
    avg_commit_size_lines       REAL,
    merge_commit_ratio          REAL,
    files_touched_diversity     REAL,
    commit_message_length_mean  REAL,
    pr_blocking_time_hours      REAL DEFAULT 0,
    pr_size_lines               REAL DEFAULT 0,
    test_failure_density        REAL DEFAULT 0,
    build_time_minutes          REAL DEFAULT 0,

    flow_score                  REAL,
    anomaly_label               INTEGER,
    anomaly_score_raw           REAL,
    shap_values_json            TEXT,
    recommendation_rule         TEXT,
    recommendation_llm          TEXT,

    created_at                  TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_IDX_DEV_DAY = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_dd_unique ON developer_day_features(developer_email, feature_date, repo_name);",
    "CREATE INDEX IF NOT EXISTS idx_dd_date ON developer_day_features(feature_date);",
    "CREATE INDEX IF NOT EXISTS idx_dd_score ON developer_day_features(flow_score);",
]

_CREATE_SIMULATION_RESULTS = """
CREATE TABLE IF NOT EXISTS simulation_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    developer_email         TEXT NOT NULL,
    simulation_date         TEXT NOT NULL,
    input_changes_json      TEXT NOT NULL,
    current_flow_score      REAL NOT NULL,
    projected_flow_score    REAL NOT NULL,
    hours_recovered_weekly  REAL,
    created_at              TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_INGESTION_RUNS = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_url         TEXT NOT NULL,
    repo_name        TEXT,
    since_date       TEXT,
    commits_ingested INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'in_progress',
    started_at       TEXT DEFAULT (datetime('now')),
    completed_at     TEXT
);
"""


def init_db() -> None:
    """Create all tables and indexes if they do not already exist."""
    logger.info("Initializing SQLite database at %s", DB_PATH)
    with get_connection() as conn:
        conn.execute(_CREATE_RAW_COMMITS)
        for idx in _CREATE_IDX_RAW_COMMITS:
            conn.execute(idx)

        conn.execute(_CREATE_DEV_DAY_FEATURES)
        for idx in _CREATE_IDX_DEV_DAY:
            conn.execute(idx)

        conn.execute(_CREATE_SIMULATION_RESULTS)
        conn.execute(_CREATE_INGESTION_RUNS)
    logger.info("Database initialized.")


# ---------------------------------------------------------------------------
# Ingestion run helpers
# ---------------------------------------------------------------------------

def record_ingestion_run(
    repo_url: str,
    since_date: str,
    status: str,
    commits_ingested: int = 0,
    run_id: int | None = None,
) -> int:
    """Insert or update an ingestion run record. Returns the run ID."""
    with get_connection() as conn:
        if run_id is None:
            repo_name = repo_url.rstrip("/").split("/")[-1]
            cur = conn.execute(
                "INSERT INTO ingestion_runs (repo_url, repo_name, since_date, status) VALUES (?,?,?,?)",
                (repo_url, repo_name, since_date, status),
            )
            return cur.lastrowid
        else:
            completed_at = datetime.utcnow().isoformat() if status in ("complete", "failed") else None
            conn.execute(
                """UPDATE ingestion_runs
                   SET status=?, commits_ingested=?, completed_at=?
                   WHERE id=?""",
                (status, commits_ingested, completed_at, run_id),
            )
            return run_id


# ---------------------------------------------------------------------------
# Raw commit persistence
# ---------------------------------------------------------------------------

def save_raw_commits(raw_df: pd.DataFrame, ci_df: pd.DataFrame | None = None) -> None:
    """
    Persist raw commit DataFrame to the raw_commits table.
    Clears existing commits for the same repo_name first so re-runs
    don't accumulate duplicate data from previous ingestions.
    """
    if raw_df.empty:
        logger.warning("save_raw_commits received an empty DataFrame.")
        return

    raw_df = raw_df.copy()
    if "repo_name" not in raw_df.columns:
        raw_df["repo_name"] = "unknown"

    repo_name = raw_df["repo_name"].iloc[0]

    # Delete stale data for this repo before inserting fresh commits
    # This prevents the 42966-rows-from-old-runs problem
    with get_connection() as conn:
        deleted = conn.execute(
            "DELETE FROM raw_commits WHERE repo_name = ?", (repo_name,)
        ).rowcount
        if deleted > 0:
            logger.info("Cleared %d stale commits for repo '%s'.", deleted, repo_name)

        # Also clear stale features for this repo
        conn.execute(
            "DELETE FROM developer_day_features WHERE repo_name = ?", (repo_name,)
        )

    rows = [
        (
            row["commit_hash"],
            row["author_email"],
            row["author_name"],
            int(row["timestamp_utc"]),
            row["commit_date"],
            int(row["commit_hour"]),
            int(row["commit_weekday"]),
            int(row.get("insertions", 0)),
            int(row.get("deletions", 0)),
            json.dumps(row.get("files_changed", [])),
            row.get("commit_message", ""),
            int(row.get("is_merge", 0)),
            row.get("repo_name", "unknown"),
        )
        for _, row in raw_df.iterrows()
    ]

    with get_connection() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO raw_commits
               (commit_hash, author_email, author_name, timestamp_utc,
                commit_date, commit_hour, commit_weekday, insertions, deletions,
                files_changed, commit_message, is_merge, repo_name)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        inserted = conn.execute("SELECT changes()").fetchone()[0]
    logger.info("Saved %d new raw commits (skipped duplicates).", inserted)


def get_raw_commits() -> pd.DataFrame:
    """Return all raw commits as a pandas DataFrame."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM raw_commits ORDER BY timestamp_utc ASC"
        ).fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    # Deserialize files_changed JSON column
    df["files_changed"] = df["files_changed"].apply(
        lambda x: json.loads(x) if x else []
    )
    return df


# ---------------------------------------------------------------------------
# Feature persistence
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "daily_commit_count", "session_duration_minutes",
    "inter_commit_gap_mean", "inter_commit_gap_variance",
    "late_night_ratio", "weekend_ratio", "avg_commit_size_lines",
    "merge_commit_ratio", "files_touched_diversity",
    "commit_message_length_mean", "pr_blocking_time_hours",
    "pr_size_lines", "test_failure_density", "build_time_minutes",
]


def save_features(meta_df: pd.DataFrame) -> None:
    """
    Upsert developer-day features into developer_day_features table.
    meta_df must include metadata columns + all FEATURE_COLS.
    """
    if meta_df.empty:
        return

    def _safe(val: Any) -> Any:
        """Convert NaN/Inf to None for SQLite."""
        if val is None:
            return None
        try:
            f = float(val)
            return None if (np.isnan(f) or np.isinf(f)) else f
        except (TypeError, ValueError):
            return val

    rows = []
    for _, row in meta_df.iterrows():
        feature_vals = [_safe(row.get(c, 0)) for c in FEATURE_COLS]
        rows.append((
            row["developer_email"],
            row.get("developer_name", row["developer_email"]),
            row["feature_date"],
            row.get("repo_name", "unknown"),
            *feature_vals,
        ))

    placeholders = ",".join(["?"] * (4 + len(FEATURE_COLS)))
    col_names = "developer_email, developer_name, feature_date, repo_name, " + ", ".join(FEATURE_COLS)

    with get_connection() as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO developer_day_features ({col_names}) VALUES ({placeholders})",
            rows,
        )
    logger.info("Saved %d developer-day feature rows.", len(rows))


def update_model_results(
    meta_df: pd.DataFrame,
    flow_scores: np.ndarray,
    anomaly_labels: np.ndarray,
    anomaly_scores_raw: np.ndarray,
    shap_matrix: np.ndarray,
) -> None:
    """Write ML results (scores, labels, SHAP) back to developer_day_features."""
    feature_names = FEATURE_COLS
    rows = []
    for i, (_, row) in enumerate(meta_df.iterrows()):
        shap_payload = [
            {"feature": feature_names[j], "shap_value": float(shap_matrix[i, j])}
            for j in range(len(feature_names))
        ]
        rows.append((
            float(flow_scores[i]),
            int(anomaly_labels[i]),
            float(anomaly_scores_raw[i]),
            json.dumps(shap_payload),
            row["developer_email"],
            row["feature_date"],
            row.get("repo_name", "unknown"),
        ))

    with get_connection() as conn:
        conn.executemany(
            """UPDATE developer_day_features
               SET flow_score=?, anomaly_label=?, anomaly_score_raw=?, shap_values_json=?
               WHERE developer_email=? AND feature_date=? AND repo_name=?""",
            rows,
        )
    logger.info("Updated model results for %d rows.", len(rows))


# ---------------------------------------------------------------------------
# Query helpers (consumed by api.py)
# ---------------------------------------------------------------------------

def get_flow_scores(
    developer: str | None = None,
    since: str | None = None,
    until: str | None = None,
    aggregate: bool = False,
) -> list[dict]:
    """
    Return flow score rows filtered by optional developer and date range.
    If aggregate=True, return daily team averages instead.
    """
    params: list[Any] = []
    where_clauses = ["flow_score IS NOT NULL"]

    if developer:
        where_clauses.append("developer_email = ?")
        params.append(developer)
    if since:
        where_clauses.append("feature_date >= ?")
        params.append(since)
    if until:
        where_clauses.append("feature_date <= ?")
        params.append(until)

    where_sql = " AND ".join(where_clauses)

    if aggregate:
        sql = f"""
            SELECT
                feature_date as date,
                'team' as developer,
                AVG(flow_score) as flow_score,
                SUM(CASE WHEN anomaly_label = -1 THEN 1 ELSE 0 END) as anomaly_count,
                COUNT(*) as dev_count
            FROM developer_day_features
            WHERE {where_sql}
            GROUP BY feature_date
            ORDER BY feature_date ASC
        """
    else:
        sql = f"""
            SELECT
                feature_date as date,
                developer_email as developer,
                developer_name,
                flow_score,
                anomaly_label,
                anomaly_score_raw,
                daily_commit_count,
                session_duration_minutes,
                repo_name
            FROM developer_day_features
            WHERE {where_sql}
            ORDER BY feature_date ASC, developer_email ASC
        """

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(r) for r in rows]


def get_shap_values(date: str, developer_email: str) -> dict | None:
    """Return the full feature row including SHAP JSON for a specific developer-day."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT * FROM developer_day_features
               WHERE feature_date = ? AND developer_email = ?
               LIMIT 1""",
            (date, developer_email),
        ).fetchone()
    return dict(row) if row else None


def get_trend(
    since: str | None = None,
    developer: str | None = None,
    window: int = 7,
) -> list[dict]:
    """
    Return rolling average flow scores for the trend line chart.
    Uses a Python-side rolling window since SQLite lacks native window functions
    in older builds.
    """
    params: list[Any] = ["flow_score IS NOT NULL"]
    filters = ["flow_score IS NOT NULL"]
    param_vals: list[Any] = []

    if since:
        filters.append("feature_date >= ?")
        param_vals.append(since)
    if developer:
        filters.append("developer_email = ?")
        param_vals.append(developer)

    where_sql = " AND ".join(filters)

    with get_connection() as conn:
        rows = conn.execute(
            f"""SELECT feature_date as date,
                       AVG(flow_score) as avg_score,
                       SUM(CASE WHEN anomaly_label=-1 THEN 1 ELSE 0 END) as anomaly_count
                FROM developer_day_features
                WHERE {where_sql}
                GROUP BY feature_date
                ORDER BY feature_date ASC""",
            param_vals,
        ).fetchall()

    if not rows:
        return []

    df = pd.DataFrame([dict(r) for r in rows])
    df["rolling_avg_score"] = df["avg_score"].rolling(window=window, min_periods=1).mean()

    return df[["date", "rolling_avg_score", "anomaly_count"]].to_dict(orient="records")


def get_all_developers() -> list[str]:
    """Return distinct developer emails present in developer_day_features."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT developer_email FROM developer_day_features ORDER BY developer_email"
        ).fetchall()
    return [r["developer_email"] for r in rows]


def get_summary_stats() -> dict:
    """Return high-level summary stats for the /api/status endpoint."""
    with get_connection() as conn:
        stats = conn.execute(
            """SELECT
                 COUNT(*) as total_rows,
                 AVG(flow_score) as mean_score,
                 SUM(CASE WHEN anomaly_label = -1 THEN 1 ELSE 0 END) as anomaly_count,
                 MIN(feature_date) as date_start,
                 MAX(feature_date) as date_end,
                 COUNT(DISTINCT developer_email) as developer_count
               FROM developer_day_features
               WHERE flow_score IS NOT NULL"""
        ).fetchone()
        raw_count = conn.execute("SELECT COUNT(*) as n FROM raw_commits").fetchone()
        last_run = conn.execute(
            "SELECT repo_url, status FROM ingestion_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "total_developer_days": stats["total_rows"] if stats else 0,
        "mean_flow_score": round(stats["mean_score"], 1) if stats and stats["mean_score"] else None,
        "anomaly_count": stats["anomaly_count"] if stats else 0,
        "date_start": stats["date_start"] if stats else None,
        "date_end": stats["date_end"] if stats else None,
        "developer_count": stats["developer_count"] if stats else 0,
        "raw_commit_count": raw_count["n"] if raw_count else 0,
        "repo": last_run["repo_url"] if last_run else None,
        "ingestion_status": last_run["status"] if last_run else "no_runs",
    }


# ---------------------------------------------------------------------------
# Simulation persistence
# ---------------------------------------------------------------------------

def save_simulation(
    developer_email: str,
    simulation_date: str,
    changes: dict,
    current_score: float,
    projected_score: float,
    hours_recovered: float,
) -> int:
    """Persist a simulator run and return its ID."""
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO simulation_results
               (developer_email, simulation_date, input_changes_json,
                current_flow_score, projected_flow_score, hours_recovered_weekly)
               VALUES (?,?,?,?,?,?)""",
            (developer_email, simulation_date, json.dumps(changes),
             current_score, projected_score, hours_recovered),
        )
        return cur.lastrowid