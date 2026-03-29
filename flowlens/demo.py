"""
demo.py — Synthetic demo dataset generator for FlowLens.

Generates realistic developer activity data when --demo flag is used,
avoiding the need to clone a live repository during presentations.

The synthetic data intentionally includes:
  - Normal "flow" periods (high scores)
  - Anomalous "crunch" periods (low scores, mirroring a release sprint)
  - Cross-track B: test failure spikes correlated with low flow periods
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from flowlens.db import save_raw_commits, init_db

logger = logging.getLogger("flowlens.demo")

DEMO_PARQUET = Path(__file__).resolve().parents[1] / "data" / "demo_dataset.parquet"

# Demo developers (anonymised)
DEMO_DEVELOPERS = [
    ("alice@demo.local", "Alice Chen"),
    ("bob@demo.local", "Bob Kumar"),
    ("charlie@demo.local", "Charlie Lopez"),
    ("diana@demo.local", "Diana Park"),
]


def load_or_generate_demo(cfg: dict) -> None:
    """
    Load pre-generated Parquet demo data if it exists, otherwise generate
    synthetic commit data and save it to the database.
    """
    if DEMO_PARQUET.exists():
        logger.info("Loading cached demo dataset from %s", DEMO_PARQUET)
        df = pd.read_parquet(DEMO_PARQUET)
        save_raw_commits(df)
        return

    logger.info("Generating synthetic demo dataset …")
    df = _generate_synthetic_commits(
        developers=DEMO_DEVELOPERS,
        days=90,
        rng_seed=42,
    )

    # Cache for faster future loads
    DEMO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEMO_PARQUET, index=False)
    logger.info("Demo dataset saved to %s (%d commits).", DEMO_PARQUET, len(df))

    save_raw_commits(df)


def _generate_synthetic_commits(
    developers: list[tuple[str, str]],
    days: int,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic commit records for multiple developers over N days.

    Intentionally injects anomalous periods:
      - Days 55-65: "Release sprint crunch" — large PRs, late nights, slow CI
      - Days 70-75: "Flaky CI incident" — high test failures
    """
    rng = np.random.default_rng(rng_seed)
    records = []
    base_date = datetime.utcnow() - timedelta(days=days)
    repo_name = "demo-project"

    for day_offset in range(days):
        current_date = base_date + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5
        is_crunch = 55 <= day_offset <= 65  # Injected anomaly period
        is_flaky_ci = 70 <= day_offset <= 75

        for email, name in developers:
            # Fewer commits on weekends and per-developer variation
            if is_weekend and rng.random() < 0.6:
                continue

            base_commits = rng.integers(3, 12)
            if is_crunch:
                base_commits = rng.integers(8, 20)  # More commits under pressure

            for _ in range(int(base_commits)):
                # Commit timestamp distribution
                if is_crunch:
                    # Late night bias during crunch
                    hour = int(rng.choice(
                        list(range(8, 24)) + list(range(0, 4)),
                        p=[0.04] * 16 + [0.06] * 4,
                    ))
                else:
                    hour = int(rng.integers(9, 19))

                ts_base = current_date.replace(hour=hour, minute=0, second=0)
                ts = ts_base + timedelta(minutes=int(rng.integers(0, 59)))

                # Commit size — larger during crunch
                insertions = int(rng.integers(5, 300 if is_crunch else 80))
                deletions = int(rng.integers(0, insertions // 2))

                # Test failure proxy via commit message hints
                msg_templates = [
                    "feat: add {}", "fix: resolve {}", "refactor: cleanup {}",
                    "docs: update {}", "test: add coverage for {}",
                    "chore: bump dependencies", "ci: update pipeline config",
                ]
                if is_flaky_ci:
                    msg_templates += [
                        "fix: retry flaky test", "ci: skip flaky test temporarily",
                        "fix: test race condition",
                    ]
                msg = rng.choice(msg_templates).format(
                    rng.choice(["auth", "dashboard", "api", "models", "utils", "core"])
                )

                n_files = int(rng.integers(1, 8 if is_crunch else 4))
                files = [f"src/{rng.choice(['models', 'views', 'utils', 'tests'])}/{rng.integers(1,20)}.py"
                         for _ in range(n_files)]

                records.append({
                    "commit_hash": _fake_hash(email, ts, rng),
                    "author_email": email,
                    "author_name": name,
                    "timestamp_utc": int(ts.timestamp()),
                    "commit_date": ts.strftime("%Y-%m-%d"),
                    "commit_hour": ts.hour,
                    "commit_weekday": ts.weekday(),
                    "insertions": insertions,
                    "deletions": deletions,
                    "files_changed": files,
                    "commit_message": str(msg),
                    "is_merge": int(rng.random() < 0.1),
                    "repo_name": repo_name,
                })

    df = pd.DataFrame(records)
    logger.info("Generated %d synthetic commits across %d developers.", len(df), len(developers))
    return df


def _fake_hash(email: str, ts: datetime, rng: np.random.Generator) -> str:
    """Generate a plausible 40-char hex hash."""
    import hashlib
    seed = f"{email}{ts.isoformat()}{rng.integers(0, 99999)}"
    return hashlib.sha1(seed.encode()).hexdigest()