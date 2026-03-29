"""
run.py — FlowLens entry point.

Parses CLI arguments, orchestrates ingestion → feature engineering →
ML training → Flask server startup.

Usage examples:
    python run.py --repo https://github.com/facebook/react
    python run.py --repo /path/to/local/repo --since 2025-01-01
    python run.py --demo
    python run.py --repo https://github.com/facebook/react --ci-logs ./ci_exports/
"""

import argparse
import os
import sys
import webbrowser
import threading
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on the path before local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()  # Read .env file if present

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flowlens.run")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path = PROJECT_ROOT / "config.yaml") -> dict:
    """Load YAML config. Returns default dict on missing file."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    logger.warning("config.yaml not found — using built-in defaults.")
    return {}


def print_banner() -> None:
    banner = r"""
  _____ _               _
 |  ___| | _____      _| |    ___ _ __  ___
 | |_  | |/ _ \ \ /\ / / |   / _ \ '_ \/ __|
 |  _| | | (_) \ V  V /| |__|  __/ | | \__ \
 |_|   |_|\___/ \_/\_/ |_____\___|_| |_|___/

 X-ray your developer experience.
    """
    print(banner)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowlens",
        description="FlowLens — AI-powered developer flow intelligence engine.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Git repo URL (https://github.com/...) or local path.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Limit ingestion to commits after this date. Format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--ci-logs",
        type=str,
        default=None,
        dest="ci_logs",
        help="Directory containing JUnit XML CI log files (enables Cross-Track B).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Load pre-baked demo dataset instead of ingesting a live repo.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the Flask server (default: 5000).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,   # None means: read from config.yaml clone_depth
        help="Shallow clone depth — number of recent commits to fetch. "
             "Overrides config.yaml clone_depth. Use 0 for full history. "
             "Default: value from config.yaml (500).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser after startup.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_ingestion(repo: str, since: str | None, ci_logs: str | None, cfg: dict, depth: int = 500) -> None:
    """
    Step 1 — ingest commits (and optional CI logs) into SQLite raw_commits table.
    """
    from flowlens.db import init_db, save_raw_commits, record_ingestion_run
    from flowlens.ingest import ingest_repo, parse_ci_logs

    init_db()

    since_date: datetime | None = None
    if since:
        since_date = datetime.strptime(since, "%Y-%m-%d")
    else:
        lookback = cfg.get("ingestion", {}).get("default_days_lookback", 90)
        since_date = datetime.utcnow() - timedelta(days=lookback)

    logger.info("▶ Ingesting commits from: %s (since %s)", repo, since_date.date())
    run_id = record_ingestion_run(repo_url=repo, since_date=str(since_date.date()), status="in_progress")

    try:
        raw_df = ingest_repo(
            repo_source=repo,
            since_date=since_date,
            batch_size=cfg.get("ingestion", {}).get("batch_size", 5000),
            depth=depth,
        )
        logger.info("  ✓ %d commits extracted.", len(raw_df))

        # Optional CI log ingestion for Cross-Track B
        ci_df = None
        if ci_logs and Path(ci_logs).is_dir():
            logger.info("  ▶ Parsing CI logs from: %s", ci_logs)
            ci_df = parse_ci_logs(ci_logs)
            logger.info("  ✓ %d CI runs parsed.", len(ci_df))

        save_raw_commits(raw_df, ci_df=ci_df)
        record_ingestion_run(
            repo_url=repo,
            since_date=str(since_date.date()),
            status="complete",
            commits_ingested=len(raw_df),
            run_id=run_id,
        )
        logger.info("  ✓ Raw commits written to database.")

    except Exception as exc:
        record_ingestion_run(repo_url=repo, since_date=str(since_date.date()), status="failed", run_id=run_id)
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise


def run_feature_engineering(cfg: dict) -> tuple:
    """
    Step 2 — transform raw commits → developer-day feature matrix.
    Returns (X_scaled, X_raw, meta_df).
    """
    from flowlens.db import get_raw_commits, save_features
    from flowlens.features import compute_developer_day_features

    logger.info("▶ Engineering features …")
    raw_df = get_raw_commits()

    if raw_df.empty:
        logger.error("No raw commits found in database. Did ingestion complete?")
        sys.exit(1)

    min_commits = cfg.get("features", {}).get("min_commits_per_day", 2)
    min_days = cfg.get("features", {}).get("min_days_per_developer", 5)

    X_scaled, X_raw, meta_df, feature_scaler = compute_developer_day_features(
        raw_df,
        min_commits_per_day=min_commits,
        min_days_per_developer=min_days,
    )

    logger.info("  ✓ Feature matrix: %d developer-day rows × %d features.", *X_scaled.shape)
    save_features(meta_df)
    logger.info("  ✓ Features written to database.")
    return X_scaled, X_raw, meta_df, feature_scaler


def run_model_training(X_scaled, X_raw, meta_df, cfg: dict, feature_scaler) -> None:
    """
    Step 3 — train Isolation Forest, compute SHAP, write scores to DB.
    """
    from flowlens.db import update_model_results
    from flowlens.model import train_model, compute_flow_scores, compute_shap_values

    model_cfg = cfg.get("model", {})

    # ── Train model with spinner (IsolationForest has no per-step progress)
    print()
    with tqdm(
        total=4,
        desc="  🤖 Training ML pipeline",
        unit=" steps",
        colour="yellow",
        bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}]",
        dynamic_ncols=True,
    ) as pbar:
        pbar.set_postfix_str("Fitting IsolationForest …")
        model, scaler, flow_scaler = train_model(
            X_scaled,
            n_estimators=model_cfg.get("n_estimators", 200),
            contamination=model_cfg.get("contamination", 0.1),
            random_state=model_cfg.get("random_state", 42),
        )
        pbar.update(1)

        pbar.set_postfix_str("Computing flow scores …")
        flow_scores, anomaly_labels, anomaly_scores_raw = compute_flow_scores(
            model, X_scaled, flow_scaler
        )
        pbar.update(1)

        pbar.set_postfix_str("Computing SHAP values …")
        shap_matrix = compute_shap_values(model, X_scaled)
        pbar.update(1)

        pbar.set_postfix_str("Saving results to database …")
        update_model_results(meta_df, flow_scores, anomaly_labels, anomaly_scores_raw, shap_matrix)
        # IMPORTANT: cache feature_scaler (from features.py), NOT the model's internal scaler.
        # The simulator must use the same scaler that was used during feature engineering
        # to correctly scale new raw feature vectors before scoring them.
        _cache_model_objects(model, feature_scaler, flow_scaler, X_scaled, meta_df)
        pbar.update(1)

    anomaly_count = (anomaly_labels == -1).sum()
    print(f"\n  ✅ Model ready — Mean flow score: {flow_scores.mean():.1f}/100 · Anomalies detected: {anomaly_count}\n")


def _cache_model_objects(model, scaler, flow_scaler, X_scaled, meta_df) -> None:
    """Store trained objects in a module-level registry for the Flask app."""
    import flowlens.model as _model_module
    _model_module._TRAINED_MODEL = model
    _model_module._TRAINED_SCALER = scaler
    _model_module._TRAINED_FLOW_SCALER = flow_scaler
    _model_module._X_SCALED = X_scaled
    _model_module._META_DF = meta_df


def load_demo_data(cfg: dict) -> None:
    """Load pre-baked demo dataset from Parquet or generate synthetic data."""
    from flowlens.db import init_db
    from flowlens.demo import load_or_generate_demo

    init_db()
    logger.info("▶ Loading demo dataset …")
    load_or_generate_demo(cfg)
    logger.info("  ✓ Demo dataset ready.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()
    cfg = load_config()
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.demo and not args.repo:
        parser.error("Either --repo <url|path> or --demo is required.")

    # -----------------------------------------------------------------------
    # Pipeline
    # -----------------------------------------------------------------------
    if args.demo:
        load_demo_data(cfg)
    else:
        # Resolve depth: CLI flag overrides config.yaml, config overrides hardcoded default
        depth = args.depth if args.depth is not None else cfg.get("ingestion", {}).get("clone_depth", 500)
        run_ingestion(args.repo, args.since, args.ci_logs, cfg, depth)

    X_scaled, X_raw, meta_df, feature_scaler = run_feature_engineering(cfg)
    run_model_training(X_scaled, X_raw, meta_df, cfg, feature_scaler)

    # -----------------------------------------------------------------------
    # Start Flask server
    # -----------------------------------------------------------------------
    from flowlens.api import create_app

    port = args.port
    app = create_app(cfg)

    auto_browser = cfg.get("server", {}).get("auto_open_browser", True)
    if auto_browser and not args.no_browser:
        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open_browser, daemon=True).start()

    logger.info("🚀 FlowLens ready → http://localhost:%d  (Dashboard: http://localhost:%d/dashboard)", port, port)
    app.run(
        host=cfg.get("server", {}).get("host", "0.0.0.0"),
        port=port,
        debug=cfg.get("server", {}).get("debug", False),
        use_reloader=False,
    )


if __name__ == "__main__":
    main()