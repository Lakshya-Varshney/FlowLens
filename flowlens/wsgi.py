"""
wsgi.py — Gunicorn entry point for FlowLens on Railway.

Runs the full demo pipeline (DB init → demo data → features → model)
BEFORE gunicorn starts serving requests, so the DB tables always exist.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("flowlens.wsgi")

# ---------------------------------------------------------------------------
# Run the full pipeline synchronously before Flask starts
# ---------------------------------------------------------------------------

import yaml

cfg = {}
cfg_path = PROJECT_ROOT / "config.yaml"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

try:
    logger.info("▶ Initialising database …")
    from flowlens.db import init_db
    init_db()

    logger.info("▶ Loading demo dataset …")
    from flowlens.demo import load_or_generate_demo
    load_or_generate_demo(cfg)

    logger.info("▶ Engineering features …")
    from flowlens.db import get_raw_commits, save_features
    from flowlens.features import compute_developer_day_features

    raw_df = get_raw_commits()
    min_commits = cfg.get("features", {}).get("min_commits_per_day", 2)
    min_days    = cfg.get("features", {}).get("min_days_per_developer", 5)

    X_scaled, X_raw, meta_df, feature_scaler = compute_developer_day_features(
        raw_df,
        min_commits_per_day=min_commits,
        min_days_per_developer=min_days,
    )
    save_features(meta_df)
    logger.info("  ✓ Feature matrix: %d developer-day rows.", len(meta_df))

    logger.info("▶ Training model …")
    from flowlens.model import train_model, compute_flow_scores, compute_shap_values
    from flowlens.db import update_model_results
    import flowlens.model as _model_module

    model_cfg = cfg.get("model", {})
    model, scaler, flow_scaler = train_model(
        X_scaled,
        n_estimators=model_cfg.get("n_estimators", 200),
        contamination=model_cfg.get("contamination", 0.1),
        random_state=model_cfg.get("random_state", 42),
    )
    flow_scores, anomaly_labels, anomaly_scores_raw = compute_flow_scores(model, X_scaled, flow_scaler)
    shap_matrix = compute_shap_values(model, X_scaled)
    update_model_results(meta_df, flow_scores, anomaly_labels, anomaly_scores_raw, shap_matrix)

    # Cache model objects so the simulator works
    _model_module._TRAINED_MODEL      = model
    _model_module._TRAINED_SCALER     = scaler
    _model_module._TRAINED_FLOW_SCALER = flow_scaler
    _model_module._X_SCALED           = X_scaled
    _model_module._META_DF            = meta_df

    logger.info("✅ FlowLens pipeline complete — handing off to gunicorn.")

except Exception as exc:
    logger.error("Pipeline failed during startup: %s", exc, exc_info=True)
    # Don't crash — let Flask start anyway so /api/status can report the error
    # rather than Railway seeing a failed deploy

# ---------------------------------------------------------------------------
# Create the Flask app — gunicorn imports this as `wsgi:app`
# ---------------------------------------------------------------------------

from flowlens.api import create_app
app = create_app(cfg)