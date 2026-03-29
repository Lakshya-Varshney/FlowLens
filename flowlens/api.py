"""
api.py — Flask REST API for FlowLens.

Endpoints:
  GET  /                                    — homepage (home.html)
  GET  /dashboard                           — main dashboard (index.html)
  POST /api/ingest                          — start background ingestion
  GET  /api/ingestion-status               — poll ingestion progress
  GET  /api/status                          — system health + dataset summary
  GET  /api/flow-scores                    — heatmap data
  GET  /api/shap/<date>/<developer_email>  — SHAP explanation
  GET  /api/recommend/<date>/<developer>   — recommendations
  POST /api/simulate                        — before/after simulator
  GET  /api/trend                           — rolling average trend line
  GET  /api/developers                      — list of all developers

All responses: {"ok": bool, "data": ..., "error": ...}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from urllib.parse import unquote

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from flowlens import model as _model_module

logger = logging.getLogger("flowlens.api")

# ---------------------------------------------------------------------------
# Background ingestion state — shared between endpoint and worker thread
# ---------------------------------------------------------------------------

_ingestion_state: dict = {
    "status":           "idle",   # idle | in_progress | complete | error
    "step":             "",
    "percent":          0,
    "message":          "",
    "timestamp":        "",
    "job_id":           None,
    "error":            None,
    "commits_ingested": None,
    "developer_count":  None,
    "developer_days":   None,
    "repo":             None,
}
_ingestion_lock = threading.Lock()


def _update_state(**kwargs) -> None:
    """Thread-safe state update. Always stamps timestamp."""
    with _ingestion_lock:
        _ingestion_state.update(kwargs)
        _ingestion_state["timestamp"] = datetime.utcnow().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------

def _run_ingestion_pipeline(repo_url: str, cfg: dict) -> None:
    """
    Full pipeline run in a daemon thread.
    Calls existing ingest → features → model functions and updates _ingestion_state.
    """
    try:
        from flowlens.db import init_db, save_raw_commits, save_features, record_ingestion_run
        from flowlens.ingest import ingest_repo, parse_ci_logs
        from flowlens.features import compute_developer_day_features
        from flowlens.model import train_model, compute_flow_scores, compute_shap_values
        from flowlens.db import update_model_results

        init_db()

        # ── STEP 1: Clone / fetch repo ──────────────────────────────
        _update_state(step="cloning", percent=5,
                      message=f"Opening repository: {repo_url}")

        lookback = cfg.get("ingestion", {}).get("default_days_lookback", 90)
        since_date = datetime.utcnow() - timedelta(days=lookback)
        depth = cfg.get("ingestion", {}).get("depth", 500)

        run_id = record_ingestion_run(
            repo_url=repo_url,
            since_date=str(since_date.date()),
            status="in_progress",
        )

        # ── STEP 2: Extract commits ─────────────────────────────────
        _update_state(step="ingesting", percent=15,
                      message="Extracting commit history…")

        raw_df = ingest_repo(
            repo_source=repo_url,
            since_date=since_date,
            batch_size=cfg.get("ingestion", {}).get("batch_size", 5000),
            depth=depth,
        )

        save_raw_commits(raw_df)

        _update_state(
            step="ingesting", percent=30,
            message=f"Extracted {len(raw_df):,} commit events",
            commits_ingested=len(raw_df),
        )

        # ── STEP 3: Feature engineering ─────────────────────────────
        _update_state(step="features", percent=44,
                      message="Engineering developer-day feature matrix…")

        min_commits = cfg.get("features", {}).get("min_commits_per_day", 2)
        min_days    = cfg.get("features", {}).get("min_days_per_developer", 5)

        X_scaled, X_raw, meta_df, feature_scaler = compute_developer_day_features(
            raw_df,
            min_commits_per_day=min_commits,
            min_days_per_developer=min_days,
        )

        save_features(meta_df)

        devs = meta_df["developer_email"].nunique()
        days = len(meta_df)
        _update_state(
            step="features", percent=55,
            message=f"Feature matrix: {days:,} developer-days · {devs} developers",
            developer_count=devs,
            developer_days=days,
        )

        # ── STEP 4: Scaling ──────────────────────────────────────────
        _update_state(step="scaling", percent=60,
                      message="StandardScaler applied · zero mean · unit variance")

        # ── STEP 5: Train Isolation Forest ───────────────────────────
        _update_state(step="training", percent=68,
                      message="Training IsolationForest (n_estimators=200)…")

        model_cfg = cfg.get("model", {})
        model, _, flow_scaler = train_model(
            X_scaled,
            n_estimators=model_cfg.get("n_estimators", 200),
            contamination=model_cfg.get("contamination", 0.1),
            random_state=model_cfg.get("random_state", 42),
        )

        flow_scores, anomaly_labels, anomaly_scores_raw = compute_flow_scores(
            model, X_scaled, flow_scaler
        )

        _update_state(step="training", percent=78,
                      message="Model trained · anomaly scores computed")

        # ── STEP 6: SHAP values ──────────────────────────────────────
        _update_state(step="shap", percent=83,
                      message="Computing SHAP explanations for anomalous days…")

        shap_matrix = compute_shap_values(model, X_scaled)

        anomaly_count = int((anomaly_labels == -1).sum())
        _update_state(step="shap", percent=88,
                      message=f"SHAP complete · {anomaly_count} anomalies explained")

        # ── STEP 7: Insights & save ──────────────────────────────────
        _update_state(step="insights", percent=92,
                      message="Generating rule-based recommendations…")

        update_model_results(meta_df, flow_scores, anomaly_labels,
                             anomaly_scores_raw, shap_matrix)

        # ── STEP 8: Cache model objects for simulator ────────────────
        _update_state(step="saving", percent=97,
                      message="Writing results to flowlens.db…")

        # Store trained objects in model module registry for simulator
        import flowlens.model as _mm
        _mm._TRAINED_MODEL = model
        _mm._TRAINED_SCALER = feature_scaler
        _mm._TRAINED_FLOW_SCALER = flow_scaler
        _mm._X_SCALED = X_scaled
        _mm._META_DF = meta_df

        record_ingestion_run(
            repo_url=repo_url,
            since_date=str(since_date.date()),
            status="complete",
            commits_ingested=len(raw_df),
            run_id=run_id,
        )

        # ── DONE ─────────────────────────────────────────────────────
        _update_state(
            status="complete",
            step="ready",
            percent=100,
            message="FlowLens analysis complete ✓",
            developer_count=devs,
            developer_days=days,
        )
        logger.info("Background ingestion complete: %d dev-days, %d devs.", days, devs)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _update_state(
            status="error",
            step="error",
            percent=0,
            error=str(exc),
            message=f"Ingestion failed: {exc}",
        )
        logger.error("Background ingestion failed: %s", exc)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: dict | None = None) -> Flask:
    cfg = cfg or {}
    frontend_dir = str(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "frontend"
    )

    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    CORS(app)
    app.config["FLOWLENS_CFG"] = cfg

    # -----------------------------------------------------------------------
    # HTML routes
    # -----------------------------------------------------------------------

    @app.route("/")
    def home():
        """Landing page — repo input form."""
        return send_from_directory(frontend_dir, "home.html")

    @app.route("/dashboard")
    def dashboard():
        """Main flow analysis dashboard."""
        return send_from_directory(frontend_dir, "index.html")

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    def ok(data: object) -> tuple:
        return jsonify({"ok": True, "data": data}), 200

    def err(message: str, status: int = 400) -> tuple:
        return jsonify({"ok": False, "error": message}), status

    # -----------------------------------------------------------------------
    # POST /api/ingest
    # -----------------------------------------------------------------------

    @app.route("/api/ingest", methods=["POST"])
    def api_ingest():
        """
        Start background ingestion for a given GitHub repo URL.
        Returns immediately with a job_id; poll /api/ingestion-status for progress.
        """
        data = request.get_json(silent=True) or {}
        repo_url = (data.get("repo_url") or "").strip()

        if not repo_url:
            return err("repo_url is required", 400)

        # Reject if already running
        with _ingestion_lock:
            if _ingestion_state["status"] == "in_progress":
                return err("Ingestion already in progress. Wait for it to complete.", 409)

        job_id = f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        _update_state(
            status="in_progress", step="cloning", percent=0,
            message="", job_id=job_id, error=None, repo=repo_url,
            commits_ingested=None, developer_count=None, developer_days=None,
        )

        thread = threading.Thread(
            target=_run_ingestion_pipeline,
            args=(repo_url, cfg),
            daemon=True,
        )
        thread.start()

        logger.info("Background ingestion started: job_id=%s repo=%s", job_id, repo_url)
        return ok({"job_id": job_id, "message": "Ingestion started"})

    # -----------------------------------------------------------------------
    # GET /api/ingestion-status
    # -----------------------------------------------------------------------

    @app.route("/api/ingestion-status")
    def api_ingestion_status():
        """Polled by the frontend every ~1.2s during ingestion."""
        with _ingestion_lock:
            state = dict(_ingestion_state)

        if state["status"] == "idle":
            return ok({
                "status":  "idle",
                "step":    "",
                "percent": 0,
                "message": "No ingestion running",
            })

        if state["status"] == "error":
            return ok({
                "status":  "error",
                "step":    "error",
                "percent": 0,
                "error":   state.get("error") or "Unknown error",
                "message": state.get("message") or "",
            })

        payload = {
            "status":    state["status"],
            "step":      state["step"],
            "percent":   state["percent"],
            "message":   state["message"],
            "timestamp": state["timestamp"],
            "job_id":    state["job_id"],
        }

        if state["status"] == "complete":
            payload["commits_ingested"] = state["commits_ingested"]
            payload["developer_count"]  = state["developer_count"]
            payload["developer_days"]   = state["developer_days"]

        return ok(payload)

    # -----------------------------------------------------------------------
    # GET /api/status
    # -----------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        from flowlens.db import get_summary_stats, has_any_data

        stats = get_summary_stats()
        db_ready = has_any_data()

        return ok({
            "status": "ready" if (_model_module.is_model_ready() or db_ready) else "idle",
            "repo": stats.get("repo"),
            "total_developer_days": stats.get("total_developer_days", 0),
            "anomaly_count": stats.get("anomaly_count", 0),
            "mean_flow_score": stats.get("mean_flow_score"),
            "developer_count": stats.get("developer_count", 0),
            "raw_commit_count": stats.get("raw_commit_count", 0),
            "date_range": {
                "start": stats.get("date_start"),
                "end":   stats.get("date_end"),
            },
            "model": "IsolationForest",
            "cross_track_b": _has_ci_data(),
            "llm_available": bool(os.environ.get("HF_TOKEN")),
        })

    # -----------------------------------------------------------------------
    # GET /api/flow-scores
    # -----------------------------------------------------------------------

    @app.route("/api/flow-scores")
    def api_flow_scores():
        from flowlens.db import get_flow_scores, get_all_developers

        developer = request.args.get("developer")
        since     = request.args.get("since", _default_since())
        until     = request.args.get("until", _today())
        aggregate = request.args.get("aggregate", "false").lower() == "true"

        scores = get_flow_scores(developer=developer, since=since,
                                 until=until, aggregate=aggregate)

        if scores:
            all_scores = [s.get("flow_score") or 0 for s in scores]
            anomalies  = [s for s in scores if s.get("anomaly_label") == -1]
            best  = max(scores, key=lambda s: s.get("flow_score") or 0)
            worst = min(scores, key=lambda s: s.get("flow_score") or 100)
            summary = {
                "mean_flow_score": round(sum(all_scores) / len(all_scores), 1),
                "anomaly_count": len(anomalies),
                "best_day":  {"date": best.get("date"),  "score": best.get("flow_score")},
                "worst_day": {"date": worst.get("date"), "score": worst.get("flow_score")},
            }
        else:
            summary = {"mean_flow_score": None, "anomaly_count": 0}

        return ok({
            "scores":     scores,
            "developers": get_all_developers(),
            "summary":    summary,
        })

    # -----------------------------------------------------------------------
    # GET /api/shap/<date>/<developer_email>
    # -----------------------------------------------------------------------

    @app.route("/api/shap/<date>/<path:developer_email>")
    def api_shap(date: str, developer_email: str):
        from flowlens.db import get_shap_values
        from flowlens.insights import build_plain_text_summary

        developer_email = unquote(developer_email)
        row = get_shap_values(date, developer_email)
        if not row:
            return err(f"No data for developer={developer_email}, date={date}", 404)

        raw_shap_json = row.get("shap_values_json")
        if not raw_shap_json:
            return err("SHAP values not yet computed for this row.", 404)

        try:
            shap_entries = json.loads(raw_shap_json)
        except json.JSONDecodeError:
            return err("Corrupt SHAP data in database.", 500)

        shap_sorted = sorted(shap_entries,
                             key=lambda x: abs(x.get("shap_value", 0)),
                             reverse=True)
        enriched = _enrich_shap_entries(shap_sorted, row, developer_email, date)

        flow_score = float(row.get("flow_score") or 0)
        plain_summary = build_plain_text_summary(
            developer=developer_email, date=date,
            flow_score=flow_score, top_shap=enriched[:3],
        )

        return ok({
            "date":             date,
            "developer":        developer_email,
            "developer_name":   row.get("developer_name", developer_email),
            "flow_score":       round(flow_score, 1),
            "anomaly_label":    row.get("anomaly_label"),
            "anomaly_score_raw": row.get("anomaly_score_raw"),
            "shap_values":      enriched,
            "plain_text_summary": plain_summary,
            "top_feature":      enriched[0]["feature"] if enriched else None,
        })

    # -----------------------------------------------------------------------
    # GET /api/recommend/<date>/<developer_email>
    # -----------------------------------------------------------------------

    @app.route("/api/recommend/<date>/<path:developer_email>")
    def api_recommend(date: str, developer_email: str):
        from flowlens.db import get_shap_values
        from flowlens.insights import (
            generate_all_rule_recommendations,
            generate_llm_recommendation,
        )

        developer_email = unquote(developer_email)
        row = get_shap_values(date, developer_email)
        if not row:
            return err(f"No data for developer={developer_email}, date={date}", 404)

        try:
            shap_entries = json.loads(row.get("shap_values_json") or "[]")
        except json.JSONDecodeError:
            shap_entries = []

        shap_sorted = sorted(shap_entries,
                             key=lambda x: abs(x.get("shap_value", 0)),
                             reverse=True)
        enriched    = _enrich_shap_entries(shap_sorted, row, developer_email, date)
        flow_score  = float(row.get("flow_score") or 0)
        rule_recs   = generate_all_rule_recommendations(shap_sorted[:3])

        llm_rec = generate_llm_recommendation(
            developer=developer_email, date=date,
            top_shap_features=enriched[:3],
            rule_recommendations=rule_recs,
            flow_score=flow_score,
        )

        hours_per_point = (cfg.get("scoring", {}).get("hours_per_flow_point", 0.5))
        hours_estimate  = round((100 - flow_score) * hours_per_point, 1)

        return ok({
            "date":                       date,
            "developer":                  developer_email,
            "flow_score":                 round(flow_score, 1),
            "rule_based_recommendations": rule_recs,
            "llm_recommendation":         llm_rec,
            "llm_available":              bool(os.environ.get("HF_TOKEN")),
            "hours_recoverable_estimate": hours_estimate,
            "top_feature": shap_sorted[0]["feature"] if shap_sorted else None,
        })

    # -----------------------------------------------------------------------
    # POST /api/simulate
    # -----------------------------------------------------------------------

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        from flowlens.simulator import simulate_changes

        body       = request.get_json(silent=True) or {}
        developer  = body.get("developer")
        date       = body.get("date")
        changes    = body.get("changes", {})

        if not developer or not date:
            return err("Both 'developer' and 'date' are required.")
        if not changes:
            return err("'changes' dict is required and must not be empty.")

        hours_per_point = cfg.get("scoring", {}).get("hours_per_flow_point", 0.5)

        try:
            result = simulate_changes(
                developer_email=developer, date=date,
                changes=changes, hours_per_flow_point=hours_per_point,
            )
            return ok(result)
        except ValueError as exc:
            return err(str(exc), 404)
        except RuntimeError as exc:
            return err(str(exc), 503)
        except Exception as exc:
            logger.error("Simulation failed: %s", exc, exc_info=True)
            return err(f"Simulation error: {exc}", 500)

    # -----------------------------------------------------------------------
    # GET /api/trend
    # -----------------------------------------------------------------------

    @app.route("/api/trend")
    def api_trend():
        from flowlens.db import get_trend

        since     = request.args.get("since", _default_since())
        developer = request.args.get("developer")
        window    = int(request.args.get("window", 7))

        trend = get_trend(since=since, developer=developer, window=window)
        return ok({"trend": trend})

    # -----------------------------------------------------------------------
    # GET /api/developers
    # -----------------------------------------------------------------------

    @app.route("/api/developers")
    def api_developers():
        from flowlens.db import get_all_developers
        devs = get_all_developers()
        return ok({"developers": devs, "count": len(devs)})

    # -----------------------------------------------------------------------
    # GET /api/health
    # -----------------------------------------------------------------------

    @app.route("/api/health")
    def api_health():
        return jsonify({"ok": True, "service": "flowlens", "version": "1.0.0"}), 200

    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _default_since() -> str:
    return (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")


def _has_ci_data() -> bool:
    from flowlens.db import get_connection
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM developer_day_features WHERE test_failure_density > 0 LIMIT 1"
            ).fetchone()
        return row is not None
    except Exception:
        return False


def _get_column_baselines(developer_email: str) -> dict[str, float]:
    from flowlens.db import get_connection
    from flowlens.features import FEATURE_NAMES

    baselines: dict[str, float] = {}
    col_avgs = ", ".join(f"AVG({col}) as {col}" for col in FEATURE_NAMES)
    try:
        with get_connection() as conn:
            row = conn.execute(
                f"SELECT {col_avgs} FROM developer_day_features WHERE developer_email = ?",
                (developer_email,),
            ).fetchone()
        if row:
            baselines = {
                col: (float(row[col]) if row[col] is not None else 0.0)
                for col in FEATURE_NAMES
            }
    except Exception as exc:
        logger.warning("Could not compute baselines: %s", exc)
    return baselines


def _enrich_shap_entries(
    shap_sorted: list[dict],
    row: dict,
    developer_email: str,
    date: str,
) -> list[dict]:
    baselines = _get_column_baselines(developer_email)
    return [
        {
            "feature":       entry.get("feature", ""),
            "shap_value":    round(float(entry.get("shap_value", 0)), 4),
            "actual_value":  round(float(row.get(entry.get("feature", "")) or 0), 3),
            "baseline_mean": round(float(baselines.get(entry.get("feature", ""), 0)), 3),
        }
        for entry in shap_sorted
    ]