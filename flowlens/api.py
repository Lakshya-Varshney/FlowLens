"""
api.py — Flask REST API for FlowLens.

Endpoints:
  GET  /api/status                         — health + dataset summary
  GET  /api/flow-scores                    — heatmap data
  GET  /api/shap/<date>/<developer_email>  — SHAP explanation for one anomaly
  GET  /api/recommend/<date>/<developer>   — recommendation (rule + optional LLM)
  POST /api/simulate                       — before/after simulator
  GET  /api/trend                          — rolling average trend line
  GET  /api/developers                     — list of all developers

All responses follow the envelope: {"ok": bool, "data": ..., "error": ...}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from urllib.parse import unquote

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from flowlens import model as _model_module

logger = logging.getLogger("flowlens.api")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: dict | None = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        cfg: Loaded config.yaml dict (passed in from run.py).

    Returns:
        Configured Flask app instance.
    """
    cfg = cfg or {}
    frontend_dir = str(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "frontend"
    )

    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    CORS(app)  # Allow cross-origin for local development

    # Store config in app context
    app.config["FLOWLENS_CFG"] = cfg

    # -----------------------------------------------------------------------
    # Static frontend route
    # -----------------------------------------------------------------------

    @app.route("/")
    def index():
        return send_from_directory(frontend_dir, "index.html")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def ok(data: object) -> tuple:
        return jsonify({"ok": True, "data": data}), 200

    def err(message: str, status: int = 400) -> tuple:
        return jsonify({"ok": False, "error": message}), status

    # -----------------------------------------------------------------------
    # GET /api/status
    # -----------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        """
        Returns the current system status and dataset summary statistics.
        """
        from flowlens.db import get_summary_stats

        stats = get_summary_stats()
        return ok({
            "status": "ready" if _model_module.is_model_ready() else "initializing",
            "repo": stats.get("repo"),
            "total_developer_days": stats.get("total_developer_days", 0),
            "anomaly_count": stats.get("anomaly_count", 0),
            "mean_flow_score": stats.get("mean_flow_score"),
            "developer_count": stats.get("developer_count", 0),
            "raw_commit_count": stats.get("raw_commit_count", 0),
            "date_range": {
                "start": stats.get("date_start"),
                "end": stats.get("date_end"),
            },
            "model": "IsolationForest",
            "cross_track_b": _has_ci_data(),
            "llm_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        })

    # -----------------------------------------------------------------------
    # GET /api/flow-scores
    # -----------------------------------------------------------------------

    @app.route("/api/flow-scores")
    def api_flow_scores():
        """
        Returns per-developer-day flow scores for the heatmap calendar.

        Query params:
          developer  — filter by email (optional)
          since      — YYYY-MM-DD (optional, default 90 days ago)
          until      — YYYY-MM-DD (optional, default today)
          aggregate  — 'true' to return daily team averages
        """
        from flowlens.db import get_flow_scores, get_all_developers

        developer = request.args.get("developer")
        since = request.args.get("since", _default_since())
        until = request.args.get("until", _today())
        aggregate = request.args.get("aggregate", "false").lower() == "true"

        scores = get_flow_scores(
            developer=developer,
            since=since,
            until=until,
            aggregate=aggregate,
        )

        # Summary stats
        if scores:
            all_scores = [s.get("flow_score") or 0 for s in scores]
            anomalies = [s for s in scores if s.get("anomaly_label") == -1]
            best = max(scores, key=lambda s: s.get("flow_score") or 0)
            worst = min(scores, key=lambda s: s.get("flow_score") or 100)
            summary = {
                "mean_flow_score": round(sum(all_scores) / len(all_scores), 1),
                "anomaly_count": len(anomalies),
                "best_day": {"date": best.get("date"), "score": best.get("flow_score")},
                "worst_day": {"date": worst.get("date"), "score": worst.get("flow_score")},
            }
        else:
            summary = {"mean_flow_score": None, "anomaly_count": 0}

        return ok({
            "scores": scores,
            "developers": get_all_developers(),
            "summary": summary,
        })

    # -----------------------------------------------------------------------
    # GET /api/shap/<date>/<developer_email>
    # -----------------------------------------------------------------------

    @app.route("/api/shap/<date>/<path:developer_email>")
    def api_shap(date: str, developer_email: str):
        """
        Returns SHAP feature attributions for a specific developer-day.
        """
        from flowlens.db import get_shap_values
        from flowlens.features import FEATURE_NAMES
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

        # Sort by absolute SHAP value descending (most impactful first)
        shap_entries_sorted = sorted(
            shap_entries,
            key=lambda x: abs(x.get("shap_value", 0)),
            reverse=True,
        )

        # Enrich with actual vs baseline values
        enriched = _enrich_shap_entries(shap_entries_sorted, row, developer_email, date)

        flow_score = float(row.get("flow_score") or 0)
        plain_summary = build_plain_text_summary(
            developer=developer_email,
            date=date,
            flow_score=flow_score,
            top_shap=enriched[:3],
        )

        return ok({
            "date": date,
            "developer": developer_email,
            "developer_name": row.get("developer_name", developer_email),
            "flow_score": round(flow_score, 1),
            "anomaly_label": row.get("anomaly_label"),
            "anomaly_score_raw": row.get("anomaly_score_raw"),
            "shap_values": enriched,
            "plain_text_summary": plain_summary,
            "top_feature": enriched[0]["feature"] if enriched else None,
        })

    # -----------------------------------------------------------------------
    # GET /api/recommend/<date>/<developer_email>
    # -----------------------------------------------------------------------

    @app.route("/api/recommend/<date>/<path:developer_email>")
    def api_recommend(date: str, developer_email: str):
        """
        Returns rule-based and (optionally) LLM-synthesised recommendations.
        """
        from flowlens.db import get_shap_values
        from flowlens.insights import (
            generate_all_rule_recommendations,
            generate_llm_recommendation,
            build_plain_text_summary,
        )

        developer_email = unquote(developer_email)

        row = get_shap_values(date, developer_email)
        if not row:
            return err(f"No data for developer={developer_email}, date={date}", 404)

        raw_shap_json = row.get("shap_values_json", "[]")
        try:
            shap_entries = json.loads(raw_shap_json or "[]")
        except json.JSONDecodeError:
            shap_entries = []

        shap_sorted = sorted(shap_entries, key=lambda x: abs(x.get("shap_value", 0)), reverse=True)
        enriched = _enrich_shap_entries(shap_sorted, row, developer_email, date)

        flow_score = float(row.get("flow_score") or 0)
        rule_recs = generate_all_rule_recommendations(shap_sorted[:3])

        # Attempt LLM recommendation
        llm_rec = generate_llm_recommendation(
            developer=developer_email,
            date=date,
            top_shap_features=enriched[:3],
            rule_recommendations=rule_recs,
            flow_score=flow_score,
        )

        # Estimate recoverable hours
        hours_per_point = (
            app.config["FLOWLENS_CFG"]
            .get("scoring", {})
            .get("hours_per_flow_point", 0.5)
        )
        max_improvement = 100 - flow_score
        hours_estimate = round(max_improvement * hours_per_point, 1)

        return ok({
            "date": date,
            "developer": developer_email,
            "flow_score": round(flow_score, 1),
            "rule_based_recommendations": rule_recs,
            "llm_recommendation": llm_rec,
            "llm_available": bool(os.environ.get("GOOGLE_API_KEY")),
            "hours_recoverable_estimate": hours_estimate,
            "top_feature": shap_sorted[0]["feature"] if shap_sorted else None,
        })

    # -----------------------------------------------------------------------
    # POST /api/simulate
    # -----------------------------------------------------------------------

    @app.route("/api/simulate", methods=["POST"])
    def api_simulate():
        """
        Runs the Before/After simulator.

        Expected JSON body:
          {
            "developer": "alice@example.com",
            "date": "2026-03-14",
            "changes": {
              "pr_review_time_hours": 4.0,
              "max_pr_size_lines": 200,
              "build_time_minutes": 5.0
            }
          }
        """
        from flowlens.simulator import simulate_changes

        body = request.get_json(silent=True) or {}
        developer = body.get("developer")
        date = body.get("date")
        changes = body.get("changes", {})

        if not developer or not date:
            return err("Both 'developer' and 'date' are required fields.")

        if not changes:
            return err("'changes' dict is required and must not be empty.")

        hours_per_point = (
            app.config["FLOWLENS_CFG"]
            .get("scoring", {})
            .get("hours_per_flow_point", 0.5)
        )

        try:
            result = simulate_changes(
                developer_email=developer,
                date=date,
                changes=changes,
                hours_per_flow_point=hours_per_point,
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
        """
        Returns rolling 7-day average team flow score for the trend chart.

        Query params:
          since     — YYYY-MM-DD (optional, default 90 days ago)
          developer — filter by email (optional)
          window    — rolling window size in days (default 7)
        """
        from flowlens.db import get_trend

        since = request.args.get("since", _default_since())
        developer = request.args.get("developer")
        window = int(request.args.get("window", 7))

        trend = get_trend(since=since, developer=developer, window=window)
        return ok({"trend": trend})

    # -----------------------------------------------------------------------
    # GET /api/developers
    # -----------------------------------------------------------------------

    @app.route("/api/developers")
    def api_developers():
        """Returns the list of distinct developer emails in the dataset."""
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
    """Check if any CI test failure data exists in the database."""
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
    """
    Compute per-developer baseline means for each feature column.
    Used to enrich SHAP entries with 'actual vs baseline' context.
    """
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
            baselines = {col: (float(row[col]) if row[col] is not None else 0.0) for col in FEATURE_NAMES}
    except Exception as exc:
        logger.warning("Could not compute baselines: %s", exc)

    return baselines


def _enrich_shap_entries(
    shap_sorted: list[dict],
    row: dict,
    developer_email: str,
    date: str,
) -> list[dict]:
    """
    Enrich SHAP entries with actual feature values and per-developer baselines.
    """
    baselines = _get_column_baselines(developer_email)
    enriched = []
    for entry in shap_sorted:
        feat = entry.get("feature", "")
        enriched.append({
            "feature": feat,
            "shap_value": round(float(entry.get("shap_value", 0)), 4),
            "actual_value": round(float(row.get(feat) or 0), 3),
            "baseline_mean": round(float(baselines.get(feat, 0)), 3),
        })
    return enriched