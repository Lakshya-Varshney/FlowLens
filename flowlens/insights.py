"""
insights.py — Recommendation engine for FlowLens.

Two-layer approach:
  Layer 1: Rule-based mapping from top SHAP feature → concrete recommendation.
  Layer 2: Optional LLM synthesis via Google AI Studio API (when GOOGLE_API_KEY is set).

The rule-based layer always runs first and never requires an API key.
The LLM layer enriches the recommendation with context-aware language.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("flowlens.insights")

# Path for recommendation cache (avoids redundant LLM calls)
_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "recommendations_cache.json"

# ---------------------------------------------------------------------------
# Rule-based recommendation map
# Maps the most impactful SHAP feature name → actionable team recommendation
# ---------------------------------------------------------------------------

RECOMMENDATION_MAP: dict[str, str] = {
    "pr_blocking_time_hours": (
        "Reduce PR review latency. "
        "Set a team SLA of <4 hours for first review. "
        "Consider a rotating review-duty engineer each sprint day. "
        "Large PRs waiting for review are the #1 flow killer in this data."
    ),
    "inter_commit_gap_variance": (
        "Work sessions are fragmented. "
        "Implement 90-minute focus blocks with Slack notifications paused. "
        "High gap variance correlates with meeting overload — audit the calendar for this period."
    ),
    "late_night_ratio": (
        "High late-night coding detected. "
        "This developer is likely overloaded or blocked during core hours. "
        "Investigate task assignment and blockers from this period. "
        "Sustainable pace: protect core-hours productivity."
    ),
    "pr_size_lines": (
        "PRs are too large for efficient review. "
        "Enforce a soft limit of 200 lines changed per PR. "
        "Large PRs block the author's flow while waiting for reviewers and "
        "increase the probability of review delays."
    ),
    "test_failure_density": (
        "Frequent CI failures disrupted commit flow. "
        "Investigate flaky tests from this period (Cross-Track B). "
        "Add pre-commit test filtering to catch failures earlier. "
        "Target: <2% flaky rate in the test suite."
    ),
    "build_time_minutes": (
        "Slow build times created flow-breaking wait loops. "
        "Profile the CI pipeline for caching opportunities. "
        "Target: <5 minutes for incremental builds. "
        "Consider parallelising test stages."
    ),
    "merge_commit_ratio": (
        "Excessive merge commits indicate long-lived branches. "
        "Enforce trunk-based development or short-lived feature branches "
        "(max 2 days). Long-lived branches cause integration pain and context switching."
    ),
    "session_duration_minutes": (
        "Session durations are abnormally long or short. "
        "Long sessions without breaks reduce code quality. "
        "Very short sessions indicate constant interruptions. "
        "Encourage Pomodoro-style 90-minute focus blocks."
    ),
    "daily_commit_count": (
        "Commit frequency is anomalous. "
        "Very low commit count may indicate blockage. "
        "Very high commit count may indicate frantic hotfixing. "
        "Aim for consistent, small, meaningful commits."
    ),
    "inter_commit_gap_mean": (
        "Average time between commits is abnormal. "
        "Unusually long average gaps suggest blockers (waiting for reviews, "
        "environment issues, or unclear requirements). "
        "Investigate what caused the delays on this day."
    ),
    "weekend_ratio": (
        "High weekend commit activity detected. "
        "This is a sustainability risk. "
        "Review sprint planning to ensure the team is not carrying over work "
        "consistently into weekends."
    ),
    "avg_commit_size_lines": (
        "Average commit size is large. "
        "Encourage smaller, atomic commits that are easier to review and roll back. "
        "Large commits are harder to review and more likely to introduce bugs."
    ),
    "files_touched_diversity": (
        "Developer touched an unusually high number of distinct files. "
        "This may indicate context-switching across unrelated areas. "
        "Consider better task focus or pair programming for complex cross-cutting work."
    ),
    "commit_message_length_mean": (
        "Commit messages are unusually short or very long. "
        "Short messages (< 20 chars) suggest rushed commits under stress. "
        "Establish a commit message template: type(scope): brief description."
    ),
}

# Default fallback if feature isn't in the map
_DEFAULT_RECOMMENDATION = (
    "An anomalous pattern was detected in the developer's workflow. "
    "Review the feature breakdown above for specific contributing factors. "
    "Consider a team retrospective to identify root causes."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rule_recommendation(top_shap_feature: str) -> str:
    """
    Return a rule-based recommendation for the given top SHAP feature.

    Args:
        top_shap_feature: Name of the feature with the largest (most negative) SHAP value.

    Returns:
        Human-readable recommendation string.
    """
    return RECOMMENDATION_MAP.get(top_shap_feature, _DEFAULT_RECOMMENDATION)


def generate_all_rule_recommendations(shap_entries: list[dict]) -> list[str]:
    """
    Generate rule-based recommendations for the top-N SHAP features.

    Args:
        shap_entries: List of dicts with keys 'feature' and 'shap_value',
                      sorted by abs(shap_value) descending.

    Returns:
        List of recommendation strings (one per top feature, de-duplicated).
    """
    recs: list[str] = []
    seen: set[str] = set()

    for entry in shap_entries[:3]:  # top 3 features
        feat = entry.get("feature", "")
        rec = RECOMMENDATION_MAP.get(feat)
        if rec and rec not in seen:
            recs.append(rec)
            seen.add(rec)

    if not recs:
        recs.append(_DEFAULT_RECOMMENDATION)

    return recs


def generate_llm_recommendation(
    developer: str,
    date: str,
    top_shap_features: list[dict],
    rule_recommendations: list[str],
    flow_score: float,
) -> Optional[str]:
    """
    Call a model via HuggingFace Inference Providers (OpenAI-compatible router).

    Uses https://router.huggingface.co/v1 — HuggingFace's unified inference
    router that automatically picks the best available provider for the model.

    Free tier: ~$0 for small daily usage with a free HF account.
    Model: Qwen/Qwen2.5-72B-Instruct — reliable, free-tier friendly, excellent
    at structured reasoning and actionable advice.

    Requires HF_TOKEN in environment (free at huggingface.co/settings/tokens).
    Token must have "Make calls to Inference Providers" permission enabled.

    Falls back to None when:
      - HF_TOKEN is not set
      - The API call fails
      - The result is already cached
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.debug("HF_TOKEN not set — skipping LLM recommendation.")
        return None

    # Check cache first — avoids redundant API calls on re-runs
    cache_key = f"{developer}_{date}"
    cached = _read_cache(cache_key)
    if cached:
        logger.debug("Using cached LLM recommendation for %s", cache_key)
        return cached

    prompt = _build_llm_prompt(developer, date, top_shap_features, rule_recommendations, flow_score)

    try:
        from openai import OpenAI

        # HuggingFace Inference Providers — OpenAI-compatible router
        # Automatically selects best available provider for the model
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",   # free tier, fast, great reasoning
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a developer experience (DX) expert. "
                        "Give concise, specific, actionable advice. "
                        "Always complete your full response."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=500,
            temperature=0.4,
        )

        recommendation_text = response.choices[0].message.content.strip()

        _write_cache(cache_key, recommendation_text)
        logger.info("HuggingFace recommendation generated for %s on %s.", developer, date)
        return recommendation_text

    except ImportError:
        logger.warning(
            "openai package not installed. "
            "Run: pip install openai"
        )
        return None
    except Exception as exc:
        logger.warning("HuggingFace Inference Providers call failed: %s", exc)
        return None


def build_plain_text_summary(
    developer: str,
    date: str,
    flow_score: float,
    top_shap: list[dict],
) -> str:
    """
    Build a concise plain-text summary of the anomaly without an LLM.

    Args:
        developer:  Developer email.
        date:       YYYY-MM-DD.
        flow_score: Current flow score.
        top_shap:   Top SHAP entries (dicts with 'feature', 'shap_value',
                    'actual_value', 'baseline_mean').

    Returns:
        Summary string.
    """
    if not top_shap:
        return f"{developer}'s flow on {date} was anomalous (score: {flow_score:.0f}/100)."

    top = top_shap[0]
    feature_label = top["feature"].replace("_", " ")
    direction = "above" if top.get("shap_value", 0) < 0 else "below"
    actual = top.get("actual_value", "N/A")
    baseline = top.get("baseline_mean", "N/A")

    summary = (
        f"{developer}'s flow on {date} scored {flow_score:.0f}/100 — "
        f"primarily disrupted by {feature_label} "
        f"({_fmt(actual)} vs baseline {_fmt(baseline)}, {direction} normal)."
    )
    if len(top_shap) > 1:
        second = top_shap[1]["feature"].replace("_", " ")
        summary += f" Secondary factor: {second}."

    return summary


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_llm_prompt(
    developer: str,
    date: str,
    top_shap: list[dict],
    rule_recs: list[str],
    flow_score: float,
) -> str:
    feature_lines = "\n".join([
        f"- {e['feature'].replace('_', ' ')}: "
        f"{abs(e.get('shap_value', 0)) * 100:.0f}% "
        f"{'above' if e.get('shap_value', 0) < 0 else 'below'} baseline"
        for e in top_shap[:3]
    ])
    rule_lines = "\n".join(f"- {r}" for r in rule_recs)

    return f"""You are a developer experience (DX) expert analysing a software team's workflow.

Developer: {developer}
Date: {date}
Flow Score: {flow_score:.0f}/100 (scale: 0=severely disrupted, 100=optimal focus)

ML anomaly detection identified these primary contributing factors (ranked by impact):
{feature_lines}

Initial rule-based recommendations:
{rule_lines}

Respond in exactly this structure — plain text only, no markdown, no asterisks, no bold:

WHAT HAPPENED:
Write 2 sentences explaining what likely disrupted this developer's productivity on {date}. Be specific and empathetic.

PRIORITISED ACTIONS:
1. [Action title]: One sentence explaining what to do and why it helps.
2. [Action title]: One sentence explaining what to do and why it helps.
3. [Action title]: One sentence explaining what to do and why it helps.

HOURS RECOVERED:
One sentence estimating developer-hours recovered per week if these actions are taken.

Keep the total response under 200 words. No markdown formatting."""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _read_cache(key: str) -> Optional[str]:
    """Read a cached recommendation from disk."""
    try:
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH) as f:
                cache = json.load(f)
            return cache.get(key)
    except Exception:
        pass
    return None


def _write_cache(key: str, value: str) -> None:
    """Write a recommendation to the disk cache."""
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache: dict = {}
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH) as f:
                cache = json.load(f)
        cache[key] = value
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to write recommendation cache: %s", exc)


def _fmt(val) -> str:
    """Format a numeric value for display."""
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return str(val)