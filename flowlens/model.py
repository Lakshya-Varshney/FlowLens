"""
model.py — ML anomaly detection core for FlowLens.

Implements:
  - Isolation Forest training (unsupervised, no labels needed)
  - Flow Score conversion (raw anomaly score → 0-100 scale)
  - SHAP explainability (TreeExplainer for feature-level diagnosis)

The trained model objects are stored in module-level globals so the
Flask API and simulator can access them without re-training.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

try:
    import shap as _shap_lib
    _SHAP_AVAILABLE = True
except ImportError:
    _shap_lib = None
    _SHAP_AVAILABLE = False
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger("flowlens.model")

# ---------------------------------------------------------------------------
# Module-level model registry
# These are populated by run.py after training and consumed by api.py / simulator.py
# ---------------------------------------------------------------------------
_TRAINED_MODEL: IsolationForest | None = None
_TRAINED_SCALER: StandardScaler | None = None
_TRAINED_FLOW_SCALER: MinMaxScaler | None = None
_X_SCALED: np.ndarray | None = None
_META_DF = None  # pd.DataFrame — set by run.py


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train_model(
    X_scaled: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.1,
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler, MinMaxScaler]:
    """
    Train an Isolation Forest anomaly detector on the scaled feature matrix.

    Args:
        X_scaled:      Pre-scaled feature matrix (shape: n_rows × n_features).
        n_estimators:  Number of isolation trees.
        contamination: Expected fraction of anomalous rows (~10% is a safe default).
        random_state:  Seed for reproducibility.

    Returns:
        (fitted IsolationForest, StandardScaler placeholder, fitted MinMaxScaler for flow scores)

    Note:
        The StandardScaler returned here is a NEW one fitted on X_scaled so the
        simulator can access it. The scaler used for feature engineering lives in
        features.py; this one is stored in the registry for the simulator.
    """
    logger.info(
        "Training IsolationForest: n_estimators=%d, contamination=%.2f, n_samples=%d",
        n_estimators, contamination, X_scaled.shape[0],
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Build a MinMaxScaler to convert raw anomaly scores → 0-100 flow scores
    raw_scores = model.score_samples(X_scaled)
    flow_scaler = MinMaxScaler(feature_range=(0, 100))
    flow_scaler.fit(raw_scores.reshape(-1, 1))

    # Return None for scaler — the correct feature scaler comes from features.py
    # and is passed in from run.py via _cache_model_objects.
    # Returning a scaler fitted on X_scaled (already scaled data) would be wrong.
    logger.info("IsolationForest training complete.")
    return model, None, flow_scaler


def compute_flow_scores(
    model: IsolationForest,
    X_scaled: np.ndarray,
    flow_scaler: MinMaxScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Flow Scores for every row in the feature matrix.

    Args:
        model:        Fitted IsolationForest.
        X_scaled:     Scaled feature matrix.
        flow_scaler:  Fitted MinMaxScaler (0-100 range).

    Returns:
        flow_scores      — float array, 0 (worst) to 100 (best)
        anomaly_labels   — int array, -1 (anomalous) or 1 (normal)
        anomaly_scores   — float array, raw IsolationForest scores (more negative = worse)
    """
    anomaly_scores_raw = model.score_samples(X_scaled)
    anomaly_labels = model.predict(X_scaled)

    # IsolationForest: more negative score = more anomalous.
    # We INVERT so that higher flow_score = better health.
    flow_scores = flow_scaler.transform(anomaly_scores_raw.reshape(-1, 1)).flatten()

    logger.debug(
        "Flow scores — min=%.1f max=%.1f mean=%.1f anomalies=%d",
        flow_scores.min(), flow_scores.max(), flow_scores.mean(),
        (anomaly_labels == -1).sum(),
    )
    return flow_scores, anomaly_labels, anomaly_scores_raw


def compute_shap_values(
    model: IsolationForest,
    X_scaled: np.ndarray,
    max_rows: int = 2000,
) -> np.ndarray:
    """
    Compute SHAP values for the feature matrix using TreeExplainer.

    For very large datasets we subsample to max_rows to keep runtime reasonable,
    then use those SHAP values as an approximation for the full set.
    Note: we compute SHAP for all rows but use a background sample for efficiency.

    Args:
        model:     Fitted IsolationForest.
        X_scaled:  Scaled feature matrix.
        max_rows:  Maximum rows for background sample.

    Returns:
        shap_matrix — numpy array of shape (n_rows, n_features)
    """
    n_rows = X_scaled.shape[0]

    # Build background sample for the explainer
    if n_rows > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_rows, size=max_rows, replace=False)
        background = X_scaled[idx]
    else:
        background = X_scaled

    try:
        if not _SHAP_AVAILABLE:
            raise ImportError("shap library not installed")
        explainer = _shap_lib.TreeExplainer(model, data=background, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)
    except Exception as exc:
        # SHAP may fail for very small datasets; fall back to zeros
        logger.warning("SHAP computation failed (%s) — returning zero matrix.", exc)
        shap_values = np.zeros_like(X_scaled)

    return shap_values


# ---------------------------------------------------------------------------
# Accessors for the module-level registry
# ---------------------------------------------------------------------------

def get_trained_model() -> IsolationForest | None:
    """Return the trained IsolationForest, or None if not yet trained."""
    return _TRAINED_MODEL


def get_flow_scaler() -> MinMaxScaler | None:
    """Return the fitted MinMaxScaler, or None if not yet trained."""
    return _TRAINED_FLOW_SCALER


def get_scaler() -> StandardScaler | None:
    """Return the fitted StandardScaler, or None if not yet trained."""
    return _TRAINED_SCALER


def get_meta_df():
    """Return the metadata DataFrame aligned with the trained model."""
    return _META_DF


def is_model_ready() -> bool:
    """Return True when all model objects are loaded and ready."""
    return all([
        _TRAINED_MODEL is not None,
        _TRAINED_FLOW_SCALER is not None,
        _X_SCALED is not None,
    ])


# ---------------------------------------------------------------------------
# Inference helpers (used by simulator.py and api.py)
# ---------------------------------------------------------------------------

def score_single_row(x_raw_row: np.ndarray) -> float:
    """
    Compute the flow score for a single raw (unscaled) feature vector.

    Args:
        x_raw_row: 1D numpy array of length n_features (unscaled values
                   in the same units as the original features, e.g. minutes,
                   ratios, counts — NOT pre-scaled).

    Returns:
        Flow score in range [0, 100].
    """
    if not is_model_ready():
        raise RuntimeError("Model is not trained yet.")

    # Use the feature scaler from features.py — this is the correct scaler
    # that transforms raw feature values into the same space the model was
    # trained on. Using any other scaler would produce wrong scores.
    x_scaled = _TRAINED_SCALER.transform(x_raw_row.reshape(1, -1))
    raw_score = _TRAINED_MODEL.score_samples(x_scaled)

    # Transform to 0-100 flow score
    # Clip raw_score to the range the flow_scaler was fitted on to prevent
    # out-of-range values from being clipped to exactly 0 or 100.
    min_seen = _TRAINED_FLOW_SCALER.data_min_[0]
    max_seen = _TRAINED_FLOW_SCALER.data_max_[0]
    clipped_score = float(np.clip(raw_score, min_seen, max_seen))

    flow_score = float(
        _TRAINED_FLOW_SCALER.transform([[clipped_score]]).flatten()[0]
    )
    return max(0.0, min(100.0, flow_score))