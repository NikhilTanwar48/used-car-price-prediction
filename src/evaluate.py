# car-price-predictor/src/evaluate.py
"""Generate evaluation plots for the trained forecasting pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.data_loader import load_config, load_vehicle_dataframe
from src.preprocessing import load_preprocessor, transform_features

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_bundle(model_path: Path) -> dict[str, Any]:
    obj: dict[str, Any] | Any = joblib.load(model_path)
    if isinstance(obj, dict) and "estimator" in obj:
        return obj
    return {"estimator": obj, "model_name": "unknown", "train_residual_std": 0.0}


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.45, edgecolor=None)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Predicted price")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Price")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.45, edgecolor=None)
    lims = [
        float(min(y_true.min(), y_pred.min())),
        float(max(y_true.max(), y_pred.max())),
    ]
    plt.plot(lims, lims, "r--", lw=2)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_feature_importance(
    bundle: dict[str, Any],
    feature_names: np.ndarray,
    out_path: Path,
) -> None:
    model = bundle["estimator"]
    model_name = bundle.get("model_name", "")
    importances: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        title = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()
        title = "Absolute Linear Coefficients"
    else:
        logger.warning("Model %s lacks importance metrics; skipping plot.", model_name)
        return

    order = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10, 6))
    labels = [feature_names[i] for i in order][::-1]
    values = importances[order][::-1]
    plt.barh(labels, values, color="#2563eb")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_pipeline(config_path: Path | None = None) -> dict[str, str]:
    """
    Recreate hold-out predictions and persist diagnostic figures.

    Args:
        config_path: Optional YAML configuration path.

    Returns:
        Mapping of figure key to file path.
    """
    cfg = load_config(config_path)
    root = _project_root()

    df = load_vehicle_dataframe(cfg)
    target_col = "selling_price"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train_raw, X_test_raw, _y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["train_test"]["test_size"]),
        random_state=int(cfg["train_test"]["random_state"]),
    )

    models_dir = root / Path(cfg["paths"]["models_dir"])
    fe_path = models_dir / cfg["paths"]["feature_engineering_filename"]
    engineer = joblib.load(fe_path)
    X_test_eng = engineer.transform(X_test_raw)

    preprocessor = load_preprocessor(models_dir / cfg["paths"]["preprocessor_filename"])
    model_path = models_dir / cfg["paths"]["model_filename"]
    bundle = _load_bundle(model_path)

    X_test_mat = transform_features(preprocessor, X_test_eng)
    preds = bundle["estimator"].predict(X_test_mat)

    figures_dir = root / Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    residual_path = figures_dir / "residuals.png"
    scatter_path = figures_dir / "predicted_vs_actual.png"
    fi_path = figures_dir / "feature_importance.png"

    y_true = y_test.to_numpy()
    _plot_residuals(y_true, preds, residual_path)
    _plot_pred_vs_actual(y_true, preds, scatter_path)

    feature_names = preprocessor.get_feature_names_out()
    _plot_feature_importance(bundle, feature_names, fi_path)

    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    logger.info("Evaluation RMSE on reconstructed hold-out: %.4f", rmse)

    return {
        "residuals": str(residual_path),
        "scatter": str(scatter_path),
        "feature_importance": str(fi_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    outputs = evaluate_pipeline()
    logger.info("Saved figures: %s", outputs)
