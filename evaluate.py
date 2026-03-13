"""Evaluation script for Sepsis Detection Model."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, setup_logger
from src.data_loader import load_data, prepare_data
from src.features import TimeSeriesFeatureEngineer, FeatureSelector
from src.models import GradientBoostingPipeline
import joblib


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main evaluation function."""
    # Load configuration
    config = load_config()
    logger = setup_logger(
        log_file=config["logging"]["log_file"], log_level=config["logging"]["log_level"]
    )

    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)

    # Load model
    model_path = config["training"]["model_save_path"]
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using train.py")
        return

    logger.info(f"Loading model from {model_path}")
    pipeline = GradientBoostingPipeline()
    pipeline.load(model_path)

    # Load feature selector if it exists
    feature_selector = None
    selector_path = Path("models/feature_selector.pkl")
    if selector_path.exists():
        logger.info("Loading feature selector...")
        feature_selector = joblib.load(selector_path)

    # Load test data
    logger.info(f"Loading data from {config['data']['raw_data_path']}")
    try:
        df = load_data(config["data"]["raw_data_path"])
    except FileNotFoundError:
        logger.error(f"Data file not found: {config['data']['raw_data_path']}")
        return

    # Feature engineering
    logger.info("Performing feature engineering...")
    feature_engineer = TimeSeriesFeatureEngineer(
        time_window_hours=config["features"]["time_window_hours"],
        lookback_hours=config["features"]["lookback_hours"],
        include_static=config["features"]["include_static_features"],
        include_temporal=config["features"]["include_temporal_features"],
    )

    df_features = feature_engineer.create_features(df)

    # Prepare data
    X, y = prepare_data(df_features)

    # Apply feature selection if available
    if feature_selector is not None:
        logger.info("Applying feature selection...")
        X = feature_selector.transform(X)

    # Make predictions
    logger.info("Making predictions...")
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "average_precision": average_precision_score(y, y_pred_proba),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results:")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y, y_pred))

    # Save plots
    if config["evaluation"]["save_plots"]:
        plots_path = Path(config["evaluation"]["plots_path"])
        plots_path.mkdir(parents=True, exist_ok=True)

        logger.info("Generating evaluation plots...")
        plot_confusion_matrix(y, y_pred, plots_path / "confusion_matrix.png")
        plot_roc_curve(y, y_pred_proba, plots_path / "roc_curve.png")
        plot_precision_recall_curve(y, y_pred_proba, plots_path / "pr_curve.png")
        logger.info(f"Plots saved to {plots_path}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = Path(config["evaluation"]["plots_path"]) / "evaluation_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
