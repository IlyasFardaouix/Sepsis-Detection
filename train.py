"""Training script for Sepsis Detection Model."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, setup_logger
from src.data_loader import load_data, prepare_data, split_data
from src.features import TimeSeriesFeatureEngineer, FeatureSelector
from src.models import GradientBoostingPipeline, ImbalancedDataHandler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    logger = setup_logger(
        log_file=config["logging"]["log_file"], log_level=config["logging"]["log_level"]
    )

    logger.info("=" * 60)
    logger.info("Starting Sepsis Detection Model Training")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading data from {config['data']['raw_data_path']}")
    try:
        df = load_data(config["data"]["raw_data_path"])
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Data shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Data file not found: {config['data']['raw_data_path']}")
        logger.info("Please generate sample data using generate_sample_data.py")
        return

    # Feature engineering
    logger.info("Performing time-series feature engineering...")
    feature_engineer = TimeSeriesFeatureEngineer(
        time_window_hours=config["features"]["time_window_hours"],
        lookback_hours=config["features"]["lookback_hours"],
        include_static=config["features"]["include_static_features"],
        include_temporal=config["features"]["include_temporal_features"],
    )

    df_features = feature_engineer.create_features(df)
    logger.info(f"Feature engineering complete. New shape: {df_features.shape}")

    # Prepare data
    logger.info("Preparing data for modeling...")
    X, y = prepare_data(df_features)

    # Check class distribution
    if y is not None:
        class_dist = y.value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
        logger.info(f"Class imbalance ratio: {class_dist[0] / class_dist[1]:.2f}")

    # Split data
    logger.info("Splitting data into train/validation/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        test_size=config["data"]["train_test_split"],
        validation_size=config["data"]["validation_split"],
        random_state=config["data"]["random_seed"],
        stratify=config["training"]["stratified"],
    )

    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Feature selection
    if config["features"]["feature_selection"]:
        logger.info("Performing feature selection...")
        feature_selector = FeatureSelector(
            method="mutual_info",
            max_features=config["features"]["max_features"],
            random_state=config["data"]["random_seed"],
        )
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_val = feature_selector.transform(X_val) if len(X_val) > 0 else X_val
        X_test = feature_selector.transform(X_test)
        logger.info(f"Selected {X_train.shape[1]} features")

        # Save feature selector
        joblib.dump(feature_selector, "models/feature_selector.pkl")

    # Handle imbalanced data
    logger.info(
        f"Handling imbalanced data using: {config['imbalanced_learning']['method']}"
    )
    imbalanced_handler = ImbalancedDataHandler(
        method=config["imbalanced_learning"]["method"],
        k_neighbors=config["imbalanced_learning"]["k_neighbors"],
        sampling_strategy=config["imbalanced_learning"]["sampling_strategy"],
        random_state=config["data"]["random_seed"],
    )

    X_train_resampled, y_train_resampled = imbalanced_handler.fit_resample(
        X_train, y_train
    )
    logger.info(f"After resampling - Train set: {X_train_resampled.shape[0]} samples")
    logger.info(
        f"Class distribution after resampling:\n{pd.Series(y_train_resampled).value_counts()}"
    )

    # Get class weights if using class_weight method
    class_weights = imbalanced_handler.get_class_weights()
    if class_weights:
        logger.info(f"Class weights: {class_weights}")

    # Calculate scale_pos_weight for XGBoost if needed
    if config["model"]["scale_pos_weight"] is None and class_weights is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        if pos_count > 0:
            scale_pos_weight = neg_count / pos_count
            logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
        else:
            scale_pos_weight = 1.0
    else:
        scale_pos_weight = config["model"]["scale_pos_weight"]

    # Update model config with scale_pos_weight
    model_params = {
        "max_depth": config["model"]["max_depth"],
        "learning_rate": config["model"]["learning_rate"],
        "n_estimators": config["model"]["n_estimators"],
        "subsample": config["model"]["subsample"],
        "colsample_bytree": config["model"]["colsample_bytree"],
        "min_child_weight": config["model"]["min_child_weight"],
    }

    if scale_pos_weight and config["model"]["algorithm"] == "xgboost":
        model_params["scale_pos_weight"] = scale_pos_weight

    # Create and train model
    logger.info(f"Training {config['model']['algorithm']} model...")
    pipeline = GradientBoostingPipeline(
        algorithm=config["model"]["algorithm"],
        model_params=model_params,
        random_state=config["data"]["random_seed"],
    )

    # Train
    pipeline.fit(
        X_train_resampled,
        y_train_resampled,
        X_val=X_val if len(X_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
        class_weights=class_weights,
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
    )

    logger.info("Model training complete!")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    logger.info("\n" + "=" * 60)
    logger.info("Test Set Performance:")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")

    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Save model
    if config["training"]["save_model"]:
        model_path = config["training"]["model_save_path"]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        pipeline.save(model_path)
        logger.info(f"Model saved to {model_path}")

    # Feature importance
    logger.info("Extracting feature importance...")
    feature_importance = pipeline.get_feature_importance()

    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    sns.barplot(data=top_features, x="importance", y="feature")
    plt.title("Top 20 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    importance_path = config["training"]["feature_importance_path"]
    Path(importance_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(importance_path, dpi=300, bbox_inches="tight")
    logger.info(f"Feature importance plot saved to {importance_path}")
    plt.close()

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = Path(config["evaluation"]["plots_path"]) / "test_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Cross-validation
    logger.info("Performing cross-validation...")
    cv_results = pipeline.cross_validate(
        X_train_resampled,
        y_train_resampled,
        cv=config["training"]["n_folds"],
        scoring="roc_auc",
        class_weights=class_weights,
    )

    logger.info(f"CV ROC-AUC: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
