"""
Example usage script for Sepsis Detection Pipeline.

This script demonstrates how to use the various components of the pipeline.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_config, setup_logger
from src.data_loader import load_data, prepare_data, split_data
from src.features import TimeSeriesFeatureEngineer, FeatureSelector
from src.models import GradientBoostingPipeline, ImbalancedDataHandler
from sklearn.metrics import roc_auc_score, classification_report


def main():
    """Example usage of the sepsis detection pipeline."""
    
    # 1. Load configuration
    print("=" * 60)
    print("1. Loading Configuration")
    print("=" * 60)
    config = load_config()
    print("Configuration loaded successfully!")
    
    # 2. Load data
    print("\n" + "=" * 60)
    print("2. Loading Data")
    print("=" * 60)
    try:
        df = load_data(config['data']['raw_data_path'])
        print(f"Data shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
    except FileNotFoundError:
        print(f"Data file not found. Please run generate_sample_data.py first.")
        return
    
    # 3. Feature engineering
    print("\n" + "=" * 60)
    print("3. Feature Engineering")
    print("=" * 60)
    feature_engineer = TimeSeriesFeatureEngineer(
        time_window_hours=config['features']['time_window_hours'],
        lookback_hours=config['features']['lookback_hours']
    )
    
    df_features = feature_engineer.create_features(df)
    print(f"Features shape: {df_features.shape}")
    print(f"Number of features: {len(df_features.columns)}")
    
    # 4. Prepare data
    print("\n" + "=" * 60)
    print("4. Preparing Data")
    print("=" * 60)
    X, y = prepare_data(df_features)
    print(f"Features shape: {X.shape}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    print(f"\nClass imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}")
    
    # 5. Split data
    print("\n" + "=" * 60)
    print("5. Splitting Data")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=config['data']['train_test_split'],
        validation_size=config['data']['validation_split'],
        random_state=config['data']['random_seed']
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 6. Feature selection (optional)
    if config['features']['feature_selection']:
        print("\n" + "=" * 60)
        print("6. Feature Selection")
        print("=" * 60)
        feature_selector = FeatureSelector(
            method="mutual_info",
            max_features=config['features']['max_features'],
            random_state=config['data']['random_seed']
        )
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_val = feature_selector.transform(X_val) if len(X_val) > 0 else X_val
        X_test = feature_selector.transform(X_test)
        print(f"Selected {X_train.shape[1]} features")
    
    # 7. Handle imbalanced data
    print("\n" + "=" * 60)
    print("7. Handling Imbalanced Data")
    print("=" * 60)
    imbalanced_handler = ImbalancedDataHandler(
        method=config['imbalanced_learning']['method'],
        k_neighbors=config['imbalanced_learning']['k_neighbors'],
        random_state=config['data']['random_seed']
    )
    
    X_train_resampled, y_train_resampled = imbalanced_handler.fit_resample(X_train, y_train)
    print(f"After resampling: {X_train_resampled.shape[0]} samples")
    print(f"\nClass distribution after resampling:")
    print(pd.Series(y_train_resampled).value_counts())
    
    # 8. Train model
    print("\n" + "=" * 60)
    print("8. Training Model")
    print("=" * 60)
    pipeline = GradientBoostingPipeline(
        algorithm=config['model']['algorithm'],
        model_params={
            'max_depth': config['model']['max_depth'],
            'learning_rate': config['model']['learning_rate'],
            'n_estimators': config['model']['n_estimators'],
            'subsample': config['model']['subsample'],
            'colsample_bytree': config['model']['colsample_bytree']
        },
        random_state=config['data']['random_seed']
    )
    
    class_weights = imbalanced_handler.get_class_weights()
    pipeline.fit(
        X_train_resampled, y_train_resampled,
        X_val=X_val if len(X_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
        class_weights=class_weights,
        early_stopping_rounds=config['model']['early_stopping_rounds']
    )
    print("Model trained successfully!")
    
    # 9. Feature importance
    print("\n" + "=" * 60)
    print("9. Feature Importance")
    print("=" * 60)
    feature_importance = pipeline.get_feature_importance()
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # 10. Evaluate on test set
    print("\n" + "=" * 60)
    print("10. Evaluation on Test Set")
    print("=" * 60)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 60)
    print("Example Usage Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

