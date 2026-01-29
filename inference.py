"""Inference script for Sepsis Detection Model."""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config, setup_logger
from src.features import TimeSeriesFeatureEngineer
from src.models import GradientBoostingPipeline
import joblib


def predict_sepsis(patient_data: pd.DataFrame,
                   model_path: str = "models/sepsis_model.pkl",
                   feature_selector_path: str = "models/feature_selector.pkl",
                   threshold: float = 0.5) -> pd.DataFrame:
    """
    Predict sepsis for patient data.
    
    Args:
        patient_data: DataFrame with patient time-series data
        model_path: Path to trained model
        feature_selector_path: Path to feature selector
        threshold: Probability threshold for positive prediction
        
    Returns:
        DataFrame with predictions
    """
    # Load configuration
    config = load_config()
    
    # Load model
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    pipeline = GradientBoostingPipeline()
    pipeline.load(model_path)
    
    # Load feature selector if it exists
    feature_selector = None
    if Path(feature_selector_path).exists():
        feature_selector = joblib.load(feature_selector_path)
    
    # Feature engineering
    feature_engineer = TimeSeriesFeatureEngineer(
        time_window_hours=config['features']['time_window_hours'],
        lookback_hours=config['features']['lookback_hours'],
        include_static=config['features']['include_static_features'],
        include_temporal=config['features']['include_temporal_features']
    )
    
    df_features = feature_engineer.create_features(patient_data)
    
    # Prepare features
    exclude_cols = ['patient_id', 'time', 'sepsis_label']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    X = df_features[feature_cols].copy()
    X = X.fillna(X.median())
    
    # Apply feature selection if available
    if feature_selector is not None:
        X = feature_selector.transform(X)
    
    # Make predictions
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]
    
    # Create results DataFrame
    results = df_features[['patient_id', 'time']].copy()
    results['sepsis_probability'] = probabilities
    results['sepsis_prediction'] = (probabilities >= threshold).astype(int)
    results['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return results


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sepsis Detection Inference')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to patient data CSV file')
    parser.add_argument('--model', type=str, default='models/sepsis_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='results/predictions.csv',
                       help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for positive prediction')
    
    args = parser.parse_args()
    
    # Setup logger
    config = load_config()
    logger = setup_logger(
        log_file=config['logging']['log_file'],
        log_level=config['logging']['log_level']
    )
    
    logger.info("=" * 60)
    logger.info("Sepsis Detection Inference")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    try:
        patient_data = pd.read_csv(args.data)
        logger.info(f"Loaded {len(patient_data)} records")
    except FileNotFoundError:
        logger.error(f"Data file not found: {args.data}")
        return
    
    # Make predictions
    logger.info("Making predictions...")
    results = predict_sepsis(
        patient_data,
        model_path=args.model,
        threshold=args.threshold
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Prediction Summary:")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {len(results)}")
    logger.info(f"High risk patients: {(results['risk_level'] == 'High').sum()}")
    logger.info(f"Medium risk patients: {(results['risk_level'] == 'Medium').sum()}")
    logger.info(f"Low risk patients: {(results['risk_level'] == 'Low').sum()}")
    logger.info(f"Predicted sepsis cases: {results['sepsis_prediction'].sum()}")
    logger.info(f"\nAverage sepsis probability: {results['sepsis_probability'].mean():.4f}")
    logger.info(f"Max sepsis probability: {results['sepsis_probability'].max():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Inference Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

