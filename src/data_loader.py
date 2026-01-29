"""Data loading and preprocessing utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split


def load_data(filepath: str, 
              patient_id_col: str = 'patient_id',
              time_col: str = 'time',
              target_col: str = 'sepsis_label') -> pd.DataFrame:
    """
    Load sepsis detection data.
    
    Args:
        filepath: Path to data file
        patient_id_col: Column name for patient ID
        time_col: Column name for time
        target_col: Column name for target variable
        
    Returns:
        Loaded DataFrame
    """
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Construct full path
    if not Path(filepath).is_absolute():
        filepath = project_root / filepath
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert time column to datetime if it's not already
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    return df


def prepare_data(df: pd.DataFrame,
                 patient_id_col: str = 'patient_id',
                 time_col: str = 'time',
                 target_col: str = 'sepsis_label',
                 exclude_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling.
    
    Args:
        df: Input DataFrame
        patient_id_col: Column name for patient ID
        time_col: Column name for time
        target_col: Column name for target variable
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if exclude_cols is None:
        exclude_cols = [patient_id_col, time_col, target_col]
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y


def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2,
               validation_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        validation_size: Proportion of validation set (from remaining after test)
        random_state: Random seed
        stratify: Whether to stratify by target
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    stratify_param = y if stratify else None
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Second split: train vs val
    if validation_size > 0:
        stratify_param = y_temp if stratify else None
        val_size_adjusted = validation_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param
        )
    else:
        X_train, X_val = X_temp, pd.DataFrame()
        y_train, y_val = y_temp, pd.Series()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

