"""Time-series feature engineering for sepsis detection."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler


class TimeSeriesFeatureEngineer:
    """
    Extract time-series features from clinical data.

    This class creates features from time-series data including:
    - Rolling statistics (mean, std, min, max, etc.)
    - Trend features
    - Change features
    - Statistical features
    - Rate of change features
    """

    def __init__(
        self,
        time_window_hours: int = 6,
        lookback_hours: int = 24,
        include_static: bool = True,
        include_temporal: bool = True,
    ):
        """
        Initialize feature engineer.

        Args:
            time_window_hours: Window size for rolling features
            lookback_hours: How far back to look for features
            include_static: Whether to include static features
            include_temporal: Whether to include temporal features
        """
        self.time_window_hours = time_window_hours
        self.lookback_hours = lookback_hours
        self.include_static = include_static
        self.include_temporal = include_temporal

    def extract_rolling_features(
        self, df: pd.DataFrame, value_col: str, time_col: str = "time"
    ) -> pd.DataFrame:
        """
        Extract rolling window features.

        Args:
            df: DataFrame with time series data
            value_col: Column name for values
            time_col: Column name for time

        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame()

        # Sort by time
        df_sorted = df.sort_values(time_col).copy()

        # Rolling statistics
        window_size = self.time_window_hours

        features[f"{value_col}_rolling_mean"] = (
            df_sorted[value_col].rolling(window=window_size, min_periods=1).mean()
        )

        features[f"{value_col}_rolling_std"] = (
            df_sorted[value_col]
            .rolling(window=window_size, min_periods=1)
            .std()
            .fillna(0)
        )

        features[f"{value_col}_rolling_min"] = (
            df_sorted[value_col].rolling(window=window_size, min_periods=1).min()
        )

        features[f"{value_col}_rolling_max"] = (
            df_sorted[value_col].rolling(window=window_size, min_periods=1).max()
        )

        features[f"{value_col}_rolling_median"] = (
            df_sorted[value_col].rolling(window=window_size, min_periods=1).median()
        )

        # Rolling percentiles
        features[f"{value_col}_rolling_q25"] = (
            df_sorted[value_col]
            .rolling(window=window_size, min_periods=1)
            .quantile(0.25)
        )

        features[f"{value_col}_rolling_q75"] = (
            df_sorted[value_col]
            .rolling(window=window_size, min_periods=1)
            .quantile(0.75)
        )

        # Coefficient of variation
        features[f"{value_col}_rolling_cv"] = features[f"{value_col}_rolling_std"] / (
            features[f"{value_col}_rolling_mean"] + 1e-6
        )

        return features

    def extract_trend_features(
        self, df: pd.DataFrame, value_col: str, time_col: str = "time"
    ) -> pd.DataFrame:
        """
        Extract trend features.

        Args:
            df: DataFrame with time series data
            value_col: Column name for values
            time_col: Column name for time

        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame()
        df_sorted = df.sort_values(time_col).copy()

        # Linear trend (slope)
        window = min(self.lookback_hours, len(df_sorted))
        if window > 1:
            x = np.arange(window)
            y = df_sorted[value_col].tail(window).values

            if len(y) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                features[f"{value_col}_trend_slope"] = [slope] * len(df_sorted)
                features[f"{value_col}_trend_r2"] = [r_value**2] * len(df_sorted)
            else:
                features[f"{value_col}_trend_slope"] = [0] * len(df_sorted)
                features[f"{value_col}_trend_r2"] = [0] * len(df_sorted)
        else:
            features[f"{value_col}_trend_slope"] = [0] * len(df_sorted)
            features[f"{value_col}_trend_r2"] = [0] * len(df_sorted)

        return features

    def extract_change_features(
        self, df: pd.DataFrame, value_col: str, time_col: str = "time"
    ) -> pd.DataFrame:
        """
        Extract change features (deltas, rates).

        Args:
            df: DataFrame with time series data
            value_col: Column name for values
            time_col: Column name for time

        Returns:
            DataFrame with change features
        """
        features = pd.DataFrame()
        df_sorted = df.sort_values(time_col).copy()

        # Absolute change
        features[f"{value_col}_delta_1h"] = df_sorted[value_col].diff(1).fillna(0)
        features[f"{value_col}_delta_3h"] = df_sorted[value_col].diff(3).fillna(0)
        features[f"{value_col}_delta_6h"] = df_sorted[value_col].diff(6).fillna(0)

        # Percentage change
        features[f"{value_col}_pct_change_1h"] = (
            df_sorted[value_col].pct_change(1).fillna(0)
        )
        features[f"{value_col}_pct_change_3h"] = (
            df_sorted[value_col].pct_change(3).fillna(0)
        )

        # Rate of change
        time_diff = df_sorted[time_col].diff(1)
        value_diff = df_sorted[value_col].diff(1)
        features[f"{value_col}_rate_of_change"] = (
            value_diff / (time_diff + 1e-6)
        ).fillna(0)

        return features

    def extract_statistical_features(
        self, df: pd.DataFrame, value_col: str, time_col: str = "time"
    ) -> pd.DataFrame:
        """
        Extract statistical features over lookback window.

        Args:
            df: DataFrame with time series data
            value_col: Column name for values
            time_col: Column name for time

        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame()
        df_sorted = df.sort_values(time_col).copy()

        window = min(self.lookback_hours, len(df_sorted))

        # Recent statistics
        recent_values = df_sorted[value_col].tail(window)

        features[f"{value_col}_recent_mean"] = [recent_values.mean()] * len(df_sorted)
        features[f"{value_col}_recent_std"] = [recent_values.std()] * len(df_sorted)
        features[f"{value_col}_recent_min"] = [recent_values.min()] * len(df_sorted)
        features[f"{value_col}_recent_max"] = [recent_values.max()] * len(df_sorted)
        features[f"{value_col}_recent_range"] = [
            recent_values.max() - recent_values.min()
        ] * len(df_sorted)

        # Skewness and kurtosis
        if len(recent_values) > 2:
            features[f"{value_col}_recent_skew"] = [stats.skew(recent_values)] * len(
                df_sorted
            )
            features[f"{value_col}_recent_kurtosis"] = [
                stats.kurtosis(recent_values)
            ] * len(df_sorted)
        else:
            features[f"{value_col}_recent_skew"] = [0] * len(df_sorted)
            features[f"{value_col}_recent_kurtosis"] = [0] * len(df_sorted)

        return features

    def create_features(
        self,
        df: pd.DataFrame,
        patient_id_col: str = "patient_id",
        time_col: str = "time",
        target_col: str = "sepsis_label",
        numeric_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create all features for the dataset.

        Args:
            df: Input DataFrame
            patient_id_col: Column name for patient ID
            time_col: Column name for time
            target_col: Column name for target variable
            numeric_cols: List of numeric columns to create features for

        Returns:
            DataFrame with engineered features
        """
        if numeric_cols is None:
            # Auto-detect numeric columns (excluding ID, time, and target)
            exclude_cols = [patient_id_col, time_col, target_col]
            numeric_cols = [
                col
                for col in df.columns
                if col not in exclude_cols and df[col].dtype in ["int64", "float64"]
            ]

        all_features = []

        # Group by patient
        for patient_id, patient_df in df.groupby(patient_id_col):
            patient_features = patient_df[[patient_id_col, time_col]].copy()

            if target_col in patient_df.columns:
                patient_features[target_col] = patient_df[target_col].values

            # Extract features for each numeric column
            for col in numeric_cols:
                if col in patient_df.columns:
                    # Rolling features
                    rolling_feat = self.extract_rolling_features(
                        patient_df, col, time_col
                    )
                    patient_features = pd.concat(
                        [patient_features, rolling_feat], axis=1
                    )

                    # Trend features
                    trend_feat = self.extract_trend_features(patient_df, col, time_col)
                    patient_features = pd.concat([patient_features, trend_feat], axis=1)

                    # Change features
                    change_feat = self.extract_change_features(
                        patient_df, col, time_col
                    )
                    patient_features = pd.concat(
                        [patient_features, change_feat], axis=1
                    )

                    # Statistical features
                    stat_feat = self.extract_statistical_features(
                        patient_df, col, time_col
                    )
                    patient_features = pd.concat([patient_features, stat_feat], axis=1)

            # Add static features if enabled
            if self.include_static:
                static_cols = [
                    c
                    for c in patient_df.columns
                    if c not in [patient_id_col, time_col, target_col] + numeric_cols
                ]
                for col in static_cols:
                    if col in patient_df.columns:
                        patient_features[col] = patient_df[col].values

            all_features.append(patient_features)

        # Combine all patients
        result_df = pd.concat(all_features, ignore_index=True)

        # Fill NaN values
        result_df = result_df.fillna(0)

        return result_df
