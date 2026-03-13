"""Feature selection utilities."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional


class FeatureSelector:
    """
    Feature selection for sepsis detection.

    Supports multiple feature selection methods:
    - Univariate selection (SelectKBest)
    - Mutual information
    - Recursive Feature Elimination (RFE)
    - Model-based selection
    """

    def __init__(
        self,
        method: str = "mutual_info",
        max_features: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize feature selector.

        Args:
            method: Selection method ('mutual_info', 'f_classif', 'rfe', 'model_based')
            max_features: Maximum number of features to select
            random_state: Random seed
        """
        self.method = method
        self.max_features = max_features
        self.random_state = random_state
        self.selector = None
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit feature selector.

        Args:
            X: Feature matrix
            y: Target vector
        """
        if self.method == "mutual_info":
            self.selector = SelectKBest(
                score_func=mutual_info_classif, k=min(self.max_features, X.shape[1])
            )
        elif self.method == "f_classif":
            self.selector = SelectKBest(
                score_func=f_classif, k=min(self.max_features, X.shape[1])
            )
        elif self.method == "rfe":
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            self.selector = RFE(
                estimator=estimator,
                n_features_to_select=min(self.max_features, X.shape[1]),
            )
        elif self.method == "model_based":
            estimator = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            estimator.fit(X, y)
            self.selector = SelectFromModel(
                estimator=estimator, max_features=self.max_features, prefit=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.selector.fit(X, y)

        # Get selected feature names
        if hasattr(self.selector, "get_support"):
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
        else:
            self.selected_features = X.columns.tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted selector.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selector is None:
            raise ValueError("Selector must be fitted before transform")

        X_transformed = self.selector.transform(X)

        # Return as DataFrame with selected feature names
        return pd.DataFrame(
            X_transformed, columns=self.selected_features, index=X.index
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform features.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.method in ["mutual_info", "f_classif"]:
            scores = self.selector.scores_
        elif self.method == "rfe":
            scores = self.selector.estimator_.feature_importances_
        elif self.method == "model_based":
            scores = self.selector.estimator_.feature_importances_
        else:
            scores = np.ones(X.shape[1])

        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": scores}
        ).sort_values("importance", ascending=False)

        return importance_df
