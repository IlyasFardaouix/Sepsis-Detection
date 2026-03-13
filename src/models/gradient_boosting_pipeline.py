"""Gradient Boosting ML Pipeline for Sepsis Detection."""

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


class GradientBoostingPipeline:
    """
    Gradient Boosting pipeline for sepsis detection.

    Supports multiple gradient boosting algorithms:
    - XGBoost
    - LightGBM
    - CatBoost
    """

    def __init__(
        self,
        algorithm: str = "xgboost",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """
        Initialize gradient boosting pipeline.

        Args:
            algorithm: Algorithm to use ('xgboost', 'lightgbm', 'catboost')
            model_params: Model hyperparameters
            random_state: Random seed
        """
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

        # Default parameters
        default_params = {
            "xgboost": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 6,
                "learning_rate": 0.01,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "random_state": random_state,
                "n_jobs": -1,
                "tree_method": "hist",
            },
            "lightgbm": {
                "objective": "binary",
                "metric": "auc",
                "max_depth": 6,
                "learning_rate": 0.01,
                "n_estimators": 1000,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            },
            "catboost": {
                "objective": "Logloss",
                "eval_metric": "AUC",
                "depth": 6,
                "learning_rate": 0.01,
                "iterations": 1000,
                "subsample": 0.8,
                "colsample_bylevel": 0.8,
                "min_child_samples": 20,
                "random_state": random_state,
                "verbose": False,
                "thread_count": -1,
            },
        }

        # Merge with provided parameters
        if model_params:
            default_params[self.algorithm].update(model_params)

        self.model_params = default_params[self.algorithm]

    def _create_model(self, class_weights: Optional[Dict] = None):
        """Create the gradient boosting model."""
        params = self.model_params.copy()

        # Handle class weights
        if class_weights is not None:
            if self.algorithm == "xgboost":
                # XGBoost uses scale_pos_weight
                if 0 in class_weights and 1 in class_weights:
                    params["scale_pos_weight"] = class_weights[0] / class_weights[1]
            elif self.algorithm == "lightgbm":
                params["class_weight"] = class_weights
            elif self.algorithm == "catboost":
                params["class_weights"] = list(class_weights.values())

        # Create model
        if self.algorithm == "xgboost":
            self.model = xgb.XGBClassifier(**params)
        elif self.algorithm == "lightgbm":
            self.model = lgb.LGBMClassifier(**params)
        elif self.algorithm == "catboost":
            self.model = CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        class_weights: Optional[Dict] = None,
        early_stopping_rounds: int = 50,
    ):
        """
        Train the model.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            class_weights: Class weights for imbalanced data
            early_stopping_rounds: Early stopping rounds
        """
        # Store feature names
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Create model
        self._create_model(class_weights)

        # Prepare validation set
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(
                X_val_scaled, columns=X_val.columns, index=X_val.index
            )

            if self.algorithm == "xgboost":
                eval_set = [(X_val_scaled, y_val)]
            elif self.algorithm == "lightgbm":
                eval_set = [(X_val_scaled, y_val)]
            elif self.algorithm == "catboost":
                eval_set = [(X_val_scaled, y_val)]

        # Train model
        if eval_set is not None and early_stopping_rounds > 0:
            if self.algorithm == "xgboost":
                self.model.fit(
                    X_scaled,
                    y,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            elif self.algorithm == "lightgbm":
                self.model.fit(
                    X_scaled,
                    y,
                    eval_set=eval_set,
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds),
                        lgb.log_evaluation(0),
                    ],
                )
            elif self.algorithm == "catboost":
                self.model.fit(
                    X_scaled,
                    y,
                    eval_set=eval_set,
                    early_stopping_rounds=early_stopping_rounds,
                )
        else:
            self.model.fit(X_scaled, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        if self.algorithm == "xgboost":
            importance = self.model.feature_importances_
        elif self.algorithm == "lightgbm":
            importance = self.model.feature_importances_
        elif self.algorithm == "catboost":
            importance = self.model.feature_importances_
        else:
            importance = np.zeros(len(self.feature_names))

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "roc_auc",
        class_weights: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            scoring: Scoring metric
            class_weights: Class weights

        Returns:
            Dictionary with CV results
        """
        # Create model
        self._create_model(class_weights)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model,
            X_scaled,
            y,
            cv=StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            ),
            scoring=scoring,
            n_jobs=-1,
        )

        return {
            "mean": cv_scores.mean(),
            "std": cv_scores.std(),
            "scores": cv_scores.tolist(),
        }

    def save(self, filepath: str):
        """
        Save the model.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "algorithm": self.algorithm,
            "model_params": self.model_params,
        }

        joblib.dump(model_data, filepath)

    def load(self, filepath: str):
        """
        Load the model.

        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.algorithm = model_data["algorithm"]
        self.model_params = model_data["model_params"]
