"""Handling imbalanced datasets for sepsis detection."""
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, SMOTEENN
from imblearn.combine import SMOTETomek
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Tuple


class ImbalancedDataHandler:
    """
    Handle imbalanced datasets using various techniques.
    
    Supports:
    - SMOTE (Synthetic Minority Oversampling Technique)
    - ADASYN (Adaptive Synthetic Sampling)
    - SMOTEENN (SMOTE + Edited Nearest Neighbours)
    - Class weights
    """
    
    def __init__(self, 
                 method: str = "smote",
                 k_neighbors: int = 5,
                 sampling_strategy: str = "auto",
                 random_state: int = 42):
        """
        Initialize imbalanced data handler.
        
        Args:
            method: Method to use ('smote', 'adasyn', 'smoteenn', 'class_weight', 'none')
            k_neighbors: Number of neighbors for SMOTE/ADASYN
            sampling_strategy: Sampling strategy (auto, float, dict)
            random_state: Random seed
        """
        self.method = method.lower()
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.sampler = None
        self.class_weights = None
    
    def fit_resample(self, X, y) -> Tuple:
        """
        Fit and resample the data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Resampled X and y
        """
        if self.method == "none":
            return X, y
        
        elif self.method == "class_weight":
            # Calculate class weights
            classes = np.unique(y)
            weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=y
            )
            self.class_weights = dict(zip(classes, weights))
            return X, y
        
        elif self.method == "smote":
            self.sampler = SMOTE(
                k_neighbors=self.k_neighbors,
                sampling_strategy=self._parse_sampling_strategy(y),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.method == "adasyn":
            self.sampler = ADASYN(
                n_neighbors=self.k_neighbors,
                sampling_strategy=self._parse_sampling_strategy(y),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.method == "smoteenn":
            self.sampler = SMOTEENN(
                sampling_strategy=self._parse_sampling_strategy(y),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif self.method == "smotetomek":
            self.sampler = SMOTETomek(
                sampling_strategy=self._parse_sampling_strategy(y),
                random_state=self.random_state,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.sampler is not None:
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        
        return X, y
    
    def _parse_sampling_strategy(self, y) -> str:
        """
        Parse sampling strategy.
        
        Args:
            y: Target vector
            
        Returns:
            Parsed sampling strategy
        """
        if self.sampling_strategy == "auto":
            return "auto"
        elif isinstance(self.sampling_strategy, (int, float)):
            return self.sampling_strategy
        elif isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        else:
            return "auto"
    
    def get_class_weights(self) -> Optional[dict]:
        """
        Get computed class weights.
        
        Returns:
            Dictionary of class weights or None
        """
        return self.class_weights

