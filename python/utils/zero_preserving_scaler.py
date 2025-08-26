"""
zero_preserving_scaler.py

Custom scaler that preserves zero values for zero-inflated data.
This is crucial for zero-inflated modeling where zeros have special meaning.

The ZeroPreservingStandardScaler:
1. Keeps zeros exactly as zeros (no transformation)
2. Only scales non-zero values using standard scaling
3. Maintains the zero-inflated structure necessary for hybrid modeling

This approach is essential for:
- Zero-inflated regression models
- Hybrid classification+regression tasks
- Maintaining interpretability of "true zeros" vs "small values"
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Union


class ZeroPreservingStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standard scaler that preserves zero values for zero-inflated data.
    
    For zero-inflated targets, regular StandardScaler destroys the zero-inflation
    property by transforming zeros to negative values. This scaler:
    1. Identifies zero values and keeps them as zeros
    2. Applies standard scaling only to non-zero values
    3. Maintains the zero-inflated structure
    
    This is essential for hybrid models that use:
    - Binary classification to predict zero/non-zero
    - Regression to predict the magnitude of non-zero values
    """
    
    def __init__(self):
        self.scaler_ = StandardScaler()
        self.is_fitted_ = False
        self.nonzero_mean_ = None
        self.nonzero_std_ = None
    
    def fit(self, X, y=None):
        """
        Fit the scaler on non-zero values only.
        
        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples,)
            y: Ignored, present for API compatibility
            
        Returns:
            self: Fitted scaler
        """
        X = self._validate_input(X)
        
        # Find non-zero values for fitting
        nonzero_mask = X != 0
        nonzero_values = X[nonzero_mask]
        
        if len(nonzero_values) == 0:
            # All zeros - create dummy scaler
            self.nonzero_mean_ = 0.0
            self.nonzero_std_ = 1.0
        else:
            # Fit standard scaler on non-zero values only
            self.scaler_.fit(nonzero_values.reshape(-1, 1))
            self.nonzero_mean_ = self.scaler_.mean_[0]
            self.nonzero_std_ = self.scaler_.scale_[0]
        
        self.is_fitted_ = True
        return self
    
    def partial_fit(self, X, y=None):
        """
        Incrementally fit the scaler on non-zero values.
        
        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples,)
            y: Ignored, present for API compatibility
            
        Returns:
            self: Fitted scaler
        """
        X = self._validate_input(X)
        
        # Find non-zero values for partial fitting
        nonzero_mask = X != 0
        nonzero_values = X[nonzero_mask]
        
        if len(nonzero_values) > 0:
            if not self.is_fitted_:
                self.scaler_ = StandardScaler()
                self.scaler_.partial_fit(nonzero_values.reshape(-1, 1))
                self.is_fitted_ = True
            else:
                self.scaler_.partial_fit(nonzero_values.reshape(-1, 1))
            
            self.nonzero_mean_ = self.scaler_.mean_[0]
            self.nonzero_std_ = self.scaler_.scale_[0]
        
        return self
    
    def fit_transform(self, X, y=None):
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples,)
            y: Ignored, present for API compatibility
            
        Returns:
            X_transformed: Transformed data with zeros preserved, same shape as input
        """
        return self.fit(X, y).transform(X)
    
    def transform(self, X):
        """
        Transform data preserving zeros.
        
        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples,)
            
        Returns:
            X_transformed: Transformed data with zeros preserved, same shape as input
        """
        if not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Store original input info for shape preservation
        original_input = X
        input_is_dataframe = isinstance(X, pd.DataFrame)
        input_is_series = isinstance(X, pd.Series)
        
        X = self._validate_input(X)
        X_transformed = X.copy()
        
        # Only transform non-zero values
        nonzero_mask = X != 0
        if np.any(nonzero_mask):
            nonzero_values = X[nonzero_mask]
            # Apply standard scaling to non-zero values
            scaled_nonzero = self.scaler_.transform(nonzero_values.reshape(-1, 1)).flatten()
            X_transformed[nonzero_mask] = scaled_nonzero
        
        # Return in the same format as input
        if input_is_dataframe:
            return pd.DataFrame(X_transformed.reshape(-1, 1), 
                              columns=original_input.columns, 
                              index=original_input.index)
        elif input_is_series:
            return pd.Series(X_transformed, 
                           name=original_input.name, 
                           index=original_input.index)
        else:
            return X_transformed
    
    def inverse_transform(self, X):
        """
        Inverse transform data preserving zeros.
        
        Args:
            X: array-like of shape (n_samples, n_features) or (n_samples,)
            
        Returns:
            X_original: Inverse transformed data with zeros preserved, same shape as input
        """
        if not self.is_fitted_:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Store original input info for shape preservation
        original_input = X
        input_is_dataframe = isinstance(X, pd.DataFrame)
        input_is_series = isinstance(X, pd.Series)
        
        X = self._validate_input(X)
        X_original = X.copy()
        
        # Only inverse transform non-zero values
        nonzero_mask = X != 0
        if np.any(nonzero_mask):
            nonzero_values = X[nonzero_mask]
            # Apply inverse standard scaling to non-zero values
            original_nonzero = self.scaler_.inverse_transform(nonzero_values.reshape(-1, 1)).flatten()
            X_original[nonzero_mask] = original_nonzero
        
        # Return in the same format as input
        if input_is_dataframe:
            return pd.DataFrame(X_original.reshape(-1, 1), 
                              columns=original_input.columns, 
                              index=original_input.index)
        elif input_is_series:
            return pd.Series(X_original, 
                           name=original_input.name, 
                           index=original_input.index)
        else:
            return X_original
    
    def _validate_input(self, X):
        """Validate and prepare input data."""
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("DataFrame input must have exactly 1 column")
            X = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            X = X.values
        else:
            X = np.asarray(X)
        
        if X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        elif X.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        
        return X
    
    def __repr__(self):
        if self.is_fitted_:
            return f"ZeroPreservingStandardScaler(fitted=True, nonzero_mean={self.nonzero_mean_:.4f}, nonzero_std={self.nonzero_std_:.4f})"
        else:
            return "ZeroPreservingStandardScaler(fitted=False)"
