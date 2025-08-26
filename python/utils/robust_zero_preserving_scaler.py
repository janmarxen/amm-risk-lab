"""
Robust Zero-Preserving Scaler using quantile-based scaling instead of mean/std.
This is more robust to distribution shifts and outliers.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RobustZeroPreservingScaler(BaseEstimator, TransformerMixin):
    """
    Robust scaler that preserves zeros and uses quantile-based scaling.
    
    For zero-inflated data, this:
    1. Preserves zeros as zeros
    2. Uses median and IQR for scaling (more robust than mean/std)
    3. Handles distribution shifts better than standard scaling
    """
    
    def __init__(self, quantile_range=(25.0, 75.0)):
        """
        Parameters:
        -----------
        quantile_range : tuple
            The quantile range for robust scaling (default: IQR)
        """
        self.quantile_range = quantile_range
        self.median_ = None
        self.scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self._samples = None  # For incremental fitting
    
    def fit(self, X, y=None):
        """Fit the scaler to the data."""
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_array = X.values
        elif isinstance(X, pd.Series):
            X_array = X.values.reshape(-1, 1)
        else:
            X_array = np.asarray(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
        
        self.n_features_in_ = X_array.shape[1]
        
        # Calculate robust statistics only on non-zero values
        self.median_ = np.zeros(self.n_features_in_)
        self.scale_ = np.ones(self.n_features_in_)
        
        for i in range(self.n_features_in_):
            col_data = X_array[:, i]
            nonzero_mask = col_data != 0
            
            if np.any(nonzero_mask):
                nonzero_data = col_data[nonzero_mask]
                self.median_[i] = np.median(nonzero_data)
                
                q_min, q_max = np.percentile(nonzero_data, self.quantile_range)
                scale = q_max - q_min
                if scale > 0:
                    self.scale_[i] = scale
        
        return self
    
    def partial_fit(self, X, y=None):
        """
        Incremental fit for online learning.
        For robust scaling, we'll collect samples and recompute statistics.
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        elif isinstance(X, pd.Series):
            X_array = X.values.reshape(-1, 1)
        else:
            X_array = np.asarray(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
        
        # Initialize if this is the first call
        if self.median_ is None:
            self.n_features_in_ = X_array.shape[1]
            self.median_ = np.zeros(self.n_features_in_)
            self.scale_ = np.ones(self.n_features_in_)
            if isinstance(X, pd.DataFrame):
                self.feature_names_in_ = X.columns.tolist()
            self._samples = [[] for _ in range(self.n_features_in_)]
        
        # Collect non-zero samples for each feature
        for i in range(self.n_features_in_):
            col_data = X_array[:, i]
            nonzero_data = col_data[col_data != 0]
            if len(nonzero_data) > 0:
                self._samples[i].extend(nonzero_data.tolist())
        
        # Recompute robust statistics with all collected samples
        for i in range(self.n_features_in_):
            if len(self._samples[i]) > 0:
                samples_array = np.array(self._samples[i])
                self.median_[i] = np.median(samples_array)
                
                q_min, q_max = np.percentile(samples_array, self.quantile_range)
                scale = q_max - q_min
                if scale > 0:
                    self.scale_[i] = scale
        
        return self
    
    def transform(self, X):
        """Transform the data using fitted parameters."""
        # Convert to numpy array if needed, preserving DataFrame structure
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            return_df = True
            df_index = X.index
            df_columns = X.columns
        elif isinstance(X, pd.Series):
            X_array = X.values.reshape(-1, 1)
            return_df = False
        else:
            X_array = np.asarray(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            return_df = False
        
        # Apply robust scaling while preserving zeros
        X_scaled = np.zeros_like(X_array, dtype=float)
        
        for i in range(self.n_features_in_):
            col_data = X_array[:, i]
            nonzero_mask = col_data != 0
            
            # Keep zeros as zeros
            X_scaled[~nonzero_mask, i] = 0.0
            
            # Scale non-zero values
            if np.any(nonzero_mask):
                X_scaled[nonzero_mask, i] = (col_data[nonzero_mask] - self.median_[i]) / self.scale_[i]
        
        # Return in same format as input
        if return_df:
            return pd.DataFrame(X_scaled, index=df_index, columns=df_columns)
        elif isinstance(X, pd.Series):
            return pd.Series(X_scaled.flatten(), index=X.index, name=X.name)
        else:
            return X_scaled.squeeze() if X_scaled.shape[1] == 1 and X.ndim == 1 else X_scaled
    
    def inverse_transform(self, X):
        """Inverse transform the scaled data."""
        # Convert to numpy array if needed, preserving DataFrame structure
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            return_df = True
            df_index = X.index
            df_columns = X.columns
        elif isinstance(X, pd.Series):
            X_array = X.values.reshape(-1, 1)
            return_df = False
        else:
            X_array = np.asarray(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            return_df = False
        
        # Apply inverse robust scaling while preserving zeros
        X_unscaled = np.zeros_like(X_array, dtype=float)
        
        for i in range(self.n_features_in_):
            col_data = X_array[:, i]
            zero_mask = col_data == 0
            
            # Keep zeros as zeros
            X_unscaled[zero_mask, i] = 0.0
            
            # Inverse scale non-zero values
            if np.any(~zero_mask):
                X_unscaled[~zero_mask, i] = col_data[~zero_mask] * self.scale_[i] + self.median_[i]
        
        # Return in same format as input
        if return_df:
            return pd.DataFrame(X_unscaled, index=df_index, columns=df_columns)
        elif isinstance(X, pd.Series):
            return pd.Series(X_unscaled.flatten(), index=X.index, name=X.name)
        else:
            return X_unscaled.squeeze() if X_unscaled.shape[1] == 1 and X.ndim == 1 else X_unscaled
    
    def fit_transform(self, X, y=None):
        """Fit the scaler and transform the data."""
        return self.fit(X, y).transform(X)


# Test function
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Create test data with zeros and distribution shift
    train_data = pd.DataFrame({
        'volume': [0, 0, 2.5, 3.0, 2.8, 0, 2.9, 2.7, 0, 2.6]
    })
    
    test_data = pd.DataFrame({
        'volume': [0, 0, 0.4, 0.5, 0.3, 0, 0.6, 0.4, 0, 0.5]
    })
    
    print("=== ROBUST ZERO-PRESERVING SCALER TEST ===")
    print(f"Train data mean (non-zero): {train_data[train_data['volume'] != 0]['volume'].mean():.3f}")
    print(f"Test data mean (non-zero): {test_data[test_data['volume'] != 0]['volume'].mean():.3f}")
    
    # Fit on training data
    scaler = RobustZeroPreservingScaler()
    scaler.fit(train_data)
    
    print(f"Scaler median: {scaler.median_[0]:.3f}")
    print(f"Scaler scale (IQR): {scaler.scale_[0]:.3f}")
    
    # Transform test data
    test_scaled = scaler.transform(test_data)
    print(f"Test data scaled mean: {test_scaled['volume'].mean():.3f}")
    print(f"Test data scaled std: {test_scaled['volume'].std():.3f}")
    
    # Check zero preservation
    original_zeros = (test_data['volume'] == 0).sum()
    scaled_zeros = (test_scaled['volume'] == 0).sum()
    print(f"Zero preservation: {original_zeros} -> {scaled_zeros} ✓" if original_zeros == scaled_zeros else "FAILED")
