import pickle

def save_scalers(feature_scaler, target_reg_scaler, path):
    """
    Save feature and target scalers to a file using pickle.
    Args:
        feature_scaler: Fitted feature scaler (e.g., StandardScaler)
        target_reg_scaler: Fitted target scaler (e.g., StandardScaler)
        path: Path to save the scalers (should end with .pkl)
    """
    with open(path, 'wb') as f:
        pickle.dump({'feature_scaler': feature_scaler, 'target_reg_scaler': target_reg_scaler}, f)

def load_scalers(path):
    """
    Load feature and target scalers from a pickle file.
    Args:
        path: Path to the saved scalers (.pkl)
    Returns:
        (feature_scaler, target_reg_scaler)
    """
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers['feature_scaler'], scalers['target_reg_scaler']
