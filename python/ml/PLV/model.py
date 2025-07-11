import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np

class ZeroInflatedLSTM(nn.Module):
    def __init__(self, input_size, n_lags=1, lstm_units=32, dense_units=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.shared_dense = nn.Linear(lstm_units, dense_units)
        # Classifier head
        self.classifier = nn.Linear(dense_units, 1)
        # Regressor head
        self.regressor_dense = nn.Linear(dense_units, dense_units)
        self.regressor = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.feature_scaler = None
        self.target_reg_scaler = None
        self._input_size = input_size
        self.n_lags = n_lags

    def scale_X(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        n_samples, n_lags, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.feature_scaler.transform(X_reshaped).reshape(n_samples, n_lags, n_features)
        return torch.tensor(X_scaled, dtype=torch.float32)

    def scale_y_reg(self, y_reg):
        if isinstance(y_reg, torch.Tensor):
            y_reg = y_reg.numpy()
        y_reg_scaled = self.target_reg_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        return torch.tensor(y_reg_scaled, dtype=torch.float32)

    def inverse_scale_y_reg(self, y_reg_scaled):
        if isinstance(y_reg_scaled, torch.Tensor):
            y_reg_scaled = y_reg_scaled.cpu().numpy()
        return self.target_reg_scaler.inverse_transform(y_reg_scaled.reshape(-1, 1)).flatten()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.relu(self.shared_dense(x))
        cls_out = self.sigmoid(self.classifier(x))
        reg_x = self.relu(self.regressor_dense(x))
        reg_out = self.regressor(reg_x)
        return cls_out, reg_out

    def fit(self, train_dataset, epochs=20, batch_size=64, lr=0.001, verbose=1):
        """
        Train the model using a PyTorch Dataset and DataLoader, with incremental scaling.
        Args:
            train_dataset: PyTorch Dataset (X, y_cls, y_reg)
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress if True
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        # Initialize scalers for incremental fitting
        self.feature_scaler = StandardScaler()
        self.target_reg_scaler = StandardScaler()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            for X, y_cls, y_reg in loader:
                # Incrementally fit scalers on this batch
                X_np = X.numpy()
                n_samples, n_lags, n_features = X_np.shape
                X_reshaped = X_np.reshape(-1, n_features)
                self.feature_scaler.partial_fit(X_reshaped)
                y_reg_np = y_reg.numpy().reshape(-1, 1)
                self.target_reg_scaler.partial_fit(y_reg_np)
                # Transform batch using current scalers
                X_scaled = self.feature_scaler.transform(X_reshaped).reshape(n_samples, n_lags, n_features)
                y_reg_scaled = self.target_reg_scaler.transform(y_reg_np).flatten()
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
                y_cls = y_cls.to(device).unsqueeze(1)
                y_reg_tensor = torch.tensor(y_reg_scaled, dtype=torch.float32, device=device).unsqueeze(1)
                optimizer.zero_grad()
                cls_pred, reg_pred = self(X_tensor)
                loss = custom_zi_loss(cls_pred, reg_pred, y_cls, y_reg_tensor)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)
            if verbose and (epoch % 5 == 0 or epoch == epochs-1):
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_dataset):.8f}")
        return self

    def evaluate(self, val_dataset):
        """
        Evaluate the model on a validation dataset using custom_zi_loss (BCE + masked MSE).
        Args:
            val_dataset: PyTorch Dataset (X, y_cls, y_reg)
        Returns:
            loss: Average custom_zi_loss over the validation set
        """
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
            for X, y_cls, y_reg in loader:
                X = self.scale_X(X).to(device)
                y_cls = y_cls.to(device).unsqueeze(1)
                y_reg = self.scale_y_reg(y_reg).to(device).unsqueeze(1)
                cls_pred, reg_pred = self(X)
                loss = custom_zi_loss(cls_pred, reg_pred, y_cls, y_reg)
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
        avg_loss = total_loss / n_samples if n_samples > 0 else float('inf')
        return avg_loss

    def predict(self, X):
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            n_samples, n_lags, n_features = X.shape
            X_scaled = self.feature_scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, n_lags, n_features)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            cls_pred, reg_pred = self(X_tensor)
            cls_pred = (cls_pred.cpu().numpy().flatten() > 0.5).astype(int)
            reg_pred = reg_pred.cpu().numpy().flatten()
            reg_pred = self.target_reg_scaler.inverse_transform(reg_pred.reshape(-1, 1)).flatten()
        return reg_pred, cls_pred


def custom_zi_loss(cls_pred, reg_pred, y_cls, y_reg):
    bce = nn.BCELoss()(cls_pred, y_cls)
    mask = (y_cls == 0).float()
    if mask.sum() > 0:
        mse = ((reg_pred.squeeze() - y_reg.squeeze()) ** 2 * mask).sum() / (mask.sum() + 1e-6)
    else:
        mse = torch.tensor(0.0, device=reg_pred.device)
    return bce + mse