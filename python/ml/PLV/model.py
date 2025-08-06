import abc
import torch
import torch.nn as nn
import torch.distributed as dist
import time

class ZeroInflatedTSModule(nn.Module, abc.ABC):
    """
    Abstract base class for zero-inflated time series models.
    Provides common scaling, training, evaluation, and prediction utilities for time series models
    with both classification and regression heads. Multi-task architecture for 2 targets.
    """

    def fit_distributed(self, train_loader, epochs=20, lr=0.001, verbose=1, val_loader=None, early_stopping_patience=10, device=None):
        """
        Distributed training loop using DataLoader and DDP. Assumes model is already wrapped in DDP and on correct device.
        Args:
            train_loader: DataLoader for training data (already scaled)
            epochs: Number of epochs
            lr: Learning rate
            verbose: Print progress if True (should be rank 0 only)
            val_loader: Optional validation DataLoader
            early_stopping_patience: Number of epochs to wait for improvement
            device: torch.device
        Returns:
            self: Trained model
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.train()
            total_loss = 0
            for batch in train_loader:
                # Multi-task batch: (X, y_cls_1, y_reg_1, y_cls_2, y_reg_2)
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = batch
                X_tensor = X.to(device)
                y_cls_1 = y_cls_1.to(device).unsqueeze(1)
                y_reg_1 = y_reg_1.to(device).unsqueeze(1)
                y_cls_2 = y_cls_2.to(device).unsqueeze(1)
                y_reg_2 = y_reg_2.to(device).unsqueeze(1)

                optimizer.zero_grad()
                cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = self(X_tensor)
                loss = self.__class__.custom_zi_loss(
                    cls_pred_1, reg_pred_1, y_cls_1, y_reg_1,
                    cls_pred_2, reg_pred_2, y_cls_2, y_reg_2
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)

            # Distributed average loss
            total_loss_tensor = torch.tensor(total_loss, device=device)
            n_samples_tensor = torch.tensor(len(train_loader.dataset), device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (total_loss_tensor / n_samples_tensor).item() if n_samples_tensor.item() > 0 else float('inf')

            # Validation & early stopping
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate_distributed(val_loader, device=device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                epoch_time = time.time() - epoch_start_time
                if verbose and dist.get_rank() == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.12f}, Val Loss: {val_loss:.12f}, Time: {epoch_time:.2f}s")

                if patience_counter >= early_stopping_patience:
                    if verbose and dist.get_rank() == 0:
                        print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.12f}")
                    if best_state is not None:
                        self.load_state_dict(best_state)
                    break
            else:
                epoch_time = time.time() - epoch_start_time
                if verbose and dist.get_rank() == 0 and (epoch % 5 == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.12f}, Time: {epoch_time:.2f}s")

        # Restore best weights if early stopping was used
        if val_loader is not None and best_state is not None:
            self.load_state_dict(best_state)
        return self


    def evaluate_distributed(self, val_loader, device=None):
        """
        Distributed evaluation loop using DataLoader and DDP. Returns global average loss.
        Args:
            val_loader: Validation DataLoader
            device: torch.device
        Returns:
            float: Global average loss
        """
        self.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                # Multi-task batch: (X, y_cls_1, y_reg_1, y_cls_2, y_reg_2)
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = batch
                X = X.to(device)
                y_cls_1 = y_cls_1.to(device).unsqueeze(1)
                y_reg_1 = y_reg_1.to(device).unsqueeze(1)
                y_cls_2 = y_cls_2.to(device).unsqueeze(1)
                y_reg_2 = y_reg_2.to(device).unsqueeze(1)
                
                cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = self(X)
                loss = self.custom_zi_loss(
                    cls_pred_1, reg_pred_1, y_cls_1, y_reg_1,
                    cls_pred_2, reg_pred_2, y_cls_2, y_reg_2
                )
                
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
        total_loss_tensor = torch.tensor(total_loss, device=device)
        n_samples_tensor = torch.tensor(n_samples, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (total_loss_tensor / n_samples_tensor).item() if n_samples_tensor.item() > 0 else float('inf')
        return avg_loss

    def fit(self, train_loader, epochs=20, lr=0.001, verbose=1, val_loader=None, early_stopping_patience=10):
        """
        Train the model using a PyTorch DataLoader.
        Args:
            train_loader: PyTorch DataLoader (X, y_cls, y_reg)
            epochs: Number of epochs
            lr: Learning rate
            verbose: Print progress if True
            val_loader: Optional validation DataLoader for early stopping
            early_stopping_patience: Number of epochs to wait for improvement
        Returns:
            self: Trained model
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        print(f"[fit] Using device: {device} (type: {device.type})")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                # Multi-task batch: (X, y_cls_1, y_reg_1, y_cls_2, y_reg_2)
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = batch
                X_tensor = X.to(device)
                y_cls_1 = y_cls_1.to(device).unsqueeze(1)
                y_reg_1 = y_reg_1.to(device).unsqueeze(1)
                y_cls_2 = y_cls_2.to(device).unsqueeze(1)
                y_reg_2 = y_reg_2.to(device).unsqueeze(1)

                optimizer.zero_grad()
                cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = self(X_tensor)
                loss = self.__class__.custom_zi_loss(
                    cls_pred_1, reg_pred_1, y_cls_1, y_reg_1,
                    cls_pred_2, reg_pred_2, y_cls_2, y_reg_2
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X.size(0)
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader.dataset):.12f}, Val Loss: {val_loss:.12f}")
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.12f}")
                    if best_state is not None:
                        self.load_state_dict(best_state)
                    break
            else:
                if verbose and (epoch % 5 == 0 or epoch == epochs-1):
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader.dataset):.12f}")
        # Restore best weights if early stopping was used
        if val_loader is not None and best_state is not None:
            self.load_state_dict(best_state)
        return self

    @staticmethod
    def custom_zi_loss(cls_pred_1, reg_pred_1, y_cls_1, y_reg_1, cls_pred_2, reg_pred_2, y_cls_2, y_reg_2, task_weights=None):
        """
        Custom loss for zero-inflated time series models with multi-task support.
        Combines binary cross-entropy for zero-class and masked MSE for regression for both tasks.
        
        Args:
            cls_pred_1: Classification predictions for task 1
            reg_pred_1: Regression predictions for task 1  
            y_cls_1: True zero-class labels for task 1
            y_reg_1: True regression targets for task 1
            cls_pred_2: Classification predictions for task 2
            reg_pred_2: Regression predictions for task 2
            y_cls_2: True zero-class labels for task 2
            y_reg_2: True regression targets for task 2
            task_weights: List of weights [w1, w2] for tasks (default: [0.5, 0.5])
        Returns:
            torch.Tensor: Combined loss
        """
        if task_weights is None:
            task_weights = [0.5, 0.5]
        
        # Task 1 loss
        bce_1 = nn.BCELoss()(cls_pred_1, y_cls_1)
        mask_1 = (y_cls_1 == 0).float()
        if mask_1.sum() > 0:
            mse_1 = ((reg_pred_1.squeeze() - y_reg_1.squeeze()) ** 2 * mask_1).sum() / (mask_1.sum() + 1e-6)
        else:
            mse_1 = torch.tensor(0.0, device=reg_pred_1.device)
        task_1_loss = bce_1 + mse_1
        
        # Task 2 loss
        bce_2 = nn.BCELoss()(cls_pred_2, y_cls_2)
        mask_2 = (y_cls_2 == 0).float()
        if mask_2.sum() > 0:
            mse_2 = ((reg_pred_2.squeeze() - y_reg_2.squeeze()) ** 2 * mask_2).sum() / (mask_2.sum() + 1e-6)
        else:
            mse_2 = torch.tensor(0.0, device=reg_pred_2.device)
        task_2_loss = bce_2 + mse_2
        
        total_loss = task_weights[0] * task_1_loss + task_weights[1] * task_2_loss
        return total_loss

    def evaluate(self, val_loader):
        """
        Evaluate the model on a validation DataLoader and return average loss.
        Args:
            val_loader: Validation DataLoader (expects pre-scaled data)
        Returns:
            float: Average loss
        """
        device = next(self.parameters()).device
        self.eval()
        total_loss = 0
        n_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                # Multi-task batch: (X, y_cls_1, y_reg_1, y_cls_2, y_reg_2)
                X, y_cls_1, y_reg_1, y_cls_2, y_reg_2 = batch
                X = X.to(device)
                y_cls_1 = y_cls_1.to(device).unsqueeze(1)
                y_reg_1 = y_reg_1.to(device).unsqueeze(1)
                y_cls_2 = y_cls_2.to(device).unsqueeze(1)
                y_reg_2 = y_reg_2.to(device).unsqueeze(1)
                
                cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = self(X)
                loss = self.custom_zi_loss(
                    cls_pred_1, reg_pred_1, y_cls_1, y_reg_1,
                    cls_pred_2, reg_pred_2, y_cls_2, y_reg_2
                )
                
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
        avg_loss = total_loss / n_samples if n_samples > 0 else float('inf')
        return avg_loss

    def predict(self, X):
        """
        Predict regression and classification outputs for input X for both tasks.
        Args:
            X: Input tensor or ndarray of shape (n_samples, n_lags, n_features).
        Returns:
            tuple: (reg_pred_1, cls_pred_1, reg_pred_2, cls_pred_2) for both tasks
        """
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
            else:
                X_tensor = X.to(device)
            cls_pred_1, reg_pred_1, cls_pred_2, reg_pred_2 = self(X_tensor)
            
            # Convert to numpy and apply thresholds
            cls_pred_1 = (cls_pred_1.cpu().numpy().flatten() > 0.5).astype(int)
            reg_pred_1 = reg_pred_1.cpu().numpy().flatten()
            cls_pred_2 = (cls_pred_2.cpu().numpy().flatten() > 0.5).astype(int)
            reg_pred_2 = reg_pred_2.cpu().numpy().flatten()
            
        return reg_pred_1, cls_pred_1, reg_pred_2, cls_pred_2

class ZeroInflatedTransformer(ZeroInflatedTSModule):
    """
    Transformer-based zero-inflated time series model for multi-task learning.
    Uses a transformer encoder to process sequential input data, with shared dense layers and separate
    heads for classification (zero/non-zero) and regression (value prediction) for 2 tasks.
    
    Args:
        input_size (int): Number of input features.
        n_lags (int): Number of lag steps.
        d_model (int): Transformer model dimension.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dense_units (int): Number of units in shared dense layer.
        dropout (float): Dropout rate.
    """
    def __init__(self, input_size, n_lags=1, d_model=32, num_heads=2, num_layers=2, dense_units=16, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.n_lags = n_lags
        self.d_model = d_model
        
        # Shared layers
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_lags, d_model))
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_model*2, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared_dense = nn.Linear(d_model, dense_units)
        
        # Task 1 heads
        self.classifier_1 = nn.Linear(dense_units, 1)
        self.regressor_dense_1 = nn.Linear(dense_units, dense_units)
        self.regressor_1 = nn.Linear(dense_units, 1)
        
        # Task 2 heads
        self.classifier_2 = nn.Linear(dense_units, 1)
        self.regressor_dense_2 = nn.Linear(dense_units, dense_units)
        self.regressor_2 = nn.Linear(dense_units, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Use last token
        shared_repr = self.relu(self.shared_dense(x))
        
        # Task 1 outputs
        cls_out_1 = self.sigmoid(self.classifier_1(shared_repr))
        reg_x_1 = self.relu(self.regressor_dense_1(shared_repr))
        reg_out_1 = self.regressor_1(reg_x_1)
        
        # Task 2 outputs
        cls_out_2 = self.sigmoid(self.classifier_2(shared_repr))
        reg_x_2 = self.relu(self.regressor_dense_2(shared_repr))
        reg_out_2 = self.regressor_2(reg_x_2)
        
        return cls_out_1, reg_out_1, cls_out_2, reg_out_2

