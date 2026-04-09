import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IncomeClassifier(nn.Module):
    """
    Feedforward neural network for binary income classification.

    Architecture
    ────────────
    Input(99) → Linear(128) → ReLU → Dropout(p)
              → Linear(64)  → ReLU → Dropout(p)
              → Linear(32)  → ReLU
              → Linear(1)   → Sigmoid

    Design notes
    ────────────
    • Three hidden layers give enough depth to capture interactions without overfitting.
    • ReLU is the default activation; the constructor also accepts
      'leaky_relu' and 'elu' for comparison.
    • Dropout (default 0.4) on the first two hidden layers is the
      primary regulariser.
    • Sigmoid output → weighted BCELoss during training.
    • No BatchNorm: on this imbalanced tabular dataset with weighted
      loss, per-batch normalisation introduced instability in the
      validation statistics and worsened generalisation.
    """

    ACTIVATIONS = {
        'relu':       nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'elu':        nn.ELU(alpha=1.0),
    }

    def __init__(self, input_dim: int, dropout_rate: float = 0.4,
                 activation: str = 'relu'):
        super().__init__()

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(self.ACTIVATIONS)}"
            )

        act = self.ACTIVATIONS[activation]

        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_dim, 128),
            act,
            nn.Dropout(p=dropout_rate),

            # Hidden layer 2
            nn.Linear(128, 64),
            act,
            nn.Dropout(p=dropout_rate),

            # Hidden layer 3
            nn.Linear(64, 32),
            act,

            # Output — Sigmoid for binary classification
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class DeepLearningPipeline:
    """
    PyTorch training pipeline for the Adult Census Income dataset.

    Parameters
    ──────────
    input_dim    : number of features after preprocessing (99)
    dropout_rate : dropout probability on hidden layers 1 & 2 (default 0.5)
    activation   : hidden-layer activation — 'relu' | 'leaky_relu' | 'elu'
    lr           : learning rate for Adam optimiser (default 0.0003)
    batch_size   : mini-batch size (default 128)
    max_epochs   : hard upper limit on training epochs (default 200)
    patience     : early-stopping patience (default 5)
    """

    def __init__(self, input_dim: int, dropout_rate: float=0.4, activation: str='relu', lr: float=0.0003,
                 batch_size: int=64, max_epochs: int=200, patience: int=10,):
        self.input_dim    = input_dim
        self.dropout_rate = dropout_rate
        self.activation   = activation
        self.lr           = lr
        self.batch_size   = batch_size
        self.max_epochs   = max_epochs
        self.patience     = patience

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Device: {self.device}")

        self.model           = None
        self.train_loss_hist = []
        self.val_loss_hist   = []
        self.best_epoch      = 0


    # Helpers
    def _to_tensors(self, X, y=None):
        X_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X,
            dtype=torch.float32
        ).to(self.device)
        if y is None:
            return X_t
        y_t = torch.tensor(
            y.values if isinstance(y, pd.Series) else y,
            dtype=torch.float32
        ).to(self.device)
        return X_t, y_t

    def _make_loader(self, X_t, y_t, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False
        )

    def _pos_weight(self, y_train) -> torch.Tensor:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        w = n_neg / n_pos
        logging.info(
            f"Class balance — Negative: {n_neg} | "
            f"Positive: {n_pos} | pos_weight: {w:.2f}"
        )
        return torch.tensor(w, dtype=torch.float32).to(self.device)

    def _weighted_bce(self, y_pred, y_true, pos_weight):
        loss    = nn.BCELoss(reduction='none')(y_pred, y_true)
        weights = torch.where(y_true == 1, pos_weight, torch.ones_like(y_true))
        return (loss * weights).mean()

    @torch.no_grad()
    def _evaluate(self, X, y, split_name: str) -> dict:
        self.model.eval()
        X_t, y_t = self._to_tensors(X, y)
        y_prob = self.model(X_t).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = y_t.cpu().numpy().astype(int)

        metrics = {
            'accuracy' : accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall'   : recall_score(y_true, y_pred, zero_division=0),
            'f1'       : f1_score(y_true, y_pred, zero_division=0),
            'roc_auc'  : roc_auc_score(y_true, y_prob),
        }
        logging.info(f"--- {split_name} Metrics ---")
        for name, val in metrics.items():
            logging.info(f"  {name:>10}: {val:.4f}")
        return metrics


    # Plot
    def _plot_loss_curves(self):
        os.makedirs('plots', exist_ok=True)

        epochs = range(1, len(self.train_loss_hist) + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, self.train_loss_hist,
                 label='Train Loss', color='steelblue', lw=2)
        plt.plot(epochs, self.val_loss_hist,
                 label='Validation Loss', color='tomato', lw=2)
        plt.axvline(
            x=self.best_epoch, color='grey', linestyle='--',
            label=f'Best epoch ({self.best_epoch})'
        )
        plt.xlabel('Epoch')
        plt.ylabel('Weighted BCE Loss')
        plt.title(
            f'Neural Network — Loss Curves\n'
            f'(activation={self.activation}, dropout={self.dropout_rate}, '
            f'lr={self.lr})'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/neural_loss_curves.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/neural_loss_curves.png'")


    # Training loop
    def _train(self, train_loader, X_val_t, y_val_t, pos_weight):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # best_val_loss    = float('inf')
        best_val_f1      = 0.0
        epochs_no_improv = 0
        best_weights     = None

        for epoch in range(1, self.max_epochs + 1):

            # Train
            self.model.train()
            epoch_loss = 0.0
            for X_b, y_b in train_loader:
                optimiser.zero_grad()
                y_pred = self.model(X_b)
                loss   = self._weighted_bce(y_pred, y_b, pos_weight)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(y_b)

            avg_train_loss = epoch_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = self._weighted_bce(
                    val_pred, y_val_t, pos_weight
                ).item()
                val_pred_np = (val_pred.cpu().numpy() >= 0.5).astype(int)
                val_true_np = y_val_t.cpu().numpy().astype(int)
                val_f1 = f1_score(val_true_np, val_pred_np, zero_division=0)

            self.train_loss_hist.append(avg_train_loss)
            self.val_loss_hist.append(val_loss)
            # Early stopping based on F1 score
            if epoch % 10 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:>4}/{self.max_epochs} | "
                    f"Train: {avg_train_loss:.4f} | "
                    f"Val: {val_loss:.4f}"
                )

            if val_f1 > best_val_f1 + 1e-5:
                best_val_f1 = val_f1
                epochs_no_improv = 0
                best_weights = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                self.best_epoch = epoch
            else:
                epochs_no_improv += 1
                if epochs_no_improv >= self.patience:
                    logging.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch: {self.best_epoch} "
                        f"(val F1: {best_val_f1:.4f})"
                    )
                    break
        self.model.load_state_dict(best_weights)
        logging.info(f"Best weights restored from epoch {self.best_epoch}.")


    def run_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame,y_val: pd.Series,):
        logging.info("Starting Neural Network Pipeline (PyTorch)")
        logging.info(
            f"Config — activation={self.activation} | "
            f"dropout={self.dropout_rate} | lr={self.lr} | "
            f"batch_size={self.batch_size} | patience={self.patience}"
        )

        # Build model
        self.model = IncomeClassifier(
            input_dim=self.input_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(self.device)

        logging.info(f"Architecture:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Trainable parameters: {total_params:,}")

        # Prepare data
        X_train_t, y_train_t  = self._to_tensors(X_train, y_train)
        X_val_t,   y_val_t    = self._to_tensors(X_val,   y_val)
        train_loader          = self._make_loader(X_train_t, y_train_t, shuffle=True)
        pos_weight            = self._pos_weight(y_train)

        # Train
        self._train(train_loader, X_val_t, y_val_t, pos_weight)

        # Classification report
        self.model.eval()
        with torch.no_grad():
            y_val_prob = self.model(X_val_t).cpu().numpy()
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        logging.info(
            "Classification Report (Validation Set):\n" +
            classification_report(
                y_val.values, y_val_pred,
                target_names=['<=50K', '>50K']
            )
        )

        # Metrics
        self._evaluate(X_train, y_train, split_name='Train')
        val_metrics = self._evaluate(X_val, y_val, split_name='Validation')

        # Plot
        self._plot_loss_curves()

        # Save
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), 'models/neural_network.pt')
        logging.info("Model saved to 'models/neural_network.pt'.")

        return val_metrics
