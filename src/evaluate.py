import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shutil
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, roc_curve
)
from src.train_neural import IncomeClassifier


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EvaluationPipeline:
    """
    Loads both saved models, evaluates them on the held-out test set,
    produces all required plots, prints a side-by-side comparison table,
    and designates the best model.

    Expected saved artifacts
    ────────────────────────
    models/classical_model.pkl  — fitted XGBClassifier
    models/neural_network.pt    — IncomeClassifier state_dict
    models/scaler.pkl           — fitted StandardScaler (not used here,
                                   but verified to exist)
    """

    CLASS_NAMES = ['<=50K', '>50K']
    METRICS     = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    def __init__(self, input_dim: int, dropout_rate: float = 0.4,
                 activation: str = 'relu'):
        self.input_dim    = input_dim
        self.dropout_rate = dropout_rate
        self.activation   = activation
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_classical(self):
        path = 'models/classical_model.pkl'
        logging.info(f"Loading classical model from '{path}'...")
        model = joblib.load(path)
        return model

    def _load_neural(self):
        path = 'models/neural_network.pt'
        logging.info(f"Loading neural network from '{path}'...")
        model = IncomeClassifier(
            input_dim=self.input_dim,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    def _predict_classical(self, model, X):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        return y_pred, y_prob

    def _predict_neural(self, model, X):
        X_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            y_prob = model(X_t).cpu().numpy()

        y_pred = (y_prob >= 0.5).astype(int)
        return y_pred, y_prob

    def _compute_metrics(self, y_true, y_pred, y_prob):
        return {
            'accuracy' : accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall'   : recall_score(y_true, y_pred, zero_division=0),
            'f1'       : f1_score(y_true, y_pred, zero_division=0),
            'roc_auc'  : roc_auc_score(y_true, y_prob),
        }

    def _plot_confusion_matrices(self, y_true, y_pred_clf, y_pred_nn):
        """Side-by-side confusion matrices for both models."""
        os.makedirs('plots', exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Confusion Matrices — Test Set', fontsize=14, fontweight='bold')

        for ax, y_pred, title in zip(
            axes,
            [y_pred_clf, y_pred_nn],
            ['XGBoost (Classical)', 'Neural Network']
        ):
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=self.CLASS_NAMES)
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(title, fontsize=12)

        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/confusion_matrices.png'")

    def _plot_roc_curves(self, y_true, y_prob_clf, y_prob_nn):
        """Overlapping ROC curves for both models on the same axes."""
        os.makedirs('plots', exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        for y_prob, label, color in [
            (y_prob_clf, 'XGBoost',        'steelblue'),
            (y_prob_nn,  'Neural Network', 'tomato'),
        ]:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, label=f'{label}  (AUC = {auc:.3f})', color=color, lw=2)

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — Test Set', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/roc_curves.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/roc_curves.png'")

    def _plot_metrics_comparison(self, clf_metrics: dict, nn_metrics: dict):
        """Grouped bar chart comparing all five metrics side by side."""
        os.makedirs('plots', exist_ok=True)

        x = np.arange(len(self.METRICS))
        width = 0.35

        clf_vals = [clf_metrics[m] for m in self.METRICS]
        nn_vals  = [nn_metrics[m]  for m in self.METRICS]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, clf_vals, width,
                       label='XGBoost',        color='steelblue', alpha=0.85)
        bars2 = ax.bar(x + width / 2, nn_vals,  width,
                       label='Neural Network', color='tomato',    alpha=0.85)

        # Value labels on top of each bar
        for bar in bars1 + bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=9
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in self.METRICS])
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison — Test Set Metrics', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/metrics_comparison.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/metrics_comparison.png'")

    def _print_comparison_table(self, clf_metrics: dict, nn_metrics: dict):
        sep = "-" * 55
        logging.info(sep)
        logging.info("FINAL TEST SET COMPARISON")
        logging.info(sep)
        logging.info(f"{'Metric':<12} {'XGBoost':>10} {'Neural Net':>12} {'Winner':>12}")
        logging.info(sep)

        for metric in self.METRICS:
            clf_val = clf_metrics[metric]
            nn_val = nn_metrics[metric]
            winner = 'XGBoost' if clf_val >= nn_val else 'Neural Net'
            logging.info(
                f"{metric:<12} {clf_val:>10.4f} {nn_val:>12.4f} {winner:>12}"
            )

        logging.info(sep)

    def _designate_best_model(self, clf_metrics: dict, nn_metrics: dict):
        """
        Compares models on ROC-AUC (primary) and F1 (tiebreaker).
        Copies the winner to models/best_model.pkl.

        ROC-AUC is chosen as the primary metric because the dataset is
        imbalanced (76/24) — it measures discrimination across all
        thresholds and is not distorted by class skew.
        """
        clf_score = clf_metrics['roc_auc']
        nn_score  = nn_metrics['roc_auc']

        logging.info(
            f"Best model selection — "
            f"XGBoost ROC-AUC: {clf_score:.4f} | "
            f"Neural Net ROC-AUC: {nn_score:.4f}"
        )

        if clf_score >= nn_score:
            best_name = 'XGBoost'
            shutil.copy('models/classical_model.pkl', 'models/best_model.pkl')
            logging.info("Best model: XGBoost → copied to 'models/best_model.pkl'")
        else:
            best_name = 'Neural Network'
            best_bundle = {
                'type':         'neural_network',
                'input_dim':    self.input_dim,
                'dropout_rate': self.dropout_rate,
                'activation':   self.activation,
                'weights_path': 'models/neural_network.pt',
            }
            import joblib as jl
            jl.dump(best_bundle, 'models/best_model.pkl')
            logging.info("Best model: Neural Network → descriptor saved to 'models/best_model.pkl'")

        return best_name


    def run_pipeline(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Full evaluation pipeline:
          1. Load both saved models
          2. Generate predictions on X_test
          3. Compute all five metrics for each model
          4. Plot confusion matrices, ROC curves, metrics bar chart
          5. Print side-by-side comparison table
          6. Designate and save best_model.pkl
        """
        logging.info("Starting Evaluation Pipeline...")

        os.makedirs('plots',  exist_ok=True)
        os.makedirs('models', exist_ok=True)

        y_true = (
            y_test.values if isinstance(y_test, pd.Series) else y_test
        ).astype(int)


        clf_model = self._load_classical()
        nn_model  = self._load_neural()


        y_pred_clf, y_prob_clf = self._predict_classical(clf_model, X_test)
        y_pred_nn,  y_prob_nn  = self._predict_neural(nn_model,  X_test)


        logging.info("XGBoost Test Metrics")
        clf_metrics = self._compute_metrics(y_true, y_pred_clf, y_prob_clf)

        logging.info("Neural Network Test Metrics")
        nn_metrics = self._compute_metrics(y_true, y_pred_nn, y_prob_nn)

        # Plots
        self._plot_confusion_matrices(y_true, y_pred_clf, y_pred_nn)
        self._plot_roc_curves(y_true, y_prob_clf, y_prob_nn)
        self._plot_metrics_comparison(clf_metrics, nn_metrics)

        # Comparison table
        self._print_comparison_table(clf_metrics, nn_metrics)

        # Best model
        best_name = self._designate_best_model(clf_metrics, nn_metrics)
        logging.info(f"Evaluation complete. Best model: {best_name}")

        return clf_metrics, nn_metrics, best_name
