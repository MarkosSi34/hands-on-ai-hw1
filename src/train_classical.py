import os
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ClassicalPipeline:
    """
    Classical ML pipeline for the Adult Census Income dataset.

    Model  : XGBoost Binary Classifier
    Task   : Predict whether income > $50K/yr  (0 = ≤50K, 1 = >50K)

    Design choices
    ──────────────
    • XGBoost is preferred over Random Forest on this dataset because it
      handles class imbalance via scale_pos_weight, supports early stopping
      with a validation eval_set, and consistently outperforms simpler
      tree ensembles on tabular income data.
    • scale_pos_weight = n_negative / n_positive corrects for the ~76/24
      class imbalance without resampling.
    • Early stopping (patience=20) prevents overfitting
    • The fitted model is saved to models/classical_model.pkl.
    """

    def __init__(self):
        self.model = None

    def _build_model(self, scale_pos_weight: float):
        """Instantiate XGBClassifier with reproducible, sensible defaults."""
        return XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            early_stopping_rounds=20,
            use_label_encoder=False,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )

    def _evaluate(self, X, y, split_name: str):
        """Return a dict of classification metrics for one split."""
        y_pred      = self.model.predict(X)
        y_prob      = self.model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy' : accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall'   : recall_score(y, y_pred, zero_division=0),
            'f1'       : f1_score(y, y_pred, zero_division=0),
            'roc_auc'  : roc_auc_score(y, y_prob),
        }

        logging.info(f"--- {split_name} Metrics ---")
        for name, val in metrics.items():
            logging.info(f"  {name:>10}: {val:.4f}")

        return metrics

    def _plot_feature_importance(self, feature_names: list, top_n: int = 20):
        """
        Bar chart of the top-N features by XGBoost gain importance.
        Saved to plots/classical_feature_importance.png
        """
        os.makedirs('plots', exist_ok=True)

        importance = self.model.get_booster().get_score(importance_type='gain')

        if all(k.startswith('f') and k[1:].isdigit() for k in importance):
            importance = {
                feature_names[int(k[1:])]: v
                for k, v in importance.items()
            }

        imp_series = (
            pd.Series(importance)
              .sort_values(ascending=False)
              .head(top_n)
        )

        plt.figure(figsize=(10, 7))
        sns.barplot(x=imp_series.values, y=imp_series.index, palette='viridis')
        plt.xlabel('Gain Importance')
        plt.title(f'XGBoost — Top {top_n} Feature Importances (Gain)')
        plt.tight_layout()
        plt.savefig('plots/classical_feature_importance.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/classical_feature_importance.png'")

    def _plot_learning_curve(self):
        """
        Training vs. validation metric across boosting rounds.
        Saved to plots/classical_learning_curve.png
        """
        os.makedirs('plots', exist_ok=True)

        results = self.model.evals_result()
        metric_name = list(results['validation_0'].keys())[0]
        train_vals = results['validation_0'][metric_name]
        val_vals   = results['validation_1'][metric_name]

        epochs = range(1, len(train_vals) + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, train_vals, label=f'Train {metric_name}',      color='steelblue')
        plt.plot(epochs, val_vals,   label=f'Validation {metric_name}', color='tomato')
        plt.axvline(
            x=self.model.best_iteration,
            color='grey', linestyle='--',
            label=f'Best iteration ({self.model.best_iteration})'
        )
        plt.xlabel('Boosting Round')
        plt.ylabel(metric_name)
        plt.title(f'XGBoost — Training vs Validation {metric_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/classical_learning_curve.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/classical_learning_curve.png'")

    def run_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,):
        """
        Full training pipeline:
          1. Build model
          2. Fit with early stopping on validation set
          3. Evaluate on train & val
          4. Plot feature importance + learning curve
          5. Save model to models/classical_model.pkl

        Returns
        ───────
        val_metrics : dict  — validation metrics for model comparison in evaluate.py
        """

        logging.info("Starting Classical ML Pipeline (XGBoost)")

        # Class imbalance correction
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / n_pos
        logging.info(
            f"Class balance — Negative (≤50K): {n_neg} | "
            f"Positive (>50K): {n_pos} | "
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

        # Build & train
        self.model = self._build_model(scale_pos_weight)
        logging.info("Fitting XGBoost (early stopping patience=20)...")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        logging.info(
            f"Training complete. "
            f"Best iteration: {self.model.best_iteration} / {self.model.n_estimators}"
        )

        # Detailed classification report on validation set
        y_val_pred = self.model.predict(X_val)
        logging.info("Classification Report (Validation Set):\n" +
                     classification_report(y_val, y_val_pred,
                                           target_names=['<=50K', '>50K']))

        # Evaluate both splits
        self._evaluate(X_train, y_train, split_name='Train')
        val_metrics = self._evaluate(X_val, y_val,   split_name='Validation')

        # Plots
        feature_names = list(X_train.columns)
        self._plot_feature_importance(feature_names, top_n=20)
        self._plot_learning_curve()

        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/classical_model.pkl')
        logging.info("Model saved to 'models/classical_model.pkl'.")

        return val_metrics