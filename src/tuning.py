import os
import logging
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ModelTuner:
    """
    Hyperparameter tuning for the classical (XGBoost) model.

    Uses RandomizedSearchCV with Stratified K-Fold cross-validation
    on X_train, then confirms on X_val before touching X_test.

    Saves:
      • models/tuned_classical_model.pkl   (XGBoost best estimator)
      • plots/tuning_xgb_results.png       (top-N CV scores bar chart)
    """

    def __init__(self):
        logging.info("ModelTuner initialised.")

    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     param_grid: dict, n_iter:  int = 20, cv_folds: int = 5) :
        """
        RandomizedSearchCV with StratifiedKFold.

        Parameters
        ──────────
        X_train    : scaled training features
        y_train    : binary target
        param_grid : dict of parameter lists, e.g.
                     {'max_depth': [3,5,7],
                      'learning_rate': [0.01,0.1],
                      'n_estimators': [100,200]}
        n_iter     : number of random combinations to try
        cv_folds   : number of stratified CV folds

        Returns
        ───────
        best_estimator : fitted XGBClassifier with best params
        """
        logging.info("=" * 55)
        logging.info("Starting XGBoost Hyperparameter Tuning")
        logging.info(f"Search space: {param_grid}")
        logging.info(f"n_iter={n_iter}, cv={cv_folds}-fold stratified")
        logging.info("=" * 55)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw   = n_neg / n_pos

        base_model = XGBClassifier(
            scale_pos_weight=spw,
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=cv,
            refit=True,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        search.fit(
            X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            y_train.values if isinstance(y_train, pd.Series)    else y_train
        )

        logging.info(f"Best params  : {search.best_params_}")
        logging.info(f"Best CV AUC  : {search.best_score_:.4f}")

        # Plot top results
        # self._plot_xgb_results(search.cv_results_)

        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(search.best_estimator_, 'models/tuned_classical_model.pkl')
        logging.info("Tuned XGBoost saved to 'models/tuned_classical_model.pkl'.")

        logging.info("=" * 55)
        logging.info("XGBoost tuning complete.")
        logging.info("=" * 55)

        return search.best_estimator_

    def _plot_xgb_results(self, cv_results: dict, top_n: int = 15):
        """Bar chart of top-N mean CV AUC scores across parameter combos."""
        os.makedirs('plots', exist_ok=True)

        results_df = pd.DataFrame(cv_results)
        results_df = results_df.sort_values('mean_test_score', ascending=False).head(top_n)

        labels = [
            f"md={r.get('param_max_depth','')} | "
            f"lr={r.get('param_learning_rate','')} | "
            f"n={r.get('param_n_estimators','')} | "
            f"ss={r.get('param_subsample','')} | "
            f"cs={r.get('param_colsample_bytree','')}"
            for _, r in results_df.iterrows()
        ]

        plt.figure(figsize=(11, 6))
        plt.barh(
            range(len(results_df)),
            results_df['mean_test_score'],
            xerr=results_df['std_test_score'],
            color='steelblue', alpha=0.85,
            error_kw={'elinewidth': 1, 'capsize': 3}
        )
        plt.yticks(range(len(results_df)), labels, fontsize=8)
        plt.xlabel('Mean CV ROC-AUC (± std)')
        plt.title(f'XGBoost Tuning — Top {top_n} Configurations')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('plots/tuning_xgb_results.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/tuning_xgb_results.png'")

    @staticmethod
    def _metrics(model, X, y):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_prob),
        }

    def compare_base_vs_tuned(self, X_val, y_val):
        logging.info("=" * 60)
        logging.info("Base vs Tuned XGBoost Comparison (Validation Set)")
        logging.info("=" * 60)

        base_model = joblib.load('models/classical_model.pkl')
        tuned_model = joblib.load('models/tuned_classical_model.pkl')

        base_m = self._metrics(base_model, X_val, y_val)
        tuned_m = self._metrics(tuned_model, X_val, y_val)

        sep = "-" * 60
        logging.info(sep)
        logging.info(f"{'Metric':<12} {'Base':>10} {'Tuned':>10} {'Diff':>10} {'Winner':>10}")
        logging.info(sep)
        for key in base_m:
            diff = tuned_m[key] - base_m[key]
            winner = "Tuned" if diff > 0 else ("Base" if diff < 0 else "Tie")
            logging.info(
                f"{key:<12} {base_m[key]:>10.4f} {tuned_m[key]:>10.4f} "
                f"{diff:>+10.4f} {winner:>10}"
            )
        logging.info(sep)

    def evaluate_on_test(self, X_test, y_test):
        base_model = joblib.load('models/classical_model.pkl')
        tuned_model = joblib.load('models/tuned_classical_model.pkl')

        base_m = self._metrics(base_model, X_test, y_test)
        tuned_m = self._metrics(tuned_model, X_test, y_test)

        sep = "-" * 60
        logging.info("=" * 60)
        logging.info("Base vs Tuned XGBoost Comparison (Test Set)")
        logging.info("=" * 60)
        logging.info(sep)
        logging.info(f"{'Metric':<12} {'Base':>10} {'Tuned':>10} {'Diff':>10} {'Winner':>10}")
        logging.info(sep)
        for key in base_m:
            diff = tuned_m[key] - base_m[key]
            winner = "Tuned" if diff > 0 else ("Base" if diff < 0 else "Tie")
            logging.info(
                f"{key:<12} {base_m[key]:>10.4f} {tuned_m[key]:>10.4f} "
                f"{diff:>+10.4f} {winner:>10}"
            )
        logging.info(sep)
        return tuned_m