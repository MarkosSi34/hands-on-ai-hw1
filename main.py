import argparse
import logging
import warnings
from src.preprocessing import Preprocessing
from src.train_classical import ClassicalPipeline
from src.train_neural import DeepLearningPipeline
from src.evaluate import EvaluationPipeline
from src.tuning import ModelTuner
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(
        description="Adult Census Income — ML Pipeline (Binary Classification)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['xgb', 'nn', 'both'],
        default='both',
        help="Which model pipeline to run (xgb, nn, or both). Default: both."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="If flagged, runs hyperparameter tuning for the XGBoost model."
    )
    args = parser.parse_args()

    logging.info("Initializing Adult Census Income ML Pipeline...")

    preprocessor = Preprocessing(src_data_path="data/adult_census_income.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_pipeline()

    if args.tune:
        tuner = ModelTuner()

        xgb_grid = {
            'max_depth':        [3, 5, 6, 7, 9],
            'learning_rate':    [0.01, 0.05, 0.1, 0.2],
            'n_estimators':     [100, 300, 500, 700],
            'subsample':        [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        tuner.tune_xgboost(X_train, y_train, xgb_grid)
        tuner.compare_base_vs_tuned(X_val, y_val)
        tuner.evaluate_on_test(X_test, y_test)
        logging.info("Tuning mode finished. Exiting pipeline.")
        return

    if args.model in ['xgb', 'both']:
        classical_pipeline = ClassicalPipeline()
        classical_pipeline.run_pipeline(X_train, y_train, X_val, y_val)
        classical_pipeline._evaluate(X_test, y_test, split_name='Test')
        logging.info("Classical ML Pipeline complete.")

    if args.model in ['nn', 'both']:
        nn_pipeline = DeepLearningPipeline(input_dim=X_train.shape[1])
        nn_pipeline.run_pipeline(X_train, y_train, X_val, y_val)
        nn_pipeline._evaluate(X_test, y_test, split_name='Test')
        logging.info("Neural Network Pipeline complete.")

    if args.model == 'both':
        eval_pipeline = EvaluationPipeline(input_dim=X_train.shape[1])
        eval_pipeline.run_pipeline(X_test, y_test)


if __name__ == "__main__":
    main()