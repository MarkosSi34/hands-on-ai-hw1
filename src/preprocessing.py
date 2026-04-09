import os
import logging
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Preprocessing:
    """
    Full preprocessing pipeline for the UCI Adult Census Income dataset.

    Domain  : Socioeconomic / Demographics
    Task    : Binary Classification — predict whether income > $50K/yr
    Source  : https://archive.ics.uci.edu/dataset/2/adult
    """
    TARGET = 'income'
    NUMERICAL_COLS = [
        'age', 'fnlwgt', 'education_num',
        'capital_gain', 'capital_loss', 'hours_per_week'
    ]
    OHE_COLS = [
        'workclass', 'education', 'marital_status',
        'occupation', 'relationship', 'race', 'native_country'
    ]
    LABEL_COLS = ['sex']

    def __init__(self, src_data_path: str):
        logging.info(f"Loading dataset from '{src_data_path}'...")
        self.data = pd.read_csv(src_data_path)

        # Strip accidental whitespace from string columns
        str_cols = self.data.select_dtypes('object').columns
        self.data[str_cols] = self.data[str_cols].apply(lambda c: c.str.strip())
        # Replace '?' with NaN
        self.data.replace('?', np.nan, inplace=True)

        logging.info(f"Dataset loaded. Shape: {self.data.shape}")

        # Encode target immediately so it is always a clean 0/1 integer
        # '>50K' → 1  |  '<=50K' → 0
        self.data[self.TARGET] = (self.data[self.TARGET] == '>50K').astype(int)

        # Partition placeholders
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None


    @property
    def X(self):
        """All columns except the target."""
        return self.data.drop(columns=[self.TARGET])

    @property
    def y(self):
        """Target column."""
        return self.data[self.TARGET]

    @property
    def _datasets(self):
        """Apply only on training set"""
        return [self.X_train, self.X_val, self.X_test]

    def split_data(self):
        """
        Stratified 80/10/10 split.
        Split FIRST — all preprocessing statistics are computed on X_train only.
        """
        logging.info("Starting 80/10/10 Stratified Data Split...")

        # Hold out 10 % as test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.10,
            stratify=self.y,
            random_state=42
        )

        # From the remaining 90 %, hold out ~11 % → yields 10 % of total
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.1111,
            stratify=y_temp,
            random_state=42
        )

        logging.info(
            f"Split complete. "
            f"Train: {self.X_train.shape[0]} rows | "
            f"Val: {self.X_val.shape[0]} rows | "
            f"Test: {self.X_test.shape[0]} rows"
        )

    def impute_missing_values(self):
        """
        Columns with missing values (originally encoded as '?')
        Statistics computed on X_train only, then applied to all splits.
        """
        logging.info("Starting Imputation of Missing Values...")

        missing_cols = [
            col for col in self.X_train.columns if self.X_train[col].isna().any()
        ]

        fill_values = {}
        for col in missing_cols:
            mode_val = self.X_train[col].mode()[0]
            fill_values[col] = mode_val
            logging.info(f"Mode of '{col}' from X_train: '{mode_val}'")

        for df in self._datasets:
            for col, val in fill_values.items():
                df[col] = df[col].fillna(val)

        remaining = self.X_train.isna().sum().sum()
        self._fill_values = fill_values
        logging.info(f"Imputation complete. Missing values remaining in X_train: {remaining}")

    def treat_outliers(self):
        """
        IQR bounds computed on X_train only, then applied (clip) to all splits.

        Columns treated:
          - age, fnlwgt, capital_gain, capital_loss, hours_per_week

        Note: 'education_num' is an ordinal scale (1–16) with no true outliers
        and is intentionally excluded.
        """
        logging.info("Starting Outlier Treatment (IQR Winsorizing)...")

        outlier_cols = [
            col for col in self.NUMERICAL_COLS
            if self.X_train[col].nunique() > 20
        ]
        self._iqr_bounds = {}
        for col in outlier_cols:
            Q1 = self.X_train[col].quantile(0.25)
            Q3 = self.X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                logging.info(f"Skipping '{col}': IQR is 0 (sparse feature)")
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self._iqr_bounds[col] = (lower, upper)
            logging.info(f"IQR bounds for '{col}': [{lower:.2f}, {upper:.2f}]")

            for df in self._datasets:
                df[col] = df[col].clip(lower=lower, upper=upper)

        logging.info("Outlier treatment complete.")

    def encode_categorical(self):
        """
        - OHE_COLS  (nominal, multi-value) → OneHotEncoder(drop='first')
        - LABEL_COLS (binary: sex)          → LabelEncoder  (Female=0, Male=1)

        Encoder fitted on X_train only, then applied to all splits.
        'education_num' already encodes education ordinally → 'education' (text)
        is still one-hot encoded for completeness, but both are kept.
        """
        logging.info("Starting Categorical Encoding...")
        self._group_rare_countries(top_n=5)
        # Binary categorical → LabelEncoder
        label_cols = [
            col for col in self.X_train.select_dtypes('object').columns
            if self.X_train[col].nunique() == 2
        ]

        # Multi-value categorical → OneHotEncoder
        ohe_cols = [
            col for col in self.X_train.select_dtypes('object').columns
            if self.X_train[col].nunique() > 2
        ]
        self._label_encoders = {}
        for col in label_cols:
            le = LabelEncoder()
            le.fit(self.X_train[col])
            self._label_encoders[col] = le
            for df in self._datasets:
                df[col] = le.transform(df[col])

        # One-Hot encode nominal columns
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(self.X_train[ohe_cols])

        new_cols = self.encoder.get_feature_names_out(ohe_cols)
        logging.info(f"  OHE produced {len(new_cols)} new columns.")

        self.X_train = self._apply_ohe(self.X_train, new_cols, ohe_cols)
        self.X_val = self._apply_ohe(self.X_val, new_cols, ohe_cols)
        self.X_test = self._apply_ohe(self.X_test, new_cols, ohe_cols)

        logging.info("Categorical encoding complete.")

    def _group_rare_countries(self, top_n: int = 5):
        """Group infrequent native_country values into 'Other' (fitted on X_train)."""
        logging.info(f"Grouping rare native_country values (keeping top {top_n})...")

        top_countries = (
            self.X_train['native_country']
            .value_counts()
            .head(top_n)
            .index.tolist()
        )
        self._top_countries = top_countries
        for df in self._datasets:
            df['native_country'] = df['native_country'].where(
                df['native_country'].isin(top_countries), other='Other'
            )

        logging.info(f"Kept: {top_countries + ['Other']}")

    def _apply_ohe(self, df: pd.DataFrame, new_cols, ohe_cols):
        """Helper: transform OHE cols and stitch back onto the DataFrame."""
        encoded_array = self.encoder.transform(df[ohe_cols])
        encoded_df    = pd.DataFrame(encoded_array, columns=new_cols, index=df.index)
        return pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    def engineer_features(self):
        """
        Two new domain-relevant features:

        1. capital_net  = capital_gain - capital_loss
           Rationale: raw gain and loss carry redundant information separately;
           their net value is the economically meaningful signal for income level.

        2. hours_education_interaction = hours_per_week * education_num
           Rationale: combines work effort with education level — someone who
           works long hours AND has high education is much more likely to earn
           >50K than either feature alone would suggest.

        Original columns are kept; new columns are appended.
        """
        logging.info("Starting Feature Engineering...")

        for df in self._datasets:
            df['capital_net'] = df['capital_gain'] - df['capital_loss']
            df['hours_education_interaction'] = df['hours_per_week'] * df['education_num']

        logging.info(
            "Feature engineering complete. "
            "Added 'capital_net' and 'hours_education_interaction'."
        )

    def scale_features(self):
        """
        StandardScaler fitted on X_train only, then applied to all splits.
        Saved to models/scaler.pkl
        """
        logging.info("Starting Feature Scaling...")

        self.scaler = StandardScaler()
        cols = self.X_train.columns

        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=cols, index=self.X_train.index
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=cols, index=self.X_val.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=cols, index=self.X_test.index
        )

        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        logging.info("Feature scaling complete. Scaler saved to 'models/scaler.pkl'.")

    def exploratory_pca(self):
        """
        PCA run on the scaled X_train for exploratory insight only.
        Does NOT replace the feature matrix — X_train is unchanged after this step.

        Outputs:
          - plots/pca_scree_plot.png   : cumulative explained variance
          - plots/pca_2d_scatter.png   : 2D projection coloured by income class
        """
        logging.info("Starting Exploratory PCA...")

        os.makedirs('plots', exist_ok=True)

        pca = PCA(random_state=42)
        X_pca = pca.fit_transform(self.X_train)

        # Scree plot
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(cumulative_var) + 1), cumulative_var,
                 marker='o', linestyle='--', color='steelblue')
        plt.axhline(y=0.90, color='red', linestyle='-', linewidth=1)
        plt.text(1.2, 0.91, '90 % Variance Threshold', color='red', fontsize=10)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Scree Plot — Adult Census Income')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('plots/pca_scree_plot.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/pca_scree_plot.png'")

        # 2D scatter plot
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1],
            hue=self.y_train.values,
            palette={0: 'steelblue', 1: 'tomato'},
            alpha=0.4, s=15, linewidth=0
        )
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('2D PCA Projection — Coloured by Income Class')
        plt.legend(title='Income  (0: ≤50K  |  1: >50K)')
        plt.tight_layout()
        plt.savefig('plots/pca_2d_scatter.png', dpi=150)
        plt.close()
        logging.info("Saved 'plots/pca_2d_scatter.png'")

        # Loadings report
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.X_train.columns
        )

        n_components_90 = int(np.argmax(cumulative_var >= 0.90)) + 1
        logging.info(
            f"Components needed for 90 % variance: {n_components_90} "
            f"(out of {pca.n_components_})"
        )

        for pc in ['PC1', 'PC2']:
            top = loadings[pc].abs().sort_values(ascending=False).head(5)
            logging.info(f"Top 5 features for {pc}:\n{top.to_string()}")

    def run_pipeline(self):
        """
        Executes all preprocessing steps in the correct order:
          split → impute → outliers → encode → engineer → scale → PCA
        """
        self.split_data()
        self.impute_missing_values()
        self.treat_outliers()
        self.encode_categorical()
        self.engineer_features()
        self.scale_features()
        self.exploratory_pca()

        logging.info(
            f"Final feature matrix shape — "
            f"Train: {self.X_train.shape} | "
            f"Val: {self.X_val.shape} | "
            f"Test: {self.X_test.shape}"
        )

        # Save preprocessing artifacts for API
        joblib.dump(self.encoder, 'models/ohe_encoder.pkl')
        joblib.dump({
            'fill_values': self._fill_values,
            'iqr_bounds': self._iqr_bounds,
            'top_countries': self._top_countries,
            'label_encoders': self._label_encoders,
        }, 'models/preprocessing_meta.pkl')
        logging.info("Preprocessing artifacts saved for API.")

        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test
        )