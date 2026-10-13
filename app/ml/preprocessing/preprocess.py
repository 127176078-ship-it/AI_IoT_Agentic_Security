
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET_CANDIDATES = (
    "label",
    "attack_cat",
)


# ============================================================
# OUTLIER CLIPPING
# ============================================================

class OutlierClipper:
    """Clip numeric features using bounds learned from training data."""

    def __init__(self, lower=0.005, upper=0.995):
        if not 0 <= lower < upper <= 1:
            raise ValueError(
                "Bounds must satisfy 0 <= lower < upper <= 1."
            )

        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.lower_bounds_ = X.quantile(
            self.lower,
            numeric_only=True,
        )

        self.upper_bounds_ = X.quantile(
            self.upper,
            numeric_only=True,
        )

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if self.lower_bounds_ is None:
            raise RuntimeError(
                "OutlierClipper must be fitted before transform."
            )

        for column in self.lower_bounds_.index:
            if column in X.columns:
                X[column] = X[column].clip(
                    lower=self.lower_bounds_[column],
                    upper=self.upper_bounds_[column],
                )

        return X


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df):
    """Validate that the supplied dataset is a usable DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    if len(df.columns) == 0:
        raise ValueError(
            "Input DataFrame has no columns."
        )

    if not df.columns.is_unique:
        raise ValueError(
            "Input DataFrame contains duplicate column names."
        )

    return True


def find_target_column(df):
    """Find the available UNSW-NB15 target column."""

    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "UNSW-NB15 target column was not found. "
        "Expected one of: label, attack_cat."
    )


def remove_target_columns(df):
    """Remove target labels from model input features."""

    return df.drop(
        columns=list(TARGET_CANDIDATES),
        errors="ignore",
    ).copy()


# ============================================================
# FEATURE SELECTION
# ============================================================

def identify_id_like_columns(df):
    """Identify identifier and address fields unsuitable as features."""

    drop_columns = []

    exact_id_names = {
        "id",
        "id_",
        "_id",
    }

    address_columns = {
        "srcip",
        "dstip",
        "source_ip",
        "destination_ip",
        "timestamp",
    }

    for column in df.columns:
        normalized = str(column).strip().lower()

        if (
            normalized in exact_id_names
            or normalized in address_columns
        ):
            drop_columns.append(column)

    return sorted(
        set(drop_columns),
        key=lambda value: str(value),
    )


def identify_low_information_columns(df):
    """Identify columns with excessive missing values or no variation."""

    drop_columns = []

    for column in df.columns:
        missing_ratio = float(
            df[column].isna().mean()
        )

        unique_count = int(
            df[column].nunique(dropna=False)
        )

        if missing_ratio > 0.50:
            drop_columns.append(column)

        elif unique_count <= 1:
            drop_columns.append(column)

    return sorted(
        set(drop_columns),
        key=lambda value: str(value),
    )


def select_feature_columns(df):
    """Select features using the supplied training DataFrame."""

    validate_dataframe(df)

    work = remove_target_columns(df)

    id_columns = identify_id_like_columns(work)

    work = work.drop(
        columns=id_columns,
        errors="ignore",
    )

    low_information_columns = identify_low_information_columns(
        work
    )

    work = work.drop(
        columns=low_information_columns,
        errors="ignore",
    )

    if work.empty:
        raise ValueError(
            "No usable UNSW-NB15 features remain "
            "after feature selection."
        )

    return list(work.columns)


# ============================================================
# PREPROCESSOR CONSTRUCTION
# ============================================================

def create_one_hot_encoder():
    """Support current and older scikit-learn versions."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def fit_preprocessor(df, feature_columns=None):
    """
    Fit preprocessing on training data only.

    Returns:
        fitted ColumnTransformer,
        selected original feature columns
    """

    validate_dataframe(df)

    if feature_columns is None:
        feature_columns = select_feature_columns(df)
    else:
        feature_columns = list(feature_columns)

    if not feature_columns:
        raise ValueError(
            "Feature column list cannot be empty."
        )

    X = remove_target_columns(df)

    for column in feature_columns:
        if column not in X.columns:
            X[column] = np.nan

    X = X[feature_columns].copy()

    numeric_columns = list(
        X.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "clipper",
                    OutlierClipper(),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    create_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    preprocessor.fit(X)

    return preprocessor, feature_columns


# ============================================================
# TRANSFORMATION
# ============================================================

def prepare_dataframe(df, feature_columns):
    """Align a DataFrame to the fitted feature schema."""

    validate_dataframe(df)

    work = remove_target_columns(df)

    for column in feature_columns:
        if column not in work.columns:
            work[column] = np.nan

    return work[feature_columns].copy()


def transform_with_fitted_preprocessor(
    df,
    preprocessor,
    feature_columns,
):
    """Transform data using an already-fitted preprocessor."""

    X = prepare_dataframe(
        df,
        feature_columns,
    )

    transformed = preprocessor.transform(X)

    transformed = np.asarray(
        transformed,
        dtype=np.float32,
    )

    if not np.isfinite(transformed).all():
        raise ValueError(
            "Preprocessing produced NaN or infinite values."
        )

    return transformed


def get_feature_names(preprocessor):
    """Return transformed feature names when supported."""

    try:
        return list(
            preprocessor.get_feature_names_out()
        )
    except (AttributeError, ValueError, TypeError):
        names = []

        for transformer_name, transformer, columns in (
            preprocessor.transformers_
        ):
            if transformer_name == "remainder":
                continue

            if transformer == "drop":
                continue

            if hasattr(
                transformer,
                "get_feature_names_out",
            ):
                try:
                    generated = transformer.get_feature_names_out(
                        columns
                    )

                    names.extend(
                        list(generated)
                    )
                    continue

                except (AttributeError, ValueError, TypeError):
                    pass

            names.extend(list(columns))

        return names


# ============================================================
# BINARY LABEL CONVERSION
# ============================================================

def convert_binary_label(series):
    """Convert UNSW-NB15 binary labels into 0 = normal, 1 = attack."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        values = numeric.astype(int)

        if set(values.unique()).issubset({0, 1}):
            return values

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    result = normalized.map(
        {
            "normal": 0,
            "benign": 0,
            "attack": 1,
        }
    )

    if result.isna().any():
        invalid = sorted(
            normalized[result.isna()].unique().tolist()
        )

        raise ValueError(
            "Unable to convert UNSW-NB15 labels to binary values. "
            f"Unexpected labels: {invalid[:10]}"
        )

    return result.astype(int)


# ============================================================
# PREPROCESSING TEST
# ============================================================

def main():
    print("=" * 70)
    print("UNSW-NB15 PREPROCESSING TEST")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        print(
            f"[ERROR] Training file not found:\n{TRAIN_FILE}"
        )
        return 1

    if not TEST_FILE.exists():
        print(
            f"[ERROR] Testing file not found:\n{TEST_FILE}"
        )
        return 1

    try:
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)

        validate_dataframe(train_df)
        validate_dataframe(test_df)

        print(f"Training rows      : {len(train_df)}")
        print(f"Testing rows       : {len(test_df)}")

        target = find_target_column(train_df)

        if target not in test_df.columns:
            raise ValueError(
                f"Target column '{target}' is missing "
                "from the test dataset."
            )

        # Check that both splits contain binary labels.
        train_labels = convert_binary_label(
            train_df["label"]
            if "label" in train_df.columns
            else train_df[target]
        )

        test_labels = convert_binary_label(
            test_df["label"]
            if "label" in test_df.columns
            else test_df[target]
        )

        # Fit feature selection and preprocessing on training data only.
        feature_columns = select_feature_columns(
            train_df
        )

        preprocessor, feature_columns = fit_preprocessor(
            train_df,
            feature_columns,
        )

        # Transform samples from both splits with the same fitted object.
        train_sample = transform_with_fitted_preprocessor(
            train_df.head(5),
            preprocessor,
            feature_columns,
        )

        test_sample = transform_with_fitted_preprocessor(
            test_df.head(5),
            preprocessor,
            feature_columns,
        )

        transformed_feature_names = get_feature_names(
            preprocessor
        )

        if train_sample.shape[1] != test_sample.shape[1]:
            raise ValueError(
                "Training and testing transformed feature counts differ."
            )

        if (
            len(transformed_feature_names)
            != train_sample.shape[1]
        ):
            print(
                "[WARNING] Transformed feature names do not match "
                "the transformed matrix width."
            )

        print(f"Target column      : {target}")
        print(f"Original features   : {len(feature_columns)}")
        print(
            "Encoded features   : "
            f"{train_sample.shape[1]}"
        )
        print(f"Training sample    : {train_sample.shape}")
        print(f"Testing sample     : {test_sample.shape}")
        print(
            "Training labels    : "
            f"{len(train_labels)} rows"
        )
        print(
            "Testing labels     : "
            f"{len(test_labels)} rows"
        )
        print("Preprocessing      : SUCCESS")

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
    ) as exc:
        print(f"[ERROR] Preprocessing failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())