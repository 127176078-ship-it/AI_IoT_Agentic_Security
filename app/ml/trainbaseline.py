
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

from app.ml.preprocessing.preprocess import (
    find_target_column,
    convert_binary_label,
    fit_preprocessor,
    transform_with_fitted_preprocessor,
    select_feature_columns,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

TOP_N_FEATURES = 15
CLASSIFICATION_THRESHOLD = 0.7
RANDOM_STATE = 42


# ============================================================
# MODEL
# ============================================================

def create_model():
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        scale_pos_weight=1.13,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# ENCODED FEATURE MAPPING
# ============================================================

def get_encoded_feature_mapping(preprocessor):
    """
    Map every transformed model input to its original feature.

    Supports OneHotEncoder and OrdinalEncoder.
    """

    mapping = []
    names = []

    for transformer_name, transformer, columns in (
        preprocessor.transformers_
    ):
        if transformer_name == "remainder":
            continue

        if transformer == "drop":
            continue

        if transformer_name == "numeric":
            for column in columns:
                mapping.append(column)
                names.append(f"numeric__{column}")

        elif transformer_name == "categorical":
            encoder = transformer.named_steps["encoder"]

            if isinstance(encoder, OneHotEncoder):
                encoded_names = encoder.get_feature_names_out(
                    columns
                )

                position = 0

                for column, categories in zip(
                    columns,
                    encoder.categories_,
                ):
                    for _ in categories:
                        mapping.append(column)
                        names.append(
                            f"categorical__{encoded_names[position]}"
                        )
                        position += 1

            elif isinstance(encoder, OrdinalEncoder):
                encoded_names = encoder.get_feature_names_out(
                    columns
                )

                for column, name in zip(
                    columns,
                    encoded_names,
                ):
                    mapping.append(column)
                    names.append(f"categorical__{name}")

            else:
                raise TypeError(
                    "Unsupported categorical encoder: "
                    f"{type(encoder).__name__}"
                )

    return mapping, names


# ============================================================
# FEATURE RANKING
# ============================================================

def rank_original_features(
    model,
    preprocessor,
    feature_columns,
):
    """Aggregate transformed-feature importance by original column."""

    mapping, _ = get_encoded_feature_mapping(
        preprocessor
    )

    importances = model.feature_importances_

    if len(mapping) != len(importances):
        raise RuntimeError(
            f"Feature mapping has {len(mapping)} inputs, "
            f"but the model has {len(importances)} importances."
        )

    totals = {
        column: 0.0
        for column in feature_columns
    }

    for original_column, importance in zip(
        mapping,
        importances,
    ):
        totals[original_column] += float(importance)

    ranking = sorted(
        totals.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if len(ranking) < TOP_N_FEATURES:
        raise RuntimeError(
            f"Only {len(ranking)} features available; "
            f"{TOP_N_FEATURES} are required."
        )

    return ranking


# ============================================================
# DATASET VALIDATION
# ============================================================

def load_datasets():
    """Load the official UNSW-NB15 train/test files."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}"
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training and testing datasets must not be empty."
        )

    # Standard official UNSW-NB15 split sizes.
    if len(train_df) == 175341 and len(test_df) == 82332:
        raise ValueError(
            "Dataset row counts appear reversed. Expected 82,332 "
            "training rows and 175,341 testing rows. Verify the "
            "contents of both CSV files before continuing."
        )

    if len(train_df) != 82332 or len(test_df) != 175341:
        print(
            "[WARNING] Row counts differ from the standard UNSW-NB15 "
            "split. Verify the dataset files and their provenance."
        )

    target_column = find_target_column(train_df)

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target '{target_column}' is missing from the test file."
        )

    # Prefer the binary label for binary intrusion detection.
    if "label" in train_df.columns and "label" in test_df.columns:
        target_column = "label"

    y_train = convert_binary_label(
        train_df[target_column]
    ).astype(int)

    y_test = convert_binary_label(
        test_df[target_column]
    ).astype(int)

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be binary 0/1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Testing labels must be binary 0/1.")

    return train_df, test_df, target_column, y_train, y_test


# ============================================================
# TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("UNSW-NB15 XGBOOST - TOP 15 ORIGINAL FEATURES")
    print("=" * 70)

    train_df, test_df, target_column, y_train, y_test = (
        load_datasets()
    )

    print(f"Training rows              : {len(train_df)}")
    print(f"Testing rows               : {len(test_df)}")

    print(
        f"Training attack rows       : {int(y_train.sum())}"
    )
    print(
        f"Testing attack rows        : {int(y_test.sum())}"
    )

    # --------------------------------------------------------
    # STAGE 1: Fit ranking preprocessor on training data only
    # --------------------------------------------------------

    all_features = select_feature_columns(
        train_df
    )

    ranking_preprocessor, all_features = fit_preprocessor(
        train_df,
        all_features,
    )

    X_train_all = transform_with_fitted_preprocessor(
        train_df,
        ranking_preprocessor,
        all_features,
    )

    print(
        f"Original features available: {len(all_features)}"
    )
    print(
        f"Initial encoded inputs     : {X_train_all.shape[1]}"
    )

    print("\nTraining feature-ranking model...")

    ranking_model = create_model()

    ranking_model.fit(
        X_train_all,
        y_train,
    )

    ranking = rank_original_features(
        ranking_model,
        ranking_preprocessor,
        all_features,
    )

    selected_features = [
        column
        for column, _ in ranking[:TOP_N_FEATURES]
    ]

    print("\nTOP ORIGINAL FEATURES")
    print("-" * 70)

    for index, (column, importance) in enumerate(
        ranking[:TOP_N_FEATURES],
        start=1,
    ):
        print(
            f"{index:2}. {column}: {importance:.6f}"
        )

    # --------------------------------------------------------
    # STAGE 2: Fit final preprocessor on selected features
    # --------------------------------------------------------

    preprocessor, selected_features = fit_preprocessor(
        train_df,
        selected_features,
    )

    X_train = transform_with_fitted_preprocessor(
        train_df,
        preprocessor,
        selected_features,
    )

    X_test = transform_with_fitted_preprocessor(
        test_df,
        preprocessor,
        selected_features,
    )

    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            "Training and testing feature dimensions do not match."
        )

    feature_mapping, feature_names = get_encoded_feature_mapping(
        preprocessor
    )

    if len(feature_names) != X_train.shape[1]:
        raise RuntimeError(
            "Feature-name count does not match transformed data."
        )

    # --------------------------------------------------------
    # STAGE 3: Train final model
    # --------------------------------------------------------

    model = create_model()

    print("\nFINAL MODEL")
    print("-" * 70)
    print(
        f"Selected original features: {len(selected_features)}"
    )
    print(
        f"Encoded model inputs      : {X_train.shape[1]}"
    )
    print(
        f"Classification threshold  : {CLASSIFICATION_THRESHOLD}"
    )

    print("\nTraining final XGBoost model...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # STAGE 4: Evaluate on held-out test data
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print("\nMODEL RESULTS")
    print("-" * 70)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")

    print(
        "\nConfusion Matrix (rows=true, columns=predicted):"
    )
    print(matrix)

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # STAGE 5: Save artifacts and metrics
    # --------------------------------------------------------

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        ARTIFACTS / "xgboost_model.joblib",
    )

    joblib.dump(
        preprocessor,
        ARTIFACTS / "preprocessor.joblib",
    )

    joblib.dump(
        selected_features,
        ARTIFACTS / "feature_columns.joblib",
    )

    joblib.dump(
        feature_names,
        ARTIFACTS / "feature_names.joblib",
    )

    joblib.dump(
        ranking,
        ARTIFACTS / "feature_importance_ranking.joblib",
    )

    metadata = {
        "dataset": "UNSW-NB15",
        "target_column": target_column,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "selection_method": (
            "XGBoost importance aggregated by original feature"
        ),
        "selected_original_feature_count": len(
            selected_features
        ),
        "selected_original_features": selected_features,
        "transformed_feature_count": int(X_train.shape[1]),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "encoder": type(
            preprocessor.named_transformers_["categorical"]
            .named_steps["encoder"]
        ).__name__ if "categorical" in preprocessor.named_transformers_
        else "none",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        metadata,
        ARTIFACTS / "baseline_metadata.joblib",
    )

    metrics = {
        **metadata,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
            output_dict=True,
        ),
    }

    with open(
        REPORTS / "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\nArtifacts saved:")
    print(f"  Model   : {ARTIFACTS / 'xgboost_model.joblib'}")
    print(f"  Pipeline: {ARTIFACTS / 'preprocessor.joblib'}")
    print(f"  Metrics : {REPORTS / 'baseline_metrics.json'}")
    print("\nTraining and evaluation completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())