from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

from app.ml.preprocessing.preprocess import (
    convert_binary_label,
    transform_with_fitted_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"

TRAIN_FILE = (
    DATA_DIR / "UNSW_NB15_training-set.csv"
)


def main():

    print("=" * 70)
    print("UNSW-NB15 ANOMALY MODEL TRAINING")
    print("=" * 70)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    preprocessor_path = (
        ARTIFACTS / "preprocessor.joblib"
    )

    feature_columns_path = (
        ARTIFACTS / "feature_columns.joblib"
    )

    if not preprocessor_path.exists():
        raise FileNotFoundError(
            "preprocessor.joblib not found. "
            "Run trainbaseline.py first."
        )

    if not feature_columns_path.exists():
        raise FileNotFoundError(
            "feature_columns.joblib not found. "
            "Run trainbaseline.py first."
        )

    df = pd.read_csv(
        TRAIN_FILE
    )

    preprocessor = joblib.load(
        preprocessor_path
    )

    feature_columns = joblib.load(
        feature_columns_path
    )

    labels = convert_binary_label(
        df["label"]
    )

    normal_df = df.loc[
        labels == 0
    ].copy()

    if normal_df.empty:
        raise ValueError(
            "No normal UNSW-NB15 traffic found."
        )

    max_rows = 30000

    if len(normal_df) > max_rows:

        rng = np.random.default_rng(
            42
        )

        indices = rng.choice(
            len(normal_df),
            size=max_rows,
            replace=False,
        )

        normal_df = normal_df.iloc[
            indices
        ].copy()

    X_normal = (
        transform_with_fitted_preprocessor(
            normal_df,
            preprocessor,
            feature_columns,
        )
    )

    print(
        f"Normal training rows : {len(X_normal)}"
    )

    print(
        f"Feature count         : {X_normal.shape[1]}"
    )

    isolation_forest = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining Isolation Forest..."
    )

    isolation_forest.fit(
        X_normal
    )

    isolation_scores = (
        isolation_forest.decision_function(
            X_normal
        )
    )

    isolation_threshold = float(
        np.percentile(
            isolation_scores,
            1.0,
        )
    )

    hidden_1 = max(
        8,
        min(
            64,
            X_normal.shape[1] // 2,
        ),
    )

    hidden_2 = max(
        4,
        min(
            32,
            X_normal.shape[1] // 4,
        ),
    )

    autoencoder = MLPRegressor(
        hidden_layer_sizes=(
            hidden_1,
            hidden_2,
            hidden_1,
        ),
        activation="relu",
        solver="adam",
        max_iter=80,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    print(
        "Training reconstruction model..."
    )

    autoencoder.fit(
        X_normal,
        X_normal,
    )

    reconstructed = (
        autoencoder.predict(
            X_normal
        )
    )

    reconstruction_errors = np.mean(
        np.square(
            X_normal - reconstructed
        ),
        axis=1,
    )

    reconstruction_threshold = float(
        np.percentile(
            reconstruction_errors,
            99.0,
        )
    )

    ARTIFACTS.mkdir(
        exist_ok=True
    )

    joblib.dump(
        isolation_forest,
        ARTIFACTS / "isolation_forest.joblib",
    )

    joblib.dump(
        autoencoder,
        ARTIFACTS / "autoencoder.joblib",
    )

    metadata = {
        "dataset": "UNSW-NB15",
        "training_rows": int(
            len(X_normal)
        ),
        "isolation_threshold": isolation_threshold,
        "reconstruction_threshold": reconstruction_threshold,
        "hidden_layer_sizes": (
            hidden_1,
            hidden_2,
            hidden_1,
        ),
    }

    joblib.dump(
        metadata,
        ARTIFACTS / "anomaly_metadata.joblib",
    )

    print("\nANOMALY MODEL RESULTS")
    print("-" * 70)

    print(
        f"Isolation threshold    : "
        f"{isolation_threshold:.6f}"
    )

    print(
        f"Reconstruction threshold: "
        f"{reconstruction_threshold:.6f}"
    )

    print(
        "\nAnomaly artifacts saved successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )