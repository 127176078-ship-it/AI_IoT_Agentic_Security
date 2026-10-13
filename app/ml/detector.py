from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = PROJECT_ROOT / "artifacts"


class IntrusionDetector:

    def __init__(self):

        required = [
            "xgboost_model.joblib",
            "preprocessor.joblib",
            "feature_columns.joblib",
            "feature_names.joblib",
            "isolation_forest.joblib",
            "autoencoder.joblib",
            "anomaly_metadata.joblib",
        ]

        missing = [
            name
            for name in required
            if not (
                ARTIFACTS / name
            ).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Missing model artifacts: "
                + ", ".join(missing)
            )

        self.model = joblib.load(
            ARTIFACTS
            / "xgboost_model.joblib"
        )

        self.preprocessor = joblib.load(
            ARTIFACTS
            / "preprocessor.joblib"
        )

        self.feature_columns = joblib.load(
            ARTIFACTS
            / "feature_columns.joblib"
        )

        self.feature_names = joblib.load(
            ARTIFACTS
            / "feature_names.joblib"
        )

        self.isolation_forest = joblib.load(
            ARTIFACTS
            / "isolation_forest.joblib"
        )

        self.autoencoder = joblib.load(
            ARTIFACTS
            / "autoencoder.joblib"
        )

        self.anomaly_metadata = joblib.load(
            ARTIFACTS
            / "anomaly_metadata.joblib"
        )

        self.classification_threshold = (
            self._load_classification_threshold()
        )

    def _load_classification_threshold(self):

        metadata_path = (
            ARTIFACTS
            / "baseline_metadata.joblib"
        )

        if metadata_path.exists():

            metadata = joblib.load(
                metadata_path
            )

            return float(
                metadata.get(
                    "classification_threshold",
                    0.5,
                )
            )

        return 0.5

    def _prepare(self, dataframe):

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "Detector input must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Detector received an empty DataFrame."
            )

        X = dataframe.drop(
            columns=[
                "label",
                "attack_cat",
            ],
            errors="ignore",
        ).copy()

        for column in self.feature_columns:

            if column not in X.columns:
                X[column] = np.nan

        X = X[
            self.feature_columns
        ]

        transformed = (
            self.preprocessor.transform(
                X
            )
        )

        return np.asarray(
            transformed,
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_anomaly_evidence(
        isolation_score,
        isolation_threshold,
        reconstruction_error,
        reconstruction_threshold,
    ):

        if isolation_score < isolation_threshold:
            if isolation_threshold != 0:
                isolation_strength = min(
                    1.0,
                    abs(
                        isolation_score
                        / isolation_threshold
                    ),
                )
            else:
                isolation_strength = 1.0
        else:
            isolation_strength = 0.0

        if reconstruction_error > reconstruction_threshold:
            if reconstruction_threshold != 0:
                reconstruction_strength = min(
                    1.0,
                    reconstruction_error
                    / reconstruction_threshold
                    - 1.0,
                )
            else:
                reconstruction_strength = 1.0
        else:
            reconstruction_strength = 0.0

        return max(
            isolation_strength,
            reconstruction_strength,
        )

    def predict(self, dataframe):

        X = self._prepare(
            dataframe
        )

        attack_probability = (
            self.model.predict_proba(
                X
            )[:, 1]
        )

        known_attack = (
            attack_probability
            >= self.classification_threshold
        )

        isolation_scores = (
            self.isolation_forest
            .decision_function(X)
        )

        isolation_threshold = float(
            self.anomaly_metadata[
                "isolation_threshold"
            ]
        )

        isolation_anomaly = (
            isolation_scores
            < isolation_threshold
        )

        reconstructed = (
            self.autoencoder.predict(
                X
            )
        )

        reconstruction_errors = np.mean(
            np.square(
                X - reconstructed
            ),
            axis=1,
        )

        reconstruction_threshold = float(
            self.anomaly_metadata[
                "reconstruction_threshold"
            ]
        )

        reconstruction_anomaly = (
            reconstruction_errors
            > reconstruction_threshold
        )

        unknown_anomaly = (
            ~known_attack
            & (
                isolation_anomaly
                | reconstruction_anomaly
            )
        )

        anomaly_strength = []

        for i in range(len(X)):

            anomaly_strength.append(
                self._normalize_anomaly_evidence(
                    float(
                        isolation_scores[i]
                    ),
                    isolation_threshold,
                    float(
                        reconstruction_errors[i]
                    ),
                    reconstruction_threshold,
                )
            )

        anomaly_strength = np.asarray(
            anomaly_strength,
            dtype=float,
        )

        confidence = np.maximum(
            attack_probability,
            anomaly_strength,
        )

        detection_type = np.where(
            known_attack,
            "KNOWN_ATTACK",
            np.where(
                unknown_anomaly,
                "UNKNOWN_ANOMALY",
                "NORMAL",
            ),
        )

        result = pd.DataFrame(
            {
                "attack_probability":
                    attack_probability.astype(float),

                "known_attack":
                    known_attack.astype(bool),

                "isolation_score":
                    isolation_scores.astype(float),

                "isolation_anomaly":
                    isolation_anomaly.astype(bool),

                "reconstruction_error":
                    reconstruction_errors.astype(float),

                "reconstruction_anomaly":
                    reconstruction_anomaly.astype(bool),

                "unknown_anomaly":
                    unknown_anomaly.astype(bool),

                "anomaly_strength":
                    anomaly_strength,

                "confidence":
                    confidence,

                "detection_type":
                    detection_type,
            }
        )

        return result

    def predict_file(self, file_path):

        dataframe = pd.read_csv(
            file_path
        )

        return self.predict(
            dataframe
        )


def main():

    testing_file = (
        PROJECT_ROOT
        / "data"
        / "UNSW_NB15_testing-set.csv"
    )

    reports = (
        PROJECT_ROOT
        / "reports"
    )

    reports.mkdir(
        exist_ok=True
    )

    print("=" * 70)
    print("AI-IoT-Agentic-Security Detector")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Artifacts    : {ARTIFACTS}"
    )

    try:

        detector = IntrusionDetector()

        results = (
            detector.predict_file(
                testing_file
            )
        )

        print(
            "\nModels loaded successfully."
        )

        print(
            "\nDetection summary:"
        )

        print(
            results[
                "detection_type"
            ].value_counts()
        )

        print(
            "\nFirst 5 detections:"
        )

        print(
            results.head()
            .to_string(
                index=False
            )
        )

        output = (
            reports
            / "detector_results.csv"
        )

        results.to_csv(
            output,
            index=False,
        )

        print(
            f"\nResults saved: {output}"
        )

        return 0

    except Exception as exc:

        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )