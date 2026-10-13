from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from app.ml.detector import (
    IntrusionDetector,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORTS = PROJECT_ROOT / "reports"

TEST_FILE = (
    DATA_DIR / "UNSW_NB15_testing-set.csv"
)


def evaluate():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_FILE}"
        )

    detector = IntrusionDetector()

    df = pd.read_csv(
        TEST_FILE
    )

    if "label" not in df.columns:
        raise ValueError(
            "UNSW-NB15 testing file must contain 'label'."
        )

    y_true = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    if y_true.isna().any():
        raise ValueError(
            "Testing labels contain non-numeric values."
        )

    y_true = y_true.astype(int)

    result = detector.predict(
        df
    )

    y_pred = (
        result["known_attack"]
        .astype(int)
        .to_numpy()
    )

    probability = (
        result["attack_probability"]
        .to_numpy()
    )

    metrics = {
        "dataset": "UNSW-NB15",

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),

        "classification_report":
            classification_report(
                y_true,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),

        "detection_summary":
            result[
                "detection_type"
            ].value_counts().to_dict(),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
    except ValueError:
        metrics["roc_auc"] = None

    ARTIFACTS.mkdir(
        exist_ok=True
    )

    REPORTS.mkdir(
        exist_ok=True
    )

    joblib.dump(
        metrics,
        ARTIFACTS
        / "evaluation_metrics.joblib",
    )

    with open(
        REPORTS / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    return metrics


def main():

    print("=" * 70)
    print("AI-IoT-AGENTIC-SECURITY EVALUATION")
    print("=" * 70)

    try:

        metrics = evaluate()

        print(
            f"Accuracy  : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall    : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score  : {metrics['f1']:.4f}"
        )

        if metrics["roc_auc"] is not None:
            print(
                f"ROC-AUC   : {metrics['roc_auc']:.4f}"
            )

        print(
            "\nEvaluation completed successfully."
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