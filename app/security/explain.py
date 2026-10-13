from pathlib import Path
import uuid

import matplotlib.pyplot as plt

from app.security.shap_analysis import (
    SHAPAnalyzer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS = (
    PROJECT_ROOT / "reports"
)

REPORTS.mkdir(
    exist_ok=True
)


class SecurityExplainer:

    def __init__(self):

        self.shap_analyzer = (
            SHAPAnalyzer()
        )

    def explain(
        self,
        dataframe,
        evaluation,
    ):

        features = (
            self.shap_analyzer
            .explain_dataframe(
                dataframe,
                top_k=10,
            )
        )

        file_name = (
            "security_explanation_"
            f"{uuid.uuid4().hex[:8]}.png"
        )

        image_path = (
            REPORTS / file_name
        )

        names = [
            item["feature"]
            for item in features
        ]

        impacts = [
            item["impact"]
            for item in features
        ]

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            names[::-1],
            impacts[::-1],
        )

        plt.xlabel(
            "SHAP impact on attack score"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "ML Feature Influence"
        )

        plt.tight_layout()

        plt.savefig(
            image_path,
            dpi=150,
        )

        plt.close()

        return {
            "zero_trust_decision":
                evaluation[
                    "zero_trust_decision"
                ],

            "risk_score":
                evaluation[
                    "risk_score"
                ],

            "severity":
                evaluation[
                    "severity"
                ],

            "influencing_features":
                features,

            "visualization":
                str(image_path),
        }


def main():

    print("=" * 60)
    print("SECURITY EXPLANATION MODULE")
    print("=" * 60)

    print(
        "SHAP explanation module ready."
    )

    print(
        "A trained model is required for explanation."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )