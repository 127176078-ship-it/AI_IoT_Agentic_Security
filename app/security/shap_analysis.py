from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"


# ============================================================
# SHAP EXPLAINER
# ============================================================

class SHAPExplainer:
    """
    Explain predictions produced by the trained XGBoost model.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.explainer = None

        self._load_artifacts()

    # --------------------------------------------------------
    # LOAD MODEL ARTIFACTS
    # --------------------------------------------------------

    def _load_artifacts(self) -> None:
        """Load trained model, preprocessing pipeline and feature names."""

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"XGBoost model not found:\n{MODEL_PATH}"
            )

        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)

        if FEATURE_NAMES_PATH.exists():
            self.feature_names = joblib.load(FEATURE_NAMES_PATH)

        # TreeExplainer is appropriate for XGBoost.
        self.explainer = shap.TreeExplainer(self.model)

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        
        if dataframe is None:
            raise ValueError("Input dataframe cannot be None.")

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if dataframe.empty:
            raise ValueError("Input dataframe is empty.")

        df = dataframe.copy()

        # These are target columns and must not be used as input features.
        for column in ["label", "attack_cat"]:
            if column in df.columns:
                df = df.drop(columns=[column])

        return df

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    def _transform(self, dataframe: pd.DataFrame):
        """Apply the same preprocessing pipeline used during training."""

        df = self._prepare_dataframe(dataframe)

        transformed = self.preprocessor.transform(df)

        # Convert sparse matrix to dense matrix if necessary.
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        return transformed

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    def _get_feature_names(self, transformed_data) -> list[str]:
        
        if self.feature_names is not None:
            try:
                names = list(self.feature_names)

                if len(names) == transformed_data.shape[1]:
                    return [str(name) for name in names]
            except Exception:
                pass

        # Try sklearn's feature-name API.
        try:
            names = self.preprocessor.get_feature_names_out()

            if len(names) == transformed_data.shape[1]:
                return [str(name) for name in names]
        except Exception:
            pass

        # Final safe fallback.
        return [
            f"feature_{index}"
            for index in range(transformed_data.shape[1])
        ]

    # --------------------------------------------------------
    # NORMALIZE SHAP OUTPUT
    # --------------------------------------------------------

    def _extract_shap_values(self, transformed_data):
        
        shap_result = self.explainer(transformed_data)

        values = shap_result.values

        values = np.asarray(values)

        # Normal binary classification output:
        # (samples, features)
        if values.ndim == 2:
            return values

        # Some versions may return:
        # (samples, features, classes)
        if values.ndim == 3:
            # Use the attack/positive class.
            if values.shape[2] >= 2:
                return values[:, :, 1]

            return values[:, :, 0]

        raise ValueError(
            f"Unsupported SHAP output shape: {values.shape}"
        )

    # --------------------------------------------------------
    # EXPLAIN
    # --------------------------------------------------------

    def explain(self, dataframe: pd.DataFrame) -> dict[str, Any]:
    
        transformed = self._transform(dataframe)

        if transformed.shape[0] == 0:
            raise ValueError("No rows available for SHAP analysis.")

        feature_names = self._get_feature_names(transformed)

        shap_values = self._extract_shap_values(transformed)

        # Explain the first supplied record.
        row_values = shap_values[0]

        # Prediction probability from XGBoost.
        try:
            probabilities = self.model.predict_proba(transformed)

            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                attack_probability = float(probabilities[0, 1])
            else:
                attack_probability = float(probabilities[0])

        except Exception:
            prediction = self.model.predict(transformed)

            attack_probability = float(prediction[0])

        # Pair feature names with SHAP values.
        feature_impacts = []

        for index, shap_value in enumerate(row_values):

            feature_name = (
                feature_names[index]
                if index < len(feature_names)
                else f"feature_{index}"
            )

            try:
                value = float(shap_value)
            except (TypeError, ValueError):
                value = 0.0

            feature_impacts.append(
                {
                    "feature": str(feature_name),
                    "shap_value": value,
                    "absolute_impact": abs(value),
                    "direction": (
                        "increases_attack_prediction"
                        if value > 0
                        else "decreases_attack_prediction"
                        if value < 0
                        else "neutral"
                    ),
                }
            )

        # Most influential features first.
        feature_impacts.sort(
            key=lambda item: item["absolute_impact"],
            reverse=True,
        )

        # Keep top 10 for the report/visualization.
        top_features = feature_impacts[:10]

        # Model prediction.
        try:
            prediction_value = int(
                self.model.predict(transformed)[0]
            )
        except Exception:
            prediction_value = None

        return {
            "prediction": prediction_value,
            "attack_probability": attack_probability,
            "influencing_features": top_features,
            "total_features_analyzed": len(feature_impacts),
        }


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

Explainer = SHAPExplainer
SHAPAnalyzer = SHAPExplainer


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():
    print("=" * 60)
    print("SHAP EXPLAINABILITY TEST")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Preprocessor : {PREPROCESSOR_PATH}")

    try:
        explainer = SHAPExplainer()

        print()
        print("[OK] XGBoost model loaded.")
        print("[OK] Preprocessor loaded.")
        print("[OK] SHAP TreeExplainer initialized.")
        print()
        print("[SUCCESS] SHAP explainability module is ready.")

    except Exception as exc:
        print()
        print("[ERROR] SHAP initialization failed.")
        print()
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()