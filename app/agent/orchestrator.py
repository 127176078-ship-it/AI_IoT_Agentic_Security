from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


# ============================================================
# ORCHESTRATOR
# ============================================================

class Orchestrator:
    
    def __init__(self):
        self.detector = None
        self.triage_agent = None
        self.planner = None
        self.retriever = None
        self.risk_evaluator = None
        self.shap_analyzer = None
        self.ollama = None

        self._load_components()

    # ========================================================
    # COMPONENT LOADING
    # ========================================================

    def _load_components(self):
        """Load project components safely."""

        # ----------------------------------------------------
        # ML DETECTOR
        # ----------------------------------------------------

        try:
            from app.ml.detector import IntrusionDetector as IDSDetector

            self.detector = IDSDetector()

        except Exception as error:
            print(
                f"[WARNING] IDS detector could not be loaded: "
                f"{error}"
            )

        # ----------------------------------------------------
        # TRIAGE
        # ----------------------------------------------------

        try:
            from app.agent.triage import TriageAgent

            self.triage_agent = TriageAgent()

        except Exception:

            try:
                from app.agent.triage import triage_request

                self.triage_agent = triage_request

            except Exception as error:
                print(
                    f"[WARNING] Triage component unavailable: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # PLANNER
        # ----------------------------------------------------

        try:
            from app.agent.planner import Planner

            self.planner = Planner()

        except Exception:

            try:
                from app.agent.planner import create_plan

                self.planner = create_plan

            except Exception as error:
                print(
                    f"[WARNING] Planner component unavailable: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # RETRIEVER
        # ----------------------------------------------------

        try:
            from app.rag.retrieve import SecurityRetriever

            self.retriever = SecurityRetriever()

        except Exception:

            try:
                from app.rag.retrieve import retrieve_security_context

                self.retriever = retrieve_security_context

            except Exception as error:
                print(
                    f"[WARNING] Retrieval component unavailable: "
                    f"{error}"
                )

        # ----------------------------------------------------
        # RISK EVALUATOR
        # ----------------------------------------------------

        try:
            from app.agent.evaluator import RiskEvaluator

            self.risk_evaluator = RiskEvaluator()

        except Exception as error:
            print(
                f"[WARNING] Risk evaluator unavailable: "
                f"{error}"
            )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        try:
            from app.security.shap_analysis import SHAPAnalyzer

            self.shap_analyzer = SHAPAnalyzer()

        except Exception as error:
            print(
                f"[WARNING] SHAP analyzer unavailable: "
                f"{error}"
            )

        # ----------------------------------------------------
        # OLLAMA
        # ----------------------------------------------------

        try:
            from app.llm.ollama_client import OllamaClient

            self.ollama = OllamaClient()

        except Exception as error:
            print(
                f"[WARNING] Ollama unavailable: "
                f"{error}"
            )

    # ========================================================
    # DATA LOADING
    # ========================================================

    def _load_default_data(self) -> pd.DataFrame:
        """
        Load the UNSW-NB15 testing set when the API request
        does not provide explicit records.
        """

        test_file = DATA_DIR / "UNSW_NB15_testing-set.csv"

        if not test_file.exists():
            raise FileNotFoundError(
                f"UNSW-NB15 testing dataset not found: "
                f"{test_file}"
            )

        dataframe = pd.read_csv(test_file)

        if dataframe.empty:
            raise ValueError(
                "UNSW-NB15 testing dataset is empty."
            )

        return dataframe

    # ========================================================
    # DATA NORMALIZATION
    # ========================================================

    def _normalize_data(
        self,
        data: Any,
    ) -> pd.DataFrame:
        """
        Convert API input into a DataFrame.

        Supported input:
            None
            list[dict]
            dict
            pandas.DataFrame
        """

        if data is None:
            return self._load_default_data()

        if isinstance(data, pd.DataFrame):
            dataframe = data.copy()

        elif isinstance(data, list):
            dataframe = pd.DataFrame(data)

        elif isinstance(data, dict):

            # Single record
            dataframe = pd.DataFrame([data])

        else:
            raise TypeError(
                "data must be a list, dictionary, "
                "DataFrame, or None."
            )

        if dataframe.empty:
            raise ValueError(
                "No data records were supplied for analysis."
            )

        return dataframe

    # ========================================================
    # TRIAGE
    # ========================================================

    def _run_triage(
        self,
        query: str,
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:

        if self.triage_agent is None:
            return {
                "status": "TRIAGED",
                "request_type": "SECURITY_ANALYSIS",
                "requires_traffic_analysis": True,
            }

        try:

            if callable(self.triage_agent):

                result = self.triage_agent(query)

            elif hasattr(
                self.triage_agent,
                "triage",
            ):

                result = self.triage_agent.triage(query)

            else:

                result = {
                    "status": "TRIAGED",
                    "request_type": "SECURITY_ANALYSIS",
                }

            if isinstance(result, dict):
                return result

            return {
                "status": "TRIAGED",
                "request_type": "SECURITY_ANALYSIS",
                "raw_result": str(result),
            }

        except Exception as error:

            return {
                "status": "TRIAGE_FALLBACK",
                "request_type": "SECURITY_ANALYSIS",
                "error": str(error),
            }

    # ========================================================
    # PLANNING
    # ========================================================

    def _create_plan(
        self,
        triage: dict[str, Any],
    ) -> dict[str, Any]:

        if self.planner is None:
            return {
                "status": "PLANNED",
                "steps": [
                    "validate_input",
                    "run_ml_detection",
                    "retrieve_security_context",
                    "evaluate_risk",
                    "generate_explanation",
                    "verify_result",
                ],
            }

        try:

            if callable(self.planner):
                result = self.planner(triage)

            elif hasattr(
                self.planner,
                "create_plan",
            ):
                result = self.planner.create_plan(triage)

            else:
                result = {
                    "status": "PLANNED",
                    "steps": [],
                }

            if isinstance(result, dict):
                return result

            return {
                "status": "PLANNED",
                "steps": [],
                "raw_result": str(result),
            }

        except Exception as error:

            return {
                "status": "PLAN_FALLBACK",
                "steps": [],
                "error": str(error),
            }

    # ========================================================
    # ML DETECTION
    # ========================================================

    def _run_detection(
        self,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:

        if self.detector is None:
            return [
                {
                    "status": "DETECTION_UNAVAILABLE",
                    "detection_type": "UNKNOWN",
                    "confidence": 0.0,
                }
            ]

        try:

            # Preferred detector API
            if hasattr(
                self.detector,
                "predict",
            ):

                result = self.detector.predict(dataframe)

            elif hasattr(
                self.detector,
                "detect",
            ):

                result = self.detector.detect(dataframe)

            elif hasattr(
                self.detector,
                "analyze",
            ):

                result = self.detector.analyze(dataframe)

            else:
                raise AttributeError(
                    "Detector does not expose predict(), "
                    "detect(), or analyze()."
                )

            # DataFrame result
            if isinstance(result, pd.DataFrame):
                return result.to_dict(
                    orient="records"
                )

            # List result
            if isinstance(result, list):
                return result

            # Dictionary result
            if isinstance(result, dict):
                return [result]

            return [
                {
                    "raw_result": str(result),
                }
            ]

        except Exception as error:

            return [
                {
                    "status": "DETECTION_ERROR",
                    "detection_type": "UNKNOWN",
                    "confidence": 0.0,
                    "error": str(error),
                }
            ]

    # ========================================================
    # SECURITY CONTEXT
    # ========================================================

    def _retrieve_context(
        self,
        query: str,
        detection: dict[str, Any],
    ) -> dict[str, Any]:

        if self.retriever is None:
            return {
                "context_type": "LOCAL_SECURITY_KNOWLEDGE",
                "confidence": 0.0,
                "knowledge": (
                    "No retrieval component was available."
                ),
            }

        try:

            if callable(self.retriever):

                result = self.retriever(
                    query,
                    detection,
                )

            elif hasattr(
                self.retriever,
                "retrieve",
            ):

                result = self.retriever.retrieve(
                    query,
                    detection,
                )

            else:
                result = {
                    "context_type": "LOCAL_SECURITY_KNOWLEDGE",
                    "confidence": 0.0,
                }

            if isinstance(result, dict):
                return result

            return {
                "context_type": "LOCAL_SECURITY_KNOWLEDGE",
                "confidence": 0.0,
                "knowledge": str(result),
            }

        except Exception as error:

            return {
                "context_type": "RETRIEVAL_ERROR",
                "confidence": 0.0,
                "error": str(error),
            }

    # ========================================================
    # RISK EVALUATION
    # ========================================================

    def _evaluate_risk(
        self,
        detection: dict[str, Any],
        triage: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        if self.risk_evaluator is None:
            return {
                "severity": "UNKNOWN",
                "risk_score": 0.0,
                "zero_trust_decision": "VERIFY",
                "verification_score": 0.0,
                "evidence_count": 0,
            }

        try:

            result = self.risk_evaluator.evaluate(
                detection=detection,
                triage=triage,
                threat_context=context,
            )

            if isinstance(result, dict):
                return result

            return {
                "severity": "UNKNOWN",
                "risk_score": 0.0,
                "zero_trust_decision": "VERIFY",
                "raw_result": str(result),
            }

        except Exception as error:

            return {
                "severity": "UNKNOWN",
                "risk_score": 0.0,
                "zero_trust_decision": "VERIFY",
                "error": str(error),
            }

    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    def _explain(
        self,
        dataframe: pd.DataFrame,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:

        if self.shap_analyzer is None:
            return {
                "status": "SHAP_UNAVAILABLE",
                "influencing_features": [],
            }

        try:

            # Current SHAP module API
            if hasattr(
                self.shap_analyzer,
                "explain",
            ):

                result = self.shap_analyzer.explain(
                    dataframe
                )

            else:
                return {
                    "status": "SHAP_METHOD_UNAVAILABLE",
                    "influencing_features": [],
                }

            if isinstance(result, dict):

                result.setdefault(
                    "risk_score",
                    evaluation.get(
                        "risk_score",
                        0.0,
                    ),
                )

                result.setdefault(
                    "severity",
                    evaluation.get(
                        "severity",
                        "UNKNOWN",
                    ),
                )

                result.setdefault(
                    "zero_trust_decision",
                    evaluation.get(
                        "zero_trust_decision",
                        "VERIFY",
                    ),
                )

                return result

            return {
                "status": "SHAP_RESULT",
                "result": str(result),
            }

        except Exception as error:

            return {
                "status": "SHAP_ERROR",
                "influencing_features": [],
                "error": str(error),
            }

    # ========================================================
    # OLLAMA
    # ========================================================

    def _generate_llm_explanation(
        self,
        query: str,
        detection: dict[str, Any],
        evaluation: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:

        if self.ollama is None:
            return {
                "status": "OLLAMA_UNAVAILABLE",
                "response": "",
            }

        prompt = f"""
You are a cybersecurity analysis assistant.

Use ONLY the evidence supplied below.

User request:
{query}

ML detection:
{detection}

Risk evaluation:
{evaluation}

Security context:
{context}

Explain:
1. What the ML system detected.
2. Whether the record is normal, a known attack,
   or an unknown anomaly.
3. Why the Zero Trust decision was made.
4. Important uncertainty or limitations.

Do not invent attack names, vulnerabilities,
IoCs, or evidence that is not supplied.

Keep the response concise.
"""

        try:

            if hasattr(
                self.ollama,
                "safe_generate",
            ):

                response = self.ollama.safe_generate(
                    prompt
                )

            elif hasattr(
                self.ollama,
                "generate",
            ):

                response = self.ollama.generate(
                    prompt
                )

            else:
                return {
                    "status": "OLLAMA_METHOD_UNAVAILABLE",
                    "response": "",
                }

            if isinstance(response, dict):
                return {
                    "status": "GENERATED",
                    **response,
                }

            return {
                "status": "GENERATED",
                "response": str(response),
            }

        except Exception as error:

            return {
                "status": "OLLAMA_ERROR",
                "response": "",
                "error": str(error),
            }

    # ========================================================
    # PUBLIC RUN METHOD
    # ========================================================

    def run(
        self,
        query: str = "",
        data: Any = None,
    ) -> dict[str, Any]:
      
        query = (
            str(query).strip()
            if query is not None
            else ""
        )

        if not query:
            query = "Perform a security analysis."

        # ----------------------------------------------------
        # 1. DATA
        # ----------------------------------------------------

        dataframe = self._normalize_data(data)

        # ----------------------------------------------------
        # 2. TRIAGE
        # ----------------------------------------------------

        triage = self._run_triage(
            query=query,
            dataframe=dataframe,
        )

        # ----------------------------------------------------
        # 3. PLAN
        # ----------------------------------------------------

        plan = self._create_plan(
            triage
        )

        # ----------------------------------------------------
        # 4. ML DETECTION
        # ----------------------------------------------------

        detections = self._run_detection(
            dataframe
        )

        representative_detection = (
            detections[0]
            if detections
            else {
                "detection_type": "UNKNOWN",
                "confidence": 0.0,
            }
        )

        # ----------------------------------------------------
        # 5. SECURITY CONTEXT
        # ----------------------------------------------------

        threat_context = self._retrieve_context(
            query=query,
            detection=representative_detection,
        )

        # ----------------------------------------------------
        # 6. RISK + ZERO TRUST
        # ----------------------------------------------------

        evaluation = self._evaluate_risk(
            detection=representative_detection,
            triage=triage,
            context=threat_context,
        )

        # ----------------------------------------------------
        # 7. SHAP
        # ----------------------------------------------------

        explanation = self._explain(
            dataframe=dataframe,
            evaluation=evaluation,
        )

        # ----------------------------------------------------
        # 8. OLLAMA
        # ----------------------------------------------------

        llm_response = self._generate_llm_explanation(
            query=query,
            detection=representative_detection,
            evaluation=evaluation,
            context=threat_context,
        )

        # ----------------------------------------------------
        # 9. TRAFFIC SUMMARY
        # ----------------------------------------------------

        total_records = len(detections)

        known_attack_records = sum(
            1
            for item in detections
            if item.get(
                "known_attack",
                False,
            )
        )

        unknown_anomaly_records = sum(
            1
            for item in detections
            if item.get(
                "unknown_anomaly",
                False,
            )
        )

        normal_records = sum(
            1
            for item in detections
            if item.get(
                "detection_type",
                "",
            ) == "NORMAL"
        )

        attack_probabilities = [
            float(
                item.get(
                    "attack_probability",
                    0.0,
                )
                or 0.0
            )
            for item in detections
        ]

        anomaly_strengths = [
            float(
                item.get(
                    "anomaly_strength",
                    0.0,
                )
                or 0.0
            )
            for item in detections
        ]

        traffic_summary = {
            "total_records": total_records,
            "normal_records": normal_records,
            "known_attack_records": (
                known_attack_records
            ),
            "unknown_anomaly_records": (
                unknown_anomaly_records
            ),
            "max_attack_probability": (
                max(attack_probabilities)
                if attack_probabilities
                else 0.0
            ),
            "max_anomaly_strength": (
                max(anomaly_strengths)
                if anomaly_strengths
                else 0.0
            ),
        }

        # ----------------------------------------------------
        # 10. FINAL RESULT
        # ----------------------------------------------------

        return {
            "status": "success",
            "dataset": "UNSW-NB15",

            "query": query,

            "triage": triage,

            "plan": plan,

            "representative_detection": (
                representative_detection
            ),

            "traffic_summary": traffic_summary,

            "threat_context": threat_context,

            "evaluation": evaluation,

            "explanation": explanation,

            "llm_response": llm_response,

            "records_analyzed": len(dataframe),
        }

    # ========================================================
    # BACKWARD COMPATIBILITY
    # ========================================================

    def analyze(
        self,
        dataframe: Any,
        query: str = "Perform a security analysis.",
    ) -> dict[str, Any]:
        """
        Backward-compatible method for older project code.
        """

        return self.run(
            query=query,
            data=dataframe,
        )


# ============================================================
# MODULE-LEVEL HELPER
# ============================================================

def analyze(
    dataframe: Any,
    query: str = "Perform a security analysis.",
) -> dict[str, Any]:
    """
    Convenience function.
    """

    orchestrator = Orchestrator()

    return orchestrator.run(
        query=query,
        data=dataframe,
    )


# ============================================================
# DIRECT TEST
# ============================================================

def main():

    print("=" * 60)
    print("AI-IoT-Agentic-Security Orchestrator")
    print("=" * 60)

    try:

        orchestrator = Orchestrator()

        result = orchestrator.run(
            query="Perform a security analysis."
        )

        print()
        print("[SUCCESS] Orchestrator executed.")
        print(
            "Records analyzed:",
            result.get(
                "records_analyzed",
                0,
            ),
        )

        evaluation = result.get(
            "evaluation",
            {},
        )

        print(
            "Severity:",
            evaluation.get(
                "severity",
                "UNKNOWN",
            ),
        )

        print(
            "Zero Trust:",
            evaluation.get(
                "zero_trust_decision",
                "VERIFY",
            ),
        )

    except Exception as error:

        print()
        print(
            "[ERROR] Orchestrator test failed:"
        )
        print(error)


if __name__ == "__main__":
    main()