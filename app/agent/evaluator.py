from typing import Dict, Any


class RiskEvaluator:

    def evaluate(
        self,
        detection: Dict[str, Any],
        triage: Dict[str, Any],
        threat_context: Dict[str, Any],
        verification: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        attack_probability = float(
            detection.get(
                "attack_probability",
                0.0,
            )
        )

        anomaly_strength = float(
            detection.get(
                "anomaly_strength",
                0.0,
            )
        )

        unknown_anomaly = bool(
            detection.get(
                "unknown_anomaly",
                False,
            )
        )

        context_confidence = float(
            threat_context.get(
                "confidence",
                0.0,
            )
        )

        verification_score = 1.0

        if verification is not None:
            verification_score = float(
                verification.get(
                    "evidence_score",
                    0.0,
                )
            )

        evidence_values = [
            attack_probability,
            anomaly_strength,
            context_confidence,
            verification_score,
        ]

        risk_score = sum(
            evidence_values
        ) / len(
            evidence_values
        )

        if unknown_anomaly:
            risk_score = max(
                risk_score,
                anomaly_strength,
            )

        risk_score = max(
            0.0,
            min(
                1.0,
                risk_score,
            ),
        )

        if risk_score >= 0.75:
            severity = "HIGH"
            zero_trust_decision = "DENY"

        elif risk_score >= 0.40:
            severity = "MEDIUM"
            zero_trust_decision = "VERIFY"

        else:
            severity = "LOW"
            zero_trust_decision = (
                "ALLOW_WITH_MONITORING"
            )

        if verification is not None:
            if not verification.get(
                "verified",
                False,
            ):
                zero_trust_decision = "VERIFY"

        return {
            "risk_score":
                float(risk_score),

            "severity":
                severity,

            "zero_trust_decision":
                zero_trust_decision,

            "evidence_count":
                len(evidence_values),

            "verification_score":
                verification_score,
        }


def main():

    evaluator = RiskEvaluator()

    result = evaluator.evaluate(
        {
            "attack_probability": 0.85,
            "anomaly_strength": 0.70,
            "unknown_anomaly": False,
        },
        {
            "request_type":
                "SECURITY_ANALYSIS"
        },
        {
            "confidence": 0.80
        },
    )

    print("=" * 70)
    print("RISK EVALUATOR")
    print("=" * 70)

    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )