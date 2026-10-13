from typing import Dict, Any


class LLMEvidenceVerifier:

    def verify(
        self,
        llm_response: str,
        detection: Dict[str, Any],
        threat_context: Dict[str, Any],
    ) -> Dict[str, Any]:

        response = str(
            llm_response or ""
        ).strip()

        if not response:
            return {
                "verified": False,
                "evidence_score": 0.0,
                "reason":
                    "LLM response was empty.",
            }

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

        context_confidence = float(
            threat_context.get(
                "confidence",
                0.0,
            )
        )

        evidence_score = (
            attack_probability
            + anomaly_strength
            + context_confidence
        ) / 3.0

        evidence_score = max(
            0.0,
            min(
                1.0,
                evidence_score,
            ),
        )

        minimum_supported_evidence = (
            evidence_score >= 0.40
        )

        verified = (
            len(response) >= 20
            and minimum_supported_evidence
        )

        if verified:
            reason = (
                "LLM explanation is supported "
                "by available ML and security-context evidence."
            )
        else:
            reason = (
                "Insufficient evidence to independently "
                "support the generated explanation."
            )

        return {
            "verified": verified,
            "evidence_score":
                float(evidence_score),
            "reason": reason,
        }


def main():

    verifier = (
        LLMEvidenceVerifier()
    )

    result = verifier.verify(
        "The model detected suspicious traffic.",
        {
            "attack_probability": 0.80,
            "anomaly_strength": 0.60,
        },
        {
            "confidence": 0.70,
        },
    )

    print("=" * 60)
    print("LLM EVIDENCE VERIFIER")
    print("=" * 60)

    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )