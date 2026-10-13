from typing import Dict, Any

from app.rag.knowledge import (
    get_security_knowledge,
)


class ThreatIntelligenceRetriever:

    def retrieve(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:

        knowledge = (
            get_security_knowledge()
        )

        confidence = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        anomaly_strength = float(
            detection.get(
                "anomaly_strength",
                0.0,
            )
        )

        attack_probability = float(
            detection.get(
                "attack_probability",
                0.0,
            )
        )

        context_confidence = (
            confidence
            + anomaly_strength
            + attack_probability
        ) / 3.0

        if bool(
            detection.get(
                "unknown_anomaly",
                False,
            )
        ):
            context_type = (
                "ANOMALY_SECURITY_CONTEXT"
            )

        elif bool(
            detection.get(
                "known_attack",
                False,
            )
        ):
            context_type = (
                "KNOWN_ATTACK_SECURITY_CONTEXT"
            )

        else:
            context_type = (
                "NORMAL_TRAFFIC_SECURITY_CONTEXT"
            )

        return {
            "context_type":
                context_type,

            "confidence":
                float(
                    context_confidence
                ),

            "knowledge":
                knowledge,

            "source":
                "UNSW-NB15 security knowledge base",
        }


def main():

    retriever = (
        ThreatIntelligenceRetriever()
    )

    result = retriever.retrieve(
        {
            "confidence": 0.80,
            "attack_probability": 0.75,
            "anomaly_strength": 0.60,
            "unknown_anomaly": True,
            "known_attack": False,
        }
    )

    print("=" * 60)
    print("SECURITY CONTEXT RETRIEVER")
    print("=" * 60)

    print(
        result["context_type"]
    )

    print(
        f"Confidence: {result['confidence']:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )