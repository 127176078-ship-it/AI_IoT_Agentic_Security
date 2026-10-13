KNOWLEDGE = {
    "zero_trust": {
        "principle":
            "Every access decision should be continuously "
            "verified using available evidence rather than "
            "implicitly trusted."
    },

    "anomaly_detection": {
        "principle":
            "Anomaly detection identifies observations that "
            "differ from learned normal traffic behaviour."
    },

    "model_uncertainty": {
        "principle":
            "Security decisions should consider model confidence "
            "and independent anomaly evidence."
    },

    "iot_security": {
        "principle":
            "IoT traffic should be continuously monitored because "
            "connected devices can expose changing network behaviour."
    },

    "evidence_verification": {
        "principle":
            "Security explanations should be grounded in observed "
            "machine-learning evidence and retrieved context."
    },
}


def get_security_knowledge():
    return KNOWLEDGE


def main():

    print("=" * 60)
    print("SECURITY KNOWLEDGE")
    print("=" * 60)

    for category, content in KNOWLEDGE.items():

        print(
            f"\n{category}:"
        )

        print(
            content["principle"]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )