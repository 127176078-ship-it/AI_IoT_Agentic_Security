MITRE_MAP = {
    "reconnaissance": ["T1595"],
    "exploits": ["T1190"],
    "fuzzers": ["T1190"],
    "backdoor": ["T1059"],
    "shellcode": ["T1055"],
    "worms": ["T1210"],
    "dos": [],
    "generic": [],
    "normal": [],
    "benign": [],
}


def map_attack(
    attack_category: str,
):

    key = str(
        attack_category or ""
    ).strip().lower()

    techniques = MITRE_MAP.get(
        key
    )

    if techniques is None:

        techniques = [
            "UNKNOWN"
        ]

        confidence = "UNKNOWN"

    elif techniques:

        confidence = "LOW"

    else:

        confidence = "NOT_MAPPED"

    return {
        "attack_category":
            attack_category,

        "mitre_techniques":
            techniques,

        "confidence":
            confidence,

        "source":
            "conservative UNSW-NB15-to-MITRE mapping",
    }


def main():

    print("=" * 60)
    print("MITRE MAPPING")
    print("=" * 60)

    print(
        map_attack(
            "reconnaissance"
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )