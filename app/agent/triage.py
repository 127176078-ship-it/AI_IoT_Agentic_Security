from typing import Dict, Any


class TriageAgent:

    def triage(
        self,
        user_request: str,
        has_traffic_data: bool = False,
    ) -> Dict[str, Any]:

        request = str(
            user_request or ""
        ).strip()

        if not request:
            raise ValueError(
                "User request cannot be empty."
            )

        request_length = len(
            request
        )

        if has_traffic_data:
            request_type = (
                "SECURITY_ANALYSIS"
            )
        elif request_length > 120:
            request_type = (
                "COMPLEX_SECURITY_QUERY"
            )
        else:
            request_type = (
                "DIRECT_SECURITY_QUERY"
            )

        return {
            "request_type": request_type,
            "requires_traffic_analysis":
                bool(has_traffic_data),
            "request_length":
                request_length,
            "status":
                "TRIAGED",
        }


def main():

    agent = TriageAgent()

    result = agent.triage(
        "Analyze the supplied IoT traffic for security threats.",
        has_traffic_data=True,
    )

    print("=" * 70)
    print("TRIAGE AGENT")
    print("=" * 70)

    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )