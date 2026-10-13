from typing import Dict, Any, List


class PlannerAgent:

    def create_plan(
        self,
        triage: Dict[str, Any],
        has_traffic_data: bool,
        live_capture_requested: bool = False,
    ) -> Dict[str, Any]:

        request_type = triage.get(
            "request_type"
        )

        steps: List[str] = []

        if has_traffic_data:

            steps.extend(
                [
                    "validate_traffic_schema",
                    "run_intrusion_detection",
                    "assess_anomaly_evidence",
                    "retrieve_security_context",
                    "generate_feature_explanation",
                    "map_mitre_context",
                    "calculate_risk",
                    "apply_zero_trust_policy",
                    "generate_llm_explanation",
                    "verify_llm_evidence",
                    "generate_final_response",
                ]
            )

        elif live_capture_requested:

            steps.extend(
                [
                    "capture_live_packets",
                    "validate_capture_result",
                    "retrieve_security_context",
                    "generate_final_response",
                ]
            )

        elif request_type == "DIRECT_SECURITY_QUERY":

            steps.extend(
                [
                    "retrieve_security_context",
                    "generate_final_response",
                ]
            )

        else:

            steps.extend(
                [
                    "retrieve_security_context",
                    "generate_llm_explanation",
                    "verify_llm_evidence",
                    "generate_final_response",
                ]
            )

        return {
            "request_type": request_type,
            "steps": steps,
            "step_count": len(steps),
        }


def main():

    planner = PlannerAgent()

    result = planner.create_plan(
        {
            "request_type":
                "SECURITY_ANALYSIS"
        },
        has_traffic_data=True,
    )

    print("=" * 70)
    print("PLANNER AGENT")
    print("=" * 70)

    for index, step in enumerate(
        result["steps"],
        start=1,
    ):
        print(
            f"{index}. {step}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )