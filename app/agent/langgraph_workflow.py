from pathlib import Path
from typing import TypedDict, Any

import pandas as pd

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.agent.orchestrator import (
    SecurityOrchestrator,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)


class SecurityState(TypedDict, total=False):

    dataframe: Any
    user_request: str
    result: dict
    error: str


def build_workflow():

    orchestrator = (
        SecurityOrchestrator()
    )

    graph = StateGraph(
        SecurityState
    )

    def security_analysis(
        state,
    ):

        try:

            result = (
                orchestrator.analyze(
                    state["dataframe"],
                    state.get(
                        "user_request",
                        "Analyze the supplied UNSW-NB15 traffic.",
                    ),
                )
            )

            return {
                "result": result
            }

        except Exception as exc:

            return {
                "error":
                    f"{type(exc).__name__}: {exc}"
            }

    graph.add_node(
        "security_analysis",
        security_analysis,
    )

    graph.add_edge(
        START,
        "security_analysis",
    )

    graph.add_edge(
        "security_analysis",
        END,
    )

    return graph.compile()


def main():

    test_file = (
        PROJECT_ROOT
        / "data"
        / "UNSW_NB15_testing-set.csv"
    )

    print("=" * 70)
    print("LANGGRAPH SECURITY WORKFLOW")
    print("=" * 70)

    try:

        dataframe = pd.read_csv(
            test_file
        ).head(3)

        workflow = (
            build_workflow()
        )

        state = workflow.invoke(
            {
                "dataframe":
                    dataframe,

                "user_request":
                    "Analyze the supplied UNSW-NB15 traffic.",
            }
        )

        if state.get("error"):

            print(
                f"[ERROR] {state['error']}"
            )

            return 1

        result = state[
            "result"
        ]

        print(
            "\nLangGraph workflow completed."
        )

        print(
            f"Request ID: "
            f"{result['request_id']}"
        )

        print(
            f"Zero Trust: "
            f"{result['evaluation']['zero_trust_decision']}"
        )

        print(
            f"Severity: "
            f"{result['evaluation']['severity']}"
        )

        return 0

    except Exception as exc:

        print(
            f"[ERROR] {type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )