from flask import (
    Blueprint,
    jsonify,
    request,
)

import pandas as pd

from app.agent.orchestrator import (
    SecurityOrchestrator,
)


api = Blueprint(
    "api",
    __name__,
)

orchestrator = None


def get_orchestrator():

    global orchestrator

    if orchestrator is None:

        orchestrator = (
            SecurityOrchestrator()
        )

    return orchestrator


@api.get(
    "/health"
)
def health():

    return jsonify(
        {
            "status":
                "ok",

            "system":
                "AI-IoT-Agentic-Security",

            "dataset":
                "UNSW-NB15",
        }
    )


@api.post(
    "/analyze"
)
def analyze():

    try:

        body = request.get_json(
            silent=True
        )

        if not body:

            return jsonify(
                {
                    "error":
                        "JSON body required."
                }
            ), 400

        rows = body.get(
            "data"
        )

        user_request = body.get(
            "query",
            "Analyze the supplied UNSW-NB15 traffic.",
        )

        if not isinstance(
            rows,
            list,
        ):

            return jsonify(
                {
                    "error":
                        "'data' must be a list."
                }
            ), 400

        if not rows:

            return jsonify(
                {
                    "error":
                        "'data' cannot be empty."
                }
            ), 400

        dataframe = pd.DataFrame(
            rows
        )

        result = (
            get_orchestrator()
            .analyze(
                dataframe,
                user_request,
            )
        )

        return jsonify(
            result
        )

    except Exception as exc:

        return jsonify(
            {
                "error":
                    type(exc).__name__,

                "message":
                    str(exc),
            }
        ), 500