"""
app/api/routes.py

REST API routes for AI-IoT-Agentic-Security.

Endpoints:
    GET  /health
    POST /analyze

The API accepts security-analysis requests and passes them
to the agentic orchestration layer.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request


# ============================================================
# BLUEPRINT
# ============================================================

api = Blueprint("api", __name__)


# ============================================================
# ORCHESTRATOR
# ============================================================

_orchestrator = None


def get_orchestrator():
    """
    Lazily create the orchestrator.

    Lazy loading is intentional:
        - Flask can start without loading every ML/SHAP component.
        - SHAP and other models are loaded only when /analyze
          actually needs them.
    """

    global _orchestrator

    if _orchestrator is None:
        from app.agent.orchestrator import Orchestrator

        _orchestrator = Orchestrator()

    return _orchestrator


# ============================================================
# HEALTH
# ============================================================

@api.get("/health")
def health():
    """API health check."""

    return jsonify(
        {
            "system": "AI-IoT-Agentic-Security",
            "dataset": "UNSW-NB15",
            "status": "ok",
        }
    )


# ============================================================
# ANALYZE
# ============================================================

@api.post("/analyze")
def analyze():
    """
    Analyze an IoT/network security request.

    Supported JSON forms:

    1. Direct user request:

        {
            "query": "Analyze this traffic for intrusion"
        }

    2. Tabular security data:

        {
            "data": [
                {
                    "dur": 0.5,
                    "spkts": 10,
                    "dpkts": 8
                }
            ]
        }

    3. Combined:

        {
            "query": "Perform a security analysis",
            "data": [...]
        }
    """

    try:
        payload = request.get_json(silent=True)

        if payload is None:
            return jsonify(
                {
                    "status": "error",
                    "message": "Request body must contain valid JSON.",
                }
            ), 400

        if not isinstance(payload, dict):
            return jsonify(
                {
                    "status": "error",
                    "message": "JSON body must be an object.",
                }
            ), 400

        query = payload.get("query", "")
        data = payload.get("data")

        # ----------------------------------------------------
        # VALIDATE INPUT
        # ----------------------------------------------------

        if data is None and not query:
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "Provide either 'query' or 'data' "
                        "in the JSON request."
                    ),
                }
            ), 400

        if data is not None:

            if not isinstance(data, list):
                return jsonify(
                    {
                        "status": "error",
                        "message": "'data' must be a list of records.",
                    }
                ), 400

            if len(data) == 0:
                return jsonify(
                    {
                        "status": "error",
                        "message": "'data' cannot be empty.",
                    }
                ), 400

        # ----------------------------------------------------
        # GET ORCHESTRATOR
        # ----------------------------------------------------

        orchestrator = get_orchestrator()

        # ----------------------------------------------------
        # EXECUTE AGENTIC ANALYSIS
        # ----------------------------------------------------

        result = orchestrator.run(
            query=query,
            data=data,
        )

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return jsonify(
            {
                "status": "success",
                "dataset": "UNSW-NB15",
                "result": result,
            }
        ), 200

    except Exception as exc:

        # Do not expose a complete internal traceback through
        # the public API.
        return jsonify(
            {
                "status": "error",
                "message": str(exc),
            }
        ), 500