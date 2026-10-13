"""
app/tools/iot_intrusion.py

Main Flask entry point for the AI-IoT-Agentic-Security IDS.

Run:
    python -m app.tools.iot_intrusion
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

from app.api.routes import api


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "UNSW-NB15"

HOST = os.getenv("FLASK_HOST", "127.0.0.1")
PORT = int(os.getenv("FLASK_PORT", "5000"))


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

def create_app() -> Flask:
    app = Flask(__name__)

    # Register API routes.
    app.register_blueprint(api)

    # --------------------------------------------------------
    # ROOT ROUTE
    # --------------------------------------------------------

    @app.get("/")
    def home():
        return jsonify(
            {
                "system": "AI-IoT-Agentic-Security",
                "dataset": DATASET_NAME,
                "mode": "Agentic AI + ML IDS",
                "status": "running",
                "api": {
                    "health": "/health",
                    "analyze": "/analyze",
                },
            }
        )

    # --------------------------------------------------------
    # HEALTH ROUTE
    # --------------------------------------------------------

    @app.get("/health")
    def health():
        return jsonify(
            {
                "system": "AI-IoT-Agentic-Security",
                "dataset": DATASET_NAME,
                "status": "ok",
            }
        )

    return app


# ============================================================
# APPLICATION OBJECT
# ============================================================

app = create_app()


# ============================================================
# START SERVER
# ============================================================

def main():
    print("=" * 60)
    print("AI-IoT-AGENTIC-SECURITY")
    print("=" * 60)
    print(f"Dataset : {DATASET_NAME}")
    print("Mode    : Agentic AI + ML IDS")
    print(f"API     : http://{HOST}:{PORT}")
    print("=" * 60)
    print()
    print("Starting Flask server...")
    print("Press CTRL+C to stop.")
    print()

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()