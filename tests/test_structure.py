from pathlib import Path


def test_required_structure():

    root = Path(
        __file__
    ).resolve().parents[1]

    required = [

        "app/__init__.py",

        "app/agent/__init__.py",
        "app/agent/orchestrator.py",
        "app/agent/triage.py",
        "app/agent/planner.py",
        "app/agent/evaluator.py",
        "app/agent/langgraph_workflow.py",

        "app/ml/__init__.py",
        "app/ml/preprocessing/__init__.py",
        "app/ml/preprocessing/preprocess.py",
        "app/ml/train_baseline.py",
        "app/ml/train_anomaly.py",
        "app/ml/detector.py",
        "app/ml/evaluate.py",

        "app/security/__init__.py",
        "app/security/shap_analysis.py",
        "app/security/explain.py",
        "app/security/mitre.py",

        "app/rag/__init__.py",
        "app/rag/knowledge.py",
        "app/rag/retriever.py",

        "app/llm/__init__.py",
        "app/llm/ollama_client.py",

        "app/tools/__init__.py",
        "app/tools/iot_intrusion.py",
        "app/tools/scapy_capture.py",
        "app/tools/web_search.py",
        "app/tools/google_tools.py",

        "app/response/__init__.py",
        "app/response/report.py",
        "app/response/verifier.py",

        "app/api/__init__.py",
        "app/api/server.py",

        "main.py"
    ]

    missing = [
        path
        for path in required
        if not (
            root / path
        ).exists()
    ]

    assert not missing, (
        "Missing files: "
        + str(missing)
    )