from pathlib import Path
from datetime import datetime, timezone
import json

from app.ml.evaluate import evaluate


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS = (
    PROJECT_ROOT / "reports"
)


def write_report(
    incident=None,
):

    REPORTS.mkdir(
        exist_ok=True
    )

    metrics = evaluate()

    report = {
        "title":
            "AI-IoT-Agentic-Security",

        "dataset":
            "UNSW-NB15 only",

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "metrics":
            metrics,

        "security_design": {
            "zero_trust":
                True,

            "multi_model_fusion":
                True,

            "anomaly_detection":
                True,

            "llm_evidence_verification":
                True,

            "human_approval_required":
                True,
        },

        "incident":
            incident or {},

        "remediation":
            "HUMAN_APPROVAL_REQUIRED",
    }

    output = (
        REPORTS
        / "final_report.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"\nReport saved: {output}"
    )

    return report


def main():

    try:

        write_report()

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