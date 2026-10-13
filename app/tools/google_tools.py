from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


CREDENTIALS_FILE = (
    PROJECT_ROOT
    / "credentials"
    / "google_credentials.json"
)


class GoogleTools:

    def credentials_available(self):

        return (
            CREDENTIALS_FILE.exists()
        )

    def send_gmail(
        self,
        recipient,
        subject,
        message,
    ):

        if not self.credentials_available():

            return {
                "status":
                    "unavailable",

                "reason":
                    "Google credentials not configured.",
            }

        return {
            "status":
                "credentials_available",

            "message":
                "Google API execution can be enabled.",
        }

    def create_calendar_event(
        self,
        title,
        start_time,
        end_time,
    ):

        if not self.credentials_available():

            return {
                "status":
                    "unavailable",

                "reason":
                    "Google credentials not configured.",
            }

        return {
            "status":
                "credentials_available",

            "message":
                "Google Calendar API execution can be enabled.",
        }


def main():

    tools = (
        GoogleTools()
    )

    print("=" * 60)
    print("GOOGLE TOOLS")
    print("=" * 60)

    if tools.credentials_available():

        print(
            "Google credentials: FOUND"
        )

    else:

        print(
            "Google credentials: NOT CONFIGURED"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )