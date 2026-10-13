from flask import Flask

from app.api.server import api


def create_app():

    app = Flask(
        __name__
    )

    app.register_blueprint(
        api
    )

    return app


def main():

    print("=" * 70)
    print("AI-IoT-AGENTIC-SECURITY")
    print("=" * 70)

    print(
        "Dataset : UNSW-NB15"
    )

    print(
        "Mode    : Agentic AI + ML IDS"
    )

    print(
        "API     : http://127.0.0.1:5000"
    )

    print("=" * 70)

    app = create_app()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )


if __name__ == "__main__":
    main()