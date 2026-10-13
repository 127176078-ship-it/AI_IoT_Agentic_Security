import os
import requests
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
).rstrip("/")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:1b"
)


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:
    """
    Lightweight client for communicating with a local Ollama server.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = 180
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        self.tags_url = f"{self.base_url}/api/tags"
        self.generate_url = f"{self.base_url}/api/generate"

        # Disable environment proxy settings.
        #
        # This is useful on Windows because requests can sometimes
        # inherit proxy settings that interfere with localhost.
        self.session = requests.Session()
        self.session.trust_env = False

    # ========================================================
    # SERVER CHECK
    # ========================================================

    def is_available(self) -> bool:
        """
        Check whether the Ollama server is reachable.
        """

        try:
            response = self.session.get(
                self.tags_url,
                timeout=5
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    # ========================================================
    # MODEL CHECK
    # ========================================================

    def model_available(self) -> bool:
        """
        Check whether the configured Ollama model exists locally.
        """

        try:
            response = self.session.get(
                self.tags_url,
                timeout=5
            )

            response.raise_for_status()

            data = response.json()

            models = data.get("models", [])

            for model_info in models:
                model_name = model_info.get("name", "")

                if model_name == self.model:
                    return True

                # Also allow matching the model without a tag.
                #
                # Example:
                # configured: llama3.2
                # installed : llama3.2:1b
                if model_name.split(":")[0] == self.model.split(":")[0]:
                    return True

            return False

        except (requests.RequestException, ValueError):
            return False

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0
    ) -> str:
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

    
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,

            # Release the model after a short period.
            # This helps reduce RAM usage.
            "keep_alive": "1m",

            "options": {
                # Deterministic output is useful for security
                # explanations and verification.
                "temperature": temperature,

                # Small context because the system has 8 GB RAM.
                "num_ctx": 512,

                # Keep generated output short.
                "num_predict": 64
            }
        }

        # Add system message only when supplied.
        if system:
            payload["system"] = system

        try:

            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            data = response.json()

            result = data.get("response", "")

            if not result or not result.strip():
                raise RuntimeError(
                    "Ollama returned an empty response."
                )

            return result.strip()

        except requests.Timeout as exc:

            raise RuntimeError(
                "Ollama generation timed out. "
                "The local model may be loading or the system "
                "may be low on available RAM."
            ) from exc

        except requests.ConnectionError as exc:

            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure the Ollama server is running."
            ) from exc

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Ollama request failed: {exc}"
            ) from exc

        except ValueError as exc:

            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

    # ========================================================
    # SAFE GENERATION
    # ========================================================

    def safe_generate(
        self,
        prompt: str,
        system: str | None = None
    ) -> str:
       
        try:

            return self.generate(
                prompt=prompt,
                system=system
            )

        except Exception as exc:

            return (
                "[LLM UNAVAILABLE] "
                f"Local Ollama reasoning could not be completed: {exc}"
            )


# ============================================================
# CONNECTION TEST
# ============================================================

def test_ollama():
    """
    Simple command-line test for Ollama connectivity,
    model availability, and generation.
    """

    print("=" * 60)
    print("OLLAMA CONNECTION TEST")
    print("=" * 60)

    print(f"URL   : {OLLAMA_BASE_URL}")
    print(f"Model : {OLLAMA_MODEL}")

    client = OllamaClient()

    # --------------------------------------------------------
    # SERVER TEST
    # --------------------------------------------------------

    if client.is_available():

        print("[OK] Ollama server is reachable.")

    else:

        print("[ERROR] Ollama server is not reachable.")

        print()
        print("Make sure Ollama is running.")
        print("Expected server:")
        print("http://127.0.0.1:11434")

        return

    # --------------------------------------------------------
    # MODEL TEST
    # --------------------------------------------------------

    if client.model_available():

        print(
            f"[OK] Model '{client.model}' is available."
        )

    else:

        print(
            f"[ERROR] Model '{client.model}' "
            "was not found."
        )

        print()
        print("Run:")
        print("ollama list")

        return

    # --------------------------------------------------------
    # GENERATION TEST
    # --------------------------------------------------------

    print()
    print("Testing lightweight generation...")
    print("Please wait...")

    try:

        result = client.generate(
            "Reply with exactly: OK"
        )

        print()
        print("[OK] Ollama generated a response.")
        print()
        print("Response:")
        print(result)
        print()

        print("[SUCCESS] Ollama integration is ready.")

    except Exception as exc:

        print()
        print("[ERROR] Ollama generation failed.")
        print()
        print(f"Reason: {exc}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_ollama()