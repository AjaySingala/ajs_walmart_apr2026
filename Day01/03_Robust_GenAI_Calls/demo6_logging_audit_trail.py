# demo6_logging_audit_trail.py

import time
import json
import logging
import concurrent.futures
from datetime import datetime, UTC
from dotenv import load_dotenv
from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
client = OpenAI()
MODEL_NAME = os.getenv("MODEL_NAME")

# Configure logging (writes to file)
logging.basicConfig(
    filename="llm_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def log_event(event_type: str, data: dict):
    """
    Logs structured events as JSON for easy parsing and audit
    """
    log_entry = {
        "event_type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "data": data
    }

    logging.info(json.dumps(log_entry))


def llm_call(prompt: str) -> str:
    """
    Core LLM call function
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def call_llm_with_logging(prompt: str, retries: int = 3, timeout: int = 5) -> str:
    """
    Robust LLM wrapper with:
    - Retry
    - Timeout
    - Logging (prompt, response, errors, latency)
    """
    start_time = time.time()

    log_event("REQUEST", {"prompt": prompt})

    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(llm_call, prompt)
                result = future.result(timeout=timeout)

                latency = round(time.time() - start_time, 2)

                log_event("RESPONSE", {
                    "response": result,
                    "latency_sec": latency,
                    "attempt": attempt + 1
                })

                return result

        except concurrent.futures.TimeoutError:
            log_event("TIMEOUT", {"attempt": attempt + 1})

        except Exception as e:
            log_event("ERROR", {
                "attempt": attempt + 1,
                "error": str(e)
            })

        # Backoff before retry
        time.sleep(2 ** attempt)

    # Final fallback
    fallback = (
        "We're currently experiencing high demand. "
        "Please try again shortly."
    )

    latency = round(time.time() - start_time, 2)

    log_event("FALLBACK", {
        "message": fallback,
        "total_latency_sec": latency
    })

    return fallback


if __name__ == "__main__":
    prompt = "How does AI improve retail demand forecasting?"

    output = call_llm_with_logging(prompt)

    print("\nLLM Response:\n", output)
    print("\nCheck 'llm_audit.log' for detailed logs.")
