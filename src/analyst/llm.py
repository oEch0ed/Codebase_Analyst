"""Thin wrapper around the Gemini API."""

import logging

from google import genai
from google.genai import types
from google.genai.interactions import Interaction

logger = logging.getLogger(__name__)

# Free tier allows 100 embeddings/minute and flash models sometimes 503 under load:
# wait it out (10s, 20s, 40s, 60s, 60s) instead of failing.
# 429 - Too many requests
# 503 - overload on serverside
RETRY = types.HttpOptions(
    retry_options=types.HttpRetryOptions(
        attempts=6, initial_delay=10, max_delay=60, http_status_codes=[429, 503]
    )
)


def make_client(api_key: str) -> genai.Client:
    """Gemini client that retries on rate limits and overload."""
    return genai.Client(api_key=api_key, http_options=RETRY)


def ask_model(question: str, *, api_key: str, model: str) -> str:
    client = make_client(api_key)
    logger.info("Sending query to %s", model)
    interaction = client.interactions.create(model=model, input=question)
    # create() is typed to maybe return a stream; we never request one
    if not isinstance(interaction, Interaction):
        raise TypeError(f"Expected an Interaction, got {type(interaction).__name__}")
    if interaction.output_text is None:
        raise RuntimeError(f"{model} returned no text output")
    return interaction.output_text
