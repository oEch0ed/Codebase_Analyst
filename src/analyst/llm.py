"""Thin wrapper around the Gemini API."""

import logging

from google import genai
from google.genai.interactions import Interaction

logger = logging.getLogger(__name__)


def ask(question: str, *, api_key: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    logger.info("Sending query to %s", model)
    interaction = client.interactions.create(model=model, input=question)
    # create() is typed to maybe return a stream; we never request one
    if not isinstance(interaction, Interaction):
        raise TypeError(f"Expected an Interaction, got {type(interaction).__name__}")
    if interaction.output_text is None:
        raise RuntimeError(f"{model} returned no text output")
    return interaction.output_text
