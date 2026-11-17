import logging
import sys


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,  # replace any handlers set earlier (matters in tests)
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request noise
