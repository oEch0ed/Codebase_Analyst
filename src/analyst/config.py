from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/analyst/config.py -> project root, so .env is found from any cwd
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_file_lines: int = Field(default=1000, gt=0)

    gemini_api_key: SecretStr | None = None
    model: str = "gemini-3.5-flash"
