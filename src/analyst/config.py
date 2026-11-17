from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    max_file_lines: int = Field(default=1000, gt=0)

    gemini_api_key: SecretStr | None = None
    model: str = "gemini-3.5-flash"
