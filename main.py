import argparse
from google import genai
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str
    model: str = "gemini-3.5-flash"


def main():
    parser = argparse.ArgumentParser(description="Ask Gemini a question.")
    parser.add_argument("query", nargs="+", help="the prompt to send")
    args = parser.parse_args()

    settings = Settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    interaction = client.interactions.create(
        model=settings.model,
        input=" ".join(args.query),
    )
    print(interaction.output_text)


if __name__ == "__main__":
    main()
