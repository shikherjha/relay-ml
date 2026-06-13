from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Relay ML"
    app_version: str = "0.1.0"
    port: int = 8001
    aws_region: str = "ap-south-1"
    bedrock_model_t2: str = ""
    bedrock_model_t3: str = ""
    openai_api_key: str = ""
    cnn_model_path: Path = Path("./models/grade_cnn_v1.pt")
    confidence_threshold_t2: float = 0.85
    confidence_threshold_t3: float = 0.75

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
