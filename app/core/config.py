from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Relay ML"
    app_version: str = "0.1.0"
    port: int = 8001
    aws_region: str = "ap-south-1"
    grading_mode: Literal["cnn", "bedrock_only", "mock"] = "bedrock_only"
    bedrock_model_t1: str = "amazon.nova-lite-v1:0"
    bedrock_model_t2: str = ""
    bedrock_model_t3: str = ""
    openai_api_key: str = ""
    cnn_model_path: Path = Path("./models/grade_cnn_v1.pt")
    fit_aggregates_path: Path = Path("./data/processed/fit_aggregates.json")
    confidence_threshold_t2: float = 0.85
    confidence_threshold_t3: float = 0.75
    embedding_model: str = "all-MiniLM-L6-v2"
    wish_score_model_path: Path = Path("./models/wish_logreg_v1.pkl")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
