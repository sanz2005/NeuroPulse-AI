"""
NeuroPulse AI — Backend Configuration
Loads environment variables with defaults.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = (
        "sqlite+aiosqlite:///./neuropulse.db"
    )

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL:  str = "redis://localhost:6379"

    # JWT
    SECRET_KEY: str = "neuropulse-secret-key-change-in-production"
    ALGORITHM:  str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # AI Service
    MODEL_DEVICE:     str = "cpu"
    SAVED_MODELS_DIR: str = "ai/saved_models"

    # Signal Streaming
    SIGNAL_SAMPLE_RATE: int = 256
    STREAM_WINDOW_SIZE: int = 256
    STREAM_STEP_SIZE:   int = 64

    # Data paths
    DATA_PROCESSED_DIR: str = "data/processed"

    class Config:
        env_file = ".env"


settings = Settings()