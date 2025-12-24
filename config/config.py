from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "llama_factory_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Use SQLite for testing if set to True
    USE_SQLITE: bool = True
    SQLITE_DB_PATH: str = "./data/llama_factory.db"

    # Application Configuration
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 7860
    DEBUG: bool = True

    # LLaMA Factory Configuration
    BASE_MODEL_PATH: str = "./models/base"
    TRAINED_MODEL_PATH: str = "./models/trained"
    DATASET_PATH: str = "./data/datasets"

    # Training Configuration
    MAX_CONCURRENT_TRAININGS: int = 2
    DEFAULT_BATCH_SIZE: int = 4
    DEFAULT_LEARNING_RATE: float = 5e-5

    # Logging
    LOG_LEVEL: str = "INFO"
    WANDB_API_KEY: Optional[str] = None

    @property
    def database_url(self) -> str:
        """Get database URL (PostgreSQL or SQLite)"""
        if self.USE_SQLITE:
            return f"sqlite:///{self.SQLITE_DB_PATH}"
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
