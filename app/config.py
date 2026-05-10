from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    database_url: str
    sync_database_url: str
    app_env: str = "dev"
    log_level: str = "INFO"
    worker_poll_seconds: int = 1
    max_tool_retries: int = 2

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
