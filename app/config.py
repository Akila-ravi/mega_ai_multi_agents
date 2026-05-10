from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    database_url: str
    sync_database_url: str
    app_env: str = "dev"
    log_level: str = "INFO"
    worker_poll_seconds: int = 1
    max_tool_retries: int = 2
    rag_top_k: int = 8
    retrieval_merge_cap: int = 6
    # FAISS + OpenAI embeddings (optional). Empty path disables local FAISS load.
    vectorstore_path: str = ""
    rag_faiss_k: int = 4

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


settings = Settings()
