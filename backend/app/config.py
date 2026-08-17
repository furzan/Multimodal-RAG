from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"
SQLITE_DB_PATH = DATA_DIR / "docstore.db"


class Settings(BaseSettings):
    # --- Groq ---
    groq_api_key: str = ""
    groq_llm_model: str = "llama-3.3-70b-versatile"  

    # --- Google Gemini ---
    google_api_key: str = ""
    gemini_vision_model: str = "gemini-2.0-flash"      
    gemini_embedding_model: str = "models/embedding-001" 

    # --- Pinecone ---
    pinecone_api_key: str = ""
    pinecone_index_name: str = "multimodal-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_embedding_dim: int = 768  

    # --- Redis (semantic cache) ---
    redis_url: str = "redis://localhost:6379"
    redis_cache_index_name: str = "llmcache"
    semantic_cache_distance_threshold: float = 0.2  

    # --- SQLite ---
    sqlite_db_path: str = str(SQLITE_DB_PATH)

    # --- App / misc ---
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Ensure required data directories exist at import time.
for _dir in (DATA_DIR, UPLOADS_DIR, EXTRACTED_IMAGES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
