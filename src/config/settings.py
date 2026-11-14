"""
Configuration management using Pydantic Settings.
Handles environment variables, paths, and model configurations.
"""

from pathlib import Path
from typing import Optional, Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Environment
    ENV: Literal["development", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    DEBUG: bool = Field(default=True)

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data"
    )
    CHROMA_DB_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "chroma_10q_db"
    )

    # PDF Configuration
    PDF_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "10q.pdf")
    AUTOMOTIVE_INPUT_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "automotive_corpus" / "incoming_pdfs"
    )
    AUTOMOTIVE_PARSED_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "automotive_corpus" / "parsed"
    )
    MINERU_OUTPUT_FORMAT: Literal["md", "markdown", "json"] = Field(default="markdown")
    MINERU_BACKEND: Literal["pipeline", "auto"] = Field(default="pipeline")
    CHUNK_SIZE: int = Field(default=500, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=100, ge=0, le=500)

    # LLM Configuration
    PRIMARY_LLM: Literal["gemini", "ollama"] = Field(default="gemini")
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-3.5-flash-lite")
    
    # Fallback LLM
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3:8b")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "mps"] = Field(default="cpu")

    # Vectorstore
    VECTORSTORE_TYPE: Literal["chroma", "milvus"] = Field(default="chroma")
    CHROMA_COLLECTION_NAME: str = Field(default="automotive_docs")

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1024, le=65535)
    API_WORKERS: int = Field(default=4, ge=1)

    # RAGAS Evaluation
    RAGAS_LLM: str = Field(default="gemini")
    RAGAS_EMBEDDINGS: str = Field(default="huggingface")

    # Automotive Data Configuration
    AUTOMOTIVE_CORPUS_PATH: Optional[Path] = Field(default=None)
    MIN_CONTEXT_RELEVANCE: float = Field(default=0.7, ge=0.0, le=1.0)
    MIN_FAITHFULNESS: float = Field(default=0.6, ge=0.0, le=1.0)
    MIN_ANSWER_CORRECTNESS: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8-sig"  # Handle Windows UTF-16LE
        case_sensitive = True

    def validate_paths(self) -> None:
        """Ensure required directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.AUTOMOTIVE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.AUTOMOTIVE_PARSED_DIR.mkdir(parents=True, exist_ok=True)
        if self.AUTOMOTIVE_CORPUS_PATH:
            self.AUTOMOTIVE_CORPUS_PATH.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_paths()
    return _settings
