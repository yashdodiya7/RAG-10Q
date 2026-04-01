"""
Embeddings factory for multi-model support.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

from src.config import get_settings


class EmbeddingsFactory:
    """Factory for creating embeddings instances."""

    @staticmethod
    def get_embeddings(model: str = "huggingface") -> Embeddings:
        """
        Get embeddings model.
        
        Args:
            model: Embedding model type ('huggingface' or 'gemini').
            
        Returns:
            Embeddings instance.
        """
        settings = get_settings()
        
        if model == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.GEMINI_API_KEY,
            )
        elif model == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
            )
        else:
            raise ValueError(f"Unknown embeddings model: {model}")

    @staticmethod
    def get_default_embeddings() -> Embeddings:
        """Get default embeddings (HuggingFace - lightweight & fast)."""
        return EmbeddingsFactory.get_embeddings("huggingface")
