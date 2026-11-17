"""
LLM factory for multi-model support.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseLanguageModel

from src.config import get_settings


class LLMFactory:
    """Factory for creating LLM instances."""

    @staticmethod
    def get_llm(model: str = "gemini", temperature: float = 0.7) -> BaseLanguageModel:
        """
        Get language model.

        Args:
            model: Model type ('gemini' or 'ollama').
            temperature: Sampling temperature (Gemini). Use 0 for agents and extraction.

        Returns:
            Language model instance.
        """
        settings = get_settings()

        if model == "gemini":
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
            )
        elif model == "ollama":
            return OllamaLLM(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
            )
        else:
            raise ValueError(f"Unknown LLM model: {model}")

    @staticmethod
    def get_primary_llm() -> BaseLanguageModel:
        """Get primary configured LLM."""
        settings = get_settings()
        return LLMFactory.get_llm(settings.PRIMARY_LLM)

    @staticmethod
    def get_fallback_llm() -> BaseLanguageModel:
        """Get fallback LLM (Ollama)."""
        return LLMFactory.get_llm("ollama")
