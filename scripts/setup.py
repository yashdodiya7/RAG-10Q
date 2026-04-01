#!/usr/bin/env python
"""
Setup script: Complete workflow for modularization, data setup, and first run.
Run: python scripts/setup.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.data import DocumentLoader, AutomotiveDocumentProcessor
from src.rag import VectorstoreManager


def setup_directories():
    """Create necessary directories."""
    settings = get_settings()
    
    print("📁 Creating directories...")
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    automotive_corpus = settings.DATA_DIR / "automotive_corpus"
    automotive_corpus.mkdir(parents=True, exist_ok=True)
    (automotive_corpus / "vehicle_specs").mkdir(parents=True, exist_ok=True)
    (automotive_corpus / "safety_reports").mkdir(parents=True, exist_ok=True)
    (automotive_corpus / "maintenance_guides").mkdir(parents=True, exist_ok=True)
    (automotive_corpus / "regulations").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Directories ready")
    return automotive_corpus


def verify_environment():
    """Verify all required environment variables."""
    settings = get_settings()
    
    print("🔍 Verifying environment...")
    print(f"  • Environment: {settings.ENV}")
    print(f"  • Primary LLM: {settings.PRIMARY_LLM}")
    print(f"  • Gemini API Key: {'SET' if settings.GEMINI_API_KEY else 'NOT SET'}")
    print(f"  • Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  • PDF Path: {settings.PDF_PATH} ({'EXISTS' if settings.PDF_PATH.exists() else 'NOT FOUND'})")
    
    if not settings.GEMINI_API_KEY:
        print("\n⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Set it in .env file: GEMINI_API_KEY=your-key-here")
    
    return settings.GEMINI_API_KEY is not None


def test_imports():
    """Verify all imports work."""
    print("\n✓ Testing imports...")
    try:
        from src.config import get_settings
        from src.data import DocumentLoader
        from src.rag import VectorstoreManager, EmbeddingsFactory
        from src.agents import LLMFactory, RAGChain
        from src.eval import RAGEvaluator, TESLA_10Q_DATASET
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup workflow."""
    print("\n" + "=" * 60)
    print("🚗 AUTOMOTIVE RAG - MODULARIZATION SETUP")
    print("=" * 60)
    
    if not test_imports():
        print("\n❌ Dependency check failed!")
        sys.exit(1)
    
    automotive_corpus = setup_directories()

    has_api_key = verify_environment()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
