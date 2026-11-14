"""Data package."""

from .mineru_parser import MinerUParser

try:
    from .document_loader import (
        AutomotiveDocumentProcessor,
        DocumentLoader,
        format_docs,
    )
except ModuleNotFoundError:
    # Allow metadata-only scripts to run without optional LangChain extras.
    AutomotiveDocumentProcessor = None
    DocumentLoader = None
    format_docs = None

__all__ = [
    "MinerUParser",
    "DocumentLoader",
    "AutomotiveDocumentProcessor",
    "format_docs",
]
