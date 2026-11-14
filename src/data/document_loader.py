"""
Document loading and text splitting for RAG.
Handles PDF ingestion and automotive document processing.
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import get_settings
from src.data.mineru_parser import MinerUParser


class DocumentLoader:
    """Load and process documents for RAG."""

    def __init__(self):
        """Initialize document loader."""
        self.settings = get_settings()
        self.mineru_parser = MinerUParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_pdf(self, pdf_path: Optional[Path] = None) -> List[Document]:
        """
        Load PDF document.
        
        Args:
            pdf_path: Path to PDF file. Defaults to configured PDF_PATH.
            
        Returns:
            List of loaded documents.
        """
        path = pdf_path or self.settings.PDF_PATH
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        try:
            documents = self.mineru_parser.parse_pdf(
                input_path=path,
                output_dir=self.settings.AUTOMOTIVE_PARSED_DIR,
                backend=self.settings.MINERU_BACKEND,
            )
            if documents:
                return documents
        except Exception as exc:
            print(f"MinerU parsing failed for {path}: {exc}. Falling back to PyPDFLoader.")

        loader = PyPDFLoader(str(path))
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split.
            
        Returns:
            List of chunked documents.
        """
        chunks = self.splitter.split_documents(documents)
        return chunks

    def load_and_split_pdf(self, pdf_path: Optional[Path] = None) -> List[Document]:
        """
        Load and split PDF in one step.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of chunked documents.
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks

    def step1_extract_text_blocks(
        self,
        pdf_path: str,
        output_json_path: Optional[str] = None,
    ) -> List[dict]:
        """
        Extract MinerU text blocks and save a content_list JSON file.

        This is the first parsing step for the automotive corpus pipeline.
        """
        return self.mineru_parser.step1_extract_text_blocks(
            pdf_path=pdf_path,
            output_json_path=output_json_path,
            backend=self.settings.MINERU_BACKEND,
        )

    def step2_generate_structure_metadata(
        self,
        content_json_path: str,
        output_json_path: Optional[str] = None,
    ) -> dict:
        """
        Generate heading-level structure metadata from a content_list JSON file.

        This step is metadata-only and does not perform chunking.
        """
        return self.mineru_parser.step2_generate_structure_metadata(
            content_json_path=content_json_path,
            output_json_path=output_json_path,
        )

    def load_automotive_corpus(self, corpus_dir: Optional[Path] = None) -> List[Document]:
        """
        Load automotive documents from structured corpus directory.
        
        Expected directory structure:
        corpus_dir/
            ├── incoming_pdfs/
            ├── parsed/
            ├── vehicle_specs/
            ├── safety_reports/
            ├── maintenance_guides/
            └── regulations/
        
        Args:
            corpus_dir: Path to corpus directory.
            
        Returns:
            List of documents with metadata.
        """
        corpus_path = corpus_dir or self.settings.AUTOMOTIVE_CORPUS_PATH
        
        if not corpus_path or not corpus_path.exists():
            raise FileNotFoundError(f"Automotive corpus not found: {corpus_path}")
        
        all_documents = []
        
        for pdf_file in corpus_path.rglob("*.pdf"):
            try:
                docs = self.load_pdf(pdf_file)
                
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["doc_type"] = pdf_file.parent.name
                    doc.metadata["corpus"] = "automotive"
                
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")
                continue
        
        return all_documents


class AutomotiveDocumentProcessor:
    """Process and enrich automotive documents with structured metadata."""

    @staticmethod
    def add_automotive_metadata(
        documents: List[Document],
        doc_type: str,
        source: str,
        region: str = "US",
        language: str = "en",
    ) -> List[Document]:
        """
        Add automotive-specific metadata to documents.
        
        Args:
            documents: List of documents.
            doc_type: Type of document (e.g., 'safety_report', 'spec_sheet').
            source: Data source (e.g., 'NHTSA', 'manufacturer').
            region: Geographic region (US, EU, CN, etc.).
            language: Document language code.
            
        Returns:
            Documents with enriched metadata.
        """
        for doc in documents:
            doc.metadata.update({
                "doc_type": doc_type,
                "source": source,
                "region": region,
                "language": language,
            })
        return documents


def format_docs(docs: List[Document]) -> str:
    """
    Format documents into a single string for prompts.
    
    Args:
        docs: List of documents to format.
        
    Returns:
        Formatted string with document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)
