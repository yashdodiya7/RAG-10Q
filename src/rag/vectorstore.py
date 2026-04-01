"""
Vectorstore management for Chroma.
"""

from pathlib import Path
from typing import Optional, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.config import get_settings
from src.rag.embeddings import EmbeddingsFactory


class VectorstoreManager:
    """Manage Chroma vectorstore lifecycle."""

    def __init__(self, embeddings_model: str = "huggingface"):
        """
        Initialize vectorstore manager.
        
        Args:
            embeddings_model: Which embeddings to use.
        """
        self.settings = get_settings()
        self.embeddings = EmbeddingsFactory.get_embeddings(embeddings_model)
        self.vectorstore: Optional[Chroma] = None

    def create_vectorstore(
        self,
        documents: List[Document],
        force_recreate: bool = False,
    ) -> Chroma:
        """
        Create or load vectorstore.
        
        Args:
            documents: Documents to add to vectorstore.
            force_recreate: Delete existing DB and recreate.
            
        Returns:
            Chroma vectorstore instance.
        """
        db_path = self.settings.CHROMA_DB_DIR
        
        if force_recreate and db_path.exists():
            import shutil
            shutil.rmtree(db_path)
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.settings.CHROMA_COLLECTION_NAME,
            persist_directory=str(db_path),
        )
        
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Load existing vectorstore from disk.
        
        Returns:
            Chroma vectorstore instance.
        """
        db_path = self.settings.CHROMA_DB_DIR
        
        if not db_path.exists():
            raise FileNotFoundError(f"Vectorstore not found at {db_path}")
        
        self.vectorstore = Chroma(
            collection_name=self.settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(db_path),
        )
        
        return self.vectorstore

    def get_or_create_vectorstore(
        self,
        documents: Optional[List[Document]] = None,
    ) -> Chroma:
        """
        Get existing vectorstore or create new one.
        
        Args:
            documents: Documents to add if creating new.
            
        Returns:
            Chroma vectorstore instance.
        """
        db_path = self.settings.CHROMA_DB_DIR
        
        if db_path.exists():
            return self.load_vectorstore()
        elif documents:
            return self.create_vectorstore(documents)
        else:
            raise ValueError(
                "Vectorstore not found and no documents provided to create it"
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing vectorstore.
        
        Args:
            documents: Documents to add.
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Search vectorstore by similarity.
        
        Args:
            query: Query text.
            k: Number of results to return.
            score_threshold: Minimum similarity score.
            
        Returns:
            List of relevant documents.
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        if score_threshold is not None:
            return self.vectorstore.similarity_search_with_score(
                query, k=k, score_threshold=score_threshold
            )
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 4):
        """
        Get a retriever from vectorstore.
        
        Args:
            k: Number of documents to retrieve.
            
        Returns:
            Vectorstore retriever.
        """
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
