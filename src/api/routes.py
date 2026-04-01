"""
API routes for chat interface.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional

from src.agents import RAGChain
from src.rag import VectorstoreManager, EmbeddingsFactory

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=4, ge=1, le=20)


class ChatResponse(BaseModel):
    """Chat response model."""
    question: str
    answer: str
    sources: Optional[list] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for automotive RAG queries.
    
    Args:
        request: Chat request with question.
        
    Returns:
        Chat response with answer and sources.
    """
    try:
        # Initialize vectorstore and chain
        vs_manager = VectorstoreManager()
        retriever = vs_manager.get_retriever(k=request.top_k)
        
        chain = RAGChain(retriever=retriever)
        answer = chain.invoke(request.question)
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=None,  # TODO: track source documents
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
