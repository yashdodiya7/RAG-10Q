"""
Multi-agent RAG orchestration using LangGraph.
"""

from typing import Optional, List

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from src.config import get_settings
from src.agents.llm_factory import LLMFactory
from src.rag import VectorstoreManager, EmbeddingsFactory
from src.data import format_docs


class RAGChain:
    """Simple RAG chain composed with LCEL."""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        retriever=None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            llm: Language model. Defaults to Gemini.
            retriever: Retriever. If None, creates from vectorstore.
        """
        self.llm = llm or LLMFactory.get_primary_llm()
        self.retriever = retriever
        self.settings = get_settings()

    def build_chain(self):
        """
        Build LCEL chain: context | prompt | llm | parser
        
        Returns:
            LCEL runnable chain.
        """
        # Retriever
        if self.retriever is None:
            vs_manager = VectorstoreManager()
            self.retriever = vs_manager.get_retriever(k=4)

        # Prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert automotive consultant with deep knowledge of electric vehicles, 
safety regulations, maintenance requirements, and industry standards.

Use the provided context to answer the user's question accurately and thoroughly.
If the context doesn't contain relevant information, state that clearly.

Context:
{context}

Question: {question}

Answer:""",
        )

        # LCEL chain composition
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
        )

        return chain

    def invoke(self, question: str) -> str:
        """
        Run the RAG chain.
        
        Args:
            question: User question.
            
        Returns:
            LLM response.
        """
        chain = self.build_chain()
        result = chain.invoke(question)
        
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def stream(self, question: str):
        """
        Stream RAG chain responses.
        
        Args:
            question: User question.
            
        Yields:
            Response chunks.
        """
        chain = self.build_chain()
        for chunk in chain.stream(question):
            if hasattr(chunk, "content"):
                yield chunk.content
            else:
                yield str(chunk)


class MultiAgentRAG:
    """Multi-agent RAG system with specialized agents."""

    def __init__(self):
        """Initialize multi-agent system."""
        self.settings = get_settings()
        self.llm = LLMFactory.get_primary_llm()
        # TO BE IMPLEMENTED: LangGraph state machine + agent graph

    def route_query(self, question: str) -> str:
        """
        Route query to appropriate agent.
        Agents:
        - Retrieval Agent: Fetch relevant documents
        - Rerank Agent: Filter by relevance
        - Synthesis Agent: Generate answer
        - Citation Agent: Verify sources
        """
        # TODO: LangGraph routing logic
        pass
