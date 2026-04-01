"""
RAGAS evaluation framework for RAG system.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config import get_settings
from src.agents import LLMFactory
from src.rag import EmbeddingsFactory


@dataclass
class EvalResult:
    """Evaluation result container."""
    
    faithfulness_score: float
    answer_relevancy_score: float
    context_precision_score: float
    context_recall_score: float
    answer_correctness_score: float
    
    def __repr__(self) -> str:
        return f"""
Evaluation Results:
  Faithfulness: {self.faithfulness_score:.3f}
  Answer Relevancy: {self.answer_relevancy_score:.3f}
  Context Precision: {self.context_precision_score:.3f}
  Context Recall: {self.context_recall_score:.3f}
  Answer Correctness: {self.answer_correctness_score:.3f}
"""

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness_score,
            "answer_relevancy": self.answer_relevancy_score,
            "context_precision": self.context_precision_score,
            "context_recall": self.context_recall_score,
            "answer_correctness": self.answer_correctness_score,
        }

    @property
    def average_score(self) -> float:
        """Calculate average evaluation score."""
        scores = self.to_dict()
        return sum(scores.values()) / len(scores)


class RAGEvaluator:
    """Evaluate RAG system using RAGAS metrics."""

    def __init__(self):
        """Initialize evaluator."""
        self.settings = get_settings()
        self.llm = LangchainLLMWrapper(LLMFactory.get_primary_llm())
        self.embeddings = LangchainEmbeddingsWrapper(
            EmbeddingsFactory.get_default_embeddings()
        )

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvalResult:
        """
        Evaluate RAG system.
        
        Args:
            questions: List of test questions.
            answers: List of generated answers.
            contexts: List of retrieved context chunks for each question.
            ground_truths: List of ground truth answers.
            
        Returns:
            EvalResult with all metrics.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        if ground_truths:
            metrics.append(answer_correctness)
        
        # Run evaluation
        results = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        print("Raw Evaluation Results:", results)
        
        eval_result = EvalResult(
            faithfulness_score=float(results["faithfulness"] if results["faithfulness"] is not None else 0.0),
            answer_relevancy_score=float(results["answer_relevancy"]),
            context_precision_score=float(results["context_precision"]),
            context_recall_score=float(results["context_recall"]),
            answer_correctness_score=(
                float(results["answer_correctness"])
                if "answer_correctness" in results
                else 0.0
            ),
        )
        
        return eval_result

    def check_quality_gates(self, eval_result: EvalResult) -> Dict[str, bool]:
        """
        Check if evaluation meets minimum quality standards.
        
        Args:
            eval_result: Evaluation results.
            
        Returns:
            Dictionary of gate checks (metric: pass/fail).
        """
        return {
            "faithfulness_gate": (
                eval_result.faithfulness_score
                >= self.settings.MIN_FAITHFULNESS
            ),
            "context_precision_gate": (
                eval_result.context_precision_score
                >= self.settings.MIN_CONTEXT_RELEVANCE
            ),
            "answer_correctness_gate": (
                eval_result.answer_correctness_score
                >= self.settings.MIN_ANSWER_CORRECTNESS
            ),
        }
