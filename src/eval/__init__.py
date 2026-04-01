"""Evaluation package."""

from .evaluator import RAGEvaluator, EvalResult
from .datasets import TESLA_10Q_DATASET, AutomotiveDataset

__all__ = [
    "RAGEvaluator",
    "EvalResult",
    "TESLA_10Q_DATASET",
    "AutomotiveDataset",
]
