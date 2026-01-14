"""Data models for the prompt optimizer."""
from models.feedback import Feedback, BatchFeedback, Issue
from models.specification import OutputSpecification, Rule
from models.result import OptimizationResult, Iteration
from models.test_case import CanonicalTestCase, TestCaseInput, ExpectedOutput, EvaluationConfig

# Rebuild models to resolve forward references
Iteration.model_rebuild()
OptimizationResult.model_rebuild()

__all__ = [
    "Feedback",
    "BatchFeedback",
    "Issue",
    "OutputSpecification",
    "Rule",
    "OptimizationResult",
    "Iteration",
    "CanonicalTestCase",
    "TestCaseInput",
    "ExpectedOutput",
    "EvaluationConfig",
]
