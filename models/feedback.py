"""Feedback and critique models."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Issue(BaseModel):
    """A specific issue found in the output."""
    severity: str = Field(description="CRITICAL | HIGH | MEDIUM | LOW")
    issue: str = Field(description="What's wrong")
    evidence: str = Field(description="Specific example from output")
    impact: str = Field(description="Why this matters")


class Feedback(BaseModel):
    """Feedback for a single test case output."""
    score: float = Field(ge=0, le=100, description="Score from 0-100")
    critique: str = Field(description="Natural language explanation")
    issues: List[Issue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list, description="Specific prompt changes to fix issues")


class BatchFeedback(BaseModel):
    """Aggregated feedback across multiple test cases."""
    test_case_feedbacks: List[Feedback] = Field(description="Feedback for each test case")
    average_score: float = Field(ge=0, le=100)
    critical_issues: List[Issue] = Field(default_factory=list)
    high_priority_suggestions: List[str] = Field(default_factory=list)
    
    def get_consolidated_critique(self) -> str:
        """Generate a consolidated critique from all test cases."""
        critiques = [f"Test {i+1}: {fb.critique}" for i, fb in enumerate(self.test_case_feedbacks)]
        return "\n\n".join(critiques)
