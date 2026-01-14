"""Optimization result models."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Iteration(BaseModel):
    """Results from a single optimization iteration."""
    iteration_number: int
    prompt: str
    average_score: float
    test_scores: List[float]
    feedback: "BatchFeedback"  # Forward reference
    timestamp: datetime = Field(default_factory=datetime.now)
    cost_usd: float = 0.0


class OptimizationResult(BaseModel):
    """Final result of an optimization run."""
    initial_prompt: str
    best_prompt: str
    best_score: float
    initial_score: float
    improvement: float = Field(description="Percentage improvement")
    iterations: List[Iteration]
    total_cost_usd: float = 0.0
    converged: bool = False
    convergence_reason: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata including enhancement results")
    current_prompt: Optional[str] = Field(default=None, description="Final prompt from last iteration")
    current_score: Optional[float] = Field(default=None, description="Final score from last iteration")
    
    @property
    def num_iterations(self) -> int:
        """Number of iterations performed."""
        return len(self.iterations)
