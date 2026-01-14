"""Output specification models."""
from typing import List, Optional, Callable, Any, Dict
from pydantic import BaseModel, Field


class Rule(BaseModel):
    """A rule for output validation."""
    rule_id: str
    name: str
    description: str
    severity: str = Field(description="CRITICAL | HIGH | MEDIUM | LOW")
    check_function: Optional[Callable] = None


class OutputSpecification(BaseModel):
    """Specification for expected output format and quality."""
    
    task_name: str
    task_description: str
    
    # Deterministic checks
    syntax_rules: List[Rule] = Field(default_factory=list)
    
    # LLM-judged quality
    semantic_rules: List[Rule] = Field(default_factory=list)
    
    # How to score 0-100
    scoring_rubric: str
    
    # Optional examples
    example_good_output: Optional[Any] = None
    example_bad_output: Optional[Any] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
