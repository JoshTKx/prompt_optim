"""Canonical test case models."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ExpectedOutput(BaseModel):
    """Expected output specification for a test case."""
    type: str = Field(description="structured | unstructured | mixed")
    format: Optional[str] = Field(None, description="json | xml | text | code")
    validation: Optional[str] = Field(None, description="valid_json_parse | constraint_checker | etc")
    constraints: List[str] = Field(default_factory=list)


class TestCaseInput(BaseModel):
    """A single test input/output pair."""
    input: str
    expected_output: ExpectedOutput


class EvaluationConfig(BaseModel):
    """Evaluation configuration for a test case."""
    method: str = Field(description="deterministic | llm_judge | hybrid")
    validators: List[str] = Field(default_factory=list)
    rubric: str = Field(default="five_point_standard")
    pass_criteria: Dict[str, Any] = Field(default_factory=dict)


class CanonicalTestCase(BaseModel):
    """A canonical test case from the test suite."""
    id: str
    name: str
    category: str
    archetype: str
    difficulty: str
    
    task: Dict[str, Any] = Field(description="Task configuration including initial_prompt")
    test_cases: List[TestCaseInput]
    evaluation: EvaluationConfig
    optimization_challenge: str = Field(description="What makes this test challenging")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def initial_prompt(self) -> str:
        """Extract initial prompt from task."""
        return self.task.get("initial_prompt", "")
    
    @property
    def target_model(self) -> str:
        """Extract target model from task."""
        model = self.task.get("target_model", "openai/gpt-4o-mini")
        # Convert legacy model names to OpenRouter format
        if model == "gpt-4o-mini":
            model = "openai/gpt-4o-mini"
        elif model == "deepseek-chat":
            model = "deepseek/deepseek-chat"
        elif model.startswith("claude") and not model.startswith("anthropic/"):
            model = f"anthropic/{model}"
        return model
    
    @property
    def context(self) -> Optional[str]:
        """Extract context from task if present."""
        return self.task.get("context")
