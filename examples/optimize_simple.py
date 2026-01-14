"""Example: Optimize a simple prompt."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestrator import Orchestrator
from models.specification import OutputSpecification, Rule
from models.test_case import CanonicalTestCase, TestCaseInput, ExpectedOutput, EvaluationConfig

# Create a simple test case
test_case = CanonicalTestCase(
    id="EXAMPLE-001",
    name="Simple JSON Output",
    category="instruction_following",
    archetype="A3_constraint_heavy",
    difficulty="easy",
    task={
        "initial_prompt": "Generate a list of 3 fruits in JSON format.",
        "target_model": "openai/gpt-4o-mini",
        "context": None
    },
    test_cases=[
        TestCaseInput(
            input="List 3 fruits",
            expected_output=ExpectedOutput(
                type="structured",
                format="json",
                validation="valid_json_parse",
                constraints=["no_markdown_fencing", "array_length_equals_3"]
            )
        )
    ],
    evaluation=EvaluationConfig(
        method="deterministic",
        validators=["json_parser", "constraint_checker"],
        rubric="five_point_standard",
        pass_criteria={"minimum_score": 4}
    ),
    optimization_challenge="Models often wrap JSON in markdown. Need explicit constraint.",
    metadata={}
)

# Create specification
specification = OutputSpecification(
    task_name="JSON Fruit List",
    task_description="Generate a JSON array of exactly 3 fruits without markdown",
    syntax_rules=[
        Rule(
            rule_id="json_format",
            name="JSON Format",
            description="Output must be valid JSON",
            severity="CRITICAL"
        ),
        Rule(
            rule_id="no_markdown",
            name="No Markdown",
            description="Do not wrap JSON in markdown code blocks",
            severity="HIGH"
        ),
        Rule(
            rule_id="array_length",
            name="Array Length",
            description="Array must contain exactly 3 items",
            severity="HIGH"
        )
    ],
    semantic_rules=[],
    scoring_rubric="Score 0-100 based on: valid JSON (40 points), no markdown (30 points), correct length (30 points)"
)

# Run optimization
if __name__ == "__main__":
    print("Starting optimization...")
    optimizer = Orchestrator()
    
    result = optimizer.optimize(
        initial_prompt=test_case.initial_prompt,
        test_case=test_case,
        specification=specification,
        max_iterations=5
    )
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Initial Score: {result.initial_score:.1f}/100")
    print(f"Best Score: {result.best_score:.1f}/100")
    print(f"Improvement: {result.improvement:.1f}%")
    print(f"Total Cost: ${result.total_cost_usd:.4f}")
    print(f"\nInitial Prompt:\n{result.initial_prompt}")
    print(f"\nOptimized Prompt:\n{result.best_prompt}")
    print(f"\nIterations: {result.num_iterations}")
