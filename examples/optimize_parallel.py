"""Example: Optimize multiple prompts in parallel."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.parallel_orchestrator import ParallelOrchestrator
from models.specification import OutputSpecification, Rule
from models.test_case import CanonicalTestCase, TestCaseInput, ExpectedOutput, EvaluationConfig

# Create multiple test cases
test_case_1 = CanonicalTestCase(
    id="PARALLEL-001",
    name="JSON Output",
    category="instruction_following",
    archetype="A3_constraint_heavy",
    difficulty="easy",
    task={
        "initial_prompt": "Generate a list of 3 fruits in JSON format.",
        "target_model": "google/gemini-3-flash-preview",
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
    optimization_challenge="Models often wrap JSON in markdown.",
    metadata={}
)

test_case_2 = CanonicalTestCase(
    id="PARALLEL-002",
    name="XML Output",
    category="instruction_following",
    archetype="A3_constraint_heavy",
    difficulty="easy",
    task={
        "initial_prompt": "Generate a list of 3 colors in XML format.",
        "target_model": "google/gemini-3-flash-preview",
        "context": None
    },
    test_cases=[
        TestCaseInput(
            input="List 3 colors",
            expected_output=ExpectedOutput(
                type="structured",
                format="xml",
                validation="valid_xml_parse",
                constraints=["valid_xml_structure"]
            )
        )
    ],
    evaluation=EvaluationConfig(
        method="deterministic",
        validators=["xml_parser"],
        rubric="five_point_standard",
        pass_criteria={"minimum_score": 4}
    ),
    optimization_challenge="Models must generate valid XML.",
    metadata={}
)

# Create specifications
spec_1 = OutputSpecification(
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
        )
    ],
    semantic_rules=[],
    scoring_rubric="Score 0-100 based on valid JSON (40 points), no markdown (30 points), correct length (30 points)"
)

spec_2 = OutputSpecification(
    task_name="XML Color List",
    task_description="Generate valid XML with 3 colors",
    syntax_rules=[
        Rule(
            rule_id="xml_format",
            name="XML Format",
            description="Output must be valid XML",
            severity="CRITICAL"
        )
    ],
    semantic_rules=[],
    scoring_rubric="Score 0-100 based on valid XML structure (100 points)"
)

# Run parallel optimization
if __name__ == "__main__":
    print("Starting parallel optimization...")
    
    # Initialize parallel orchestrator
    parallel_optimizer = ParallelOrchestrator(max_workers=2)
    
    # Prepare tasks
    tasks = [
        {
            "initial_prompt": test_case_1.initial_prompt,
            "test_case": test_case_1,
            "specification": spec_1,
            "max_iterations": 5,
            "prompt_id": "parallel_task_1"
        },
        {
            "initial_prompt": test_case_2.initial_prompt,
            "test_case": test_case_2,
            "specification": spec_2,
            "max_iterations": 5,
            "prompt_id": "parallel_task_2"
        }
    ]
    
    # Run optimizations in parallel
    results = parallel_optimizer.optimize_multiple(tasks)
    
    # Print results
    print(f"\n{'='*60}")
    print("PARALLEL OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        if result is None:
            print(f"\nTask {i}: FAILED")
            continue
        
        print(f"\nTask {i}:")
        print(f"  Initial Score: {result.initial_score:.1f}/100")
        print(f"  Best Score: {result.best_score:.1f}/100")
        print(f"  Improvement: {result.improvement:.1f}%")
        print(f"  Total Cost: ${result.total_cost_usd:.4f}")
        print(f"  Iterations: {result.num_iterations}")
