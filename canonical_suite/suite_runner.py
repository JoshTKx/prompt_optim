"""Runner for executing optimizer against canonical test suite."""
from typing import List, Dict, Any
from core.orchestrator import Orchestrator
from models.test_case import CanonicalTestCase
from models.specification import OutputSpecification, Rule
from canonical_suite.suite_loader import CanonicalTestSuite
from canonical_suite.evaluator import TestEvaluator
from utils.logging_utils import setup_logging

logger = setup_logging()


class SuiteRunner:
    """Runs optimizer against canonical test suite."""
    
    def __init__(self):
        self.optimizer = Orchestrator()
        self.evaluator = TestEvaluator()
    
    def run_test_case(
        self,
        test_case: CanonicalTestCase
    ) -> Dict[str, Any]:
        """
        Run optimizer on a single test case.
        
        Returns:
            {
                "test_id": str,
                "initial_score": float,
                "optimized_score": float,
                "improvement": float,
                "passed": bool,
                "result": OptimizationResult
            }
        """
        logger.info(f"Running test case: {test_case.id} - {test_case.name}")
        
        # Create specification from test case
        specification = self._create_specification(test_case)
        
        # Run optimization
        result = self.optimizer.optimize(
            initial_prompt=test_case.initial_prompt,
            test_case=test_case,
            specification=specification
        )
        
        # Evaluate final result
        # (In practice, we'd re-run the optimized prompt)
        optimized_score = result.best_score
        
        # Check pass criteria
        min_score = test_case.evaluation.pass_criteria.get("minimum_score", 4) * 20
        passed = optimized_score >= min_score
        
        return {
            "test_id": test_case.id,
            "name": test_case.name,
            "category": test_case.category,
            "initial_score": result.initial_score,
            "optimized_score": optimized_score,
            "improvement": result.improvement,
            "passed": passed,
            "result": result
        }
    
    def _create_specification(self, test_case: CanonicalTestCase) -> OutputSpecification:
        """Create OutputSpecification from test case."""
        # Extract rules from test case
        syntax_rules = []
        semantic_rules = []
        
        # Add rules based on expected outputs
        for tc in test_case.test_cases:
            expected = tc.expected_output
            
            if expected.format == "json":
                syntax_rules.append(Rule(
                    rule_id="json_format",
                    name="JSON Format",
                    description="Output must be valid JSON",
                    severity="CRITICAL"
                ))
            
            for constraint in expected.constraints:
                syntax_rules.append(Rule(
                    rule_id=f"constraint_{constraint}",
                    name=f"Constraint: {constraint}",
                    description=f"Must satisfy: {constraint}",
                    severity="HIGH" if "no_" in constraint else "MEDIUM"
                ))
        
        # Create rubric from evaluation config
        rubric = f"Evaluate based on: {test_case.evaluation.rubric}. "
        rubric += f"Method: {test_case.evaluation.method}. "
        rubric += f"Pass criteria: {test_case.evaluation.pass_criteria}"
        
        return OutputSpecification(
            task_name=test_case.name,
            task_description=test_case.optimization_challenge,
            syntax_rules=syntax_rules,
            semantic_rules=semantic_rules,
            scoring_rubric=rubric
        )
