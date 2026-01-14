"""Evaluator for canonical test cases using 5-point rubric."""
from typing import Dict, Any, List
from models.test_case import CanonicalTestCase
from models.specification import OutputSpecification, Rule
from utils.validators import OutputValidator
from utils.logging_utils import setup_logging

logger = setup_logging()


class TestEvaluator:
    """Evaluates test case outputs using deterministic and LLM-based methods."""
    
    def __init__(self):
        self.validator = OutputValidator()
    
    def evaluate(
        self,
        test_case: CanonicalTestCase,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate outputs for a test case.
        
        Returns:
            {
                "score": 0-100,
                "passed": bool,
                "details": {...}
            }
        """
        method = test_case.evaluation.method
        
        if method == "deterministic":
            return self._evaluate_deterministic(test_case, outputs)
        elif method == "llm_judge":
            return self._evaluate_llm_judge(test_case, outputs)
        else:
            # Hybrid: combine both
            det_result = self._evaluate_deterministic(test_case, outputs)
            llm_result = self._evaluate_llm_judge(test_case, outputs)
            
            # Combine scores (weighted average)
            combined_score = (det_result["score"] * 0.6) + (llm_result["score"] * 0.4)
            
            return {
                "score": combined_score,
                "passed": combined_score >= test_case.evaluation.pass_criteria.get("minimum_score", 4) * 20,
                "details": {
                    "deterministic": det_result,
                    "llm_judge": llm_result
                }
            }
    
    def _evaluate_deterministic(
        self,
        test_case: CanonicalTestCase,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate using deterministic validators."""
        scores = []
        errors = []
        
        for i, result in enumerate(outputs):
            if not result.get("success", False):
                scores.append(0)
                errors.append(f"Test {i+1}: Execution failed")
                continue
            
            output_text = result["output"]
            expected = result["expected"]
            
            # Run validators
            is_valid, validation_errors = self.validator.validate(output_text, expected)
            
            if is_valid:
                scores.append(100)
            else:
                scores.append(0)
                errors.extend([f"Test {i+1}: {e}" for e in validation_errors])
        
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = test_case.evaluation.pass_criteria.get("minimum_score", 4) * 20
        
        return {
            "score": avg_score,
            "passed": avg_score >= min_score,
            "errors": errors,
            "individual_scores": scores
        }
    
    def _evaluate_llm_judge(
        self,
        test_case: CanonicalTestCase,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate using LLM judge (placeholder - would use Judge component)."""
        # This would integrate with the Judge component
        # For now, return a placeholder
        return {
            "score": 50.0,  # Placeholder
            "passed": False,
            "note": "LLM judge evaluation not yet integrated"
        }
