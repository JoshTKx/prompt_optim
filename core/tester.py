"""Tester component - runs prompts on test cases."""
from typing import List, Dict, Any
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.test_case import CanonicalTestCase
from utils.logging_utils import setup_logging

logger = setup_logging()


class Tester:
    """Runs prompts on test cases and collects outputs."""
    
    def __init__(self, llm_client: LLMClient, cost_tracker: CostTracker):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
    
    def test_prompt(
        self,
        prompt: str,
        test_case: CanonicalTestCase,
        context: str = None
    ) -> List[Dict[str, Any]]:
        """
        Run prompt on all test inputs for a test case.
        
        Returns:
            List of outputs, one per test input
        """
        results = []
        target_model = test_case.target_model
        
        for test_input in test_case.test_cases:
            # Construct full prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\n{prompt}"
            
            # Add the specific test input
            user_input = f"{full_prompt}\n\nInput: {test_input.input}"
            
            try:
                # Generate output
                output = self.llm_client.complete(
                    model=target_model,
                    prompt=user_input,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Estimate cost (rough estimate - actual token counting would be better)
                # For now, approximate: prompt ~500 tokens, output ~200 tokens
                estimated_cost = self.llm_client.estimate_cost(
                    model=target_model,
                    prompt_tokens=500,
                    completion_tokens=200
                )
                self.cost_tracker.add_cost("target_model", estimated_cost)
                
                results.append({
                    "input": test_input.input,
                    "output": output,
                    "expected": test_input.expected_output.dict(),
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"Error testing prompt: {e}")
                results.append({
                    "input": test_input.input,
                    "output": None,
                    "expected": test_input.expected_output.dict(),
                    "success": False,
                    "error": str(e)
                })
        
        return results
