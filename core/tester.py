"""Tester component - runs prompts on test cases."""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.test_case import CanonicalTestCase
from config.optimization_config import OptimizationConfig
from utils.logging_utils import setup_logging

logger = setup_logging()


class Tester:
    """Runs prompts on test cases and collects outputs."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        cost_tracker: CostTracker,
        max_parallel_tests: Optional[int] = None,
        enable_parallel: Optional[bool] = None
    ):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.max_parallel_tests = max_parallel_tests or OptimizationConfig.MAX_PARALLEL_TESTS
        self.enable_parallel = enable_parallel if enable_parallel is not None else OptimizationConfig.ENABLE_PARALLEL_TESTS
    
    def test_prompt(
        self,
        prompt: str,
        test_case: CanonicalTestCase,
        context: str = None,
        parallel: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Run prompt on all test inputs for a test case.
        
        Args:
            prompt: Prompt to test
            test_case: Test case with inputs
            context: Optional context
            parallel: Override parallel setting (defaults to instance setting)
        
        Returns:
            List of outputs, one per test input
        """
        use_parallel = parallel if parallel is not None else self.enable_parallel
        
        if use_parallel and len(test_case.test_cases) > 1:
            return self._test_prompt_parallel(prompt, test_case, context)
        else:
            return self._test_prompt_sequential(prompt, test_case, context)
    
    def _test_prompt_sequential(
        self,
        prompt: str,
        test_case: CanonicalTestCase,
        context: str = None
    ) -> List[Dict[str, Any]]:
        """Run tests sequentially (original behavior)."""
        results = []
        target_model = test_case.target_model
        
        # Add progress bar for test case execution
        for test_input in tqdm(
            test_case.test_cases,
            desc="Running test cases",
            unit="test",
            leave=False,
            disable=None  # Auto-detect TTY
        ):
            result = self._run_single_test(prompt, test_input, target_model, context)
            results.append(result)
        
        return results
    
    def _test_prompt_parallel(
        self,
        prompt: str,
        test_case: CanonicalTestCase,
        context: str = None
    ) -> List[Dict[str, Any]]:
        """Run tests in parallel."""
        target_model = test_case.target_model
        results = [None] * len(test_case.test_cases)
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_tests) as executor:
            # Submit all tests
            future_to_index = {
                executor.submit(
                    self._run_single_test,
                    prompt,
                    test_input,
                    target_model,
                    context
                ): idx
                for idx, test_input in enumerate(test_case.test_cases)
            }
            
            # Collect results in order
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel test execution: {e}")
                    test_input = test_case.test_cases[idx]
                    results[idx] = {
                        "input": test_input.input,
                        "output": None,
                        "expected": test_input.expected_output.dict(),
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _run_single_test(
        self,
        prompt: str,
        test_input,
        target_model: str,
        context: str = None
    ) -> Dict[str, Any]:
        """Run a single test input."""
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
            
            return {
                "input": test_input.input,
                "output": output,
                "expected": test_input.expected_output.dict(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error testing prompt: {e}")
            return {
                "input": test_input.input,
                "output": None,
                "expected": test_input.expected_output.dict(),
                "success": False,
                "error": str(e)
            }
