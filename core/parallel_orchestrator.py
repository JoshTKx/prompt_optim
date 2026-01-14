"""Parallel orchestrator for running multiple optimizations simultaneously."""
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from core.orchestrator import Orchestrator
from models.specification import OutputSpecification
from models.test_case import CanonicalTestCase
from models.result import OptimizationResult
from config.optimization_config import OptimizationConfig
from utils.logging_utils import setup_logging, set_correlation_id
from tqdm import tqdm

logger = setup_logging()


class ParallelOrchestrator:
    """Orchestrator that can run multiple optimizations in parallel."""
    
    def __init__(
        self,
        max_workers: int = None,
        **orchestrator_kwargs  # Pass through to Orchestrator
    ):
        """
        Initialize parallel orchestrator.
        
        Args:
            max_workers: Number of parallel optimizations (defaults to config)
            **orchestrator_kwargs: Arguments to pass to each Orchestrator instance
        """
        self.max_workers = max_workers or OptimizationConfig.MAX_PARALLEL_OPTIMIZATIONS
        self.orchestrator_kwargs = orchestrator_kwargs
        
        logger.info(
            "ParallelOrchestrator initialized",
            max_workers=self.max_workers,
            orchestrator_options=list(orchestrator_kwargs.keys())
        )
    
    def optimize_multiple(
        self,
        optimization_tasks: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, OptimizationResult], None]] = None
    ) -> List[OptimizationResult]:
        """
        Optimize multiple prompts in parallel.
        
        Args:
            optimization_tasks: List of dicts with keys:
                - initial_prompt: str
                - test_case: CanonicalTestCase
                - specification: OutputSpecification
                - max_iterations: int (optional)
                - target_score: float (optional)
                - prompt_id: str (optional)
                - run_id: str (optional)
            progress_callback: Optional callback function(task_id, result) called when each task completes
        
        Returns:
            List of OptimizationResult objects (in same order as input tasks)
        """
        if not optimization_tasks:
            return []
        
        logger.info(
            "Starting parallel optimization",
            num_tasks=len(optimization_tasks),
            max_workers=self.max_workers
        )
        
        # Create results list with None placeholders to maintain order
        results: List[Optional[OptimizationResult]] = [None] * len(optimization_tasks)
        completed_count = 0
        
        # Use ThreadPoolExecutor for I/O-bound LLM API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for idx, task in enumerate(optimization_tasks):
                # Generate run_id if not provided
                run_id = task.get("run_id") or str(uuid.uuid4())
                
                # Create orchestrator instance for this task
                orchestrator = Orchestrator(**self.orchestrator_kwargs)
                
                # Submit task
                future = executor.submit(
                    self._run_optimization,
                    orchestrator=orchestrator,
                    initial_prompt=task["initial_prompt"],
                    test_case=task["test_case"],
                    specification=task["specification"],
                    max_iterations=task.get("max_iterations"),
                    target_score=task.get("target_score"),
                    prompt_id=task.get("prompt_id"),
                    run_id=run_id
                )
                future_to_index[future] = idx
            
            # Progress bar for parallel execution
            with tqdm(total=len(optimization_tasks), desc="Parallel Optimization", unit="task") as pbar:
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    task = optimization_tasks[idx]
                    task_id = task.get("prompt_id") or task.get("test_case", {}).id if hasattr(task.get("test_case"), "id") else f"task_{idx}"
                    
                    try:
                        result = future.result()
                        results[idx] = result
                        completed_count += 1
                        
                        logger.info(
                            "Parallel optimization completed",
                            task_id=task_id,
                            index=idx,
                            completed=completed_count,
                            total=len(optimization_tasks),
                            final_score=result.best_score if result else None
                        )
                        
                        # Call progress callback if provided
                        if progress_callback and result:
                            progress_callback(task_id, result)
                        
                        pbar.update(1)
                        pbar.set_postfix_str(f"Completed: {completed_count}/{len(optimization_tasks)}")
                        
                    except Exception as e:
                        logger.error(
                            f"Optimization failed for task {task_id}",
                            error=str(e),
                            index=idx,
                            exc_info=True
                        )
                        # Store failed result (None) to maintain order
                        results[idx] = None
                        completed_count += 1
                        pbar.update(1)
        
        # Filter out None results (failed tasks) or keep them for error tracking
        # For now, we'll keep None to maintain order and let caller handle failures
        return results
    
    @staticmethod
    def _run_optimization(
        orchestrator: Orchestrator,
        initial_prompt: str,
        test_case: CanonicalTestCase,
        specification: OutputSpecification,
        max_iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        prompt_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> OptimizationResult:
        """Run a single optimization task."""
        return orchestrator.optimize(
            initial_prompt=initial_prompt,
            test_case=test_case,
            specification=specification,
            max_iterations=max_iterations,
            target_score=target_score,
            prompt_id=prompt_id,
            run_id=run_id
        )
    
    def optimize_batch(
        self,
        test_cases: List[CanonicalTestCase],
        specifications: List[OutputSpecification],
        initial_prompts: Optional[List[str]] = None,
        max_iterations: Optional[int] = None,
        target_score: Optional[float] = None
    ) -> List[OptimizationResult]:
        """
        Convenience method to optimize a batch of test cases.
        
        Args:
            test_cases: List of test cases to optimize
            specifications: List of specifications (one per test case)
            initial_prompts: Optional list of initial prompts (uses test_case.initial_prompt if not provided)
            max_iterations: Max iterations for all optimizations
            target_score: Target score for all optimizations
        
        Returns:
            List of OptimizationResult objects
        """
        if len(test_cases) != len(specifications):
            raise ValueError("test_cases and specifications must have the same length")
        
        tasks = []
        for i, (test_case, specification) in enumerate(zip(test_cases, specifications)):
            initial_prompt = initial_prompts[i] if initial_prompts and i < len(initial_prompts) else test_case.initial_prompt
            
            tasks.append({
                "initial_prompt": initial_prompt,
                "test_case": test_case,
                "specification": specification,
                "max_iterations": max_iterations,
                "target_score": target_score,
                "prompt_id": f"prompt_{test_case.id}"
            })
        
        return self.optimize_multiple(tasks)
