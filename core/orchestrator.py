"""Orchestrator - main optimization loop."""
import uuid
import time
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Import models to trigger model_rebuild
from models import BatchFeedback, Iteration, OptimizationResult
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.specification import OutputSpecification
from models.result import OptimizationResult, Iteration
from models.test_case import CanonicalTestCase
from core.tester import Tester
from core.judge import Judge
from core.reviser import Reviser
from core.history import OptimizationHistory
from config.optimization_config import OptimizationConfig
from utils.logging_utils import setup_logging, set_correlation_id
from utils.prompt_versioning import PromptVersionManager
from utils.metrics import get_metrics_collector
from utils.error_handling import handle_errors, ErrorSeverity
from utils.golden_set import GoldenSetManager
from utils.negative_constraints import get_negative_constraints_library
from utils.checker import Checker
from utils.multi_turn import MultiTurnTester
from utils.result_saver import save_optimization_result

logger = setup_logging()
metrics = get_metrics_collector()
version_manager = PromptVersionManager()


class Orchestrator:
    """Main orchestrator for prompt optimization."""
    
    def __init__(
        self,
        use_golden_set: bool = True,
        use_negative_constraints: bool = True,
        use_checker: bool = True,
        use_multi_turn: bool = False,  # Only for Security/Adversarial categories
        save_interval: int = 1,  # Save after every N iterations (1 = every iteration)
        save_progress: bool = True  # Enable periodic saving
    ):
        """
        Initialize orchestrator with enhancement options.
        
        Args:
            use_golden_set: Use dynamic golden set for examples
            use_negative_constraints: Apply negative constraints
            use_checker: Use checker prompt for output validation
            use_multi_turn: Use multi-turn testing (expensive, for Security category)
            save_interval: Save progress after every N iterations (default: 1 = every iteration)
            save_progress: Enable periodic saving of optimization state
        """
        self.llm_client = LLMClient()
        self.cost_tracker = CostTracker()
        self.tester = Tester(self.llm_client, self.cost_tracker)
        self.judge = Judge(self.llm_client, self.cost_tracker)
        self.reviser = Reviser(self.llm_client, self.cost_tracker)
        
        # Enhancements
        self.use_golden_set = use_golden_set
        self.use_negative_constraints = use_negative_constraints
        self.use_checker = use_checker
        self.use_multi_turn = use_multi_turn
        
        if use_golden_set:
            self.golden_set = GoldenSetManager()
        else:
            self.golden_set = None
        
        if use_negative_constraints:
            self.negative_constraints = get_negative_constraints_library()
        else:
            self.negative_constraints = None
        
        if use_checker:
            self.checker = Checker(self.llm_client, self.cost_tracker)
        else:
            self.checker = None
        
        if use_multi_turn:
            self.multi_turn_tester = MultiTurnTester(self.llm_client, self.cost_tracker)
        else:
            self.multi_turn_tester = None
        
        # Progress saving
        self.save_interval = save_interval
        self.save_progress = save_progress
    
    @handle_errors(severity=ErrorSeverity.HIGH, log_error=True, reraise=True)
    def optimize(
        self,
        initial_prompt: str,
        test_case: CanonicalTestCase,
        specification: OutputSpecification,
        max_iterations: int = None,
        target_score: float = None,
        prompt_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> OptimizationResult:
        """
        Optimize a prompt through iterative improvement.
        
        Args:
            initial_prompt: Starting prompt
            test_case: Canonical test case to optimize against
            specification: Output specification
            max_iterations: Maximum iterations (defaults to config)
            target_score: Target score to reach (defaults to config)
            prompt_id: Optional prompt identifier for versioning
            run_id: Optional run identifier for correlation
        
        Returns:
            OptimizationResult with best prompt and trajectory
        """
        # Set up correlation ID for request tracking
        if run_id is None:
            run_id = str(uuid.uuid4())
        set_correlation_id(run_id)
        
        # Generate prompt_id if not provided
        if prompt_id is None:
            prompt_id = f"prompt_{test_case.id}_{run_id[:8]}"
        
        max_iterations = max_iterations or OptimizationConfig.MAX_ITERATIONS
        target_score = target_score or OptimizationConfig.TARGET_SCORE
        
        # Use config values for saving if not explicitly set
        if not hasattr(self, 'save_interval'):
            self.save_interval = OptimizationConfig.SAVE_INTERVAL
        if not hasattr(self, 'save_progress'):
            self.save_progress = OptimizationConfig.SAVE_PROGRESS
        
        history = OptimizationHistory()
        current_prompt = initial_prompt
        
        # Version initial prompt
        initial_version = version_manager.create_version(
            prompt_id=prompt_id,
            prompt_text=initial_prompt,
            metadata={
                "test_case_id": test_case.id,
                "test_case_name": test_case.name,
                "run_id": run_id
            },
            change_summary="Initial prompt"
        )
        
        logger.info(
            "Starting optimization",
            prompt_id=prompt_id,
            run_id=run_id,
            test_case_id=test_case.id,
            max_iterations=max_iterations,
            target_score=target_score,
            initial_prompt_length=len(initial_prompt)
        )
        metrics.increment("optimization.started", tags={"test_case": test_case.id})
        
        # Initial evaluation
        initial_results = self.tester.test_prompt(
            prompt=initial_prompt,
            test_case=test_case,
            context=test_case.context
        )
        initial_feedback = self.judge.evaluate_batch(initial_results, specification)
        initial_score = initial_feedback.average_score
        
        logger.info(
            "Initial evaluation complete",
            initial_score=initial_score,
            test_results_count=len(initial_results)
        )
        metrics.gauge("optimization.initial_score", initial_score, tags={"test_case": test_case.id})
        
        # Track best score for progress bar
        best_score_so_far = initial_score
        best_prompt_so_far = initial_prompt
        
        # Optimization loop with progress bar
        # Use disable=None to auto-detect if we're in a TTY
        # Use leave=True to keep the bar after completion
        progress_bar = tqdm(
            total=max_iterations,
            initial=0,
            desc=f"Optimizing",
            unit="iter",
            ncols=120,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, score={postfix}]",
            file=sys.stdout,  # Explicitly use stdout (logging uses stderr)
            disable=None  # Auto-detect TTY
        )
        progress_bar.set_postfix_str(f"initial={initial_score:.1f}")
        
        iteration_num = 0
        while iteration_num < max_iterations:
            iteration_num += 1
            iteration_start_time = time.time()
            
            logger.info(
                "Starting iteration",
                iteration=iteration_num,
                max_iterations=max_iterations,
                current_prompt_hash=hashlib.sha256(current_prompt.encode()).hexdigest()[:8]
            )
            metrics.increment("optimization.iteration.started", tags={"iteration": str(iteration_num)})
            
            # Update progress bar - show we're starting iteration
            progress_bar.set_description(f"Optimizing [iter {iteration_num}/{max_iterations}]")
            progress_bar.set_postfix_str("testing...")
            progress_bar.refresh()
            
            # Test current prompt
            test_results = self.tester.test_prompt(
                prompt=current_prompt,
                test_case=test_case,
                context=test_case.context
            )
            
            # Apply checker if enabled
            if self.use_checker and self.checker:
                progress_bar.set_postfix_str("checking...")
                progress_bar.refresh()
                for result in test_results:
                    if result.get("success") and result.get("output"):
                        checked = self.checker.check(
                            output=result["output"],
                            specification=specification,
                            test_input=result.get("input"),
                            auto_fix=True
                        )
                        if checked["fixed"]:
                            result["output"] = checked["checked"]
                            result["checker_fixed"] = True
                            metrics.increment("checker.fixes_applied", tags={"iteration": str(iteration_num)})
            
            # Check negative constraints
            if self.use_negative_constraints and self.negative_constraints:
                for result in test_results:
                    if result.get("success") and result.get("output"):
                        violations = self.negative_constraints.check(result["output"])
                        if violations:
                            result["negative_constraint_violations"] = [v.name for v in violations]
                            logger.warning(
                                "Negative constraint violations",
                                violations=[v.name for v in violations],
                                severity=[v.severity for v in violations]
                            )
            
            # Judge outputs
            progress_bar.set_postfix_str("judging...")
            progress_bar.refresh()
            feedback = self.judge.evaluate_batch(test_results, specification)
            current_score = feedback.average_score
            
            # Update progress bar with score
            if current_score > best_score_so_far:
                best_score_so_far = current_score
                best_prompt_so_far = current_prompt
            
            progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score_so_far:.1f}")
            progress_bar.refresh()
            
            # Capture golden examples
            if self.use_golden_set and self.golden_set:
                for i, result in enumerate(test_results):
                    if result.get("success") and result.get("output"):
                        score = feedback.test_case_feedbacks[i].score if i < len(feedback.test_case_feedbacks) else current_score
                        self.golden_set.capture(
                            test_case_id=test_case.id,
                            prompt=current_prompt,
                            input_text=result.get("input", ""),
                            output=result["output"],
                            score=score,
                            metadata={"iteration": iteration_num, "run_id": run_id}
                        )
            
            logger.info(
                "Iteration evaluation complete",
                iteration=iteration_num,
                score=current_score,
                critical_issues=len(feedback.critical_issues),
                suggestions_count=len(feedback.high_priority_suggestions)
            )
            metrics.gauge("optimization.iteration.score", current_score, tags={"iteration": str(iteration_num)})
            metrics.histogram("optimization.score", current_score)
            
            # Record iteration
            iteration = Iteration(
                iteration_number=iteration_num,
                prompt=current_prompt,
                average_score=current_score,
                test_scores=[fb.score for fb in feedback.test_case_feedbacks],
                feedback=feedback,
                cost_usd=self.cost_tracker.total_cost
            )
            history.add_iteration(iteration)
            
            # Check convergence
            converged, reason = history.has_converged(
                improvement_threshold=OptimizationConfig.SCORE_IMPROVEMENT_THRESHOLD,
                window=OptimizationConfig.CONVERGENCE_WINDOW
            )
            
            # Periodic saving (before potential break)
            if self.save_progress and iteration_num % self.save_interval == 0:
                temp_result = OptimizationResult(
                    initial_prompt=initial_prompt,
                    best_prompt=best_prompt_so_far,
                    best_score=best_score_so_far,
                    initial_score=initial_score,
                    improvement=((best_score_so_far - initial_score) / initial_score * 100) if initial_score > 0 else 0,
                    iterations=history.get_all(),
                    total_cost_usd=self.cost_tracker.total_cost,
                    converged=converged,
                    convergence_reason=reason if converged else None,
                    metadata={
                        "saved_during_optimization": True,
                        "current_iteration": iteration_num,
                        "max_iterations": max_iterations
                    }
                )
                try:
                    save_path = save_optimization_result(
                        result=temp_result,
                        run_id=run_id,
                        prompt_id=prompt_id
                    )
                    logger.debug("Progress saved", filepath=str(save_path), iteration=iteration_num)
                except Exception as e:
                    logger.warning(f"Failed to save progress: {e}", iteration=iteration_num)
            
            if converged:
                logger.info("Optimization converged", reason=reason, iteration=iteration_num)
                metrics.increment("optimization.converged", tags={"reason": reason})
                progress_bar.update(1)  # Update to show completion
                progress_bar.set_postfix_str(f"score={current_score:.1f} ✓ converged")
                progress_bar.refresh()
                break
            
            if current_score >= target_score:
                logger.info(
                    "Target score reached",
                    score=current_score,
                    target=target_score,
                    iteration=iteration_num
                )
                metrics.increment("optimization.target_reached")
                progress_bar.update(1)  # Update to show completion
                progress_bar.set_postfix_str(f"score={current_score:.1f} ✓ target reached")
                progress_bar.refresh()
                break
            
            # Improve prompt
            if iteration_num < max_iterations:  # Don't revise on last iteration
                progress_bar.set_postfix_str("revising...")
                progress_bar.refresh()
                previous_prompt = current_prompt
                current_prompt = self.reviser.improve_prompt(
                    current_prompt=current_prompt,
                    feedback=feedback,
                    iteration=iteration_num,
                    history=history.get_all()
                )
            
            # Update progress bar to show iteration complete
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score_so_far:.1f}")
            progress_bar.refresh()
            
            # Version the new prompt
            change_summary = f"Iteration {iteration_num}: {reason if converged else 'Improvement based on feedback'}"
            new_version = version_manager.create_version(
                prompt_id=prompt_id,
                prompt_text=current_prompt,
                metadata={
                    "iteration": iteration_num,
                    "score": current_score,
                    "run_id": run_id
                },
                change_summary=change_summary
            )
            
            # Get diff
            if iteration_num > 1:
                diff = version_manager.diff_versions(prompt_id, new_version.version - 1, new_version.version)
                logger.debug("Prompt changed", changes=diff["lines_changed"], diff_preview=diff["diff"][:5])
            
            iteration_duration = time.time() - iteration_start_time
            logger.info(
                "Iteration complete",
                iteration=iteration_num,
                new_prompt_length=len(current_prompt),
                duration_seconds=iteration_duration,
                prompt_version=new_version.version
            )
            metrics.histogram("optimization.iteration.duration", iteration_duration)
        
        # Close progress bar
        progress_bar.close()
        
        # Get best iteration
        best_iteration = history.get_best_iteration()
        
        # Calculate improvement
        improvement = ((best_iteration.average_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        
        result = OptimizationResult(
            initial_prompt=initial_prompt,
            best_prompt=best_iteration.prompt,
            best_score=best_iteration.average_score,
            initial_score=initial_score,
            improvement=improvement,
            iterations=history.get_all(),
            total_cost_usd=self.cost_tracker.total_cost,
            converged=converged,
            convergence_reason=reason if converged else None
        )
        
        total_duration = time.time() - (history.get_all()[0].timestamp.timestamp() if history.get_all() else time.time())
        
        # Multi-turn testing for Security/Adversarial categories
        multi_turn_result = None
        if self.use_multi_turn and self.multi_turn_tester and test_case.category == "security":
            logger.info("Running multi-turn test for security category")
            multi_turn_result = self.multi_turn_tester.test_multi_turn(
                prompt=best_iteration.prompt,
                test_case=test_case,
                turns=3
            )
            metrics.record(
                "multi_turn.final_score",
                multi_turn_result["final_score"],
                tags={"test_case": test_case.id}
            )
        
        # Golden set statistics
        golden_stats = None
        if self.use_golden_set and self.golden_set:
            golden_stats = self.golden_set.get_statistics()
            logger.info("Golden set statistics", **golden_stats)
        
        logger.info(
            "Optimization complete",
            best_score=best_iteration.average_score,
            initial_score=initial_score,
            improvement=improvement,
            iterations=result.num_iterations,
            total_cost=self.cost_tracker.total_cost,
            duration_seconds=total_duration,
            converged=converged,
            prompt_id=prompt_id,
            run_id=run_id,
            enhancements={
                "golden_set": self.use_golden_set,
                "negative_constraints": self.use_negative_constraints,
                "checker": self.use_checker,
                "multi_turn": self.use_multi_turn
            }
        )
        
        metrics.gauge("optimization.final_score", best_iteration.average_score)
        metrics.gauge("optimization.improvement", improvement)
        metrics.gauge("optimization.total_cost", self.cost_tracker.total_cost)
        metrics.histogram("optimization.duration", total_duration)
        metrics.increment("optimization.completed", tags={"converged": str(converged)})
        
        # Add enhancement results to result metadata
        if not hasattr(result, 'metadata') or result.metadata is None:
            result.metadata = {}
        result.metadata.update({
            "golden_set_stats": golden_stats,
            "multi_turn_result": multi_turn_result,
            "enhancements_used": {
                "golden_set": self.use_golden_set,
                "negative_constraints": self.use_negative_constraints,
                "checker": self.use_checker,
                "multi_turn": self.use_multi_turn
            }
        })
        
        # Final save
        if self.save_progress:
            try:
                final_save_path = save_optimization_result(
                    result=result,
                    run_id=run_id,
                    prompt_id=prompt_id
                )
                logger.info("Final result saved", filepath=str(final_save_path))
            except Exception as e:
                logger.warning(f"Failed to save final result: {e}")
        
        return result
