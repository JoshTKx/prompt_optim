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
        save_progress: bool = True,  # Enable periodic saving
        max_budget_usd: float = None  # Maximum budget per optimization run
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
            max_budget_usd: Maximum budget in USD per optimization run (defaults to config)
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
        
        # Budget tracking
        self.max_budget = max_budget_usd or OptimizationConfig.MAX_BUDGET_PER_OPTIMIZATION
        self.start_cost = 0.0
    
    @handle_errors(severity=ErrorSeverity.HIGH, log_error=True, reraise=True)
    def optimize(
        self,
        initial_prompt: str,
        test_case: CanonicalTestCase,
        specification: OutputSpecification,
        max_iterations: int = None,
        target_score: float = None,
        prompt_id: Optional[str] = None,
        run_id: Optional[str] = None,
        accept_threshold: float = 2.0
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
            accept_threshold: Minimum score improvement required to accept a new prompt (default: 2.0)
        
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
        start_time = time.time()
        
        # Track starting cost
        self.start_cost = self.llm_client.total_cost
        
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
        
        # Track best score and prompt for regression protection
        best_score = initial_score
        best_prompt = initial_prompt
        no_improvement_count = 0
        
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
                "Testing iteration",
                iteration=iteration_num,
                max_iterations=max_iterations,
                current_prompt_hash=hashlib.sha256(current_prompt.encode()).hexdigest()[:8]
            )
            metrics.increment("optimization.iteration.started", tags={"iteration": str(iteration_num)})
            
            # Update progress bar - show we're starting iteration
            progress_bar.set_description(f"Optimizing [iter {iteration_num}/{max_iterations}]")
            progress_bar.set_postfix_str("testing...")
            progress_bar.refresh()
            
            # STEP 1: TEST the current prompt
            test_results = self.tester.test_prompt(
                prompt=current_prompt,
                test_case=test_case,
                context=test_case.context
            )
            
            # STEP 1a: Check negative constraints FIRST (free pattern matching)
            # This runs before expensive checker to catch issues early
            violations_found = False
            violation_count = 0
            if self.use_negative_constraints and self.negative_constraints:
                for result in test_results:
                    if result.get("success") and result.get("output"):
                        violations = self.negative_constraints.check(result["output"])
                        if violations:
                            violations_found = True
                            violation_count += len(violations)
                            result["negative_constraint_violations"] = [v.name for v in violations]
                            logger.warning(
                                "Negative constraint violations detected",
                                violations=[v.name for v in violations],
                                severity=[v.severity for v in violations],
                                test_input=result.get("input", "")[:50]
                            )
            
            # STEP 1b: Apply checker ONLY if violations found or explicitly enabled
            # This saves cost by skipping checker when negative constraints are clean
            if self.use_checker and self.checker:
                if violations_found:
                    logger.info(
                        f"Negative constraints: {violation_count} violations found, running checker to auto-fix",
                        violation_count=violation_count
                    )
                    progress_bar.set_postfix_str("checking (fixing violations)...")
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
                                result["checker_triggered_by_violations"] = True
                                metrics.increment("checker.fixes_applied", tags={"iteration": str(iteration_num), "trigger": "violations"})
                else:
                    logger.debug(
                        "Negative constraints: clean, skipping checker",
                        iteration=iteration_num
                    )
                    metrics.increment("checker.skipped", tags={"reason": "no_violations", "iteration": str(iteration_num)})
            
            # STEP 2: EVALUATE results (get feedback)
            progress_bar.set_postfix_str("judging...")
            progress_bar.refresh()
            feedback = self.judge.evaluate_batch(test_results, specification)
            current_score = feedback.average_score
            
            # REGRESSION TESTING: Test against golden set if enabled
            regression_result = None
            if self.use_golden_set and self.golden_set and self.golden_set.regression_enabled:
                progress_bar.set_postfix_str("regression check...")
                progress_bar.refresh()
                regression_result = self.golden_set.test_regression(
                    prompt=current_prompt,
                    llm_client=self.llm_client,
                    judge=self.judge,
                    specification=specification,
                    cost_tracker=self.cost_tracker,
                    target_model=test_case.target_model
                )
            
            # Store the tested prompt before potentially reverting (for logging)
            tested_prompt = current_prompt
            previous_best_score = best_score
            improvement_accepted = False
            
            # STEP 3: DECIDE if this is better than best_prompt
            # Check both current_score improvement AND regression pass rate
            score_improved = current_score > best_score + accept_threshold
            regression_passed = True
            
            if regression_result and regression_result.get("enabled"):
                regression_pass_rate = regression_result.get("regression_pass_rate", 1.0)
                regression_passed = regression_pass_rate >= self.golden_set.regression_pass_rate
                
                if not regression_passed:
                    logger.warning(
                        f"Regression check failed: {regression_result['passing_golden']}/{regression_result['total_golden']} passed ({regression_pass_rate:.1%} < {self.golden_set.regression_pass_rate:.1%})",
                        passing=regression_result['passing_golden'],
                        total=regression_result['total_golden'],
                        pass_rate=regression_pass_rate,
                        threshold=self.golden_set.regression_pass_rate
                    )
            
            if score_improved and regression_passed:
                # Significant improvement - accept the new prompt
                best_prompt = current_prompt
                best_score = current_score
                no_improvement_count = 0
                improvement_accepted = True
                logger.info(
                    f"✓ ACCEPTED: {previous_best_score:.1f} → {current_score:.1f} (+{current_score - previous_best_score:.1f})",
                    iteration=iteration_num,
                    previous_best=previous_best_score,
                    new_score=current_score,
                    improvement=current_score - previous_best_score
                )
            else:
                # No significant improvement or regression failed - keep previous best
                rejection_reason = []
                if not score_improved:
                    rejection_reason.append(f"score {current_score:.1f} <= best {best_score:.1f} + threshold {accept_threshold:.1f}")
                if not regression_passed:
                    rejection_reason.append(f"regression check failed")
                
                no_improvement_count += 1
                logger.info(
                    f"✗ REJECTED: keeping best (score: {best_score:.1f}, current: {current_score:.1f}) - {', '.join(rejection_reason)}",
                    iteration=iteration_num,
                    best_score=best_score,
                    current_score=current_score,
                    threshold=accept_threshold,
                    no_improvement_count=no_improvement_count,
                    regression_passed=regression_passed
                )
                
                # Check for significant regression
                if current_score < best_score - 5:
                    logger.warning(
                        f"Significant regression detected: {best_score:.1f} → {current_score:.1f} (drop: {best_score - current_score:.1f})",
                        iteration=iteration_num,
                        best_score=best_score,
                        current_score=current_score,
                        regression_amount=best_score - current_score
                    )
                
                # Revert to best prompt for next iteration
                # After reversion, current_prompt == best_prompt, and feedback describes current_prompt
                current_prompt = best_prompt
            
            # Update progress bar with score
            progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f}")
            progress_bar.refresh()
            
            # Note: Golden set accumulation happens AFTER optimization completes
            # (not during iterations to avoid storing all iterations)
            
            # Calculate iteration cost (from LLM client total cost)
            total_cost_so_far = self.llm_client.total_cost - self.start_cost
            previous_total = sum(
                self.cost_tracker.by_iteration.get(i, 0.0) for i in range(1, iteration_num)
            )
            iteration_cost = total_cost_so_far - previous_total
            self.cost_tracker.add_cost("iteration", iteration_cost, iteration=iteration_num)
            
            # Check budget
            if total_cost_so_far > self.max_budget * 0.8:
                logger.warning(
                    f"⚠ Budget warning: ${total_cost_so_far:.2f} / ${self.max_budget:.2f} (80% threshold)",
                    total_cost=total_cost_so_far,
                    budget=self.max_budget,
                    iteration=iteration_num
                )
            
            if total_cost_so_far > self.max_budget:
                logger.error(
                    f"❌ Budget exceeded: ${total_cost_so_far:.2f} / ${self.max_budget:.2f}",
                    total_cost=total_cost_so_far,
                    budget=self.max_budget,
                    iteration=iteration_num
                )
                progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f} ⚠ budget exceeded")
                progress_bar.refresh()
                break
            
            # Get previous score for comparison
            previous_score = history.get_all()[-1].average_score if len(history.get_all()) > 0 else initial_score
            
            # Count issues by severity
            critical_count = len(feedback.critical_issues)
            high_count = len(feedback.high_priority_suggestions) if hasattr(feedback, 'high_priority_suggestions') else 0
            
            # Determine if accepted (use the flag set in regression protection logic)
            accepted = improvement_accepted
            
            # STEP 4: LOG the iteration
            # Structured iteration summary (print to stdout for visibility)
            print(f"""
{'='*60}
ITERATION {iteration_num}/{max_iterations}
{'='*60}
Scores:
  Current:  {current_score:.1f}
  Previous: {previous_score:.1f}
  Best:     {best_score:.1f}
  Change:   {current_score - previous_score:+.1f}

Issues Found:
  Critical: {critical_count}
  High:     {high_count}

Cost:
  This Iteration: ${iteration_cost:.4f}
  Total So Far:   ${total_cost_so_far:.4f} / ${self.max_budget:.2f}

Decision: {'✓ ACCEPTED (improved)' if accepted else '✗ REJECTED (no improvement)'}
{'='*60}
""")
            
            logger.info(
                "Iteration evaluation complete",
                iteration=iteration_num,
                score=current_score,
                critical_issues=critical_count,
                suggestions_count=len(feedback.high_priority_suggestions)
            )
            metrics.gauge("optimization.iteration.score", current_score, tags={"iteration": str(iteration_num)})
            metrics.histogram("optimization.score", current_score)
            metrics.gauge("optimization.iteration.cost", iteration_cost, tags={"iteration": str(iteration_num)})
            
            # Record iteration (use tested_prompt, not current_prompt which may have been reverted)
            iteration = Iteration(
                iteration_number=iteration_num,
                prompt=tested_prompt,
                average_score=current_score,
                test_scores=[fb.score for fb in feedback.test_case_feedbacks],
                feedback=feedback,
                cost_usd=self.cost_tracker.total_cost
            )
            history.add_iteration(iteration)
            
            # STEP 5: CHECK convergence criteria
            converged, reason = history.has_converged(
                improvement_threshold=OptimizationConfig.SCORE_IMPROVEMENT_THRESHOLD,
                window=OptimizationConfig.CONVERGENCE_WINDOW
            )
            
            # Periodic saving (before potential break)
            if self.save_progress and iteration_num % self.save_interval == 0:
                temp_result = OptimizationResult(
                    initial_prompt=initial_prompt,
                    best_prompt=best_prompt,
                    best_score=best_score,
                    initial_score=initial_score,
                    improvement=((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0,
                    iterations=history.get_all(),
                    total_cost_usd=self.cost_tracker.total_cost,
                    converged=converged,
                    convergence_reason=reason if converged else None,
                    current_prompt=current_prompt,
                    current_score=current_score,
                    metadata={
                        "saved_during_optimization": True,
                        "current_iteration": iteration_num,
                        "max_iterations": max_iterations,
                        "accept_threshold": accept_threshold,
                        "no_improvement_count": no_improvement_count
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
            
            # Early stopping if no improvement for 2 iterations
            if no_improvement_count >= 2:
                logger.info(
                    "Converged (no improvement for 2 iterations)",
                    iteration=iteration_num,
                    best_score=best_score,
                    no_improvement_count=no_improvement_count
                )
                metrics.increment("optimization.converged", tags={"reason": "no_improvement"})
                progress_bar.update(1)  # Update to show completion
                progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f} ✓ converged")
                progress_bar.refresh()
                break
            
            if converged:
                logger.info("Optimization converged", reason=reason, iteration=iteration_num)
                metrics.increment("optimization.converged", tags={"reason": reason})
                progress_bar.update(1)  # Update to show completion
                progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f} ✓ converged")
                progress_bar.refresh()
                break
            
            if best_score >= target_score:
                logger.info(
                    "Target score reached",
                    score=best_score,
                    target=target_score,
                    iteration=iteration_num
                )
                metrics.increment("optimization.target_reached")
                progress_bar.update(1)  # Update to show completion
                progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f} ✓ target reached")
                progress_bar.refresh()
                break
            
            # STEP 6: REVISE the current_prompt for the NEXT iteration
            # This happens at the END of the loop, after all decisions and convergence checks
            # At this point:
            # - current_prompt is the prompt we just tested (or best_prompt if we reverted)
            # - feedback describes the prompt we just tested
            # - If we accepted: current_prompt == tested_prompt, feedback describes it (correct pairing)
            # - If we rejected: current_prompt == best_prompt (after reversion), but feedback describes rejected prompt
            #   In this case, we use feedback from the best iteration (which tested best_prompt) for consistency
            # - The revised prompt will be tested in the next iteration
            if iteration_num < max_iterations:  # Don't revise on last iteration
                # Determine which feedback to use for revision
                revision_feedback = feedback
                revision_score = current_score
                
                # If we reverted (rejected the improvement), use feedback from best iteration
                # This ensures feedback-prompt consistency: we revise best_prompt using feedback on best_prompt
                if not improvement_accepted and current_prompt == best_prompt:
                    # Find the iteration that tested best_prompt (the one with best_score)
                    best_iteration = None
                    for iter in history.get_all():
                        if iter.prompt == best_prompt and abs(iter.average_score - best_score) < 0.01:
                            best_iteration = iter
                            break
                    
                    if best_iteration and best_iteration.feedback:
                        revision_feedback = best_iteration.feedback
                        revision_score = best_score
                        logger.debug(
                            "Using feedback from best iteration for revision",
                            best_iteration_num=best_iteration.iteration_number,
                            best_score=best_score
                        )
                    else:
                        # Fallback: use current feedback (describes rejected prompt, but provides useful context)
                        logger.debug(
                            "Using current feedback for revision (best iteration not found, feedback provides context)"
                        )
                
                logger.info(
                    "Revising for next iteration",
                    iteration=iteration_num,
                    next_iteration=iteration_num + 1,
                    prompt_being_revised_hash=hashlib.sha256(current_prompt.encode()).hexdigest()[:8],
                    feedback_source="best_iteration" if not improvement_accepted and current_prompt == best_prompt and revision_feedback != feedback else "current_iteration"
                )
                progress_bar.set_postfix_str("revising...")
                progress_bar.refresh()
                
                # Revise current_prompt using appropriate feedback
                # After reversion, current_prompt == best_prompt, and we use feedback on best_prompt
                # ALWAYS pass current_score to ensure score-based temperature (not iteration-based)
                revised_prompt = self.reviser.improve_prompt(
                    current_prompt=current_prompt,  # The prompt we're revising (best_prompt after reversion, or accepted prompt)
                    feedback=revision_feedback,  # Feedback on the prompt being revised (ensures consistency)
                    iteration=iteration_num,
                    history=history.get_all(),
                    current_score=revision_score  # Score of the prompt being revised (used for temperature strategy)
                )
                
                # The revised prompt becomes current_prompt for next iteration
                current_prompt = revised_prompt
                
                # Version the new prompt
                change_summary = f"Iteration {iteration_num}: Revision for iteration {iteration_num + 1}"
                new_version = version_manager.create_version(
                    prompt_id=prompt_id,
                    prompt_text=current_prompt,
                    metadata={
                        "iteration": iteration_num,
                        "next_iteration": iteration_num + 1,
                        "score": current_score,
                        "run_id": run_id
                    },
                    change_summary=change_summary
                )
                
                # Get diff
                if iteration_num > 1:
                    diff = version_manager.diff_versions(prompt_id, new_version.version - 1, new_version.version)
                    logger.debug("Prompt changed", changes=diff["lines_changed"], diff_preview=diff["diff"][:5])
            
            # Update progress bar to show iteration complete
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"score={current_score:.1f}, best={best_score:.1f}")
            progress_bar.refresh()
            
            iteration_duration = time.time() - iteration_start_time
            logger.info(
                "Iteration complete",
                iteration=iteration_num,
                new_prompt_length=len(current_prompt),
                duration_seconds=iteration_duration
            )
            metrics.histogram("optimization.iteration.duration", iteration_duration)
        
        # Close progress bar
        progress_bar.close()
        
        # Get best iteration (for backward compatibility, but we use tracked best_prompt/best_score)
        best_iteration = history.get_best_iteration()
        
        # Use tracked best_prompt and best_score (may differ from best_iteration if regression protection rejected improvements)
        final_best_prompt = best_prompt
        final_best_score = best_score
        
        # Calculate improvement
        improvement = ((final_best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        
        # Get current prompt and score from last iteration
        last_iteration = history.get_all()[-1] if history.get_all() else None
        final_current_prompt = last_iteration.prompt if last_iteration else current_prompt
        final_current_score = last_iteration.average_score if last_iteration else current_score
        
        result = OptimizationResult(
            initial_prompt=initial_prompt,
            best_prompt=final_best_prompt,
            best_score=final_best_score,
            initial_score=initial_score,
            improvement=improvement,
            iterations=history.get_all(),
            total_cost_usd=self.cost_tracker.total_cost,
            converged=converged,
            convergence_reason=reason if converged else None,
            current_prompt=final_current_prompt,
            current_score=final_current_score
        )
        
        total_duration = time.time() - start_time
        total_cost_final = self.llm_client.total_cost - self.start_cost
        
        # Multi-turn testing for Security/Adversarial categories
        # NOTE: Multi-turn testing runs AFTER optimization completes (post-optimization validation only)
        # If multi-turn robustness is critical, include multi-turn test cases in your main test suite
        # so the optimizer can improve prompts for multi-turn scenarios during optimization
        multi_turn_result = None
        if self.use_multi_turn and self.multi_turn_tester and test_case.category == "security":
            logger.warning(
                "Multi-turn testing is post-optimization validation only. "
                "To optimize for multi-turn scenarios, include multi-turn test cases in your main test suite.",
                test_case_id=test_case.id,
                category=test_case.category
            )
            logger.info("Running multi-turn test for security category")
            multi_turn_result = self.multi_turn_tester.test_multi_turn(
                prompt=final_best_prompt,  # Use tracked best prompt
                test_case=test_case,
                turns=3
            )
            metrics.record(
                "multi_turn.final_score",
                multi_turn_result["final_score"],
                tags={"test_case": test_case.id}
            )
        
        # Golden set accumulation: Add best result if it meets threshold
        # This runs AFTER optimization completes, storing only the best prompt
        golden_set_added = 0
        golden_set_updated = 0
        golden_set_skipped = 0
        if self.use_golden_set and self.golden_set and final_best_score >= self.golden_set.threshold:
            # Re-test the best prompt to get actual outputs for golden set
            # This ensures we have the actual outputs that achieved the score
            logger.info("Re-testing best prompt for golden set accumulation", prompt_hash=hash(final_best_prompt) % 10000)
            test_results = self.tester.test_prompt(
                prompt=final_best_prompt,
                test_case=test_case,
                context=test_case.context
            )
            
            # Find the test case that scored highest (or use first if all similar)
            best_test_idx = 0
            best_test_score = final_best_score
            if test_results and len(test_case.test_cases) > 0:
                # Use the first test case's input/output structure
                # In future, we could store all test cases, but for now store the first
                test_input_dict = {
                    "input": test_case.test_cases[0].input,
                    "test_case_index": 0
                }
                
                # Get the actual output from test results
                actual_output = test_results[0].get("output", "") if test_results else ""
                
                # Store expected_output as the actual output that achieved the score
                # (not the test case's expected_output spec)
                expected_output_dict = {
                    "actual_output": actual_output,
                    "score": final_best_score,
                    "test_case_spec": test_case.test_cases[0].expected_output.dict() if test_case.test_cases else {}
                }
                
                golden_set_result = self.golden_set.add_or_update(
                    test_case_id=test_case.id,
                    test_input=test_input_dict,
                    expected_output=expected_output_dict,
                    best_prompt=final_best_prompt,
                    score=final_best_score,
                    optimization_run_id=run_id
                )
                
                if golden_set_result == "added":
                    golden_set_added = 1
                elif golden_set_result == "updated":
                    golden_set_updated = 1
                else:
                    golden_set_skipped = 1
                
                logger.info(
                    f"Golden set: {golden_set_result}",
                    test_case_id=test_case.id,
                    score=final_best_score
                )
        
        # Golden set statistics
        golden_stats = None
        if self.use_golden_set and self.golden_set:
            golden_stats = self.golden_set.get_statistics()
            golden_stats.update({
                "added_this_run": golden_set_added,
                "updated_this_run": golden_set_updated,
                "skipped_this_run": golden_set_skipped
            })
            logger.info("Golden set statistics", **golden_stats)
        
        # Final summary
        print(f"""
{'='*60}
OPTIMIZATION COMPLETE
{'='*60}
Results:
  Initial Score:  {initial_score:.1f}
  Final Score:    {final_best_score:.1f}
  Current Score:  {final_current_score:.1f}
  Improvement:    {improvement:+.1f}%
  
Performance:
  Total Iterations: {result.num_iterations}
  Converged:        {'Yes' if converged else 'No'}
  Convergence Reason: {reason if converged else 'N/A'}
  
Cost & Time:
  Total Cost:  ${total_cost_final:.4f}
  Budget:      ${self.max_budget:.2f}
  Total Time:  {total_duration:.1f}s
  Avg/Iteration: ${total_cost_final/result.num_iterations:.4f} (${total_duration/result.num_iterations:.1f}s)
  
Enhancements Used:
  Golden Set:        {'Yes' if self.use_golden_set else 'No'}
  Negative Constraints: {'Yes' if self.use_negative_constraints else 'No'}
  Checker:          {'Yes' if self.use_checker else 'No'}
  Multi-turn:       {'Yes' if self.use_multi_turn else 'No'}
  
Golden Set:
  Total Entries:    {golden_stats.get('total_entries', 0) if golden_stats else 0}
  Added This Run:   {golden_stats.get('added_this_run', 0) if golden_stats else 0}
  Updated This Run:  {golden_stats.get('updated_this_run', 0) if golden_stats else 0}
  Threshold:        {golden_stats.get('threshold', 85.0) if golden_stats else 85.0}
{'='*60}
""")
        
        logger.info(
            "Optimization complete",
            best_score=final_best_score,
            current_score=final_current_score,
            initial_score=initial_score,
            improvement=improvement,
            iterations=result.num_iterations,
            total_cost=total_cost_final,
            duration_seconds=total_duration,
            converged=converged,
            prompt_id=prompt_id,
            run_id=run_id,
            accept_threshold=accept_threshold,
            enhancements={
                "golden_set": self.use_golden_set,
                "negative_constraints": self.use_negative_constraints,
                "checker": self.use_checker,
                "multi_turn": self.use_multi_turn
            }
        )
        
        metrics.gauge("optimization.final_score", final_best_score)
        metrics.gauge("optimization.current_score", final_current_score)
        metrics.gauge("optimization.improvement", improvement)
        metrics.gauge("optimization.total_cost", total_cost_final)
        metrics.histogram("optimization.duration", total_duration)
        metrics.increment("optimization.completed", tags={"converged": str(converged)})
        
        # Add enhancement results to result metadata
        if not hasattr(result, 'metadata') or result.metadata is None:
            result.metadata = {}
        result.metadata.update({
            "golden_set_stats": golden_stats,
            "multi_turn_result": multi_turn_result,
            "accept_threshold": accept_threshold,
            "no_improvement_count": no_improvement_count,
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
        
        # Export metrics
        try:
            metrics_path = metrics.export()
            logger.info("Metrics exported", filepath=metrics_path)
        except Exception as e:
            logger.warning(f"Failed to export metrics: {e}")
        
        return result
