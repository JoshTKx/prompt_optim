"""Reviser component - improves prompts based on feedback."""
from pathlib import Path
from typing import List, Optional, Dict, Any
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.feedback import BatchFeedback
from models.result import Iteration
from config.llm_config import LLMConfig
from config.optimization_config import OptimizationConfig
from utils.logging_utils import setup_logging
from utils.system_prompt_manager import SystemPromptManager

logger = setup_logging()
prompt_manager = SystemPromptManager()


class Reviser:
    """Improves prompts based on judge feedback."""
    
    def __init__(self, llm_client: LLMClient, cost_tracker: CostTracker):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.model = LLMConfig.REVISER_MODEL
        
        # Load reviser system prompt with versioning
        prompt_path = Path(__file__).parent.parent / "prompts" / "reviser_system.txt"
        self.system_prompt, self.prompt_version, self.prompt_hash = prompt_manager.load_prompt(
            prompt_name="reviser_system",
            prompt_path=prompt_path
        )
        
        logger.info(
            "Reviser initialized",
            model=self.model,
            prompt_version=self.prompt_version,
            prompt_hash=self.prompt_hash
        )
    
    def improve_prompt(
        self,
        current_prompt: str,
        feedback: BatchFeedback,
        iteration: int,
        history: List[Iteration],
        current_score: Optional[float] = None
    ) -> str:
        """
        Generate improved prompt that addresses critique.
        
        Args:
            current_prompt: Current prompt text
            feedback: Aggregated feedback from judge
            iteration: Current iteration number
            history: Previous iterations (to avoid repeating failures)
            current_score: Current average score (for adaptive strategy)
        
        Returns:
            Improved prompt text
        """
        # Get adaptive revision strategy based on current score
        # ALWAYS use score-based temperature (not iteration-based)
        # Score reflects prompt quality better than iteration number
        if current_score is not None:
            strategy = self._get_revision_strategy(current_score)
            temperature = strategy["temperature"]
            system_addition = strategy["system_addition"]
        else:
            # Fallback: use feedback score if current_score not provided
            # This should rarely happen, but ensures we always have a score
            fallback_score = feedback.average_score if hasattr(feedback, 'average_score') else 50.0
            logger.warning(
                "current_score not provided to reviser, using feedback score as fallback",
                fallback_score=fallback_score,
                iteration=iteration
            )
            strategy = self._get_revision_strategy(fallback_score)
            temperature = strategy["temperature"]
            system_addition = strategy["system_addition"]
        
        # Build system prompt with adaptive strategy
        adaptive_system_prompt = self.system_prompt
        if system_addition:
            adaptive_system_prompt = f"{self.system_prompt}\n\n{system_addition}"
        
        # Build improvement prompt
        improvement_prompt = f"""Improve the following prompt based on the evaluation feedback.

CURRENT PROMPT:
{current_prompt}

EVALUATION FEEDBACK:
Average Score: {feedback.average_score:.1f}/100

Consolidated Critique:
{feedback.get_consolidated_critique()}

CRITICAL ISSUES:
{self._format_issues(feedback.critical_issues)}

HIGH-PRIORITY SUGGESTIONS:
{chr(10).join(f'- {s}' for s in feedback.high_priority_suggestions[:5])}

ITERATION: {iteration}
{self._format_history(history)}

Generate an improved prompt that addresses the issues above. Return ONLY the improved prompt text."""

        try:
            improved_prompt = self.llm_client.complete(
                model=self.model,
                prompt=improvement_prompt,
                system_prompt=adaptive_system_prompt,
                temperature=temperature,
                max_tokens=OptimizationConfig.REVISER_MAX_TOKENS
            )
            
            # Estimate cost
            estimated_cost = self.llm_client.estimate_cost(
                model=self.model,
                prompt_tokens=1500,
                completion_tokens=800
            )
            self.cost_tracker.add_cost("reviser", estimated_cost)
            
            # Clean up output (remove any markdown or explanations)
            improved_prompt = improved_prompt.strip()
            if improved_prompt.startswith("```"):
                # Remove code fences
                lines = improved_prompt.split("\n")
                improved_prompt = "\n".join(lines[1:-1]) if len(lines) > 2 else improved_prompt
            
            return improved_prompt.strip()
            
        except Exception as e:
            logger.error(f"Error in reviser: {e}")
            # Return original prompt on error
            return current_prompt
    
    def _format_issues(self, issues: List) -> str:
        """Format issues for prompt."""
        if not issues:
            return "None"
        return "\n".join([
            f"- {issue.severity}: {issue.issue}\n  Evidence: {issue.evidence}\n  Impact: {issue.impact}"
            for issue in issues[:5]  # Limit to top 5
        ])
    
    def _format_history(self, history: List[Iteration]) -> str:
        """Format iteration history to avoid repeating failures."""
        if not history:
            return ""
        
        history_text = "PREVIOUS ITERATIONS:\n"
        for iter in history[-3:]:  # Last 3 iterations
            history_text += f"Iteration {iter.iteration_number}: Score {iter.average_score:.1f}\n"
        
        return history_text
    
    def _get_revision_strategy(self, current_score: float) -> Dict[str, Any]:
        """
        Return revision strategy based on current performance.
        
        Args:
            current_score: Current average score (0-100)
        
        Returns:
            Dictionary with temperature, system_addition, and max_changes
        """
        if current_score >= 85:
            # Conservative - already excellent
            return {
                "temperature": 0.3,
                "system_addition": f"""
The current prompt is performing excellently (score: {current_score:.1f}).
Make ONLY minimal, targeted changes to address specific low-severity issues.
Preserve all working elements. Avoid restructuring.
Focus on fixing only the specific problems mentioned in the critique.
                """,
                "max_changes": 2
            }
        elif current_score >= 70:
            # Balanced - good but room for improvement
            return {
                "temperature": 0.5,
                "system_addition": f"""
The current prompt is performing well (score: {current_score:.1f}) but has room for improvement.
Make focused changes to address identified issues while preserving strengths.
Be selective about changes - don't fix what isn't broken.
                """,
                "max_changes": 4
            }
        else:
            # Aggressive - needs major work
            return {
                "temperature": 0.7,
                "system_addition": f"""
The current prompt has significant issues (score: {current_score:.1f}).
Be bold in restructuring and adding necessary content.
Address all critical and high-severity issues comprehensively.
Major changes are acceptable and expected.
                """,
                "max_changes": 8
            }
