"""Reviser component - improves prompts based on feedback."""
from pathlib import Path
from typing import List
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
        history: List[Iteration]
    ) -> str:
        """
        Generate improved prompt that addresses critique.
        
        Args:
            current_prompt: Current prompt text
            feedback: Aggregated feedback from judge
            iteration: Current iteration number
            history: Previous iterations (to avoid repeating failures)
        
        Returns:
            Improved prompt text
        """
        # Determine temperature based on iteration
        if iteration <= 2:
            temperature = OptimizationConfig.TEMPERATURE_SCHEDULE["early"]
        elif iteration <= 4:
            temperature = OptimizationConfig.TEMPERATURE_SCHEDULE["mid"]
        else:
            temperature = OptimizationConfig.TEMPERATURE_SCHEDULE["late"]
        
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
                system_prompt=self.system_prompt,
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
