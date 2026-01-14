"""Checker Prompt - validate outputs before final delivery."""
from typing import Dict, Any, Optional, List
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.specification import OutputSpecification
from utils.logging_utils import setup_logging, get_correlation_id
from utils.metrics import get_metrics_collector
from utils.negative_constraints import get_negative_constraints_library
from pathlib import Path

logger = setup_logging()
metrics = get_metrics_collector()


class Checker:
    """Validates outputs using a checker prompt before delivery."""
    
    def __init__(self, llm_client: LLMClient, cost_tracker: CostTracker):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.negative_constraints = get_negative_constraints_library()
        
        # Load checker system prompt
        prompt_path = Path(__file__).parent.parent / "prompts" / "checker_system.txt"
        if prompt_path.exists():
            with open(prompt_path, "r") as f:
                self.system_prompt = f.read()
        else:
            # Default checker prompt
            self.system_prompt = """You are a quality checker for LLM outputs. Your job is to validate outputs before they are delivered to users.

Check for:
1. Format compliance (JSON, XML, etc.)
2. Constraint violations (length, case, etc.)
3. Common errors (markdown in JSON, explanations when not needed)
4. Security issues (PII leakage, prompt injection)

If the output is CORRECT, return it unchanged.
If the output has ISSUES, return a corrected version.

Return ONLY the corrected output (no explanations, no markdown)."""
    
    def check(
        self,
        output: str,
        specification: OutputSpecification,
        test_input: Optional[str] = None,
        auto_fix: bool = True
    ) -> Dict[str, Any]:
        """
        Check and optionally fix an output.
        
        Args:
            output: Output to check
            specification: Output specification
            test_input: Optional test input for context
            auto_fix: Whether to automatically fix issues
        
        Returns:
            {
                "original": str,
                "checked": str,
                "fixed": bool,
                "violations": List[str],
                "fixes_applied": List[str]
            }
        """
        # First, check negative constraints
        violations = self.negative_constraints.check(output)
        violation_summary = self.negative_constraints.get_violation_summary(violations)
        
        # If no violations and auto_fix is False, return early
        if not violations and not auto_fix:
            return {
                "original": output,
                "checked": output,
                "fixed": False,
                "violations": [],
                "fixes_applied": []
            }
        
        # Run checker prompt if violations found or auto_fix enabled
        if violations or auto_fix:
            checked_output = self._run_checker_prompt(
                output=output,
                specification=specification,
                test_input=test_input,
                violations=violations
            )
            
            fixed = checked_output != output
            
            if fixed:
                logger.info(
                    "Checker fixed output",
                    violations_count=len(violations),
                    correlation_id=get_correlation_id()
                )
                metrics.increment("checker.fixes_applied")
            
            return {
                "original": output,
                "checked": checked_output,
                "fixed": fixed,
                "violations": [v.name for v in violations],
                "fixes_applied": [v.name for v in violations] if fixed else []
            }
        
        return {
            "original": output,
            "checked": output,
            "fixed": False,
            "violations": [v.name for v in violations],
            "fixes_applied": []
        }
    
    def _run_checker_prompt(
        self,
        output: str,
        specification: OutputSpecification,
        test_input: Optional[str],
        violations: List
    ) -> str:
        """Run checker LLM to validate and fix output."""
        checker_prompt = f"""Check and fix the following output.

OUTPUT TO CHECK:
{output}

SPECIFICATION:
- Task: {specification.task_name}
- Description: {specification.task_description}

SYNTAX RULES:
{self._format_rules(specification.syntax_rules)}

VIOLATIONS DETECTED:
{self.negative_constraints.get_violation_summary(violations) if violations else "None"}

{"TEST INPUT: " + test_input if test_input else ""}

Return ONLY the corrected output. If no fixes needed, return the original output unchanged."""

        try:
            # Use a fast, cheap model for checking (e.g., GPT-4o-mini)
            checked = self.llm_client.complete(
                model="openai/gpt-4o-mini",  # Fast and cheap for checking
                prompt=checker_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000
            )
            
            # Estimate cost
            estimated_cost = self.llm_client.estimate_cost(
                model="openai/gpt-4o-mini",
                prompt_tokens=500,
                completion_tokens=200
            )
            self.cost_tracker.add_cost("checker", estimated_cost)
            
            return checked.strip()
        
        except Exception as e:
            logger.error(f"Checker prompt failed: {e}")
            # Return original on error
            return output
    
    def _format_rules(self, rules: List) -> str:
        """Format rules for prompt."""
        if not rules:
            return "None"
        return "\n".join([
            f"- {rule.name} ({rule.severity}): {rule.description}"
            for rule in rules
        ])
