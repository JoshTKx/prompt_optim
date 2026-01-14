"""Judge component - evaluates outputs using LLM."""
import json
from typing import List, Dict, Any
from pathlib import Path
from utils.llm_client import LLMClient
from utils.cost_tracker import CostTracker
from models.feedback import Feedback, Issue, BatchFeedback
from models.specification import OutputSpecification
from config.llm_config import LLMConfig
from config.optimization_config import OptimizationConfig
from utils.logging_utils import setup_logging
from utils.system_prompt_manager import SystemPromptManager

logger = setup_logging()
prompt_manager = SystemPromptManager()


class Judge:
    """Evaluates outputs and provides feedback."""
    
    def __init__(self, llm_client: LLMClient, cost_tracker: CostTracker):
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.model = LLMConfig.JUDGE_MODEL
        
        # Load judge system prompt with versioning
        prompt_path = Path(__file__).parent.parent / "prompts" / "judge_system.txt"
        self.system_prompt, self.prompt_version, self.prompt_hash = prompt_manager.load_prompt(
            prompt_name="judge_system",
            prompt_path=prompt_path
        )
        
        logger.info(
            "Judge initialized",
            model=self.model,
            prompt_version=self.prompt_version,
            prompt_hash=self.prompt_hash
        )
    
    def evaluate_output(
        self,
        test_input: Dict[str, Any],
        actual_output: str,
        specification: OutputSpecification
    ) -> Feedback:
        """
        Evaluate a single output.
        
        Args:
            test_input: The test input that was used
            actual_output: The actual output from the model
            specification: Output specification
        
        Returns:
            Feedback object with score and critique
        """
        # Build evaluation prompt
        eval_prompt = f"""Evaluate the following output against the specification.

TASK: {specification.task_name}
DESCRIPTION: {specification.task_description}

TEST INPUT: {test_input.get('input', 'N/A')}

ACTUAL OUTPUT:
{actual_output}

SPECIFICATION:
- Task: {specification.task_name}
- Rubric: {specification.scoring_rubric}

SYNTAX RULES:
{self._format_rules(specification.syntax_rules)}

SEMANTIC RULES:
{self._format_rules(specification.semantic_rules)}

Provide your evaluation in JSON format as specified."""
        
        try:
            # Get judgment from LLM
            response = self.llm_client.complete(
                model=self.model,
                prompt=eval_prompt,
                system_prompt=self.system_prompt,
                temperature=OptimizationConfig.JUDGE_TEMPERATURE,
                max_tokens=OptimizationConfig.JUDGE_MAX_TOKENS,
                json_mode=False  # DeepSeek doesn't support json_mode, parse manually
            )
            
            # Parse JSON response
            # Remove markdown if present
            response_text = response.strip()
            if response_text.startswith("```"):
                # Extract content between code fences
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                else:
                    # Fallback: remove first and last lines if they're code fences
                    lines = response_text.split("\n")
                    if lines[0].strip().startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip().endswith("```"):
                        lines = lines[:-1]
                    response_text = "\n".join(lines).strip()
            
            # Try to parse JSON with repair attempts
            judgment = None
            parse_attempts = [
                response_text,  # Try original first
                self._repair_json(response_text),  # Try repaired version
            ]
            
            for attempt_text in parse_attempts:
                try:
                    judgment = json.loads(attempt_text)
                    break
                except json.JSONDecodeError as e:
                    continue
            
            # If still failed, try to extract JSON object using better regex
            if judgment is None:
                logger.warning(f"JSON parse error after repair attempts, trying extraction")
                # Better regex that handles nested objects (but still limited for strings)
                import re
                # Try to find JSON object boundaries more intelligently
                # Look for opening brace followed by content
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        extracted = json_match.group(0)
                        # Try to repair the extracted JSON
                        extracted = self._repair_json(extracted)
                        judgment = json.loads(extracted)
                    except:
                        pass
            
            # If still failed, try with retry to LLM with more tokens
            if judgment is None:
                logger.warning(f"JSON parse failed, retrying judge with increased max_tokens")
                try:
                    # Retry with more tokens to avoid truncation
                    retry_response = self.llm_client.complete(
                        model=self.model,
                        prompt=eval_prompt,
                        system_prompt=self.system_prompt,
                        temperature=OptimizationConfig.JUDGE_TEMPERATURE,
                        max_tokens=OptimizationConfig.JUDGE_MAX_TOKENS * 2,  # Double tokens
                        json_mode=False
                    )
                    
                    # Clean and parse retry response
                    retry_text = retry_response.strip()
                    if retry_text.startswith("```"):
                        parts = retry_text.split("```")
                        if len(parts) >= 3:
                            retry_text = parts[1]
                            if retry_text.startswith("json"):
                                retry_text = retry_text[4:]
                            retry_text = retry_text.strip()
                    
                    # Try parsing retry
                    retry_text = self._repair_json(retry_text)
                    judgment = json.loads(retry_text)
                except Exception as retry_error:
                    logger.error(f"JSON parse error even after retry: {retry_error}")
                    raise ValueError(f"Could not parse JSON from judge response: {str(retry_error)}")
            
            # Estimate cost
            estimated_cost = self.llm_client.estimate_cost(
                model=self.model,
                prompt_tokens=1000,
                completion_tokens=500
            )
            self.cost_tracker.add_cost("judge", estimated_cost)
            
            # Convert to Feedback model
            issues = [
                Issue(**issue) for issue in judgment.get("issues", [])
            ]
            
            return Feedback(
                score=float(judgment.get("score", 0)),
                critique=judgment.get("critique", ""),
                issues=issues,
                suggestions=judgment.get("suggestions", [])
            )
            
        except Exception as e:
            logger.error(f"Error in judge evaluation: {e}")
            # Return default feedback on error
            return Feedback(
                score=0.0,
                critique=f"Evaluation error: {str(e)}",
                issues=[Issue(
                    severity="CRITICAL",
                    issue="Evaluation failed",
                    evidence=str(e),
                    impact="Cannot assess output quality"
                )],
                suggestions=[]
            )
    
    def evaluate_batch(
        self,
        test_results: List[Dict[str, Any]],
        specification: OutputSpecification
    ) -> BatchFeedback:
        """
        Evaluate multiple outputs and aggregate feedback.
        
        Args:
            test_results: List of test results from Tester
            specification: Output specification
        
        Returns:
            BatchFeedback with aggregated scores and issues
        """
        feedbacks = []
        
        for result in test_results:
            if not result.get("success", False):
                # Failed test case
                feedbacks.append(Feedback(
                    score=0.0,
                    critique="Test execution failed",
                    issues=[Issue(
                        severity="CRITICAL",
                        issue="Execution error",
                        evidence=result.get("error", "Unknown error"),
                        impact="Cannot evaluate output"
                    )],
                    suggestions=[]
                ))
                continue
            
            feedback = self.evaluate_output(
                test_input={"input": result["input"]},
                actual_output=result["output"],
                specification=specification
            )
            feedbacks.append(feedback)
        
        # Aggregate
        scores = [fb.score for fb in feedbacks]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        # Collect critical issues
        critical_issues = []
        for fb in feedbacks:
            critical_issues.extend([issue for issue in fb.issues if issue.severity == "CRITICAL"])
        
        # Collect high-priority suggestions
        high_priority_suggestions = []
        for fb in feedbacks:
            high_priority_suggestions.extend(fb.suggestions)
        
        return BatchFeedback(
            test_case_feedbacks=feedbacks,
            average_score=average_score,
            critical_issues=critical_issues,
            high_priority_suggestions=high_priority_suggestions
        )
    
    def _repair_json(self, content: str) -> str:
        """Attempt to repair common JSON issues including unterminated strings."""
        import re
        
        # Fix trailing commas in objects and arrays
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # Track state
        depth = 0
        array_depth = 0
        in_string = False
        escape_next = False
        last_complete_pos = -1
        
        for i, char in enumerate(content):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        last_complete_pos = i
                        break
                elif char == '[':
                    array_depth += 1
                elif char == ']':
                    array_depth -= 1
        
        # If we found a complete JSON object, use it
        if last_complete_pos > 0 and depth == 0:
            return content[:last_complete_pos + 1]
        
        # If we're in an unterminated string, try to fix it
        if in_string:
            # Find the last complete key-value pair or array element
            # Look for pattern: ...", "key": "value that got cut off
            last_quote = content.rfind('"')
            if last_quote > 0:
                # Check what comes before this quote
                # Look for : "pattern (start of a value)
                colon_quote_pattern = content.rfind(': "')
                if colon_quote_pattern > 0 and colon_quote_pattern < last_quote:
                    # We have an unterminated string value
                    # Try to close it and the structure
                    safe_pos = colon_quote_pattern + 2  # Right after : "
                    # Close the string
                    content = content[:safe_pos] + '""'
                    # Close any open structures
                    temp_depth = depth
                    temp_array_depth = array_depth
                    while temp_depth > 0:
                        content += '}'
                        temp_depth -= 1
                    while temp_array_depth > 0:
                        content += ']'
                        temp_array_depth -= 1
                    return content
        
        # If JSON is incomplete but not in a string, try to close structures
        if not in_string and (depth > 0 or array_depth > 0):
            while depth > 0:
                content += '}'
                depth -= 1
            while array_depth > 0:
                content += ']'
                array_depth -= 1
        
        return content
    
    def _format_rules(self, rules: List) -> str:
        """Format rules for prompt."""
        if not rules:
            return "None"
        return "\n".join([
            f"- {rule.name} ({rule.severity}): {rule.description}"
            for rule in rules
        ])
