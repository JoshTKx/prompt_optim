"""Validators for output checking."""
import json
import re
from typing import Any, Dict, List, Optional, Callable


class JSONValidator:
    """Validates JSON output."""
    
    @staticmethod
    def validate(text: str) -> tuple[bool, Optional[str]]:
        """
        Validate JSON.
        
        Returns:
            (is_valid, error_message)
        """
        # Remove markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            # Remove ```json or ``` markers
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
            text = text.strip()
        
        try:
            json.loads(text)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"


class ConstraintValidator:
    """Validates constraints on output."""
    
    @staticmethod
    def validate(text: str, constraints: List[str]) -> tuple[bool, List[str]]:
        """
        Validate constraints.
        
        Args:
            text: Output text
            constraints: List of constraint names
        
        Returns:
            (all_passed, failed_constraints)
        """
        failed = []
        
        for constraint in constraints:
            if not ConstraintValidator._check_constraint(text, constraint):
                failed.append(constraint)
        
        return len(failed) == 0, failed
    
    @staticmethod
    def _check_constraint(text: str, constraint: str) -> bool:
        """Check a single constraint."""
        constraint_lower = constraint.lower()
        
        # No markdown fencing
        if "no_markdown_fencing" in constraint_lower:
            return not (text.strip().startswith("```") or text.strip().endswith("```"))
        
        # Array length constraints
        if "array_length_equals" in constraint_lower:
            match = re.search(r'array_length_equals_(\d+)', constraint_lower)
            if match:
                expected_length = int(match.group(1))
                try:
                    data = json.loads(text.strip().strip("```"))
                    if isinstance(data, list):
                        return len(data) == expected_length
                except:
                    pass
            return False
        
        # Case constraints
        if "only_lowercase" in constraint_lower:
            return text.islower()
        
        # Word exclusion
        if constraint.startswith("no_word_"):
            excluded_word = constraint.replace("no_word_", "")
            return excluded_word.lower() not in text.lower()
        
        # Default: constraint not recognized
        return True


class OutputValidator:
    """Main validator that combines multiple validators."""
    
    def __init__(self):
        self.json_validator = JSONValidator()
        self.constraint_validator = ConstraintValidator()
    
    def validate(
        self,
        output: str,
        expected_output: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate output against expected specification.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # JSON validation
        if expected_output.get("format") == "json":
            is_valid, error = self.json_validator.validate(output)
            if not is_valid:
                errors.append(error)
        
        # Constraint validation
        constraints = expected_output.get("constraints", [])
        if constraints:
            all_passed, failed = self.constraint_validator.validate(output, constraints)
            if not all_passed:
                errors.append(f"Failed constraints: {', '.join(failed)}")
        
        return len(errors) == 0, errors
