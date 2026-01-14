"""Negative Constraints - patterns to avoid in outputs."""
import re
from typing import List, Dict, Any, Optional, Pattern
from dataclasses import dataclass
from enum import Enum
from utils.logging_utils import setup_logging
from utils.metrics import get_metrics_collector

logger = setup_logging()
metrics = get_metrics_collector()


class ConstraintType(Enum):
    """Types of negative constraints."""
    REGEX = "regex"
    KEYWORD = "keyword"
    PHRASE = "phrase"
    PATTERN = "pattern"


@dataclass
class NegativeConstraint:
    """A negative constraint pattern."""
    constraint_id: str
    name: str
    description: str
    pattern: str
    constraint_type: ConstraintType
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    category: Optional[str] = None  # e.g., "security", "formatting", "hallucination"
    
    def matches(self, text: str) -> bool:
        """Check if text matches this constraint (should be avoided)."""
        if self.constraint_type == ConstraintType.REGEX:
            return bool(re.search(self.pattern, text, re.IGNORECASE))
        elif self.constraint_type == ConstraintType.KEYWORD:
            return self.pattern.lower() in text.lower()
        elif self.constraint_type == ConstraintType.PHRASE:
            return self.pattern.lower() in text.lower()
        elif self.constraint_type == ConstraintType.PATTERN:
            # Custom pattern matching logic
            return self._pattern_match(text)
        return False
    
    def _pattern_match(self, text: str) -> bool:
        """Custom pattern matching (extensible)."""
        # Example: detect markdown code blocks when JSON is expected
        if "markdown" in self.pattern.lower() and "json" in self.pattern.lower():
            return bool(re.search(r'```(?:json)?\s*\n', text))
        return False


class NegativeConstraintsLibrary:
    """Library of negative constraints to avoid."""
    
    def __init__(self):
        self.constraints: List[NegativeConstraint] = []
        self._load_default_constraints()
    
    def add_constraint(self, constraint: NegativeConstraint):
        """Add a constraint to the library."""
        # Check for duplicates
        if any(c.constraint_id == constraint.constraint_id for c in self.constraints):
            logger.warning(f"Constraint {constraint.constraint_id} already exists, skipping")
            return
        
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {constraint.name}")
    
    def check(self, text: str, category: Optional[str] = None) -> List[NegativeConstraint]:
        """
        Check text against all constraints.
        
        Args:
            text: Text to check
            category: Optional category filter
        
        Returns:
            List of matching constraints (violations)
        """
        violations = []
        constraints_to_check = self.constraints
        
        if category:
            constraints_to_check = [c for c in constraints_to_check if c.category == category]
        
        for constraint in constraints_to_check:
            if constraint.matches(text):
                violations.append(constraint)
                metrics.increment(
                    "negative_constraints.violated",
                    tags={"constraint": constraint.constraint_id, "severity": constraint.severity}
                )
        
        return violations
    
    def get_violation_summary(self, violations: List[NegativeConstraint]) -> str:
        """Generate summary of violations."""
        if not violations:
            return "No violations detected."
        
        critical = [v for v in violations if v.severity == "CRITICAL"]
        high = [v for v in violations if v.severity == "HIGH"]
        
        summary = []
        if critical:
            summary.append(f"CRITICAL: {len(critical)} violations")
            summary.extend([f"  - {v.name}: {v.description}" for v in critical[:3]])
        if high:
            summary.append(f"HIGH: {len(high)} violations")
            summary.extend([f"  - {v.name}: {v.description}" for v in high[:3]])
        
        return "\n".join(summary)
    
    def _load_default_constraints(self):
        """Load default negative constraints."""
        default_constraints = [
            # Formatting violations
            NegativeConstraint(
                constraint_id="no_markdown_json",
                name="No Markdown in JSON",
                description="JSON output should not be wrapped in markdown code blocks",
                pattern=r'```(?:json)?\s*\n.*?```',
                constraint_type=ConstraintType.REGEX,
                severity="CRITICAL",
                category="formatting"
            ),
            NegativeConstraint(
                constraint_id="no_explanation_json",
                name="No Explanation with JSON",
                description="JSON output should not include explanatory text",
                pattern=r'^(?:Here|This|The).*?\{',
                constraint_type=ConstraintType.REGEX,
                severity="HIGH",
                category="formatting"
            ),
            
            # Security violations
            NegativeConstraint(
                constraint_id="no_system_prompt_leak",
                name="No System Prompt Leakage",
                description="Output should not reveal system prompts or instructions",
                pattern=r'(?:system|instruction|prompt).*?(?:reveal|show|print|display)',
                constraint_type=ConstraintType.REGEX,
                severity="CRITICAL",
                category="security"
            ),
            NegativeConstraint(
                constraint_id="no_pii_leak",
                name="No PII Leakage",
                description="Output should not leak personally identifiable information",
                pattern=r'\b(?:\d{3}-\d{2}-\d{4}|\d{3}\.\d{2}\.\d{4})\b',  # SSN pattern
                constraint_type=ConstraintType.REGEX,
                severity="CRITICAL",
                category="security"
            ),
            
            # Hallucination patterns
            NegativeConstraint(
                constraint_id="no_unsupported_claims",
                name="No Unsupported Claims",
                description="Output should not make unsupported factual claims",
                pattern=r'\b(?:definitely|certainly|proven|fact).*?(?:is|are|was|were)\b',
                constraint_type=ConstraintType.REGEX,
                severity="MEDIUM",
                category="hallucination"
            ),
            
            # Common error patterns
            NegativeConstraint(
                constraint_id="no_apology_format",
                name="No Apology in Format Errors",
                description="Output should not apologize for format errors, just fix them",
                pattern=r'\b(?:sorry|apologize|my mistake).*?(?:format|output|error)',
                constraint_type=ConstraintType.REGEX,
                severity="LOW",
                category="formatting"
            ),
        ]
        
        for constraint in default_constraints:
            self.add_constraint(constraint)
        
        logger.info(f"Loaded {len(self.constraints)} default negative constraints")


# Global library instance
_global_library: Optional[NegativeConstraintsLibrary] = None


def get_negative_constraints_library() -> NegativeConstraintsLibrary:
    """Get global negative constraints library."""
    global _global_library
    if _global_library is None:
        _global_library = NegativeConstraintsLibrary()
    return _global_library
