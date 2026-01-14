"""Analyzes test suite coverage across dimensions."""
from typing import Dict, Any, List
from canonical_suite.suite_loader import CanonicalTestSuite
from models.test_case import CanonicalTestCase


class CoverageReport:
    """Coverage analysis report."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        return self.data


class CoverageAnalyzer:
    """Analyzes test suite coverage."""
    
    def analyze_coverage(self, test_suite: CanonicalTestSuite) -> CoverageReport:
        """
        Analyze coverage across dimensions.
        
        Returns:
            CoverageReport with heatmap data
        """
        test_cases = test_suite.test_cases
        
        # By archetype class
        by_archetype = {}
        for tc in test_cases:
            arch = tc.metadata.get("archetype_class", "unknown")
            by_archetype[arch] = by_archetype.get(arch, 0) + 1
        
        # By domain
        by_domain = {}
        for tc in test_cases:
            domain = tc.metadata.get("domain", "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1
        
        # By difficulty
        by_difficulty = {}
        for tc in test_cases:
            diff = tc.difficulty
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        
        # By category
        by_category = {}
        for tc in test_cases:
            cat = tc.category
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # By output format
        by_format = {}
        for tc in test_cases:
            for test_input in tc.test_cases:
                fmt = test_input.expected_output.format or "text"
                by_format[fmt] = by_format.get(fmt, 0) + 1
        
        return CoverageReport({
            "total_tests": len(test_cases),
            "by_archetype": by_archetype,
            "by_domain": by_domain,
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "by_format": by_format,
            "coverage_matrix": {
                "archetype_classes": len(by_archetype),
                "domains": len(by_domain),
                "difficulties": len(by_difficulty),
                "categories": len(by_category)
            }
        })
