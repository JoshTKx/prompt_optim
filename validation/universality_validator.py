"""Universality validator - tests optimizer against canonical suite."""
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import json
from canonical_suite.suite_loader import CanonicalTestSuite
from canonical_suite.suite_runner import SuiteRunner
from utils.logging_utils import setup_logging

logger = setup_logging()


class UniversalityReport:
    """Report from universality validation."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    @property
    def total_tests(self) -> int:
        return self.data.get("total_tests", 0)
    
    @property
    def tests_passed(self) -> int:
        return self.data.get("tests_passed", 0)
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.tests_passed / self.total_tests
    
    def to_dict(self) -> Dict[str, Any]:
        return self.data
    
    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)


class UniversalityValidator:
    """Validates optimizer universality against canonical test suite."""
    
    def __init__(self):
        self.runner = SuiteRunner()
    
    def validate_optimizer(
        self,
        test_suite: CanonicalTestSuite,
        sample_size: int = None,
        max_iterations: int = 5
    ) -> UniversalityReport:
        """
        Run optimizer on canonical test cases to validate universality.
        
        Args:
            test_suite: CanonicalTestSuite instance
            sample_size: Number of tests to run (None = all)
            max_iterations: Max iterations per test
        
        Returns:
            UniversalityReport with pass/fail statistics
        """
        logger.info(f"Starting universality validation on {len(test_suite)} test cases")
        
        test_cases = list(test_suite.test_cases)
        if sample_size:
            test_cases = test_cases[:sample_size]
        
        results = []
        passed = 0
        failed = []
        
        by_category = {}
        by_archetype = {}
        
        total_cost = 0.0
        start_time = datetime.now()
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test {i}/{len(test_cases)}: {test_case.id}")
            
            try:
                result = self.runner.run_test_case(test_case)
                results.append(result)
                
                if result["passed"]:
                    passed += 1
                else:
                    failed.append(test_case.id)
                
                # Track by category
                category = test_case.category
                if category not in by_category:
                    by_category[category] = {"passed": 0, "total": 0}
                by_category[category]["total"] += 1
                if result["passed"]:
                    by_category[category]["passed"] += 1
                
                # Track by archetype
                archetype_class = test_case.metadata.get("archetype_class", "unknown")
                if archetype_class not in by_archetype:
                    by_archetype[archetype_class] = {"passed": 0, "total": 0}
                by_archetype[archetype_class]["total"] += 1
                if result["passed"]:
                    by_archetype[archetype_class]["passed"] += 1
                
                # Track cost
                total_cost += result["result"].total_cost_usd
                
            except Exception as e:
                logger.error(f"Error running test {test_case.id}: {e}")
                failed.append(test_case.id)
                results.append({
                    "test_id": test_case.id,
                    "error": str(e),
                    "passed": False
                })
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        report_data = {
            "total_tests": len(test_cases),
            "tests_passed": passed,
            "pass_rate": passed / len(test_cases) if test_cases else 0.0,
            "by_category": by_category,
            "by_archetype": by_archetype,
            "failed_tests": failed,
            "cost_total": round(total_cost, 2),
            "time_total_hours": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "test_id": r["test_id"],
                    "passed": r["passed"],
                    "initial_score": r.get("initial_score", 0),
                    "optimized_score": r.get("optimized_score", 0),
                    "improvement": r.get("improvement", 0)
                }
                for r in results
            ]
        }
        
        return UniversalityReport(report_data)
