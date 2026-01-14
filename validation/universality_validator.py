"""Universality validator - tests optimizer against canonical suite."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
from canonical_suite.suite_loader import CanonicalTestSuite
from canonical_suite.suite_runner import SuiteRunner
from core.parallel_orchestrator import ParallelOrchestrator
from config.optimization_config import OptimizationConfig
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
    
    def save(self, filepath: str, keep_only_latest: bool = True):
        """
        Save report to JSON file.
        
        Args:
            filepath: Path to save the report
            keep_only_latest: If True, delete old universality reports before saving
        """
        filepath_obj = Path(filepath)
        output_dir = filepath_obj.parent
        
        # Delete old universality reports if keep_only_latest is True
        if keep_only_latest:
            pattern = "universality_report_*.json"
            deleted_count = 0
            for old_file in output_dir.glob(pattern):
                if old_file != filepath_obj:  # Don't delete the file we're about to create
                    try:
                        old_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete old report {old_file}: {e}")
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old universality report(s)")
        
        # Save the new report
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)


class UniversalityValidator:
    """Validates optimizer universality against canonical test suite."""
    
    def __init__(self, use_parallel: bool = True, max_parallel_workers: Optional[int] = None):
        """
        Initialize validator.
        
        Args:
            use_parallel: Whether to use parallel optimization
            max_parallel_workers: Max parallel workers (defaults to config)
        """
        self.runner = SuiteRunner()
        self.use_parallel = use_parallel
        self.max_parallel_workers = max_parallel_workers or OptimizationConfig.MAX_PARALLEL_OPTIMIZATIONS
        
        if self.use_parallel:
            self.parallel_orchestrator = ParallelOrchestrator(max_workers=self.max_parallel_workers)
        else:
            self.parallel_orchestrator = None
    
    def validate_optimizer(
        self,
        test_suite: CanonicalTestSuite,
        sample_size: int = None,
        max_iterations: int = 5,
        use_parallel: Optional[bool] = None
    ) -> UniversalityReport:
        """
        Run optimizer on canonical test cases to validate universality.
        
        Args:
            test_suite: CanonicalTestSuite instance
            sample_size: Number of tests to run (None = all)
            max_iterations: Max iterations per test
            use_parallel: Override parallel setting (defaults to instance setting)
        
        Returns:
            UniversalityReport with pass/fail statistics
        """
        use_parallel_exec = use_parallel if use_parallel is not None else self.use_parallel
        
        logger.info(
            f"Starting universality validation on {len(test_suite)} test cases",
            parallel=use_parallel_exec,
            max_workers=self.max_parallel_workers if use_parallel_exec else 1
        )
        
        test_cases = list(test_suite.test_cases)
        if sample_size:
            test_cases = test_cases[:sample_size]
        
        start_time = datetime.now()
        
        if use_parallel_exec and self.parallel_orchestrator:
            # Use parallel execution
            results = self._validate_parallel(test_cases, max_iterations)
        else:
            # Use sequential execution (original behavior)
            results = self._validate_sequential(test_cases, max_iterations)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        # Process results
        passed = sum(1 for r in results if r.get("passed", False))
        failed = [r["test_id"] for r in results if not r.get("passed", False)]
        
        by_category = {}
        by_archetype = {}
        total_cost = 0.0
        
        for result in results:
            # Track by category
            category = result.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"passed": 0, "total": 0}
            by_category[category]["total"] += 1
            if result.get("passed", False):
                by_category[category]["passed"] += 1
            
            # Track by archetype
            archetype_class = result.get("archetype_class", "unknown")
            if archetype_class not in by_archetype:
                by_archetype[archetype_class] = {"passed": 0, "total": 0}
            by_archetype[archetype_class]["total"] += 1
            if result.get("passed", False):
                by_archetype[archetype_class]["passed"] += 1
            
            # Track cost
            if "result" in result and result["result"]:
                total_cost += result["result"].total_cost_usd
        
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
    
    def _validate_sequential(
        self,
        test_cases: List,
        max_iterations: int
    ) -> List[Dict[str, Any]]:
        """Validate using sequential execution (original behavior)."""
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Running test {i}/{len(test_cases)}: {test_case.id}")
            
            try:
                result = self.runner.run_test_case(test_case)
                result["category"] = test_case.category
                result["archetype_class"] = test_case.metadata.get("archetype_class", "unknown")
                results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_case.id}: {e}")
                results.append({
                    "test_id": test_case.id,
                    "name": test_case.name,
                    "category": test_case.category,
                    "error": str(e),
                    "passed": False,
                    "initial_score": 0,
                    "optimized_score": 0,
                    "improvement": 0
                })
        
        return results
    
    def _validate_parallel(
        self,
        test_cases: List,
        max_iterations: int
    ) -> List[Dict[str, Any]]:
        """Validate using parallel execution."""
        # Prepare optimization tasks
        tasks = []
        for test_case in test_cases:
            specification = self.runner._create_specification(test_case)
            tasks.append({
                "initial_prompt": test_case.initial_prompt,
                "test_case": test_case,
                "specification": specification,
                "max_iterations": max_iterations,
                "prompt_id": f"prompt_{test_case.id}"
            })
        
        # Run optimizations in parallel
        optimization_results = self.parallel_orchestrator.optimize_multiple(tasks)
        
        # Process results into expected format
        results = []
        for i, (test_case, opt_result) in enumerate(zip(test_cases, optimization_results)):
            if opt_result is None:
                # Failed optimization
                results.append({
                    "test_id": test_case.id,
                    "name": test_case.name,
                    "category": test_case.category,
                    "error": "Optimization failed",
                    "passed": False,
                    "initial_score": 0,
                    "optimized_score": 0,
                    "improvement": 0,
                    "archetype_class": test_case.metadata.get("archetype_class", "unknown")
                })
                continue
            
            # Evaluate result
            optimized_score = opt_result.best_score
            min_score = test_case.evaluation.pass_criteria.get("minimum_score", 4) * 20
            passed = optimized_score >= min_score
            
            results.append({
                "test_id": test_case.id,
                "name": test_case.name,
                "category": test_case.category,
                "initial_score": opt_result.initial_score,
                "optimized_score": optimized_score,
                "improvement": opt_result.improvement,
                "passed": passed,
                "result": opt_result,
                "archetype_class": test_case.metadata.get("archetype_class", "unknown")
            })
        
        return results
