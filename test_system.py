#!/usr/bin/env python3
"""Comprehensive system test without API calls."""
import sys
import json
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    errors = []
    
    try:
        from config.llm_config import LLMConfig
        print("  ✓ LLMConfig")
    except Exception as e:
        errors.append(f"LLMConfig: {e}")
        print(f"  ✗ LLMConfig: {e}")
    
    try:
        from config.optimization_config import OptimizationConfig
        print("  ✓ OptimizationConfig")
    except Exception as e:
        errors.append(f"OptimizationConfig: {e}")
        print(f"  ✗ OptimizationConfig: {e}")
    
    try:
        from models.specification import OutputSpecification, Rule
        print("  ✓ Models (Specification)")
    except Exception as e:
        errors.append(f"Models: {e}")
        print(f"  ✗ Models: {e}")
    
    try:
        from models.feedback import Feedback, BatchFeedback, Issue
        print("  ✓ Models (Feedback)")
    except Exception as e:
        errors.append(f"Feedback: {e}")
        print(f"  ✗ Models (Feedback): {e}")
    
    try:
        from models.test_case import CanonicalTestCase
        print("  ✓ Models (TestCase)")
    except Exception as e:
        errors.append(f"TestCase: {e}")
        print(f"  ✗ Models (TestCase): {e}")
    
    try:
        from utils.golden_set import GoldenSetManager, GoldenExample
        print("  ✓ GoldenSetManager")
    except Exception as e:
        errors.append(f"GoldenSet: {e}")
        print(f"  ✗ GoldenSetManager: {e}")
    
    try:
        from utils.negative_constraints import get_negative_constraints_library
        lib = get_negative_constraints_library()
        print(f"  ✓ NegativeConstraintsLibrary ({len(lib.constraints)} constraints)")
    except Exception as e:
        errors.append(f"NegativeConstraints: {e}")
        print(f"  ✗ NegativeConstraintsLibrary: {e}")
    
    try:
        from utils.prompt_versioning import PromptVersionManager
        print("  ✓ PromptVersionManager")
    except Exception as e:
        errors.append(f"PromptVersioning: {e}")
        print(f"  ✗ PromptVersionManager: {e}")
    
    try:
        from utils.metrics import MetricsCollector
        print("  ✓ MetricsCollector")
    except Exception as e:
        errors.append(f"Metrics: {e}")
        print(f"  ✗ MetricsCollector: {e}")
    
    try:
        from utils.cost_tracker import CostTracker
        print("  ✓ CostTracker")
    except Exception as e:
        errors.append(f"CostTracker: {e}")
        print(f"  ✗ CostTracker: {e}")
    
    try:
        from utils.validators import JSONValidator, ConstraintValidator
        print("  ✓ Validators")
    except Exception as e:
        errors.append(f"Validators: {e}")
        print(f"  ✗ Validators: {e}")
    
    try:
        from utils.logging_utils import setup_logging
        logger = setup_logging()
        print("  ✓ Logging")
    except Exception as e:
        errors.append(f"Logging: {e}")
        print(f"  ✗ Logging: {e}")
    
    # Note: LLMClient requires openai package, skip for now
    print("  ⚠ LLMClient (requires openai package)")
    
    return errors


def test_golden_set():
    """Test golden set functionality."""
    print("\nTesting Golden Set...")
    try:
        from utils.golden_set import GoldenSetManager, GoldenExample
        from datetime import datetime
        
        manager = GoldenSetManager()
        
        # Test should_capture logic
        should_capture = manager.should_capture("TEST-001", 90.0, "test output")
        print(f"  ✓ should_capture logic: {should_capture}")
        
        # Test statistics
        stats = manager.get_statistics()
        print(f"  ✓ Statistics: {stats['total_examples']} examples")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_negative_constraints():
    """Test negative constraints."""
    print("\nTesting Negative Constraints...")
    try:
        from utils.negative_constraints import get_negative_constraints_library
        
        lib = get_negative_constraints_library()
        
        # Test constraint checking
        violations = lib.check("Here is some JSON: ```json\n{}\n```")
        print(f"  ✓ Found {len(violations)} violations in test text")
        
        # Test summary
        summary = lib.get_violation_summary(violations)
        print(f"  ✓ Violation summary generated")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_validators():
    """Test validators."""
    print("\nTesting Validators...")
    try:
        from utils.validators import JSONValidator, ConstraintValidator
        
        # Test JSON validator
        is_valid, error = JSONValidator.validate('{"test": "value"}')
        print(f"  ✓ JSON validator: valid={is_valid}")
        
        is_invalid, _ = JSONValidator.validate('{"invalid": json}')
        print(f"  ✓ JSON validator: invalid={is_invalid}")
        
        # Test constraint validator
        passed, failed = ConstraintValidator.validate("lowercase text", ["only_lowercase"])
        print(f"  ✓ Constraint validator: passed={passed}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_test_cases():
    """Test loading test cases."""
    print("\nTesting Test Cases...")
    try:
        from canonical_suite.suite_loader import CanonicalTestSuite
        
        test_dir = Path("canonical_suite/tests")
        if test_dir.exists():
            suite = CanonicalTestSuite.load(str(test_dir))
            print(f"  ✓ Loaded {len(suite)} test cases")
            
            # Test filtering
            by_category = suite.get_by_category("instruction_following")
            print(f"  ✓ Filtered by category: {len(by_category)} tests")
            
            return True
        else:
            print("  ⚠ Test directory not found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_prompt_versioning():
    """Test prompt versioning."""
    print("\nTesting Prompt Versioning...")
    try:
        from utils.prompt_versioning import PromptVersionManager
        
        manager = PromptVersionManager()
        
        # Create a test version
        version = manager.create_version(
            prompt_id="test_prompt",
            prompt_text="Test prompt",
            metadata={"test": True}
        )
        print(f"  ✓ Created version {version.version}")
        
        # Get latest
        latest = manager.get_latest("test_prompt")
        print(f"  ✓ Retrieved latest version: {latest.version}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_metrics():
    """Test metrics collection."""
    print("\nTesting Metrics...")
    try:
        from utils.metrics import get_metrics_collector
        
        metrics = get_metrics_collector()
        
        # Record some metrics
        metrics.record("test.metric", 42.0, tags={"test": "true"})
        metrics.increment("test.counter")
        metrics.gauge("test.gauge", 100.0)
        
        summary = metrics.get_summary()
        print(f"  ✓ Metrics summary: {summary['total_metrics']} metrics")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\nTesting Configuration...")
    try:
        from config.llm_config import LLMConfig
        from config.optimization_config import OptimizationConfig
        
        # Test config access
        print(f"  ✓ Judge model: {LLMConfig.JUDGE_MODEL}")
        print(f"  ✓ Reviser model: {LLMConfig.REVISER_MODEL}")
        print(f"  ✓ Max iterations: {OptimizationConfig.MAX_ITERATIONS}")
        print(f"  ✓ Target score: {OptimizationConfig.TARGET_SCORE}")
        
        # Test headers
        headers = LLMConfig.get_headers()
        print(f"  ✓ Headers method: {len(headers)} headers")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PROMPT OPTIMIZER SYSTEM TEST")
    print("=" * 60)
    
    results = {}
    
    # Test imports
    import_errors = test_imports()
    results["imports"] = len(import_errors) == 0
    
    # Test functionality
    results["golden_set"] = test_golden_set()
    results["negative_constraints"] = test_negative_constraints()
    results["validators"] = test_validators()
    results["test_cases"] = test_test_cases()
    results["prompt_versioning"] = test_prompt_versioning()
    results["metrics"] = test_metrics()
    results["config"] = test_config()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if import_errors:
        print("\n⚠ Import Errors (require dependencies):")
        for error in import_errors:
            print(f"  - {error}")
        print("\nTo fix: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
