"""
Test script to validate optimizer improvements on a small sample.

Usage:
    python test_optimizer_improvements.py
    
This runs the optimizer on a test case with 3 iterations max
to verify:
- No regressions occur
- Convergence works properly  
- Cost stays reasonable
- Improvements are shown
"""

import json
import sys
import logging
from pathlib import Path

from core.orchestrator import Orchestrator
from models.test_case import CanonicalTestCase
from models.specification import OutputSpecification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_no_regression(result) -> tuple[bool, str]:
    """Verify scores never decreased significantly"""
    
    if len(result.iterations) < 2:
        return True, "Not enough iterations to check regression"
    
    for i in range(len(result.iterations) - 1):
        curr = result.iterations[i].average_score
        next_score = result.iterations[i+1].average_score
        
        if next_score < curr - 5:
            return False, f"Regression in iteration {i+1}: {curr:.1f} → {next_score:.1f}"
    
    return True, "No regressions detected"


def check_convergence(result) -> tuple[bool, str]:
    """Verify convergence logic works"""
    
    # If already excellent (>85), should stop early
    if result.best_score >= 85 and len(result.iterations) > 3:
        return False, f"Score was {result.best_score:.1f} but ran {len(result.iterations)} iterations (should stop early)"
    
    # If converged flag is set, should have reason
    if result.converged and not result.convergence_reason:
        return False, "Marked as converged but no reason given"
    
    return True, "Convergence logic working correctly"


def check_cost_reasonable(result, max_cost: float = 1.0) -> tuple[bool, str]:
    """Verify cost didn't explode"""
    
    if result.total_cost_usd > max_cost:
        return False, f"Cost ${result.total_cost_usd:.2f} exceeds ${max_cost:.2f} limit"
    
    return True, f"Cost ${result.total_cost_usd:.2f} within budget"


def check_improvement(result) -> tuple[bool, str]:
    """Verify some improvement was made"""
    
    if result.best_score <= result.initial_score + 1:
        return False, f"No meaningful improvement: {result.initial_score:.1f} → {result.best_score:.1f}"
    
    return True, f"Improved by {result.best_score - result.initial_score:.1f} points"


def main():
    print("="*60)
    print("TESTING OPTIMIZER IMPROVEMENTS")
    print("="*60)
    
    # Load a sample test case from canonical suite
    from canonical_suite.suite_loader import CanonicalTestSuite
    
    test_suite_dir = Path(__file__).parent / "canonical_suite" / "tests"
    
    if not test_suite_dir.exists():
        print(f"❌ Test directory not found: {test_suite_dir}")
        return 1
    
    try:
        suite = CanonicalTestSuite.load(str(test_suite_dir))
        if len(suite) == 0:
            print("❌ No test cases found in canonical_suite/tests/")
            return 1
        
        # Use first test case
        test_case = suite.test_cases[0]
        print(f"\n✓ Loaded test case: {test_case.id} ({test_case.name})")
    except Exception as e:
        print(f"❌ Failed to load test case: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create specification from test case
    from canonical_suite.suite_runner import SuiteRunner
    runner = SuiteRunner()
    spec = runner._create_specification(test_case)
    
    # Run optimization
    print("\nRunning optimization with improvements...")
    print("-"*60)
    
    orchestrator = Orchestrator(
        max_budget_usd=1.0,
        use_golden_set=True,
        use_checker=True
    )
    
    initial_prompt = "You are a helpful assistant. Please respond to the user's request."
    
    try:
        result = orchestrator.optimize(
            initial_prompt=initial_prompt,
            test_case=test_case,
            specification=spec,
            max_iterations=3,
            accept_threshold=2.0
        )
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    checks = {
        "No Regressions": check_no_regression(result),
        "Convergence Logic": check_convergence(result),
        "Cost Reasonable": check_cost_reasonable(result, max_cost=1.0),
        "Shows Improvement": check_improvement(result)
    }
    
    all_passed = True
    for check_name, (passed, message) in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{check_name:20s}: {status}")
        print(f"  {message}")
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Initial Score: {result.initial_score:.1f}")
    print(f"Final Score:   {result.best_score:.1f}")
    print(f"Current Score: {result.current_score:.1f}")
    print(f"Improvement:   {result.best_score - result.initial_score:+.1f}")
    print(f"Total Cost:    ${result.total_cost_usd:.4f}")
    print(f"Iterations:    {len(result.iterations)}")
    print(f"Converged:     {'Yes' if result.converged else 'No'}")
    if result.converged:
        print(f"Reason:        {result.convergence_reason}")
    print("="*60)
    
    if all_passed:
        print("\n✓ All checks PASSED! Optimizer improvements are working.")
        print("Ready to run on full dataset.")
        return 0
    else:
        print("\n✗ Some checks FAILED. Review logs and fix issues before full run.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
