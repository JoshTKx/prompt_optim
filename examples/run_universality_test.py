"""Example: Run universality validation against canonical test suite."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from canonical_suite.suite_loader import CanonicalTestSuite
from validation.universality_validator import UniversalityValidator
from validation.coverage_analyzer import CoverageAnalyzer
from pathlib import Path

if __name__ == "__main__":
    # Load test suite
    test_dir = Path(__file__).parent.parent / "canonical_suite" / "tests"
    suite = CanonicalTestSuite.load(str(test_dir))
    
    print(f"Loaded {len(suite)} test cases")
    
    # Analyze coverage
    print("\nAnalyzing test suite coverage...")
    analyzer = CoverageAnalyzer()
    coverage = analyzer.analyze_coverage(suite)
    print(f"Coverage Report:")
    print(f"  Total Tests: {coverage.data['total_tests']}")
    print(f"  By Category: {coverage.data['by_category']}")
    print(f"  By Archetype: {coverage.data['by_archetype']}")
    print(f"  By Domain: {coverage.data['by_domain']}")
    
    # Run universality validation
    print("\nRunning universality validation...")
    print("(This will take a while - running optimizer on all test cases)")
    
    validator = UniversalityValidator()
    report = validator.validate_optimizer(
        test_suite=suite,
        sample_size=40  # Start with 20 tests for quick validation
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("UNIVERSALITY VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Tests Passed: {report.tests_passed}")
    print(f"Pass Rate: {report.pass_rate:.1%}")
    print(f"\nBy Category:")
    for cat, stats in report.data["by_category"].items():
        print(f"  {cat}: {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']:.1%})")
    print(f"\nBy Archetype:")
    for arch, stats in report.data["by_archetype"].items():
        print(f"  {arch}: {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']:.1%})")
    print(f"\nFailed Tests: {report.data['failed_tests']}")
    print(f"Total Cost: ${report.data['cost_total']:.2f}")
    print(f"Total Time: {report.data['time_total_hours']:.2f} hours")
    
    # Save report
    output_dir = Path(__file__).parent.parent / "outputs" / "universality_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"universality_report_{report.data['timestamp'][:10]}.json"
    report.save(str(report_path))
    print(f"\nReport saved to: {report_path}")
