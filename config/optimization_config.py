"""Optimization configuration settings."""
import os
from dotenv import load_dotenv

load_dotenv()


class OptimizationConfig:
    """Configuration for optimization parameters."""
    
    # Iteration limits
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    TARGET_SCORE: float = float(os.getenv("TARGET_SCORE", "95.0"))
    
    # Cost management
    COST_BUDGET: float = float(os.getenv("COST_BUDGET", "2.0"))  # USD per run
    MAX_BUDGET_PER_OPTIMIZATION: float = float(os.getenv("MAX_BUDGET_PER_OPTIMIZATION", "5.0"))  # Max budget per optimization
    MAX_TOKENS_DEFAULT: int = int(os.getenv("MAX_TOKENS_DEFAULT", "2000"))  # Default max tokens for LLM calls
    
    # Convergence thresholds
    SCORE_IMPROVEMENT_THRESHOLD: float = 2.0  # Minimum improvement to continue
    CONVERGENCE_WINDOW: int = 2  # Iterations without improvement to stop
    
    # Temperature schedules for reviser
    TEMPERATURE_SCHEDULE: dict = {
        "early": 0.7,      # Iterations 1-2: explore boldly
        "mid": 0.5,        # Iterations 3-4: refine
        "late": 0.3        # Iterations 5+: conservative tweaks
    }
    
    # Judge settings
    JUDGE_TEMPERATURE: float = 0.2  # Low temperature for consistent judging
    JUDGE_MAX_TOKENS: int = 4000  # Increased from 2000 to prevent truncation
    
    # Reviser settings
    REVISER_MAX_TOKENS: int = 4000
    
    # Progress saving
    SAVE_INTERVAL: int = int(os.getenv("SAVE_INTERVAL", "1"))  # Save after every N iterations
    SAVE_PROGRESS: bool = os.getenv("SAVE_PROGRESS", "true").lower() == "true"  # Enable periodic saving
    
    # Parallelism settings
    MAX_PARALLEL_OPTIMIZATIONS: int = int(os.getenv("MAX_PARALLEL_OPTIMIZATIONS", "5"))  # Max concurrent optimizations
    MAX_PARALLEL_TESTS: int = int(os.getenv("MAX_PARALLEL_TESTS", "5"))  # Max concurrent test executions
    ENABLE_PARALLEL_TESTS: bool = os.getenv("ENABLE_PARALLEL_TESTS", "true").lower() == "true"  # Enable parallel test execution
    
    # Golden set settings
    GOLDEN_SET_THRESHOLD: float = float(os.getenv("GOLDEN_SET_THRESHOLD", "85.0"))  # Minimum score to add to golden set
    REGRESSION_CHECK_ENABLED: bool = os.getenv("REGRESSION_CHECK_ENABLED", "true").lower() == "true"  # Enable regression testing
    REGRESSION_PASS_RATE_THRESHOLD: float = float(os.getenv("REGRESSION_PASS_RATE_THRESHOLD", "0.90"))  # Minimum pass rate (90%)
    GOLDEN_SET_PATH: str = os.getenv("GOLDEN_SET_PATH", "outputs/golden_set/golden_set.json")  # Path to golden set file