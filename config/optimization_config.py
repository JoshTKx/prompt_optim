"""Optimization configuration settings."""
import os
from dotenv import load_dotenv

load_dotenv()


class OptimizationConfig:
    """Configuration for optimization parameters."""
    
    # Iteration limits
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    TARGET_SCORE: float = float(os.getenv("TARGET_SCORE", "85.0"))
    
    # Cost management
    COST_BUDGET: float = float(os.getenv("COST_BUDGET", "2.0"))  # USD per run
    
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
    JUDGE_MAX_TOKENS: int = 2000
    
    # Reviser settings
    REVISER_MAX_TOKENS: int = 4000
    
    # Progress saving
    SAVE_INTERVAL: int = int(os.getenv("SAVE_INTERVAL", "1"))  # Save after every N iterations
    SAVE_PROGRESS: bool = os.getenv("SAVE_PROGRESS", "true").lower() == "true"  # Enable periodic saving
