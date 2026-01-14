"""Utility for saving optimization results periodically."""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from models.result import OptimizationResult
from utils.logging_utils import setup_logging

logger = setup_logging()


def save_optimization_result(
    result: OptimizationResult,
    run_id: str,
    prompt_id: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save optimization result to JSON file.
    
    Args:
        result: OptimizationResult to save
        run_id: Run identifier
        prompt_id: Optional prompt identifier
        output_dir: Output directory (defaults to outputs/optimization_runs/)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "optimization_runs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    if prompt_id:
        filename = f"{prompt_id}_{run_id}.json"
    else:
        filename = f"optimization_{run_id}.json"
    
    filepath = output_dir / filename
    
    # Convert result to dict (Pydantic model)
    result_dict = result.model_dump(mode='json')
    
    # Add metadata
    result_dict["_metadata"] = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "saved_at": result.timestamp.isoformat() if result.timestamp else None
    }
    
    # Save to file
    with open(filepath, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    logger.debug(
        "Saved optimization result",
        filepath=str(filepath),
        run_id=run_id,
        iterations=len(result.iterations),
        best_score=result.best_score
    )
    
    return filepath


def load_optimization_result(filepath: Path) -> Optional[OptimizationResult]:
    """
    Load optimization result from JSON file.
    
    Args:
        filepath: Path to saved result file
    
    Returns:
        OptimizationResult or None if loading fails
    """
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Remove metadata
        data.pop("_metadata", None)
        
        # Reconstruct from dict
        return OptimizationResult(**data)
    except Exception as e:
        logger.error(f"Failed to load optimization result: {e}", filepath=str(filepath))
        return None
