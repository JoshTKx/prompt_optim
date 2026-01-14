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
    Save optimization result to JSON file, organized by test ID in subdirectories.
    
    Args:
        result: OptimizationResult to save
        run_id: Run identifier
        prompt_id: Optional prompt identifier (e.g., "prompt_EDGE-001" or "prompt_IF-001")
        output_dir: Output directory (defaults to outputs/optimization_runs/)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / "optimization_runs"
    
    # Extract test ID from prompt_id to organize into folders
    # Format: "prompt_EDGE-001_abc12345" -> "EDGE-001"
    # Format: "prompt_GAP-MENTAL_HEALTH-001_def67890" -> "GAP-MENTAL_HEALTH-001"
    test_id = None
    if prompt_id:
        # Remove "prompt_" prefix if present
        test_id_str = prompt_id.replace("prompt_", "")
        # Split by underscore - test_id is everything except the last part (which is run_id)
        parts = test_id_str.split("_")
        if len(parts) >= 2:
            # Rejoin all parts except the last one (run_id) to get the full test_id
            # This handles cases like "GAP-MENTAL_HEALTH-001" which may have underscores
            test_id = "_".join(parts[:-1])
        elif len(parts) == 1:
            # If no underscore, the whole thing is the test_id
            test_id = parts[0]
    
    # Create subdirectory based on test_id
    if test_id:
        test_dir = output_dir / test_id
        test_dir.mkdir(parents=True, exist_ok=True)
        save_dir = test_dir
    else:
        # If no test_id, save to a "misc" folder
        misc_dir = output_dir / "misc"
        misc_dir.mkdir(parents=True, exist_ok=True)
        save_dir = misc_dir
    
    # Create filename
    if prompt_id:
        filename = f"{prompt_id}_{run_id}.json"
    else:
        filename = f"optimization_{run_id}.json"
    
    filepath = save_dir / filename
    
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
