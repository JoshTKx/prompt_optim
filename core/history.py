"""History tracking for optimization runs."""
from typing import List
from models.result import Iteration


class OptimizationHistory:
    """Tracks optimization trajectory."""
    
    def __init__(self):
        self.iterations: List[Iteration] = []
    
    def add_iteration(self, iteration: Iteration):
        """Add an iteration to history."""
        self.iterations.append(iteration)
    
    def get_best_iteration(self) -> Iteration:
        """Get iteration with highest score."""
        if not self.iterations:
            raise ValueError("No iterations in history")
        return max(self.iterations, key=lambda i: i.average_score)
    
    def has_converged(
        self,
        improvement_threshold: float = 2.0,
        window: int = 2
    ) -> tuple[bool, str]:
        """
        Check if optimization has converged.
        
        Returns:
            (has_converged, reason)
        """
        if len(self.iterations) < window + 1:
            return False, "Not enough iterations"
        
        # Check if score improved in last window
        recent_scores = [i.average_score for i in self.iterations[-window:]]
        previous_score = self.iterations[-window-1].average_score
        
        max_recent = max(recent_scores)
        improvement = max_recent - previous_score
        
        if improvement < improvement_threshold:
            return True, f"No significant improvement in last {window} iterations"
        
        # Check if we've hit target
        if max_recent >= 95.0:  # Near perfect
            return True, "Score near perfect (95+)"
        
        return False, "Still improving"
    
    def get_all(self) -> List[Iteration]:
        """Get all iterations."""
        return self.iterations.copy()
