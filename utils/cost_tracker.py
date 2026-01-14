"""Cost tracking for optimization runs."""
from typing import Dict, Optional, Any
from datetime import datetime


class CostTracker:
    """Tracks API costs across optimization runs."""
    
    def __init__(self):
        self.total_cost: float = 0.0
        self.by_component: Dict[str, float] = {
            "judge": 0.0,
            "reviser": 0.0,
            "target_model": 0.0
        }
        self.by_iteration: Dict[int, float] = {}
    
    def add_cost(
        self,
        component: str,
        cost: float,
        iteration: Optional[int] = None
    ):
        """Record a cost."""
        self.total_cost += cost
        if component in self.by_component:
            self.by_component[component] += cost
        
        if iteration is not None:
            if iteration not in self.by_iteration:
                self.by_iteration[iteration] = 0.0
            self.by_iteration[iteration] += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "by_component": {k: round(v, 4) for k, v in self.by_component.items()},
            "by_iteration": {k: round(v, 4) for k, v in self.by_iteration.items()}
        }
    
    def reset(self):
        """Reset tracker."""
        self.total_cost = 0.0
        self.by_component = {"judge": 0.0, "reviser": 0.0, "target_model": 0.0}
        self.by_iteration = {}
