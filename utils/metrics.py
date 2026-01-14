"""Metrics and monitoring utilities."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "unit": self.unit
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics collector.
        
        Args:
            storage_path: Path to store metrics (defaults to outputs/metrics)
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "outputs" / "metrics"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._metrics: List[Metric] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        self._metrics.append(metric)
    
    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        self._counters[name] += value
        self.record(f"{name}_total", self._counters[name], tags=tags, unit="count")
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        self._gauges[name] = value
        self.record(name, value, tags=tags)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value."""
        self._histograms[name].append(value)
        self.record(name, value, tags=tags)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    "count": len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "avg": sum(values) / len(values) if values else None
                }
                for name, values in self._histograms.items()
            },
            "total_metrics": len(self._metrics)
        }
    
    def export(self, filepath: Optional[str] = None) -> str:
        """
        Export metrics to JSON.
        
        Args:
            filepath: Optional file path (auto-generates if None)
        
        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filepath = str(self.storage_path / f"metrics_{timestamp}.json")
        
        data = {
            "summary": self.get_summary(),
            "metrics": [m.to_dict() for m in self._metrics[-1000:]]  # Last 1000 metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def clear(self):
        """Clear all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
