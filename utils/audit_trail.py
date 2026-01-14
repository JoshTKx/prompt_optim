"""Audit trail for tracking all operations."""
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class AuditEventType(Enum):
    """Types of audit events."""
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    PROMPT_VERSIONED = "prompt_versioned"
    TEST_EXECUTED = "test_executed"
    JUDGE_EVALUATED = "judge_evaluated"
    PROMPT_REVISED = "prompt_revised"
    ERROR_OCCURRED = "error_occurred"
    CONFIG_CHANGED = "config_changed"


@dataclass
class AuditEvent:
    """An audit trail event."""
    event_type: AuditEventType
    timestamp: datetime
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.user_id:
            data["user_id"] = self.user_id
        if self.details:
            data["details"] = self.details
        return data


class AuditTrail:
    """Manages audit trail for all operations."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize audit trail.
        
        Args:
            storage_path: Path to store audit logs (defaults to outputs/audit)
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "outputs" / "audit"
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._events: list[AuditEvent] = []
    
    def log(
        self,
        event_type: AuditEventType,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            user_id=user_id,
            details=details or {}
        )
        
        self._events.append(event)
        
        # Write to file (append mode)
        audit_file = self.storage_path / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100
    ) -> list[AuditEvent]:
        """Get audit events with filters."""
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]
        
        return events[-limit:]
    
    def export(self, filepath: str, event_type: Optional[AuditEventType] = None):
        """Export audit trail to JSON file."""
        events = self.get_events(event_type=event_type, limit=10000)
        data = [e.to_dict() for e in events]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath


# Global audit trail instance
_global_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    """Get global audit trail instance."""
    global _global_audit_trail
    if _global_audit_trail is None:
        _global_audit_trail = AuditTrail()
    return _global_audit_trail
