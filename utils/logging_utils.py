"""Production-grade structured logging utilities."""
import logging
import sys
import json
import uuid
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path
from contextvars import ContextVar

# Context variable for correlation IDs
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_data["correlation_id"] = corr_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)


class ContextLogger:
    """Logger with context management."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set context variables for logging."""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear context variables."""
        self._context.clear()
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add context to log record."""
        result = extra.copy()
        result.update(self._context)
        return result
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, extra=self._add_context(kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, extra=self._add_context(kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        # Extract special logging parameters that can't be in extra
        exc_info = kwargs.pop('exc_info', None)
        stack_info = kwargs.pop('stack_info', None)
        stacklevel = kwargs.pop('stacklevel', None)
        
        # Pass remaining kwargs as extra context
        self.logger.error(
            message,
            extra=self._add_context(kwargs),
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, extra=self._add_context(kwargs))
    
    def exception(self, message: str, **kwargs):
        """Log exception with context."""
        self.logger.exception(message, extra=self._add_context(kwargs))


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> ContextLogger:
    """
    Set up production-grade structured logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_type: "json" for structured JSON, "text" for human-readable
        log_file: Optional file path for file logging
    
    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger("prompt_optimizer")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Clear existing handlers
    
    # Console handler - use stderr to avoid conflicts with tqdm progress bars
    console_handler = logging.StreamHandler(sys.stderr)
    
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return ContextLogger(logger)


def set_correlation_id(corr_id: Optional[str] = None) -> str:
    """
    Set correlation ID for request tracking.
    
    Args:
        corr_id: Optional correlation ID (generates new if None)
    
    Returns:
        Correlation ID
    """
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)
    return corr_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()
