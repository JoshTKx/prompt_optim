"""Production-grade error handling with retries and circuit breakers."""
import time
from typing import Callable, TypeVar, Optional, List, Any
from functools import wraps
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerState:
    """State of circuit breaker."""
    failures: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    timeout: float = 60.0  # seconds


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.state = CircuitBreakerState(
            failure_threshold=failure_threshold,
            timeout=timeout
        )
        self.expected_exception = expected_exception
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state.state == "open":
            if self._should_attempt_reset():
                self.state.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset circuit breaker."""
        if self.state.last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.state.last_failure_time).total_seconds()
        return elapsed >= self.state.timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state.state == "half_open":
            self.state.state = "closed"
        self.state.failures = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.state.failures += 1
        self.state.last_failure_time = datetime.utcnow()
        
        if self.state.failures >= self.state.failure_threshold:
            self.state.state = "open"


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        on_retry: Optional callback called on each retry
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        break
                    
                    # Calculate delay
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    log_error: bool = True,
    reraise: bool = True
):
    """
    Decorator for error handling with logging.
    
    Args:
        severity: Error severity level
        log_error: Whether to log the error
        reraise: Whether to re-raise the exception
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    from utils.logging_utils import setup_logging
                    logger = setup_logging()
                    logger.error(
                        f"Error in {func.__name__}: {str(e)}",
                        severity=severity.value,
                        function=func.__name__,
                        exception_type=type(e).__name__
                    )
                
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator
