"""
Retry mechanisms and decorators for the refunc library.

This module provides sophisticated retry logic with exponential backoff,
jitter, and custom retry conditions for robust error recovery.
"""

import time
import random
import functools
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from .core import RefuncError, OperationError


class RetryError(OperationError):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(
        self, 
        original_error: Exception,
        attempts: int,
        total_time: float,
        operation_name: Optional[str] = None
    ):
        context = {
            "attempts": attempts,
            "total_time_seconds": round(total_time, 2),
            "operation_name": operation_name or "unknown"
        }
        
        suggestion = "Check the underlying cause and consider increasing retry limits."
        
        super().__init__(
            f"Operation failed after {attempts} attempts",
            context=context,
            suggestion=suggestion,
            original_error=original_error
        )


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self.non_retryable_exceptions = non_retryable_exceptions or []
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False
            
        # Check non-retryable exceptions first
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
                
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0, delay)


def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator that adds retry logic to a function.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: List of exceptions that should trigger retries
        non_retryable_exceptions: List of exceptions that should not trigger retries
        on_retry: Callback function called on each retry attempt
    
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        break
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        time.sleep(delay)
            
            # All attempts failed
            total_time = time.time() - start_time
            if last_exception is None:
                last_exception = Exception("Unknown error occurred")
            
            raise RetryError(
                original_error=last_exception,
                attempts=config.max_attempts,
                total_time=total_time,
                operation_name=func.__name__
            )
        
        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations."""
    
    def __init__(self, config: RetryConfig, operation_name: Optional[str] = None):
        self.config = config
        self.operation_name = operation_name
        self.attempt = 0
        self.start_time = None
        self.last_exception = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # No exception occurred
            
        self.attempt += 1
        self.last_exception = exc_val
        
        if self.config.should_retry(exc_val, self.attempt):
            if self.attempt < self.config.max_attempts:
                delay = self.config.calculate_delay(self.attempt)
                time.sleep(delay)
                return True  # Suppress the exception, retry
        
        # Convert to RetryError if all attempts exhausted
        if self.attempt >= self.config.max_attempts:
            total_time = time.time() - (self.start_time or 0)
            raise RetryError(
                original_error=exc_val,
                attempts=self.attempt,
                total_time=total_time,
                operation_name=self.operation_name
            )
        
        return False  # Don't suppress the exception