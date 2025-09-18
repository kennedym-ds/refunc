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
                    
                    # Check if this exception type is retryable (ignoring attempt count)
                    is_exception_type_retryable = False
                    for exc_type in config.retryable_exceptions:
                        if isinstance(e, exc_type):
                            is_exception_type_retryable = True
                            break
                    
                    # Check if it's explicitly non-retryable
                    is_explicitly_non_retryable = False
                    for exc_type in config.non_retryable_exceptions:
                        if isinstance(e, exc_type):
                            is_explicitly_non_retryable = True
                            break
                    
                    # If explicitly non-retryable or not in retryable types, raise immediately
                    if is_explicitly_non_retryable or not is_exception_type_retryable:
                        raise e
                    
                    # If we've reached max attempts, don't retry anymore
                    if attempt >= config.max_attempts:
                        break
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    delay = config.calculate_delay(attempt)
                    time.sleep(delay)
            
            # All retry attempts failed for a retryable exception
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
    """Context manager for retryable operations with automatic retry."""
    
    def __init__(self, config: RetryConfig, operation_name: Optional[str] = None):
        self.config = config
        self.operation_name = operation_name
        self.attempts_made = 0
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # Success, no exception
            
        self.attempts_made += 1
        
        # Check if we should retry this exception
        if not self.config.should_retry(exc_val, self.attempts_made):
            # Non-retryable exception, let it propagate
            return False
            
        # Check if we have more attempts left
        if self.attempts_made < self.config.max_attempts:
            # Sleep and suppress the exception to retry
            delay = self.config.calculate_delay(self.attempts_made)
            time.sleep(delay)
            return True  # Suppress exception, continue
        
        # All attempts exhausted
        total_time = time.time() - (self.start_time or 0)
        raise RetryError(
            original_error=exc_val,
            attempts=self.attempts_made,
            total_time=total_time,
            operation_name=self.operation_name
        )
    
    def execute(self, operation_func):
        """Execute an operation with retries."""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.attempts_made = attempt
                return operation_func()
            except Exception as e:
                if not self.config.should_retry(e, attempt):
                    raise e
                    
                if attempt < self.config.max_attempts:
                    delay = self.config.calculate_delay(attempt)
                    time.sleep(delay)
                else:
                    total_time = time.time() - (self.start_time or 0)
                    raise RetryError(
                        original_error=e,
                        attempts=attempt,
                        total_time=total_time,
                        operation_name=self.operation_name
                    )