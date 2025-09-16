"""
Intelligent caching system for data and model operations.

This module provides a sophisticated caching system with memory and disk-based
storage, automatic cache invalidation, and compression support.
"""

import os
import pickle
import hashlib
import time
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, Tuple
from functools import wraps
import threading


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(
        self, 
        value: Any, 
        created_at: float,
        access_count: int = 0,
        last_accessed: Optional[float] = None,
        size_bytes: Optional[int] = None
    ):
        self.value = value
        self.created_at = created_at
        self.access_count = access_count
        self.last_accessed = last_accessed or created_at
        self.size_bytes = size_bytes
    
    def access(self) -> Any:
        """Access the cached value and update metadata."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at
    
    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        max_memory_mb: Optional[float] = None
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size_bytes = 0
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        if not self.ttl_seconds:
            return
            
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.created_at > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self) -> None:
        """Remove least recently used entry."""
        if not self._cache:
            return
            
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        self._remove_entry(lru_key)
    
    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update size tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            if entry.size_bytes:
                self._total_size_bytes -= entry.size_bytes
    
    def _enforce_limits(self) -> None:
        """Enforce cache size and memory limits."""
        # Remove expired entries first
        self._evict_expired()
        
        # Enforce memory limit
        while (self.max_memory_bytes and 
               self._total_size_bytes > self.max_memory_bytes and 
               self._cache):
            self._evict_lru()
        
        # Enforce entry count limit
        while len(self._cache) > self.max_size:
            self._evict_lru()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            self._evict_expired()
            
            if key in self._cache:
                return self._cache[key].access()
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value into cache."""
        with self._lock:
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
            
            # Enforce limits
            self._enforce_limits()
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "entry_count": len(self._cache),
                "total_size_mb": self._total_size_bytes / (1024 * 1024),
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024) if self.max_memory_bytes else None,
                "ttl_seconds": self.ttl_seconds,
            }


class DiskCache:
    """Disk-based cache with compression support."""
    
    def __init__(
        self, 
        cache_dir: Union[str, Path],
        compress: bool = True,
        max_size_mb: Optional[float] = None
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None
        self._lock = threading.RLock()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        extension = ".pkl.gz" if self.compress else ".pkl"
        return self.cache_dir / f"{safe_key}{extension}"
    
    def _save_to_disk(self, path: Path, value: Any) -> None:
        """Save value to disk with optional compression."""
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.compress:
            with gzip.open(path, 'wb') as f:
                f.write(data)
        else:
            with open(path, 'wb') as f:
                f.write(data)
    
    def _load_from_disk(self, path: Path) -> Any:
        """Load value from disk with optional decompression."""
        if self.compress:
            with gzip.open(path, 'rb') as f:
                data = f.read()
        else:
            with open(path, 'rb') as f:
                data = f.read()
        
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            
            if path.exists():
                try:
                    return self._load_from_disk(path)
                except Exception:
                    # Remove corrupted cache file
                    path.unlink()
                    return None
            
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value into disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            
            try:
                self._save_to_disk(path, value)
                self._enforce_size_limit()
            except Exception:
                # Clean up on failure
                if path.exists():
                    path.unlink()
                raise
    
    def remove(self, key: str) -> bool:
        """Remove entry from disk cache."""
        with self._lock:
            path = self._get_cache_path(key)
            
            if path.exists():
                path.unlink()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.pkl*"):
                cache_file.unlink()
    
    def _enforce_size_limit(self) -> None:
        """Enforce disk cache size limit."""
        if not self.max_size_bytes:
            return
        
        # Get all cache files with their sizes and modification times
        cache_files = []
        total_size = 0
        
        for cache_file in self.cache_dir.glob("*.pkl*"):
            try:
                stat = cache_file.stat()
                cache_files.append((cache_file, stat.st_size, stat.st_mtime))
                total_size += stat.st_size
            except OSError:
                continue
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by modification time (oldest first) and remove files
        cache_files.sort(key=lambda x: x[2])
        
        for cache_file, size, _ in cache_files:
            cache_file.unlink()
            total_size -= size
            
            if total_size <= self.max_size_bytes:
                break


def cache_result(
    cache_key_fn: Optional[Callable] = None,
    ttl_seconds: Optional[float] = None,
    use_disk: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    compress: bool = True
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        cache_key_fn: Function to generate cache key from args/kwargs
        ttl_seconds: Time-to-live for cache entries
        use_disk: Whether to use disk-based caching
        cache_dir: Directory for disk cache (if use_disk=True)
        compress: Whether to compress disk cache files
    
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        if use_disk:
            if cache_dir is None:
                cache_dir_path = Path.home() / ".refunc_cache" / func.__name__
            else:
                cache_dir_path = Path(cache_dir)
            cache = DiskCache(cache_dir_path, compress=compress)
        else:
            cache = MemoryCache(ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            if cache_key_fn:
                cache_key = cache_key_fn(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Expose cache for manual operations
        wrapper._cache = cache
        return wrapper
    
    return decorator