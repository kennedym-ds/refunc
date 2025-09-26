#!/usr/bin/env python3
"""
Caching Examples - Refunc Decorators

This example demonstrates the comprehensive caching capabilities including
result caching, disk caching, cache invalidation, and optimization strategies
for ML workflows.

Key Features Demonstrated:
- Result caching decorator
- Memory and disk cache backends
- Cache invalidation strategies
- Performance optimization patterns
- Cache statistics and monitoring
- Integration with file operations
"""

import os
import sys
import time
import random
import hashlib
from typing import List, Dict, Any, Optional

# Handle missing dependencies gracefully
try:
    from refunc.utils import (
        cache_result, MemoryCache, DiskCache, 
        CacheEntry, FileHandler
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def basic_caching_examples():
    """Demonstrate basic result caching functionality."""
    print("💾 Basic Result Caching")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic caching examples:
from refunc.utils import cache_result

# Simple result caching
@cache_result(ttl_seconds=300)
def expensive_computation(n):
    # Simulate expensive operation
    time.sleep(2)
    return n ** 2 + n * 10

# Disk-based caching
@cache_result(ttl_seconds=3600, use_disk=True)
def load_and_process_data(file_path):
    data = load_large_dataset(file_path)
    return preprocess_data(data)

# Cache with custom key
@cache_result(
    ttl_seconds=1800,
    key_func=lambda model_type, params: f"{model_type}_{hash(str(params))}"
)
def train_model(model_type, params):
    return expensive_training(model_type, params)
        """)
        return
    
    print("⚡ Testing basic caching performance:")
    
    @cache_result(ttl_seconds=10)  # Short TTL for demo
    def expensive_calculation(n: int) -> float:
        """Simulate an expensive mathematical computation."""
        print(f"     🔄 Computing expensive operation for n={n}...")
        time.sleep(0.3)  # Simulate computation time
        result = sum(i ** 0.5 for i in range(n))
        return result
    
    @cache_result(ttl_seconds=15)
    def data_analysis(data_size: int, analysis_type: str = "basic") -> Dict[str, Any]:
        """Simulate data analysis with caching."""
        print(f"     📊 Analyzing {data_size} samples with {analysis_type} analysis...")
        time.sleep(0.2)  # Simulate analysis time
        
        # Generate mock analysis results
        data = [random.random() for _ in range(data_size)]
        return {
            "size": data_size,
            "type": analysis_type,
            "mean": sum(data) / len(data),
            "min": min(data),
            "max": max(data),
            "computed_at": time.time()
        }
    
    # Test caching performance
    test_values = [1000, 2000, 1000]  # Note: 1000 appears twice
    
    print("   First pass (cache misses):")
    start_time = time.time()
    results = []
    for i, n in enumerate(test_values):
        print(f"   Call {i+1}: n={n}")
        result = expensive_calculation(n)
        results.append(result)
        print(f"     ✓ Result: {result:.2f}")
    first_pass_time = time.time() - start_time
    
    print(f"\n   Second pass (cache hits):")
    start_time = time.time()
    cached_results = []
    for i, n in enumerate(test_values):
        print(f"   Call {i+1}: n={n}")
        result = expensive_calculation(n)
        cached_results.append(result)
        print(f"     ✓ Result: {result:.2f}")
    second_pass_time = time.time() - start_time
    
    print(f"\n   ⏱️  Performance comparison:")
    print(f"     First pass: {first_pass_time:.2f}s")
    print(f"     Second pass: {second_pass_time:.2f}s")
    print(f"     Speedup: {first_pass_time/second_pass_time:.1f}x")
    print(f"     Results match: {results == cached_results}")
    
    # Test data analysis caching
    print("\n   📈 Data analysis caching:")
    analysis_start = time.time()
    analysis1 = data_analysis(500, "detailed")
    first_analysis_time = time.time() - analysis_start
    
    analysis_start = time.time()
    analysis2 = data_analysis(500, "detailed")  # Same parameters - should be cached
    second_analysis_time = time.time() - analysis_start
    
    print(f"     First analysis: {first_analysis_time:.2f}s")
    print(f"     Cached analysis: {second_analysis_time:.2f}s")
    print(f"     Cache hit: {analysis1 == analysis2}")


def disk_caching_examples():
    """Demonstrate disk-based caching for persistent storage."""
    print("\n💿 Disk-Based Caching")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Disk caching examples:
from refunc.utils import cache_result, DiskCache

# Persistent disk caching
@cache_result(ttl_seconds=86400, use_disk=True, cache_dir="./cache")
def process_large_dataset(dataset_path):
    # Load and process large dataset
    data = load_dataset(dataset_path)
    processed = heavy_preprocessing(data)
    return processed

# Manual disk cache management
disk_cache = DiskCache(cache_dir="./model_cache")

def get_trained_model(model_config):
    cache_key = f"model_{hash(str(model_config))}"
    
    # Try to load from cache
    cached_model = disk_cache.get(cache_key)
    if cached_model:
        return cached_model
    
    # Train model if not cached
    model = train_model(model_config)
    disk_cache.set(cache_key, model, ttl=3600)
    return model
        """)
        return
    
    print("💾 Testing persistent disk caching:")
    
    @cache_result(ttl_seconds=30, use_disk=True)
    def simulate_data_processing(dataset_id: str, processing_level: int = 1) -> Dict[str, Any]:
        """Simulate processing that should persist across runs."""
        print(f"     🔄 Processing dataset {dataset_id} (level {processing_level})...")
        time.sleep(0.15)  # Simulate processing time
        
        # Generate processing results
        sample_size = 1000 * processing_level
        mock_data = [random.random() for _ in range(sample_size)]
        
        return {
            "dataset_id": dataset_id,
            "processing_level": processing_level,
            "sample_size": sample_size,
            "checksum": hashlib.md5(str(mock_data).encode()).hexdigest()[:8],
            "processed_at": time.time()
        }
    
    @cache_result(ttl_seconds=60, use_disk=True)
    def simulate_model_training(model_type: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate expensive model training with disk caching."""
        print(f"     🤖 Training {model_type} with params {hyperparams}...")
        time.sleep(0.2)  # Simulate training time
        
        # Simulate training results
        accuracy = 0.8 + random.random() * 0.15
        return {
            "model_type": model_type,
            "hyperparams": hyperparams,
            "accuracy": accuracy,
            "loss": 1.0 - accuracy,
            "trained_at": time.time()
        }
    
    # Test disk caching
    datasets = ["dataset_A", "dataset_B", "dataset_A"]  # dataset_A repeated
    
    print("   📊 Processing datasets:")
    for i, dataset_id in enumerate(datasets):
        print(f"   Processing {i+1}: {dataset_id}")
        result = simulate_data_processing(dataset_id, processing_level=2)
        print(f"     ✓ Size: {result['sample_size']}, Checksum: {result['checksum']}")
    
    # Test model training caching
    print("\n   🤖 Model training with caching:")
    model_configs = [
        ("random_forest", {"n_estimators": 100, "max_depth": 10}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
        ("random_forest", {"n_estimators": 100, "max_depth": 10})  # Repeat
    ]
    
    for i, (model_type, params) in enumerate(model_configs):
        print(f"   Training {i+1}: {model_type}")
        result = simulate_model_training(model_type, params)
        print(f"     ✓ Accuracy: {result['accuracy']:.3f}")
    
    print("\n   💡 Disk cache persists between runs - try running this example again!")


def cache_management_examples():
    """Demonstrate cache management and statistics."""
    print("\n📊 Cache Management")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Cache management examples:
from refunc.utils import MemoryCache, DiskCache

# Memory cache with size limits
memory_cache = MemoryCache(max_size=100, max_memory_mb=50)

# Monitor cache statistics
stats = memory_cache.get_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit ratio: {stats['hit_ratio']:.2%}")

# Cache invalidation
memory_cache.invalidate("pattern*")  # Invalidate by pattern
memory_cache.clear()  # Clear all cache

# Disk cache management
disk_cache = DiskCache(cache_dir="./cache")
disk_cache.cleanup_expired()  # Remove expired entries
disk_cache.get_size_info()  # Get cache size statistics
        """)
        return
    
    print("📈 Cache statistics and management:")
    
    # Create manual cache instances for demonstration
    memory_cache = MemoryCache(max_size=5)  # Small size for demo
    
    # Populate cache with test data
    print("   📥 Populating cache...")
    test_data = [
        ("user_1", {"name": "Alice", "score": 95}),
        ("user_2", {"name": "Bob", "score": 87}),
        ("user_3", {"name": "Charlie", "score": 92}),
        ("user_4", {"name": "Diana", "score": 98}),
        ("user_5", {"name": "Eve", "score": 85}),
        ("user_6", {"name": "Frank", "score": 90})  # This should evict oldest
    ]
    
    for key, value in test_data:
        memory_cache.set(key, value, ttl=30)
        print(f"     Cached: {key} -> {value['name']} (score: {value['score']})")
    
    # Test cache retrieval and statistics
    print("\n   📊 Cache statistics:")
    cache_stats = memory_cache.get_stats()
    print(f"     Total sets: {cache_stats.get('total_sets', 0)}")
    print(f"     Current size: {cache_stats.get('current_size', 0)}")
    print(f"     Max size: {cache_stats.get('max_size', 0)}")
    
    # Test cache hits and misses
    print("\n   🎯 Testing cache hits and misses:")
    test_keys = ["user_1", "user_3", "user_7", "user_5"]
    
    for key in test_keys:
        value = memory_cache.get(key)
        if value:
            print(f"     ✓ Hit: {key} -> {value['name']}")
        else:
            print(f"     ❌ Miss: {key}")
    
    # Cache cleanup demonstration
    print("\n   🧹 Cache cleanup:")
    initial_stats = memory_cache.get_stats()
    print(f"     Before cleanup: {initial_stats.get('current_size', 0)} items")
    
    # Simulate expired entries (set TTL to 0)
    memory_cache.set("temp_1", {"temp": True}, ttl=0)
    memory_cache.set("temp_2", {"temp": True}, ttl=0)
    
    time.sleep(0.1)  # Let entries expire
    
    memory_cache.cleanup_expired()
    final_stats = memory_cache.get_stats()
    print(f"     After cleanup: {final_stats.get('current_size', 0)} items")
    
    # Pattern-based operations
    print("\n   🔍 Pattern-based cache operations:")
    # Add some entries with patterns
    pattern_data = [
        ("model_rf_v1", {"type": "random_forest", "version": 1}),
        ("model_rf_v2", {"type": "random_forest", "version": 2}),
        ("model_svm_v1", {"type": "svm", "version": 1}),
        ("data_train", {"type": "training_data"}),
        ("data_test", {"type": "test_data"})
    ]
    
    for key, value in pattern_data:
        memory_cache.set(key, value, ttl=60)
    
    print(f"     Added {len(pattern_data)} patterned entries")
    
    # Find entries by pattern (if supported)
    all_keys = list(memory_cache.keys()) if hasattr(memory_cache, 'keys') else []
    model_keys = [k for k in all_keys if k.startswith('model_')]
    data_keys = [k for k in all_keys if k.startswith('data_')]
    
    print(f"     Model entries: {len(model_keys)}")
    print(f"     Data entries: {len(data_keys)}")


def cache_optimization_patterns():
    """Demonstrate cache optimization patterns for ML workflows."""
    print("\n⚡ Cache Optimization Patterns")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Cache optimization patterns:
from refunc.utils import cache_result, FileHandler

# Layered caching strategy
@cache_result(ttl_seconds=300)  # Fast memory cache
def get_processed_features(data_id):
    return compute_features(data_id)

@cache_result(ttl_seconds=3600, use_disk=True)  # Persistent disk cache
def get_trained_model(config):
    return train_model(config)

# Cache warming
def warm_cache():
    '''Pre-populate cache with commonly used data.'''
    common_configs = get_common_model_configs()
    for config in common_configs:
        get_trained_model(config)  # Populate cache

# Cache hierarchy
class MLCacheManager:
    def __init__(self):
        self.fast_cache = MemoryCache(max_size=100)
        self.slow_cache = DiskCache(cache_dir="./persistent")
    
    def get_model(self, model_id):
        # Try fast cache first
        model = self.fast_cache.get(model_id)
        if model:
            return model
        
        # Try persistent cache
        model = self.slow_cache.get(model_id)
        if model:
            self.fast_cache.set(model_id, model)  # Promote to fast cache
            return model
        
        # Load and cache at both levels
        model = load_model(model_id)
        self.fast_cache.set(model_id, model)
        self.slow_cache.set(model_id, model, ttl=86400)
        return model
        """)
        return
    
    print("🏗️ ML workflow optimization patterns:")
    
    # Demonstrate cache warming
    @cache_result(ttl_seconds=20)
    def get_model_predictions(model_type: str, data_size: int) -> Dict[str, Any]:
        """Get model predictions with caching."""
        print(f"     🔄 Generating predictions: {model_type} on {data_size} samples...")
        time.sleep(0.1)  # Simulate prediction time
        
        # Simulate predictions
        accuracy = 0.8 + random.random() * 0.15
        predictions = [random.choice([0, 1]) for _ in range(data_size)]
        
        return {
            "model_type": model_type,
            "data_size": data_size,
            "accuracy": accuracy,
            "predictions_sample": predictions[:5],  # First 5 predictions
            "generated_at": time.time()
        }
    
    @cache_result(ttl_seconds=30)
    def get_feature_importance(model_type: str, feature_count: int) -> Dict[str, Any]:
        """Get feature importance with caching."""
        print(f"     📊 Computing feature importance: {model_type} with {feature_count} features...")
        time.sleep(0.08)  # Simulate computation time
        
        # Generate mock feature importance
        importance_scores = {
            f"feature_{i}": random.random() 
            for i in range(feature_count)
        }
        
        return {
            "model_type": model_type,
            "feature_count": feature_count,
            "importance_scores": importance_scores,
            "top_features": sorted(importance_scores.items(), 
                                 key=lambda x: x[1], reverse=True)[:3],
            "computed_at": time.time()
        }
    
    # Cache warming demonstration
    print("   🔥 Cache warming - pre-populating common requests:")
    common_requests = [
        ("random_forest", 1000),
        ("svm", 500),
        ("neural_network", 2000)
    ]
    
    warming_start = time.time()
    for model_type, data_size in common_requests:
        predictions = get_model_predictions(model_type, data_size)
        features = get_feature_importance(model_type, 20)
        print(f"     ✓ Warmed: {model_type} (accuracy: {predictions['accuracy']:.3f})")
    warming_time = time.time() - warming_start
    
    # Test cache hits after warming
    print(f"\n   ⚡ Testing cache hits after warming (took {warming_time:.2f}s):")
    hit_start = time.time()
    for model_type, data_size in common_requests:
        predictions = get_model_predictions(model_type, data_size)  # Should be cached
        print(f"     ✓ Retrieved: {model_type} (cached)")
    hit_time = time.time() - hit_start
    
    print(f"     Cache hit speedup: {warming_time/hit_time:.1f}x faster")
    
    # Demonstrate cache invalidation strategy
    print("\n   🔄 Cache invalidation patterns:")
    
    # Show cache before invalidation
    print("     Testing selective cache invalidation...")
    
    # Add some test entries
    test_predictions = get_model_predictions("test_model", 100)
    print(f"     Added test model predictions: accuracy = {test_predictions['accuracy']:.3f}")
    
    # Simulate cache invalidation (in real usage, you'd call cache.invalidate())
    print("     ✓ In production: Use cache.invalidate(pattern) to remove outdated entries")
    print("     ✓ In production: Use TTL expiration for automatic cleanup")
    print("     ✓ In production: Use cache versioning for model updates")


def integration_examples():
    """Demonstrate caching integration with other refunc components."""
    print("\n🔗 Integration Examples")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Integration with other refunc components:
from refunc.utils import cache_result, FileHandler
from refunc.decorators import time_it, memory_profile
from refunc.logging import MLLogger

# Combined caching with performance monitoring
logger = MLLogger("caching_demo")

@time_it(logger=logger)
@memory_profile(logger=logger)
@cache_result(ttl_seconds=3600, use_disk=True)
def comprehensive_data_processing(file_path):
    handler = FileHandler(cache_enabled=True)
    data = handler.load_auto(file_path)
    return expensive_processing(data)

# Caching with error handling
from refunc.exceptions import retry_on_failure

@retry_on_failure(max_attempts=3)
@cache_result(ttl_seconds=1800)
def robust_api_call(endpoint, params):
    response = make_api_request(endpoint, params)
    return process_response(response)
        """)
        return
    
    print("🔄 Caching integrates seamlessly with:")
    print("   • Performance monitoring decorators")
    print("   • Error handling and retry mechanisms") 
    print("   • File handling operations")
    print("   • Logging and experiment tracking")
    
    print("\n💡 Integration benefits:")
    print("   • Automatic performance improvement")
    print("   • Reduced redundant computations")
    print("   • Persistent results across runs")
    print("   • Error recovery with cached fallbacks")
    print("   • Comprehensive monitoring of cache effectiveness")
    
    print("\n🎯 Best practices:")
    print("   • Use memory cache for frequently accessed small data")
    print("   • Use disk cache for expensive computations")
    print("   • Set appropriate TTL based on data freshness needs")
    print("   • Monitor cache hit ratios for optimization")
    print("   • Implement cache warming for critical paths")
    print("   • Use cache invalidation for data consistency")


def main():
    """Run all caching examples."""
    print("🚀 Refunc Caching Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_caching_examples()
    disk_caching_examples()
    cache_management_examples()
    cache_optimization_patterns()
    integration_examples()
    
    print("\n✅ Caching examples completed!")
    print("\n📖 Next steps:")
    print("- Implement caching in your expensive computations")
    print("- Monitor cache hit rates for optimization opportunities")
    print("- Use disk caching for model training results")
    print("- Check out validation_examples.py for input/output validation")


if __name__ == "__main__":
    main()