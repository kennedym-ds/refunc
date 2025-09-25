#!/usr/bin/env python3
"""
Performance Monitoring Examples - Refunc Decorators

This example demonstrates the comprehensive performance monitoring decorators
including timing, memory profiling, system monitoring, and combined tracking
for ML workflows.

Key Features Demonstrated:
- Timing decorators and profiling
- Memory usage monitoring
- System resource tracking
- GPU monitoring (when available)
- Combined performance metrics
- Integration with logging
"""

import os
import sys
import time
import random
from typing import List, Optional, Any

# Handle missing dependencies gracefully
try:
    from refunc.decorators import (
        time_it, memory_profile, system_monitor, performance_monitor,
        validate_inputs, validate_outputs,
        TimingResult, MemoryResult, MonitoringResult,
        quick_monitor, full_monitor, debug_monitor
    )
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def timing_examples():
    """Demonstrate timing decorators and profiling."""
    print("⏱️ Timing Decorators")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Timing decorator examples:
from refunc.decorators import time_it, timer, TimingProfiler

# Basic timing decorator
@time_it
def data_processing():
    # Process data
    return processed_data

# Advanced timing with custom name
@time_it(name="model_training", unit="ms")
def train_model():
    # Training code
    return model

# Timing with statistics
@time_it(collect_stats=True, warmup_calls=3)
def inference():
    return model.predict(data)

# Manual timing profiler
profiler = TimingProfiler("ml_pipeline")
with profiler.time("data_loading"):
    data = load_data()

with profiler.time("preprocessing"):
    data = preprocess(data)

print(profiler.get_summary())
        """)
        return
    
    print("⏰ Basic timing examples:")
    
    # Basic timing decorator
    @time_it
    def simulate_data_loading():
        """Simulate data loading operation."""
        time.sleep(0.1 + random.random() * 0.05)  # 100-150ms
        return list(range(1000))
    
    @time_it(name="feature_engineering", unit="ms")
    def simulate_feature_engineering(data):
        """Simulate feature engineering."""
        time.sleep(0.05 + random.random() * 0.02)  # 50-70ms
        return [x * 2 for x in data[:100]]
    
    @time_it(collect_stats=True)
    def simulate_model_inference(features):
        """Simulate model inference with statistics collection."""
        time.sleep(0.02 + random.random() * 0.01)  # 20-30ms
        return sum(features) / len(features)
    
    # Test timing decorators
    print("   📊 Loading data...")
    data = simulate_data_loading()
    print(f"   ✓ Loaded {len(data)} samples")
    
    print("   🔧 Engineering features...")
    features = simulate_feature_engineering(data)
    print(f"   ✓ Created {len(features)} features")
    
    print("   🤖 Running inference (multiple times for stats)...")
    for i in range(3):
        result = simulate_model_inference(features)
        print(f"   Run {i+1}: prediction = {result:.2f}")
    
    # Manual profiler example
    if hasattr(sys.modules.get('refunc.decorators', {}), 'TimingProfiler'):
        from refunc.decorators import TimingProfiler
        
        print("\n📈 Manual profiler example:")
        profiler = TimingProfiler("ml_pipeline_demo")
        
        with profiler.time("data_validation"):
            time.sleep(0.03)  # Simulate validation
        
        with profiler.time("model_prediction"):
            time.sleep(0.02)  # Simulate prediction
        
        with profiler.time("result_formatting"):
            time.sleep(0.01)  # Simulate formatting
        
        print("   Pipeline timing summary:")
        summary = profiler.get_summary()
        for operation, timing in summary.items():
            print(f"     {operation}: {timing}")


def memory_monitoring_examples():
    """Demonstrate memory profiling and monitoring."""
    print("\n💾 Memory Monitoring")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Memory monitoring examples:
from refunc.decorators import memory_profile, MemoryMonitor

# Basic memory profiling
@memory_profile
def data_processing():
    large_data = [0] * 1000000  # Allocate memory
    return process(large_data)

# Advanced memory monitoring
@memory_profile(track_allocations=True, detailed=True)
def memory_intensive_operation():
    return complex_computation()

# Memory leak detection
monitor = MemoryMonitor()
with monitor.track("potential_leak"):
    result = run_operation()

if monitor.detect_leak():
    print("Memory leak detected!")
        """)
        return
    
    print("🧠 Memory profiling examples:")
    
    @memory_profile
    def simulate_data_processing():
        """Simulate memory-intensive data processing."""
        # Allocate some memory
        large_list = list(range(100000))  # ~400KB for integers
        large_dict = {i: str(i) * 10 for i in range(10000)}  # String data
        
        # Process data
        result = sum(large_list)
        
        # Clean up (Python GC will handle this)
        del large_list, large_dict
        
        return result
    
    @memory_profile(detailed=True)
    def simulate_model_training():
        """Simulate model training with detailed memory tracking."""
        # Simulate training data
        training_data = [[random.random() for _ in range(50)] for _ in range(1000)]
        
        # Simulate model parameters
        model_weights = [[random.random() for _ in range(50)] for _ in range(10)]
        
        # Simulate training iterations
        for epoch in range(3):
            # Simulate forward pass
            predictions = []
            for sample in training_data:
                pred = sum(w[i] * sample[i] for w in model_weights for i in range(min(50, len(sample))))
                predictions.append(pred)
            
            # Simulate gradient computation (more memory allocation)
            gradients = [[random.random() for _ in range(50)] for _ in range(10)]
            
            # Update weights
            for i, grad_layer in enumerate(gradients):
                for j, grad in enumerate(grad_layer):
                    model_weights[i][j] -= 0.01 * grad
        
        return {"accuracy": 0.85, "loss": 0.23}
    
    # Test memory monitoring
    print("   📊 Processing data with memory tracking...")
    result = simulate_data_processing()
    print(f"   ✓ Processing result: {result}")
    
    print("   🤖 Training model with detailed memory tracking...")
    training_result = simulate_model_training()
    print(f"   ✓ Training completed: {training_result}")
    
    # Memory snapshot example
    if hasattr(sys.modules.get('refunc.decorators', {}), 'MemoryMonitor'):
        print("\n📸 Memory snapshot example:")
        from refunc.decorators import MemoryMonitor
        
        monitor = MemoryMonitor()
        snapshot1 = monitor.take_snapshot("before_allocation")
        
        # Allocate memory
        test_data = [list(range(1000)) for _ in range(100)]
        
        snapshot2 = monitor.take_snapshot("after_allocation")
        
        print(f"   Memory before: {snapshot1.get('memory_mb', 'N/A')} MB")
        print(f"   Memory after: {snapshot2.get('memory_mb', 'N/A')} MB")
        
        # Clean up
        del test_data


def system_monitoring_examples():
    """Demonstrate system resource monitoring."""
    print("\n🖥️ System Monitoring")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# System monitoring examples:
from refunc.decorators import system_monitor, get_system_info

# System resource monitoring
@system_monitor
def cpu_intensive_task():
    return heavy_computation()

# Detailed system monitoring
@system_monitor(track_gpu=True, track_network=True)
def ml_training():
    return train_neural_network()

# Manual system info
system_info = get_system_info()
print(f"CPU cores: {system_info['cpu_cores']}")
print(f"Memory: {system_info['memory_gb']} GB")
        """)
        return
    
    print("🔧 System resource monitoring:")
    
    @system_monitor
    def simulate_cpu_task():
        """Simulate CPU-intensive computation."""
        # CPU-intensive calculation
        result = 0
        for i in range(500000):
            result += i ** 0.5
        return result
    
    @system_monitor(detailed=True)
    def simulate_mixed_workload():
        """Simulate mixed CPU/memory workload."""
        # CPU work
        cpu_result = sum(range(100000))
        
        # Memory work
        memory_data = [list(range(100)) for _ in range(1000)]
        memory_result = sum(len(sublist) for sublist in memory_data)
        
        # I/O simulation (creating temporary data)
        temp_data = "\n".join(str(i) for i in range(1000))
        
        return {
            "cpu_result": cpu_result,
            "memory_result": memory_result,
            "io_size": len(temp_data)
        }
    
    # Test system monitoring
    print("   💻 Running CPU-intensive task...")
    cpu_result = simulate_cpu_task()
    print(f"   ✓ CPU task result: {cpu_result:.0f}")
    
    print("   ⚙️ Running mixed workload with detailed monitoring...")
    mixed_result = simulate_mixed_workload()
    print(f"   ✓ Mixed workload results: {mixed_result}")
    
    # System info example
    if hasattr(sys.modules.get('refunc.decorators', {}), 'get_system_info'):
        print("\n🖥️ System information:")
        from refunc.decorators import get_system_info
        
        system_info = get_system_info()
        print(f"   CPU cores: {system_info.get('cpu_cores', 'N/A')}")
        print(f"   Memory: {system_info.get('memory_gb', 'N/A')} GB")
        print(f"   Platform: {system_info.get('platform', 'N/A')}")


def combined_monitoring_examples():
    """Demonstrate combined performance monitoring."""
    print("\n🔄 Combined Performance Monitoring")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Combined monitoring examples:
from refunc.decorators import performance_monitor, full_monitor

# Complete performance monitoring
@performance_monitor
def ml_pipeline():
    data = load_data()
    model = train_model(data)
    return evaluate_model(model)

# Full monitoring with all metrics
@full_monitor
def comprehensive_task():
    return complex_ml_workflow()

# Quick monitoring for development
@quick_monitor
def debug_function():
    return experimental_algorithm()
        """)
        return
    
    print("📊 Combined performance monitoring:")
    
    @performance_monitor
    def simulate_ml_pipeline():
        """Simulate complete ML pipeline with combined monitoring."""
        print("     📥 Loading data...")
        time.sleep(0.05)  # Data loading
        data = list(range(1000))
        
        print("     🔧 Preprocessing...")
        time.sleep(0.03)  # Preprocessing
        processed_data = [x * 2 for x in data]
        
        print("     🤖 Training model...")
        time.sleep(0.08)  # Model training
        model_weights = [random.random() for _ in range(10)]
        
        print("     📊 Evaluating...")
        time.sleep(0.02)  # Evaluation
        accuracy = 0.85 + random.random() * 0.1
        
        return {
            "data_size": len(data),
            "model_params": len(model_weights),
            "accuracy": accuracy
        }
    
    @performance_monitor(detailed=True)
    def simulate_data_science_workflow():
        """Simulate data science workflow with detailed monitoring."""
        # Data analysis phase
        analysis_data = []
        for i in range(5):
            batch = [random.random() for _ in range(1000)]
            analysis_data.extend(batch)
            time.sleep(0.01)  # Processing time
        
        # Statistical computation
        mean_val = sum(analysis_data) / len(analysis_data)
        variance = sum((x - mean_val) ** 2 for x in analysis_data) / len(analysis_data)
        
        # Result compilation
        results = {
            "samples": len(analysis_data),
            "mean": mean_val,
            "variance": variance,
            "std_dev": variance ** 0.5
        }
        
        return results
    
    # Test combined monitoring
    print("   🚀 Running ML pipeline with combined monitoring...")
    pipeline_result = simulate_ml_pipeline()
    print(f"   ✓ Pipeline completed: accuracy = {pipeline_result['accuracy']:.3f}")
    
    print("   📈 Running data science workflow...")
    ds_result = simulate_data_science_workflow()
    print(f"   ✓ Analysis completed: {ds_result['samples']} samples, "
          f"mean = {ds_result['mean']:.3f}")


def validation_integration():
    """Demonstrate integration with validation decorators."""
    print("\n✅ Validation Integration")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Validation with performance monitoring:
from refunc.decorators import (
    performance_monitor, validate_inputs, validate_outputs
)

@performance_monitor
@validate_inputs(types={'data': list, 'threshold': float})
@validate_outputs(types=dict)
def process_with_validation(data, threshold=0.5):
    '''Process data with input/output validation and monitoring.'''
    filtered_data = [x for x in data if x > threshold]
    return {
        'original_count': len(data),
        'filtered_count': len(filtered_data),
        'filter_ratio': len(filtered_data) / len(data)
    }

# Type validation with timing
@time_it
@validate_inputs(
    types={'X': (list, tuple), 'y': (list, tuple)},
    shapes={'X': (None, None), 'y': (None,)}
)
def train_with_validation(X, y):
    return fit_model(X, y)
        """)
        return
    
    print("🔍 Performance monitoring with validation:")
    
    @performance_monitor
    @validate_inputs(types={'data': list, 'multiplier': (int, float)})
    @validate_outputs(types=dict)
    def process_with_validation(data: List[float], multiplier: float = 2.0):
        """Process data with validation and performance monitoring."""
        time.sleep(0.02)  # Simulate processing
        
        processed = [x * multiplier for x in data]
        result = {
            'original_count': len(data),
            'processed_count': len(processed),
            'sum_original': sum(data),
            'sum_processed': sum(processed),
            'multiplier_used': multiplier
        }
        
        return result
    
    # Test valid inputs
    print("   ✅ Testing with valid inputs...")
    test_data = [1.0, 2.5, 3.7, 4.2, 5.1]
    result = process_with_validation(test_data, 1.5)
    print(f"   ✓ Valid processing: {result['processed_count']} items processed")
    
    # Test validation errors
    print("   ❌ Testing validation errors...")
    try:
        # This should fail validation (wrong type)
        bad_result = process_with_validation("invalid_data", 2.0)
        print("   Unexpected success!")
    except Exception as e:
        print(f"   ✓ Caught validation error: {type(e).__name__}")
    
    try:
        # This should fail validation (wrong multiplier type)
        bad_result = process_with_validation(test_data, "invalid_multiplier")
        print("   Unexpected success!")
    except Exception as e:
        print(f"   ✓ Caught validation error: {type(e).__name__}")


def main():
    """Run all performance monitoring examples."""
    print("🚀 Refunc Performance Monitoring Examples")
    print("=" * 65)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    timing_examples()
    memory_monitoring_examples()
    system_monitoring_examples()
    combined_monitoring_examples()
    validation_integration()
    
    print("\n✅ Performance monitoring examples completed!")
    print("\n📖 Next steps:")
    print("- Integrate performance decorators into your ML workflows")
    print("- Monitor resource usage in production systems")
    print("- Combine with logging for comprehensive monitoring")
    print("- Check out caching_examples.py for optimization patterns")


if __name__ == "__main__":
    main()