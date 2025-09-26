#!/usr/bin/env python3
"""
File Handling Examples - Refunc Utilities

This example demonstrates the FileHandler utilities for efficient file operations,
caching, and format detection in ML workflows.

Key Features Demonstrated:
- FileHandler basic operations
- Automatic format detection
- Caching mechanisms
- File format validation
- Batch file operations
"""

import os
import sys
from pathlib import Path
import tempfile
import json
import time

# Handle missing dependencies gracefully
try:
    from refunc.utils import FileHandler, FileFormat, cache_result
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False


def create_sample_data():
    """Create sample data files for demonstration."""
    temp_dir = Path(tempfile.mkdtemp(prefix="refunc_file_example_"))
    print(f"📁 Created temporary directory: {temp_dir}")
    
    # Create sample data
    sample_data = {
        "experiment": "demo",
        "metrics": {"accuracy": 0.85, "loss": 0.15},
        "parameters": {"learning_rate": 0.001, "batch_size": 32},
        "data_info": {"samples": 1000, "features": 20}
    }
    
    # Create files in different formats
    (temp_dir / "config.json").write_text(json.dumps(sample_data, indent=2))
    (temp_dir / "data.txt").write_text("Sample text data for processing\nLine 2\nLine 3")
    (temp_dir / "README.md").write_text("# Sample Project\nThis is a test file.")
    
    # Create a CSV-like file
    csv_content = "name,value,category\nitem1,100,A\nitem2,200,B\nitem3,150,A"
    (temp_dir / "data.csv").write_text(csv_content)
    
    return temp_dir


def basic_file_operations():
    """Demonstrate basic FileHandler operations."""
    print("🔧 Basic File Operations")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic FileHandler usage:
from refunc.utils import FileHandler

# Initialize FileHandler
handler = FileHandler()

# Load files automatically by format
data = handler.load_auto('config.json')
text = handler.load_auto('document.txt')

# Save files in different formats
handler.save_auto(data, 'output.json')
handler.save_auto(data, 'output.yaml')  # Auto-converts format
        """)
        return
    
    temp_dir = create_sample_data()
    
    try:
        # Initialize FileHandler
        handler = FileHandler()
        
        # Load files automatically
        config_file = temp_dir / "config.json"
        print(f"📖 Loading: {config_file.name}")
        config_data = handler.load_auto(str(config_file))
        print(f"✓ Loaded config: {config_data['experiment']}")
        
        # Load text file
        text_file = temp_dir / "data.txt"
        print(f"📖 Loading: {text_file.name}")
        text_data = handler.load_auto(str(text_file))
        print(f"✓ Loaded text ({len(text_data)} chars): {text_data[:30]}...")
        
        # Save in different format
        output_file = temp_dir / "config.yaml"
        print(f"💾 Saving as YAML: {output_file.name}")
        handler.save_auto(config_data, str(output_file))
        print("✓ Saved successfully")
        
        # Verify the saved file
        loaded_yaml = handler.load_auto(str(output_file))
        print(f"✓ Verified YAML load: {loaded_yaml['experiment']}")
        
    except Exception as e:
        print(f"❌ Error in basic operations: {e}")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🗑️  Cleaned up temporary directory")


def format_detection_examples():
    """Demonstrate format detection capabilities."""
    print("\n🔍 Format Detection")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Format detection examples:
from refunc.utils import FileFormat, validate_file_format

# Detect format by extension
format_info = FileFormat.detect_from_path('data.csv')
print(f"Detected format: {format_info}")

# Validate file format
is_valid = validate_file_format('data.csv', FileFormat.CSV)
print(f"Valid CSV: {is_valid}")

# Get format information
info = get_format_info(FileFormat.JSON)
print(f"JSON info: {info}")
        """)
        return
    
    temp_dir = create_sample_data()
    
    try:
        # Test format detection
        test_files = list(temp_dir.glob("*"))
        
        for file_path in test_files:
            print(f"📄 File: {file_path.name}")
            
            # Detect format
            detected_format = FileFormat.detect_from_path(str(file_path))
            print(f"   Format: {detected_format}")
            
            # Get format info
            if detected_format != FileFormat.UNKNOWN:
                from refunc.utils import get_format_info
                info = get_format_info(detected_format)
                print(f"   Extensions: {info.get('extensions', [])}")
                print(f"   MIME type: {info.get('mime_type', 'N/A')}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error in format detection: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def caching_examples():
    """Demonstrate caching capabilities."""
    print("\n💾 Caching Examples")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Caching examples:
from refunc.utils import cache_result, FileHandler

# Cache decorator for expensive operations
@cache_result(ttl_seconds=300)
def expensive_computation(data):
    # Simulate expensive operation
    time.sleep(1)
    return sum(data) / len(data)

# FileHandler with caching enabled
handler = FileHandler(cache_enabled=True, use_disk_cache=True)

# Cached file operations
data = handler.load_auto('large_file.csv')  # Cached after first load
        """)
        return
    
    # Demonstrate caching decorator
    @cache_result(ttl_seconds=5)  # Short TTL for demo
    def expensive_calculation(n):
        """Simulate an expensive calculation."""
        print(f"  🔄 Computing expensive operation for n={n}...")
        time.sleep(0.5)  # Simulate computation time
        return n ** 2 + n * 10
    
    # Test caching
    print("First call (should compute):")
    start_time = time.time()
    result1 = expensive_calculation(5)
    duration1 = time.time() - start_time
    print(f"  ✓ Result: {result1} (took {duration1:.2f}s)")
    
    print("\nSecond call (should be cached):")
    start_time = time.time()
    result2 = expensive_calculation(5)
    duration2 = time.time() - start_time
    print(f"  ✓ Result: {result2} (took {duration2:.2f}s)")
    
    print(f"\nCaching speedup: {duration1/duration2:.1f}x faster")
    
    # Demonstrate FileHandler caching
    temp_dir = create_sample_data()
    
    try:
        print("\n📁 FileHandler Caching:")
        handler = FileHandler(cache_enabled=True)
        
        config_file = temp_dir / "config.json"
        
        print("First load (from disk):")
        start_time = time.time()
        data1 = handler.load_auto(str(config_file))
        duration1 = time.time() - start_time
        print(f"  ✓ Loaded in {duration1:.3f}s")
        
        print("Second load (from cache):")
        start_time = time.time()
        data2 = handler.load_auto(str(config_file))
        duration2 = time.time() - start_time
        print(f"  ✓ Loaded in {duration2:.3f}s")
        
        print(f"  📊 Same data: {data1 == data2}")
        
    except Exception as e:
        print(f"❌ Error in caching demo: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def batch_operations_example():
    """Demonstrate batch file operations."""
    print("\n📦 Batch Operations")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Batch operations example:
from refunc.utils import FileHandler

handler = FileHandler()

# Find files by format
csv_files = handler.find_files_by_format('./data', FileFormat.CSV)
json_files = handler.find_files_by_format('./config', FileFormat.JSON)

# Process multiple files
results = []
for file_path in csv_files:
    data = handler.load_auto(file_path)
    processed = process_data(data)
    results.append(processed)

# Save batch results
for i, result in enumerate(results):
    handler.save_auto(result, f'output_{i}.json')
        """)
        return
    
    temp_dir = create_sample_data()
    
    try:
        # Create additional files
        for i in range(3):
            sample_data = {"batch_id": i, "value": i * 10}
            (temp_dir / f"batch_{i}.json").write_text(json.dumps(sample_data))
        
        handler = FileHandler()
        
        # Find files by pattern
        json_files = list(temp_dir.glob("*.json"))
        print(f"📁 Found {len(json_files)} JSON files")
        
        # Process batch
        results = []
        for file_path in json_files:
            print(f"  📖 Processing: {file_path.name}")
            data = handler.load_auto(str(file_path))
            
            # Simple processing
            if isinstance(data, dict) and 'batch_id' in data:
                processed = {
                    "original": data,
                    "processed_value": data.get('value', 0) * 2,
                    "timestamp": time.time()
                }
                results.append(processed)
        
        # Save batch results
        output_file = temp_dir / "batch_results.json"
        handler.save_auto(results, str(output_file))
        print(f"💾 Saved {len(results)} processed results")
        
        # Verify batch results
        loaded_results = handler.load_auto(str(output_file))
        print(f"✓ Verified: {len(loaded_results)} results loaded")
        
    except Exception as e:
        print(f"❌ Error in batch operations: {e}")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all file handling examples."""
    print("🚀 Refunc File Handling Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    print()
    
    # Run examples
    basic_file_operations()
    format_detection_examples()
    caching_examples()
    batch_operations_example()
    
    print("\n✅ File handling examples completed!")
    print("\n📖 Next steps:")
    print("- Try modifying the examples with your own data")
    print("- Explore the FileHandler API documentation")
    print("- Check out logging_setup.py for ML experiment tracking")


if __name__ == "__main__":
    main()