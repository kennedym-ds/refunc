#!/usr/bin/env python3
"""
Benchmark execution script for refunc.

This script runs performance benchmarks and handles regression detection,
baseline management, and performance reporting.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json
from typing import List, Dict, Any
import os


def run_benchmarks(
    benchmark_dir: Path = None,
    output_file: Path = None,
    compare_baseline: bool = True,
    save_baseline: bool = False,
    regression_threshold: float = 20.0
) -> Dict[str, Any]:
    """
    Run performance benchmarks.
    
    Args:
        benchmark_dir: Directory containing benchmark files
        output_file: Output file for benchmark results
        compare_baseline: Whether to compare against baseline
        save_baseline: Whether to save results as new baseline
        regression_threshold: Regression threshold percentage
        
    Returns:
        Dictionary containing benchmark results and analysis
    """
    project_root = Path(__file__).parent.parent
    benchmark_dir = benchmark_dir or project_root / "benchmarks"
    output_file = output_file or project_root / "benchmarks" / ".benchmarks" / "results.json"
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Running performance benchmarks...")
    print(f"📂 Benchmark directory: {benchmark_dir}")
    print(f"📄 Output file: {output_file}")
    
    # Build pytest command
    cmd = [
        "python", "-m", "pytest",
        str(benchmark_dir),
        "--benchmark-only",
        "--benchmark-json", str(output_file),
        "--benchmark-columns", "min,max,mean,stddev,outliers,ops,rounds",
        "--benchmark-group-by", "group",
        "--benchmark-sort", "mean",
        "-v"
    ]
    
    # Add regression comparison if requested
    if compare_baseline:
        baseline_file = project_root / "benchmarks" / ".benchmarks" / "baseline.json"
        if baseline_file.exists():
            cmd.extend(["--benchmark-compare", str(baseline_file)])
            cmd.extend(["--benchmark-compare-fail", f"mean:{regression_threshold}%"])
    
    try:
        # Run benchmarks
        print("⏱️  Executing benchmarks...")
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        # Parse results
        results = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "benchmarks_completed": True
        }
        
        # Load benchmark data if available
        if output_file.exists():
            with open(output_file, 'r') as f:
                benchmark_data = json.load(f)
            results["benchmark_data"] = benchmark_data
            results["total_benchmarks"] = len(benchmark_data.get("benchmarks", []))
        
        # Handle regression detection
        if compare_baseline and result.returncode != 0:
            print("⚠️  Performance regressions detected!")
            results["regressions_detected"] = True
        elif result.returncode == 0:
            print("✅ All benchmarks completed successfully")
            results["regressions_detected"] = False
        
        # Save as baseline if requested
        if save_baseline and output_file.exists():
            baseline_file = project_root / "benchmarks" / ".benchmarks" / "baseline.json"
            import shutil
            shutil.copy2(output_file, baseline_file)
            print(f"💾 Baseline saved to: {baseline_file}")
            results["baseline_saved"] = True
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark execution failed: {e}")
        return {
            "exit_code": 1,
            "error": str(e),
            "benchmarks_completed": False
        }


def analyze_benchmark_results(results_file: Path) -> Dict[str, Any]:
    """Analyze benchmark results and generate insights."""
    if not results_file.exists():
        return {"error": "Results file not found"}
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    benchmarks = data.get("benchmarks", [])
    
    if not benchmarks:
        return {"error": "No benchmark data found"}
    
    # Calculate statistics
    mean_times = [b["stats"]["mean"] for b in benchmarks]
    total_time = sum(mean_times)
    
    analysis = {
        "total_benchmarks": len(benchmarks),
        "total_execution_time": total_time,
        "average_benchmark_time": total_time / len(benchmarks) if benchmarks else 0,
        "fastest_benchmark": min(benchmarks, key=lambda x: x["stats"]["mean"]),
        "slowest_benchmark": max(benchmarks, key=lambda x: x["stats"]["mean"]),
        "high_variance_benchmarks": [
            b for b in benchmarks 
            if b["stats"]["stddev"] / b["stats"]["mean"] > 0.1  # > 10% coefficient of variation
        ]
    }
    
    return analysis


def generate_performance_report(
    results_file: Path,
    output_format: str = "text"
) -> str:
    """Generate a formatted performance report."""
    analysis = analyze_benchmark_results(results_file)
    
    if "error" in analysis:
        return f"Error: {analysis['error']}"
    
    if output_format == "json":
        return json.dumps(analysis, indent=2)
    
    # Text format report
    report = []
    report.append("="*60)
    report.append("📊 PERFORMANCE BENCHMARK REPORT")
    report.append("="*60)
    report.append(f"Total Benchmarks: {analysis['total_benchmarks']}")
    report.append(f"Total Execution Time: {analysis['total_execution_time']:.4f}s")
    report.append(f"Average Benchmark Time: {analysis['average_benchmark_time']:.4f}s")
    report.append("")
    
    report.append("🏆 FASTEST BENCHMARK:")
    fastest = analysis['fastest_benchmark']
    report.append(f"  {fastest['name']}: {fastest['stats']['mean']:.6f}s")
    report.append("")
    
    report.append("🐌 SLOWEST BENCHMARK:")
    slowest = analysis['slowest_benchmark']
    report.append(f"  {slowest['name']}: {slowest['stats']['mean']:.6f}s")
    report.append("")
    
    if analysis['high_variance_benchmarks']:
        report.append("⚠️  HIGH VARIANCE BENCHMARKS:")
        for bench in analysis['high_variance_benchmarks']:
            cv = bench['stats']['stddev'] / bench['stats']['mean']
            report.append(f"  {bench['name']}: CV={cv:.2%}")
        report.append("")
    
    report.append("="*60)
    
    return "\n".join(report)


def main():
    """Main entry point for benchmark execution."""
    parser = argparse.ArgumentParser(description="Run performance benchmarks for refunc")
    parser.add_argument("--benchmark-dir", type=Path, help="Directory containing benchmarks")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--no-compare", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as new baseline")
    parser.add_argument("--threshold", type=float, default=20.0, help="Regression threshold percentage")
    parser.add_argument("--report", choices=["text", "json"], default="text", help="Report format")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    output_file = args.output or project_root / "benchmarks" / ".benchmarks" / "results.json"
    
    if args.analyze_only:
        # Only analyze existing results
        if not output_file.exists():
            print(f"❌ Results file not found: {output_file}")
            sys.exit(1)
        
        report = generate_performance_report(output_file, args.report)
        print(report)
        return
    
    # Run benchmarks
    results = run_benchmarks(
        benchmark_dir=args.benchmark_dir,
        output_file=output_file,
        compare_baseline=not args.no_compare,
        save_baseline=args.save_baseline,
        regression_threshold=args.threshold
    )
    
    if not results["benchmarks_completed"]:
        print("❌ Benchmarks failed to complete")
        if "error" in results:
            print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Generate report
    if output_file.exists():
        report = generate_performance_report(output_file, args.report)
        print("\n" + report)
    
    # Exit with error if regressions detected
    if results.get("regressions_detected", False):
        print("\n❌ Performance regressions detected - failing build")
        sys.exit(1)
    
    print("\n✅ All benchmarks completed successfully")


if __name__ == "__main__":
    main()