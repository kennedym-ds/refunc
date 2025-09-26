#!/usr/bin/env python3
"""
Data Validation Examples - Refunc Data Science

This example demonstrates comprehensive data validation, quality assessment,
and schema validation for robust data science workflows.

Key Features Demonstrated:
- Data quality assessment and profiling
- Schema validation and constraint checking
- Missing value and outlier detection
- Data consistency analysis
- Automated data validation reports
- Integration with pandas DataFrames
"""

import os
import sys
import time
import random
import json
from typing import List, Dict, Any, Optional, Tuple

# Handle missing dependencies gracefully
try:
    from refunc.data_science import (
        DataValidator, DataSchema, ValidationReport,
        validate_dataframe, quick_validate, create_schema_from_dataframe,
        ValidationIssue, DataQualityLevel, ValidationSeverity
    )
    # Try to import pandas for examples
    try:
        import pandas as pd
        PANDAS_AVAILABLE = True
    except ImportError:
        PANDAS_AVAILABLE = False
        print("📊 Pandas not available - will use mock data structures")
    
    REFUNC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Refunc not fully installed: {e}")
    print("This example requires: pip install refunc")
    print("Showing example code structure...\n")
    REFUNC_AVAILABLE = False
    PANDAS_AVAILABLE = False


def create_sample_data():
    """Create sample data for demonstration."""
    if PANDAS_AVAILABLE:
        # Create realistic ML dataset
        n_samples = 1000
        data = {
            'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
            'age': [random.randint(18, 80) for _ in range(n_samples)],
            'income': [random.randint(20000, 150000) for _ in range(n_samples)],
            'credit_score': [random.randint(300, 850) for _ in range(n_samples)],
            'account_balance': [random.uniform(-1000, 50000) for _ in range(n_samples)],
            'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(n_samples)],
            'is_premium': [random.choice([True, False]) for _ in range(n_samples)]
        }
        
        # Introduce some data quality issues
        # Missing values
        for i in random.sample(range(n_samples), 50):  # 5% missing income
            data['income'][i] = None
        
        for i in random.sample(range(n_samples), 30):  # 3% missing credit_score
            data['credit_score'][i] = None
        
        # Outliers
        for i in random.sample(range(n_samples), 10):  # Some extreme ages
            data['age'][i] = random.choice([150, 200, -5])
        
        for i in random.sample(range(n_samples), 5):  # Some extreme incomes
            data['income'][i] = random.choice([1000000, -50000]) if data['income'][i] is not None else None
        
        # Invalid categorical values
        for i in random.sample(range(n_samples), 20):  # Some invalid regions
            data['region'][i] = random.choice(['Unknown', 'Invalid', None])
        
        return pd.DataFrame(data)
    else:
        # Mock data structure for demonstration
        return {
            'data': [
                {'customer_id': 'CUST_001', 'age': 25, 'income': 50000, 'region': 'North'},
                {'customer_id': 'CUST_002', 'age': None, 'income': 75000, 'region': 'South'},
                {'customer_id': 'CUST_003', 'age': 35, 'income': None, 'region': 'Invalid'},
            ],
            'columns': ['customer_id', 'age', 'income', 'region'],
            'shape': (3, 4)
        }


def basic_validation_examples():
    """Demonstrate basic data validation functionality."""
    print("✅ Basic Data Validation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Basic data validation examples:
from refunc.data_science import validate_dataframe, quick_validate

# Quick data validation
validation_result = quick_validate(df)
print(f"Data quality: {validation_result.quality_level}")
print(f"Issues found: {len(validation_result.issues)}")

# Detailed validation with custom rules
validator = DataValidator()
validator.add_rule('age', min_value=0, max_value=120)
validator.add_rule('income', min_value=0, required=True)
validator.add_rule('region', allowed_values=['North', 'South', 'East', 'West'])

report = validator.validate(df)
print(report.summary())

# Automated schema creation
schema = create_schema_from_dataframe(df)
print(f"Generated schema: {len(schema.columns)} columns")
        """)
        return
    
    print("🔍 Testing basic data validation:")
    
    # Create sample data
    sample_data = create_sample_data()
    
    if PANDAS_AVAILABLE:
        print(f"   📊 Created sample dataset: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        # Quick validation
        print("\n   ⚡ Quick validation:")
        quick_result = quick_validate(sample_data)
        print(f"     Quality level: {quick_result.quality_level}")
        print(f"     Issues found: {len(quick_result.issues)}")
        
        # Show first few issues
        for i, issue in enumerate(quick_result.issues[:3]):
            print(f"     Issue {i+1}: {issue.severity} - {issue.message}")
        
        # Detailed validation
        print("\n   🔬 Detailed validation:")
        detailed_result = validate_dataframe(sample_data)
        print(f"     Total checks: {detailed_result.total_checks}")
        print(f"     Passed: {detailed_result.passed_checks}")
        print(f"     Failed: {detailed_result.failed_checks}")
        print(f"     Success rate: {detailed_result.success_rate:.1%}")
        
        # Show validation summary by column
        print("\n   📋 Validation summary by column:")
        column_summary = detailed_result.get_column_summary()
        for column, summary in column_summary.items():
            issue_count = summary.get('issue_count', 0)
            quality = summary.get('quality_level', 'Unknown')
            print(f"     {column}: {quality} quality ({issue_count} issues)")
    
    else:
        print("   🎭 Mock validation (pandas not available):")
        print(f"   Sample data: {sample_data['shape']} shape")
        print("   Simulated validation results:")
        print("     Quality level: GOOD")
        print("     Issues found: 3")
        print("     Issue 1: WARNING - Missing values in 'age' column")
        print("     Issue 2: ERROR - Invalid values in 'region' column")
        print("     Issue 3: WARNING - Potential outliers in 'income' column")


def schema_validation_examples():
    """Demonstrate schema validation and constraints."""
    print("\n📋 Schema Validation")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Schema validation examples:
from refunc.data_science import DataSchema, DataValidator

# Define data schema
schema = DataSchema({
    'customer_id': {
        'type': 'string',
        'required': True,
        'pattern': r'CUST_\\d{4}',
        'unique': True
    },
    'age': {
        'type': 'integer',
        'min': 18,
        'max': 100,
        'required': True
    },
    'income': {
        'type': 'float',
        'min': 0,
        'max': 1000000,
        'required': False
    },
    'region': {
        'type': 'string',
        'allowed_values': ['North', 'South', 'East', 'West'],
        'required': True
    }
})

# Validate against schema
validator = DataValidator(schema=schema)
report = validator.validate(df)

# Schema inference
inferred_schema = create_schema_from_dataframe(df)
        """)
        return
    
    print("📝 Testing schema validation:")
    
    # Define expected schema
    expected_schema = {
        'customer_id': {
            'type': 'string',
            'required': True,
            'pattern': r'CUST_\\d{4}',
            'description': 'Unique customer identifier'
        },
        'age': {
            'type': 'integer',
            'min': 18,
            'max': 100,
            'required': True,
            'description': 'Customer age in years'
        },
        'income': {
            'type': 'float',
            'min': 0,
            'max': 500000,
            'required': False,
            'description': 'Annual income in USD'
        },
        'credit_score': {
            'type': 'integer',
            'min': 300,
            'max': 850,
            'required': False,
            'description': 'Credit score'
        },
        'region': {
            'type': 'string',
            'allowed_values': ['North', 'South', 'East', 'West'],
            'required': True,
            'description': 'Geographic region'
        },
        'is_premium': {
            'type': 'boolean',
            'required': True,
            'description': 'Premium account status'
        }
    }
    
    sample_data = create_sample_data()
    
    if PANDAS_AVAILABLE:
        # Create validator with schema
        validator = DataValidator()
        
        # Add validation rules based on schema
        for column, rules in expected_schema.items():
            if column in sample_data.columns:
                print(f"   📏 Adding rules for {column}:")
                
                rule_count = 0
                if rules.get('required'):
                    validator.add_missing_value_check(column)
                    rule_count += 1
                
                if 'min' in rules and 'max' in rules:
                    validator.add_range_check(column, rules['min'], rules['max'])
                    rule_count += 1
                
                if 'allowed_values' in rules:
                    validator.add_categorical_check(column, rules['allowed_values'])
                    rule_count += 1
                
                if 'pattern' in rules:
                    validator.add_pattern_check(column, rules['pattern'])
                    rule_count += 1
                
                print(f"     Added {rule_count} validation rules")
        
        # Run schema validation
        print(f"\n   🔬 Running schema validation...")
        schema_result = validator.validate(sample_data)
        
        print(f"     Schema compliance: {schema_result.success_rate:.1%}")
        print(f"     Critical issues: {len([i for i in schema_result.issues if i.severity == 'CRITICAL'])}")
        print(f"     Warning issues: {len([i for i in schema_result.issues if i.severity == 'WARNING'])}")
        
        # Show schema violations by type
        violation_types = {}
        for issue in schema_result.issues:
            issue_type = issue.check_type
            violation_types[issue_type] = violation_types.get(issue_type, 0) + 1
        
        print(f"\n   📊 Violation types:")
        for violation_type, count in violation_types.items():
            print(f"     {violation_type}: {count} issues")
        
        # Schema inference example
        print(f"\n   🔍 Schema inference from data:")
        inferred_schema = create_schema_from_dataframe(sample_data)
        print(f"     Inferred {len(inferred_schema.columns)} column schemas")
        
        for column in sample_data.columns[:3]:  # Show first 3 columns
            column_schema = inferred_schema.get_column_schema(column)
            print(f"     {column}: {column_schema.get('type', 'unknown')} "
                  f"(nullable: {column_schema.get('nullable', True)})")
    
    else:
        print("   🎭 Mock schema validation (pandas not available):")
        print("   Expected schema: 6 columns with type and constraint rules")
        print("   Validation results:")
        print("     Schema compliance: 78.5%")
        print("     Critical issues: 2 (invalid age values)")
        print("     Warning issues: 8 (missing values, invalid regions)")
        print("   Violation types:")
        print("     Range violations: 12")
        print("     Missing values: 80")
        print("     Invalid categories: 20")


def data_quality_assessment():
    """Demonstrate comprehensive data quality assessment."""
    print("\n📊 Data Quality Assessment")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Data quality assessment examples:
from refunc.data_science import DataValidator, DataQualityLevel

# Comprehensive quality assessment
validator = DataValidator()
quality_report = validator.assess_quality(df)

print(f"Overall quality: {quality_report.overall_quality}")
print(f"Completeness: {quality_report.completeness:.1%}")
print(f"Validity: {quality_report.validity:.1%}")
print(f"Consistency: {quality_report.consistency:.1%}")
print(f"Accuracy: {quality_report.accuracy:.1%}")

# Quality metrics by column
for column in df.columns:
    metrics = quality_report.get_column_metrics(column)
    print(f"{column}: {metrics.quality_score:.2f}")

# Automated quality improvement suggestions
suggestions = quality_report.get_improvement_suggestions()
for suggestion in suggestions:
    print(f"Suggestion: {suggestion.action} - {suggestion.reason}")
        """)
        return
    
    print("📈 Testing comprehensive quality assessment:")
    
    sample_data = create_sample_data()
    
    if PANDAS_AVAILABLE:
        # Manual quality assessment metrics
        print("   🔍 Computing quality metrics...")
        
        quality_metrics = {}
        
        for column in sample_data.columns:
            column_data = sample_data[column]
            total_count = len(column_data)
            
            # Completeness (non-null ratio)
            non_null_count = column_data.notna().sum()
            completeness = non_null_count / total_count
            
            # Validity (for specific column types)
            validity = 1.0  # Default to valid
            
            if column == 'age':
                valid_ages = column_data[(column_data >= 0) & (column_data <= 120)]
                validity = len(valid_ages) / non_null_count if non_null_count > 0 else 0
            
            elif column == 'income':
                valid_incomes = column_data[column_data >= 0]
                validity = len(valid_incomes) / non_null_count if non_null_count > 0 else 0
            
            elif column == 'credit_score':
                valid_scores = column_data[(column_data >= 300) & (column_data <= 850)]
                validity = len(valid_scores) / non_null_count if non_null_count > 0 else 0
            
            elif column == 'region':
                valid_regions = column_data[column_data.isin(['North', 'South', 'East', 'West'])]
                validity = len(valid_regions) / non_null_count if non_null_count > 0 else 0
            
            # Overall quality score (simple average)
            quality_score = (completeness + validity) / 2
            
            quality_metrics[column] = {
                'completeness': completeness,
                'validity': validity,
                'quality_score': quality_score,
                'total_records': total_count,
                'valid_records': int(non_null_count * validity)
            }
        
        # Display quality metrics
        print(f"\n   📊 Quality metrics by column:")
        for column, metrics in quality_metrics.items():
            print(f"     {column}:")
            print(f"       Completeness: {metrics['completeness']:.1%}")
            print(f"       Validity: {metrics['validity']:.1%}")
            print(f"       Quality Score: {metrics['quality_score']:.2f}")
            print(f"       Valid Records: {metrics['valid_records']}/{metrics['total_records']}")
        
        # Overall dataset quality
        overall_completeness = sum(m['completeness'] for m in quality_metrics.values()) / len(quality_metrics)
        overall_validity = sum(m['validity'] for m in quality_metrics.values()) / len(quality_metrics)
        overall_quality = (overall_completeness + overall_validity) / 2
        
        print(f"\n   🎯 Overall dataset quality:")
        print(f"     Completeness: {overall_completeness:.1%}")
        print(f"     Validity: {overall_validity:.1%}")
        print(f"     Overall Quality: {overall_quality:.2f}")
        
        # Quality level classification
        if overall_quality >= 0.9:
            quality_level = "EXCELLENT"
        elif overall_quality >= 0.8:
            quality_level = "GOOD"
        elif overall_quality >= 0.6:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        print(f"     Quality Level: {quality_level}")
        
        # Generate improvement suggestions
        print(f"\n   💡 Quality improvement suggestions:")
        suggestions = []
        
        for column, metrics in quality_metrics.items():
            if metrics['completeness'] < 0.95:
                missing_pct = (1 - metrics['completeness']) * 100
                suggestions.append(f"Address {missing_pct:.1f}% missing values in '{column}'")
            
            if metrics['validity'] < 0.90:
                invalid_pct = (1 - metrics['validity']) * 100
                suggestions.append(f"Fix {invalid_pct:.1f}% invalid values in '{column}'")
        
        for i, suggestion in enumerate(suggestions[:5], 1):  # Show top 5
            print(f"     {i}. {suggestion}")
        
        # Data profiling summary
        print(f"\n   📋 Data profiling summary:")
        print(f"     Total records: {len(sample_data):,}")
        print(f"     Total columns: {len(sample_data.columns)}")
        
        # Column type distribution
        numeric_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = sample_data.select_dtypes(include=['object', 'bool']).columns
        
        print(f"     Numeric columns: {len(numeric_cols)}")
        print(f"     Categorical columns: {len(categorical_cols)}")
        
        # Missing value summary
        total_missing = sample_data.isnull().sum().sum()
        total_cells = len(sample_data) * len(sample_data.columns)
        missing_rate = total_missing / total_cells
        
        print(f"     Missing values: {total_missing:,} ({missing_rate:.1%})")
    
    else:
        print("   🎭 Mock quality assessment (pandas not available):")
        print("   Quality metrics by column:")
        print("     customer_id: Completeness: 100.0%, Validity: 100.0%, Score: 1.00")
        print("     age: Completeness: 95.0%, Validity: 98.0%, Score: 0.97")
        print("     income: Completeness: 95.0%, Validity: 99.5%, Score: 0.97")
        print("     region: Completeness: 98.0%, Validity: 80.0%, Score: 0.89")
        print("   Overall dataset quality:")
        print("     Completeness: 97.0%")
        print("     Validity: 94.4%")
        print("     Overall Quality: 0.96")
        print("     Quality Level: EXCELLENT")


def automated_validation_reports():
    """Demonstrate automated validation report generation."""
    print("\n📄 Automated Validation Reports")
    print("=" * 50)
    
    if not REFUNC_AVAILABLE:
        print("""
# Automated validation reports:
from refunc.data_science import DataValidator, ValidationReport

# Generate comprehensive report
validator = DataValidator()
report = validator.generate_report(df)

# Export report in different formats
report.to_html('validation_report.html')
report.to_json('validation_report.json')
report.to_csv('validation_summary.csv')

# Interactive report features
print(report.executive_summary())
print(report.detailed_findings())
print(report.recommendations())

# Report customization
custom_report = ValidationReport(
    include_visualizations=True,
    severity_threshold='WARNING',
    max_issues_per_column=10
)
        """)
        return
    
    print("📊 Testing automated report generation:")
    
    sample_data = create_sample_data()
    
    # Generate validation report data
    validation_summary = {
        'dataset_info': {
            'name': 'customer_data_sample',
            'records': len(sample_data) if PANDAS_AVAILABLE else 1000,
            'columns': len(sample_data.columns) if PANDAS_AVAILABLE else 6,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': '2.4 MB'
        },
        'validation_results': {
            'total_checks': 45,
            'passed_checks': 32,
            'failed_checks': 13,
            'success_rate': 71.1,
            'critical_issues': 2,
            'warning_issues': 11
        },
        'quality_metrics': {
            'completeness': 94.8,
            'validity': 89.2,
            'consistency': 96.5,
            'overall_quality': 93.5
        },
        'column_issues': [
            {'column': 'age', 'issue_type': 'outliers', 'count': 10, 'severity': 'WARNING'},
            {'column': 'income', 'issue_type': 'missing_values', 'count': 50, 'severity': 'WARNING'},
            {'column': 'credit_score', 'issue_type': 'missing_values', 'count': 30, 'severity': 'WARNING'},
            {'column': 'region', 'issue_type': 'invalid_values', 'count': 20, 'severity': 'CRITICAL'}
        ],
        'recommendations': [
            'Investigate and correct age outliers (10 records with age > 120)',
            'Implement imputation strategy for missing income values (5.0%)',
            'Review data collection process for region field (2.0% invalid)',
            'Consider additional validation rules for credit_score range'
        ]
    }
    
    # Display executive summary
    print("   📋 Executive Summary:")
    info = validation_summary['dataset_info']
    results = validation_summary['validation_results']
    quality = validation_summary['quality_metrics']
    
    print(f"     Dataset: {info['name']} ({info['records']:,} records, {info['columns']} columns)")
    print(f"     Validation Date: {info['timestamp']}")
    print(f"     Overall Quality Score: {quality['overall_quality']:.1f}%")
    print(f"     Validation Success Rate: {results['success_rate']:.1f}%")
    print(f"     Critical Issues: {results['critical_issues']}")
    print(f"     Warning Issues: {results['warning_issues']}")
    
    # Display quality breakdown
    print(f"\n   📊 Quality Breakdown:")
    print(f"     Completeness: {quality['completeness']:.1f}%")
    print(f"     Validity: {quality['validity']:.1f}%")
    print(f"     Consistency: {quality['consistency']:.1f}%")
    
    # Display top issues
    print(f"\n   ⚠️  Top Issues by Impact:")
    for i, issue in enumerate(validation_summary['column_issues'], 1):
        severity_icon = "🔴" if issue['severity'] == 'CRITICAL' else "🟡"
        print(f"     {i}. {severity_icon} {issue['column']}: {issue['issue_type']} ({issue['count']} records)")
    
    # Display recommendations
    print(f"\n   💡 Key Recommendations:")
    for i, recommendation in enumerate(validation_summary['recommendations'][:3], 1):
        print(f"     {i}. {recommendation}")
    
    # Simulate report export
    print(f"\n   📤 Report Export Options:")
    report_formats = ['HTML', 'JSON', 'CSV', 'PDF']
    for fmt in report_formats:
        filename = f"validation_report_{int(time.time())}.{fmt.lower()}"
        print(f"     ✓ {fmt} report: {filename}")
    
    print(f"\n   🔗 Integration capabilities:")
    print(f"     • Email notifications for critical issues")
    print(f"     • Dashboard integration via API")
    print(f"     • Automated remediation workflows")
    print(f"     • Historical trend analysis")
    print(f"     • Custom validation rule templates")


def main():
    """Run all data validation examples."""
    print("🚀 Refunc Data Validation Examples")
    print("=" * 60)
    
    if not REFUNC_AVAILABLE:
        print("ℹ️  Running in demo mode (showing code structure)")
    else:
        print("ℹ️  Running with full Refunc functionality")
    
    if not PANDAS_AVAILABLE:
        print("⚠️  Pandas not available - using mock data structures")
    
    print()
    
    # Set random seed for reproducible examples
    random.seed(42)
    
    # Run examples
    basic_validation_examples()
    schema_validation_examples()
    data_quality_assessment()
    automated_validation_reports()
    
    print("\n✅ Data validation examples completed!")
    print("\n📖 Next steps:")
    print("- Implement data validation in your data pipelines")
    print("- Create custom validation schemas for your datasets")
    print("- Set up automated quality monitoring")
    print("- Check out preprocessing_pipeline.py for data transformation")


if __name__ == "__main__":
    main()