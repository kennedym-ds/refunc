"""
Data validation and quality assessment utilities.

This module provides comprehensive data validation tools including
schema validation, data quality checks, constraint validation,
and data integrity assessment.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import Counter
import re

from ..exceptions import RefuncError, ValidationError


class DataQualityLevel(Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    severity: ValidationSeverity
    message: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    rule_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of validation issue."""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.column:
            parts.append(f"Column '{self.column}':")
        parts.append(self.message)
        
        if self.row_indices and len(self.row_indices) <= 5:
            parts.append(f"(rows: {self.row_indices})")
        elif self.row_indices:
            parts.append(f"(rows: {self.row_indices[:5]}... and {len(self.row_indices)-5} more)")
            
        return " ".join(parts)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    quality_score: float
    quality_level: DataQualityLevel
    total_issues: int
    issues_by_severity: Dict[ValidationSeverity, int]
    issues: List[ValidationIssue]
    dataset_stats: Dict[str, Any] = field(default_factory=dict)
    column_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.issues_by_severity = Counter([issue.severity for issue in self.issues])
        self.total_issues = len(self.issues)
        
        # Determine quality level based on score
        if self.quality_score >= 0.9:
            self.quality_level = DataQualityLevel.EXCELLENT
        elif self.quality_score >= 0.8:
            self.quality_level = DataQualityLevel.GOOD
        elif self.quality_score >= 0.6:
            self.quality_level = DataQualityLevel.FAIR
        elif self.quality_score >= 0.4:
            self.quality_level = DataQualityLevel.POOR
        else:
            self.quality_level = DataQualityLevel.CRITICAL
    
    def summary(self) -> str:
        """Get formatted summary of validation report."""
        lines = [
            "Data Validation Report",
            "=" * 30,
            f"Overall Status: {'✅ VALID' if self.is_valid else '❌ INVALID'}",
            f"Quality Score: {self.quality_score:.2%}",
            f"Quality Level: {self.quality_level.value.upper()}",
            f"Total Issues: {self.total_issues}"
        ]
        
        if self.issues_by_severity:
            lines.append("\nIssues by Severity:")
            for severity, count in self.issues_by_severity.items():
                lines.append(f"  {severity.value.title()}: {count}")
        
        if self.dataset_stats:
            lines.append(f"\nDataset: {self.dataset_stats.get('rows', 0)} rows × {self.dataset_stats.get('columns', 0)} columns")
            
        return "\n".join(lines)
    
    def get_issues(self, severity: Optional[ValidationSeverity] = None, column: Optional[str] = None) -> List[ValidationIssue]:
        """Get filtered list of issues."""
        filtered_issues = self.issues
        
        if severity:
            filtered_issues = [issue for issue in filtered_issues if issue.severity == severity]
        
        if column:
            filtered_issues = [issue for issue in filtered_issues if issue.column == column]
            
        return filtered_issues


@dataclass
class DataSchema:
    """Data schema definition for validation."""
    columns: Dict[str, Dict[str, Any]]
    required_columns: List[str] = field(default_factory=list)
    optional_columns: List[str] = field(default_factory=list)
    primary_key: Optional[List[str]] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate_column_def(self, column_name: str, column_def: Dict[str, Any]) -> None:
        """Validate column definition."""
        required_fields = ['dtype']
        for field in required_fields:
            if field not in column_def:
                raise ValidationError(f"Column '{column_name}' missing required field: {field}")


class DataValidator:
    """Comprehensive data validation engine."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize data validator.
        
        Args:
            strict_mode: Whether to treat warnings as errors
        """
        self.strict_mode = strict_mode
        self._validation_rules = []
        self._custom_rules = {}
        
        # Register default validation rules
        self._register_default_rules()
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        schema: Optional[DataSchema] = None,
        rules: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            schema: Optional schema to validate against
            rules: Specific validation rules to apply
            
        Returns:
            ValidationReport with results
        """
        issues = []
        
        # Basic dataset statistics
        dataset_stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        # Column-level statistics
        column_stats = {}
        for col in df.columns:
            column_stats[col] = self._analyze_column(df[col])
        
        # Schema validation
        if schema:
            issues.extend(self._validate_schema(df, schema))
        
        # Apply validation rules
        rules_to_apply = rules or [rule['name'] for rule in self._validation_rules]
        for rule_name in rules_to_apply:
            try:
                rule_issues = self._apply_rule(df, rule_name)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to apply rule '{rule_name}': {str(e)}",
                    rule_name=rule_name
                ))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, issues)
        
        # Determine overall validity
        critical_errors = [issue for issue in issues 
                          if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        is_valid = len(critical_errors) == 0
        
        if self.strict_mode:
            warnings_count = len([issue for issue in issues if issue.severity == ValidationSeverity.WARNING])
            is_valid = is_valid and warnings_count == 0
        
        return ValidationReport(
            is_valid=is_valid,
            quality_score=quality_score,
            quality_level=DataQualityLevel.GOOD,  # Will be set in __post_init__
            total_issues=len(issues),
            issues_by_severity={},  # Will be set in __post_init__
            issues=issues,
            dataset_stats=dataset_stats,
            column_stats=column_stats
        )
    
    def _validate_schema(self, df: pd.DataFrame, schema: DataSchema) -> List[ValidationIssue]:
        """Validate DataFrame against schema."""
        issues = []
        
        # Check required columns
        missing_required = set(schema.required_columns) - set(df.columns)
        if missing_required:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Missing required columns: {list(missing_required)}",
                rule_name="required_columns"
            ))
        
        # Check column data types
        for col_name, col_def in schema.columns.items():
            if col_name not in df.columns:
                continue
                
            expected_dtype = col_def.get('dtype')
            actual_dtype = str(df[col_name].dtype)
            
            if expected_dtype and not self._is_compatible_dtype(actual_dtype, expected_dtype):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Expected dtype '{expected_dtype}', got '{actual_dtype}'",
                    column=col_name,
                    rule_name="dtype_validation"
                ))
        
        # Check primary key constraints
        if schema.primary_key:
            pk_issues = self._validate_primary_key(df, schema.primary_key)
            issues.extend(pk_issues)
        
        # Check custom constraints
        for constraint in schema.constraints:
            constraint_issues = self._validate_constraint(df, constraint)
            issues.extend(constraint_issues)
        
        return issues
    
    def _validate_primary_key(self, df: pd.DataFrame, primary_key: List[str]) -> List[ValidationIssue]:
        """Validate primary key constraints."""
        issues = []
        
        # Check if primary key columns exist
        missing_pk_cols = set(primary_key) - set(df.columns)
        if missing_pk_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Primary key columns missing: {list(missing_pk_cols)}",
                rule_name="primary_key"
            ))
            return issues
        
        # Check for duplicates
        pk_data = df[primary_key]
        duplicates = pk_data.duplicated()
        if duplicates.any():
            duplicate_rows = df.index[duplicates].tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Primary key constraint violated: duplicate values found",
                row_indices=duplicate_rows,
                rule_name="primary_key"
            ))
        
        # Check for null values in primary key
        null_mask = pk_data.isnull().any(axis=1)
        if null_mask.any():
            null_rows = df.index[null_mask].tolist()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Primary key contains null values",
                row_indices=null_rows,
                rule_name="primary_key"
            ))
        
        return issues
    
    def _validate_constraint(self, df: pd.DataFrame, constraint: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate custom constraint."""
        issues = []
        
        constraint_type = constraint.get('type')
        constraint_column = constraint.get('column')
        constraint_condition = constraint.get('condition')
        
        if constraint_type == 'range':
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            
            if constraint_column in df.columns:
                col_data = df[constraint_column]
                
                if min_val is not None:
                    violations = col_data < min_val
                    if violations.any():
                        violation_rows = df.index[violations].tolist()
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Values below minimum {min_val}",
                            column=constraint_column,
                            row_indices=violation_rows,
                            rule_name="range_constraint"
                        ))
                
                if max_val is not None:
                    violations = col_data > max_val
                    if violations.any():
                        violation_rows = df.index[violations].tolist()
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Values above maximum {max_val}",
                            column=constraint_column,
                            row_indices=violation_rows,
                            rule_name="range_constraint"
                        ))
        
        elif constraint_type == 'custom' and constraint_condition:
            try:
                # Evaluate custom condition
                violations = ~df.eval(constraint_condition)
                if violations.any():
                    violation_rows = df.index[violations].tolist()
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Custom constraint violated: {constraint_condition}",
                        row_indices=violation_rows,
                        rule_name="custom_constraint"
                    ))
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Failed to evaluate constraint: {str(e)}",
                    rule_name="custom_constraint"
                ))
        
        return issues
    
    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        self._validation_rules = [
            {
                'name': 'missing_values',
                'description': 'Check for missing values',
                'function': self._check_missing_values
            },
            {
                'name': 'duplicates',
                'description': 'Check for duplicate rows',
                'function': self._check_duplicates
            },
            {
                'name': 'data_types',
                'description': 'Validate data types',
                'function': self._check_data_types
            },
            {
                'name': 'outliers',
                'description': 'Detect outliers',
                'function': self._check_outliers
            },
            {
                'name': 'consistency',
                'description': 'Check data consistency',
                'function': self._check_consistency
            },
            {
                'name': 'completeness',
                'description': 'Check data completeness',
                'function': self._check_completeness
            }
        ]
    
    def _apply_rule(self, df: pd.DataFrame, rule_name: str) -> List[ValidationIssue]:
        """Apply a specific validation rule."""
        # Find rule by name
        rule = None
        for r in self._validation_rules:
            if r['name'] == rule_name:
                rule = r
                break
        
        if not rule:
            if rule_name in self._custom_rules:
                return self._custom_rules[rule_name](df)
            else:
                raise ValidationError(f"Unknown validation rule: {rule_name}")
        
        return rule['function'](df)
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for missing values."""
        issues = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = missing_count / len(df) * 100
            
            if missing_count > 0:
                severity = ValidationSeverity.INFO
                if missing_percent > 50:
                    severity = ValidationSeverity.CRITICAL
                elif missing_percent > 20:
                    severity = ValidationSeverity.ERROR
                elif missing_percent > 5:
                    severity = ValidationSeverity.WARNING
                
                missing_rows = df.index[df[col].isnull()].tolist()
                issues.append(ValidationIssue(
                    severity=severity,
                    message=f"{missing_count} missing values ({missing_percent:.1f}%)",
                    column=col,
                    row_indices=missing_rows if len(missing_rows) <= 100 else missing_rows[:100],
                    rule_name="missing_values",
                    details={'missing_count': missing_count, 'missing_percent': missing_percent}
                ))
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for duplicate rows."""
        issues = []
        
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            duplicate_percent = duplicate_count / len(df) * 100
            severity = ValidationSeverity.WARNING
            if duplicate_percent > 10:
                severity = ValidationSeverity.ERROR
            
            duplicate_rows = df.index[duplicates].tolist()
            issues.append(ValidationIssue(
                severity=severity,
                message=f"{duplicate_count} duplicate rows ({duplicate_percent:.1f}%)",
                row_indices=duplicate_rows,
                rule_name="duplicates",
                details={'duplicate_count': duplicate_count, 'duplicate_percent': duplicate_percent}
            ))
        
        return issues
    
    def _check_data_types(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check data types for potential issues."""
        issues = []
        
        for col in df.columns:
            col_data = df[col]
            
            # Check for mixed types in object columns
            if col_data.dtype == 'object':
                types = set(type(x).__name__ for x in col_data.dropna().values)
                if len(types) > 1:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Mixed data types detected: {types}",
                        column=col,
                        rule_name="data_types"
                    ))
            
            # Check for potential numeric columns stored as strings
            if col_data.dtype == 'object':
                numeric_pattern = re.compile(r'^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$')
                non_null_values = col_data.dropna().astype(str)
                if len(non_null_values) > 0:
                    numeric_matches = non_null_values.str.match(numeric_pattern)
                    numeric_percent = numeric_matches.sum() / len(non_null_values) * 100
                    
                    if numeric_percent > 80:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Column appears to be numeric but stored as text ({numeric_percent:.1f}% numeric)",
                            column=col,
                            rule_name="data_types"
                        ))
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for outliers in numeric columns."""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Use IQR method for outlier detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_percent = outlier_count / len(col_data) * 100
                severity = ValidationSeverity.INFO
                if outlier_percent > 10:
                    severity = ValidationSeverity.WARNING
                
                outlier_rows = df.index[df[col].isin(col_data[outliers])].tolist()
                issues.append(ValidationIssue(
                    severity=severity,
                    message=f"{outlier_count} outliers detected ({outlier_percent:.1f}%)",
                    column=col,
                    row_indices=outlier_rows[:50],  # Limit to first 50
                    rule_name="outliers",
                    details={
                        'outlier_count': outlier_count,
                        'outlier_percent': outlier_percent,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                ))
        
        return issues
    
    def _check_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for data consistency issues."""
        issues = []
        
        # Check for inconsistent string formats in object columns
        object_columns = df.select_dtypes(include=['object']).columns
        
        for col in object_columns:
            col_data = df[col].dropna().astype(str)
            if len(col_data) == 0:
                continue
            
            # Check for mixed case issues
            mixed_case_count = 0
            unique_values = set(col_data.values)
            for value in unique_values:
                similar_values = [v for v in unique_values if v.lower() == value.lower() and v != value]
                if similar_values:
                    mixed_case_count += len(similar_values) + 1
                    break
            
            if mixed_case_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Inconsistent capitalization detected",
                    column=col,
                    rule_name="consistency"
                ))
            
            # Check for leading/trailing whitespace
            has_whitespace = col_data.str.startswith(' ') | col_data.str.endswith(' ')
            whitespace_count = has_whitespace.sum()
            
            if whitespace_count > 0:
                whitespace_rows = df.index[df[col].isin(col_data[has_whitespace])].tolist()
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"{whitespace_count} values with leading/trailing whitespace",
                    column=col,
                    row_indices=whitespace_rows[:20],
                    rule_name="consistency"
                ))
        
        return issues
    
    def _check_completeness(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check data completeness."""
        issues = []
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        # Avoid division by zero
        if total_cells == 0:
            completeness_percent = 100.0
        else:
            completeness_percent = (total_cells - missing_cells) / total_cells * 100
        
        if completeness_percent < 95:
            severity = ValidationSeverity.INFO
            if completeness_percent < 80:
                severity = ValidationSeverity.WARNING
            if completeness_percent < 60:
                severity = ValidationSeverity.ERROR
            
            issues.append(ValidationIssue(
                severity=severity,
                message=f"Dataset completeness: {completeness_percent:.1f}%",
                rule_name="completeness",
                details={'completeness_percent': completeness_percent}
            ))
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score."""
        if len(issues) == 0:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.05,
            ValidationSeverity.WARNING: 0.2,
            ValidationSeverity.ERROR: 0.5,
            ValidationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 0.1) for issue in issues)
        max_possible_penalty = len(issues) * 1.0  # Assume all critical
        
        # Normalize penalty
        normalized_penalty = min(total_penalty / max_possible_penalty, 1.0) if max_possible_penalty > 0 else 0
        
        return max(0.0, 1.0 - normalized_penalty)
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column."""
        stats = {
            'dtype': str(series.dtype),
            'non_null_count': series.count(),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': series.nunique() / len(series) * 100 if len(series) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(series):
            stats.update({
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'median': series.median(),
                'q1': series.quantile(0.25),
                'q3': series.quantile(0.75)
            })
        
        return stats
    
    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected."""
        # Simplified compatibility check
        compatibility_map = {
            'int64': ['int32', 'int64', 'float64'],
            'float64': ['float32', 'float64', 'int32', 'int64'],
            'object': ['object', 'string'],
            'bool': ['bool'],
            'datetime64[ns]': ['datetime64[ns]', 'datetime64']
        }
        
        return actual in compatibility_map.get(expected, [expected])
    
    def add_custom_rule(self, name: str, rule_function: Callable[[pd.DataFrame], List[ValidationIssue]]) -> None:
        """Add a custom validation rule."""
        self._custom_rules[name] = rule_function
    
    def remove_custom_rule(self, name: str) -> None:
        """Remove a custom validation rule."""
        if name in self._custom_rules:
            del self._custom_rules[name]


# Convenience functions
def validate_dataframe(
    df: pd.DataFrame,
    schema: Optional[DataSchema] = None,
    strict: bool = False
) -> ValidationReport:
    """Validate a DataFrame with default settings."""
    validator = DataValidator(strict_mode=strict)
    return validator.validate_dataframe(df, schema)


def quick_validate(df: pd.DataFrame) -> bool:
    """Quick validation - returns True if data is valid."""
    validator = DataValidator()
    report = validator.validate_dataframe(df)
    return report.is_valid


def create_schema_from_dataframe(df: pd.DataFrame) -> DataSchema:
    """Create a schema definition from an existing DataFrame."""
    columns = {}
    for col in df.columns:
        columns[col] = {
            'dtype': str(df[col].dtype),
            'nullable': df[col].isnull().any(),
            'unique_count': df[col].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            columns[col].update({
                'min': df[col].min(),
                'max': df[col].max()
            })
    
    return DataSchema(
        columns=columns,
        required_columns=list(df.columns)
    )