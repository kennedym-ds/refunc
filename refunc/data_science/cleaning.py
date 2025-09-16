"""
Data cleaning and quality improvement utilities.

This module provides comprehensive data cleaning tools including
duplicate removal, data type correction, format standardization,
and automated data quality improvements.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Tuple, Pattern
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import re
from datetime import datetime, date
from dateutil import parser as date_parser

from ..exceptions import RefuncError, ValidationError
from ..logging import get_logger


class CleaningOperation(Enum):
    """Types of cleaning operations."""
    REMOVE_DUPLICATES = "remove_duplicates"
    STANDARDIZE_TEXT = "standardize_text"
    FIX_DATA_TYPES = "fix_data_types"
    CLEAN_NUMERIC = "clean_numeric"
    STANDARDIZE_DATES = "standardize_dates"
    REMOVE_OUTLIERS = "remove_outliers"
    NORMALIZE_STRINGS = "normalize_strings"
    VALIDATE_FORMATS = "validate_formats"


@dataclass
class CleaningResult:
    """Result of a data cleaning operation."""
    operation: CleaningOperation
    success: bool
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    changes_made: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get summary of cleaning result."""
        status = "✅" if self.success else "❌"
        shape_change = f"{self.original_shape} → {self.final_shape}"
        return f"{status} {self.operation.value}: {self.changes_made} changes, {shape_change}"


@dataclass
class CleaningReport:
    """Comprehensive cleaning report."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    total_changes: int
    operations_performed: List[CleaningResult]
    execution_time: float
    data_quality_before: float
    data_quality_after: float
    
    def summary(self) -> str:
        """Get formatted summary."""
        successful_ops = sum(1 for op in self.operations_performed if op.success)
        total_ops = len(self.operations_performed)
        
        lines = [
            "Data Cleaning Report",
            "=" * 30,
            f"Shape Change: {self.original_shape} → {self.final_shape}",
            f"Total Changes: {self.total_changes}",
            f"Operations: {successful_ops}/{total_ops} successful",
            f"Quality Improvement: {self.data_quality_before:.2%} → {self.data_quality_after:.2%}",
            f"Execution Time: {self.execution_time:.3f}s"
        ]
        
        if self.operations_performed:
            lines.append("\nOperations Performed:")
            for result in self.operations_performed:
                lines.append(f"  • {result.summary()}")
        
        return "\n".join(lines)


class DataCleaner:
    """Comprehensive data cleaning engine."""
    
    def __init__(self, aggressive_cleaning: bool = False):
        """
        Initialize data cleaner.
        
        Args:
            aggressive_cleaning: Whether to apply more aggressive cleaning rules
        """
        self.aggressive_cleaning = aggressive_cleaning
        self.logger = get_logger("data_cleaner")
        
        # Common patterns for data validation
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_pattern = re.compile(r'^[\+]?[1-9][\d]{0,15}$')
        self.url_pattern = re.compile(r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$')
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        operations: Optional[List[CleaningOperation]] = None,
        column_types: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df: DataFrame to clean
            operations: Specific operations to perform (None for all)
            column_types: Expected data types for columns
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        start_time = datetime.now()
        original_df = df.copy()
        current_df = df.copy()
        results = []
        total_changes = 0
        
        # Default operations
        if operations is None:
            operations = [
                CleaningOperation.REMOVE_DUPLICATES,
                CleaningOperation.STANDARDIZE_TEXT,
                CleaningOperation.FIX_DATA_TYPES,
                CleaningOperation.CLEAN_NUMERIC,
                CleaningOperation.STANDARDIZE_DATES,
                CleaningOperation.NORMALIZE_STRINGS
            ]
            
            if self.aggressive_cleaning:
                operations.extend([
                    CleaningOperation.REMOVE_OUTLIERS,
                    CleaningOperation.VALIDATE_FORMATS
                ])
        
        # Calculate initial quality score
        quality_before = self._calculate_quality_score(current_df)
        
        # Perform cleaning operations
        for operation in operations:
            try:
                original_shape = current_df.shape
                
                if operation == CleaningOperation.REMOVE_DUPLICATES:
                    current_df, changes = self._remove_duplicates(current_df)
                elif operation == CleaningOperation.STANDARDIZE_TEXT:
                    current_df, changes = self._standardize_text(current_df)
                elif operation == CleaningOperation.FIX_DATA_TYPES:
                    current_df, changes = self._fix_data_types(current_df, column_types)
                elif operation == CleaningOperation.CLEAN_NUMERIC:
                    current_df, changes = self._clean_numeric(current_df)
                elif operation == CleaningOperation.STANDARDIZE_DATES:
                    current_df, changes = self._standardize_dates(current_df)
                elif operation == CleaningOperation.REMOVE_OUTLIERS:
                    current_df, changes = self._remove_outliers(current_df)
                elif operation == CleaningOperation.NORMALIZE_STRINGS:
                    current_df, changes = self._normalize_strings(current_df)
                elif operation == CleaningOperation.VALIDATE_FORMATS:
                    current_df, changes = self._validate_formats(current_df)
                else:
                    changes = 0
                
                final_shape = current_df.shape
                total_changes += changes
                
                result = CleaningResult(
                    operation=operation,
                    success=True,
                    original_shape=original_shape,
                    final_shape=final_shape,
                    changes_made=changes
                )
                
                results.append(result)
                self.logger.info(f"Completed {operation.value}: {changes} changes made")
                
            except Exception as e:
                result = CleaningResult(
                    operation=operation,
                    success=False,
                    original_shape=current_df.shape,
                    final_shape=current_df.shape,
                    changes_made=0,
                    errors=[str(e)]
                )
                
                results.append(result)
                self.logger.error(f"Failed {operation.value}: {str(e)}")
        
        # Calculate final quality score
        quality_after = self._calculate_quality_score(current_df)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        report = CleaningReport(
            original_shape=original_df.shape,
            final_shape=current_df.shape,
            total_changes=total_changes,
            operations_performed=results,
            execution_time=execution_time,
            data_quality_before=quality_before,
            data_quality_after=quality_after
        )
        
        return current_df, report
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows."""
        original_count = len(df)
        df_cleaned = df.drop_duplicates()
        changes = original_count - len(df_cleaned)
        
        if changes > 0:
            self.logger.info(f"Removed {changes} duplicate rows")
        
        return df_cleaned, changes
    
    def _standardize_text(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Standardize text data."""
        df_cleaned = df.copy()
        changes = 0
        
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            original_values = df_cleaned[col].copy()
            
            # Remove leading/trailing whitespace
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
            # Standardize case for short strings (likely categorical)
            if df_cleaned[col].str.len().median() < 50:  # Assume categorical if short
                most_common_case = df_cleaned[col].value_counts().index[0] if len(df_cleaned[col].value_counts()) > 0 else None
                if most_common_case:
                    if most_common_case.isupper():
                        df_cleaned[col] = df_cleaned[col].str.upper()
                    elif most_common_case.islower():
                        df_cleaned[col] = df_cleaned[col].str.lower()
                    elif most_common_case.istitle():
                        df_cleaned[col] = df_cleaned[col].str.title()
            
            # Count changes
            changes += (original_values != df_cleaned[col]).sum()
        
        return df_cleaned, changes
    
    def _fix_data_types(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, int]:
        """Fix and optimize data types."""
        df_cleaned = df.copy()
        changes = 0
        
        for col in df_cleaned.columns:
            original_dtype = df_cleaned[col].dtype
            
            # Use provided type hints
            if column_types and col in column_types:
                target_type = column_types[col]
                try:
                    if target_type in ['int', 'integer']:
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').astype('Int64')
                    elif target_type in ['float', 'numeric']:
                        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    elif target_type in ['datetime', 'date']:
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                    elif target_type in ['bool', 'boolean']:
                        df_cleaned[col] = df_cleaned[col].astype('boolean')
                    elif target_type in ['category', 'categorical']:
                        df_cleaned[col] = df_cleaned[col].astype('category')
                    
                    if df_cleaned[col].dtype != original_dtype:
                        changes += 1
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to {target_type}: {str(e)}")
            
            # Auto-detect numeric columns stored as strings
            elif df_cleaned[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df_cleaned[col], errors='coerce')
                numeric_ratio = numeric_series.notna().sum() / len(df_cleaned[col])
                
                if numeric_ratio > 0.8:  # If >80% can be converted to numeric
                    # Check if integers
                    if numeric_series.dropna().apply(lambda x: x.is_integer()).all():
                        df_cleaned[col] = numeric_series.astype('Int64')
                    else:
                        df_cleaned[col] = numeric_series
                    changes += 1
                
                # Try to convert to datetime
                elif self._is_likely_datetime(df_cleaned[col]):
                    datetime_series = pd.to_datetime(df_cleaned[col], errors='coerce')
                    datetime_ratio = datetime_series.notna().sum() / len(df_cleaned[col])
                    
                    if datetime_ratio > 0.8:
                        df_cleaned[col] = datetime_series
                        changes += 1
        
        return df_cleaned, changes
    
    def _clean_numeric(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Clean numeric data."""
        df_cleaned = df.copy()
        changes = 0
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            original_values = df_cleaned[col].copy()
            
            # Replace infinite values with NaN
            inf_mask = np.isinf(df_cleaned[col])
            if inf_mask.any():
                df_cleaned.loc[inf_mask, col] = np.nan
                changes += inf_mask.sum()
            
            # Round very small numbers to zero (potential floating point errors)
            if df_cleaned[col].dtype == 'float64':
                very_small_mask = (abs(df_cleaned[col]) < 1e-10) & (df_cleaned[col] != 0)
                if very_small_mask.any():
                    df_cleaned.loc[very_small_mask, col] = 0
                    changes += very_small_mask.sum()
        
        return df_cleaned, changes
    
    def _standardize_dates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Standardize date columns."""
        df_cleaned = df.copy()
        changes = 0
        
        # Find potential date columns
        potential_date_columns = []
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                if self._is_likely_datetime(df_cleaned[col]):
                    potential_date_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                potential_date_columns.append(col)
        
        for col in potential_date_columns:
            try:
                original_dtype = df_cleaned[col].dtype
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                
                if df_cleaned[col].dtype != original_dtype:
                    changes += 1
                    
            except Exception as e:
                self.logger.warning(f"Could not standardize dates in {col}: {str(e)}")
        
        return df_cleaned, changes
    
    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove outliers from numeric columns."""
        df_cleaned = df.copy()
        original_count = len(df_cleaned)
        
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series([False] * len(df_cleaned), index=df_cleaned.index)
        
        for col in numeric_columns:
            col_data = df_cleaned[col].dropna()
            if len(col_data) == 0:
                continue
                
            # Use IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Skip if no variation
                continue
                
            lower_bound = Q1 - 3 * IQR  # More conservative than 1.5
            upper_bound = Q3 + 3 * IQR
            
            col_outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
            outlier_mask |= col_outliers
        
        # Remove outlier rows
        df_cleaned = df_cleaned[~outlier_mask]
        changes = original_count - len(df_cleaned)
        
        return df_cleaned, changes
    
    def _normalize_strings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Normalize string formatting."""
        df_cleaned = df.copy()
        changes = 0
        
        text_columns = df_cleaned.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            original_values = df_cleaned[col].copy()
            
            # Convert to string and handle NaN
            df_cleaned[col] = df_cleaned[col].astype(str)
            
            # Replace 'nan' strings with actual NaN
            nan_mask = df_cleaned[col].str.lower() == 'nan'
            df_cleaned.loc[nan_mask, col] = np.nan
            
            # Normalize whitespace
            df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters for categorical-like columns
            if df_cleaned[col].nunique() < len(df_cleaned) * 0.1:  # Likely categorical
                df_cleaned[col] = df_cleaned[col].str.replace(r'[^\w\s]', '', regex=True)
            
            # Count changes
            changes += (original_values.astype(str) != df_cleaned[col].astype(str)).sum()
        
        return df_cleaned, changes
    
    def _validate_formats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Validate and correct common data formats."""
        df_cleaned = df.copy()
        changes = 0
        
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                col_data = df_cleaned[col].dropna().astype(str)
                
                # Email validation
                if col.lower() in ['email', 'e_mail', 'email_address']:
                    invalid_emails = ~col_data.str.match(self.email_pattern)
                    if invalid_emails.any():
                        df_cleaned.loc[df_cleaned[col].isin(col_data[invalid_emails]), col] = np.nan
                        changes += invalid_emails.sum()
                
                # Phone number validation
                elif col.lower() in ['phone', 'telephone', 'phone_number']:
                    # Clean phone numbers
                    cleaned_phones = col_data.str.replace(r'[^\d+]', '', regex=True)
                    df_cleaned[col] = cleaned_phones
                    changes += (col_data != cleaned_phones).sum()
                
                # URL validation
                elif col.lower() in ['url', 'website', 'link']:
                    invalid_urls = ~col_data.str.match(self.url_pattern)
                    if invalid_urls.any():
                        df_cleaned.loc[df_cleaned[col].isin(col_data[invalid_urls]), col] = np.nan
                        changes += invalid_urls.sum()
        
        return df_cleaned, changes
    
    def _is_likely_datetime(self, series: pd.Series) -> bool:
        """Check if a series likely contains datetime data."""
        if len(series.dropna()) == 0:
            return False
        
        # Sample a few values and try to parse them
        sample_size = min(10, len(series.dropna()))
        sample_values = series.dropna().astype(str).sample(sample_size).tolist()
        
        successful_parses = 0
        for value in sample_values:
            try:
                date_parser.parse(value)
                successful_parses += 1
            except:
                continue
        
        return successful_parses / len(sample_values) > 0.7
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate a data quality score (0-1)."""
        if len(df) == 0:
            return 0.0
        
        scores = []
        
        # Completeness score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        scores.append(completeness)
        
        # Consistency score (based on data types)
        type_consistency = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed types in object columns
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    types = set(type(x).__name__ for x in non_null_values.values)
                    type_consistency += 1 / len(types)  # Lower score for more mixed types
                else:
                    type_consistency += 1
            else:
                type_consistency += 1
        
        type_consistency /= len(df.columns) if len(df.columns) > 0 else 1
        scores.append(type_consistency)
        
        # Duplicate score
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        duplicate_score = 1 - duplicate_ratio
        scores.append(duplicate_score)
        
        return np.mean(scores)


# Convenience functions
def quick_clean(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """Quick data cleaning with default settings."""
    cleaner = DataCleaner(aggressive_cleaning=aggressive)
    cleaned_df, _ = cleaner.clean_dataframe(df)
    return cleaned_df


def clean_with_report(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, CleaningReport]:
    """Clean data and return detailed report."""
    cleaner = DataCleaner(aggressive_cleaning=aggressive)
    return cleaner.clean_dataframe(df)


def remove_duplicates_advanced(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    ignore_case: bool = True
) -> pd.DataFrame:
    """Advanced duplicate removal with case-insensitive option."""
    if ignore_case and subset:
        # Create temporary columns with lowercase values for comparison
        temp_df = df.copy()
        temp_cols = []
        
        for col in subset:
            if col in df.columns and df[col].dtype == 'object':
                temp_col = f"_temp_{col}_lower"
                temp_df[temp_col] = df[col].astype(str).str.lower()
                temp_cols.append(temp_col)
            else:
                temp_cols.append(col)
        
        # Remove duplicates based on temporary columns
        mask = ~temp_df.duplicated(subset=temp_cols, keep=keep)
        result = df[mask].copy()
    else:
        result = df.drop_duplicates(subset=subset, keep=keep)
    
    return result


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case."""
    df_cleaned = df.copy()
    
    new_columns = []
    for col in df_cleaned.columns:
        # Convert to string and lowercase
        new_col = str(col).lower()
        
        # Replace spaces and special characters with underscores
        new_col = re.sub(r'[^\w]', '_', new_col)
        
        # Remove multiple consecutive underscores
        new_col = re.sub(r'_+', '_', new_col)
        
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        
        new_columns.append(new_col)
    
    df_cleaned.columns = new_columns
    return df_cleaned


def detect_encoding_issues(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect potential encoding issues in text columns."""
    issues = {}
    
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        col_issues = []
        
        for idx, value in df[col].dropna().items():
            if isinstance(value, str):
                # Check for common encoding issues
                if '�' in value:
                    col_issues.append(f"Row {idx}: Contains replacement character")
                
                # Check for mixed encodings
                try:
                    value.encode('ascii')
                except UnicodeEncodeError:
                    if any(ord(char) > 127 for char in value):
                        col_issues.append(f"Row {idx}: Contains non-ASCII characters")
        
        if col_issues:
            issues[col] = col_issues[:10]  # Limit to first 10 issues
    
    return issues