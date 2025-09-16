"""
Data Science operations router.

Provides endpoints for:
- Data validation
- Data cleaning
- Data profiling
- Data transformations
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import pandas as pd
import io
import json
import time

from ...data_science.validation import DataValidator
from ...data_science.cleaning import DataCleaner
from ...data_science.profiling import DataProfiler
from ..models.responses import (
    ValidationReportResponse,
    DataProfileResponse,
    CleaningReportResponse,
    ProcessingRequest
)

router = APIRouter()

@router.post("/validate", response_model=ValidationReportResponse)
async def validate_data(
    file: UploadFile = File(...),
    validation_rules: Optional[str] = Form(None)
) -> ValidationReportResponse:
    """
    Validate uploaded data file.
    
    Args:
        file: Data file to validate (CSV, JSON, Excel)
        validation_rules: Optional JSON string of validation configuration
    
    Returns:
        Validation report with issues and quality metrics
    """
    try:
        start_time = time.time()
        
        # Read the uploaded file
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Parse validation rules if provided
        rules = {}
        if validation_rules:
            rules = json.loads(validation_rules)
        
        # Create validator and run validation
        validator = DataValidator()
        report = validator.validate(df, **rules)
        
        execution_time = time.time() - start_time
        
        # Convert to response model
        return ValidationReportResponse(
            issues=[
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "column": issue.column,
                    "row_indices": issue.row_indices,
                    "rule_name": issue.rule_name,
                    "details": issue.details
                }
                for issue in report.issues
            ],
            total_issues=report.total_issues,
            quality_score=report.quality_score,
            quality_level=report.quality_level.value,
            issues_by_severity={k.value: v for k, v in report.issues_by_severity.items()},
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@router.post("/profile", response_model=DataProfileResponse)
async def profile_data(
    file: UploadFile = File(...),
    profile_options: Optional[str] = Form(None)
) -> DataProfileResponse:
    """
    Generate comprehensive data profile.
    
    Args:
        file: Data file to profile (CSV, JSON, Excel)
        profile_options: Optional JSON string of profiling configuration
    
    Returns:
        Detailed data profile with statistics and insights
    """
    try:
        # Read the uploaded file
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Parse profile options if provided
        options = {}
        if profile_options:
            options = json.loads(profile_options)
        
        # Create profiler and generate profile
        profiler = DataProfiler()
        profile = profiler.profile(df, **options)
        
        # Convert to response model
        return DataProfileResponse(
            shape=list(profile.shape),
            missing_percentage=profile.missing_percentage,
            duplicate_percentage=profile.duplicate_percentage,
            completeness_score=profile.completeness_score,
            consistency_score=profile.consistency_score,
            data_quality_score=profile.data_quality_score,
            columns={
                name: {
                    "dtype": str(col.dtype),
                    "missing_count": col.missing_count,
                    "unique_count": col.unique_count,
                    "has_outliers": col.has_outliers,
                    "statistics": col.statistics
                }
                for name, col in profile.columns.items()
            },
            insights=profile.insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Profiling failed: {str(e)}")

@router.post("/clean", response_model=CleaningReportResponse)
async def clean_data(
    file: UploadFile = File(...),
    cleaning_options: Optional[str] = Form(None)
) -> CleaningReportResponse:
    """
    Clean uploaded data file.
    
    Args:
        file: Data file to clean (CSV, JSON, Excel)
        cleaning_options: Optional JSON string of cleaning configuration
    
    Returns:
        Cleaning report with operations performed and quality improvement
    """
    try:
        start_time = time.time()
        
        # Read the uploaded file
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Parse cleaning options if provided
        options = {}
        if cleaning_options:
            options = json.loads(cleaning_options)
        
        # Create cleaner and clean data
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.clean(df, **options)
        
        execution_time = time.time() - start_time
        
        # Convert to response model
        return CleaningReportResponse(
            original_shape=list(report.original_shape),
            final_shape=list(report.final_shape),
            total_changes=report.total_changes,
            operations_performed=[
                {
                    "operation": op.operation.value if hasattr(op.operation, 'value') else str(op.operation),
                    "success": op.success,
                    "changes_made": op.changes_made,
                    "message": op.message,
                    "execution_time": op.execution_time
                }
                for op in report.operations_performed
            ],
            execution_time=execution_time,
            data_quality_before=report.data_quality_before,
            data_quality_after=report.data_quality_after
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cleaning failed: {str(e)}")

@router.post("/validate-json")
async def validate_json_data(request: ProcessingRequest) -> ValidationReportResponse:
    """
    Validate JSON data directly.
    
    Args:
        request: Processing request with data and options
    
    Returns:
        Validation report
    """
    try:
        start_time = time.time()
        
        # Convert data to DataFrame
        if isinstance(request.data, list):
            df = pd.DataFrame(request.data)
        else:
            df = pd.DataFrame([request.data])
        
        # Create validator and run validation
        validator = DataValidator()
        options = request.options or {}
        report = validator.validate(df, **options)
        
        execution_time = time.time() - start_time
        
        # Convert to response model
        return ValidationReportResponse(
            issues=[
                {
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "column": issue.column,
                    "row_indices": issue.row_indices,
                    "rule_name": issue.rule_name,
                    "details": issue.details
                }
                for issue in report.issues
            ],
            total_issues=report.total_issues,
            quality_score=report.quality_score,
            quality_level=report.quality_level.value,
            issues_by_severity={k.value: v for k, v in report.issues_by_severity.items()},
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON validation failed: {str(e)}")

def _read_file_content(content: bytes, filename: str) -> pd.DataFrame:
    """Helper function to read file content into DataFrame."""
    try:
        if filename.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif filename.endswith('.json'):
            return pd.read_json(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(content))
        elif filename.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(content))
        else:
            # Try CSV as default
            return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Unable to read file {filename}: {str(e)}")