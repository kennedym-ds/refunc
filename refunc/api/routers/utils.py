"""
Utility operations router.

Provides endpoints for:
- File handling operations
- Caching utilities
- Format conversions
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
import pandas as pd
import io
import json
import zipfile
import tempfile
import os
from pydantic import BaseModel

from ...utils.file_handler import FileHandler
from ...utils.cache import MemoryCache
from ...utils.formats import FormatRegistry

router = APIRouter()

class FileConversionRequest(BaseModel):
    """Request model for file format conversion."""
    source_format: str
    target_format: str
    options: Optional[Dict[str, Any]] = None

class CacheRequest(BaseModel):
    """Request model for cache operations."""
    key: str
    data: Optional[Any] = None
    ttl: Optional[int] = None  # Time to live in seconds

@router.post("/files/convert")
async def convert_file_format(
    file: UploadFile = File(...),
    target_format: str = Form(...),
    options: Optional[str] = Form(None)
) -> StreamingResponse:
    """
    Convert file from one format to another.
    
    Args:
        file: Source file to convert
        target_format: Target format (csv, json, excel, parquet)
        options: Optional JSON string of conversion options
    
    Returns:
        Converted file as download
    """
    try:
        # Read the uploaded file
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Parse options
        conversion_options = {}
        if options:
            conversion_options = json.loads(options)
        
        # Convert to target format
        output_buffer = io.BytesIO()
        
        if target_format.lower() == "csv":
            df.to_csv(output_buffer, index=False, **conversion_options)
            media_type = "text/csv"
            extension = "csv"
        elif target_format.lower() == "json":
            df.to_json(output_buffer, orient="records", **conversion_options)
            media_type = "application/json"
            extension = "json"
        elif target_format.lower() == "excel":
            df.to_excel(output_buffer, index=False, **conversion_options)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            extension = "xlsx"
        elif target_format.lower() == "parquet":
            df.to_parquet(output_buffer, index=False, **conversion_options)
            media_type = "application/octet-stream"
            extension = "parquet"
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        output_buffer.seek(0)
        
        # Generate filename
        base_name = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename
        output_filename = f"{base_name}.{extension}"
        
        return StreamingResponse(
            io.BytesIO(output_buffer.read()),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File conversion failed: {str(e)}")

@router.post("/files/info")
async def get_file_info(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Get detailed information about uploaded file.
    
    Args:
        file: File to analyze
    
    Returns:
        File information and metadata
    """
    try:
        content = await file.read()
        df = _read_file_content(content, file.filename)
        
        # Basic file info
        file_info = {
            "filename": file.filename,
            "size_bytes": len(content),
            "content_type": file.content_type,
        }
        
        # Data info
        data_info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Sample data
        sample_data = {
            "head": df.head(5).to_dict('records'),
            "tail": df.tail(5).to_dict('records')
        }
        
        # Statistical summary for numeric columns
        numeric_summary = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_summary = df[numeric_cols].describe().to_dict()
        
        return {
            "file_info": file_info,
            "data_info": data_info,
            "sample_data": sample_data,
            "numeric_summary": numeric_summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File analysis failed: {str(e)}")

@router.post("/files/merge")
async def merge_files(
    files: List[UploadFile] = File(...),
    merge_type: str = Form("concat"),  # concat, join, merge
    options: Optional[str] = Form(None)
) -> StreamingResponse:
    """
    Merge multiple files into one.
    
    Args:
        files: List of files to merge
        merge_type: Type of merge operation (concat, join, merge)
        options: Optional JSON string of merge options
    
    Returns:
        Merged file as download
    """
    try:
        if len(files) < 2:
            raise ValueError("At least 2 files are required for merging")
        
        # Read all files
        dataframes = []
        for file in files:
            content = await file.read()
            df = _read_file_content(content, file.filename)
            dataframes.append(df)
        
        # Parse options
        merge_options = {}
        if options:
            merge_options = json.loads(options)
        
        # Perform merge based on type
        if merge_type == "concat":
            result_df = pd.concat(dataframes, **merge_options)
        elif merge_type == "join":
            result_df = dataframes[0]
            for df in dataframes[1:]:
                result_df = result_df.join(df, **merge_options)
        elif merge_type == "merge":
            result_df = dataframes[0]
            for df in dataframes[1:]:
                result_df = pd.merge(result_df, df, **merge_options)
        else:
            raise ValueError(f"Unsupported merge type: {merge_type}")
        
        # Return as CSV
        output_buffer = io.BytesIO()
        result_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output_buffer.read()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=merged_data.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File merging failed: {str(e)}")

@router.post("/cache/set")
async def cache_set(request: CacheRequest) -> Dict[str, Any]:
    """
    Store data in cache.
    
    Args:
        request: Cache set request
    
    Returns:
        Cache operation result
    """
    try:
        cache_manager = MemoryCache()
        
        # Store data in cache
        success = cache_manager.set(
            key=request.key,
            value=request.data,
            ttl=request.ttl
        )
        
        return {
            "operation": "set",
            "key": request.key,
            "success": success,
            "ttl": request.ttl
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cache set failed: {str(e)}")

@router.get("/cache/get/{key}")
async def cache_get(key: str) -> Dict[str, Any]:
    """
    Retrieve data from cache.
    
    Args:
        key: Cache key to retrieve
    
    Returns:
        Cached data or None if not found
    """
    try:
        cache_manager = MemoryCache()
        
        # Get data from cache
        data = cache_manager.get(key)
        
        return {
            "operation": "get",
            "key": key,
            "data": data,
            "found": data is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cache get failed: {str(e)}")

@router.delete("/cache/delete/{key}")
async def cache_delete(key: str) -> Dict[str, Any]:
    """
    Delete data from cache.
    
    Args:
        key: Cache key to delete
    
    Returns:
        Cache operation result
    """
    try:
        cache_manager = MemoryCache()
        
        # Delete from cache
        success = cache_manager.delete(key)
        
        return {
            "operation": "delete",
            "key": key,
            "success": success
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cache delete failed: {str(e)}")

@router.get("/cache/stats")
async def cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Cache statistics and metrics
    """
    try:
        cache_manager = MemoryCache()
        
        # Get cache stats
        stats = cache_manager.get_stats()
        
        return {
            "cache_stats": stats,
            "cache_type": type(cache_manager).__name__
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cache stats failed: {str(e)}")

@router.post("/files/compress")
async def compress_files(files: List[UploadFile] = File(...)) -> StreamingResponse:
    """
    Compress multiple files into a ZIP archive.
    
    Args:
        files: List of files to compress
    
    Returns:
        ZIP archive as download
    """
    try:
        # Create temporary ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                content = await file.read()
                zip_file.writestr(file.filename, content)
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=compressed_files.zip"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File compression failed: {str(e)}")

@router.get("/formats/supported")
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get list of supported file formats and operations.
    
    Returns:
        Dictionary of supported formats and operations
    """
    return {
        "input_formats": {
            "csv": "Comma-separated values",
            "json": "JavaScript Object Notation",
            "excel": "Microsoft Excel (.xlsx, .xls)",
            "parquet": "Apache Parquet",
            "txt": "Plain text (will attempt CSV parsing)"
        },
        "output_formats": {
            "csv": "Comma-separated values",
            "json": "JavaScript Object Notation", 
            "excel": "Microsoft Excel (.xlsx)",
            "parquet": "Apache Parquet"
        },
        "operations": {
            "convert": "Convert between file formats",
            "merge": "Merge multiple files",
            "compress": "Create ZIP archive",
            "analyze": "Get file information and statistics"
        },
        "merge_types": {
            "concat": "Concatenate files vertically",
            "join": "Join files horizontally by index",
            "merge": "Merge files on common columns"
        }
    }

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