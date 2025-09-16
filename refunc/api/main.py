"""
Main FastAPI application for refunc REST API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any

from .routers import data_science, math_stats, ml_ops, utils
from .models.responses import HealthResponse, ErrorResponse

# Create FastAPI app
app = FastAPI(
    title="Refunc REST API",
    description="A comprehensive REST API for ML utilities and data science operations",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_science.router, prefix="/api/v1/data-science", tags=["Data Science"])
app.include_router(math_stats.router, prefix="/api/v1/math-stats", tags=["Math & Statistics"])
app.include_router(ml_ops.router, prefix="/api/v1/ml", tags=["Machine Learning"])
app.include_router(utils.router, prefix="/api/v1/utils", tags=["Utilities"])

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint - API health check."""
    return HealthResponse(
        status="healthy",
        message="Refunc REST API is running",
        version="0.1.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="All systems operational",
        version="0.1.0"
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="ValueError",
            message=str(exc),
            status_code=400
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            status_code=500
        ).dict()
    )

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the FastAPI server."""
    uvicorn.run(
        "refunc.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()