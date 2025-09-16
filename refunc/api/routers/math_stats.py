"""
Math & Statistics operations router.

Provides endpoints for:
- Numerical computations
- Root finding
- Statistical analysis
- Optimization
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from pydantic import BaseModel

from ...math_stats.numerical import RootFinder
from ...math_stats.statistics import StatisticsEngine
from ...math_stats.optimization import Optimizer
from ..models.responses import RootFindingResultResponse

router = APIRouter()

class RootFindingRequest(BaseModel):
    """Request model for root finding operations."""
    function_str: str  # Function as a string, e.g., "x**2 - 4"
    bracket: Optional[List[float]] = None  # [min, max] for bracketed methods
    x0: Optional[float] = None  # Initial guess
    method: str = "brentq"
    tolerance: float = 1e-8
    max_iterations: int = 100

class StatisticsRequest(BaseModel):
    """Request model for statistical operations."""
    data: List[float]
    operation: str  # "descriptive", "normality_test", "outliers", etc.
    options: Optional[Dict[str, Any]] = None

class OptimizationRequest(BaseModel):
    """Request model for optimization operations."""
    function_str: str  # Objective function as string
    x0: List[float]  # Initial guess
    method: str = "BFGS"
    bounds: Optional[List[List[float]]] = None  # [[min1, max1], [min2, max2], ...]
    constraints: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None

@router.post("/root-finding", response_model=RootFindingResultResponse)
async def find_root(request: RootFindingRequest) -> RootFindingResultResponse:
    """
    Find roots of mathematical functions.
    
    Args:
        request: Root finding configuration
    
    Returns:
        Root finding result with convergence information
    """
    try:
        # Create function from string
        func = _string_to_function(request.function_str)
        
        # Create root finder and find root
        root_finder = RootFinder(default_tolerance=request.tolerance)
        result = root_finder.find_root(
            func=func,
            bracket=tuple(request.bracket) if request.bracket else None,
            x0=request.x0,
            method=request.method,
            tolerance=request.tolerance,
            max_iterations=request.max_iterations
        )
        
        # Convert to response model
        return RootFindingResultResponse(
            root=result.root,
            function_value=result.function_value,
            iterations=result.iterations,
            converged=result.converged,
            method=result.method,
            tolerance=result.tolerance,
            message=result.message
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Root finding failed: {str(e)}")

@router.post("/statistics")
async def compute_statistics(request: StatisticsRequest) -> Dict[str, Any]:
    """
    Perform statistical analysis on data.
    
    Args:
        request: Statistical analysis configuration
    
    Returns:
        Statistical analysis results
    """
    try:
        data = np.array(request.data)
        analyzer = StatisticsEngine()
        options = request.options or {}
        
        if request.operation == "descriptive":
            result = analyzer.compute_descriptive_stats(data)
            return {
                "operation": "descriptive",
                "results": result._asdict() if hasattr(result, '_asdict') else dict(result)
            }
        
        elif request.operation == "normality_test":
            result = analyzer.normality_test(data, **options)
            return {
                "operation": "normality_test",
                "results": result._asdict() if hasattr(result, '_asdict') else dict(result)
            }
        
        elif request.operation == "outliers":
            outliers, clean_data = analyzer.outlier_detection(data, **options)
            return {
                "operation": "outliers",
                "results": {
                    "outliers": outliers.tolist(),
                    "clean_data": clean_data.tolist(),
                    "num_outliers": len(outliers)
                }
            }
        
        elif request.operation == "correlation":
            if len(data.shape) == 1:
                raise ValueError("Correlation requires 2D data")
            result = analyzer.correlation_matrix(data, **options)
            return {
                "operation": "correlation",
                "results": result
            }
        
        else:
            raise ValueError(f"Unknown operation: {request.operation}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Statistical analysis failed: {str(e)}")

@router.post("/optimize")
async def optimize_function(request: OptimizationRequest) -> Dict[str, Any]:
    """
    Optimize mathematical functions.
    
    Args:
        request: Optimization configuration
    
    Returns:
        Optimization results
    """
    try:
        # Create function from string
        func = _string_to_function(request.function_str, multivariate=True)
        
        # Create optimizer
        optimizer = Optimizer()
        options = request.options or {}
        
        # Convert bounds format if provided
        bounds = None
        if request.bounds:
            bounds = [(b[0], b[1]) for b in request.bounds]
        
        # Run optimization
        result = optimizer.minimize(
            func=func,
            x0=np.array(request.x0),
            method=request.method,
            bounds=bounds,
            constraints=request.constraints,
            **options
        )
        
        return {
            "success": result.success,
            "x": result.x.tolist(),
            "fun": float(result.fun),
            "nit": int(result.nit),
            "message": result.message,
            "method": request.method
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization failed: {str(e)}")

@router.post("/solve-system")
async def solve_system(
    equations: List[str],
    x0: List[float],
    tolerance: float = 1e-8,
    max_iterations: int = 100
) -> Dict[str, Any]:
    """
    Solve system of nonlinear equations.
    
    Args:
        equations: List of equation strings
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
    
    Returns:
        System solution
    """
    try:
        # Create system function
        def system_func(x):
            results = []
            for eq_str in equations:
                # Replace variables with actual values
                expr = eq_str
                for i, val in enumerate(x):
                    expr = expr.replace(f'x{i}', str(val))
                    expr = expr.replace(f'x[{i}]', str(val))
                results.append(eval(expr))
            return np.array(results)
        
        # Solve system
        root_finder = RootFinder()
        solution, converged, message = root_finder.solve_system(
            equations=system_func,
            x0=np.array(x0),
            tolerance=tolerance,
            max_iterations=max_iterations
        )
        
        return {
            "solution": solution.tolist(),
            "converged": converged,
            "message": message,
            "tolerance": tolerance
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"System solving failed: {str(e)}")

@router.get("/functions/examples")
async def get_function_examples() -> Dict[str, Any]:
    """
    Get examples of function strings for different operations.
    
    Returns:
        Dictionary of example functions
    """
    return {
        "root_finding": {
            "quadratic": "x**2 - 4",
            "trigonometric": "np.sin(x) - 0.5",
            "exponential": "np.exp(x) - 2",
            "polynomial": "x**3 - 2*x - 5"
        },
        "optimization": {
            "quadratic_2d": "x[0]**2 + x[1]**2",
            "rosenbrock": "100*(x[1] - x[0]**2)**2 + (1 - x[0])**2",
            "sphere": "sum(x**2)",
            "rastrigin": "10*len(x) + sum(x**2 - 10*np.cos(2*np.pi*x))"
        },
        "system_equations": {
            "simple": ["x[0]**2 + x[1]**2 - 1", "x[0] - x[1]"],
            "nonlinear": ["x[0] + x[1] - 3", "x[0]**2 + x[1]**2 - 9"]
        },
        "notes": {
            "variables": "Use 'x' for single variable, 'x[0], x[1], ...' for multiple variables",
            "functions": "Available: np.sin, np.cos, np.exp, np.log, np.sqrt, etc.",
            "operators": "Standard: +, -, *, /, **, %"
        }
    }

def _string_to_function(func_str: str, multivariate: bool = False) -> Callable:
    """Convert string representation to callable function."""
    import numpy as np
    
    # Security check - only allow safe mathematical operations
    allowed_names = {
        "__builtins__": {},
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": abs,
        "sum": sum,
        "len": len,
        "pow": pow,
        "pi": np.pi,
        "e": np.e
    }
    
    if multivariate:
        def func(x):
            return eval(func_str, allowed_names, {"x": x})
    else:
        def func(x):
            return eval(func_str, allowed_names, {"x": x})
    
    return func