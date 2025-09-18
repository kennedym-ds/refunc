"""
Numerical optimization utilities for ML workflows.

This module provides optimization algorithms, objective function utilities,
constraint handling, and optimization result analysis.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import (
    minimize, minimize_scalar, differential_evolution, basinhopping,
    brute, shgo, dual_annealing, OptimizeResult
)
from scipy.optimize import LinearConstraint, NonlinearConstraint, Bounds

from ..exceptions import RefuncError, ValidationError


class OptimizationMethod(Enum):
    """Optimization methods."""
    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    CG = "CG"
    BFGS = "BFGS"
    L_BFGS_B = "L-BFGS-B"
    TNC = "TNC"
    COBYLA = "COBYLA"
    SLSQP = "SLSQP"
    TRUST_CONSTR = "trust-constr"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BASIN_HOPPING = "basinhopping"
    BRUTE_FORCE = "brute"
    SHGO = "shgo"
    DUAL_ANNEALING = "dual_annealing"


class ConstraintType(Enum):
    """Constraint types."""
    EQUALITY = "eq"
    INEQUALITY = "ineq"
    BOUNDS = "bounds"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


@dataclass
class OptimizationBounds:
    """Bounds for optimization variables."""
    lower: List[float]
    upper: List[float]
    
    def __post_init__(self):
        """Validate bounds."""
        if len(self.lower) != len(self.upper):
            raise ValidationError("Lower and upper bounds must have same length")
        
        for i, (lb, ub) in enumerate(zip(self.lower, self.upper)):
            if lb >= ub:
                raise ValidationError(f"Lower bound must be less than upper bound for variable {i}")
    
    @property
    def n_variables(self) -> int:
        """Number of variables."""
        return len(self.lower)
    
    def to_scipy_bounds(self) -> Bounds:
        """Convert to scipy Bounds object."""
        return Bounds(self.lower, self.upper)


@dataclass
class OptimizationConstraint:
    """Optimization constraint definition."""
    type: ConstraintType
    function: Callable
    jacobian: Optional[Callable] = None
    args: Tuple = ()
    kwargs: Dict = None
    bounds: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Initialize constraint."""
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class OptimizationResult:
    """Results from optimization."""
    success: bool
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    method: str
    message: str
    jac: Optional[np.ndarray] = None
    hess: Optional[np.ndarray] = None
    optimization_trace: Optional[List[Dict]] = None
    convergence_info: Optional[Dict] = None
    
    @property
    def summary(self) -> str:
        """Get formatted summary of results."""
        lines = [
            f"Optimization Result (Method: {self.method})",
            "=" * 50,
            f"Success: {self.success}",
            f"Optimal value: {self.fun:.8f}",
            f"Optimal point: {self.x}",
            f"Iterations: {self.nit}",
            f"Function evaluations: {self.nfev}",
            f"Message: {self.message}"
        ]
        
        if self.convergence_info:
            lines.append("\nConvergence Information:")
            for key, value in self.convergence_info.items():
                lines.append(f"  {key}: {value}")
                
        return "\n".join(lines)


class ObjectiveFunction:
    """Wrapper for objective functions with additional features."""
    
    def __init__(
        self,
        func: Callable,
        gradient: Optional[Callable] = None,
        hessian: Optional[Callable] = None,
        args: Tuple = (),
        kwargs: Dict = None
    ):
        """
        Initialize objective function.
        
        Args:
            func: Objective function to minimize
            gradient: Gradient function (optional)
            hessian: Hessian function (optional)
            args: Additional arguments for function
            kwargs: Additional keyword arguments for function
        """
        self.func = func
        self.gradient = gradient
        self.hessian = hessian
        self.args = args
        self.kwargs = kwargs or {}
        
        # Tracking
        self.call_count = 0
        self.gradient_count = 0
        self.hessian_count = 0
        self.evaluation_history = []
        
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate objective function."""
        try:
            result = self.func(x, *self.args, **self.kwargs)
            self.call_count += 1
            self.evaluation_history.append({
                'x': x.copy(),
                'f': result,
                'iteration': self.call_count
            })
            return result
        except Exception as e:
            raise RefuncError(f"Error evaluating objective function: {e}")
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient."""
        if self.gradient is None:
            raise ValidationError("Gradient function not provided")
        
        try:
            result = self.gradient(x, *self.args, **self.kwargs)
            self.gradient_count += 1
            return result
        except Exception as e:
            raise RefuncError(f"Error evaluating gradient: {e}")
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Hessian."""
        if self.hessian is None:
            raise ValidationError("Hessian function not provided")
        
        try:
            result = self.hessian(x, *self.args, **self.kwargs)
            self.hessian_count += 1
            return result
        except Exception as e:
            raise RefuncError(f"Error evaluating Hessian: {e}")
    
    def reset_counters(self):
        """Reset evaluation counters."""
        self.call_count = 0
        self.gradient_count = 0
        self.hessian_count = 0
        self.evaluation_history = []


class Optimizer:
    """Comprehensive optimization engine."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize optimizer.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self._optimization_history = []
        
    def minimize(
        self,
        objective: Union[Callable, ObjectiveFunction],
        x0: Union[List, np.ndarray],
        method: Union[str, OptimizationMethod] = OptimizationMethod.BFGS,
        bounds: Optional[OptimizationBounds] = None,
        constraints: List[OptimizationConstraint] = None,
        options: Dict = None,
        callback: Callable = None
    ) -> OptimizationResult:
        """
        Minimize objective function.
        
        Args:
            objective: Objective function to minimize
            x0: Initial guess
            method: Optimization method
            bounds: Variable bounds
            constraints: Optimization constraints
            options: Method-specific options
            callback: Callback function
            
        Returns:
            OptimizationResult object
        """
        x0 = np.array(x0)
        
        if isinstance(method, OptimizationMethod):
            method_str = method.value
        else:
            method_str = method
            
        if options is None:
            options = {}
            
        # Prepare objective function
        if isinstance(objective, ObjectiveFunction):
            obj_func = objective
        else:
            obj_func = ObjectiveFunction(objective)
            
        # Reset counters
        obj_func.reset_counters()
        
        # Prepare constraints for scipy
        scipy_constraints = []
        if constraints:
            for constraint in constraints:
                if constraint.type == ConstraintType.EQUALITY:
                    scipy_constraints.append({
                        'type': 'eq',
                        'fun': constraint.function,
                        'jac': constraint.jacobian,
                        'args': constraint.args
                    })
                elif constraint.type == ConstraintType.INEQUALITY:
                    scipy_constraints.append({
                        'type': 'ineq',
                        'fun': constraint.function,
                        'jac': constraint.jacobian,
                        'args': constraint.args
                    })
                    
        # Prepare bounds
        scipy_bounds = bounds.to_scipy_bounds() if bounds else None
        
        # Choose optimization method
        try:
            if method_str in ["differential_evolution", "basinhopping", "brute", "shgo", "dual_annealing"]:
                result = self._global_optimize(
                    obj_func, x0, method_str, scipy_bounds, 
                    scipy_constraints, options, callback
                )
            else:
                # Check if method supports bounds
                if scipy_bounds and method_str in ["BFGS", "CG", "Newton-CG"]:
                    # Switch to L-BFGS-B which supports bounds
                    import warnings
                    warnings.warn(f"Method {method_str} cannot handle bounds. Switching to L-BFGS-B.", RuntimeWarning)
                    method_str = "L-BFGS-B"
                
                result = minimize(
                    fun=obj_func,
                    x0=x0,
                    method=method_str,
                    jac=obj_func.grad if obj_func.gradient else None,
                    hess=obj_func.hess if obj_func.hessian else None,
                    bounds=scipy_bounds,
                    constraints=scipy_constraints,
                    options=options,
                    callback=callback
                )
                
        except Exception as e:
            raise RefuncError(f"Optimization failed: {e}")
        
        # Create result object
        optimization_result = OptimizationResult(
            success=result.success,
            x=result.x,
            fun=result.fun,
            nit=result.nit if hasattr(result, 'nit') else -1,
            nfev=result.nfev if hasattr(result, 'nfev') else obj_func.call_count,
            method=method_str,
            message=result.message,
            jac=result.jac if hasattr(result, 'jac') else None,
            hess=result.hess_inv if hasattr(result, 'hess_inv') else None,
            optimization_trace=obj_func.evaluation_history,
            convergence_info=self._analyze_convergence(obj_func.evaluation_history)
        )
        
        self._optimization_history.append(optimization_result)
        
        if self.verbose:
            print(optimization_result.summary)
            
        return optimization_result
    
    def minimize_scalar(
        self,
        objective: Callable,
        bounds: Optional[Tuple[float, float]] = None,
        method: str = "brent",
        options: Dict = None
    ) -> OptimizationResult:
        """
        Minimize scalar objective function.
        
        Args:
            objective: Scalar objective function
            bounds: Bounds for optimization
            method: Scalar optimization method
            options: Method-specific options
            
        Returns:
            OptimizationResult object
        """
        if options is None:
            options = {}
        
        # Choose appropriate method based on bounds
        if bounds is not None and method == "brent":
            method = "bounded"  # Use bounded method when bounds are provided
            
        try:
            result = minimize_scalar(
                fun=objective,
                bounds=bounds,
                method=method,
                options=options
            )
        except Exception as e:
            raise RefuncError(f"Scalar optimization failed: {e}")
            
        return OptimizationResult(
            success=result.success,
            x=np.array([result.x]),
            fun=result.fun,
            nit=result.nit if hasattr(result, 'nit') else -1,
            nfev=result.nfev if hasattr(result, 'nfev') else -1,
            method=method,
            message=result.message if hasattr(result, 'message') else "Success"
        )
    
    def multi_start_optimization(
        self,
        objective: Union[Callable, ObjectiveFunction],
        bounds: OptimizationBounds,
        n_starts: int = 10,
        method: Union[str, OptimizationMethod] = OptimizationMethod.L_BFGS_B,
        random_state: int = None
    ) -> List[OptimizationResult]:
        """
        Perform multi-start optimization.
        
        Args:
            objective: Objective function
            bounds: Variable bounds
            n_starts: Number of random starting points
            method: Optimization method
            random_state: Random seed
            
        Returns:
            List of optimization results
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        results = []
        
        for i in range(n_starts):
            # Generate random starting point
            x0 = np.random.uniform(
                low=bounds.lower,
                high=bounds.upper,
                size=bounds.n_variables
            )
            
            try:
                result = self.minimize(
                    objective=objective,
                    x0=x0,
                    method=method,
                    bounds=bounds
                )
                results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Start {i+1} failed: {e}")
                continue
                
        if not results:
            raise RefuncError("All optimization starts failed")
            
        # Sort by objective value
        results.sort(key=lambda x: x.fun)
        
        return results
    
    def parameter_sweep(
        self,
        objective: Callable,
        parameter_ranges: Dict[str, np.ndarray],
        fixed_params: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform parameter sweep optimization.
        
        Args:
            objective: Objective function
            parameter_ranges: Dictionary of parameter ranges
            fixed_params: Fixed parameter values
            
        Returns:
            Tuple of (parameter_combinations, objective_values)
        """
        if fixed_params is None:
            fixed_params = {}
            
        # Create parameter grid
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        from itertools import product
        param_combinations = list(product(*param_values))
        
        objective_values = []
        
        for combination in param_combinations:
            params = dict(zip(param_names, combination))
            params.update(fixed_params)
            
            try:
                # Convert to array format expected by objective
                x = np.array(list(combination))
                value = objective(x)
                objective_values.append(value)
            except Exception as e:
                objective_values.append(np.inf)
                if self.verbose:
                    print(f"Evaluation failed for {params}: {e}")
                    
        return np.array(param_combinations), np.array(objective_values)
    
    def _global_optimize(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        method: str,
        bounds: Optional[Bounds],
        constraints: List,
        options: Dict,
        callback: Callable
    ) -> OptimizeResult:
        """Handle global optimization methods."""
        if method == "differential_evolution":
            if bounds is None:
                raise ValidationError("Differential evolution requires bounds")
            return differential_evolution(
                func=objective,
                bounds=list(zip(bounds.lb, bounds.ub)),
                **options,
                callback=callback
            )
            
        elif method == "basinhopping":
            return basinhopping(
                func=objective,
                x0=x0,
                **options,
                callback=callback
            )
            
        elif method == "dual_annealing":
            if bounds is None:
                raise ValidationError("Dual annealing requires bounds")
            return dual_annealing(
                func=objective,
                bounds=list(zip(bounds.lb, bounds.ub)),
                **options,
                callback=callback
            )
            
        elif method == "shgo":
            if bounds is None:
                raise ValidationError("SHGO requires bounds")
            return shgo(
                func=objective,
                bounds=list(zip(bounds.lb, bounds.ub)),
                constraints=constraints,
                **options,
                callback=callback
            )
            
        else:
            raise ValidationError(f"Unknown global optimization method: {method}")
    
    def _analyze_convergence(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        if not history:
            return {}
            
        objective_values = [entry['f'] for entry in history]
        
        convergence_info = {
            'initial_value': objective_values[0],
            'final_value': objective_values[-1],
            'improvement': objective_values[0] - objective_values[-1],
            'improvement_ratio': (objective_values[0] - objective_values[-1]) / abs(objective_values[0]) if objective_values[0] != 0 else 0,
            'n_evaluations': len(objective_values)
        }
        
        # Check for convergence plateaus
        if len(objective_values) > 10:
            recent_values = objective_values[-10:]
            convergence_info['recent_std'] = np.std(recent_values)
            convergence_info['converged'] = convergence_info['recent_std'] < 1e-8
            
        return convergence_info


# Convenience functions
def minimize_function(
    func: Callable,
    x0: Union[List, np.ndarray],
    method: str = "BFGS",
    **kwargs
) -> OptimizationResult:
    """Minimize a function using specified method."""
    optimizer = Optimizer()
    return optimizer.minimize(func, x0, method, **kwargs)


def find_minimum_scalar(
    func: Callable,
    bounds: Tuple[float, float] = None,
    method: str = "brent"
) -> OptimizationResult:
    """Find minimum of scalar function."""
    optimizer = Optimizer()
    return optimizer.minimize_scalar(func, bounds, method)


def global_minimize(
    func: Callable,
    bounds: List[Tuple[float, float]],
    method: str = "differential_evolution",
    **kwargs
) -> OptimizationResult:
    """Global optimization with bounds."""
    optimizer = Optimizer()
    bounds_obj = OptimizationBounds(
        lower=[b[0] for b in bounds],
        upper=[b[1] for b in bounds]
    )
    return optimizer.minimize(func, np.mean(bounds, axis=1), method, bounds_obj, **kwargs)


def multi_start_minimize(
    func: Callable,
    bounds: List[Tuple[float, float]],
    n_starts: int = 10,
    **kwargs
) -> List[OptimizationResult]:
    """Multi-start optimization."""
    optimizer = Optimizer()
    bounds_obj = OptimizationBounds(
        lower=[b[0] for b in bounds],
        upper=[b[1] for b in bounds]
    )
    return optimizer.multi_start_optimization(func, bounds_obj, n_starts, **kwargs)