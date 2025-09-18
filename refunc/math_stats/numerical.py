"""
Numerical computation utilities.

This module provides numerical integration, differentiation,
interpolation, and special mathematical functions.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, special, linalg
from scipy.optimize import fsolve, root_scalar, brentq

# Numerical derivative implementation (replaces deprecated scipy.misc.derivative)
def derivative(func, x0, dx=1e-5, n=1, order=3, args=()):
    """
    Numerical derivative implementation.
    
    This replaces the deprecated scipy.misc.derivative function.
    """
    if n == 1:
        # First derivative using central difference
        return (func(x0 + dx, *args) - func(x0 - dx, *args)) / (2 * dx)
    elif n == 2:
        # Second derivative
        return (func(x0 + dx, *args) - 2*func(x0, *args) + func(x0 - dx, *args)) / (dx**2)
    else:
        raise NotImplementedError("Higher order derivatives not implemented")

from ..exceptions import RefuncError, ValidationError


class IntegrationMethod(Enum):
    """Numerical integration methods."""
    QUAD = "quad"
    FIXED_QUAD = "fixed_quad"
    QUADRATURE = "quadrature"
    ROMBERG = "romberg"
    SIMPSON = "simpson"
    TRAPEZOID = "trapezoid"
    MONTE_CARLO = "monte_carlo"


class InterpolationMethod(Enum):
    """Interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    RBF = "rbf"
    NEAREST = "nearest"


@dataclass
class IntegrationResult:
    """Results from numerical integration."""
    value: float
    error: Optional[float]
    method: str
    intervals: Optional[int] = None
    function_evaluations: Optional[int] = None
    success: bool = True
    message: str = ""
    
    def summary(self) -> str:
        """Get formatted summary of integration result."""
        lines = [
            f"Integration Result (Method: {self.method})",
            "=" * 40,
            f"Value: {self.value:.10f}",
            f"Success: {self.success}"
        ]
        
        if self.error is not None:
            lines.append(f"Error estimate: {self.error:.2e}")
        if self.intervals is not None:
            lines.append(f"Intervals: {self.intervals}")
        if self.function_evaluations is not None:
            lines.append(f"Function evaluations: {self.function_evaluations}")
        if self.message:
            lines.append(f"Message: {self.message}")
            
        return "\n".join(lines)


@dataclass
class RootFindingResult:
    """Results from root finding."""
    root: float
    function_value: float
    iterations: int
    converged: bool
    method: str
    tolerance: float
    message: str = ""
    
    def summary(self) -> str:
        """Get formatted summary of root finding result."""
        lines = [
            f"Root Finding Result (Method: {self.method})",
            "=" * 40,
            f"Root: {self.root:.10f}",
            f"Function value: {self.function_value:.2e}",
            f"Converged: {self.converged}",
            f"Iterations: {self.iterations}",
            f"Tolerance: {self.tolerance:.2e}"
        ]
        
        if self.message:
            lines.append(f"Message: {self.message}")
            
        return "\n".join(lines)


class NumericalIntegrator:
    """Comprehensive numerical integration engine."""
    
    def __init__(self, default_tolerance: float = 1e-8):
        """
        Initialize numerical integrator.
        
        Args:
            default_tolerance: Default integration tolerance
        """
        self.default_tolerance = default_tolerance
        
    def integrate(
        self,
        func: Callable,
        a: float,
        b: float,
        method: Union[str, IntegrationMethod] = IntegrationMethod.QUAD,
        args: Tuple = (),
        tolerance: Optional[float] = None,
        max_subdivisions: int = 50
    ) -> IntegrationResult:
        """
        Integrate function over interval [a, b].
        
        Args:
            func: Function to integrate
            a: Lower integration limit
            b: Upper integration limit
            method: Integration method
            args: Additional arguments for function
            tolerance: Integration tolerance
            max_subdivisions: Maximum number of subdivisions
            
        Returns:
            IntegrationResult object
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        if isinstance(method, IntegrationMethod):
            method_str = method.value
        else:
            method_str = method
            
        try:
            if method_str == "quad":
                result, error = integrate.quad(
                    func, a, b, args=args, 
                    epsabs=tolerance, limit=max_subdivisions
                )
                return IntegrationResult(
                    value=result,
                    error=error,
                    method=method_str,
                    success=True
                )
                
            elif method_str == "fixed_quad":
                result, _ = integrate.fixed_quad(func, a, b, args=args, n=max_subdivisions)
                return IntegrationResult(
                    value=result,
                    error=None,
                    method=method_str,
                    intervals=max_subdivisions,
                    success=True
                )
                
            elif method_str == "romberg":
                # romberg method was removed in newer scipy, use quad instead
                result, error = integrate.quad(func, a, b, args=args, epsabs=tolerance)
                return IntegrationResult(
                    value=result,
                    error=error,
                    method="quad_romberg_fallback",
                    success=True
                )
                
            elif method_str == "simpson":
                # Use simpson's rule with adaptive subdivision
                n = max_subdivisions
                x = np.linspace(a, b, 2*n + 1)
                y = np.array([func(xi, *args) for xi in x])
                result = integrate.simpson(y, x=x)
                
                return IntegrationResult(
                    value=float(result),
                    error=None,
                    method=method_str,
                    intervals=n,
                    function_evaluations=len(x),
                    success=True
                )
                
            elif method_str == "trapezoid":
                # Use trapezoidal rule
                n = max_subdivisions
                x = np.linspace(a, b, n + 1)
                y = np.array([func(xi, *args) for xi in x])
                result = integrate.trapezoid(y, x)
                
                return IntegrationResult(
                    value=result,
                    error=None,
                    method=method_str,
                    intervals=n,
                    function_evaluations=len(x),
                    success=True
                )
                
            elif method_str == "monte_carlo":
                return self._monte_carlo_integration(func, a, b, args, tolerance, max_subdivisions)
                
            else:
                raise ValidationError(f"Unknown integration method: {method_str}")
                
        except Exception as e:
            return IntegrationResult(
                value=np.nan,
                error=None,
                method=method_str,
                success=False,
                message=str(e)
            )
    
    def integrate_2d(
        self,
        func: Callable,
        x_bounds: Tuple[float, float],
        y_bounds: Union[Tuple[float, float], Callable],
        method: str = "dblquad",
        args: Tuple = (),
        tolerance: Optional[float] = None
    ) -> IntegrationResult:
        """
        Integrate function over 2D region.
        
        Args:
            func: Function f(y, x) to integrate
            x_bounds: (x_min, x_max) bounds
            y_bounds: (y_min, y_max) or callable functions
            method: Integration method
            args: Additional arguments
            tolerance: Integration tolerance
            
        Returns:
            IntegrationResult object
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        try:
            if method == "dblquad":
                if callable(y_bounds):
                    raise ValidationError("Use y_bounds as tuple for dblquad")
                    
                integration_result = integrate.dblquad(
                    func, x_bounds[0], x_bounds[1],
                    y_bounds[0], y_bounds[1],
                    args=args, epsabs=tolerance
                )
                result, error = integration_result[0], integration_result[1]
                
            else:
                raise ValidationError(f"Unknown 2D integration method: {method}")
                
            return IntegrationResult(
                value=result,
                error=error,
                method=method,
                success=True
            )
            
        except Exception as e:
            return IntegrationResult(
                value=np.nan,
                error=None,
                method=method,
                success=False,
                message=str(e)
            )
    
    def _monte_carlo_integration(
        self,
        func: Callable,
        a: float,
        b: float,
        args: Tuple,
        tolerance: float,
        n_samples: int
    ) -> IntegrationResult:
        """Monte Carlo integration."""
        np.random.seed(42)  # For reproducibility
        
        # Generate random samples
        x_samples = np.random.uniform(a, b, n_samples)
        y_samples = np.array([func(x, *args) for x in x_samples])
        
        # Estimate integral
        estimate = (b - a) * np.mean(y_samples)
        
        # Estimate error using standard error
        error = (b - a) * np.std(y_samples) / np.sqrt(n_samples)
        
        return IntegrationResult(
            value=float(estimate),
            error=float(error),
            method="monte_carlo",
            function_evaluations=n_samples,
            success=True
        )


class NumericalDifferentiator:
    """Numerical differentiation utilities."""
    
    def __init__(self, default_dx: float = 1e-5):
        """
        Initialize numerical differentiator.
        
        Args:
            default_dx: Default step size for finite differences
        """
        self.default_dx = default_dx
        
    def derivative(
        self,
        func: Callable,
        x: float,
        order: int = 1,
        dx: Optional[float] = None,
        method: str = "central"
    ) -> float:
        """
        Compute numerical derivative of function at point.
        
        Args:
            func: Function to differentiate
            x: Point to evaluate derivative
            order: Order of derivative (1, 2, 3, ...)
            dx: Step size
            method: Finite difference method
            
        Returns:
            Derivative value
        """
        if dx is None:
            dx = self.default_dx
            
        try:
            if method == "central":
                return derivative(func, x, dx=dx, n=order, order=2*order+1)
            elif method == "forward":
                return derivative(func, x, dx=dx, n=order, order=order+1)
            elif method == "backward":
                return derivative(func, x, dx=dx, n=order, order=order+1)
            else:
                raise ValidationError(f"Unknown differentiation method: {method}")
                
        except Exception as e:
            raise RefuncError(f"Failed to compute derivative: {e}")
    
    def gradient(
        self,
        func: Callable,
        x: np.ndarray,
        dx: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute numerical gradient of multivariate function.
        
        Args:
            func: Function to differentiate
            x: Point to evaluate gradient
            dx: Step size
            
        Returns:
            Gradient vector
        """
        if dx is None:
            dx = self.default_dx
            
        x = np.array(x)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += dx
            x_minus[i] -= dx
            
            grad[i] = (func(x_plus) - func(x_minus)) / (2 * dx)
            
        return grad
    
    def hessian(
        self,
        func: Callable,
        x: np.ndarray,
        dx: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute numerical Hessian matrix.
        
        Args:
            func: Function to differentiate
            x: Point to evaluate Hessian
            dx: Step size
            
        Returns:
            Hessian matrix
        """
        if dx is None:
            dx = self.default_dx
            
        x = np.array(x)
        n = len(x)
        hessian = np.zeros((n, n))
        
        f0 = func(x)
        
        # Diagonal elements (second derivatives)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += dx
            x_minus[i] -= dx
            
            hessian[i, i] = (func(x_plus) - 2*f0 + func(x_minus)) / (dx**2)
            
        # Off-diagonal elements (mixed derivatives)
        for i in range(n):
            for j in range(i+1, n):
                x_pp = x.copy()  # ++
                x_pm = x.copy()  # +-
                x_mp = x.copy()  # -+
                x_mm = x.copy()  # --
                
                x_pp[i] += dx
                x_pp[j] += dx
                x_pm[i] += dx
                x_pm[j] -= dx
                x_mp[i] -= dx
                x_mp[j] += dx
                x_mm[i] -= dx
                x_mm[j] -= dx
                
                mixed_deriv = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * dx**2)
                hessian[i, j] = mixed_deriv
                hessian[j, i] = mixed_deriv
                
        return hessian


class RootFinder:
    """Root finding and equation solving utilities."""
    
    def __init__(self, default_tolerance: float = 1e-8):
        """
        Initialize root finder.
        
        Args:
            default_tolerance: Default convergence tolerance
        """
        self.default_tolerance = default_tolerance
        
    def find_root(
        self,
        func: Callable,
        bracket: Optional[Tuple[float, float]] = None,
        x0: Optional[float] = None,
        method: str = "brentq",
        tolerance: Optional[float] = None,
        max_iterations: int = 100
    ) -> RootFindingResult:
        """
        Find root of function.
        
        Args:
            func: Function to find root of
            bracket: Bracketing interval [a, b] where f(a)*f(b) < 0
            x0: Initial guess (for non-bracketing methods)
            method: Root finding method
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            RootFindingResult object
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        try:
            if method == "brentq":
                if bracket is None:
                    raise ValidationError("Brent method requires bracket")
                
                root_value = brentq(func, bracket[0], bracket[1], xtol=tolerance, maxiter=max_iterations)
                # Extract the root value - brentq returns a scalar
                if isinstance(root_value, tuple):
                    root_scalar = root_value[0]
                else:
                    root_scalar = root_value
                    
                result = RootFindingResult(
                    root=float(root_scalar),
                    function_value=func(root_scalar),
                    iterations=-1,  # brentq doesn't return iteration count
                    converged=True,
                    method=method,
                    tolerance=tolerance
                )
                
            elif method == "newton":
                if x0 is None:
                    raise ValidationError("Newton method requires initial guess")
                
                sol = root_scalar(func, x0=x0, method='newton', xtol=tolerance, maxiter=max_iterations)
                result = RootFindingResult(
                    root=sol.root,
                    function_value=sol.function_calls,
                    iterations=sol.iterations,
                    converged=sol.converged,
                    method=method,
                    tolerance=tolerance,
                    message=sol.flag if hasattr(sol, 'flag') else ""
                )
                
            elif method == "secant":
                if x0 is None:
                    raise ValidationError("Secant method requires initial guess")
                    
                sol = root_scalar(func, x0=x0, method='secant', xtol=tolerance, maxiter=max_iterations)
                result = RootFindingResult(
                    root=sol.root,
                    function_value=func(sol.root),
                    iterations=sol.iterations,
                    converged=sol.converged,
                    method=method,
                    tolerance=tolerance
                )
                
            else:
                raise ValidationError(f"Unknown root finding method: {method}")
                
        except Exception as e:
            result = RootFindingResult(
                root=np.nan,
                function_value=np.nan,
                iterations=0,
                converged=False,
                method=method,
                tolerance=tolerance,
                message=str(e)
            )
            
        return result
    
    def solve_system(
        self,
        equations: Callable,
        x0: np.ndarray,
        tolerance: Optional[float] = None,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, bool, str]:
        """
        Solve system of nonlinear equations.
        
        Args:
            equations: Function returning residuals
            x0: Initial guess
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Tuple of (solution, converged, message)
        """
        if tolerance is None:
            tolerance = self.default_tolerance
            
        try:
            solution_result = fsolve(equations, x0, xtol=tolerance, maxfev=max_iterations)
            # Extract solution array from result
            if isinstance(solution_result, tuple):
                solution = solution_result[0]
            else:
                solution = solution_result
            
            # Ensure it's a numpy array
            solution = np.array(solution)
            residual = equations(solution)
            converged = np.allclose(residual, 0, atol=tolerance)
            
            return solution, converged, "Success" if converged else "Not converged"
            
        except Exception as e:
            return np.full_like(x0, np.nan), False, str(e)


class Interpolator:
    """Interpolation and curve fitting utilities."""
    
    def __init__(self):
        """Initialize interpolator."""
        pass
        
    def interpolate_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: Union[str, InterpolationMethod] = InterpolationMethod.LINEAR,
        **kwargs
    ) -> Callable:
        """
        Create 1D interpolation function.
        
        Args:
            x: X coordinates
            y: Y coordinates
            method: Interpolation method
            **kwargs: Additional method-specific arguments
            
        Returns:
            Interpolation function
        """
        if isinstance(method, InterpolationMethod):
            method_str = method.value
        else:
            method_str = method
            
        x = np.array(x)
        y = np.array(y)
        
        if len(x) != len(y):
            raise ValidationError("x and y must have same length")
            
        try:
            if method_str == "linear":
                return interpolate.interp1d(x, y, kind='linear', **kwargs)
            elif method_str == "cubic":
                return interpolate.interp1d(x, y, kind='cubic', **kwargs)
            elif method_str == "spline":
                degree = kwargs.get('degree', 3)
                smoothing = kwargs.get('smoothing', 0)
                return interpolate.UnivariateSpline(x, y, k=degree, s=smoothing)
            elif method_str == "polynomial":
                degree = kwargs.get('degree', len(x) - 1)
                coeffs = np.polyfit(x, y, degree)
                return np.poly1d(coeffs)
            elif method_str == "nearest":
                return interpolate.interp1d(x, y, kind='nearest', **kwargs)
            else:
                raise ValidationError(f"Unknown interpolation method: {method_str}")
                
        except Exception as e:
            raise RefuncError(f"Interpolation failed: {e}")
    
    def interpolate_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        method: str = "linear"
    ) -> Callable:
        """
        Create 2D interpolation function.
        
        Args:
            x: X coordinates
            y: Y coordinates
            z: Z values
            method: Interpolation method
            
        Returns:
            2D interpolation function
        """
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        
        try:
            if method == "linear":
                return interpolate.LinearNDInterpolator(
                    np.column_stack((x, y)), z
                )
            elif method == "cubic":
                return interpolate.CloughTocher2DInterpolator(
                    np.column_stack((x, y)), z
                )
            elif method == "rbf":
                return interpolate.Rbf(x, y, z)
            else:
                raise ValidationError(f"Unknown 2D interpolation method: {method}")
                
        except Exception as e:
            raise RefuncError(f"2D interpolation failed: {e}")


class SpecialFunctions:
    """Special mathematical functions and utilities."""
    
    @staticmethod
    def gamma_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute gamma function."""
        return special.gamma(x)
    
    @staticmethod
    def beta_function(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute beta function."""
        return special.beta(a, b)
    
    @staticmethod
    def error_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute error function."""
        return special.erf(x)
    
    @staticmethod
    def bessel_j(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Bessel function of the first kind."""
        return special.jv(n, x)
    
    @staticmethod
    def legendre_polynomial(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Legendre polynomial."""
        return special.eval_legendre(n, x)
    
    @staticmethod
    def hermite_polynomial(n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Hermite polynomial."""
        return special.eval_hermite(n, x)


# Convenience functions
def integrate_function(
    func: Callable,
    a: float,
    b: float,
    method: str = "quad",
    **kwargs
) -> IntegrationResult:
    """Integrate function over interval."""
    integrator = NumericalIntegrator()
    return integrator.integrate(func, a, b, method, **kwargs)


def find_function_root(
    func: Callable,
    bracket: Optional[Tuple[float, float]] = None,
    x0: Optional[float] = None,
    method: str = "brentq",
    **kwargs
) -> RootFindingResult:
    """Find root of function."""
    finder = RootFinder()
    return finder.find_root(func, bracket, x0, method, **kwargs)


def create_interpolator(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "linear",
    **kwargs
) -> Callable:
    """Create interpolation function."""
    interpolator = Interpolator()
    return interpolator.interpolate_1d(x, y, method, **kwargs)


def numerical_derivative(
    func: Callable,
    x: float,
    order: int = 1,
    **kwargs
) -> float:
    """Compute numerical derivative."""
    differentiator = NumericalDifferentiator()
    return differentiator.derivative(func, x, order, **kwargs)