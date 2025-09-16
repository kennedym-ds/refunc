"""
Distribution analysis and fitting utilities.

This module provides tools for fitting distributions to data,
generating samples, and performing distribution-based analysis.
"""

import warnings
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma, factorial

from ..exceptions import RefuncError, ValidationError


class DistributionFamily(Enum):
    """Common distribution families."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"


@dataclass
class DistributionFit:
    """Results from distribution fitting."""
    distribution_name: str
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    goodness_of_fit: Dict[str, float]
    sample_size: int
    fitted_distribution: Any
    
    @property
    def parameter_summary(self) -> str:
        """Get formatted parameter summary."""
        params = ", ".join([f"{k}={v:.4f}" for k, v in self.parameters.items()])
        return f"{self.distribution_name}({params})"
    
    def summary(self) -> str:
        """Get formatted summary of fit results."""
        lines = [
            f"Distribution: {self.distribution_name}",
            f"Parameters: {self.parameter_summary}",
            f"Log-likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            f"Sample size: {self.sample_size}"
        ]
        
        if self.goodness_of_fit:
            lines.append("Goodness of fit:")
            for test, value in self.goodness_of_fit.items():
                lines.append(f"  {test}: {value:.4f}")
                
        return "\n".join(lines)


@dataclass
class DistributionComparison:
    """Results from comparing multiple distribution fits."""
    fits: List[DistributionFit]
    best_fit: DistributionFit
    ranking_method: str
    
    def summary(self) -> str:
        """Get formatted summary of comparison."""
        lines = [
            f"Distribution Comparison (ranked by {self.ranking_method})",
            "=" * 50,
            f"Best fit: {self.best_fit.parameter_summary}"
        ]
        
        lines.append("\nAll fits ranked:")
        for i, fit in enumerate(self.fits, 1):
            criterion_value = getattr(fit, self.ranking_method.lower())
            lines.append(f"{i}. {fit.parameter_summary} ({self.ranking_method.upper()}: {criterion_value:.4f})")
            
        return "\n".join(lines)


class DistributionAnalyzer:
    """Comprehensive distribution analysis and fitting."""
    
    # Common continuous distributions
    CONTINUOUS_DISTRIBUTIONS = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'exponential': stats.expon,
        'gamma': stats.gamma,
        'beta': stats.beta,
        'uniform': stats.uniform,
        'weibull': stats.weibull_min,
        'chi2': stats.chi2,
        't': stats.t,
        'f': stats.f,
        'pareto': stats.pareto,
        'logistic': stats.logistic,
        'laplace': stats.laplace,
        'cauchy': stats.cauchy,
        'rayleigh': stats.rayleigh,
        'gumbel': stats.gumbel_r
    }
    
    # Common discrete distributions
    DISCRETE_DISTRIBUTIONS = {
        'poisson': stats.poisson,
        'binomial': stats.binom,
        'negative_binomial': stats.nbinom,
        'geometric': stats.geom,
        'hypergeometric': stats.hypergeom,
        'multinomial': stats.multinomial
    }
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize distribution analyzer.
        
        Args:
            confidence_level: Confidence level for intervals
        """
        self.confidence_level = confidence_level
        
    def fit_distribution(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distribution: Union[str, stats.rv_continuous, stats.rv_discrete],
        method: str = "mle"
    ) -> DistributionFit:
        """
        Fit a single distribution to data.
        
        Args:
            data: Input data
            distribution: Distribution name or scipy.stats distribution
            method: Fitting method (mle, moments, lse)
            
        Returns:
            DistributionFit object
        """
        data = np.array(data)
        if len(data) == 0:
            raise ValidationError("Cannot fit distribution to empty data")
            
        # Get distribution object
        if isinstance(distribution, str):
            if distribution in self.CONTINUOUS_DISTRIBUTIONS:
                dist = self.CONTINUOUS_DISTRIBUTIONS[distribution]
                dist_name = distribution
            elif distribution in self.DISCRETE_DISTRIBUTIONS:
                dist = self.DISCRETE_DISTRIBUTIONS[distribution]
                dist_name = distribution
            else:
                raise ValidationError(f"Unknown distribution: {distribution}")
        else:
            dist = distribution
            dist_name = distribution.name if hasattr(distribution, 'name') else str(distribution)
            
        # Fit parameters
        try:
            if method == "mle":
                params = dist.fit(data)
            elif method == "moments":
                # Method of moments (simplified implementation)
                params = self._fit_method_of_moments(data, dist)
            else:
                raise ValidationError(f"Unknown fitting method: {method}")
                
        except Exception as e:
            raise RefuncError(f"Failed to fit {dist_name} distribution: {e}")
            
        # Calculate goodness of fit metrics
        log_likelihood = np.sum(dist.logpdf(data, *params) if hasattr(dist, 'logpdf') 
                               else dist.logpmf(data, *params))
        
        # Handle potential infinite log-likelihood
        if not np.isfinite(log_likelihood):
            log_likelihood = -np.inf
            
        n_params = len(params)
        n_samples = len(data)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_samples) - 2 * log_likelihood
        
        # Goodness of fit tests
        goodness_of_fit = self._calculate_goodness_of_fit(data, dist, params)
        
        # Create parameter dictionary
        if hasattr(dist, 'shapes') and dist.shapes:
            shape_names = dist.shapes.split(',')
            param_dict = {name.strip(): param for name, param in zip(shape_names, params[:-2])}
            param_dict['loc'] = params[-2]
            param_dict['scale'] = params[-1]
        else:
            if len(params) == 2:
                param_dict = {'loc': params[0], 'scale': params[1]}
            elif len(params) == 1:
                param_dict = {'param': params[0]}
            else:
                param_dict = {f'param_{i}': param for i, param in enumerate(params)}
                
        return DistributionFit(
            distribution_name=dist_name,
            parameters=param_dict,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            goodness_of_fit=goodness_of_fit,
            sample_size=n_samples,
            fitted_distribution=dist(*params)
        )
    
    def fit_best_distribution(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distributions: List[str] = None,
        ranking_method: str = "aic"
    ) -> DistributionComparison:
        """
        Fit multiple distributions and find the best one.
        
        Args:
            data: Input data
            distributions: List of distribution names to try
            ranking_method: Criterion for ranking (aic, bic, log_likelihood)
            
        Returns:
            DistributionComparison object
        """
        if distributions is None:
            # Determine if data is likely discrete or continuous
            data_array = np.array(data)
            if np.allclose(data_array, np.round(data_array)) and np.all(data_array >= 0):
                # Likely discrete
                distributions = ['poisson', 'binomial', 'negative_binomial']
            else:
                # Likely continuous
                distributions = ['normal', 'lognormal', 'exponential', 'gamma', 'weibull']
                
        fits = []
        failed_fits = []
        
        for dist_name in distributions:
            try:
                fit = self.fit_distribution(data, dist_name)
                fits.append(fit)
            except Exception as e:
                failed_fits.append((dist_name, str(e)))
                warnings.warn(f"Failed to fit {dist_name}: {e}")
                
        if not fits:
            raise RefuncError("Failed to fit any distributions")
            
        # Rank fits
        reverse = ranking_method == "log_likelihood"  # Higher is better for log-likelihood
        fits.sort(key=lambda x: getattr(x, ranking_method), reverse=reverse)
        best_fit = fits[0]
        
        return DistributionComparison(
            fits=fits,
            best_fit=best_fit,
            ranking_method=ranking_method
        )
    
    def generate_samples(
        self,
        distribution_fit: DistributionFit,
        n_samples: int,
        random_state: int = None
    ) -> np.ndarray:
        """
        Generate samples from fitted distribution.
        
        Args:
            distribution_fit: Fitted distribution
            n_samples: Number of samples to generate
            random_state: Random seed
            
        Returns:
            Generated samples
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        return distribution_fit.fitted_distribution.rvs(size=n_samples)
    
    def calculate_quantiles(
        self,
        distribution_fit: DistributionFit,
        quantiles: List[float]
    ) -> Dict[float, float]:
        """
        Calculate quantiles for fitted distribution.
        
        Args:
            distribution_fit: Fitted distribution
            quantiles: List of quantile values (0-1)
            
        Returns:
            Dictionary mapping quantiles to values
        """
        return {q: distribution_fit.fitted_distribution.ppf(q) for q in quantiles}
    
    def calculate_probability(
        self,
        distribution_fit: DistributionFit,
        value: float,
        is_cumulative: bool = True
    ) -> float:
        """
        Calculate probability for a value.
        
        Args:
            distribution_fit: Fitted distribution
            value: Value to calculate probability for
            is_cumulative: Whether to return cumulative probability
            
        Returns:
            Probability value
        """
        if is_cumulative:
            return distribution_fit.fitted_distribution.cdf(value)
        else:
            if hasattr(distribution_fit.fitted_distribution, 'pdf'):
                return distribution_fit.fitted_distribution.pdf(value)
            else:
                return distribution_fit.fitted_distribution.pmf(value)
    
    def _fit_method_of_moments(
        self,
        data: np.ndarray,
        distribution: stats.rv_continuous
    ) -> Tuple[float, ...]:
        """Fit distribution using method of moments (simplified)."""
        # This is a simplified implementation
        # For most distributions, fall back to MLE
        return distribution.fit(data)
    
    def _calculate_goodness_of_fit(
        self,
        data: np.ndarray,
        distribution: Union[stats.rv_continuous, stats.rv_discrete],
        params: Tuple[float, ...]
    ) -> Dict[str, float]:
        """Calculate goodness of fit statistics."""
        goodness = {}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, lambda x: distribution.cdf(x, *params))
            goodness['ks_statistic'] = ks_stat
            goodness['ks_p_value'] = ks_p
            
            # Anderson-Darling test (for normal distribution)
            if distribution == stats.norm:
                ad_stat, ad_crit, ad_sig = stats.anderson(data, dist='norm')
                goodness['anderson_darling'] = ad_stat
                
        except Exception as e:
            warnings.warn(f"Could not calculate some goodness of fit metrics: {e}")
            
        return goodness
    
    def probability_plot(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distribution_fit: DistributionFit
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate data for probability plot (Q-Q plot).
        
        Args:
            data: Observed data
            distribution_fit: Fitted distribution
            
        Returns:
            Tuple of (theoretical_quantiles, sample_quantiles, correlation)
        """
        data = np.array(data)
        n = len(data)
        
        # Calculate plotting positions
        plotting_positions = (np.arange(1, n + 1) - 0.5) / n
        
        # Theoretical quantiles
        theoretical_quantiles = distribution_fit.fitted_distribution.ppf(plotting_positions)
        
        # Sample quantiles (sorted data)
        sample_quantiles = np.sort(data)
        
        # Calculate correlation
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        
        return theoretical_quantiles, sample_quantiles, correlation


class BayesianDistributionAnalyzer:
    """Bayesian approach to distribution analysis."""
    
    def __init__(self, n_samples: int = 10000):
        """
        Initialize Bayesian analyzer.
        
        Args:
            n_samples: Number of MCMC samples
        """
        self.n_samples = n_samples
        
    def bayesian_model_comparison(
        self,
        data: Union[List, np.ndarray, pd.Series],
        distributions: List[str],
        prior_weights: List[float] = None
    ) -> Dict[str, float]:
        """
        Compare distributions using Bayesian model comparison.
        
        Args:
            data: Input data
            distributions: List of distribution names
            prior_weights: Prior model probabilities
            
        Returns:
            Dictionary of posterior model probabilities
        """
        # Simplified implementation using AIC/BIC approximation
        analyzer = DistributionAnalyzer()
        
        if prior_weights is None:
            prior_weights = [1.0 / len(distributions)] * len(distributions)
            
        log_evidences = []
        successful_distributions = []
        successful_priors = []
        
        for i, dist_name in enumerate(distributions):
            try:
                fit = analyzer.fit_distribution(data, dist_name)
                # Approximate log evidence using BIC
                log_evidence = -0.5 * fit.bic
                log_evidences.append(log_evidence)
                successful_distributions.append(dist_name)
                successful_priors.append(prior_weights[i])
            except Exception:
                continue
                
        if not log_evidences:
            raise RefuncError("Could not fit any distributions")
            
        log_evidences = np.array(log_evidences)
        successful_priors = np.array(successful_priors)
        
        # Calculate posterior probabilities
        log_posteriors = log_evidences + np.log(successful_priors)
        max_log_posterior = np.max(log_posteriors)
        log_posteriors -= max_log_posterior  # Numerical stability
        
        posteriors = np.exp(log_posteriors)
        posteriors /= np.sum(posteriors)
        
        return dict(zip(successful_distributions, posteriors))


# Convenience functions
def fit_distribution(
    data: Union[List, np.ndarray, pd.Series],
    distribution: str = "normal"
) -> DistributionFit:
    """Fit a distribution to data."""
    analyzer = DistributionAnalyzer()
    return analyzer.fit_distribution(data, distribution)


def find_best_distribution(
    data: Union[List, np.ndarray, pd.Series],
    distributions: List[str] = None
) -> DistributionComparison:
    """Find the best fitting distribution."""
    analyzer = DistributionAnalyzer()
    return analyzer.fit_best_distribution(data, distributions)


def generate_from_distribution(
    distribution_name: str,
    parameters: Dict[str, float],
    n_samples: int,
    random_state: int = None
) -> np.ndarray:
    """Generate samples from a parameterized distribution."""
    # This is a simplified implementation
    analyzer = DistributionAnalyzer()
    
    if distribution_name in analyzer.CONTINUOUS_DISTRIBUTIONS:
        dist_class = analyzer.CONTINUOUS_DISTRIBUTIONS[distribution_name]
    elif distribution_name in analyzer.DISCRETE_DISTRIBUTIONS:
        dist_class = analyzer.DISCRETE_DISTRIBUTIONS[distribution_name]
    else:
        raise ValidationError(f"Unknown distribution: {distribution_name}")
        
    # Convert parameters to args (simplified)
    if 'loc' in parameters and 'scale' in parameters:
        args = (parameters['loc'], parameters['scale'])
    else:
        args = tuple(parameters.values())
        
    if random_state is not None:
        np.random.seed(random_state)
        
    return dist_class.rvs(*args, size=n_samples)