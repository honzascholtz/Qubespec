#!/usr/bin/env python3
"""
JAX-based prior distributions for QubeSpec

Converted from original NumPy/SciPy implementations to pure JAX functions.
All functions are JIT-compatible for use with BlackJAX.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from typing import Dict, List, Tuple, Union
from jax.typing import ArrayLike
import numpy as np

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


@jit
def log_uniform_prior(x: float, low: float, high: float) -> float:
    """
    Log-uniform prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    low : float
        Lower bound (in log10 space)
    high : float
        Upper bound (in log10 space)
        
    Returns
    -------
    float
        Log prior probability
    """
    log_x = jnp.log10(x)
    return jnp.where(
        (log_x >= low) & (log_x <= high),
        -jnp.log(jnp.log(10.0)) - jnp.log(x) - jnp.log(high - low),
        -jnp.inf
    )


@jit
def uniform_prior(x: float, low: float, high: float) -> float:
    """
    Uniform prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    low : float
        Lower bound
    high : float
        Upper bound
        
    Returns
    -------
    float
        Log prior probability
    """
    return jnp.where(
        (x >= low) & (x <= high),
        -jnp.log(high - low),
        -jnp.inf
    )


@jit
def normal_prior(x: float, mean: float, std: float) -> float:
    """
    Normal prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    mean : float
        Mean
    std : float
        Standard deviation
        
    Returns
    -------
    float
        Log prior probability
    """
    return jsp.stats.norm.logpdf(x, mean, std)


@jit
def truncated_normal_prior(x: float, mean: float, std: float, 
                          low: float, high: float) -> float:
    """
    Truncated normal prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    mean : float
        Mean
    std : float
        Standard deviation
    low : float
        Lower bound
    high : float
        Upper bound
        
    Returns
    -------
    float
        Log prior probability
    """
    return jnp.where(
        (x >= low) & (x <= high),
        jsp.stats.norm.logpdf(x, mean, std) - jnp.log(
            jsp.stats.norm.cdf(high, mean, std) - jsp.stats.norm.cdf(low, mean, std)
        ),
        -jnp.inf
    )


@jit
def log_normal_prior(x: float, mean: float, std: float) -> float:
    """
    Log-normal prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    mean : float
        Mean in log space
    std : float
        Standard deviation in log space
        
    Returns
    -------
    float
        Log prior probability
    """
    return jnp.where(
        x > 0,
        jsp.stats.norm.logpdf(jnp.log10(x), mean, std) - jnp.log(jnp.log(10.0)) - jnp.log(x),
        -jnp.inf
    )


@jit
def truncated_log_normal_prior(x: float, mean: float, std: float,
                              low: float, high: float) -> float:
    """
    Truncated log-normal prior distribution.
    
    Parameters
    ----------
    x : float
        Parameter value
    mean : float
        Mean in log space
    std : float
        Standard deviation in log space
    low : float
        Lower bound in log space
    high : float
        Upper bound in log space
        
    Returns
    -------
    float
        Log prior probability
    """
    log_x = jnp.log10(x)
    return jnp.where(
        (x > 0) & (log_x >= low) & (log_x <= high),
        jsp.stats.norm.logpdf(log_x, mean, std) - jnp.log(jnp.log(10.0)) - jnp.log(x) - jnp.log(
            jsp.stats.norm.cdf(high, mean, std) - jsp.stats.norm.cdf(low, mean, std)
        ),
        -jnp.inf
    )


def create_prior_function(prior_dict: Dict[str, List]) -> callable:
    """
    Create a JAX-compatible prior function from QubeSpec prior dictionary.
    
    Parameters
    ----------
    prior_dict : Dict[str, List]
        Dictionary of priors in QubeSpec format:
        {param_name: [initial_value, 'prior_type', param1, param2, ...]}
        
    Returns
    -------
    callable
        JAX-compatible log prior function
    """
    param_names = list(prior_dict.keys())
    
    # Convert prior specifications to JAX-compatible format
    prior_specs = []
    for name in param_names:
        spec = prior_dict[name]
        prior_type = spec[1]
        
        if prior_type == 'uniform':
            prior_specs.append(('uniform', spec[2], spec[3]))
        elif prior_type == 'loguniform':
            prior_specs.append(('loguniform', spec[2], spec[3]))
        elif prior_type == 'normal':
            prior_specs.append(('normal', spec[2], spec[3]))
        elif prior_type == 'normal_hat':
            prior_specs.append(('truncated_normal', spec[2], spec[3], spec[4], spec[5]))
        elif prior_type == 'lognormal':
            prior_specs.append(('lognormal', spec[2], spec[3]))
        elif prior_type == 'lognormal_hat':
            prior_specs.append(('truncated_lognormal', spec[2], spec[3], spec[4], spec[5]))
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
    
    @jit
    def log_prior(params: ArrayLike) -> float:
        """
        Evaluate log prior for parameter vector.
        
        Parameters
        ----------
        params : ArrayLike
            Parameter vector
            
        Returns
        -------
        float
            Log prior probability
        """
        log_prob = 0.0
        
        for i, (prior_type, *args) in enumerate(prior_specs):
            param_val = params[i]
            
            if prior_type == 'uniform':
                log_prob += uniform_prior(param_val, args[0], args[1])
            elif prior_type == 'loguniform':
                log_prob += log_uniform_prior(param_val, args[0], args[1])
            elif prior_type == 'normal':
                log_prob += normal_prior(param_val, args[0], args[1])
            elif prior_type == 'truncated_normal':
                log_prob += truncated_normal_prior(param_val, args[0], args[1], args[2], args[3])
            elif prior_type == 'lognormal':
                log_prob += log_normal_prior(param_val, args[0], args[1])
            elif prior_type == 'truncated_lognormal':
                log_prob += truncated_log_normal_prior(param_val, args[0], args[1], args[2], args[3])
                
        return log_prob
    
    return log_prior


def create_uniform_prior_bounds(prior_dict: Dict[str, List]) -> Tuple[ArrayLike, ArrayLike]:
    """
    Create uniform prior bounds for nested sampling.
    
    Parameters
    ----------
    prior_dict : Dict[str, List]
        Dictionary of priors in QubeSpec format
        
    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Lower and upper bounds arrays
    """
    param_names = list(prior_dict.keys())
    lower_bounds = []
    upper_bounds = []
    
    for name in param_names:
        spec = prior_dict[name]
        prior_type = spec[1]
        
        if prior_type == 'uniform':
            lower_bounds.append(spec[2])
            upper_bounds.append(spec[3])
        elif prior_type == 'loguniform':
            lower_bounds.append(10**spec[2])
            upper_bounds.append(10**spec[3])
        elif prior_type == 'normal':
            # Use 4-sigma bounds for normal priors
            lower_bounds.append(spec[2] - 4*spec[3])
            upper_bounds.append(spec[2] + 4*spec[3])
        elif prior_type == 'normal_hat':
            lower_bounds.append(spec[4])
            upper_bounds.append(spec[5])
        elif prior_type == 'lognormal':
            # Use 4-sigma bounds in log space
            lower_bounds.append(10**(spec[2] - 4*spec[3]))
            upper_bounds.append(10**(spec[2] + 4*spec[3]))
        elif prior_type == 'lognormal_hat':
            lower_bounds.append(10**spec[4])
            upper_bounds.append(10**spec[5])
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
    
    return jnp.array(lower_bounds), jnp.array(upper_bounds)


def create_prior_transform(prior_dict: Dict[str, List]) -> callable:
    """
    Create a prior transform function for nested sampling.
    
    This transforms uniform [0,1] samples to the parameter space.
    
    Parameters
    ----------
    prior_dict : Dict[str, List]
        Dictionary of priors in QubeSpec format
        
    Returns
    -------
    callable
        Prior transform function
    """
    lower_bounds, upper_bounds = create_uniform_prior_bounds(prior_dict)
    
    @jit
    def prior_transform(unit_cube: ArrayLike) -> ArrayLike:
        """
        Transform uniform [0,1] samples to parameter space.
        
        Parameters
        ----------
        unit_cube : ArrayLike
            Uniform samples in [0,1]
            
        Returns
        -------
        ArrayLike
            Transformed parameters
        """
        return lower_bounds + unit_cube * (upper_bounds - lower_bounds)
    
    return prior_transform


# Default QubeSpec priors converted to JAX format
DEFAULT_PRIORS = {
    'z': [0, 'normal_hat', 0, 0.003, 0.01, 0.01],
    'cont': [0, 'loguniform', -4, 1],
    'cont_grad': [0, 'normal', 0, 0.3],
    'Hal_peak': [0, 'loguniform', -3, 1],
    'BLR_Hal_peak': [0, 'loguniform', -3, 1],
    'NII_peak': [0, 'loguniform', -3, 1],
    'Nar_fwhm': [300, 'uniform', 100, 900],
    'BLR_fwhm': [4000, 'uniform', 2000, 9000],
    'zBLR': [0, 'normal', 0, 0.003],
    'Hal_out_peak': [0, 'loguniform', -3, 1],
    'NII_out_peak': [0, 'loguniform', -3, 1],
    'outflow_fwhm': [600, 'uniform', 300, 1500],
    'outflow_vel': [-50, 'normal', 0, 300],
    'OIII_peak': [0, 'loguniform', -3, 1],
    'OIII_out_peak': [0, 'loguniform', -3, 1],
    'Hbeta_peak': [0, 'loguniform', -3, 1],
    'Hbeta_out_peak': [0, 'loguniform', -3, 1],
    'Fe_peak': [0, 'loguniform', -3, 1],
    'Fe_fwhm': [3000, 'uniform', 2000, 6000],
    'SIIr_peak': [0, 'loguniform', -3, 1],
    'SIIb_peak': [0, 'loguniform', -3, 1],
    'BLR_Hbeta_peak': [0, 'loguniform', -3, 1]
}