#!/usr/bin/env python3
"""
JAX-based likelihood functions for QubeSpec

Converted from original NumPy implementations to pure JAX functions.
All functions are JIT-compatible for use with BlackJAX.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from jax.typing import ArrayLike

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


@jit
def gaussian_log_likelihood(model: ArrayLike, data: ArrayLike, error: ArrayLike) -> float:
    """
    Compute Gaussian log-likelihood.
    
    Parameters
    ----------
    model : ArrayLike
        Model spectrum
    data : ArrayLike
        Observed data
    error : ArrayLike
        Error array
        
    Returns
    -------
    float
        Log-likelihood
    """
    # Handle NaN values by setting them to very large errors
    valid_mask = jnp.isfinite(data) & jnp.isfinite(error) & (error > 0)
    
    # Use nansum equivalent behavior
    chi2 = jnp.sum(
        jnp.where(valid_mask, ((data - model) / error) ** 2, 0.0)
    )
    
    # Add normalization term (optional for parameter estimation)
    n_valid = jnp.sum(valid_mask)
    log_norm = -0.5 * n_valid * jnp.log(2.0 * jnp.pi) - jnp.sum(
        jnp.where(valid_mask, jnp.log(error), 0.0)
    )
    
    return log_norm - 0.5 * chi2


def create_log_likelihood_function(model_func: Callable, wavelength: ArrayLike, 
                                  flux: ArrayLike, error: ArrayLike) -> Callable:
    """
    Create a JAX-compatible log-likelihood function.
    
    Parameters
    ----------
    model_func : Callable
        JAX-compatible model function
    wavelength : ArrayLike
        Wavelength array
    flux : ArrayLike
        Flux array
    error : ArrayLike
        Error array
        
    Returns
    -------
    Callable
        JAX-compatible log-likelihood function
    """
    @jit
    def log_likelihood(params: ArrayLike) -> float:
        """
        Evaluate log-likelihood for parameter vector.
        
        Parameters
        ----------
        params : ArrayLike
            Parameter vector
            
        Returns
        -------
        float
            Log-likelihood
        """
        # Evaluate model
        model_spectrum = model_func(wavelength, *params)
        
        # Compute likelihood
        return gaussian_log_likelihood(model_spectrum, flux, error)
    
    return log_likelihood


def create_log_posterior_function(model_func: Callable, wavelength: ArrayLike,
                                 flux: ArrayLike, error: ArrayLike, log_prior: Callable) -> Callable:
    """
    Create a JAX-compatible log-posterior function.
    
    Parameters
    ----------
    model_func : Callable
        JAX-compatible model function
    wavelength : ArrayLike
        Wavelength array
    flux : ArrayLike
        Flux array
    error : ArrayLike
        Error array
    log_prior : Callable
        JAX-compatible log-prior function
        
    Returns
    -------
    Callable
        JAX-compatible log-posterior function
    """
    log_likelihood = create_log_likelihood_function(model_func, wavelength, flux, error)
    
    @jit
    def log_posterior(params: ArrayLike) -> float:
        """
        Evaluate log-posterior for parameter vector.
        
        Parameters
        ----------
        params : ArrayLike
            Parameter vector
            
        Returns
        -------
        float
            Log-posterior probability
        """
        lp = log_prior(params)
        
        # Return -inf if prior is zero
        return jnp.where(
            jnp.isfinite(lp),
            lp + log_likelihood(params),
            -jnp.inf
        )
    
    return log_posterior


@jit
def chi_squared(model: ArrayLike, data: ArrayLike, error: ArrayLike) -> float:
    """
    Compute chi-squared statistic.
    
    Parameters
    ----------
    model : ArrayLike
        Model spectrum
    data : ArrayLike
        Observed data
    error : ArrayLike
        Error array
        
    Returns
    -------
    float
        Chi-squared value
    """
    valid_mask = jnp.isfinite(data) & jnp.isfinite(error) & (error > 0)
    
    return jnp.sum(
        jnp.where(valid_mask, ((data - model) / error) ** 2, 0.0)
    )


@jit
def reduced_chi_squared(model: ArrayLike, data: ArrayLike, error: ArrayLike, n_params: int) -> float:
    """
    Compute reduced chi-squared statistic.
    
    Parameters
    ----------
    model : ArrayLike
        Model spectrum
    data : ArrayLike
        Observed data
    error : ArrayLike
        Error array
    n_params : int
        Number of model parameters
        
    Returns
    -------
    float
        Reduced chi-squared value
    """
    valid_mask = jnp.isfinite(data) & jnp.isfinite(error) & (error > 0)
    n_valid = jnp.sum(valid_mask)
    
    chi2 = chi_squared(model, data, error)
    dof = n_valid - n_params
    
    return chi2 / jnp.maximum(dof, 1.0)  # Avoid division by zero


@jit
def bic_score(model: ArrayLike, data: ArrayLike, error: ArrayLike, n_params: int) -> float:
    """
    Compute Bayesian Information Criterion (BIC).
    
    Parameters
    ----------
    model : ArrayLike
        Model spectrum
    data : ArrayLike
        Observed data
    error : ArrayLike
        Error array
    n_params : int
        Number of model parameters
        
    Returns
    -------
    float
        BIC score
    """
    valid_mask = jnp.isfinite(data) & jnp.isfinite(error) & (error > 0)
    n_valid = jnp.sum(valid_mask)
    
    chi2 = chi_squared(model, data, error)
    
    return chi2 + n_params * jnp.log(n_valid)


@jit
def aic_score(model: ArrayLike, data: ArrayLike, error: ArrayLike, n_params: int) -> float:
    """
    Compute Akaike Information Criterion (AIC).
    
    Parameters
    ----------
    model : ArrayLike
        Model spectrum
    data : ArrayLike
        Observed data
    error : ArrayLike
        Error array
    n_params : int
        Number of model parameters
        
    Returns
    -------
    float
        AIC score
    """
    chi2 = chi_squared(model, data, error)
    
    return chi2 + 2 * n_params