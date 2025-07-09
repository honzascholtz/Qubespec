#!/usr/bin/env python3
"""
BlackJAX nested sampling interface for QubeSpec

This module provides the main interface for running nested sampling
fits on QubeSpec data using BlackJAX.
"""

import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, vmap
import blackjax
from typing import Dict, List, Tuple, Callable, Optional
from jax.typing import ArrayLike
import numpy as np
from anesthetic import NestedSamples
import time
import tqdm

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

from .priors import create_prior_function, create_prior_transform
from .likelihood import create_log_likelihood_function, create_log_posterior_function
from ..Models_JAX.core_models import (
    halpha_oiii_model, halpha_oiii_outflow_model, 
    halpha_model, oiii_model
)


class NestedSamplingResults:
    """
    Container for nested sampling results.
    
    Attributes
    ----------
    samples : NestedSamples
        Anesthetic NestedSamples object
    logz : float
        Log evidence
    logz_err : float
        Log evidence error
    info : Dict
        Additional information
    """
    
    def __init__(self, samples: NestedSamples, logz: float, logz_err: float, info: Dict):
        self.samples = samples
        self.logz = logz
        self.logz_err = logz_err
        self.info = info
        
    def get_summary(self) -> Dict:
        """Get parameter summary statistics."""
        return {
            'means': self.samples.mean().to_dict(),
            'stds': self.samples.std().to_dict(),
            'medians': self.samples.median().to_dict(),
            'percentiles': {
                '16': self.samples.quantile(0.16).to_dict(),
                '84': self.samples.quantile(0.84).to_dict()
            }
        }


class JAXNestedSampler:
    """
    JAX-based nested sampling fitter for QubeSpec.
    
    This class provides a high-level interface for fitting spectral models
    using BlackJAX nested sampling.
    """
    
    def __init__(self, rng_key: ArrayLike, num_live: int = 500, num_delete: int = 50):
        """
        Initialize the nested sampler.
        
        Parameters
        ----------
        rng_key : ArrayLike
            JAX random key
        num_live : int
            Number of live points
        num_delete : int
            Number of points to delete at each iteration
        """
        self.rng_key = rng_key
        self.num_live = num_live
        self.num_delete = num_delete
        
    def fit_spectrum(self, 
                    wavelength: ArrayLike, 
                    flux: ArrayLike, 
                    error: ArrayLike,
                    model_name: str,
                    priors: Dict[str, List],
                    param_names: List[str],
                    num_inner_steps_multiplier: int = 5,
                    convergence_criterion: float = -3.0) -> NestedSamplingResults:
        """
        Fit a spectrum using nested sampling.
        
        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength array
        flux : ArrayLike
            Flux array
        error : ArrayLike
            Error array
        model_name : str
            Name of the model to fit
        priors : Dict[str, List]
            Prior specifications
        param_names : List[str]
            Parameter names
        num_inner_steps_multiplier : int
            Multiplier for num_dims to set the number of inner MCMC steps
        convergence_criterion : float
            Convergence criterion based on logZ - logZ_live
            
        Returns
        -------
        NestedSamplingResults
            Results object
        """
        start_time = time.time()
        
        # 1. Get model and create JAX-native likelihood function
        model_func = self._get_model_function(model_name)
        log_likelihood = create_log_likelihood_function(model_func, wavelength, flux, error)
        
        # 2. Define prior bounds and parameter dictionaries
        lower_bounds, upper_bounds = self._create_uniform_bounds(priors)
        prior_bounds = {param_names[i]: (lower_bounds[i], upper_bounds[i]) 
                       for i in range(len(param_names))}
        
        # It's highly recommended to use dicts for the likelihood to avoid ordering bugs
        dict_log_likelihood = self._create_dict_likelihood(log_likelihood, param_names)

        # 3. Initialize the nested sampler
        key_init, self.rng_key = random.split(self.rng_key)
        particles, logprior_fn = blackjax.ns.utils.uniform_prior(key_init, self.num_live, prior_bounds)
        
        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=dict_log_likelihood,
            num_delete=self.num_delete,
            num_inner_steps=len(param_names) * num_inner_steps_multiplier
        )
        
        # 4. JIT compile the init and step functions
        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)
        
        # 5. Run the nested sampling loop with the CORRECT convergence criterion
        live = init_fn(particles)
        dead = []
        
        # Use tqdm for progress monitoring, which is very helpful for debugging hangs
        pbar = tqdm.tqdm(desc="Nested Sampling", unit=" dead points")
        
        iteration = 0
        while not live.logZ_live - live.logZ < convergence_criterion:
            key_step, self.rng_key = random.split(self.rng_key)
            live, dead_info = step_fn(key_step, live)
            
            # The second return value *is* the dead points info
            dead.append(dead_info)
            
            # Update progress bar
            pbar.update(self.num_delete)
            pbar.set_postfix({'logZ': f'{live.logZ:.2f}', 'logZ_live': f'{live.logZ_live:.2f}'})
            
            iteration += 1
            # Optional: Add a safety break for pathologically non-converging fits
            if iteration > 20000:  # A very high number
                print("Warning: Max iterations reached without convergence.")
                break
                
        pbar.close()

        # 6. CRUCIAL: Finalize the run by adding the remaining live points
        dead = blackjax.ns.utils.finalise(live, dead)
        end_time = time.time()
        
        # 7. Process results with anesthetic using the correct logL values
        data = jnp.vstack([dead.particles[key] for key in param_names]).T
        
        nested_samples = NestedSamples(
            data=np.array(data),
            logL=np.array(dead.loglikelihood),
            logL_birth=np.array(dead.loglikelihood_birth),
            columns=param_names,
            logzero=jnp.nan  # Important for anesthetic
        )
        
        logz = nested_samples.logZ()
        logz_err = nested_samples.logZ(nsamples=100).std()
        
        # Create info dictionary
        info_dict = {
            'num_iterations': iteration,
            'runtime': end_time - start_time,
            'model_name': model_name,
            'num_live': self.num_live,
            'num_delete': self.num_delete,
            'convergence_criterion': convergence_criterion,
        }
        
        return NestedSamplingResults(nested_samples, logz, logz_err, info_dict)
    
    def _get_model_function(self, model_name: str) -> Callable:
        """Get the appropriate model function."""
        model_registry = {
            'halpha_oiii': halpha_oiii_model,
            'halpha_oiii_outflow': halpha_oiii_outflow_model,
            'halpha': halpha_model,
            'oiii': oiii_model
        }
        
        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_registry.keys())}")
        
        return model_registry[model_name]
    
    def _create_uniform_bounds(self, priors: Dict[str, List]) -> Tuple[ArrayLike, ArrayLike]:
        """Create uniform bounds for nested sampling from priors."""
        from .priors import create_uniform_prior_bounds
        return create_uniform_prior_bounds(priors)
    
    def _create_dict_likelihood(self, log_likelihood: Callable, param_names: List[str]) -> Callable:
        """Convert array-based likelihood to dict-based for BlackJAX."""
        def dict_likelihood(params_dict):
            # Convert dict to array in the correct order
            params_array = jnp.array([params_dict[name] for name in param_names])
            return log_likelihood(params_array)
        return dict_likelihood
    
    def fit_cube_vectorized(self,
                           wavelength: ArrayLike,
                           flux_cube: ArrayLike,
                           error_cube: ArrayLike,
                           model_name: str,
                           priors: Dict[str, List],
                           param_names: List[str],
                           batch_size: Optional[int] = None,
                           max_iterations: int = 5000,
                           tolerance: float = 0.01) -> List[NestedSamplingResults]:
        """
        Fit an entire cube using vectorized operations.
        
        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength array
        flux_cube : ArrayLike
            Flux cube (n_spaxels, n_wavelength)
        error_cube : ArrayLike
            Error cube (n_spaxels, n_wavelength)
        model_name : str
            Name of the model to fit
        priors : Dict[str, List]
            Prior specifications
        param_names : List[str]
            Parameter names
        batch_size : Optional[int]
            Batch size for processing (None for full cube)
        max_iterations : int
            Maximum number of iterations per spaxel
        tolerance : float
            Convergence tolerance
            
        Returns
        -------
        List[NestedSamplingResults]
            List of results for each spaxel
        """
        n_spaxels = flux_cube.shape[0]
        
        if batch_size is None:
            batch_size = n_spaxels
        
        results = []
        
        # Process in batches
        for i in range(0, n_spaxels, batch_size):
            end_idx = min(i + batch_size, n_spaxels)
            batch_flux = flux_cube[i:end_idx]
            batch_error = error_cube[i:end_idx]
            
            # Fit each spaxel in the batch
            batch_results = []
            for j in range(batch_flux.shape[0]):
                result = self.fit_spectrum(
                    wavelength=wavelength,
                    flux=batch_flux[j],
                    error=batch_error[j],
                    model_name=model_name,
                    priors=priors,
                    param_names=param_names,
                    max_iterations=max_iterations,
                    tolerance=tolerance
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Progress update
            print(f"Processed {end_idx}/{n_spaxels} spaxels")
        
        return results


def create_default_priors(model_name: str, z: float) -> Tuple[Dict[str, List], List[str]]:
    """
    Create default priors for a given model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    z : float
        Redshift
        
    Returns
    -------
    Tuple[Dict[str, List], List[str]]
        Priors dictionary and parameter names
    """
    # Base priors
    base_priors = {
        'z': [z, 'normal_hat', z, 200/3e5*(1+z), z-1000/3e5*(1+z), z+1000/3e5*(1+z)],
        'cont': [0.01, 'loguniform', -4, 1],
        'cont_grad': [0.0, 'normal', 0, 0.3],
        'nar_fwhm': [300, 'uniform', 100, 900]
    }
    
    # Model-specific priors
    if model_name == 'halpha':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 1],
            'nii_peak': [0.03, 'loguniform', -3, 1],
            'sii_r_peak': [0.02, 'loguniform', -3, 1],
            'sii_b_peak': [0.02, 'loguniform', -3, 1]
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'nar_fwhm', 'sii_r_peak', 'sii_b_peak']
        
    elif model_name == 'oiii':
        priors = {
            **base_priors,
            'oiii_peak': [0.1, 'loguniform', -3, 1],
            'hbeta_peak': [0.03, 'loguniform', -3, 1]
        }
        param_names = ['z', 'cont', 'cont_grad', 'oiii_peak', 'nar_fwhm', 'hbeta_peak']
        
    elif model_name == 'halpha_oiii':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 1],
            'nii_peak': [0.03, 'loguniform', -3, 1],
            'sii_r_peak': [0.02, 'loguniform', -3, 1],
            'sii_b_peak': [0.02, 'loguniform', -3, 1],
            'oiii_peak': [0.1, 'loguniform', -3, 1],
            'hbeta_peak': [0.03, 'loguniform', -3, 1]
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'nar_fwhm', 
                      'sii_r_peak', 'sii_b_peak', 'oiii_peak', 'hbeta_peak']
        
    elif model_name == 'halpha_oiii_outflow':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 1],
            'nii_peak': [0.03, 'loguniform', -3, 1],
            'oiii_peak': [0.1, 'loguniform', -3, 1],
            'hbeta_peak': [0.03, 'loguniform', -3, 1],
            'sii_r_peak': [0.02, 'loguniform', -3, 1],
            'sii_b_peak': [0.02, 'loguniform', -3, 1],
            'outflow_fwhm': [600, 'uniform', 300, 1500],
            'outflow_vel': [-50, 'normal', 0, 300],
            'hal_out_peak': [0.03, 'loguniform', -3, 1],
            'nii_out_peak': [0.01, 'loguniform', -3, 1],
            'oiii_out_peak': [0.03, 'loguniform', -3, 1],
            'hbeta_out_peak': [0.01, 'loguniform', -3, 1]
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'oiii_peak', 'hbeta_peak',
                      'sii_r_peak', 'sii_b_peak', 'nar_fwhm', 'outflow_fwhm', 'outflow_vel',
                      'hal_out_peak', 'nii_out_peak', 'oiii_out_peak', 'hbeta_out_peak']
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return priors, param_names