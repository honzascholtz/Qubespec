#!/usr/bin/env python3
"""
JAX-based nested sampling for QubeSpec using BlackJAX.
Follows the workshop_nested_sampling.py pattern exactly.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import blackjax
import numpy as np
import time
import tqdm
from typing import Dict, List, Tuple, Optional, Callable
from jax.typing import ArrayLike
from anesthetic import NestedSamples
from ..Models_JAX.core_models import halpha_oiii_model, halpha_oiii_outflow_model, halpha_model, oiii_model

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


class NestedSamplingResults:
    """Container for nested sampling results."""
    
    def __init__(self, nested_samples: NestedSamples, logz: float, logz_err: float, info: Dict):
        self.nested_samples = nested_samples
        self.logz = logz
        self.logz_err = logz_err
        self.info = info
        self.samples = nested_samples  # For backward compatibility
        
    def get_summary(self) -> Dict:
        """Get parameter summary statistics."""
        means = {}
        stds = {}
        
        for param in self.nested_samples.columns:
            means[param] = float(self.nested_samples[param].mean())
            stds[param] = float(self.nested_samples[param].std())
            
        return {'means': means, 'stds': stds}


class JAXNestedSampler:
    """
    JAX-based nested sampler using BlackJAX.
    Follows the workshop_nested_sampling.py pattern exactly.
    """
    
    def __init__(self, rng_key: ArrayLike, num_live: int = 200, num_delete: int = 10):
        """
        Initialize the JAX nested sampler.
        
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
            Convergence criterion for nested sampling
            
        Returns
        -------
        NestedSamplingResults
            Results object
        """
        start_time = time.time()
        
        # 1. Get the model function
        model_func = self._get_model_function(model_name)
        
        # 2. Create likelihood function that accepts dict parameters (like workshop)
        def spectral_loglikelihood(params):
            """Log-likelihood for spectral model fitting."""
            try:
                # Convert log parameters back to linear for model evaluation
                linear_params = self._convert_to_linear_params(params)
                model_flux = model_func(wavelength, **linear_params)
                
                # Chi-squared likelihood
                chi2 = jnp.sum(((flux - model_flux) / error) ** 2)
                return -0.5 * chi2
            except Exception:
                return -jnp.inf
        
        # 3. Create prior bounds (following workshop pattern)
        prior_bounds = self._create_prior_bounds(priors)
        
        # 4. Initialize nested sampler (exactly like workshop)
        key_init, self.rng_key = random.split(self.rng_key)
        particles, logprior_fn = blackjax.ns.utils.uniform_prior(
            key_init, self.num_live, prior_bounds
        )
        
        nested_sampler = blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=spectral_loglikelihood,
            num_delete=self.num_delete,
            num_inner_steps=len(param_names) * num_inner_steps_multiplier
        )
        
        # 5. JIT compile the init and step functions
        init_fn = jax.jit(nested_sampler.init)
        step_fn = jax.jit(nested_sampler.step)
        
        # 6. Run the nested sampling loop (exactly like workshop)
        live = init_fn(particles)
        dead = []
        
        with tqdm.tqdm(desc="Nested Sampling", unit=" dead points") as pbar:
            while not live.logZ_live - live.logZ < convergence_criterion:
                key_step, self.rng_key = random.split(self.rng_key)
                live, dead_info = step_fn(key_step, live)
                dead.append(dead_info)
                pbar.update(self.num_delete)
        
        # 7. Finalize the run (exactly like workshop)
        dead = blackjax.ns.utils.finalise(live, dead)
        end_time = time.time()
        
        # 8. Process results with anesthetic (exactly like workshop)
        # Get the parameter names for the results (including log_ prefixes)
        result_columns = list(prior_bounds.keys())
        data = jnp.vstack([dead.particles[key] for key in result_columns]).T
        
        nested_samples = NestedSamples(
            data=np.array(data),
            logL=np.array(dead.loglikelihood),
            logL_birth=np.array(dead.loglikelihood_birth),
            columns=result_columns,
            logzero=jnp.nan
        )
        
        logz = nested_samples.logZ()
        logz_err = nested_samples.logZ(nsamples=100).std()
        
        # Create info dictionary
        info_dict = {
            'num_iterations': len(dead.particles[result_columns[0]]),
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
    
    def _create_prior_bounds(self, priors: Dict[str, List]) -> Dict[str, Tuple[float, float]]:
        """Create prior bounds for BlackJAX, handling log-uniform parameters."""
        bounds = {}
        
        for param_name, prior_spec in priors.items():
            if len(prior_spec) >= 4:
                prior_type = prior_spec[1]
                
                if prior_type == 'loguniform':
                    # For log-uniform, use log bounds and log_ prefix
                    log_min, log_max = prior_spec[2], prior_spec[3]
                    bounds[f"log_{param_name}"] = (log_min, log_max)
                elif prior_type == 'uniform':
                    # For uniform, use linear bounds
                    min_val, max_val = prior_spec[2], prior_spec[3]
                    bounds[param_name] = (min_val, max_val)
                elif prior_type == 'normal' or prior_type == 'normal_hat':
                    # For normal, use mean ± 3*sigma as bounds
                    mean, std = prior_spec[2], prior_spec[3]
                    bounds[param_name] = (mean - 3*std, mean + 3*std)
                    
        return bounds
    
    def _convert_to_linear_params(self, log_params: Dict[str, float]) -> Dict[str, float]:
        """Convert log parameters back to linear space for model evaluation."""
        linear_params = {}
        
        for key, value in log_params.items():
            if key.startswith('log_'):
                # Convert log parameter to linear
                param_name = key[4:]  # Remove 'log_' prefix
                linear_params[param_name] = 10 ** value
            else:
                # Keep linear parameters as-is
                linear_params[key] = value
                
        return linear_params


def create_default_priors(model_name: str, z: float) -> Tuple[Dict[str, List], List[str]]:
    """
    Create default priors for QubeSpec models.
    
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
    # Base priors - use log-uniform for positive parameters that span orders of magnitude
    base_priors = {
        'z': [z, 'uniform', z - 0.1, z + 0.1],
        'cont': [0.01, 'loguniform', -3, -1],  # log10(0.001) to log10(0.1)
        'cont_grad': [0.0, 'uniform', -0.5, 0.5],
        'nar_fwhm': [300, 'uniform', 100, 900]
    }
    
    # Model-specific priors
    if model_name == 'halpha':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 0],    # log10(0.001) to log10(1.0)
            'nii_peak': [0.03, 'loguniform', -3, -0.3], # log10(0.001) to log10(0.5)
            'sii_r_peak': [0.02, 'loguniform', -3, -0.7], # log10(0.001) to log10(0.2)
            'sii_b_peak': [0.02, 'loguniform', -3, -0.7]  # log10(0.001) to log10(0.2)
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'nar_fwhm', 'sii_r_peak', 'sii_b_peak']
        
    elif model_name == 'oiii':
        priors = {
            **base_priors,
            'oiii_peak': [0.1, 'loguniform', -3, 0],
            'hbeta_peak': [0.03, 'loguniform', -3, -0.5]
        }
        param_names = ['z', 'cont', 'cont_grad', 'oiii_peak', 'hbeta_peak', 'nar_fwhm']
        
    elif model_name == 'halpha_oiii':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 0],
            'nii_peak': [0.03, 'loguniform', -3, -0.3],
            'sii_r_peak': [0.02, 'loguniform', -3, -0.7],
            'sii_b_peak': [0.02, 'loguniform', -3, -0.7],
            'oiii_peak': [0.1, 'loguniform', -3, 0],
            'hbeta_peak': [0.03, 'loguniform', -3, -0.5]
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'nar_fwhm', 
                      'sii_r_peak', 'sii_b_peak', 'oiii_peak', 'hbeta_peak']
        
    elif model_name == 'halpha_oiii_outflow':
        priors = {
            **base_priors,
            'hal_peak': [0.1, 'loguniform', -3, 0],
            'nii_peak': [0.03, 'loguniform', -3, -0.3],
            'oiii_peak': [0.1, 'loguniform', -3, 0],
            'hbeta_peak': [0.03, 'loguniform', -3, -0.5],
            'sii_r_peak': [0.02, 'loguniform', -3, -0.7],
            'sii_b_peak': [0.02, 'loguniform', -3, -0.7],
            'outflow_fwhm': [600, 'uniform', 300, 1500],
            'outflow_vel': [-50, 'uniform', -500, 500],
            'hal_out_peak': [0.03, 'loguniform', -3, -0.5],
            'nii_out_peak': [0.01, 'loguniform', -3, -1],
            'oiii_out_peak': [0.03, 'loguniform', -3, -0.5],
            'hbeta_out_peak': [0.01, 'loguniform', -3, -1]
        }
        param_names = ['z', 'cont', 'cont_grad', 'hal_peak', 'nii_peak', 'oiii_peak', 'hbeta_peak',
                      'sii_r_peak', 'sii_b_peak', 'nar_fwhm', 'outflow_fwhm', 'outflow_vel',
                      'hal_out_peak', 'nii_out_peak', 'oiii_out_peak', 'hbeta_out_peak']
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return priors, param_names