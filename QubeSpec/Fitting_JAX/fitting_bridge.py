#!/usr/bin/env python3
"""
Bridge class for JAX-based fitting that maintains compatibility with existing QubeSpec interface.

This class provides a drop-in replacement for the original Fitting class
while using JAX + BlackJAX nested sampling under the hood.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pickle
import time

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

from .nested_sampling import JAXNestedSampler, create_default_priors, NestedSamplingResults
from .likelihood import chi_squared, reduced_chi_squared, bic_score


class FittingJAX:
    """
    JAX-based fitting class that maintains compatibility with the original QubeSpec Fitting interface.
    
    This class serves as a bridge between the original emcee-based fitting and the new
    JAX + BlackJAX nested sampling implementation.
    """
    
    def __init__(self, wave: np.ndarray, flux: np.ndarray, error: np.ndarray, z: float,
                 N: int = 5000, ncpu: int = 1, progress: bool = True, 
                 sampler: str = 'nested', priors: Optional[Dict] = None,
                 rng_seed: int = 42):
        """
        Initialize the JAX-based fitting class.
        
        Parameters
        ----------
        wave : np.ndarray
            Wavelength array
        flux : np.ndarray
            Flux array
        error : np.ndarray
            Error array
        z : float
            Redshift
        N : int
            Number of iterations (adapted for nested sampling)
        ncpu : int
            Number of CPUs (not used in JAX version)
        progress : bool
            Show progress
        sampler : str
            Sampler type ('nested' only)
        priors : Optional[Dict]
            Prior specifications
        rng_seed : int
            Random seed for reproducibility
        """
        self.wave = jnp.array(wave)
        self.flux = jnp.array(flux)
        self.error = jnp.array(error)
        self.z = z
        self.N = N
        self.ncpu = ncpu
        self.progress = progress
        self.sampler = sampler
        self.priors = priors or {}
        
        # Initialize JAX random key
        self.rng_key = random.PRNGKey(rng_seed)
        
        # Initialize nested sampler
        # Scale num_live based on N parameter to maintain similar computational cost
        num_live = min(500, max(100, N // 10))
        self.nested_sampler = JAXNestedSampler(
            rng_key=self.rng_key, 
            num_live=num_live,
            num_delete=min(50, num_live // 10)
        )
        
        # Results storage (compatible with original interface)
        self.chains = {}
        self.props = {}
        self.flat_samples = None
        self.like_chains = None
        self.yeval = None
        self.chi2 = None
        self.BIC = None
        self.fitted_model = None
        self.labels = []
        self.model = None
        self.nested_results = None
        
    def set_model(self, model_name: str) -> None:
        """Set the model to be fitted."""
        self.model = model_name
        
    def set_priors(self, priors: Union[str, Dict]) -> None:
        """Set the priors for fitting."""
        if priors == 'default':
            self.priors, self.labels = create_default_priors(self.model, self.z)
        else:
            self.priors = priors
        
    def fitting_collapse_Halpha_OIII(self, models: str = 'Single_only', plot: int = 0) -> None:
        """
        Fit Halpha + [OIII] model to collapsed spectrum.
        
        Parameters
        ----------
        models : str
            Model type ('Single_only', 'Outflow_only', 'Outflow')
        plot : int
            Plot results (not implemented in JAX version)
        """
        # Map model names
        model_mapping = {
            'Single_only': 'halpha_oiii',
            'Outflow_only': 'halpha_oiii_outflow',
            'Outflow': 'halpha_oiii_outflow'  # Default to outflow for now
        }
        
        if models not in model_mapping:
            raise ValueError(f"Model '{models}' not supported. Available: {list(model_mapping.keys())}")
        
        model_name = model_mapping[models]
        self.model = models
        
        # Create priors and parameter names
        priors, param_names = create_default_priors(model_name, self.z)
        
        # Update with user-provided priors
        if self.priors:
            for param, spec in self.priors.items():
                if param in priors:
                    priors[param] = spec
        
        self.labels = param_names
        
        # Run nested sampling
        self.nested_results = self.nested_sampler.fit_spectrum(
            wavelength=self.wave,
            flux=self.flux,
            error=self.error,
            model_name=model_name,
            priors=priors,
            param_names=param_names,
            max_iterations=self.N,
            tolerance=0.01
        )
        
        # Convert results to original format
        self._convert_results_to_original_format()
        
    def fitting_collapse_Halpha(self, models: str = 'Single_only', plot: int = 0) -> None:
        """
        Fit Halpha model to collapsed spectrum.
        
        Parameters
        ----------
        models : str
            Model type
        plot : int
            Plot results
        """
        model_name = 'halpha'
        self.model = models
        
        # Create priors and parameter names
        priors, param_names = create_default_priors(model_name, self.z)
        
        # Update with user-provided priors
        if self.priors:
            for param, spec in self.priors.items():
                if param in priors:
                    priors[param] = spec
        
        self.labels = param_names
        
        # Run nested sampling
        self.nested_results = self.nested_sampler.fit_spectrum(
            wavelength=self.wave,
            flux=self.flux,
            error=self.error,
            model_name=model_name,
            priors=priors,
            param_names=param_names,
            max_iterations=self.N,
            tolerance=0.01
        )
        
        # Convert results to original format
        self._convert_results_to_original_format()
        
    def fitting_collapse_OIII(self, models: str = 'Single_only', plot: int = 0) -> None:
        """
        Fit [OIII] model to collapsed spectrum.
        
        Parameters
        ----------
        models : str
            Model type
        plot : int
            Plot results
        """
        model_name = 'oiii'
        self.model = models
        
        # Create priors and parameter names
        priors, param_names = create_default_priors(model_name, self.z)
        
        # Update with user-provided priors
        if self.priors:
            for param, spec in self.priors.items():
                if param in priors:
                    priors[param] = spec
        
        self.labels = param_names
        
        # Run nested sampling
        self.nested_results = self.nested_sampler.fit_spectrum(
            wavelength=self.wave,
            flux=self.flux,
            error=self.error,
            model_name=model_name,
            priors=priors,
            param_names=param_names,
            max_iterations=self.N,
            tolerance=0.01
        )
        
        # Convert results to original format
        self._convert_results_to_original_format()
        
    def fitting_general(self, fitted_model, labels, logprior=None, nwalkers=64, 
                       skip_check=False, zscale=0.001, odd=False) -> None:
        """
        General fitting method (not fully implemented in JAX version).
        
        This method would require converting arbitrary user models to JAX,
        which is complex. For now, it raises an error directing users to
        use the specific fitting methods.
        """
        raise NotImplementedError(
            "General fitting with arbitrary models not yet implemented in JAX version. "
            "Please use specific fitting methods like fitting_collapse_Halpha_OIII()."
        )
        
    def _convert_results_to_original_format(self) -> None:
        """Convert nested sampling results to original emcee-compatible format."""
        if self.nested_results is None or self.nested_results.samples is None:
            return
        
        # Get samples from nested sampling
        samples = self.nested_results.samples
        
        # Create chains dictionary (compatible with original format)
        self.chains = {'name': self.model}
        self.flat_samples = np.array(samples.values)
        
        for i, label in enumerate(self.labels):
            self.chains[label] = self.flat_samples[:, i]
        
        # Create properties dictionary
        summary = self.nested_results.get_summary()
        self.props = {'name': self.model, 'popt': []}
        
        for label in self.labels:
            mean = summary['means'][label]
            std = summary['stds'][label]
            self.props[label] = np.array([mean, std, std])  # [mean, -err, +err]
            self.props['popt'].append(mean)
        
        # Calculate model evaluation and statistics
        if len(self.props['popt']) > 0:
            model_func = self.nested_sampler._get_model_function(
                self.nested_results.info['model_name']
            )
            self.yeval = np.array(model_func(self.wave, *self.props['popt']))
            
            # Calculate fit statistics
            self.chi2 = float(chi_squared(
                jnp.array(self.yeval), self.flux, self.error
            ))
            self.BIC = float(bic_score(
                jnp.array(self.yeval), self.flux, self.error, len(self.props['popt'])
            ))
        
        # Store log-likelihood chains (approximate from weights)
        if hasattr(samples, 'logL'):
            self.like_chains = np.array(samples.logL)
        
        # Store fitted model function
        self.fitted_model = self.nested_sampler._get_model_function(
            self.nested_results.info['model_name']
        )
        
    def save(self, file_path: str) -> None:
        """Save the fitting results."""
        # Convert JAX arrays to numpy for pickling
        save_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, jnp.ndarray):
                save_dict[key] = np.array(value)
            elif key == 'nested_sampler':
                # Don't save the sampler object
                continue
            else:
                save_dict[key] = value
        
        with open(file_path, "wb") as file:
            pickle.dump(save_dict, file)
    
    def load(self, file_path: str) -> None:
        """Load fitting results."""
        with open(file_path, "rb") as file:
            loaded_dict = pickle.load(file)
        
        # Update instance with loaded data
        for key, value in loaded_dict.items():
            if key == 'nested_sampler':
                continue
            setattr(self, key, value)
        
        # Recreate nested sampler if needed
        if not hasattr(self, 'nested_sampler'):
            self.nested_sampler = JAXNestedSampler(
                rng_key=random.PRNGKey(42),
                num_live=min(500, max(100, self.N // 10))
            )
    
    def corner(self):
        """Create corner plot using anesthetic."""
        if self.nested_results is None or self.nested_results.samples is None:
            raise ValueError("No results available for plotting")
        
        # Use anesthetic's built-in plotting
        return self.nested_results.samples.plot_2d(self.labels)
    
    def get_evidence(self) -> tuple:
        """
        Get Bayesian evidence.
        
        Returns
        -------
        tuple
            (log evidence, log evidence error)
        """
        if self.nested_results is None:
            return None, None
        
        return self.nested_results.logz, self.nested_results.logz_err
    
    def get_nested_samples(self):
        """Get the anesthetic NestedSamples object."""
        if self.nested_results is None:
            return None
        
        return self.nested_results.samples


# Factory function to create appropriate fitting class
def create_fitting_class(use_jax: bool = True, **kwargs):
    """
    Factory function to create either JAX or original fitting class.
    
    Parameters
    ----------
    use_jax : bool
        Whether to use JAX implementation
    **kwargs
        Arguments passed to fitting class
        
    Returns
    -------
    Fitting class instance
    """
    if use_jax:
        return FittingJAX(**kwargs)
    else:
        # Import and return original class
        from ..Fitting.fits_r import Fitting
        return Fitting(**kwargs)