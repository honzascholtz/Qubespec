#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spaxel fitting module for IFU spectroscopic data.

This module provides parallel fitting capabilities for emission line analysis
across IFU datacubes. It supports multiple emission line configurations
(H-alpha, [OIII], combined fits) and various models (single component, 
outflows, broad-line regions).

Modernized: 2025
@author: jscholtz
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import pickle
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tqdm
from astropy.table import Table
import multiprocess as mp
from multiprocess import Pool

from ..Fitting import Fitting
from .. import Utils as sp


# =============================================================================
# Enums and Constants
# =============================================================================

class ModelType(Enum):
    """Supported spectral models."""
    SINGLE = "Single"
    BLR = "BLR"
    BLR_SIMPLE = "BLR_simple"
    OUTFLOW_BOTH = "outflow_both"
    BLR_BOTH = "BLR_both"


class FittingMode(Enum):
    """Type of emission line fitting."""
    OIII = "OIII"
    HALPHA = "Halpha"
    HALPHA_OIII = "Halpha_OIII"
    GENERAL = "general"


class SamplerType(Enum):
    """MCMC sampler types."""
    EMCEE = "emcee"
    NUMPYRO = "numpyro"
    DYNESTY = "dynesty"


# Default prior configurations
DEFAULT_PRIORS_OIII = {
    'z': [0, 'normal', 0, 0.003],
    'cont': [0, 'loguniform', -4, 1],
    'cont_grad': [0, 'normal', 0, 0.3],
    'Nar_fwhm': [300, 'uniform', 100, 900],
    'BLR_fwhm': [4000, 'uniform', 2000, 9000],
    'zBLR': [0, 'normal', 0, 0.003],
    'outflow_fwhm': [600, 'uniform', 300, 1500],
    'outflow_vel': [-50, 'normal', 0, 300],
    'OIII_peak': [0, 'loguniform', -4, 1],
    'OIII_out_peak': [0, 'loguniform', -4, 1],
    'Hbeta_peak': [0, 'loguniform', -4, 1],
    'Hbeta_out_peak': [0, 'loguniform', -4, 1],
    'BLR_Hbeta_peak': [0, 'loguniform', -4, 1],
}

DEFAULT_PRIORS_HALPHA = {
    'z': [0, 'normal', 0, 0.003],
    'cont': [0, 'loguniform', -4, 1],
    'cont_grad': [0, 'normal', 0, 0.3],
    'Hal_peak': [0, 'loguniform', -4, 1],
    'BLR_Hal_peak': [0, 'loguniform', -4, 1],
    'NII_peak': [0, 'loguniform', -4, 1],
    'Nar_fwhm': [300, 'uniform', 100, 900],
    'BLR_fwhm': [4000, 'uniform', 2000, 9000],
    'zBLR': [0, 'normal', 0, 0.003],
    'SIIr_peak': [0, 'loguniform', -3, 1],
    'SIIb_peak': [0, 'loguniform', -3, 1],
    'Hal_out_peak': [0, 'loguniform', -4, 1],
    'NII_out_peak': [0, 'loguniform', -4, 1],
    'outflow_fwhm': [600, 'uniform', 300, 1500],
    'outflow_vel': [-50, 'normal', 0, 300],
}

DEFAULT_PRIORS_COMBINED = {**DEFAULT_PRIORS_HALPHA, **DEFAULT_PRIORS_OIII}


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class FittingConfig:
    """Configuration for spaxel fitting operations."""
    
    # Model configuration
    model_type: ModelType = ModelType.SINGLE
    fitting_mode: FittingMode = FittingMode.OIII
    sampler: SamplerType = SamplerType.EMCEE
    
    # MCMC parameters
    n_samples: int = 10000
    n_walkers: int = 64
    
    # Computational parameters
    n_cores: int = field(default_factory=lambda: max(1, mp.cpu_count() - 2))
    
    # File I/O
    suffix: str = ''
    save_suffix: str = ''
    
    # Priors
    priors: Dict[str, List] = field(default_factory=dict)
    
    # Additional parameters
    template: int = 0
    wavelength_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    show_progress: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.n_cores < 1:
            self.n_cores = 1
        
        # Set default priors based on fitting mode if not provided
        if not self.priors:
            self.priors = self._get_default_priors()
    
    def _get_default_priors(self) -> Dict[str, List]:
        """Get default priors for the fitting mode."""
        prior_map = {
            FittingMode.OIII: DEFAULT_PRIORS_OIII,
            FittingMode.HALPHA: DEFAULT_PRIORS_HALPHA,
            FittingMode.HALPHA_OIII: DEFAULT_PRIORS_COMBINED,
            FittingMode.GENERAL: {},
        }
        return prior_map.get(self.fitting_mode, {}).copy()
    
    @property
    def model_name(self) -> str:
        """Get model type as string."""
        return self.model_type.value
    
    @property
    def sampler_name(self) -> str:
        """Get sampler as string."""
        return self.sampler.value


# =============================================================================
# Helper Functions
# =============================================================================

def load_unwrapped_cube(cube, suffix: str = '') -> List:
    """
    Load unwrapped cube data from pickle file.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    suffix : str, optional
        Additional suffix for filename
        
    Returns
    -------
    list
        List of (i, j, flux, error, wavelength, z) tuples for each spaxel
    """
    filename = f"{cube.savepath}{cube.ID}_{cube.band}_Unwrapped_cube{suffix}.txt"
    
    if not Path(filename).exists():
        raise FileNotFoundError(f"Unwrapped cube not found: {filename}")
    
    with open(filename, "rb") as fp:
        unwrapped_cube = pickle.load(fp)
    
    return unwrapped_cube


def save_fit_results(cube, results: List, mode: str, suffix: str = '',
                    save_suffix: str = '') -> None:
    """
    Save fit results to pickle file.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    results : list
        List of fit results
    mode : str
        Fitting mode identifier
    suffix : str, optional
        Input filename suffix
    save_suffix : str, optional
        Additional output filename suffix
    """
    output_filename = (f"{cube.savepath}{cube.ID}_{cube.band}_"
                      f"spaxel_fit_raw_{mode}{suffix}{save_suffix}.txt")
    
    with open(output_filename, "wb") as fp:
        pickle.dump(results, fp)
    
    print(f"Saved results to: {output_filename}")


def load_fit_results(cube, mode: str, suffix: str = '') -> List:
    """
    Load fit results from pickle file.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    mode : str
        Fitting mode identifier
    suffix : str, optional
        Filename suffix
        
    Returns
    -------
    list
        List of fit results
    """
    filename = (f"{cube.savepath}{cube.ID}_{cube.band}_"
               f"spaxel_fit_raw_{mode}{suffix}.txt")
    
    with open(filename, "rb") as fp:
        results = pickle.load(fp)
    
    return results


# =============================================================================
# Base Spaxel Fitter Class
# =============================================================================

class BaseSpaxelFitter(ABC):
    """
    Base class for spaxel fitting operations.
    
    This abstract class provides common functionality for fitting emission
    lines across IFU datacubes, including parallel processing, error handling,
    and result management.
    
    Attributes
    ----------
    config : FittingConfig
        Configuration parameters for fitting
    status : str
        Current status of the fitter
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        """
        Initialize base fitter.
        
        Parameters
        ----------
        config : FittingConfig, optional
            Fitting configuration (creates default if not provided)
        """
        self.config = config if config is not None else FittingConfig()
        self.status = 'initialized'
    
    def fit_cube(self, cube, **kwargs) -> List:
        """
        Fit all spaxels in a cube using parallel processing.
        
        Parameters
        ----------
        cube : Cube
            QubeSpec Cube instance
        **kwargs : dict
            Additional configuration parameters (merged with self.config)
            
        Returns
        -------
        list
            Fit results for all spaxels
        """
        # Update configuration from kwargs
        self._update_config_from_kwargs(**kwargs)
        
        # Load unwrapped cube
        start_time = time.time()
        unwrapped_cube = load_unwrapped_cube(cube, self.config.suffix)
        print(f"Loaded {len(unwrapped_cube)} spaxels")
        
        # Set up progress bar
        if self.config.show_progress:
            progress_bar = tqdm.tqdm
        else:
            progress_bar = lambda x, total=0, desc='': x
        
        # Perform fitting
        if self.config.debug_mode:
            warnings.warn(
                '\u001b[5;33mDebug mode - no multiprocessing!\033[0;0m',
                UserWarning
            )
            results = [
                self.fit_single_spaxel(spaxel_data, progress=True)
                for spaxel_data in progress_bar(
                    unwrapped_cube,
                    total=len(unwrapped_cube),
                    desc="Fitting spaxels"
                )
            ]
        else:
            with Pool(self.config.n_cores) as pool:
                results = list(progress_bar(
                    pool.imap(self.fit_single_spaxel, unwrapped_cube),
                    total=len(unwrapped_cube),
                    desc="Fitting spaxels"
                ))
        
        # Save results
        save_fit_results(
            cube, results,
            self.config.fitting_mode.value,
            self.config.suffix,
            self.config.save_suffix
        )
        
        elapsed = time.time() - start_time
        print(f"Cube fitted in {elapsed:.2f} seconds")
        
        self.status = 'completed'
        return results
    
    def fit_single_spaxel(self, spaxel_data: Tuple, progress: bool = False) -> List:
        """
        Fit a single spaxel.
        
        This method implements the actual fitting logic and should be
        overridden by subclasses for specific fitting modes.
        
        Parameters
        ----------
        spaxel_data : tuple
            (i, j, flux, error, wavelength, z) for the spaxel
        progress : bool, optional
            Whether to show progress for this fit
            
        Returns
        -------
        list
            [i, j, fit_result(s)] or [i, j, error_dict]
        """
        i, j, flux, error, wavelength, z = spaxel_data
        
        # Apply wavelength selection if specified
        if len(self.config.wavelength_indices) > 0:
            idx = self.config.wavelength_indices
            flux = flux[idx]
            error = error[idx]
            wavelength = wavelength[idx]
        
        try:
            if self.config.model_type == ModelType.SINGLE:
                result = self._fit_single_model(
                    wavelength, flux, error, z, progress
                )
                return [i, j, result]
            
            elif self.config.model_type in [ModelType.OUTFLOW_BOTH, ModelType.BLR_BOTH]:
                result_single, result_complex = self._fit_two_models(
                    wavelength, flux, error, z, progress
                )
                return [i, j, result_single, result_complex]
            
            elif self.config.model_type in [ModelType.BLR, ModelType.BLR_SIMPLE]:
                result = self._fit_single_model(
                    wavelength, flux, error, z, progress
                )
                return [i, j, result]
            
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        except Exception as exc:
            print(f"Failed to fit spaxel ({i}, {j}): {exc}")
            n_results = 2 if self.config.model_type in [
                ModelType.OUTFLOW_BOTH, ModelType.BLR_BOTH
            ] else 1
            return [i, j] + [{'Failed fit': str(exc)}] * n_results
    
    @abstractmethod
    def _fit_single_model(self, wavelength: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, z: float,
                         progress: bool = False):
        """
        Fit a single model to spaxel data.
        
        Must be implemented by subclasses.
        """
        pass
    
    def _fit_two_models(self, wavelength: np.ndarray, flux: np.ndarray,
                       error: np.ndarray, z: float,
                       progress: bool = False) -> Tuple:
        """
        Fit two models and compare (e.g., single vs outflow).
        
        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array
        flux : np.ndarray
            Flux array
        error : np.ndarray
            Error array
        z : float
            Redshift
        progress : bool, optional
            Show progress
            
        Returns
        -------
        tuple
            (simple_model_result, complex_model_result)
        """
        # Determine model names
        if self.config.model_type == ModelType.OUTFLOW_BOTH:
            simple_model = 'gal'
            complex_model = 'outflow'
        elif self.config.model_type == ModelType.BLR_BOTH:
            simple_model = 'BLR_simple'
            complex_model = 'BLR'
        else:
            raise ValueError(f"Two-model fitting not supported for {self.config.model_type}")
        
        # Fit simple model
        result_simple = self._fit_with_model_name(
            wavelength, flux, error, z, simple_model, progress
        )
        
        # Fit complex model
        result_complex = self._fit_with_model_name(
            wavelength, flux, error, z, complex_model, progress
        )
        
        return result_simple, result_complex
    
    def _fit_with_model_name(self, wavelength: np.ndarray, flux: np.ndarray,
                            error: np.ndarray, z: float, model_name: str,
                            progress: bool = False):
        """
        Helper to fit with a specific model name.
        
        This is used internally by _fit_two_models.
        """
        # Temporarily change model type
        original_model = self.config.model_type
        
        # Map model name to ModelType
        model_map = {
            'gal': ModelType.SINGLE,
            'outflow': ModelType.SINGLE,
            'BLR': ModelType.BLR,
            'BLR_simple': ModelType.BLR_SIMPLE,
        }
        
        self.config.model_type = model_map.get(model_name, ModelType.SINGLE)
        
        # Store model name for use in fitting method
        self._current_model_name = model_name
        
        # Fit
        result = self._fit_single_model(wavelength, flux, error, z, progress)
        
        # Restore original model type
        self.config.model_type = original_model
        self._current_model_name = None
        
        return result
    
    def refit_spaxels(self, cube, spaxel_coords: List[Tuple[int, int]],
                     **kwargs) -> None:
        """
        Refit specific spaxels (top-up fitting).
        
        Parameters
        ----------
        cube : Cube
            QubeSpec Cube instance
        spaxel_coords : list of tuple
            List of (i, j) coordinates to refit
        **kwargs : dict
            Additional configuration parameters
        """
        self._update_config_from_kwargs(**kwargs)
        
        # Load existing results
        results = load_fit_results(
            cube,
            self.config.fitting_mode.value,
            self.config.suffix
        )
        
        # Build coordinate lookup
        coord_map = {}
        for idx, result in enumerate(results):
            i, j = result[0], result[1]
            coord_map[(i, j)] = idx
        
        # Refit specified spaxels
        for i_target, j_target in tqdm.tqdm(spaxel_coords, desc="Refitting"):
            idx = coord_map.get((i_target, j_target))
            
            if idx is None:
                warnings.warn(f"Spaxel ({i_target}, {j_target}) not found in results")
                continue
            
            # Extract spaxel data from existing result
            old_result = results[idx]
            if len(old_result) >= 3 and hasattr(old_result[2], 'fluxs'):
                fit_obj = old_result[2]
                spaxel_data = (
                    i_target, j_target,
                    fit_obj.fluxs, fit_obj.error,
                    fit_obj.wave, cube.z
                )
                
                # Refit
                new_result = self.fit_single_spaxel(spaxel_data, progress=True)
                results[idx] = new_result
                
                # Plot if successful
                if len(new_result) >= 3 and hasattr(new_result[2], 'yeval'):
                    self._plot_fit_result(new_result[2], i_target, j_target)
            else:
                warnings.warn(f"Cannot refit spaxel ({i_target}, {j_target}): invalid data")
        
        # Save updated results
        save_fit_results(
            cube, results,
            self.config.fitting_mode.value,
            self.config.suffix,
            self.config.save_suffix
        )
    
    def _plot_fit_result(self, fit_result, i: int, j: int) -> Figure:
        """
        Plot fit result for a single spaxel.
        
        Parameters
        ----------
        fit_result : Fitting
            Fit result object
        i, j : int
            Spaxel coordinates
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(1, figsize=(10, 5))
        
        ax.plot(fit_result.wave, fit_result.flux,
               drawstyle='steps-mid', label='Data')
        ax.plot(fit_result.wave, fit_result.yeval,
               'r--', label='Model')
        
        ax.text(0.05, 0.95, f'x={j}, y={i}',
               transform=ax.transAxes, va='top')
        ax.legend()
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        
        plt.tight_layout()
        return fig
    
    def _update_config_from_kwargs(self, **kwargs):
        """Update configuration from keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif key in ['models', 'model']:
                # Handle backward compatibility
                if isinstance(value, str):
                    self.config.model_type = ModelType(value)
            elif key == 'Ncores':
                self.config.n_cores = value
            elif key == 'add':
                self.config.suffix = value
            elif key == 'add_save':
                self.config.save_suffix = value
            elif key == 'N':
                self.config.n_samples = value
            elif key == 'progress':
                self.config.show_progress = value
            elif key == 'debug':
                self.config.debug_mode = value


# =============================================================================
# Specific Fitting Classes
# =============================================================================

class OIIIFitter(BaseSpaxelFitter):
    """
    Fitter for [OIII] emission line region.
    
    Fits [OIII]λλ4959,5007 and H-beta with optional outflow components
    and broad-line regions.
    
    Examples
    --------
    >>> from spaxel_fitting_refactored import OIIIFitter, FittingConfig, ModelType
    >>> 
    >>> config = FittingConfig(
    ...     model_type=ModelType.OUTFLOW_BOTH,
    ...     n_cores=8,
    ...     n_samples=5000
    ... )
    >>> 
    >>> fitter = OIIIFitter(config)
    >>> results = fitter.fit_cube(cube)
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        """Initialize OIII fitter."""
        if config is None:
            config = FittingConfig(fitting_mode=FittingMode.OIII)
        elif config.fitting_mode != FittingMode.OIII:
            config.fitting_mode = FittingMode.OIII
        
        super().__init__(config)
    
    def _fit_single_model(self, wavelength: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, z: float,
                         progress: bool = False):
        """Fit OIII model to spaxel data."""
        # Create Fitting object
        fit_obj = Fitting(
            wavelength, flux, error, z,
            N=self.config.n_samples,
            progress=progress,
            priors=self.config.priors,
            sampler=self.config.sampler_name
        )
        
        # Determine model name
        if hasattr(self, '_current_model_name'):
            model_name = self._current_model_name
        else:
            model_map = {
                ModelType.SINGLE: 'gal',
                ModelType.BLR: 'BLR',
                ModelType.BLR_SIMPLE: 'BLR_simple',
            }
            model_name = model_map.get(self.config.model_type, 'gal')
        
        # Fit
        fit_obj.fitting_OIII(model=model_name)
        fit_obj.fitted_model = 0
        
        return fit_obj


class HalphaFitter(BaseSpaxelFitter):
    """
    Fitter for H-alpha emission line region.
    
    Fits H-alpha, [NII]λλ6548,6583, and [SII]λλ6716,6731 with optional
    outflow components and broad-line regions.
    
    Examples
    --------
    >>> config = FittingConfig(
    ...     fitting_mode=FittingMode.HALPHA,
    ...     model_type=ModelType.SINGLE
    ... )
    >>> fitter = HalphaFitter(config)
    >>> results = fitter.fit_cube(cube)
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        """Initialize H-alpha fitter."""
        if config is None:
            config = FittingConfig(fitting_mode=FittingMode.HALPHA)
        elif config.fitting_mode != FittingMode.HALPHA:
            config.fitting_mode = FittingMode.HALPHA
        
        super().__init__(config)
    
    def _fit_single_model(self, wavelength: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, z: float,
                         progress: bool = False):
        """Fit H-alpha model to spaxel data."""
        fit_obj = Fitting(
            wavelength, flux, error, z,
            N=self.config.n_samples,
            progress=progress,
            priors=self.config.priors,
            sampler=self.config.sampler_name
        )
        
        # Determine model name
        if hasattr(self, '_current_model_name'):
            model_name = self._current_model_name
        else:
            model_map = {
                ModelType.SINGLE: 'gal',
                ModelType.BLR: 'BLR',
                ModelType.BLR_SIMPLE: 'BLR_simple',
            }
            model_name = model_map.get(self.config.model_type, 'gal')
        
        # Fit
        fit_obj.fitting_Halpha(model=model_name)
        fit_obj.fitted_model = 0
        
        return fit_obj
    
    def fit_cube_chunked(self, cube, chunk_size: int = 800, **kwargs) -> None:
        """
        Fit cube in chunks (useful for very large cubes).
        
        Parameters
        ----------
        cube : Cube
            QubeSpec Cube instance
        chunk_size : int, optional
            Number of spaxels per chunk
        **kwargs : dict
            Additional configuration parameters
        """
        self._update_config_from_kwargs(**kwargs)
        
        # Load data
        unwrapped_cube = load_unwrapped_cube(cube, self.config.suffix)
        n_spaxels = len(unwrapped_cube)
        n_chunks = int(np.ceil(n_spaxels / chunk_size))
        
        print(f"Fitting {n_spaxels} spaxels in {n_chunks} chunks")
        
        # Process each chunk
        for i_chunk in range(n_chunks):
            start_idx = i_chunk * chunk_size
            end_idx = min((i_chunk + 1) * chunk_size, n_spaxels)
            
            chunk_data = unwrapped_cube[start_idx:end_idx]
            
            print(f"Processing chunk {i_chunk + 1}/{n_chunks} "
                  f"(spaxels {start_idx}-{end_idx})")
            
            # Fit chunk
            with Pool(self.config.n_cores) as pool:
                chunk_results = list(tqdm.tqdm(
                    pool.imap(self.fit_single_spaxel, chunk_data),
                    total=len(chunk_data),
                    desc=f"Chunk {i_chunk + 1}"
                ))
            
            # Save chunk results
            chunk_filename = (f"{cube.savepath}{cube.ID}_{cube.band}_"
                            f"spaxel_fit_raw_Halpha{self.config.suffix}"
                            f"_chunk{i_chunk + 1}.txt")
            
            with open(chunk_filename, "wb") as fp:
                pickle.dump(chunk_results, fp)
            
            print(f"Saved chunk {i_chunk + 1} to: {chunk_filename}")


class HalphaOIIIFitter(BaseSpaxelFitter):
    """
    Fitter for combined H-alpha + [OIII] regions.
    
    Simultaneously fits both H-alpha complex and [OIII] complex, allowing
    for tied kinematics and better constraint of outflow properties.
    
    Examples
    --------
    >>> config = FittingConfig(
    ...     fitting_mode=FittingMode.HALPHA_OIII,
    ...     model_type=ModelType.OUTFLOW_BOTH
    ... )
    >>> fitter = HalphaOIIIFitter(config)
    >>> results = fitter.fit_cube(cube)
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        """Initialize combined H-alpha + OIII fitter."""
        if config is None:
            config = FittingConfig(fitting_mode=FittingMode.HALPHA_OIII)
        elif config.fitting_mode != FittingMode.HALPHA_OIII:
            config.fitting_mode = FittingMode.HALPHA_OIII
        
        super().__init__(config)
    
    def _fit_single_model(self, wavelength: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, z: float,
                         progress: bool = False):
        """Fit combined H-alpha + OIII model to spaxel data."""
        fit_obj = Fitting(
            wavelength, flux, error, z,
            N=self.config.n_samples,
            progress=progress,
            priors=self.config.priors,
            sampler=self.config.sampler_name
        )
        
        # Determine model name
        if hasattr(self, '_current_model_name'):
            model_name = self._current_model_name
        else:
            model_map = {
                ModelType.SINGLE: 'gal',
                ModelType.BLR: 'BLR',
                ModelType.BLR_SIMPLE: 'BLR_simple',
            }
            model_name = model_map.get(self.config.model_type, 'gal')
        
        # Fit
        fit_obj.fitting_Halpha_OIII(model=model_name)
        fit_obj.fitted_model = 0
        
        return fit_obj


class GeneralFitter(BaseSpaxelFitter):
    """
    General-purpose fitter for arbitrary emission line models.
    
    This flexible fitter accepts custom model functions and can be used
    for any emission line configuration not covered by the specific fitters.
    
    Attributes
    ----------
    fitted_model : callable
        Model function to fit
    labels : list
        Parameter names
    logprior : callable
        Log-prior function
    
    Examples
    --------
    >>> def my_model(wave, z, cont, line_peak, line_fwhm):
    ...     # Custom model implementation
    ...     return flux
    >>> 
    >>> def my_logprior(params, priors):
    ...     # Custom prior implementation
    ...     return log_prob
    >>> 
    >>> config = FittingConfig(
    ...     fitting_mode=FittingMode.GENERAL,
    ...     priors={'z': [0, 'normal', 0, 0.003], ...}
    ... )
    >>> 
    >>> fitter = GeneralFitter(
    ...     config,
    ...     fitted_model=my_model,
    ...     labels=['z', 'cont', 'line_peak', 'line_fwhm'],
    ...     logprior=my_logprior
    ... )
    >>> results = fitter.fit_cube(cube)
    """
    
    def __init__(self, config: Optional[FittingConfig] = None,
                 fitted_model: Optional[Callable] = None,
                 labels: Optional[List[str]] = None,
                 logprior: Optional[Callable] = None):
        """
        Initialize general fitter.
        
        Parameters
        ----------
        config : FittingConfig, optional
            Fitting configuration
        fitted_model : callable, optional
            Model function
        labels : list of str, optional
            Parameter names
        logprior : callable, optional
            Log-prior function
        """
        if config is None:
            config = FittingConfig(fitting_mode=FittingMode.GENERAL)
        elif config.fitting_mode != FittingMode.GENERAL:
            config.fitting_mode = FittingMode.GENERAL
        
        super().__init__(config)
        
        self.fitted_model = fitted_model
        self.labels = labels
        self.logprior = logprior
    
    def _fit_single_model(self, wavelength: np.ndarray, flux: np.ndarray,
                         error: np.ndarray, z: float,
                         progress: bool = False):
        """Fit general model to spaxel data."""
        if self.fitted_model is None:
            raise ValueError("fitted_model must be provided for GeneralFitter")
        if self.labels is None:
            raise ValueError("labels must be provided for GeneralFitter")
        if self.logprior is None:
            raise ValueError("logprior must be provided for GeneralFitter")
        
        fit_obj = Fitting(
            wavelength, flux, error, z,
            N=self.config.n_samples,
            progress=progress,
            priors=self.config.priors,
            sampler=self.config.sampler_name
        )
        
        fit_obj.fitting_general(
            self.fitted_model,
            self.labels,
            self.logprior,
            nwalkers=self.config.n_walkers
        )
        fit_obj.fitted_model = 0
        
        return fit_obj


# =============================================================================
# Backward Compatibility Classes
# =============================================================================

class OIII:
    """Backward compatibility wrapper for OIIIFitter."""
    
    def __init__(self):
        """Initialize OIII fitter."""
        self.status = 'ok'
        self._fitter = None
    
    def Spaxel_fitting(self, Cube, models='Single', add='', template=0,
                      sampler='emcee', Ncores=(mp.cpu_count() - 1),
                      priors=None, **kwargs):
        """Fit OIII spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_OIII
        
        config = FittingConfig(
            fitting_mode=FittingMode.OIII,
            model_type=ModelType(models),
            sampler=SamplerType(sampler),
            n_cores=Ncores,
            suffix=add,
            template=template,
            priors=priors,
            **kwargs
        )
        
        self._fitter = OIIIFitter(config)
        return self._fitter.fit_cube(Cube)
    
    def Spaxel_toptup(self, Cube, to_fit, add='', sampler='emcee',
                     Ncores=(mp.cpu_count() - 2), models='Single',
                     priors=None, **kwargs):
        """Refit specific spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_OIII
        
        if self._fitter is None:
            config = FittingConfig(
                fitting_mode=FittingMode.OIII,
                model_type=ModelType(models),
                sampler=SamplerType(sampler),
                n_cores=Ncores,
                suffix=add,
                priors=priors,
                **kwargs
            )
            self._fitter = OIIIFitter(config)
        
        self._fitter.refit_spaxels(Cube, to_fit, **kwargs)


class Halpha:
    """Backward compatibility wrapper for HalphaFitter."""
    
    def __init__(self):
        """Initialize H-alpha fitter."""
        self.status = 'ok'
        self._fitter = None
    
    def Spaxel_fitting(self, Cube, models='Single', sampler='emcee',
                      add='', Ncores=(mp.cpu_count() - 1),
                      priors=None, **kwargs):
        """Fit H-alpha spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_HALPHA
        
        config = FittingConfig(
            fitting_mode=FittingMode.HALPHA,
            model_type=ModelType(models),
            sampler=SamplerType(sampler),
            n_cores=Ncores,
            suffix=add,
            priors=priors,
            **kwargs
        )
        
        self._fitter = HalphaFitter(config)
        return self._fitter.fit_cube(Cube)
    
    def Spaxel_fitting_big(self, Cube, models='Single', sampler='emcee',
                          add='', Ncores=(mp.cpu_count() - 1),
                          priors=None, **kwargs):
        """Fit large cube in chunks (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_HALPHA
        
        config = FittingConfig(
            fitting_mode=FittingMode.HALPHA,
            model_type=ModelType(models),
            sampler=SamplerType(sampler),
            n_cores=Ncores,
            suffix=add,
            priors=priors,
            **kwargs
        )
        
        self._fitter = HalphaFitter(config)
        self._fitter.fit_cube_chunked(Cube, chunk_size=800)
    
    def Spaxel_toptup(self, Cube, to_fit, add='', Ncores=(mp.cpu_count() - 2),
                     sampler='emcee', models='Single', priors=None, **kwargs):
        """Refit specific spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_HALPHA
        
        if self._fitter is None:
            config = FittingConfig(
                fitting_mode=FittingMode.HALPHA,
                model_type=ModelType(models),
                sampler=SamplerType(sampler),
                n_cores=Ncores,
                suffix=add,
                priors=priors,
                **kwargs
            )
            self._fitter = HalphaFitter(config)
        
        self._fitter.refit_spaxels(Cube, to_fit, **kwargs)


class Halpha_OIII:
    """Backward compatibility wrapper for HalphaOIIIFitter."""
    
    def __init__(self):
        """Initialize combined H-alpha + OIII fitter."""
        self.status = 'ok'
        self._fitter = None
    
    def Spaxel_fitting(self, Cube, add='', Ncores=(mp.cpu_count() - 2),
                      sampler='emcee', models='Single', priors=None, **kwargs):
        """Fit combined H-alpha + OIII spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_COMBINED
        
        config = FittingConfig(
            fitting_mode=FittingMode.HALPHA_OIII,
            model_type=ModelType(models),
            sampler=SamplerType(sampler),
            n_cores=Ncores,
            suffix=add,
            priors=priors,
            **kwargs
        )
        
        self._fitter = HalphaOIIIFitter(config)
        return self._fitter.fit_cube(Cube)
    
    def Spaxel_toptup(self, Cube, to_fit, add='', models='Single',
                     sampler='emcee', Ncores=(mp.cpu_count() - 2),
                     priors=None, **kwargs):
        """Refit specific spaxels (backward compatible interface)."""
        if priors is None:
            priors = DEFAULT_PRIORS_COMBINED
        
        if self._fitter is None:
            config = FittingConfig(
                fitting_mode=FittingMode.HALPHA_OIII,
                model_type=ModelType(models),
                sampler=SamplerType(sampler),
                n_cores=Ncores,
                suffix=add,
                priors=priors,
                **kwargs
            )
            self._fitter = HalphaOIIIFitter(config)
        
        self._fitter.refit_spaxels(Cube, to_fit, **kwargs)


class general:
    """Backward compatibility wrapper for GeneralFitter."""
    
    def __init__(self):
        """Initialize general fitter."""
        self.status = 'ok'
        self._fitter = None
    
    def Spaxel_fitting(self, Cube, fitted_model, labels, priors, logprior,
                      sampler='emcee', nwalkers=64, use=np.array([]),
                      N=10000, add='', add_save='',
                      Ncores=(mp.cpu_count() - 2), **kwargs):
        """Fit general model to spaxels (backward compatible interface)."""
        config = FittingConfig(
            fitting_mode=FittingMode.GENERAL,
            sampler=SamplerType(sampler),
            n_walkers=nwalkers,
            wavelength_indices=use,
            n_samples=N,
            suffix=add,
            save_suffix=add_save,
            n_cores=Ncores,
            priors=priors,
            **kwargs
        )
        
        self._fitter = GeneralFitter(config, fitted_model, labels, logprior)
        return self._fitter.fit_cube(Cube)
    
    def Spaxel_topup(self, Cube, to_fit, fitted_model, labels, priors,
                    logprior, nwalkers=64, use=np.array([]), N=10000,
                    add='', add_save='', Ncores=(mp.cpu_count() - 2), **kwargs):
        """Refit specific spaxels (backward compatible interface)."""
        config = FittingConfig(
            fitting_mode=FittingMode.GENERAL,
            n_walkers=nwalkers,
            wavelength_indices=use,
            n_samples=N,
            suffix=add,
            save_suffix=add_save,
            n_cores=Ncores,
            priors=priors,
            **kwargs
        )
        
        if self._fitter is None:
            self._fitter = GeneralFitter(config, fitted_model, labels, logprior)
        
        self._fitter.refit_spaxels(Cube, to_fit, **kwargs)


# =============================================================================
# Additional Utilities
# =============================================================================

def Spaxel_ppxf(Cube, ncpu: int = 2):
    """
    Run pPXF fitting on spaxel data.
    
    This function interfaces with the nirspecxf package for pPXF-based
    emission line fitting.
    
    Parameters
    ----------
    Cube : Cube
        QubeSpec Cube instance
    ncpu : int, optional
        Number of CPUs to use
    """
    import glob
    import yaml
    from yaml.loader import SafeLoader
    import nirspecxf
    
    # Load template configuration
    template_path = ('/Users/jansen/My Drive/MyPython/Qubespec/QubeSpec/'
                    'jadify_temp/r100_jades_deep_hst_v3.1.1_template.yaml')
    
    with open(template_path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    
    # Update paths
    data['dirs']['data_dir'] = f"{Cube.savepath}PRISM_spaxel/"
    data['dirs']['output_dir'] = f"{Cube.savepath}PRISM_spaxel/"
    data['ppxf']['redshift_table'] = f"{Cube.savepath}PRISM_1D/redshift_1D.csv"
    
    # Save configuration
    config_path = f"{Cube.savepath}PRISM_spaxel/R100_1D_setup_test.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=True)
    
    # Create redshift catalog
    from . import jadify_temp as pth
    PATH_TO_jadify = pth.__path__[0] + '/'
    filename = PATH_TO_jadify + 'red_table_template.csv'
    redshift_cat = Table.read(filename)
    
    # Get spaxel IDs
    files = glob.glob(f"{Cube.savepath}PRISM_spaxel/prism_clear/*.fits")
    IDs = np.array([int(Path(f).stem[:6]) for f in files], dtype=int)
    
    # Create redshift table
    redshift_cat_mod = Table()
    redshift_cat_mod['ID'] = IDs
    redshift_cat_mod['z_visinsp'] = np.full(len(IDs), Cube.z)
    redshift_cat_mod['z_phot'] = np.full(len(IDs), Cube.z)
    redshift_cat_mod['z_bagp'] = np.full(len(IDs), Cube.z)
    redshift_cat_mod['flag'] = np.full(len(IDs), redshift_cat['flag'][0], dtype='<U6')
    
    output_cat_path = f"{Cube.savepath}PRISM_spaxel/redshift_spaxel.csv"
    redshift_cat_mod.write(output_cat_path, overwrite=True)
    
    # Run pPXF fitting
    config100 = nirspecxf.NIRSpecConfig(
        f"{Cube.savepath}PRISM_spaxel/R100_1D_setup_manual.yaml"
    )
    
    print(f"Running pPXF on {len(IDs)} spaxels with {ncpu} cores")
    nirspecxf.process_multi(ncpu, IDs, config100)
    
    # Merge results
    print('Fitting done, merging results')
    nirspecxf.data_prods.merge_em_lines_tables(
        f"{Cube.savepath}PRISM_spaxel/res/*R100_em_lines.fits",
        f"{Cube.savepath}PRISM_spaxel/spaxel_R100_ppxf_emlines.fits"
    )
    
    print("pPXF fitting complete")