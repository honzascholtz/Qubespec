#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopic fitting module for astronomical emission lines.

This module provides tools for fitting emission line spectra using MCMC methods,
supporting various emission line configurations including broad line regions (BLR),
outflows, and multiple components.

@author: jscholtz
Refactored: 2025
"""

import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.optimize import curve_fit
from astropy.modeling.powerlaws import PowerLaw1D

# Local imports
from ..Models import (
    OIII_models as O_models,
    Halpha_OIII_models as HO_models,
    QSO_models as QSO_models,
    Halpha_models as H_models,
    Full_optical as FO_models,
    Custom_model
)
from .. import Utils as sp
from .priors import *

warnings.filterwarnings("ignore")


# =============================================================================
# Constants
# =============================================================================
@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants in SI units."""
    c: float = 3.0e8  # Speed of light (m/s)
    h: float = 6.62e-34  # Planck constant (J⋅s)
    k: float = 1.38e-23  # Boltzmann constant (J/K)
    c_kms: float = 299792.458  # Speed of light (km/s)


@dataclass(frozen=True)
class FittingDefaults:
    """Default values for fitting parameters."""
    N_WALKERS: int = 64
    N_ITERATIONS: int = 5000
    DISCARD_FRACTION: float = 0.5
    THIN: int = 15
    DEFAULT_FWHM: float = 300.0
    DEFAULT_BLR_FWHM: float = 4000.0
    DEFAULT_OUTFLOW_FWHM: float = 600.0
    DEFAULT_OUTFLOW_VEL: float = -50.0
    REDSHIFT_TOLERANCE: float = 0.001
    REDSHIFT_WINDOW_KMS: float = 200.0
    REDSHIFT_RANGE_KMS: float = 1000.0


CONSTANTS = PhysicalConstants()
DEFAULTS = FittingDefaults()


# =============================================================================
# Helper Functions
# =============================================================================
def gaussian(x: np.ndarray, k: float, mu: float, fwhm: float) -> np.ndarray:
    """
    Gaussian function.
    
    Parameters
    ----------
    x : array-like
        Input wavelength array
    k : float
        Amplitude
    mu : float
        Center wavelength
    fwhm : float
        Full width at half maximum (in velocity units)
    
    Returns
    -------
    array-like
        Gaussian profile
    """
    sigma = fwhm / CONSTANTS.c_kms / 2.35 * mu
    return k * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sanitize_spectrum(
    flux: np.ndarray,
    error: np.ndarray,
    fill_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean up spectrum arrays by handling NaN, inf, and masked values.
    
    Parameters
    ----------
    flux : array-like
        Flux array
    error : array-like
        Error array
    fill_value : float, optional
        Value to use for invalid flux values
    
    Returns
    -------
    flux_clean, error_clean : tuple of arrays
        Sanitized flux and error arrays
    """
    flux = np.array(flux, dtype=float)
    error = np.array(error, dtype=float)
    
    # Handle masked arrays
    if isinstance(flux, np.ma.MaskedArray):
        flux = flux.filled(fill_value)
    if isinstance(error, np.ma.MaskedArray):
        error = error.filled(np.nan)
    
    # Replace NaN/inf in flux
    flux[~np.isfinite(flux)] = fill_value
    
    # Replace bad errors with large values
    median_error = np.nanmedian(error)
    error[~np.isfinite(error)] = 10000 * median_error
    error[error == 0] = 10000 * median_error
    
    return flux, error


def calculate_redshift_priors(
    z: float,
    prior_type: str = 'normal_hat',
    velocity_window: float = DEFAULTS.REDSHIFT_WINDOW_KMS,
    velocity_range: float = DEFAULTS.REDSHIFT_RANGE_KMS
) -> List:
    """
    Calculate sensible redshift priors based on a central value.
    
    Parameters
    ----------
    z : float
        Central redshift value
    prior_type : str
        Type of prior distribution
    velocity_window : float
        Velocity window for prior width (km/s)
    velocity_range : float
        Total velocity range for prior bounds (km/s)
    
    Returns
    -------
    list
        Prior specification [initial, type, mean, sigma, lower, upper]
    """
    dz_window = velocity_window / CONSTANTS.c_kms * (1 + z)
    dz_range = velocity_range / CONSTANTS.c_kms * (1 + z)
    
    return [
        z,
        prior_type,
        z,
        dz_window,
        z - dz_range,
        z + dz_range
    ]


# =============================================================================
# Main Fitting Class
# =============================================================================
class Fitting:
    """
    Class for fitting astronomical emission line spectra.
    
    This class provides methods for fitting various emission line configurations
    using MCMC (emcee) or least-squares methods.
    
    Parameters
    ----------
    wave : array-like
        Observed wavelength array (microns)
    flux : array-like
        Flux density array
    error : array-like
        Error array
    z : float
        Source redshift
    N : int, optional
        Number of MCMC iterations (default: 5000)
    ncpu : int, optional
        Number of CPUs for parallel processing (default: 1)
    progress : bool, optional
        Show progress bar (default: True)
    sampler : str, optional
        Fitting method: 'emcee' or 'leastsq' (default: 'emcee')
    priors : dict, optional
        Dictionary of prior specifications
    
    Attributes
    ----------
    chains : dict
        MCMC chains for fitted parameters
    props : dict
        Fitted parameter values and uncertainties
    yeval : array
        Model evaluation on full wavelength grid
    chi2 : float
        Chi-squared statistic
    BIC : float
        Bayesian Information Criterion
    """
    
    def __init__(
        self,
        wave: np.ndarray = np.array([]),
        flux: np.ndarray = np.array([]),
        error: np.ndarray = np.array([]),
        z: float = 1.0,
        N: int = DEFAULTS.N_ITERATIONS,
        ncpu: int = 1,
        progress: bool = True,
        sampler: str = 'emcee',
        priors: Optional[Dict] = None
    ):
        # Store input parameters
        self.z = z
        self.N = N
        self.ncpu = ncpu
        self.progress = progress
        self.sampler = sampler
        
        # Initialize priors
        self._initialize_priors(priors)
        
        # Store spectral data
        self.waves = wave.copy()
        self.wave = wave.copy()
        self.fluxs = flux.copy()
        self.errors = error.copy()
        self.error = error.copy()
        
        # Initialize result containers
        self.model = None
        self.template = None
        self.fitted_model = None
        self.labels = []
        self.chains = {}
        self.props = {}
        self.yeval = None
        self.chi2 = np.nan
        self.BIC = np.nan
    
    def _initialize_priors(self, user_priors: Optional[Dict] = None):
        """Initialize default priors and update with user values."""
        self.priors = {
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
            'BLR_Hbeta_peak': [0, 'loguniform', -3, 1],
        }
        
        if user_priors is not None:
            self.priors.update(user_priors)
    
    def _setup_redshift_prior(self):
        """Set up redshift prior if not already configured."""
        if self.priors['z'][0] == 0:
            self.priors['z'] = calculate_redshift_priors(self.z)
    
    def _prepare_data(self, fit_region: Optional[Tuple[float, float]] = None) -> None:
        """
        Prepare data for fitting by cleaning and selecting region.
        
        Parameters
        ----------
        fit_region : tuple, optional
            Wavelength range (min, max) for fitting
        """
        # Clean up masked/NaN values
        self.fluxs[np.isnan(self.fluxs)] = 0
        
        if hasattr(self.fluxs, 'mask'):
            valid_mask = ~self.fluxs.mask
            self.flux = self.fluxs.data[valid_mask]
            self.wave = self.waves[valid_mask]
        else:
            self.flux = self.fluxs.copy()
            self.wave = self.waves.copy()
        
        # Sanitize errors
        self.flux, self.error = sanitize_spectrum(self.flux, self.error)
        
        # Select fitting region if specified
        if fit_region is not None:
            self.fit_loc = np.where(
                (self.wave >= fit_region[0]) & (self.wave <= fit_region[1])
            )[0]
        else:
            self.fit_loc = np.arange(len(self.wave))
        
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
    
    def _estimate_peak_flux(
        self,
        line_center: float,
        window: float = 20.0
    ) -> float:
        """
        Estimate peak flux near an emission line.
        
        Parameters
        ----------
        line_center : float
            Central wavelength (Angstroms)
        window : float
            Window size around center (Angstroms)
        
        Returns
        -------
        float
            Estimated peak flux
        """
        center_observed = line_center * (1 + self.z) / 1e4
        mask = np.abs(self.wave - center_observed) < (window * (1 + self.z) / 1e4)
        
        if np.any(mask):
            return np.abs(np.max(self.flux[mask]))
        else:
            return np.abs(np.median(self.flux[self.fit_loc]))
    
    def _initialize_walkers(
        self,
        pos_l: np.ndarray,
        nwalkers: int = DEFAULTS.N_WALKERS
    ) -> np.ndarray:
        """
        Initialize MCMC walker positions.
        
        Parameters
        ----------
        pos_l : array
            Initial parameter values
        nwalkers : int
            Number of walkers
        
        Returns
        -------
        array
            Walker starting positions (nwalkers x ndim)
        """
        pos = np.random.normal(pos_l, np.abs(pos_l * 0.1), (nwalkers, len(pos_l)))
        pos[:, 0] = np.random.normal(self.z, DEFAULTS.REDSHIFT_TOLERANCE, nwalkers)
        return pos
    
    def _validate_initial_conditions(self, pos_l: np.ndarray) -> None:
        """
        Validate initial parameter values against priors.
        
        Parameters
        ----------
        pos_l : array
            Initial parameter values
        
        Raises
        ------
        ValueError
            If initial conditions violate priors
        """
        pr_code = self.prior_create()
        lp = self.log_prior_fce(pos_l, pr_code)
        
        if not np.isfinite(lp) or lp == -np.inf:
            print("Prior evaluation failed for initial conditions:")
            print(logprior_general_test(pos_l, pr_code, self.labels))
            raise ValueError(
                "Initial conditions outside prior bounds. "
                "Check your prior specifications."
            )
    
    def _run_mcmc(
        self,
        pos: np.ndarray,
        skip_check: bool = False
    ) -> emcee.EnsembleSampler:
        """
        Run MCMC sampling.
        
        Parameters
        ----------
        pos : array
            Initial walker positions
        skip_check : bool
            Skip initial state check
        
        Returns
        -------
        emcee.EnsembleSampler
            Fitted sampler object
        """
        nwalkers, ndim = pos.shape
        
        if self.ncpu == 1:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general
            )
            sampler.run_mcmc(
                pos, self.N,
                progress=self.progress,
                skip_initial_state_check=skip_check
            )
        else:
            from multiprocess import Pool
            with Pool(self.ncpu) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_general, pool=pool
                )
                sampler.run_mcmc(pos, self.N, progress=self.progress)
        
        return sampler
    
    def _extract_chains(self, sampler: emcee.EnsembleSampler) -> None:
        """Extract and store MCMC chains from sampler."""
        discard = int(DEFAULTS.DISCARD_FRACTION * self.N)
        self.flat_samples = sampler.get_chain(
            discard=discard, thin=DEFAULTS.THIN, flat=True
        )
        self.like_chains = sampler.get_log_prob(
            discard=discard, thin=DEFAULTS.THIN, flat=True
        )
        
        self.chains = {'name': self.model}
        for i, label in enumerate(self.labels):
            self.chains[label] = self.flat_samples[:, i]
    
    def _run_leastsq(self, pos_l: np.ndarray) -> None:
        """
        Run least-squares fitting.
        
        Parameters
        ----------
        pos_l : array
            Initial parameter values
        """
        # Remove invalid data points
        valid = ~(np.isnan(self.flux_fitloc) | np.isinf(self.flux_fitloc))
        self.error_fitloc[np.isnan(self.error_fitloc)] = 1e6
        
        # Perform fit
        popt, pcov = curve_fit(
            self.fitted_model,
            self.wave_fitloc[valid],
            self.flux_fitloc[valid],
            p0=pos_l,
            sigma=self.error_fitloc[valid],
            bounds=self.bounds_est()
        )
        errs = np.sqrt(np.diag(pcov))
        
        # Store results
        self.props = {'name': self.model, 'popt': popt}
        self.chains = {'name': self.model}
        
        for i, name in enumerate(self.labels):
            self.props[name] = [popt[i], errs[i], errs[i]]
            self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
    
    def _calculate_statistics(self) -> None:
        """Calculate goodness-of-fit statistics."""
        residuals = self.flux_fitloc - self.yeval_fitloc
        self.chi2 = np.nansum((residuals / self.error_fitloc) ** 2)
        n_params = len(self.props['popt'])
        n_data = len(self.flux_fitloc)
        self.BIC = self.chi2 + n_params * np.log(n_data)
    
    # =============================================================================
    # Primary Fitting Methods
    # =============================================================================
    
    def fitting_Halpha(
        self,
        model: str = 'gal',
        nwalkers: int = DEFAULTS.N_WALKERS
    ) -> None:
        """
        Fit Hα + [NII] + [SII] emission lines.
        
        Parameters
        ----------
        model : str
            Model configuration:
            - 'gal': Simple galaxy model
            - 'outflow': Galaxy with outflow component
            - 'BLR_simple': Simple broad line region
            - 'BLR': BLR with outflow
            - 'QSO_BKPL': QSO with broken power-law BLR
        nwalkers : int, optional
            Number of MCMC walkers
        """
        self.model = model
        self.template = None
        self._setup_redshift_prior()
        
        # Define fitting region around Hα
        fit_region = (
            (6564.52 - 170) * (1 + self.z) / 1e4,
            (6564.52 + 200) * (1 + self.z) / 1e4
        )
        self._prepare_data(fit_region)
        
        # Estimate peak flux
        peak = self._estimate_peak_flux(6564.52)
        cont = np.median(self.flux[self.fit_loc])
        if cont < 0:
            cont = 0.01
        
        # Configure model
        model_configs = {
            'gal': self._setup_halpha_gal,
            'outflow': self._setup_halpha_outflow,
            'BLR_simple': self._setup_halpha_blr_simple,
            'BLR': self._setup_halpha_blr,
            'QSO_BKPL': self._setup_halpha_qso_bkpl,
        }
        
        if model not in model_configs:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(model_configs.keys())}"
            )
        
        pos_l = model_configs[model](peak, cont)
        
        # Validate and fit
        self._validate_initial_conditions(pos_l)
        self.pos_l = pos_l
        
        if self.sampler == 'emcee':
            pos = self._initialize_walkers(pos_l, nwalkers)
            self.pos = pos
            sampler = self._run_mcmc(pos)
            self._extract_chains(sampler)
            self.props = self.prop_calc()
        elif self.sampler == 'leastsq':
            self._run_leastsq(pos_l)
        else:
            raise ValueError("Sampler must be 'emcee' or 'leastsq'")
        
        # Evaluate model and calculate statistics
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        self._calculate_statistics()
    
    def _setup_halpha_gal(self, peak: float, cont: float) -> np.ndarray:
        """Setup galaxy model for Hα fitting."""
        self.fitted_model = H_models.Halpha
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak',
            'Nar_fwhm', 'SIIr_peak', 'SIIb_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha'}
        
        pos_l = np.array([
            self.z, cont, 0.01, peak / 1.3, peak / 10,
            self.priors['Nar_fwhm'][0], peak / 6, peak / 6
        ])
        
        # Override with user priors if set
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_outflow(self, peak: float, cont: float) -> np.ndarray:
        """Setup outflow model for Hα fitting."""
        self.fitted_model = H_models.Halpha_outflow
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
            'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak',
            'outflow_fwhm', 'outflow_vel'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_wth_out'}
        
        pos_l = np.array([
            self.z, cont, 0.01, peak / 2, peak / 4, self.priors['Nar_fwhm'][0],
            peak / 6, peak / 6, peak / 8, peak / 8,
            self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0]
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_blr_simple(self, peak: float, cont: float) -> np.ndarray:
        """Setup simple BLR model for Hα fitting."""
        if self.priors['zBLR'][0] == 0:
            self.priors['zBLR'][0] = self.z
        
        self.fitted_model = H_models.Halpha_wBLR
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'BLR_Hal_peak', 'NII_peak',
            'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_wth_BLR'}
        
        pos_l = np.array([
            self.z, cont, 0.001, peak / 2, peak / 4, peak / 4,
            self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],
            self.priors['zBLR'][0], peak / 6, peak / 6
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_blr(self, peak: float, cont: float) -> np.ndarray:
        """Setup BLR with outflow model for Hα fitting."""
        self.fitted_model = H_models.Halpha_BLR_outflow
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'BLR_Hal_peak', 'NII_peak',
            'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak',
            'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_wth_BLR'}
        
        pos_l = np.array([
            self.z, cont, 0.001, peak / 2, peak / 4, peak / 4,
            self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],
            self.priors['zBLR'][0], peak / 6, peak / 6,
            peak / 6, peak / 6, 700, -100
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_qso_bkpl(self, peak: float, cont: float) -> np.ndarray:
        """Setup QSO broken power-law model for Hα fitting."""
        self.fitted_model = QSO_models.Hal_QSO_BKPL
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
            'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel',
            'BLR_Hal_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2', 'BLR_sig'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_QSO_BKPL'}
        
        pos_l = np.array([
            self.z, cont, 0.01, peak / 2, peak / 4, self.priors['Nar_fwhm'][0],
            peak / 6, peak / 6, self.priors['outflow_fwhm'][0],
            self.priors['outflow_vel'][0], peak, self.priors['zBLR'][0],
            self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],
            self.priors['BLR_sig'][0]
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def fitting_OIII(
        self,
        model: str = 'gal',
        Fe_template: Optional[str] = None,
        expand_prism: bool = False,
        nwalkers: int = DEFAULTS.N_WALKERS
    ) -> None:
        """
        Fit [OIII] + Hβ emission lines.
        
        Parameters
        ----------
        model : str
            Model configuration:
            - 'gal' or 'gal_simple': Simple galaxy model
            - 'outflow' or 'outflow_simple': Galaxy with outflow
            - 'BLR_simple': Simple broad line region
            - 'BLR_outflow': BLR with outflow
            - 'QSO_BKPL': QSO with broken power-law
        Fe_template : str, optional
            Iron template name (Tsuzuki, BG92, Veron)
        expand_prism : bool, optional
            Expand wavelength range for PRISM data
        nwalkers : int, optional
            Number of MCMC walkers
        """
        self.model = model
        self.template = Fe_template
        self._setup_redshift_prior()
        
        if self.priors['zBLR'][0] == 0:
            self.priors['zBLR'] = calculate_redshift_priors(self.z)
        
        # Define fitting region
        if expand_prism:
            fit_region = (
                4600 * (1 + self.z) / 1e4,
                5600 * (1 + self.z) / 1e4
            )
        else:
            fit_region = (
                4700 * (1 + self.z) / 1e4,
                5100 * (1 + self.z) / 1e4
            )
        
        self._prepare_data(fit_region)
        
        # Estimate peaks
        peak_oiii = self._estimate_peak_flux(5007)
        peak_hbeta = self._estimate_peak_flux(4861)
        cont = np.abs(np.median(self.flux[self.fit_loc]))
        
        # Configure model
        model_configs = {
            'gal': self._setup_oiii_gal,
            'gal_simple': self._setup_oiii_gal,
            'outflow': self._setup_oiii_outflow,
            'outflow_simple': self._setup_oiii_outflow,
            'BLR_simple': self._setup_oiii_blr_simple,
            'BLR_outflow': self._setup_oiii_blr_outflow,
            'QSO_BKPL': self._setup_oiii_qso_bkpl,
        }
        
        if model not in model_configs:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(model_configs.keys())}"
            )
        
        pos_l = model_configs[model](peak_oiii, peak_hbeta, cont)
        
        # Validate and fit
        self._validate_initial_conditions(pos_l)
        
        if self.sampler == 'emcee':
            pos = self._initialize_walkers(pos_l, nwalkers)
            sampler = self._run_mcmc(pos)
            self._extract_chains(sampler)
            self.props = self.prop_calc()
        elif self.sampler == 'leastsq':
            self._run_leastsq(pos_l)
        else:
            raise ValueError("Sampler must be 'emcee' or 'leastsq'")
        
        # Evaluate model and calculate statistics
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        self._calculate_statistics()
    
    def _setup_oiii_gal(
        self,
        peak_oiii: float,
        peak_hbeta: float,
        cont: float
    ) -> np.ndarray:
        """Setup galaxy model for [OIII] fitting."""
        self.fitted_model = O_models.OIII_gal
        self.log_prior_fce = logprior_general
        self.labels = ['z', 'cont', 'cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak']
        self.pr_code = self.prior_create()
        self.res = {'name': 'OIII_simple'}
        
        pos_l = np.array([
            self.z, cont, 0.001, peak_oiii / 2,
            self.priors['Nar_fwhm'][0], peak_hbeta
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_oiii_outflow(
        self,
        peak_oiii: float,
        peak_hbeta: float,
        cont: float
    ) -> np.ndarray:
        """Setup outflow model for [OIII] fitting."""
        self.fitted_model = O_models.OIII_outflow
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'OIII_peak', 'OIII_out_peak',
            'Nar_fwhm', 'outflow_fwhm', 'outflow_vel',
            'Hbeta_peak', 'Hbeta_out_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'OIII_outflow_simple'}
        
        pos_l = np.array([
            self.z, cont, 0.001, peak_oiii / 2, peak_oiii / 6,
            self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
            self.priors['outflow_vel'][0], peak_hbeta, peak_hbeta / 3
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_oiii_blr_simple(
        self,
        peak_oiii: float,
        peak_hbeta: float,
        cont: float
    ) -> np.ndarray:
        """Setup simple BLR model for [OIII] fitting."""
        if self.template:
            self.labels = [
                'z', 'cont', 'cont_grad', 'OIII_peak', 'Nar_fwhm',
                'Hbeta_peak', 'zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm',
                'Fe_peak', 'Fe_fwhm'
            ]
            self.fitted_model = O_models.OIII_fe_models(self.template).OIII_gal_BLR_Fe
            pos_l = np.array([
                self.z, cont, 0.001, peak_oiii / 2, self.priors['Nar_fwhm'][0],
                peak_hbeta, self.z, peak_hbeta / 2, self.priors['BLR_fwhm'][0],
                cont, self.priors['Fe_fwhm'][0]
            ])
        else:
            self.labels = [
                'z', 'cont', 'cont_grad', 'OIII_peak', 'Nar_fwhm',
                'Hbeta_peak', 'zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm'
            ]
            self.fitted_model = O_models.OIII_gal_BLR
            pos_l = np.array([
                self.z, cont, 0.001, peak_oiii / 2, self.priors['Nar_fwhm'][0],
                peak_hbeta, self.z, peak_hbeta / 2, self.priors['BLR_fwhm'][0]
            ])
        
        self.log_prior_fce = logprior_general
        self.pr_code = self.prior_create()
        self.res = {'name': 'OIII_BLR_simple'}
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_oiii_blr_outflow(
        self,
        peak_oiii: float,
        peak_hbeta: float,
        cont: float
    ) -> np.ndarray:
        """Setup BLR with outflow model for [OIII] fitting."""
        if self.template:
            self.labels = [
                'z', 'cont', 'cont_grad', 'OIII_peak', 'OIII_out_peak',
                'Nar_fwhm', 'outflow_fwhm', 'outflow_vel',
                'Hbeta_peak', 'Hbeta_out_peak', 'zBLR', 'BLR_Hbeta_peak',
                'BLR_fwhm', 'Fe_peak', 'Fe_fwhm'
            ]
            self.fitted_model = O_models.OIII_fe_models(self.template).OIII_outflow_BLR_Fe
            pos_l = np.array([
                self.z, cont, 0.001, peak_oiii / 2, peak_oiii / 6,
                self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
                self.priors['outflow_vel'][0], peak_hbeta, peak_hbeta / 3,
                self.z, peak_hbeta / 2, self.priors['BLR_fwhm'][0],
                cont, self.priors['Fe_fwhm'][0]
            ])
        else:
            self.labels = [
                'z', 'cont', 'cont_grad', 'OIII_peak', 'OIII_out_peak',
                'Nar_fwhm', 'outflow_fwhm', 'outflow_vel',
                'Hbeta_peak', 'Hbeta_out_peak', 'zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm'
            ]
            self.fitted_model = O_models.OIII_outflow_BLR
            pos_l = np.array([
                self.z, cont, 0.001, peak_oiii / 2, peak_oiii / 6,
                self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
                self.priors['outflow_vel'][0], peak_hbeta, peak_hbeta / 3,
                self.z, peak_hbeta / 2, self.priors['BLR_fwhm'][0]
            ])
        
        self.log_prior_fce = logprior_general
        self.pr_code = self.prior_create()
        self.res = {'name': 'BLR_outflow'}
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_oiii_qso_bkpl(
        self,
        peak_oiii: float,
        peak_hbeta: float,
        cont: float
    ) -> np.ndarray:
        """Setup QSO broken power-law model for [OIII] fitting."""
        self.fitted_model = QSO_models.OIII_QSO_BKPL
        self.log_prior_fce = logprior_general_scipy
        self.labels = [
            'z', 'cont', 'cont_grad', 'OIII_peak', 'OIII_out_peak',
            'Nar_fwhm', 'outflow_fwhm', 'outflow_vel',
            'BLR_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2', 'BLR_sig',
            'Hb_nar_peak', 'Hb_out_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'OIII_QSO_BKP'}
        
        pos_l = np.array([
            self.z, cont, 0.001, peak_oiii / 2, peak_oiii / 6,
            self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
            self.priors['outflow_vel'][0], peak_hbeta, self.priors['zBLR'][0],
            self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],
            self.priors['BLR_sig'][0], peak_hbeta / 4, peak_hbeta / 4
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def fitting_Halpha_OIII(
        self,
        model: str = 'gal',
        template: Optional[str] = None,
        nwalkers: int = DEFAULTS.N_WALKERS
    ) -> None:
        """
        Fit Hα + [OIII] + Hβ + [NII] + [SII] emission lines.
        
        Parameters
        ----------
        model : str
            Model configuration:
            - 'gal': Simple galaxy model
            - 'outflow': Galaxy with outflow component
            - 'BLR': BLR with outflow
            - 'BLR_simple': Simple broad line region
            - 'QSO_BKPL': QSO with broken power-law
        template : str, optional
            Iron template name for BLR models
        nwalkers : int, optional
            Number of MCMC walkers
        """
        self.model = model
        self.template = template
        self._setup_redshift_prior()
        
        # Setup zBLR prior if needed
        if 'zBLR' in self.priors and self.priors['zBLR'][2] == 0:
            self.priors['zBLR'][2] = self.z
        
        # Define fitting region (OIII + Halpha regions)
        self._prepare_data()  # First get clean data
        
        # Select OIII region
        oiii_region = np.where(
            (self.wave >= 4700 * (1 + self.z) / 1e4) &
            (self.wave <= 5100 * (1 + self.z) / 1e4)
        )[0]
        
        # Select Halpha region
        halpha_region = np.where(
            (self.wave >= (6564.52 - 170) * (1 + self.z) / 1e4) &
            (self.wave <= (6564.52 + 200) * (1 + self.z) / 1e4)
        )[0]
        
        # Combine regions
        self.fit_loc = np.append(oiii_region, halpha_region)
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        
        # Estimate peak fluxes
        peak_oiii = self._estimate_peak_flux(5007)
        peak_hal = self._estimate_peak_flux(6564.52)
        cont = np.median(self.flux[self.fit_loc])
        if cont < 0:
            cont = np.abs(cont)
        
        # Configure model
        model_configs = {
            'gal': self._setup_halpha_oiii_gal,
            'outflow': self._setup_halpha_oiii_outflow,
            'BLR': self._setup_halpha_oiii_blr,
            'BLR_simple': self._setup_halpha_oiii_blr_simple,
            'QSO_BKPL': self._setup_halpha_oiii_qso_bkpl,
        }
        
        if model not in model_configs:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(model_configs.keys())}"
            )
        
        pos_l = model_configs[model](peak_hal, peak_oiii, cont)
        
        # Validate and fit
        self._validate_initial_conditions(pos_l)
        
        if self.sampler == 'emcee':
            pos = self._initialize_walkers(pos_l, nwalkers)
            sampler = self._run_mcmc(pos)
            self._extract_chains(sampler)
            self.props = self.prop_calc()
        elif self.sampler == 'leastsq':
            self._run_leastsq(pos_l)
        else:
            raise ValueError("Sampler must be 'emcee' or 'leastsq'")
        
        # Evaluate model and calculate statistics
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        self._calculate_statistics()
    
    def _setup_halpha_oiii_gal(
        self,
        peak_hal: float,
        peak_oiii: float,
        cont: float
    ) -> np.ndarray:
        """Setup galaxy model for Halpha+OIII fitting."""
        self.fitted_model = HO_models.Halpha_OIII
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
            'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_OIII'}
        
        pos_l = np.array([
            self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
            self.priors['Nar_fwhm'][0], peak_hal * 0.15, peak_hal * 0.2,
            peak_oiii * 0.8, peak_hal * 0.2
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_oiii_outflow(
        self,
        peak_hal: float,
        peak_oiii: float,
        cont: float
    ) -> np.ndarray:
        """Setup outflow model for Halpha+OIII fitting."""
        self.fitted_model = HO_models.Halpha_OIII_outflow
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak',
            'Hbeta_peak', 'SIIr_peak', 'SIIb_peak', 'Nar_fwhm',
            'outflow_fwhm', 'outflow_vel', 'Hal_out_peak', 'NII_out_peak',
            'OIII_out_peak', 'Hbeta_out_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_OIII_outflow'}
        
        pos_l = np.array([
            self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
            peak_oiii * 0.8, peak_hal * 0.2, peak_hal * 0.2, peak_hal * 0.2,
            self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
            self.priors['outflow_vel'][0], peak_hal * 0.3, peak_hal * 0.3,
            peak_oiii * 0.2, peak_hal * 0.05
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_oiii_blr(
        self,
        peak_hal: float,
        peak_oiii: float,
        cont: float
    ) -> np.ndarray:
        """Setup BLR with outflow model for Halpha+OIII fitting."""
        self.fitted_model = HO_models.Halpha_OIII_BLR
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak',
            'Hbeta_peak', 'SIIr_peak', 'SIIb_peak', 'Nar_fwhm',
            'outflow_fwhm', 'outflow_vel', 'Hal_out_peak', 'NII_out_peak',
            'OIII_out_peak', 'Hbeta_out_peak', 'BLR_fwhm', 'zBLR',
            'BLR_Hal_peak', 'BLR_Hbeta_peak'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_OIII_BLR'}
        
        pos_l = np.array([
            self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
            peak_oiii * 0.8, peak_hal * 0.3, peak_hal * 0.2, peak_hal * 0.2,
            self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],
            self.priors['outflow_vel'][0], peak_hal * 0.3, peak_hal * 0.3,
            peak_oiii * 0.2, peak_hal * 0.1, self.priors['BLR_fwhm'][0],
            self.priors['zBLR'][0], peak_hal * 0.3, peak_hal * 0.1
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_oiii_blr_simple(
        self,
        peak_hal: float,
        peak_oiii: float,
        cont: float
    ) -> np.ndarray:
        """Setup simple BLR model for Halpha+OIII fitting."""
        if self.template:
            self.labels = [
                'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak',
                'Hbeta_peak', 'SIIr_peak', 'SIIb_peak', 'Nar_fwhm',
                'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak',
                'Fe_peak', 'Fe_FWHM'
            ]
            # Note: Would need Fe template model here
            # For now, using simple model
            self.fitted_model = HO_models.Halpha_OIII_BLR_simple
            pos_l = np.array([
                self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
                peak_oiii * 0.8, peak_hal * 0.3, peak_hal * 0.2, peak_hal * 0.2,
                self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],
                self.priors['zBLR'][0], peak_hal * 0.3, peak_hal * 0.1,
                cont, self.priors['Fe_fwhm'][0]
            ])
        else:
            self.labels = [
                'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak',
                'Hbeta_peak', 'SIIr_peak', 'SIIb_peak', 'Nar_fwhm',
                'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak'
            ]
            self.fitted_model = HO_models.Halpha_OIII_BLR_simple
            pos_l = np.array([
                self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
                peak_oiii * 0.8, peak_hal * 0.3, peak_hal * 0.2, peak_hal * 0.2,
                self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],
                self.priors['zBLR'][0], peak_hal * 0.3, peak_hal * 0.1
            ])
        
        self.log_prior_fce = logprior_general
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_OIII_BLR_simple'}
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l
    
    def _setup_halpha_oiii_qso_bkpl(
        self,
        peak_hal: float,
        peak_oiii: float,
        cont: float
    ) -> np.ndarray:
        """Setup QSO broken power-law model for Halpha+OIII fitting."""
        self.fitted_model = QSO_models.Halpha_OIII_QSO_BKPL
        self.log_prior_fce = logprior_general
        self.labels = [
            'z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak',
            'Hbeta_peak', 'Nar_fwhm', 'Hal_out_peak', 'NII_out_peak',
            'OIII_out_peak', 'Hbeta_out_peak', 'outflow_fwhm', 'outflow_vel',
            'Hal_BLR_peak', 'Hbeta_BLR_peak', 'BLR_vel', 'BLR_alp1',
            'BLR_alp2', 'BLR_sig'
        ]
        self.pr_code = self.prior_create()
        self.res = {'name': 'Halpha_OIII_BLR'}
        
        pos_l = np.array([
            self.z, cont, -0.1, peak_hal * 0.7, peak_hal * 0.3,
            peak_oiii * 0.8, peak_oiii * 0.3, self.priors['Nar_fwhm'][0],
            peak_hal * 0.2, peak_hal * 0.3, peak_oiii * 0.4, peak_oiii * 0.2,
            self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0],
            peak_hal * 0.4, peak_oiii * 0.4, self.priors['BLR_vel'][0],
            self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],
            self.priors['BLR_sig'][0]
        ])
        
        for i, label in enumerate(self.labels):
            if self.priors[label][0] != 0:
                pos_l[i] = self.priors[label][0]
        
        return pos_l

    
    def fitting_general(
        self,
        fitted_model: Callable,
        labels: List[str],
        logprior: Optional[Callable] = None,
        nwalkers: int = DEFAULTS.N_WALKERS,
        skip_check: bool = False,
        zscale: float = DEFAULTS.REDSHIFT_TOLERANCE,
        odd: bool = False
    ) -> None:
        """
        Fit a general custom model.
        
        This is a flexible method that allows you to fit any custom model function
        with your own parameter labels and prior specifications.
        
        Parameters
        ----------
        fitted_model : callable
            Model function to fit. Should accept (wavelength, *params) and return flux.
        labels : list of str
            List of parameter names in the same order as in fitted_model
        logprior : callable, optional
            Prior evaluation function. If None, uses logprior_general
        nwalkers : int, optional
            Number of MCMC walkers
        skip_check : bool, optional
            Skip initial state check in MCMC
        zscale : float, optional
            Standard deviation for redshift walker initialization
        odd : bool, optional
            If True, ensure odd number of data points (for some models)
        
        Examples
        --------
        >>> def my_model(wave, z, cont, line_amp, line_width):
        ...     # Custom model implementation
        ...     return flux
        >>> 
        >>> labels = ['z', 'cont', 'line_amp', 'line_width']
        >>> priors = {
        ...     'z': [2.5, 'normal', 2.5, 0.01],
        ...     'cont': [0.1, 'uniform', 0, 1],
        ...     'line_amp': [0.5, 'loguniform', -2, 1],
        ...     'line_width': [300, 'uniform', 100, 1000]
        ... }
        >>> 
        >>> fitter = SpectralFitter(wave, flux, error, z=2.5, priors=priors)
        >>> fitter.fit_general(my_model, labels)
        """
        self.template = None
        self.labels = labels
        self.fitted_model = fitted_model
        self.log_prior_fce = logprior if logprior else logprior_general
        
        # Setup redshift prior
        self._setup_redshift_prior()
        
        # Prepare data
        self._prepare_data()
        
        # Handle odd number of points if requested
        if odd and len(self.flux_fitloc) % 2 == 0:
            self.flux_fitloc = self.flux_fitloc[:-1]
            self.wave_fitloc = self.wave_fitloc[:-1]
            self.error_fitloc = self.error_fitloc[:-1]
        
        # Create prior code
        self.pr_code = self.prior_create()
        
        # Initialize parameters
        pos_l = self._initialize_general_parameters()
        self.pos_l = pos_l
        
        # Validate initial conditions
        self._validate_initial_conditions(pos_l)
        
        # Run fitting
        if self.sampler == 'emcee':
            pos = self._initialize_walkers(pos_l, nwalkers)
            # Override redshift initialization with custom scale
            pos[:, 0] = np.random.normal(self.z, zscale, nwalkers)
            self.pos = pos
            
            sampler = self._run_mcmc(pos, skip_check=skip_check)
            self._extract_chains(sampler)
            self.props = self.prop_calc()
            
        elif self.sampler == 'leastsq':
            self._run_leastsq(pos_l)
        else:
            raise ValueError("Sampler must be 'emcee' or 'leastsq'")
        
        # Evaluate model
        try:
            self.yeval = self.fitted_model(self.wave, *self.props['popt'])
        except Exception:
            self.yeval = np.zeros_like(self.wave)
        
        try:
            self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        except Exception:
            self.yeval_fitloc = np.zeros_like(self.wave_fitloc)
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _initialize_general_parameters(self) -> np.ndarray:
        """
        Initialize parameters for general fitting.
        
        Uses prior initial values, with smart defaults for common parameter types.
        
        Returns
        -------
        array
            Initial parameter values
        """
        pos_l = np.zeros(len(self.labels))
        
        for i, name in enumerate(self.labels):
            # Use prior initial value if set
            if self.priors[name][0] != 0:
                pos_l[i] = self.priors[name][0]
            # Smart defaults for common parameter patterns
            elif '_peak' in name:
                # Peak amplitude: use 5-10x median error
                pos_l[i] = np.nanmean(self.error_fitloc) * np.random.uniform(5, 10)
            elif name == 'cont':
                # Continuum: use 5x median flux
                pos_l[i] = np.nanmedian(self.flux_fitloc) * 5
            # Otherwise keep zero (from priors)
        
        return pos_l
    
    def fit_custom(
        self,
        model_inputs: Dict,
        model_name: str,
        nwalkers: int = DEFAULTS.N_WALKERS,
        template: Optional[str] = None
    ) -> None:
        """
        Fit using custom model configuration.
        
        This method uses a dictionary-based model specification for complex
        multi-component models.
        
        Parameters
        ----------
        model_inputs : dict
            Dictionary specifying the model configuration
        model_name : str
            Name for the custom model
        nwalkers : int, optional
            Number of MCMC walkers
        template : str, optional
            Template name if applicable
        
        Examples
        --------
        >>> model_inputs = {
        ...     'components': [
        ...         {'name': 'halpha', 'type': 'gaussian'},
        ...         {'name': 'nii', 'type': 'doublet'}
        ...     ],
        ...     'continuum': 'linear'
        ... }
        >>> 
        >>> fitter = SpectralFitter(wave, flux, error, z=2.5)
        >>> fitter.fit_custom(model_inputs, 'my_custom_model')
        """
        self.template = template
        self.model_inputs = model_inputs
        self.model_name = model_name
        
        # Setup redshift prior
        self._setup_redshift_prior()
        
        # Prepare data
        self._prepare_data()
        
        # Create and fit custom model
        self.Model = Custom_model.Model(self.model_name, model_inputs)
        self.Model.fit_to_data(
            self.wave_fitloc,
            self.flux_fitloc,
            self.error_fitloc,
            N=self.N,
            nwalkers=nwalkers,
            ncpu=1
        )
        
        # Extract results
        self.labels = self.Model.labels
        self.chains = self.Model.chains
        self.props = self.Model.props
        self.yeval = self.Model.calculate_values(self.waves)
        self.comps = self.Model.lines
    
    # =============================================================================
    # Utility Methods
    # =============================================================================
    
    def log_probability_general(self, theta: np.ndarray) -> float:
        """
        Calculate log probability for MCMC.
        
        Parameters
        ----------
        theta : array
            Parameter values
        
        Returns
        -------
        float
            Log probability
        """
        # Check prior
        lp = self.log_prior_fce(theta, self.pr_code)
        if not np.isfinite(lp):
            return -np.inf
        
        # Evaluate model
        try:
            model_flux = self.fitted_model(self.wave_fitloc, *theta)
        except Exception:
            try:
                model_flux = self.fitted_model(self.wave_fitloc, theta)
            except Exception:
                return -np.inf
        
        # Calculate log likelihood
        residuals = self.flux_fitloc - model_flux
        sigma2 = self.error_fitloc ** 2
        log_likelihood = -0.5 * np.nansum(residuals ** 2 / sigma2)
        
        return lp + log_likelihood
    
    def prior_create(self) -> np.ndarray:
        """
        Create prior code array from prior dictionary.
        
        Returns
        -------
        array
            Prior code array (n_params x 5)
        """
        prior_codes = {
            'normal': 0,
            'uniform': 1,
            'lognormal': 2,
            'loguniform': 3,
            'normal_hat': 4,
            'lognormal_hat': 5,
        }
        
        pr_code = np.zeros((len(self.labels), 5))
        
        for i, label in enumerate(self.labels):
            prior_spec = self.priors[label]
            prior_type = prior_spec[1]
            
            if prior_type not in prior_codes:
                raise ValueError(f"Unknown prior type '{prior_type}' for {label}")
            
            pr_code[i, 0] = prior_codes[prior_type]
            pr_code[i, 1] = prior_spec[2]
            pr_code[i, 2] = prior_spec[3]
            
            # Additional parameters for hat priors
            if prior_type in ('normal_hat', 'lognormal_hat'):
                pr_code[i, 3] = prior_spec[4]
                pr_code[i, 4] = prior_spec[5]
        
        return pr_code
    
    def prop_calc(self) -> Dict:
        """
        Calculate parameter properties from MCMC chains.
        
        Returns
        -------
        dict
            Dictionary with median values and uncertainties
        """
        labels = [k for k in self.chains.keys() if k != 'name']
        res_dict = {'name': self.chains['name'], 'popt': []}
        
        for label in labels:
            chain = self.chains[label]
            p50, p16, p84 = np.percentile(chain, [50, 16, 84])
            
            res_dict[label] = np.array([p50, p50 - p16, p84 - p50])
            res_dict['popt'].append(p50)
        
        return res_dict
    
    def bounds_est(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate parameter bounds from priors.
        
        Returns
        -------
        lower, upper : tuple of arrays
            Lower and upper bounds for each parameter
        """
        lower = []
        upper = []
        
        for label in self.labels:
            prior = self.priors[label]
            prior_type = prior[1]
            
            if prior_type == 'normal':
                lower.append(prior[2] - 4 * prior[3])
                upper.append(prior[2] + 4 * prior[3])
            elif prior_type == 'lognormal':
                lower.append(10 ** (prior[2] - 4 * prior[3]))
                upper.append(10 ** (prior[2] + 4 * prior[3]))
            elif prior_type == 'uniform':
                lower.append(prior[2])
                upper.append(prior[3])
            elif prior_type == 'loguniform':
                lower.append(10 ** prior[2])
                upper.append(10 ** prior[3])
            elif prior_type == 'normal_hat':
                lower.append(prior[4])
                upper.append(prior[5])
            elif prior_type == 'lognormal_hat':
                lower.append(10 ** prior[4])
                upper.append(10 ** prior[5])
            else:
                raise ValueError(f"Unknown prior type: {prior_type}")
        
        self.bounds = (np.array(lower), np.array(upper))
        return self.bounds
    
    
    def bic_calc(self, obs_wave: np.ndarray) -> float:
        """
        Calculate BIC for a given observed wavelength array.
        
        This is useful for comparing model fits on specific wavelength ranges.
        
        Parameters
        ----------
        obs_wave : array
            Observed wavelength array to calculate BIC over
        
        Returns
        -------
        float
            Bayesian Information Criterion
        """
        # Evaluate model on observed wavelengths
        yeval_obs = self.fitted_model(obs_wave, *self.props['popt'])
        
        # Find overlap with fitting region
        use = (self.wave_fitloc >= np.min(obs_wave)) & (self.wave_fitloc <= np.max(obs_wave))
        
        flux_setup = self.flux_fitloc[use]
        error_setup = self.error_fitloc[use]
        
        # Calculate chi-squared
        chi2 = np.nansum(((flux_setup - yeval_obs) / error_setup) ** 2)
        
        # Calculate BIC
        n_params = len(self.props['popt'])
        n_data = len(flux_setup)
        BIC = chi2 + n_params * np.log(n_data)
        
        return BIC
    
    def save(self, filepath: str) -> None:
        """
        Save fitter state to file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filepath: str) -> None:
        """
        Load fitter state from file.
        
        Parameters
        ----------
        filepath : str
            Input file path
        """
        import pickle
        with open(filepath, "rb") as f:
            self.__dict__ = pickle.load(f)

    def corner(self,):
        import corner

        fig = corner.corner(
            sp.unwrap_chain(self.chains), 
            labels = self.labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        return fig



# =============================================================================
# Convenience Functions
# =============================================================================

def fit_emission_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    redshift: float,
    line_region: str = 'halpha',
    model: str = 'gal',
    **kwargs
) -> Fitting:
    """
    Convenience function for fitting emission lines.
    
    This function provides a simple one-line interface for common fitting tasks.
    
    Parameters
    ----------
    wavelength : array
        Wavelength array (microns)
    flux : array
        Flux array
    error : array
        Error array
    redshift : float
        Source redshift
    line_region : str, optional
        Region to fit:
        - 'halpha': Hα + [NII] + [SII]
        - 'oiii': [OIII] + Hβ
        - 'halpha_oiii': Combined Hα and [OIII] regions
        - 'full_optical': Full optical spectrum
    model : str, optional
        Model type (depends on line_region)
    **kwargs
        Additional arguments passed to SpectralFitter constructor
        
    Returns
    -------
    SpectralFitter
        Fitted spectral fitter object with results
        
    Examples
    --------
    Simple Hα fitting:
    
    >>> fitter = fit_emission_lines(wave, flux, error, z=2.5,
    ...                             line_region='halpha', model='gal')
    >>> print(f"Hα flux: {fitter.props['Hal_peak'][0]:.3e}")
    
    [OIII] with outflow:
    
    >>> fitter = fit_emission_lines(wave, flux, error, z=2.5,
    ...                             line_region='oiii', model='outflow')
    
    Custom priors:
    
    >>> priors = {'Nar_fwhm': [400, 'uniform', 200, 800]}
    >>> fitter = fit_emission_lines(wave, flux, error, z=2.5,
    ...                             line_region='halpha', model='gal',
    ...                             priors=priors)
    """
    fitter = Fitting(
        wave=wavelength,
        flux=flux,
        error=error,
        z=redshift,
        **kwargs
    )
    
    if line_region == 'halpha':
        fitter.fit_halpha(model=model)
    elif line_region == 'oiii':
        fitter.fit_oiii(model=model)
    elif line_region == 'halpha_oiii':
        fitter.fit_halpha_oiii(model=model)
    elif line_region == 'full_optical':
        fitter.fit_full_optical(model=model)
    else:
        raise ValueError(
            f"Unknown line region: {line_region}. "
            f"Available: 'halpha', 'oiii', 'halpha_oiii', 'full_optical'"
        )
    
    return fitter


def quick_fit(
    wavelength: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    redshift: float,
    line_region: str = 'halpha',
    **kwargs
) -> Dict:
    """
    Quick fit with automatic model selection.
    
    This function performs a fit and returns only the fitted parameters,
    useful for quick analysis or batch processing.
    
    Parameters
    ----------
    wavelength : array
        Wavelength array (microns)
    flux : array
        Flux array
    error : array
        Error array
    redshift : float
        Source redshift
    line_region : str, optional
        Region to fit: 'halpha', 'oiii', 'halpha_oiii', or 'full_optical'
    **kwargs
        Additional arguments passed to SpectralFitter
        
    Returns
    -------
    dict
        Dictionary with fitted parameters and uncertainties
        
    Examples
    --------
    >>> results = quick_fit(wave, flux, error, z=2.5, line_region='halpha')
    >>> print(f"Hα = {results['Hal_peak'][0]:.2e} ± {results['Hal_peak'][1]:.2e}")
    """
    fitter = fit_emission_lines(
        wavelength, flux, error, redshift,
        line_region=line_region,
        model='gal',  # Default to simple galaxy model
        **kwargs
    )
    
    return fitter.props


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def fit_multiple_spectra(
    spectra: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    line_region: str = 'halpha',
    model: str = 'gal',
    parallel: bool = False,
    n_jobs: int = -1,
    **kwargs
) -> List[Fitting]:
    """
    Fit multiple spectra with the same configuration.
    
    Parameters
    ----------
    spectra : list of tuples
        List of (wavelength, flux, error, redshift) tuples
    line_region : str, optional
        Region to fit
    model : str, optional
        Model type
    parallel : bool, optional
        Use parallel processing (requires joblib)
    n_jobs : int, optional
        Number of parallel jobs (-1 for all CPUs)
    **kwargs
        Additional arguments passed to SpectralFitter
        
    Returns
    -------
    list of SpectralFitter
        List of fitted objects
        
    Examples
    --------
    >>> spectra_list = [
    ...     (wave1, flux1, error1, z1),
    ...     (wave2, flux2, error2, z2),
    ...     (wave3, flux3, error3, z3),
    ... ]
    >>> 
    >>> fitters = fit_multiple_spectra(spectra_list, line_region='halpha',
    ...                               model='gal', parallel=True)
    >>> 
    >>> for i, fitter in enumerate(fitters):
    ...     print(f"Spectrum {i}: χ² = {fitter.chi2:.1f}, BIC = {fitter.BIC:.1f}")
    """
    def _fit_single(spec_data):
        wave, flux, error, z = spec_data
        return fit_emission_lines(wave, flux, error, z,
                                 line_region=line_region,
                                 model=model, **kwargs)
    
    if parallel:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_single)(spec) for spec in spectra
            )
        except ImportError:
            print("Warning: joblib not available, running in serial mode")
            results = [_fit_single(spec) for spec in spectra]
    else:
        results = [_fit_single(spec) for spec in spectra]
    
    return results