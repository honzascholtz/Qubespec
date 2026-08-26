#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopic plotting utilities for emission line analysis.

This module provides comprehensive plotting functions for visualizing fitted 
spectroscopic data from the QubeSpec package. It includes functions for plotting
individual emission line regions (OIII, H-alpha, H-beta), combined spectral fits,
and spatial maps from Integral Field Unit (IFU) data.

Key Features
------------
- Emission line plotting: OIII, H-alpha, H-beta regions with model components
- IFU kinematic maps: velocity, FWHM, flux, and SNR distributions  
- Model decomposition: narrow/broad components, outflows, BLR, continuum, FeII
- Publication-ready plots with emission line identification
- Residual analysis and error visualization
- Flexible coordinate systems (rest-frame, observed-frame, offset coordinates)

Main Functions
--------------
plotting_OIII : Plot OIII emission line region with model components
plotting_Halpha : Plot H-alpha region with [NII], [SII] components  
plotting_Halpha_OIII : Combined H-alpha and OIII region plotting
OIII_map_plotting : Create 4-panel OIII kinematic maps from IFU data
Plot_results_Halpha_OIII : Comprehensive 6x3 grid of emission line maps
add_lines_paper : Add emission line identification for publication plots

Typical Usage
-------------
>>> from QubeSpec.Plotting import plots
>>> import matplotlib.pyplot as plt
>>> 
>>> # Plot OIII region from fit results
>>> fig, ax = plt.subplots()
>>> plots.plotting_OIII(fit_results, ax, errors=True, template='BG92')
>>> 
>>> # Create IFU kinematic maps
>>> fig = plots.OIII_map_plotting('Object_ID', 'maps.fits', 
...                               fwhmrange=(200, 600), velrange=(-300, 200))

Dependencies
------------
- numpy: Numerical computations
- matplotlib: Plotting and visualization  
- astropy: FITS file handling, WCS, physical units
- scipy: Curve fitting and optimization
- QubeSpec.Models: Spectral line models and templates

Created on Thu Aug 17 10:11:38 2017
Modernized: 2025
@author: jscholtz
"""

from typing import Optional, Tuple, List, Union
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
from astropy.coordinates import SkyCoord
from astropy.modeling.powerlaws import PowerLaw1D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from scipy.optimize import curve_fit
import glob

# Import local models
from ..Models import OIII_models as O_models

# Physical constants
C_LIGHT = 3.0e8  # m/s (kept as 3e8 for consistency with original)
H_PLANCK = 6.62e-34  # J·s
K_BOLTZMANN = 1.38e-23  # J/K
SPEED_OF_LIGHT_KMS = 3e5  # km/s

# Common symbols
ARROW = '$\u2193$'


# =============================================================================
# Core Gaussian Model
# =============================================================================

def gaussian(x: np.ndarray, amplitude: float, center: float, fwhm: float) -> np.ndarray:
    """
    Compute a Gaussian profile.
    
    Parameters
    ----------
    x : np.ndarray
        Wavelength array in Angstroms
    amplitude : float
        Peak amplitude of the Gaussian
    center : float
        Central wavelength in Angstroms
    fwhm : float
        Full Width at Half Maximum in km/s
    
    Returns
    -------
    np.ndarray
        Gaussian profile evaluated at x
    """
    sigma = fwhm / SPEED_OF_LIGHT_KMS * center / 2.35482
    exponent = -((x - center)**2) / (2 * sigma**2)
    return amplitude * np.exp(exponent)


# Backward compatibility alias
gauss = gaussian


# =============================================================================
# Helper Functions for Plotting
# =============================================================================

def _extract_masked_data(fluxs, wv_rest: np.ndarray, 
                         wave_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract unmasked flux data within a specified wavelength range.
    
    Parameters
    ----------
    fluxs : MaskedArray
        Flux data (may be masked)
    wv_rest : np.ndarray
        Rest-frame wavelength array
    wave_range : tuple
        (min_wave, max_wave) in Angstroms
    
    Returns
    -------
    flux : np.ndarray
        Unmasked flux values
    wv_rst_sc : np.ndarray
        Rest wavelength for unmasked points
    fit_loc_sc : np.ndarray
        Indices within specified range
    """

    flux = fluxs[~fluxs.mask].copy()
    try:
        wv_rst_sc = wv_rest[~fluxs.mask]
    except:
        wv_rst_sc = wv_rest.copy()

    fit_loc_sc = np.where((wv_rst_sc > wave_range[0]) & (wv_rst_sc < wave_range[1]))[0]
    return flux, wv_rst_sc, fit_loc_sc


def _plot_data_with_errors(ax: Axes, wv_rest: np.ndarray, fluxs, 
                           wave_range: Tuple[float, float], 
                           error: Optional[np.ndarray] = None,
                           show_errors: bool = False,
                           fit_loc: Optional[np.ndarray] = None) -> Tuple:
    """
    Plot spectroscopic data with optional error shading.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis object
    wv_rest : np.ndarray
        Rest-frame wavelength
    fluxs : MaskedArray
        Flux data
    wave_range : tuple
        (min_wave, max_wave) for plotting
    error : np.ndarray, optional
        Flux uncertainties
    show_errors : bool
        Whether to show error shading
    fit_loc : np.ndarray, optional
        Indices for full wavelength range
        
    Returns
    -------
    flux : np.ndarray
        Unmasked flux
    wv_rst_sc : np.ndarray
        Rest wavelength (unmasked)
    fit_loc_sc : np.ndarray
        Indices within range
    y_tot_rs : np.ndarray, optional
        Model evaluated at unmasked points (if available)
    """
    # Plot full masked data in grey
    if fit_loc is not None:
        ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], 
                color='grey', drawstyle='steps-mid', alpha=0.2)
    
    # Extract and plot unmasked data
    flux, wv_rst_sc, fit_loc_sc = _extract_masked_data(fluxs, wv_rest, wave_range)
    ax.plot(wv_rst_sc[fit_loc_sc], flux[fit_loc_sc], 
            drawstyle='steps-mid', label='data')
    
    # Add error shading if requested
    if show_errors and error is not None:
        ax.fill_between(wv_rst_sc[fit_loc_sc], 
                        flux[fit_loc_sc] - error[fit_loc_sc],
                        flux[fit_loc_sc] + error[fit_loc_sc], 
                        alpha=0.3, color='k', step='mid')
    
    return flux, wv_rst_sc, fit_loc_sc


def _plot_residuals(axres: Axes, wv_rst_sc: np.ndarray, residuals: np.ndarray,
                    rms: float, error: Optional[np.ndarray] = None,
                    residual_type: str = 'rms',
                    fit_loc_sc: Optional[np.ndarray] = None) -> None:
    """
    Plot fit residuals with optional error or RMS shading.
    
    Parameters
    ----------
    axres : Axes
        Axis for residual plot
    wv_rst_sc : np.ndarray
        Rest wavelength array
    residuals : np.ndarray
        Residual values (data - model)
    rms : float
        RMS of residuals
    error : np.ndarray, optional
        Flux uncertainties
    residual_type : str
        'rms' or 'error' for shading type
    fit_loc_sc : np.ndarray, optional
        Indices for data selection
    """
    axres.plot(wv_rst_sc, residuals, drawstyle='steps-mid')
    axres.set_ylim(-3 * rms, 3 * rms)
    axres.hlines(0, wv_rst_sc.min(), wv_rst_sc.max(), 
                 color='black', linestyle='dashed')
    
    if residual_type == 'rms':
        axres.fill_between(wv_rst_sc, rms, -rms, 
                          facecolor='grey', alpha=0.2, step='mid')
    elif residual_type == 'error' and error is not None and fit_loc_sc is not None:
        axres.fill_between(wv_rst_sc, 
                          residuals - error[fit_loc_sc],
                          residuals + error[fit_loc_sc], 
                          alpha=0.3, color='k', step='mid')


def _setup_emission_line_axis(ax: Axes, y_tot: np.ndarray, 
                               xlim: Tuple[float, float]) -> None:
    """
    Configure axis limits and appearance for emission line plots.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    y_tot : np.ndarray
        Total model flux
    xlim : tuple
        (xmin, xmax) for wavelength limits
    """
    flt = np.where((np.isfinite(y_tot)) & (y_tot > 0))[0]
    if len(flt) > 0:
        ymax = np.nanmax(y_tot[flt])
        ax.set_ylim(-0.1 * ymax, ymax * 1.1)
    ax.set_xlim(xlim)
    ax.tick_params(direction='in')


def _compute_shifted_wavelength(rest_wavelength: float, z: float, 
                                velocity: float = 0.0) -> float:
    """
    Compute observed wavelength with redshift and velocity shift.
    
    Parameters
    ----------
    rest_wavelength : float
        Rest-frame wavelength in Angstroms
    z : float
        Redshift
    velocity : float, optional
        Additional velocity offset in km/s
        
    Returns
    -------
    float
        Observed wavelength in microns
    """
    wave_obs = rest_wavelength * (1 + z) / 1e4
    if velocity != 0:
        wave_obs += velocity / SPEED_OF_LIGHT_KMS * wave_obs
    return wave_obs


# =============================================================================
# Emission Line Plotting Functions
# =============================================================================

def plotting_OIII(res, ax: Axes, errors: bool = False, 
                  template: Union[int, str] = 0,
                  residual: str = 'none', 
                  axres: Optional[Axes] = None) -> None:
    """
    Plot OIII emission line region with fitted components.
    
    This function creates a detailed plot of the OIII emission line region
    (4700-5050 Å rest frame) showing the data, total model fit, and individual
    spectral components including narrow and outflow OIII lines, H-beta,
    iron templates, and continuum.
    
    Parameters
    ----------
    res : FitResult
        Object containing fit results with attributes:
        - props: dict of fit parameters including redshift, line peaks, FWHM
        - wave: observed wavelength array in microns
        - fluxs: flux array (may be masked) in arbitrary units
        - error: flux uncertainties
        - yeval: evaluated model flux
    ax : Axes
        Matplotlib axis for main plot
    errors : bool, optional
        Whether to show error shading on the data, by default False
    template : int or str, optional
        FeII template: 0 (none), 'BG92', 'Tsuzuki', or 'Veron', by default 0
    residual : str, optional
        Residual plot type: 'none', 'rms', or 'error', by default 'none'
    axres : Axes, optional
        Axis for residual plot (required if residual != 'none')
        
    Notes
    -----
    The function plots the following components in different colors/styles:
    - Data: black step function
    - Total model: red dashed line
    - Narrow OIII: green dashed
    - H-beta: orange dashed  
    - Outflow components: blue dashed
    - Continuum: lime green dashed
    - FeII template: magenta dashed
    - BLR components: various colors/styles
    """
    # Extract fit parameters and data
    sol = res.props
    keys = list(sol.keys())
    z = sol['z'][0]  # Redshift
    wave = res.wave  # Observed wavelength in microns
    fluxs = res.fluxs  # Flux data (potentially masked)
    error = res.error  # Flux uncertainties
    
    # Define wavelength range for OIII region (4700-5200 Å rest frame)
    wv_rest = wave / (1 + z) * 1e4  # Convert to rest-frame wavelength in Angstroms
    fit_loc = np.where((wv_rest > 4700) & (wv_rest < 5200))[0]
    wave_range = (4700, 5200)
    
    # Plot observed data with error handling for masked arrays
    try:
        flux, wv_rst_sc, fit_loc_sc = _plot_data_with_errors(
            ax, wv_rest, fluxs, wave_range, error, errors, fit_loc
        )
        # Extract model at unmasked wavelengths for residual calculation
        y_tot_rs = res.yeval[~fluxs.mask][fit_loc_sc]
    except (AttributeError, TypeError):
        # Fallback for non-masked data or older data formats
        ax.plot(wv_rest, res.flux, drawstyle='steps-mid', label='data')
        flux, wv_rst_sc, fit_loc_sc = None, None, None
        y_tot_rs = None
    
    # Plot total model fit (red dashed line)
    y_tot = res.yeval[fit_loc]
    ax.plot(wv_rest[fit_loc], y_tot, 'r--', label='Total model')
    
    # Configure axis appearance and limits
    _setup_emission_line_axis(ax, y_tot, (4700, 5050))
    
    # Calculate emission line wavelengths at object redshift
    oiii_r = _compute_shifted_wavelength(5008.24, z)  # [OIII] λ5008 in microns
    oiii_b = _compute_shifted_wavelength(4960.3, z)   # [OIII] λ4960 in microns
    fwhm = sol['Nar_fwhm'][0]  # Narrow line FWHM in km/s
    
    # Plot narrow OIII doublet (3:1 flux ratio, green dashed lines)
    ax.plot(wv_rest[fit_loc], 
            gaussian(wave[fit_loc], sol['OIII_peak'][0] / 3, oiii_b, fwhm) + 
            gaussian(wave[fit_loc], sol['OIII_peak'][0], oiii_r, fwhm),
            color='green', linestyle='dashed', label='[OIII] narrow')
    
    # Plot H-beta narrow component if present (orange dashed)
    if 'Hbeta_peak' in keys:
        hbeta = _compute_shifted_wavelength(4862.6, z)
        ax.plot(wv_rest[fit_loc], 
                gaussian(wave[fit_loc], sol['Hbeta_peak'][0], hbeta, fwhm),
                color='orange', linestyle='dashed', label='H$\\beta$ narrow')
    
    # Plot outflow/broad components if present (blue dashed)
    if 'outflow_fwhm' in keys:
        # Calculate outflow wavelengths with velocity offset
        oiii_r_out = _compute_shifted_wavelength(5008.24, z, sol['outflow_vel'][0])
        oiii_b_out = _compute_shifted_wavelength(4960.3, z, sol['outflow_vel'][0])
        fwhm_out = sol['outflow_fwhm'][0]  # Outflow FWHM in km/s
        
        # Plot OIII outflow doublet
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['OIII_out_peak'][0] / 3, oiii_b_out, fwhm_out) +
                gaussian(wave[fit_loc], sol['OIII_out_peak'][0], oiii_r_out, fwhm_out),
                color='blue', linestyle='dashed', label='[OIII] outflow')
        
        # Plot H-beta outflow component if present
        if 'Hbeta_out_peak' in keys:
            hbeta_out = _compute_shifted_wavelength(4862.6, z, sol['outflow_vel'][0])
            ax.plot(wv_rest[fit_loc],
                    gaussian(wave[fit_loc], sol['Hbeta_out_peak'][0], hbeta_out, fwhm_out),
                    color='blue', linestyle='dashed', alpha=0.7, label='H$\\beta$ outflow')
    
    # Plot continuum and FeII templates if iron lines are fitted
    if 'Fe_peak' in keys:
        # Power-law continuum (lime green dashed)
        ax.plot(wv_rest[fit_loc], 
                PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0], oiii_r, 
                                   alpha=sol['cont_grad'][0]),
                linestyle='dashed', color='limegreen', label='Continuum')
        
        # FeII template selection and plotting (magenta dashed)
        fe_templates = {
            'BG92': O_models.Fem.FeII_BG92,      # Boroson & Green 1992
            'Tsuzuki': O_models.Fem.FeII_Tsuzuki, # Tsuzuki et al. 2006
            'Veron': O_models.Fem.FeII_Veron     # Veron-Cetty et al. 2004
        }
        
        if template in fe_templates:
            fe_model = fe_templates[template](wave[fit_loc], z, sol['Fe_fwhm'][0])
            ax.plot(wv_rest[fit_loc], sol['Fe_peak'][0] * fe_model,
                    linestyle='dashed', color='magenta', label=f'FeII {template}')
    
    # Plot separate narrow-line components (for complex fits)
    if 'Hb_nar_peak' in keys:
        hbeta = _compute_shifted_wavelength(4862.6, z)
        hbeta_out = _compute_shifted_wavelength(4862.6, z, sol['outflow_vel'][0])
        
        # Separate narrow and outflow H-beta components (dotted lines)
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['Hb_nar_peak'][0], hbeta, sol['Nar_fwhm'][0]),
                color='orange', linestyle='dotted', alpha=0.7)
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['Hb_out_peak'][0], hbeta_out, sol['outflow_fwhm'][0]),
                color='orange', linestyle='dotted', alpha=0.7)
    
    # Plot Broad Line Region (BLR) components for QSOs
    if 'zBLR' in keys:
        hbeta_blr = _compute_shifted_wavelength(4862.6, sol['zBLR'][0])
        
        if 'BLR_alp1' in keys:
            # Double-peaked BLR profile using broken power-law
            from ..Models.QSO_models import BKPLG
            hbeta_blr_model = BKPLG(wave[fit_loc], sol['BLR_peak'][0], hbeta_blr,
                                    sol['BLR_sig'][0], sol['BLR_alp1'][0], sol['BLR_alp2'][0])
            ax.plot(wv_rest[fit_loc], hbeta_blr_model, 
                    color='orange', linestyle='dashed', alpha=0.8, label='H$\\beta$ BLR')
        else:
            # Simple Gaussian BLR component
            ax.plot(wv_rest[fit_loc],
                    gaussian(wave[fit_loc], sol['BLR_Hbeta_peak'][0], hbeta_blr, sol['BLR_fwhm'][0]),
                    color='orange', linestyle='dashed', alpha=0.8, label='H$\\beta$ BLR')
    
    # Plot residuals if requested and data available
    if residual != 'none' and axres is not None and y_tot_rs is not None:
        residuals = flux[fit_loc_sc] - y_tot_rs
        rms = np.sqrt(np.mean(residuals**2))  # Root mean square of residuals
        _plot_residuals(axres, wv_rst_sc[fit_loc_sc], residuals, rms, 
                       error, residual, fit_loc_sc)


# =============================================================================
# H-alpha Region Plotting
# =============================================================================

def plotting_Halpha(res, ax: Axes, errors: bool = False,
                    residual: str = 'none', 
                    axres: Optional[Axes] = None) -> None:
    """
    Plot H-alpha emission line region with fitted components.
    
    This function creates a detailed plot of the H-alpha emission line region
    (around 6564 Å rest frame) showing the data, total model fit, and individual
    spectral components including H-alpha, [NII] doublet, [SII] doublet, and
    any outflow or BLR components.
    
    Parameters
    ----------
    res : FitResult
        Object containing fit results with spectroscopic fitting parameters
    ax : Axes
        Matplotlib axis for main plot
    errors : bool, optional
        Whether to show error shading on the data, by default False
    residual : str, optional
        Residual plot type: 'none', 'rms', or 'error', by default 'none'
    axres : Axes, optional
        Axis for residual plot (required if residual != 'none')
        
    Notes
    -----
    The function plots the following components in different colors/styles:
    - Data: black step function
    - Total model: red dashed line
    - H-alpha: orange dashed
    - [NII] λλ6549,6585: dark green dashed (3:1 ratio)
    - [SII] λλ6717,6731: dark blue dashed
    - Outflow components: magenta dashed
    - BLR H-alpha: various colors/styles
    """
    # Extract fit parameters and data
    sol = res.props
    z = sol['popt'][0]  # Redshift from fit
    keys = list(sol.keys())  # Available fit parameters
    wave = res.wave  # Observed wavelength in microns
    fluxs = res.fluxs  # Flux data (potentially masked)
    error = res.error  # Flux uncertainties
    
    # Define wavelength range for H-alpha region (6000-7000 Å rest frame)
    wv_rest = wave / (1 + z) * 1e4  # Convert to rest-frame wavelength in Angstroms
    fit_loc = np.where((wv_rest > 6000) & (wv_rest < 7500))[0]  # Extended range for [SII]
    wave_range = (6000, 7000)  # Plotting range around H-alpha
    
    # Plot observed data with error handling for masked arrays
    try:
        flux, wv_rst_sc, fit_loc_sc = _plot_data_with_errors(
            ax, wv_rest, fluxs, wave_range, error, errors, fit_loc
        )
        # Extract model at unmasked wavelengths for residual calculation
        y_tot_rs = res.yeval[~fluxs.mask][fit_loc_sc]
    except (AttributeError, TypeError):
        # Fallback for non-masked data or older data formats
        ax.plot(wv_rest, res.flux, drawstyle='steps-mid', label='data')
        flux, wv_rst_sc, fit_loc_sc = None, None, None
        y_tot_rs = None
    
    # Plot total model fit (red dashed line)
    y_tot = res.yeval[fit_loc]
    ax.plot(wv_rest[fit_loc], y_tot, 'r--', label='Total model')
    
    # Configure axis appearance and limits (focused on H-alpha)
    ax.set_ylim(-0.1 * max(y_tot), max(y_tot) * 1.1)
    ax.set_xlim(6564.52 - 250, 6564.52 + 250)  # ±250 Å around H-alpha
    ax.tick_params(direction='in')
    
    # Calculate emission line wavelengths at object redshift
    hal_wv = _compute_shifted_wavelength(6564.52, z)   # H-alpha in microns
    nii_r = _compute_shifted_wavelength(6585.27, z)    # [NII] λ6585 (red component)
    nii_b = _compute_shifted_wavelength(6549.86, z)    # [NII] λ6549 (blue component)
    sii_r = _compute_shifted_wavelength(6732.67, z)    # [SII] λ6732 (red component)  
    sii_b = _compute_shifted_wavelength(6718.29, z)    # [SII] λ6718 (blue component)
    
    # Plot narrow H-alpha component (orange dashed)
    fwhm = sol['Nar_fwhm'][0]  # Narrow line FWHM in km/s
    ax.plot(wv_rest[fit_loc], 
            gaussian(wave[fit_loc], sol['Hal_peak'][0], hal_wv, fwhm),
            color='orange', linestyle='dashed', label='H$\\alpha$ narrow')
    
    # Plot [NII] doublet if present (dark green dashed, 3:1 flux ratio)
    if 'NII_peak' in keys:
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_peak'][0], nii_r, fwhm),
                color='darkgreen', linestyle='dashed', label='[NII] $\\lambda$6585')
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_peak'][0] / 3, nii_b, fwhm),
                color='darkgreen', linestyle='dashed', alpha=0.7, label='[NII] $\\lambda$6549')
    
    # Plot [SII] doublet if present (dark blue dashed)
    if 'SIIr_peak' in keys:
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['SIIr_peak'][0], sii_r, fwhm),
                color='darkblue', linestyle='dashed', label='[SII] $\\lambda$6732')
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['SIIb_peak'][0], sii_b, fwhm),
                color='darkblue', linestyle='dashed', alpha=0.7, label='[SII] $\\lambda$6718')
    
    # Plot BLR H-alpha component if present (for QSOs)
    if 'zBLR' in keys:
        blr_wv = _compute_shifted_wavelength(6564.52, sol['zBLR'][0])
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['BLR_Hal_peak'][0], blr_wv, sol['BLR_fwhm'][0]),
                color='darkorange', linestyle='dashed', alpha=0.8, label='H$\\alpha$ BLR')
    
    # Plot outflow components if present (magenta dashed)
    if 'Hal_out_peak' in keys:
        out_fwhm = sol['outflow_fwhm'][0]  # Outflow FWHM in km/s
        out_vel = sol['outflow_vel'][0]    # Outflow velocity in km/s
        
        # Calculate outflow wavelengths with velocity offset
        hal_wv_out = _compute_shifted_wavelength(6564.52, z, out_vel)
        nii_r_out = _compute_shifted_wavelength(6585.27, z, out_vel)
        nii_b_out = _compute_shifted_wavelength(6549.86, z, out_vel)
        
        # Plot H-alpha outflow
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['Hal_out_peak'][0], hal_wv_out, out_fwhm),
                color='magenta', linestyle='dashed', label='H$\\alpha$ outflow')
        
        # Plot [NII] outflow doublet
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_out_peak'][0], nii_r_out, out_fwhm),
                color='magenta', linestyle='dashed', alpha=0.7)
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_out_peak'][0] / 3, nii_b_out, out_fwhm),
                color='magenta', linestyle='dashed', alpha=0.5)
    
    # Plot double-peaked BLR profile if present (advanced QSO modeling)
    if 'BLR_alp1' in keys:
        from ..Models.QSO_models import BKPLG
        ha_blr_wv = _compute_shifted_wavelength(6563, sol['zBLR'][0])
        ha_blr = BKPLG(wave[fit_loc], sol['BLR_Hal_peak'][0], ha_blr_wv,
                       sol['BLR_sig'][0], sol['BLR_alp1'][0], sol['BLR_alp2'][0])
        ax.plot(wv_rest[fit_loc], ha_blr, 
                color='magenta', linestyle='dashed', alpha=0.8, label='H$\\alpha$ BLR (double-peak)')
    
    # Plot residuals if requested and data available
    if residual != 'none' and axres is not None and y_tot_rs is not None:
        residuals = flux[fit_loc_sc] - y_tot_rs
        rms = np.sqrt(np.mean(residuals**2))
        axres.set_ylim(-2 * rms, 2 * rms)  # Tighter limits for H-alpha residuals
        _plot_residuals(axres, wv_rst_sc[fit_loc_sc], residuals, rms,
                       error, residual, fit_loc_sc)


# =============================================================================
# Combined H-alpha + OIII Plotting
# =============================================================================

def plotting_Halpha_OIII(res, ax: Axes, errors: bool = False,
                         residual: str = 'none',
                         axres: Optional[Axes] = None,
                         template: Union[int, str] = 0) -> None:
    """
    Plot combined H-alpha and OIII region with all emission lines.
    
    This function is useful for visualizing full optical spectra with
    multiple emission line regions simultaneously.
    
    Parameters
    ----------
    res : FitResult
        Object containing fit results
    ax : Axes
        Matplotlib axis for main plot
    errors : bool, optional
        Whether to show error shading
    residual : str, optional
        Residual plot type: 'none', 'rms', or 'error'
    axres : Axes, optional
        Axis for residual plot
    template : int or str, optional
        FeII template for continuum modeling
    """
    sol = res.props
    keys = list(sol.keys())
    z = sol['popt'][0]
    wave = res.wave
    fluxs = res.fluxs
    error = res.error
    
    # Full wavelength range
    wv_rest = wave / (1 + z) * 1e4
    fit_loc = np.where((wv_rest > 100) & (wv_rest < 16000))[0]
    wave_range = (4700, 5200)
    
    # Plot data
    try:
        flux, wv_rst_sc, fit_loc_sc = _plot_data_with_errors(
            ax, wv_rest, fluxs, wave_range, error, errors, fit_loc
        )
        y_tot_rs = res.yeval[~fluxs.mask][fit_loc_sc]
    except Exception as e:
        ax.plot(wv_rest, res.flux, drawstyle='steps-mid', label='data')
        print(f"Warning in plotting: {e}")
        flux, wv_rst_sc, fit_loc_sc = None, None, None
        y_tot_rs = None
    
    # Plot total model
    y_tot = res.yeval[fit_loc]
    ax.plot(wv_rest[fit_loc], y_tot, 'r--')
    
    # Configure axis
    ax.set_ylim(-0.1 * max(y_tot), max(y_tot) * 1.1)
    ax.tick_params(direction='in')
    
    # H-alpha region lines
    hal_wv = _compute_shifted_wavelength(6564.52, z)
    nii_r = _compute_shifted_wavelength(6585.27, z)
    nii_b = _compute_shifted_wavelength(6549.86, z)
    sii_r = _compute_shifted_wavelength(6732.67, z)
    sii_b = _compute_shifted_wavelength(6718.29, z)
    
    fwhm = sol['Nar_fwhm'][0]
    ax.plot(wv_rest[fit_loc],
            gaussian(wave[fit_loc], sol['Hal_peak'][0], hal_wv, fwhm),
            color='orange', linestyle='dashed')
    
    if 'NII_peak' in keys:
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_peak'][0], nii_r, fwhm),
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['NII_peak'][0] / 3, nii_b, fwhm),
                color='darkgreen', linestyle='dashed')
    
    if 'SIIr_peak' in keys:
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['SIIr_peak'][0], sii_r, fwhm),
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['SIIb_peak'][0], sii_b, fwhm),
                color='darkblue', linestyle='dashed')
    
    # OIII region lines
    oiii_r = _compute_shifted_wavelength(5008.24, z)
    oiii_b = _compute_shifted_wavelength(4960.3, z)
    
    ax.plot(wv_rest[fit_loc],
            gaussian(wave[fit_loc], sol['OIII_peak'][0] / 3, oiii_b, fwhm) +
            gaussian(wave[fit_loc], sol['OIII_peak'][0], oiii_r, fwhm),
            color='green', linestyle='dashed')
    
    # H-beta
    hbeta = _compute_shifted_wavelength(4862.6, z)
    ax.plot(wv_rest[fit_loc],
            gaussian(wave[fit_loc], sol['Hbeta_peak'][0], hbeta, fwhm),
            color='orange', linestyle='dashed')
    
    # FeII templates
    if 'Fe_peak' in keys:
        ax.plot(wv_rest[fit_loc],
                PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0], oiii_r,
                                   alpha=sol['cont_grad'][0]),
                linestyle='dashed', color='limegreen')
        
        fe_templates = {
            'BG92': O_models.Fem.FeII_BG92,
            'Tsuzuki': O_models.Fem.FeII_Tsuzuki,
            'Veron': O_models.Fem.FeII_Veron
        }
        
        if template in fe_templates:
            fe_model = fe_templates[template](wave[fit_loc], z, sol['Fe_fwhm'][0])
            ax.plot(wv_rest[fit_loc], sol['Fe_peak'][0] * fe_model,
                    linestyle='dashed', color='magenta')
    
    # [OI] line
    if 'OI_peak' in keys:
        oi = _compute_shifted_wavelength(6302.0, z)
        ax.plot(wv_rest[fit_loc],
                gaussian(wave[fit_loc], sol['OI_peak'][0], oi, fwhm),
                color='green', linestyle='dashed')
    
    # Outflow components
    if 'outflow_vel' in keys:
        out_fwhm = sol['outflow_fwhm'][0]
        out_vel = sol['outflow_vel'][0]
        
        # Compute shifted wavelengths
        hal_out = _compute_shifted_wavelength(6564.52, z, out_vel)
        nii_r_out = _compute_shifted_wavelength(6585.27, z, out_vel)
        nii_b_out = _compute_shifted_wavelength(6549.86, z, out_vel)
        oiii_r_out = _compute_shifted_wavelength(5008.24, z, out_vel)
        oiii_b_out = _compute_shifted_wavelength(4960.3, z, out_vel)
        hbeta_out = _compute_shifted_wavelength(4862.6, z, out_vel)
        oi_out = _compute_shifted_wavelength(6302.0, z, out_vel)
        sii_r_out = _compute_shifted_wavelength(6732.67, z, out_vel)
        sii_b_out = _compute_shifted_wavelength(6718.29, z, out_vel)
        
        # Compute total outflow
        outflow = (
            gaussian(wave[fit_loc], sol['Hal_out_peak'][0], hal_out, out_fwhm) +
            gaussian(wave[fit_loc], sol['NII_out_peak'][0], nii_r_out, out_fwhm) +
            gaussian(wave[fit_loc], sol['NII_out_peak'][0] / 3, nii_b_out, out_fwhm) +
            gaussian(wave[fit_loc], sol['OIII_out_peak'][0], oiii_r_out, out_fwhm) +
            gaussian(wave[fit_loc], sol['OIII_out_peak'][0] / 3, oiii_b_out, out_fwhm) +
            gaussian(wave[fit_loc], sol['Hbeta_out_peak'][0], hbeta_out, out_fwhm)
        )
        
        if 'OI_peak' in keys:
            outflow += gaussian(wave[fit_loc], sol['OI_out_peak'][0], oi_out, out_fwhm)
        
        ax.plot(wv_rest[fit_loc], outflow, color='magenta', linestyle='dashed')
    
    # BLR components
    if 'BLR_fwhm' in keys:
        blr_fwhm = sol['BLR_fwhm'][0]
        blr_hal = _compute_shifted_wavelength(6564.52, sol['zBLR'][0])
        blr_hbe = _compute_shifted_wavelength(4862.6, sol['zBLR'][0])
        
        hal_blr = gaussian(wave[fit_loc], sol['BLR_Hal_peak'][0], blr_hal, blr_fwhm)
        hbe_blr = gaussian(wave[fit_loc], sol['BLR_Hbeta_peak'][0], blr_hbe, blr_fwhm)
        
        ax.plot(wv_rest[fit_loc], hal_blr + hbe_blr, 
                linestyle='dashed', color='lightblue')


# =============================================================================
# General Plotting Function
# =============================================================================

def plotting_general(wave: np.ndarray, fluxs, ax: Axes, sol: dict,
                     fitted_model, error: np.ndarray = np.array([1]),
                     residual: str = 'none', 
                     axres: Union[str, Axes] = 'none') -> None:
    """
    General-purpose plotting function for arbitrary spectral fits.
    
    Parameters
    ----------
    wave : np.ndarray
        Observed wavelength array
    fluxs : MaskedArray
        Flux data
    ax : Axes
        Main plot axis
    sol : dict
        Solution dictionary with 'popt' key
    fitted_model : callable
        Model function that takes (wave, *popt)
    error : np.ndarray, optional
        Flux uncertainties
    residual : str, optional
        Residual type: 'none', 'rms', or 'error'
    axres : Axes or str, optional
        Residual plot axis
    """
    popt = sol['popt']
    z = popt[0]
    
    wv_rest = wave / (1 + z) * 1e4
    fit_loc = np.where((wv_rest > 100) & (wv_rest < 16000))[0]
    
    # Plot data
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], 
            color='grey', drawstyle='steps-mid', alpha=0.2)
    
    flux = fluxs.data[~fluxs.mask]
    wv_rst_sc = wv_rest[~fluxs.mask]
    fit_loc_sc = np.where((wv_rst_sc > 100) & (wv_rst_sc < 16000))[0]
    
    ax.plot(wv_rst_sc[fit_loc_sc], flux[fit_loc_sc], drawstyle='steps-mid')
    
    # Plot model
    y_tot = fitted_model(wave[fit_loc], *popt)
    y_tot_rs = fitted_model(wv_rst_sc[fit_loc_sc] * (1 + z) / 1e4, *popt)
    
    ax.plot(wv_rest[fit_loc], y_tot, 'r--')
    ax.set_ylim(-0.1 * max(y_tot), max(y_tot) * 1.1)
    ax.tick_params(direction='in')
    
    # Plot residuals
    if axres != 'none' and isinstance(axres, Axes):
        residuals = flux[fit_loc_sc] - y_tot_rs
        rms = np.sqrt(np.mean(residuals**2))
        _plot_residuals(axres, wv_rst_sc[fit_loc_sc], residuals, rms,
                       error, residual, fit_loc_sc)


# =============================================================================
# Full Optical Spectrum Plotting
# =============================================================================

def plotting_optical(res, ax: Axes, error: np.ndarray = np.array([1]),
                     template: Union[int, str] = 0,
                     residual: str = 'none',
                     axres: Optional[Axes] = None) -> None:
    """
    Plot full optical spectrum with all emission line components.
    
    Parameters
    ----------
    res : FitResult
        Fit result object
    ax : Axes
        Main plot axis
    error : np.ndarray, optional
        Flux uncertainties
    template : int or str, optional
        FeII template selection
    residual : str, optional
        Residual type
    axres : Axes, optional
        Residual plot axis
    """
    sol = res.props
    keys = list(sol.keys())
    z = sol['popt'][0]
    wave = res.wave
    fluxs = res.fluxs
    error = res.error
    
    wv_rest = wave / (1 + z) * 1e4
    
    # Plot full spectrum
    ax.plot(wv_rest, fluxs.data, color='grey', drawstyle='steps-mid', alpha=0.2)
    
    flux = fluxs.data[~fluxs.mask]
    wv_rst_sc = wv_rest[~fluxs.mask]
    
    ax.plot(wv_rst_sc, flux, drawstyle='steps-mid', label='data')
    if len(error) != 1:
        ax.fill_between(wv_rst_sc, flux - error, flux + error, 
                        alpha=0.3, color='k')
    
    # Plot model
    y_tot = res.yeval
    y_tot_rs = res.yeval[~fluxs.mask]
    ax.plot(wv_rest, y_tot, 'r--')
    
    # Configure axis
    ax.set_ylim(-0.1 * np.nanmax(y_tot), np.nanmax(y_tot) * 1.1)
    ax.tick_params(direction='in')
    ax.set_xlim(4700, 5050)
    
    # Plot OIII components
    oiii_r = _compute_shifted_wavelength(5008.24, z)
    oiii_b = _compute_shifted_wavelength(4960.3, z)
    fwhm = sol['Nar_fwhm'][0]
    
    ax.plot(wv_rest,
            gaussian(wave, sol['OIII_peak'][0] / 3, oiii_b, fwhm) +
            gaussian(wave, sol['OIII_peak'][0], oiii_r, fwhm),
            color='green', linestyle='dashed')
    
    # Plot H-beta
    if 'Hbeta_peak' in keys:
        hbeta = _compute_shifted_wavelength(4862.6, z)
        ax.plot(wv_rest,
                gaussian(wave, sol['Hbeta_peak'][0], hbeta, fwhm),
                color='orange', linestyle='dashed')
    
    # Plot outflow components
    if 'outflow_fwhm' in keys:
        oiii_r_out = _compute_shifted_wavelength(5008.24, z, sol['outflow_vel'][0])
        oiii_b_out = _compute_shifted_wavelength(4960.3, z, sol['outflow_vel'][0])
        fwhm_out = sol['outflow_fwhm'][0]
        
        ax.plot(wv_rest,
                gaussian(wave, sol['OIII_out_peak'][0] / 3, oiii_b_out, fwhm_out) +
                gaussian(wave, sol['OIII_out_peak'][0], oiii_r_out, fwhm_out),
                color='blue', linestyle='dashed')
        
        if 'Hbeta_out_peak' in keys:
            hbeta_out = _compute_shifted_wavelength(4862.6, z, sol['outflow_vel'][0])
            ax.plot(wv_rest,
                    gaussian(wave, sol['Hbeta_out_peak'][0], hbeta_out, fwhm_out),
                    color='blue', linestyle='dashed')
    
    # Plot residuals
    if residual != 'none' and axres is not None:
        residuals = flux - y_tot_rs
        rms = np.sqrt(np.mean(residuals**2))
        axres.set_ylim(-3 * rms, 3 * rms)
        axres.hlines(0, 4600, 5600, color='black', linestyle='dashed')
        _plot_residuals(axres, wv_rst_sc, residuals, rms, error, residual)


# =============================================================================
# IFU Map Plotting
# =============================================================================

def OIII_map_plotting(ID: str, path: str, 
                      fwhmrange: Tuple[float, float] = (300, 500),
                      velrange: Tuple[float, float] = (-400, 100),
                      flux_max: float = 0) -> Figure:
    """
    Create OIII kinematic maps from IFU data.
    
    This function generates a 2x2 grid of maps showing the spatial distribution
    of OIII emission properties across the field of view from Integral Field
    Unit (IFU) spectroscopy data.
    
    Parameters
    ----------
    ID : str
        Object identifier for the title/labeling
    path : str
        Path to FITS file containing emission line maps with extensions:
        - 'OIII': flux and SNR maps  
        - 'OIII_kin': velocity and FWHM maps
        - 'PRIMARY': header with astrometry
    fwhmrange : tuple, optional
        (min, max) for FWHM colorbar in km/s, by default (300, 500)
    velrange : tuple, optional
        (min, max) for velocity colorbar in km/s, by default (-400, 100)
    flux_max : float, optional
        Maximum flux for colorbar (0 = auto-scale from central pixel), by default 0
        
    Returns
    -------
    Figure
        Matplotlib figure with 4-panel OIII kinematic map
        
    Notes
    -----
    The four panels show:
    1. Flux map (top-left): OIII emission line flux
    2. Velocity map (bottom-left): Line-of-sight velocity offsets 
    3. FWHM map (bottom-right): Line width (velocity dispersion)
    4. SNR map (top-right): Signal-to-noise ratio of the line detection
    
    All maps use arcsecond offset coordinates from the central position.
    """
    # Load IFU data from FITS file
    with pyfits.open(path, memmap=False) as hdulist:
        map_oiii = hdulist['OIII'].data      # Shape: (2, ny, nx) - [SNR, flux]
        map_oiii_ki = hdulist['OIII_kin'].data  # Shape: (2, ny, nx) - [vel, fwhm]
        IFU_header = hdulist['PRIMARY'].header
    
    # Extract central position and determine flux scaling
    x = int(IFU_header['X_cent'])  # Central x pixel
    y = int(IFU_header['Y_cent'])  # Central y pixel
    
    if flux_max == 0:
        flux_max = map_oiii[1, y, x]  # Use central pixel flux as maximum
    
    # Calculate coordinate system for plotting
    deg_per_pix = IFU_header['CDELT2']  # Degrees per pixel
    arc_per_pix = deg_per_pix * 3600    # Arcseconds per pixel
    
    # Calculate offset coordinates from center
    offsets_low = -np.array([x, y])
    offsets_hig = np.array(map_oiii[1].shape[1:3]) - np.array([x, y])
    
    # Define extent for matplotlib imshow (xmin, xmax, ymin, ymax)
    lim = np.array([offsets_low[0], offsets_hig[0],
                    offsets_low[1], offsets_hig[1]])
    lim_sc = lim * arc_per_pix  # Convert to arcseconds
    
    # Create 2x2 figure layout for the four maps
    f = plt.figure(figsize=(10, 10))
    ax1 = f.add_axes([0.1, 0.55, 0.38, 0.38])   # Top-left: Flux map
    ax2 = f.add_axes([0.1, 0.1, 0.38, 0.38])    # Bottom-left: Velocity map  
    ax3 = f.add_axes([0.55, 0.1, 0.38, 0.38])   # Bottom-right: FWHM map
    ax4 = f.add_axes([0.55, 0.55, 0.38, 0.38])  # Top-right: SNR map
    
    # Flux map (top-left): Shows spatial distribution of OIII flux
    flx = ax1.imshow(map_oiii[1], vmax=flux_max, origin='lower', 
                     extent=lim_sc, cmap='viridis')
    ax1.set_title('OIII Flux Map')
    colorbar(f, ax1, flx, 'Flux (arbitrary units)')
    
    # Velocity map (bottom-left): Shows line-of-sight velocity field 
    vel = ax2.imshow(map_oiii_ki[0], cmap='coolwarm', origin='lower',
                     vmin=velrange[0], vmax=velrange[1], extent=lim_sc)
    ax2.set_title('OIII Velocity Offset Map')
    ax2.set_xlabel('RA Offset (arcsec)')
    ax2.set_ylabel('Dec Offset (arcsec)')
    colorbar(f, ax2, vel, 'Velocity (km/s)')
    
    # FWHM map (bottom-right): Shows velocity dispersion/line width
    fw = ax3.imshow(map_oiii_ki[1], vmin=fwhmrange[0], vmax=fwhmrange[1],
                    origin='lower', extent=lim_sc, cmap='plasma')
    ax3.set_title('OIII FWHM Map')
    ax3.set_xlabel('RA Offset (arcsec)')
    colorbar(f, ax3, fw, 'FWHM (km/s)')
    
    # SNR map (top-right): Shows signal-to-noise ratio of line detection
    snr = ax4.imshow(map_oiii[0], vmin=3, vmax=20, origin='lower', 
                     extent=lim_sc, cmap='hot')
    ax4.set_title('OIII SNR Map')
    colorbar(f, ax4, snr, 'SNR')
    
    return f


def Plot_results_Halpha_OIII(file: str, center: List[int] = [27, 27],
                             fwhmrange: Tuple[float, float] = (100, 500),
                             velrange: Tuple[float, float] = (-100, 100),
                             flux_max: float = 0,
                             o3offset: float = 0,
                             extent: np.ndarray = np.array([0])) -> Tuple[Figure, np.ndarray]:
    """
    Create comprehensive emission line map plots from IFU data.
    
    This function generates a detailed 6×3 grid of maps showing the spatial
    distribution of multiple emission lines from Integral Field Unit (IFU)
    spectroscopy. Each row shows a different emission line, with columns 
    showing SNR, flux, and kinematics.
    
    Parameters
    ----------
    file : str
        Path to FITS file containing emission line maps with extensions:
        - 'Halpha', 'NII', 'Hbeta', 'OIII', 'OI', 'SIIr', 'SIIb': line maps
        - 'Hal_kin': H-alpha kinematics  
        - 'Av': dust extinction map
        - 'PRIMARY': header with astrometry
    center : list, optional
        [x, y] pixel coordinates of field center, by default [27, 27]
    fwhmrange : tuple, optional
        (min, max) FWHM range for colorbars in km/s, by default (100, 500)
    velrange : tuple, optional
        (min, max) velocity range for colorbars in km/s, by default (-100, 100)
    flux_max : float, optional
        Maximum flux for colorbars (0 = auto-scale from central pixel), by default 0
    o3offset : float, optional
        Additional velocity offset for OIII maps in km/s, by default 0
    extent : np.ndarray, optional
        [xmin, xmax, ymin, ymax] for axis limits in arcsec, by default np.array([0])
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object with all emission line maps
    axes : np.ndarray
        2D array of axes objects (6×3 grid)
        
    Notes
    -----
    The 6×3 grid layout shows:
    - Row 0: H-alpha (SNR, flux, velocity)
    - Row 1: [NII] (SNR, flux, FWHM) 
    - Row 2: H-beta (SNR, flux)
    - Row 3: [OIII] (SNR, flux)
    - Row 4: [OI] (SNR, flux)
    - Row 5: [SII] (SNR, ratio)
    
    All spatial coordinates are converted to arcsecond offsets from the
    specified center position. Flux maps show emission line strength,
    while kinematic maps reveal velocity structure and line widths.
    """
    with pyfits.open(file, memmap=False) as hdulist:
        map_hal = hdulist['Halpha'].data
        map_nii = hdulist['NII'].data
        map_hb = hdulist['Hbeta'].data
        map_oiii = hdulist['OIII'].data
        map_oi = hdulist['OI'].data
        map_siir = hdulist['SIIr'].data
        map_siib = hdulist['SIIb'].data
        map_hal_ki = hdulist['Hal_kin'].data
        Av = hdulist['Av'].data
        IFU_header = hdulist['PRIMARY'].header
    
    x, y = center[0], center[1]
    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix * 3600
    
    offsets_low = -np.array(center)
    offsets_hig = np.array(map_hal.shape[1:]) - np.array(center)
    
    lim = np.array([offsets_low[0], offsets_hig[0],
                    offsets_low[1], offsets_hig[1]])
    lim_sc = lim * arc_per_pix
    
    if flux_max == 0:
        flux_max = map_hal[1, y, x]
    
    # Create figure
    f, axes = plt.subplots(6, 3, figsize=(10, 20), sharex=True, sharey=True)
    
    # H-alpha SNR
    im = axes[0, 0].imshow(map_hal[0], vmin=3, vmax=20, origin='lower', extent=lim_sc)
    axes[0, 0].set_title('Hal SNR map')
    colorbar(f, axes[0, 0], im, 'SNR')
    axes[0, 0].set_ylabel('Dec offset (arcsec)')
    
    # H-alpha flux
    im = axes[0, 1].imshow(map_hal[1], vmax=flux_max, origin='lower', extent=lim_sc)
    axes[0, 1].set_title('Halpha Flux map')
    colorbar(f, axes[0, 1], im, 'Flux (arbitrary units)')
    axes[0, 1].set_ylabel('Dec offset (arcsec)')
    
    # H-alpha velocity
    im = axes[0, 2].imshow(map_hal_ki[0], cmap='coolwarm', origin='lower',
                           vmin=velrange[0], vmax=velrange[1], extent=lim_sc)
    axes[0, 2].set_title('Hal Velocity offset map')
    colorbar(f, axes[0, 2], im, 'Velocity (km/s)')
    axes[0, 2].set_ylabel('Dec offset (arcsec)')
    
    # H-alpha FWHM
    im = axes[1, 2].imshow(map_hal_ki[1], vmin=fwhmrange[0], vmax=fwhmrange[1],
                           origin='lower', extent=lim_sc)
    axes[1, 2].set_title('Hal FWHM map')
    colorbar(f, axes[1, 2], im, 'FWHM (km/s)')
    
    # [NII] SNR
    im = axes[1, 0].imshow(map_nii[0], vmin=3, vmax=10, origin='lower', extent=lim_sc)
    axes[1, 0].set_title('[NII] SNR')
    colorbar(f, axes[1, 0], im, 'SNR')
    axes[1, 0].set_ylabel('Dec offset (arcsec)')
    
    # [NII] flux
    im = axes[1, 1].imshow(map_nii[1], vmax=flux_max, origin='lower', extent=lim_sc)
    axes[1, 1].set_title('[NII] map')
    colorbar(f, axes[1, 1], im, 'Flux')
    axes[1, 1].set_ylabel('Dec offset (arcsec)')
    
    # H-beta SNR
    im = axes[2, 0].imshow(map_hb[0], vmin=3, vmax=10, origin='lower', extent=lim_sc)
    axes[2, 0].set_title('Hbeta SNR')
    colorbar(f, axes[2, 0], im, 'SNR')
    axes[2, 0].set_ylabel('Dec offset (arcsec)')
    
    # H-beta flux
    im = axes[2, 1].imshow(map_hb[1], vmax=flux_max, origin='lower', extent=lim_sc)
    axes[2, 1].set_title('Hbeta map')
    colorbar(f, axes[2, 1], im, 'Flux')
    axes[2, 1].set_ylabel('Dec offset (arcsec)')
    
    # [OIII] SNR
    im = axes[3, 0].imshow(map_oiii[0], vmin=3, vmax=20, origin='lower', extent=lim_sc)
    axes[3, 0].set_title('[OIII] SNR')
    colorbar(f, axes[3, 0], im, 'SNR')
    axes[3, 0].set_ylabel('Dec offset (arcsec)')
    
    # [OIII] flux
    im = axes[3, 1].imshow(map_oiii[1], vmax=flux_max, origin='lower', extent=lim_sc)
    axes[3, 1].set_title('[OIII] map')
    colorbar(f, axes[3, 1], im, 'Flux')
    axes[3, 1].set_ylabel('Dec offset (arcsec)')
    
    # [OI] SNR
    im = axes[4, 0].imshow(map_oi[0], vmin=3, vmax=10, origin='lower', extent=lim_sc)
    axes[4, 0].set_title('OI SNR')
    colorbar(f, axes[4, 0], im, 'SNR')
    axes[4, 0].set_ylabel('Dec offset (arcsec)')
    
    # [OI] flux
    im = axes[4, 1].imshow(map_oi[1], origin='lower', extent=lim_sc)
    axes[4, 1].set_title('OI Flux')
    colorbar(f, axes[4, 1], im, 'Flux')
    axes[4, 1].set_ylabel('Dec offset (arcsec)')
    
    # [SII] SNR
    im = axes[5, 0].imshow(map_siir[0], vmin=3, vmax=10, origin='lower', extent=lim_sc)
    axes[5, 0].set_title('[SII] SNR')
    colorbar(f, axes[5, 0], im, 'SNR')
    axes[5, 0].set_xlabel('RA offset (arcsec)')
    axes[5, 0].set_ylabel('Dec offset (arcsec)')
    
    # [SII] ratio
    im = axes[5, 1].imshow(map_siir[1] / map_siib[1], vmin=0.3, vmax=1.5,
                           origin='lower', extent=lim_sc)
    axes[5, 1].set_title('[SII]r/[SII]b')
    colorbar(f, axes[5, 1], im, 'Ratio')
    axes[5, 1].set_xlabel('RA offset (arcsec)')
    axes[5, 1].set_ylabel('Dec offset (arcsec)')
    
    # Set extent if provided
    if len(extent) > 1:
        axes[0, 0].set_xlim(extent[0], extent[1])
        axes[0, 0].set_ylim(extent[2], extent[3])
    
    plt.tight_layout()
    
    return f, axes


# =============================================================================
# Utility Functions
# =============================================================================

def colorbar(f: Figure, ax: Axes, im, label: str, fontsize: int = 12) -> None:
    """
    Add a colorbar to an axis.
    
    Parameters
    ----------
    f : Figure
        Matplotlib figure
    ax : Axes
        Axis to add colorbar to
    im : mappable
        Image/mappable object
    label : str
        Colorbar label
    fontsize : int, optional
        Label font size
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    f.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(label, fontsize=fontsize)


def overide_axes_labels(fig: Figure, ax: Axes, lims,
                        showx: int = 1, showy: int = 1,
                        labelx: int = 1, labely: int = 1,
                        color: str = 'k', fewer_x: int = 0,
                        pruney: int = 0, prunex: int = 0,
                        tick_color: str = 'k', tickin: int = 0,
                        labelsize: int = 12, white: int = 0) -> None:
    """
    Override WCS axis labels with relative coordinate labels.
    
    This function is useful for replacing celestial coordinates with
    offset coordinates in arcseconds from a reference position.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Axis with WCS coordinates
    lims : tuple or list
        Either (xmin, xmax, ymin, ymax) in arcsec, or (img_wcs, img_hdr)
    showx, showy : int, optional
        Whether to show x/y tick labels
    labelx, labely : int, optional
        Whether to show x/y axis labels
    color : str, optional
        Label color
    fewer_x : int, optional
        Skip every other x tick
    pruney, prunex : int, optional
        Remove first and last ticks
    tick_color : str, optional
        Tick color
    tickin : int, optional
        0=out, 1=in(white), 2=in(black)
    labelsize : int, optional
        Font size for labels
    white : int, optional
        Use white labels
    """
    # Parse limits
    if len(lims) == 2:
        img_wcs, hdr = lims
        o = np.array(img_wcs.all_pix2world(1, 1, 1))
        o = SkyCoord(o[0], o[1], unit="deg")
        p1 = np.array(img_wcs.all_pix2world(hdr['NAXIS1'], 1, 1))
        p1 = SkyCoord(p1[0], p1[1], unit="deg")
        p2 = np.array(img_wcs.all_pix2world(1, hdr['NAXIS2'], 1))
        p2 = SkyCoord(p2[0], p2[1], unit="deg")
        
        arcsec_size = np.array([o.separation(p2).arcsec,
                                o.separation(p1).arcsec]) / 2.
        
        lims = [-arcsec_size[1], arcsec_size[1],
                -arcsec_size[0], arcsec_size[0]]
    
    # Hide WCS labels
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    
    # Add new axis overlay
    if showx or showy:
        newax = fig.add_axes(ax.get_position(), frameon=False)
        
        plt.xlim(lims[0], lims[1])
        plt.ylim(lims[2], lims[3])
        
        newax.xaxis.label.set_color(color)
        newax.tick_params(axis='x', colors=color, color=tick_color)
        newax.yaxis.label.set_color(color)
        newax.tick_params(axis='y', colors=color, color=tick_color)
        
        if not showx:
            newax.axes.xaxis.set_ticklabels([])
        if not showy:
            newax.axes.yaxis.set_ticklabels([])
        
        if labely:
            plt.ylabel('arcsec', fontsize=labelsize)
        if labelx:
            plt.xlabel('arcsec', fontsize=labelsize)
        
        if fewer_x:
            newax.set_xticks(newax.get_xticks()[::2])
        if pruney:
            newax.set_yticks(newax.get_yticks()[1:-1])
        if prunex:
            newax.set_xticks(newax.get_xticks()[1:-1])
        
        if tickin == 1:
            newax.tick_params(axis='both', direction='in', color='white')
        elif tickin == 2:
            newax.tick_params(axis='both', direction='in', color='black')
        
        if white == 1:
            newax.xaxis.label.set_color('white')
            newax.yaxis.label.set_color('white')
            plt.setp(newax.get_yticklabels(), color="white")
            plt.setp(newax.get_xticklabels(), color="white")


def add_lines_paper(ax, wave1d, spec1d, redshift, show_labels=True, twodim=False,
                    fontsize=18, subset_name={}, dz=None, extra=False):
    """
    Add emission line labels to spectroscopic plots for publication.
    
    This function overlays vertical dashed lines and labels at the expected
    positions of common emission lines, accounting for redshift. Useful for
    identifying spectral features in published plots.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axis to add lines to
    wave1d : array-like
        Wavelength array in microns
    spec1d : array-like  
        Flux array (used for positioning labels)
    redshift : float
        Object redshift for line positioning
    show_labels : bool, optional
        Whether to show text labels, by default True
    twodim : bool, optional
        Use 2D spectrum formatting (partial lines), by default False
    fontsize : int, optional
        Font size for labels, by default 18
    subset_name : dict, optional
        Subset of line names to show (empty = all), by default {}
    dz : float, optional
        Redshift uncertainty for shaded regions, by default None
    extra : bool, optional
        Include additional lines (not implemented), by default False
        
    Notes
    -----
    Common lines included: Lyman-alpha, CIV, HeII, CIII], [OIII], H-delta,
    H-gamma, [OII], [NeIII], [OIII]λ4363, plus others in UV and optical.
    
    Line wavelengths are converted from rest-frame Angstroms to observed
    microns using: λ_obs = λ_rest * (1+z) / 10^4
    """
    # Import matplotlib for colormap handling  
    import matplotlib
    
    # Define emission line dictionary: {name: (wavelength_Å, y_offset, y_position)}
    # y_offset: horizontal label offset, y_position: vertical position (0-1)
    emlines = {
        'Ly$\\alpha$'   : ( 1215.,  +0.06,  0.9),      # Lyman-alpha
        'NIV'   : ( 1488.,  -0.05,  0.9),              # N IV
        'CIII]'   : ( 1908.734,  -0.0,  0.9),          # C III] semi-forbidden
        'CIV' : ( 1550., -0.05, 0.9),                  # C IV doublet  
        'HeII' : (1640., -0.05, 0.9),                  # He II
        '[OIII]' : (1663., +0.05, 0.9),                # [O III] UV
        'NIII]' : (1752., +0.05, 0.9),                 # N III] semi-forbidden
        'H$\\delta$'         : ( 4102.860, -0.00, 0.9), # H-delta
        'H$\\gamma$'          : ( 4341.647,  0.0, 0.9),  # H-gamma  
        '[OII]'   : ( 3727.,  -0.0,  0.9),             # [O II] doublet
        '[NeIII]' : ( np.array([3869., 3968.]), -0.05, 0.9), # [Ne III] doublet
        'H$\\delta$'         : ( 4102.860, -0.00, 0.9),      # H-delta (duplicate)
        'H$\\gamma$'         : ( 4340.471, -0.04, 0.7),      # H-gamma (updated)
        r'[OIII]$\lambda$4363': ( 4363.471, 0.04, 0.7),     # [O III] auroral line
    }
    
    # Set up colormap for line identification  
    n_lines = len(emlines.items())
    cmap = matplotlib.colormaps['nipy_spectral']  # Spectral colormap
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=n_lines-0.8)
    
    # Loop through each emission line
    for i, (line, lineprop) in enumerate(emlines.items()):
        waves, offset, y_position = lineprop
        waves = np.atleast_1d(waves)  # Handle both single lines and doublets
        
        # Convert rest wavelength to observed wavelength in microns
        waves *= (1+redshift)/1.e4
        wave = np.nanmean(waves)  # Use mean for doublets
        
        # Check if line falls within wavelength range of spectrum
        if not 1.001*wave1d[0] < wave < wave1d[-1]*0.999: 
            continue
        if subset_name and line not in subset_name: 
            continue
            
        # Get color for this line
        color = cmap(norm(float(i)))
        
        # Find spectral region around line for label positioning
        where_line = np.argmin(np.abs(wave1d-wave))
        where_line = slice(where_line-5, where_line+6, 1)
        
        # Set label vertical alignment based on position
        va = 'center'
        y_position = y_position * ax.get_ylim()[1]
        if y_position < 0.05: va = 'bottom'
        if y_position > 0.90: va = 'top'
        
        dwave = 0.  # Additional wavelength offset (not used)
        
        # Draw line markers
        if twodim:
            # For 2D spectra: draw partial lines at top and bottom
            for w in waves:
                ax.axvline(w, 0., 0.33, color='w', lw=1.5, alpha=1.0, ls='--')
                ax.axvline(w, 0.65, 1.0, color='w', lw=1.5, alpha=1.0, ls='--')
        else:
            # For 1D spectra: draw full vertical lines
            for w in waves:
                ax.axvline(w, color=color, lw=1.5, alpha=1.0, ls='--', zorder=0)
            
            # Add uncertainty region if redshift error provided
            if dz:
                wmin, wmax = np.min(waves), np.max(waves)
                wmin = wmin*(1.-dz/(1+redshift))
                wmax = wmax*(1.+dz/(1+redshift))
                ax.axvspan(wmin, wmax, facecolor=color, edgecolor='none', alpha=0.2, zorder=0)
        
        # Add text labels if requested
        if show_labels:
            if ((wave+dwave+offset) < ax.get_xlim()[1]) & ((wave+dwave+offset) > ax.get_xlim()[0]):
                ax.text(
                    wave+dwave+offset, y_position, line,
                    color=color, va=va, ha='center',
                    fontsize=fontsize,
                    rotation='vertical',  # Vertical text for better readability
                    bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'),
                    zorder=5,  # Place text on top
                    )
                
def add_lines_paper_r1000(ax, wave1d, spec1d, redshift, show_labels=True, twodim=False,
    fontsize=18, subset_name={}, dz=None, extra=False):

    import matplotlib
    emlines = {
        'Ly$\\alpha$'   : ( 1215.,  +0.0,  0.8),
        #'NV'   : ( 1240.,  0.05,  0.9),
        'CIII]'   : ( 1907.734,  -0.0,  0.8),
         #'SiIV': [1400., 0, 0.9],
        'CIV' : ( 1550., -0.0, 0.8),
        'HeII' : (1640., -0.0, 0.8),
        '[OIII]' : (1663., +0., 0.8),
        #'NIII]' : (1752., +0.05, 0.9),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.8),
        'H$\\gamma$'          : ( 4341.647,  0.0, 0.8),
        '[OII]'   : ( 3727.,  -0.0,  0.8),
        '[NeIII]' : ( np.array([3869., 3968.]), -0.0, 0.8),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.8),
        'H$\\gamma$'         : ( 4340.471	, -0.0, 0.8),
        r'[OIII]$\lambda$4363'         : ( 4353.471	, 0.04, 0.7),
         'H$\\beta$'         : ( 4862.683, -0.00, 0.8),
        r'[OIII]$\lambda$5008'         : ( 5008.24, 0.0, 0.6),
        'H$\\alpha$'         : ( 6564.61, -0.00, 0.8),
    }
    n_lines = len(emlines.items())
    cmap = matplotlib.colormaps['nipy_spectral']
    norm = matplotlib.colors.Normalize(vmin= -0.5, vmax=n_lines-0.8)
    
    for i,(line,lineprop) in enumerate(emlines.items()):
        waves, offset, y_position = lineprop
        waves = np.atleast_1d(waves)
        waves *= (1+redshift)/1.e4
        wave = np.nanmean(waves)
        if not 1.001*wave1d[0]<wave<wave1d[-1]*0.999: continue
        if subset_name and line not in subset_name: continue
        color = cmap(norm(float(i)))
        where_line = np.argmin(np.abs(wave1d-wave))
        where_line = slice(where_line-5, where_line+6, 1)
        #data_at_line = (
        #    np.min(spec1d[where_line]) if pre_dash
        #    else np.max(spec1d[where_line])
        #)
        va = 'center'
        y_position = y_position*ax.get_ylim()[1]
        if y_position<0.05: va='bottom'
        if y_position>0.90: va='top'
        
        dwave = 0.
        if twodim:
            for w in waves:
                ax.axvline(w, 0., 0.33, color='w', lw=1.5, alpha=1.0, ls='--')
                ax.axvline(w, 0.65, 1.0, color='w', lw=1.5, alpha=1.0, ls='--')
        else:
            for w in waves:
                ax.axvline(w, color=color, lw=1.5, alpha=1.0, ls='--', zorder=0)
            if dz:
                wmin, wmax = np.min(waves), np.max(waves)
                wmin = wmin*(1.-dz/(1+redshift))
                wmax = wmax*(1.+dz/(1+redshift))
                ax.axvspan(wmin, wmax, facecolor=color, edgecolor='none', alpha=0.2, zorder=0)
        if show_labels:
            if ((wave+dwave+offset)< ax.get_xlim()[1]) & ((wave+dwave+offset)> ax.get_xlim()[0]):
                ax.text(
                    wave+dwave+offset, y_position, line,
                    color=color, va=va, ha='center',
                    fontsize=fontsize,
                    rotation='vertical',
                    bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'),
                    zorder=5,
                    )
                
def add_lines_paper_r1000_mul(axes, wave1d, spec1d, redshift, show_labels=True, twodim=False,
    fontsize=18, subset_name={}, dz=None, extra=False):

    import matplotlib
    emlines = {
        'Ly$\\alpha$'   : ( 1215.,  +0.0,  0.8),
        #'NV'   : ( 1240.,  0.05,  0.9),
        'CIII]'   : ( 1907.734,  -0.0,  0.8),
         #'SiIV': [1400., 0, 0.9],
        'CIV' : ( 1550., -0.0, 0.8),
        'HeII' : (1640., -0.0, 0.8),
        '[OIII]' : (1663., +0., 0.8),
        #'NIII]' : (1752., +0.05, 0.9),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.8),
        'H$\\gamma$'          : ( 4341.647,  0.0, 0.8),
        '[OII]'   : ( 3727.,  -0.0,  0.8),
        '[NeIII]' : ( np.array([3869., 3968.]), -0.0, 0.8),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.8),
        'H$\\gamma$'         : ( 4340.471	, -0.0, 0.8),
        r'[OIII]$\lambda$4363'         : ( 4353.471	, 0.04, 0.7),
         'H$\\beta$'         : ( 4862.683, -0.00, 0.8),
        r'[OIII]$\lambda$5008'         : ( 5008.24, 0.0, 0.6),
        'H$\\alpha$'         : ( 6564.61, -0.00, 0.9),
    }
    n_lines = len(emlines.items())
    cmap = matplotlib.colormaps['nipy_spectral']
    norm = matplotlib.colors.Normalize(vmin= -0.5, vmax=n_lines-0.8)
    
    for i,(line,lineprop) in enumerate(emlines.items()):
        waves, offset, y_position = lineprop
        waves = np.atleast_1d(waves)
        waves *= (1+redshift)/1.e4
        wave = np.nanmean(waves)
        if not 1.001*wave1d[0]<wave<wave1d[-1]*0.999: continue
        if subset_name and line not in subset_name: continue
        color = cmap(norm(float(i)))
        where_line = np.argmin(np.abs(wave1d-wave))
        where_line = slice(where_line-5, where_line+6, 1)
        #data_at_line = (
        #    np.min(spec1d[where_line]) if pre_dash
        #    else np.max(spec1d[where_line])
        #)
        va = 'center'
        for ax in axes:
            y_position = y_position*ax.get_ylim()[1]
            if y_position<0.05: va='bottom'
            if y_position>0.80: va='top'
            
            dwave = 0.
        
            if twodim:
                for w in waves:
                    ax.axvline(w, 0., 0.33, color='w', lw=1.5, alpha=1.0, ls='--')
                    ax.axvline(w, 0.65, 1.0, color='w', lw=1.5, alpha=1.0, ls='--')
            else:
                for w in waves:
                    ax.axvline(w, color=color, lw=1.5, alpha=1.0, ls='--', zorder=0)
                if dz:
                    wmin, wmax = np.min(waves), np.max(waves)
                    wmin = wmin*(1.-dz/(1+redshift))
                    wmax = wmax*(1.+dz/(1+redshift))
                    ax.axvspan(wmin, wmax, facecolor=color, edgecolor='none', alpha=0.2, zorder=0)
            if show_labels:
                if ((wave+dwave+offset)< ax.get_xlim()[1]) & ((wave+dwave+offset)> ax.get_xlim()[0]):
                    ax.text(
                        wave+dwave+offset, y_position, line,
                        color=color, va=va, ha='center',
                        fontsize=fontsize,
                        rotation='vertical',
                        bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                                alpha=1.0, edgecolor='none'),
                        zorder=5,
                        )
                
def add_lines_paper_baxes(ax, wave1d, spec1d, redshift, show_labels=True, twodim=False,
    fontsize=18, subset_name={}, dz=None, extra=False):

    import matplotlib
    emlines = {
        'Ly$\\alpha$'   : ( 1215.,  +0.06,  0.9),
        #'NV'   : ( 1240.,  0.05,  0.9),
        'CIII]'   : ( 1908.734,  -0.0,  0.9),
         #'SiIV': [1400., 0, 0.9],
        'CIV' : ( 1550., -0.05, 0.9),
        'HeII' : (1640., -0.05, 0.9),
        '[OIII]' : (1663., +0.05, 0.9),
        #'NIII]' : (1752., +0.05, 0.9),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.9),
        'H$\\gamma$'          : ( 4341.647,  0.0, 0.9),
        '[OII]'   : ( 3727.,  -0.0,  0.9),
        '[NeIII]' : ( np.array([3869., 3968.]), -0.05, 0.9),
        'H$\\delta$'         : ( 4102.860, -0.00, 0.9),
    }
    n_lines = len(emlines.items())
    cmap = matplotlib.colormaps['nipy_spectral']
    norm = matplotlib.colors.Normalize(vmin= -0.5, vmax=n_lines-0.8)
    
    for i,(line,lineprop) in enumerate(emlines.items()):
        waves, offset, y_position = lineprop
        waves = np.atleast_1d(waves)
        waves *= (1+redshift)/1.e4
        wave = np.nanmean(waves)
        if not 1.001*wave1d[0]<wave<wave1d[-1]*0.999: continue
        if subset_name and line not in subset_name: continue
        color = cmap(norm(float(i)))
        where_line = np.argmin(np.abs(wave1d-wave))
        where_line = slice(where_line-5, where_line+6, 1)
        #data_at_line = (
        #    np.min(spec1d[where_line]) if pre_dash
        #    else np.max(spec1d[where_line])
        #)
        va = 'center'
        y_position = y_position*ax.get_ylim()[0][1]
        if y_position<0.05: va='bottom'
        if y_position>0.90: va='top'
        
        dwave = 0.
        if twodim:
            for w in waves:
                ax.axvline(w, 0., 0.33, color='w', lw=1.5, alpha=1.0, ls='--')
                ax.axvline(w, 0.65, 1.0, color='w', lw=1.5, alpha=1.0, ls='--')
        else:
            for w in waves:
                ax.axvline(w, color=color, lw=1.5, alpha=1.0, ls='--', zorder=0)
            if dz:
                wmin, wmax = np.min(waves), np.max(waves)
                wmin = wmin*(1.-dz/(1+redshift))
                wmax = wmax*(1.+dz/(1+redshift))
                ax.axvspan(wmin, wmax, facecolor=color, edgecolor='none', alpha=0.2, zorder=0)
        if show_labels:
            if ((wave+dwave+offset)< ax.get_xlim()[1][1]) & ((wave+dwave+offset)> ax.get_xlim()[0][0]):
                ax.text(
                    wave+dwave+offset, y_position, line,
                    color=color, va=va, ha='center',
                    fontsize=fontsize,
                    rotation='vertical',
                    bbox=dict(boxstyle='Round,pad=0.01', facecolor='white',
                            alpha=1.0, edgecolor='none'),
                    zorder=5,
                    )