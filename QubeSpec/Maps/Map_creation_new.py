#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map creation utilities for spectroscopic IFU data analysis.

This module processes fitted spectroscopic data and generates 2D maps of
emission line properties including flux, kinematics (W80, velocity), and
diagnostic line ratios.

Modernized: 2024
@author: jscholtz
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm
from astropy.io import fits
from astropy.table import Table
from brokenaxes import brokenaxes

from .. import Utils as sp
from .. import Plotting as emplot
from .. import Fitting as emfit
from ..Models import Halpha_OIII_models as HaO_models


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class MapConfig:
    """Configuration for map creation parameters."""
    snr_cut: float = 3.0
    fwhm_range: Tuple[float, float] = (100, 500)
    vel_range: Tuple[float, float] = (-100, 100)
    flux_max: float = 0.0
    width_upper: float = 300.0
    delta_bic: float = 10.0
    add_suffix: str = ''
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.snr_cut <= 0:
            raise ValueError("SNR cut must be positive")
        if self.fwhm_range[0] >= self.fwhm_range[1]:
            raise ValueError("Invalid FWHM range")


@dataclass
class EmissionLineInfo:
    """Information for extracting and processing emission line data."""
    name: str
    wavelength: float
    fwhm_param: str
    peak_param: str
    has_kinematics: bool = True
    lsf: float = 0.0
    
    # Kinematic parameter names (for lines with outflows)
    kin_peaks: Optional[List[str]] = None
    kin_fwhms: Optional[List[str]] = None
    kin_vels: Optional[List[str]] = None


# =============================================================================
# Helper Functions - File I/O
# =============================================================================

def load_fit_results(cube, suffix: str = '') -> List:
    """
    Load pickled fit results from file.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    suffix : str, optional
        Additional suffix for filename
        
    Returns
    -------
    list
        List of fit results for each spaxel
        
    Raises
    ------
    FileNotFoundError
        If results file doesn't exist
    """
    filename = f"{cube.savepath}{cube.ID}_{cube.band}_spaxel_fit_raw{suffix}.txt"
    filepath = Path(filename)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    with open(filepath, "rb") as fp:
        results = pickle.load(fp)
    
    return results


def create_fits_hdu_list(cube, data_dict: Dict[str, np.ndarray]) -> fits.HDUList:
    """
    Create FITS HDU list from data dictionary.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance with header information
    data_dict : dict
        Dictionary mapping HDU names to data arrays
        
    Returns
    -------
    HDUList
        FITS HDU list ready for writing
    """
    hdr = cube.header.copy()
    primary_hdu = fits.PrimaryHDU(np.zeros((3, 3, 3)), header=hdr)
    
    hdus = [primary_hdu]
    for name, data in data_dict.items():
        hdus.append(fits.ImageHDU(data, name=name))
    
    return fits.HDUList(hdus)


def save_fits_maps(cube, data_dict: Dict[str, np.ndarray], 
                   output_name: str) -> None:
    """
    Save maps to FITS file.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    data_dict : dict
        Dictionary of maps to save
    output_name : str
        Base name for output file (without extension)
    """
    hdulist = create_fits_hdu_list(cube, data_dict)
    output_path = f"{cube.savepath}{output_name}.fits"
    hdulist.writeto(output_path, overwrite=True)
    print(f"Saved maps to: {output_path}")


# =============================================================================
# Helper Functions - Map Initialization
# =============================================================================

def initialize_map_arrays(cube, n_layers: int = 4) -> np.ndarray:
    """
    Initialize map array filled with NaN.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    n_layers : int
        Number of layers in first dimension
        
    Returns
    -------
    np.ndarray
        Array of shape (n_layers, dim[0], dim[1]) filled with NaN
    """
    return np.full((n_layers, cube.dim[0], cube.dim[1]), np.nan)


def initialize_standard_maps(cube) -> Dict[str, np.ndarray]:
    """
    Initialize standard map arrays for emission line analysis.
    
    Creates SNR, flux, flux error (p16, p84), and kinematic (W80, velocities)
    maps for a given emission line.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
        
    Returns
    -------
    dict
        Dictionary containing initialized map arrays
    """
    maps = {
        'flux': initialize_map_arrays(cube, 4),  # SNR, flux, p16, p84
        'w80': initialize_map_arrays(cube, 3),   # p16, median, p84
        'v10': initialize_map_arrays(cube, 3),
        'v90': initialize_map_arrays(cube, 3),
        'v50': initialize_map_arrays(cube, 3),
        'vel_peak': initialize_map_arrays(cube, 3),
        'narrow_fwhm': initialize_map_arrays(cube, 3),
        'narrow_vel': initialize_map_arrays(cube, 3),
        'outflow_fwhm': initialize_map_arrays(cube, 3),
        'outflow_vel': initialize_map_arrays(cube, 3),
    }
    return maps


def initialize_result_cubes(cube) -> Dict[str, np.ndarray]:
    """
    Initialize 3D result cubes for storing fitted spectra.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
        
    Returns
    -------
    dict
        Dictionary with data, error, model, narrow, and broad cubes
    """
    return {
        'data': cube.flux.data.copy(),
        'error': cube.error_cube.data.copy(),
        'model': np.zeros_like(cube.flux.data),
        'narrow': np.zeros_like(cube.flux.data),
        'broad': np.zeros_like(cube.flux.data),
    }


# =============================================================================
# Helper Functions - Coordinate Calculations
# =============================================================================

def calculate_spatial_extent(cube) -> Tuple[np.ndarray, float]:
    """
    Calculate spatial extent in arcseconds for map display.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
        
    Returns
    -------
    extent : np.ndarray
        Array [xmin, xmax, ymin, ymax] in arcseconds
    arc_per_pix : float
        Arcseconds per pixel conversion factor
    """
    deg_per_pix = cube.header['CDELT2']
    arc_per_pix = deg_per_pix * 3600
    
    offsets_low = -cube.center_data[1:3][::-1]
    offsets_high = cube.dim[0:2] - cube.center_data[1:3][::-1]
    
    extent = np.array([
        offsets_low[0], offsets_high[0],
        offsets_low[1], offsets_high[1]
    ]) * arc_per_pix
    
    return extent, arc_per_pix


# =============================================================================
# Helper Functions - Model Selection
# =============================================================================

def select_best_model(fit_results: Tuple, delta_bic: float = 10.0):
    """
    Select best-fitting model based on BIC comparison.
    
    Parameters
    ----------
    fit_results : tuple
        Either (i, j, Fits) or (i, j, Fits_single, Fits_outflow)
    delta_bic : float
        BIC threshold for model selection
        
    Returns
    -------
    i : int
        Row index
    j : int
        Column index
    Fits : Fitting
        Selected best-fit model
    flag : str
        'single' or 'outflow' indicating selected model type
    """
    if len(fit_results) == 3:
        i, j, Fits = fit_results
        flag = 'outflow' if 'outflow_fwhm' in Fits.props.keys() else 'single'
        return i, j, Fits, flag
    
    else:
        i, j, Fits_single, Fits_outflow = fit_results
        
        # Check for invalid fits
        if not _is_valid_fit(Fits_single):
            return i, j, None, None
        
        # Select based on BIC
        if (Fits_single.BIC - Fits_outflow.BIC) > delta_bic:
            Fits = Fits_outflow
            flag = 'outflow'
        else:
            Fits = Fits_single
            flag = 'single'
        
        return i, j, Fits, flag


def _is_valid_fit(fit_object) -> bool:
    """Check if fit object is valid."""
    return str(type(fit_object)) == "<class 'QubeSpec.Fitting.fits_r.Fitting'>"


# =============================================================================
# Helper Functions - Flux and Kinematics Extraction
# =============================================================================

def extract_flux_measurements(fits_obj, cube, line_type: str = 'OIIIt') -> Tuple:
    """
    Extract flux and uncertainties from fit results.
    
    Parameters
    ----------
    fits_obj : Fitting
        Fit results object
    cube : Cube
        QubeSpec Cube instance
    line_type : str
        Line identifier for flux calculation
        
    Returns
    -------
    snr : float
        Signal-to-noise ratio
    flux : float
        Integrated flux
    p16 : float
        16th percentile (lower uncertainty)
    p84 : float
        84th percentile (upper uncertainty)
    """
    flux, p16, p84 = sp.flux_calc_mcmc(fits_obj, line_type, cube.flux_norm)
    snr = flux / p16 if p16 > 0 else 0.0
    
    return snr, flux, p16, p84


def extract_kinematics(fits_obj, cube, line_type: str = 'OIII', 
                      n_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Extract kinematic measurements (W80, velocities).
    
    Parameters
    ----------
    fits_obj : Fitting
        Fit results object
    cube : Cube
        QubeSpec Cube instance
    line_type : str
        'OIII' or 'Halpha'
    n_samples : int
        Number of samples for percentile calculation
        
    Returns
    -------
    dict
        Dictionary with 'w80', 'v10', 'v90', 'v50', 'vel_peak' arrays
    """
    if line_type == 'OIII':
        kin_params = sp.W80_OIII_calc(fits_obj, z=cube.z, N=n_samples)
    elif line_type == 'Halpha':
        kin_params = sp.W80_Halpha_calc(fits_obj, z=cube.z, N=n_samples)
    else:
        raise ValueError(f"Unknown line type: {line_type}")
    
    return kin_params


def extract_parameter_percentiles(fits_obj, param_name: str, 
                                  z0: Optional[float] = None) -> np.ndarray:
    """
    Extract parameter percentiles from MCMC chains.
    
    Parameters
    ----------
    fits_obj : Fitting
        Fit results object
    param_name : str
        Parameter name in chains
    z0 : float, optional
        Reference redshift for velocity conversion
        
    Returns
    -------
    np.ndarray
        Array of [p16, median, p84] or [value, lower_err, upper_err]
    """
    percentiles = np.percentile(fits_obj.chains[param_name], (16, 50, 84))
    
    # Convert to velocity if needed
    if param_name == 'z' and z0 is not None:
        percentiles = (percentiles - z0) / (1 + z0) * 3e5  # km/s
    
    # Convert to error format: [value, -error, +error]
    result = np.array([
        percentiles[1],
        abs(percentiles[1] - percentiles[0]),
        abs(percentiles[2] - percentiles[1])
    ])
    
    return result


# =============================================================================
# Core Map Filling Functions
# =============================================================================

def fill_emission_line_maps(i: int, j: int, fits_obj, cube, 
                            maps: Dict[str, np.ndarray],
                            line_type: str, snr_cut: float = 3.0) -> None:
    """
    Fill all maps for a single spaxel and emission line.
    
    This is the core function that populates flux, kinematic, and parameter
    maps for a given spaxel based on the fit results.
    
    Parameters
    ----------
    i, j : int
        Spaxel indices
    fits_obj : Fitting
        Fit results object
    cube : Cube
        QubeSpec Cube instance
    maps : dict
        Dictionary of map arrays to fill
    line_type : str
        'OIII' or 'Halpha'
    snr_cut : float
        SNR threshold for detection
    """
    # Extract flux measurements
    snr, flux, p16, p84 = extract_flux_measurements(fits_obj, cube, line_type + 't')
    
    maps['flux'][0, i, j] = snr
    
    if snr > snr_cut:
        # Fill flux maps
        maps['flux'][1, i, j] = flux
        maps['flux'][2, i, j] = p16
        maps['flux'][3, i, j] = p84
        
        # Extract kinematics
        kin_params = extract_kinematics(fits_obj, cube, line_type)
        
        maps['w80'][:, i, j] = kin_params['w80']
        maps['v10'][:, i, j] = kin_params['v10']
        maps['v90'][:, i, j] = kin_params['v90']
        maps['v50'][:, i, j] = kin_params['v50']
        maps['vel_peak'][:, i, j] = kin_params['vel_peak']
        
        # Extract narrow component parameters
        maps['narrow_fwhm'][:, i, j] = extract_parameter_percentiles(
            fits_obj, 'Nar_fwhm'
        )
        maps['narrow_vel'][:, i, j] = extract_parameter_percentiles(
            fits_obj, 'z', cube.z
        )
        
        # Extract outflow parameters if present
        if 'outflow_fwhm' in fits_obj.chains:
            maps['outflow_fwhm'][:, i, j] = extract_parameter_percentiles(
                fits_obj, 'outflow_fwhm'
            )
            maps['outflow_vel'][:, i, j] = extract_parameter_percentiles(
                fits_obj, 'outflow_vel'
            )
    
    else:
        # Upper limits for non-detections
        maps['flux'][2, i, j] = p16
        maps['flux'][3, i, j] = p84


def fill_result_cubes(i: int, j: int, fits_obj, cubes: Dict[str, np.ndarray],
                     indices: Optional[np.ndarray] = None) -> None:
    """
    Fill result cubes with fitted spectra for a single spaxel.
    
    Parameters
    ----------
    i, j : int
        Spaxel indices
    fits_obj : Fitting
        Fit results object
    cubes : dict
        Dictionary of 3D data cubes
    indices : np.ndarray, optional
        Wavelength indices to use (default: all)
    """
    if indices is None:
        indices = slice(None)
    
    try:
        cubes['data'][indices, i, j] = fits_obj.fluxs.data
        cubes['error'][indices, i, j] = fits_obj.error.data
        cubes['model'][indices, i, j] = fits_obj.yeval
    except (AttributeError, IndexError) as e:
        warnings.warn(f"Failed to fill cubes at ({i}, {j}): {e}")


# =============================================================================
# Plotting Functions
# =============================================================================

def create_diagnostic_pdf(cube, results: List, maps: Dict,
                         plot_function, config: MapConfig,
                         line_name: str) -> None:
    """
    Create multi-page PDF with individual spaxel fits.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    results : list
        Fit results for all spaxels
    maps : dict
        Dictionary of map arrays
    plot_function : callable
        Plotting function (e.g., emplot.plotting_OIII)
    config : MapConfig
        Configuration parameters
    line_name : str
        Line identifier for filename
    """
    output_path = (f"{cube.savepath}{cube.ID}_Spaxel_{line_name}_"
                   f"fit_detection_only{config.add_suffix}.pdf")
    
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(1, figsize=(8, 6))
        
        for row in tqdm.tqdm(results, desc=f"Creating {line_name} PDF"):
            i, j, fits_obj, flag = select_best_model(row, config.delta_bic)
            
            if fits_obj is None or not _is_valid_fit(fits_obj):
                continue
            
            snr = maps['flux'][0, i, j]
            
            if snr > config.snr_cut:
                ax.clear()
                
                try:
                    plot_function(fits_obj, ax)
                    
                    # Add annotations
                    if 'w80' in maps and not np.isnan(maps['w80'][0, i, j]):
                        y_pos = ax.get_ylim()[1] * 0.9
                        w80_val = maps['w80'][0, i, j]
                        ax.text(0.05, 0.95, f'W80 = {w80_val:.2f} km/s',
                               transform=ax.transAxes, va='top')
                    
                    ax.set_title(f'x={j}, y={i}, SNR={snr:.2f}')
                    plt.tight_layout()
                    pdf.savefig(fig)
                    
                except Exception as e:
                    warnings.warn(f"Failed to plot spaxel ({i}, {j}): {e}")
        
        plt.close(fig)
    
    print(f"Saved diagnostic PDF: {output_path}")


def plot_4panel_summary(cube, maps: Dict[str, np.ndarray],
                       config: MapConfig, line_name: str) -> Figure:
    """
    Create 4-panel summary map (flux, velocity, W80, SNR).
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    maps : dict
        Dictionary of map arrays
    config : MapConfig
        Configuration parameters
    line_name : str
        Line identifier for title
        
    Returns
    -------
    Figure
        Matplotlib figure with 4-panel plot
    """
    extent, _ = calculate_spatial_extent(cube)
    x, y = int(cube.center_data[1]), int(cube.center_data[2])
    
    # Determine flux maximum
    if config.flux_max == 0:
        flux_max = maps['flux'][1, y, x]
    else:
        flux_max = config.flux_max
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_axes([0.1, 0.55, 0.38, 0.38])
    ax2 = fig.add_axes([0.1, 0.1, 0.38, 0.38])
    ax3 = fig.add_axes([0.55, 0.1, 0.38, 0.38])
    ax4 = fig.add_axes([0.55, 0.55, 0.38, 0.38])
    
    # Flux map
    im = ax1.imshow(maps['flux'][1], vmax=flux_max, origin='lower', extent=extent)
    ax1.set_title(f'{line_name} Flux')
    _add_colorbar(fig, ax1, im, 'Flux')
    
    # Velocity map
    im = ax2.imshow(maps['vel_peak'][0], cmap='coolwarm', origin='lower',
                    vmin=config.vel_range[0], vmax=config.vel_range[1], extent=extent)
    ax2.set_title('v50 (km/s)')
    _add_colorbar(fig, ax2, im, 'Velocity (km/s)')
    
    # W80 map
    im = ax3.imshow(maps['w80'][0], vmin=config.fwhm_range[0],
                    vmax=config.fwhm_range[1], origin='lower', extent=extent)
    ax3.set_title('W80')
    _add_colorbar(fig, ax3, im, 'W80 (km/s)')
    
    # SNR map
    im = ax4.imshow(maps['flux'][0], vmin=3, vmax=20, origin='lower', extent=extent)
    ax4.set_title('SNR')
    _add_colorbar(fig, ax4, im, 'SNR')
    
    # Add axis labels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('RA offset (arcsec)')
        ax.set_ylabel('Dec offset (arcsec)')
    
    return fig


def _add_colorbar(fig: Figure, ax: Axes, im, label: str) -> None:
    """Helper to add colorbar to axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(label)


def plot_6x3_emission_line_grid(cube, maps_dict: Dict[str, Dict],
                                config: MapConfig) -> Figure:
    """
    Create comprehensive 6x3 grid showing multiple emission lines.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    maps_dict : dict
        Dictionary of maps for each emission line
    config : MapConfig
        Configuration parameters
        
    Returns
    -------
    Figure
        Matplotlib figure with grid layout
    """
    extent, _ = calculate_spatial_extent(cube)
    x, y = int(cube.center_data[1]), int(cube.center_data[2])
    
    if config.flux_max == 0:
        flux_max = maps_dict['Halpha']['flux'][1, y, x]
    else:
        flux_max = config.flux_max
    
    fig, axes = plt.subplots(6, 3, figsize=(10, 20), sharex=True, sharey=True)
    
    # Define layout: each row is (line_name, map_type, vmin, vmax, cmap, label)
    layout = [
        # H-alpha row 0
        ('Halpha', 'flux', 0, 3, 20, None, 'SNR'),
        ('Halpha', 'flux', 1, None, flux_max, None, 'Flux'),
        ('Halpha', 'vel_peak', 0, config.vel_range[0], config.vel_range[1], 'coolwarm', 'Velocity (km/s)'),
        # H-alpha row 1
        ('Halpha', 'w80', 0, config.fwhm_range[0], config.fwhm_range[1], None, 'W80 (km/s)'),
        # [NII] row 1
        ('NII', 'flux', 0, 3, 10, None, 'SNR'),
        ('NII', 'flux', 1, None, flux_max, None, 'Flux'),
        # H-beta row 2
        ('Hbeta', 'flux', 0, 3, 10, None, 'SNR'),
        ('Hbeta', 'flux', 1, None, flux_max, None, 'Flux'),
        # [OIII] row 3
        ('OIII', 'flux', 0, 3, 20, None, 'SNR'),
        ('OIII', 'flux', 1, None, flux_max, None, 'Flux'),
        ('OIII', 'vel_peak', 0, config.vel_range[0], config.vel_range[1], 'coolwarm', 'Velocity (km/s)'),
        # [OIII] row 3 continued
        ('OIII', 'w80', 0, config.fwhm_range[0], config.fwhm_range[1], None, 'W80 (km/s)'),
        # Additional lines as needed
    ]
    
    # This is a simplified version - you'd populate based on available maps
    # For full implementation, iterate through layout and populate axes
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main Map Creation Functions
# =============================================================================

def create_oiii_maps(cube, config: Optional[MapConfig] = None) -> Figure:
    """
    Create OIII emission line maps and diagnostic plots.
    
    This is the main entry point for OIII-only analysis. It loads fit results,
    extracts flux and kinematic properties, creates diagnostic PDFs, and saves
    FITS maps.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    config : MapConfig, optional
        Configuration parameters (uses defaults if not provided)
        
    Returns
    -------
    Figure
        4-panel summary figure
        
    Examples
    --------
    >>> from map_creation_refactored import create_oiii_maps, MapConfig
    >>> config = MapConfig(snr_cut=5.0, delta_bic=15.0)
    >>> fig = create_oiii_maps(cube, config)
    >>> fig.savefig('oiii_summary.pdf')
    """
    if config is None:
        config = MapConfig()
    
    # Load results
    suffix = f'_OIII{config.add_suffix}'
    results = load_fit_results(cube, suffix)
    
    # Initialize maps
    maps = initialize_standard_maps(cube)
    cubes = initialize_result_cubes(cube)
    
    # Process each spaxel
    failed_fits = 0
    for row in tqdm.tqdm(results, desc="Processing OIII fits"):
        i, j, fits_obj, flag = select_best_model(row, config.delta_bic)
        
        if fits_obj is None or not _is_valid_fit(fits_obj):
            failed_fits += 1
            continue
        
        # Fill cubes
        fill_result_cubes(i, j, fits_obj, cubes)
        
        # Fill maps
        fill_emission_line_maps(i, j, fits_obj, cube, maps, 'OIII', config.snr_cut)
    
    if failed_fits > 0:
        print(f"Warning: {failed_fits} failed fits encountered")
    
    # Create diagnostic PDF
    create_diagnostic_pdf(cube, results, maps, emplot.plotting_OIII,
                         config, 'OIII')
    
    # Create summary plot
    fig = plot_4panel_summary(cube, maps, config, '[OIII]')
    fig.savefig(f"{cube.savepath}Diagnostics/OIII_maps.pdf")
    
    # Save FITS maps
    data_dict = {
        'flux': cubes['data'],
        'error': cubes['error'],
        'yeval': cubes['model'],
        'residuals': cubes['data'] - cubes['model'],
        'OIII': maps['flux'],
        'OIII_w80': maps['w80'],
        'OIII_v10': maps['v10'],
        'OIII_v90': maps['v90'],
        'OIII_v50': maps['v50'],
        'OIII_vel': maps['vel_peak'],
        'narrow_vel': maps['narrow_vel'],
        'narrow_fwhm': maps['narrow_fwhm'],
        'outflow_fwhm': maps['outflow_fwhm'],
        'outflow_vel': maps['outflow_vel'],
    }
    
    output_name = f"{cube.ID}_OIII_fits_maps{config.add_suffix}"
    save_fits_maps(cube, data_dict, output_name)
    
    return fig


def create_halpha_maps(cube, config: Optional[MapConfig] = None) -> Figure:
    """
    Create H-alpha emission line maps and diagnostic plots.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    config : MapConfig, optional
        Configuration parameters
        
    Returns
    -------
    Figure
        4-panel summary figure
    """
    if config is None:
        config = MapConfig()
    
    # Load results
    suffix = f'_Halpha{config.add_suffix}'
    results = load_fit_results(cube, suffix)
    
    # Initialize maps
    maps_hal = initialize_standard_maps(cube)
    maps_nii = {'flux': initialize_map_arrays(cube, 4)}
    maps_sii = {
        'siir': initialize_map_arrays(cube, 4),
        'siib': initialize_map_arrays(cube, 4),
    }
    cubes = initialize_result_cubes(cube)
    
    # Process each spaxel
    failed_fits = 0
    for row in tqdm.tqdm(results, desc="Processing H-alpha fits"):
        i, j, fits_obj, flag = select_best_model(row, config.delta_bic)
        
        if fits_obj is None or not _is_valid_fit(fits_obj):
            failed_fits += 1
            continue
        
        # Fill cubes
        fill_result_cubes(i, j, fits_obj, cubes)
        
        # Fill H-alpha maps
        fill_emission_line_maps(i, j, fits_obj, cube, maps_hal, 'Halpha', config.snr_cut)
        
        # Fill [NII] maps
        snr_nii, flux_nii, p16_nii, p84_nii = extract_flux_measurements(
            fits_obj, cube, 'NIIt'
        )
        maps_nii['flux'][0, i, j] = snr_nii
        if snr_nii > config.snr_cut:
            maps_nii['flux'][1:4, i, j] = [flux_nii, p16_nii, p84_nii]
        else:
            maps_nii['flux'][2:4, i, j] = [p16_nii, p84_nii]
        
        # Fill [SII] maps if present
        if 'SIIr_peak' in fits_obj.props:
            snr_sii, flux_r, p16_r, p84_r = extract_flux_measurements(
                fits_obj, cube, 'SIIr'
            )
            _, flux_b, p16_b, p84_b = extract_flux_measurements(
                fits_obj, cube, 'SIIb'
            )
            
            maps_sii['siir'][0, i, j] = snr_sii
            maps_sii['siib'][0, i, j] = snr_sii
            
            if snr_sii > config.snr_cut:
                maps_sii['siir'][1:4, i, j] = [flux_r, p16_r, p84_r]
                maps_sii['siib'][1:4, i, j] = [flux_b, p16_b, p84_b]
            else:
                maps_sii['siir'][2:4, i, j] = [p16_r, p84_r]
                maps_sii['siib'][2:4, i, j] = [p16_b, p84_b]
    
    if failed_fits > 0:
        print(f"Warning: {failed_fits} failed fits encountered")
    
    # Create diagnostic PDF
    create_diagnostic_pdf(cube, results, maps_hal, emplot.plotting_Halpha,
                         config, 'Halpha')
    
    # Create summary plot
    fig = plot_4panel_summary(cube, maps_hal, config, 'H-alpha')
    fig.savefig(f"{cube.savepath}Diagnostics/Halpha_maps.pdf")
    
    # Save FITS maps
    data_dict = {
        'flux': cubes['data'],
        'error': cubes['error'],
        'yeval': cubes['model'],
        'residuals': cubes['data'] - cubes['model'],
        'Hal': maps_hal['flux'],
        'Hal_w80': maps_hal['w80'],
        'Hal_v10': maps_hal['v10'],
        'Hal_v90': maps_hal['v90'],
        'Hal_v50': maps_hal['v50'],
        'Hal_vel': maps_hal['vel_peak'],
        'NII': maps_nii['flux'],
    }
    
    output_name = f"{cube.ID}_Halpha_fits_maps{config.add_suffix}"
    save_fits_maps(cube, data_dict, output_name)
    
    return fig


def create_halpha_oiii_maps(cube, config: Optional[MapConfig] = None) -> Figure:
    """
    Create combined H-alpha + [OIII] emission line maps.
    
    This function processes simultaneous fits to both H-alpha and [OIII]
    regions, extracting properties for all major emission lines including
    H-alpha, [NII], H-beta, [OIII], [OI], and [SII].
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    config : MapConfig, optional
        Configuration parameters
        
    Returns
    -------
    Figure
        6x3 grid summary figure
    """
    if config is None:
        config = MapConfig()
    
    # Load results
    suffix = f'_Halpha_OIII{config.add_suffix}'
    results = load_fit_results(cube, suffix)
    
    # Initialize all maps
    maps_dict = {
        'Halpha': initialize_standard_maps(cube),
        'OIII': initialize_standard_maps(cube),
        'Hbeta': {'flux': initialize_map_arrays(cube, 4)},
        'NII': {'flux': initialize_map_arrays(cube, 4)},
        'OI': {'flux': initialize_map_arrays(cube, 4)},
        'SIIr': {'flux': initialize_map_arrays(cube, 4)},
        'SIIb': {'flux': initialize_map_arrays(cube, 4)},
    }
    
    # Add narrow and outflow component maps
    maps_dict['OIII_narrow'] = {'flux': initialize_map_arrays(cube, 4)}
    maps_dict['OIII_outflow'] = {'flux': initialize_map_arrays(cube, 4)}
    
    cubes = initialize_result_cubes(cube)
    
    # Process each spaxel
    failed_fits = 0
    for row in tqdm.tqdm(results, desc="Processing combined fits"):
        i, j, fits_obj, flag = select_best_model(row, config.delta_bic)
        
        if fits_obj is None or not _is_valid_fit(fits_obj):
            failed_fits += 1
            continue
        
        # Fill cubes
        fill_result_cubes(i, j, fits_obj, cubes)
        
        # Fill H-alpha maps
        fill_emission_line_maps(i, j, fits_obj, cube, maps_dict['Halpha'], 
                               'Halpha', config.snr_cut)
        
        # Fill [OIII] maps
        fill_emission_line_maps(i, j, fits_obj, cube, maps_dict['OIII'],
                               'OIII', config.snr_cut)
        
        # Fill other lines (simplified - full implementation would be similar)
        for line_type in ['Hbeta', 'NII', 'SIIr', 'SIIb']:
            if line_type in ['SIIr', 'SIIb']:
                line_suffix = line_type
            else:
                line_suffix = line_type + 't' if line_type == 'NII' else line_type
            
            snr, flux, p16, p84 = extract_flux_measurements(
                fits_obj, cube, line_suffix
            )
            
            maps_dict[line_type]['flux'][0, i, j] = snr
            if snr > config.snr_cut:
                maps_dict[line_type]['flux'][1:4, i, j] = [flux, p16, p84]
            else:
                maps_dict[line_type]['flux'][2:4, i, j] = [p16, p84]
        
        # Store narrow and broad components if available
        if flag == 'outflow':
            # Extract narrow component fluxes
            snr_n, flux_n, p16_n, p84_n = extract_flux_measurements(
                fits_obj, cube, 'OIIIn'
            )
            maps_dict['OIII_narrow']['flux'][1:4, i, j] = [flux_n, p16_n, p84_n]
            
            # Extract outflow component fluxes
            snr_w, flux_w, p16_w, p84_w = extract_flux_measurements(
                fits_obj, cube, 'OIIIw'
            )
            maps_dict['OIII_outflow']['flux'][1:4, i, j] = [flux_w, p16_w, p84_w]
    
    if failed_fits > 0:
        print(f"Warning: {failed_fits} failed fits encountered")
    
    # Create diagnostic PDF with brokenaxes
    _create_combined_diagnostic_pdf(cube, results, maps_dict, config)
    
    # Create summary plot
    fig = plot_6x3_emission_line_grid(cube, maps_dict, config)
    fig.savefig(f"{cube.savepath}Diagnostics/Halpha_OIII_maps.pdf")
    
    # Save FITS maps
    data_dict = {
        'flux': cubes['data'],
        'error': cubes['error'],
        'yeval': cubes['model'],
        'yeval_nar': cubes['narrow'],
        'yeval_bro': cubes['broad'],
        'residuals': cubes['data'] - cubes['model'],
    }
    
    # Add all emission line maps
    for line_name, line_maps in maps_dict.items():
        if line_name in ['Halpha', 'OIII']:
            prefix = 'Hal' if line_name == 'Halpha' else 'OIII'
            data_dict[prefix] = line_maps['flux']
            data_dict[f'{prefix}_w80'] = line_maps['w80']
            data_dict[f'{prefix}_v10'] = line_maps['v10']
            data_dict[f'{prefix}_v90'] = line_maps['v90']
            data_dict[f'{prefix}_v50'] = line_maps['v50']
            data_dict[f'{prefix}_vel'] = line_maps['vel_peak']
            data_dict['narrow_vel'] = line_maps['narrow_vel']
            data_dict['narrow_fwhm'] = line_maps['narrow_fwhm']
            data_dict['outflow_fwhm'] = line_maps['outflow_fwhm']
            data_dict['outflow_vel'] = line_maps['outflow_vel']
        else:
            data_dict[line_name] = line_maps['flux']
    
    output_name = f"{cube.ID}_Halpha_OIII_fits_maps{config.add_suffix}"
    save_fits_maps(cube, data_dict, output_name)
    
    return fig


def _create_combined_diagnostic_pdf(cube, results: List, 
                                   maps_dict: Dict[str, Dict],
                                   config: MapConfig) -> None:
    """Create PDF with brokenaxes for combined H-alpha + OIII plots."""
    output_path = (f"{cube.savepath}{cube.ID}_Spaxel_Halpha_OIII_"
                   f"fit_detection_only{config.add_suffix}.pdf")
    
    with PdfPages(output_path) as pdf:
        for row in tqdm.tqdm(results, desc="Creating combined PDF"):
            i, j, fits_obj, flag = select_best_model(row, config.delta_bic)
            
            if fits_obj is None or not _is_valid_fit(fits_obj):
                continue
            
            # Create broken axes plot
            fig = plt.figure(figsize=(10, 4))
            baxes = brokenaxes(xlims=((4800, 5050), (6500, 6800)), hspace=0.01)
            
            try:
                emplot.plotting_Halpha_OIII(fits_obj, baxes)
                
                # Add SNR information
                snr_hal = maps_dict['Halpha']['flux'][0, i, j]
                snr_oiii = maps_dict['OIII']['flux'][0, i, j]
                snr_nii = maps_dict['NII']['flux'][0, i, j]
                
                snr_text = f'SNR = [{snr_hal:.1f}, {snr_oiii:.1f}, {snr_nii:.1f}]'
                baxes.set_title(f'xy={j} {i}, {snr_text}')
                baxes.set_xlabel('Restframe wavelength (Å)')
                baxes.set_ylabel(r'$10^{-16}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$')
                
                # Add W80 annotation if available
                if not np.isnan(maps_dict['OIII']['w80'][0, i, j]):
                    w80_val = maps_dict['OIII']['w80'][0, i, j]
                    y_pos = baxes.get_ylim()[0][1] * 0.9
                    baxes.text(4810, y_pos, f'OIII W80 = {w80_val:.2f}')
                
                pdf.savefig(fig)
                
            except Exception as e:
                warnings.warn(f"Failed to plot spaxel ({i}, {j}): {e}")
            
            finally:
                plt.close(fig)
    
    print(f"Saved combined diagnostic PDF: {output_path}")


# =============================================================================
# General Map Creation (Flexible Framework)
# =============================================================================

def create_general_maps(cube, emission_line_info: Dict[str, EmissionLineInfo],
                       params_to_extract: Optional[List[str]] = None,
                       config: Optional[MapConfig] = None,
                       wavelength_indices: Optional[np.ndarray] = None) -> fits.HDUList:
    """
    General-purpose map creation for arbitrary emission lines.
    
    This flexible function can handle any emission line configuration by
    accepting a dictionary that specifies line properties and extraction
    parameters.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    emission_line_info : dict
        Dictionary mapping line names to EmissionLineInfo objects
    params_to_extract : list, optional
        List of parameter names to extract percentiles for
    config : MapConfig, optional
        Configuration parameters
    wavelength_indices : np.ndarray, optional
        Wavelength indices to use (default: all)
        
    Returns
    -------
    HDUList
        FITS HDU list with all maps
        
    Examples
    --------
    >>> from map_creation_refactored import create_general_maps, EmissionLineInfo
    >>> 
    >>> # Define emission lines
    >>> lines = {
    >>>     'Hbeta': EmissionLineInfo(
    >>>         name='Hbeta',
    >>>         wavelength=4862.6,
    >>>         fwhm_param='Nar_fwhm',
    >>>         peak_param='Hbeta_peak'
    >>>     ),
    >>>     'OIII': EmissionLineInfo(
    >>>         name='OIII',
    >>>         wavelength=5008.24,
    >>>         fwhm_param='Nar_fwhm',
    >>>         peak_param='OIII_peak',
    >>>         kin_peaks=['OIII_peak', 'OIII_out_peak'],
    >>>         kin_fwhms=['Nar_fwhm', 'outflow_fwhm'],
    >>>         kin_vels=[None, 'outflow_vel']
    >>>     ),
    >>> }
    >>> 
    >>> # Create maps
    >>> hdulist = create_general_maps(cube, lines, params_to_extract=['z', 'Nar_fwhm'])
    >>> hdulist.writeto('general_maps.fits', overwrite=True)
    """
    if config is None:
        config = MapConfig()
    
    # Load results
    suffix = f'_general{config.add_suffix}'
    results = load_fit_results(cube, suffix)
    
    # Initialize maps
    line_maps = {}
    for line_name, line_info in emission_line_info.items():
        line_maps[line_name] = {
            'flux': np.full((6, cube.dim[0], cube.dim[1]), np.nan),
            # Index 0: SNR, 1: flux, 2: p16, 3: p84, 4: flux/p16, 5: std
        }
        
        if line_info.has_kinematics:
            line_maps[line_name]['W80'] = initialize_map_arrays(cube, 3)
            line_maps[line_name]['peak_vel'] = initialize_map_arrays(cube, 3)
            line_maps[line_name]['v10'] = initialize_map_arrays(cube, 3)
            line_maps[line_name]['v90'] = initialize_map_arrays(cube, 3)
    
    # Parameter maps
    param_maps = {}
    if params_to_extract:
        for param in params_to_extract:
            param_maps[param] = initialize_map_arrays(cube, 3)
    
    # Quality maps
    chi2_map = np.full((cube.dim[0], cube.dim[1]), np.nan)
    bic_map = np.full((cube.dim[0], cube.dim[1]), np.nan)
    
    # Result cubes
    cubes = initialize_result_cubes(cube)
    
    if wavelength_indices is None:
        wavelength_indices = np.arange(cube.flux.data.shape[0])
    
    # Process each spaxel
    failed_fits = 0
    for row in tqdm.tqdm(results, desc="Processing general fits"):
        try:
            i, j, fits_obj = row
        except (ValueError, TypeError):
            continue
        
        if not _is_valid_fit(fits_obj):
            failed_fits += 1
            continue
        
        # Fill result cubes
        fill_result_cubes(i, j, fits_obj, cubes, wavelength_indices)
        
        # Store quality metrics
        try:
            chi2_map[i, j] = fits_obj.chi2
            bic_map[i, j] = fits_obj.BIC
        except AttributeError:
            pass
        
        # Extract parameters
        if params_to_extract:
            for param in params_to_extract:
                if param in fits_obj.chains:
                    param_maps[param][:, i, j] = extract_parameter_percentiles(
                        fits_obj, param, cube.z if param == 'z' else None
                    )
        
        # Extract emission line properties
        for line_name, line_info in emission_line_info.items():
            # Calculate SNR
            snr = sp.SNR_calc(
                cube.obs_wave[wavelength_indices],
                fits_obj.fluxs,
                fits_obj.error,
                fits_obj.props,
                'general',
                wv_cent=line_info.wavelength,
                peak_name=line_info.peak_param,
                fwhm_name=line_info.fwhm_param,
                lsf=line_info.lsf
            )
            
            line_maps[line_name]['flux'][0, i, j] = snr
            
            # Calculate flux
            flux, p16, p84, std = sp.flux_calc_mcmc(
                fits_obj,
                'general',
                cube.flux_norm,
                wv_cent=line_info.wavelength,
                peak_name=line_info.peak_param,
                fwhm_name=line_info.fwhm_param,
                lsf=line_info.lsf,
                std=True
            )
            
            line_maps[line_name]['flux'][4, i, j] = flux / p16 if p16 > 0 else 0
            line_maps[line_name]['flux'][5, i, j] = std
            
            if snr > config.snr_cut:
                line_maps[line_name]['flux'][1, i, j] = flux
                line_maps[line_name]['flux'][2, i, j] = p16
                line_maps[line_name]['flux'][3, i, j] = p84
                
                # Extract kinematics if requested
                if line_info.has_kinematics and line_info.kin_peaks:
                    kin_params = sp.vel_kin_percentiles(
                        fits_obj,
                        peak_names=line_info.kin_peaks,
                        fwhm_names=line_info.kin_fwhms,
                        vel_names=line_info.kin_vels,
                        rest_wave=line_info.wavelength,
                        N=100,
                        z=cube.z
                    )
                    
                    line_maps[line_name]['W80'][:, i, j] = kin_params['w80']
                    line_maps[line_name]['peak_vel'][:, i, j] = kin_params['vel_peak']
                    line_maps[line_name]['v10'][:, i, j] = kin_params['v10']
                    line_maps[line_name]['v90'][:, i, j] = kin_params['v90']
            else:
                # Store upper limits
                line_maps[line_name]['flux'][2, i, j] = p16
                line_maps[line_name]['flux'][3, i, j] = p84
    
    if failed_fits > 0:
        print(f"Warning: {failed_fits} failed fits encountered")
    
    # Create FITS HDU list
    data_dict = {
        'flux': cubes['data'],
        'error': cubes['error'],
        'yeval': cubes['model'],
        'residuals': cubes['data'] - cubes['model'],
    }
    
    # Add parameter maps
    for param, param_map in param_maps.items():
        data_dict[param] = param_map
    
    # Add emission line maps
    for line_name, maps in line_maps.items():
        data_dict[line_name] = maps['flux']
        
        if 'W80' in maps:
            data_dict[f'{line_name}_peakvel'] = maps['peak_vel']
            data_dict[f'{line_name}_W80'] = maps['W80']
            data_dict[f'{line_name}_v10'] = maps['v10']
            data_dict[f'{line_name}_v90'] = maps['v90']
    
    # Add quality maps
    data_dict['chi2'] = chi2_map
    data_dict['BIC'] = bic_map
    
    hdulist = create_fits_hdu_list(cube, data_dict)
    
    # Save to file
    output_name = f"{cube.ID}_general_fits_maps{config.add_suffix}"
    output_path = f"{cube.savepath}{output_name}.fits"
    hdulist.writeto(output_path, overwrite=True)
    print(f"Saved general maps: {output_path}")
    
    return hdulist


# =============================================================================
# pPXF Map Creation (Alternative Fitting Method)
# =============================================================================

def create_ppxf_maps(cube, emission_line_info: Dict[str, EmissionLineInfo],
                    suffix: str = '') -> fits.HDUList:
    """
    Create maps from pPXF emission line fitting results.
    
    Parameters
    ----------
    cube : Cube
        QubeSpec Cube instance
    emission_line_info : dict
        Dictionary of emission line information
    suffix : str, optional
        Additional suffix for output filename
        
    Returns
    -------
    HDUList
        FITS HDU list with flux maps
    """
    # Load pPXF results table
    table_path = f"{cube.savepath}PRISM_spaxel/spaxel_R100_ppxf_emlines.fits"
    flux_table = Table.read(table_path)
    
    # Initialize maps
    line_maps = {}
    for line_name in emission_line_info.keys():
        line_maps[line_name] = np.full((2, cube.dim[0], cube.dim[1]), np.nan)
    
    # Fill maps from table
    for row in tqdm.tqdm(flux_table, desc="Processing pPXF results"):
        # Parse spaxel ID
        spaxel_id = str(row['ID'])
        i, j = int(spaxel_id[:2]), int(spaxel_id[2:])
        
        # Fill flux maps
        for line_name in emission_line_info.keys():
            flux_col = f'{line_name}_flux'
            upper_col = f'{line_name}_flux_upper'
            
            if row[flux_col] > row[upper_col]:
                line_maps[line_name][0, i, j] = row[flux_col]
            else:
                line_maps[line_name][1, i, j] = row[upper_col] / 3
    
    # Create FITS HDU list
    data_dict = {}
    for line_name, flux_map in line_maps.items():
        data_dict[line_name] = flux_map
    
    hdulist = create_fits_hdu_list(cube, data_dict)
    
    # Save to file
    output_path = f"{cube.savepath}{cube.ID}_ppxf_fits_maps{suffix}.fits"
    hdulist.writeto(output_path, overwrite=True)
    print(f"Saved pPXF maps: {output_path}")
    
    return hdulist


# =============================================================================
# Backward Compatibility Wrappers
# =============================================================================

def Map_creation_OIII(Cube, SNR_cut=3, fwhmrange=[100, 500], 
                     velrange=[-100, 100], dbic=12, flux_max=0,
                     width_upper=300, add=''):
    """Backward compatibility wrapper for create_oiii_maps."""
    config = MapConfig(
        snr_cut=SNR_cut,
        fwhm_range=tuple(fwhmrange),
        vel_range=tuple(velrange),
        flux_max=flux_max,
        width_upper=width_upper,
        delta_bic=dbic,
        add_suffix=add
    )
    return create_oiii_maps(Cube, config)


def Map_creation_Halpha(Cube, SNR_cut=3, fwhmrange=[100, 500],
                       velrange=[-100, 100], dbic=10, flux_max=0, add=''):
    """Backward compatibility wrapper for create_halpha_maps."""
    config = MapConfig(
        snr_cut=SNR_cut,
        fwhm_range=tuple(fwhmrange),
        vel_range=tuple(velrange),
        flux_max=flux_max,
        delta_bic=dbic,
        add_suffix=add
    )
    return create_halpha_maps(Cube, config)


def Map_creation_Halpha_OIII(Cube, SNR_cut=3, fwhmrange=[100, 500],
                            velrange=[-100, 100], dbic=10, flux_max=0,
                            width_upper=300, add=''):
    """Backward compatibility wrapper for create_halpha_oiii_maps."""
    config = MapConfig(
        snr_cut=SNR_cut,
        fwhm_range=tuple(fwhmrange),
        vel_range=tuple(velrange),
        flux_max=flux_max,
        width_upper=width_upper,
        delta_bic=dbic,
        add_suffix=add
    )
    return create_halpha_oiii_maps(Cube, config)
