#!/usr/bin/env python3
"""
JAX-based core spectral models for QubeSpec

Converted from original NumPy/Numba implementations to pure JAX functions.
All functions are JIT-compatible and can be vectorized with jax.vmap.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit
from jax.typing import ArrayLike
from typing import Union

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)

# Physical constants
PI = jnp.pi
E = jnp.e
C = 3e8  # speed of light in m/s

# Emission line rest wavelengths (in Angstroms)
HALPHA_REST = 6564.52
NII_R_REST = 6585.27
NII_B_REST = 6549.86
OIII_R_REST = 5008.24
OIII_B_REST = 4960.3
HBETA_REST = 4862.6
SII_R_REST = 6732.67
SII_B_REST = 6718.29


@jit
def gauss_jax(x: ArrayLike, amplitude: float, center: float, sigma: float) -> ArrayLike:
    """
    JAX-compatible Gaussian function.
    
    Parameters
    ----------
    x : ArrayLike
        Wavelength array
    amplitude : float
        Peak amplitude
    center : float
        Center wavelength
    sigma : float
        Standard deviation (not FWHM)
        
    Returns
    -------
    ArrayLike
        Gaussian profile
    """
    return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2)


@jit
def fwhm_to_sigma(fwhm: float, center_wavelength: float) -> float:
    """
    Convert FWHM in km/s to sigma in wavelength units.
    
    Parameters
    ----------
    fwhm : float
        Full width at half maximum in km/s
    center_wavelength : float
        Center wavelength in microns
        
    Returns
    -------
    float
        Sigma in wavelength units
    """
    # Convert km/s to fractional velocity, then to wavelength
    velocity_fraction = (fwhm / 1000.0) / (C / 1000.0)  # km/s to c
    sigma_wavelength = velocity_fraction * center_wavelength / 2.35482
    return sigma_wavelength


@jit
def powerlaw_continuum(x: ArrayLike, amplitude: float, pivot: float, alpha: float) -> ArrayLike:
    """
    JAX-compatible power-law continuum.
    
    Replaces astropy.modeling.powerlaws.PowerLaw1D.evaluate
    
    Parameters
    ----------
    x : ArrayLike
        Wavelength array
    amplitude : float
        Amplitude at pivot wavelength
    pivot : float
        Pivot wavelength
    alpha : float
        Power-law index
        
    Returns
    -------
    ArrayLike
        Power-law continuum
    """
    return amplitude * (x / pivot) ** alpha


@jit
def halpha_oiii_model(x: ArrayLike, z: float, cont: float, cont_grad: float,
                      hal_peak: float, nii_peak: float, nar_fwhm: float,
                      sii_r_peak: float, sii_b_peak: float,
                      oiii_peak: float, hbeta_peak: float) -> ArrayLike:
    """
    JAX-compatible Halpha + [OIII] model.
    
    Converted from QubeSpec/Models/Halpha_OIII_models.py:Halpha_OIII
    
    Parameters
    ----------
    x : ArrayLike
        Observed wavelength array in microns
    z : float
        Redshift
    cont : float
        Continuum amplitude
    cont_grad : float
        Continuum gradient
    hal_peak : float
        Halpha peak flux
    nii_peak : float
        [NII] peak flux
    nar_fwhm : float
        Narrow line FWHM in km/s
    sii_r_peak : float
        [SII] 6731 peak flux
    sii_b_peak : float
        [SII] 6717 peak flux
    oiii_peak : float
        [OIII] 5007 peak flux
    hbeta_peak : float
        Hbeta peak flux
        
    Returns
    -------
    ArrayLike
        Model spectrum
    """
    # Calculate observed wavelengths in microns
    hal_wv = HALPHA_REST * (1 + z) / 1e4
    nii_r_wv = NII_R_REST * (1 + z) / 1e4
    nii_b_wv = NII_B_REST * (1 + z) / 1e4
    sii_r_wv = SII_R_REST * (1 + z) / 1e4
    sii_b_wv = SII_B_REST * (1 + z) / 1e4
    oiii_r_wv = OIII_R_REST * (1 + z) / 1e4
    oiii_b_wv = OIII_B_REST * (1 + z) / 1e4
    hbeta_wv = HBETA_REST * (1 + z) / 1e4
    
    # Calculate sigmas from FWHM
    hal_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)
    nii_r_sigma = fwhm_to_sigma(nar_fwhm, nii_r_wv)
    nii_b_sigma = fwhm_to_sigma(nar_fwhm, nii_b_wv)
    sii_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)  # Use Halpha for [SII]
    oiii_sigma = fwhm_to_sigma(nar_fwhm, oiii_r_wv)
    hbeta_sigma = fwhm_to_sigma(nar_fwhm, hbeta_wv)
    
    # Continuum
    continuum = powerlaw_continuum(x, cont, hal_wv, cont_grad)
    
    # Halpha region
    hal_line = gauss_jax(x, hal_peak, hal_wv, hal_sigma)
    nii_r_line = gauss_jax(x, nii_peak, nii_r_wv, nii_r_sigma)
    nii_b_line = gauss_jax(x, nii_peak / 3.0, nii_b_wv, nii_b_sigma)
    sii_r_line = gauss_jax(x, sii_r_peak, sii_r_wv, sii_sigma)
    sii_b_line = gauss_jax(x, sii_b_peak, sii_b_wv, sii_sigma)
    
    # [OIII] region
    oiii_r_line = gauss_jax(x, oiii_peak, oiii_r_wv, oiii_sigma)
    oiii_b_line = gauss_jax(x, oiii_peak / 3.0, oiii_b_wv, oiii_sigma)
    hbeta_line = gauss_jax(x, hbeta_peak, hbeta_wv, hbeta_sigma)
    
    return (continuum + hal_line + nii_r_line + nii_b_line + sii_r_line + sii_b_line +
            oiii_r_line + oiii_b_line + hbeta_line)


@jit
def halpha_oiii_outflow_model(x: ArrayLike, z: float, cont: float, cont_grad: float,
                              hal_peak: float, nii_peak: float, oiii_peak: float,
                              hbeta_peak: float, sii_r_peak: float, sii_b_peak: float,
                              nar_fwhm: float, outflow_fwhm: float, outflow_vel: float,
                              hal_out_peak: float, nii_out_peak: float,
                              oiii_out_peak: float, hbeta_out_peak: float) -> ArrayLike:
    """
    JAX-compatible Halpha + [OIII] + outflow model.
    
    Parameters
    ----------
    x : ArrayLike
        Observed wavelength array in microns
    z : float
        Redshift
    cont : float
        Continuum amplitude
    cont_grad : float
        Continuum gradient
    hal_peak : float
        Halpha narrow peak flux
    nii_peak : float
        [NII] narrow peak flux
    oiii_peak : float
        [OIII] narrow peak flux
    hbeta_peak : float
        Hbeta narrow peak flux
    sii_r_peak : float
        [SII] 6731 peak flux
    sii_b_peak : float
        [SII] 6717 peak flux
    nar_fwhm : float
        Narrow line FWHM in km/s
    outflow_fwhm : float
        Outflow FWHM in km/s
    outflow_vel : float
        Outflow velocity in km/s
    hal_out_peak : float
        Halpha outflow peak flux
    nii_out_peak : float
        [NII] outflow peak flux
    oiii_out_peak : float
        [OIII] outflow peak flux
    hbeta_out_peak : float
        Hbeta outflow peak flux
        
    Returns
    -------
    ArrayLike
        Model spectrum with outflow components
    """
    # Calculate observed wavelengths in microns
    hal_wv = HALPHA_REST * (1 + z) / 1e4
    nii_r_wv = NII_R_REST * (1 + z) / 1e4
    nii_b_wv = NII_B_REST * (1 + z) / 1e4
    sii_r_wv = SII_R_REST * (1 + z) / 1e4
    sii_b_wv = SII_B_REST * (1 + z) / 1e4
    oiii_r_wv = OIII_R_REST * (1 + z) / 1e4
    oiii_b_wv = OIII_B_REST * (1 + z) / 1e4
    hbeta_wv = HBETA_REST * (1 + z) / 1e4
    
    # Calculate outflow wavelengths (velocity shifted)
    velocity_fraction = (outflow_vel / 1000.0) / (C / 1000.0)  # km/s to c
    hal_out_wv = hal_wv * (1 + velocity_fraction)
    nii_r_out_wv = nii_r_wv * (1 + velocity_fraction)
    nii_b_out_wv = nii_b_wv * (1 + velocity_fraction)
    oiii_r_out_wv = oiii_r_wv * (1 + velocity_fraction)
    oiii_b_out_wv = oiii_b_wv * (1 + velocity_fraction)
    hbeta_out_wv = hbeta_wv * (1 + velocity_fraction)
    
    # Calculate sigmas
    hal_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)
    nii_r_sigma = fwhm_to_sigma(nar_fwhm, nii_r_wv)
    nii_b_sigma = fwhm_to_sigma(nar_fwhm, nii_b_wv)
    sii_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)
    oiii_sigma = fwhm_to_sigma(nar_fwhm, oiii_r_wv)
    hbeta_sigma = fwhm_to_sigma(nar_fwhm, hbeta_wv)
    
    # Outflow sigmas
    hal_out_sigma = fwhm_to_sigma(outflow_fwhm, hal_out_wv)
    nii_r_out_sigma = fwhm_to_sigma(outflow_fwhm, nii_r_out_wv)
    nii_b_out_sigma = fwhm_to_sigma(outflow_fwhm, nii_b_out_wv)
    oiii_out_sigma = fwhm_to_sigma(outflow_fwhm, oiii_r_out_wv)
    hbeta_out_sigma = fwhm_to_sigma(outflow_fwhm, hbeta_out_wv)
    
    # Continuum
    continuum = powerlaw_continuum(x, cont, hal_wv, cont_grad)
    
    # Narrow components
    hal_nar = gauss_jax(x, hal_peak, hal_wv, hal_sigma)
    nii_r_nar = gauss_jax(x, nii_peak, nii_r_wv, nii_r_sigma)
    nii_b_nar = gauss_jax(x, nii_peak / 3.0, nii_b_wv, nii_b_sigma)
    sii_r_nar = gauss_jax(x, sii_r_peak, sii_r_wv, sii_sigma)
    sii_b_nar = gauss_jax(x, sii_b_peak, sii_b_wv, sii_sigma)
    oiii_r_nar = gauss_jax(x, oiii_peak, oiii_r_wv, oiii_sigma)
    oiii_b_nar = gauss_jax(x, oiii_peak / 3.0, oiii_b_wv, oiii_sigma)
    hbeta_nar = gauss_jax(x, hbeta_peak, hbeta_wv, hbeta_sigma)
    
    # Outflow components
    hal_out = gauss_jax(x, hal_out_peak, hal_out_wv, hal_out_sigma)
    nii_r_out = gauss_jax(x, nii_out_peak, nii_r_out_wv, nii_r_out_sigma)
    nii_b_out = gauss_jax(x, nii_out_peak / 3.0, nii_b_out_wv, nii_b_out_sigma)
    oiii_r_out = gauss_jax(x, oiii_out_peak, oiii_r_out_wv, oiii_out_sigma)
    oiii_b_out = gauss_jax(x, oiii_out_peak / 3.0, oiii_b_out_wv, oiii_out_sigma)
    hbeta_out = gauss_jax(x, hbeta_out_peak, hbeta_out_wv, hbeta_out_sigma)
    
    return (continuum + hal_nar + nii_r_nar + nii_b_nar + sii_r_nar + sii_b_nar +
            oiii_r_nar + oiii_b_nar + hbeta_nar + hal_out + nii_r_out + nii_b_out +
            oiii_r_out + oiii_b_out + hbeta_out)


@jit
def halpha_model(x: ArrayLike, z: float, cont: float, cont_grad: float,
                 hal_peak: float, nii_peak: float, nar_fwhm: float,
                 sii_r_peak: float, sii_b_peak: float) -> ArrayLike:
    """
    JAX-compatible Halpha-only model.
    
    Parameters
    ----------
    x : ArrayLike
        Observed wavelength array in microns
    z : float
        Redshift
    cont : float
        Continuum amplitude
    cont_grad : float
        Continuum gradient
    hal_peak : float
        Halpha peak flux
    nii_peak : float
        [NII] peak flux
    nar_fwhm : float
        Narrow line FWHM in km/s
    sii_r_peak : float
        [SII] 6731 peak flux
    sii_b_peak : float
        [SII] 6717 peak flux
        
    Returns
    -------
    ArrayLike
        Model spectrum
    """
    # Calculate observed wavelengths in microns
    hal_wv = HALPHA_REST * (1 + z) / 1e4
    nii_r_wv = NII_R_REST * (1 + z) / 1e4
    nii_b_wv = NII_B_REST * (1 + z) / 1e4
    sii_r_wv = SII_R_REST * (1 + z) / 1e4
    sii_b_wv = SII_B_REST * (1 + z) / 1e4
    
    # Calculate sigmas from FWHM
    hal_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)
    nii_r_sigma = fwhm_to_sigma(nar_fwhm, nii_r_wv)
    nii_b_sigma = fwhm_to_sigma(nar_fwhm, nii_b_wv)
    sii_sigma = fwhm_to_sigma(nar_fwhm, hal_wv)
    
    # Continuum
    continuum = powerlaw_continuum(x, cont, hal_wv, cont_grad)
    
    # Emission lines
    hal_line = gauss_jax(x, hal_peak, hal_wv, hal_sigma)
    nii_r_line = gauss_jax(x, nii_peak, nii_r_wv, nii_r_sigma)
    nii_b_line = gauss_jax(x, nii_peak / 3.0, nii_b_wv, nii_b_sigma)
    sii_r_line = gauss_jax(x, sii_r_peak, sii_r_wv, sii_sigma)
    sii_b_line = gauss_jax(x, sii_b_peak, sii_b_wv, sii_sigma)
    
    return continuum + hal_line + nii_r_line + nii_b_line + sii_r_line + sii_b_line


@jit
def oiii_model(x: ArrayLike, z: float, cont: float, cont_grad: float,
               oiii_peak: float, nar_fwhm: float, hbeta_peak: float) -> ArrayLike:
    """
    JAX-compatible [OIII] + Hbeta model.
    
    Parameters
    ----------
    x : ArrayLike
        Observed wavelength array in microns
    z : float
        Redshift
    cont : float
        Continuum amplitude
    cont_grad : float
        Continuum gradient
    oiii_peak : float
        [OIII] 5007 peak flux
    nar_fwhm : float
        Narrow line FWHM in km/s
    hbeta_peak : float
        Hbeta peak flux
        
    Returns
    -------
    ArrayLike
        Model spectrum
    """
    # Calculate observed wavelengths in microns
    oiii_r_wv = OIII_R_REST * (1 + z) / 1e4
    oiii_b_wv = OIII_B_REST * (1 + z) / 1e4
    hbeta_wv = HBETA_REST * (1 + z) / 1e4
    
    # Calculate sigmas from FWHM
    oiii_sigma = fwhm_to_sigma(nar_fwhm, oiii_r_wv)
    hbeta_sigma = fwhm_to_sigma(nar_fwhm, hbeta_wv)
    
    # Continuum
    continuum = powerlaw_continuum(x, cont, oiii_r_wv, cont_grad)
    
    # Emission lines
    oiii_r_line = gauss_jax(x, oiii_peak, oiii_r_wv, oiii_sigma)
    oiii_b_line = gauss_jax(x, oiii_peak / 3.0, oiii_b_wv, oiii_sigma)
    hbeta_line = gauss_jax(x, hbeta_peak, hbeta_wv, hbeta_sigma)
    
    return continuum + oiii_r_line + oiii_b_line + hbeta_line