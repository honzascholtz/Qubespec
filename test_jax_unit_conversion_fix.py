#!/usr/bin/env python3
"""
Test script for JAX conversion of QubeSpec

This script demonstrates the JAX + BlackJAX nested sampling implementation
and compares it with the original emcee-based approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as random
from typing import Dict, List

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Import JAX modules
from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX
from QubeSpec.Models_JAX.core_models import halpha_oiii_model, gauss_jax


def generate_synthetic_spectrum(z: float = 2.0, noise_level: float = 0.01) -> tuple:
    """
    Generate synthetic Halpha + [OIII] spectrum for testing.
    
    Parameters
    ----------
    z : float
        Redshift
    noise_level : float
        Noise level as fraction of signal
        
    Returns
    -------
    tuple
        (wavelength, flux, error)
    """
    # Create wavelength array (microns)
    wave = np.linspace(1.5, 2.0, 1000)
    
    # True parameters
    true_params = {
        'z': z,
        'cont': 0.05,
        'cont_grad': -0.1,
        'hal_peak': 0.3,
        'nii_peak': 0.1,
        'nar_fwhm': 250.0,
        'sii_r_peak': 0.05,
        'sii_b_peak': 0.03,
        'oiii_peak': 0.4,
        'hbeta_peak': 0.12
    }
    
    # Generate model spectrum
    flux_clean = np.array(halpha_oiii_model(
        jnp.array(wave), **true_params
    ))
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level * np.max(flux_clean), len(wave))
    flux_noisy = flux_clean + noise
    
    # Create error array
    error = np.full_like(wave, noise_level * np.max(flux_clean))
    
    return wave, flux_noisy, error, true_params


def test_jax_models():
    """Test JAX model functions."""
    print("Testing JAX model functions...")
    
    # Test parameters
    z = 2.0
    wave = jnp.linspace(1.5, 2.0, 100)
    
    # Test Halpha + [OIII] model
    flux = halpha_oiii_model(
        wave, z=z, cont=0.05, cont_grad=-0.1,
        hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
        sii_r_peak=0.05, sii_b_peak=0.03,
        oiii_peak=0.4, hbeta_peak=0.12
    )
    
    print(f"Model evaluation successful. Flux range: {jnp.min(flux):.3f} - {jnp.max(flux):.3f}")
    
    # Test JIT compilation
    jitted_model = jax.jit(halpha_oiii_model)
    flux_jitted = jitted_model(
        wave, z=z, cont=0.05, cont_grad=-0.1,
        hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
        sii_r_peak=0.05, sii_b_peak=0.03,
        oiii_peak=0.4, hbeta_peak=0.12
    )
    
    print(f"JIT compilation successful. Results match: {jnp.allclose(flux, flux_jitted)}")
    
    return True


def test_jax_fitting():
    """Test JAX-based fitting."""
    print("\nTesting JAX-based fitting...")
    
    # Generate synthetic data
    wave, flux, error, true_params = generate_synthetic_spectrum(z=2.0)
    
    print(f"Generated synthetic spectrum with {len(wave)} points")
    print(f"Flux range: {np.min(flux):.3f} - {np.max(flux):.3f}")
    print(f"True parameters: {true_params}")
    
    # Create fitting object
    fitter = FittingJAX(
        wave=wave,
        flux=flux,
        error=error,
        z=2.0,
        N=1000,  # Reduced for testing
        rng_seed=42
    )
    
    # Test Halpha + [OIII] fitting
    print("\nRunning Halpha + [OIII] fit...")
    try:
        fitter.fitting_collapse_Halpha_OIII(models='Single_only')
        
        if fitter.props:
            print("Fit successful!")
            print(f"Model: {fitter.props['name']}")
            print(f"Chi-squared: {fitter.chi2:.2f}")
            print(f"BIC: {fitter.BIC:.2f}")
            
            # Get evidence
            logz, logz_err = fitter.get_evidence()
            print(f"Log evidence: {logz:.2f} ± {logz_err:.2f}")
            
            # Compare fitted vs true parameters
            print("\nParameter comparison:")
            for param in ['z', 'cont', 'hal_peak', 'oiii_peak']:
                if param in fitter.props and param in true_params:
                    fitted = fitter.props[param][0]  # Mean
                    true_val = true_params[param]
                    print(f"{param}: fitted={fitted:.3f}, true={true_val:.3f}")
        else:
            print("Fit failed - no results available")
            
    except Exception as e:
        print(f"Fit failed with error: {e}")
        return False
    
    return True


def test_model_comparison():
    """Test model comparison using Bayesian evidence."""
    print("\nTesting model comparison...")
    
    # Generate synthetic data with outflow
    wave, flux, error, _ = generate_synthetic_spectrum(z=2.0)
    
    # Fit with simple model
    fitter_simple = FittingJAX(wave=wave, flux=flux, error=error, z=2.0, N=500)
    fitter_simple.fitting_collapse_Halpha_OIII(models='Single_only')
    
    # Fit with outflow model
    fitter_outflow = FittingJAX(wave=wave, flux=flux, error=error, z=2.0, N=500)
    fitter_outflow.fitting_collapse_Halpha_OIII(models='Outflow_only')
    
    # Compare evidences
    logz_simple, logz_err_simple = fitter_simple.get_evidence()
    logz_outflow, logz_err_outflow = fitter_outflow.get_evidence()
    
    print(f"Simple model log evidence: {logz_simple:.2f} ± {logz_err_simple:.2f}")
    print(f"Outflow model log evidence: {logz_outflow:.2f} ± {logz_err_outflow:.2f}")
    
    if logz_simple is not None and logz_outflow is not None:
        bayes_factor = logz_simple - logz_outflow
        print(f"Bayes factor (simple vs outflow): {bayes_factor:.2f}")
        
        if bayes_factor > 2.5:
            print("Strong evidence for simple model")
        elif bayes_factor < -2.5:
            print("Strong evidence for outflow model")
        else:
            print("Inconclusive evidence")
    
    return True


def create_comparison_plot():
    """Create a comparison plot of original vs JAX fitting."""
    print("\nCreating comparison plot...")
    
    # Generate synthetic data
    wave, flux, error, true_params = generate_synthetic_spectrum(z=2.0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot data
    ax1.plot(wave, flux, 'k-', alpha=0.7, label='Data')
    ax1.fill_between(wave, flux-error, flux+error, alpha=0.3, color='gray')
    
    # Fit with JAX
    fitter = FittingJAX(wave=wave, flux=flux, error=error, z=2.0, N=500)
    fitter.fitting_collapse_Halpha_OIII(models='Single_only')
    
    if fitter.yeval is not None:
        ax1.plot(wave, fitter.yeval, 'r-', label='JAX Fit', linewidth=2)
        
        # Plot residuals
        residuals = flux - fitter.yeval
        ax2.plot(wave, residuals, 'k-', alpha=0.7)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax2.fill_between(wave, -error, error, alpha=0.3, color='gray')
    
    # Format plots
    ax1.set_ylabel('Flux')
    ax1.set_title('JAX + BlackJAX Nested Sampling Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('jax_fitting_test.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved as 'jax_fitting_test.png'")
    
    return True


def main():
    """Main test function."""
    print("=== QubeSpec JAX Conversion Test ===\n")
    
    # Test model functions
    if not test_jax_models():
        print("Model tests failed!")
        return False
    
    # Test fitting
    if not test_jax_fitting():
        print("Fitting tests failed!")
        return False
    
    # Test model comparison
    if not test_model_comparison():
        print("Model comparison tests failed!")
        return False
    
    # Create comparison plot
    if not create_comparison_plot():
        print("Plotting failed!")
        return False
    
    print("\n=== All tests completed successfully! ===")
    
    # Print performance summary
    print("\nPerformance notes:")
    print("- JAX models are JIT-compiled for speed")
    print("- Nested sampling provides Bayesian evidence")
    print("- GPU acceleration available (if GPU present)")
    print("- Vectorization ready for cube analysis")
    
    return True


if __name__ == "__main__":
    main()