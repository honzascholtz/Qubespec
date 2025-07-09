#!/usr/bin/env python3
"""
Simple JAX test to verify basic functionality without nested sampling.
"""

import sys
import numpy as np
import jax.numpy as jnp
import jax

# Add path for imports
sys.path.insert(0, '/home/will/jens/Qubespec')

def test_jax_models():
    """Test JAX model functionality."""
    print("Testing JAX models...")
    
    from QubeSpec.Models_JAX.core_models import halpha_oiii_model
    from QubeSpec.Fitting_JAX.priors import create_prior_function
    from QubeSpec.Fitting_JAX.likelihood import create_log_likelihood_function
    
    # Create test data
    wave = jnp.linspace(1.5, 2.0, 100)
    
    # Generate synthetic spectrum
    flux_true = halpha_oiii_model(
        wave, z=2.0, cont=0.05, cont_grad=-0.1,
        hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
        sii_r_peak=0.05, sii_b_peak=0.03,
        oiii_peak=0.4, hbeta_peak=0.12
    )
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.001, len(wave))
    flux_noisy = np.array(flux_true) + noise
    error = np.full_like(wave, 0.001)
    
    print(f"✓ Generated synthetic data: {len(wave)} points")
    print(f"✓ Flux range: {np.min(flux_noisy):.3f} to {np.max(flux_noisy):.3f}")
    
    # Test priors
    priors = {
        'z': [2.0, 'normal', 2.0, 0.01],
        'cont': [0.05, 'loguniform', -4, 1],
        'cont_grad': [-0.1, 'normal', 0, 0.3],
        'hal_peak': [0.3, 'uniform', 0.0, 1.0],
        'nii_peak': [0.1, 'uniform', 0.0, 1.0],
        'nar_fwhm': [250.0, 'uniform', 100, 900],
        'sii_r_peak': [0.05, 'uniform', 0.0, 1.0],
        'sii_b_peak': [0.03, 'uniform', 0.0, 1.0],
        'oiii_peak': [0.4, 'uniform', 0.0, 1.0],
        'hbeta_peak': [0.12, 'uniform', 0.0, 1.0]
    }
    
    log_prior = create_prior_function(priors)
    test_params = jnp.array([2.0, 0.05, -0.1, 0.3, 0.1, 250.0, 0.05, 0.03, 0.4, 0.12])
    log_prob = log_prior(test_params)
    print(f"✓ Prior evaluation: log_prob = {log_prob}")
    
    # Test likelihood
    log_likelihood = create_log_likelihood_function(halpha_oiii_model, wave, flux_noisy, error)
    log_like = log_likelihood(test_params)
    print(f"✓ Likelihood evaluation: log_like = {log_like}")
    
    # Test JIT compilation
    jitted_model = jax.jit(halpha_oiii_model)
    flux_jitted = jitted_model(wave, *test_params)
    print(f"✓ JIT compilation works: max diff = {jnp.max(jnp.abs(flux_true - flux_jitted))}")
    
    return True

def test_fitting_bridge():
    """Test the fitting bridge class."""
    print("\nTesting fitting bridge...")
    
    from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX
    
    # Create test data
    wave = np.linspace(1.5, 2.0, 100)
    flux = np.sin(wave) + 0.05
    error = np.full_like(wave, 0.01)
    
    # Initialize fitter
    fitter = FittingJAX(wave, flux, error, z=2.0, N=100)
    print("✓ FittingJAX initialized successfully")
    
    # Test parameter setup
    fitter.set_model('halpha')
    print("✓ Model set to halpha")
    
    # Test prior setup
    fitter.set_priors('default')
    print("✓ Default priors set")
    
    return True

def main():
    """Run all simple tests."""
    print("=== Simple JAX Tests ===\n")
    
    try:
        test_jax_models()
        test_fitting_bridge()
        print("\n✓ All simple tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)