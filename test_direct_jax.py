#!/usr/bin/env python3
"""
Test JAX modules directly without importing the full QubeSpec package.
"""

import sys
import traceback
import importlib.util

def test_direct_model_import():
    """Test importing models directly."""
    print("Testing direct JAX model import...")
    
    try:
        # Load the module directly
        spec = importlib.util.spec_from_file_location(
            "core_models", 
            "/home/will/jens/Qubespec/QubeSpec/Models_JAX/core_models.py"
        )
        core_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_models)
        
        print("✓ core_models module loaded successfully")
        
        # Test basic function
        import jax.numpy as jnp
        x = jnp.linspace(0, 10, 100)
        y = core_models.gauss_jax(x, 1.0, 5.0, 1.0)
        print(f"✓ gauss_jax function works: max = {jnp.max(y):.3f}")
        
        return True, core_models
        
    except Exception as e:
        print(f"✗ Direct model import failed: {e}")
        traceback.print_exc()
        return False, None

def test_direct_model_evaluation(core_models):
    """Test model evaluation directly."""
    print("\nTesting direct model evaluation...")
    
    try:
        import jax.numpy as jnp
        import jax
        
        # Create test data
        wave = jnp.linspace(1.5, 2.0, 100)
        
        # Test model evaluation
        flux = core_models.halpha_oiii_model(
            wave, z=2.0, cont=0.05, cont_grad=-0.1,
            hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
            sii_r_peak=0.05, sii_b_peak=0.03,
            oiii_peak=0.4, hbeta_peak=0.12
        )
        
        print(f"✓ halpha_oiii_model works: flux range = {jnp.min(flux):.3f} to {jnp.max(flux):.3f}")
        
        # Test JIT compilation
        jitted_model = jax.jit(core_models.halpha_oiii_model)
        flux_jitted = jitted_model(
            wave, z=2.0, cont=0.05, cont_grad=-0.1,
            hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
            sii_r_peak=0.05, sii_b_peak=0.03,
            oiii_peak=0.4, hbeta_peak=0.12
        )
        
        print(f"✓ JIT compilation works: results match = {jnp.allclose(flux, flux_jitted)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Direct model evaluation failed: {e}")
        traceback.print_exc()
        return False

def test_direct_priors_import():
    """Test importing priors directly."""
    print("\nTesting direct priors import...")
    
    try:
        # Load the module directly
        spec = importlib.util.spec_from_file_location(
            "priors", 
            "/home/will/jens/Qubespec/QubeSpec/Fitting_JAX/priors.py"
        )
        priors = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(priors)
        
        print("✓ priors module loaded successfully")
        
        # Test uniform prior
        log_prob = priors.uniform_prior(0.5, 0.0, 1.0)
        print(f"✓ uniform_prior works: log_prob = {log_prob}")
        
        # Test normal prior  
        log_prob = priors.normal_prior(0.0, 0.0, 1.0)
        print(f"✓ normal_prior works: log_prob = {log_prob}")
        
        return True, priors
        
    except Exception as e:
        print(f"✗ Direct priors import failed: {e}")
        traceback.print_exc()
        return False, None

def test_prior_functions(priors):
    """Test prior function creation."""
    print("\nTesting prior function creation...")
    
    try:
        import jax.numpy as jnp
        
        # Test prior dictionary
        prior_dict = {
            'z': [2.0, 'normal', 2.0, 0.01],
            'cont': [0.05, 'loguniform', -4, 1],
            'hal_peak': [0.3, 'uniform', 0.0, 1.0]
        }
        
        # Create prior function
        log_prior = priors.create_prior_function(prior_dict)
        print("✓ Prior function created successfully")
        
        # Test evaluation
        params = jnp.array([2.0, 0.05, 0.3])
        log_prob = log_prior(params)
        print(f"✓ Prior evaluation works: log_prob = {log_prob}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prior function test failed: {e}")
        traceback.print_exc()
        return False

def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    
    try:
        import numpy as np
        import jax.numpy as jnp
        
        # Load models
        spec = importlib.util.spec_from_file_location(
            "core_models", 
            "/home/will/jens/Qubespec/QubeSpec/Models_JAX/core_models.py"
        )
        core_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_models)
        
        # Create wavelength array
        wave = jnp.linspace(1.5, 2.0, 1000)
        
        # Generate synthetic spectrum
        flux_clean = core_models.halpha_oiii_model(
            wave, z=2.0, cont=0.05, cont_grad=-0.1,
            hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
            sii_r_peak=0.05, sii_b_peak=0.03,
            oiii_peak=0.4, hbeta_peak=0.12
        )
        
        # Add noise
        rng = np.random.default_rng(42)
        noise_level = 0.01
        noise = rng.normal(0, noise_level * np.max(flux_clean), len(wave))
        flux_noisy = np.array(flux_clean) + noise
        
        # Create error array
        error = np.full_like(wave, noise_level * np.max(flux_clean))
        
        print(f"✓ Synthetic data generated: {len(wave)} points")
        print(f"✓ Flux range: {np.min(flux_noisy):.3f} to {np.max(flux_noisy):.3f}")
        print(f"✓ Error level: {np.mean(error):.3f}")
        
        return True, wave, flux_noisy, error
        
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
        traceback.print_exc()
        return False, None, None, None

def main():
    """Run all tests."""
    print("=== Direct JAX Module Tests ===\n")
    
    # Test model import
    success, core_models = test_direct_model_import()
    if not success:
        print("✗ Cannot continue without models")
        return False
    
    # Test model evaluation
    success = test_direct_model_evaluation(core_models)
    if not success:
        print("✗ Model evaluation failed")
        return False
    
    # Test priors import
    success, priors = test_direct_priors_import()
    if not success:
        print("✗ Cannot continue without priors")
        return False
    
    # Test prior functions
    success = test_prior_functions(priors)
    if not success:
        print("✗ Prior functions failed")
        return False
    
    # Test synthetic data
    success, wave, flux, error = test_synthetic_data_generation()
    if not success:
        print("✗ Synthetic data generation failed")
        return False
    
    print("\n=== All direct tests passed! ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)