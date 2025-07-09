#!/usr/bin/env python3
"""
Test the corrected nested sampling implementation.
"""

import sys
import numpy as np
import jax.numpy as jnp
import jax.random as random

# Add path for imports
sys.path.insert(0, '/home/will/jens/Qubespec')

def test_nested_sampling():
    """Test nested sampling with a simple model."""
    print("Testing nested sampling with corrected API...")
    
    from QubeSpec.Fitting_JAX.nested_sampling import JAXNestedSampler, create_default_priors
    
    # Create simple synthetic data  
    wave = jnp.linspace(1.5, 2.0, 50)  # Smaller dataset for faster testing
    
    # Simple model parameters
    true_params = {
        'z': 2.0,
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
    
    # Generate synthetic spectrum
    from QubeSpec.Models_JAX.core_models import halpha_oiii_model
    flux_true = halpha_oiii_model(wave, **true_params)
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.002, len(wave))
    flux_noisy = np.array(flux_true) + noise
    error = np.full_like(wave, 0.002)
    
    print(f"✓ Generated synthetic data: {len(wave)} points")
    print(f"✓ True parameters: {true_params}")
    
    # Create nested sampler with smaller live points for faster testing
    rng_key = random.PRNGKey(42)
    sampler = JAXNestedSampler(rng_key, num_live=50, num_delete=10)
    
    # Create default priors
    priors, param_names = create_default_priors('halpha_oiii', z=2.0)
    
    print(f"✓ Created priors for {len(param_names)} parameters")
    print(f"✓ Parameter names: {param_names}")
    
    # Run nested sampling (this should not hang now)
    try:
        print("\nRunning nested sampling...")
        result = sampler.fit_spectrum(
            wavelength=wave,
            flux=flux_noisy,
            error=error,
            model_name='halpha_oiii',
            priors=priors,
            param_names=param_names,
            num_inner_steps_multiplier=3,  # Smaller for faster testing
            convergence_criterion=-2.0     # Less stringent for testing
        )
        
        print(f"✓ Nested sampling completed!")
        print(f"✓ Number of iterations: {result.info['num_iterations']}")
        print(f"✓ Runtime: {result.info['runtime']:.2f} seconds")
        print(f"✓ Log evidence: {result.logz:.2f} ± {result.logz_err:.2f}")
        
        # Check parameter recovery
        if result.samples is not None:
            summary = result.get_summary()
            print(f"✓ Parameter recovery:")
            for param in param_names[:5]:  # Show first 5 parameters
                if param in summary['means']:
                    mean = summary['means'][param]
                    std = summary['stds'][param]
                    true_val = true_params.get(param, 'N/A')
                    print(f"  {param}: {mean:.3f} ± {std:.3f} (true: {true_val})")
        
        return True
        
    except Exception as e:
        print(f"✗ Nested sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run nested sampling test."""
    print("=== Nested Sampling Test ===\n")
    
    try:
        success = test_nested_sampling()
        if success:
            print("\n✓ Nested sampling test passed!")
        else:
            print("\n✗ Nested sampling test failed!")
        return success
    except Exception as e:
        print(f"\n✗ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)