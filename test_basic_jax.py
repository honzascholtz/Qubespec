#!/usr/bin/env python3
"""
Basic JAX test to identify issues step by step.
"""

import sys
import traceback

def test_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except Exception as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import jax
        import jax.numpy as jnp
        print("✓ jax imported successfully")
    except Exception as e:
        print(f"✗ jax import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except Exception as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    return True

def test_jax_basic():
    """Test basic JAX functionality."""
    print("\nTesting basic JAX functionality...")
    
    try:
        import jax
        import jax.numpy as jnp
        
        # Test basic operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sin(x)
        print(f"✓ Basic JAX operations work: sin([1,2,3]) = {y}")
        
        # Test JIT compilation
        @jax.jit
        def test_func(x):
            return jnp.sum(x ** 2)
        
        result = test_func(x)
        print(f"✓ JIT compilation works: sum(x^2) = {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX basic test failed: {e}")
        traceback.print_exc()
        return False

def test_jax_model_import():
    """Test importing our JAX models."""
    print("\nTesting JAX model imports...")
    
    try:
        # Add current directory to path for imports
        sys.path.insert(0, '/home/will/jens/Qubespec')
        
        from QubeSpec.Models_JAX.core_models import gauss_jax
        print("✓ gauss_jax imported successfully")
        
        # Test the function
        import jax.numpy as jnp
        x = jnp.linspace(0, 10, 100)
        y = gauss_jax(x, 1.0, 5.0, 1.0)
        print(f"✓ gauss_jax function works: max = {jnp.max(y):.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX model import failed: {e}")
        traceback.print_exc()
        return False

def test_jax_model_evaluation():
    """Test JAX model evaluation."""
    print("\nTesting JAX model evaluation...")
    
    try:
        sys.path.insert(0, '/home/will/jens/Qubespec')
        
        from QubeSpec.Models_JAX.core_models import halpha_oiii_model
        import jax.numpy as jnp
        import jax
        
        # Create test data
        wave = jnp.linspace(1.5, 2.0, 100)
        
        # Test model evaluation
        flux = halpha_oiii_model(
            wave, z=2.0, cont=0.05, cont_grad=-0.1,
            hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
            sii_r_peak=0.05, sii_b_peak=0.03,
            oiii_peak=0.4, hbeta_peak=0.12
        )
        
        print(f"✓ halpha_oiii_model works: flux range = {jnp.min(flux):.3f} to {jnp.max(flux):.3f}")
        
        # Test JIT compilation
        jitted_model = jax.jit(halpha_oiii_model)
        flux_jitted = jitted_model(
            wave, z=2.0, cont=0.05, cont_grad=-0.1,
            hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
            sii_r_peak=0.05, sii_b_peak=0.03,
            oiii_peak=0.4, hbeta_peak=0.12
        )
        
        print(f"✓ JIT compilation works: results match = {jnp.allclose(flux, flux_jitted)}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX model evaluation failed: {e}")
        traceback.print_exc()
        return False

def test_priors_import():
    """Test importing priors module."""
    print("\nTesting priors import...")
    
    try:
        sys.path.insert(0, '/home/will/jens/Qubespec')
        
        from QubeSpec.Fitting_JAX.priors import uniform_prior, normal_prior
        import jax.numpy as jnp
        
        # Test uniform prior
        log_prob = uniform_prior(0.5, 0.0, 1.0)
        print(f"✓ uniform_prior works: log_prob = {log_prob}")
        
        # Test normal prior  
        log_prob = normal_prior(0.0, 0.0, 1.0)
        print(f"✓ normal_prior works: log_prob = {log_prob}")
        
        return True
        
    except Exception as e:
        print(f"✗ Priors import failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Basic JAX Test Suite ===\n")
    
    tests = [
        test_imports,
        test_jax_basic,
        test_jax_model_import,
        test_jax_model_evaluation,
        test_priors_import
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("✓ All basic tests passed!")
        return True
    else:
        print("✗ Some tests failed. Check output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)