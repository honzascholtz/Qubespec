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

# Create synthetic data
wave = jnp.linspace(1.5, 2.0, 100)  # Smaller for speed

true_params = {
    'z': 2.0,
    'cont': 0.03,
    'cont_grad': -0.05,
    'hal_peak': 0.4,
    'nii_peak': 0.12,
    'nar_fwhm': 280.0,
    'sii_r_peak': 0.06,
    'sii_b_peak': 0.04,
    'oiii_peak': 0.35,
    'hbeta_peak': 0.1
}

from QubeSpec.Models_JAX.core_models import halpha_oiii_model

# Generate clean spectrum
flux_clean = halpha_oiii_model(wave, **true_params)

# Add noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.003, len(wave))
flux_noisy = np.array(flux_clean) + noise
error = np.full_like(wave, 0.003)

# Test nested sampling
from QubeSpec.Fitting_JAX.nested_sampling import JAXNestedSampler, create_default_priors

print("=== JAX Parameter Recovery Test ===")
rng_key = random.PRNGKey(42)
sampler = JAXNestedSampler(rng_key, num_live=50, num_delete=10)

priors, param_names = create_default_priors('halpha_oiii', z=2.0)
print(f"Priors: {priors}")
print(f"Parameter names: {param_names}")

result = sampler.fit_spectrum(
    wavelength=wave,
    flux=flux_noisy,
    error=error,
    model_name='halpha_oiii',
    priors=priors,
    param_names=param_names,
    num_inner_steps_multiplier=2,
    convergence_criterion=-1.5  # Quick convergence for testing
)

print(f"\\nNested sampling completed!")
print(f"Log evidence: {result.logz:.3f} ± {result.logz_err:.3f}")
print(f"Runtime: {result.info['runtime']:.1f}s")
print(f"Iterations: {result.info['num_iterations']}")

# Show the actual parameter columns
print(f"\\nResult columns: {result.nested_samples.columns}")

# Get summary
summary = result.get_summary()
print(f"\\nParameter summary:")
for param, mean_val in summary['means'].items():
    std_val = summary['stds'][param]
    print(f"  {param}: {mean_val:.3f} ± {std_val:.3f}")

# Compare with true values (converting log parameters back)
print(f"\\nComparison with true values:")
for param in param_names:
    # Check if this parameter was fit in log space
    log_param = f"log_{param}"
    if log_param in summary['means']:
        recovered_val = 10 ** summary['means'][log_param]
        recovered_std = recovered_val * np.log(10) * summary['stds'][log_param]
        print(f"  {param}: {recovered_val:.3f} ± {recovered_std:.3f} (true: {true_params[param]:.3f})")
    elif param in summary['means']:
        recovered_val = summary['means'][param]
        recovered_std = summary['stds'][param]
        print(f"  {param}: {recovered_val:.3f} ± {recovered_std:.3f} (true: {true_params[param]:.3f})")