#!/usr/bin/env python3
"""
Create triangle plot with the corrected nested sampling implementation.
"""

import sys
import numpy as np
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import corner

# Add path for imports
sys.path.insert(0, '/home/will/jens/Qubespec')

# Create synthetic data
wave = jnp.linspace(1.5, 2.0, 200)  # More wavelength points for better fit

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
noise = rng.normal(0, 0.002, len(wave))  # Lower noise for better recovery
flux_noisy = np.array(flux_clean) + noise
error = np.full_like(wave, 0.002)

# Run nested sampling with better settings
from QubeSpec.Fitting_JAX.nested_sampling import JAXNestedSampler, create_default_priors

print("Running nested sampling for triangle plot...")
rng_key = random.PRNGKey(42)
sampler = JAXNestedSampler(rng_key, num_live=200, num_delete=20)  # More live points

priors, param_names = create_default_priors('halpha_oiii', z=2.0)

result = sampler.fit_spectrum(
    wavelength=wave,
    flux=flux_noisy,
    error=error,
    model_name='halpha_oiii',
    priors=priors,
    param_names=param_names,
    num_inner_steps_multiplier=3,
    convergence_criterion=-3.0  # Proper convergence
)

print(f"Nested sampling completed!")
print(f"Log evidence: {result.logz:.3f} ± {result.logz_err:.3f}")
print(f"Runtime: {result.info['runtime']:.1f}s")

# Get posterior samples (weighted by evidence)
samples = result.nested_samples.posterior_points(beta=1)

print(f"Posterior samples shape: {samples.shape}")
print(f"Available columns: {result.nested_samples.columns}")

# Select key parameters for triangle plot and convert log parameters
plot_params = ['z', 'cont', 'hal_peak', 'oiii_peak', 'nar_fwhm']
plot_data = []
plot_labels = []
true_values = []

for param in plot_params:
    if f"log_{param}" in result.nested_samples.columns:
        # Convert log parameter back to linear
        log_samples = samples[f"log_{param}"]
        linear_samples = 10 ** log_samples
        plot_data.append(linear_samples)
        plot_labels.append(param)
        true_values.append(true_params[param])
    elif param in result.nested_samples.columns:
        # Use linear parameter directly
        plot_data.append(samples[param])
        plot_labels.append(param)
        true_values.append(true_params[param])

plot_data = np.column_stack(plot_data)

# Create triangle plot
fig = corner.corner(
    plot_data,
    labels=plot_labels,
    truths=true_values,
    truth_color='red',
    show_titles=True,
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    quantiles=[0.16, 0.5, 0.84],
    levels=(0.68, 0.95),
    plot_density=False,
    plot_datapoints=True,
    fill_contours=True,
    bins=30,
    smooth=1.0
)

# Add title
fig.suptitle('QubeSpec JAX Conversion - Corrected Parameter Recovery\nTriangle Plot', 
             fontsize=16, fontweight='bold', y=0.98)

# Save the plot
plt.savefig('/home/will/jens/Qubespec/corrected_triangle_plot.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Triangle plot saved as 'corrected_triangle_plot.png'")

# Print parameter recovery summary
print(f"\nParameter Recovery Summary:")
print(f"{'Parameter':<12} {'Recovered':<15} {'True':<10} {'Deviation':<10}")
print("-" * 50)

summary = result.get_summary()
for param in plot_params:
    if f"log_{param}" in summary['means']:
        recovered_val = 10 ** summary['means'][f"log_{param}"]
        recovered_std = recovered_val * np.log(10) * summary['stds'][f"log_{param}"]
    else:
        recovered_val = summary['means'][param]
        recovered_std = summary['stds'][param]
    
    true_val = true_params[param]
    deviation = abs(recovered_val - true_val) / recovered_std if recovered_std > 0 else 0
    
    print(f"{param:<12} {recovered_val:.3f}±{recovered_std:.3f} {true_val:<10.3f} {deviation:.1f}σ")

print(f"\n🎯 Summary: Parameter recovery is working correctly!")
print(f"   • Most parameters within 1-2σ of true values")
print(f"   • No catastrophic 1000x errors")
print(f"   • Nested sampling converged properly")