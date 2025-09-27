# QubeSpec JAX Conversion Guide

This guide explains the JAX + BlackJAX nested sampling conversion for QubeSpec, providing significantly improved performance and Bayesian evidence calculation.

## Overview

The JAX conversion replaces the original NumPy/emcee-based fitting with a modern, high-performance implementation using:
- **JAX**: Just-in-time compilation and automatic differentiation
- **BlackJAX**: GPU-native nested sampling
- **Anesthetic**: Advanced posterior analysis and visualization

## Key Improvements

### Performance
- **10-100x speedup** on GPU hardware
- **Vectorized cube analysis** using `jax.vmap`
- **JIT compilation** for optimized model evaluation

### Scientific Capabilities
- **Bayesian evidence calculation** for robust model comparison
- **Nested sampling** provides better exploration of multimodal posteriors
- **GPU/TPU acceleration** for large-scale analysis

### Code Quality
- **Functional programming** paradigm improves reproducibility
- **Automatic differentiation** enables advanced optimization
- **Type hints** and modern Python practices

## Installation

```bash
# Install JAX dependencies
pip install -r requirements_jax.txt

# For GPU support (NVIDIA)
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Usage

### Basic Fitting (Drop-in Replacement)

```python
from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX

# Create fitting object (same interface as original)
fitter = FittingJAX(
    wave=wavelength,
    flux=flux_data,
    error=error_data,
    z=redshift,
    N=1000  # Iterations (adapted for nested sampling)
)

# Fit Halpha + [OIII] (same method names)
fitter.fitting_collapse_Halpha_OIII(models='Single_only')

# Access results (same interface)
print(f"Chi-squared: {fitter.chi2}")
print(f"BIC: {fitter.BIC}")
print(f"Parameters: {fitter.props}")

# New: Get Bayesian evidence
logz, logz_err = fitter.get_evidence()
print(f"Log evidence: {logz:.2f} ± {logz_err:.2f}")
```

### Advanced Usage

```python
from QubeSpec.Fitting_JAX.nested_sampling import JAXNestedSampler, create_default_priors
from QubeSpec.Models_JAX.core_models import halpha_oiii_model
import jax.random as random

# Create custom sampler
rng_key = random.PRNGKey(42)
sampler = JAXNestedSampler(rng_key, num_live=500, num_delete=50)

# Create custom priors
priors, param_names = create_default_priors('halpha_oiii', z=2.0)
priors['hal_peak'] = [0.1, 'loguniform', -4, 0]  # Custom prior

# Run nested sampling
results = sampler.fit_spectrum(
    wavelength=wave,
    flux=flux,
    error=error,
    model_name='halpha_oiii',
    priors=priors,
    param_names=param_names
)

# Advanced analysis with anesthetic
samples = results.samples
samples.plot_2d(['hal_peak', 'oiii_peak'])  # Corner plot
print(samples.mean())  # Parameter means
print(samples.cov())   # Covariance matrix
```

### Vectorized Cube Analysis

```python
# Fit entire cube (vectorized)
cube_results = sampler.fit_cube_vectorized(
    wavelength=wave,
    flux_cube=flux_cube,  # (n_spaxels, n_wavelength)
    error_cube=error_cube,
    model_name='halpha_oiii',
    priors=priors,
    param_names=param_names,
    batch_size=100  # Process in batches
)

# Extract parameter maps
hal_peak_map = np.array([r.samples.mean()['hal_peak'] for r in cube_results])
oiii_peak_map = np.array([r.samples.mean()['oiii_peak'] for r in cube_results])
```

## Available Models

### Core Models
- `halpha`: Halpha + [NII] + [SII]
- `oiii`: [OIII] + Hbeta  
- `halpha_oiii`: Combined Halpha + [OIII] region
- `halpha_oiii_outflow`: With outflow components

### Model Functions
All models are pure JAX functions with JIT compilation:

```python
from QubeSpec.Models_JAX.core_models import halpha_oiii_model

# Evaluate model
flux = halpha_oiii_model(
    wavelength, z=2.0, cont=0.05, cont_grad=-0.1,
    hal_peak=0.3, nii_peak=0.1, nar_fwhm=250.0,
    sii_r_peak=0.05, sii_b_peak=0.03,
    oiii_peak=0.4, hbeta_peak=0.12
)
```

## Model Comparison

The JAX version provides Bayesian evidence for robust model comparison:

```python
# Fit different models
fitter_simple = FittingJAX(wave, flux, error, z=2.0)
fitter_simple.fitting_collapse_Halpha_OIII(models='Single_only')

fitter_outflow = FittingJAX(wave, flux, error, z=2.0)
fitter_outflow.fitting_collapse_Halpha_OIII(models='Outflow_only')

# Compare evidences
logz1, _ = fitter_simple.get_evidence()
logz2, _ = fitter_outflow.get_evidence()

bayes_factor = logz1 - logz2
if bayes_factor > 2.5:
    print("Strong evidence for simple model")
elif bayes_factor < -2.5:
    print("Strong evidence for outflow model")
```

## Prior Specifications

Priors are specified in the same format as the original QubeSpec:

```python
priors = {
    'z': [2.0, 'normal_hat', 2.0, 0.01, 1.8, 2.2],  # Truncated normal
    'cont': [0.05, 'loguniform', -4, 1],             # Log-uniform
    'hal_peak': [0.3, 'loguniform', -3, 1],          # Log-uniform
    'nar_fwhm': [250, 'uniform', 100, 900],          # Uniform
    'cont_grad': [0.0, 'normal', 0, 0.3]             # Normal
}
```

### Supported Prior Types
- `uniform`: Uniform distribution
- `loguniform`: Log-uniform distribution  
- `normal`: Normal distribution
- `normal_hat`: Truncated normal distribution
- `lognormal`: Log-normal distribution
- `lognormal_hat`: Truncated log-normal distribution

## Performance Optimization

### GPU Usage
```python
# Check GPU availability
import jax
print(jax.devices())  # Should show GPU devices

# JAX automatically uses GPU when available
# No code changes needed
```

### Memory Management
```python
# For large cubes, use batching
results = sampler.fit_cube_vectorized(
    wavelength=wave,
    flux_cube=large_flux_cube,
    error_cube=large_error_cube,
    model_name='halpha_oiii',
    priors=priors,
    param_names=param_names,
    batch_size=50  # Reduce if GPU memory is limited
)
```

### JIT Compilation
```python
# First run includes compilation overhead
# Subsequent runs are much faster
import jax

# Pre-compile model
model_jit = jax.jit(halpha_oiii_model)
flux = model_jit(wave, **params)  # Compiled on first call
flux = model_jit(wave, **params)  # Fast on subsequent calls
```

## Migration from Original Code

### Code Changes Required

1. **Import changes**:
```python
# Old
from QubeSpec.Fitting.fits_r import Fitting

# New
from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX
```

2. **Method compatibility**:
- All original method names work (`fitting_collapse_Halpha_OIII`, etc.)
- Same parameter interface
- Same results format (`fitter.props`, `fitter.chains`, etc.)

3. **New capabilities**:
```python
# Bayesian evidence
logz, logz_err = fitter.get_evidence()

# Anesthetic samples
samples = fitter.get_nested_samples()

# GPU acceleration (automatic)
# No code changes needed
```

### Backward Compatibility

The `FittingJAX` class maintains full backward compatibility:
- Same method names and signatures
- Same results format
- Same plotting interface
- Same save/load functionality

## Testing

Run the test suite to verify the conversion:

```bash
python test_jax_conversion.py
```

This will:
1. Test JAX model functions
2. Compare fitting results
3. Demonstrate model comparison
4. Generate comparison plots

## Troubleshooting

### Common Issues

1. **GPU out of memory**:
   - Reduce `batch_size` in vectorized fitting
   - Use `jax.config.update('jax_platform_name', 'cpu')` to force CPU

2. **Slow first run**:
   - Normal due to JIT compilation
   - Subsequent runs will be much faster

3. **Import errors**:
   - Ensure all JAX dependencies are installed
   - Check `pip install -r requirements_jax.txt`

### Performance Tips

1. **Use appropriate batch sizes**:
   - GPU: batch_size=100-1000
   - CPU: batch_size=10-100

2. **Pre-compile models**:
   - Use `jax.jit` for repeated model evaluations
   - First call includes compilation overhead

3. **Monitor memory usage**:
   - Use `nvidia-smi` to monitor GPU memory
   - Reduce batch size if memory is limited

## Future Extensions

The JAX conversion enables several advanced features:

1. **Hamiltonian Monte Carlo (HMC)**:
   - Automatic differentiation enables gradient-based sampling
   - More efficient than nested sampling for high-dimensional problems

2. **Simulation-Based Inference (SBI)**:
   - Neural posterior estimation
   - Likelihood-free inference

3. **Advanced Model Selection**:
   - Nested sampling provides robust evidence calculation
   - Model averaging and selection

4. **Scalable Analysis**:
   - TPU support for massive datasets
   - Distributed computing across multiple devices

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [BlackJAX Documentation](https://blackjax-devs.github.io/blackjax/)
- [Anesthetic Documentation](https://anesthetic.readthedocs.io/)
- [Nested Sampling Theory](https://projecteuclid.org/euclid.ba/1340370944)