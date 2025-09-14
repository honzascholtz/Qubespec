# JAX Analysis Summary - Key Files

## Console Output Issues
```
hal_peak: 493.554 ± 226.649 (true: 0.400) [2.2σ]
nar_fwhm: 5.037 ± 2.835 (true: 280.000) [97.0σ]
nii_peak: 4.949 ± 2.962 (true: 0.120) [1.6σ]
```

## Key Problems:
1. Performance only 1.1x speedup, not 10-100x
2. Parameter recovery catastrophically wrong (1000x larger values)
3. Model seems to fit noise, not signal
4. Most parameters show massive deviations (97σ!)

## Core Model Function (QubeSpec/Models_JAX/core_models.py)
```python
@jit
def fwhm_to_sigma(fwhm: float, center_wavelength: float) -> float:
    """
    Convert FWHM in km/s to sigma in wavelength units.
    """
    # Convert km/s to fractional velocity, then to wavelength
    velocity_fraction = (fwhm / 1000.0) / (C / 1000.0)  # km/s to c
    sigma_wavelength = velocity_fraction * center_wavelength / 2.35482
    return sigma_wavelength
```

## Nested Sampling Implementation
```python
# In nested_sampling.py
def _create_dict_likelihood(self, log_likelihood: Callable, param_names: List[str]) -> Callable:
    """Convert array-based likelihood to dict-based for BlackJAX."""
    def dict_likelihood(params_dict):
        # Convert dict to array in the correct order
        params_array = jnp.array([params_dict[name] for name in param_names])
        return log_likelihood(params_array)
    return dict_likelihood
```

## Vectorization Issue
```python
# This is NOT vectorized - it's a Python loop!
for j in range(batch_flux.shape[0]):
    result = self.fit_spectrum(
        wavelength=wavelength,
        flux=batch_flux[j],
        error=batch_error[j],
        model_name=model_name,
        priors=priors,
        param_names=param_names,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    batch_results.append(result)
```

## Working Workshop Example
```python
# From workshop_nested_sampling.py
def line_loglikelihood(params):
    """Log-likelihood for linear model with Gaussian noise."""
    m, c, sigma = params["m"], params["c"], params["sigma"]
    y_model = m * x + c
    # Vectorized normal log-likelihood
    return jax.scipy.stats.multivariate_normal.logpdf(y, y_model, sigma**2)

# Proper BlackJAX usage
particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)
nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=line_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)
```

## Questions for Analysis:
1. Is the fwhm_to_sigma function causing unit conversion errors?
2. Why is parameter recovery so catastrophically wrong?
3. Is the dict to array conversion in _create_dict_likelihood correct?
4. Why is performance only 1.1x instead of 10-100x?
5. How should true vectorization be implemented?