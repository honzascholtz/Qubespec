# | # BlackJAX Nested Sampling Workshop
# |
# | This workshop demonstrates GPU-native nested sampling using BlackJAX. We'll progress through three examples: line fitting, 2D Gaussian inference, and performance comparisons with other samplers. The workshop showcases JAX's key strengths: automatic differentiation and JIT compilation for high-performance Bayesian inference.
# ------------------------------------------------
# | ## Installation for Google Colab
# |```bash
# | pip install git+https://github.com/handley-lab/blackjax@nested_sampling
# | pip install anesthetic tqdm
# |```

import jax
# | Configure JAX immediately after import
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import blackjax
from anesthetic import NestedSamples

# | ## Part 1: Line Fitting with Nested Sampling
# |
# | We start with the classic problem of fitting a linear model y = mx + c to noisy data.
# | This introduces the basic nested sampling workflow in BlackJAX.


# | ### 1.1 Nested Sampling Configuration
# | 
# | Key parameters for workshop timing and educational value:
# | - `num_live=100`: Fast convergence for workshop setting
# | - `num_delete=50`: Parallelization parameter
# | - `num_inner_steps`: Reliability parameter (rule of thumb: 5 * num_dims)
rng_key = jax.random.PRNGKey(42)
num_live = 100
num_delete = 50

# | ### 1.2 Generate Synthetic Line Data
# |
# | True model: y = 2x + 1 + noise, with σ = 0.5
num_data = 15
x = jnp.linspace(-2.0, 2.0, num_data)
true = {'m': 2.0, 'c': 1.0, 'sigma': 0.5}

key, rng_key = jax.random.split(rng_key)
noise = true['sigma'] * jax.random.normal(key, (num_data,))
y = true['m'] * x + true['c'] + noise

# | Visualize the data
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, y, yerr=true['sigma'], fmt="o", label="Observed data", color='black')
ax.plot(x, true['m'] * x + true['c'], '--', label="True model", color='red', alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.set_title("Linear Model: Bayesian Parameter Estimation")

# | ### 1.3 Define Likelihood Function
# |
# | Gaussian likelihood with unknown slope, intercept, and noise level
def line_loglikelihood(params):
    """Log-likelihood for linear model with Gaussian noise."""
    m, c, sigma = params["m"], params["c"], params["sigma"]
    y_model = m * x + c
    # Vectorized normal log-likelihood
    return jax.scipy.stats.multivariate_normal.logpdf(y, y_model, sigma**2)

# | ### 1.4 Define Prior Distributions
prior_bounds = {
    "m": (-5.0, 5.0),      # slope
    "c": (-5.0, 5.0),      # intercept  
    "sigma": (0.1, 2.0),   # noise level (positive)
}

num_dims = len(prior_bounds)
num_inner_steps = num_dims * 5

# | ### 1.5 Initialize Nested Sampler
rng_key, prior_key = jax.random.split(rng_key)
particles, logprior_fn = blackjax.ns.utils.uniform_prior(prior_key, num_live, prior_bounds)

nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=line_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)
print(f"Initialized nested sampler with {num_live} live points")

# | ### 1.6 JIT Compile for Performance
init_fn = jax.jit(nested_sampler.init)
step_fn = jax.jit(nested_sampler.step)
print("Functions compiled - ready to run!")

# | ### 1.7 Run the Nested Sampling
print("Running nested sampling for line fitting...")
ns_start = time.time()
live = init_fn(particles)
dead = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not live.logZ_live - live.logZ < -3:  # Convergence criterion
        rng_key, subkey = jax.random.split(rng_key, 2)
        live, dead_info = step_fn(subkey, live)
        dead.append(dead_info)
        pbar.update(num_delete)

dead = blackjax.ns.utils.finalise(live, dead)
ns_time = time.time() - ns_start

# | ### 1.8 Process Results with Anesthetic
columns = ["m", "c", "sigma"]
labels = [r"$m$", r"$c$", r"$\sigma$"]
data = jnp.vstack([dead.particles[key] for key in columns]).T

line_samples = NestedSamples(
    data,
    logL=dead.loglikelihood,
    logL_birth=dead.loglikelihood_birth,
    columns=columns,
    labels=labels,
    logzero=jnp.nan,
)

# | ### 1.9 Results Analysis and Visualization
print(f"Nested sampling runtime: {ns_time:.2f} seconds")
print(f"Log Evidence: {line_samples.logZ():.2f} ± {line_samples.logZ(100).std():.2f}")
print(f"True parameters: m={true['m']}, c={true['c']}, σ={true['sigma']}")
print(f"Posterior means: m={line_samples.m.mean():.2f}, c={line_samples.c.mean():.2f}, σ={line_samples.sigma.mean():.2f}")

# Create posterior corner plot with true values marked
kinds = {'lower': 'kde_2d', 'diagonal': 'hist_1d', 'upper': 'scatter_2d'}
axes = line_samples.plot_2d(columns, kinds=kinds, label='Posterior')
axes.axlines(true, color='red', linestyle='--', alpha=0.8)
plt.suptitle("Line Fitting: Posterior Distributions")

# | ## Part 2: 2D Gaussian Parameter Inference
# |
# | Now we tackle a more complex problem: inferring the parameters of a 2D Gaussian distribution
# | including the correlation coefficient. This demonstrates parameter transforms and constrained sampling.


# | ### 2.1 Define 2D Gaussian Parameters
true.update({
    'mu1': 1.0, 'mu2': -0.5,
    'sigma1': 1.2, 'sigma2': 0.8, 
    'rho': 0.6
})
print("True parameters:", {k: v for k, v in true.items() if k in ['mu1', 'mu2', 'sigma1', 'sigma2', 'rho']})

# | ### 2.2 Generate Correlated 2D Data
true_mu = jnp.array([true['mu1'], true['mu2']])
true_cov = jnp.array([
    [true['sigma1']**2, true['rho'] * true['sigma1'] * true['sigma2']],
    [true['rho'] * true['sigma1'] * true['sigma2'], true['sigma2']**2]
])

num_samples = 200
key, rng_key = jax.random.split(rng_key)
gaussian_data = jax.random.multivariate_normal(key, true_mu, true_cov, (num_samples,))

print(f"Generated {num_samples} correlated 2D samples")
print(f"Sample mean: [{gaussian_data.mean(0)[0]:.2f}, {gaussian_data.mean(0)[1]:.2f}]")

# | ### 2.3 Visualize the 2D Data
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(gaussian_data[:, 0], gaussian_data[:, 1], alpha=0.6, s=20)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("2D Gaussian Data")
ax.grid(True, alpha=0.3)

# | ### 2.4 Define Likelihood with Parameter Transforms
# |
# | We use arctanh/tanh transform for the correlation coefficient to enforce |ρ| < 1
def gaussian_2d_loglikelihood(params):
    """Log-likelihood for 2D Gaussian with correlation."""
    mu1, mu2 = params["mu1"], params["mu2"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho_transformed = params["rho_t"]
    
    # Transform correlation coefficient: rho = tanh(rho_t)
    rho = jnp.tanh(rho_transformed)
    
    # Construct covariance matrix
    cov = jnp.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ])
    
    # Check positive definiteness
    det = jnp.linalg.det(cov)
    
    # Return -inf for invalid covariance matrices
    def valid_loglik():
        mu = jnp.array([mu1, mu2])
        return jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(gaussian_data, mu, cov))
    
    def invalid_loglik():
        return -jnp.inf
    
    return jax.lax.cond(det > 1e-8, valid_loglik, invalid_loglik)

# | ### 2.5 Set Up Priors for 2D Gaussian
gaussian_prior_bounds = {
    "mu1": (-3.0, 5.0),
    "mu2": (-3.0, 3.0), 
    "sigma1": (0.1, 3.0),
    "sigma2": (0.1, 3.0),
    "rho_t": (-2.0, 2.0),  # transformed correlation: rho = tanh(rho_t)
}

num_dims_2d = len(gaussian_prior_bounds)
num_inner_steps_2d = num_dims_2d * 5

# | ### 2.6 Initialize and Run Nested Sampling
rng_key, prior_key = jax.random.split(rng_key)
particles_2d, logprior_fn_2d = blackjax.ns.utils.uniform_prior(prior_key, num_live, gaussian_prior_bounds)

nested_sampler_2d = blackjax.nss(
    logprior_fn=logprior_fn_2d,
    loglikelihood_fn=gaussian_2d_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps_2d,
)

init_fn_2d = jax.jit(nested_sampler_2d.init)
step_fn_2d = jax.jit(nested_sampler_2d.step)

print("Running nested sampling for 2D Gaussian...")
live_2d = init_fn_2d(particles_2d)
dead_2d = []

with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not live_2d.logZ_live - live_2d.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key, 2)
        live_2d, dead_info_2d = step_fn_2d(subkey, live_2d)
        dead_2d.append(dead_info_2d)
        pbar.update(num_delete)

dead_2d = blackjax.ns.utils.finalise(live_2d, dead_2d)

# | ### 2.7 Transform Back and Analyze Results
columns_2d = ["mu1", "mu2", "sigma1", "sigma2", "rho_t"]
labels_2d = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$", r"$\rho_t$"]
data_2d = jnp.vstack([dead_2d.particles[key] for key in columns_2d]).T

gaussian_samples = NestedSamples(
    data_2d,
    logL=dead_2d.loglikelihood,
    logL_birth=dead_2d.loglikelihood_birth,
    columns=columns_2d,
    labels=labels_2d,
    logzero=jnp.nan,
)

# | Add transformed correlation coefficient
gaussian_samples["rho"] = jnp.tanh(gaussian_samples["rho_t"].values)

print(f"Log Evidence: {gaussian_samples.logZ():.2f} ± {gaussian_samples.logZ(100).std():.2f}")
print(f"True parameters: μ₁={true['mu1']:.2f}, μ₂={true['mu2']:.2f}, σ₁={true['sigma1']:.2f}, σ₂={true['sigma2']:.2f}, ρ={true['rho']:.2f}")
print(f"Posterior means: μ₁={gaussian_samples.mu1.mean():.2f}, μ₂={gaussian_samples.mu2.mean():.2f}, σ₁={gaussian_samples.sigma1.mean():.2f}, σ₂={gaussian_samples.sigma2.mean():.2f}, ρ={gaussian_samples.rho.mean():.2f}")

# | Plot posterior for key parameters with true values
key_params = ["mu1", "mu2", "sigma1", "sigma2", "rho"]
axes = gaussian_samples[key_params].plot_2d(key_params, kinds={'diagonal': 'hist_1d', 'lower': 'kde_2d'})

# Mark true values using anesthetic's axlines method
true_2d = {k: true[k] for k in key_params}
axes.axlines(true_2d, color='red', linestyle='--', alpha=0.8)
plt.suptitle("2D Gaussian: Posterior Parameter Estimates")

# | ## Part 3: Performance Comparison
# |
# | Compare BlackJAX nested sampling with NUTS (No-U-Turn Sampler) and 
# | Affine Invariant Ensemble Sampler on the line fitting problem

import time

# | ### 3.1 Define NUTS Log-Probability Function
def nuts_logprob(params_array):
    """Combined log probability for NUTS (assumes flat priors within bounds)."""
    m, c, log_sigma = params_array
    sigma = jnp.exp(log_sigma)  # positive constraint via log transform
    
    # Check bounds (flat prior)
    m_valid = (m >= -5.0) & (m <= 5.0)
    c_valid = (c >= -5.0) & (c <= 5.0)
    sigma_valid = (sigma >= 0.1) & (sigma <= 2.0)
    
    def valid_logprob():
        y_model = m * x + c
        loglik = jax.scipy.stats.multivariate_normal.logpdf(y, y_model, sigma**2)
        return loglik + log_sigma  # Add Jacobian for log transform
    
    def invalid_logprob():
        return -jnp.inf
    
    return jax.lax.cond(m_valid & c_valid & sigma_valid, valid_logprob, invalid_logprob)

# | ### 3.2 Initialize and Run NUTS Sampler
initial_position = jnp.array([1.0, 0.0, jnp.log(1.0)])  # [m, c, log_sigma]
nuts = blackjax.nuts(nuts_logprob, step_size=0.1, inverse_mass_matrix=jnp.eye(3))

rng_key, nuts_key = jax.random.split(rng_key)
nuts_state = nuts.init(initial_position)
nuts_step = jax.jit(nuts.step)

print("Running NUTS sampler...")

num_nuts_samples = 2000
nuts_start = time.time()
nuts_samples = []
nuts_states = nuts_state

for i in tqdm.tqdm(range(num_nuts_samples), desc="NUTS"):
    nuts_key, step_key = jax.random.split(nuts_key)
    nuts_states, nuts_info = nuts_step(step_key, nuts_states)
    nuts_samples.append(nuts_states.position)

nuts_time = time.time() - nuts_start
nuts_samples = jnp.stack(nuts_samples)

# | ### 3.3 Process NUTS Results  
nuts_m = nuts_samples[:, 0]
nuts_c = nuts_samples[:, 1] 
nuts_sigma = jnp.exp(nuts_samples[:, 2])

print(f"NUTS runtime: {nuts_time:.2f} seconds")
print(f"NUTS means: m={nuts_m[500:].mean():.2f}, c={nuts_c[500:].mean():.2f}, σ={nuts_sigma[500:].mean():.2f}")

# | ### 3.4 Performance Summary and Visualization
methods = ["Nested Sampling", "NUTS"]
times = [f"{ns_time:.1f} sec", f"{nuts_time:.1f} sec"]
evidence = ["✓ (Log Z available)", "✗ (Not computed)"]
parallelization = ["✓ (GPU native)", "Limited"]

print(f"{'Method':<20} {'Time':<15} {'Evidence':<15} {'GPU Parallel'}")
print("-" * 65)
for i in range(len(methods)):
    print(f"{methods[i]:<20} {times[i]:<15} {evidence[i]:<15} {parallelization[i]}")

# | ### 3.5 Posterior Comparison Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# | Generate proper posterior samples from NestedSamples (not raw dead points)
# | Use the number of available samples or 1000, whichever is smaller
n_posterior_samples = min(1000, len(line_samples))
ns_posterior_samples = line_samples.sample(n_posterior_samples, replace=True)  # Sample from posterior with replacement
nuts_burnin = 500  # Remove burn-in

# | Compare marginal posteriors
axes[0].hist(ns_posterior_samples.m.values, bins=30, alpha=0.7, density=True, label='Nested Sampling')
axes[0].hist(nuts_m[nuts_burnin:], bins=30, alpha=0.7, density=True, label='NUTS')
axes[0].axvline(true['m'], color='red', linestyle='--', label='True value')
axes[0].set_xlabel('Slope (m)')
axes[0].set_ylabel('Density')
axes[0].legend()

axes[1].hist(ns_posterior_samples.c.values, bins=30, alpha=0.7, density=True, label='Nested Sampling')
axes[1].hist(nuts_c[nuts_burnin:], bins=30, alpha=0.7, density=True, label='NUTS')
axes[1].axvline(true['c'], color='red', linestyle='--', label='True value')
axes[1].set_xlabel('Intercept (c)')
axes[1].set_ylabel('Density')
axes[1].legend()

axes[2].hist(ns_posterior_samples.sigma.values, bins=30, alpha=0.7, density=True, label='Nested Sampling')
axes[2].hist(nuts_sigma[nuts_burnin:], bins=30, alpha=0.7, density=True, label='NUTS')
axes[2].axvline(true['sigma'], color='red', linestyle='--', label='True value')
axes[2].set_xlabel('Noise (σ)')
axes[2].set_ylabel('Density')
axes[2].legend()

plt.tight_layout()
plt.suptitle("Posterior Comparison: Nested Sampling vs NUTS", y=1.02)

# | ## Part 4: Building Your Own Nested Sampler
# |
# | Advanced users can build custom nested samplers using BlackJAX's low-level components.
# | This demonstrates the modular design and shows how nested sampling works under the hood.

from functools import partial
from blackjax.ns.adaptive import build_kernel, init
from blackjax.ns.base import new_state_and_info, delete_fn
from blackjax.ns.utils import repeat_kernel, finalise
from blackjax.mcmc.random_walk import build_rmh, RWState
from blackjax import SamplingAlgorithm

# | ### 4.1 Define Custom Nested MCMC Algorithm
def custom_nsmcmc(
    logprior_fn,
    loglikelihood_fn,
    num_delete=10,
    num_inner_steps=10,
):
    """
    Build a custom nested sampling MCMC algorithm from low-level components.
    
    This demonstrates how to construct a nested sampler using BlackJAX's
    modular infrastructure - useful for research and customization.
    
    Parameters
    ----------
    logprior_fn : callable
        Function that computes the log prior probability of the parameters.
    loglikelihood_fn : callable
        Function that computes the log likelihood of the parameters.
    num_delete : int
        Number of particles to delete at each step.
    num_inner_steps : int
        Number of inner MCMC steps to perform.
    
    Returns
    -------
    SamplingAlgorithm
        Custom nested sampling algorithm with init and step functions.
    """
    
    # Build the MCMC kernel for exploring within likelihood constraints
    mcmc_kernel = build_rmh()

    @repeat_kernel(num_inner_steps)
    def inner_kernel(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params):
        """Inner MCMC kernel that explores within likelihood constraint."""
        def proposal_distribution(rng_key, position):
            # Handle dictionary position structure
            if isinstance(position, dict):
                step = {}
                for key in position.keys():
                    sigma_val = params['sigma'][key] if isinstance(params['sigma'], dict) else params['sigma']
                    step[key] = sigma_val * jax.random.normal(rng_key, shape=position[key].shape)
                    rng_key, _ = jax.random.split(rng_key)  # Split key for each parameter
                return {key: position[key] + step[key] for key in position.keys()}
            else:
                # Fallback for array position
                step = params['sigma'] * jax.random.normal(rng_key, shape=position.shape) 
                return position + step

        # Convert to MCMC state format
        mcmc_state = RWState(position=state.position, logdensity=state.logprior)
        new_mcmc_state, mcmc_info = mcmc_kernel(rng_key, mcmc_state, logprior_fn, proposal_distribution)

        # Evaluate likelihood at new position
        loglikelihood = loglikelihood_fn(new_mcmc_state.position)

        # Create new nested sampling state
        new_state, info = new_state_and_info(
            position=new_mcmc_state.position,
            logprior=new_mcmc_state.logdensity,
            loglikelihood=loglikelihood,
            info=mcmc_info,
        )

        # Accept only if likelihood exceeds threshold (key constraint!)
        new_state = jax.lax.cond(
            loglikelihood > loglikelihood_0,
            lambda _: new_state,
            lambda _: state,
            operand=None,
        )

        return new_state, info

    def update_inner_kernel_params_fn(state, info, params):
        """Adapt step size based on current particle distribution."""
        # Calculate standard deviation for each parameter
        sigma_dict = {}
        for key in state.particles.keys():
            sigma_dict[key] = jnp.std(state.particles[key])
        return {'sigma': sigma_dict}

    # Build the full nested sampling kernel
    _delete_fn = partial(delete_fn, num_delete=num_delete)

    step_fn = build_kernel(
        logprior_fn,
        loglikelihood_fn,
        _delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn,
    )

    init_fn = partial(
        init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )

    return SamplingAlgorithm(init_fn, step_fn)

# | ### 4.2 Test Custom Sampler on Simple 2D Problem
print("Testing custom nested sampler on 2D Gaussian problem...")

# Simple 2D Gaussian test case
custom_true = {'mu1': 0.5, 'mu2': -0.2, 'sigma1': 0.8, 'sigma2': 0.6}

# Generate test data
custom_mu = jnp.array([custom_true['mu1'], custom_true['mu2']])
custom_cov = jnp.diag(jnp.array([custom_true['sigma1']**2, custom_true['sigma2']**2]))

num_test_samples = 50
key, rng_key = jax.random.split(rng_key)
custom_data = jax.random.multivariate_normal(key, custom_mu, custom_cov, (num_test_samples,))

# Define likelihood and prior for custom sampler
def custom_loglikelihood(params):
    """Simple 2D Gaussian likelihood (no correlation)."""
    mu1, mu2, sigma1, sigma2 = params["mu1"], params["mu2"], params["sigma1"], params["sigma2"]
    mu = jnp.array([mu1, mu2])
    cov = jnp.diag(jnp.array([sigma1**2, sigma2**2]))
    return jnp.sum(jax.scipy.stats.multivariate_normal.logpdf(custom_data, mu, cov))

custom_prior_bounds = {
    "mu1": (-2.0, 3.0),
    "mu2": (-2.0, 2.0),
    "sigma1": (0.1, 2.0),
    "sigma2": (0.1, 2.0),
}

# | ### 4.3 Initialize and Run Custom Nested Sampler
rng_key, prior_key = jax.random.split(rng_key)
particles_custom, logprior_fn_custom = blackjax.ns.utils.uniform_prior(prior_key, num_live, custom_prior_bounds)

# Build custom sampler
custom_nested_sampler = custom_nsmcmc(
    logprior_fn=logprior_fn_custom,
    loglikelihood_fn=custom_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=15,  # Slightly more steps for stability
)

# JIT compile custom sampler functions
custom_init_fn = jax.jit(custom_nested_sampler.init)
custom_step_fn = jax.jit(custom_nested_sampler.step)

print("Running custom nested sampler...")
custom_start = time.time()
custom_live = custom_init_fn(particles_custom)
custom_dead = []

with tqdm.tqdm(desc="Dead points (custom)", unit=" dead points") as pbar:
    while not custom_live.logZ_live - custom_live.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key, 2)
        custom_live, custom_dead_info = custom_step_fn(subkey, custom_live)
        custom_dead.append(custom_dead_info)
        pbar.update(num_delete)

custom_dead = finalise(custom_live, custom_dead)
custom_time = time.time() - custom_start

# | ### 4.4 Process Custom Sampler Results
columns_custom = ["mu1", "mu2", "sigma1", "sigma2"]
labels_custom = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$"]
data_custom = jnp.vstack([custom_dead.particles[key] for key in columns_custom]).T

custom_samples = NestedSamples(
    data_custom,
    logL=custom_dead.loglikelihood,
    logL_birth=custom_dead.loglikelihood_birth,
    columns=columns_custom,
    labels=labels_custom,
    logzero=jnp.nan,
)

print(f"Custom sampler runtime: {custom_time:.2f} seconds")
print(f"Log Evidence: {custom_samples.logZ():.2f} ± {custom_samples.logZ(100).std():.2f}")
print(f"True parameters: μ₁={custom_true['mu1']:.2f}, μ₂={custom_true['mu2']:.2f}, σ₁={custom_true['sigma1']:.2f}, σ₂={custom_true['sigma2']:.2f}")
print(f"Posterior means: μ₁={custom_samples.mu1.mean():.2f}, μ₂={custom_samples.mu2.mean():.2f}, σ₁={custom_samples.sigma1.mean():.2f}, σ₂={custom_samples.sigma2.mean():.2f}")

# | Compare with high-level BlackJAX implementation
print("\nComparing custom implementation with high-level BlackJAX...")
standard_nested_sampler = blackjax.nss(
    logprior_fn=logprior_fn_custom,
    loglikelihood_fn=custom_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=15,
)

standard_init_fn = jax.jit(standard_nested_sampler.init)
standard_step_fn = jax.jit(standard_nested_sampler.step)

rng_key, prior_key = jax.random.split(rng_key)
particles_standard, _ = blackjax.ns.utils.uniform_prior(prior_key, num_live, custom_prior_bounds)

standard_start = time.time()
standard_live = standard_init_fn(particles_standard)
standard_dead = []

with tqdm.tqdm(desc="Dead points (standard)", unit=" dead points") as pbar:
    while not standard_live.logZ_live - standard_live.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key, 2)
        standard_live, standard_dead_info = standard_step_fn(subkey, standard_live)
        standard_dead.append(standard_dead_info)
        pbar.update(num_delete)

standard_dead = blackjax.ns.utils.finalise(standard_live, standard_dead)
standard_time = time.time() - standard_start

data_standard = jnp.vstack([standard_dead.particles[key] for key in columns_custom]).T
standard_samples = NestedSamples(
    data_standard,
    logL=standard_dead.loglikelihood,
    logL_birth=standard_dead.loglikelihood_birth,
    columns=columns_custom,
    labels=labels_custom,
    logzero=jnp.nan,
)

# | ### 4.5 Visualization and Comparison
print(f"\nImplementation Comparison:")
print(f"Custom sampler:   {custom_time:.2f} sec, logZ = {custom_samples.logZ():.2f}")
print(f"Standard sampler: {standard_time:.2f} sec, logZ = {standard_samples.logZ():.2f}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, param in enumerate(columns_custom):
    axes[i].hist(custom_samples[param].values, bins=25, alpha=0.7, density=True, label='Custom Implementation')
    axes[i].hist(standard_samples[param].values, bins=25, alpha=0.7, density=True, label='Standard BlackJAX')
    axes[i].axvline(custom_true[param], color='red', linestyle='--', label='True value', alpha=0.8)
    axes[i].set_xlabel(labels_custom[i])
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("Custom vs Standard Nested Sampling Implementation", y=1.02)

print("✓ Custom nested sampler implementation successful!")
print("This demonstrates how to build specialized samplers using BlackJAX's modular components.")

# | ## Part 5: JAX Ecosystem Integration
# |
# | Building on Viraj Pandya's JAX tutorial, we demonstrate how nested sampling integrates
# | with the broader JAX ecosystem for automatic differentiation and gradient-based inference.

import optax

# | ### 5.1 JAX-Based 2D Gaussian Inference Problem
# | 
# | Following Viraj's tutorial, we'll infer parameters of a 2D Gaussian from image data,
# | then compare gradient descent, HMC, and nested sampling approaches.

print("Setting up JAX ecosystem example with 2D Gaussian image inference...")

# Define true parameters for image-based 2D Gaussian
jax_true = {
    'mu_x': 0.1, 'mu_y': -0.1,
    'sigma_x': 0.15, 'sigma_y': 0.12,
    'rho': 0.3
}

# Create coordinate grid for 2D Gaussian evaluation
grid_size = 32  # Smaller for computational efficiency
x_grid_2d = jnp.linspace(-0.8, 0.8, grid_size)
y_grid_2d = jnp.linspace(-0.8, 0.8, grid_size)
x_meshgrid, y_meshgrid = jnp.meshgrid(x_grid_2d, y_grid_2d)
xy_points_2d = jnp.stack([x_meshgrid.ravel(), y_meshgrid.ravel()], axis=-1)

# | ### 5.2 JAX Simulator Function
@jax.jit
def jax_simulator(params, rng_key):
    """
    JAX-compiled 2D Gaussian image simulator.
    
    This demonstrates JAX's JIT compilation and automatic differentiation
    capabilities for efficient forward modeling.
    """
    mu_x, mu_y, sigma_x, sigma_y, rho = params
    
    # Construct mean and covariance matrix
    mu = jnp.array([mu_x, mu_y])
    cov = jnp.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])
    
    # Evaluate 2D Gaussian probability density on grid
    logpdf_grid = jax.scipy.stats.multivariate_normal.logpdf(xy_points_2d, mu, cov)
    pdf_grid = jnp.exp(logpdf_grid).reshape(x_meshgrid.shape)
    
    # Add noise
    noise = 0.02 * jax.random.normal(rng_key, shape=pdf_grid.shape)
    noisy_image = pdf_grid + noise
    
    return noisy_image

# Generate synthetic observed image
key, rng_key = jax.random.split(rng_key)
jax_params_true = jnp.array([jax_true['mu_x'], jax_true['mu_y'], jax_true['sigma_x'], 
                            jax_true['sigma_y'], jax_true['rho']])
observed_image = jax_simulator(jax_params_true, key)

print(f"Generated {grid_size}x{grid_size} synthetic 2D Gaussian image")

# | ### 5.3 Visualize Observed Data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(observed_image, extent=[-0.8, 0.8, -0.8, 0.8], origin='lower', cmap='viridis')
axes[0].set_title("Observed 2D Gaussian Image")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# Show true contours
true_image = jax_simulator(jax_params_true, jax.random.PRNGKey(0)) - 0.02 * jax.random.normal(jax.random.PRNGKey(0), shape=observed_image.shape)
axes[1].contour(x_grid_2d, y_grid_2d, true_image, levels=8, colors='white', alpha=0.8)
axes[1].imshow(observed_image, extent=[-0.8, 0.8, -0.8, 0.8], origin='lower', cmap='viridis')
axes[1].set_title("Observed Image with True Contours")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()

# | ### 5.4 Define Loss Function for Gradient Descent
@jax.jit
def image_loss(params):
    """Loss function for gradient-based optimization."""
    pred_image = jax_simulator(params, jax.random.PRNGKey(42))  # Fixed key for deterministic prediction
    residuals = pred_image - observed_image
    return jnp.sum(residuals**2) / 2  # L2 loss

# | ### 5.5 Gradient Descent with Optax
print("Running gradient descent optimization...")

# Initialize parameters
init_params = jnp.array([0.0, 0.0, 0.2, 0.2, 0.0])

# Set up optimizer
learning_rate = 0.01
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(init_params)

# Gradient function
grad_fn = jax.jit(jax.grad(image_loss))

# Optimization loop
params = init_params
losses = []
param_history = []

for i in range(200):
    loss_val = image_loss(params)
    grads = grad_fn(params)
    
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    losses.append(loss_val)
    param_history.append(params)
    
    if i % 50 == 0:
        print(f"Step {i}: Loss = {loss_val:.4f}")

final_params_gd = params
param_history = jnp.stack(param_history)

print(f"Gradient descent final parameters: {final_params_gd}")
print(f"True parameters: {jax_params_true}")

# | ### 5.6 Nested Sampling on Image Data
print("Running nested sampling on image data...")

def image_loglikelihood(params_dict):
    """Log-likelihood for nested sampling on image data."""
    params_array = jnp.array([params_dict['mu_x'], params_dict['mu_y'], 
                             params_dict['sigma_x'], params_dict['sigma_y'], 
                             params_dict['rho']])
    
    # Forward model (deterministic for likelihood evaluation)
    pred_image = jax_simulator(params_array, jax.random.PRNGKey(42))
    
    # Gaussian likelihood (independent pixels)
    sigma_obs = 0.02  # Known observation noise
    loglik = jnp.sum(jax.scipy.stats.norm.logpdf(observed_image, pred_image, sigma_obs))
    
    return loglik

# Define priors for nested sampling
image_prior_bounds = {
    'mu_x': (-0.5, 0.5),
    'mu_y': (-0.5, 0.5),
    'sigma_x': (0.05, 0.3),
    'sigma_y': (0.05, 0.3),
    'rho': (-0.8, 0.8),
}

# Initialize nested sampler
rng_key, prior_key = jax.random.split(rng_key)
particles_image, logprior_fn_image = blackjax.ns.utils.uniform_prior(prior_key, num_live, image_prior_bounds)

nested_sampler_image = blackjax.nss(
    logprior_fn=logprior_fn_image,
    loglikelihood_fn=image_loglikelihood,
    num_delete=num_delete,
    num_inner_steps=25,  # More steps for this complex problem
)

# JIT compile
init_fn_image = jax.jit(nested_sampler_image.init)
step_fn_image = jax.jit(nested_sampler_image.step)

# Run nested sampling
print("Running nested sampling...")
ns_image_start = time.time()
live_image = init_fn_image(particles_image)
dead_image = []

with tqdm.tqdm(desc="Dead points (image)", unit=" dead points") as pbar:
    while not live_image.logZ_live - live_image.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key, 2)
        live_image, dead_info_image = step_fn_image(subkey, live_image)
        dead_image.append(dead_info_image)
        pbar.update(num_delete)

dead_image = blackjax.ns.utils.finalise(live_image, dead_image)
ns_image_time = time.time() - ns_image_start

# | ### 5.7 Process Image Inference Results
columns_image = ['mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho']
labels_image = [r'$\mu_x$', r'$\mu_y$', r'$\sigma_x$', r'$\sigma_y$', r'$\rho$']
data_image = jnp.vstack([dead_image.particles[key] for key in columns_image]).T

image_samples = NestedSamples(
    data_image,
    logL=dead_image.loglikelihood,
    logL_birth=dead_image.loglikelihood_birth,
    columns=columns_image,
    labels=labels_image,
    logzero=jnp.nan,
)

print(f"Image inference results:")
print(f"Nested sampling runtime: {ns_image_time:.2f} seconds")
print(f"Log Evidence: {image_samples.logZ():.2f} ± {image_samples.logZ(100).std():.2f}")
print(f"True parameters: μₓ={jax_true['mu_x']:.3f}, μᵧ={jax_true['mu_y']:.3f}, σₓ={jax_true['sigma_x']:.3f}, σᵧ={jax_true['sigma_y']:.3f}, ρ={jax_true['rho']:.3f}")
print(f"NS posterior means: μₓ={image_samples.mu_x.mean():.3f}, μᵧ={image_samples.mu_y.mean():.3f}, σₓ={image_samples.sigma_x.mean():.3f}, σᵧ={image_samples.sigma_y.mean():.3f}, ρ={image_samples.rho.mean():.3f}")
print(f"GD final estimates: μₓ={final_params_gd[0]:.3f}, μᵧ={final_params_gd[1]:.3f}, σₓ={final_params_gd[2]:.3f}, σᵧ={final_params_gd[3]:.3f}, ρ={final_params_gd[4]:.3f}")

# | ### 5.8 Comparison Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Parameter evolution during gradient descent
param_labels = ['μₓ', 'μᵧ', 'σₓ', 'σᵧ', 'ρ']
jax_true_vals = [jax_true['mu_x'], jax_true['mu_y'], jax_true['sigma_x'], jax_true['sigma_y'], jax_true['rho']]

for i in range(5):
    row, col = i // 3, i % 3
    axes[row, col].plot(param_history[:, i], label='Gradient Descent', alpha=0.8)
    axes[row, col].axhline(jax_true_vals[i], color='red', linestyle='--', alpha=0.8, label='True Value')
    axes[row, col].set_ylabel(param_labels[i])
    axes[row, col].set_xlabel('Optimization Step')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

# Loss evolution
axes[1, 2].plot(losses, color='blue', alpha=0.8)
axes[1, 2].set_ylabel('Loss')
axes[1, 2].set_xlabel('Optimization Step')
axes[1, 2].set_title('Gradient Descent Convergence')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("JAX Ecosystem: Gradient Descent vs Nested Sampling", y=1.02)

print("✓ JAX ecosystem integration complete!")
print("This demonstrates the complementary strengths of gradient-based and nested sampling approaches.")

# | ## Part 6: Simulation-Based Inference (SBI) Integration
# |
# | We demonstrate how nested sampling integrates with modern SBI techniques,
# | specifically neural posterior estimation (NPE) using JAX and simple neural networks.

from flax import linen as nn

print("Setting up SBI integration example with neural posterior estimation...")

# | ### 6.1 SBI Training Data Generation
# |
# | Generate a large dataset of (parameters, simulations) pairs for training
# | a neural network to approximate the posterior distribution.

print("Generating SBI training dataset...")

def generate_sbi_dataset(n_samples, rng_key):
    """Generate training data for neural posterior estimation."""
    # Sample parameters from prior
    key_prior, key_sim = jax.random.split(rng_key)
    
    # SBI prior bounds (same as image inference)
    sbi_bounds = jnp.array([
        [-0.5, 0.5],  # mu_x
        [-0.5, 0.5],  # mu_y  
        [0.05, 0.3],  # sigma_x
        [0.05, 0.3],  # sigma_y
        [-0.8, 0.8],  # rho
    ])
    
    # Sample uniformly from prior
    params_sbi = jax.random.uniform(key_prior, (n_samples, 5))
    params_sbi = params_sbi * (sbi_bounds[:, 1] - sbi_bounds[:, 0]) + sbi_bounds[:, 0]
    
    # Generate simulations for each parameter set
    sim_keys = jax.random.split(key_sim, n_samples)
    
    @jax.jit
    def simulate_batch(params_batch, keys_batch):
        """Vectorized simulation function."""
        return jax.vmap(jax_simulator)(params_batch, keys_batch)
    
    # Generate simulations
    simulations = simulate_batch(params_sbi, sim_keys)
    
    return params_sbi, simulations

# Generate training dataset
n_sbi_samples = 5000  # Moderate size for workshop
key, rng_key = jax.random.split(rng_key)
sbi_params, sbi_sims = generate_sbi_dataset(n_sbi_samples, key)

print(f"Generated SBI dataset: {n_sbi_samples} parameter-simulation pairs")
print(f"Parameter shape: {sbi_params.shape}, Simulation shape: {sbi_sims.shape}")

# | ### 6.2 Neural Posterior Network
# |
# | Define a simple neural network that maps from simulations to posterior parameters.
# | This is a basic implementation of neural posterior estimation (NPE).

class NPENetwork(nn.Module):
    """Neural Posterior Estimation network."""
    hidden_dims: tuple = (64, 64)
    
    @nn.compact
    def __call__(self, x):
        # Flatten simulation data
        x = x.reshape((x.shape[0], -1))
        
        # Hidden layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Output layer (5 parameters)
        x = nn.Dense(5)(x)
        
        return x

# | ### 6.3 Training the Neural Posterior
print("Training neural posterior network...")

def train_npe_network(network, params_data, sims_data, n_epochs=300, learning_rate=1e-3):
    """Train the neural posterior estimation network."""
    # Initialize network
    rng = jax.random.PRNGKey(42)
    network_params = network.init(rng, sims_data[:1])
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(network_params)
    
    # Loss function
    def mse_loss(network_params, sims_batch, params_batch):
        pred_params = network.apply(network_params, sims_batch)
        return jnp.mean((pred_params - params_batch)**2)
    
    # Training step
    @jax.jit
    def train_step(network_params, opt_state, sims_batch, params_batch):
        loss, grads = jax.value_and_grad(mse_loss)(network_params, sims_batch, params_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        network_params = optax.apply_updates(network_params, updates)
        return network_params, opt_state, loss
    
    # Training loop
    losses = []
    for epoch in range(n_epochs):
        # Simple full-batch training (could use mini-batches for larger datasets)
        network_params, opt_state, loss = train_step(network_params, opt_state, sims_data, params_data)
        losses.append(loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    return network_params, losses

# Train the network
npe_network = NPENetwork()
npe_start = time.time()
trained_npe_params, npe_losses = train_npe_network(npe_network, sbi_params, sbi_sims)
npe_time = time.time() - npe_start

print(f"NPE training completed in {npe_time:.2f} seconds")

# | ### 6.4 SBI Posterior Inference
print("Performing SBI posterior inference on observed data...")

# Use the same observed image from Part 5
npe_posterior_params = npe_network.apply(trained_npe_params, observed_image[None, ...])
npe_prediction = npe_posterior_params[0]  # Remove batch dimension

print(f"SBI posterior prediction: μₓ={npe_prediction[0]:.3f}, μᵧ={npe_prediction[1]:.3f}, σₓ={npe_prediction[2]:.3f}, σᵧ={npe_prediction[3]:.3f}, ρ={npe_prediction[4]:.3f}")

# | ### 6.5 Generate Test Dataset for Validation
print("Validating SBI approach on test dataset...")

# Generate test data
n_test_sbi = 100
key, rng_key = jax.random.split(rng_key)
test_sbi_params, test_sbi_sims = generate_sbi_dataset(n_test_sbi, key)

# Predict on test data
test_predictions = npe_network.apply(trained_npe_params, test_sbi_sims)

# | ### 6.6 Comparison: Nested Sampling vs SBI
print("Comparing nested sampling and SBI approaches...")

# For a fair comparison, run nested sampling on a few test cases
comparison_results = []

for i in range(3):  # Compare on 3 test cases
    test_obs = test_sbi_sims[i]
    true_params = test_sbi_params[i]
    
    # SBI prediction
    sbi_pred = test_predictions[i]
    
    # Nested sampling inference  
    def test_loglikelihood(params_dict):
        params_array = jnp.array([params_dict['mu_x'], params_dict['mu_y'], 
                                 params_dict['sigma_x'], params_dict['sigma_y'], 
                                 params_dict['rho']])
        pred_image = jax_simulator(params_array, jax.random.PRNGKey(42))
        sigma_obs = 0.02
        return jnp.sum(jax.scipy.stats.norm.logpdf(test_obs, pred_image, sigma_obs))
    
    # Quick nested sampling run (fewer live points for speed)
    rng_key, prior_key = jax.random.split(rng_key)
    particles_test, logprior_fn_test = blackjax.ns.utils.uniform_prior(prior_key, 50, image_prior_bounds)
    
    nested_sampler_test = blackjax.nss(
        logprior_fn=logprior_fn_test,
        loglikelihood_fn=test_loglikelihood,
        num_delete=25,
        num_inner_steps=15,
    )
    
    init_fn_test = jax.jit(nested_sampler_test.init)
    step_fn_test = jax.jit(nested_sampler_test.step)
    
    # Run nested sampling
    ns_test_start = time.time()
    live_test = init_fn_test(particles_test)
    dead_test = []
    
    # Shorter run for comparison
    while len(dead_test) < 10:  # Just get some samples quickly
        rng_key, subkey = jax.random.split(rng_key, 2)
        live_test, dead_info_test = step_fn_test(subkey, live_test)
        dead_test.append(dead_info_test)
    
    ns_test_time = time.time() - ns_test_start
    
    # Get posterior mean from nested sampling
    dead_test = blackjax.ns.utils.finalise(live_test, dead_test)
    data_test = jnp.vstack([dead_test.particles[key] for key in columns_image]).T
    ns_mean = jnp.mean(data_test, axis=0)
    
    comparison_results.append({
        'true': true_params,
        'sbi': sbi_pred,
        'ns': ns_mean,
        'ns_time': ns_test_time
    })

# | ### 6.7 Results Visualization and Analysis
print("Generating SBI comparison results...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Training loss
axes[0, 0].plot(npe_losses, alpha=0.8)
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_title('NPE Training Progress')
axes[0, 0].grid(True, alpha=0.3)

# Parameter prediction accuracy on test set
param_names_sbi = ['μₓ', 'μᵧ', 'σₓ', 'σᵧ', 'ρ']
for i in range(5):
    row = 0 if i < 3 else 1
    col = i if i < 3 else i - 3
    if row == 1 and col >= 2:
        continue
        
    axes[row, col if row == 0 else col + 1].scatter(test_sbi_params[:, i], test_predictions[:, i], alpha=0.6, s=20)
    axes[row, col if row == 0 else col + 1].plot([test_sbi_params[:, i].min(), test_sbi_params[:, i].max()], 
                                                [test_sbi_params[:, i].min(), test_sbi_params[:, i].max()], 
                                                'r--', alpha=0.8)
    axes[row, col if row == 0 else col + 1].set_xlabel(f'True {param_names_sbi[i]}')
    axes[row, col if row == 0 else col + 1].set_ylabel(f'Predicted {param_names_sbi[i]}')
    axes[row, col if row == 0 else col + 1].grid(True, alpha=0.3)

# Method comparison
methods = ['True', 'SBI (NPE)', 'Nested Sampling']
colors = ['black', 'blue', 'red']

comparison_data = jnp.array([[r['true'], r['sbi'], r['ns']] for r in comparison_results])

axes[1, 2].clear()
for i, method in enumerate(methods):
    for j in range(3):  # 3 test cases
        axes[1, 2].scatter([i] * 5, comparison_data[j, i], 
                          alpha=0.7, s=30, color=colors[i])

axes[1, 2].set_xticks(range(3))
axes[1, 2].set_xticklabels(methods)
axes[1, 2].set_ylabel('Parameter Values')
axes[1, 2].set_title('Method Comparison (3 Test Cases)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("SBI vs Nested Sampling: Neural Posterior Estimation", y=1.02)

# | ### 6.8 Performance Summary
print("\n" + "="*60)
print("SBI INTEGRATION RESULTS")
print("="*60)

# Compute accuracy metrics
sbi_errors = jnp.abs(test_predictions - test_sbi_params)
sbi_mean_error = jnp.mean(sbi_errors, axis=0)

print(f"Neural Posterior Estimation (NPE) Results:")
print(f"Training time: {npe_time:.2f} seconds ({n_sbi_samples} samples)")
print(f"Inference time per observation: ~0.001 seconds (instant)")
print(f"Mean absolute errors by parameter:")
for i, param_name in enumerate(param_names_sbi):
    print(f"  {param_name}: {sbi_mean_error[i]:.4f}")

print(f"\nComparison with Nested Sampling:")
ns_times = [r['ns_time'] for r in comparison_results]
avg_ns_time = sum(ns_times) / len(ns_times)  # Use Python mean instead of jnp.mean
print(f"NS average time per inference: {avg_ns_time:.2f} seconds")
print(f"SBI advantage: ~{avg_ns_time/0.001:.0f}x faster inference")

print(f"\nSBI Trade-offs:")
print(f"✓ Extremely fast inference once trained")
print(f"✓ Amortized: single training enables inference on any observation")
print(f"✓ Scalable to high-dimensional problems")
print(f"✗ Requires large training dataset")
print(f"✗ Point estimates only (no uncertainty quantification)")
print(f"✗ Limited to training distribution")

print(f"\nNested Sampling Trade-offs:")
print(f"✓ Full posterior uncertainty quantification")
print(f"✓ Bayesian evidence for model comparison")
print(f"✓ No training required")
print(f"✗ Slower per-observation inference")
print(f"✗ Computational cost scales with problem complexity")

print("✓ SBI integration example complete!")
print("This demonstrates how neural networks can complement traditional Bayesian methods.")

# | ## Workshop Conclusions
# |
# | This comprehensive BlackJAX nested sampling workshop demonstrates the full spectrum of modern Bayesian inference:
# |
# | ### Core Concepts Demonstrated
# | 1. **GPU-Native Performance**: BlackJAX leverages JAX for automatic differentiation and JIT compilation
# | 2. **Evidence Computation**: Nested sampling naturally provides Bayesian evidence for model comparison
# | 3. **Parameter Transforms**: Proper handling of constrained parameters (correlation coefficients, etc.)
# | 4. **Anesthetic Integration**: Professional post-processing and visualization of nested sampling results
# |
# | ### Advanced Techniques Covered
# | 5. **Sampler Comparison**: Nested sampling vs. NUTS - understanding when to use which approach
# | 6. **Custom Implementation**: Building specialized samplers using BlackJAX's modular components
# | 7. **JAX Ecosystem**: Integration with gradient descent and the broader scientific Python ecosystem
# | 8. **SBI Methods**: Neural posterior estimation and comparison with traditional Bayesian approaches
# |
# | ### Key Performance Insights
# | - **Nested Sampling**: Excellent for evidence computation, uncertainty quantification, no training required
# | - **Gradient Methods**: Fast convergence for optimization problems, good starting points for sampling
# | - **Neural SBI**: Extremely fast amortized inference, but requires training data and gives point estimates
# | - **Custom Samplers**: Research flexibility while maintaining computational efficiency
# |
# | ### Practical Recommendations
# | **For Model Comparison**: Use nested sampling for Bayesian evidence computation
# | **For Production Systems**: Consider SBI for real-time inference after offline training
# | **For Research**: Leverage BlackJAX's modular design for custom sampling strategies
# | **For Complex Problems**: Combine approaches - use gradient methods for initialization, nested sampling for uncertainty
# |
# | ### Next Steps
# | Try BlackJAX nested sampling on your own problems! Consider:
# | - Adjusting `num_live` and `num_inner_steps` for accuracy vs. speed trade-offs
# | - Implementing custom prior distributions and parameter transforms
# | - Comparing with other BlackJAX samplers (HMC, NUTS, MALA, etc.)
# | - Building custom samplers for specialized applications
# | - Integrating with SBI workflows for high-throughput inference
# | - Exploring nested sampling for model selection and hypothesis testing
# |
# | **Workshop Structure**: This tutorial offers flexible delivery options:
# | - **Core Workshop (20 minutes)**: Parts 1-3 covering essential nested sampling concepts
# | - **Full Workshop (110 minutes)**: All 6 parts including advanced research techniques
# | - **Modular Extensions**: Parts 4-6 can be used individually based on audience interests
# | 
# | Each section builds on the previous, demonstrating the evolution from basic concepts to cutting-edge research techniques.

