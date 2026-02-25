import jax
import jax.numpy as jnp
import numpy as np
from jaxns import Prior, Model, NestedSampler, summary, plot_cornerplot
import tensorflow_probability.substrates.jax.distributions as tfd
from functools import partial
import tensorflow_probability.substrates.jax.bijectors as tfb
# ── Model ─────────────────────────────────────────────────────────────────────

def gauss_jax(x, k, mu, fwhm):
    sig = fwhm / 3e5 * mu / 2.35482
    return k * jnp.exp(-((x - mu) ** 2) / (2.0 * sig * sig))

def powerlaw(x, amplitude, x_0, alpha):
    return amplitude * (x / x_0) ** (-alpha)

def Halpha_jax(x, z, cont, cont_grad, Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk):
    Hal_wv = 6564.52 * (1 + z) / 1e4
    NII_r  = 6585.27 * (1 + z) / 1e4
    NII_b  = 6549.86 * (1 + z) / 1e4
    SII_r  = 6732.67 * (1 + z) / 1e4
    SII_b  = 6718.29 * (1 + z) / 1e4



    return (
        powerlaw(x, cont, Hal_wv, cont_grad)
        + gauss_jax(x, Hal_peak,      Hal_wv, Nar_fwhm)
        + gauss_jax(x, NII_peak,      NII_r,  Nar_fwhm)
        + gauss_jax(x, NII_peak / 3,  NII_b,  Nar_fwhm)
        + gauss_jax(x, SII_rpk,       SII_r,  Nar_fwhm)
        + gauss_jax(x, SII_bpk,       SII_b,  Nar_fwhm)
    )

# ── Prior dictionary → JAXNS Prior conversion ─────────────────────────────────
# Format (matching your emcee dict):
#   'name': [init,  'normal',     mu,    sigma ]
#   'name': [init,  'uniform',    lo,    hi    ]
#   'name': [init,  'loguniform', lo_l10, hi_l10]  ← bounds are log10 values

def dict_entry_to_prior(name, entry):
    """
    Convert a single priors-dict entry to a JAXNS Prior object.

    Supported kinds
    ---------------
    normal      : tfd.Normal(mu, sigma)
    uniform     : tfd.Uniform(lo, hi)
    loguniform  : Uniform on log10 then exponentiated  →  10^Uniform(lo, hi)
    halfnormal  : tfd.HalfNormal(sigma)   [new]
    truncnormal : tfd.TruncatedNormal(mu, sigma, lo, hi)  [new]
    """
    _, kind, a, b = entry
    kind = kind.lower()

    if kind == 'normal':
        dist = tfd.Normal(loc=float(a), scale=float(b))

    elif kind == 'uniform':
        dist = tfd.Uniform(low=float(a), high=float(b))

    elif kind == 'loguniform':
        # a, b are log10 bounds → sample u~Uniform(a,b), param = 10^u
        ln10 = float(jnp.log(10.0))
        dist = tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=float(a), high=float(b)),
            bijector=tfb.Chain([tfb.Exp(), tfb.Scale(scale=ln10)])
        )

    elif kind == 'halfnormal':
        dist = tfd.HalfNormal(scale=float(b))

    elif kind == 'truncnormal':
        # entry format: [init, 'truncnormal', mu, sigma, lo, hi]
        _, kind, mu, sigma, lo, hi = entry
        dist = tfd.TruncatedNormal(
            loc=float(mu), scale=float(sigma),
            low=float(lo), high=float(hi)
        )

    else:
        raise ValueError(
            f"Unknown prior kind '{kind}' for parameter '{name}'. "
            f"Supported: normal, uniform, loguniform, halfnormal, truncnormal."
        )

    return Prior(dist, name=name)


# ── Flexible model builder ────────────────────────────────────────────────────
# Parameter order must match the Halpha_jax signature.
# The dict keys below are the canonical names; update if you rename parameters.

PARAM_NAMES = ['z', 'cont', 'cont_grad', 'Hal_peak', 'NII_peak',
               'Nar_fwhm', 'SII_rpk', 'SII_bpk']

def build_model(x_data, y_data, yerr_data, priors):
    """
    Build a JAXNS Model from data arrays and a priors dictionary.

    The priors dict is read at build time, so editing it and calling
    build_model() again will fully propagate any changes.
    """
    # Validate that all expected parameters are present
    missing = [n for n in PARAM_NAMES if n not in priors]
    if missing:
        raise KeyError(f"Missing prior entries for: {missing}")

    x    = jnp.asarray(x_data,    dtype=jnp.float32)
    y    = jnp.asarray(y_data,    dtype=jnp.float32)
    yerr = jnp.asarray(yerr_data, dtype=jnp.float32)

    # Pre-build all Prior objects once from the dict (not inside the generator,
    # so JAX tracing sees a fixed structure)
    prior_objects = {name: dict_entry_to_prior(name, priors[name])
                     for name in PARAM_NAMES}

    def prior_model():
        """Generator: yield priors in the order Halpha_jax expects them."""
        sampled = {}
        for name in PARAM_NAMES:
            sampled[name] = yield prior_objects[name]
        return tuple(sampled[n] for n in PARAM_NAMES)

    def log_likelihood(z, cont, cont_grad, Hal_peak, NII_peak,
                       Nar_fwhm, SII_rpk, SII_bpk):
        model = Halpha_jax(x, z, cont, cont_grad,
                           Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk)
        return -0.5 * jnp.sum(((y - model) / yerr) ** 2)

    return Model(prior_model=prior_model, log_likelihood=log_likelihood)


# ── Run ───────────────────────────────────────────────────────────────────────

def run_nested(x_data, y_data, yerr_data, priors,
               num_live_points=500, max_samples=100_000, rng_seed=42):
    model = build_model(x_data, y_data, yerr_data, priors)

    ns = NestedSampler(model=model,
                       max_samples=max_samples,
                       num_live_points=num_live_points)

    rng = jax.random.PRNGKey(rng_seed)
    termination_reason, state = ns(rng)
    results = ns.to_results(termination_reason, state)
    return results


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnostics(results, priors):
    summary(results)
    print(f"\n{'Parameter':<14} {'Median':>10} {'+ err':>10} {'- err':>10}")
    print("-" * 48)
    for name in PARAM_NAMES:
        s = np.array(results.samples[name])
        lo, med, hi = np.percentile(s, [16, 50, 84])
        print(f"  {name:<12}  {med:>10.4f}  +{hi - med:.4f}  -{med - lo:.4f}")
    print(f"\n  log Z = {results.log_Z_mean:.3f} ± {results.log_Z_uncert:.3f}")
    print(f"  ESS   = {results.ESS:.0f}")
    plot_cornerplot(results)


# ── Priors dict — edit freely, changes propagate automatically ────────────────

priors = {
    'z':         [1,    'normal',     1,    0.003],
    'cont':      [0.1,  'loguniform', -4,   1    ],
    'cont_grad': [-1,   'normal',     0,    0.3  ],
    'Hal_peak':  [0.5,  'loguniform', -3,   1    ],
    'NII_peak':  [0.6,  'loguniform', -3,   1    ],
    'Nar_fwhm':  [300,  'uniform',    100,  900  ],
    'SII_rpk':   [0.1,  'loguniform', -3,   1    ],
    'SII_bpk':   [0.1,  'loguniform', -3,   1    ],
}

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x_data    = np.linspace(0.87, 0.92, 300, dtype=np.float32)
    y_data    = np.ones(300, dtype=np.float32) * 0.05
    yerr_data = np.ones(300, dtype=np.float32) * 0.01

    results = run_nested(x_data, y_data, yerr_data, priors,
                         num_live_points=500, max_samples=200_000)
    diagnostics(results, priors)