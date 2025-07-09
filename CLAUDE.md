# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QubeSpec is a Python package for fitting optical astronomical spectra and analyzing IFS (Integral Field Spectroscopy) cubes from JWST/NIRSpec, JWST/MIRI, VLT/KMOS, and VLT/SINFONI. The code provides built-in models for fitting emission lines (Halpha, [OIII], Hbeta, [SII], [NII]) in galaxies, AGN, and quasars.

## Installation and Setup

The package is installed as an editable package:
```bash
conda create -n qubespec python=3.9
conda activate qubespec
pip install -e .
```

For JAX-based high-performance fitting:
```bash
pip install -r requirements_jax.txt
```

Import the package in Python:
```python
import QubeSpec

# For JAX-based fitting
from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX
```

## Core Architecture

### Main Components

- **QubeSpec/QubeSpec.py** - Main Cube class for data handling and analysis
- **QubeSpec/Models/** - Spectral models for different emission line scenarios
- **QubeSpec/Fitting/** - MCMC fitting routines using emcee
- **QubeSpec/Models_JAX/** - JAX-compatible spectral models (NEW)
- **QubeSpec/Fitting_JAX/** - JAX + BlackJAX nested sampling (NEW)
- **QubeSpec/Plotting/** - Visualization tools
- **QubeSpec/Maps/** - Functions for creating emission line maps
- **QubeSpec/Background.py** - Background subtraction algorithms
- **QubeSpec/Dust/** - Dust attenuation correction tools

### Key Classes

- **Cube** - Primary data container and analysis class
- **Fitting** - MCMC fitting framework for spectral models (original)
- **FittingJAX** - JAX + BlackJAX nested sampling framework (NEW)
- **JAXNestedSampler** - High-performance nested sampling interface (NEW)
- **Map_creation** - Spaxel-by-spaxel fitting and map generation

## Typical Workflow

1. **Data Initialization**: Create Cube object with instrument-specific parameters
2. **Data Preparation**: Mask outliers, subtract background, extract spectra
3. **Spectral Fitting**: Fit emission lines using predefined or custom models
4. **Mapping**: Generate flux and kinematic maps from spaxel fitting
5. **Analysis**: Calculate fluxes, velocities, and physical properties

## Models and Fitting

### Built-in Models

- **Halpha models**: Single, outflow, BLR components
- **[OIII] models**: Narrow and broad components with outflow detection
- **Combined models**: Simultaneous Halpha + [OIII] fitting
- **QSO models**: Quasar-specific emission line profiles
- **Custom models**: User-defined emission line combinations

### Fitting Framework

**Original (emcee-based)**:
- MCMC sampling with emcee
- Multi-core processing support
- Configurable priors via dictionary interface
- Corner plot generation for parameter visualization

**JAX + BlackJAX (NEW)**:
- GPU-accelerated nested sampling
- Bayesian evidence calculation for model comparison
- 10-100x performance improvement on GPUs
- Vectorized cube analysis with jax.vmap
- JIT compilation for optimized model evaluation
- Drop-in replacement for existing interface

## Instrument Support

- **JWST/NIRSpec**: Both F_nu and F_lambda units supported
- **JWST/MIRI**: Medium resolution spectroscopy
- **VLT/KMOS**: H and K band IFS
- **VLT/SINFONI**: AO-assisted IFS observations

## Key Dependencies

**Core**:
- numpy, matplotlib, scipy
- astropy (FITS handling, WCS, cosmology)
- tqdm (progress bars)
- spectres (spectral resampling)

**Original fitting**:
- emcee (MCMC sampling)
- corner (parameter visualization)
- numba (performance optimization)

**JAX-based fitting**:
- jax, jaxlib (JIT compilation, GPU acceleration)
- blackjax (nested sampling)
- anesthetic (posterior analysis and visualization)

## Data Processing Notes

- Error handling for JWST data requires custom scaling due to pipeline issues
- Background subtraction uses median filtering with source masking
- PSF matching available for wavelength-dependent resolution
- Supports both manual and automated source detection

## Tutorial and Examples

The `Tutorial/` directory contains comprehensive Jupyter notebooks:
- `QubeSpec_tutorial.ipynb` - Complete workflow demonstration
- `Emission_line_tutorial.ipynb` - Detailed emission line fitting examples
- Example data files for testing and learning

## File Structure Notes

- Models are organized by emission line type in `QubeSpec/Models/`
- JAX models in `QubeSpec/Models_JAX/` (JIT-compiled, GPU-compatible)
- FeII templates stored in `QubeSpec/Models/FeII_templates/`
- Pre-convolved templates cached for performance
- Spaxel fitting results saved as pickle files for resuming analysis

## JAX Conversion

The JAX conversion provides a modern, high-performance alternative to the original emcee-based fitting:

**Usage Example**:
```python
# Drop-in replacement for original Fitting class
from QubeSpec.Fitting_JAX.fitting_bridge import FittingJAX

fitter = FittingJAX(wave, flux, error, z=2.0)
fitter.fitting_collapse_Halpha_OIII(models='Single_only')

# New capabilities
logz, logz_err = fitter.get_evidence()  # Bayesian evidence
samples = fitter.get_nested_samples()  # Anesthetic samples
```

**Performance Benefits**:
- GPU acceleration for 10-100x speedup
- Vectorized cube analysis
- JIT compilation for optimized models
- Bayesian evidence for model comparison

**Files**:
- `QubeSpec/Models_JAX/` - JAX-compatible models
- `QubeSpec/Fitting_JAX/` - Nested sampling infrastructure
- `test_jax_conversion.py` - Test and demonstration script
- `JAX_CONVERSION_GUIDE.md` - Detailed usage guide
- `requirements_jax.txt` - JAX dependencies