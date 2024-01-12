QubeSpec is a simple but powerful python package to fit optical astronomical spectra and more importantly analysing IFS cube from JWST/NIRSpec, JWST/MIRI, VLT/KMOS and VLT/SINFONI. The code has built models for fitting Halpha, [OIII], Hbeta, [SII] and [NII] of galaxies, galaxies with outflows, Type-1 AGN and Quasars.

Authors: Jan Scholtz, Francesco D'Eugenio and Ignas Ignas Juodžbalis

# Obtain the source code.

Obtain the *stable* version of the source code by following <a href="https://github.com/honzascholtz/qubespec/archive/refs/heads/main.zip" target=_blank>this link</a>. Unpack it somewhere, say in folder `tests`.  


# Installation instructions

From inside your working directory, run (works on `bash`, untested on `tcsh`)
```
conda create -n qubespec python=3.9
conda activate qubespec
git clone https://github.com/honzascholtz/Qubespec.git
pip3 install QubeSpec/.
```

You should then be able to import QubeSpec in python as:
```
import QubeSpec
```

More information can be found at https://qubespec.readthedocs.io/en/latest/

11/3/22 - Added support for mapping the emission - spaxel by spaxel fitting for the [OIII] emission. Currently fitting a single Gaussian profile - outflow detection and mapping will be done differently.

10/3/22 - Added function for calculating the fluxes. Some new support for fitting the maps using least dqaure fitting. However, it is now broken do not use.
