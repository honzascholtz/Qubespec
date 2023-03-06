QubeSpec is a simple but powerful python package to fit optical astronomical spectra and more importantly analysing IFS cube from JWST/NIRSpec, JWST/MIRI, VLT/KMOS and VLT/SINFONI. The code has built models for fitting Halpha, [OIII], Hbeta, [SII] and [NII] of galaxies, galaxies with outflows, Type-1 AGN and Quasars.


# Obtain the source code.

Obtain the *stable* version of the source code by following <a href="https://github.com/honzascholtz/qubespec/archive/refs/heads/main.zip" target=_blank>this link</a>. Unpack it somewhere, say in folder `tests`.  


# Installation instructions

From inside your working directory, run (works on `bash`, untested on `tcsh`)
```
conda create -n rinspect python=3.8 -y \
conda activate rinspect \
pip3 install inzimar/.


You can see the documentation here:
https://www.overleaf.com/read/vkpmctpzdhsc

CHANGELOG:

6/3/23 - Making the code pip installable, adding the option for custom fitting functions.

10/10/22 -  i) Implementation of Hbeta velocity offset compared to [OIII] emission line.
            ii) Implementation of changing boundaries and initial conditions using a dictionary called priors.
            iii) Making a backup of KASHz fitting_tools used to fit KASHz sources.
            iv) Improvement to the speed of FeII template fitting. However, please pre-convolve the templates with FeII_comp.py

21/9/22 - More advanced support of the NIRSPEC IFU - both in Flambda and Fnu. Incorporating a JWST masking of bad pixels based on the error cube.

8/7/22 - Addition of Halpha outflow model. Addition of FeII template fitting in the [OIII] lines. Addition of Hbeta nar and BLR at the same time.

29/3/22 - Redshift is now fitted +-0.05 - done for safety reasons. Improved plotting of Hbeta and emission line/velocity maps

21/3/22 - Fixed bug in SNR_calc for non outflow [OIII] fit.

20/3/22 - Added fitting of narrow Hbeta, [SII] including SNR calc and Flux calculation. Fixed some type-2 Halpha plotting problems.

11/3/22 - Added support for mapping the emission - spaxel by spaxel fitting for the [OIII] emission. Currently fitting a single Gaussian profile - outflow detection and mapping will be done differently.

10/3/22 - Added function for calculating the fluxes. Some new support for fitting the maps using least dqaure fitting. However, it is now broken do not use.
