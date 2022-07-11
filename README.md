Data analysis pipeline to analyze data from VLT/KMOS, VLT/SINFONI and soon JWST/NIRSPEC to map emission lines in AGN and incactive host galaxies, create flux and kinematic maps as well as modelling outflows in Halpha and [OIII]. 

Code in IFU_tools_class is the main body of the code and example to run it is in cubes_prep.py

Graph_setup sets up matplotlib to make plots that Dave and Chris like. 

WHEN UPDATING THE FITTING_TOOLS_MCMC.PY PLEASE EDIT 35 TO POINT THE CODE TO THE CORRECT LOCATION OF THE FeII TEMPLATES. 

CHANGELOG:

8/7/22 - Addition of Halpha outflow model. Addition of feII template fitting in the [OIII] lines. Addition of Hbeta nar and BLR at the same time.

29/3/22 - redhsift is now fitted +-0.05 - done for safety reasons. Improved plotting of Hbeta and emission line/velocity maps

21/3/22 - Fixed bug in SNR_calc for non outflow [OIII] fit. 

20/3/22 - Added fitting of narrow Hbeta, [SII] including SNR calc and Flux calculation. Fixed some type-2 Halpha plotting problems. 

11/3/22 - Added support for mapping the emission - spaxel by spaxel fitting for the [OIII] emission. Currently fitting a single Gaussian profile - outflow detection and mapping will be done differently. 

10/3/22 - Added function for calculating the fluxes. Some new support for fitting the maps using least dqaure fitting. However, it is now broken do not use. 
