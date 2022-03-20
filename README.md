Basic tools to analyse KMOS cubes from KASHz

Code in IFU_tools_class is the main body of the code and example to run it is in cubes_prep.py

Graph_setup sets up matplotlib to make plots that Dave and Chris like. 

CHANGELOG:

20/3 - Added fitting of narrow Hbeta, [SII] including SNR calc and Flux calculation. Fixed some type-2 Halpha plotting problems. 

11/3 - Added support for mapping the emission - spaxel by spaxel fitting for the [OIII] emission. Currently fitting a single Gaussian profile - outflow detection and mapping will be done differently. 

10/3 - Added function for calculating the fluxes. Some new support for fitting the maps using least dqaure fitting. However, it is now broken do not use. 
