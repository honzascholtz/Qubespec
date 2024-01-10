import numpy as np
import QubeSpec.Visulizations as viz

PATH='/Users/jansen/JADES/GA_NIFS/'

Viz = viz.Visualize(PATH+'Results/GS551/GS551_R2700_general_fits_maps.fits',\
                     ['HAL','OIII','OIII_kin'])

#Viz = viz.Visualize(PATH+'Results/GS551/GS551_R2700_Halpha_OIII_fits_maps.fits',\
#                     ['HALPHA','OIII','OIII_kin'])
Viz.showme()