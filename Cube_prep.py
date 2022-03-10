#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 09:17:06 2022

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.optimize import curve_fit

import Graph_setup as gst 

nan= float('nan')

pi= np.pi
e= np.e

plt.close('all')
c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

Ken98= (4.5*10**-44)
Conversion2Chabrier=1.7 # Also Madau
Calzetti12= 2.8*10**-44
arrow = u'$\u2193$' 


PATH='/Users/jansen/My Drive/Astro/'
fsz = gst.graph_format()


import IFU_tools_class as IFU 

plot_it=0


'''

IFU_cube_path = PATH +'KMOS_SIN/KMOS_data/H_band/COMBINE_SHIFT_cdfs_39849_x_r.fits' 
instrument = 'KMOS'
ID = 'XID_208'
z= 1.61


    

Cube = IFU.Cube(IFU_cube_path, z, ID, instrument, 'path', 'H')

Cube.mask_emission()
Cube.mask_sky( 1.5)
Cube.collapse_white(plot_it)
Cube.find_center( plot_it)

Cube.choose_pixels(plot_it, rad= 0.3)#, flg='K587')

Cube.stack_sky(plot_it, expand=0)
Cube.D1_spectra_collapse( plot_it, addsave='_inner')


Cube.fitting_collapse_Halpha( plot_it)

wave = Cube.obs_wave.copy()
flux = Cube.D1_spectrum.copy()
error = Cube.D1_spectrum_er.copy



plt.show()
'''


import importlib
importlib.reload(IFU )

IFU_cube_path = PATH +'KMOS_SIN/KMOS_data/H_band/COMBINE_SHIFT_SCI_RECONSTRUCTED_GS3_19791_H.fits' 
#IFU_cube_path = PATH +'KMOS_SIN/KMOS_data/YJ_band/COMBINE_SHIFT_AGN6.fits' 
instrument = 'KMOS'
ID = 'XID_587'
z= 2.2246
#z= 1.61
Band = 'H'


Save_path = 'path'
'''

IFU_cube_path = PATH +'KMOS_SIN/KMOS_data/YJ_band/COMBINE_SHIFT_cdfs_42968_x.fits'
instrument = 'KMOS'
ID = 'XID_208'
z= 1.61
Band = 'YJ'
'''
Cube = IFU.Cube(IFU_cube_path, z, ID, instrument,Save_path , Band)

Cube.mask_emission()
Cube.mask_sky( 1.5)
Cube.collapse_white(plot_it)
Cube.find_center( plot_it)

Cube.choose_pixels(plot_it, rad= 0.3)#, flg='K587')

Cube.stack_sky(plot_it, expand=0)
Cube.D1_spectra_collapse( plot_it, addsave='_inner')


Cube.fitting_collapse_OIII( plot_it)
Cube.report()

print('Flux total ', IFU.flux_calc(Cube.D1_fit_results, 'OIIIt'))
print('Flux narrow ', IFU.flux_calc(Cube.D1_fit_results, 'OIIIn'))
print('Flux wide ', IFU.flux_calc(Cube.D1_fit_results, 'OIIIw'))



plt.show()

