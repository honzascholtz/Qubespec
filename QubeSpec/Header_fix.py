#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:24:16 2023

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


import shutil
from astropy.io import fits
JWST_FILE=PATH+'JWST/Data/GTO/HFL3/HFLS3_NIRSPEC_jw1264_o012_woff_px0.1_emsm_wOD_snr95.0-3_wCR4_wCTX1000.pmap_g395h-f290lp_extfluxcal_s3d.fits'
altered_file=JWST_FILE.replace(PATH+'JWST/Data/GTO/HFL3/HFLS3_NIRSPEC_jw1264_o012_woff_px0.1_emsm_wOD_snr95.0-3_wCR4_wCTX1000.pmap_g395h-f290lp_extfluxcal_s3d.fits',\
                               PATH+'JWST/Data/GTO/HFL3/HFLS3_NIRSPEC_jw1264_o012_woff_px0.1_emsm_wOD_snr95.0-3_wCR4_wCTX1000.pmap_g395h-f290lp_extfluxcal_s3d_shifted.fits')
shutil.copyfile(JWST_FILE, altered_file)
[CRV1,CRV2,CRP1,CRP2]=[256.6990468517151, 58.773266819964604, 32, 30]
fits.setval(altered_file, 'CRPIX1', value=float(CRP1),ext=1)
fits.setval(altered_file, 'CRPIX2', value=float(CRP2),ext=1)
fits.setval(altered_file, 'CRVAL1', value=float(CRV1),ext=1)
fits.setval(altered_file, 'CRVAL2', value=float(CRV2),ext=1)

