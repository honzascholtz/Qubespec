#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:23:35 2022

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

from astropy.modeling.powerlaws import PowerLaw1D
import numba

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

def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)

    y= k* e**expo

    return y
from astropy.modeling.powerlaws import PowerLaw1D

def Full_optical(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, Hgamma_peak, Hdelta_peak, NeIII_peak, OII_peak, OII_rat,OIIIc_peak, HeI_peak,HeII_peak, Nar_fwhm):
    # Halpha side of things
    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4
    
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_fwhm)
    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_fwhm)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_fwhm)
    
    Hgamma_wv = 4341.647191*(1+z)/1e4
    Hdelta_wv = 4102.859855*(1+z)/1e4
    
    Hgamma_nar = gauss(x, Hgamma_peak, Hgamma_wv, Nar_fwhm)
    Hdelta_nar = gauss(x, Hdelta_peak, Hdelta_wv, Nar_fwhm)
    
    
    # [OIII] side of things
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta, Nar_fwhm)
    
    NeIII = gauss(x, NeIII_peak, 3869.68*(1+z)/1e4, Nar_fwhm ) + gauss(x, 0.322*NeIII_peak, 3968.68*(1+z)/1e4, Nar_fwhm)
    
    OII = gauss(x, OII_peak, 3727.1*(1+z)/1e4, Nar_fwhm )  + gauss(x, OII_rat*OII_peak, 3729.875*(1+z)/1e4, Nar_fwhm) 
    
    OIIIc = gauss(x, OIIIc_peak, 4364.436*(1+z)/1e4, Nar_fwhm )
    HeI = gauss(x, HeI_peak, 3889.73*(1+z)/1e4, Nar_fwhm )
    HeII = gauss(x, HeII_peak, 4686.0*(1+z)/1e4, Nar_fwhm )

    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)

    return contm+Hal_nar+NII_nar_r+NII_nar_b + OIII_nar + Hbeta_nar + Hgamma_nar + Hdelta_nar + NeIII+ OII + OIIIc+ HeI+HeII

def Full_optical_outflow(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak,\
                          Hgamma_peak, Hdelta_peak, NeIII_peak, OII_peak, OII_rat,OIIIc_peak, \
                            HeI_peak,HeII_peak, Nar_fwhm,\
                                Hal_out_peak, OIII_out_peak, NII_out_peak, Hbeta_out_peak, \
                                  outflow_vel, outflow_fwhm):
    # Halpha side of things
    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4
    
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_fwhm)
    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_fwhm)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_fwhm)
    
    Hgamma_wv = 4341.647191*(1+z)/1e4
    Hdelta_wv = 4102.859855*(1+z)/1e4
    
    Hgamma_nar = gauss(x, Hgamma_peak, Hgamma_wv, Nar_fwhm)
    Hdelta_nar = gauss(x, Hdelta_peak, Hdelta_wv, Nar_fwhm)
    
    ##### Halpha outflow side
    Hal_out = gauss(x, Hal_out_peak, Hal_wv+outflow_vel/3e5*Hal_wv, outflow_fwhm)
    NII_out_r = gauss(x, NII_out_peak, NII_r+outflow_vel/3e5*NII_r, outflow_fwhm)
    NII_out_b = gauss(x, NII_out_peak/3, NII_b+outflow_vel/3e5*NII_b, outflow_fwhm)
    
    # [OIII] side of things
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta, Nar_fwhm)

    OIII_out = gauss(x, OIII_out_peak, OIIIr+outflow_vel/3e5*OIIIr, outflow_fwhm) + gauss(x, OIII_out_peak/3, OIIIb+outflow_vel/3e5*OIIIb, outflow_fwhm)
    Hbeta_out = gauss(x, Hbeta_out_peak, Hbeta+outflow_vel/3e5*Hbeta, outflow_fwhm)
    
    NeIII = gauss(x, NeIII_peak, 3869.68*(1+z)/1e4, Nar_fwhm ) + gauss(x, 0.322*NeIII_peak, 3968.68*(1+z)/1e4, Nar_fwhm)
    
    OII = gauss(x, OII_peak, 3727.1*(1+z)/1e4, Nar_fwhm )  + gauss(x, OII_rat*OII_peak, 3729.875*(1+z)/1e4, Nar_fwhm) 
    
    OIIIc = gauss(x, OIIIc_peak, 4364.436*(1+z)/1e4, Nar_fwhm )
    HeI = gauss(x, HeI_peak, 3889.73*(1+z)/1e4, Nar_fwhm )
    HeII = gauss(x, HeII_peak, 4686.0*(1+z)/1e4, Nar_fwhm )

    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)

    return contm+Hal_nar+NII_nar_r+NII_nar_b + OIII_nar + Hbeta_nar + Hgamma_nar + Hdelta_nar + NeIII+ OII + OIIIc+ HeI+HeII + \
      Hal_out + NII_out_b + NII_out_r + OIII_out + Hbeta_out
