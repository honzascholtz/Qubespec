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

PATH_TO_FeII = '/Users/jansen/My Drive/Astro/General_data/FeII_templates/'


def find_nearest(array, value):
    """ Find the location of an array closest to a value 
	
	"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def gauss(x, k, mu,sig):

    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

##=============================================================================
# Fitting Halpha + [OIII]
# =============================================================================
def Halpha_OIII(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak,  OIII_fwhm, Hbeta_peak, OIII_vel, OI_peak):
    # Halpha side of things
    Hal_wv = 6562.8*(1+z)/1e4     
    NII_r = 6583.*(1+z)/1e4
    NII_b = 6548.*(1+z)/1e4
    
    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482
    
    SII_r = 6731.*(1+z)/1e4   
    SII_b = 6716.*(1+z)/1e4   
    
    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_vel_hal)
    
    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_vel_niir)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_vel_niib)
    
    SII_rg = gauss(x, SII_rpk, SII_r, Nar_vel_hal)
    SII_bg = gauss(x, SII_bpk, SII_b, Nar_vel_hal)
    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)
    
    # [OIII] side of things
    
    OIIIr = 5008.*(1+z)/1e4
    OIIIr = OIIIr-OIII_vel/3e5*OIIIr
    OIIIb = 4960.*(1+z)/1e4   
    OIIIb = OIIIb-OIII_vel/3e5*OIIIb
    
    Hbeta = 4861.*(1+z)/1e4 
    
    OIII_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, OIII_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, OIII_fwhm)
    
    Hbeta_fwhm = Nar_fwhm/3e5*Hbeta/2.35482
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta, Hbeta_fwhm )
    
    OI = 6300.*(1+z)/1e4
    OI = OI-OIII_vel/3e5*OI
    OI_nar = gauss(x, OI_peak, OI, OIII_fwhm)
    
    
    return contm+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg+ OIII_nar + Hbeta_nar + OI_nar


def log_prior_Halpha_OIII(theta, priors):
    z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak,  OIII_fwhm, Hbeta_peak, OIII_vel, OI_peak = theta
    
    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak/Hbeta_peak<(2.86/1.35)):
        return -np.inf
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['Hal_peak'][1] < np.log10(Hal_peak) < priors['Hal_peak'][2] \
            and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2]\
                and priors['NII_peak'][1] < np.log10(NII_peak) < priors['NII_peak'][2]\
                    and priors['Nar_fwhm'][1] < Nar_fwhm <priors['Nar_fwhm'][2] \
                        and priors['SII_rpk'][1] < np.log10(SII_rpk) < priors['SII_rpk'][2] and priors['SII_bpk'][1] < np.log10(SII_bpk)<priors['SII_bpk'][2]\
                            and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIIIn_fwhm'][1] < OIII_fwhm <priors['OIIIn_fwhm'][2]\
                                and  priors['OIII_vel'][1]<OIII_vel<priors['OIII_vel'][2]\
                                    and priors['OI_peak'][1] < np.log10(OI_peak) < priors['OI_peak'][2]\
                                        and 0.44<(SII_rpk/SII_bpk)<1.45:
                                            return 0.0 
    
    return -np.inf