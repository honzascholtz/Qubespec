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

from . import FeII_templates as pth
PATH_TO_FeII = pth.__path__[0]+ '/'

def find_nearest(array, value):
    """ Find the location of an array closest to a value

	"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)
    y= k* e**expo
    return y

def OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak):
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = OIIIr- (48.*(1+z)/1e4)

    Hbeta = 4862.6*(1+z)/1e4
    OIII_nar = gauss(x, OIIIn_peak, OIIIr,OIII_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, OIII_fwhm)

    Hbeta_wv = Hbeta 
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta_wv, OIII_fwhm )
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_nar

def OIII_outflow(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak):

    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
    z_out = z+ out_vel/3e5*(1+z)
    y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)

    return y

def OIII_gal_BLR(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak,\
                  zBLR, Hbeta_blr_peak, BLR_fwhm):
    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)

    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )
    return y

def OIII_outflow_BLR(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak,\
                      zBLR, Hbeta_blr_peak, BLR_fwhm):

    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
    z_out = z+ out_vel/3e5*(1+z)
    y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)
    
    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

    return y

# =============================================================================
# FeII code
# =============================================================================
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.interpolate import interp1d
#Loading the template
Veron_d = pyfits.getdata(PATH_TO_FeII+ 'Veron-cetty_2004.fits')
Veron_hd = pyfits.getheader(PATH_TO_FeII+'Veron-cetty_2004.fits')
Veron_wv = np.arange(Veron_hd['CRVAL1'], Veron_hd['CRVAL1']+ Veron_hd['NAXIS1'])

Tsuzuki = np.loadtxt(PATH_TO_FeII+'FeII_Tsuzuki_opttemp.txt')
Tsuzuki_d = Tsuzuki[:,1]
Tsuzuki_wv = Tsuzuki[:,0]

BG92 = np.loadtxt(PATH_TO_FeII+'bg92.con')
BG92_d = BG92[:,1]
BG92_wv = BG92[:,0]

with open(PATH_TO_FeII+'Preconvolved_FeII.txt', "rb") as fp:
    Templates= pickle.load(fp)

def FeII_Veron(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['Veron_dat'][:,index]

    fce = interp1d(Veron_wv*(1+z)/1e4, convolved , kind='cubic')

    return fce(wave)

def FeII_Tsuzuki(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['Tsuzuki_dat'][:,index]

    fce = interp1d(Tsuzuki_wv*(1+z)/1e4, convolved , kind='cubic')

    return fce(wave)

def FeII_BG92(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['BG92_dat'][:,index]

    fce = interp1d(BG92_wv*(1+z)/1e4, convolved , kind='cubic')

    return fce(wave)

def OIII_gal_BLR_Fe(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak,\
                  zBLR, Hbeta_blr_peak, BLR_fwhm,\
                     FeII_peak, FeII_fwhm, template):
    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)

    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron

    y += FeII_peak*FeII_fce(x, z, FeII_fwhm)

    return y

def OIII_outflow_BLR_Fe(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak,\
                      zBLR, Hbeta_blr_peak, BLR_fwhm,\
                         FeII_peak, FeII_fwhm, template):

    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
    z_out = z+ out_vel/3e5*(1+z)
    y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)
    
    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron

    y += FeII_peak*FeII_fce(x, z, FeII_fwhm)

    return y
