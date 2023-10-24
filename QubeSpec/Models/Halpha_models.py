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

from scipy.stats import norm, uniform

def gauss(x, k, mu,sig):

    expo= -((x-mu)**2)/(2*sig*sig)

    y= k* e**expo

    return y

# =============================================================================
#  Function for fitting Halpha with BLR
# =============================================================================
def Halpha_wBLR(x,z,cont, cont_grad, Hal_peak, BLR_peak, NII_peak, Nar_fwhm, BLR_fwhm, zBLR, SII_rpk, SII_bpk):
    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4

    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4

    Nar_sig= Nar_fwhm/3e5*Hal_wv/2.35482
    BLR_sig = BLR_fwhm/3e5*Hal_wv/2.35482

    BLR_wv = 6564.52*(1+zBLR)/1e4

    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)
    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_sig)
    Hal_blr = gauss(x, BLR_peak, BLR_wv, BLR_sig)

    NII_rg = gauss(x, NII_peak, NII_r, Nar_sig)
    NII_bg = gauss(x, NII_peak/3, NII_b, Nar_sig)

    SII_rg = gauss(x, SII_rpk, SII_r, Nar_sig)
    SII_bg = gauss(x, SII_bpk, SII_b, Nar_sig)

    return contm + Hal_nar + Hal_blr + NII_rg + NII_bg + SII_rg + SII_bg

@numba.njit
def log_prior_Halpha_BLR(theta, priors):
    z, cont, cont_grad ,Hal_peak, BLR_peak, NII_peak, Nar_fwhm, BLR_fwhm, BLR_offset, SII_rpk, SII_bpk  = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak<SII_rpk) | (Hal_peak<SII_bpk) | (BLR_peak<0):
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0] ==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1]))
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))

    return results



# =============================================================================
# Function to fit just narrow Halpha
# =============================================================================
def Halpha(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk):
    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4

    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482

    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4

    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_vel_hal)

    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_vel_niir)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_vel_niib)

    SII_rg = gauss(x, SII_rpk, SII_r, Nar_vel_hal)
    SII_bg = gauss(x, SII_bpk, SII_b, Nar_vel_hal)
    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)

    return contm+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg

@numba.njit
def log_prior_Halpha(theta, priors):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0):
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0] ==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1]))
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))

    return results

# =============================================================================
# Function to fit  Halpha with outflow
# =============================================================================
def Halpha_outflow(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel):

    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4

    Hal_wv_vel = 6564.52*(1+z)/1e4 + outflow_vel/3e5*Hal_wv
    NII_r_vel = 6585.27*(1+z)/1e4 + outflow_vel/3e5*NII_r
    NII_b_vel = 6549.86*(1+z)/1e4 + outflow_vel/3e5*NII_b


    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482

    out_vel_hal = outflow_fwhm/3e5*Hal_wv/2.35482
    out_vel_niir = outflow_fwhm/3e5*NII_r/2.35482
    out_vel_niib = outflow_fwhm/3e5*NII_b/2.35482

    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4

    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_vel_hal)
    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_vel_niir)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_vel_niib)

    Hal_out = gauss(x, Hal_out_peak, Hal_wv_vel, out_vel_hal)
    NII_out_r = gauss(x, NII_out_peak, NII_r_vel, out_vel_niir)
    NII_out_b = gauss(x, NII_out_peak/3, NII_b_vel, out_vel_niib)

    outflow = Hal_out+ NII_out_r + NII_out_b

    SII_rg = gauss(x, SII_rpk, SII_r, Nar_vel_hal)
    SII_bg = gauss(x, SII_bpk, SII_b, Nar_vel_hal)
    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)
    return contm+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg + outflow

@numba.njit
def log_prior_Halpha_outflow(theta, priors):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel = theta
    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak<SII_rpk) | (Hal_peak<SII_bpk):
        return -np.inf
    if Hal_peak<Hal_out_peak:
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0] ==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1]))
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))

    return results
