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

import numba

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
def Halpha_OIII(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak, Hbeta_peak):
    # Halpha side of things
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

    # [OIII] side of things

    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4

    Hbeta = 4862.6*(1+z)/1e4

    OIII_fwhm = Nar_fwhm/3e5*OIIIr/2.35482
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, OIII_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, OIII_fwhm)

    Hbeta_fwhm = Nar_fwhm/3e5*Hbeta/2.35482
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta, Hbeta_fwhm )

    return contm+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg+ OIII_nar + Hbeta_nar 

@numba.njit
def log_prior_Halpha_OIII(theta, priors):
    z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak, Hbeta_peak = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak/Hbeta_peak<(2.86/1.35)) | \
        (SII_rpk > Hal_peak) | (SII_bpk> Hal_peak) :
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf

    return results


def Halpha_OIII_outflow(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                        Nar_fwhm, outflow_fwhm, outflow_vel, \
                        Hal_out_peak, NII_out_peak, OIII_out_peak, Hbeta_out_peak):
    # Halpha side of things
    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4
    OIII_r = 5008.24*(1+z)/1e4
    OIII_b = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4
    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4

    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482
    Nar_vel_oiiir = Nar_fwhm/3e5*OIII_r/2.35482
    Nar_vel_oiiib = Nar_fwhm/3e5*OIII_b/2.35482
    Nar_vel_hbe = Nar_fwhm/3e5*Hbeta/2.35482

    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)

    Nar = gauss(x, Hal_peak, Hal_wv, Nar_vel_hal)  + \
        gauss(x, NII_peak, NII_r, Nar_vel_niir)+  gauss(x, NII_peak/3, NII_b, Nar_vel_niib) + \
        gauss(x, SII_rpk, SII_r, Nar_vel_hal) + gauss(x, SII_bpk, SII_b, Nar_vel_hal)  +\
        gauss(x, OIIIn_peak, OIII_r, Nar_vel_oiiir) + gauss(x, OIIIn_peak/3, OIII_b, Nar_vel_oiiib)+\
        gauss(x, Hbeta_peak, Hbeta, Nar_vel_hbe )



    Out_vel_hal = outflow_fwhm/3e5*Hal_wv/2.35482
    Out_vel_niir = outflow_fwhm/3e5*NII_r/2.35482
    Out_vel_niib = outflow_fwhm/3e5*NII_b/2.35482
    Out_vel_oiiir = outflow_fwhm/3e5*OIII_r/2.35482
    Out_vel_oiiib = outflow_fwhm/3e5*OIII_b/2.35482
    Out_vel_hbe = outflow_fwhm/3e5*Hbeta/2.35482


    Hal_wv = 6564.52*(1+z)/1e4   + outflow_vel/3e5*Hal_wv
    NII_r = 6585.27*(1+z)/1e4 + outflow_vel/3e5*NII_r
    NII_b = 6549.86*(1+z)/1e4 + outflow_vel/3e5*NII_b
    OIII_r = 5008.24*(1+z)/1e4 + outflow_vel/3e5*OIII_r
    OIII_b = 4960.3*(1+z)/1e4  + outflow_vel/3e5*OIII_b
    Hbeta = 4862.6*(1+z)/1e4  + outflow_vel/3e5* Hbeta
    SII_r = 6732.67*(1+z)/1e4  + outflow_vel/3e5*SII_r
    SII_b = 6716.*(1+z)/1e4   + outflow_vel/3e5*SII_b


    Outflow =  gauss(x, Hal_out_peak, Hal_wv, Out_vel_hal)  + \
        gauss(x, NII_out_peak, NII_r, Out_vel_niir)+  gauss(x, NII_out_peak/3, NII_b, Out_vel_niib) + \
        gauss(x, OIII_out_peak, OIII_r, Out_vel_oiiir) + gauss(x, OIII_out_peak/3, OIII_b, Out_vel_oiiib)+\
        gauss(x, Hbeta_out_peak, Hbeta, Out_vel_hbe )


    return contm+Nar+ Outflow

@numba.njit
def log_prior_Halpha_OIII_outflow(theta, priors):
    z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                            Nar_fwhm, outflow_fwhm, outflow_vel, \
                            Hal_out_peak, NII_out_peak, OIII_out_peak, Hbeta_out_peak = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak/Hbeta_peak<(2.86/1.35))|\
        (SII_rpk > Hal_peak) | (SII_bpk> Hal_peak):
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf

    return results






def Halpha_OIII_BLR(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                        Nar_fwhm, outflow_fwhm, outflow_vel, \
                        Hal_out_peak, NII_out_peak, OIII_out_peak,  Hbeta_out_peak,\
                        BLR_fwhm, zBLR, BLR_hal_peak, BLR_hbe_peak):



    NLR = Halpha_OIII(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak, Hbeta_peak)


    deltaz = outflow_vel/3e5*(1+z)

    zout = z+ deltaz
    Outflow = Halpha_OIII(x, zout, 0,0,  Hal_out_peak, NII_out_peak, outflow_fwhm , 0, 0, OIII_out_peak, Hbeta_out_peak)

    Hal_wv = 6564.52*(1+z)/1e4
    Hbe_wv = 4862.6*(1+z)/1e4
    BLR_sig_hal = BLR_fwhm/3e5*Hal_wv/2.35482
    BLR_sig_hbe = BLR_fwhm/3e5*Hbe_wv/2.35482

    BLR_wv_hal = 6564.52*(1+zBLR)/1e4
    BLR_wv_hbe = 4862.6*(1+zBLR)/1e4

    Hal_blr = gauss(x, BLR_hal_peak, BLR_wv_hal, BLR_sig_hal)
    Hbe_blr = gauss(x, BLR_hbe_peak, BLR_wv_hbe, BLR_sig_hbe)


    return NLR+Outflow+Hal_blr + Hbe_blr


@numba.njit
def log_prior_Halpha_OIII_BLR(theta, priors):
    z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                            Nar_fwhm, outflow_fwhm, outflow_vel, \
                            Hal_out_peak, NII_out_peak, OIII_out_peak,  Hbeta_out_peak,\
                            BLR_fwhm, zBLR, BLR_hal_peak, BLR_hbe_peak = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak/Hbeta_peak<(2.86/1.35)) \
        | (BLR_hal_peak/BLR_hbe_peak<(2.86/1.35)) | (OIII_out_peak>OIIIn_peak) | (NII_out_peak>NII_peak) \
            | (Hbeta_out_peak>Hbeta_peak) | (Hal_out_peak>Hal_peak) | (BLR_hal_peak<0) | (BLR_hbe_peak<0)  :
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf

    return results


def Halpha_OIII_BLR_simple(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                        Nar_fwhm,\
                        BLR_fwhm, zBLR, BLR_hal_peak, BLR_hbe_peak):

    NLR = Halpha_OIII(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, OIIIn_peak, Hbeta_peak, 0)

    Hal_wv = 6564.52*(1+z)/1e4
    Hbe_wv = 4862.6*(1+z)/1e4
    BLR_sig_hal = BLR_fwhm/3e5*Hal_wv/2.35482
    BLR_sig_hbe = BLR_fwhm/3e5*Hbe_wv/2.35482

    BLR_wv_hal = 6564.52*(1+zBLR)/1e4
    BLR_wv_hbe = 4862.6*(1+zBLR)/1e4

    Hal_blr = gauss(x, BLR_hal_peak, BLR_wv_hal, BLR_sig_hal)
    Hbe_blr = gauss(x, BLR_hbe_peak, BLR_wv_hbe, BLR_sig_hbe)


    return NLR+Hal_blr + Hbe_blr


@numba.njit
def log_prior_Halpha_OIII_BLR_simple(theta, priors):
    z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, SII_rpk, SII_bpk,\
                            Nar_fwhm, \
                            BLR_fwhm, zBLR, BLR_hal_peak, BLR_hbe_peak = theta

    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak/Hbeta_peak<(2.86/1.35)) \
        | (BLR_hal_peak/BLR_hbe_peak<(2.86/1.35))  \
           | (BLR_hal_peak<0) | (BLR_hbe_peak<0)  :
        return -np.inf

    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf

    return results
