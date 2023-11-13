#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:19:37 2022

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
import pickle
from scipy.optimize import curve_fit


from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.powerlaws import BrokenPowerLaw1D


nan= float('nan')

pi= np.pi
e= np.e

c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

Ken98= (4.5*10**-44)
Conversion2Chabrier=1.7 # Also Madau
Calzetti12= 2.8*10**-44
arrow = u'$\u2193$'

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
from scipy.stats import norm, uniform

import numba
# =============================================================================
# FeII code
# =============================================================================
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.interpolate import interp1d
#Loading the template
from . import FeII_templates as pth
PATH_TO_FeII = pth.__path__[0]+ '/'

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


# =============================================================================
# High lum QSO model
# =============================================================================
def OIII_QSO(x, z, cont,cont_grad,\
             OIIIn_peak, OIIIw_peak, OIII_fwhm,\
             OIII_out, out_vel,\
             Hb_BLR1_peak, Hb_BLR2_peak, Hb_BLR_fwhm1, Hb_BLR_fwhm2, Hb_BLR_vel,\
             Hb_nar_peak, Hb_out_peak):

    ############################################
    # OIII
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482

    Nar_fwhm_hb = OIII_fwhm/3e5*Hbeta/2.35482
    Out_fwhm_hb = OIII_out/3e5*Hbeta/2.35482

    out_vel_wv = out_vel/3e5*OIIIr


    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)

    ############################################
    # Hbeta BLR
    Hb_BLR_vel_wv = Hb_BLR_vel/3e5*Hbeta
    Hb_BLR1_fwhm_wv = Hb_BLR_fwhm1/3e5*Hbeta/2.35482
    Hb_BLR2_fwhm_wv = Hb_BLR_fwhm2/3e5*Hbeta/2.35482


    Hbeta_BLR_wv = Hbeta+Hb_BLR_vel_wv
    Hbeta_BLR = gauss(x, Hb_BLR1_peak, Hbeta_BLR_wv, Hb_BLR1_fwhm_wv) +\
                gauss(x, Hb_BLR2_peak, Hbeta_BLR_wv, Hb_BLR2_fwhm_wv)

    ############################################
    # Hbeta NLR
    out_vel_wv_hb = out_vel/3e5*Hbeta

    Hbeta_NLR = gauss(x, Hb_nar_peak, Hbeta, Nar_fwhm_hb)  + \
                gauss(x, Hb_out_peak, Hbeta+out_vel_wv_hb , Out_fwhm_hb)

    ############################################
    # Continuum
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)

    return contm+ OIII_nar + OIII_out + Hbeta_BLR + Hbeta_NLR



@numba.njit
def log_prior_OIII_QSO(theta,priors):
    z, cont,cont_grad,OIIIn_peak, OIIIw_peak, OIII_fwhm,OIII_out, out_vel,\
        Hb_BLR1_peak, Hb_BLR2_peak, Hb_BLR_fwhm1, Hb_BLR_fwhm2, Hb_BLR_vel,\
            Hb_nar_peak, Hb_out_peak, = theta

    if Hb_out_peak>Hb_nar_peak:
        return -np.inf

    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hb_BLR1_peak'][1] < np.log10(Hb_BLR1_peak)< priors['Hb_BLR1_peak'][2] and priors['Hb_BLR2_peak'][1] < np.log10(Hb_BLR2_peak)< priors['Hb_BLR2_peak'][2]  \
                    and priors['Hb_BLR1_fwhm'][1] < Hb_BLR_fwhm1< priors['Hb_BLR1_fwhm'][2] and priors['Hb_BLR2_fwhm'][1] < Hb_BLR_fwhm2< priors['Hb_BLR2_fwhm'][2]\
                        and priors['Hb_BLR_vel'][1]<Hb_BLR_vel<priors['Hb_BLR_vel'][2] \
                            and  priors['Hb_nar_peak'][1] < np.log10(Hb_nar_peak)<priors['Hb_nar_peak'][2] and  priors['Hb_out_peak'][1] < np.log10(Hb_out_peak)<priors['Hb_out_peak'][2]:
                                return 0.0

    return -np.inf


def BKPLG(x, peak,center,sig, a1,a2):
    gk = Gaussian1DKernel(stddev=sig)

    BKP = BrokenPowerLaw1D.evaluate(x,1, center , a1,a2)

    convolved = convolve(BKP, gk)

    return convolved/max(convolved)*peak

# =============================================================================
# High lum QSO model broken power law BLR
# =============================================================================
def OIII_QSO_BKPL(x, z, cont,cont_grad,\
             OIIIn_peak, OIIIw_peak, OIII_fwhm,\
             OIII_out, out_vel,\
             Hb_BLR_peak, zBLR, Hb_BLR_alp1, Hb_BLR_alp2, Hb_BLR_sig,\
             Hb_nar_peak, Hb_out_peak):

    ############################################
    # OIII
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482

    Nar_fwhm_hb = OIII_fwhm/3e5*Hbeta/2.35482
    Out_fwhm_hb = OIII_out/3e5*Hbeta/2.35482

    out_vel_wv = out_vel/3e5*OIIIr


    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)

    ############################################
    # Hbeta BLR
    Hb_BLR_vel_wv = Hbeta = 4862.6*(1+zBLR)/1e4

    Hbeta_BLR_wv = Hbeta+Hb_BLR_vel_wv
    Hbeta_BLR = BKPLG(x, Hb_BLR_peak, Hbeta_BLR_wv, Hb_BLR_sig, Hb_BLR_alp1, Hb_BLR_alp2)


    ############################################
    # Hbeta NLR
    out_vel_wv_hb = out_vel/3e5*Hbeta

    Hbeta_NLR = gauss(x, Hb_nar_peak, Hbeta, Nar_fwhm_hb)  + \
                gauss(x, Hb_out_peak, Hbeta+out_vel_wv_hb , Out_fwhm_hb)

    ############################################
    # Continuum
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)

    return contm+ OIII_nar + OIII_out + Hbeta_BLR + Hbeta_NLR


@numba.njit
def log_prior_OIII_QSO_BKPL(theta,priors):
    z, cont,cont_grad,OIIIn_peak, OIIIw_peak, OIII_fwhm,OIII_out, out_vel,\
        Hb_BLR_peak, Hb_BLR_vel, Hb_BLR_alp1, Hb_BLR_alp2, Hb_BLR_sig,\
            Hb_nar_peak, Hb_out_peak, = theta

    if OIIIn_peak<OIIIw_peak:
        return -np.inf
    if Hb_out_peak>Hb_nar_peak:
        return -np.inf

    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hb_BLR_peak'][1] < np.log10(Hb_BLR_peak)< priors['Hb_BLR_peak'][2] and priors['Hb_BLR_sig'][1] < Hb_BLR_sig< priors['Hb_BLR_sig'][2] \
                    and priors['Hb_BLR_alp1'][1] < Hb_BLR_alp1< priors['Hb_BLR_alp1'][2] and priors['Hb_BLR_alp2'][1] < Hb_BLR_alp2< priors['Hb_BLR_alp2'][2] \
                        and priors['Hb_BLR_vel'][1]<Hb_BLR_vel<priors['Hb_BLR_vel'][2] \
                            and  priors['Hb_nar_peak'][1] < np.log10(Hb_nar_peak)<priors['Hb_nar_peak'][2] and  priors['Hb_out_peak'][1] < np.log10(Hb_out_peak)<priors['Hb_out_peak'][2]:
                                return 0.0

    return -np.inf


@numba.njit
def log_prior_OIII_QSO_BKPL(theta, priors):
    z, cont,cont_grad,OIIIn_peak, OIIIw_peak, OIII_fwhm,OIII_out, out_vel,\
        Hb_BLR_peak, Hb_BLR_vel, Hb_BLR_alp1, Hb_BLR_alp2, Hb_BLR_sig,\
            Hb_nar_peak, Hb_out_peak, = theta.copy()
    '''
    if OIIIn_peak<OIIIw_peak:
        return -np.inf
    if Hb_out_peak>Hb_nar_peak:
        return -np.inf
    '''
    logprior = sum([ f.logpdf(t) for f,t in zip(priors, theta)])

    return logprior

# =============================================================================
# High lum QSO Fe model
# =============================================================================
def OIII_Fe_QSO(x, z, cont,cont_grad,\
             OIIIn_peak, OIIIw_peak, OIII_fwhm,\
             OIII_out, out_vel,\
             Hb_BLR1_peak, Hb_BLR2_peak, Hb_BLR_fwhm1, Hb_BLR_fwhm2, Hb_BLR_vel,\
             Hb_nar_peak, Hb_out_peak, FeII_peak, FeII_fwhm, template):

    ############################################
    # OIII
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4

    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482

    Nar_fwhm_hb = OIII_fwhm/3e5*Hbeta/2.35482
    Out_fwhm_hb = OIII_out/3e5*Hbeta/2.35482

    out_vel_wv = out_vel/3e5*OIIIr


    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)

    ############################################
    # Hbeta BLR
    Hb_BLR_vel_wv = Hb_BLR_vel/3e5*Hbeta
    Hb_BLR1_fwhm_wv = Hb_BLR_fwhm1/3e5*Hbeta/2.35482
    Hb_BLR2_fwhm_wv = Hb_BLR_fwhm2/3e5*Hbeta/2.35482


    Hbeta_BLR_wv = Hbeta+Hb_BLR_vel_wv
    Hbeta_BLR = gauss(x, Hb_BLR1_peak, Hbeta_BLR_wv, Hb_BLR1_fwhm_wv) +\
                gauss(x, Hb_BLR2_peak, Hbeta_BLR_wv, Hb_BLR2_fwhm_wv)

    ############################################
    # Hbeta NLR
    out_vel_wv_hb = out_vel/3e5*Hbeta

    Hbeta_NLR = gauss(x, Hb_nar_peak, Hbeta, Nar_fwhm_hb)  + \
                gauss(x, Hb_out_peak, Hbeta+out_vel_wv_hb , Out_fwhm_hb)

    ###################################
    # FeII
    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron
    FeII = FeII_peak*FeII_fce(x, z, FeII_fwhm)
    ############################################
    # Continuum
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)

    return contm+ OIII_nar + OIII_out + Hbeta_BLR + Hbeta_NLR+ FeII

@numba.njit
def log_prior_OIII_Fe_QSO(theta,priors):
    z, cont,cont_grad,OIIIn_peak, OIIIw_peak, OIII_fwhm,OIII_out, out_vel,\
        Hb_BLR1_peak, Hb_BLR2_peak, Hb_BLR_fwhm1, Hb_BLR_fwhm2, Hb_BLR_vel,\
            Hb_nar_peak, Hb_out_peak,FeII_peak, FeII_fwhm, = theta

    if Hb_out_peak>Hb_nar_peak:
        return -np.inf

    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hb_BLR1_peak'][1] < np.log10(Hb_BLR1_peak)< priors['Hb_BLR1_peak'][2] and priors['Hb_BLR2_peak'][1] < np.log10(Hb_BLR2_peak)< priors['Hb_BLR2_peak'][2]  \
                    and priors['Hb_BLR1_fwhm'][1] < Hb_BLR_fwhm1< priors['Hb_BLR1_fwhm'][2] and priors['Hb_BLR2_fwhm'][1] < Hb_BLR_fwhm2< priors['Hb_BLR2_fwhm'][2]\
                        and priors['Hb_BLR_vel'][1]<Hb_BLR_vel<priors['Hb_BLR_vel'][2] \
                            and  priors['Hb_nar_peak'][1] < np.log10(Hb_nar_peak)<priors['Hb_nar_peak'][2] and  priors['Hb_out_peak'][1] < np.log10(Hb_out_peak)<priors['Hb_out_peak'][2]\
                                and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                                    return 0.0

    return -np.inf


# =============================================================================
# Halpha QSO model
# =============================================================================
def Hal_QSO_BKPL(x, z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel,Ha_BLR_peak, zBLR, Ha_BLR_alp1, Ha_BLR_alp2, Ha_BLR_sig):


    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4

    Hal_wv_vel = 6564.52*(1+z)/1e4 + outflow_vel/3e5*Hal_wv
    NII_r_vel = 6585.27*(1+z)/1e4 + outflow_vel/3e5*Hal_wv
    NII_b_vel = 6549.86*(1+z)/1e4 + outflow_vel/3e5*Hal_wv

    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482

    out_vel_hal = outflow_fwhm/3e5*Hal_wv/2.35482
    out_vel_niir = outflow_fwhm/3e5*NII_r/2.35482
    out_vel_niib = outflow_fwhm/3e5*NII_b/2.35482

    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_vel_hal)
    NII_nar_r = gauss(x, NII_peak, NII_r, Nar_vel_niir)
    NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_vel_niib)

    Hal_out = gauss(x, Hal_out_peak, Hal_wv_vel, out_vel_hal)
    NII_out_r = gauss(x, NII_out_peak, NII_r_vel, out_vel_niir)
    NII_out_b = gauss(x, NII_out_peak/3, NII_b_vel, out_vel_niib)

    outflow = Hal_out+ NII_out_r + NII_out_b


    ############################################
    # Hbeta BLR
    Ha_BLR_vel_wv = 6564.52*(1+zBLR)/1e4

    Ha_BLR_wv = Hal_wv+Ha_BLR_vel_wv
    Ha_BLR = BKPLG(x, Ha_BLR_peak, Ha_BLR_wv, Ha_BLR_sig, Ha_BLR_alp1, Ha_BLR_alp2)

    ############################################
    # Continuum
    contm = PowerLaw1D.evaluate(x, cont, Hal_wv, alpha=cont_grad)

    return contm+Hal_nar+NII_nar_r+NII_nar_b + outflow+ Ha_BLR


@numba.njit
def Halpha_OIII_QSO_BKPL(x, z, cont,cont_grad, Hal_peak, NII_peak, OIII_peak,Hbeta_peak, Nar_fwhm, \
                      Hal_out_peak, NII_out_peak,OIII_out_peak, Hbeta_out_peak,\
                      outflow_fwhm, outflow_vel,\
                      Hal_BLR_peak, Hbeta_BLR_peak,  zBLR, BLR_alp1, BLR_alp2, BLR_sig):

    Hal_part = Hal_QSO_BKPL(x, z, 0, 0, Hal_peak, NII_peak, Nar_fwhm, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel, Hal_BLR_peak, zBLR, BLR_alp1, BLR_alp2, BLR_sig)

    OIII_part = OIII_QSO_BKPL(x, z, cont,cont_grad,\
                 OIII_peak, OIII_out_peak, Nar_fwhm,\
                 outflow_fwhm, outflow_vel,\
                 Hbeta_BLR_peak, zBLR, BLR_alp1, BLR_alp2, BLR_sig,\
                 Hbeta_peak, Hbeta_out_peak)

    return Hal_part + OIII_part
