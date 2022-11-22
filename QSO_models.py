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
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.optimize import curve_fit

import Graph_setup as gst 

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


PATH='/Users/jansen/My Drive/Astro/'
fsz = gst.graph_format()

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

PATH_TO_FeII = '/Users/jansen/My Drive/Astro/General_data/FeII_templates/'

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


def log_likelihood_OIII_QSO(theta, x, y, yerr):
    
    model = OIII_QSO(x,*theta)
    sigma2 = yerr*yerr
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


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


def log_probability_OIII_QSO(theta, x, y, yerr, priors):
    lp = log_prior_OIII_QSO(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_QSO(theta, x, y, yerr)  


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
             Hb_BLR_peak, Hb_BLR_vel, Hb_BLR_alp1, Hb_BLR_alp2, Hb_BLR_sig,\
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


uniform.logpdf(np.random.normal(size=15), *np.random.normal(size=(2, 15)))

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
    
    '''
    zcont=0.01
    logpriors = np.zeros_like(theta)
    
    logpriors[0] = norm.logpdf(z,priors['z'][0],zcont)
    logpriors[1] = uniform.logpdf(np.log10(cont), priors['cont'][1],priors['cont'][2])  
    logpriors[2] = norm.logpdf(cont_grad,priors['cont_grad'][1], priors['cont_grad'][2])
    
    logpriors[3] = uniform.logpdf(np.log10(OIIIn_peak), priors['OIIIn_peak'][1],priors['OIIIn_peak'][2])
    logpriors[4] = uniform.logpdf(np.log10(OIIIw_peak), priors['OIIIw_peak'][1],priors['OIIIw_peak'][2])
      
    logpriors[5] = uniform.logpdf(OIII_fwhm, priors['OIII_fwhm'][1],priors['OIII_fwhm'][2])
    logpriors[6] = uniform.logpdf(OIII_out, priors['OIII_out'][1],priors['OIII_out'][2])
    logpriors[7] = norm.logpdf(out_vel, priors['out_vel'][1],priors['out_vel'][2])
    
    logpriors[8] = uniform.logpdf(np.log10(Hb_BLR_peak), priors['Hb_BLR_peak'][1],priors['Hb_BLR_peak'][2])
    logpriors[9] = uniform.logpdf(np.log10(Hb_nar_peak), priors['Hb_nar_peak'][1],priors['Hb_nar_peak'][2])
    logpriors[10] = uniform.logpdf(np.log10(Hb_out_peak), priors['Hb_out_peak'][1],priors['Hb_out_peak'][2])
    
    logpriors[11] = norm.logpdf(Hb_BLR_vel,priors['Hb_BLR_vel'][1], priors['Hb_BLR_vel'][2])
    logpriors[12] = uniform.logpdf(Hb_BLR_alp1, priors['Hb_BLR_alp1'][1],priors['Hb_BLR_alp1'][2])
    logpriors[13] = uniform.logpdf(Hb_BLR_alp2, priors['Hb_BLR_alp2'][1],priors['Hb_BLR_alp2'][2])
    logpriors[14] = uniform.logpdf(Hb_BLR_sig, priors['Hb_BLR_sig'][1],priors['Hb_BLR_sig'][2])
    
    logprior = np.sum(logpriors)
    '''
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
def Hal_QSO_BKPL(x, z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel,Ha_BLR_peak, Ha_BLR_vel, Ha_BLR_alp1, Ha_BLR_alp2, Ha_BLR_sig):
            
    
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
    Ha_BLR_vel_wv = Ha_BLR_vel/3e5*Hal_wv
    
    Ha_BLR_wv = Hal_wv+Ha_BLR_vel_wv 
    Ha_BLR = BKPLG(x, Ha_BLR_peak, Ha_BLR_wv, Ha_BLR_sig, Ha_BLR_alp1, Ha_BLR_alp2)
    
    
     
    ############################################
    # Continuum
    contm = PowerLaw1D.evaluate(x, cont, Hal_wv, alpha=cont_grad)
    
    return contm+Hal_nar+NII_nar_r+NII_nar_b + outflow+ Ha_BLR


def log_prior_Halpha_QSO_BKPL(theta, priors):
    
    return sum([ f.logpdf(t) for f,t in zip(priors, theta)])





