#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:35:45 2017

@author: jscholtz
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.optimize import curve_fit

import emcee
import corner
from astropy.modeling.powerlaws import PowerLaw1D
nan= float('nan')

pi= np.pi
e= np.e

c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

arrow = u'$\u2193$' 

N = 10000
PATH_TO_FeII = '/Users/jansen/My Drive/Astro/General_data/FeII_templates/'

version = 'Main'    

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

# =============================================================================
#  Function for fitting Halpha with BLR
# =============================================================================
def Halpha_wBLR(x,z,cont, cont_grad, Hal_peak, BLR_peak, NII_peak, Nar_fwhm, BLR_fwhm, BLR_offset, SII_rpk, SII_bpk):
    Hal_wv = 6562.8*(1+z)/1e4     
    NII_r = 6583.*(1+z)/1e4
    NII_b = 6548.*(1+z)/1e4
    
    SII_r = 6731.*(1+z)/1e4   
    SII_b = 6716.*(1+z)/1e4   
    
    Nar_sig= Nar_fwhm/3e5*Hal_wv/2.35482
    BLR_sig = BLR_fwhm/3e5*Hal_wv/2.35482
    
    BLR_wv = Hal_wv + BLR_offset/3e5*Hal_wv
    
    contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)
    Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_sig)
    Hal_blr = gauss(x, BLR_peak, BLR_wv, BLR_sig)
    
    NII_rg = gauss(x, NII_peak, NII_r, Nar_sig)
    NII_bg = gauss(x, NII_peak/3, NII_b, Nar_sig)
    
    SII_rg = gauss(x, SII_rpk, SII_r, Nar_sig)
    SII_bg = gauss(x, SII_bpk, SII_b, Nar_sig)
    
    return contm + Hal_nar + Hal_blr + NII_rg + NII_bg + SII_rg + SII_bg
    

def log_likelihood_Halpha_BLR(theta, x, y, yerr):
    
    model = Halpha_wBLR(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_Halpha_BLR(theta, priors):
    z, cont, cont_grad ,Hal_peak, BLR_peak, NII_peak, Nar_fwhm, BLR_fwhm, BLR_offset, SII_rpk, SII_bpk  = theta
    
    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak<SII_rpk) | (Hal_peak<SII_bpk) | (BLR_peak<0):
        return -np.inf
   
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['Hal_peak'][1] < np.log10(Hal_peak) < priors['Hal_peak'][2] and priors['Nar_fwhm'][1] < Nar_fwhm <priors['Nar_fwhm'][2] and priors['NII_peak'][1] < np.log10(NII_peak) < priors['NII_peak'][2]\
            and priors['BLR_peak'][1] < np.log10(BLR_peak) < priors['BLR_peak'][2] and priors['BLR_fwhm'][1] < BLR_fwhm <priors['BLR_fwhm'][2] and priors['BLR_offset'][1] < BLR_offset <priors['BLR_offset'][2]\
                and priors['SII_rpk'][1] < np.log10(SII_rpk) < priors['SII_rpk'][2] and priors['SII_bpk'][1] < np.log10(SII_bpk)<priors['SII_bpk'][2]\
                    and 0.44<(SII_rpk/SII_bpk)<1.45:
                        return 0.0 

    return -np.inf
'''
def log_prior_Halpha_BLR(theta, pr):
    z, cont, cont_grad ,Hal_peak, BLR_peak, NII_peak, Nar_fwhm, BLR_fwhm, BLR_offset, SII_rpk, SII_bpk  = theta
    
    
    zcont=0.05
    priors = []
    
    priors.append(uniform.logpdf(z,pr['z'][1], pr['z'][2]))
    priors.append(uniform.logpdf(np.log10(cont), -4,3))
    priors.append(uniform.logpdf(np.log10(Hal_peak),  -4, 3 ))
    priors.append(uniform.logpdf(np.log10(NII_peak),  -4, 3 ))
    priors.append(uniform.logpdf(Nar_fwhm, 100,1000 ))
    priors.append(norm.logpdf(cont_grad, 0, 0.1))
    priors.append(uniform.logpdf(np.log10(SII_bpk), -4, 3))
    priors.append(uniform.logpdf(np.log10(SII_rpk), -4, 3))
    
    priors.append(uniform.logpdf(np.log10(BLR_peak),  -4, 3 ))
    priors.append(uniform.logpdf(BLR_fwhm, 2000,9000 ))
    priors.append(norm.logpdf(BLR_offset, 0, 200))
    
    logprior = np.sum(priors)
    
    if logprior==np.nan:
        return -np.inf
    else:
        return logprior
'''
def log_probability_Halpha_BLR(theta, x, y, yerr, priors):
    lp = log_prior_Halpha_BLR(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_Halpha_BLR(theta, x, y, yerr)
    
from scipy.stats import norm, uniform

# =============================================================================
# Function to fit just narrow Halpha
# =============================================================================
def Halpha(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk):
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
    
    return contm+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg

def log_likelihood_Halpha(theta, x, y, yerr):
    
    model = Halpha(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_Halpha(theta, priors):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk = theta
    
    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0):
        return -np.inf
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['Hal_peak'][1] < np.log10(Hal_peak) < priors['Hal_peak'][2] and priors['Nar_fwhm'][1] < Nar_fwhm <priors['Nar_fwhm'][2] and priors['NII_peak'][1] < np.log10(NII_peak) < priors['NII_peak'][2]\
            and priors['SII_rpk'][1] < np.log10(SII_rpk) < priors['SII_rpk'][2] and priors['SII_bpk'][1] < np.log10(SII_bpk)<priors['SII_bpk'][2]\
                and 0.44<(SII_rpk/SII_bpk)<1.45:
                    return 0.0 
    
    return -np.inf

'''
def log_prior_Halpha(theta, zguess, zcont):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk = theta
    
    
    priors = np.zeros_like(theta)
    
    priors[0] = uniform.logpdf(z,zguess-zcont,zguess+zcont)
    priors[1] = uniform.logpdf(np.log10(cont), -4,3)
    priors[2] = uniform.logpdf(np.log10(Hal_peak),  -4, 3 )
    priors[3] = uniform.logpdf(np.log10(NII_peak),  -4, 3 )
    priors[4] = uniform.logpdf(Nar_fwhm, 100,1000 )
    priors[5] = norm.logpdf(cont_grad, 0, 0.1)
    priors[6] = uniform.logpdf(np.log10(SII_bpk), -4, 3)
    priors[7] = uniform.logpdf(np.log10(SII_rpk), -4, 3)
    
    logprior = np.sum(priors)
    
    if logprior==np.nan:
        return -np.inf
    else:
        return logprior
'''
def log_probability_Halpha(theta, x, y, yerr, priors):
    lp = log_prior_Halpha(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_Halpha(theta, x, y, yerr)


# =============================================================================
# Function to fit  Halpha with outflow
# =============================================================================
def Halpha_outflow(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel):
    Hal_wv = 6562.8*(1+z)/1e4     
    NII_r = 6583.*(1+z)/1e4
    NII_b = 6548.*(1+z)/1e4
    
    Hal_wv_vel = 6562.8*(1+z)/1e4 + outflow_vel/3e5*Hal_wv 
    NII_r_vel = 6583.*(1+z)/1e4 + outflow_vel/3e5*Hal_wv 
    NII_b_vel = 6548.*(1+z)/1e4 + outflow_vel/3e5*Hal_wv 
    
    
    Nar_vel_hal = Nar_fwhm/3e5*Hal_wv/2.35482
    Nar_vel_niir = Nar_fwhm/3e5*NII_r/2.35482
    Nar_vel_niib = Nar_fwhm/3e5*NII_b/2.35482
    
    out_vel_hal = outflow_fwhm/3e5*Hal_wv/2.35482
    out_vel_niir = outflow_fwhm/3e5*NII_r/2.35482
    out_vel_niib = outflow_fwhm/3e5*NII_b/2.35482
    
    SII_r = 6731.*(1+z)/1e4   
    SII_b = 6716.*(1+z)/1e4   
    
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

def log_likelihood_Halpha_outflow(theta, x, y, yerr):
    
    model = Halpha_outflow(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_Halpha_outflow(theta, priors):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk, Hal_out_peak, NII_out_peak, outflow_fwhm, outflow_vel = theta
    if (Hal_peak<0) | (NII_peak<0) | (SII_rpk<0) | (SII_bpk<0) | (Hal_peak<SII_rpk) | (Hal_peak<SII_bpk):
        return -np.inf
    if Hal_peak<Hal_out_peak:
        return -np.inf
   
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['Hal_peak'][1] < np.log10(Hal_peak) < priors['Hal_peak'][2] and priors['Nar_fwhm'][1] < Nar_fwhm <priors['Nar_fwhm'][2] and priors['NII_peak'][1] < np.log10(NII_peak) < priors['NII_peak'][2]\
            and priors['SII_rpk'][1] < np.log10(SII_rpk) < priors['SII_rpk'][2] and priors['SII_bpk'][1] < np.log10(SII_bpk) <priors['SII_bpk'][2]\
                and priors['Hal_out_peak'][1] < np.log10(Hal_out_peak) < priors['Hal_out_peak'][2] and priors['outflow_fwhm'][1] < outflow_fwhm <priors['outflow_fwhm'][2] \
                    and priors['NII_out_peak'][1] < np.log10(NII_out_peak) < priors['NII_out_peak'][2] and priors['outflow_vel'][1] < outflow_vel <priors['outflow_vel'][2]\
                        and 0.44<(SII_rpk/SII_bpk)<1.45:
                            return 0.0 
    
    return -np.inf

def log_probability_Halpha_outflow(theta, x, y, yerr, priors):
    lp = log_prior_Halpha_outflow(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_Halpha_outflow(theta, x, y, yerr)



  
    
# =============================================================================
#    functions to fit [OIII] only with outflow
# =============================================================================
def OIII_outflow(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm, Hbeta_vel):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4861.*(1+z)/1e4 
    
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482
    
    out_vel_wv = out_vel/3e5*OIIIr
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)
    
    Hbeta_fwhm = Hbeta_fwhm/3e5*Hbeta/2.35482
    
    Hbeta_wv = Hbeta + Hbeta_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta_wv, Hbeta_fwhm )
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + OIII_out + Hbeta_nar
    


def log_likelihood_OIII_outflow(theta, x, y, yerr):
    
    model = OIII_outflow(x,*theta)
    sigma2 = yerr*yerr
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_outflow(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm, Hbeta_vel, = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]:
                    return 0.0 
    
    return -np.inf

def log_probability_OIII_outflow(theta, x, y, yerr, priors):
    lp = log_prior_OIII_outflow(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_outflow(theta, x, y, yerr)  



# =============================================================================
#    functions to fit [OIII] only with outflow with nar Hbeta
# =============================================================================
def OIII_outflow_narHb(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm,Hbetab_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4861.*(1+z)/1e4 
    
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482
    
    out_vel_wv = out_vel/3e5*OIIIr
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)
    
    Hbetab_fwhm = Hbetab_fwhm/3e5*Hbeta/2.35482
    Hbetab_wv = Hbeta + Hbetab_vel/3e5*Hbeta
    Hbetab_blr = gauss(x, Hbetab_peak, Hbetab_wv, Hbetab_fwhm )
     
    Hbetan_fwhm = Hbetan_fwhm/3e5*Hbeta/2.35482
    Hbetan_wv = Hbeta + Hbetan_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbetan_peak, Hbetan_wv, Hbetan_fwhm )
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + OIII_out + Hbetab_blr + Hbeta_nar
    


def log_likelihood_OIII_outflow_narHb(theta, x, y, yerr):
    
    model = OIII_outflow_narHb(x,*theta)
    sigma2 = yerr*yerr
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_outflow_narHb(theta,priors):
    #zguess = np.loadtxt('zguess.txt')
    
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel, = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                    and  priors['Hbetan_peak'][1] < np.log10(Hbetan_peak)<priors['Hbetan_peak'][2] and priors['Hbetan_fwhm'][1]<Hbetan_fwhm<priors['Hbetan_fwhm'][2] and  priors['Hbetan_vel'][1]<Hbetan_vel<priors['Hbetan_vel'][2]:
                        return 0.0 
    
    return -np.inf

def log_probability_OIII_outflow_narHb(theta, x, y, yerr, priors):
    lp = log_prior_OIII_outflow_narHb(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_outflow_narHb(theta, x, y, yerr)  
    

# =============================================================================
#  Function to fit [OIII] without outflow with hbeta
# =============================================================================
def OIII(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak, Hbeta_fwhm, Hbeta_vel):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4861.*(1+z)/1e4 
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    
    Hbeta_fwhm = Hbeta_fwhm/3e5*Hbeta/2.35482
    
    Hbeta_wv = Hbeta + Hbeta_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta_wv, Hbeta_fwhm )
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_nar
    


def log_likelihood_OIII(theta, x, y, yerr):
    
    model = OIII(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm, Hbeta_vel = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]:
                return 0.0 
    
    return -np.inf

def log_probability_OIII(theta, x, y, yerr,priors):
    lp = log_prior_OIII(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII(theta, x, y, yerr)  
    

# =============================================================================
#  Function to fit [OIII] without outflow with dual hbeta
# =============================================================================
def OIII_dual_hbeta(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4861.*(1+z)/1e4 
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    
    Hbetab_fwhm = Hbetab_fwhm/3e5*Hbeta/2.35482
    Hbetab_wv = Hbeta + Hbetab_vel/3e5*Hbeta
    Hbeta_blr = gauss(x, Hbetab_peak, Hbetab_wv, Hbetab_fwhm )
    
    Hbetan_fwhm = Hbetan_fwhm/3e5*Hbeta/2.35482
    Hbetan_wv = Hbeta + Hbetan_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbetan_peak, Hbetan_wv, Hbetan_fwhm )
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_blr + Hbeta_nar
    


def log_likelihood_OIII_dual_hbeta(theta, x, y, yerr):
    
    model = OIII_dual_hbeta(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_dual_hbeta(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel= theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                and priors['Hbetan_peak'][1] < np.log10(Hbetan_peak)< priors['Hbetan_peak'][2] and  priors['Hbetan_fwhm'][1]<Hbetan_fwhm<priors['Hbetan_fwhm'][2] and  priors['Hbetan_vel'][1]<Hbetan_vel<priors['Hbetan_vel'][2]:
                    return 0.0 
    
    return -np.inf

def log_probability_OIII_dual_hbeta(theta, x, y, yerr,priors):
    lp = log_prior_OIII_dual_hbeta(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_dual_hbeta(theta, x, y, yerr)  
    
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

'''
def FeII_Veron(wave,z, FWHM_feii):
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008/2.35)

    convolved = convolve(Veron_d, gk)
    convolved = convolved/max(convolved[(Veron_wv<5400) &(Veron_wv>4900)])

    fce = interp1d(Veron_wv*(1+z)/1e4, convolved , kind='cubic')
    
    return fce(wave)

def FeII_Tsuzuki(wave,z, FWHM_feii):
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008/2.35)

    convolved = convolve(Tsuzuki_d, gk)
    convolved = convolved/max(convolved[(Tsuzuki_wv<5400) &(Tsuzuki_wv>4900)])

    fce = interp1d(Tsuzuki_wv*(1+z)/1e4, convolved , kind='cubic')
    
    return fce(wave)

def FeII_BG92(wave,z, FWHM_feii):
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008/2.35)

    convolved = convolve(BG92_d, gk)
    convolved = convolved/max(convolved[(BG92_wv<5400) &(BG92_wv>4900)])
    
    fce = interp1d(BG92_wv*(1+z)/1e4, convolved , kind='cubic')
    return fce(wave)

'''

# =============================================================================
#    functions to fit [OIII] only with outflow
# =============================================================================
def OIII_outflow_Fe(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm,Hbetab_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4861.*(1+z)/1e4 
    
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482
    
    out_vel_wv = out_vel/3e5*OIIIr
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)
    
    Hbetab_fwhm = Hbetab_fwhm/3e5*Hbeta/2.35482
    Hbetab_wv = Hbeta + Hbetab_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbetab_peak, Hbetab_wv, Hbetab_fwhm )
    
    
    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron
    
    FeII = FeII_peak*FeII_fce(x, z, FeII_fwhm)
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + OIII_out + Hbeta_nar + FeII
    


def log_likelihood_OIII_outflow_Fe(theta, x, y, yerr, template):
    
    model = OIII_outflow_Fe(x,*theta, template)
    sigma2 = yerr*yerr
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_outflow_Fe(theta,priors):
    #zguess = np.loadtxt('zguess.txt')
    
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, FeII_peak, FeII_fwhm = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2]and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                    and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                        return 0.0 
    
    return -np.inf

def log_probability_OIII_outflow_Fe(theta, x, y, yerr, priors,template):
    lp = log_prior_OIII_outflow_Fe(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_outflow_Fe(theta, x, y, yerr, template)  
    
# =============================================================================
#    functions to fit [OIII] only with outflow with nar Hbeta with Fe
# =============================================================================
def OIII_outflow_Fe_narHb(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm,Hbetan_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4861.*(1+z)/1e4 
    
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    Out_fwhm = OIII_out/3e5*OIIIr/2.35482
    
    out_vel_wv = out_vel/3e5*OIIIr
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    OIII_out = gauss(x, OIIIw_peak, OIIIr+out_vel_wv, Out_fwhm) + gauss(x, OIIIw_peak/3, OIIIb+out_vel_wv, Out_fwhm)
    
    Hbetab_fwhm = Hbetab_fwhm/3e5*Hbeta/2.35482
    Hbetab_wv = Hbeta + Hbetab_vel/3e5*Hbeta
    Hbeta_blr = gauss(x, Hbetab_peak, Hbetab_wv, Hbetab_fwhm )
      
    Hbetan_fwhm = Hbetan_fwhm/3e5*Hbeta/2.35482
    Hbetan_wv = Hbeta + Hbetan_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbetan_peak, Hbetan_wv, Hbetan_fwhm )
    
    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron
    
    FeII = FeII_peak*FeII_fce(x, z, FeII_fwhm)
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    
    return contm+ OIII_nar + OIII_out + Hbeta_blr + Hbeta_nar+ FeII
    


def log_likelihood_OIII_outflow_Fe_narHb(theta, x, y, yerr, template):
    
    model = OIII_outflow_Fe_narHb(x,*theta, template)
    sigma2 = yerr*yerr
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_outflow_Fe_narHb(theta,priors):
    #zguess = np.loadtxt('zguess.txt')
    
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel, FeII_peak, FeII_fwhm = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['OIIIw_peak'][1] < np.log10(OIIIw_peak) < priors['OIIIw_peak'][2] and priors['OIII_out'][1] < OIII_out <priors['OIII_out'][2]  and priors['out_vel'][1]<out_vel< priors['out_vel'][2] \
                and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                    and  priors['Hbetan_peak'][1] < np.log10(Hbetan_peak)<priors['Hbetan_peak'][2] and priors['Hbetan_fwhm'][1]<Hbetan_fwhm<priors['Hbetan_fwhm'][2] and  priors['Hbetan_vel'][1]<Hbetan_vel<priors['Hbetan_vel'][2]\
                        and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                          return 0.0 
                    
    
    return -np.inf

def log_probability_OIII_outflow_Fe_narHb(theta, x, y, yerr, priors,template):
    lp = log_prior_OIII_outflow_Fe_narHb(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_outflow_Fe_narHb(theta, x, y, yerr, template)  
        

# =============================================================================
#  Function to fit [OIII] without outflow with Fe
# =============================================================================
def OIII_Fe(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4861.*(1+z)/1e4 
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    
    Hbeta_fwhm = Hbeta_fwhm/3e5*Hbeta/2.35482
    
    Hbeta_nar = gauss(x,Hbeta_peak, Hbeta, Hbeta_fwhm)
    
    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron
    
    FeII = FeII_peak*FeII_fce(x, z, FeII_fwhm)
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    
    return contm+ OIII_nar + Hbeta_nar + FeII
    


def log_likelihood_OIII_Fe(theta, x, y, yerr, template):
    
    model = OIII_Fe(x,*theta, template)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_Fe(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm, Hbeta_vel, FeII_peak, FeII_fwhm = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] \
            and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2] \
                and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] \
                    and priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                        and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                          return 0.0 
    
    return -np.inf

def log_probability_OIII_Fe(theta, x, y, yerr,priors, template):
    lp = log_prior_OIII_Fe(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_Fe(theta, x, y, yerr, template)  



# =============================================================================
#  Function to fit [OIII] without outflow with dual hbeta and FeII
# =============================================================================
def OIII_dual_hbeta_Fe(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm,Hbetan_vel,FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4861.*(1+z)/1e4 
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    
    Hbetab_fwhm = Hbetab_fwhm/3e5*Hbeta/2.35482
    Hbetab_wv = Hbeta + Hbetab_vel/3e5*Hbeta
    Hbeta_blr = gauss(x, Hbetab_peak, Hbetab_wv, Hbetab_fwhm )
      
    Hbetan_fwhm = Hbetan_fwhm/3e5*Hbeta/2.35482
    Hbetan_wv = Hbeta + Hbetan_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbetan_peak, Hbetan_wv, Hbetan_fwhm )
    
    if template=='BG92':
        FeII_fce = FeII_BG92
    if template=='Tsuzuki':
        FeII_fce = FeII_Tsuzuki
    if template=='Veron':
        FeII_fce = FeII_Veron
    
    FeII = FeII_peak*FeII_fce(x, z, FeII_fwhm)
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_blr + Hbeta_nar+ FeII
    


def log_likelihood_OIII_dual_hbeta_Fe(theta, x, y, yerr, template):
    
    model = OIII_dual_hbeta_Fe(x,*theta, template)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_OIII_dual_hbeta_Fe(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel, FeII_peak, FeII_fwhm = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                and priors['Hbetan_peak'][1] < np.log10(Hbetan_peak)< priors['Hbetan_peak'][2] and  priors['Hbetan_fwhm'][1]<Hbetan_fwhm<priors['Hbetan_fwhm'][2]and  priors['Hbetan_vel'][1]<Hbetan_vel<priors['Hbetan_vel'][2]\
                    and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                        return 0.0 
    
    return -np.inf

def log_probability_OIII_dual_hbeta_Fe(theta, x, y, yerr,priors, template):
    lp = log_prior_OIII_dual_hbeta_Fe(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_OIII_dual_hbeta_Fe(theta, x, y, yerr, template)  


# =============================================================================
#  Primary function to fit Halpha both with or without BLR - data prep and fit 
# =============================================================================
def fitting_Halpha(wave, fluxs, error,z, BLR=1,zcont=0.05, progress=True ,priors= {'cont':[0,-3,1],\
                                                                                   'cont_grad':[0,-0.01,0.01], \
                                                                                   'Hal_peak':[0,-3,1],\
                                                                                   'BLR_peak':[0,-3,1],\
                                                                                   'NII_peak':[0,-3,1],\
                                                                                   'Nar_fwhm':[300,100,900],\
                                                                                   'BLR_fwhm':[4000,2000,9000],\
                                                                                   'BLR_offset':[-200,-900,600],\
                                                                                    'SII_rpk':[0,-3,1],\
                                                                                    'SII_bpk':[0,-3,1],\
                                                                                    'Hal_out_peak':[0,-3,1],\
                                                                                    'NII_out_peak':[0,-3,1],\
                                                                                    'outflow_fwhm':[600,300,1500],\
                                                                                    'outflow_vel':[-50, -300,300]}):
    
    priors['z'] = [z, z-zcont, z+zcont]
    fluxs[np.isnan(fluxs)] = 0
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>(6562.8-170)*(1+z)/1e4)&(wave<(6562.8+170)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    znew = wave_zoom[peak_loc]/0.6562-1
    if abs(znew-z)<zcont:
        z= znew
    peak = np.ma.max(flux_zoom)
    nwalkers=32
    
    if BLR==1:
        pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/4, peak/4, priors['Nar_fwhm'][0], priors['BLR_fwhm'][0],priors['BLR_offset'][0],peak/6, peak/6])
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
    
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_Halpha_BLR, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
            
        labels=('z', 'cont','cont_grad', 'Hal_peak','BLR_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'BLR_offset', 'SIIr_peak', 'SIIb_peak')

        
        
        
        fitted_model = Halpha_wBLR
        
        res = {'name': 'Halpha_wth_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    
    if BLR==0:
        
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4,priors['Nar_fwhm'][0],peak/6, peak/6 ])
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_Halpha, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
    
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak')
        
        fitted_model = Halpha
        
        res = {'name': 'Halpha_wth_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
            
    if BLR==-1:
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4, priors['Nar_fwhm'][0],peak/6, peak/6,peak/8, peak/8, priors['outflow_fwhm'][0],priors['outflow_vel'][0] ])
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
    
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_Halpha_outflow, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
    
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel')
        
        fitted_model = Halpha_outflow
        
        res = {'name': 'Halpha_wth_out'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    
            
        
    return res, fitted_model
    
# =============================================================================
# Primary function to fit [OIII] with and without outflows. 
# =============================================================================
    
def fitting_OIII(wave, fluxs, error,z, outflow=0, template=0, Hbeta_dual=0, progress=True, \
                                                                 priors= {'cont':[0,-3,1],\
                                                                'cont_grad':[0,-0.01,0.01], \
                                                                'OIIIn_peak':[0,-3,1],\
                                                                'OIIIw_peak':[0,-3,1],\
                                                                'OIII_fwhm':[300,100,900],\
                                                                'OIII_out':[700,600,2500],\
                                                                'out_vel':[-200,-900,600],\
                                                                'Hbeta_peak':[0,-3,1],\
                                                                'Hbeta_fwhm':[200,120,7000],\
                                                                'Hbeta_vel':[10,-200,200],\
                                                                'Hbetan_peak':[0,-3,1],\
                                                                'Hbetan_fwhm':[300,120,700],\
                                                                'Hbetan_vel':[10,-100,100],\
                                                                'Fe_peak':[0,-3,2],\
                                                                'Fe_fwhm':[3000,2000,6000]}):
    
    
    
    priors['z'] = [z, z-0.05, z+0.05]
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    
    sel=  np.where((wave<5025*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.argmax(flux_zoom)
    peak = (np.max(flux_zoom))
    
    selb =  np.where((wave<4880*(1+z)/1e4)& (wave>4820*(1+z)/1e4))[0]
    flux_zoomb = flux[selb]
    wave_zoomb = wave[selb]
    try:
        peak_loc_beta = np.argmax(flux_zoomb)
        peak_beta = (np.max(flux_zoomb))
    except:
        peak_beta = peak/3
    
    
    nwalkers=32
    if outflow==1: 
        if template==0:
            if Hbeta_dual == 0:
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0]])
                
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
               
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability_OIII_outflow, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
                
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel')
                
                fitted_model = OIII_outflow
                
                res = {'name': 'OIII_outflow'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
            else:
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0],\
                                peak_beta, priors['Hbeta_fwhm'][0], priors['Hbeta_vel'][0],peak_beta, priors['Hbetan_fwhm'][0], priors['Hbetan_vel'][0]])
                
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_outflow_narHb, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                    
                labels= ('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel')
                
                fitted_model = OIII_outflow_narHb
                
                res = {'name': 'OIII_outflow_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
         
        else:
            if Hbeta_dual == 0:
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                np.median(flux[fit_loc]), priors['Fe_fwhm'][0]])
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
               
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_outflow_Fe, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, template))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                
                fitted_model = OIII_outflow_Fe
                
                res = {'name': 'OIII_outflow_Fe'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
            else:
                pos_l = np.array([z,np.median(flux[fit_loc])/2,0.001, peak/2, peak/4, 300., 600.,-100, \
                                peak_beta/2, 4000,priors['Hbeta_vel'][0],peak_beta/2, 600,priors['Hbetan_vel'][0],\
                                np.median(flux[fit_loc]), 2000])
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_outflow_Fe_narHb, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, template))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                    
                labels= ('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel', 'Fe_peak', 'Fe_fwhm')
                
                fitted_model = OIII_outflow_Fe_narHb
                
                res = {'name': 'OIII_outflow_Fe_narHb'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
            
    if outflow==0: 
        if template==0:
            if Hbeta_dual == 0:
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0]]) 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel')
                
                
                fitted_model = OIII
                
                res = {'name': 'OIII'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
        
            else:
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta/4, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                peak_beta/4, priors['Hbetan_fwhm'][0], priors['Hbetan_vel'][0]])
                
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_dual_hbeta, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
            
                sampler.run_mcmc(pos, N, progress=progress);
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel')
                
                fitted_model = OIII_dual_hbeta
                res = {'name': 'OIII_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
         
        else:
            if Hbeta_dual == 0:
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],np.median(flux[fit_loc]), priors['Fe_fwhm'][0]]) 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_Fe, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, template))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                
                
                fitted_model = OIII_Fe
                
                res = {'name': 'OIII_Fe'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                
                    
            else:
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta/2, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                peak_beta/2, priors['Hbetan_fwhm'][0],priors['Hbetan_vel'][0], \
                                np.median(flux[fit_loc]), priors['Fe_fwhm'][0]]) 
                    
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_dual_hbeta_Fe, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, template))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel','Fe_peak', 'Fe_fwhm')
                
                
                fitted_model = OIII_dual_hbeta_Fe
                
                res = {'name': 'OIII_Fe_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
        
    return res, fitted_model


# =============================================================================
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

def log_likelihood_Halpha_OIII(theta, x, y, yerr): 
    model = Halpha_OIII(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_probability_Halpha_OIII(theta, x, y, yerr,priors):
    lp = log_prior_Halpha_OIII(theta,priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_Halpha_OIII(theta, x, y, yerr)  

def fitting_Halpha_OIII(wave, fluxs, error,z,zcont=0.01, progress=True, priors= {'cont':[0,-3,1],\
                                                                'cont_grad':[0,-10.,10], \
                                                                'OIIIn_peak':[0,-3,1],\
                                                                'OIIIn_fwhm':[300,100,900],\
                                                                'OIII_vel':[-100,-600,600],\
                                                                'Hbeta_peak':[0,-3,1],\
                                                                'Hal_peak':[0,-3,1],\
                                                                'NII_peak':[0,-3,1],\
                                                                'Nar_fwhm':[300,150,900],\
                                                                'SII_rpk':[0,-3,1],\
                                                                'SII_bpk':[0,-3,1],\
                                                                'OI_peak':[0,-3,1]}):
    priors['z'] = [z, z-zcont, z+zcont]
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    fit_loc = np.append(fit_loc, np.where((wave>(6300-50)*(1+z)/1e4)&(wave<(6300+50)*(1+z)/1e4))[0])
    fit_loc = np.append(fit_loc, np.where((wave>(6562.8-170)*(1+z)/1e4)&(wave<(6562.8+170)*(1+z)/1e4))[0])
    
# =============================================================================
#     Finding the initial conditions
# =============================================================================
    sel=  np.where((wave<5025*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc_OIII = np.argmax(flux_zoom)
    peak_OIII = (np.max(flux_zoom))
    
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    znew = wave_zoom[peak_loc]/0.6562-1
    if abs(znew-z)<zcont:
        z= znew
    peak_hal = np.ma.max(flux_zoom)
    
# =============================================================================
#   Setting up fitting  
# =============================================================================
    nwalkers=32
    pos_l = np.array([z,np.median(flux[fit_loc]), -1, peak_hal*0.7, peak_hal*0.3, priors['Nar_fwhm'][0], peak_hal*0.2, peak_hal*0.2, peak_OIII*0.8,  priors['OIIIn_fwhm'][0], peak_hal*0.2, priors['OIII_vel'][0], peak_hal*0.3])
    
    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
    pos[:,0] = np.random.normal(z,0.001, nwalkers)
   
    nwalkers, ndim = pos.shape
    
    sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_Halpha_OIII, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
    
    sampler.run_mcmc(pos, N, progress=progress);
    
    flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)

        
    labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SII_rpk', 'SII_bpk', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'OIII_vel', 'OI_peak')
    
    fitted_model = Halpha_OIII
    
    res = {'name': 'Halpha_OIII'}
    for i in range(len(labels)):
        res[labels[i]] = flat_samples[:,i]
        
    return res, fitted_model


def Fitting_OIII_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    with open('/Users/jansen/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    
    flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, outflow=0, progress=False, priors=priors)
    cube_res  = [i,j,prop_calc(flat_samples_sig)]
    return cube_res

def Fitting_Halpha_OIII_unwrap(lst, progress=False):
    with open('/Users/jansen/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    print(i)
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    
    flat_samples_sig, fitted_model_sig = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors)
    cube_res  = [i,j,prop_calc(flat_samples_sig), flat_samples_sig]
    return cube_res

def Fitting_OIII_2G_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, outflow=1, progress=False)
    cube_res  = [i,j,prop_calc(flat_samples_sig)]
    
    return cube_res

import time

def Fitting_Halpha_unwrap(lst): 
    
    with open('/Users/jansen/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    print(priors)  
    i,j,flx_spax_m, error, wave, z = lst
    print(i)
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    flat_samples_sig, fitted_model_sig = fitting_Halpha(wave,flx_spax_m,error,z, zcont=deltaz, BLR=0, progress=False, priors=priors)
    cube_res  = [i,j,prop_calc(flat_samples_sig)]
    
    return cube_res
    
    
def prop_calc(results):  
    labels = list(results.keys())[1:]
    res_plt = []
    res_dict = {'name': results['name']}
    for lbl in labels:
        
        array = results[lbl]
        
        p50,p16,p84 = np.percentile(array, (50,16,84))
        p16 = p50-p16
        p84 = p84-p50
        
        res_plt.append(p50)
        res_dict[lbl] = np.array([p50,p16,p84])
        
    res_dict['popt'] = res_plt
    return res_dict


# =============================================================================
# Fit single Gaussian
# =============================================================================
def Single_gauss(x, z, cont,cont_grad,  Hal_peak, NII_peak, Nar_fwhm, SII_rpk, SII_bpk):
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
    
    return cont+x*cont_grad+Hal_nar+NII_nar_r+NII_nar_b + SII_rg + SII_bg

def log_likelihood_single(theta, x, y, yerr):
    
    model = Halpha(x,*theta)
    sigma2 = yerr*yerr#yerr ** 2 + model ** 2 #* np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))


def log_prior_single(theta, zguess, zcont):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk = theta
    if (zguess-zcont) < z < (zguess+zcont) and -3 < np.log10(cont)<1 and -3<np.log10(Hal_peak)<1 and -3<np.log10(NII_peak)<1 \
        and 150 < Nar_fwhm<900 and -0.01<cont_grad<0.01 and 0<SII_bpk<0.5 and 0<SII_rpk<0.5:
            return 0.0
    
    return -np.inf

def log_probability_single(theta, x, y, yerr, zguess, zcont=0.05):
    lp = log_prior_Halpha(theta,zguess,zcont)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_Halpha(theta, x, y, yerr)
