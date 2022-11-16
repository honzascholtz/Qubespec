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
def log_prior_Halpha(theta, priors):
    z, cont,cont_grad, Hal_peak, NII_peak, Nar_fwhm,  SII_rpk, SII_bpk = theta
    zcont=0.01
    
    logpriors = np.zeros_like(theta)
    
    logpriors[0] = norm.logpdf(z,priors['z'][0],zcont)
    logpriors[1] = uniform.logpdf(np.log10(cont), -3,5)
    logpriors[2] = uniform.logpdf(np.log10(Hal_peak), -3,5)
    logpriors[3] = uniform.logpdf(np.log10(NII_peak), -3,5 )
    logpriors[4] = uniform.logpdf(Nar_fwhm, 100, 900 )
    logpriors[5] = norm.logpdf(cont_grad, 0, 0.1)
    logpriors[6] = uniform.logpdf(np.log10(SII_bpk),-3,53)
    logpriors[7] = uniform.logpdf(np.log10(SII_rpk), -3,5)
    
    logprior = np.einsum('i',logpriors)
    
    return logprior
'''
def log_probability_Halpha(theta, x, y, yerr, priors):
    lp = log_prior_Halpha(theta,priors)
    #if not np.isfinite(lp):
    #    return -np.inf
    #lp[~np.isfinite(lp)] = -np.inf
    
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