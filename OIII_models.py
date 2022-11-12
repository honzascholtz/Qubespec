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
