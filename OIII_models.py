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

import Fe_path as fpt
PATH_TO_FeII = fpt.PATH_TO_FeII

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
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4862.6*(1+z)/1e4 
    
    
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

def log_prior_OIII_outflow(theta,priors):
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm, Hbeta_vel, = theta
    
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
#    functions to fit [OIII] only with outflow with nar Hbeta
# =============================================================================
def OIII_outflow_narHb(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm,Hbetab_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    Hbeta = 4862.6*(1+z)/1e4 
    
    
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

def log_prior_OIII_outflow_narHb(theta,priors):
    #zguess = np.loadtxt('zguess.txt')
    
    z, cont, cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel, = theta
    
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
#  Function to fit [OIII] without outflow with hbeta
# =============================================================================
def OIII(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak, Hbeta_fwhm, Hbeta_vel):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4862.6*(1+z)/1e4 
    
    Nar_fwhm = OIII_fwhm/3e5*OIIIr/2.35482
    
    OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
    
    Hbeta_fwhm = Hbeta_fwhm/3e5*Hbeta/2.35482
    
    Hbeta_wv = Hbeta + Hbeta_vel/3e5*Hbeta
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta_wv, Hbeta_fwhm )
    
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_nar
    
def log_prior_OIII(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm, Hbeta_vel = theta
    
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
#  Function to fit [OIII] without outflow with dual hbeta
# =============================================================================
def OIII_dual_hbeta(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = OIIIr- (48.*(1+z)/1e4)
    
    Hbeta = 4862.6*(1+z)/1e4 
    
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

def log_prior_OIII_dual_hbeta(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel= theta
    
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
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008.24/2.35)

    convolved = convolve(Veron_d, gk)
    convolved = convolved/max(convolved[(Veron_wv<5400) &(Veron_wv>4900)])

    fce = interp1d(Veron_wv*(1+z)/1e4, convolved , kind='cubic')
    
    return fce(wave)

def FeII_Tsuzuki(wave,z, FWHM_feii):
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008.24/2.35)

    convolved = convolve(Tsuzuki_d, gk)
    convolved = convolved/max(convolved[(Tsuzuki_wv<5400) &(Tsuzuki_wv>4900)])

    fce = interp1d(Tsuzuki_wv*(1+z)/1e4, convolved , kind='cubic')
    
    return fce(wave)

def FeII_BG92(wave,z, FWHM_feii):
    gk = Gaussian1DKernel(stddev=FWHM_feii/3e5*5008.24/2.35)

    convolved = convolve(BG92_d, gk)
    convolved = convolved/max(convolved[(BG92_wv<5400) &(BG92_wv>4900)])
    
    fce = interp1d(BG92_wv*(1+z)/1e4, convolved , kind='cubic')
    return fce(wave)

'''

# =============================================================================
#    functions to fit [OIII] only with outflow
# =============================================================================
def OIII_outflow_Fe(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm,Hbetab_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4 
    
    
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

# =============================================================================
#    functions to fit [OIII] only with outflow with nar Hbeta with Fe
# =============================================================================
def OIII_outflow_Fe_narHb(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm,Hbetan_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = 4960.3*(1+z)/1e4
    Hbeta = 4862.6*(1+z)/1e4 
    
    
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

# =============================================================================
#  Function to fit [OIII] without outflow with Fe
# =============================================================================
def OIII_Fe(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = 4960.3*(1+z)/1e4
    
    Hbeta = 4862.6*(1+z)/1e4 
    
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

# =============================================================================
#  Function to fit [OIII] without outflow with dual hbeta and FeII
# =============================================================================
def OIII_dual_hbeta_Fe(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbetab_peak, Hbetab_fwhm, Hbetab_vel, Hbetan_peak, Hbetan_fwhm,Hbetan_vel,FeII_peak, FeII_fwhm, template):
    OIIIr = 5008.24*(1+z)/1e4   
    OIIIb = 4960.3*(1+z)/1e4
    
    Hbeta = 4862.6*(1+z)/1e4 
    
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
    
def log_prior_OIII_dual_hbeta_Fe(theta,priors):
    
    z, cont, cont_grad, OIIIn_peak, OIII_fwhm, Hbeta_peak, Hbeta_fwhm,Hbeta_vel, Hbetan_peak, Hbetan_fwhm, Hbetan_vel, FeII_peak, FeII_fwhm = theta
    
    if priors['z'][1] < z < priors['z'][2] and priors['cont'][1] < np.log10(cont)<priors['cont'][2]  and priors['cont_grad'][1]< cont_grad<priors['cont_grad'][2]  \
        and priors['OIIIn_peak'][1] < np.log10(OIIIn_peak) < priors['OIIIn_peak'][2] and priors['OIII_fwhm'][1] < OIII_fwhm <priors['OIII_fwhm'][2]\
            and priors['Hbeta_peak'][1] < np.log10(Hbeta_peak)< priors['Hbeta_peak'][2] and  priors['Hbeta_fwhm'][1]<Hbeta_fwhm<priors['Hbeta_fwhm'][2] and  priors['Hbeta_vel'][1]<Hbeta_vel<priors['Hbeta_vel'][2]\
                and priors['Hbetan_peak'][1] < np.log10(Hbetan_peak)< priors['Hbetan_peak'][2] and  priors['Hbetan_fwhm'][1]<Hbetan_fwhm<priors['Hbetan_fwhm'][2]and  priors['Hbetan_vel'][1]<Hbetan_vel<priors['Hbetan_vel'][2]\
                    and priors['Fe_fwhm'][1]<FeII_fwhm<priors['Fe_fwhm'][2] and priors['Fe_peak'][1] < np.log10(FeII_peak)<priors['Fe_peak'][2]:
                        return 0.0 
    
    return -np.inf