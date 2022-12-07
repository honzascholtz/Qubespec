#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:36:26 2017

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import wcs
from astropy.nddata import Cutout2D

from matplotlib.backends.backend_pdf import PdfPages
import pickle
import emcee
import corner
import tqdm
import os

from astropy import stats

import multiprocessing as mp
from multiprocessing import Pool
from astropy.modeling.powerlaws import PowerLaw1D

import Plotting_tools_v2 as emplot
import Fitting_tools_mcmc as emfit
from brokenaxes import brokenaxes

from astropy.utils.exceptions import AstropyWarning
import astropy.constants, astropy.cosmology, astropy.units, astropy.wcs

def switch(read_module):
    emfit = __import__(read_module)
    emplot = __import__('Plotting_tools_v2_KASHz')
    return emfit, emplot
    
def test():
    print(emfit)
    print(emfit.version)


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


OIIIr = 5008.24
OIIIb = 4960.3
Hal = 6564.52
NII_r = 6585.27
NII_b = 6549.86
Hbe = 4862.6

SII_r = 6731
SII_b = 6718.29
import time


from Halpha_OIII_models import *
# =============================================================================
# Useful function 
# =============================================================================

def gauss(x,k,mu,sig):
    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

def find_nearest(array, value):
    """ Find the location of an array closest to a value 
	
	"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_circular_mask(h, w, center=None, radius=None):
    """ Creates a circular mask input - size of the array (height, width), optional center of the circular
    aperture and the radius in pixels
	
	"""

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return mask

def error_calc(array):
    """ calculates, 50th, 16th and 84 percintile of an array
	
	"""
    p50,p16,p84 = np.percentile(array, (50,16,84))
    p16 = p50-p16
    p84 = p84-p50
    
    return p50, p16,p84

def conf(aray):
    """ old version of finding 16th and 84th percintile
	
	"""
    
    sorted_array= np.array(sorted(aray))
    leng= (float(len(aray))/100)*16
    leng= int(leng)
    
    
    hgh = sorted_array[-leng]
    low = sorted_array[leng]
    
    return low, hgh

def twoD_Gaussian(dm, amplitude, xo, yo, sigma_x, sigma_y, theta, offset): 
    """ 2D Gaussian array used to find center of emission 
	
	"""
    x, y = dm
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def Av_calc(Falpha, Fbeta):
    """ Calculating Av based on Halpha and Hbeta emission 
	
	"""
    
    Av = 1.964*4.12*np.log10(Falpha/Fbeta/2.86)
    
    return Av

def Flux_cor(Flux, Av, lam= 0.6563):
    """ Correcting a line flux based on Av
	
	"""
    
    Ebv = Av/4.12
    Ahal = 3.325*Ebv
    
    F = Flux*10**(0.4*Ahal)
    
    return F

def smooth(image,sm):
    """ Gaussian 2D smoothning for maps
	
	"""
    from astropy.convolution import Gaussian2DKernel
    from scipy.signal import convolve as scipy_convolve
    from astropy.convolution import convolve
    
    gauss_kernel = Gaussian2DKernel(sm)

    con_im = convolve(image, gauss_kernel)
    
    #con_im = con_im#*image/image
    
    return con_im  

def prop_calc(results): 
    """ Take the dictionary with the results chains and calculates the values 
    and 1 sigma confidence interval
	
	"""
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
        

def SNR_calc(wave,flux, error, dictsol, mode):
    """ Calculates the SNR of a line
    wave - observed wavelength
    flux - flux of the spectrum
    error - error on the spectrum
    dictsol - spectral fitting results in dictionary form 
    mode - which emission line do you want to caluclate the SNR for: OIII, Hn, Hblr, NII,
           Hb, SII
	
	"""
    sol = dictsol['popt']
    wave = wave[np.invert(flux.mask)]
    flux = flux.data[np.invert(flux.mask)]
    
    wv_or = wave.copy()
    keys = list(dictsol.keys())
    if mode =='OIII':
        center = OIIIr*(1+sol[0])/1e4
        if 'OIIIw_fwhm' in keys:
            fwhms = sol[5]/3e5*center
            fwhm = dictsol['OIIIw_fwhm'][0]/3e5*center
            
            center = OIIIr*(1+sol[0])/1e4
            centerw = OIIIr*(1+sol[0])/1e4 + sol[7]/3e5*center
            
            contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
            model = flux-contfce 
        elif 'Nar_fwhm' in keys:
            fwhm = dictsol['Nar_fwhm'][0]/3e5*center
            
            center = OIIIr*(1+sol[0])/1e4
            
            contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
            model = flux-contfce 
        else:   
            fwhm = sol[4]/3e5*center
            
            model = flux- PowerLaw1D.evaluate(wave,sol[1],center, alpha=sol[2])
            
            
    elif mode =='Hn':
        center = Hal*(1+sol[0])/1e4
        
        fwhm = dictsol['Nar_fwhm'][0]/3e5*center
        model = gauss(wave, dictsol['Hal_peak'][0], center, fwhm/2.35)
        
    
    elif mode =='Hblr':
        center = Hal*(1+sol[0])/1e4
        
        fwhm = sol[7]/3e5*center
        model = gauss(wave, sol[4], center, fwhm/2.35)
            
    elif mode =='NII':
        center = NII_r*(1+sol[0])/1e4
        fwhm = dictsol['Nar_fwhm'][0]/3e5*center
        model = gauss(wave, dictsol['NII_peak'][0], center, fwhm/2.35)
        
            
    
    elif mode =='Hb':
        center = Hbe*(1+sol[0])/1e4
        if 'Hbetan_fwhm' in keys:
            fwhm = dictsol['Hbetan_fwhm'][0]/3e5*center
            model = gauss(wave, dictsol['Hbetan_peak'][0], center, fwhm/2.35)
        elif 'Nar_fwhm' in keys:
            fwhm = dictsol['Nar_fwhm'][0]/3e5*center
            model = gauss(wave, dictsol['Hbeta_peak'][0], center, fwhm/2.35)
        else:
            fwhm = dictsol['Hbeta_fwhm'][0]/3e5*center
            model = gauss(wave, dictsol['Hbeta_peak'][0], center, fwhm/2.35)
    
    elif mode =='SII':
        center = SII_r*(1+sol[0])/1e4
        
        fwhm = dictsol['Nar_fwhm'][0]/3e5*center
        try:
            model_r = gauss(wave, dictsol['SIIr_peak'][0], center, fwhm/2.35) 
            model_b = gauss(wave, dictsol['SIIb_peak'][0], center, fwhm/2.35) 
        except:
            model_r = gauss(wave, dictsol['SIIr_peak'][0], center, fwhm/2.35) 
            model_b = gauss(wave, dictsol['SIIb_peak'][0], center, fwhm/2.35) 
        
        model = model_r + model_b
        
        center = 6724*(1+sol[0])/1e4
        
        use = np.where((wave< center+fwhm*1)&(wave> center-fwhm*1))[0]   
        flux_l = model[use]
        std = np.mean(error[use])
        
        n = len(use)
        SNR = (sum(flux_l)/np.sqrt(n)) * (1./std)
        
        if SNR < 0:
            SNR=0
        
        return SNR
    
    else:
        raise Exception('Sorry mode in SNR_calc not understood')
    
    use = np.where((wave< center+fwhm*1)&(wave> center-fwhm*1))[0] 
    flux_l = model[use]
    std = np.mean(error[use])
    
    n = len(use)
    SNR = (sum(flux_l)/np.sqrt(n)) * (1./std)
    if SNR < 0:
        SNR=0
    
    return SNR  

def BIC_calc(wave,fluxm,error, model, results, mode, template=0):
    """ calculates BIC
	
	"""
    popt = results['popt']
    z= popt[0]
    
    if mode=='OIII':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>4800*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        if template==0:
            y_model = model(wave, *popt)
        else:
            y_model = model(wave, *popt, template)
        chi2 = sum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    if mode=='Halpha':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>(6564.52-200)*(1+z)/1e4)&(wave<(6564.52+300)*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = sum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    if mode=='Halpha_OIII':
        fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
        fit_loc = np.append(fit_loc, np.where((wave>(6300-50)*(1+z)/1e4)&(wave<(6300+50)*(1+z)/1e4))[0])
        fit_loc = np.append(fit_loc, np.where((wave>(6564.52-170)*(1+z)/1e4)&(wave<(6564.52+170)*(1+z)/1e4))[0])
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>(6564.52-200)*(1+z)/1e4)&(wave<(6564.52+300)*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = sum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
        
        
    
    return chi2, BIC

def unwrap_chain(res):
    keys = list(res.keys())[1:]
    
    chains = np.zeros(( len(res[keys[0]]), len(keys) ))
    
    for i in range(len(keys)):
        
        chains[:,i] = res[keys[i]]
        
    return chains
    

def flux_calc(res, mode, norm=1e-13):
    keys = list(res.keys())
    if mode=='OIIIt':
        
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if 'OIIIw_peak' in keys:
            o3 = 5008.24*(1+res['z'][0])/1e4
            
            o3n = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
            o3w = gauss(wave, res['OIIIw_peak'][0], o3, res['OIIIw_fwhm'][0]/2.355/3e5*o3  )*1.333
            
            model = o3n+o3w
        if 'OIII_out_peak' in keys:
            o3 = 5008.24*(1+res['z'][0])/1e4
            
            o3n = gauss(wave, res['OIIIn_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
            o3w = gauss(wave, res['OIII_out_peak'][0], o3, res['outflow_fwhm'][0]/2.355/3e5*o3  )*1.333
            
            model = o3n+o3w
            
        else:
            o3 = 5008.24*(1+res['z'][0])/1e4
            try:
                model = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
            except:
                model = gauss(wave, res['OIIIn_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
            
    elif mode=='OIIIn':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        
        o3 = 5008.24*(1+res['z'][0])/1e4
        try:
            model = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
        except:
            model = gauss(wave, res['OIIIn_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
        
    elif mode=='OIIIw':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if 'OIIIw_peak' in keys:
            o3 = 5008.24*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIIIw_peak'][0], o3, res['OIIIw_fwhm'][0]/2.355/3e5*o3  )*1.333
        else:
            model = np.zeros_like(wave)
    
    elif mode=='Hat':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        hn = 6565*(1+res['z'][0])/1e4
        
        model = gauss(wave, res['Hal_peak'][0], hn, res['Nar_fwhm'][0]/2.355/3e5*hn  )
        
        if 'outflow_fwhm' in list(res.keys()):
            model = gauss(wave, res['Hal_peak'][0], hn, res['Nar_fwhm'][0]/2.355/3e5*hn  ) + \
                gauss(wave, res['Hal_out_peak'][0], hn, res['outflow_fwhm'][0]/2.355/3e5*hn  )
    
    elif mode=='Han':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        hn = 6565*(1+res['z'][0])/1e4
        
        model = gauss(wave, res['Hal_peak'][0], hn, res['Nar_fwhm'][0]/2.355/3e5*hn  )
        
        if 'outflow_fwhm' in list(res.keys()):
            model = gauss(wave, res['Hal_peak'][0], hn, res['Nar_fwhm'][0]/2.355/3e5*hn  )
    
    elif mode=='Hal_BLR':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        hn = 6565*(1+res['z'][0])/1e4
        if 'BLR_fwhm' in keys:
            model = gauss(wave, res['BLR_peak'][0], hn, res['BLR_fwhm'][0]/2.355/3e5*hn  )
            
        if 'BLR_alp1' in keys:
            from QSO_models import BKPLG
            model = BKPLG(wave, res['BLR_peak'][0], hn, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
            
        else:
            model = np.zeros_like(wave)
    
    elif mode=='NIIt':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        nii = 6583*(1+res['z'][0])/1e4
        model = gauss(wave, res['NII_peak'][0], nii, res['Nar_fwhm'][0]/2.355/3e5*nii  )*1.333
        
        if 'outflow_fwhm' in list(res.keys()):
            model = gauss(wave, res['NII_peak'][0], nii, res['Nar_fwhm'][0]/2.355/3e5*nii  ) + \
                gauss(wave, res['NII_out_peak'][0], nii, res['outflow_fwhm'][0]/2.355/3e5*nii  )
    
    elif mode=='NII':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        nii = 6583*(1+res['z'][0])/1e4
        model = gauss(wave, res['NII_peak'][0], nii, res['Nar_fwhm'][0]/2.355/3e5*nii  )*1.333
        
                
    elif mode=='NIIo':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        nii = 6583*(1+res['z'][0])/1e4
        model = gauss(wave, res['NII_out_peak'][0], nii, res['outflow_fwhm'][0]/2.355/3e5*nii  )*1.333
        
    elif mode=='Hbeta':
        wave = np.linspace(4800,4900,300)*(1+res['z'][0])/1e4
        hbeta = 4862.6*(1+res['z'][0])/1e4
        try:
            model = gauss(wave, res['Hbeta_peak'][0], hbeta, res['Hbeta_fwhm'][0]/2.355/3e5*hbeta  )
        except:
            model = gauss(wave, res['Hbeta_peak'][0], hbeta, res['Nar_fwhm'][0]/2.355/3e5*hbeta  )
    
    elif mode=='Hbe_BLR':
        wave = np.linspace(4800,4900,300)*(1+res['z'][0])/1e4
        hbeta = 4862.6*(1+res['z'][0])/1e4
        if 'BLR_alp1' in keys:
            from QSO_models import BKPLG
            model = BKPLG(wave, res['BLR_peak'][0], hbeta, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
    
    elif mode=='Hbetaw':
        wave = np.linspace(4800,4900,300)*(1+res['z'][0])/1e4
        hbeta = 4862.6*(1+res['z'][0])/1e4
        model = gauss(wave, res['Hbeta_peak'][0], hbeta, res['Hbeta_fwhm'][0]/2.355/3e5*hbeta  )
    
    elif mode=='Hbetan':
        wave = np.linspace(4800,4900,300)*(1+res['z'][0])/1e4
        hbeta = 4862.6*(1+res['z'][0])/1e4
        model = gauss(wave, res['Hbetan_peak'][0], hbeta, res['Hbetan_fwhm'][0]/2.355/3e5*hbeta  )
    
    elif mode=='OI':
        wave = np.linspace(6250,6350,300)*(1+res['z'][0])/1e4
        OI = 6302*(1+res['z'][0])/1e4
        model = gauss(wave, res['OI_peak'][0], OI, res['Nar_fwhm'][0]/2.355/3e5*OI  )    
    
    elif mode=='SIIr':
        SII_r = 6732.67*(1+res['z'][0])/1e4   
        
        
        wave = np.linspace(6600,6800,200)*(1+res['z'][0])/1e4
        try:  
            model_r = gauss(wave, res['SIIr_peak'][0], SII_r, res['Nar_fwhm'][0]/2.355/3e5*SII_r  )
        except:
            model_r = gauss(wave, res['SIIr_peak'][0], SII_r, res['Nar_fwhm'][0]/2.355/3e5*SII_r  )
        
        import scipy.integrate as scpi
            
        Flux_r = scpi.simps(model_r, wave)*norm
       
        return Flux_r
    
    elif mode=='SIIb':
        SII_b = 6718.29*(1+res['z'][0])/1e4   
        
        wave = np.linspace(6600,6800,200)*(1+res['z'][0])/1e4
        try:
            model_b = gauss(wave, res['SIIb_peak'][0], SII_b, res['Nar_fwhm'][0]/2.355/3e5*SII_b  )
        except:
            model_b = gauss(wave, res['SIIb_peak'][0], SII_b, res['Nar_fwhm'][0]/2.355/3e5*SII_b  )
        
        import scipy.integrate as scpi
            
        
        Flux_b = scpi.simps(model_b, wave)*norm
        
        return Flux_b
    
    else:
        raise Exception('Sorry mode in Flux not understood')
        
    import scipy.integrate as scpi
        
    Flux = scpi.simps(model, wave)*norm
        
    return Flux

def flux_calc_mcmc(res,chains, mode, norm=1e-13):
    
    labels = list(chains.keys())

    popt = np.zeros_like(res['popt'])
    N=100
    Fluxes = []
    res_new = {'name': res['name']}
    for j in range(N):
        for i in range(len(popt)): 
            popt[i] = np.random.choice(chains[labels[i+1]],1)
            
            res_new[labels[i+1]] = [popt[i], 0,0 ]
        res_new['popt'] = popt
        Fluxes.append(flux_calc(res_new, mode,norm))
    
    p50,p16,p84 = np.percentile(Fluxes, (50,16,84))
    p16 = p50-p16
    p84 = p84-p50
    return p50, p16, p84
        
def W80_OIII_calc( function, sol, chains, plot):
    popt = sol['popt']     
    
    import scipy.integrate as scpi
    
    cent =  5008.24*(1+popt[0])/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    N= 100
    
    v10s = np.zeros(N)
    v50s = np.zeros(N)
    v90s = np.zeros(N)
    w80s = np.zeros(N)
    
    if 'OIIIw_fwhm' in sol:
        OIIIr = 5008.24*(1+popt[0])/1e4
        
        fwhms = np.random.choice(chains['OIIIn_fwhm'], N)/3e5/2.35*OIIIr
        fwhmws = np.random.choice(chains['OIIIw_fwhm'], N)/3e5/2.35*OIIIr
        
        OIIIrws = cent + np.random.choice(chains['out_vel'], N)/3e5*OIIIr
        
        peakn = np.random.choice(chains['OIIIn_peak'], N)
        peakw = np.random.choice(chains['OIIIw_peak'], N)
        
        
        for i in range(N):
            y = gauss(wvs, peakn[i],OIIIr, fwhms[i]) + gauss(wvs, peakw[i], OIIIrws[i], fwhmws[i])
            
            Int = np.zeros(Ni-1)
    
            for j in range(Ni-1):

                Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

            Int = Int/max(Int)   

            wv10 = wvs[find_nearest(Int, 0.1)]
            wv90 = wvs[find_nearest(Int, 0.9)]
            wv50 = wvs[find_nearest(Int, 0.5)]

            v10 = (wv10-cent)/cent*3e5
            v90 = (wv90-cent)/cent*3e5
            v50 = (wv50-cent)/cent*3e5
            
            w80 = v90-v10
            
            v10s[i] = v10
            v90s[i] = v90
            v50s[i] = v50
            w80s[i] = w80
    
    elif 'Nar_fwhm' in sol:
        OIIIr = 5008.24*(1+popt[0])/1e4
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*OIIIr
        fwhmws = np.random.choice(chains['outflow_fwhm'], N)/3e5/2.35*OIIIr
        
        OIIIrws = cent + np.random.choice(chains['outflow_vel'], N)/3e5*OIIIr
        
        peakn = np.random.choice(chains['OIIIn_peak'], N)
        peakw = np.random.choice(chains['OIII_out_peak'], N)
        
        
        for i in range(N):
            y = gauss(wvs, peakn[i],OIIIr, fwhms[i]) + gauss(wvs, peakw[i], OIIIrws[i], fwhmws[i])
            
            Int = np.zeros(Ni-1)
    
            for j in range(Ni-1):

                Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

            Int = Int/max(Int)   

            wv10 = wvs[find_nearest(Int, 0.1)]
            wv90 = wvs[find_nearest(Int, 0.9)]
            wv50 = wvs[find_nearest(Int, 0.5)]

            v10 = (wv10-cent)/cent*3e5
            v90 = (wv90-cent)/cent*3e5
            v50 = (wv50-cent)/cent*3e5
            
            w80 = v90-v10
            
            v10s[i] = v10
            v90s[i] = v90
            v50s[i] = v50
            w80s[i] = w80
            
    else:
        OIIIr = 5008.24*(1+popt[0])/1e4
        
        fwhms = np.random.choice(chains['OIIIn_fwhm'], N)/3e5/2.35*OIIIr
        peakn = np.random.choice(chains['OIIIn_peak'], N)
        
        
        for i in range(N):
            y = gauss(wvs, peakn[i],OIIIr, fwhms[i])
            
            Int = np.zeros(Ni-1)
    
            for j in range(Ni-1):

                Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

            Int = Int/max(Int)   

            wv10 = wvs[find_nearest(Int, 0.1)]
            wv90 = wvs[find_nearest(Int, 0.9)]
            wv50 = wvs[find_nearest(Int, 0.5)]

            v10 = (wv10-cent)/cent*3e5
            v90 = (wv90-cent)/cent*3e5
            v50 = (wv50-cent)/cent*3e5
            
            w80 = v90-v10
            
            v10s[i] = v10
            v90s[i] = v90
            v50s[i] = v50
            w80s[i] = w80
            
           
    return error_calc(v10s),error_calc(v90s),error_calc(w80s), error_calc(v50s)


def W80_OIII_calc_single( function, sol, plot, z=0):
    popt = sol['popt']  
    
    if z==0:
        z = popt[0]
  
    
    import scipy.integrate as scpi
    
    cent =  5008.24*(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    
 
    if 'OIIIw_fwhm' in sol:
        OIIIr = 5008.24*(1+sol['z'][0])/1e4
        
        fwhms = sol['OIIIn_fwhm'][0]/3e5/2.35*OIIIr
        fwhmws =sol['OIIIw_fwhm'][0]/3e5/2.35*OIIIr
        
        OIIIrws = cent + sol['out_vel'][0]/3e5*OIIIr
        
        peakn = sol['OIIIn_peak'][0]
        peakw = sol['OIIIw_peak'][0]
        
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
          
    
    elif 'Nar_fwhm' in sol:
        OIIIr = 5008.24*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
        peakn = sol['OIIIn_peak'][0]
        try:
            fwhmws = sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
            OIIIrws = cent + sol['outflow_vel'][0]/3e5*OIIIr
        
            peakw = sol['OIII_out_peak'][0]
            y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
        except:
            y = gauss(wvs, peakn,OIIIr, fwhms) 
            
            
            
    else:
        OIIIr = 5008.24*(1+sol['z'][0])/1e4
        
        fwhms = sol['OIIIn_fwhm'][0]/3e5/2.35*OIIIr
        peakn = sol['OIIIn_peak'][0]
        
        
        y = gauss(wvs, peakn,OIIIr, fwhms)
            
         
    Int = np.zeros(Ni-1)

    for j in range(Ni-1):
        Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

    Int = Int/max(Int)   

    wv10 = wvs[find_nearest(Int, 0.1)]
    wv90 = wvs[find_nearest(Int, 0.9)]
    wv50 = wvs[find_nearest(Int, 0.5)]

    v10 = (wv10-cent)/cent*3e5
    v90 = (wv90-cent)/cent*3e5
    v50 = (wv50-cent)/cent*3e5
    
    w80 = v90-v10
           
    return v10,v90, w80, v50




def W80_Halpha_calc( function, sol, chains, plot,z=0):
    popt = sol['popt'] 
    if z==0:
        z=popt[0]
    
    import scipy.integrate as scpi
    
    cent =  6564.52**(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    N= 100
    
    v10s = np.zeros(N)
    v50s = np.zeros(N)
    v90s = np.zeros(N)
    w80s = np.zeros(N)
    
    if 'outflow_fwhm' in sol:
        Halpha = 6564.52*(1+chains['z'])/1e4
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*Halpha
        fwhmws = np.random.choice(chains['outflow_fwhm'], N)/3e5/2.35*Halpha
        
        Halpha_out = Halpha+ np.random.choice(chains['outflow_vel'], N)/3e5*Halpha
        
        peakn = np.random.choice(chains['Hal_peak'], N)
        peakw = np.random.choice(chains['Hal_out_peak'], N)
        
        
        for i in range(N):
            y = gauss(wvs, peakn[i],Halpha, fwhms[i]) + gauss(wvs, peakw[i], Halpha_out[i], fwhmws[i])
            
            Int = np.zeros(Ni-1)
    
            for j in range(Ni-1):

                Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

            Int = Int/max(Int)   

            wv10 = wvs[find_nearest(Int, 0.1)]
            wv90 = wvs[find_nearest(Int, 0.9)]
            wv50 = wvs[find_nearest(Int, 0.5)]

            v10 = (wv10-cent)/cent*3e5
            v90 = (wv90-cent)/cent*3e5
            v50 = (wv50-cent)/cent*3e5
            
            w80 = v90-v10
            
            v10s[i] = v10
            v90s[i] = v90
            w80s[i] = w80
    
    else:
        Halpha = 6564.52*(1+z)/1e4
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*Halpha
        peakn = np.random.choice(chains['Hal_peak'], N)
        
        
        for i in range(N):
            y = gauss(wvs, peakn[i], Halpha, fwhms[i])
            
            Int = np.zeros(Ni-1)
    
            for j in range(Ni-1):

                Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

            Int = Int/max(Int)   
            
            wv10 = wvs[find_nearest(Int, 0.1)]
            wv90 = wvs[find_nearest(Int, 0.9)]
            wv50 = wvs[find_nearest(Int, 0.5)]

            v10 = (wv10-cent)/cent*3e5
            v90 = (wv90-cent)/cent*3e5
            v50 = (wv50-cent)/cent*3e5
            
            w80 = v90-v10
            
            v10s[i] = v10
            v90s[i] = v90
            v50s[i] = v50
            w80s[i] = w80
            
           
    return error_calc(v10s),error_calc(v90s),error_calc(w80s), error_calc(v50s)

def W80_Halpha_calc_single( function, sol, plot, z=0):
    popt = sol['popt']  
    
    if z==0:
        z = popt[0]
  
    
    import scipy.integrate as scpi
    
    cent =  6564.52*(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    
    
    if 'outflow_fwhm' in sol:
        OIIIr = 6564.52*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
        fwhmws =sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
        
        OIIIrws = cent + sol['outflow_vel'][0]/3e5*OIIIr
        
        peakn = sol['Hal_peak'][0]
        peakw = sol['Hal_out_peak'][0]
        
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
         
    else:
        OIIIr = 6564.52*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
        peakn = sol['Hal_peak'][0]
        
        
        y = gauss(wvs, peakn,OIIIr, fwhms)
            
         
    Int = np.zeros(Ni-1)

    for j in range(Ni-1):
        Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) 

    Int = Int/max(Int)   

    wv10 = wvs[find_nearest(Int, 0.1)]
    wv90 = wvs[find_nearest(Int, 0.9)]
    wv50 = wvs[find_nearest(Int, 0.5)]

    v10 = (wv10-cent)/cent*3e5
    v90 = (wv90-cent)/cent*3e5
    v50 = (wv50-cent)/cent*3e5
    
    w80 = v90-v10
    
    return v10,v90, w80, v50


def W80_NII_calc_single( function, sol, plot, z=0):
    popt = sol['popt']  
    
    if z==0:
        z = popt[0]
  
    
    import scipy.integrate as scpi
    
    cent =  5008.24*(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    
    
    if 'outflow_fwhm' in sol:
        OIIIr = 5008.24*(1+z)/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
        fwhmws =sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
        
        OIIIrws = cent + sol['out_vel'][0]/3e5*OIIIr
        
        peakn = sol['NII_peak'][0]
        peakw = sol['NII_out_peak'][0]
        
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
         
    else:
        OIIIr = 5008.24*(1+z)/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
        peakn = sol['NII_peak'][0]
        
        
        y = gauss(wvs, peakn,OIIIr, fwhms)
            
         
    Int = np.zeros(Ni-1)

    for j in range(Ni-1):
        Int[j] = scpi.simps(y[:j+1], wvs[:j+1]) * 1e-13

    Int = Int/max(Int)   

    wv10 = wvs[find_nearest(Int, 0.1)]
    wv90 = wvs[find_nearest(Int, 0.9)]
    wv50 = wvs[find_nearest(Int, 0.5)]

    v10 = (wv10-cent)/cent*3e5
    v90 = (wv90-cent)/cent*3e5
    v50 = (wv50-cent)/cent*3e5
    
    w80 = v90-v10
           
    return v10,v90, w80, v50

    

# ============================================================================
#  Main class
# =============================================================================
class Cube:
    
    def __init__(self, Full_path, z, ID, flag, savepath, Band, norm=1e-13):
        import importlib
        importlib.reload(emfit )
    
        filemarker = fits.open(Full_path)
        
        #print (Full_path)
        if flag=='KMOS':
            
            header = filemarker[1].header # FITS header in HDU 1
            flux_temp  = filemarker[1].data/norm
                             
            filemarker.close()  # FITS HDU file marker closed
        
        elif flag=='Sinfoni':
            header = filemarker[0].header # FITS header in HDU 1
            flux_temp  = filemarker[0].data/norm
                             
            filemarker.close()  # FITS HDU file marker closed
        
        elif flag=='NIRSPEC_IFU':
            with fits.open(Full_path, memmap=False) as hdulist:
                flux_temp = hdulist['SCI'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                error = hdulist['ERR'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                w = wcs.WCS(hdulist[1].header)
                header = hdulist[1].header
                cube_wcs = astropy.wcs.WCS(hdulist['SCI'].header)
                wave = cube_wcs.all_pix2world(0., 0., np.arange(cube_wcs._naxis[2]), 0)[2]
                wave *= astropy.units.Unit(hdulist['SCI'].header['CUNIT3'])
                if wave.unit==astropy.units.m:
                    wave = wave.to('um')
                else:
                    wave *= 1.e6 # Somehow, units are autoconverted to m
                
                error *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                flux_temp *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                flux_temp = flux_temp.to('1 erg/(s cm2 AA arcsec2)')/0.01
                error = error.to('1 erg/(s cm2 AA arcsec2)')/0.01
                flux_temp = flux_temp.value
                self.error_cube = error.value
                
        elif flag=='NIRSPEC_IFU_fl':
            with fits.open(Full_path, memmap=False) as hdulist:
                flux_temp = hdulist['SCI'].data/norm*10000
                self.error_cube = hdulist['ERR'].data/norm*10000
                w = wcs.WCS(hdulist[1].header)
                header = hdulist[1].header
        
        elif flag=='MIRI':
            with fits.open(Full_path, memmap=False) as hdulist:
                flux_temp = hdulist['SCI'].data/1e4
                self.error_cube = hdulist['ERR'].data/1e4
                w = wcs.WCS(hdulist[1].header)
                header = hdulist[1].header
                
        else:
            raise Exception('Instrument flag not understood')
        
        flux = np.ma.masked_invalid(flux_temp)   #  deal with NaN
       
        # Number of spatial pixels
        n_xpixels = header['NAXIS1']
        n_ypixels = header['NAXIS2']
        dim = [n_ypixels, n_xpixels]
    
        #  Number of spectral pixels
        n_spixels = header['NAXIS3']
        dim = [n_ypixels, n_xpixels, n_spixels]
        
            
        try:
            x = header['CDELT1']
        except:
            header['CDELT1'] = header['CD1_1']
        
        
        try:
            x = header['CDELT2']
        except:
            header['CDELT2'] = header['CD2_2']
            
        
        try:
            x = header['CDELT3']
        except:
            header['CDELT3'] = header['CD3_3']
        
        wave = header['CRVAL3'] + (np.arange(n_spixels) - (header['CRPIX3'] - 1.0))*header['CDELT3']
         
        
        try:
            deg_per_pix_x = abs(header['CDELT1'])
            
        except:
            deg_per_pix_x = header['CDELT1']
            
            
        arc_per_pix_x = 1.*deg_per_pix_x*3600
        Xpix = header['NAXIS1']
        Xph = Xpix*arc_per_pix_x
        
        try:
            deg_per_pix_y = abs(header['CDELT2'])
        
        except:
            deg_per_pix_y = header['CDELT2']
            header['CD2_2'] = header['CDELT2']
            
        arc_per_pix_y = deg_per_pix_y*3600
        Ypix = header['NAXIS2']
        Yph = Ypix*arc_per_pix_y
        
        
        
        self.flux_norm= norm
        self.dim = dim
        self.z = z
        self.obs_wave = wave
        self.flux = flux
        self.ID = ID
        self.instrument = flag
        if flag=='NIRSPEC_IFU_fl':
            self.instrument= 'NIRSPEC_IFU'
        self.savepath = savepath
        self.header= header
        self.phys_size = np.array([Xph, Yph])
        self.band = Band
        
        #hdu = fits.PrimaryHDU(self.error_cube, header=header)
        #hdulist = fits.HDUList([hdu])
        #hdulist.writeto('/Users/jansen/Cube_temp.fits', overwrite=True)
    
    def add_res(self, line_cat):
        
        self.cat = line_cat
        

    def mask_emission(self):
        '''This function masks out all the OIII and HBeta emission
        '''
        z= self.z
        OIIIa=  501./1e3*(1+z)
        OIIIb=  496./1e3*(1+z)
        Hbeta=  485./1e3*(1+z)
        width = 300/3e5*OIIIa
    
        mask =  self.flux.mask.copy()
        wave= self.obs_wave
    
        OIIIa_loc = np.where((wave<OIIIa+width)&(wave>OIIIa-width))[0]
        OIIIb_loc = np.where((wave<OIIIb+width)&(wave>OIIIb-width))[0]
        Hbeta_loc = np.where((wave<Hbeta+width)&(wave>Hbeta-width))[0]
    
        mask[OIIIa_loc,:,:] = True
        mask[OIIIb_loc,:,:] = True
        mask[Hbeta_loc,:,:] = True
        
        self.em_line_mask= mask
        

    def mask_sky(self,sig, mode=0):
            
        bins = np.linspace(0,2048,5)
        bins= np.array(bins, dtype=int)
        
        flux = np.ma.array(data=self.flux.data.copy(), mask= self.em_line_mask.copy())    
        mask =  self.em_line_mask.copy()  
        dim = self.dim
        wave = self.obs_wave
        x=0
        
        if mode=='Hsin':
            use = np.where((wave>1.81)&(wave<1.46))[0]
            mask[use,:,:] = True 
        
        for i in range(dim[0]):
            for j in range(dim[1]):
                stds = np.ma.std(flux[:,i,j])
                y = np.ma.mean(flux[:,i,j])
                
                         
                sky = np.where((flux[:,i,j]< (y-stds*sig)) | (flux[:,i,j]> (y+stds*sig)))[0]
                    
                mask[sky,i,j] = True
                
                if i==20 and j==20:
                    print(sky)
        
                
        self.sky_line_mask_em = mask
        
        flux = np.ma.array(data=flux.data, mask=mask)
        
        noise_spax = np.ma.std(flux, axis=(0))
                


    def collapse_white(self, plot):
        try:
            flux = np.ma.array(self.flux.data, mask=self.sky_line_mask_em) 
        except:
            flux = self.flux
        
        median = np.ma.median(flux, axis=(0))  
        
        ID = self.ID
        if ID== 'ALESS_75':
            wave = self.obs_wave.copy()
            
            use = np.where((wave < 1.7853) &(wave > 1.75056))[0]
            
            use = np.append(use, np.where((wave < 2.34) &(wave > 2.31715))[0] )
            
            median = np.ma.median(flux[use,:,:], axis=(0))  
            
       
        self.Median_stack_white = median
        
      
        if plot==1:
            plt.figure()
            plt.imshow(median,  origin='lower')
            plt.colorbar()
            
        

    def find_center(self, plot, extra_mask=0, manual=np.array([0])):
        '''
        Input: 
            Storage, 
            image name to be loaded
            Plot it?
            extra mask of the weird bright features
            manual - If manual is on then It will just return the manual inputs. Use if there is no continuum detected
        '''
        
        shapes = self.dim
        data = self.Median_stack_white
        
        # Create the x inputs for the curve_fit
        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)   
        
        import scipy.optimize as opt
        
        if len(manual)==1:
            
            # If there is no extra mask -create a dummy extra mask
            if len(np.shape(extra_mask))<1.:
                extra_mask = self.flux.mask.copy()
                extra_mask = extra_mask[0,:,:]
                extra_mask[:,:] = False
        
            #############
            # Masking the edges based on the pixel scale. 
            edges = self.flux.mask.copy() # Mask to collapse
            edges = edges[0,:,:]
            edges[:,:] = False
            try:
                pixel_scale = 1./(self.header['CD2_2']*3600)
            
            except:
                pixel_scale = 1./(self.header['CDELT2']*3600)
                
        
            if pixel_scale < 7:    
                edges[:,0] = True
                edges[:,1] = True
                edges[:, -1] = True
                edges[:, -2] = True
                edges[1,:] = True
                edges[0,:] = True
                edges[-1,:] = True
                edges[-2,:] = True
                
                print ('Masking edges based on 0.2 scale')
                
            else:
                edges[:,:5] = True
                edges[:, -6:] = True        
                edges[:5,:] = True        
                edges[-6:,:] = True
                
                print ('Masking edges based on 0.1 scale')
            
            # Combining the edge and extra mask 
            comb = np.logical_or(extra_mask, edges)
            
            # Setting all other nan values to 0 to avoid any troubles
            data.data[np.isnan(data.data)] = 0
            data= data.data
            
            # combining the data and masks
            masked_im = np.ma.array(data= data, mask=comb)
            
            # Finding the center of the contiunuum 
            loc = np.ma.where(masked_im == masked_im.max())
            print ('location of the peak on the continuum',loc)
            
            #plt.figure()
            #plt.title('Find center - plot')
            #plt.imshow(masked_im, origin='low')
            
            # Setting the            
            initial_guess = (data[loc[1][0], loc[0][0]],loc[1][0],loc[0][0],1,1,0,0)
            
            print ('Initial guesses', initial_guess)
            
            try:
                dm = (x,y)
                popt, pcov = opt.curve_fit(twoD_Gaussian, dm, data.ravel(), p0=initial_guess)
                
                er = np.sqrt(np.diag(pcov))
            
                print ('Cont loc ', popt[1:3])
                print ('Cont loc er', er[1:3])
                
            except:
                print('Failed fit to the continuum - switching to the center of the cube')
                popt = np.zeros(7)
                popt[1] = int(self.dim[0]/2); popt[2] = int(self.dim[1]/2) 
            
            if (popt[1]<3) | (popt[2]<3):
                print('Failed fit to the continuum - switching to the center of the cube')
                popt = np.zeros(7)
                popt[1] = int(self.dim[0]/2); popt[2] = int(self.dim[1]/2) 
                
            self.center_data = popt      
            
            
            
        else:
            manual = np.append(data[int(manual[0]), int(manual[1])], manual)
            manual = np.append(manual, np.array([2.,2.,0.5,0. ]))
            self.center_data = manual
        
        
    
    def choose_pixels(self, plot, rad= 0.6, flg=1, mask_manual=[0]):
        ''' Choosing the pixels that will collapse into the 1D spectrum. Also this mask is used later
        '''
        center =  self.center_data[1:3].copy()
        shapes = self.dim
        
        print ('Center of cont', center)
        
        print ('Extracting spectrum from diameter', rad*2, 'arcseconds')
        
        # Creating a mask for all spaxels. 
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        print('Pixel scale:', arc)
        print ('radius ', arc*rad)
        
        
        # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc*rad:
                    mask_catch[:,ix,iy] = False
                    
        if len(mask_manual) !=1:
            for i in range(shapes[2]):
                mask_catch[i,:,:] = mask_manual.copy()
        
        # 587 have special extended stuff that I dont want to mask
        if flg=='K587':
            mask_catch[:,11:29 ,3:36 ] = False
            print ('extracting flux K band XID 587')
        
        elif flg=='H587':
            mask_catch[:,12:29 ,2:36 ] = False
            print ('extracting flux H band XID 587')
        
        elif flg=='K751':
            mask_catch[:,3:31 ,5:34 ] = False
            print ('extracting flux K band XID 751')
        
        elif flg=='K587sin':
            mask_catch[:,20:65 ,25:50 ] = False
            print ('extracting flux H band XID 587 sinfoni')
        
        self.Signal_mask = mask_catch
        
        
        if plot==1:
            plt.figure()
            plt.title('Selected Spaxels for 1D spectrum + Contours from 2D Gaus')
            plt.imshow(np.ma.array(data=self.Median_stack_white, mask=self.Signal_mask[0,:,:]), origin='lower')
            plt.colorbar()
    
            shapes = self.dim
            x = np.linspace(0, shapes[1]-1, shapes[1])
            y = np.linspace(0, shapes[0]-1, shapes[0])
            x, y = np.meshgrid(x, y)
        
            data_fit = twoD_Gaussian((x,y), *self.center_data)
    
            plt.contour(x, y, data_fit.reshape(shapes[0], shapes[1]), 8, colors='w')
    
            
    
    def background_sub_spec(self, center, rad=0.6, plot=0):
        # Creating a mask for all spaxels.
        shapes = self.dim
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        
        
        # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc*rad:
                    mask_catch[:,ix,iy] = False
        
        mask_spax = mask_catch.copy()
        # Loading mask of the sky lines an bad features in the spectrum 
        mask_sky_1D = self.sky_clipped_1D.copy()
        total_mask = np.logical_or( mask_spax, self.sky_clipped)
        
        flux = np.ma.array(data=self.flux.data, mask= total_mask) 
        
        Sky = np.ma.median(flux, axis=(1,2))
        Sky = np.ma.array(data = Sky.data, mask=mask_sky_1D)
        
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                self.flux[:,ix,iy] = self.flux[:,ix,iy] - Sky
        
        if plot==1:
            plt.figure()
            plt.title('Sky spectrum')
            
            plt.plot(self.obs_wave, np.ma.array(data= Sky , mask=self.sky_clipped_1D), drawstyle='steps-mid')
            
        
            plt.ylabel('Flux')
            plt.xlabel('Observed wavelength')
    
    
    def D1_spectra_collapse(self, plot, addsave='', err_range=[0], boundary=2.4):
        '''
        This function collapses the Cube to form a 1D spectrum of the galaxy
        '''
        # Loading the Signal mask - selects spaxel to collapse to 1D spectrum 
        mask_spax = self.Signal_mask.copy()
        # Loading mask of the sky lines an bad features in the spectrum 
        mask_sky_1D = self.sky_clipped_1D.copy()
        
        if self.instrument=='NIRSPEC_IFU':
            total_mask = np.logical_or( mask_spax, self.sky_clipped)
        else:
            total_mask = np.logical_or( mask_spax, self.flux.mask)
        # M
        flux = np.ma.array(data=self.flux.data, mask= total_mask) 
        
        D1_spectra = np.ma.sum(flux, axis=(1,2))
        D1_spectra = np.ma.array(data = D1_spectra.data, mask=mask_sky_1D)
        
        wave= self.obs_wave
        
        if plot==1:
            plt.figure()
            plt.title('Collapsed 1D spectrum from D1_spectra_collapse fce')
            
            plt.plot(wave, D1_spectra.data, drawstyle='steps-mid', color='grey')
            plt.plot(wave, np.ma.array(data= D1_spectra, mask=self.sky_clipped_1D), drawstyle='steps-mid')
            
        
            plt.ylabel('Flux')
            plt.xlabel('Observed wavelength')
       
        
        self.D1_spectrum = D1_spectra
        if self.instrument=='NIRSPEC_IFU':
            if len(err_range)==2:
                error = stats.sigma_clipped_stats(D1_spectra[(err_range[0]<self.obs_wave) \
                                                             &(self.obs_wave<err_range[1])],sigma=3)[2] \
                                                        *np.ones(len(D1_spectra))
                                                        
            elif len(err_range)==4:
                error1 = stats.sigma_clipped_stats(D1_spectra[(err_range[0]<self.obs_wave) \
                                                             &(self.obs_wave<err_range[1])],sigma=3)[2]
                error2 = stats.sigma_clipped_stats(D1_spectra[(err_range[1]<self.obs_wave) \
                                                             &(self.obs_wave<err_range[2])],sigma=3)[2]
                
                error = np.zeros(len(D1_spectra))
                error[self.obs_wave<boundary] = error1
                error[self.obs_wave>boundary] = error2
            
            self.D1_spectrum_er = stats.sigma_clipped_stats(D1_spectra[(err_range[0]<self.obs_wave) &(self.obs_wave<err_range[1])],sigma=3)[2]*np.ones(len(self.D1_spectrum))
            
        else:  
            self.D1_spectrum_er = stats.sigma_clipped_stats(D1_spectra,sigma=3)[2]*np.ones(len(self.D1_spectrum)) #STD_calc(wave/(1+self.z)*1e4,self.D1_spectrum, self.band)* np.ones(len(self.D1_spectrum))
        
        print(self.D1_spectrum_er)
        if self.ID =='cdfs_220':
            self.D1_spectrum_er = 0.05*np.ones(len(self.D1_spectrum))
        if self.ID =='cid_346':
            self.D1_spectrum_er = 0.005*np.ones(len(self.D1_spectrum))
        
        if self.ID =='cdfs_584':
            self.D1_spectrum_er = 0.02*np.ones(len(self.D1_spectrum))
            
        
        Save_spec = np.zeros((4,len(D1_spectra)))
        
        Save_spec[0,:] = wave
        Save_spec[1,:] = self.D1_spectrum
        Save_spec[2,:] = self.D1_spectrum_er.copy()
        Save_spec[3,:] = mask_sky_1D
        

        
        np.savetxt(self.savepath+self.ID+'_'+self.band+addsave+'_1Dspectrum.txt', Save_spec)
        


    def mask_JWST(self, plot, threshold=1e11, spe_ma=[], dtype=bool):
        
        sky_clipped =  self.flux.mask.copy()
        sky_clipped[self.error_cube>threshold] = True 
        
        sky_clipped_1D = self.flux.mask.copy()[:,10,10].copy()
        sky_clipped_1D[:] = False
        sky_clipped_1D[spe_ma] = True
        self.sky_clipped_1D = sky_clipped_1D
        self.sky_clipped = sky_clipped
        self.Sky_stack_mask = self.flux.mask.copy()
        
        
        
    def stack_sky(self,plot, spe_ma=np.array([], dtype=bool), expand=0):
        header = self.header
        ######
        # Finding the center of the Object
        center =  self.center_data[1:3].copy()
        
        ######
        # Defining other variable to be used
        wave = self.obs_wave.copy()#*1e4/(1+z) # Wavelength
        flux = self.flux.copy()   # Flux 
        
        shapes = self.dim # The shapes of the image
        
        mask_nan = self.flux.mask.copy()
        
        mask_collapse = self.flux.mask.copy() # Mask to collapse 
        mask_collapse[:,:,:] = False
        
        
        
        header  = self.header
        
        try:
            Radius = np.round(1.2/(header['CD2_2']*3600))
        
        except:
            Radius = np.round(1.2/(header['CDELT2']*3600))
            
        
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< Radius:
                    mask_collapse[:,ix,iy] = True
        
        try:
            arc = (header['CD2_2']*3600)
        except:
            arc = (header['CDELT2']*3600)
            
            
        if arc< 0.17:
            print ('Masking special corners')
            mask_collapse[:,:,0] = True
            mask_collapse[:,:,1] = True
            mask_collapse[:,:,2] = True
            mask_collapse[:,:,3] = True
            
            mask_collapse[:,:,-1] = True
            mask_collapse[:,:,-2] = True
            mask_collapse[:,:,-3] = True
            mask_collapse[:,:,-4] = True
            
            mask_collapse[:,0,:] = True
            mask_collapse[:,1,:] = True
            mask_collapse[:,2,:] = True
            mask_collapse[:,3,:] = True
            
            mask_collapse[:,-1,:] = True
            mask_collapse[:,-2,:] = True
            mask_collapse[:,-3,:] = True
            mask_collapse[:,-4,:] = True
        
        elif arc> 0.17:
            print ('Masking special corners')
            mask_collapse[:,:,0] = True
            mask_collapse[:,:,-1] = True  
            mask_collapse[:,0,:] = True       
            mask_collapse[:,-1,:] = True
            
        # For Hband sinfoni data I subtracted the frames from each other to imitate the sky frame. Therefore there are two negative objects in the cube. I am masking those as well.
        if (self.band =='Hsin'):
            
            mask_collapse[:,76-10:76+8 ,83-10:83+10 ] = True #mask_collapse[:,2:10 ,0:12 ] = True #
            mask_collapse[:,21-10:21+8 ,27-10:27+10 ] = True
            print ('HB89 Hsin masking of negative objects')
            
        
        mask_collapse = np.logical_or(mask_collapse, mask_nan)
                  
        self.Sky_stack_mask = mask_collapse
        ######
        
        # Collapsing the spaxels into sky spectrum
        stacked_sky = np.ma.sum(np.ma.array(data=flux.data, mask=mask_collapse), axis=(1,2)) 
        
        # Creating a brand new mask to mask weird features in the sky_spectrum 
        std_mask = mask_collapse[:,7,7].copy()
        std_mask[:] = False
        
        weird = np.array([], dtype=int)
        
        if self.band=='YJ':
            print ('Masking based on YJ band')
            # Actually masking the weird features: Edges + that 1.27 annoying bit
            weird = np.where((wave>1.269) &(wave<1.285))[0]
            #weird = np.where((wave>1.28499) &(wave<1.285))[0]
            weird = np.append(weird, np.where((wave<1.010))[0])
            weird = np.append(weird, np.where((wave>1.35))[0])
            weird = np.append(weird, np.where((wave>1.31168)&(wave< 1.3140375))[0])
        
            # Masking the stacked sky not to intefere with calculating std
            std_mask[weird] = True   
        
        elif  self.band=='H':
            print ('Masking based on H band')
            weird = np.where((wave<1.45))[0]
            weird = np.append(weird, np.where((wave>1.85))[0])
            weird = np.append(weird, np.where((wave>1.78351) &(wave<1.7885))[0])
        
        elif self.band=='K':
            print ('Masking based on K band')
            weird = np.where((wave<1.945))[0]
            weird = np.append(weird, np.where((wave>2.4))[0])
            
        elif self.band=='Hsin':
            print ('Masking based on H band with Sinfoni')
            weird = np.where((wave>1.82))[0]
            weird = np.append(weird, np.where((wave<1.45))[0])
            print (len(wave), len(weird))
        
        elif self.band=='Ysin':
            print ('Masking based on Y band with Sinfoni')
            weird = np.where((wave>1.35))[0]
            weird = np.append(weird, np.where((wave<1.11))[0])
            print (len(wave), len(weird))
        
        elif self.band=='Ksin':
            print ('Masking based on K band with Sinfoni')
            weird = np.where((wave>2.4))[0]
            weird = np.append(weird, np.where((wave<1.945))[0])
            print (len(wave), len(weird))
            
        elif self.band=='HKsin':
            print ('Masking based on HK band with Sinfoni')
            weird = np.where((wave>2.4))[0]
            weird = np.append(weird, np.where((wave<1.5))[0])
            print (len(wave), len(weird))
           
        weird = np.append(weird, spe_ma)
            
        std_mask[weird] = True

        stacked_sky_mask = np.ma.array(data = stacked_sky, mask=std_mask)
        
        
        # Masking sky lines 
        y = np.zeros_like(stacked_sky)
            
        clip_w = 1.5
        
        ssma = stacked_sky_mask.data[np.invert(stacked_sky_mask.mask)]
        low, hgh = conf(ssma)   
        
        
        sky = np.where((stacked_sky<y+ low*clip_w) | (stacked_sky> y + hgh*clip_w))[0]
        
        sky_clipped =  self.flux.mask.copy()
        sky_clipped = sky_clipped[:,7,7]
        sky_clipped[sky] = True          # Masking the sky features
        sky_clipped[weird] = True        # Masking the weird features
            
          
        # Storing the 1D sky line mask into cube to be used later
        mask_sky = self.flux.mask.copy()
        

        
        if (self.ID=='xuds_316') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.202 ) & (wave > 1.199))[0]] = False
            
        if (self.ID=='cdfs_328') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.2658 ) & (wave > 1.2571))[0]] = False
        
        if (self.ID=='cdfs_220') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.286 ) & (wave > 1.2571))[0]] = False
        
        if (self.ID=='cdfs_751') & (self.band=='K'):
            sky_clipped[np.where((wave< 1.99 ) )[0]] = False
        
        if (self.ID=='cdfs_485') & (self.band=='H'):
            sky_clipped[np.where((wave< 1.71577 ) & (wave > 1.71300))[0]] = False
        
        if (self.ID=='xuds_398') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.2976 ) & (wave > 1.26))[0]] = False
        if (self.ID=='xuds_479') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.04806 ) & (wave > 1.03849))[0]] = False
            
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                mask_sky[:,ix,iy] = sky_clipped
        
        # Storing the 1D and 3D sky masks
        self.sky_clipped = mask_sky
        self.sky_clipped_1D = sky_clipped 
        
        self.Stacked_sky = stacked_sky
        
        
        np.savetxt(self.savepath+'clp.txt', sky_clipped)
        

        
        if plot==1:
            mask_spax = self.Signal_mask.copy()
            mask_sky = self.sky_clipped.copy()
        
            total_mask = np.logical_or(mask_spax, mask_sky)
        
        
            flux_new = np.ma.array(data=self.flux.data, mask= total_mask) 
            D1_spectra_new = np.ma.sum(flux_new, axis=(1,2))
        
            flux_old = np.ma.array(data=self.flux.data, mask=mask_spax)
            D1_spectra_old = np.ma.sum(flux_old, axis=(1,2))
            #######
            # Plotting the spaxels used to assemble the sky
            
            plt.figure()
            plt.imshow(np.ma.array(data=self.Median_stack_white, mask=mask_collapse[0,:,:]), origin='lower')
            plt.colorbar()
            plt.title('Removing the Galaxy')
        
        
            #######
            # 3 panel plot
            # 1 panel - The stacked spectrum
            # 2 panel - clipped sky lines
            # 3 panel - The collpsed 1D spectrum of the galaxy - No masking
            # 4 panel - The collpsed 1D spectrum of the galaxy - Final masking
            
            f, (ax1,ax2, ax3, ax4) = plt.subplots(4, sharex=True,sharey=True, figsize=(10,15))
            
            ax1.set_title('Stacked spectrum outside the galaxy')
        
            ax1.plot(wave, y+np.ones_like(stacked_sky)*hgh, 'g--')
            ax1.plot(wave, y+np.ones_like(stacked_sky)*low, 'g--')
            
            ax1.plot(wave, y+np.ones_like(stacked_sky)*hgh*clip_w, 'r--')
            ax1.plot(wave, y+np.ones_like(stacked_sky)*low*clip_w, 'r--')
            
            ax1.plot(wave, (stacked_sky), drawstyle='steps-mid', color='grey')
            ax1.plot(wave, np.ma.array(data=stacked_sky,mask=std_mask), drawstyle='steps-mid')
            ax1.set_ylabel('Sky spec')
            
            ax1.set_ylim(np.ma.min(np.ma.array(data=stacked_sky,mask=std_mask)), np.ma.max(np.ma.array(data=stacked_sky,mask=std_mask)))
        
            ax2.plot(wave, np.ma.array(data=stacked_sky, mask=sky_clipped), drawstyle='steps-mid')
            ax2.set_ylabel('Clipped sky')
            
            ax2.set_ylim(np.ma.min(np.ma.array(data=stacked_sky,mask=std_mask)), np.ma.max(np.ma.array(data=stacked_sky,mask=std_mask)))
        
            
            ax3.set_ylabel('Old spec')
            ax3.plot(wave, D1_spectra_old, drawstyle='steps-mid')
            
            ax4.plot(wave, D1_spectra_new, drawstyle='steps-mid')
            ax4.set_ylabel('New spec')
            
            ax1.set_ylim(-0.05,0.4)
            
            plt.tight_layout()
    
    def fitting_collapse_Halpha(self, plot, AGN = 'BLR', progress=True,er_scale=1, N=6000, priors= {'z':[0, 'normal', 0,0.003],\
                                                                                       'cont':[0,'loguniform',-3,1],\
                                                                                       'cont_grad':[0,'normal',0,0.3], \
                                                                                       'Hal_peak':[0,'loguniform',-3,1],\
                                                                                       'BLR_peak':[0,'loguniform',-3,1],\
                                                                                       'NII_peak':[0,'loguniform',-3,1],\
                                                                                       'Nar_fwhm':[300,'uniform',100,900],\
                                                                                       'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                       'BLR_offset':[-100,'normal',0,200],\
                                                                                        'SIIr_peak':[0,'loguniform',-3,1],\
                                                                                        'SIIb_peak':[0,'loguniform',-3,1],\
                                                                                        'Hal_out_peak':[0,'loguniform',-3,1],\
                                                                                        'NII_out_peak':[0,'loguniform',-3,1],\
                                                                                        'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                        'outflow_vel':[-50,'normal', 0,300]}):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()/er_scale
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        
        if AGN=='BLR':
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha(wave,flux,error,z,N=N, BLR=0, progress=progress, priors=priors)
    
            prop_sig = prop_calc(flat_samples_sig)
            
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            flat_samples_blr, fitted_model_blr = emfit.fitting_Halpha(wave,flux,error,z,N=N, BLR=1, progress=progress, priors=priors)
            prop_blr = prop_calc(flat_samples_blr)
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'Halpha')
            chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_blr, prop_blr, 'Halpha')
            
            
            if BICM-BICS <-2:
                print('Delta BIC' , BICM-BICS, ' ')
                print('BICM', BICM)
                self.D1_fit_results = prop_blr
                self.D1_fit_chain = flat_samples_blr
                self.D1_fit_model = fitted_model_blr
                
                self.z = prop_blr['popt'][0]
                
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
                self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = BICM-BICS
                
            else:
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                
                self.dBIC = BICM-BICS
                
            
            if (self.ID=='cid_111') | (self.ID=='xuds_254') | (self.ID=='xuds_379') | (self.ID=='xuds_235') | (self.ID=='sxds_620')\
                | (self.ID=='cdfs_751') | (self.ID=='cdfs_704') | (self.ID=='cdfs_757') | (self.ID=='sxds_787') | (self.ID=='sxds_1093')\
                    | (self.ID=='xuds_186') | (self.ID=='cid_1445') | (self.ID=='cdfs_38')| (self.ID=='cdfs_485')\
                        | (self.ID=='cdfs_588')  | (self.ID=='cid_932')  | (self.ID=='xuds_317'):
                    
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = BICM-BICS
                
                
            if (self.ID=='xuds_168') :
                 print('Delta BIC' , BICM-BICS, ' ')
                 print('BICM', BICM)
                 self.D1_fit_results = prop_blr
                 self.D1_fit_chain = flat_samples_blr
                 self.D1_fit_model = fitted_model_blr
                 self.z = prop_blr['popt'][0]
                 
                 self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
                 self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                 self.dBIC = BICM-BICS
                 
        if AGN=='Outflow':
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha(wave,flux,error,z,N=N, BLR=0, progress=progress,priors=priors)
    
            prop_sig = prop_calc(flat_samples_sig)
            
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            flat_samples_out, fitted_model_out = emfit.fitting_Halpha(wave,flux,error,z,N=N, BLR=-1, progress=progress, priors=priors)
            prop_out = prop_calc(flat_samples_out)
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'Halpha')
            chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_out, prop_out, 'Halpha')
            
            
            if BICM-BICS <-2:
                print('Delta BIC' , BICM-BICS, ' ')
                print('BICM', BICM)
                self.D1_fit_results = prop_out
                self.D1_fit_chain = flat_samples_out
                self.D1_fit_model = fitted_model_out 
                
                self.z = prop_out['popt'][0]
                
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = BICM-BICS
                
            else:
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                
                self.dBIC = BICM-BICS
                
        if AGN=='Single_only':
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha(wave,flux,error,z, BLR=0,N=N, progress=progress, priors=priors)
    
            prop_sig = prop_calc(flat_samples_sig)
            
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'Halpha')
        
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
                
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                
            self.dBIC = 3
            
        if AGN=='Outflow_only':
        
            flat_samples, fitted_model = emfit.fitting_Halpha(wave,flux,error,z,N=N, BLR=-1, progress=progress, priors=priors)
            prop = prop_calc(flat_samples)
            
           
            self.D1_fit_results = prop
            self.D1_fit_chain = flat_samples
            self.D1_fit_model = fitted_model
            
            self.z = prop['popt'][0]
                
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 10
            
        if AGN=='BLR_only':
            
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha(wave,flux,error,z, BLR=1,N=N, progress=progress, priors=priors)
    
            prop_sig = prop_calc(flat_samples_sig)
            
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'Halpha')
        
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
                
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                
            self.dBIC = 3
            
        if AGN=='QSO_BKPL':
            
            flat_samples, fitted_model = emfit.fitting_Halpha(wave,flux,error,z, BLR='QSO_BKPL',N=N, progress=progress, priors=priors)
            prop = prop_calc(flat_samples)
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model, prop, 'Halpha')
        
            self.D1_fit_results = prop
            self.D1_fit_chain = flat_samples
            self.D1_fit_model = fitted_model
            self.z = prop['popt'][0]
                
            self.SNR =  10
            self.SNR_sii =  10
                
            self.dBIC = 3
        
        labels= list(self.D1_fit_chain.keys())[1:]
            
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print('SNR hal ', self.SNR)
        print('SNR SII ', self.SNR_sii)
        
        
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_Halpha(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error', axres=ax2)
        
        self.fit_plot = [f,ax1,ax2]
        
        if (AGN=='BLR') | (AGN=='Outflow'):
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_Halpha(wave, flux, ax1a, prop_sig , fitted_model_sig)
            try:
                emplot.plotting_Halpha(wave, flux, ax2a, prop_blr , fitted_model_blr)
            except:
                emplot.plotting_Halpha(wave, flux, ax2a, prop_out , fitted_model_out)
            
            
    def fitting_collapse_Halpha_OIII(self, plot, progress=True,N=6000,outflow='Single_only', priors={'z':[0,'normal', 0, 0.003],\
                                                                                                     'cont':[0,'loguniform', -3,1],\
                                                                                                     'cont_grad':[0,'normal', 0,0.2],\
                                                                                                     'Hal_peak':[0,'loguniform', -3,1],\
                                                                                                     'NII_peak':[0,'loguniform', -3,1],\
                                                                                                     'Nar_fwhm':[300,'uniform', 200,900],\
                                                                                                     'SIIr_peak':[0,'loguniform', -3,1],\
                                                                                                     'SIIb_peak':[0,'loguniform', -3,1],\
                                                                                                     'OIIIn_peak':[0,'loguniform', -3,1],\
                                                                                                     'Hbeta_peak':[0,'loguniform', -3,1],\
                                                                                                     'OI_peak':[0,'loguniform', -3,1],\
                                                                                                     'outflow_fwhm':[450,'uniform', 300,900],\
                                                                                                     'outflow_vel':[-50,'normal', -50,100],\
                                                                                                     'Hal_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'NII_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'OIII_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'OI_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'Hbeta_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'zBLR':[0,'normal', 0,0.003],\
                                                                                                     'BLR_fwhm':[4000,'normal', 5000,500],\
                                                                                                     'BLR_hal_peak':[0,'loguniform', -3,1],\
                                                                                                     'BLR_hbe_peak':[0,'loguniform', -3,1],\
                                                                                                     }):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        if outflow=='Single_only':   
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha_OIII(wave,flux,error,z,N=N, progress=progress, outflow=0, priors=priors)
        
            prop_sig = prop_calc(flat_samples_sig)
                
                
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            
            self.SNR_hal =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
    
            
            self.dBIC = 3
            
            
        elif outflow=='Outflow_only':   
            flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha_OIII(wave,flux,error,z,N=N, progress=progress, outflow=1, priors=priors)
        
            prop_sig = prop_calc(flat_samples_sig)
                
                
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            
            self.SNR_hal =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
            
            
            self.dBIC = 3
            
        elif outflow=='BLR':   
             flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha_OIII(wave,flux,error,z,N=N, progress=progress, outflow='BLR', priors=priors)
         
             prop_sig = prop_calc(flat_samples_sig)
                 
                 
             y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
             chi2S = sum(((flux.data-y_model_sig)/error)**2)
             BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
             
             self.D1_fit_results = prop_sig
             self.D1_fit_chain = flat_samples_sig
             self.D1_fit_model = fitted_model_sig
             self.z = prop_sig['popt'][0]
             
             self.SNR_hal =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
             
             
             self.dBIC = 3
        
        elif outflow=='QSO_BKPL':   
             flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha_OIII(wave,flux,error,z,N=N, progress=progress, outflow='QSO_BKPL', priors=priors)
         
             prop_sig = prop_calc(flat_samples_sig)
                 
                 
             y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
             chi2S = sum(((flux.data-y_model_sig)/error)**2)
             BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
             
             self.D1_fit_results = prop_sig
             self.D1_fit_chain = flat_samples_sig
             self.D1_fit_model = fitted_model_sig
             self.z = prop_sig['popt'][0]
             '''
             self.SNR_hal =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_OIII =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
             '''
             
             self.dBIC = 3
        
        else:
            raise Exception('outflow variable in fitting_collapse_Halpha_OIII not understood.')
        labels= list(self.D1_fit_chain.keys())[1:]
            
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        
        f = plt.figure(figsize=(10,4))
        if outflow=='QSO_BKPL':
            baxes = brokenaxes(xlims=((4700,5050),(6200,6800)),  hspace=.01)
        
        else:
            baxes = brokenaxes(xlims=((4800,5050),(6250,6350),(6400,6800)),  hspace=.01)
        
        emplot.plotting_Halpha_OIII(wave, flux, baxes, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error')
        baxes.set_xlabel('Restframe wavelength (ang)')
        baxes.set_ylabel(r'$10^{-16}$ ergs/s/cm2/mic')
        
        self.fit_plot = [f,baxes]
        
        
        
    def fitting_collapse_OIII(self,  plot, outflow='both',template=0, Hbeta_dual=0, N=6000,priors= {'z': [0,'normal',0, 0.003],\
                                                                                                    'cont':[0,'loguniform',-3,1],\
                                                                                                    'cont_grad':[0,'normal',0,0.2], \
                                                                                                    'OIIIn_peak':[0,'loguniform',-3,1],\
                                                                                                    'OIIIw_peak':[0,'loguniform',-3,1],\
                                                                                                    'OIIIn_fwhm':[300,'uniform', 100,900],\
                                                                                                    'OIIIw_fwhm':[700,'uniform',600,2500],\
                                                                                                    'out_vel':[-50,'normal',0,200],\
                                                                                                    'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbeta_fwhm':[400,'uniform',120,7000],\
                                                                                                    'Hbeta_vel':[10,'normal', 0,200],\
                                                                                                    'Hbetan_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbetan_fwhm':[300,'uniform',120,700],\
                                                                                                    'Hbetan_vel':[10,'normal', 0,100],\
                                                                                                    'Fe_peak':[0,'loguniform',-3,1],\
                                                                                                    'Fe_fwhm':[3000,'uniform',2000,6000]}):
        
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        ID = self.ID
        
        
        if outflow=='both':
            flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=0,N=N, priors=priors, Hbeta_dual=Hbeta_dual, template=template)
            prop_sig = prop_calc(flat_samples_sig)
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow=1,N=N, priors=priors, Hbeta_dual=Hbeta_dual ,template=template)
            prop_out = prop_calc(flat_samples_out)
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'OIII')
            chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_out, prop_out, 'OIII')
            
            
            if BICM-BICS <-2:
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_out
                self.D1_fit_chain = flat_samples_out
                self.D1_fit_model = fitted_model_out
                self.z = prop_out['popt'][0]
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
                
                
            else:
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
                
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm')
             
            if (ID=='cdfs_751') | (ID=='cid_40') | (ID=='xuds_068') | (ID=='cdfs_51') | (ID=='cdfs_614')\
                | (ID=='xuds_190') | (ID=='cdfs_979') | (ID=='cdfs_301')| (ID=='cid_453') | (ID=='cid_61') | (ID=='cdfs_254')  | (ID=='cdfs_427'):
                    
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
                
                
                
            if ID=='cid_346':
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_out
                self.D1_fit_chain = flat_samples_out
                self.D1_fit_model = fitted_model_out
                self.z = prop_out['popt'][0]
                self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
            
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_OIII(wave, flux, ax1a, prop_sig , fitted_model_sig)
            emplot.plotting_OIII(wave, flux, ax2a, prop_out ,fitted_model_out)
            
            
        elif outflow=='single':
            
            flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=0,N=N, priors=priors, Hbeta_dual=Hbeta_dual, template=template)
            prop_sig = prop_calc(flat_samples_sig)
            
            y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
            chi2S = sum(((flux.data-y_model_sig)/error)**2)
            BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
            
            chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'OIII')
            
            
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = 3
            
        elif outflow=='outflow_only':
            
            flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow=1,N=N, priors=priors, Hbeta_dual=Hbeta_dual, template=template)
            prop_out = prop_calc(flat_samples_out)
            
            
            self.D1_fit_results = prop_out
            self.D1_fit_chain = flat_samples_out
            self.D1_fit_model = fitted_model_out
            self.z = prop_out['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = 3
        
        elif outflow=='QSO':
            flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow='QSO',N=N,template=template, priors=priors)
            prop_out = prop_calc(flat_samples_out)
            
            self.D1_fit_results = prop_out
            self.D1_fit_chain = flat_samples_out
            self.D1_fit_model = fitted_model_out
            self.z = prop_out['popt'][0]
            self.SNR =  10
            self.SNR_hb = 10
            self.dBIC = 3
        
        elif outflow=='QSO_bkp':
            flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow='QSO_bkp',N=N,template=template, priors=priors)
            prop_out = prop_calc(flat_samples_out)
            
            self.D1_fit_results = prop_out
            self.D1_fit_chain = flat_samples_out
            self.D1_fit_model = fitted_model_out
            self.z = prop_out['popt'][0]
            self.SNR =  10
            self.SNR_hb = 10
            self.dBIC = 3
        
        labels= list(self.D1_fit_results.keys())
        print(labels)
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels[1:],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print(self.SNR)
        print(self.SNR_hb)
        
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_OIII(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error', axres=ax2, template=template)
        
        self.fit_plot = [f,ax1,ax2]  
            
    
    def fitting_collapse_single_G(self,  plot, outflow=0):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        ID = self.ID
        
        flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=0)
        prop_sig = prop_calc(flat_samples_sig)
        
        y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
        chi2S = sum(((flux.data-y_model_sig)/error)**2)
        BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
        
        flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow=1)
        prop_out = prop_calc(flat_samples_out)
        
        chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'OIII')
        chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_out, prop_out, 'OIII')
        
        
        if BICM-BICS <-2:
            print('Delta BIC' , BICM-BICS, ' ')
            self.D1_fit_results = prop_out
            self.D1_fit_chain = flat_samples_out
            self.D1_fit_model = fitted_model_out
            self.z = prop_out['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = BICM-BICS
            
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel', 'Hbeta_peak', 'Hbeta_fwhm')
        
        else:
            print('Delta BIC' , BICM-BICS, ' ')
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = BICM-BICS
            
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm')
         
            
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print(self.SNR)
        print(self.SNR_hb)
        
        
        f, ax1 = plt.subplots(1)
        emplot.plotting_OIII(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model)
        
        g, (ax1a,ax2a) = plt.subplots(2)
        emplot.plotting_OIII(wave, flux, ax1a, prop_sig , fitted_model_sig)
        emplot.plotting_OIII(wave, flux, ax2a, prop_out ,fitted_model_out)
        
        self.fit_plot = [f,ax1]
    
    def save_fit_info(self):
        results = [self.D1_fit_results, self.dBIC, self.SNR]
        
        with open(self.savepath+self.ID+'_'+self.band+'_best_fit.txt', "wb") as fp:
            pickle.dump( results,fp)     
        
        with open(self.savepath+self.ID+'_'+self.band+'_best_chain.txt', "wb") as fp:
            pickle.dump( self.D1_fit_chain,fp)     
        

    def astrometry_correction_HST(self, img_file):
        '''
        Correcting the Astrometry of the Cube. Fits a 2D Gaussian to the HST image and assumes that the 
        HST and Cube centroids are in the same location. 
        '''
        
        
        img=fits.getdata(img_file)
        img_wcs= wcs.WCS(img_file).celestial
        hdr=fits.getheader(img_file)
        
        # Sie of the new image - same as the cube
        new_size = self.phys_size# size of the cube
        # Finding the pixel scale of the HST
        try:
            pixscale=abs(hdr['CD1_1']*3600)
        
        except:
            pixscale=abs(hdr['CDELT1']*3600)
            
        
        # Loading the Catalogue coordinates - Chris sometimes uses ra and sometimes RA
        Cat = self.cat
        try:
            Ra_opt = Cat['ra']
            Dec_opt = Cat['dec']
        
        except:
            Ra_opt = Cat['RA']
            Dec_opt = Cat['DEC']
        
        
        # Finding the position of the Galaxy in pix scale
        opt_world= np.array([[Ra_opt,Dec_opt]])
        opt_pixcrd = img_wcs.wcs_world2pix(opt_world, 0) # WCS transform
        opt_x= (opt_pixcrd[0,0]) # X pixel
        opt_y= (opt_pixcrd[0,1]) # Y pixel
        
        position = np.array([opt_x, opt_y])
    
    
        # Cutting out an image from the bigger image
        cutout = Cutout2D(img, position, new_size/pixscale, wcs=img_wcs,mode='partial')
           
        # Extracting the new image and the new wcs solution 
        img=(cutout.data).copy()
        img_wcs=cutout.wcs
        
        
        
        # To avoid weird things on side of the stamps
        img[np.isnan(img)] = 0
      
        # Finding the dimensions of the new image - need to create new XY grid for the 2D Gaussian 
        shapes = np.array(np.shape(img))
        
        # Finding the initial condition- location of the maximum of the image
        loc = np.where(img == img.max()) # Assumes that our AGN is the brightest thing in the image
        
        initial_guess = (np.max(img),loc[1][0],loc[0][0],5. , 5. ,0,0)
        
        # XID 522 in KMOS is just damn aweful 
        if self.ID == 'XID_522':
            print( 'XID_522: Changing initial conditions for Astrometry corrections')
            initial_guess = (img[34,32],34,32,5. , 5. ,0,0)
            
            print (initial_guess)
            
            
        #print 'Initial guesses', initial_guess
        import scipy.optimize as opt
        
        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)   
        
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), img.ravel(), p0=initial_guess)
        er = np.sqrt(np.diag(pcov))
        print ('HST Cont', popt[1:3])
        print ('HST Cont er', er[1:3])
        popt_HST  = popt.copy()
        # Extrating the XY coordinates of the center. Will be plotted later to check accuracy
        center_C= popt[1:3]
        
        center_C1 = np.zeros(2)
        center_C1[0] = center_C[1]
        center_C1[1] = center_C[0]
        
        # Finding the Accurate HST based optical coordinates
        center_global =  img_wcs.wcs_pix2world(np.array([center_C]), 0)[0] 
        
        cube_center = self.center_data[1:3]    
    
        # Old Header for plotting purposes
        Header_cube_old = self.header.copy()
        
        # New Header 
        Header_cube_new = self.header.copy()
        Header_cube_new['CRPIX1'] = cube_center[0]+1
        Header_cube_new['CRPIX2'] = cube_center[1]+1
        Header_cube_new['CRVAL1'] = center_global[0]
        Header_cube_new['CRVAL2'] = center_global[1]
        
        
        # Saving new coordinates and the new header
        self.HST_cent = center_global
        self.header = Header_cube_new
        
        ###############
        # DO check plots:    
        f  = plt.figure(figsize=(10,7))
        ax = f.add_subplot(121, projection=img_wcs)
        
        rms = np.std(img - img.mean(axis=0))
        
        ax.imshow((img), origin='low')#, vmin=0, vmax=3*rms)
        ax.set_autoscale_on(False)
        ax.plot(center_C[0], center_C[1], 'r*')
        
       
        cube_wcs= wcs.WCS(Header_cube_new).celestial  
        cont_map = self.Median_stack_white
        popt_cube = self.center_data
        
        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)  
        
        
        data_fit = twoD_Gaussian((x,y), *popt_cube)
        
        ax.contour( data_fit.reshape(shapes[0], shapes[1]), levels=(max(data_fit)*0.68,max(data_fit)*0.98),transform= ax.get_transform(cube_wcs), colors='r')
        
        cube_wcs= wcs.WCS(Header_cube_old).celestial  
        
          
        ax.contour( data_fit.reshape(shapes[0], shapes[1]), levels=(max(data_fit)*0.68,max(data_fit)*0.98),transform= ax.get_transform(cube_wcs), colors='g')
          
    
        popt = cube_center
        cube_wcs= wcs.WCS(Header_cube_new).celestial
        
        ax = f.add_subplot(122, projection=cube_wcs)  
        ax.imshow(cont_map, vmin=0, vmax= cont_map[int(popt[1]), int(popt[0])], origin='low')
        
        ax.plot(popt[0], popt[1],'r*')
        
        ax.contour(img, transform= ax.get_transform(img_wcs), colors='red',levels=(popt_HST[0]*0.68,popt_HST[0]*0.95), alpha=0.9)
        
        ax.set_xlim(40,60)
        ax.set_ylim(40,57)
        


    def astrometry_correction_GAIA(self, path_to_gaia):
        '''
        Correcting the Astrometry of the Cube. Fits a 2D Gaussian to the HST image and assumes that the 
        HST and Cube centroids are in the same location. 
        '''
        
        from astropy.table import Table
        Gaia = Table.read(path_to_gaia , format='ipac')
        
        cube_center = self.center_data[1:3]    
    
        # Old Header for plotting purposes
        Header_cube_old = self.header.copy()
        
        # New Header 
        Header_cube_new = self.header.copy()
        Header_cube_new['CRPIX1'] = cube_center[0]+1
        Header_cube_new['CRPIX2'] = cube_center[1]+1
        Header_cube_new['CRVAL1'] = float(Gaia['ra'])
        Header_cube_new['CRVAL2'] = float(Gaia['dec'])
        
        center_global= np.array([float(Gaia['ra']), float(Gaia['dec'])])
        
        # Saving new coordinates and the new header
        self.HST_center_glob = center_global
        self.header = Header_cube_new
    
    def report(self):
        
        results = self.D1_fit_results
        print('')
        print('Model: ', results['name'])
        print('SNR of the line: ', self.SNR)
        print('dBIC of ', self.dBIC)
        
        for key in results.keys():
            
            if key == 'name' or key =='popt':
                print('')
            else:
                print(key, results[key])
                
     
    def unwrap_cube(self, rad=0.4, sp_binning='Nearest', instrument='KMOS', add='', mask_manual=0, binning_pix=1, err_range=[0], boundary=2.4):
        flux = self.flux.copy()
        Mask= self.sky_clipped_1D
        shapes = self.dim
        
        
        ThD_mask = self.sky_clipped.copy()
        z = self.z
        wv_obs = self.obs_wave.copy()
         
        msk = Mask.copy()
        Spax_mask = self.Sky_stack_mask[0,:,:]
        
        if (self.ID=='XID_587'):
            print ('Masking special corners')
            Spax_mask[:,0] = False
            Spax_mask[:,1] = False
            
            Spax_mask[:,-1] = False
            Spax_mask[:,-2] = False
            
            Spax_mask[0,:] = False
            Spax_mask[1,:] = False
            
            Spax_mask[-1,:] = False
            Spax_mask[-2,:] = False
            
# =============================================================================
#   Unwrapping the cube    
# =============================================================================
        try:
            arc = (self.header['CD2_2']*3600)
        
        except:
            arc = (self.header['CDELT2']*3600)
         
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
           
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
        if instrument=='NIRSPEC':
            upper_lim = 0            
            step = 1   
            step = binning_pix
        if instrument=='NIRSPEC05':
            upper_lim = 0            
            step = 2  
            step = binning_pix
        x = range(shapes[0]-upper_lim)
        y = range(shapes[1]-upper_lim) 
        
        
        print(rad/arc)
        h, w = self.dim[:2]
        center= self.center_data[1:3]
        
        mask = create_circular_mask(h, w, center= center, radius= rad/arc)
        mask = np.invert(mask)
        try:
            if mask_manual.all !=1:
                mask=mask_manual
        except:
            print('Circular mask')
        
        Spax_mask = np.logical_or(np.invert(Spax_mask),mask)
        if self.instrument=='NIRSPEC_IFU':
            Spax_mask = mask.copy()
        
        plt.figure()
        
        plt.imshow(np.ma.array(data=self.Median_stack_white, mask=Spax_mask), origin='lower')
        
        Unwrapped_cube = []        
        for i in tqdm.tqdm(x):
            #i= i+step
            
            for j in y:
                #j=j+step
                if Spax_mask[i,j]==False:
                    #print i,j
                    
                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    if sp_binning=='Nearest':
                        Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                    if sp_binning=='Single':
                        Spax_mask_pick[:, i, j] = False
                    
                    if self.instrument=='NIRSPEC_IFU':
                        total_mask = np.logical_or(Spax_mask_pick, self.sky_clipped)
                        flx_spax_t = np.ma.array(data=flux.data,mask=total_mask)
                        
                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=self.sky_clipped_1D)  
                        '''
                        error_calc = np.ma.array(data=self.error_cube, mask=total_mask)
                        error = np.zeros_like(self.obs_wave)
                        
                        error = np.sqrt(np.ma.sum(error_calc**2, axis=(1,2)))
                        #for index in range(len(self.obs_wave)):
                        #    error[index] = np.sqrt(np.ma.sum(error_calc[index,:,:]**2))
                        
                        error[np.isnan(error)] = 0
                        error[error==0] = np.median(error)
                        error = error/np.sqrt(np.where(Spax_mask_pick==False)[0])
                        '''
                        if len(err_range)==2:
                            error = stats.sigma_clipped_stats(flx_spax_m[(err_range[0]<self.obs_wave) \
                                                                         &(self.obs_wave<err_range[1])],sigma=3)[2] \
                                                                    *np.ones(len(flx_spax_m))
                                                                    
                        elif len(err_range)==4:
                            error1 = stats.sigma_clipped_stats(flx_spax_m[(err_range[0]<self.obs_wave) \
                                                                         &(self.obs_wave<err_range[1])],sigma=3)[2]
                            error2 = stats.sigma_clipped_stats(flx_spax_m[(err_range[1]<self.obs_wave) \
                                                                         &(self.obs_wave<err_range[2])],sigma=3)[2]
                            
                            error = np.zeros(len(flx_spax_m))
                            error[self.obs_wave<boundary] = error1
                            error[self.obs_wave>boundary] = error2
                        
                    else:
                        flx_spax_t = np.ma.array(data=flux.data,mask=Spax_mask_pick)
                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)  
                        
                        error = stats.sigma_clipped_stats(flx_spax_m,sigma=3)[2] * np.ones(len(flx_spax))
                        
                    Unwrapped_cube.append([i,j,flx_spax_m, error,wv_obs, z])
                    

        print(len(Unwrapped_cube))
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "wb") as fp:
            pickle.dump(Unwrapped_cube, fp)      
          
    def Spaxel_fitting_OIII_MCMC_mp(self,Ncores=(mp.cpu_count() - 1), priors= {'cont':[0,-3,1],\
                                                  'cont_grad':[0,-0.01,0.01], \
                                                  'OIIIn_peak':[0,-3,1],\
                                                  'OIIIw_peak':[0,-3,1],\
                                                  'OIII_fwhm':[300,100,900],\
                                                  'OIII_out':[700,600,2500],\
                                                  'out_vel':[-200,-900,600],\
                                                  'Hbeta_peak':[0,-3,1],\
                                                  'Hbeta_fwhm':[200,120,800],\
                                                  'Hbeta_vel':[10,-200,200],\
                                                  'Hbetan_peak':[0,-3,1],\
                                                  'Hbetan_fwhm':[300,120,700],\
                                                  'Hbetan_vel':[10,-100,100],\
                                                  'Fe_peak':[0,-3,2],\
                                                  'Fe_fwhm':[3000,2000,6000]}):
        
            
        import pickle
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        obs_wave = self.obs_wave.copy()
        z = self.z
        
        if Ncores<1:
            Ncores=1
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)       
        #for i in range(len(Unwrapped_cube)):   
            #results.append( emfit.Fitting_OIII_unwrap(Unwrapped_cube[i], self.obs_wave, self.z))
        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_OIII_unwrap, Unwrapped_cube )    
        
        
         
        self.spaxel_fit_raw = cube_res
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
    
    def Spaxel_fitting_OIII_2G_MCMC_mp(self, Ncores=(mp.cpu_count() - 1), priors= {'cont':[0,-3,1],\
                                                                    'cont_grad':[0,-0.01,0.01], \
                                                                    'OIIIn_peak':[0,-3,1],\
                                                                    'OIIIw_peak':[0,-3,1],\
                                                                    'OIII_fwhm':[300,100,900],\
                                                                    'OIII_out':[900,600,2500],\
                                                                    'out_vel':[-200,-900,600],\
                                                                    'Hbeta_peak':[0,-3,1],\
                                                                    'Hbeta_fwhm':[200,120,7000],\
                                                                    'Hbeta_vel':[10,-200,200],\
                                                                    'Hbetan_peak':[0,-3,1],\
                                                                    'Hbetan_fwhm':[300,120,700],\
                                                                    'Hbetan_vel':[10,-100,100],\
                                                                    'Fe_peak':[0,-3,2],\
                                                                    'Fe_fwhm':[3000,2000,6000]}):
        import pickle
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
        
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)    
                
        print('import of the unwrap cube - done')
        print(len(Unwrapped_cube))
        
        if Ncores<1:
            Ncores=1
        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_OIII_2G_unwrap, Unwrapped_cube )    
        
        
        self.spaxel_fit_raw = cube_res
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII_2G.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
    
    def Spaxel_fitting_Halpha_MCMC_mp(self, add='',Ncores=(mp.cpu_count() - 1),priors={'cont':[0,-3,1],\
                                                     'cont_grad':[0,-0.01,0.01], \
                                                     'Hal_peak':[0,-3,1],\
                                                     'BLR_peak':[0,-3,1],\
                                                     'NII_peak':[0,-3,1],\
                                                     'Nar_fwhm':[300,100,900],\
                                                     'BLR_fwhm':[4000,2000,9000],\
                                                     'BLR_offset':[-200,-900,600],\
                                                     'SIIr_peak':[0,-3,1],\
                                                     'SIIb_peak':[0,-3,1],\
                                                     'Hal_out_peak':[0,-3,1],\
                                                     'NII_out_peak':[0,-3,1],\
                                                     'outflow_fwhm':[600,300,1500],\
                                                     'outflow_vel':[-50, -300,300]}):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)    
            
        
        if Ncores<1:
            Ncores=1
            
        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_Halpha_unwrap, Unwrapped_cube)
        
        
         
        self.spaxel_fit_raw = cube_res
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))
        
    def Spaxel_fitting_Halpha_OIII_MCMC_mp(self,add='',Ncores=(mp.cpu_count() - 2),BLR='None', priors= {'cont':[0,-3,1],\
                                                          'cont_grad':[0,-10.,10], \
                                                          'OIIIn_peak':[0,-3,1],\
                                                           'OIIIn_fwhm':[300,100,900],\
                                                           'OIII_vel':[-100,-600,600],\
                                                           'Hbeta_peak':[0,-3,1],\
                                                           'Hal_peak':[0,-3,1],\
                                                           'NII_peak':[0,-3,1],\
                                                           'Nar_fwhm':[300,150,900],\
                                                           'SIIr_peak':[0,-3,1],\
                                                           'SIIb_peak':[0,-3,1],\
                                                           'OI_peak':[0,-3,1]}):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        #results = []
        #for i in range(len(Unwrapped_cube)):   
        #    results.append( emfit.Fitting_Halpha_unwrap(Unwrapped_cube[i]))
        #cube_res = results
        
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)     
        
        if Ncores<1:
            Ncores=1
        if BLR=='None':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_unwrap, Unwrapped_cube)
        elif BLR=='BLR':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_AGN_unwrap, Unwrapped_cube)
        
        elif BLR=='outflow_both':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_outflowboth_unwrap, Unwrapped_cube)
        else:
            raise Exception('Mode {BLR} not understood. Options are: None, BLR and outflow_both')
            
        self.spaxel_fit_raw = cube_res
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))
    


    
    def Map_creation_OIII(self, SNR_cut=3, fwhmrange = [100,500], velrange=[-100,100]):
        z0 = self.z
    
        wvo3 = 5008.24*(1+z0)/1e4
        wvhb = 4862.6*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw.txt', "rb") as fp:
            results= pickle.load(fp)
            
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_vel = np.zeros(self.dim[:2])
        map_vel[:,:] = np.nan
        
        map_fwhm = np.zeros(self.dim[:2])
        map_fwhm[:,:] = np.nan
        
        map_flux = np.zeros(self.dim[:2])
        map_flux[:,:] = np.nan
        
        map_snr = np.zeros(self.dim[:2])
        map_snr[:,:] = np.nan
        
        map_snrhb = np.zeros(self.dim[:2])
        map_snrhb[:,:] = np.nan
        
        map_flux_hb= np.zeros(self.dim[:2])
        map_flux_hb[:,:] = np.nan
        
        map_vel_hb = np.zeros(self.dim[:2])
        map_vel_hb[:,:] = np.nan
        
        map_fwhm_hb = np.zeros(self.dim[:2])
        map_fwhm_hb[:,:] = np.nan
        
        # =============================================================================
        #        Filling these maps  
        # =============================================================================
        f,ax= plt.subplots(1)
        import Plotting_tools_v2 as emplot
        
        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_OIII_fit_detection_only.pdf')
        Resid_cube = self.flux.data.copy()
        
        for row in range(len(results)):
            i,j, res_spx = results[row]
            i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]
            
            z = res_spx['popt'][0]
            SNR = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'OIII')
            map_snr[i,j]= SNR
            if (SNR>SNR_cut):
                
                map_vel[i,j] = ((5008.24*(1+z)/1e4)-wvo3)/wvo3*3e5
                map_fwhm[i,j] = res_spx['popt'][4]
                map_flux[i,j] = flux_calc(res_spx, 'OIIIt', self.flux_norm)
                
                
                Resid_cube[:,i,j] = flx_spax_m.data-emfit.OIII(self.obs_wave, *res_spx['popt'])
            else:
                Resid_cube[:,i,j] = flx_spax_m.data
            
            emplot.plotting_OIII(self.obs_wave, flx_spax_m, ax, res_spx, emfit.OIII, error=error)
            ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
            ax.set_xlim(4830,5030)
            plt.tight_layout()
            Spax.savefig()  
            ax.clear()
                
            
            SNRhb = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hb')
            map_snrhb[i,j]= SNRhb
            if (SNRhb>SNR_cut) & (SNRhb<100):
                
                map_vel_hb[i,j] = ((4862.6*(1+z)/1e4)-wvhb)/wvhb*3e5
                map_fwhm_hb[i,j] = res_spx['Hbeta_fwhm'][0]
                map_flux_hb[i,j] = flux_calc(res_spx, 'Hbeta', self.flux_norm)
                
                
                
        Spax.close() 
        
        self.Flux_map = map_flux
        self.Vel_map = map_vel
        self.FWHM_map = map_fwhm
        self.SNR_map = map_snr
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x = int(self.center_data[1]); y= int(self.center_data[2])
        
        
        IFU_header = self.header
        
        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600
        
        
        Offsets_low = -self.center_data[1:3]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3]
        
        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])
    
        lim_sc = lim*arc_per_pix
        
# =============================================================================
#         OIII fitting
# =============================================================================
        f = plt.figure( figsize=(10,10))        
        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])
        
        flx = ax1.imshow(map_flux,vmax=map_flux[y,x], origin='lower', extent= lim_sc)
        ax1.set_title('Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        
        #lims = 
        #emplot.overide_axes_labels(f, axes[0,0], lims)
        
        
        vel = ax2.imshow(map_vel, cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
        ax2.set_title('Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        
        fw = ax3.imshow(map_fwhm,vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')
        
        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y 
        
        Line_info = np.zeros((8,self.dim[0],self.dim[1]))
        Line_info[0,:,:] = map_flux
        Line_info[1,:,:] = map_vel
        Line_info[2,:,:] = map_fwhm
        Line_info[3,:,:] = map_snr
        
        Line_info[4,:,:] = map_flux_hb
        Line_info[5,:,:] = map_vel_hb
        Line_info[6,:,:] = map_fwhm_hb
        Line_info[7,:,:] = map_snrhb
        
        # =============================================================================
        #         Hbeta plotting
        # =============================================================================
        f = plt.figure( figsize=(10,10))
        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])
        
        flx = ax1.imshow(map_flux_hb,vmax=map_flux_hb[y,x], origin='lower', extent= lim_sc)
        ax1.set_title('Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        
        #lims = 
        #emplot.overide_axes_labels(f, axes[0,0], lims)
        
        
        vel = ax2.imshow(map_vel_hb, cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
        ax2.set_title('Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        
        fw = ax3.imshow(map_fwhm_hb,vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        snr = ax4.imshow(map_snrhb,vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')
    
        
        
        
        prhdr = hdr
        hdu = fits.PrimaryHDU(Line_info, header=prhdr)
        hdulist = fits.HDUList([hdu])
        
        
        hdulist.writeto(self.savepath+self.ID+'_OIII_fits_maps.fits', overwrite=True)
        
        hdu = fits.PrimaryHDU(Resid_cube, header=prhdr)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(self.savepath+self.ID+'_OIII_fits_resid_cube.fits', overwrite=True)
        
    def Map_creation_OIII_2G(self):
        z0 = self.z
    
        wvo3 = 5008.24*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII_2G.txt', "rb") as fp:
            results= pickle.load(fp)
            
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_vel = np.zeros(self.dim[:2])
        map_vel[:,:] = np.nan
        
        map_fwhm = np.zeros(self.dim[:2])
        map_fwhm[:,:] = np.nan
        
        map_flux = np.zeros(self.dim[:2])
        map_flux[:,:] = np.nan
        
        map_snr = np.zeros(self.dim[:2])
        map_snr[:,:] = np.nan
        # =============================================================================
        #        Filling these maps  
        # =============================================================================
        f,ax= plt.subplots(1)
        import Plotting_tools_v2 as emplot
        
        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_OIII_fit_detection_only_2G.pdf')
        
        
        for row in range(len(results)):
            i,j, res_spx = results[row]
            i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]
            
            z = res_spx['popt'][0]
            SNR = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'OIII')
            map_snr[i,j]= SNR
            if SNR>3:
                
                map_vel[i,j] = ((5008.24*(1+z)/1e4)-wvo3)/wvo3*3e5
                map_fwhm[i,j] = W80_OIII_calc(emfit.OIII_outflow, res_spx, chains, 0)#res_spx['OIIIn_fwhm'][0]
                map_flux[i,j] = flux_calc(res_spx, 'OIIIt',self.flux_norm)
                
                
                emplot.plotting_OIII(self.obs_wave, flx_spax_m, ax, res_spx, emfit.OIII_outflow)
                ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
                plt.tight_layout()
                Spax.savefig()  
                ax.clear()
          
        Spax.close() 
        
        self.Flux_map = map_flux
        self.Vel_map = map_vel
        self.FWHM_map = map_fwhm
        self.SNR_map = map_snr
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x = int(self.center_data[1]); y= int(self.center_data[2])
        f = plt.figure( figsize=(10,10))
        
        IFU_header = self.header
        
        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600
        
        
        Offsets_low = -self.center_data[1:3]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3]
        
        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])
    
        lim_sc = lim*arc_per_pix
        
        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])
        
        flx = ax1.imshow(map_flux,vmax=map_flux[y,x], origin='lower', extent= lim_sc)
        ax1.set_title('Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        
        #lims = 
        #emplot.overide_axes_labels(f, axes[0,0], lims)
        
        
        vel = ax2.imshow(map_vel, cmap='coolwarm', origin='lower', vmin=-200, vmax=200, extent= lim_sc)
        ax2.set_title('Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        
        fw = ax3.imshow(map_fwhm,vmin=100, origin='lower', extent= lim_sc)
        ax3.set_title('FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')
        
        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y 
        
        Line_info = np.zeros((4,self.dim[0],self.dim[1]))
        Line_info[0,:,:] = map_flux
        Line_info[1,:,:] = map_vel
        Line_info[2,:,:] = map_fwhm
        Line_info[3,:,:] = map_snr
        
        prhdr = hdr
        hdu = fits.PrimaryHDU(Line_info, header=prhdr)
        hdulist = fits.HDUList([hdu])
        
        
        hdulist.writeto(self.savepath+self.ID+'_OIII_fits_maps_2G.fits', overwrite=True)

    def Map_creation_Halpha(self, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, add=''):
        z0 = self.z
    
        wvo3 = 6563*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
            results= pickle.load(fp)
            
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_vel = np.zeros(self.dim[:2])
        map_vel[:,:] = np.nan
        
        map_fwhm = np.zeros(self.dim[:2])
        map_fwhm[:,:] = np.nan
        
        map_flux = np.zeros(self.dim[:2])
        map_flux[:,:] = np.nan
        
        map_snr = np.zeros(self.dim[:2])
        map_snr[:,:] = np.nan
        
        map_nii = np.zeros(self.dim[:2])
        map_nii[:,:] = np.nan
        
        # =============================================================================
        #        Filling these maps  
        # =============================================================================
        gf,ax= plt.subplots(1)
        import Plotting_tools_v2 as emplot
        
        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_Halpha_fit_detection_only.pdf')
        
        
        for row in range(len(results)):
            
            i,j, res_spx = results[row]
            i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]
            
            z = res_spx['popt'][0]
            SNR = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hn')
            map_snr[i,j]= SNR
            if SNR>SNR_cut:
                
                map_vel[i,j] = ((6563*(1+z)/1e4)-wvo3)/wvo3*3e5
                map_fwhm[i,j] = res_spx['popt'][5]
                map_flux[i,j] = flux_calc(res_spx, 'Hat',self.flux_norm)
                map_nii[i,j] = flux_calc(res_spx, 'NIIt', self.flux_norm)
                
                
            emplot.plotting_Halpha(self.obs_wave, flx_spax_m, ax, res_spx, emfit.Halpha, error=error)
            ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
            
            if res_spx['Hal_peak'][0]<3*error[0]:
                ax.set_ylim(-error[0], 5*error[0])
            if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
                ax.set_ylim(-error[0], 5*error[0])
            Spax.savefig()  
            ax.clear()
        plt.close(gf)
        Spax.close() 
        
        self.Flux_map = map_flux
        self.Vel_map = map_vel
        self.FWHM_map = map_fwhm
        self.SNR_map = map_snr
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    
        x = int(self.center_data[1]); y= int(self.center_data[2])
        f = plt.figure( figsize=(10,10))
        
        IFU_header = self.header
        
        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600
        
        
        Offsets_low = -self.center_data[1:3]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3]
        
        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])
    
        lim_sc = lim*arc_per_pix
        
        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])
        
        if flux_max==0:
            flx_max = map_flux[y,x]
        else:
            flx_max = flux_max
        
        smt=0.0000001
        print(lim_sc)
        flx = ax1.imshow(smooth(map_flux,smt),vmax=flx_max, origin='lower', extent= lim_sc)
        ax1.set_title('Halpha Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        cax.set_ylabel('Flux (arbitrary units)')
        ax1.set_xlabel('RA offset (arcsecond)')
        ax1.set_ylabel('Dec offset (arcsecond)')
        
        #lims = 
        #emplot.overide_axes_labels(f, axes[0,0], lims)
        
        
        vel = ax2.imshow(smooth(map_vel,smt), cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
        ax2.set_title('Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        cax.set_ylabel('Velocity (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        
        fw = ax3.imshow(smooth(map_fwhm,smt),vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        cax.set_ylabel('FWHM (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')
        
        cax.set_ylabel('SNR')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        fnii,axnii = plt.subplots(1)
        axnii.set_title('[NII] map')
        fw= axnii.imshow(map_nii, vmax=flx_max ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fnii.colorbar(fw, cax=cax, orientation='vertical')
        
        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y 
        
        Line_info = np.zeros((5,self.dim[0],self.dim[1]))
        Line_info[0,:,:] = map_flux
        Line_info[1,:,:] = map_vel
        Line_info[2,:,:] = map_fwhm
        Line_info[3,:,:] = map_snr
        Line_info[4,:,:] = map_nii
        
        prhdr = hdr
        hdu = fits.PrimaryHDU(Line_info, header=prhdr)
        hdulist = fits.HDUList([hdu])
        
        
        hdulist.writeto(self.savepath+self.ID+'_Halpha_fits_maps.fits', overwrite=True)
        
        return f 
        
    
    def Map_creation_Halpha_OIII(self, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='',modelfce = Halpha_OIII):
        z0 = self.z
        failed_fits=0
        wv_hal = 6564.52*(1+z0)/1e4
        wv_oiii = 5008.24*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
            results= pickle.load(fp)
            
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        
        map_hal = np.zeros((4,self.dim[0], self.dim[1]))
        map_hal[:,:,:] = np.nan
        
        map_nii = np.zeros((4,self.dim[0], self.dim[1]))
        map_nii[:,:,:] = np.nan
        
        map_hb = np.zeros((4,self.dim[0], self.dim[1]))
        map_hb[:,:,:] = np.nan
        
        map_oiii = np.zeros((4,self.dim[0], self.dim[1]))
        map_oiii[:,:,:] = np.nan
        
        map_siir = np.zeros((4,self.dim[0], self.dim[1]))
        map_siir[:,:,:] = np.nan
        
        map_siib = np.zeros((4,self.dim[0], self.dim[1]))
        map_siib[:,:,:] = np.nan
        
        map_hal_ki = np.zeros((4,self.dim[0], self.dim[1]))
        map_hal_ki[:,:,:] = np.nan
        
        map_oiii_ki = np.zeros((5,self.dim[0], self.dim[1]))
        map_oiii_ki[:,:,:] = np.nan
        
        map_oi = np.zeros((4,self.dim[0], self.dim[1]))
        map_oi[:,:,:] = np.nan
        
        # =============================================================================
        #        Filling these maps  
        # =============================================================================
        
        
        import Plotting_tools_v2 as emplot
   
        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_Halpha_OIII_fit_detection_only.pdf')
        
        
        for row in tqdm.tqdm(range(len(results))):
            
            try:
                i,j, res_spx,chains,wave,flx_spax_m,error = results[row]
            except:
                i,j, res_spx,chains= results[row]
                i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]
            
            lists = list(res_spx.keys())
            if 'Failed fit' in lists:
                failed_fits+=1
                continue
            
            z = res_spx['popt'][0]
            
            if 'zBLR' in lists:
                modelfce = Halpha_OIII_BLR
            elif 'outflow_vel' not in lists:
                modelfce = Halpha_OIII
            elif 'outflow_vel' in lists and 'zBLR' not in lists:
                modelfce = Halpha_OIII_outflow
            
# =============================================================================
#             Halpha
# =============================================================================
            
            flux_hal, p16_hal,p84_hal = flux_calc_mcmc(res_spx, chains, 'Hat', self.flux_norm)
            SNR_hal = flux_hal/p16_hal
            map_hal[0,i,j]= SNR_hal
            
            SNR_hal = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hn')
            SNR_oiii = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'OIII')
            SNR_nii = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'NII')
            SNR_hb = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hb')
            
            if SNR_hal>SNR_cut:
                map_hal[1,i,j] = flux_hal.copy()
                map_hal[2,i,j] = p16_hal.copy()
                map_hal[3,i,j] = p84_hal.copy()
                
                if 'Hal_out_peak' in list(res_spx.keys()):
                    map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = W80_Halpha_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0] 
                    
                else:
                    map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = W80_Halpha_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0] 
                    
                
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                map_hal[3,i,j] = -SNR_cut*error[-1]*dl*np.sqrt(n)
                
            
                
            
# =============================================================================
#             Plotting 
# =============================================================================
            f = plt.figure(figsize=(10,4))
            baxes = brokenaxes(xlims=((4800,5050),(6250,6350),(6500,6800)),  hspace=.01)
            emplot.plotting_Halpha_OIII(self.obs_wave, flx_spax_m, baxes, res_spx, modelfce)
            
            #if res_spx['Hal_peak'][0]<3*error[0]:
            #    baxes.set_ylim(-error[0], 5*error[0])
            #if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
            #    baxes.set_ylim(-error[0], 5*error[0])
            
            SNRs = np.array([SNR_hal])
            
# =============================================================================
#             NII
# =============================================================================
            #SNR = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'NII')
            flux_NII, p16_NII,p84_NII = flux_calc_mcmc(res_spx, chains, 'NIIt', self.flux_norm)
            
            map_nii[0,i,j]= SNR_nii
            if SNR_nii>SNR_cut:
                map_nii[1,i,j] = flux_NII.copy()
                map_nii[2,i,j] = p16_NII.copy()
                map_nii[3,i,j] = p84_NII.copy()
            
            else:
                
                
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                map_nii[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
# =============================================================================
#             OIII
# =============================================================================
            flux_oiii, p16_oiii,p84_oiii = flux_calc_mcmc(res_spx, chains, 'OIIIt', self.flux_norm)
            
            map_oiii[0,i,j]= SNR_oiii
            
            if SNR_oiii>SNR_cut:
                map_oiii[1,i,j] = flux_oiii.copy()
                map_oiii[2,i,j] = p16_oiii.copy()
                map_oiii[3,i,j] = p84_oiii.copy()
                
                
                if 'OIII_out_peak' in list(res_spx.keys()):
                    map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = W80_OIII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0] 
                    
                else:
                    map_oiii_ki[0,i,j] = ((5008.24*(1+z)/1e4)-wv_oiii)/wv_oiii*3e5
                    map_oiii_ki[1,i,j] = res_spx['Nar_fwhm'][0]
                    map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = W80_OIII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0] 
                    
                    
            else:
                
                
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(5008.24*(1+self.z)/1e4)/dl
                map_oiii[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)
             
# =============================================================================
#             Hbeta
# =============================================================================
            flux_hb, p16_hb,p84_hb = flux_calc_mcmc(res_spx, chains, 'Hbeta', self.flux_norm)

            map_hb[0,i,j]= SNR_hb.copy()
            if SNR_hb>SNR_cut:
                map_hb[1,i,j] = flux_hb.copy()
                map_hb[2,i,j] = p16_hb.copy()
                map_hb[3,i,j] = p84_hb.copy()
            
            else:
            
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(4860*(1+self.z)/1e4)/dl
                map_hb[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)

# =============================================================================
#             OI
# =============================================================================
            if 'OI_peak' in list(res_spx.keys()):  
                flux_oi, p16_oi,p84_oi = flux_calc_mcmc(res_spx, chains, 'OI', self.flux_norm)
                SNR_oi=  flux_oi/p16_oi
                SNRs = np.append(SNRs, SNR_oi)
                map_oi[0,i,j]= SNR_oi.copy()
                if SNR_oi>SNR_cut:
                    map_oi[1,i,j] = flux_oi.copy()
                    map_oi[2,i,j] = p16_oi.copy()
                    map_oi[3,i,j] = p84_oi.copy()
                
                else:
                    
                    dl = self.obs_wave[1]-self.obs_wave[0]
                    n = width_upper/3e5*(6300*(1+self.z)/1e4)/dl
                    map_oi[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
                  
# =============================================================================
#           SII
# =============================================================================
            fluxr, p16r,p84r = flux_calc_mcmc(res_spx, chains, 'SIIr', self.flux_norm)
            fluxb, p16b,p84b = flux_calc_mcmc(res_spx, chains, 'SIIb', self.flux_norm)

            SNR_SII = SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'SII')
            
            if SNR_SII>SNR_cut:
                map_siir[0,i,j] = SNR_SII.copy()
                map_siib[0,i,j] = SNR_SII.copy()
                
                map_siir[1,i,j] = fluxr.copy()
                map_siir[2,i,j] = p16r.copy()
                map_siir[3,i,j] = p84r.copy()
                
                map_siib[1,i,j] = fluxb.copy()
                map_siib[2,i,j] = p16b.copy()
                map_siib[3,i,j] = p84b.copy()
            
            else:
                
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6731*(1+self.z)/1e4)/dl
                map_siir[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
                map_siib[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
                



            baxes.set_title('xy='+str(j)+' '+ str(i) + ', SNR = '+ str(np.round([SNR_hal, SNR_oiii, SNR_nii, SNR_SII],1)))
            baxes.set_xlabel('Restframe wavelength (ang)')
            baxes.set_ylabel(r'$10^{-16}$ ergs/s/cm2/mic')
            wv0 = 5008.24*(1+z0)
            wv0 = wv0/(1+z)
            baxes.vlines(wv0, 0,10, linestyle='dashed', color='k')
            Spax.savefig()  
            plt.close(f)
        
        print('Failed fits', failed_fits)
        Spax.close() 
# =============================================================================
#         Calculating Avs
# =============================================================================
        Av = Av_calc(map_hal[1,:,:],map_hb[1,:,:])
# =============================================================================
#         Plotting maps
# =============================================================================
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        x = int(self.center_data[1]); y= int(self.center_data[2])
        IFU_header = self.header
        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600
        
        Offsets_low = -self.center_data[1:3]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3]
        
        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])
    
        lim_sc = lim*arc_per_pix
        
        if flux_max==0:
            flx_max = map_hal[y,x]
        else:
            flx_max = flux_max
        
        smt=0.0000001
        print(lim_sc)
        
# =============================================================================
#         Plotting Stuff
# =============================================================================
        f,axes = plt.subplots(6,3, figsize=(10,20))
        ax1 = axes[0,0]
        # =============================================================================
        # Halpha SNR
        snr = ax1.imshow(map_hal[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax1.set_title('Hal SNR map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')
        ax1.set_xlabel('RA offset (arcsecond)')
        ax1.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # Halpha flux
        ax1 = axes[0,1]
        flx = ax1.imshow(map_hal[1,:,:],vmax=flx_max, origin='lower', extent= lim_sc)
        ax1.set_title('Halpha Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        cax.set_ylabel('Flux (arbitrary units)')
        ax1.set_xlabel('RA offset (arcsecond)')
        ax1.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # Halpha  velocity
        ax2 = axes[0,2]
        vel = ax2.imshow(map_hal_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
        ax2.set_title('Hal Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        cax.set_ylabel('Velocity (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # Halpha fwhm
        ax3 = axes[1,2]
        fw = ax3.imshow(map_hal_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('Hal FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        cax.set_ylabel('FWHM (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # [NII] SNR
        axes[1,0].set_title('[NII] SNR')
        fw= axes[1,0].imshow(map_nii[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[1,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[1,0].set_xlabel('RA offset (arcsecond)')
        axes[1,0].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # [NII] flux
        axes[1,1].set_title('[NII] map')
        fw= axes[1,1].imshow(map_nii[1,:,:] ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[1,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[1,1].set_xlabel('RA offset (arcsecond)')
        axes[1,1].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # Hbeta] SNR
        axes[2,0].set_title('Hbeta SNR')
        fw= axes[2,0].imshow(map_hb[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[2,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[2,0].set_xlabel('RA offset (arcsecond)')
        axes[2,0].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # Hbeta flux
        axes[2,1].set_title('Hbeta map')
        fw= axes[2,1].imshow(map_hb[1,:,:] ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[2,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[2,1].set_xlabel('RA offset (arcsecond)')
        axes[2,1].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # [OIII] SNR
        axes[3,0].set_title('[OIII] SNR')
        fw= axes[3,0].imshow(map_oiii[0,:,:],vmin=3, vmax=20,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[3,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[3,0].set_xlabel('RA offset (arcsecond)')
        axes[3,0].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # [OIII] flux
        axes[3,1].set_title('[OIII] map')
        fw= axes[3,1].imshow(map_oiii[1,:,:] ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[3,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[3,1].set_xlabel('RA offset (arcsecond)')
        axes[3,1].set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # OIII  velocity
        ax2 = axes[2,2]
        vel = ax2.imshow(map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
        ax2.set_title('OIII Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')
        
        cax.set_ylabel('Velocity (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # OIII fwhm
        ax3 = axes[3,2]
        fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('OIII FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        cax.set_ylabel('FWHM (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')
        
        # =============================================================================
        # OI SNR
        ax3 = axes[4,0]
        ax3.set_title('OI SNR')
        fw = ax3.imshow(map_oi[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        # =============================================================================
        # OI flux
        ax3 = axes[4,1]
        ax3.set_title('OI Flux')
        fw = ax3.imshow(map_oi[1,:,:], origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        # =============================================================================
        # SII SNR
        ax3 = axes[5,0]
        ax3.set_title('[SII] SNR')
        fw = ax3.imshow(map_siir[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        # =============================================================================
        # SII Ratio
        ax3 = axes[5,1]
        ax3.set_title('[SII]r/[SII]b')
        fw = ax3.imshow(map_siir[1,:,:]/map_siib[1,:,:] ,vmin=0.3, vmax=1.5, origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        
        
        plt.tight_layout()
        
        self.map_hal = map_hal
        
        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y 
        
        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=self.header)
        hal_hdu = fits.ImageHDU(map_hal, name='Halpha')
        nii_hdu = fits.ImageHDU(map_nii, name='NII')
        hbe_hdu = fits.ImageHDU(map_hb, name='Hbeta')
        oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
        oi_hdu = fits.ImageHDU(map_oi, name='OI')
        
        siir_hdu = fits.ImageHDU(map_siir, name='SIIr')
        siib_hdu = fits.ImageHDU(map_siib, name='SIIb')
        
        hal_kin_hdu = fits.ImageHDU(map_hal_ki, name='Hal_kin')
        oiii_kin_hdu = fits.ImageHDU(map_oiii_ki, name='OIII_kin')
        Av_hdu = fits.ImageHDU(Av, name='Av')
        
        hdulist = fits.HDUList([primary_hdu, hal_hdu, nii_hdu, hbe_hdu, oiii_hdu,hal_kin_hdu,oi_hdu,siir_hdu,oiii_kin_hdu, siib_hdu, Av_hdu ])
        hdulist.writeto(self.savepath+self.ID+'_Halpha_OIII_fits_maps.fits', overwrite=True)
        
        return f 
    
    def Regional_Spec(self, center, rad, err_range=None, manual_mask=None, boundary=None):
        
        center =  center
        shapes = self.dim
        
        print ('Center of cont', center)
        print ('Extracting spectrum from diameter', rad*2, 'arcseconds')
        
        # Creating a mask for all spaxels. 
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        print('Pixel scale:', arc)
        print ('radius ', arc*rad)
        
        # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc*rad:
                    mask_catch[:,ix,iy] = False

        # Loading mask of the sky lines an bad features in the spectrum 
        mask_sky_1D = self.sky_clipped_1D.copy()
        
        if self.instrument=='NIRSPEC_IFU':
            total_mask = np.logical_or( mask_catch, self.sky_clipped)
        else:
            total_mask = np.logical_or( mask_catch, self.flux.mask)
        # M
        flux = np.ma.array(data=self.flux.data, mask= total_mask) 
        
        D1_spectrum = np.ma.sum(flux, axis=(1,2))
        D1_spectrum = np.ma.array(data = D1_spectrum.data, mask=mask_sky_1D)
        
        if self.instrument=='NIRSPEC_IFU':
            if len(err_range)==2:
                
                D1_spectrum_er = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) &(self.obs_wave<err_range[1])],sigma=3)[2]*np.ones(len(D1_spectrum))   
            elif len(err_range) ==4:
                error1 = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) \
                                                              &(self.obs_wave<err_range[1])],sigma=3)[2]
                
                error2 = stats.sigma_clipped_stats(D1_spectrum[(err_range[1]<self.obs_wave) \
                                                              &(self.obs_wave<err_range[2])],sigma=3)[2]
                    
                D1_spectrum_er = np.zeros(len(D1_spectrum))
                D1_spectrum_er[self.obs_wave<boundary] = error1
                D1_spectrum_er[self.obs_wave>boundary] = error2
        else:  
            D1_spectrum_er = stats.sigma_clipped_stats(D1_spectrum,sigma=3)[2]*np.ones(len(D1_spectrum)) #STD_calc(wave/(1+self.z)*1e4,self.D1_spectrum, self.band)* np.ones(len(self.D1_spectrum))
        
        
        return D1_spectrum, D1_spectrum_er, mask_catch
        
        
           