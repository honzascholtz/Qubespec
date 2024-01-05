#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 12:22:03 2022

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

nan= float('nan')

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


OIIIr = 5008.24
OIIIb = 4960.3
Hal = 6564.52
NII_r = 6585.27
NII_b = 6549.86
Hbe = 4862.6

SII_r = 6731
SII_b = 6718.29

from astropy.modeling.powerlaws import PowerLaw1D

def test():
    x=1

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
    """ Legacy - old version of finding 16th and 84th percintile
	
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
    """ Legacy - Now part of Fitting class.  Take the dictionary with the results chains and calculates the values 
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
        

def SNR_calc(wave,flux, error, dictsol, mode, wv_cent=5008, peak_name='', fwhm_name=''):
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
    keys = list(dictsol.keys())
    
    if mode=='general':
        center = wv_cent*(1+dictsol['z'][0])/1e4
        fwhm = dictsol[fwhm_name][0]/3e5*center
        contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
        model = gauss(wave, dictsol[peak_name][0],center, fwhm/2.215)
    elif mode =='OIII':
        center = OIIIr*(1+sol[0])/1e4
        if 'outflow_fwhm' in keys:
            fwhms = sol[5]/3e5*center
            fwhm = dictsol['outflow_fwhm'][0]/3e5*center
            
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
        std = error[use]
        
        n = len(use)
        SNR = np.nansum(flux_l)/np.sqrt(np.nansum(std**2))
        
        if SNR < 0:
            SNR=0
        
        return SNR
    
    else:
        raise Exception('Sorry mode in SNR_calc not understood')
    
    use = np.where((wave< center+fwhm*1)&(wave> center-fwhm*1))[0] 
    flux_l = model[use]
    std = error[use]
    
    n = len(use)
    SNR =np.nansum(flux_l)/np.sqrt(np.nansum(std**2))
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
        chi2 = np.nansum(((flux-y_model)/error)**2)
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
        chi2 = np.nansum(((flux-y_model)/error)**2)
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
        chi2 = np.nansum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
       
    return chi2, BIC

def unwrap_chain(res):
    keys = list(res.keys())[1:]  
    chains = np.zeros(( len(res[keys[0]]), len(keys) ))  
    for i in range(len(keys)):  
        chains[:,i] = res[keys[i]]    
    return chains

def QFitsview_mask(filepath):
    mask_load = pyfits.getdata(filepath)
    mask = ~mask_load
    mask[mask==-1] = 1
    mask[mask==-2] = 0
    return mask

def flux_calc_general(wv_cent, res, fwhm_name, peak_name):
    mu = wv_cent*(1+res['z'][0])/1e4
    FWHM = res[fwhm_name][0]
    a = 1./(2*(FWHM/3e5*mu/2.35482)**2)
    return res[peak_name][0]*np.sqrt(np.pi/a)


def flux_calc(res, mode, norm=1e-13, wv_cent=5008, peak_name='', fwhm_name='', ratio_name=''):
    keys = list(res.keys())
    
    if mode=='general':
        if ratio_name=='':
            ratio=1
        else:
            ratio=res[ratio_name][0]
        flx =  ratio*flux_calc_general(wv_cent, res, fwhm_name, peak_name)
        return flx*norm
    
    elif mode=='OIIIt':
        flx =  flux_calc_general(OIIIr, res, 'Nar_fwhm', 'OIII_peak')
        if 'OIII_out_peak' in keys:  
            flx +=  flux_calc_general(OIIIr, res, 'outflow_fwhm', 'OIII_out_peak')      
        return flx*norm
            
            
    elif mode=='OIIIn':
        flx =  flux_calc_general(OIIIr, res, 'Nar_fwhm', 'OIII_peak')
        return flx*norm
        
    elif mode=='OIIIw':
        if 'OIII_out_peak' in keys:
            flx =  flux_calc_general(OIIIr, res, 'outflow_fwhm', 'OIII_out_peak')  
            return flx*norm
        else:
            return 0
    
    elif mode=='Hat':
        flx =  flux_calc_general(Hal, res, 'Nar_fwhm', 'Hal_peak')
        if 'outflow_fwhm' in list(res.keys()):
            flx +=  flux_calc_general(Hal, res, 'outflow_fwhm', 'Hal_out_peak')
        return flx*norm
    
    elif mode=='Han':
        flx = flux_calc_general(Hal, res, 'Nar_fwhm', 'Hal_peak')
        return flx*norm
        
    elif mode=='Hal_BLR':
        if 'BLR_fwhm' in keys:
            flx = flux_calc_general(Hal, res, 'BLR_fwhm', 'BLR_Hal_peak')
            return flx*norm
            
        elif 'BLR_alp1' in keys:
            from .Models.QSO_models import BKPLG
            wave = np.linspace(6300,6700,700)*(1+res['z'][0])/1e4
            model = BKPLG(wave, res['BLR_peak'][0], Hal*(1+res['z'][0])/1e4, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
            
        else:
            return 0
    
    elif mode=='NIIt':
        flx = flux_calc_general(NII_r, res, 'Nar_fwhm', 'NII_peak')
        
        if 'outflow_fwhm' in list(res.keys()):
            flx +=  flux_calc_general(NII_r, res, 'outflow_fwhm', 'NII_out_peak')
        return flx*norm
    
    elif mode=='NII':
        flx = flux_calc_general(NII_r, res, 'Nar_fwhm', 'NII_peak')
        return flx*norm
              
    elif mode=='NIIo':
        flx = flux_calc_general(NII_r, res, 'outflow_fwhm', 'NII_out_peak')
        return flx*norm
        
    elif mode=='Hbeta':     
        try:
            flx = flux_calc_general(Hbe, res, 'Nar_fwhm', 'Hbeta_peak')
        except:
            flx = flux_calc_general(Hbe, res, 'Hbeta_fwhm', 'Hbeta_peak')
        return flx*norm
    
    elif mode=='Hbe_BLR':
        if 'BLR_fwhm' in keys:
            flx = flux_calc_general(Hbe, res, 'BLR_fwhm', 'BLR_Hbeta_peak')
            return flx*norm
        
        elif 'BLR_alp1' in keys:
            wave = np.linspace(4800,4900,700)*(1+res['z'][0])/1e4
            from .Models.QSO_models import BKPLG
            model = BKPLG(wave, res['BLR_peak'][0], Hbe, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
        else:
            return 0 
        
    elif mode=='Hbetaw':
        flx = flux_calc_general(Hbe, res, 'Hbeta_fwhm', 'Hbeta_peak')
        return flx*norm
    elif mode=='Hbetan':
        flx = flux_calc_general(Hbe, res, 'Hbetan_fwhm', 'Hbetan_peak')
        return flx*norm  
    
    elif mode=='SIIr':
        flx = flux_calc_general(6732, res, 'Nar_fwhm', 'SIIr_peak')
        return flx*norm
        
    elif mode=='SIIb':
        flx = flux_calc_general(6718, res, 'Nar_fwhm', 'SIIb_peak') 
        return flx*norm
    
    else:
        raise Exception('Sorry mode in Flux calc not understood')
        
    import scipy.integrate as scpi
    
    Flux = scpi.simps(model, wave)*norm
        
    return Flux

import random
def flux_calc_mcmc(fit_obj, mode, norm=1, N=2000, wv_cent=5008, peak_name='', fwhm_name='', ratio_name=''):
    """
    Calculates flux and 68% confidence iterval. 

    Parameters
    ----------

        fit_obj - object
            Fitting class object
        
        mode - string
            modes: general, OIIIn, OIIIw, OIIIt, Han, NII, Hbeta, SIIr, SIIb
        
        norm - value
            normalization used in the QubeSpec cube class 

        N - int
            number of sampling of the chains
        
        wv_cent - float
            rest-frame wavelength in ang of the line if mode='general'
        
        peak_name - string
          if mode='general' name of the peak name to use

        fwhm_name - string
            if mode='general' name of the fwhm name to use

        ratio_name - string
            if mode='general' name of the ratio to use (e.g. in [OII])

    Returns
    -------

    array of median value and +- 1sigma
    """
    chains = fit_obj.chains
    res = fit_obj.props
    labels = list(chains.keys())

    popt = np.zeros_like(res['popt'])
    Fluxes = []
    res_new = {'name': res['name']}
    
    Nchain = len(chains['z'])
    itere = np.arange(Nchain/2,Nchain,1, dtype=int)
        
    for j in itere:
        #sel = random.randint(Nchain/2,N-1)
        for i in range(len(popt)): 
            
            popt[i] = chains[labels[i+1]][j]
            res_new[labels[i+1]] = [popt[i], 0,0 ]
        
        res_new['popt'] = popt
        Fluxes.append(flux_calc(res_new, mode,norm, wv_cent=wv_cent, peak_name=peak_name, fwhm_name=fwhm_name, ratio_name=ratio_name))
    
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
    
    if 'outflow_fwhm' in sol:
        OIIIr = 5008.24*(1+popt[0])/1e4
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*OIIIr
        fwhmws = np.random.choice(chains['outflow_fwhm'], N)/3e5/2.35*OIIIr
        
        OIIIrws = cent + np.random.choice(chains['outflow_vel'], N)/3e5*OIIIr
        
        peakn = np.random.choice(chains['OIII_peak'], N)
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
    
    elif 'Nar_fwhm' in sol:
        OIIIr = 5008.24*(1+popt[0])/1e4
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*OIIIr
        fwhmws = np.random.choice(chains['outflow_fwhm'], N)/3e5/2.35*OIIIr
        
        OIIIrws = OIIIr + np.random.choice(chains['outflow_vel'], N)/3e5*OIIIr
        
        peakn = np.random.choice(chains['OIII_peak'], N)
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
        
        fwhms = np.random.choice(chains['Nar_fwhm'], N)/3e5/2.35*OIIIr
        peakn = np.random.choice(chains['OIII_peak'], N)
        
        
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


def W80_OIII_calc_single( function, sol, plot, z=0, peak=0):
    popt = sol['popt']  
    
    if z==0:
        z = popt[0]
  
    
    import scipy.integrate as scpi
    
    cent =  5008.24*(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    
    OIIIr = 5008.24*(1+sol['z'][0])/1e4
    
    fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
    peakn = sol['OIII_peak'][0]
    if 'out_vel_n2' in sol:  
        fwhmws = sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
        OIIIrws = OIIIr + sol['outflow_vel'][0]/3e5*OIIIr
        peakw = sol['OIII_out_peak'][0]
        
        peakn2 = sol['OIIIn2_peak'][0]
        out_vel_wv_n2 = sol['out_vel_n2'][0]/3e5*OIIIr
        fwhmn2 = sol['OIII_fwhm_n2'][0]/3e5/2.35*OIIIr
        
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws) + gauss(wvs, peakn2, OIIIr +out_vel_wv_n2, fwhmn2)
    
    elif 'outflow_fwhm' in sol:  
        fwhmws = sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
        OIIIrws = OIIIr + sol['outflow_vel'][0]/3e5*OIIIr
        peakw = sol['OIII_out_peak'][0]
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
    
    else:
        y = gauss(wvs, peakn,OIIIr, fwhms) 
        
      
    peak_wv = wvs[np.argmax(y)]
    peak_vel = (peak_wv-cent)/cent*3e5
        
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
    
    if plot==1:
        plt.figure()
        plt.plot(wvs, y)
    
    if peak==0:
        return v10,v90, w80, v50
    else:
        return v10,v90, w80, v50, peak_vel




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
        Halc = 6564.52*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*Halc
        fwhmws =sol['outflow_fwhm'][0]/3e5/2.35*Halc
        
        Halcw = Halc + sol['outflow_vel'][0]/3e5*Halc
        
        peakn = sol['Hal_peak'][0]
        peakw = sol['Hal_out_peak'][0]
        
        y = gauss(wvs, peakn,Halc, fwhms) + gauss(wvs, peakw, Halcw, fwhmws)
         
    else:
        Halc = 6564.52*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*Halc
        peakn = sol['Hal_peak'][0]
        
        
        y = gauss(wvs, peakn,Halc, fwhms)
            
         
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
    
    cent = 6585.27*(1+z)/1e4
    
    bound1 =  cent + 2000/3e5*cent
    bound2 =  cent - 2000/3e5*cent
    Ni = 500
    
    wvs = np.linspace(bound2, bound1, Ni)
    
    
    if 'outflow_fwhm' in sol:
        NIIr = 6585.27*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*NIIr
        fwhmws =sol['outflow_fwhm'][0]/3e5/2.35*NIIr
        
        NIIrws = NIIr + sol['outflow_vel'][0]/3e5*NIIr
        
        peakn = sol['NII_peak'][0]
        peakw = sol['NII_out_peak'][0]
        
        y = gauss(wvs, peakn,NIIr, fwhms) + gauss(wvs, peakw, NIIrws, fwhmws)
         
    else:
        NIIr = 6585.27*(1+sol['z'][0])/1e4
        
        fwhms = sol['Nar_fwhm'][0]/3e5/2.35*NIIr
        peakn = sol['NII_peak'][0]
        
        
        y = gauss(wvs, peakn,NIIr, fwhms)
            
         
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
    
    if plot==1:
        plt.figure()
        plt.plot(wvs, y)
        plt.vlines(wv10, 0,0.1)
        plt.vlines(wv90, 0,0.1)
        plt.vlines(wv50, 0,0.1)
           
    return v10,v90, w80, v50


def flux_to_lum(flux,redshift):
    from astropy.cosmology import WMAP9 as cosmo
    import astropy.units as u 
    lum = (flux*u.erg/u.s/u.cm**2) * 4*np.pi*(cosmo.luminosity_distance(redshift))**2  
    return lum.to(u.erg/u.s)

def jadify(object_name, disp_filt, wave, flux, err=None, mask=None, verbose=True,
    overwrite=True, descr=None, author=None):
    """
    object_name : int or str **do not use "_" characters
        unique identifier. Preferrably up to eight digits. String is acceptable but
        frowned upon.
    disp_filt : string
        one of 'prism_clear', 'g140m_f070lp', 'g235m_f170lp', 'g395m_f290lp'
    wave : array or any other iterable
        in [um]
    flux : array or any other iterable. Same shape as `wave`
        in [W/m^3]
    err  : array or any other iterable. Same shape as `wave`; [optional]
        in [W/m^3]
    mask : array or any other iterable. Same sahpe as `wave`; [optional]
        0 means good pixel; 1 bad pixel
    verbose : bool
        Suppress messages
    overwrite : bool
        Force overwrite existing fits file. Ask for confirmation otherwise.
    descr : str [optional]
        Any text to add to the comments. Added to the primary HDU under 'comments'
    author : str [optional]
        Only if you want to. Added to the primary HDU under 'author'
    """
    import os
    from astropy.io import fits
    if verbose:
        print('Reminder (suppress with verbose=False):\n'
            '`wave`: [um]; in observed frame\n'
            '`flux`: [W/m^3]; 1 erg/(s cm^2 AA) = 1e7 W/m^3\n'
            '`err` : [W/m^3]; optional. If not given all 0\n'
            '`mask`: [0=good, 1=bad]; optional. If not given all 0\n'
            )
    if any(wave<0.5) or any(wave>6):
        print(f'{wave} wavelength vector seems strange. Are you sure it is in um?')
    weird_wave_disp = (
        (disp_filt=='g140m_f070lp' and (any(wave<0.5) or any(wave>2))) # Not in G140M
        or (disp_filt=='g235m_f170lp' and (any(wave<1.5) or any(wave>3.5))) # Not in G235M
        or (disp_filt=='g395m_f290lp' and (any(wave<2.5) or any(wave>6)))   # Not in G395M
        )
    if weird_wave_disp:
        print(f'{wave} wavelength vector seems strange for {disp_filt}. Are you sure it the right disperser/filter combination?')
        
    # Open dummy file.
    output_filename = f'{object_name}_{disp_filt}_v3.0_1D.fits'
    from . import jadify_temp as pth

    PATH_TO_jadify = pth.__path__[0]+ '/'
    filename = PATH_TO_jadify+ 'temp_prism_clear_v3.0_extr3_1D.fits'

    with fits.open(filename) as hdu:
        hdu['DATA'].data = flux

        hdu['ERR'].data  = (err if err is not None else np.zeros_like(flux))
        hdu['DIRTY_Data'].data = flux
        hdu['DIRTY_QUALITY'].data = (mask if mask is not None else np.zeros(flux.size, dtype=int))
        hdu['WAVELENGTH'].data = wave/1e6
        hdu['GTO_FLAG'].data = np.zeros_like(flux)
        hdu['GTO_OVERLAPPING'].data = np.zeros_like(flux)

        if descr is not None:
            hdu[0].header['COMMENT'] = str(descr)

        if os.path.isfile(output_filename) and (overwrite is False):
            proceed = input(f'{output_filename} exists and {overwrite=}. Enter y to overwrite\n')
            if proceed is not 'y': 
                print('Aborted by user')
                return
         
        # Never reach this if overwrite=False and file exists
        hdu.writeto(output_filename, overwrite=True)


def NIRSpec_IFU_PSF(wave):
    # From D'Eugenio et al 2023 - stellar kinematics
    sigma1= 0.12 + 1.9*wave * e**(-24.4/wave)
    sigma2= 0.09 + 0.2*wave * e**(-12.5/wave)                     
    return np.array([sigma1,sigma2])

def pickle_load(file_path):
    import pickle
    with open(file_path, "rb") as fp:
        return pickle.load(fp)

def pickle_save(file_path, stuff):
    import pickle
    with open(file_path, "wb") as fp:
        pickle.dump(stuff, fp)

def error_scaling(obs_wave,flux, error_var, err_range, boundary, exp=0):
    error= np.zeros_like(flux)
    from astropy import stats

    if len(err_range)==2:
        error1 = stats.sigma_clipped_stats(flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3)[2]
        
        average_var1 = stats.sigma_clipped_stats(error_var[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3)[1]
        error = error_var*(error1/average_var1)

    elif len(err_range)==4:
        error1 = stats.sigma_clipped_stats(flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3)[2]
        error2 = stats.sigma_clipped_stats(flux[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])],sigma=3)[2]
        
        average_var1 = stats.sigma_clipped_stats(error_var[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3)[1]
        average_var2 = stats.sigma_clipped_stats(error_var[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])],sigma=3)[1]
        
        error[obs_wave<boundary] = error_var[obs_wave<boundary]*(error1/average_var1)
        error[obs_wave>boundary] = error_var[obs_wave>boundary]*(error2/average_var2)
    else:
        error1 = stats.sigma_clipped_stats(flux,sigma=3)[2]
                
        average_var1 = stats.sigma_clipped_stats(flux,sigma=3)[1]
        error = error_var/(error1/average_var1)
            
    error[error==0] = np.mean(error)*10

    if exp==1:
        try:
            print('Error rescales are: ', error1/average_var1, error2/average_var2 )
        except:
            print('Error rescale is: ', error1/average_var1 )

    return error

def where(array, lmin, lmax):
    use = np.where( (array>lmin) & (array<lmax))
    return use