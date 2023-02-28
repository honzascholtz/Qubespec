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
        if 'OIII_out_peak' in keys:
            o3 = 5008.24*(1+res['z'][0])/1e4
            
            o3n = gauss(wave, res['OIII_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
            o3w = gauss(wave, res['OIII_out_peak'][0], o3, res['outflow_fwhm'][0]/2.355/3e5*o3  )*1.333
            
            model = o3n+o3w
            
        else:
            o3 = 5008.24*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIII_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
            
            
    elif mode=='OIIIn':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        
        o3 = 5008.24*(1+res['z'][0])/1e4
        model = gauss(wave, res['OIII_peak'][0], o3, res['Nar_fwhm'][0]/2.355/3e5*o3  )*1.333
        
    elif mode=='OIIIw':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if 'OIII_out_peak' in keys:
            o3 = 5008.24*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIII_out_peak'][0], o3, res['outflow_fwhm'][0]/2.355/3e5*o3  )*1.333
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
            try:
                model = gauss(wave, res['BLR_peak'][0], hn, res['BLR_fwhm'][0]/2.355/3e5*hn  )
                
            except:
                model = gauss(wave, res['BLR_Hal_peak'][0], hn, res['BLR_fwhm'][0]/2.355/3e5*hn  )
            
        elif 'BLR_alp1' in keys:
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
        if 'BLR_fwhm' in keys:
            try:
                model = gauss(wave, res['BLR_peak'][0], hbeta, res['BLR_fwhm'][0]/2.355/3e5*hn  )
                
            except:
                model = gauss(wave, res['BLR_Hbeta_peak'][0], hbeta, res['BLR_fwhm'][0]/2.355/3e5*hbeta  )
        elif 'BLR_alp1' in keys:
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

def flux_calc_mcmc(res,chains, mode, norm=1e-13,N=100):
    
    labels = list(chains.keys())

    popt = np.zeros_like(res['popt'])
    
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
    
    OIIIr = 5008.24*(1+sol['z'][0])/1e4
    
    fwhms = sol['Nar_fwhm'][0]/3e5/2.35*OIIIr
    peakn = sol['OIII_peak'][0]
    if 'outflow_fwhm' in sol:  
        fwhmws = sol['outflow_fwhm'][0]/3e5/2.35*OIIIr
        OIIIrws = OIIIr + sol['outflow_vel'][0]/3e5*OIIIr
        peakw = sol['OIII_out_peak'][0]
        y = gauss(wvs, peakn,OIIIr, fwhms) + gauss(wvs, peakw, OIIIrws, fwhmws)
    else:
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
    
    if plot==1:
        plt.figure()
        plt.plot(wvs, y)
           
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

    