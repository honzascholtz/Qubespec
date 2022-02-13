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


nan= float('nan')

pi= np.pi
e= np.e

c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

arrow = u'$\u2193$' 


up_limit_OIII = 1500.


up_lim_nar_hal = 1000.

def fitting_OIII_sig(wave, fluxs, error,z, init_sig=450., init_offset=0):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4900*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    
    sel=  np.where((wave<5025*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    
    peak_loc = np.argmax(flux_zoom)
    peak = (np.max(flux_zoom))
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] - init_offset
    
    model = LinearModel() + GaussianModel(prefix='o3r_')+ GaussianModel(prefix='o3b_')
    # Starting parameters for the fits
    #print wv
    #init_sig= 300.
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		o3r_amplitude = peak,  \
		o3b_amplitude =  peak/3,   \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		o3r_center = wv +init_offset  ,         \
		o3b_center = wv - 48.*(1+z)/1e4 +init_offset,   \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		o3r_sigma = (init_sig/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
		o3b_sigma = (init_sig/2.36/2.9979e5)*4958.92*(1+z)/1e4, 
	)

    Max_offset = 700.
    # Parameters constraints Narrow line flux > 0.0
    parameters['o3r_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3b_amplitude'].set(expr='o3r_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3r_sigma'].set(min=(100.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(up_limit_OIII/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3r_center'].set(min=5006.84*(1+z)/1e4+ 5006.84*(1+z)/1e4*(-Max_offset/2.9979e5),max=5006.84*(1+z)/1e4+ 5006.84*(1+z)/1e4*(Max_offset/2.9979e5))
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3b_sigma'].set(expr='o3r_sigma*(4958.92/5006.84)')  
    
    off = 48.*(1+z)/1e4
    parameters['o3b_center'].set(expr='o3r_center - '+str(off))
    
   
    #flux = np.array(flux[fit_loc], dtype='float64')
    #wave = np.array(wave[fit_loc], dtype='float64')
    #error = np.array(error[fit_loc], dtype='float64')

    out = model.fit(flux,params=parameters, errors=error, x=(wave))
    
    chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))/len(fit_loc)
    
    Hal_cm = 5008.*(1+z)/1e4
    #print init_sig, chi2, (out.params['o3r_fwhm'].value/Hal_cm)*2.9979e5  
    
    
    chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
    
    BIC = chi2+3*np.log(len(flux[fit_loc]))
    
    
    return out, BIC


def fitting_OIII_mul(wave, fluxs, error,z, chir=0):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4900*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    
    flux = flux[fit_loc]
    wave = wave[fit_loc]
    error = error[fit_loc]
    
 
       
    sel=  np.where((wave<5050*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 

    #plt.plot(wave, flux, drawstyle='steps-mid')
    
    
    
    model = LinearModel() + GaussianModel(prefix='o3rw_')+ GaussianModel(prefix='o3bw_') + GaussianModel(prefix='o3rn_')+ GaussianModel(prefix='o3bn_')
    # Starting parameters for the fits
    wv = 5008.*(1+z)/1e4
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		o3rw_center = wv-5.*(1+z)/1e4 ,         \
		o3bw_center = wv - (48.-5)*(1+z)/1e4,   \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		o3rw_sigma = (700.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
		o3bw_sigma = (700.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
         # pso
        o3rn_amplitude = peak*(3./4),  \
		o3bn_amplitude =  peak*(3./12),   \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		o3rn_center = wv,         \
		o3bn_center = wv - 48.*(1+z)/1e4,   \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		o3rn_sigma = (300.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
		o3bn_sigma = (300.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
         
	)

    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rw_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bw_amplitude'].set(expr='o3rw_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rw_sigma'].set(min=(100.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(up_limit_OIII/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rw_center'].set(min=5008.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-800.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5))
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bw_sigma'].set(expr='o3rw_sigma*(4958.92/5006.84)')  
    
    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rn_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bn_amplitude'].set(expr='o3rn_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rn_sigma'].set(min=(100.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(up_limit_OIII/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rn_center'].set(min=5006.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-800.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5))
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bn_sigma'].set(expr='o3rn_sigma*(4958.92/5006.84)') 
    off = 48.*(1+z)/1e4
    parameters['o3bn_center'].set(expr='o3rn_center - '+str(off))
    parameters['o3bw_center'].set(expr='o3rw_center - '+str(off))

    out = model.fit(flux,params=parameters, errors=error,x=(wave ))
    try:
        chi2 = sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
    
        BIC = chi2+6*np.log(len(flux))
    except:
        BIC = len(flux)+6*np.log(len(flux))
        
    if chir==0:
        return out 
    
    else:
        return out,chi2
    
    

def fitting_OIII_Hbeta_mul(wave, fluxs, error,z, chir=0, Hbeta=1):#, outflow_lim=150.):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4800*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    
    flux = flux[fit_loc]
    wave = wave[fit_loc]
    error = error[fit_loc]
    
 
       
    sel=  np.where((wave<5050*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 

    #plt.plot(wave, flux, drawstyle='steps-mid')
    
    
    
    model = LinearModel() + GaussianModel(prefix='o3rw_')+ GaussianModel(prefix='o3bw_') + GaussianModel(prefix='o3rn_')+ GaussianModel(prefix='o3bn_') + GaussianModel(prefix='Hb_')
    # Starting parameters for the fits
    wv = 5008.*(1+z)/1e4
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		o3rw_amplitude = peak/4,  \
		o3bw_amplitude =  peak/12,   \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		o3rw_center = wv-5.*(1+z)/1e4 ,         \
		o3bw_center = wv - (48.-5)*(1+z)/1e4,   \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		o3rw_sigma = (500.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
		o3bw_sigma = (500.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
         # pso
         o3rn_amplitude = peak*(3./4),  \
		o3bn_amplitude =  peak*(3./12),   \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		o3rn_center = wv,         \
		o3bn_center = wv - 48.*(1+z)/1e4,   \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		o3rn_sigma = (300.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
		o3bn_sigma = (300.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
        Hb_amplitude = peak/10,  \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		Hb_center =  4861.2*(1+z)/1e4  ,         \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		Hb_sigma = (300.0/2.36/2.9979e5)*4861.2*(1+z)/1e4, 
       
	)

    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rw_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bw_amplitude'].set(expr='o3rw_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rw_sigma'].set(min=(150./2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(up_limit_OIII/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rw_center'].set(min=5008.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-800.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5))
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bw_sigma'].set(expr='o3rw_sigma*(4958.92/5006.84)')  
    
    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rn_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bn_amplitude'].set(expr='o3rn_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rn_sigma'].set(min=(150.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(up_limit_OIII/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rn_center'].set(min=5006.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-800.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5))
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bn_sigma'].set(expr='o3rn_sigma*(4958.92/5006.84)') 
    off = 48.*(1+z)/1e4
    parameters['o3bn_center'].set(expr='o3rn_center - '+str(off))
    parameters['o3bw_center'].set(expr='o3rw_center - '+str(off))
    
    # Parameters constraints Narrow line flux > 0.0
    parameters['Hb_amplitude'].set(min=0.0) 
    
    if Hbeta==0:
        parameters['Hb_amplitude'].set(max= 0.00001) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Hb_sigma'].set(min=(150.0/2.36/2.9979e5)*4861.2*(1+z)/1e4,max=(700.0/2.36/2.9979e5)*4861.2*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Hb_center'].set(min=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4*(-700.0/2.9979e5),max=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4 *(700.0/2.9979e5))


    out = model.fit(flux,params=parameters, errors=error,x=(wave ))
    try:
        chi2 = sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
    
        BIC = chi2+6*np.log(len(flux))
    except:
        BIC = len(flux)+6*np.log(len(flux))
        
    if chir==0:
        return out 
    
    else:
        return out,chi2
    


def fitting_Halpha_sig(wave, fluxs, error,z, initial=0):
    from lmfit.models import LinearModel, GaussianModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>(6562.8-300)*(1+z)/1e4)&(wave<(6562.8+300)*(1+z)/1e4))[0]
      
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 
    
    
    model = LinearModel() + GaussianModel(prefix='Ha_')

    # Starting parameters for the fits
    if initial ==0:
        
        parameters = model.make_params( \
                                       #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		                              c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		Ha_amplitude = peak*(2./4),  \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		Ha_center = wv  ,         \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		Ha_sigma = (1500.0/2.36/2.9979e5)*6562.8*(1+z)/1e4)
	   
    
    else:
        parameters = model.make_params( \
                                       #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		                              c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		Ha_amplitude = peak,  \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		Ha_center = initial.params['Ha_center'].value  ,         \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		Ha_sigma = (700.0/2.36/2.9979e5)*6562.8*(1+z)/1e4)
        
        
    # Parameters constraints Narrow line flux > 0.0
    parameters['Ha_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Ha_sigma'].set(min=(100.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Ha_center'].set(min=6563.*(1+z)/1e4 + 6562.8*(1+z)/1e4*(-1000.0/2.9979e5),max=6563.*(1+z)/1e4 + 6562.8*(1+z)/1e4*(1000.0/2.9979e5))


    out = model.fit(flux[fit_loc],params=parameters, errors=error[fit_loc], x=(wave[fit_loc]))
    
        
    return out 


def fitting_Halpha_mul(wave, fluxs, error,z, wvnet=1., decompose=np.array([1]), offset=0,init_sig=300., broad=1, cont=1):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    if wvnet ==1.:
        wv = wave[np.where(wave==wave_peak)[0]] 
    
    else:
        wv = wvnet
        
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ GaussianModel(prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_')
    
    #model = LinearModel()+ GaussianModel(prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_')

    # Starting parameters for the fits
    #print wv
    if len(decompose)==1:
        sigma_s = 1000
        c_broad = wv
    
    
    elif len(decompose)==2:
        sigma_s = decompose[0]
        c_broad = decompose[1]
            
    Hal_cm = 6562.8*(1+z)/1e4
    
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
		Haw_height = peak/3,  \
		#  		
		Haw_center = wv  ,         \
		#  
		Haw_sigma = (sigma_s/2.36/2.9979e5)*Hal_cm, \
         # pso
        Han_amplitude = peak*(2./2),  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = 6583.*(1+z)/1e4, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = 6548.*(1+z)/1e4, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
	)
    if cont==0:
        parameters['intercept'].set(min=-0.0000000000001) 
        parameters['intercept'].set(max= 0.0000000000001) 
        parameters['slope'].set(min=-0.0000000000001) 
        parameters['slope'].set(max= 0.0000000000001) 
        print ('No continuum')
    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s
    
    
    if len(decompose) == 1:
        parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(7000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-400.0/2.9979e5),max=Hal_cm+ Hal_cm*(400.0/2.9979e5))
        
        #parameters['Haw_center'].set(expr='Han_center')
        
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
            
        
        
    elif len(decompose) == 2:
        parameters['Haw_sigma'].set(min=((decompose[0]-20)/2.36/2.9979e5)*Hal_cm,max=((decompose[0]+20)/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min= c_broad+ 6562.8*(-10.0/2.9979e5),max=c_broad+ 6562.8*(10.0/2.9979e5))
        
        
        #print 'Decomposing based on fixed Halpha broad center: ', c_broad, 'and width ', decompose[0] 
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
        
    # Parameters constraints Narrow line flux > 0.0
    parameters['Han_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((150.0/2.9979e5)*Hal_cm),max=(up_lim_nar_hal/2.36/2.9979e5)*Hal_cm) 
        
    
    # Velocity offsets between -800 and 800 km/s for narrow
    if wvnet== 1.:
        parameters['Han_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
    elif wvnet !=1:
        parameters['Han_center'].set(min=wvnet+ Hal_cm*(-600.0/2.9979e5),max=wvnet+ Hal_cm*(600.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    
    
    flux = np.array(flux[fit_loc], dtype='float64')
    error = np.array(error[fit_loc], dtype='float64')
    wave = np.array(wave[fit_loc], dtype='float64')
    out = model.fit(flux,params=parameters, errors=error, x=(wave))
        
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
        
    except:
        try:
            chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
        
        except:
            chi2=1
     
    Hal_cm = 6562.*(1+z)/1e4
    #print 'Broadline params of the fits ',(out.params['Haw_fwhm'].value/Hal_cm)*2.9979e5, (out.params['Haw_center'].value)
    #print 'BLR mode: chi2 ', chi2, ' N ', len(flux[fit_loc])
    #print 'BLR BIC ', chi2+7*np.log(len(flux[fit_loc]))
    return out ,chi2


def fit_continuum(wave, fluxs):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
        
    model = LinearModel()
    
    # Starting parameters for the fits

    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
        )
    
    flux[np.isnan(flux)] = 0
    out = model.fit(flux ,params=parameters, x=wave)
    
    
        
    return out


def sub_QSO(wave, fluxs, error,z, fst_out):
    from lmfit.models import LinearModel, GaussianModel, LorentzianModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
          
    sel=  np.where(((wave<(6562.8+60)*(1+z)/1e4))& (wave>(6562.8-60)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]   
    
    peak = np.ma.max(flux_zoom)
    
    wv = fst_out.params['Haw_center'].value

    model = LinearModel() + GaussianModel(prefix='Ha_')

    # Starting parameters for the fits
    parameters = model.make_params( \
         c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		Ha_amplitude = peak,  \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		Ha_center = wv  ,         \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		Ha_sigma = fst_out.params['Haw_sigma'].value)
	      
    # Parameters constraints Narrow line flux > 0.0
    parameters['Ha_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Ha_sigma'].set(min=fst_out.params['Haw_sigma'].value*0.999999,max=fst_out.params['Haw_sigma'].value*1.000001) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Ha_center'].set(min=wv*0.9999999,max=wv*1.000000000001) 

    out = model.fit(flux,params=parameters, errors=error, x=(wave))
    
        
    return out 



def fitting_Halpha_mul_outflow(wave, fluxs, error,z):
    from lmfit.models import GaussianModel, LinearModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-200)*(1+z)/1e4)&(wave<(6562.8+200)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 
    
    z = float(z)
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ GaussianModel(prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_') + GaussianModel(prefix='Nrw_') + GaussianModel(prefix='Nbw_')

            
    Hal_cm = 6562.8*(1+z)/1e4
    Nr_cm = 6583.*(1+z)/1e4
    Nb_cm = 6548.*(1+z)/1e4
    
    sigma_s = 900.
    init_sig = 400.
    wvnet= wv
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
		Haw_amplitude = peak/3,  \
		#  		
		Haw_center = wv  ,         \
		#  
		Haw_sigma = (sigma_s/2.36/2.9979e5)*Hal_cm, \
         # pso
         Han_amplitude = peak,  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = Nr_cm, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Nr_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = Nb_cm, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Nb_cm, \
         #
         Nrw_amplitude = peak*(1./6), \
         #
         Nrw_center = Nr_cm, \
         #
         Nrw_sigma = (init_sig/2.36/2.9979e5)*Nr_cm, \
         #
         Nbw_amplitude = peak/18, \
         #
         Nbw_center = Nb_cm, \
         #
         Nbw_sigma = (init_sig/2.36/2.9979e5)*Nb_cm, \
	)

    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s    
    #parameters['Haw_sigma'].set(min=(500.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(4000.0/2.36/2.9979e5)*Hal_cm) 
    
    parameters['Haw_sigma'].set(min=(600.0/2.36/2.9979e5)*Hal_cm,max=(1200.0/2.36/2.9979e5)*Hal_cm)
    #parameters['Haw_center'].set(expr='Han_center- (600./3e5*2.114534083642)')
    
    parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-1000.0/2.9979e5),max=Hal_cm+ Hal_cm*(1000.0/2.9979e5))
        
      
    # Parameters constraints Narrow line flux > 0.0
    parameters['Han_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((1./3000.0)/2.36)*Hal_cm,max=(1200.0/2.36/2.9979e5)*Hal_cm) 
    #
    parameters['Han_center'].set(min=wvnet+ Hal_cm*(-1000.0/2.9979e5),max=wvnet+ Hal_cm*(1000.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    
    ####################
    # Broad NII
    parameters['Nrw_amplitude'].set(min=0.0)
    parameters['Nrw_amplitude'].set(max=1.0e-4)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nbw_amplitude'].set(expr='Nrw_amplitude/3')  
    #
    parameters['Nrw_sigma'].set(expr='Haw_sigma*(6583/6562)')
    #
    parameters['Nbw_sigma'].set(expr='Haw_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nrw_center'].set(expr='Haw_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nbw_center'].set(expr='Haw_center - '+str(offset_b))
    
    
    out = model.fit(flux[fit_loc],params=parameters, errors=error[fit_loc], x=(wave[fit_loc]))    
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
    except:
        chi2=1
    
    print ('Outflow mode: chi2 ', chi2, ' N ', len(flux[fit_loc]))
    print ('OUtflow BIC ', chi2+8*np.log(len(flux[fit_loc]))  )
   
    return out ,chi2


import math

def gaussian(x, k, mu,sig):

    expo= -((x-mu)**2)/(sig*sig)
    
    y= k* e**expo
    
    return y

def Gaussian_BK(x, amplitude, center,sigma,a1,a2):
    from astropy.modeling.powerlaws import BrokenPowerLaw1D
    
    BK = BrokenPowerLaw1D.evaluate(x,1 ,center,a2,a1)
    
    GS = gaussian(x, 1, center,sigma)
    
    fcs = BK*GS
    fcs = fcs/max(fcs)
    y = amplitude*fcs
    return y

def Gaussian_BKc(x, amplitude, center,sigma,a1,a2):
    from astropy.modeling.powerlaws import BrokenPowerLaw1D
    
    BK = BrokenPowerLaw1D.evaluate(x,1 ,center,a2,a1)
    
    GS = gaussian(x, 1, center,sigma)
    
    fcs = np.convolve(BK, GS, 'same')
    fcs = fcs/max(fcs)
    y = amplitude*fcs
    return y

def BKP(x, amplitude, center, a1,a2):
    from astropy.modeling.powerlaws import BrokenPowerLaw1D
    
    BK = BrokenPowerLaw1D.evaluate(x,1 ,center,a2,a1)
    
    
    fcs = BK
    fcs = fcs/max(fcs)
    y = amplitude*fcs
    return y
    

import lmfit 
BkpGModel = lmfit.Model( Gaussian_BK)
'''

x = np.linspace(2,2.5, 3000)

plt.close('all')
plt.figure()

y = BKP(x, 2, 2.25, 2,-2)


plt.plot(x,y)

plt.show()
'''

def fitting_Halpha_mul_bkp(wave, fluxs, error,z, wvnet=1., decompose=1, offset=0,init_sig=300., broad=1, cont=1, Hal_up=1000.):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    
    #print ('Fitting Broken Power law Gaussian')
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    if wvnet ==1.:
        wv = wave[np.where(wave==wave_peak)[0]] 
    
    else:
        wv = wvnet
        
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ lmfit.Model(Gaussian_BK, prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_') #+ GaussianModel(prefix='X_')
    
    
    # Starting parameters for the fits
    #print wv
    if decompose==1:
        sigma_s = 4000
        c_broad = wv
    
    else:
        outo = decompose
        sigma_s = outo.params['Haw_sigma'].value/outo.params['Haw_center'].value*3e5
        c_broad = outo.params['Haw_center'].value
      
    Hal_cm = 6562.8*(1+z)/1e4
    
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
		Haw_amplitude = peak/3,  \
		#  		
		Haw_center = wv  ,         \
		#  
		Haw_sigma = (sigma_s/2.9979e5)*Hal_cm, \
         # pso
        Han_amplitude = peak*(2./2),  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = 6583.*(1+z)/1e4, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = 6548.*(1+z)/1e4, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         Haw_a1 = + 3, \
         Haw_a2 = - 3, \
	)
    if cont==0:
        parameters['intercept'].set(min=-0.0000000000001) 
        parameters['intercept'].set(max= 0.0000000000001) 
        parameters['slope'].set(min=-0.0000000000001) 
        parameters['slope'].set(max= 0.0000000000001) 
        print ('No continuum')
    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s
    
    
    if decompose == 1:
        parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-400.0/2.9979e5),max=Hal_cm+ Hal_cm*(400.0/2.9979e5))
        
        #parameters['Haw_center'].set(expr='Han_center')
        
        # Parameters constraints Narrow line flux > 0.0
        parameters['Han_amplitude'].set(min=0.0) 
        
        slp_edge = 200.
        parameters['Haw_a1'].set(min=0.0) 
        parameters['Haw_a1'].set(max=slp_edge) 
        
        parameters['Haw_a2'].set(max=0.0)
        parameters['Haw_a2'].set(min= -slp_edge)
        
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            
            #print 'No broad'
            
        
        
    elif decompose != 1:
        parameters['Haw_sigma'].set(min= 0.999*outo.params['Haw_sigma'],max=1.0001*outo.params['Haw_sigma']) 
        parameters['Haw_center'].set(min= 0.999*outo.params['Haw_center'],max=1.0001*outo.params['Haw_center']) 
        
        
        parameters['Haw_a1'].set(min= outo.params['Haw_a1'],max=outo.params['Haw_a1']+1)        
        parameters['Haw_a2'].set(min= outo.params['Haw_a2']-1,max=outo.params['Haw_a2']) 
        
        #print 'Decomposing based on fixed Halpha broad center: ', c_broad, 'and width ', decompose[0] 
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
        
    
    
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((200.0/2.9979e5)*Hal_cm),max=(Hal_up/2.36/2.9979e5)*Hal_cm) 
    
    #parameters['X_amplitude'].set(min=0)   
    #parameters['X_center'].set(min=2.07,max=2.15)
    #parameters['X_sigma'].set(min=0.2,max=(3000.0/2.36/2.9979e5)*Hal_cm)
    
    parameters['Han_amplitude'].set(min=0.0) 
    
    # Velocity offsets between -800 and 800 km/s for narrow
    if wvnet== 1.:
        parameters['Han_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
    elif wvnet !=1:
        parameters['Han_center'].set(min=wvnet+ Hal_cm*(-600.0/2.9979e5),max=wvnet+ Hal_cm*(600.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    
    
    flux = np.array(flux[fit_loc], dtype='float64')
    error = np.array(error[fit_loc], dtype='float64')
    wave = np.array(wave[fit_loc], dtype='float64')
    out = model.fit(flux,params=parameters, errors=error, x=(wave))
        
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
        
    except:
        try:
            chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
        
        except:
            chi2=1
     
    Hal_cm = 6562.*(1+z)/1e4
    #print 'Broadline params of the fits ',(out.params['Haw_fwhm'].value/Hal_cm)*2.9979e5, (out.params['Haw_center'].value)
    #print 'BLR mode: chi2 ', chi2, ' N ', len(flux[fit_loc])
    #print 'BLR BIC ', chi2+7*np.log(len(flux[fit_loc]))
    return out ,chi2


def fitting_OIII_Hbeta_qso_mul(wave, fluxs, error,z, chir=0, Hbeta=1, decompose=1, hbw=4861., offn=0, offw=0, o3n=1):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5200*(1+z)/1e4))[0]
    
    flux = flux[fit_loc]
    wave = wave[fit_loc]
    error = error[fit_loc]
    
 
       
    sel=  np.where((wave<5050*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 

    #plt.plot(wave, flux, drawstyle='steps-mid')
    
    
    
    model = LinearModel() + GaussianModel(prefix='o3rw_')+ GaussianModel(prefix='o3bw_') + GaussianModel(prefix='o3rn_')+ GaussianModel(prefix='o3bn_') + GaussianModel(prefix='Hbn_') + lmfit.Model(Gaussian_BK, prefix='Hbw_')#GaussianModel(prefix='Hbw_')
    # Starting parameters for the fits
    wv = 5008.*(1+z)/1e4
    
    if decompose==1:
        sigma_s = 800.
        c_broad = wv
    
    else:
        outo = decompose
        sigma_s = outo.params['Hbw_sigma'].value/outo.params['Hbw_center'].value*3e5
        c_broad = outo.params['Hbw_center'].value
    
    if 1==1:      
        parameters = model.make_params( \
          #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
    		c = np.ma.median(flux), \
    		#  Continuum slope; start = 0.0
    		b = -2., \
             #a = 0.0, \
    		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		o3rw_center = wv-10.*(1+z)/1e4 + offw/3e5*wv,         \
    		o3bw_center = wv - (48.+10)*(1+z)/1e4 + offw/3e5*wv,   \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		o3rw_sigma = (1100.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
    		o3bw_sigma = (1100.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
             # pso
            o3rn_height = peak*(2./4),  \
    		o3bn_height =  peak*(2./12),   \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		o3rn_center = wv+offn/3e5*wv,         \
    		o3bn_center = wv+offn/3e5*wv - 48.*(1+z)/1e4,   \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		o3rn_sigma = (300.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
    		o3bn_sigma = (300.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
            Hbn_amplitude = peak/10,  \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		Hbn_center =  4861.2*(1+z)/1e4  ,         \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		Hbn_sigma = (300.0/2.36/2.9979e5)*4861.2*(1+z)/1e4, \
            Hbw_a1 = +3., \
            Hbw_a2 = -35., \
            Hbw_amplitude = peak/10,  \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		Hbw_center =  hbw*(1+z)/1e4  ,         \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		Hbw_sigma = (sigma_s/2.9979e5)*4861.2*(1+z)/1e4, \
    	)
   
    
    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rw_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bw_amplitude'].set(expr='o3rw_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rw_sigma'].set(min=(1000./2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(2000/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rw_center'].set(min=5008.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-1000.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5)) #HB89 -700, 800, LBQS -1000, 800,  2QZJ -700,800
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bw_sigma'].set(expr='o3rw_sigma*(4958.92/5006.84)')  
    
    # Parameters constraints Narrow line flux > 0.0
    #parameters['o3rn_amplitude'].set(min=peak*0.1*(np.sqrt(2*np.pi)*0.00278)) 
    parameters['o3rn_amplitude'].set(min=0.0) # LBQS, HB89 0.0002, 2QZJ=0
    #print('height min on narrow ', peak*0.1)
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bn_amplitude'].set(expr='o3rn_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rn_sigma'].set(min=(150.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(1000./2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rn_center'].set(min=5006.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-200.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(500.0/2.9979e5)) # HB89 -500, 500, LBQS -200, 500,  2QZJ -200 500
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bn_sigma'].set(expr='o3rn_sigma*(4958.92/5006.84)') 
    off = 48.*(1+z)/1e4
    parameters['o3bn_center'].set(expr='o3rn_center - '+str(off))
    parameters['o3bw_center'].set(expr='o3rw_center - '+str(off))
    
    if o3n==0:
        parameters['o3rn_amplitude'].set(expr='o3rw_amplitude/100000000000') 
        parameters['o3bn_amplitude'].set(expr='o3bw_amplitude/100000000000') 
        print('small o3')
    else:
      parameters['o3rn_amplitude'].set(min=0.0002) # LBQS, HB89 0.0002, 2QZJ=0
    
    # Parameters constraints Narrow line flux > 0.0
    #parameters['Hbn_amplitude'].set(min=0.0) 
    parameters['Hbn_amplitude'].set(expr='Hbw_amplitude/100000000') 
    
    parameters['slope'].set(min= -3,max=3) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Hbn_sigma'].set(min=(300.0/2.36/2.9979e5)*4861.2*(1+z)/1e4,max=(700.0/2.36/2.9979e5)*4861.2*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Hbn_center'].set(min=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4*(-700.0/2.9979e5),max=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4 *(700.0/2.9979e5))
    
    if decompose==1:
        
        parameters['Hbw_amplitude'].set(min=0.0) 
        parameters['Hbw_sigma'].set(min=(2000.0/2.36/2.9979e5)*4861.2*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*4861.2*(1+z)/1e4) 
        # Velocity offsets between -500 and 500 km/s for narrow
        parameters['Hbw_center'].set(min=hbw*(1+z)/1e4 + hbw*(1+z)/1e4*(-1000.0/2.9979e5),max=hbw*(1+z)/1e4 + hbw*(1+z)/1e4 *(1000.0/2.9979e5))
        
        slp_edge = 100.
        parameters['Hbw_a1'].set(min=0.0) 
        parameters['Hbw_a1'].set(max=slp_edge) 
            
        parameters['Hbw_a2'].set(max=-20)
        parameters['Hbw_a2'].set(min= -slp_edge)
    else:
        parameters['Hbw_sigma'].set(min= 0.999*outo.params['Hbw_sigma'],max=1.0001*outo.params['Hbw_sigma']) 
        parameters['Hbw_center'].set(min= 0.999*outo.params['Hbw_center'],max=1.0001*outo.params['Hbw_center']) 
        
        parameters['Hbw_a1'].set(min= outo.params['Hbw_a1'],max=outo.params['Hbw_a1']+1)        
        parameters['Hbw_a2'].set(min= outo.params['Hbw_a2']-1,max=outo.params['Hbw_a2']) 
        

    out = model.fit(flux,params=parameters, errors=error,x=(wave ))
    try:
        chi2 = sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
    
        BIC = chi2+6*np.log(len(flux))
    except:
        BIC = len(flux)+6*np.log(len(flux))
        
    if chir==0:
        return out 
    
    else:
        chi2 = (out.eval(x=wave)- flux )**2#sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
        return out,chi2


def fitting_OIII_Hbeta_qso_sig(wave, fluxs, error,z, chir=0, Hbeta=1, decompose=1, hbw=4861., offn=0, offw=0, o3n=1):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5200*(1+z)/1e4))[0]
    
    flux = flux[fit_loc]
    wave = wave[fit_loc]
    error = error[fit_loc]
    
 
       
    sel=  np.where((wave<5050*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    wv = wave[np.where(wave==wave_peak)[0]] 

    #plt.plot(wave, flux, drawstyle='steps-mid')
    
    
    
    model = LinearModel() + GaussianModel(prefix='o3rw_')+ GaussianModel(prefix='o3bw_') + GaussianModel(prefix='Hbn_') + lmfit.Model(Gaussian_BK, prefix='Hbw_')#GaussianModel(prefix='Hbw_')
    # Starting parameters for the fits
    wv = 5008.*(1+z)/1e4
    
    if decompose==1:
        sigma_s = 800.
        c_broad = wv
    
    else:
        outo = decompose
        sigma_s = outo.params['Hbw_sigma'].value/outo.params['Hbw_center'].value*3e5
        c_broad = outo.params['Hbw_center'].value
    
    if 1==1:      
        parameters = model.make_params( \
          #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
    		c = np.ma.median(flux), \
    		#  Continuum slope; start = 0.0
    		b = -2., \
             #a = 0.0, \
    		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		o3rw_center = wv-10.*(1+z)/1e4 + offw/3e5*wv,         \
    		o3bw_center = wv - (48.+10)*(1+z)/1e4 + offw/3e5*wv,   \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		o3rw_sigma = (1100.0/2.36/2.9979e5)*5006.84*(1+z)/1e4, \
    		o3bw_sigma = (1100.0/2.36/2.9979e5)*4958.92*(1+z)/1e4, \
             # pso
            o3rw_height = peak*(2./4),  \
    		o3bw_height =  peak*(2./12),   \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		
            Hbn_amplitude = peak/10,  \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		Hbn_center =  4861.2*(1+z)/1e4  ,         \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		Hbn_sigma = (300.0/2.36/2.9979e5)*4861.2*(1+z)/1e4, \
            Hbw_a1 = +3., \
            Hbw_a2 = -35., \
            Hbw_amplitude = peak/10,  \
    		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
    		Hbw_center =  hbw*(1+z)/1e4  ,         \
    		#  [O III] 5007 sigma; start = 300 km/s FWHM			
    		Hbw_sigma = (sigma_s/2.9979e5)*4861.2*(1+z)/1e4, \
    	)
   
    
    # Parameters constraints Narrow line flux > 0.0
    parameters['o3rw_amplitude'].set(min=0.0) 
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bw_amplitude'].set(expr='o3rw_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rw_sigma'].set(min=(1000./2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(2000/2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rw_center'].set(min=5008.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-800.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(800.0/2.9979e5)) #LBQS -1000, 800, HB89 -700, 800, 2QZJ -800,800
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bw_sigma'].set(expr='o3rw_sigma*(4958.92/5006.84)')  
    
    # Parameters constraints Narrow line flux > 0.0
    #parameters['o3rn_amplitude'].set(min=peak*0.1*(np.sqrt(2*np.pi)*0.00278)) 
    parameters['o3rn_amplitude'].set(min=0.0002) 
    #print('height min on narrow ', peak*0.1)
    # [O III] 4959 amplitude = 1/3 of [O III] 5007 amplitude 
    parameters['o3bn_amplitude'].set(expr='o3rn_amplitude/3.0')
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 1000 km/s
    parameters['o3rn_sigma'].set(min=(150.0/2.36/2.9979e5)*5006.84*(1+z)/1e4,max=(1000./2.36/2.9979e5)*5006.84*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['o3rn_center'].set(min=5006.*(1+z)/1e4 + 5006.84*(1+z)/1e4*(-200.0/2.9979e5),max=5006*(1+z)/1e4+ 5006.84*(1+z)/1e4*(500.0/2.9979e5)) # LBQS -200, 500, HB89 -500, 500, 2QZJ -200 500
    # Constrain narrow line kinematics to match the [O III] line 
    parameters['o3bn_sigma'].set(expr='o3rn_sigma*(4958.92/5006.84)') 
    off = 48.*(1+z)/1e4
    parameters['o3bn_center'].set(expr='o3rn_center - '+str(off))
    parameters['o3bw_center'].set(expr='o3rw_center - '+str(off))
    
    if o3n==0:
        parameters['o3rn_amplitude'].set(expr='o3rw_amplitude/100000000000') 
        parameters['o3bn_amplitude'].set(expr='o3bw_amplitude/100000000000') 
        print('small o3')
        
    
    # Parameters constraints Narrow line flux > 0.0
    #parameters['Hbn_amplitude'].set(min=0.0) 
    parameters['Hbn_amplitude'].set(expr='Hbw_amplitude/100000000') 
    
    parameters['slope'].set(min= -3,max=3) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Hbn_sigma'].set(min=(300.0/2.36/2.9979e5)*4861.2*(1+z)/1e4,max=(700.0/2.36/2.9979e5)*4861.2*(1+z)/1e4) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Hbn_center'].set(min=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4*(-700.0/2.9979e5),max=4861.2*(1+z)/1e4 + 4861.2*(1+z)/1e4 *(700.0/2.9979e5))
    
    if decompose==1:
        
        parameters['Hbw_amplitude'].set(min=0.0) 
        parameters['Hbw_sigma'].set(min=(2000.0/2.36/2.9979e5)*4861.2*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*4861.2*(1+z)/1e4) 
        # Velocity offsets between -500 and 500 km/s for narrow
        parameters['Hbw_center'].set(min=hbw*(1+z)/1e4 + hbw*(1+z)/1e4*(-1000.0/2.9979e5),max=hbw*(1+z)/1e4 + hbw*(1+z)/1e4 *(1000.0/2.9979e5))
        
        slp_edge = 100.
        parameters['Hbw_a1'].set(min=0.0) 
        parameters['Hbw_a1'].set(max=slp_edge) 
            
        parameters['Hbw_a2'].set(max=-20)
        parameters['Hbw_a2'].set(min= -slp_edge)
    else:
        parameters['Hbw_sigma'].set(min= 0.999*outo.params['Hbw_sigma'],max=1.0001*outo.params['Hbw_sigma']) 
        parameters['Hbw_center'].set(min= 0.999*outo.params['Hbw_center'],max=1.0001*outo.params['Hbw_center']) 
        
        parameters['Hbw_a1'].set(min= outo.params['Hbw_a1'],max=outo.params['Hbw_a1']+1)        
        parameters['Hbw_a2'].set(min= outo.params['Hbw_a2']-1,max=outo.params['Hbw_a2']) 
        
        
        


    out = model.fit(flux,params=parameters, errors=error,x=(wave ))
    try:
        chi2 = sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
    
        BIC = chi2+6*np.log(len(flux))
    except:
        BIC = len(flux)+6*np.log(len(flux))
        
    if chir==0:
        return out 
    
    else:
        chi2 = (out.eval(x=wave)- flux )**2#sum(((out.eval(x=wave)- flux)**2)/(error.data**2))
        return out,chi2




def sub_QSO_bkp(wave, fluxs, error,z, fst_out):
    from lmfit.models import LinearModel, GaussianModel, LorentzianModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
          
    sel=  np.where(((wave<(6562.8+60)*(1+z)/1e4))& (wave>(6562.8-60)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]   
    
    peak = np.ma.max(flux_zoom)
    
    wv = fst_out.params['Haw_center'].value

    model = LinearModel() + lmfit.Model(Gaussian_BK, prefix='Ha_')
    
    # Starting parameters for the fits
    parameters = model.make_params( \
        c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
         #a = 0.0, \
		#  [O III] 5007 amplitude; start = 5 sigma of spectrum			
		Ha_amplitude = peak,  \
		#  [O III] 5007 peak; start = zero offset, BLR 5 Ang offset to stagger			
		Ha_center = wv  ,         \
		#  [O III] 5007 sigma; start = 300 km/s FWHM			
		Ha_sigma = fst_out.params['Haw_sigma'].value,\
        Ha_a1 = fst_out.params['Haw_a1'].value,\
        Ha_a2 = fst_out.params['Haw_a2'].value)
	      
    # Parameters constraints Narrow line flux > 0.0
    parameters['Ha_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of grating of KMOS (R >~ 3000) < max of 2500 km/s
    parameters['Ha_sigma'].set(min=fst_out.params['Haw_sigma'].value*0.999999,max=fst_out.params['Haw_sigma'].value*1.000001) 
    # Velocity offsets between -500 and 500 km/s for narrow
    parameters['Ha_center'].set(min=wv*0.9999999,max=wv*1.000000000001) 
    
    parameters['Ha_a1'].set(min= fst_out.params['Haw_a1'],max=fst_out.params['Haw_a1']+1)        
    parameters['Ha_a2'].set(min= fst_out.params['Haw_a2']-1,max=fst_out.params['Haw_a2']) 
        

    out = model.fit(flux,params=parameters, errors=error, x=(wave))
    
        
    return out 




def fitting_Halpha_mul_2QZJ(wave, fluxs, error,z, wvnet=1., decompose=np.array([1]), offset=0,init_sig=300., broad=1, cont=1):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    if wvnet ==1.:
        wv = wave[np.where(wave==wave_peak)[0]] 
    
    else:
        wv = wvnet
        
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ GaussianModel(prefix='Haw_') +  GaussianModel(prefix='Hawn_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_')
    
    #model = LinearModel()+ GaussianModel(prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_')

    # Starting parameters for the fits
    #print wv
    if len(decompose)==1:
        sigma_s = 1000
        c_broad = wv
    
    
    elif len(decompose)==2:
        sigma_s = decompose[0]
        c_broad = decompose[1]
            
    Hal_cm = 6562.8*(1+z)/1e4
    
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
		Haw_amplitude = peak/2,  \
        Hawn_amplitude = peak/2,  \
		#  		
		Haw_center = 2.2478  ,         \
        Hawn_center = 2.2391  ,         \
		#  
		Haw_sigma = (9700./2.36/2.9979e5)*Hal_cm, \
        Hawn_sigma = (3400./2.36/2.9979e5)*Hal_cm, \
         # pso
        Han_amplitude = peak*(2./2),  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = 6583.*(1+z)/1e4, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = 6548.*(1+z)/1e4, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
	)
    if cont==0:
        parameters['intercept'].set(min=-0.0000000000001) 
        parameters['intercept'].set(max= 0.0000000000001) 
        parameters['slope'].set(min=-0.0000000000001) 
        parameters['slope'].set(max= 0.0000000000001) 
        print ('No continuum')
    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    parameters['Hawn_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s
    
    
    if len(decompose) == 1:
        parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-400.0/2.9979e5),max=Hal_cm+ Hal_cm*(1700.0/2.9979e5))
        
        parameters['Hawn_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(4000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Hawn_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
        #parameters['Haw_center'].set(expr='Han_center')
        
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
            
        
        
    elif len(decompose) == 2:
        parameters['Haw_sigma'].set(min=((decompose[0]-20)/2.36/2.9979e5)*Hal_cm,max=((decompose[0]+20)/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min= c_broad+ 6562.8*(-10.0/2.9979e5),max=c_broad+ 6562.8*(10.0/2.9979e5))
        
        
        #print 'Decomposing based on fixed Halpha broad center: ', c_broad, 'and width ', decompose[0] 
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
        
    # Parameters constraints Narrow line flux > 0.0
    parameters['Han_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((200.0/2.9979e5)*Hal_cm),max=(up_lim_nar_hal/2.36/2.9979e5)*Hal_cm) 
        
    
    # Velocity offsets between -800 and 800 km/s for narrow
    if wvnet== 1.:
        parameters['Han_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
    elif wvnet !=1:
        parameters['Han_center'].set(min=wvnet+ Hal_cm*(-600.0/2.9979e5),max=wvnet+ Hal_cm*(600.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    
    
    flux = np.array(flux[fit_loc], dtype='float64')
    error = np.array(error[fit_loc], dtype='float64')
    wave = np.array(wave[fit_loc], dtype='float64')
    out = model.fit(flux,params=parameters, errors=error, x=(wave))
        
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
        
    except:
        try:
            chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
        
        except:
            chi2=1
     
    Hal_cm = 6562.*(1+z)/1e4
    #print 'Broadline params of the fits ',(out.params['Haw_fwhm'].value/Hal_cm)*2.9979e5, (out.params['Haw_center'].value)
    #print 'BLR mode: chi2 ', chi2, ' N ', len(flux[fit_loc])
    #print 'BLR BIC ', chi2+7*np.log(len(flux[fit_loc]))
    return out ,chi2



def fitting_Halpha_mul_LBQS(wave, fluxs, error,z, wvnet=1., decompose=1, offset=0,init_sig=300., broad=1, cont=1):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    if wvnet ==1.:
        wv = wave[np.where(wave==wave_peak)[0]] 
    
    else:
        wv = wvnet
        
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ lmfit.Model(Gaussian_BK, prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_') + GaussianModel(prefix='X_')
    
    
    # Starting parameters for the fits
    #print wv
    if decompose==1:
        sigma_s = 4000
        c_broad = wv
    
    else:
        outo = decompose
        sigma_s = outo.params['Haw_sigma'].value/outo.params['Haw_center'].value*3e5
        c_broad = outo.params['Haw_center'].value
      
    Hal_cm = 6562.8*(1+z)/1e4
    
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.0, \
		Haw_amplitude = peak/3,  \
		#  		
		Haw_center = wv  ,         \
		#  
		Haw_sigma = (sigma_s/2.9979e5)*Hal_cm, \
         # pso
        Han_amplitude = peak*(2./2),  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = 6583.*(1+z)/1e4, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = 6548.*(1+z)/1e4, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         Haw_a1 = + 3, \
         Haw_a2 = - 3, \
         
         X_sigma = (6000./2.9979e5)*Hal_cm , \
         X_center = 6350*(1+z)/1e4 , \
	)
    if cont==0:
        parameters['intercept'].set(min=-0.0000000000001) 
        parameters['intercept'].set(max= 0.0000000000001) 
        parameters['slope'].set(min=-0.0000000000001) 
        parameters['slope'].set(max= 0.0000000000001) 
        print ('No continuum')
    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s
    
    
    if decompose == 1:
        parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-400.0/2.9979e5),max=Hal_cm+ Hal_cm*(400.0/2.9979e5))
        
        #parameters['Haw_center'].set(expr='Han_center')
        
        # Parameters constraints Narrow line flux > 0.0
        parameters['Han_amplitude'].set(min=0.0) 
        
        slp_edge = 200.
        parameters['Haw_a1'].set(min=0.0) 
        parameters['Haw_a1'].set(max=slp_edge) 
        
        parameters['Haw_a2'].set(max=0.0)
        parameters['Haw_a2'].set(min= -slp_edge)
        
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            
            #print 'No broad'
            
        
        
    elif decompose != 1:
        parameters['Haw_sigma'].set(min= 0.999*outo.params['Haw_sigma'],max=1.0001*outo.params['Haw_sigma']) 
        parameters['Haw_center'].set(min= 0.999*outo.params['Haw_center'],max=1.0001*outo.params['Haw_center']) 
        
        
        parameters['Haw_a1'].set(min= outo.params['Haw_a1'],max=outo.params['Haw_a1']+1)        
        parameters['Haw_a2'].set(min= outo.params['Haw_a2']-1,max=outo.params['Haw_a2']) 
        
        parameters['X_sigma'].set(min= 0.999*outo.params['X_sigma'],max=1.0001*outo.params['X_sigma']) 
        parameters['X_center'].set(min= 0.999*outo.params['X_center'],max=1.0001*outo.params['X_center']) 
        
        
        #print 'Decomposing based on fixed Halpha broad center: ', c_broad, 'and width ', decompose[0] 
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
        
    
    
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((200.0/2.9979e5)*Hal_cm),max=(up_lim_nar_hal/2.36/2.9979e5)*Hal_cm) 
    
    #parameters['X_amplitude'].set(min=0)   
    #parameters['X_center'].set(min=2.07,max=2.15)
    #parameters['X_sigma'].set(min=0.2,max=(3000.0/2.36/2.9979e5)*Hal_cm)
    
    parameters['Han_amplitude'].set(min=0.0) 
    
    # Velocity offsets between -800 and 800 km/s for narrow
    if wvnet== 1.:
        parameters['Han_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
    elif wvnet !=1:
        parameters['Han_center'].set(min=wvnet+ Hal_cm*(-600.0/2.9979e5),max=wvnet+ Hal_cm*(600.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    parameters['X_center'].set(min=6361*(1+z)/1e4+ Hal_cm*(-900.0/2.9979e5),max=6361*(1+z)/1e4+ Hal_cm*(900.0/2.9979e5))
    parameters['X_sigma'].set(min=(4000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(8000.0/2.36/2.9979e5)*Hal_cm) 
    
    
    flux = np.array(flux[fit_loc], dtype='float64')
    error = np.array(error[fit_loc], dtype='float64')
    wave = np.array(wave[fit_loc], dtype='float64')
    
    out = model.fit(flux,params=parameters, errors=error, x=(wave))
        
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
        
    except:
        try:
            chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
        
        except:
            chi2=1
     
    Hal_cm = 6562.*(1+z)/1e4
    #print 'Broadline params of the fits ',(out.params['Haw_fwhm'].value/Hal_cm)*2.9979e5, (out.params['Haw_center'].value)
    #print 'BLR mode: chi2 ', chi2, ' N ', len(flux[fit_loc])
    #print 'BLR BIC ', chi2+7*np.log(len(flux[fit_loc]))
    return out ,chi2



def fitting_Halpha_mul_bkp_2QZJ(wave, fluxs, error,z, wvnet=1., decompose=1, offset=0,init_sig=300., broad=1, cont=1, cont_norm='n'):
    from lmfit.models import GaussianModel, LorentzianModel, LinearModel, QuadraticModel, PowerLawModel
    
    
    print ('Fitting Broken Power law Gaussian with fixed slope')
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    
    
    fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6562.8+20)*(1+z)/1e4))& (wave>(6562.8-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    peak = np.ma.max(flux_zoom)
    wave_peak = wave_zoom[peak_loc]
    
    if wvnet ==1.:
        wv = wave[np.where(wave==wave_peak)[0]] 
    
    else:
        wv = wvnet
        
    Hal_cm = 6562.8*(1+z)/1e4
    
    
    model = LinearModel()+ lmfit.Model(Gaussian_BK, prefix='Haw_') + GaussianModel(prefix='Han_') + GaussianModel(prefix='Nr_') + GaussianModel(prefix='Nb_') #+ GaussianModel(prefix='X_')
    
    
    # Starting parameters for the fits
    #print wv
    if decompose==1:
        sigma_s = 4000
        c_broad = wv
    
    else:
        outo = decompose
        sigma_s = outo.params['Haw_sigma'].value/outo.params['Haw_center'].value*3e5
        c_broad = outo.params['Haw_center'].value
      
    Hal_cm = 6562.8*(1+z)/1e4
    
    # Starting parameters for the fits    
    parameters = model.make_params( \
      #  Continuum level @ 5000 Ang; start = mean flux of spectrum			
		c = np.ma.median(flux), \
		#  Continuum slope; start = 0.0
		b = 0.1707, \
		Haw_amplitude = peak/3,  \
		#  		
		Haw_center = wv  ,         \
		#  
		Haw_sigma = (sigma_s/2.9979e5)*Hal_cm, \
         # pso
        Han_amplitude = peak*(2./2),  \
		#  			
		Han_center = wv,         \
		#  	
		Han_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nr_amplitude = peak*(1./6), \
         #
         Nr_center = 6583.*(1+z)/1e4, \
         #
         Nr_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         #
         Nb_amplitude = peak/18, \
         #
         Nb_center = 6548.*(1+z)/1e4, \
         #
         Nb_sigma = (init_sig/2.36/2.9979e5)*Hal_cm, \
         Haw_a1 = + 3, \
         Haw_a2 = - 3, \
	)
    if cont==0:
        parameters['intercept'].set(min=-0.0000000000001) 
        parameters['intercept'].set(max= 0.0000000000001) 
        parameters['slope'].set(min=-0.0000000000001) 
        parameters['slope'].set(max= 0.0000000000001) 
        print ('No continuum')
    # Parameters constraints Broad line flux > 0.0
    parameters['Haw_amplitude'].set(min=0.0) 
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 2500 km/s
    
    parameters['slope'].set(min=-0.0000001) 
    parameters['slope'].set(max=0.00000001) 
    
    if cont_norm !='n':
        
        parameters['intercept'].set(min=cont_norm-0.0000001) 
        parameters['intercept'].set(max=cont_norm+0.00000001) 
        
    
    
    if decompose == 1:
        parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(12000.0/2.36/2.9979e5)*Hal_cm) 
        parameters['Haw_center'].set(min=Hal_cm+ Hal_cm*(-400.0/2.9979e5),max=Hal_cm+ Hal_cm*(400.0/2.9979e5))
        
        #parameters['Haw_center'].set(expr='Han_center')
        
        # Parameters constraints Narrow line flux > 0.0
        parameters['Han_amplitude'].set(min=0.0) 
        
        slp_edge = 200.
        parameters['Haw_a1'].set(min=0.0) 
        parameters['Haw_a1'].set(max=slp_edge) 
        
        parameters['Haw_a2'].set(max=0.0)
        parameters['Haw_a2'].set(min= -slp_edge)
        
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            
            #print 'No broad'
            
        
        
    elif decompose != 1:
        parameters['Haw_sigma'].set(min= 0.999*outo.params['Haw_sigma'],max=1.0001*outo.params['Haw_sigma']) 
        parameters['Haw_center'].set(min= 0.999*outo.params['Haw_center'],max=1.0001*outo.params['Haw_center']) 
        
        
        parameters['Haw_a1'].set(min= outo.params['Haw_a1'],max=outo.params['Haw_a1']+1)        
        parameters['Haw_a2'].set(min= outo.params['Haw_a2']-1,max=outo.params['Haw_a2']) 
        
        #print 'Decomposing based on fixed Halpha broad center: ', c_broad, 'and width ', decompose[0] 
        if broad==0:
            parameters['Haw_amplitude'].set(expr='Han_amplitude/100000000') 
            #print 'No broad'
        
    
    
    # Narrow line FWHM > min resolution of YJ grating of KMOS (R >~ 3000) < max of 300 km/s
    parameters['Han_sigma'].set(min=((200.0/2.9979e5)*Hal_cm),max=(up_lim_nar_hal/2.36/2.9979e5)*Hal_cm) 
    
    #parameters['X_amplitude'].set(min=0)   
    #parameters['X_center'].set(min=2.07,max=2.15)
    #parameters['X_sigma'].set(min=0.2,max=(3000.0/2.36/2.9979e5)*Hal_cm)
    
    parameters['Han_amplitude'].set(min=0.0) 
    
    # Velocity offsets between -800 and 800 km/s for narrow
    if wvnet== 1.:
        parameters['Han_center'].set(min=Hal_cm+ Hal_cm*(-900.0/2.9979e5),max=Hal_cm+ Hal_cm*(900.0/2.9979e5))
        
    elif wvnet !=1:
        parameters['Han_center'].set(min=wvnet+ Hal_cm*(-600.0/2.9979e5),max=wvnet+ Hal_cm*(600.0/2.9979e5))
        
    #
    parameters['Nr_amplitude'].set(min=0.0)
    #parameters['Nr_amplitude'].set(expr = 'Han_amplitude/1000000000')
    #
    parameters['Nb_amplitude'].set(expr='Nr_amplitude/3')  
    #
    parameters['Nr_sigma'].set(expr='Han_sigma*(6583/6562)')
    #
    parameters['Nb_sigma'].set(expr='Han_sigma*(6548/6562)') 
    
    offset_r = (6562.-6583.)*(1+z)/1e4
    #
    parameters['Nr_center'].set(expr='Han_center - '+str(offset_r))
    
    offset_b = (6562.-6548.)*(1+z)/1e4
    #
    parameters['Nb_center'].set(expr='Han_center - '+str(offset_b))
    #parameters['Haw_sigma'].set(min=(2000.0/2.36/2.9979e5)*6562.8*(1+z)/1e4,max=(2500.0/2.36/2.9979e5)*Hal_cm) 
    
    
    flux = np.array(flux[fit_loc], dtype='float64')
    error = np.array(error[fit_loc], dtype='float64')
    wave = np.array(wave[fit_loc], dtype='float64')
    out = model.fit(flux,params=parameters, errors=error, x=(wave))
        
    try:
        chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error.data[fit_loc]**2))
        
    except:
        try:
            chi2 = sum(((out.eval(x=wave[fit_loc])- flux[fit_loc])**2)/(error[fit_loc]**2))
        
        except:
            chi2=1
     
    Hal_cm = 6562.*(1+z)/1e4
    #print 'Broadline params of the fits ',(out.params['Haw_fwhm'].value/Hal_cm)*2.9979e5, (out.params['Haw_center'].value)
    #print 'BLR mode: chi2 ', chi2, ' N ', len(flux[fit_loc])
    #print 'BLR BIC ', chi2+7*np.log(len(flux[fit_loc]))
    return out ,chi2
