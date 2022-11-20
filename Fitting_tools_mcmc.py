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
import os
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

N = 6000
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

from Halpha_models import *
from OIII_models import *
from Halpha_OIII_models import *
from QSO_models import *




# =============================================================================
#  Primary function to fit Halpha both with or without BLR - data prep and fit 
# =============================================================================
def fitting_Halpha(wave, fluxs, error,z, BLR=1,zcont=0.05, progress=True ,N=6000,priors= {'cont':[0,-3,1],\
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
    
    fit_loc = np.where((wave>(6564.52-170)*(1+z)/1e4)&(wave<(6564.52+170)*(1+z)/1e4))[0]
       
    sel=  np.where(((wave<(6564.52+20)*(1+z)/1e4))& (wave>(6564.52-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    znew = wave_zoom[peak_loc]/0.656452-1
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
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, Halpha_wBLR, log_prior_Halpha_BLR))
    
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
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, Halpha, log_prior_Halpha))
    
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
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, Halpha_outflow, log_prior_Halpha_outflow))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
    
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel')
        
        fitted_model = Halpha_outflow
        
        res = {'name': 'Halpha_wth_out'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    
    if BLR=='QSO_BKPL':
        print(priors)
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4, priors['Nar_fwhm'][0],\
                          peak/6, peak/6,\
                          priors['outflow_fwhm'][0],priors['outflow_vel'][0], \
                          peak, priors['Ha_BLR_vel'][0], priors['Ha_BLR_alp1'][0], priors['Ha_BLR_alp2'][0],priors['Ha_BLR_sig'][0], \
                          ])
            
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        priors['z'] = [z, norm(z,0.01)]
        prl = [ priors[key][1] for key in list(priors.keys()) ]
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],prl, Hal_QSO_BKPL, log_prior_Halpha_QSO_BKPL))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
        
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
                'Hal_out_peak', 'NII_out_peak', 
                'outflow_fwhm', 'outflow_vel', \
                'Ha_BLR_peak', 'Ha_BLR_vel', 'Ha_BLR_alp1', 'Ha_BLR_alp2', 'Ha_BLR_sig')
        
        fitted_model = Hal_QSO_BKPL
        
        res = {'name': 'Halpha_QSO_BKPL'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
        
    
            
        
    return res, fitted_model
    
# =============================================================================
# Primary function to fit [OIII] with and without outflows. 
# =============================================================================
    
def fitting_OIII(wave, fluxs, error,z, outflow=0, template=0, Hbeta_dual=0,N=6000, progress=True, \
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
    
    
    nwalkers=64
    if outflow==1: 
        if template==0:
            if Hbeta_dual == 0:
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0]])
                
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
               
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, OIII_outflow, log_prior_OIII_outflow))
                
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, OIII_outflow_narHb, log_prior_OIII_outflow_narHb))
                    
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors,OIII_outflow_Fe, log_prior_OIII_outflow_Fe, template))
                    
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, OIII_outflow_Fe_narHb, log_prior_OIII_outflow_Fe_narHb, template))
                    
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, OIII, log_prior_OIII))
            
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, OIII_dual_hbeta, log_prior_OIII_dual_hbeta))
            
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors,OIII_Fe, log_prior_OIII_Fe, template))
            
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
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors,OIII_dual_hbeta_Fe, log_prior_OIII_dual_hbeta_Fe,  template))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel','Fe_peak', 'Fe_fwhm')
                
                
                fitted_model = OIII_dual_hbeta_Fe
                
                res = {'name': 'OIII_Fe_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
    if outflow=='QSO':
        if template==0:
            pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                    priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0],\
                    peak_beta/2, peak_beta/2, \
                    priors['Hb_BLR1_fwhm'][0],priors['Hb_BLR2_fwhm'][0],priors['Hb_BLR_vel'][0],\
                    peak_beta/4, peak_beta/4])
               
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(z,0.001, nwalkers)
           
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_QSO, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
            
            sampler.run_mcmc(pos, N, progress=progress);
            
            flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
                
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm',\
                    'out_vel', 'Hb_BLR1_peak', 'Hb_BLR2_peak', 'Hb_BLR_fwhm1', 'Hb_BLR_fwhm2', 'Hb_BLR_vel',\
                    'Hb_nar_peak', 'Hb_out_peak')
            
            fitted_model = OIII_QSO
            
            res = {'name': 'OIII_QSO'}
            for i in range(len(labels)):
                res[labels[i]] = flat_samples[:,i]
    
        else:
            nwalkers=64
            pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                    priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0],\
                    peak_beta/2, peak_beta/2, \
                    priors['Hb_BLR1_fwhm'][0],priors['Hb_BLR2_fwhm'][0],priors['Hb_BLR_vel'][0],\
                    peak_beta/4, peak_beta/4,\
                    np.median(flux[fit_loc]), priors['Fe_fwhm'][0]])
               
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(z,0.001, nwalkers)
           
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors,OIII_Fe_QSO,log_prior_OIII_Fe_QSO, template))
            
            sampler.run_mcmc(pos, N, progress=progress);
            
            flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
                
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm',\
                    'out_vel', 'Hb_BLR1_peak', 'Hb_BLR2_peak', 'Hb_BLR_fwhm1', 'Hb_BLR_fwhm2', 'Hb_BLR_vel',\
                    'Hb_nar_peak', 'Hb_out_peak', 'Fe_peak', 'Fe_fwhm')
            
            fitted_model = OIII_Fe_QSO
            
            res = {'name': 'OIII_QSO_fe'}
            for i in range(len(labels)):
                res[labels[i]] = flat_samples[:,i]
                
    if outflow=='QSO_bkp':
        if template==0:
            pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                    priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['out_vel'][0],\
                    peak_beta, priors['Hb_BLR_vel'][0], priors['Hb_BLR_alp1'][0], priors['Hb_BLR_alp2'][0],priors['Hb_BLR_sig'][0], \
                    peak_beta/4, peak_beta/4])
            
            print(pos_l)
            priors['z'] = [z, norm(z,0.01)]
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(z,0.001, nwalkers)
            
            prl = [ priors[key][1] for key in list(priors.keys()) ]
            print(prl)
           
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],prl, OIII_QSO_BKPL,log_prior_OIII_QSO_BKPL))
            
            sampler.run_mcmc(pos, N, progress=progress);
            
            flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
                
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm',\
                    'out_vel', 'Hb_BLR_peak', 'Hb_BLR_vel', 'Hb_BLR_alp1', 'Hb_BLR_alp2','Hb_BLR_sig' ,\
                    'Hb_nar_peak', 'Hb_out_peak')
            
            fitted_model = OIII_QSO_BKPL
            
            res = {'name': 'OIII_QSO_BKP'}
            for i in range(len(labels)):
                res[labels[i]] = flat_samples[:,i]
        
        
    return res, fitted_model

def fitting_Halpha_OIII(wave, fluxs, error,z,zcont=0.01,outflow=0 ,progress=True,N=6000, priors= {'cont':[0,-3,1],\
                                                                'cont_grad':[0,-1.,1], \
                                                                'OIIIn_peak':[0,-3,1],\
                                                                'Hbeta_peak':[0,-3,1],\
                                                                'Hal_peak':[0,-3,1],\
                                                                'NII_peak':[0,-3,1],\
                                                                'Nar_fwhm':[300,150,900],\
                                                                'SII_rpk':[0,-3,1],\
                                                                'SII_bpk':[0,-3,1],\
                                                                'OI_peak':[0,-3,1],\
                                                                'outflow_fwhm':[300,300,900],\
                                                                'outflow_vel':[-100,-600,600],\
                                                                'Hal_out_peak':[0,-3,1],\
                                                                'NII_out_peak':[0,-3,1],\
                                                                'OIII_out_peak':[0,-3,1],\
                                                                'Hbeta_out_peak':[0,-3,1],\
                                                                'OI_out_peak':[0,-3,1]}):
    priors['z'] = [z, z-zcont, z+zcont]
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
    fit_loc = np.append(fit_loc, np.where((wave>(6300-50)*(1+z)/1e4)&(wave<(6300+50)*(1+z)/1e4))[0])
    fit_loc = np.append(fit_loc, np.where((wave>(6564.52-170)*(1+z)/1e4)&(wave<(6564.52+170)*(1+z)/1e4))[0])
    
# =============================================================================
#     Finding the initial conditions
# =============================================================================
    sel=  np.where((wave<5025*(1+z)/1e4)& (wave>4980*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc_OIII = np.argmax(flux_zoom)
    peak_OIII = (np.max(flux_zoom))
    
    sel=  np.where(((wave<(6564.52+20)*(1+z)/1e4))& (wave>(6564.52-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    '''
    znew = wave_zoom[peak_loc]/0.656452-1
    if abs(znew-z)<zcont:
        z= znew
    '''
    peak_hal = np.ma.max(flux_zoom)
    
# =============================================================================
#   Setting up fitting  
# =============================================================================
    if outflow==0:
        nwalkers=64
        fitted_model = Halpha_OIII
        log_prior = log_prior_Halpha_OIII
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, priors['Nar_fwhm'][0], peak_hal*0.15, peak_hal*0.2, peak_OIII*0.8,\
                          peak_hal*0.2, peak_hal*0.3])  
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
       
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, fitted_model, log_prior)) 
        sampler.run_mcmc(pos, N, progress=progress);
        
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
    
            
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SII_rpk', 'SII_bpk', 'OIIIn_peak', 'Hbeta_peak', 'OI_peak')
        
        
        
        res = {'name': 'Halpha_OIII'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
            
    if outflow==1:
        nwalkers=64
        fitted_model = Halpha_OIII_outflow
        log_prior = log_prior_Halpha_OIII_outflow
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, \
                          peak_OIII*0.8, peak_OIII*0.3, peak_hal*0.2, peak_hal*0.2, peak_hal*0.1,\
                          priors['Nar_fwhm'][0], priors['outflow_fwhm'][0], priors['outflow_vel'][0],
                          peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.1, peak_OIII*0.2])
                          
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
       
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors, fitted_model, log_prior)) 
        
        sampler.run_mcmc(pos, N, progress=progress);
        
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                                  
                                  
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIIIn_peak', 'Hbeta_peak','SII_rpk', 'SII_bpk','OI_peak',\
                'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'OI_out_peak', 'Hbeta_out_peak'   )
        
        
        
        res = {'name': 'Halpha_OIII_outflow'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
        
    return res, fitted_model

def log_probability_general(theta, x, y, yerr, priors, model, logpriorfce, template=None):
    lp = logpriorfce(theta,priors)
    
    try:
        if not np.isfinite(lp):
            return -np.inf
    except:
        lp[np.isnan(lp)] = -np.inf
    
    if template:
        evalm = model(x,*theta, template)
    else:
        evalm = model(x,*theta)
       
    sigma2 = yerr*yerr
    log_likelihood = -0.5 * np.sum((y - evalm) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))
    
    return lp + log_likelihood


def Fitting_OIII_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    
    flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, outflow=0, progress=False, priors=priors)
    cube_res  = [i,j,prop_calc(flat_samples_sig)]
    return cube_res

def Fitting_Halpha_OIII_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        flat_samples_sig, fitted_model_sig = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors)
        cube_res  = [i,j,prop_calc(flat_samples_sig), flat_samples_sig]
    except:
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
    return cube_res

def Fitting_OIII_2G_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, outflow=1, progress=False)
    cube_res  = [i,j,prop_calc(flat_samples_sig)]
    
    return cube_res

import time

def Fitting_Halpha_unwrap(lst): 
    
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
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


