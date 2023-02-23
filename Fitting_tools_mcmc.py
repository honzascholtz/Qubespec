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


def gauss(x, k, mu,sig):

    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

from Halpha_models import *
from OIII_models import *
from Halpha_OIII_models import *
from QSO_models import *
import numba
import Support as sp


# =============================================================================
#  Primary function to fit Halpha both with or without BLR - data prep and fit 
# =============================================================================
def fitting_Halpha(wave, fluxs, error,z, model='BLR',zcont=0.05, progress=True ,N=6000,priors= {'z':[0, 'normal', 0,0.003],\
                                                                                   'cont':[0,'loguniform',-3,1],\
                                                                                   'cont_grad':[0,'normal',0,0.3], \
                                                                                   'Hal_peak':[0,'loguniform',-3,1],\
                                                                                   'BLR_Hal_peak':[0,'loguniform',-3,1],\
                                                                                   'NII_peak':[0,'loguniform',-3,1],\
                                                                                   'Nar_fwhm':[300,'uniform',100,900],\
                                                                                   'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                   'zBLR':[0, 'normal', 0,0.003],\
                                                                                    'SIIr_peak':[0,'loguniform',-3,1],\
                                                                                    'SIIb_peak':[0,'loguniform',-3,1],\
                                                                                    'Hal_out_peak':[0,'loguniform',-3,1],\
                                                                                    'NII_out_peak':[0,'loguniform',-3,1],\
                                                                                    'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                    'outflow_vel':[-50,'normal', 0,300]}):
    
    if priors['z'][0]==0:
        priors['z'][0]=z
    
    try:
        if priors['zBLR'][0]==0:
            priors['zBLR'][0]=z
    except:
        lksdf=0
        
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
    
    if model=='BLR':
        
        labels=('z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak')
        
        fitted_model = Halpha_wBLR
        
        pr_code = prior_create(labels, priors)
        
        pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/4, peak/4, priors['Nar_fwhm'][0], priors['BLR_fwhm'][0],priors['zBLR'][0],peak/6, peak/6])
        
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
            
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_Halpha_BLR))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
        
        res = {'name': 'Halpha_wth_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    
    elif model=='gal':
        fitted_model = Halpha
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak')
        pr_code = prior_create(labels, priors)
        
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4,priors['Nar_fwhm'][0],peak/6, peak/6 ])
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
        
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_Halpha))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
    
        res = {'name': 'Halpha_wth_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
            
    elif model=='outflow':
        fitted_model = Halpha_outflow
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel')
        pr_code = prior_create(labels, priors)
        
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4, priors['Nar_fwhm'][0],peak/6, peak/6,peak/8, peak/8, priors['outflow_fwhm'][0],priors['outflow_vel'][0] ])
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
    
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_Halpha_outflow))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
    
        
        res = {'name': 'Halpha_wth_out'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    
    elif model=='QSO_BKPL':
        
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
                'Hal_out_peak', 'NII_out_peak', 
                'outflow_fwhm', 'outflow_vel', \
                'BLR_Hal_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2', 'BLR_sig')
            
            
        pos_l = np.array([z,np.median(flux[fit_loc]),0.01, peak/2, peak/4, priors['Nar_fwhm'][0],\
                          peak/6, peak/6,\
                          priors['outflow_fwhm'][0],priors['outflow_vel'][0], \
                          peak, priors['zBLR'][0], priors['BLR_alp1'][0], priors['BLR_alp2'][0],priors['BLR_sig'][0], \
                          ])
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0]    
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        pr_code = prior_create(labels, priors)
        
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, Hal_QSO_BKPL, logprior_general))
    
        sampler.run_mcmc(pos, N, progress=progress);
    
        flat_samples = sampler.get_chain(discard=int(0.25*N), thin=15, flat=True)
        
        
        
        fitted_model = Hal_QSO_BKPL
        
        res = {'name': 'Halpha_QSO_BKPL'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    else:
        raise Exception('model variable not understood. Available model keywords: BLR, outflow, gal, QSO_BKPL')
         
    
            
        
    return res, fitted_model
    
# =============================================================================
# Primary function to fit [OIII] with and without outflows. 
# =============================================================================

def fitting_OIII(wave, fluxs, error,z, model='gal', template=0, Hbeta_dual=0,N=6000, progress=True, \
                                                                 priors= {'z': [0,'normal',0, 0.003],\
                                                                'cont':[0,'loguniform',-3,1],\
                                                                'cont_grad':[0,'normal',0,0.2], \
                                                                'OIII_peak':[0,'loguniform',-3,1],\
                                                                'OIII_out_peak':[0,'loguniform',-3,1],\
                                                                'Nar_fwhm':[300,'uniform', 100,900],\
                                                                'outflow_fwhm':[700,'uniform',600,2500],\
                                                                'outflow_vel':[-100,'normal',0,200],\
                                                                'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                'Hbeta_fwhm':[200,'uniform',120,7000],\
                                                                'Hbeta_vel':[10,'normal', 0,200],\
                                                                'Hbetan_peak':[0,'loguniform',-3,1],\
                                                                'Hbetan_fwhm':[300,'uniform',120,700],\
                                                                'Hbetan_vel':[10,'normal', 0,100],\
                                                                'Fe_peak':[0,'loguniform',-3,1],\
                                                                'Fe_fwhm':[3000,'uniform',2000,6000]}):
    
    if priors['z'][2]==0:
        priors['z'][2]=z
    
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
    if model=='outflow': 
        if template==0:
            if Hbeta_dual == 0:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel')
                fitted_model = OIII_outflow
                
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['Nar_fwhm'][0], priors['outflow_fwhm'][0],priors['outflow_vel'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0]])
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_outflow))
                
                sampler.run_mcmc(pos, N, progress=progress);
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                res = {'name': 'OIII_outflow'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
            else:
                labels= ('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel')
                fitted_model = OIII_outflow_narHb
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['Nar_fwhm'][0], priors['outflow_fwhm'][0],priors['outflow_vel'][0],\
                                peak_beta, priors['Hbeta_fwhm'][0], priors['Hbeta_vel'][0],peak_beta, priors['Hbetan_fwhm'][0], priors['Hbetan_vel'][0]])
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_outflow_narHb))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                   
                res = {'name': 'OIII_outflow_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
         
        else:
            if Hbeta_dual == 0:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                fitted_model = OIII_outflow_Fe
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6, priors['Nar_fwhm'][0], priors['OIII_out'][0],priors['outflow_vel'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                np.median(flux[fit_loc]), priors['Fe_fwhm'][0]])
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
               
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_outflow_Fe, template))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                res = {'name': 'OIII_outflow_Fe'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
            else:
                labels= ('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel', 'Fe_peak', 'Fe_fwhm')
                fitted_model = OIII_outflow_Fe_narHb
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc])/2,0.001, peak/2, peak/4, 300., 600.,-100, \
                                peak_beta/2, 4000,priors['Hbeta_vel'][0],peak_beta/2, 600,priors['Hbetan_vel'][0],\
                                np.median(flux[fit_loc]), 2000])
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_outflow_Fe_narHb, template))
                    
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
            
                res = {'name': 'OIII_outflow_Fe_narHb'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
            
    elif model=='gal': 
        if template==0:
            if Hbeta_dual == 0:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel')
                fitted_model = OIII
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['Nar_fwhm'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0]]) 
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                  
                res = {'name': 'OIII'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
        
            else:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel')
                fitted_model = OIII_dual_hbeta
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['Nar_fwhm'][0], peak_beta/4, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                peak_beta/4, priors['Hbetan_fwhm'][0], priors['Hbetan_vel'][0]])
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_dual_hbeta))
            
                sampler.run_mcmc(pos, N, progress=progress);
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                    
                res = {'name': 'OIII_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
         
        else:
            if Hbeta_dual == 0:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                fitted_model = OIII_Fe
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],np.median(flux[fit_loc]), priors['Fe_fwhm'][0]]) 
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_Fe, template))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                
                res = {'name': 'OIII_Fe'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                
                    
            else:
                labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel','Fe_peak', 'Fe_fwhm')
                fitted_model = OIII_dual_hbeta_Fe
                pr_code = prior_create(labels, priors)
                
                pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2,  priors['OIII_fwhm'][0], peak_beta/2, priors['Hbeta_fwhm'][0],priors['Hbeta_vel'][0],\
                                peak_beta/2, priors['Hbetan_fwhm'][0],priors['Hbetan_vel'][0], \
                                np.median(flux[fit_loc]), priors['Fe_fwhm'][0]]) 
                for i in enumerate(labels):
                    pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                    
                pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                pos[:,0] = np.random.normal(z,0.001, nwalkers)
                
                nwalkers, ndim = pos.shape
                
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior_OIII_dual_hbeta_Fe,  template))
            
                sampler.run_mcmc(pos, N, progress=progress);
                
                flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
                 
                res = {'name': 'OIII_Fe_HBn'}
                for i in range(len(labels)):
                    res[labels[i]] = flat_samples[:,i]
                    
    elif model=='QSO_dg':
        if template==0:
            pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                    priors['OIII_fwhm'][0], priors['OIII_out'][0],priors['outflow_vel'][0],\
                    peak_beta/2, peak_beta/2, \
                    priors['Hb_BLR1_fwhm'][0],priors['Hb_BLR2_fwhm'][0],priors['Hb_BLR_vel'][0],\
                    peak_beta/4, peak_beta/4])
            
            for i in enumerate(labels):
                pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
                
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(z,0.001, nwalkers)
           
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_OIII_QSO, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors))
            
            sampler.run_mcmc(pos, N, progress=progress);
            
            flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
                
            labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm',\
                    'outflow_vel', 'Hb_BLR1_peak', 'Hb_BLR2_peak', 'Hb_BLR_fwhm1', 'Hb_BLR_fwhm2', 'Hb_BLR_vel',\
                    'Hb_nar_peak', 'Hb_out_peak')
            
            fitted_model = OIII_QSO
            
            res = {'name': 'OIII_QSO'}
            for i in range(len(labels)):
                res[labels[i]] = flat_samples[:,i]
    
        else:
            nwalkers=64
            pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                    priors['Nar_fwhm'][0], priors['outflow_fwhm'][0],priors['outflow_vel'][0],\
                    peak_beta/2, peak_beta/2, \
                    priors['Hb_BLR1_fwhm'][0],priors['Hb_BLR2_fwhm'][0],priors['Hb_BLR_vel'][0],\
                    peak_beta/4, peak_beta/4,\
                    np.median(flux[fit_loc]), priors['Fe_fwhm'][0]])
            for i in enumerate(labels):
                pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0]   
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(z,0.001, nwalkers)
           
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],priors,OIII_Fe_QSO,log_prior_OIII_Fe_QSO, template))
            
            sampler.run_mcmc(pos, N, progress=progress);
            
            flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
                
            labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm',\
                    'outflow_vel', 'Hb_BLR1_peak', 'Hb_BLR2_peak', 'Hb_BLR_fwhm1', 'Hb_BLR_fwhm2', 'Hb_BLR_vel',\
                    'Hb_nar_peak', 'Hb_out_peak', 'Fe_peak', 'Fe_fwhm')
            
            fitted_model = OIII_Fe_QSO
            
            res = {'name': 'OIII_QSO_fe'}
            for i in range(len(labels)):
                res[labels[i]] = flat_samples[:,i]
                
    elif model=='QSO_BKPL':
        labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm',\
                    'outflow_vel', 'BLR_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2','BLR_sig' ,\
                    'Hb_nar_peak', 'Hb_out_peak')
            
        pos_l = np.array([z,np.median(flux[fit_loc]),0.001, peak/2, peak/6,\
                priors['Nar_fwhm'][0], priors['outflow_fwhm'][0],priors['outflow_vel'][0],\
                peak_beta, priors['zBLR'][0], priors['BLR_alp1'][0], priors['BLR_alp2'][0],priors['BLR_sig'][0], \
                peak_beta/4, peak_beta/4])
        
            
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0]    
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        pos[:,9] = np.random.normal(z,0.001, nwalkers)
        
        pr_code = prior_create(labels, priors)
         
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
             nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, OIII_QSO_BKPL, logprior_general))
     
        sampler.run_mcmc(pos, N, progress=progress);   
        
        
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
        fitted_model = OIII_QSO_BKPL
        
        res = {'name': 'OIII_QSO_BKP'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    else:
        raise Exception('model variable not understood. Available model keywords: outflow, gal, QSO_BKPL')
        
    return res, fitted_model

def fitting_Halpha_OIII(wave, fluxs, error,z,zcont=0.01,model='gal' ,progress=True,N=6000,initial=np.array([0]), priors={'z':[0,'normal', 0, 0.003],\
                                                                                                 'cont':[0,'loguniform', -3,1],\
                                                                                                 'cont_grad':[0,'normal', 0,0.2],\
                                                                                                 'Hal_peak':[0,'loguniform', -3,1],\
                                                                                                 'NII_peak':[0,'loguniform', -3,1],\
                                                                                                 'Nar_fwhm':[300,'uniform', 200,900],\
                                                                                                 'SIIr_peak':[0,'loguniform', -3,1],\
                                                                                                 'SIIb_peak':[0,'loguniform', -3,1],\
                                                                                                 'OIII_peak':[0,'loguniform', -3,1],\
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
                                                                                                 'BLR_Hal_peak':[0,'loguniform', -3,1],\
                                                                                                 'BLR_Hbeta_peak':[0,'loguniform', -3,1],\
                                                                                                 }):
    
    if priors['z'][2]==0:
        priors['z'][2]=z
    try:
        if priors['zBLR'][2]==0:
            priors['zBLR'][2]=z
    except:
        lksdf=0
        
        
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
    try:
        peak_loc_OIII = np.argmax(flux_zoom)
        peak_OIII = (np.max(flux_zoom))
    except:
        peak_OIII = np.max(flux[fit_loc])
    
    sel=  np.where(((wave<(6564.52+20)*(1+z)/1e4))& (wave>(6564.52-20)*(1+z)/1e4))[0]
    flux_zoom = flux[sel]
    wave_zoom = wave[sel]
    
    peak_loc = np.ma.argmax(flux_zoom)
    
    znew = wave_zoom[peak_loc]/0.656452-1
    '''
    if abs(znew-z)<zcont:
        z= znew
        priors['z'][0] = znew
        priors['z'][2] = znew
    '''
    peak_hal = np.ma.max(flux_zoom)
    if peak_hal<0:
        peak_hal==3e-3
    
# =============================================================================
#   Setting up fitting  
# =============================================================================
    if model=='gal':
        #priors['z'] = [z, z-zcont, z+zcont]
        nwalkers=64
        fitted_model = Halpha_OIII
        log_prior = log_prior_Halpha_OIII
        
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak', 'OI_peak')
        
        pr_code = prior_create(labels, priors)
        
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, priors['Nar_fwhm'][0], peak_hal*0.15, peak_hal*0.2, peak_OIII*0.8,\
                          peak_hal*0.2, peak_OIII*0.1])  
        
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
            
        if (log_prior(pos_l, pr_code)==-np.inf):
            logprior_general_test(pos_l, pr_code, labels)
            
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible. {pos_l}')
            
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior)) 
        sampler.run_mcmc(pos, N, progress=progress);
        
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
        
        res = {'name': 'Halpha_OIII'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
            
    elif model=='outflow':
        nwalkers=64
        fitted_model = Halpha_OIII_outflow
        
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak','OI_peak',\
                'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'OI_out_peak', 'Hbeta_out_peak'   )
        
        pr_code = prior_create(labels, priors)   
        log_prior = log_prior_Halpha_OIII_outflow
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, \
                          peak_OIII*0.8, peak_hal*0.2, peak_hal*0.2, peak_hal*0.2, peak_hal*0.1,\
                          priors['Nar_fwhm'][0], priors['outflow_fwhm'][0], priors['outflow_vel'][0],
                          peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.05, peak_hal*0.05])
        
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
            
        if (log_prior(pos_l, pr_code)==-np.inf):
            
            logprior_general_test(pos_l, pr_code, labels)
            
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible. {pos_l}')
                            
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
       
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior)) 
        
        sampler.run_mcmc(pos, N, progress=progress);
        
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
        res = {'name': 'Halpha_OIII_outflow'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    elif model=='BLR':
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' ,\
                'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak')
            
        nwalkers=64
        
        if priors['BLR_Hal_peak'][2]=='error':
            priors['BLR_Hal_peak'][2]=error[-2]; priors['BLR_Hal_peak'][3]=error[-2]*2
        if priors['BLR_Hbeta_peak'][2]=='error':
            priors['BLR_Hbeta_peak'][2]=error[2]; priors['BLR_Hbeta_peak'][3]=error[2]*2
        
        pr_code = prior_create(labels, priors)   
        log_prior = log_prior_Halpha_OIII_BLR
        fitted_model = Halpha_OIII_BLR
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, \
                          peak_OIII*0.8, peak_hal*0.3 , peak_hal*0.2, peak_hal*0.2,\
                          priors['Nar_fwhm'][0], priors['outflow_fwhm'][0], priors['outflow_vel'][0],
                          peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.1,\
                          priors['BLR_fwhm'][0], priors['zBLR'][0], peak_hal*0.3, peak_hal*0.1])
            
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
             
        
        
        if (log_prior(pos_l, pr_code)==np.nan)|\
            (log_prior(pos_l, pr_code)==-np.inf):
            print(logprior_general_test(pos_l, pr_code, labels))
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible. {pos_l} ')
        
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        pos[:,-3] = np.random.normal(priors['zBLR'][0],0.00001, nwalkers)
       
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior)) 
        sampler.run_mcmc(pos, N, progress=progress);
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
          
        
        res = {'name': 'Halpha_OIII_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
        
    elif model=='BLR_simple':
        labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak')
            
        nwalkers=64
        
        if priors['BLR_Hal_peak'][2]=='error':
            priors['BLR_Hal_peak'][2]=error[-2]; priors['BLR_Hal_peak'][3]=error[-2]*2
        if priors['BLR_Hbeta_peak'][2]=='error':
            priors['BLR_Hbeta_peak'][2]=error[2]; priors['BLR_Hbeta_peak'][3]=error[2]*2
        
        pr_code = prior_create(labels, priors) 
        log_prior = log_prior_Halpha_OIII_BLR_simple
        fitted_model = Halpha_OIII_BLR_simple
        
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, \
                          peak_OIII*0.8, peak_hal*0.3 , peak_hal*0.2, peak_hal*0.2,\
                          priors['Nar_fwhm'][0], priors['BLR_fwhm'][0], priors['zBLR'][0], peak_hal*0.3, peak_hal*0.1])
            
        for i in enumerate(labels):
            pos_l[i[0]] = pos_l[i[0]] if priors[i[1]][0]==0 else priors[i[1]][0] 
             
        
        
        if (log_prior(pos_l, pr_code)==np.nan)|\
            (log_prior(pos_l, pr_code)==-np.inf):
            print(logprior_general_test(pos_l, pr_code, labels))
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible. {pos_l} ')
        
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        pos[:,-3] = np.random.normal(priors['zBLR'][0],0.00001, nwalkers)
       
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior)) 
        sampler.run_mcmc(pos, N, progress=progress);
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
          
        
        res = {'name': 'Halpha_OIII_BLR_simple'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
        
        
    elif outflow=='QSO_BKPL':
        labels =[ 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak','Hbeta_peak', 'Nar_fwhm', \
                              'Hal_out_peak', 'NII_out_peak','OIII_out_peak', 'Hbeta_out_peak',\
                              'outflow_fwhm', 'outflow_vel',\
                              'Hal_BLR_peak', 'Hbeta_BLR_peak',  'BLR_vel', 'BLR_alp1', 'BLR_alp2', 'BLR_sig']
    
    
        nwalkers=64
        pr_code = prior_create(labels, priors)   
        log_prior = logprior_general
        fitted_model =  Halpha_OIII_QSO_BKPL
        pos_l = np.array([z,np.median(flux[fit_loc]), -0.1, peak_hal*0.7, peak_hal*0.3, \
                          peak_OIII*0.8, peak_OIII*0.3,priors['Nar_fwhm'][0],\
                          peak_hal*0.2, peak_hal*0.3, peak_OIII*0.4, peak_OIII*0.2   ,\
                          priors['outflow_fwhm'][0], priors['outflow_vel'][0],\
                          peak_hal*0.4, peak_OIII*0.4, priors['BLR_vel'][0], 
                          priors['BLR_alp1'][0], priors['BLR_alp2'][0],priors['BLR_sig'][0]])
        
        if (log_prior(pos_l, pr_code)==np.nan)|\
            (log_prior(pos_l, pr_code)==-np.inf):
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible: {pos_l}')
                            
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability_general, args=(wave[fit_loc], flux[fit_loc], error[fit_loc],pr_code, fitted_model, log_prior)) 
        sampler.run_mcmc(pos, N, progress=progress);
        flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
          
        
        res = {'name': 'Halpha_OIII_BLR'}
        for i in range(len(labels)):
            res[labels[i]] = flat_samples[:,i]
    else:
        raise Exception('model variable not understood. Available model keywords: outflow, gal, QSO_BKPL')
            
    return res, fitted_model

import numba
@numba.njit
def logprior_general(theta, priors):
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


def fitting_general(wave, fluxs, error,z, priors, fitted_model, labels, logprior= logprior_general, progress=True,N=6000, nwalkers=64):
    
    if priors['z'][2]==0:
        priors['z'][2]=z
        
    flux = fluxs.data[np.invert(fluxs.mask)]
    wave = wave[np.invert(fluxs.mask)]
    
    pr_code = prior_create(labels, priors)
   
    pos_l = np.zeros(len(labels))
        
    for i in enumerate(labels):
        pos_l[i[0]] = priors[i[1]][0] 
            
    if (logprior(pos_l, pr_code)==-np.inf):
        logprior_general_test(pos_l, pr_code, labels)
            
        raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                        boundries are sensible')
            
    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
    pos[:,0] = np.random.normal(z,0.001, nwalkers)
        
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_general, args=(wave, flux, error,pr_code, fitted_model, logprior)) 
    sampler.run_mcmc(pos, N, progress=progress);
        
    flat_samples = sampler.get_chain(discard=int(0.5*N), thin=15, flat=True)
        
    res = {'name': 'Custom model'}
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



def logprior_general_test(theta, priors, labels):
    for t,p,lb in zip( theta, priors, labels):
        print(p)
        if p[0] ==0:
            results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0] ==1:
            results = np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results = np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
    
        print(lb, t, results)
        
def prior_create(labels,priors):
    pr_code = np.zeros((len(labels),3))
    
    for key in enumerate(labels):
        if priors[key[1]][1] == 'normal':
            pr_code[key[0]][0] = 0
            
        
        elif priors[key[1]][1] == 'lognormal':
            pr_code[key[0]][0] = 2
            
        elif priors[key[1]][1] == 'uniform':
            pr_code[key[0]][0] = 1
            
        elif priors[key[1]][1] == 'loguniform':
            pr_code[key[0]][0] = 3
        
        else:
            raise Exception('Sorry mode in prior type not understood: ', key )
        
        pr_code[key[0]][1] = priors[key[1]][2]
        pr_code[key[0]][2] = priors[key[1]][3]
    return pr_code

def Fitting_OIII_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    
    flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, model='gal', progress=False, priors=priors)
    cube_res  = [i,j,sp.prop_calc(flat_samples_sig)]
    return cube_res

def Fitting_Halpha_OIII_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        flat_samples_sig, fitted_model_sig = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors,N=10000)
        cube_res  = [i,j,sp.prop_calc(flat_samples_sig), flat_samples_sig,wave,flx_spax_m,error]
    except:
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
    return cube_res

def Fitting_Halpha_OIII_AGN_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        flat_samples_sig, fitted_model_sig = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors, model='BLR', N=10000)
        cube_res  = [i,j,sp.prop_calc(flat_samples_sig), flat_samples_sig,wave,flx_spax_m,error ]
    except:
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
    return cube_res

def Fitting_Halpha_OIII_outflowboth_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
     
    
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        flat_samples_sig, fitted_model_sig = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors, model='gal', N=10000)    
        flat_samples_out, fitted_model_out = fitting_Halpha_OIII(wave,flx_spax_m,error,z,zcont=deltaz, progress=progress, priors=priors, model='outflow', N=10000)
        
        BIC_sig = sp.BIC_calc(wave, flx_spax_m, error, fitted_model_sig, sp.prop_calc(flat_samples_sig), 'Halpha_OIII' )
        BIC_out = sp.BIC_calc(wave, flx_spax_m, error, fitted_model_out, sp.prop_calc(flat_samples_out), 'Halpha_OIII' )
        
        if (BIC_sig[1]-BIC_out[1])>5:
            fitted_model = fitted_model_out
            flat_samples = flat_samples_out
        else:
            fitted_model = fitted_model_sig
            flat_samples = flat_samples_sig
            
        cube_res  = [i,j,sp.prop_calc(flat_samples), flat_samples,wave,flx_spax_m,error ]
    except:
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        print('Failed fit')
    return cube_res

def Fitting_OIII_2G_unwrap(lst, priors):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
        
    i,j,flx_spax_m, error, wave, z = lst
    
    try:
        flat_samples_sig, fitted_model_sig = fitting_OIII(wave,flx_spax_m,error,z, model='gal', progress=False, priors=priors) 
        flat_samples_out, fitted_model_out = fitting_OIII(wave,flx_spax_m,error,z, model='outflow', progress=False, priors=priors)
        
        BIC_sig = sp.BIC_calc(wave, flx_spax_m, error, fitted_model_sig, sp.prop_calc(flat_samples_sig), 'OIII' )
        BIC_out = sp.BIC_calc(wave, flx_spax_m, error, fitted_model_out, sp.prop_calc(flat_samples_out), 'OIII' )
        
        if (BIC_sig[1]-BIC_out[1])>5:
            fitted_model = fitted_model_out
            flat_samples = flat_samples_out
        else:
            fitted_model = fitted_model_sig
            flat_samples = flat_samples_sig
            
        cube_res  = [i,j,sp.prop_calc(flat_samples), flat_samples,wave,flx_spax_m,error ]
    except:
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        print('Failed fit')
    return cube_res
    
import time

def Fitting_Halpha_unwrap(lst): 
    
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    
    i,j,flx_spax_m, error, wave, z = lst
    
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    flat_samples_sig, fitted_model_sig = fitting_Halpha(wave,flx_spax_m,error,z, zcont=deltaz, model='BLR', progress=False, priors=priors,N=10000)
    cube_res  = [i,j,sp.prop_calc(flat_samples_sig)]
    
    return cube_res
    
    

