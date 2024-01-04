#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:35:45 2017

@author: jscholtz
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt

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

PATH_TO_FeII = '/Users/jansen/My Drive/Astro/General_data/FeII_templates/'

version = 'Main'    


def gauss(x, k, mu,sig):

    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

import warnings
warnings.filterwarnings("ignore")

from .Models import OIII_models as O_models
from .Models import Halpha_OIII_models as HO_models
from .Models import QSO_models as QSO_models
from .Models import Halpha_models as H_models
from .Models import Custom_model
import numba
from . import Support as sp



class Fitting:
    """ Simple class containing everything that we need to fit a spectrum and also all of its results. 

    Parameters
    ----------
    wave - observed wavelength in microns
    flux - flux of the spectrum
    error` - error on the spectrum
    z - redshift of the source
    N - number of points in the chain - default 5000
    ncpu - number of cpus used to fit - I find that the overheads can be bigger what using multipleprocessing then the speed up. Experiment ok keep to 1
    progress - progress bar for the emcee bit
    prior_update - dictionary with all of the priors 
    """
       
    def __init__(self, wave='', flux='', error='', z='', N=5000,ncpu=1, progress=True, prior_update= {'z':[0, 'normal', 0,0.003]}):
        
        priors= {'z':[0, 'normal', 0,0.003],\
                'cont':[0,'loguniform',-4,1],\
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
                'outflow_vel':[-50,'normal', 0,300],\
                'OIII_peak':[0,'loguniform',-3,1],\
                'OIII_out_peak':[0,'loguniform',-3,1],\
                'Hbeta_peak':[0,'loguniform',-3,1],\
                'Hbeta_out_peak':[0,'loguniform',-3,1],\
                'Hbeta_fwhm':[200,'uniform',120,7000],\
                'Hbeta_vel':[10,'normal', 0,200],\
                'Hbetan_peak':[0,'loguniform',-3,1],\
                'Hbetan_fwhm':[300,'uniform',120,700],\
                'Hbetan_vel':[10,'normal', 0,100],\
                'Fe_peak':[0,'loguniform',-3,1],\
                'Fe_fwhm':[3000,'uniform',2000,6000],\
                'SIIr_peak':[0,'loguniform', -3,1],\
                'SIIb_peak':[0,'loguniform', -3,1],\
                'BLR_Hbeta_peak':[0,'loguniform', -3,1]}
        
        for name in list(prior_update.keys()):
            priors[name] = prior_update[name]

        self.N = N # Number of points in the chains
        self.priors = priors # storing priors
        self.progress = progress # progress bar?
        self.z = z  # redshift
        self.wave = wave # wavelength 
        self.fluxs = flux # flux density
        self.error = error # errors
        self.ncpu= ncpu # number of cpus to use in the fit 
    
    # =============================================================================
    #  Primary function to fit Halpha both with or without BLR - data prep and fit 
    # =============================================================================
    def fitting_Halpha(self, model='gal'):
        """ ok lets fit Halpha only - Models - gal, BLR, outflow, QSO_BKPL
        
        """
        self.model= model
        self.template = None
        if self.priors['z'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['z'][1]=='normal_hat') & (self.priors['z'][2]==0):
                self.priors['z'][2] = self.z
                self.priors['z'][3] = 200/3e5*(1+self.z)
                self.priors['z'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['z'][5] = self.z+1000/3e5*(1+self.z)
        try:
            if self.priors['zBLR'][0]==0:
                self.priors['zBLR'][0]=self.z
        except:
            lksdf=0
            
        self.fluxs[np.isnan(self.fluxs)] = 0
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.wave = self.wave[np.invert(self.fluxs.mask)]
         
        self.fit_loc = np.where((self.wave>(6564.52-170)*(1+self.z)/1e4)&(self.wave<(6564.52+170)*(1+self.z)/1e4))[0]
        sel=  np.where(((self.wave<(6564.52+20)*(1+self.z)/1e4))& (self.wave>(6564.52-20)*(1+self.z)/1e4))[0]
        
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        
        peak = np.ma.max(self.flux_zoom)
        nwalkers=32
        
        if self.model=='BLR':
            self.labels=['z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak']
            
            self.fitted_model = H_models.Halpha_wBLR
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
            
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.001, peak/2, peak/4, peak/4, self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],self.priors['zBLR'][0],peak/6, peak/6])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'Halpha_wth_BLR'}
        
        elif self.model=='gal':
            self.fitted_model = H_models.Halpha
            self.log_prior_fce = logprior_general
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak']
            self.pr_code = self.prior_create()
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.01, peak/2, peak/4,self.priors['Nar_fwhm'][0],peak/6, peak/6 ])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 

            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            self.res = {'name': 'Halpha_wth_BLR'}
            
                
        elif self.model=='outflow':
            self.fitted_model = H_models.Halpha_outflow
            self.log_prior_fce = logprior_general
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel']
            self.pr_code = self.prior_create()
            
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.01, peak/2, peak/4, self.priors['Nar_fwhm'][0],peak/6, peak/6,peak/8, peak/8, self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0] ])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            
            self.res = {'name': 'Halpha_wth_out'}
            
        
        elif self.model=='QSO_BKPL':
            
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
                    'Hal_out_peak', 'NII_out_peak', 
                    'outflow_fwhm', 'outflow_vel', \
                    'BLR_Hal_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2', 'BLR_sig']
                
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.01, peak/2, peak/4, self.priors['Nar_fwhm'][0],\
                              peak/6, peak/6,\
                              self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], \
                              peak, self.priors['zBLR'][0], self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],self.priors['BLR_sig'][0], \
                              ])
            
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0]    
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.pr_code = self.prior_create()
            self.fitted_model = QSO_models.Hal_QSO_BKPL
            self.log_prior_fce = logprior_general
            self.res = {'name': 'Halpha_QSO_BKPL'}
            
        else:
            raise Exception('self.model variable not understood. Available self.model keywords: BLR, outflow, gal, QSO_BKPL')
        
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            print(logprior_general_scipy_test(pos_l, self.pr_code))
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')

        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
             nwalkers, ndim, self.log_probability_general, args=())
     
        sampler.run_mcmc(pos, self.N, progress=self.progress)
        self.flat_samples = sampler.get_chain(discard=int(0.25*self.N), thin=15, flat=True)      
        
        self.chains = {'name': 'Halpha'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
            
        self.props = self.prop_calc()
        try:
            self.chi2, self.BIC = sp.BIC_calc(self.wave, self.fluxs, self.error, self.fitted_model, self.props, 'Halpha')
        except:
            self.chi2, self.BIC = np.nan, np.nan
        
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        self.yeval = self.fitted_model(self.wave, *self.props['popt'])
        
    # =============================================================================
    # Primary function to fit [OIII] with and without outflows. 
    # =============================================================================
    
    def fitting_OIII(self, model, template=0, Hbeta_dual=0, plot=0):
        """ Lets fit the OIII
        
        """
        self.model = model
        self.template = template
        self.Hbeta_dual = Hbeta_dual
        if self.priors['z'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['z'][1]=='normal_hat') & (self.priors['z'][2]==0):
                self.priors['z'][2] = self.z
                self.priors['z'][3] = 200/3e5*(1+self.z)
                self.priors['z'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['z'][5] = self.z+1000/3e5*(1+self.z)

        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.wave = self.wave[np.invert(self.fluxs.mask)]
        
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        sel=  np.where((self.wave<5025*(1+self.z)/1e4)& (self.wave>4980*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        
        peak_loc = np.argmax(self.flux_zoom)
        peak = (np.max(self.flux_zoom))
        
        selb =  np.where((self.wave<4880*(1+self.z)/1e4)& (self.wave>4820*(1+self.z)/1e4))[0]
        self.flux_zoomb = self.flux[selb]
        self.wave_zoomb = self.wave[selb]
        try:
            peak_loc_beta = np.argmax(self.flux_zoomb)
            peak_beta = (np.max(self.flux_zoomb))
        except:
            peak_beta = peak/3
        
        if plot==1:
            print(self.flux[self.fit_loc], self.error[self.fit_loc])

        nwalkers=64

        if self.model== 'outflow_simple':
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak']
            self.fitted_model = O_models.OIII_outflow_simple
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
                    
            pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_scipy_test(pos_l, self.pr_code))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
            
            self.res = {'name': 'OIII_outflow_simple'}
        
        elif self.model== 'gal_simple':
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak']
            self.fitted_model = O_models.OIII_simple
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
                    
            pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'OIII_simple'}

            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_scipy_test(pos_l, self.pr_code))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
        
        elif self.model=='outflow': 
            if self.template==0:
                if self.Hbeta_dual == 0:
                    self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel']
                    self.fitted_model = O_models.OIII_outflow
                    self.log_prior_fce = logprior_general
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0]])
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
                    self.res = {'name': 'OIII_outflow'}
                    
                else:
                    self.labels= ['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel']
                    self.fitted_model = O_models.OIII_outflow_narHb
                    self.log_prior_fce = logprior_general
                    self.pr_code =self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0],\
                                    peak_beta, self.priors['Hbeta_fwhm'][0], self.priors['Hbeta_vel'][0],peak_beta, self.priors['Hbetan_fwhm'][0], self.priors['Hbetan_vel'][0]])
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
                    
                    self.res = {'name': 'OIII_outflow_HBn'}
                    
             
            else:
                if self.Hbeta_dual == 0:
                    self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                    self.fitted_model = O_models.OIII_outflow_Fe
                    self.log_prior_fce = logprior_general
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['OIII_out'][0],self.priors['outflow_vel'][0], peak_beta, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0],\
                                    np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]])
                        
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
                    
                    self.res = {'name': 'OIII_outflow_Fe'}
                        
                else:
                    self.labels= ('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel', 'Fe_peak', 'Fe_fwhm')
                    self.fitted_model = O_models.OIII_outflow_Fe_narHb
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc])/2,0.001, peak/2, peak/4, 300., 600.,-100, \
                                    peak_beta/2, 4000,self.priors['Hbeta_vel'][0],peak_beta/2, 600,self.priors['Hbetan_vel'][0],\
                                    np.median(self.flux[self.fit_loc]), 2000])
                    
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
                
                    self.res = {'name': 'OIII_outflow_Fe_narHb'}
                    
                
        elif self.model=='gal': 
            if self.template==0:
                if self.Hbeta_dual == 0:
                    self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel')
                    self.fitted_model = O_models.OIII
                    self.log_prior_fce = logprior_general
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2,  self.priors['Nar_fwhm'][0], peak_beta, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0]]) 
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
                    
                    self.res = {'name': 'OIII'}
                    
            
                else:
                    self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel')
                    self.fitted_model = O_models.OIII_dual_hbeta
                    self.log_prior_fce = logprior_general_scipy
                    self.pr_code =self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2,  self.priors['Nar_fwhm'][0], peak_beta/4, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0],\
                                    peak_beta/4, self.priors['Hbetan_fwhm'][0], self.priors['Hbetan_vel'][0]])
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)     
                    self.res = {'name': 'OIII_HBn'}
            

            else:
                if self.Hbeta_dual == 0:
                    self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel', 'Fe_peak', 'Fe_fwhm')
                    self.fitted_model = O_models.OIII_Fe
                    self.log_prior_fce = logprior_general_scipy
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2,  self.priors['OIII_fwhm'][0], peak_beta, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0],np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]]) 
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                    
                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
                    self.res = {'name': 'OIII_Fe'}
                
                else:
                    self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'Hbeta_fwhm','Hbeta_vel','Hbetan_peak', 'Hbetan_fwhm','Hbetan_vel','Fe_peak', 'Fe_fwhm')
                    self.fitted_model = O_models.OIII_dual_hbeta_Fe
                    self.log_prior_fce = logprior_general_scipy
                    self.pr_code = self.prior_create()
                    
                    pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2,  self.priors['OIII_fwhm'][0], peak_beta/2, self.priors['Hbeta_fwhm'][0],self.priors['Hbeta_vel'][0],\
                                    peak_beta/2, self.priors['Hbetan_fwhm'][0],self.priors['Hbetan_vel'][0], \
                                    np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]]) 
                    for i in enumerate(self.labels):
                        pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 

                    pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
                    pos[:,0] = np.random.normal(self.z,0.001, nwalkers)  
                    self.res = {'name': 'OIII_Fe_HBn'}
        
                    
        elif self.model=='QSO_BKPL':
            self.labels=('z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm',\
                        'outflow_vel', 'BLR_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2','BLR_sig' ,\
                        'Hb_nar_peak', 'Hb_out_peak')
                
            pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6,\
                    self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0],\
                    peak_beta, self.priors['zBLR'][0], self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],self.priors['BLR_sig'][0], \
                    peak_beta/4, peak_beta/4])
            
                
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0]    
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            pos[:,9] = np.random.normal(self.z,0.001, nwalkers)
            
            self.pr_code = self.prior_create()
            self.fitted_model = QSO_models.OIII_QSO_BKPL
            self.log_prior_fce = logprior_general_scipy
            self.res = {'name': 'OIII_QSO_BKP'}
            
        else:
            raise Exception('self.model variable not understood. Available self.model keywords: outflow, gal, QSO_BKPL')
         
        nwalkers, ndim = pos.shape

        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        
        sampler = emcee.EnsembleSampler(
                 nwalkers, ndim, self.log_probability_general, args=())
        
        sampler.run_mcmc(pos, self.N, progress=self.progress)
        self.flat_samples = sampler.get_chain(discard=int(0.25*self.N), thin=15, flat=True)      
        
        self.chains = {'name': 'OIII'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
        
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        self.props = self.prop_calc()
        self.modeleval = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        self.chi2 = sum(((self.flux_fitloc-self.modeleval)/self.error_fitloc)**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))
        self.yeval = self.fitted_model(self.wave, *self.props['popt'])

        
        
    def fitting_Halpha_OIII(self, model, template=None):
        """ Lets fit both OIII and Halpha at the same time. 
        
        """
        self.template = template
        self.model = model
        
        if self.priors['z'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['z'][1]=='normal_hat') & (self.priors['z'][2]==0):
                self.priors['z'][2] = self.z
                self.priors['z'][3] = 200/3e5*(1+self.z)
                self.priors['z'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['z'][5] = self.z+1000/3e5*(1+self.z)
        try:
            if self.priors['zBLR'][2]==0:
                self.priors['zBLR'][2]=self.z
        except:
            lksdf=0
            
            
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.waves = self.wave.copy()
        self.wave = self.wave[np.invert(self.fluxs.mask)]
        
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        self.fit_loc = np.append(self.fit_loc, np.where((self.wave>(6300-50)*(1+self.z)/1e4)&(self.wave<(6300+50)*(1+self.z)/1e4))[0])
        self.fit_loc = np.append(self.fit_loc, np.where((self.wave>(6564.52-170)*(1+self.z)/1e4)&(self.wave<(6564.52+170)*(1+self.z)/1e4))[0])

    # =============================================================================
    #     Finding the initial conditions
    # =============================================================================
        sel=  np.where((self.wave<5025*(1+self.z)/1e4)& (self.wave>4980*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        try:
            peak_loc_OIII = np.argmax(self.flux_zoom)
            peak_OIII = (np.max(self.flux_zoom))
        except:
            peak_OIII = np.max(self.flux[self.fit_loc])
        
        sel=  np.where(((self.wave<(6564.52+20)*(1+self.z)/1e4))& (self.wave>(6564.52-20)*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        
        peak_loc = np.ma.argmax(self.flux_zoom)
        
        znew = self.wave_zoom[peak_loc]/0.656452-1
        peak_hal = np.ma.max(self.flux_zoom)
        if peak_hal<0:
            peak_hal==3e-3
        
    # =============================================================================
    #   Setting up fitting  
    # =============================================================================
        if self.model=='gal':
            #self.priors['z'] = [z, z-zcont, z+zcont]
            nwalkers=64
            self.fitted_model = HO_models.Halpha_OIII
            self.log_prior_fce = logprior_general
            self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak')
            self.pr_code = self.prior_create()
            
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)
            pos_l = np.array([self.z, cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, self.priors['Nar_fwhm'][0], peak_hal*0.15, peak_hal*0.2, peak_OIII*0.8,\
                              peak_hal*0.2])  
            
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                
            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf):
                logprior_general_test(pos_l, self.pr_code, self.labels)
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your self.priors\
                                boundries are sensible. {pos_l}')
               
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'Halpha_OIII'}
            
            
        elif self.model=='outflow':
            nwalkers=64
            self.fitted_model = HO_models.Halpha_OIII_outflow
            
            self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' )
            
            self.pr_code = self.prior_create()
            self.log_prior_fce = logprior_general
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)

            pos_l = np.array([self.z,cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, \
                              peak_OIII*0.8, peak_hal*0.2, peak_hal*0.2, peak_hal*0.2, \
                              self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0],
                              peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.05])
            
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                
            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf):
                
                logprior_general_test(pos_l, self.pr_code, self.labels)
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your self.priors\
                                boundries are sensible. {%pos_l}')
                                
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'Halpha_OIII_outflow'}
            
            
        elif self.model=='BLR':
            self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' ,\
                    'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak')
                
            nwalkers=64
            
            if self.priors['BLR_Hal_peak'][2]=='self.error':
                self.priors['BLR_Hal_peak'][2]=self.error[-2]; self.priors['BLR_Hal_peak'][3]=self.error[-2]*2
            if self.priors['BLR_Hbeta_peak'][2]=='self.error':
                self.priors['BLR_Hbeta_peak'][2]=self.error[2]; self.priors['BLR_Hbeta_peak'][3]=self.error[2]*2
            
            self.pr_code = self.prior_create()
            self.log_prior_fce = logprior_general
            self.fitted_model = HO_models.Halpha_OIII_BLR
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)
            pos_l = np.array([self.z,cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, \
                              peak_OIII*0.8, peak_hal*0.3 , peak_hal*0.2, peak_hal*0.2,\
                              self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0],
                              peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.1,\
                              self.priors['BLR_fwhm'][0], self.priors['zBLR'][0], peak_hal*0.3, peak_hal*0.1])
                
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                 
            
            
            if (self.log_prior_fce(pos_l, self.pr_code)==np.nan)|\
                (self.log_prior_fce(pos_l, self.pr_code)==-np.inf):
                print(logprior_general_test(pos_l, self.pr_code, self.labels))
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your self.priors\
                                boundries are sensible. {pos_l} ')
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            pos[:,-3] = np.random.normal(self.priors['zBLR'][0],0.00001, nwalkers)
           
            self.res = {'name': 'Halpha_OIII_BLR'}
            
        elif self.model=='BLR_simple':
            self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak')
                
            nwalkers=64
            
            if self.priors['BLR_Hal_peak'][2]=='self.error':
                self.priors['BLR_Hal_peak'][2]=self.error[-2]; self.priors['BLR_Hal_peak'][3]=self.error[-2]*2
            if self.priors['BLR_Hbeta_peak'][2]=='self.error':
                self.priors['BLR_Hbeta_peak'][2]=self.error[2]; self.priors['BLR_Hbeta_peak'][3]=self.error[2]*2
            
            self.pr_code = self.prior_create(self.labels, self.priors) 
            self.log_prior_fce = logprior_general
            self.fitted_model = HO_models.Halpha_OIII_BLR_simple
            
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)

            pos_l = np.array([self.z,cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, \
                              peak_OIII*0.8, peak_hal*0.3 , peak_hal*0.2, peak_hal*0.2,\
                              self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0], self.priors['zBLR'][0], peak_hal*0.3, peak_hal*0.1])
                
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            if (self.log_prior_fce(pos_l, self.pr_code)==np.nan)|\
                (self.log_prior_fce(pos_l, self.pr_code)==-np.inf):
                print(logprior_general_test(pos_l, self.pr_code, self.labels))
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your self.priors\
                                boundries are sensible. {pos_l} ')
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            pos[:,-3] = np.random.normal(self.priors['zBLR'][0],0.00001, nwalkers)
           
            self.res = {'name': 'Halpha_OIII_BLR_simple'}
            
            
        elif self.model=='QSO_BKPL':
            self.labels =[ 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'OIII_peak','Hbeta_peak', 'Nar_fwhm', \
                                  'Hal_out_peak', 'NII_out_peak','OIII_out_peak', 'Hbeta_out_peak',\
                                  'outflow_fwhm', 'outflow_vel',\
                                  'Hal_BLR_peak', 'Hbeta_BLR_peak',  'BLR_vel', 'BLR_alp1', 'BLR_alp2', 'BLR_sig']
        
        
            nwalkers=64
            self.pr_code = self.prior_create(self.labels, self.priors)   
            self.log_prior_fce = logprior_general
            self.fitted_model =  QSO_models.Halpha_OIII_QSO_BKPL
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)

            pos_l = np.array([self.z,cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, \
                              peak_OIII*0.8, peak_OIII*0.3,self.priors['Nar_fwhm'][0],\
                              peak_hal*0.2, peak_hal*0.3, peak_OIII*0.4, peak_OIII*0.2   ,\
                              self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0],\
                              peak_hal*0.4, peak_OIII*0.4, self.priors['BLR_vel'][0], 
                              self.priors['BLR_alp1'][0], self.priors['BLR_alp2'][0],self.priors['BLR_sig'][0]])
            
            if (self.log_prior_fce(pos_l, self.pr_code)==np.nan)|\
                (self.log_prior_fce(pos_l, self.pr_code)==-np.inf):
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your self.priors\
                                boundries are sensible: {pos_l}')
                                
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'Halpha_OIII_BLR'}
            
        else:
            raise Exception('self.model variable not understood. Available self.model keywords: outflow, gal, QSO_BKPL')
        
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]

        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general, args=()) 
        sampler.run_mcmc(pos, self.N, progress=self.progress)
        
        self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        
        self.chains = {'name': 'Halpha_OIII'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]

        self.props = self.prop_calc()
        
        self.chi2, self.BIC = sp.BIC_calc(self.waves, self.fluxs, self.error, self.fitted_model, self.props, 'Halpha_OIII')
        self.yeval = self.fitted_model(self.wave, *self.props['popt'])

        
    def fitting_general(self, fitted_model, labels, logprior=None, nwalkers=64, template=None):
        """ Fitting any general function that you pass. You need to put in fitted_model, labels and
        you can pass logprior function or number of walkers.        
        """
        self.template= template
        self.labels= labels
        if logprior !=None:
            self.log_prior_fce = logprior_general
        else: 
            self.log_prior_fce = logprior
        self.fitted_model = fitted_model
        
        if self.priors['z'][2]==0:
            self.priors['z'][2]=self.z
            
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.waves = self.wave[np.invert(self.fluxs.mask)]
        self.errors = self.error[np.invert(self.fluxs.mask)]

        self.flux_fitloc = self.flux.copy()
        self.wave_fitloc = self.waves.copy()
        self.error_fitloc = self.errors.copy()

        self.pr_code = self.prior_create()
       
        pos_l = np.zeros(len(self.labels))
            
        for i in enumerate(self.labels):
            pos_l[i[0]] = self.priors[i[1]][0] 
                
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            logprior_general_test(pos_l, self.pr_code, self.labels)
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
                
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        
        if self.ncpu==1:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general, args=()) 
            sampler.run_mcmc(pos, self.N, progress=self.progress)
        
        elif self.ncpu>1:
            from multiprocess import Pool
            with Pool(self.ncpu) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_general, args=(), pool=pool) 
            
                sampler.run_mcmc(pos, self.N, progress=self.progress)

        self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        self.chains = {'name': 'Custom model'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
        
        self.props = self.prop_calc()
        try:
            self.yeval = self.fitted_model(self.wave, *self.props['popt'])
        except:
            self.yeval = np.zeros_like(self.wave)

    def fitting_general_test(self, fitted_model, labels, logprior=None, nwalkers=64, template=None):
        """ Fitting any general function that you pass. You need to put in fitted_model, labels and
        you can pass logprior function or number of walkers.        
        """
        self.template= template
        self.labels= labels
        if logprior !=None:
            self.log_prior_fce = logprior_general_scipy
        else: 
            self.log_prior_fce = logprior
        self.fitted_model = fitted_model
        
        if self.priors['z'][2]==0:
            self.priors['z'][2]=self.z
            
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.waves = self.wave[np.invert(self.fluxs.mask)]
        self.errors = self.error[np.invert(self.fluxs.mask)]

        self.flux_fitloc = self.flux.copy()
        self.wave_fitloc = self.waves.copy()
        self.error_fitloc = self.errors.copy()

        self.pr_code = self.prior_create()
       
        pos_l = np.zeros(len(self.labels))
            
        for i in enumerate(self.labels):
            pos_l[i[0]] = self.priors[i[1]][0] 
                
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            logprior_general_test(pos_l, self.pr_code, self.labels)
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
                
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
        
        nwalkers, ndim = pos.shape
        
        if self.ncpu==1:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general, args=()) 
            sampler.run_mcmc(pos, self.N, progress=self.progress)
        
        elif self.ncpu>1:
            from multiprocess import Pool
            with Pool(self.ncpu) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_general, args=(), pool=pool) 
            
                sampler.run_mcmc(pos, self.N, progress=self.progress)

        self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        self.chains = {'name': 'Custom model'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
        
        self.props = self.prop_calc()
        try:
            self.yeval = self.fitted_model(self.wave, *self.props['popt'])
        except:
            self.yeval = np.zeros_like(self.wave)


    def fitting_custom(self, model_inputs,model_name, nwalkers=64, template=None):
        """ Fitting any custom model that you pass with a dictionary. You need to put in fitted_model, labels and
        you can pass logprior function or number of walkers.        
        """
        self.template= template
        self.model_inputs = model_inputs
        self.model_name = model_name 
        
        
        if self.priors['z'][2]==0:
            self.priors['z'][2]=self.z
            
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.waves = self.wave[np.invert(self.fluxs.mask)]
        self.errors = self.error[np.invert(self.fluxs.mask)]

        self.flux_fitloc = self.flux.copy()
        self.wave_fitloc = self.waves.copy()
        self.error_fitloc = self.errors.copy()
        
        self.Model = Custom_model.Model(self.model_name, model_inputs)
        self.Model.fit_to_data(self.wave_fitloc, self.flux_fitloc, self.error_fitloc, N=self.N, nwalkers=nwalkers, ncpu=1)

        self.labels= self.Model.labels
        self.chains = self.Model.chains
        self.props = self.Model.props
        self.yeval = self.Model.calculate_values(self.waves)
        self.comps = self.Model.lines


    def save(self, file_path):
        import pickle
        """save class as self.name.txt"""
        with open(file_path, "wb") as file:
            file.write(pickle.dumps(self.__dict__))
        
    def load(self, file_path):
        """try load self.name.txt"""
        import pickle
        with open(file_path, "rb") as file:
            dataPickle = file.read()
            self.__dict__ = pickle.loads(dataPickle)
    
    def log_probability_general(self, theta):
        """ Basic log probability function used in the emcee. Theta are the variables supplied by the emcee 
        """

        lp = self.log_prior_fce(theta,self.pr_code)
        
        try:
            if not np.isfinite(lp):
                return -np.inf
        except:
            lp[np.isnan(lp)] = -np.inf

        try:
            if self.template:
                evalm = self.fitted_model(self.wave_fitloc,*theta, self.template)
            else:
                evalm = self.fitted_model(self.wave_fitloc,*theta)
        except:
            evalm = self.fitted_model(self.wave_fitloc,theta)


        sigma2 = self.error_fitloc**2
        log_likelihood = -0.5 * np.nansum((self.flux_fitloc - evalm) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))
        
        return lp + log_likelihood
    
    def log_probability_custom(self, theta):
        """ Basic log probability function used in the emcee. Theta are the variables supplied by the emcee 
        """

        lp = self.log_prior_fce(theta,self.pr_code)
        
        try:
            if not np.isfinite(lp):
                return -np.inf
        except:
            lp[np.isnan(lp)] = -np.inf

        
        theta_dict = {}
        for i,name in enumerate(self.labels):
            theta_dict[name] = theta[i]

        evalm = self.fitted_model(self.wave_fitloc,theta_dict)


        sigma2 = self.error_fitloc**2
        log_likelihood = -0.5 * np.nansum((self.flux_fitloc - evalm) ** 2 / sigma2) #+ np.log(2*np.pi*sigma2))
        
        return lp + log_likelihood
    
    def prior_create(self):
        """ Function that takes the prior dictionary and create a priote code that the prior function as using.         
        """
        pr_code = np.zeros((len(self.labels),5))
        
        for key in enumerate(self.labels):
            match self.priors[key[1]][1]:
                case 'normal':
                    pr_code[key[0]][0] = 0
                case 'lognormal':
                    pr_code[key[0]][0] = 2
                
                case 'uniform':
                    pr_code[key[0]][0] = 1
                
                case 'loguniform':
                    pr_code[key[0]][0] = 3
            
                case 'normal_hat':
                    pr_code[key[0]][0] = 4
                    pr_code[key[0]][3] = self.priors[key[1]][4]
                    pr_code[key[0]][4] = self.priors[key[1]][5]
            
                case 'lognormal_hat':
                    pr_code[key[0]][0] = 5
                    pr_code[key[0]][3] = self.priors[key[1]][4]
                    pr_code[key[0]][4] = self.priors[key[1]][5]
                
                case _:
                    raise Exception('Sorry mode in prior type not understood: ', key )
            
            pr_code[key[0]][1] = self.priors[key[1]][2]
            pr_code[key[0]][2] = self.priors[key[1]][3]
        return pr_code
    
    def prop_calc(self): 
        """ Take the dictionary with the results chains and calculates the values 
        and 1 sigma confidence interval
        
        """
        labels = list(self.chains.keys())[1:]
        res_plt = []
        res_dict = {'name': self.chains['name']}
        for lbl in labels:
            
            array = self.chains[lbl]
            
            p50,p16,p84 = np.percentile(array, (50,16,84))
            p16 = p50-p16
            p84 = p84-p50
            
            res_plt.append(p50)
            res_dict[lbl] = np.array([p50,p16,p84])
            
        res_dict['popt'] = res_plt
        return res_dict
    
    def corner(self,):
        import corner

        fig = corner.corner(
            sp.unwrap_chain(self.chains), 
            labels = self.labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
    
        
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import truncnorm
from scipy.stats import lognorm
from scipy.stats import loguniform

#import numba
#@numba.njit
def logprior_general_scipy_test(theta, priors):
    results = []
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results.append(norm.logpdf(t, p[1], p[2]))
        elif p[0]==1:
            results.append(uniform.logpdf(t, p[1], p[2]-p[1]))
        elif p[0]==2:
            results.append(  norm.logpdf(np.log10(t), p[1], p[2]))
        elif p[0]==3:
            results.append( uniform.logpdf(np.log10(t), p[1], p[2]-p[1]))
        elif p[0]==4:
            results.append(truncnorm.logpdf(t, a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2]))
        elif p[0]==4:
            results.append( truncnorm.logpdf(np.log10(t),a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2]))
    return results

def logprior_general_scipy(theta, priors):
    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results+= norm.logpdf(t, p[1], p[2])
        elif p[0]==1:
            results+= uniform.logpdf(t, p[1], p[2]-p[1])
        elif p[0]==2:
            results+=  norm.logpdf(np.log10(t), p[1], p[2])
        elif p[0]==3:
            results+= uniform.logpdf(np.log10(t), p[1], p[2]-p[1])
        elif p[0]==4:
            results += truncnorm.logpdf(t, a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2])
        elif p[0]==4:
            results += truncnorm.logpdf(np.log10(t),a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2])
    return results


import numba
@numba.njit
def logprior_general(theta, priors):
    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==4:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf
    
    return results



def logprior_general_test(theta, priors, labels):
    for t,p,lb in zip( theta, priors, labels):
        print(p)
        if p[0] ==0:
            results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0]==1:
            results= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results = -np.inf
        elif p[0]==4:
            if p[3]<np.log10(t)<p[4]:
                results= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results = -np.inf
    
        print(lb, t, results)

def Fitting_OIII_unwrap(lst):
    
    i,j,flx_spax_m, error, wave, z = lst
    
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    
    Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, prior_update=priors)
    Fits_sig.fitting_OIII(model='gal_simple')
    Fits_sig.fitted_model = 0
    
    cube_res  = [i,j, Fits_sig.props, Fits_sig.chains,wave,flx_spax_m,error]
                 
    return cube_res

def Fitting_Halpha_OIII_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    
    try:
        Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, prior_update=priors)
        Fits_sig.fitting_Halpha_OIII(model='gal' )
        Fits_sig.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig]
    except Exception as _exc_:
        print(_exc_)
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
    return cube_res

def Fitting_Halpha_OIII_AGN_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, prior_update=priors)
        Fits_sig.fitting_Halpha_OIII(model='BLR' )
        Fits_sig.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig]
        
    except Exception as _exc_:
        print(_exc_)
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
    return cube_res

def Fitting_Halpha_OIII_outflowboth_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
     
    
    i,j,flx_spax_m, error, wave, z = lst
    deltav = 1500
    deltaz = deltav/3e5*(1+z)
    try:
        Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, prior_update=priors)
        Fits_sig.fitting_Halpha_OIII(model='gal' )
        Fits_sig.fitted_model = 0
        
        Fits_out = Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, prior_update=priors)
        Fits_out.fitting_Halpha_OIII(model='outflow' )
        Fits_out.fitted_model = 0
        
        cube_res  = [i,j,Fits_sig, Fits_out ]
    except Exception as _exc_:
        print(_exc_)
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        print('Failed fit')
    return cube_res

def Fitting_OIII_2G_unwrap(lst):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        priors= pickle.load(fp) 
        
    i,j,flx_spax_m, error, wave, z = lst
    
    try:
        Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, prior_update=priors)
        Fits_sig.fitting_OIII(model='gal_simple')
        Fits_sig.fitted_model = 0
        
        Fits_out = Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, prior_update=priors)
        Fits_out.fitting_OIII(model='outflow_simple')
        Fits_out.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig, Fits_out ]
    except Exception as _exc_:
        print(_exc_)
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
    
    Fits_sig = Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, prior_update=priors)
    Fits_sig.fitting_Halpha(model='gal')
    Fits_sig.fitted_model = 0
    
    cube_res  = [i,j,Fits_sig]
    
    return cube_res
    
def Fitting_general_unwrap(lst, progress=False):
    with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
        data= pickle.load(fp) 

    i,j,flx_spax_m, error, wave, z = lst
    use = data['use'] 
    

    if len(use)==0:
        use = np.linspace(0, len(wave)-1, len(wave), dtype=int)

    try:
        Fits_sig = Fitting(wave[use], flx_spax_m[use], error[use], z,N=data['N'],progress=progress, prior_update=data['priors'])
        Fits_sig.fitting_general(data['fitted_model'], data['labels'], data['logprior'], nwalkers=data['nwalkers'])
        Fits_sig.fitted_model = 0
      
            
        cube_res  = [i,j,Fits_sig ]
    except Exception as _exc_:
        print(_exc_)
        cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        print('Failed fit')
    '''
    f,ax = plt.subplots(1)
    
    from .Plotting_tools_v2 import plotting_OIII as ploto
    ploto(wave, flx_spax_m,ax, sp.prop_calc(flat_samples),fitted_model)
    '''
    return cube_res


