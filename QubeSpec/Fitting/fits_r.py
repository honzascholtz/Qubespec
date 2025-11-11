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

from typing import Dict, List, Tuple, Optional, Union


pi= np.pi
e= np.e

c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23


def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5/2.35*mu
    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

import warnings
warnings.filterwarnings("ignore")

from ..Models import OIII_models as O_models
from ..Models import Halpha_OIII_models as HO_models
from ..Models import QSO_models as QSO_models
from ..Models import Halpha_models as H_models
from ..Models import Full_optical as FO_models
from ..Models import Custom_model
import numba
from .. import Utils as sp

from .priors import * 

class Fitting:
    """ Simple class containing everything that we need to fit a spectrum and also all of its results. 


    Parameters
    ----------
 
    wave : array
      observed wavelength in microns

    flux : array
        flux of the spectrum

    error : array
        error on the spectrum
    z : float
        redshift of the source

    N: int - optional 
        number of points in the chain - default 5000

    ncpu: int - optional
        number of cpus used to fit - I find that the overheads can be bigger what using multipleprocessing then the speed up. Experiment or keep to 1

    progress : bool - optional
        progress bar for the emcee bit

    priors: dict - optional
        dictionary with all of the priors to update
        
    """
       
    def __init__(self, wave='', flux='', error='', z='', N=5000,ncpu=1, progress=True,sampler='emcee', priors= {'z':[0, 'normal_hat', 0,0.003,0,0]}):
        priors_update = priors.copy()
        priors= {'z':[0, 'normal_hat', 0,0.003,0.01,0.01],\
                'cont':[0,'loguniform',-4,1],\
                'cont_grad':[0,'normal',0,0.3], \
                'Hal_peak':[0,'loguniform',-3,1],\
                'BLR_Hal_peak':[0,'loguniform',-3,1],\
                'NII_peak':[0,'loguniform',-3,1],\
                'Nar_fwhm':[300,'uniform',100,900],\
                'BLR_fwhm':[4000,'uniform', 2000,9000],\
                'zBLR':[0, 'normal', 0,0.003],\
                'Hal_out_peak':[0,'loguniform',-3,1],\
                'NII_out_peak':[0,'loguniform',-3,1],\
                'outflow_fwhm':[600,'uniform', 300,1500],\
                'outflow_vel':[-50,'normal', 0,300],\
                'OIII_peak':[0,'loguniform',-3,1],\
                'OIII_out_peak':[0,'loguniform',-3,1],\
                'Hbeta_peak':[0,'loguniform',-3,1],\
                'Hbeta_out_peak':[0,'loguniform',-3,1],\
                'Fe_peak':[0,'loguniform',-3,1],\
                'Fe_fwhm':[3000,'uniform',2000,6000],\
                'SIIr_peak':[0,'loguniform', -3,1],\
                'SIIb_peak':[0,'loguniform', -3,1],\
                'BLR_Hbeta_peak':[0,'loguniform', -3,1]}
        
        for name in list(priors_update.keys()):
            priors[name] = priors_update[name]

        self.N = N # Number of points in the chains
        self.priors = priors # storing priors
        self.progress = progress # progress bar?
        self.z = z  # redshift
        self.waves = wave # wavelength 
        self.wave = wave # wavelength 
        self.fluxs = flux # flux density
        self.error = error # errors
        self.ncpu= ncpu # number of cpus to use in the fit 
        self.sampler = sampler
    
    # =============================================================================
    #  Primary function to fit Halpha both with or without BLR - data prep and fit 
    # =============================================================================
    def fitting_optical(self, model='gal'):
        """ Method to fit Halpha+[NII +[SII]]
        
        Parameters
        ----------

        model - str
            current valid models names and their variable names/also prior names:

            gal -  'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak'

            outflow - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel'
           
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
        
        self.fluxs[np.isnan(self.fluxs)] = 0
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.waves = self.waves[np.invert(self.fluxs.mask)]

        self.fit_loc = np.where((self.waves>(6564.52-170)*(1+self.z)/1e4)&(self.waves<(6564.52+170)*(1+self.z)/1e4))[0]
        sel=  np.where(((self.waves<(6564.52+20)*(1+self.z)/1e4))& (self.waves>(6564.52-20)*(1+self.z)/1e4))[0]

        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.waves[sel]

        peak = np.ma.max(self.flux_zoom)
        nwalkers=64

        if self.model=='gal':
            self.fitted_model = FO_models.Full_optical
            self.log_prior_fce = logprior_general
            self.labels= ['z', 'cont','cont_grad',  'Hal_peak', 'NII_peak', 'OIII_peak', 'Hbeta_peak','Hgamma_peak', 'Hdelta_peak','NeIII_peak','OII_peak','OII_rat','OIIIaur_peak', 'HeI_peak','HeII_peak', 'Nar_fwhm']

            self.pr_code = self.prior_create()
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.01, peak/2, peak/4,self.priors['Nar_fwhm'][0],peak/6, peak/6 ])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 

            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            self.res = {'name': 'Full_optical'}
            
                
        elif self.model=='outflow':
            self.fitted_model = FO_models.Full_optical_outflow
            self.log_prior_fce = logprior_general
            self.labels= ['z', 'cont','cont_grad',  'Hal_peak', 'NII_peak', 'OIII_peak', 'Hbeta_peak','Hgamma_peak',\
                          'Hdelta_peak','NeIII_peak','OII_peak','OII_rat','OIIIaur_peak', 'HeI_peak','HeII_peak', 'Nar_fwhm',\
                          'Hal_out_peak', 'OIII_out_peak', 'NII_out_peak', 'Hbeta_out_peak', \
                                  'outflow_vel', 'outflow_fwhm']
            
            self.pr_code = self.prior_create()
            
            cont = np.median(self.flux[self.fit_loc])
            if cont<0:
                cont=0.01
            pos_l = np.array([self.z,cont,0.01, peak/2, peak/4, self.priors['Nar_fwhm'][0],peak/6, peak/6,peak/8, peak/8, self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0] ])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'Full_optical_outflow'}
        
        self.flux_fitloc = self.flux
        self.wave_fitloc = self.wave
        self.error_fitloc = self.error
        
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            print(logprior_general_scipy_test(pos_l, self.pr_code))
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
        self.pos = pos
        self.pos_l = pos_l
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
             nwalkers, ndim, self.log_probability_general, args=())
     
        sampler.run_mcmc(pos, self.N, progress=self.progress)
        self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)      
        
        self.chains = {'name': 'Full_optical'}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
            
        self.props = self.prop_calc()
        try:
            self.chi2, self.BIC = sp.BIC_calc(self.wave, self.fluxs, self.error, self.fitted_model, self.props, 'Halpha')
        except:
            self.chi2, self.BIC = np.nan, np.nan
        
        self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])

    def fitting_Halpha(self, model='gal'):
        """ Method to fit Halpha+[NII +[SII]]
        
        Parameters
        ----------

        model - str
            current valid models names and their variable names/also prior names:

            gal -  'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak'

            outflow - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel'
            
            BLR_simple - 'z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak'

            BLR - 'z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak'

            QSO_BKPL - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm',
                    'Hal_out_peak', 'NII_out_peak', \
                    'outflow_fwhm', 'outflow_vel', \
                    'BLR_Hal_peak', 'zBLR', 'BLR_alp1', 'BLR_alp2', 'BLR_sig'
        
        
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
        self.wave = self.waves[np.invert(self.fluxs.mask)]
         
        self.fit_loc = np.where((self.wave>(6564.52-170)*(1+self.z)/1e4)&(self.wave<(6564.52+200)*(1+self.z)/1e4))[0]
        sel=  np.where(((self.wave<(6564.52+20)*(1+self.z)/1e4))& (self.wave>(6564.52-20)*(1+self.z)/1e4))[0]
        
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)
        
        peak = abs(np.ma.max(self.flux_zoom))
        nwalkers=32
        
        cont = np.median(self.flux[self.fit_loc])
        if cont<0:
            cont=0.01

        if self.model=='BLR_simple':
            self.labels=['z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak']
            
            self.fitted_model = H_models.Halpha_wBLR
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
            
            pos_l = np.array([self.z,cont,0.001, peak/2, peak/4, peak/4, self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],self.priors['zBLR'][0],peak/6, peak/6])
            self.pos_l = pos_l
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
                
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            for i,lb in enumerate(self.labels):
                if lb =='zBLR':
                    pos[:,i] = np.random.normal(self.z,0.001, nwalkers)

            self.pos =pos
            
            self.res = {'name': 'Halpha_wth_BLR'}
        
        elif self.model=='BLR':
            self.labels=['z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak',\
                         'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel']
            
            self.fitted_model = H_models.Halpha_BLR_outflow
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
            
            pos_l = np.array([self.z,cont,0.001, peak/2, peak/4, peak/4, self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],self.priors['zBLR'][0],peak/6, peak/6,\
                              peak/6, peak/6, 700, -100])
            self.pos_l = pos_l
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
            
            pos_l = np.array([self.z,cont,0.01, peak/1.3, peak/10,self.priors['Nar_fwhm'][0],peak/6, peak/6 ])
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
            raise Exception('self.model variable not understood. Available model keywords: BLR, BLR_simple, outflow, gal, QSO_BKPL')
        
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            print(logprior_general_test(pos_l, self.pr_code,self.labels))
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')

        self.init_value = pos_l
        if self.sampler =='emcee':
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general, args=())
        
            sampler.run_mcmc(pos, self.N, progress=self.progress)
            self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)      
            
            self.chains = {'name': 'Halpha'}
            for i in range(len(self.labels)):
                self.chains[self.labels[i]] = self.flat_samples[:,i]
                
            self.props = self.prop_calc()
            self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)

        elif self.sampler=='leastsq':
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(self.fitted_model, self.wave_fitloc, self.flux_fitloc, p0= pos_l, sigma=self.error_fitloc, bounds = self.bounds_est())
            errs = np.sqrt(np.diag(pcov))

            self.props = {'name': 'Halpha'}
            self.chains = {'name': 'Halpha'}
            self.props['popt'] = popt
            for i, name in enumerate(self.labels):
                self.props[name] = [popt[i], errs[i], errs[i]]
                self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
        
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')

        self.chi2 = np.nansum(((self.flux_fitloc-self.fitted_model(self.wave_fitloc, *self.props['popt']))**2)/self.error_fitloc**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))
        
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        
    # =============================================================================
    # Primary function to fit [OIII] with and without outflows. 
    # =============================================================================
    
    def fitting_OIII(self, model, Fe_template=0, plot=0, expand_prism=0):
        """ Method to fit [OIII] + Hbeta
        
        Parameters
        ----------

        model - str
            current valid models names and their variable names/also prior names:

            gal - 'z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak' - Hbeta and [OIII] kinematics are linked together

            outflow - 'z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak' - Hbeta and [OIII] kinematics are linked together
        
        template - str
            name of the FeII template you want to fit - Tsuzuki, BG92, Veron

        """
        
        self.model = model
        self.template = Fe_template
        if self.priors['z'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['z'][1]=='normal_hat') & (self.priors['z'][2]==0):
                self.priors['z'][2] = self.z
                self.priors['z'][3] = 200/3e5*(1+self.z)
                self.priors['z'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['z'][5] = self.z+1000/3e5*(1+self.z)
        
        if self.priors['zBLR'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['zBLR'][1]=='normal_hat') & (self.priors['zBLR'][2]==0):
                self.priors['zBLR'][2] = self.z
                self.priors['zBLR'][3] = 200/3e5*(1+self.z)
                self.priors['zBLR'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['zBLR'][5] = self.z+1000/3e5*(1+self.z)

        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.wave = self.waves[np.invert(self.fluxs.mask)]
        
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        if (expand_prism==1) | (expand_prism==True):
            self.fit_loc = np.where((self.wave>4600*(1+self.z)/1e4)&(self.wave<5600*(1+self.z)/1e4))[0]

        sel=  np.where((self.wave<5025*(1+self.z)/1e4)& (self.wave>4980*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        
        try:
            peak_loc = np.argmax(self.flux_zoom)
            peak = abs((np.max(self.flux_zoom)))
        except:
            peak_loc = np.argmax(self.flux[self.fit_loc])
            peak = abs((np.max(self.flux[self.fit_loc])))

        selb =  np.where((self.wave<4880*(1+self.z)/1e4)& (self.wave>4820*(1+self.z)/1e4))[0]
        self.flux_zoomb = self.flux[selb]
        self.wave_zoomb = self.wave[selb]
        try:
            peak_loc_beta = np.argmax(self.flux_zoomb)
            peak_beta = abs((np.max(self.flux_zoomb)))
        except:
            peak_beta = abs(peak/3)

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)

        nwalkers=64

        if  (self.model=='outflow_simple') | (self.model=='outflow'):
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak']
            self.fitted_model = O_models.OIII_outflow
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
            
            cont_init = abs(np.median(self.flux[self.fit_loc]))
            pos_l = np.array([self.z,cont_init,0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_test(pos_l, self.pr_code,self.labels))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
            
            self.res = {'name': 'OIII_outflow_simple'}
        
        elif (self.model== 'gal_simple') | (self.model== 'gal'):
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak']
            self.fitted_model = O_models.OIII_gal
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
            cont_init = abs(np.median(self.flux[self.fit_loc]))
            pos_l = np.array([self.z,cont_init,0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta])
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            self.res = {'name': 'OIII_simple'}

            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_test(pos_l, self.pr_code,self.labels))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
            
        elif (self.model=='BLR_simple'):
            
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm']
            
            if Fe_template != 0:
                self.labels.append('Fe_peak')
                self.labels.append('Fe_fwhm')
                self.fitted_model = O_models.OIII_fe_models(self.template)
                self.fitted_model = self.fitted_model.OIII_gal_BLR_Fe

                self.res = {'name': 'OIII_BLR_simple_fe'}
                pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta, self.z, peak_beta/2, self.priors['BLR_fwhm'][0],\
                                np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]])

            else:
                pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta, self.z, peak_beta/2, self.priors['BLR_fwhm'][0]])
                self.fitted_model = O_models.OIII_gal_BLR
                self.res = {'name': 'OIII_BLR_simple'}
                
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
                    
            
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)

            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_test(pos_l, self.pr_code,self.labels))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
            
            self.res = {'name': 'OIII_BLR_simple'}
        
        elif (self.model== 'BLR_outflow'):

            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak','zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm']
            
            if Fe_template != 0:
                self.labels.append('Fe_peak')
                self.labels.append('Fe_fwhm')
                self.fitted_model = O_models.OIII_fe_models(self.template)
                self.fitted_model = self.fitted_model.OIII_outflow_BLR_Fe

                pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3,\
                                  self.z, peak_beta/2, self.priors['BLR_fwhm'][0],\
                                  np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]])
                self.res = {'name': 'BLR_outflow_Fe'}
                
                
            else:
                self.fitted_model = O_models.OIII_outflow_BLR
                pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3,\
                                 self.z, peak_beta/2, self.priors['BLR_fwhm'][0]])
                self.res = {'name': 'BLR_outflow'}
                
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()
                        
                
            for i in enumerate(self.labels):
                pos_l[i[0]] = pos_l[i[0]] if self.priors[i[1]][0]==0 else self.priors[i[1]][0] 
            pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
            pos[:,0] = np.random.normal(self.z,0.001, nwalkers)
            
            if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
                print(logprior_general_test(pos_l, self.pr_code,self.labels))
                
                raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
                  
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
            raise Exception('self.model variable not understood. Available self.model keywords: outflow_simple, BLR_simple, BLR_outflow,  gal_simple, QSO_BKPL')
         
        nwalkers, ndim = pos.shape

        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        if self.sampler =='emcee':
            
            sampler = emcee.EnsembleSampler(
                        nwalkers, ndim, self.log_probability_general, args=())
            
            sampler.run_mcmc(pos, self.N, progress=self.progress)
            self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)      
            
            self.chains = {'name': 'OIII'}
            for i in range(len(self.labels)):
                self.chains[self.labels[i]] = self.flat_samples[:,i]
            
            self.like_chains = sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)
            self.props = self.prop_calc()
        elif self.sampler=='leastsq':
            from scipy.optimize import curve_fit

            use = pos_l < self.bounds_est()[0]
            if True in use:
                raise ValueError(f'Initial guess is outside of the lower bounds {print(self.labels[use])}')
            use = pos_l > self.bounds_est()[1]
            if True in use:
                raise ValueError(f'Initial guess is outside of the higher bounds in {print(self.labels[use])}')
            
            popt, pcov = curve_fit(self.fitted_model, self.wave_fitloc, self.flux_fitloc, p0= pos_l, sigma=self.error_fitloc, bounds = self.bounds_est())
            errs = np.sqrt(np.diag(pcov))

            self.props = {'name': 'OIII'}
            self.chains = {'name': 'OIII'}
            self.props['popt'] = popt
            for i, name in enumerate(self.labels):
                self.props[name] = [popt[i], errs[i], errs[i]]
                self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
        
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')
        
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])

        self.chi2 = np.nansum(((self.flux_fitloc-self.yeval[self.fit_loc])/self.error_fitloc)**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))
        
    def fitting_Halpha_OIII(self, model, template=0):
        """ Method to fit Halpha + [OIII] + Hbeta+ [NII] + [SII]
        
        Parameters
        ----------

        model - str
            current valid models names and their variable names/also prior names:

            gal - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak'

            outflow - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak','Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' 
            
            BLR - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' ,\
                    'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak'

            BLR_simple - 'z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak'
        
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
        self.wave = self.waves[np.invert(self.fluxs.mask)]
        
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        #self.fit_loc = np.append(self.fit_loc, np.where((self.wave>(6300-50)*(1+self.z)/1e4)&(self.wave<(6300+50)*(1+self.z)/1e4))[0])
        self.fit_loc = np.append(self.fit_loc, np.where((self.wave>(6564.52-170)*(1+self.z)/1e4)&(self.wave<(6564.52+200)*(1+self.z)/1e4))[0])

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)
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
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak']
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
            if self.template==0:
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
            
            else:
                self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                        'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'BLR_Hal_peak', 'BLR_Hbeta_peak', 'Fe_peak', 'Fe_FWHM')
                    
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
                                self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0], self.priors['zBLR'][0], peak_hal*0.3, peak_hal*0.1, cont_init, self.priors['Fe_FWHM'][0]])
                    
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

        if self.sampler =='emcee':
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.log_probability_general, args=())
        
            sampler.run_mcmc(pos, self.N, progress=self.progress)
            self.flat_samples = sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)      
            
            self.chains = {'name': 'Halpha_OIII'}
            for i in range(len(self.labels)):
                self.chains[self.labels[i]] = self.flat_samples[:,i]
                
            self.props = self.prop_calc()
        elif self.sampler=='leastsq':
            from scipy.optimize import curve_fit
            popt, pcov = curve_fit(self.fitted_model, self.wave_fitloc, self.flux_fitloc, p0= pos_l, sigma=self.error_fitloc, bounds = self.bounds_est())
            errs = np.sqrt(np.diag(pcov))

            self.props = {'name': 'Halpha_OIII'}
            self.chains = {'name': 'Halpha_OIII'}
            self.props['popt'] = popt
            for i, name in enumerate(self.labels):
                self.props[name] = [popt[i], errs[i], errs[i]]
                self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
        
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')


        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        
        self.chi2 = np.nansum(((self.flux_fitloc-self.yeval[self.fit_loc])/self.error_fitloc)**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))

        
    def fitting_general(self, fitted_model, labels, logprior=None, nwalkers=64, skip_check=False, zscale=0.001, odd=False):
        """ Fitting any general function that you pass. You need to put in fitted_model, labels and
        you can pass logprior function or number of walkers.  

        Parameters
        ----------

        fitted_model : callable
            Function to fit

        labels : list
            list of the name of the paramters in the same order as in the fitted_function

        priors: dict - optional
            dictionary with all of the priors to update
        
        logprior: callable function
            logprior evaluation function - use emfit.logprior_general or emfit.logprior_general_scipy
        
        nwalkers : int - optional
            default 64 walkers for the MCMC
                 
        """
        self.template= None
        self.labels= labels
        if logprior !=None:
            self.log_prior_fce = logprior_general
        else: 
            self.log_prior_fce = logprior
        self.fitted_model = fitted_model
        
        if self.priors['z'][2]==0:
            self.priors['z'][2]=self.z

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)
            
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.wave = self.waves[np.invert(self.fluxs.mask)]
        self.errors = self.error[np.invert(self.fluxs.mask)]

        self.flux_fitloc = self.flux.copy()
        self.wave_fitloc = self.waves.copy()
        self.error_fitloc = self.errors.copy()

        if odd==True:
            if len(self.flux_fitloc) % 2 == 0:
                self.flux_fitloc = self.flux_fitloc[:-1]
                self.wave_fitloc = self.wave_fitloc[:-1]
                self.error_fitloc = self.error_fitloc[:-1]

        self.pr_code = self.prior_create()
       
        pos_l = np.zeros(len(self.labels))
            
        for i, name in enumerate(self.labels):
            pos_l[i] = self.priors[name][0] 

            if ('_peak' in name) & (self.priors[name][0] ==0):
                pos_l[i] = np.nanmean(self.error_fitloc)*np.random.uniform(5,10)
            
            if ('cont' == name) & (self.priors[name][0] ==0):
                pos_l[i] = np.nanmedian(self.flux_fitloc)*5
            
        self.pos_l = pos_l
       
        if (self.log_prior_fce(pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(pos_l, self.pr_code)== np.nan):
            logprior_general_test(pos_l, self.pr_code, self.labels)
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
                
        pos = np.random.normal(pos_l, abs(pos_l*0.1), (nwalkers, len(pos_l)))
        pos[:,0] = np.random.normal(self.z,zscale, nwalkers)

        self.pos = pos

        if self.sampler =='emcee':
            nwalkers, ndim = pos.shape
        
            if self.ncpu==1:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_general, args=()) 
                sampler.run_mcmc(pos, self.N, progress=self.progress,skip_initial_state_check=skip_check)
            
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
        elif self.sampler=='leastsq':
            from scipy.optimize import curve_fit
            use = ~(np.isnan(self.flux_fitloc) | np.isinf(self.flux_fitloc))
            self.error_fitloc[np.isnan(self.error_fitloc)] = 1e6
            
            #print(self.bounds_est())
            #print ( np.sum( (self.wave_fitloc[use]-self.fitted_model(self.wave_fitloc[use],*pos_l ))/self.error_fitloc[use]))
            #print ( np.sum( (self.wave_fitloc[use]-self.fitted_model(self.wave_fitloc[use],*self.bounds_est()[0] ))/self.error_fitloc[use]))
            #print ( np.sum( (self.wave_fitloc[use]-self.fitted_model(self.wave_fitloc[use],*self.bounds_est()[1] ))/self.error_fitloc[use]))

            popt, pcov = curve_fit(self.fitted_model, self.wave_fitloc[use], self.flux_fitloc[use], p0= pos_l, sigma=self.error_fitloc[use], bounds = self.bounds_est())
            errs = np.sqrt(np.diag(pcov))

            self.props = {'name': 'Custom model'}
            self.chains = {'name': 'Custom model'}
            self.props['popt'] = popt
            for i, name in enumerate(self.labels):
                self.props[name] = [popt[i], errs[i], errs[i]]
                self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
        
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')

        try:
            self.yeval = self.fitted_model(self.wave, *self.props['popt'])
        except:
            self.yeval = np.zeros_like(self.wave)

        try:
            self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        except:
            self.yeval_fitloc = np.zeros_like(self.wave_fitloc)
        
        self.chi2 = np.nansum(((self.flux_fitloc-self.yeval_fitloc)/self.error_fitloc)**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))


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

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)
            
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

        #evalm = self.fitted_model(self.wave_fitloc,*theta)
        try:
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
            if self.priors[key[1]][1]== 'normal':
                pr_code[key[0]][0] = 0
            
            elif self.priors[key[1]][1]== 'lognormal':
                pr_code[key[0]][0] = 2
            
            elif self.priors[key[1]][1]== 'uniform':
                pr_code[key[0]][0] = 1
            
            elif self.priors[key[1]][1]== 'loguniform':
                pr_code[key[0]][0] = 3
            
            elif self.priors[key[1]][1]== 'normal_hat':
                pr_code[key[0]][0] = 4
                pr_code[key[0]][3] = self.priors[key[1]][4]
                pr_code[key[0]][4] = self.priors[key[1]][5]

            elif self.priors[key[1]][1]== 'lognormal_hat':
                pr_code[key[0]][0] = 5
                pr_code[key[0]][3] = self.priors[key[1]][4]
                pr_code[key[0]][4] = self.priors[key[1]][5]
                
            else:
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
        return fig
    
    def bounds_est(self):
        up = np.array([])
        do = np.array([])
  
        for key in enumerate(self.labels):
            if self.priors[key[1]][1]== 'normal':
                up = np.append(up, self.priors[key[1]][2]+4*self.priors[key[1]][3])
                do = np.append(do, self.priors[key[1]][2]-4*self.priors[key[1]][3])
            
            elif self.priors[key[1]][1]== 'lognormal':
                up = np.append(up, 10**(self.priors[key[1]][2]+4*self.priors[key[1]][3]))
                do = np.append(do, 10**(self.priors[key[1]][2]-4*self.priors[key[1]][3]))
            
            elif self.priors[key[1]][1]== 'uniform':
                up = np.append(up, self.priors[key[1]][3])
                do = np.append(do, self.priors[key[1]][2])
            
            elif self.priors[key[1]][1]== 'loguniform':
                up = np.append(up, 10**self.priors[key[1]][3])
                do = np.append(do, 10**self.priors[key[1]][2])
            
            elif self.priors[key[1]][1]== 'normal_hat':
                up = np.append(up, self.priors[key[1]][5])
                do = np.append(do, self.priors[key[1]][4])

            elif self.priors[key[1]][1]== 'lognormal_hat':
                up = np.append(up, 10**self.priors[key[1]][5])
                do = np.append(do, 10**self.priors[key[1]][4])
            
    
            else:
                raise Exception('Sorry mode in prior type not understood: ', key )
        self.bounds = (do, up)
        #print(do)
        #print(up)
        return self.bounds
    

    @staticmethod
    def gaussian(x, amplitude, center, sigma):
        """Gaussian line profile"""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    @staticmethod
    def powerlaw_continuum(x, amplitude, x0, alpha):
        """Power-law continuum"""
        return PowerLaw1D.evaluate(x, amplitude, x0, alpha)
    
    @staticmethod
    def linear_continuum(x, slope, intercept):
        """Linear continuum"""
        return slope * x + intercept
    
   
    
    def redshift_wavelength(self, rest_wavelength: float, z: float) -> float:
        """
        Apply redshift to rest wavelength.
        
        Parameters:
        -----------
        rest_wavelength : float
            Rest wavelength in Angstroms
        z : float
            Redshift parameter
            
        Returns:
        --------
        float
            Observed wavelength
        """
        return rest_wavelength * (1 + z)
    
    def parse_line_components(self, line_names: List[str], 
                            components: List[int]) -> Dict[int, List[str]]:
        """
        Parse line names and components into groups.
        
        Parameters:
        -----------
        line_names : list
            List of emission line names
        components : list
            List of component numbers (same length as line_names)
            
        Returns:
        --------
        dict
            Dictionary mapping component numbers to lists of line names
        """
        if len(line_names) != len(components):
            raise ValueError("line_names and components must have the same length")
        
        component_groups = {}
        for line_name, comp in zip(line_names, components):
            if comp not in component_groups:
                component_groups[comp] = []
            component_groups[comp].append(line_name)
        
        return component_groups
    
    
    def fit_multiple_components_with_doublets(self,
                                            priors,
                                            continuum: str = 'linear',
                                            z_init: Union[float, List[float]] = 0.0,
                                            fwhm_init: Union[float, List[float]] = 2.0) -> Dict:
        """
        Fit multiple emission lines grouped by velocity components with doublet support.
        
        Parameters:
        -----------
        line_names : list
            Names of emission lines to fit
        components : list
            Component numbers for each line (same length as line_names)
        doublets : list, optional
            Doublet names for each line (None for single lines)
        doublet_ratios : list, optional
            Custom ratios for doublets (if not using predefined ratios)
        profile : str
            Line profile for all lines
        continuum : str
            Continuum type
            
        Returns:
        --------
        dict
            Combined fitting results
        """
        # Parse components and doublets
        parsed_data = self.parse_line_components_and_doublets(
            line_names, components, doublets, doublet_ratios)
        
        component_groups = parsed_data['component_groups']
        doublet_info = parsed_data['doublet_info']
        unique_components = sorted(component_groups.keys())
        n_components = len(unique_components)
        
        # Handle initial parameter setup
        if isinstance(z_init, (int, float)):
            z_inits = [z_init] * n_components
        else:
            if len(z_init) != n_components:
                raise ValueError(f"z_init list length ({len(z_init)}) must match number of components ({n_components})")
            z_inits = z_init
            
        if isinstance(fwhm_init, (int, float)):
            fwhm_inits = [fwhm_init] * n_components
        else:
            if len(fwhm_init) != n_components:
                raise ValueError(f"fwhm_init list length ({len(fwhm_init)}) must match number of components ({n_components})")
            fwhm_inits = fwhm_init
        
        sigma_inits = [fwhm / 2.355 for fwhm in fwhm_inits]
        
        if profile == 'gaussian' and continuum == 'linear':
            def fit_func(x, *params):
                # Parameter structure for each component:
                # [single_line_amps..., doublet_primary_amps..., z, sigma]
                # Final: [slope, intercept]
                
                result = self.linear_continuum(x, params[-2], params[-1])
                param_idx = 0
                
                for comp_idx, comp_num in enumerate(unique_components):
                    comp_data = component_groups[comp_num]
                    
                    # Count parameters needed for this component
                    n_single_lines = len(comp_data['single_lines'])
                    n_doublet_groups = len(comp_data['doublets'])
                    
                    # Extract parameters for this component
                    single_amps = params[param_idx:param_idx + n_single_lines]
                    param_idx += n_single_lines
                    
                    doublet_primary_amps = params[param_idx:param_idx + n_doublet_groups]
                    param_idx += n_doublet_groups
                    
                    z_comp = params[param_idx]
                    sigma_comp = params[param_idx + 1]
                    param_idx += 2
                    
                    # Add single lines
                    for line_idx, line_name in enumerate(comp_data['single_lines']):
                        rest_wl = self.emission_lines[line_name]
                        observed_center = self.redshift_wavelength(rest_wl, z_comp)
                        result += self.gaussian(x, single_amps[line_idx], observed_center, sigma_comp)
                    
                    # Add doublet lines
                    for doublet_idx, (doublet_name, doublet_lines) in enumerate(comp_data['doublets'].items()):
                        primary_amp = doublet_primary_amps[doublet_idx]
                        doublet_ratio = doublet_info[doublet_name]['ratio']
                        doublet_line_names = doublet_info[doublet_name]['lines']
                        
                        # Determine which line is primary (stronger) and secondary
                        if len(doublet_lines) != 2:
                            raise ValueError(f"Doublet {doublet_name} must have exactly 2 lines")
                        
                        # Find primary line (first in the predefined order)
                        primary_line = doublet_line_names[0]
                        secondary_line = doublet_line_names[1]
                        
                        # Add primary line
                        rest_wl_primary = self.emission_lines[primary_line]
                        observed_center_primary = self.redshift_wavelength(rest_wl_primary, z_comp)
                        result += self.gaussian(x, primary_amp, observed_center_primary, sigma_comp)
                        
                        # Add secondary line with fixed ratio
                        secondary_amp = primary_amp / doublet_ratio
                        rest_wl_secondary = self.emission_lines[secondary_line]
                        observed_center_secondary = self.redshift_wavelength(rest_wl_secondary, z_comp)
                        result += self.gaussian(x, secondary_amp, observed_center_secondary, sigma_comp)
                
                return result
            
            # Build initial parameters and names
            p0 = []
            param_names = []
            
            for comp_idx, comp_num in enumerate(unique_components):
                comp_data = component_groups[comp_num]
                
                # Single line amplitudes
                for line_name in comp_data['single_lines']:
                    amp_guess = (np.max(flux) - np.min(flux)) / len(line_names)
                    p0.append(amp_guess)
                    param_names.append(f'{line_name}_comp{comp_num}_amplitude')
                
                # Doublet primary amplitudes
                for doublet_name in comp_data['doublets'].keys():
                    primary_amp_guess = (np.max(flux) - np.min(flux)) / len(line_names) * 2
                    p0.append(primary_amp_guess)
                    param_names.append(f'{doublet_name}_comp{comp_num}_primary_amplitude')
                
                # Component redshift and sigma
                p0.append(z_inits[comp_idx])
                p0.append(sigma_inits[comp_idx])
                param_names.append(f'component{comp_num}_redshift')
                param_names.append(f'component{comp_num}_sigma')
            
            # Continuum parameters
            p0.extend([0.0, np.median(flux)])
            param_names.extend(['slope', 'intercept'])
            
        else:
            raise ValueError(f"Configuration {profile}/{continuum} not implemented for doublet fitting")
        
        try:
            # Perform the fit
            popt, pcov = curve_fit(fit_func, wavelengths, flux, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate goodness of fit
            fitted_flux = fit_func(wavelengths, *popt)
            residuals = flux - fitted_flux
            chi_squared = np.sum(residuals**2)
            r_squared = 1 - np.sum(residuals**2) / np.sum((flux - np.mean(flux))**2)
            
            # Parse results
            result = {
                'line_names': line_names,
                'components': components,
                'doublets': doublets,
                'component_groups': component_groups,
                'doublet_info': doublet_info,
                'profile': profile,
                'continuum': continuum,
                'parameters': popt,
                'parameter_names': param_names,
                'errors': perr,
                'chi_squared': chi_squared,
                'r_squared': r_squared,
                'fitted_flux': fitted_flux,
                'residuals': residuals,
                'individual_lines': {},
                'component_params': {},
                'doublet_params': {}
            }
            
            # Extract component and line parameters
            param_idx = 0
            c = 299792.458  # km/s
            
            for comp_idx, comp_num in enumerate(unique_components):
                comp_data = component_groups[comp_num]
                n_single_lines = len(comp_data['single_lines'])
                n_doublet_groups = len(comp_data['doublets'])
                
                # Skip to component parameters
                z_param_idx = param_idx + n_single_lines + n_doublet_groups
                z_comp = popt[z_param_idx]
                sigma_comp = popt[z_param_idx + 1]
                z_comp_error = perr[z_param_idx]
                sigma_comp_error = perr[z_param_idx + 1]
                
                result['component_params'][comp_num] = {
                    'redshift': z_comp,
                    'redshift_error': z_comp_error,
                    'velocity_shift': c * z_comp,
                    'velocity_shift_error': c * z_comp_error,
                    'sigma': sigma_comp,
                    'sigma_error': sigma_comp_error,
                    'fwhm': 2.355 * sigma_comp,
                    'fwhm_error': 2.355 * sigma_comp_error,
                    'single_lines': comp_data['single_lines'],
                    'doublets': list(comp_data['doublets'].keys())
                }
                
                # Extract single line parameters
                for line_idx, line_name in enumerate(comp_data['single_lines']):
                    rest_wl = self.emission_lines[line_name]
                    observed_center = self.redshift_wavelength(rest_wl, z_comp)
                    amplitude = popt[param_idx + line_idx]
                    amplitude_error = perr[param_idx + line_idx]
                    equivalent_width = np.abs(amplitude * sigma_comp * np.sqrt(2*np.pi))
                    
                    line_result = {
                        'component': comp_num,
                        'is_doublet': False,
                        'amplitude': amplitude,
                        'amplitude_error': amplitude_error,
                        'rest_wavelength': rest_wl,
                        'observed_center': observed_center,
                        'redshift': z_comp,
                        'velocity_shift': c * z_comp,
                        'sigma': sigma_comp,
                        'fwhm': 2.355 * sigma_comp,
                        'equivalent_width': equivalent_width
                    }
                    result['individual_lines'][line_name] = line_result
                
                # Extract doublet parameters
                doublet_start_idx = param_idx + n_single_lines
                for doublet_idx, (doublet_name, doublet_lines) in enumerate(comp_data['doublets'].items()):
                    primary_amp = popt[doublet_start_idx + doublet_idx]
                    primary_amp_error = perr[doublet_start_idx + doublet_idx]
                    doublet_ratio = doublet_info[doublet_name]['ratio']
                    doublet_line_names = doublet_info[doublet_name]['lines']
                    
                    # Store doublet information
                    result['doublet_params'][f'{doublet_name}_comp{comp_num}'] = {
                        'component': comp_num,
                        'primary_amplitude': primary_amp,
                        'primary_amplitude_error': primary_amp_error,
                        'ratio': doublet_ratio,
                        'lines': doublet_line_names
                    }
                    
                    # Add individual doublet lines
                    primary_line = doublet_line_names[0]
                    secondary_line = doublet_line_names[1]
                    secondary_amp = primary_amp / doublet_ratio
                    
                    for line_name, amplitude in [(primary_line, primary_amp), 
                                               (secondary_line, secondary_amp)]:
                        rest_wl = self.emission_lines[line_name]
                        observed_center = self.redshift_wavelength(rest_wl, z_comp)
                        equivalent_width = np.abs(amplitude * sigma_comp * np.sqrt(2*np.pi))
                        
                        line_result = {
                            'component': comp_num,
                            'is_doublet': True,
                            'doublet_name': doublet_name,
                            'doublet_role': 'primary' if line_name == primary_line else 'secondary',
                            'amplitude': amplitude,
                            'amplitude_error': primary_amp_error if line_name == primary_line else primary_amp_error / doublet_ratio,
                            'rest_wavelength': rest_wl,
                            'observed_center': observed_center,
                            'redshift': z_comp,
                            'velocity_shift': c * z_comp,
                            'sigma': sigma_comp,
                            'fwhm': 2.355 * sigma_comp,
                            'equivalent_width': equivalent_width,
                            'doublet_ratio': doublet_ratio
                        }
                        result['individual_lines'][line_name] = line_result
    
            return result
            
        except Exception as e:
            print(f"Multi-component doublet fitting failed: {e}")
            return None

