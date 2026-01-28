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
       
    def __init__(self, wave=np.array([]), flux=np.array([]), error=np.array([]), z=1, N=5000,ncpu=1, progress=True,sampler='emcee', priors= {'z':[0, 'normal_hat', 0,0.003,0,0]}):
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
        self.waves = wave.copy() # wavelength 
        self.wave = wave.copy() # wavelength 
        self.fluxs = flux.copy() # flux density
        self.errors = error.copy() # errors
        self.error = error.copy() # errors
        self.ncpu= ncpu # number of cpus to use in the fit 
        self.sampler = sampler

    def _setup_(self, wv= None):  
                
        if self.priors['z'][0]==0:
            self.priors['z'][0]=self.z
            if (self.priors['z'][1]=='normal_hat') & (self.priors['z'][2]==0):
                self.priors['z'][2] = self.z
                self.priors['z'][3] = 200/3e5*(1+self.z)
                self.priors['z'][4] = self.z-1000/3e5*(1+self.z)
                self.priors['z'][5] = self.z+1000/3e5*(1+self.z)
        if 'zBLR' in self.priors.keys():
            if self.priors['zBLR'][0]==0:
                self.priors['zBLR'][0]=self.z

                if (self.priors['zBLR'][1]=='normal_hat'):
                    self.priors['zBLR'][2] = self.z
                    self.priors['zBLR'][3] = 200/3e5*(1+self.z)
                    self.priors['zBLR'][4] = self.z-1000/3e5*(1+self.z)
                    self.priors['zBLR'][5] = self.z+1000/3e5*(1+self.z)
       
        self.fluxs[np.isnan(self.fluxs)] = 0
        self.flux = self.fluxs.data[np.invert(self.fluxs.mask)]
        self.wave = self.waves[np.invert(self.fluxs.mask)]

        if isinstance(self.error, np.ma.MaskedArray) == True:
            self.error = self.error.data
        self.error[~np.isfinite(self.error)] = 10000*np.nanmedian(self.error)
        self.error[self.error==0] = 10000*np.nanmedian(self.error)

        if wv is not None:

            self.fit_loc = np.where((self.wave>(wv-170)*(1+self.z)/1e4)&(self.wave<(wv+200)*(1+self.z)/1e4))[0]
            sel=  np.where(((self.wave<(wv+20)*(1+self.z)/1e4))& (self.wave>(wv-20)*(1+self.z)/1e4))[0]

            self.flux_fitloc = self.flux[self.fit_loc]
            self.wave_fitloc = self.wave[self.fit_loc]
            self.error_fitloc = self.error[self.fit_loc]

            self.flux_zoom = self.flux[sel]
            self.wave_zoom = self.wave[sel]


        else: 
            self.flux_fitloc = self.flux.copy()
            self.wave_fitloc = self.wave.copy()
            self.error_fitloc = self.error.copy() 
            
    def _fit_emcee(self, model_name='custom'):
        nwalkers, ndim = self.pos.shape
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability_general, args=())
    
        self.sampler.run_mcmc(self.pos, self.N, progress=self.progress)
        self.flat_samples = self.sampler.get_chain(discard=int(0.5*self.N), thin=15, flat=True)      
        
        self.chains = {'name': model_name}
        for i in range(len(self.labels)):
            self.chains[self.labels[i]] = self.flat_samples[:,i]
            
        self.props = self.prop_calc()
        self.like_chains = self.sampler.get_log_prob(discard=int(0.5*self.N),thin=15, flat=True)

    def _fit_curvefit(self, model_name='custom'):

        use = self.pos_l < self.bounds_est()[0]
        if True in use:
            raise ValueError(f'Initial guess is outside of the lower bounds {print(self.labels[use])}')
        use = self.pos_l > self.bounds_est()[1]
        if True in use:
            raise ValueError(f'Initial guess is outside of the higher bounds in {print(self.labels[use])}')
        
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(self.fitted_model, self.wave_fitloc, self.flux_fitloc, p0= self.pos_l, sigma=self.error_fitloc, bounds = self.bounds_est())
        errs = np.sqrt(np.diag(pcov))

        self.props = {'name': model_name}
        self.chains = {'name': model_name}
        self.props['popt'] = popt
        for i, name in enumerate(self.labels):
            self.props[name] = [popt[i], errs[i], errs[i]]
            self.chains[name] = np.random.normal(popt[i], errs[i], size=1000)
    
    def _post_process(self):
        self.chi2 = np.nansum(((self.flux_fitloc-self.fitted_model(self.wave_fitloc, *self.props['popt']))**2)/self.error_fitloc**2)
        self.BIC = self.chi2+ len(self.props['popt'])*np.log(len(self.flux_fitloc))
        
        self.yeval = self.fitted_model(self.waves, *self.props['popt'])
        self.yeval_fitloc = self.fitted_model(self.wave_fitloc, *self.props['popt'])
        
    def _logprior_test(self):
        if (self.log_prior_fce(self.pos_l, self.pr_code)==-np.inf) | (self.log_prior_fce(self.pos_l, self.pr_code)== np.nan):
            print(logprior_general_test(self.pos_l, self.pr_code,self.labels))
                
            raise Exception('Logprior function returned nan or -inf on initial conditions. You should double check that your priors\
                            boundries are sensible')
    def _generate_init_walkers(self):
        for i, name in enumerate(self.labels):
            self.pos_l[i] = self.pos_l[i] if self.priors[name][0]==0 else self.priors[name][0]                
            self.pos = np.random.normal(self.pos_l, abs(self.pos_l*0.1), (self.nwalkers, len(self.pos_l)))
            self.pos[:,0] = np.random.normal(self.z,0.001, self.nwalkers)

            if name =='zBLR':
                    self.pos[:,i] = np.random.normal(self.z,0.001, self.nwalkers)
        
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
        self._setup_(wv=6563)
        peak = abs(np.ma.max(self.flux_zoom))
        self.nwalkers=32
        
        cont = np.median(self.flux[self.fit_loc])
        if cont<0:
            cont=0.01

        if self.model=='BLR_simple':
            self.labels=['z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak']
            
            self.fitted_model = H_models.Halpha_wBLR
            self.pos_l = np.array([self.z,cont,0.001, peak/2, peak/4, peak/4, self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],self.priors['zBLR'][0],peak/6, peak/6])
            self.res = {'name': 'Halpha_wth_BLR'}
        
        elif self.model=='BLR':
            self.labels=['z', 'cont','cont_grad', 'Hal_peak','BLR_Hal_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'zBLR', 'SIIr_peak', 'SIIb_peak',\
                         'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel']
            
            self.fitted_model = H_models.Halpha_BLR_outflow
            self.pos_l = np.array([self.z,cont,0.001, peak/2, peak/4, peak/4, self.priors['Nar_fwhm'][0], self.priors['BLR_fwhm'][0],self.priors['zBLR'][0],peak/6, peak/6,\
                              peak/6, peak/6, 700, -100])
           
            self.res = {'name': 'Halpha_wth_BLR'}
        
        elif self.model=='gal':
            self.fitted_model = H_models.Halpha
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak']
            
            self.pos_l = np.array([self.z,cont,0.01, peak/1.3, peak/10,self.priors['Nar_fwhm'][0],peak/6, peak/6 ])
            self.res = {'name': 'Halpha_wth_BLR'}
                 
        elif self.model=='outflow':
            self.fitted_model = H_models.Halpha_outflow
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'Hal_out_peak', 'NII_out_peak', 'outflow_fwhm', 'outflow_vel']
            self.pos_l = np.array([self.z,cont,0.01, peak/1.3, peak/10,self.priors['Nar_fwhm'][0],peak/6, peak/6, peak/6, peak/6, self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0]])
            self.res = {'name': 'Halpha_wth_out'}   
        else:
            raise Exception('self.model variable not understood. Available model keywords: BLR, BLR_simple, outflow, gal')
        
        self.log_prior_fce = logprior_general
        self.pr_code = self.prior_create()
        self._generate_init_walkers()
        self._logprior_test()
        if self.sampler =='emcee':
            self._fit_emcee()
        elif self.sampler=='leastsq':
            self._fit_curvefit()
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')

        self._post_process()
        
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
        self._setup_()
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        if (expand_prism==1) | (expand_prism==True):
            self.fit_loc = np.where((self.wave>4600*(1+self.z)/1e4)&(self.wave<5600*(1+self.z)/1e4))[0]

        sel=  np.where((self.wave<5025*(1+self.z)/1e4)& (self.wave>4980*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        
        try:
            peak = abs((np.max(self.flux_zoom)))
        except:
            peak = abs((np.max(self.flux[self.fit_loc])))

        selb =  np.where((self.wave<4880*(1+self.z)/1e4)& (self.wave>4820*(1+self.z)/1e4))[0]
        self.flux_zoomb = self.flux[selb]
        self.wave_zoomb = self.wave[selb]
        try:
            peak_beta = abs((np.max(self.flux_zoomb)))
        except:
            peak_beta = abs(peak/3)

        self.nwalkers=64

        if  (self.model=='outflow_simple') | (self.model=='outflow'):
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak']
            self.fitted_model = O_models.OIII_outflow
            
            cont_init = abs(np.median(self.flux[self.fit_loc]))
            self.pos_l = np.array([self.z,cont_init,0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3])
            self.res = {'name': 'OIII_outflow_simple'}
        
        elif (self.model== 'gal_simple') | (self.model== 'gal'):
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak']
            self.fitted_model = O_models.OIII_gal
            cont_init = abs(np.median(self.flux[self.fit_loc]))
            self.pos_l = np.array([self.z,cont_init,0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta])        
            self.res = {'name': 'OIII_simple'}
            
        elif (self.model=='BLR_simple'):
            
            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'Nar_fwhm', 'Hbeta_peak', 'zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm']
            
            if Fe_template != 0:
                self.labels.append('Fe_peak')
                self.labels.append('Fe_fwhm')
                self.fitted_model = O_models.OIII_fe_models(self.template)
                self.fitted_model = self.fitted_model.OIII_gal_BLR_Fe

                self.res = {'name': 'OIII_BLR_simple_fe'}
                self.pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta, self.z, peak_beta/2, self.priors['BLR_fwhm'][0],\
                                np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]])

            else:
                self.pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, self.priors['Nar_fwhm'][0], peak_beta, self.z, peak_beta/2, self.priors['BLR_fwhm'][0]])
                self.fitted_model = O_models.OIII_gal_BLR
                self.res = {'name': 'OIII_BLR_simple'}
                
            self.res = {'name': 'OIII_BLR_simple'}
        
        elif (self.model== 'BLR_outflow'):

            self.labels=['z', 'cont','cont_grad', 'OIII_peak', 'OIII_out_peak', 'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hbeta_peak', 'Hbeta_out_peak','zBLR', 'BLR_Hbeta_peak', 'BLR_fwhm']
            
            if Fe_template != 0:
                self.labels.append('Fe_peak')
                self.labels.append('Fe_fwhm')
                self.fitted_model = O_models.OIII_fe_models(self.template)
                self.fitted_model = self.fitted_model.OIII_outflow_BLR_Fe

                self.pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3,\
                                  self.z, peak_beta/2, self.priors['BLR_fwhm'][0],\
                                  np.median(self.flux[self.fit_loc]), self.priors['Fe_fwhm'][0]])
                self.res = {'name': 'BLR_outflow_Fe'}
                 
            else:
                self.fitted_model = O_models.OIII_outflow_BLR
                self.pos_l = np.array([self.z,np.median(self.flux[self.fit_loc]),0.001, peak/2, peak/6, self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0],self.priors['outflow_vel'][0], peak_beta, peak_beta/3,\
                                 self.z, peak_beta/2, self.priors['BLR_fwhm'][0]])
                self.res = {'name': 'BLR_outflow'}
                
            self.log_prior_fce = logprior_general
            self.pr_code = self.prior_create()                      
 
        else:
            raise Exception('self.model variable not understood. Available self.model keywords: outflow_simple, BLR_simple, BLR_outflow,  gal_simple')
        self.log_prior_fce = logprior_general
        self.pr_code = self.prior_create()
        self._generate_init_walkers()
        self._logprior_test()

        if self.sampler =='emcee':     
            self._fit_emcee()
        elif self.sampler=='leastsq':
            self._fit_curvefit()
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')
        
        self._post_process()
        
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
        self._setup_()
        self.fit_loc = np.where((self.wave>4700*(1+self.z)/1e4)&(self.wave<5100*(1+self.z)/1e4))[0]
        self.fit_loc = np.append(self.fit_loc, np.where((self.wave>(6564.52-170)*(1+self.z)/1e4)&(self.wave<(6564.52+200)*(1+self.z)/1e4))[0])

    
    # =============================================================================
    #     Finding the initial conditions
    # =============================================================================
        sel=  np.where((self.wave<5025*(1+self.z)/1e4)& (self.wave>4980*(1+self.z)/1e4))[0]
        self.flux_zoom = self.flux[sel]
        self.wave_zoom = self.wave[sel]
        try:
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
            self.labels=['z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak', 'OIII_peak', 'Hbeta_peak']
            
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)
            self.pos_l = np.array([self.z, cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, self.priors['Nar_fwhm'][0], peak_hal*0.15, peak_hal*0.2, peak_OIII*0.8,\
                              peak_hal*0.2])  
        
            self.res = {'name': 'Halpha_OIII'}
              
        elif self.model=='outflow':
            nwalkers=64
            self.fitted_model = HO_models.Halpha_OIII_outflow
            
            self.labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak','OIII_peak', 'Hbeta_peak','SIIr_peak', 'SIIb_peak',\
                    'Nar_fwhm', 'outflow_fwhm', 'outflow_vel', 'Hal_out_peak','NII_out_peak', 'OIII_out_peak', 'Hbeta_out_peak' )
            
            cont_init = np.median(self.flux[self.fit_loc])
            if cont_init<0:
                cont_init = abs(cont_init)

            self.pos_l = np.array([self.z,cont_init, -0.1, peak_hal*0.7, peak_hal*0.3, \
                              peak_OIII*0.8, peak_hal*0.2, peak_hal*0.2, peak_hal*0.2, \
                              self.priors['Nar_fwhm'][0], self.priors['outflow_fwhm'][0], self.priors['outflow_vel'][0],
                              peak_hal*0.3, peak_hal*0.3, peak_OIII*0.2, peak_hal*0.05])
            
            
            self.res = {'name': 'Halpha_OIII_outflow'}
                
        else:
            raise Exception('self.model variable not understood. Available self.model keywords: outflow, gal')
        
        self.flux_fitloc = self.flux[self.fit_loc]
        self.wave_fitloc = self.wave[self.fit_loc]
        self.error_fitloc = self.error[self.fit_loc]
        self.pr_code = self.prior_create()
        self.log_prior_fce = logprior_general
        self._generate_init_walkers()
        self._logprior_test()

        if self.sampler =='emcee':
            self._fit_emcee()
        elif self.sampler=='leastsq':
            self._fit_curvefit()
        else:
            raise ValueError('Sampler value not understood. Should be emcee or leastsq')

        self._post_process()
        
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
        self.nwalkers= nwalkers
        
        self._setup_()

        if odd==True:
            if len(self.flux_fitloc) % 2 == 0:
                self.flux_fitloc = self.flux_fitloc[:-1]
                self.wave_fitloc = self.wave_fitloc[:-1]
                self.error_fitloc = self.error_fitloc[:-1]

        self.pr_code = self.prior_create()
       
        self.pos_l = np.zeros(len(self.labels))
            
        for i, name in enumerate(self.labels):
            self.pos_l[i] = self.priors[name][0] 
            if ('_peak' in name) & (self.priors[name][0] ==0):
                self.pos_l[i] = np.nanmean(self.error_fitloc)*np.random.uniform(5,10)
            if ('cont' == name) & (self.priors[name][0] ==0):
                self.pos_l[i] = np.nanmedian(self.flux_fitloc)*5
            
        self._generate_init_walkers()
        self._logprior_test()

        if self.sampler =='emcee':
            self._fit_emcee()
        elif self.sampler=='leastsq':
            self._fit_curvefit()
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

    def bic_calc(self, obs_wave):
        """ Calculate BIC for a given observed wavelength array
        """
        yeval_obs = self.fitted_model(obs_wave, *self.props['popt'])
        
        use = (self.wave_fitloc>=np.min(obs_wave)) & (self.wave_fitloc<=np.max(obs_wave))

        flux_setup = self.flux_fitloc[use]
        error_setup = self.error_fitloc[use]

        chi2 = np.nansum(((flux_setup-yeval_obs)/error_setup)**2)
        BIC = chi2+ len(self.props['popt'])*np.log(len(flux_setup))
        return BIC


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