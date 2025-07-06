#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:16:53 2023

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from astropy.io import fits as pyfits

nan= float('nan')

pi= np.pi
e= np.e

c=3e8


PATH='/Users/jansen/My Drive/Astro/'

from . import Utils as sp
from . import Plotting as emplot
from . import Fitting as emfit



paths = {}
paths['medium_jwst_gn'] = ['/Users/jansen/JADES/GOODS-N/NIRSpec/medium_jwst_gn/Final_products_v3.1/', '3.1','']
paths['medium_hst_gn'] = ['/Users/jansen/JADES/GOODS-N/NIRSpec/medium_hst_gn/Final_products_v3.0/', '3.0','']


paths['deep_hst_gs'] = ['/Users/jansen/JADES/GOODS-S/NIRSpec/deep_hst_gs/Final_products_v3.0/', '3.0','']
paths['medium_hst_gs_shorts'] = ['/Users/jansen/JADES/GOODS-S/NIRSpec/medium_hst_gs_shorts/Final_products_v3.0_extr3/', '3.0','_extr3']
paths['medium_jwst_gs_1180'] = ['/Users/jansen/JADES/GOODS-S/NIRSpec/medium_jwst_gs_1180/Final_products_v3.0_extr3/', '3.0','_extr3']
paths['medium_jwst_gs'] = ['/Users/jansen/JADES/GOODS-S/NIRSpec/medium_jwst_gs/', '','']



def load_data(path):
    with pyfits.open(path, memmap=False) as hdulist:
        flux_orig = hdulist['DATA'].data*1e-7*1e4
        error =  hdulist['ERR'].data*1e-7*1e4
        flux = np.ma.masked_invalid(flux_orig.copy())
        obs_wave = hdulist['wavelength'].data*1e6
    return obs_wave, flux, error

def gauss(x, k, mu, fwhm):
    sig= fwhm/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)
    y= k* e**expo
    return y


class R1000:  
    def __init__(self, path, z, ID, wave_custom=None, version ='3.0', add='_extr3', data_hdu='DATA', rev=False):
        # Store basic stuff
        self.z = z
        self.ID = ID
        self.path = path
        self.add = add
        self.data_hdu = data_hdu
        
        if version != '':
            version= '_v'+version
        self.version = version

        # Set boundries of the R1000 boundries 
        B1M = [0.7, 2.2]
        B2M = [1.66, 4.0]
        B3M = [2.87, 5.5]
        
        # Caluclating the redshifted wavelength of Halpha and OIII
        self.Hal_wv = 0.6563*(1+self.z)
        self.O3_wv = 0.5008*(1+self.z)
        if wave_custom:
            self.wave_custom = wave_custom
            self.wave_custom_obs = self.wave_custom*(1+self.z)/1e4
        
        
        # Set to Halpha gratings
        if (self.Hal_wv<B3M[1]) and (self.Hal_wv>B3M[0]):
            self.Hal_band = 'g395m_f290lp'
        elif (self.Hal_wv<B2M[1]) and (self.Hal_wv>B2M[0]):
            self.Hal_band = 'g235m_f170lp'
        elif (self.Hal_wv<B1M[1]) and (self.Hal_wv>B1M[0]):
            self.Hal_band = 'g140m_f070lp'
        else:
            self.Hal_band = None
          
        # Set OIII gratings
        if (self.O3_wv<B3M[1]) and (self.O3_wv>B3M[0]):
            self.O3_band = 'g395m_f290lp'
        elif (self.O3_wv<B2M[1]) and (self.O3_wv>B2M[0]):
            self.O3_band = 'g235m_f170lp'
        elif (self.O3_wv<B1M[1]) and (self.O3_wv>B1M[0]):
            self.O3_band = 'g140m_f070lp'
        else:
            self.O3_band = None
            
        if wave_custom:
            # Set custom gratings
            if (self.wave_custom_obs<B3M[1]) and (self.wave_custom_obs>B3M[0]):
                self.band_custom = 'g395m_f290lp'
            elif (self.wave_custom_obs<B2M[1]) and (self.wave_custom_obs>B2M[0]):
                self.band_custom = 'g235m_f170lp'
            elif (self.wave_custom_obs<B1M[1]) and (self.wave_custom_obs>B1M[0]):
                self.band_custom = 'g140m_f070lp'
            else:
                self.band_custom = None
            if rev==True:
                if (self.wave_custom_obs<B1M[1]) and (self.wave_custom_obs>B1M[0]):
                    self.band_custom = 'g140m_f070lp'
                elif (self.wave_custom_obs<B2M[1]) and (self.wave_custom_obs>B2M[0]):
                    self.band_custom = 'g235m_f170lp'
                elif (self.wave_custom_obs<B3M[1]) and (self.wave_custom_obs>B3M[0]):
                    self.band_custom = 'g395m_f290lp'
                else:
                    self.band_custom = None

            
            self.Full_path_custom = self.path + self.band_custom +'/'+ self.ID + '_' + self.band_custom +self.version+self.add+'_1D.fits'

        else:
            self.band_custom= None
            
            
            
        
        
    def load_data(self):
        try:
            if self.Hal_band != None:  
                Full_path = self.path + self.Hal_band +'/'+ self.ID + '_' + self.Hal_band+self.version +self.add+'_1D.fits'
                with pyfits.open(Full_path, memmap=False) as hdulist:
                    flux_orig = hdulist[self.data_hdu].data*1e-7*1e4*1e15
                    self.Hal_error =  hdulist['ERR'].data*1e-7*1e4*1e15
                    self.Hal_flux = np.ma.masked_invalid(flux_orig.copy())
                    self.Hal_obs_wave = hdulist['wavelength'].data*1e6
                    
                    #plt.figure()
                    #plt.plot(self.Hal_obs_wave, self.Hal_flux, drawstyle='steps-mid')
        except:
            ls =  0 
        try: 
            if self.O3_band != None:  
                Full_path = self.path + self.O3_band +'/'+ self.ID + '_' + self.O3_band +self.version+self.add+'_1D.fits'
                with pyfits.open(Full_path, memmap=False) as hdulist:
                    flux_orig = hdulist[self.data_hdu].data*1e-7*1e4*1e15
                    self.O3_error =  hdulist['ERR'].data*1e-7*1e4*1e15
                    self.O3_flux = np.ma.masked_invalid(flux_orig.copy())
                    self.O3_obs_wave = hdulist['wavelength'].data*1e6
                    #plt.figure()
                    #plt.plot(self.O3_obs_wave, self.O3_flux, drawstyle='steps-mid')
        except:
            ls = 0
        if self.band_custom:  
            try:
                Full_path = self.path + self.band_custom +'/'+ self.ID + '_' + self.band_custom +self.version+self.add+'_1D.fits'
                with pyfits.open(Full_path, memmap=False) as hdulist:
                    flux_orig = hdulist[self.data_hdu].data*1e-7*1e4*1e15
                    self.custom_error =  hdulist['ERR'].data*1e-7*1e4*1e15
                    self.custom_flux = np.ma.masked_invalid(flux_orig.copy())
                    self.custom_obs_wave = hdulist['wavelength'].data*1e6
            except:
                Full_path = self.path + self.band_custom +'/'+ self.ID + '_' + self.band_custom +self.version+self.add+'_1D.fits'
                with pyfits.open(Full_path, memmap=False) as hdulist:
                    flux_orig = hdulist[self.data_hdu].data*1e-7*1e4*1e15
                    self.custom_error =  hdulist['ERR'].data*1e-7*1e4*1e15
                    self.custom_flux = np.ma.masked_invalid(flux_orig.copy())
                    self.custom_obs_wave = hdulist['wavelength'].data*1e6
                
                
                '''
                f,ax = plt.subplots()
                ax.plot(self.custom_obs_wave, self.custom_flux, drawstyle='steps-mid')
                ax.vlines(self.wave_custom_obs, 0,1, color='red')
                
                yer = np.ma.median( np.ma.masked_invalid( self.custom_error))
                
                ax.set_ylim(-yer, 5*yer)
                
                ax.set_title(self.ID)
                '''
                
    def Fitting_Halpha(self, N=10000, progress=True,sampler='emcee',priors= {'z':[0, 'normal', 0,0.003],\
                                                   'cont':[0,'loguniform',-5,1.7],\
                                                   'cont_grad':[0,'normal',0,0.3], \
                                                   'Hal_peak':[0,'loguniform',-3,2.],\
                                                   'Nar_fwhm':[300,'uniform',100,900],\
                                                   'SIIr_peak':[0,'loguniform',-5,1.7],\
                                                   'SIIb_peak':[0,'loguniform',-5,1.7],\
                                                   'NII_peak':[0,'loguniform',-5,1.7]}):
        if self.Hal_band != None:  
            dvstd = 300/3e5*(1+self.z)
            if priors['z'][2] ==0:
                
                priors['z'] = [self.z, 'normal', self.z, dvstd]
            
            
            self.Hal_fits = emfit.Fitting(self.Hal_obs_wave, self.Hal_flux, self.Hal_error, self.z, N=N, progress=progress, priors=priors, sampler=sampler)
            self.Hal_fits.fitting_Halpha(model='gal')
            
            self.Halpha_flux = sp.flux_calc_mcmc(self.Hal_fits, 'Han', norm=1e-15)
            self.N2_flux = sp.flux_calc_mcmc(self.Hal_fits, 'NIIt', norm=1e-15)
            self.S2b_flux = sp.flux_calc_mcmc(self.Hal_fits, 'SIIb', norm=1e-15)
            self.S2r_flux = sp.flux_calc_mcmc(self.Hal_fits, 'SIIr', norm=1e-15)
        
        else:
            self.Hal_fits = None
            self.Halpha_flux = [0,0,0]
            self.N2_flux = [0,0,0]
            self.S2b_flux = [0,0,0]
            self.S2r_flux = [0,0,0]
        
    def Fitting_O3(self, N=10000, progress=True,sampler='emcee', priors= {'z':[0, 'normal', 0,0.003],\
                                                          'cont':[0,'loguniform',-4,1.7],\
                                                          'cont_grad':[0,'normal',0,0.3], \
                                                          'Nar_fwhm':[300,'uniform',200,900],\
                                                          'outflow_fwhm':[600,'uniform', 300,1500],\
                                                          'outflow_vel':[-50,'normal', 0,300],\
                                                          'OIII_peak':[0,'loguniform',-4,2.0],\
                                                          'OIII_out_peak':[0,'loguniform',-3,1.7],\
                                                          'Hbeta_peak':[0,'loguniform',-4,1.7],\
                                                          'Hbeta_fwhm':[200,'uniform',120,900],\
                                                          'Hbeta_vel':[10,'normal', 0,10]}):

        if self.O3_band != None:  
            dvstd = 300/3e5*(1+self.z)
            if priors['z'][2] ==0:
                priors['z'] = [self.z, 'normal', self.z, dvstd]
            self.O3_fits = emfit.Fitting(self.O3_obs_wave, self.O3_flux, self.O3_error, self.z, N=N, progress=progress, priors=priors, sampler=sampler)
            self.O3_fits.fitting_OIII(model='gal')

            self.O3_flux = sp.flux_calc_mcmc(self.O3_fits, 'OIIIt', norm=1e-15)
            self.Hbeta_flux = sp.flux_calc_mcmc(self.O3_fits, 'Hbeta', norm=1e-15)
        else:
            self.O3_fits = None
            self.O3_flux = [0,0,0]
            self.O3_flux = [0,0,0]
    
    def Fitting_custom(self,model, N=10000, progress=True, labels=['z', 'cont', 'cont_grad', 'peak', 'Nar_fwhm'], plot=1, useb=[-150,150], use=np.array([0]), \
                       sampler='emcee', priors= {'z':[0, 'normal', 0,0.001],\
                                                          'cont':[0.1,'loguniform',-3,1],\
                                                          'cont_grad':[-0.1,'normal',0,0.3], \
                                                          'Nar_fwhm':[300,'uniform',100,900],\
                                                          'peak':[0.2, 'loguniform', -3,1]}):
        self.model = model
        self.labels= labels
        self.priors = priors
        if len(use) ==1:
            use = np.where( (self.custom_obs_wave>((self.wave_custom+useb[0])*(1+self.z)/1e4)) & (self.custom_obs_wave< ((self.wave_custom+useb[1])*(1+self.z)/1e4))   )[0]

        self.use = use
        self.Fitting = emfit.Fitting(self.custom_obs_wave[use], self.custom_flux[use], self.custom_error[use],z=self.z, priors=self.priors,N=10000, progress=progress, sampler=sampler)
        self.Fitting.fitting_general(self.model, labels, emfit.logprior_general)
        
        self.yeval = self.model(self.custom_obs_wave, *self.Fitting.props['popt'])
        
        
        if plot==1:
            f,ax = plt.subplots(1)
            ax.plot(self.custom_obs_wave, self.custom_flux, drawstyle='steps-mid')
            ax.plot(self.custom_obs_wave, self.yeval, 'r--' )
            ax.vlines(self.wave_custom_obs, 0,1, color='red')
            
            ax.set_xlim((self.wave_custom-200)*(1+self.z)/1e4, (self.wave_custom+150)*(1+self.z)/1e4 )
            
            ylim = 1.11*max(self.yeval)
            yer = np.ma.median( np.ma.masked_invalid( self.custom_error))
            if ylim<yer*3:
                ylim= yer*3
            
            ax.set_ylim(-yer, ylim)  
            
            ax.set_title(self.ID + ', wave = '+str(self.wave_custom))
        
        
        return self.Fitting
    
    def weed_out(self, N=1000):

        for key in self.Fitting.labels:
            self.Fitting.chains[key] = np.random.choice(self.Fitting.chains[key], N, replace=False)
        self.Fitting.like_chains = np.random.choice(self.Fitting.like_chains, N, replace=False)
                   

class R100:  
    def __init__(self, path='', z=0, ID='', version ='4.0', add='', full_path=None):
        # Store basic stuff
        self.z = z
        self.ID = ID
        self.path = path
        self.band = 'prism_clear'
        if version != '':
            version= '_v'+version
        self.version = version

        self.version = version
        self.add = add 
        self.Full_path = full_path

        if self.Full_path == None:
            self.Full_path = self.path +'prism_clear/'+ self.ID + '_prism_clear' +self.version+self.add+'_1D.fits'

        with pyfits.open(self.Full_path, memmap=False) as hdulist:
            flux_orig = hdulist['DATA'].data*1e-7*1e4*1e15
            self.error =  hdulist['ERR'].data*1e-7*1e4*1e15
            self.flux = np.ma.masked_invalid(flux_orig.copy())
            self.obs_wave = hdulist['wavelength'].data*1e6
            
         
    def load_data(self):
        if self.Full_path == None:
            self.Full_path = self.path +'prism_clear/'+ self.ID + '_prism_clear' +self.version+self.add+'_1D.fits'

        with pyfits.open(self.Full_path, memmap=False) as hdulist:
            flux_orig = hdulist['DATA'].data*1e-7*1e4*1e15
            self.error =  hdulist['ERR'].data*1e-7*1e4*1e15
            self.flux = np.ma.masked_invalid(flux_orig.copy())
            self.obs_wave = hdulist['wavelength'].data*1e6
            
            plt.figure()
            plt.plot(self.obs_wave, self.flux, drawstyle='steps-mid')
        
                  
                
    def Fitting_Halpha(self, N=10000, progress=True,priors= {'z':[0, 'normal', 0,0.003],\
                                                   'cont':[0,'loguniform',-5,1],\
                                                   'cont_grad':[0,'normal',0,0.3], \
                                                   'Hal_peak':[0,'loguniform',-3,1],\
                                                   'Nar_fwhm':[1000,'uniform',500,3000],\
                                                   'SIIr_peak':[0,'loguniform',-3,1],\
                                                   'SIIb_peak':[0,'loguniform',-3,1],\
                                                   'NII_peak':[0,'loguniform',-3,1]}):
         
        z_std = 200/3e5*(1+self.z)
        zlim = 600/3e5*(1+self.z)
        priors['z'] = [self.z, 'normal_hat', self.z, z_std, self.z-zlim, self.z+zlim]
        
        
        self.Hal_fits = emfit.Fitting(self.obs_wave, self.flux, self.error, self.z, N=N, progress=progress, priors=priors)
        self.Hal_fits.fitting_Halpha(model='gal')
        
        f,ax = plt.subplots(1)
        emplot.plotting_Halpha(self.obs_wave, self.flux, ax, self.Hal_fits.props, self.Hal_fits.fitted_model)
        
        
        
        self.Halpha_flux = sp.flux_calc_mcmc(self.Hal_fits.props, self.Hal_fits.chains, 'Han', norm=1e-15)
        self.N2_flux = sp.flux_calc_mcmc(self.Hal_fits.props, self.Hal_fits.chains, 'NIIt', norm=1e-15)
        
    def Fitting_O3(self, N=10000, progress=True, priors= {'z':[0, 'normal', 0,0.003],\
                                                          'cont':[0,'loguniform',-3,1],\
                                                          'cont_grad':[0,'normal',0,0.3], \
                                                          'Nar_fwhm':[300,'uniform',100,900],\
                                                          'outflow_fwhm':[600,'uniform', 300,1500],\
                                                          'outflow_vel':[-50,'normal', 0,300],\
                                                          'OIII_peak':[0,'loguniform',-3,1],\
                                                          'OIII_out_peak':[0,'loguniform',-3,1],\
                                                          'Hbeta_peak':[0,'loguniform',-3,1],\
                                                          'Hbeta_fwhm':[200,'uniform',120,900],\
                                                          'Hbeta_vel':[10,'normal', 0,10]}):
        
        
        priors['z'] = [self.z, 'uniform', self.z-0.01, self.z+0.01]
        self.O3_fits = emfit.Fitting(self.obs_wave, self.flux, self.error, self.z, N=N, progress=progress, priors=priors)
        
        self.O3_fits.fitting_OIII(model='gal')
        
        f,ax = plt.subplots(1)
        #ax.plot(self.O3_obs_wave, self.O3_flux)
        #ax.plot(self.O3_obs_wave, O3_fits.fitted_model(self.O3_obs_wave, *O3_fits.props['popt']), 'r--')
        emplot.plotting_OIII(self.obs_wave, self.flux, ax, self.O3_fits.props, self.O3_fits.fitted_model)
        
        self.O3_flux = sp.flux_calc_mcmc(self.O3_fits.props, self.O3_fits.chains, 'OIIIt', norm=1e-15)
        self.Hbeta_flux = sp.flux_calc_mcmc(self.O3_fits.props, self.O3_fits.chains, 'Hbeta', norm=1e-15)

    def cal_1500_mag(self, z, lmin=1475, lmax=1525):
        """
        Input: MPDAF Spectrum object, redshift
        Output: UV mag, upper bound, lower bound, F(1500), Error F(1500)

        Calculate the 1500 magnitude.
        Reference here: https://astronomy.stackexchange.com/questions/35396/how-to-convert-luminosity-at-rest-frame-wavelength-of-1450-a-to-absolute-magnitu
        """
        from astropy.stats import sigma_clipped_stats
        import astropy.units as u
        from astropy.cosmology import Planck18 as cosmo 

        self.rest_wave = self.obs_wave/(1+z)*1e4
        self.rest_flux = self.flux*1e-15/1e4*(1+z)
        use =(self.rest_wave>lmin) & (self.rest_wave<lmax)

        ### start by calculating the flux at 1500A rest-frame
        f1500, median, std = sigma_clipped_stats( self.rest_flux[use], sigma=3)
        err_f1500 = np.std(self.rest_flux[use])

        f1500 = f1500 * u.erg/u.s/u.cm**2/u.AA
        err_f1500 = err_f1500 * u.erg/u.s/u.cm**2/u.AA
        lumdist = cosmo.luminosity_distance(z).to(u.parsec)
        
        ### flux_density at a distance of 10 parsec (definition of UV mag)
        fnu_1500 = (f1500 * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                            equivalencies=u.spectral_density(1500*u.AA))
        fnu_1500_u = ((f1500 + err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                            equivalencies=u.spectral_density(1500*u.AA))
        fnu_1500_l = ((f1500 - err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                            equivalencies=u.spectral_density(1500*u.AA))
        
        ### UV absolute mag calculation
        mab_1500 = (-2.5 * np.log10(fnu_1500.value)) - 48.60
        mab_1500_u = abs((-2.5 * np.log10(fnu_1500_u.value) - 48.60) - mab_1500)
        mab_1500_l = abs((-2.5 * np.log10(fnu_1500_l.value) - 48.60) - mab_1500)

        ### UV observed mag calculation
        mab_1500_obs = mab_1500 + 5 * np.log10(lumdist/(10.*u.parsec))
        mab_1500_obs_u = mab_1500_u + 5 *np.log10 (lumdist/(10.*u.parsec))
        mab_1500_obs_l = mab_1500 + 5 * np.log10(lumdist/(10.*u.parsec))

        ### return the results
        return(mab_1500, mab_1500_u, mab_1500_l, f1500, err_f1500, mab_1500_obs, mab_1500_obs_u, mab_1500_obs_l)
    
       
      
   