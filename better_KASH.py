#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:26:39 2022

@author: pwcc62
"""

##import
import Fitting_tools_mcmc as emfit
import glob
import os
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()


from astropy.io import fits
from astropy.wcs import wcs
from astropy.nddata import Cutout2D
import astropy.units as u
from scipy import integrate

from astropy.io import fits as pyfits

#import Fitting_tools_mcmc as emfit
import Plotting_tools_v2 as emplot
from IFU_tools_class import BIC_calc
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import pickle
import corner
import emcee

import pandas as pd
from astropy.table import QTable, Table
from astropy.cosmology import WMAP9 as cosmo
import csv

from astropy import stats


##housekeeping
spectra_dir = '/cosma5/data/durham/dc-esco1/OIII_fitting/Qubespec/Spectra_27_zlim'
out_dir = '/cosma5/data/durham/dc-esco1/OIII_fitting/Qubespec/Plots/'
dict_dir = '/cosma5/data/durham/dc-esco1/OIII_fitting/Qubespec/Dict/'


def spectral_info(filename):#, radiocat='/home/pwcc62/AGN_outflows/Important_Samples/FINAL_NAMED_OPT_27_SPEC.fits'):
    

    with fits.open(filename) as hdu:
        name = filename.split('/')[-1].replace('.fits','')
        plate = name.split('-')[1]
        mjd = name.split('-')[2]
        FIBREID = name.split('-')[3]
        wave = hdu[1].data['loglam']
        #flux = hdu[1].data['flux']
        flux = np.ma.masked_invalid(hdu[1].data['flux'])
        ivar = hdu[1].data['ivar']
        error = hdu[1].data['ivar']
        z = hdu[2].data['z'][0]
        lum_dist = cosmo.luminosity_distance(z).to(u.meter)
        lum_dist = lum_dist/u.m
        wave = 10**wave
        wave = wave/1e4
        flux = flux*10
        error = 1/np.sqrt(error)
        error = error/100
        flux = flux/1000 ## try to help the code
    '''
    with fits.open(radiocat) as hdu2:
        plate_radio = hdu2[1].data['PLATE']
        mjd_radio = hdu2[1].data['MJD']
        fibreid_radio = hdu2[1].data['FIBERID']
        radio_flux = hdu2[1].data['Total_flux']
        
    FIBREID= np.int64(FIBREID)
    radio_flux_source=radio_flux[FIBREID==fibreid_radio]
    #flux = np.ma.array(flux, mask=np.zeros(len(flux)) )
    '''
        
    return wave, flux, error, z#, radio_flux_source

specfiles = glob.glob( os.path.join( spectra_dir, '*.fits' ))



## testing
#specfiles = specfiles[0:2]

name = np.zeros(0)
cont = np.zeros(0)
cont16 = np.zeros(0)
cont84 = np.zeros(0)
cont_grad = np.zeros(0)
cont_grad16 = np.zeros(0)
cont_grad84 = np.zeros(0)
OIIIn_peak = np.zeros(0)
OIIIn_peak16 = np.zeros(0)
OIIIn_peak84 = np.zeros(0)
OIIIn_fwhm = np.zeros(0)
OIIIn_fwhm16 = np.zeros(0)
OIIIn_fwhm84 = np.zeros(0)
Hbeta_peak = np.zeros(0)
Hbeta_peak16 = np.zeros(0)
Hbeta_peak84 = np.zeros(0)
Hbeta_fwhm = np.zeros(0)
Hbeta_fwhm16 = np.zeros(0)
Hbeta_fwhm84 = np.zeros(0)
Fe_peak = np.zeros(0)
Fe_peak16 = np.zeros(0)
Fe_peak84 = np.zeros(0)
Fe_fwhm = np.zeros(0)
Fe_fwhm16 = np.zeros(0)
Fe_fwhm84 = np.zeros(0)
templ = np.zeros(0)
out_flows = np.zeros(0)
H_beta = np.zeros(0)
BIC = np.zeros(0)
rms_OIII = np.zeros(0)

#specfiles = specfiles[2:3]
outflows = [0,1]
#templates = [0]
templates = [0, 'Veron', 'BG92', 'Tsuzuki']
Hbeta_duals = [0,1]
##outflows = [0,1]
##templates = ['None', 'BG92','Veron', 'Tsuzuki']
for specfile in specfiles:
    #specfile = specfiles[2]
    for template in templates:
        for Hbeta_dual in Hbeta_duals:
            for outflow in outflows:
                
                print(specfile)

    
                out_filestem = specfile.replace('.fits','')
                out_filestem = out_filestem.split('/')[-1]
                out_filestem = out_filestem + '_' + str(outflow) + '_' + str(template) + '_' + str(Hbeta_dual)
                wave, flux, error, z= spectral_info(specfile)
                
                    
                if Hbeta_dual == 1:
                    outflow = 1
                else:
                    outflow=outflow

                    
                flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=outflow, template=template, Hbeta_dual=Hbeta_dual, progress=True)
                prop_out_no = emfit.prop_calc(flat_samples_sig)
                chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_out_no, 'OIII', template=template)
                
                
                with open(dict_dir + out_filestem + '_flat_sample.pkl','wb') as f:
                    pickle.dump(flat_samples_sig, f)
                    
                with open(dict_dir + out_filestem + '_prop_out.pkl', 'wb') as f:
                    pickle.dump(prop_out_no, f)
                    
                with open(dict_dir + out_filestem + '_fitted_model.pkl', 'wb') as f:
                    pickle.dump(fitted_model_sig, f)
                
                
                #with open('flat_sample_sig.pkl','rb') as f:
                 #   loaded_dict = pickle.load(f)
                
                cont = np.append( cont, prop_out_no['cont'][0])
                cont16 = np.append( cont16, prop_out_no['cont'][1])
                cont84 = np.append( cont84, prop_out_no['cont'][2])
                cont_grad = np.append( cont_grad, prop_out_no['cont_grad'][0])
                cont_grad16 = np.append( cont_grad16, prop_out_no['cont_grad'][1])
                cont_grad84 = np.append( cont_grad84, prop_out_no['cont_grad'][2])
                OIIIn_peak = np.append( OIIIn_peak, prop_out_no['OIIIn_peak'][0])
                OIIIn_peak16 = np.append( OIIIn_peak16, prop_out_no['OIIIn_peak'][1])
                OIIIn_peak84 = np.append( OIIIn_peak84, prop_out_no['OIIIn_peak'][2])
                OIIIn_fwhm = np.append( OIIIn_fwhm, prop_out_no['OIIIn_fwhm'][0])
                OIIIn_fwhm16 = np.append( OIIIn_fwhm16, prop_out_no['OIIIn_fwhm'][1])
                OIIIn_fwhm84 = np.append( OIIIn_fwhm84, prop_out_no['OIIIn_fwhm'][2])
                Hbeta_peak = np.append(Hbeta_peak, prop_out_no['Hbeta_peak'][0])
                Hbeta_peak16 = np.append(Hbeta_peak16, prop_out_no['Hbeta_peak'][1])
                Hbeta_peak84 = np.append(Hbeta_peak84, prop_out_no['Hbeta_peak'][2])
                Hbeta_fwhm = np.append(Hbeta_fwhm, prop_out_no['Hbeta_fwhm'][0])
                Hbeta_fwhm16 = np.append(Hbeta_fwhm16, prop_out_no['Hbeta_fwhm'][1])
                Hbeta_fwhm84 = np.append(Hbeta_fwhm84, prop_out_no['Hbeta_fwhm'][2])
                out_flows = np.append(out_flows, outflow)
                templ = np.append (templ, template)
                BIC = np.append(BIC, BICS)
                H_beta = np.append(H_beta, Hbeta_dual)
                temp_name = specfile.split('/')[-1]
                temp_name= temp_name.replace('.fits','')
                name = np.append(name, temp_name)
                
                if template != 0:
                    Fe_peak = np.append(Fe_peak, prop_out_no['Fe_peak'][0])
                    Fe_peak16 = np.append(Fe_peak16, prop_out_no['Fe_peak'][1])
                    Fe_peak84 = np.append(Fe_peak84, prop_out_no['Fe_peak'][2])
                    Fe_fwhm = np.append(Fe_fwhm, prop_out_no['Fe_fwhm'][0])
                    Fe_fwhm16 = np.append(Fe_fwhm16, prop_out_no['Fe_fwhm'][1])
                    Fe_fwhm84 = np.append(Fe_fwhm84, prop_out_no['Fe_fwhm'][2])
                else:
                    Fe_peak = np.append(Fe_peak, np.nan)
                    Fe_peak16 = np.append(Fe_peak16, np.nan)
                    Fe_peak84 = np.append(Fe_peak84, np.nan)
                    Fe_fwhm = np.append(Fe_fwhm, np.nan)
                    Fe_fwhm16 = np.append(Fe_fwhm16, np.nan)
                    Fe_fwhm84 = np.append(Fe_fwhm84, np.nan)
                    
            
               
                
                print(outflow, template, Hbeta_dual)
                a, ax1 = plt.subplots(1)
                y_tot, wv_loc, plot_loc = emplot.plotting_OIII(wave, flux, ax1, prop_out_no , fitted_model_sig, template=template)
                #plt.show()
                plt.savefig(out_dir + out_filestem + '.png')

                #RESIDUALS
                #wv_rest = wave/(1+z)*1e4 
                #wv_resid = wv_rest[plot_loc]
                #y_tot_new = y_tot - min(y_tot)
                #flux_new = flux[plot_loc] - min(y_tot)
                #residuals = y_tot_new - flux_new
                #b, ax2 = plt.subplots(1)
                #sigma = np.std(residuals)
                #rms = np.sqrt(np.mean(residuals**2))
                #print(rms)
                #print(sigma)
                
                #OIII = np.where((wv_resid>4980)&(wv_resid<5030))[0] ## JUST 5007
                #resid_OIII = residuals[OIII]
                #rest_OIII = wv_resid[OIII]
                #sigma_OIII = np.std(resid_OIII)
                #RMS_OIII = np.sqrt(np.mean(resid_OIII**2))
                #ax2.plot(rest_OIII, resid_OIII, color='k')
                #ax2.plot(wv_resid, residuals, color='k')
                #ax2.fill_between(wv_resid, rms, -rms, facecolor='red', alpha=0.5)
                #plt.axvline(x=4980, color='green')
                #plt.axvline(x=5030 ,color='green')
                #print(sigma_OIII)
                #print(rms_OIII)
                #rms_OIII = np.append(rms_OIII, RMS_OIII)
                
                #ax2.plot(wv_resid, residuals, color='k')
                #ax2.fill_between(wv_resid, rms, -rms, facecolor='red', alpha=0.5)
                #ax2.fill_between(wv_resid, rms, -rms, facecolor='red', alpha=0.5)
                #ax2.fill_between(wv_resid, rms*2, -rms*2, facecolor='green', alpha=0.3)
                #ax2.fill_between(wv_resid, rms*3, -rms*3, facecolor='blue', alpha=0.3)

                #ax2.fill_between(wv_resid, sigma*2, -sigma*2, facecolor='green', alpha=0.3)
                #ax2.fill_between(wv_resid, sigma*3, -sigma*3, facecolor='blue', alpha=0.3)
                #ax2.fill_between(wv_resid, sigma*4, -sigma*4, facecolor='orange', alpha=0.3)
                #ax2.fill_between(wv_resid, sigma*5, -sigma*5, facecolor='yellow', alpha=0.3)
                #plt.xlim(4980,5030)
                #plt.savefig(out_dir + 'rms' + '_' +  out_filestem + '.png')
                #plt.show()
                
   

## make a table write to file
t = Table()
        
t['cont'] = cont
t['specfile'] = name
t['cont16'] = cont16
t['cont84'] = cont84
t['cont_grad'] = cont_grad
t['cont_grad16'] = cont_grad16
t['cont_grad84'] = cont_grad84
t['OIIIn_peak'] = OIIIn_peak
t['OIIIn_peak16'] = OIIIn_peak16
t['OIIIn_peak84'] = OIIIn_peak84
t['OIIIn_fwhm'] = OIIIn_fwhm
t['OIIIn_fwhm16'] = OIIIn_fwhm16
t['OIIIn_fwhm84'] = OIIIn_fwhm84
t['Hbeta_peak'] = Hbeta_peak
t['Hbeta_peak16'] = Hbeta_peak16
t['Hbeta_peak84'] = Hbeta_peak84
t['Hbeta_fwhm'] = Hbeta_fwhm
t['Hbeta_fwhm16'] = Hbeta_fwhm16
t['Hbeta_fwhm84'] = Hbeta_fwhm84
t['BIC'] = BIC
t['rms_OIII'] = rms_OIII
t['template'] = templ
t['outflow'] = out_flows
t['Hbeta_dual'] = H_beta

if template != 'None':
    t['Fe_peak'] = Fe_peak
    t['Fe_peak16'] = Fe_peak16
    t['Fe_peak84'] = Fe_peak84
    t['Fe_fwhm'] = Fe_fwhm
    t['Fe_fwhm16'] = Fe_fwhm16
    t['Fe_fwhm84'] = Fe_fwhm84
else:
    t['Fe_peak'] = np.nan
    t['Fe_peak16'] = np.nan
    t['Fe_peak84'] = np.nan
    t['Fe_fwhm'] = np.nan
    t['Fe_fwhm16'] = np.nan
    t['Fe_fwhm84'] = np.nan

tablename = 'fitting_results.fits'
   # tablename = 'results' + '_' + mjd + '_' + FIBREID + '_' + str(outflow) + '_' + template + '.fits'
t.write(tablename, format='fits', overwrite=True)


    
    
    
    
    



