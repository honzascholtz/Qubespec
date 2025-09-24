#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:11:38 2017

@author: jscholtz
"""

#importing modules
from ast import Raise
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.optimize import curve_fit
import glob

from astropy.coordinates import SkyCoord

from astropy.modeling.powerlaws import PowerLaw1D


nan= float('nan')

pi= np.pi
e= np.e

plt.close('all')
c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

arrow = u'$\u2193$'

from ..Models import OIII_models as O_models

def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)
    y= k* e**expo
    return y

def plotting_OIII(res, ax, errors=False, template=0, residual='none',mode='restframe',axres=None):
    sol = res.props
    popt = sol['popt']
    keys = list(sol.keys())
    z = sol['z'][0]
    wave = res.wave
    fluxs = res.fluxs
    error = res.error

    wv_rest = wave/(1+z)*1e4
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]

    if mode=='restframe':
        fit_loc = np.where((wv_rest>4700.)&(wv_rest<5200.))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700)&(wv_rst_sc<5200))[0]
    elif mode=='observedframe':
        wv_rest = wave
        fit_loc = np.where((wv_rest>(4700.*(1+z)/1e4))&(wv_rest<(5200.*(1+z)/1e4)))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700*(1+z)/1e4)&(wv_rst_sc<5200*(1+z)/1e4))[0]
    else:
        raise ValueError('mode must be restframe or observed')
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    
    if mode=='restframe':
        ax.plot(res.wave/(1+z)*1e4, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  
    
    elif mode=='observedframe':
        ax.plot(res.wave, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(res.wave[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  
    

    y_tot = res.yeval[fit_loc]
    y_tot_rs = res.yeval[fit_loc_sc]

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')

    flt = np.where((wv_rest[fit_loc]>4900)&(wv_rest[fit_loc]<5100))[0]

    ax.set_ylim(-0.1*np.nanmax(y_tot[flt]), np.nanmax(y_tot[flt])*1.1)
    ax.tick_params(direction='in')
    if mode=='restframe':
        ax.set_xlim(4700,5050 )
    elif mode=='observedframe':
        ax.set_xlim(4700*(1+z)/1e4,5050*(1+z)/1e4)
    else:
        raise ValueError('mode must be restframe or observedframe')

    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4

    fwhm = sol['Nar_fwhm'][0]

    ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['OIII_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIII_peak'][0],OIIIr, fwhm) \
                 ,color= 'green', linestyle ='dashed')

    if 'Hbeta_peak' in keys:
        Hbeta= 4862.6*(1+z)/1e4    
        fwhm = sol['Nar_fwhm'][0]
        ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['Hbeta_peak'][0],Hbeta, fwhm),\
                        color= 'orange', linestyle ='dashed')


    if 'outflow_fwhm' in keys:
        OIIIr = OIIIr+ sol['outflow_vel'][0]/3e5*OIIIr
        OIIIb = OIIIb + sol['outflow_vel'][0]/3e5*OIIIb

        fwhm = sol['outflow_fwhm'][0]

        ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['OIII_out_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIII_out_peak'][0],OIIIr, fwhm),\
                     color= 'blue', linestyle ='dashed')

        if 'Hbeta_out_peak' in keys:
            Hbeta_wv = 4862.6*(1+z)/1e4
            Hbeta_out_wv = Hbeta_wv + sol['outflow_vel'][0]/3e5*Hbeta_wv

            fwhm = sol['outflow_fwhm'][0]

            ax.plot(wv_rest[fit_loc] ,  gauss(wave[fit_loc], sol['Hbeta_out_peak'][0],Hbeta_out_wv, fwhm),\
                        color= 'blue', linestyle ='dashed')

    if 'Fe_peak' in keys:
        ax.plot(wv_rest[fit_loc], PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0],OIIIr, alpha=sol['cont_grad'][0]), linestyle='dashed', color='limegreen')

        if template=='BG92':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_BG92(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )

        if template=='Tsuzuki':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_Tsuzuki(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )

        if template=='Veron':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_Veron(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )


    if 'Hb_nar_peak' in keys:
        Hbeta= 4862.6*(1+z)/1e4

        Hbeta_NLR = gauss(wave[fit_loc], sol['Hb_nar_peak'][0],Hbeta, sol['Nar_fwhm'][0])
        Hbeta_NLR2= gauss(wave[fit_loc], sol['Hb_out_peak'][0],Hbeta+ sol['outflow_vel'][0]/3e5*Hbeta, sol['outflow_fwhm'][0])


        ax.plot(wv_rest[fit_loc] , Hbeta_NLR , color= 'orange', linestyle ='dotted')
        ax.plot(wv_rest[fit_loc] , Hbeta_NLR2, color= 'orange', linestyle ='dotted')
        #ax.plot(wv_rest[fit_loc] , Hbeta_NLR+Hbeta_NLR2, color= 'orange', linestyle ='dotted')

    if ('zBLR' in keys):
        if 'BLR_alp1' in keys:
            from ..Models.QSO_models import BKPLG
            from astropy.modeling.powerlaws import BrokenPowerLaw1D
            from astropy.convolution import Gaussian1DKernel
            from astropy.convolution import convolve

            Hbeta= 4862.6*(1+z)/1e4

            Hbeta_BLR_wv = 4862.6*(1+sol['zBLR'][0])/1e4
            Hbeta_BLR = BKPLG(wave[fit_loc], sol['BLR_peak'][0], Hbeta_BLR_wv, sol['BLR_sig'][0], sol['BLR_alp1'][0], sol['BLR_alp2'][0])
            ax.plot(wv_rest[fit_loc] , Hbeta_BLR , color= 'orange', linestyle ='dashed')
        else:
            Hbeta= 4862.6*(1+z)/1e4
            Hbeta_BLR_wv = 4862.6*(1+sol['zBLR'][0])/1e4

            Hbeta_BLR = gauss(wave[fit_loc], sol['BLR_Hbeta_peak'][0],Hbeta, sol['BLR_fwhm'][0])

            ax.plot(wv_rest[fit_loc] , Hbeta_BLR , color= 'orange', linestyle ='dashed')
            #ax.plot(wv_rest[fit_loc] , Hbeta_BLR2, color= 'orange', linestyle ='dashed')

    if residual !='none':
        resid_OIII = flux[fit_loc_sc]-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))

        axres.plot(wv_rst_sc[fit_loc_sc],resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-3*RMS_OIII, 3*RMS_OIII) ## the /3 scales to the ratio
        axres.hlines(0, 4600,5600, color='black', linestyle='dashed')
        if residual=='rms':
            axres.fill_between(wv_rst_sc[fit_loc_sc], RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2, step='mid')
        elif residual=='error':
            axres.fill_between(wv_rst_sc[fit_loc_sc],resid_OIII-error[fit_loc_sc],resid_OIII+error[fit_loc_sc], alpha=0.3, color='k', step='mid')



def plotting_Halpha( res, ax, errors=False, residual='none', axres=None, mode='restframe'):
    sol = res.props
    popt = sol['popt']
    z = popt[0]

    wave = res.wave
    fluxs = res.fluxs
    error = res.error
    keys = list(sol.keys())

    wv_rest = wave/(1+z)*1e4
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]

    if mode=='restframe':
        fit_loc = np.where((wv_rest>4700.)&(wv_rest<5200.))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700)&(wv_rst_sc<5200))[0]
    elif mode=='observedframe':
        wv_rest = wave
        fit_loc = np.where((wv_rest>(4700.*(1+z)/1e4))&(wv_rest<(5200.*(1+z)/1e4)))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700*(1+z)/1e4)&(wv_rst_sc<5200*(1+z)/1e4))[0]
    else:
        raise ValueError('mode must be restframe or observed')
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    
    if mode=='restframe':
        ax.plot(res.wave/(1+z)*1e4, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  
    
    elif mode=='observedframe':
        ax.plot(res.wave, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(res.wave[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  
    


    y_tot = res.yeval[fit_loc]

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')

    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    if mode=='restframe':
        ax.set_xlim(6564.52-150,6564.52+150 )
    elif mode=='observedframe':
        ax.set_xlim((6564.52-150)*(1+z)/1e4, (6564.52+150)*(1+z)/1e4)
    else:
        raise ValueError('mode must be restframe or observedframe')
    ax.tick_params(direction='in')

    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4

    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4

    ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['Hal_peak'][0], Hal_wv, sol['Nar_fwhm'][0]), \
            color='orange', linestyle='dashed')
    if 'NII_peak' in list(sol.keys()):
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['NII_peak'][0], NII_r, sol['Nar_fwhm'][0]), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['NII_peak'][0]/3, NII_b, sol['Nar_fwhm'][0]), \
                color='darkgreen', linestyle='dashed')

    if 'SIIr_peak' in keys:
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['SIIr_peak'][0], SII_r, sol['Nar_fwhm'][0]), \
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['SIIb_peak'][0], SII_b, sol['Nar_fwhm'][0]), \
                color='darkblue', linestyle='dashed')

    if 'zBLR' in keys:
        BLR_wv = 6564.52*(1+sol['zBLR'][0])/1e4

        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['BLR_Hal_peak'][0], BLR_wv, sol['BLR_fwhm'][0]), \
                color='darkorange', linestyle='dashed')

    if 'Hal_out_peak' in keys:
        out_vel_hal = sol['outflow_fwhm'][0]
        out_vel_niir = sol['outflow_fwhm'][0]
        out_vel_niib = sol['outflow_fwhm'][0]

        Hal_wv_vel = 6564.52*(1+z)/1e4+ sol['outflow_vel'][0]/3e5*Hal_wv
        NII_r_vel = 6585.27*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*Hal_wv
        NII_b_vel = 6549.86*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*Hal_wv

        Hal_out = gauss(wave[fit_loc],sol['Hal_out_peak'][0], Hal_wv_vel, out_vel_hal)
        NII_out_r = gauss(wave[fit_loc], sol['NII_out_peak'][0], NII_r_vel, out_vel_niir)
        NII_out_b = gauss(wave[fit_loc], sol['NII_out_peak'][0]/3, NII_b_vel, out_vel_niib)

        ax.plot(wv_rest[fit_loc], Hal_out, color='magenta', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], NII_out_r, color='magenta', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], NII_out_b, color='magenta', linestyle='dashed')

    if 'BLR_alp1' in keys:
        from ..Models.QSO_models import BKPLG
        from astropy.modeling.powerlaws import BrokenPowerLaw1D
        from astropy.convolution import Gaussian1DKernel
        from astropy.convolution import convolve

        Ha_BLR_vel_wv = 6563*(1+sol['zBLR'][0])/1e4

        Ha_BLR_wv = Hal_wv+Ha_BLR_vel_wv
        Ha_BLR = BKPLG(wave[fit_loc], sol['BLR_Hal_peak'][0], Ha_BLR_wv, sol['BLR_sig'][0], sol['BLR_alp1'][0], sol['BLR_alp2'][0])
        ax.plot(wv_rest[fit_loc] , Ha_BLR , color= 'magenta', linestyle ='dashed')

    if residual !='none':
        resid_OIII = flux[fit_loc_sc]-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))

        axres.plot(wv_rst_sc[fit_loc_sc],resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-2*RMS_OIII, 2*RMS_OIII) ## the /3 scales to the ratio
        if residual=='rms':
            axres.fill_between(wv_rst_sc[fit_loc_sc], RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2, step='mid')
        elif residual=='error':
            axres.fill_between(wv_rst_sc[fit_loc_sc],resid_OIII-error[fit_loc_sc],resid_OIII+error[fit_loc_sc], alpha=0.3, color='k', step='mid')


def plotting_Halpha_OIII(res, ax,errors=False, residual='none', axres=None,mode='restframe', template=0):
    sol = res.props
    popt = sol['popt']
    keys = list(sol.keys())
    z = popt[0]
    error =res.error

    wave = res.wave
    fluxs = res.fluxs

    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>100.)&(wv_rest<16000.))[0]

    wv_rest = wave/(1+z)*1e4
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]

    if mode=='restframe':
        fit_loc = np.where((wv_rest>4700.)&(wv_rest<5200.))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700)&(wv_rst_sc<5200))[0]
    elif mode=='observedframe':
        wv_rest = wave
        fit_loc = np.where((wv_rest>(4700.*(1+z)/1e4))&(wv_rest<(5200.*(1+z)/1e4)))[0]
        fit_loc_sc = np.where((wv_rst_sc>4700*(1+z)/1e4)&(wv_rst_sc<5200*(1+z)/1e4))[0]
    else:
        raise ValueError('mode must be restframe or observed')
    
    flux = fluxs.data[np.invert(fluxs.mask)]
    
    if mode=='restframe':
        ax.plot(res.wave/(1+z)*1e4, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  
    
    elif mode=='observedframe':
        ax.plot(res.wave, res.flux, drawstyle='steps-mid', label='data')
        ax.plot(res.wave[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)  

    #if len(error) !=1:
    #    ax.fill_between(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc]-error[fit_loc_sc],flux[fit_loc_sc]+error[fit_loc_sc], alpha=0.3, color='k')
    #    ax.fill_between(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc]-error[fit_loc_sc],flux[fit_loc_sc]+error[fit_loc_sc], alpha=0.3, color='k')
    y_tot = res.yeval[fit_loc]
    y_tot_rs = res.yeval[np.invert(fluxs.mask)][fit_loc_sc]

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')


    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    ax.tick_params(direction='in')


    Hal_wv = 6564.52*(1+z)/1e4
    NII_r = 6585.27*(1+z)/1e4
    NII_b = 6549.86*(1+z)/1e4
    SII_r = 6732.67*(1+z)/1e4
    SII_b = 6718.29*(1+z)/1e4
    ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['Hal_peak'][0], Hal_wv, sol['Nar_fwhm'][0]), \
            color='orange', linestyle='dashed')

    if 'NII_peak' in list(sol.keys()):
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['NII_peak'][0], NII_r, sol['Nar_fwhm'][0]), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['NII_peak'][0]/3, NII_b, sol['Nar_fwhm'][0]), \
                color='darkgreen', linestyle='dashed')

    if 'SIIr_peak' in list(sol.keys()):
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['SIIr_peak'][0], SII_r, sol['Nar_fwhm'][0]), \
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], sol['SIIb_peak'][0], SII_b, sol['Nar_fwhm'][0]), \
                color='darkblue', linestyle='dashed')

    OIII_r = OIIIr = 5008.24*(1+z)/1e4
    OIII_b = OIIIb = 4960.3*(1+z)/1e4

    fwhm = sol['Nar_fwhm'][0]
    try:
        ax.plot(wv_rest[fit_loc] ,    gauss(wave[fit_loc], sol['OIII_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIII_peak'][0],OIIIr, fwhm) \
            ,color= 'green', linestyle ='dashed')
    except:
        ax.plot(wv_rest[fit_loc] ,    gauss(wave[fit_loc], sol['OIII_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIII_peak'][0],OIIIr, fwhm) \
                ,color= 'green', linestyle ='dashed')

    Hbeta= 4862.6*(1+z)/1e4
    fwhm = sol['Nar_fwhm'][0]
    ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['Hbeta_peak'][0],Hbeta, fwhm),\
            color= 'orange', linestyle ='dashed')
    
    if 'Fe_peak' in keys:
        ax.plot(wv_rest[fit_loc], PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0],OIIIr, alpha=sol['cont_grad'][0]), linestyle='dashed', color='limegreen')

        if template=='BG92':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_BG92(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )

        if template=='Tsuzuki':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_Tsuzuki(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )

        if template=='Veron':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*O_models.Fem.FeII_Veron(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )

    if 'OI_peak' in list(sol.keys()):
        OI = 6302.0*(1+z)/1e4
        fwhm = sol['Nar_fwhm'][0]
        ax.plot(wv_rest[fit_loc] , gauss(wave[fit_loc], sol['OI_peak'][0],OI, fwhm), color='green', linestyle='dashed')

    if 'outflow_vel' in list(sol.keys()):
        OI = 6302.0*(1+z)/1e4

        Out_vel_hal = sol['outflow_fwhm'][0]
        Out_vel_niir = sol['outflow_fwhm'][0]
        Out_vel_niib = sol['outflow_fwhm'][0]
        Out_vel_oiiir = sol['outflow_fwhm'][0]
        Out_vel_oiiib = sol['outflow_fwhm'][0]
        Out_vel_oi = sol['outflow_fwhm'][0]
        Out_vel_hbe = sol['outflow_fwhm'][0]

        Hal_wv = 6564.52*(1+z)/1e4   + sol['outflow_vel'][0]/3e5*Hal_wv
        NII_r = 6585.27*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*NII_r
        NII_b = 6549.86*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*NII_b
        OIII_r = 5008.24*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*OIII_r
        OIII_b = 4960.3*(1+z)/1e4  + sol['outflow_vel'][0]/3e5*OIII_b
        OI = 6302.0*(1+z)/1e4 + sol['outflow_vel'][0]/3e5*OI
        Hbeta = 4862.6*(1+z)/1e4  + sol['outflow_vel'][0]/3e5* Hbeta
        SII_r = 6732.67*(1+z)/1e4  + sol['outflow_vel'][0]/3e5*SII_r
        SII_b = 6718.29*(1+z)/1e4   + sol['outflow_vel'][0]/3e5*SII_b


        Outflow =  gauss(wave[fit_loc], sol['Hal_out_peak'][0], Hal_wv, Out_vel_hal)  + \
            gauss(wave[fit_loc],  sol['NII_out_peak'][0], NII_r, Out_vel_niir)+  gauss(wave[fit_loc],  sol['NII_out_peak'][0]/3, NII_b, Out_vel_niib) + \
            gauss(wave[fit_loc],  sol['OIII_out_peak'][0], OIII_r, Out_vel_oiiir) + gauss(wave[fit_loc],  sol['OIII_out_peak'][0]/3, OIII_b, Out_vel_oiiib)+\
            gauss(wave[fit_loc],  sol['Hbeta_out_peak'][0], Hbeta, Out_vel_hbe )

        if 'OI_peak' in list(sol.keys()):
            Outflow+= gauss(wave[fit_loc], sol['OI_out_peak'][0], OI, Out_vel_oi)

        ax.plot(wv_rest[fit_loc] , Outflow, color='magenta', linestyle='dashed')


    if 'BLR_fwhm' in list(sol.keys()):
        Hal_wv = 6564.52*(1+z)/1e4
        Hbe_wv = 4862.6*(1+z)/1e4

        BLR_sig_hal = sol['BLR_fwhm'][0]
        BLR_sig_hbe = sol['BLR_fwhm'][0]

        BLR_wv_hal = 6564.52*(1+sol['zBLR'][0])/1e4
        BLR_wv_hbe = 4862.6*(1+sol['zBLR'][0])/1e4

        Hal_blr = gauss(wave[fit_loc], sol['BLR_Hal_peak'][0], BLR_wv_hal, BLR_sig_hal)
        Hbe_blr = gauss(wave[fit_loc], sol['BLR_Hbeta_peak'][0], BLR_wv_hbe, BLR_sig_hbe)

        ax.plot(wv_rest[fit_loc], Hal_blr+Hbe_blr, linestyle='dashed',color='lightblue')

def plotting_general(wave, fluxs, ax, sol,fitted_model,error=np.array([1]), residual='none', axres='none'):
    
    popt = sol['popt']
    z = popt[0]
    
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>100.)&(wv_rest<16000.))[0]
    
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)

    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>100)&(wv_rst_sc<16000))[0]   
    
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')
    
    y_tot = fitted_model(wave[fit_loc], *popt)
    y_tot_rs = fitted_model(wv_rst_sc[fit_loc_sc]*(1+z)/1e4, *popt)

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')
    

    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    ax.tick_params(direction='in')
    
        
    if axres !='none':
        resid_OIII = flux[fit_loc_sc]-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))
        
        axres.plot(wv_rst_sc[fit_loc_sc],resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-2*RMS_OIII, 2*RMS_OIII) ## the /3 scales to the ratio
        if residual=='rms':
            axres.fill_between(wv_rst_sc[fit_loc_sc], RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2)
        elif residual=='error':
            axres.fill_between(wv_rst_sc[fit_loc_sc],resid_OIII-error[fit_loc_sc],resid_OIII+error[fit_loc_sc], alpha=0.3, color='k')


def plotting_optical(res, ax, error=np.array([1]), template=0, residual='none',axres=None):
    sol = res.props
    popt = sol['popt']
    keys = list(sol.keys())
    z = popt[0]
    wave = res.wave
    fluxs = res.fluxs
    error = res.error
    wv_rest = wave/(1+z)*1e4

    ax.plot(wv_rest, fluxs.data, color='grey', drawstyle='steps-mid', alpha=0.2)

    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]


    ax.plot(wv_rst_sc,flux, drawstyle='steps-mid', label='data')
    if len(error) !=1:
        ax.fill_between(wv_rst_sc,flux-error,flux+error, alpha=0.3, color='k')

    y_tot = res.yeval
    y_tot_rs = res.yeval[np.invert(fluxs.mask)]

    ax.plot(wv_rest, y_tot, 'r--')

    ax.set_ylim(-0.1*np.nanmax(y_tot), np.nanmax(y_tot)*1.1)
    ax.tick_params(direction='in')
    ax.set_xlim(4700,5050 )

    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = 4960.3*(1+z)/1e4

    fwhm = sol['Nar_fwhm'][0]

    ax.plot(wv_rest, gauss(wave, sol['OIII_peak'][0]/3,OIIIb, fwhm)+ gauss(wave, sol['OIII_peak'][0],OIIIr, fwhm) \
                 ,color= 'green', linestyle ='dashed')

    if 'Hbeta_peak' in keys:
        Hbeta= 4862.6*(1+z)/1e4    
        fwhm = sol['Nar_fwhm'][0]
        ax.plot(wv_rest ,   gauss(wave, sol['Hbeta_peak'][0],Hbeta, fwhm),\
                        color= 'orange', linestyle ='dashed')


    if 'outflow_fwhm' in keys:
        OIIIr = OIIIr+ sol['outflow_vel'][0]/3e5*OIIIr
        OIIIb = OIIIb + sol['outflow_vel'][0]/3e5*OIIIb

        fwhm = sol['outflow_fwhm'][0]

        ax.plot(wv_rest,   gauss(wave, sol['OIII_out_peak'][0]/3,OIIIb, fwhm)+ gauss(wave, sol['OIII_out_peak'][0],OIIIr, fwhm),\
                     color= 'blue', linestyle ='dashed')

        if 'Hbeta_out_peak' in keys:
            Hbeta_wv = 4862.6*(1+z)/1e4
            Hbeta_out_wv = Hbeta_wv + sol['outflow_vel'][0]/3e5*Hbeta_wv

            fwhm = sol['outflow_fwhm'][0]

            ax.plot(wv_rest,  gauss(wave, sol['Hbeta_out_peak'][0],Hbeta_out_wv, fwhm),\
                        color= 'blue', linestyle ='dashed')

    if residual !='none':
        resid_OIII = flux-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))

        axres.plot(wv_rst_sc,resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-3*RMS_OIII, 3*RMS_OIII) ## the /3 scales to the ratio
        axres.hlines(0, 4600,5600, color='black', linestyle='dashed')
        if residual=='rms':
            axres.fill_between(wv_rst_sc, RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2)
        elif residual=='error':
            axres.fill_between(wv_rst_sc,resid_OIII-error,resid_OIII+error, alpha=0.3, color='k')



def overide_axes_labels(fig,ax,lims,showx=1,showy=1,labelx=1,labely=1,color='k',fewer_x=0,pruney=0,prunex=0,tick_color='k', tickin=0, labelsize=12, white=0):
    #over rides axis labels generated by wcsaxes in order to put relative coordinates in

    #fig is the figure being plotted
    #ax = axis to plot on
    #lims is what you want on the axis (xmin,xmax,ymin,ymax) OR (img_wcs,im_hdr)
    #labelx / label y = bool, true if want tick and axis labels on the x / y axis

    #plt.ion()

    if len(lims)==2:
        img_wcs=lims[0]
        hdr=lims[1]
        o=np.array(img_wcs.all_pix2world(1,1,1))
        o=SkyCoord(o[0],o[1], unit="deg")
        p1=np.array(img_wcs.all_pix2world(hdr['NAXIS1'],1,1))
        p1=SkyCoord(p1[0],p1[1], unit="deg")
        p2=np.array(img_wcs.all_pix2world(1,hdr['NAXIS2'],1))
        p2=SkyCoord(p2[0],p2[1], unit="deg")

        arcsec_size= np.array([o.separation(p2).arcsec,o.separation(p1).arcsec,])/2.

        lims=[-arcsec_size[1],arcsec_size[1],-arcsec_size[0],arcsec_size[0]]

    lon = ax.coords[0]
    lat = ax.coords[1]

    lon.set_ticks_visible(False)
    lon.set_ticklabel_visible(False)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)

    if showx or showy:
        #plt.draw()
        #plt.pause(0.000000001)
        newax = fig.add_axes(ax.get_position(), frameon=False)

        plt.xlim(lims[0],lims[1])
        plt.ylim(lims[2],lims[3])

        newax.xaxis.label.set_color(color)
        newax.tick_params(axis='x', colors=color)
        newax.tick_params(axis='x', color=tick_color)


        newax.yaxis.label.set_color(color)
        newax.tick_params(axis='y', colors=color)
        newax.tick_params(axis='y', color=tick_color)

        if not showx: gca().axes.xaxis.set_ticklabels([])
        if not showy: gca().axes.axes.yaxis.set_ticklabels([])

        if labely:    plt.ylabel('arcsec', fontsize=labelsize)
        if labelx:    plt.xlabel('arcsec', fontsize=labelsize)


        if fewer_x: newax.set_xticks(newax.get_xticks()[::2])

        #newax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        #newax.xaxis.set_major_locator(MaxNLocator(prune='both'))
        if pruney: newax.set_yticks(newax.get_yticks()[1:-1])
        if prunex: newax.set_xticks(newax.get_xticks()[1:-1])

    if tickin==1:
        newax.tick_params(axis='y', direction='in', color='white')
        newax.tick_params(axis='x', direction='in', color='white')

    if tickin==2:
        newax.tick_params(axis='y', direction='in', color='black')
        newax.tick_params(axis='x', direction='in', color='black')

    if white==1:
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
        newax.xaxis.label.set_color('white')
        newax.yaxis.label.set_color('white')

        #plt.ylabel('arcsec', fontsize=labelsize, color='w')
        #plt.xlabel('arcsec', fontsize=labelsize, color='w')

        plt.setp(newax.get_yticklabels(), color="white")
        plt.setp(newax.get_xticklabels(), color="white")

        #newax.xaxis.set_major_locator(MultipleLocator(4))
        #newax.yaxis.set_major_locator(MultipleLocator(4))

        #newax.xaxis.set_minor_locator(MultipleLocator(3))
        #newax.yaxis.set_minor_locator(MultipleLocator(3))

def OIII_map_plotting(ID, path, fwhmrange = [300,500], velrange=[-400,100], flux_max=0):
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    with pyfits.open(path, memmap=False) as hdulist:
        
        #map_hb = hdulist['Hbeta'].data
        map_oiii = hdulist['OIII'].data
        map_oiii_ki= hdulist['OIII_kin'].data
        

        IFU_header = hdulist['PRIMARY'].header

    
    x = int(IFU_header['X_cent']); y= int(IFU_header['Y_cent'])
    f = plt.figure( figsize=(10,10))

    if flux_max==0:
        flx_max = map_oiii[1,y,x]
    else:
        flx_max = flux_max

    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600


    Offsets_low = -np.array([x,y])
    Offsets_hig = np.shape(map_oiii[1])[1:3] -np.array([x,y])

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                     Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix

    ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
    ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
    ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
    ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])

    flx = ax1.imshow(map_oiii[1],vmax=flux_max, origin='lower', extent= lim_sc)
    ax1.set_title('Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')

    #lims =
    #emplot.overide_axes_labels(f, axes[0,0], lims)


    vel = ax2.imshow( map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')


    fw = ax3.imshow( map_oiii_ki[1,:,:],vmin=fwhmrange[0], vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    snr = ax4.imshow(map_oiii[0],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax4.set_title('SNR map')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')



def Plot_results_Halpha_OIII(file, center=[27,27], fwhmrange = [100,500], velrange=[-100,100], flux_max=0, o3offset=0, extent=np.array([0])):
    with pyfits.open(file, memmap=False) as hdulist:
        map_hal = hdulist['Halpha'].data
        map_nii = hdulist['NII'].data
        map_hb = hdulist['Hbeta'].data
        map_oiii = hdulist['OIII'].data
        map_oi = hdulist['OI'].data
        map_siir = hdulist['SIIr'].data
        map_siib = hdulist['SIIb'].data
        map_hal_ki = hdulist['Hal_kin'].data
        #map_oiii_ki= hdulist['OIII_kin'].data
        Av = hdulist['Av'].data

        IFU_header = hdulist['PRIMARY'].header

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    x = int(center[0]); y= int(center[0])

    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600

    Offsets_low =- np.array(center)
    Offsets_hig = np.shape(map_hal)[1:] - np.array(center)

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                    Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix
    print(lim_sc)

    if flux_max==0:
        flx_max = map_hal[1,y,x]
    else:
        flx_max = flux_max

# =============================================================================
#         Plotting Stuff
# =============================================================================
    f,axes = plt.subplots(6,3, figsize=(10,20), sharex=True, sharey=True)
    ax1 = axes[0,0]
    # =============================================================================
    # Halpha SNR
    snr = ax1.imshow(map_hal[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax1.set_title('Hal SNR map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha flux
    ax1 = axes[0,1]
    flx = ax1.imshow(map_hal[1,:,:],vmax=flx_max, origin='lower', extent= lim_sc)
    ax1.set_title('Halpha Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    cax.set_ylabel('Flux (arbitrary units)')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha  velocity
    ax2 = axes[0,2]
    vel = ax2.imshow(map_hal_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Hal Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha fwhm
    ax3 = axes[1,2]
    fw = ax3.imshow(map_hal_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('Hal FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] SNR
    axes[1,0].set_title('[NII] SNR')
    fw= axes[1,0].imshow(map_nii[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,0].set_xlabel('RA offset (arcsecond)')
    axes[1,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] flux
    axes[1,1].set_title('[NII] map')
    fw= axes[1,1].imshow(map_nii[1,:,:] ,vmax=flx_max,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,1].set_xlabel('RA offset (arcsecond)')
    axes[1,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta] SNR
    axes[2,0].set_title('Hbeta SNR')
    fw= axes[2,0].imshow(map_hb[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,0].set_xlabel('RA offset (arcsecond)')
    axes[2,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta flux
    axes[2,1].set_title('Hbeta map')
    fw= axes[2,1].imshow(map_hb[1,:,:] ,vmax=flx_max,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,1].set_xlabel('RA offset (arcsecond)')
    axes[2,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] SNR
    axes[3,0].set_title('[OIII] SNR')
    fw= axes[3,0].imshow(map_oiii[0,:,:],vmin=3, vmax=20,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,0].set_xlabel('RA offset (arcsecond)')
    axes[3,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] flux
    axes[3,1].set_title('[OIII] map')
    fw= axes[3,1].imshow(map_oiii[1,:,:] ,vmax=flx_max,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,1].set_xlabel('RA offset (arcsecond)')
    axes[3,1].set_ylabel('Dec offset (arcsecond)')
    '''
    # =============================================================================
    # [OIII] vel
    ax3= axes[2,2]
    ax3.set_title('[OIII] vel')
    fw = ax3.imshow(map_oiii_ki[0,:,:],vmin=velrange[0]+o3offset,vmax=velrange[1]+o3offset,cmap='coolwarm', origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # [OIII] fwhm
    ax3 = axes[3,2]
    ax3.set_title('[OIII] fwhm')
    fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # [OIII] fwhm
    ax3 = axes[4,2]
    ax3.set_title('[OIII] vel offset')
    fw = ax3.imshow(map_oiii_ki[2,:,:],vmin=0,vmax=150, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    '''
    # =============================================================================
    # OI SNR
    ax3 = axes[4,0]
    ax3.set_title('OI SNR')
    fw = ax3.imshow(map_oi[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # OI flux
    ax3 = axes[4,1]
    ax3.set_title('OI Flux')
    fw = ax3.imshow(map_oi[1,:,:], origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # SII SNR
    ax3 = axes[5,0]
    ax3.set_title('[SII] SNR')
    fw = ax3.imshow(map_siir[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # SII Ratio
    ax3 = axes[5,1]
    ax3.set_title('[SII]r/[SII]b')
    fw = ax3.imshow(map_siir[1,:,:]/map_siib[1,:,:] ,vmin=0.3, vmax=1.5, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    if len(extent) >1:
        axes[0,0].set_xlim(extent[0], extent[1])
        axes[0,0].set_ylim(extent[2], extent[3])


    plt.tight_layout()



from mpl_toolkits.axes_grid1 import make_axes_locatable

def colorbar(f, ax, im, label, fontsize=12):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.0)
    f.colorbar(im, cax=cax, orientation='vertical') 
    cax.set_ylabel(label, fontsize=fontsize)