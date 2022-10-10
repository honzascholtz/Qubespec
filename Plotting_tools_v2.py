#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:11:38 2017

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
import glob

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

PATH='/Users/jansen/Google Drive/Astro/'
from astropy.modeling.powerlaws import PowerLaw1D

import Graph_setup as gst 
fsz = gst.graph_format()

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

import Fitting_tools_mcmc as emfit

def gauss(x, k, mu,sig):

    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

def smooth(image,sm):
    
    from astropy.convolution import Gaussian2DKernel
    from scipy.signal import convolve as scipy_convolve
    from astropy.convolution import convolve
    
    gauss_kernel = Gaussian2DKernel(sm)

    con_im = convolve(image, gauss_kernel)
    
    con_im = con_im#*image/image
    
    return con_im

def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def plotting_OIII(wave, fluxs, ax, sol,fitted_model, error=np.array([1]), template=0, residual='none',axres=None):
    popt = sol['popt']
    keys = list(sol.keys())
    z = popt[0]
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>4700)&(wv_rest<5200))[0]
    
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>4700)&(wv_rst_sc<5200))[0]   
    
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')
    if len(error) !=1:
        ax.fill_between(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc]-error[fit_loc_sc],flux[fit_loc_sc]+error[fit_loc_sc], alpha=0.3, color='k')
        
        
    if template==0:
        y_tot = fitted_model(wave[fit_loc], *popt)
        y_tot_rs = fitted_model(wv_rst_sc[fit_loc_sc]*(1+z)/1e4, *popt)
    else:
        y_tot = fitted_model(wave[fit_loc], *popt, template)
        y_tot_rs = fitted_model(wv_rst_sc[fit_loc_sc]*(1+z)/1e4, *popt, template)
    
    
    if sol['Hbeta_peak'][1]>(sol['Hbeta_peak'][0]*0.6):
        Hbeta= 4861*(1+z)/1e4
        fwhm = sol['Hbeta_fwhm'][0]/3e5/2.35*Hbeta
        
        y_tot = y_tot- gauss(wv_rest[fit_loc], sol['Hbeta_peak'][0],4861, fwhm)
        
        sol['Hbeta_peak'][0] = 0
        
    ax.plot(wv_rest[fit_loc], y_tot, 'r--')
    
    flt = np.where((wv_rest[fit_loc]>4900)&(wv_rest[fit_loc]<5100))[0]

    ax.set_ylim(-0.1*max(y_tot[flt]), max(y_tot[flt])*1.1)
    ax.tick_params(direction='in')
    ax.set_xlim(4700,5050 )

    
    
    OIIIr = 5008.*(1+z)/1e4
    OIIIb = 4960.*(1+z)/1e4
    
    fwhm = sol['OIIIn_fwhm'][0]/3e5/2.35*OIIIr
    
    
    ax.plot(wv_rest[fit_loc] ,    gauss(wave[fit_loc], sol['OIIIn_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIIIn_peak'][0],OIIIr, fwhm) \
                 ,color= 'green', linestyle ='dashed')
    
        
    Hbeta= 4861.*(1+z)/1e4
    Hbeta= Hbeta + sol['Hbeta_vel'][0]/3e5*Hbeta
    fwhm = sol['Hbeta_fwhm'][0]/3e5/2.35*Hbeta
    ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['Hbeta_peak'][0],Hbeta, fwhm),\
                 color= 'orange', linestyle ='dashed')
    
    ax.plot(wv_rest[fit_loc], PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0],OIIIr, alpha=sol['cont_grad'][0]), linestyle='dashed', color='limegreen')
        
        
    if 'OIIIw_fwhm' in keys:
        OIIIr = OIIIr+ sol['out_vel'][0]/3e5*OIIIr
        OIIIb = OIIIb + sol['out_vel'][0]/3e5*OIIIb
        
        fwhm = sol['OIIIw_fwhm'][0]/3e5/2.35*OIIIr
        
        ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['OIIIw_peak'][0]/3,OIIIb, fwhm)+ gauss(wave[fit_loc], sol['OIIIw_peak'][0],OIIIr, fwhm),\
                     color= 'blue', linestyle ='dashed')
            
        
    if 'Hbetan_fwhm' in keys:
        Hbeta= 4861.*(1+z)/1e4
        Hbeta= Hbeta + sol['Hbetan_vel'][0]/3e5*Hbeta
        
        fwhm = sol['Hbetan_fwhm'][0]/3e5/2.35*Hbeta
        ax.plot(wv_rest[fit_loc] ,   gauss(wave[fit_loc], sol['Hbetan_peak'][0],Hbeta, fwhm),\
                     color= 'orange', linestyle ='dotted')
        
      
    if 'Fe_peak' in keys:
        ax.plot(wv_rest[fit_loc], PowerLaw1D.evaluate(wave[fit_loc], sol['cont'][0],OIIIr, alpha=sol['cont_grad'][0]), linestyle='dashed', color='limegreen')
        
        import Fitting_tools_mcmc as emfit
        if template=='BG92':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*emfit.FeII_BG92(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )
        
        if template=='Tsuzuki':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*emfit.FeII_Tsuzuki(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )
        
        if template=='Veron':
            ax.plot(wv_rest[fit_loc] , sol['Fe_peak'][0]*emfit.FeII_Veron(wave[fit_loc], z, sol['Fe_fwhm'][0]) , linestyle='dashed', color='magenta' )
        
    if residual !='none':
        resid_OIII = flux[fit_loc_sc]-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))
        
        axres.plot(wv_rst_sc[fit_loc_sc],resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-2*RMS_OIII, 2*RMS_OIII) ## the /3 scales to the ratio
        if residual=='rms':
            axres.fill_between(wv_rst_sc[fit_loc_sc], RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2)
        elif residual=='error':
            axres.fill_between(wv_rst_sc[fit_loc_sc],resid_OIII-error[fit_loc_sc],resid_OIII+error[fit_loc_sc], alpha=0.3, color='k')
            


def plotting_Halpha(wave, fluxs, ax, sol,fitted_model,error=np.array([1]), residual='none', axres=None):
    popt = sol['popt']
    z = popt[0]
    
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>6000.)&(wv_rest<7500.))[0]
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>6000)&(wv_rst_sc<7500))[0]   
    
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')
    if len(error) !=1:
        ax.fill_between(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc]-error[fit_loc_sc],flux[fit_loc_sc]+error[fit_loc_sc], alpha=0.3, color='k')
     
    y_tot = fitted_model(wave[fit_loc], *popt)
    y_tot_rs = fitted_model(wv_rst_sc[fit_loc_sc]*(1+z)/1e4, *popt)

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')


    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    ax.set_xlim(6562-250,6562+250 )
    ax.tick_params(direction='in')
    
    Hal_wv = 6562*(1+z)/1e4
    NII_r = 6583.*(1+z)/1e4
    NII_b = 6548.*(1+z)/1e4
    
    SII_r = 6731.*(1+z)/1e4   
    SII_b = 6716.*(1+z)/1e4   
    
    if len(popt)==11:
        BLR_wv = 6562*(1+z)/1e4+ popt[8]/3e5*Hal_wv
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[3], Hal_wv, popt[6]/3e5*Hal_wv/2.35), \
                color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4], BLR_wv, popt[7]/3e5*Hal_wv/2.35), \
                color='magenta', linestyle='dashed')
            
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[5], NII_r, popt[6]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[5]/3, NII_b, popt[6]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
            
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[9], SII_r, popt[6]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[10], SII_b, popt[6]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
            
    if len(popt)==8:
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[3], Hal_wv, popt[5]/3e5*Hal_wv/2.35), \
                color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4], NII_r, popt[5]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4]/3, NII_b, popt[5]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
            
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[6], SII_r, popt[5]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[7], SII_b, popt[5]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
           
    if len(popt)==12:
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[3], Hal_wv, popt[5]/3e5*Hal_wv/2.35), \
                color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4], NII_r, popt[5]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4]/3, NII_b, popt[5]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
            
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[6], SII_r, popt[5]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[7], SII_b, popt[5]/3e5*Hal_wv/2.35), \
                color='darkblue', linestyle='dashed')
            
        out_vel_hal = popt[10]/3e5*Hal_wv/2.35482
        out_vel_niir = popt[10]/3e5*NII_r/2.35482
        out_vel_niib = popt[10]/3e5*NII_b/2.35482
        
        Hal_wv_vel = 6562*(1+z)/1e4+ popt[11]/3e5*Hal_wv
        NII_r_vel = 6583.*(1+z)/1e4 + popt[11]/3e5*Hal_wv 
        NII_b_vel = 6548.*(1+z)/1e4 + popt[11]/3e5*Hal_wv 
        
        Hal_out = gauss(wave[fit_loc], popt[8], Hal_wv_vel, out_vel_hal)
        NII_out_r = gauss(wave[fit_loc], popt[9], NII_r_vel, out_vel_niir)
        NII_out_b = gauss(wave[fit_loc], popt[9]/3, NII_b_vel, out_vel_niib)
        
        ax.plot(wv_rest[fit_loc], Hal_out, color='magenta', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], NII_out_r, color='magenta', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], NII_out_b, color='magenta', linestyle='dashed')
    
    if residual !='none':
        resid_OIII = flux[fit_loc_sc]-y_tot_rs
        sigma_OIII = np.std(resid_OIII)
        RMS_OIII = np.sqrt(np.mean(resid_OIII**2))
        
        axres.plot(wv_rst_sc[fit_loc_sc],resid_OIII, drawstyle='steps-mid')
        axres.set_ylim(-2*RMS_OIII, 2*RMS_OIII) ## the /3 scales to the ratio
        if residual=='rms':
            axres.fill_between(wv_rst_sc[fit_loc_sc], RMS_OIII, -RMS_OIII, facecolor='grey', alpha=0.2)
        elif residual=='error':
            axres.fill_between(wv_rst_sc[fit_loc_sc],resid_OIII-error[fit_loc_sc],resid_OIII+error[fit_loc_sc], alpha=0.3, color='k')
        
        
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

def OIII_map_plotting(ID, path):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    IFU_header = pyfits.getheader(path)
    data = pyfits.getdata(path)
    
    map_flux = data[0,:,:]
    map_vel = data[1,:,:]
    map_fwhm = data[2,:,:]
    map_snr = data[3,:,:]
    
    
    x = int(IFU_header['X_cent']); y= int(IFU_header['Y_cent'])
    f = plt.figure( figsize=(10,10))
    
    
    
    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600
    
    
    Offsets_low = -np.array([x,y])
    Offsets_hig = np.shape(data)[1:3] -np.array([x,y])
    
    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                     Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix
    
    ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
    ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
    ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
    ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])
    
    flx = ax1.imshow(map_flux,vmax=map_flux[y,x], origin='lower', extent= lim_sc)
    ax1.set_title('Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    
    #lims = 
    #emplot.overide_axes_labels(f, axes[0,0], lims)
    
    
    vel = ax2.imshow(map_vel, cmap='coolwarm', origin='lower', vmin=-200, vmax=200, extent= lim_sc)
    ax2.set_title('Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')
    
    
    fw = ax3.imshow(map_fwhm,vmin=100, origin='lower', extent= lim_sc)
    ax3.set_title('FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    
    snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax4.set_title('SNR map')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')
    
    

