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


def plotting_OIII(wave, fluxs, ax, sol,fitted_model):
    popt = sol['popt']
    z = popt[0]
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>4930)&(wv_rest<5200))[0]
    
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>4930)&(wv_rst_sc<5200))[0]   
    
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')

    
    y_tot = fitted_model(wave[fit_loc], *popt)

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')


    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    ax.tick_params(direction='in')
    ax.set_xlim(4920,5050 )

    
    if len(popt)==8:
        OIIIr = 5008
        OIIIb = 4960
        
        fwhm = sol['OIIIn_fwhm'][0]/3e5/2.35*OIIIr
        
        
        plt.plot(wv_rest[fit_loc] ,   gauss(wv_rest[fit_loc], sol['OIIIn_peak'][0],OIIIr, fwhm) +\
                 gauss(wv_rest[fit_loc], sol['OIIIn_peak'][0]/3, OIIIb, fwhm) \
                     ,color= 'green', linestyle ='dashed')
         
        OIIIr = 5008+ sol['out_vel'][0]/3e5*OIIIr
        OIIIb = 4960 + sol['out_vel'][0]/3e5*OIIIb
        
        fwhm = sol['OIIIw_fwhm'][0]/3e5/2.35*OIIIr
        
        plt.plot(wv_rest[fit_loc] ,   gauss(wv_rest[fit_loc], sol['OIIIw_peak'][0],OIIIr, fwhm) +\
                 gauss(wv_rest[fit_loc], sol['OIIIw_peak'][0]/3, OIIIb, fwhm) \
                     ,color= 'blue', linestyle ='dashed')
            




def plotting_Halpha(wave, fluxs, ax, sol,fitted_model):
    popt = sol['popt']
    z = popt[0]
    print(len(popt))
    
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>6000.)&(wv_rest<7500.))[0]
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>6000)&(wv_rst_sc<7500))[0]   
    
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')

    y_tot = fitted_model(wave[fit_loc], *popt)

    ax.plot(wv_rest[fit_loc], y_tot, 'r--')


    ax.set_ylim(-0.1*max(y_tot), max(y_tot)*1.1)
    ax.set_xlim(6562-250,6562+250 )
    ax.tick_params(direction='in')
    
    Hal_wv = 6562*(1+z)/1e4
    BLR_wv = 6562*(1+z)/1e4+ popt[8]/3e5*Hal_wv
    NII_r = 6583.*(1+z)/1e4
    NII_b = 6548.*(1+z)/1e4
    if len(popt)==9:
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[3], Hal_wv, popt[6]/3e5*Hal_wv/2.35), \
                color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[4], BLR_wv, popt[7]/3e5*Hal_wv/2.35), \
                color='magenta', linestyle='dashed')
            
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[5], NII_r, popt[6]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], gauss(wave[fit_loc], popt[5]/3, NII_b, popt[6]/3e5*Hal_wv/2.35), \
                color='darkgreen', linestyle='dashed')
        
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


'''      
def plotting_Halpha_2QZJ(wave, fluxs, ax, out, mode,z, title=1, yscale='model'):
    
    #try:
    #    z =  (out.params['Han_center'].value*1e4/6562.)-1
    #except:
    #    z= (out.params['Ha_center'].value*1e4/6562.)-1
        
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>6000.)&(wv_rest<7500.))[0]
    
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>6000.)&(wv_rst_sc<6999.))[0]   
    
    #from scipy.signal import medfilt as mdf
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')

    try:
        ax.plot(wv_rest[fit_loc], out.eval(x=wave[fit_loc]), 'r--')
        #ax.set_ylim(min(flux[fit_loc]), max(out.eval(x=wave[fit_loc]))*1.1)
    
    except:
        out= out[0]
        ax.plot(wv_rest[fit_loc], out.eval(x=wave[fit_loc]), 'r--')
        
        

    ax.tick_params(direction='in')
    ax.set_xlim(6200, 6999)
    
    if yscale=='model':
        ax.set_ylim(-0.1*max(out.eval(x=wave[fit_loc])), max(out.eval(x=wave[fit_loc]))*1.1)
    
    elif yscale=='spec':
        ax.set_ylim(min(flux[fit_loc_sc])*1.1, max(flux[fit_loc_sc])*1.1)
            
    if mode=='mul':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Haw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Hawn_'], color='firebrick', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Han_'], color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nr_'], color='green', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nb_'], color='limegreen', linestyle='dashed')
        
        
        try: 
            broad = int(out.params['Haw_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        except:
            broad = int(out.params['Haw_sigma'].value/6562.8*2.9979e5/(1+z)*1e4)*2.3
            
        narow = int(out.params['Han_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        
        if title==1:    
            ax.set_title('B,N '+ str(broad)+', '+ str(narow)+' km/s'  )
    
    elif mode=='out':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Haw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nrw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nbw_'], color='blue', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Han_'],color='orange', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nr_'], color='limegreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nb_'], color='limegreen', linestyle='dashed')
        
        broad = int(out.params['Haw_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        narow = int(out.params['Han_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        
        if title==1:    
            ax.set_title('B,N '+ str(broad)+', '+ str(narow)+' km/s'  )
    
    
    elif mode=='sig':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Ha_'], 'b--')
    
    else:
        print ('mode not understood')
        
        
def plotting_Halpha_LBQS(wave, fluxs, ax, out, mode,z, title=1, yscale='model'):
    
    #try:
    #    z =  (out.params['Han_center'].value*1e4/6562.)-1
    #except:
    #    z= (out.params['Ha_center'].value*1e4/6562.)-1
        
    wv_rest = wave/(1+z)*1e4
    fit_loc = np.where((wv_rest>6000.)&(wv_rest<7500.))[0]
    
    ax.plot(wv_rest[fit_loc], fluxs.data[fit_loc], color='grey', drawstyle='steps-mid', alpha=0.2)
   
    flux = fluxs.data[np.invert(fluxs.mask)]
    wv_rst_sc= wv_rest[np.invert(fluxs.mask)]
        
    fit_loc_sc = np.where((wv_rst_sc>6000.)&(wv_rst_sc<6999.))[0]   
    
    #from scipy.signal import medfilt as mdf
    ax.plot(wv_rst_sc[fit_loc_sc],flux[fit_loc_sc], drawstyle='steps-mid')

    try:
        ax.plot(wv_rest[fit_loc], out.eval(x=wave[fit_loc]), 'r--')
        #ax.set_ylim(min(flux[fit_loc]), max(out.eval(x=wave[fit_loc]))*1.1)
    
    except:
        out= out[0]
        ax.plot(wv_rest[fit_loc], out.eval(x=wave[fit_loc]), 'r--')
        
        

    ax.tick_params(direction='in')
    ax.set_xlim(6200, 6999)
    
    if yscale=='model':
        ax.set_ylim(-0.1*max(out.eval(x=wave[fit_loc])), max(out.eval(x=wave[fit_loc]))*1.1)
    
    elif yscale=='spec':
        ax.set_ylim(min(flux[fit_loc_sc])*1.1, max(flux[fit_loc_sc])*1.1)
            
    if mode=='mul':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Haw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['X_'], color='firebrick', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Han_'], color='orange', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nr_'], color='green', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nb_'], color='limegreen', linestyle='dashed')
        
        
        try: 
            broad = int(out.params['Haw_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        except:
            broad = int(out.params['Haw_sigma'].value/6562.8*2.9979e5/(1+z)*1e4)*2.3
            
        narow = int(out.params['Han_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        
        if title==1:    
            ax.set_title('B,N '+ str(broad)+', '+ str(narow)+' km/s'  )
    
    elif mode=='out':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Haw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nrw_'], color='blue', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nbw_'], color='blue', linestyle='dashed')
        
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Han_'],color='orange', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nr_'], color='limegreen', linestyle='dashed')
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Nb_'], color='limegreen', linestyle='dashed')
        
        broad = int(out.params['Haw_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        narow = int(out.params['Han_fwhm'].value/6562.8*2.9979e5/(1+z)*1e4)
        
        if title==1:    
            ax.set_title('B,N '+ str(broad)+', '+ str(narow)+' km/s'  )
    
    
    elif mode=='sig':
        ax.plot(wv_rest[fit_loc], out.eval_components(x=wave[fit_loc])['Ha_'], 'b--')
    
    else:
        print ('mode not understood')
        
        
   

def Halpha_line_map(storage,z):
    out = storage['1D_fit_Halpha_mul']
    
    rst_w = storage['obs_wave'].copy()*1e4/(1+z)
    
    
    # Find width of the line by taking the width of the broad component
    width = out.params['Haw_fwhm'].value
    sel = width*2
    
    peak = out.params['Haw_center'].value + 6562.8
    
    
    use = np.where((rst_w<(peak+sel))&(rst_w>(peak-sel)))[0]
        
    flux_m = np.ma.array(data=storage['flux'].data.copy(), mask=storage['sky_clipped'].copy()) 
    
    flux_m = flux_m[use,:,:] 
    flux_m[np.isnan(flux_m)] = 0
    
    
    median_o = np.ma.sum(flux_m, axis=(0))  
    
    data = median_o.ravel() 
    shapes = storage['dim']
    
    
    x = np.linspace(0, shapes[1]-1, shapes[1])
    y = np.linspace(0, shapes[0]-1, shapes[0])
    x, y = np.meshgrid(x, y)   
    
    import scipy.optimize as opt
    
    initial_guess = storage['Median_stack_white_Center_data']
    initial_guess[0] = np.max(median_o.ravel())
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)
    
    print ( 'Halpha sizes', popt[:4])
    
    storage['Halpha_map_sizes'] = popt
    
    plt.figure()
    plt.title('Halpha map')
    plt.imshow(median_o, vmin=0, vmax= popt[0], origin='low') #
    plt.colorbar()

    data_fit = twoD_Gaussian((x,y), *popt)
    

    plt.contour(x, y, data_fit.reshape(shapes[0], shapes[1]), 8, colors='w')
    
    storage['Halpha_map'] = median_o 
    
    
    return storage




def Summary_plot(storage, mode, z, ID, storage_sub, hal_use='mul', prj='triple'):
    shapes = storage['dim']
    
    x = np.linspace(0, shapes[1]-1, shapes[1])
    y = np.linspace(0, shapes[0]-1, shapes[0])
    x, y = np.meshgrid(x, y)   
    
       
    # Setting up the plots
    f = plt.figure(figsize= (11,8))
    #f, axes = plt.subplots(2,4, figsize= (11,8))    
    
    
    if mode=='OIII':
        SNR = storage['1D_fit_OIII_SNR']
    
    elif mode=='H':
        SNR = storage['1D_fit_Halpha_SNR']
    
    elif mode=='Hb':
        SNR = storage['1D_fit_Hbeta_SNR']
    
    from IFU_tools_QSO import flux_measure
    fl = flux_measure(storage, mode)
    print (fl)
    #axes[0][3].set_title('Flux= %.2f e-17' %(fl/1e-17))
    
    
    ##################
    # ax1 Continuum Map
    
    try:
        storage, ax = new_HST_image_plot(storage, f,'241')
        new_ALMA_contour_plot(storage, ax, prj=prj)
        x=1
        ax.set_title(ID)
    except:
        print ('NO HST')
        ax = f.add_subplot(241)
        ax.text(0.3,0.5, 'NO HST on file')
        ax.set_title(ID)
    ##################
    #ax2 ALMA
    #ax = axes[0][1]
    if 1==1:
        ax =  new_ALMA_plot(storage, f, '242', prj=prj)
        #ax.set_title('z='+str(z))
    else:
        print ('No ALMA')
        ax = f.add_subplot(242)
        ax.text(0.3,0.5, 'NO ALMA on file')
        ax.set_title('z='+str(z))
    
    ##################
    # ax2 Continuum Map
    popt = storage['Median_stack_white_Center_data']
    
    cont_map = storage['Median_stack_white'].copy()
    
    header_cube = storage['header']
    cube_wcs= wcs.WCS(header_cube).celestial    
    
    try:
        ax = f.add_subplot(243, projection=cube_wcs)#axes[0][2]
    
    except:
        ax = f.add_subplot(243)#axes[0][2]
        
    
    
    ax.imshow(cont_map, vmin=0, vmax= cont_map[int(popt[2]), int(popt[1])], origin='low')
    
    try:
        new_ALMA_contour_plot(storage, ax, prj=prj)
    except:
        print ('Contours for Continuum not succesfull ')
    ax.tick_params(direction='in')    
    
    
    ax.set_title('SNR= %.3f' %SNR)
    
    try:
        dm= (x,y)
        data_fit = twoD_Gaussian(dm, *popt)
        ax.contour(x, y, data_fit.reshape(shapes[0], shapes[1]), 8, colors='w')
    
    except:
        print ('Contours are screwed up')
    
    # Sky spectrum mask
    mask = storage['Sky_stack_mask'].copy()
    mask = mask[300,:,:]    
    
    header_cube = storage['header']
    cube_wcs= wcs.WCS(header_cube).celestial  
    
    try:
        ax = f.add_subplot(244, projection=cube_wcs)#axes[0][2] 
    
    except:
        ax = f.add_subplot(244)#axes[0][2] 
        
    if ID !='cid_72':
        ax.imshow(np.ma.array(data=cont_map, mask=mask),vmin=0 ,origin='low')
    
    else:
        # If the cube is below zeros it will ignore the vmin=0 argument 
        ax.imshow(np.ma.array(data=cont_map, mask=mask), origin='low')
    
    ax.tick_params(direction='in')
    image_axis(ax, storage)
    ax.set_title('Flux= %.2f e-16' %(fl/1e-16))

    ##################
    # The stacked sky spectrum 
    ax = f.add_subplot(245) #axes[1][0] 
    ax.tick_params(direction='in')
    
    wave = storage['obs_wave'].copy()    
    sky_clipped = storage['sky_clipped_1D']   
    stacked_sky = storage['Stacked_sky']
    
    ax.plot(wave, (stacked_sky), drawstyle='steps-mid', color='grey')
    ax.plot(wave, np.ma.array(data=stacked_sky,mask=sky_clipped), drawstyle='steps-mid')
    
    if mode=='H':
        out = storage['1D_fit_Halpha_mul'] 
        peak = out.params['Han_center'].value
        wid = 400.*(1+z)/1e4
        
        ax.set_xlim(peak-wid, peak+wid)
    
    elif mode=='Hb':
        out = storage['1D_fit_Hbeta_mul'] 
        peak = out.params['Hbn_center'].value
        wid = 400.*(1+z)/1e4
        
        ax.set_xlim(peak-wid, peak+wid)
    
    elif mode=='OIII':
        out = storage['1D_fit_OIII_mul'] 
        peak = out.params['o3rn_center'].value
        wid = 400.*(1+z)/1e4
        
        ax.set_xlim(peak-wid, peak+wid)
   
    ##################
    # Fit to the data
    ax = f.add_subplot(246) #axes[1][1] 
    ax.tick_params(direction='in')
    
    
    wave = storage['obs_wave'].copy()
    flux = storage['1D_spectrum'].copy()
    
    if mode=='H':
        out = storage['1D_fit_Halpha_mul'] 
        outs= storage['1D_fit_Halpha_sig'] 
        
        plotting_Halpha(wave, flux, ax, out, hal_use,z)
        
        
    elif mode=='Hb':
        out = storage['1D_fit_Hbeta_mul'] 
        outs= storage['1D_fit_Hbeta_sig'] 
        
        plotting_Hbeta(wave, flux, ax, out, 'mul',z)
        ax.plot(wave/(1+z)*1e4, outs.eval(x=wave), 'k--')
    
    elif mode=='OIII':
        out = storage['1D_fit_OIII_mul']
        outs= storage['1D_fit_OIII_sig']
        
        fit_loc = np.where((wave>4800*(1+z)/1e4)&(wave<5050*(1+z)/1e4))[0]   
    
        plotting_OIII(wave, flux, ax, out, 'mul',z)
        ax.plot(wave[fit_loc]/(1+z)*1e4, outs.eval(x=wave[fit_loc]), 'k--')
        
    else:
        print ('Wrong mode')

    ##################
    # Substracted Continuum spectrum
    
    
    if storage_sub !=1:
        ax = f.add_subplot(247) #axes[1][2] 
        ax.tick_params(direction='in')
    
        wave = storage_sub['obs_wave'].copy()#*1e4/(1+z)
        flux = storage_sub['1D_spectrum'].copy()
    
        if mode=='H':
            out = storage_sub['1D_fit_Halpha_mul'] 
        
            fit_loc = np.where((wave>6562.8-300)&(wave<6562.8+300))[0]      
            plotting_Halpha(wave, flux, ax, out, 'mul',z)
    
    
        elif mode=='OIII':
            out = storage_sub['1D_fit_OIII_mul']
        
            plotting_OIII(wave, flux, ax, out, 'mul',z)
        
        elif mode=='Hb':
            try:
                out = storage_sub['1D_fit_Hbeta_mul']
    
                fit_loc = np.where((wave>4861-70)&(wave<4861+75))[0]       
                plotting_Hbeta(wave, flux, ax, out, 'mul',z)
            except:
                print ('Hbet sub is oupsy')
        
    
        else:
            print ('Wrong mode')
        
    ##################
    # The line map
    if storage_sub !=1:
        header_cube = storage['header']
        cube_wcs= wcs.WCS(header_cube).celestial  
    
        ax = f.add_subplot(248)
    
        ax.tick_params(direction='in')
    
        plot_line_map(storage,ax , mode)
        image_axis(ax, storage)
    f.tight_layout()
    
    
    

def save_obj(obj, name ):
    with open(PATH + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open(PATH + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')
    
    


def new_HST_image_plot(storage,f, axloc):
    
    img_file = '/Users/jansen/Google Drive/Astro/KMOS_SIN/HST_data/HST_'+storage['X-ray ID']+'.fits'
   
    img=pyfits.getdata(img_file)
    img_wcs= WCS(img_file).celestial
    hdr=pyfits.getheader(img_file)
    
    new_size = storage['Phys_size']
    
    print (new_size)
    
    try:
        pixscale=abs(hdr['CD1_1']*3600)
    except:
        pixscale=abs(hdr['CDELT1']*3600)
        
    print (pixscale)
    
    Cat = storage['Cat_entry']
    try:
        Ra_opt = Cat['ra']
        Dec_opt = Cat['dec']
    
    except:
        
        Ra_opt = Cat['RA']
        Dec_opt = Cat['DEC']
    
    opt_world= np.array([[Ra_opt,Dec_opt]])
    opt_pixcrd = img_wcs.wcs_world2pix(opt_world, 0) # WCS transform
    opt_x= (opt_pixcrd[0,0]) # X pixel
    opt_y= (opt_pixcrd[0,1]) # Y pixel
    
    position = np.array([opt_x, opt_y])
    
    cutout = Cutout2D(img, position, new_size/pixscale, wcs=img_wcs,mode='partial') 

    img=(cutout.data).copy()
    img_wcs=cutout.wcs
    
    cut_opt = img
    
    if type(axloc)==str:
        ax = f.add_subplot(axloc, projection=img_wcs)
    
    else:
        ax = f.add_subplot(axloc[0],axloc[1],axloc[2], projection=img_wcs)
        
    
    rms = np.std(cut_opt - cut_opt.mean(axis=0))
    
    #rms = 0.13/3
    print ('RMS of the HST', rms)
    
    
    ax.imshow((cut_opt), origin='low', vmin=0, vmax=3*rms)
    ax.set_autoscale_on(False)
    
    return storage, ax
    
def new_ALMA_contour_plot(storage,ax, infos=1, linew =2.,respond=0, both=0, prj='triple', clr='red', linestyle='dashed'):
    
    
    if storage!=1:  
        afile = storage['Cat_entry']['ALMA']
        ID = storage['X-ray ID']
    
        Cat = storage['Cat_entry']
        
        try:
            Ra_opt = Cat['ra']
            Dec_opt = Cat['dec'] 
        except:
            Ra_opt = Cat['RA']
            Dec_opt = Cat['DEC']
        new_size = storage['Phys_size']
    
    else:
        ID = infos[0]
        Ra_opt = infos[1]
        Dec_opt = infos[2]
        new_size = infos[3]
        
    
    if prj=='triple':
        folder = '/KMOS_SIN'
    
    elif prj=='4QSO':   
        folder = '/Four_Quasars'
        
        
    import glob
    afile = glob.glob(PATH+folder+'/ALMA/Final_set/'+ID+'_l*')
    
    if len(afile)>0:
        
        
        cont_file =glob.glob(PATH+folder+'/ALMA/Final_set/'+ID+'_l_clean.pbcor.*')[0]        
        cont= (pyfits.getdata(cont_file)[0,0,:,:])
        
        coor = Table.read('/Users/jansen/Google Drive/Astro/KMOS_SIN/Catalogues/Astrometry_corrections.fits')

        lnew = np.zeros(len(coor))
        lnew = np.array(lnew, dtype=str)

        for i in range(len(coor)):
            IDd = coor['XID'][i]
            
            if (IDd[0] != 'c') & (IDd[0] != 'l') & (IDd[0] != 'x')& (IDd[0] != 'A'):
                IDd = 'XID_'+ str(coor['XID'][i])
                
                lnew[i] = IDd.strip()
                
        index = np.where(lnew==ID)[0]
        
        if len(index)>0:
            coors = np.array([float(coor[index]['dra_wfc3']), float(coor[index]['ddec_wfc3']) ])
            
        else:
            coors= np.array([0,0]) 
        
        print ('fce for ALMA cont: ',ID,index, coors)
        
        
        header = pyfits.getheader(cont_file)
        if (ID[:3]=='XID') :#| (ID[:3]=='ALE'):
            header['CRVAL1'] = header['CRVAL1'] - coors[0]/3600  # Based on Elbaz. Mine +0.19
            header['CRVAL2'] = header['CRVAL2'] - coors[1]/3600 
            
            #print 'Taking into the account the CDFS offset'
           
         
        wcs_alma_header=  wcs.WCS(header)
        
           
        ###############################################################
        # Find the center of the image
        alm_world= np.array([[Ra_opt, Dec_opt,0,0]])    
        alm_pixcrd = wcs_alma_header.wcs_world2pix(alm_world, 0) # WCS transform
        
        alma_x= int(alm_pixcrd[0,0]) # X pixel
        alma_y= int(alm_pixcrd[0,1]) # Y pixel
        
        position = np.array([alma_x, alma_y])
        
        
        #print header
        pixscale = abs(header['CDELT1']*3600)
        
        cont_wcs= wcs.WCS(header).celestial
        
        cutout = Cutout2D(cont, position, new_size/pixscale, wcs=cont_wcs,mode='partial') 
        
        cont_c_d = cutout.data
        cont_c_w = cutout.wcs
        
        rms = np.std(cont_c_d- cont_c_d.mean(axis=0))
        print ('fce for ALMA cont: ','Plotting Low ress solid contours')
        print ('fce for ALMA cont: ', 'linew ', linew)
        try:
            ax.contour(cont_c_d, transform= ax.get_transform(cont_c_w), colors=clr, linestyles=linestyle ,levels=( 2.5*rms ,3*rms,4*rms, 5*rms), alpha=0.5, linewidths=linew)
        except:
            print ('ALMA Contours fail')
    
    import glob
    afile = glob.glob(PATH+folder+'/ALMA/Final_set/'+ID+'_h*')
    
    if (len(afile)>0) & (both==1):
        
        
        cont_file =glob.glob(PATH+'KMOS_SIN/ALMA/Final_set/'+ID+'_h_clean.pbcor.*')[0]        
        cont= (pyfits.getdata(cont_file)[0,0,:,:])
        
        coor = Table.read('/Users/jansen/Google Drive/Astro/KMOS_SIN/Catalogues/Astrometry_corrections.fits')

        lnew = np.zeros(len(coor))
        lnew = np.array(lnew, dtype=str)

        for i in range(len(coor)):
            IDd = coor['XID'][i]
            
            if (IDd[0] != 'c') & (IDd[0] != 'l') & (IDd[0] != 'x')& (IDd[0] != 'A'):
                IDd = 'XID_'+ str(coor['XID'][i])
                
                lnew[i] = IDd.strip()
                
        index = np.where(lnew==ID)[0]
        
        if len(index)>0:
            coors = np.array([float(coor[index]['dra_wfc3']), float(coor[index]['ddec_wfc3']) ])
            
        else:
            coors= np.array([0,0]) 
        
        print (ID,index, coors)
        
        
        header = pyfits.getheader(cont_file)
        if (ID[:3]=='XID') :#| (ID[:3]=='ALE'):
            header['CRVAL1'] = header['CRVAL1'] - coors[0]/3600  # Based on Elbaz. Mine +0.19
            header['CRVAL2'] = header['CRVAL2'] - coors[1]/3600 
            
            #print 'Taking into the account the CDFS offset'
           
         
        wcs_alma_header=  wcs.WCS(header)
           
        ###############################################################
        # Find the center of the image
        alm_world= np.array([[Ra_opt, Dec_opt,0,0]])    
        alm_pixcrd = wcs_alma_header.wcs_world2pix(alm_world, 0) # WCS transform
        alma_x= int(alm_pixcrd[0,0]) # X pixel
        alma_y= int(alm_pixcrd[0,1]) # Y pixel
        
        position = np.array([alma_x, alma_y])
        
        
        
        pixscale = abs(header['CDELT1']*3600)
        
        cont_wcs= wcs.WCS(header).celestial
        cutout = Cutout2D(cont, position, new_size/pixscale, wcs=cont_wcs,mode='partial') 
        
        cont_c_d = cutout.data
        cont_c_w = cutout.wcs
        
        rms = np.std(cont_c_d- cont_c_d.mean(axis=0))
        print ('Plotting High ress dashed contours')
        ax.contour(cont_c_d, transform= ax.get_transform(cont_c_w), colors=clr,linewidths=linew, levels=( 3*rms, 5*rms), alpha=0.5)
    
    if respond==1:
        return header['BMAJ'],header['BMIN'],header['BPA']

def new_ALMA_CO_cont(storage,ax):
    
    ID = storage['X-ray ID']
    
    #center_global = storage['HST_center_im'].copy()#storage['HST_center_glob'].copy()
    #center_HST = storage['HST_center_glob'].copy()
    #center_HST = center_HST[0]
   
    if 1==0:
        print ('No ALMA data')
    
        
    if ID =='XID_587':
        cont_file = '/Users/jansen/Astro/KMOS_SIN/ALMA/XID_587_images/XID_587_CO32.fits'
        
        cont= (pyfits.getdata(cont_file)[0,:,:,:])
              
        cont = cont[17:46,:,:]
        cont = np.sum(cont, axis=(0))
        
      
        header = pyfits.getheader(cont_file)
        
        if (ID[:3]=='XID') | (ID[:3]=='ALE'):
            header['CRVAL1'] = header['CRVAL1'] - 0.195/3600  # Based on Elbaz. Mine +0.19
            header['CRVAL2'] = header['CRVAL2'] + 0.23/3600 
            print ('Taking into the account the CDFS offset')
           
         
        wcs_alma_header=  wcs.WCS(header)
        Cat = storage['Cat_entry']
        try:
            Ra_opt = Cat['ra']
            Dec_opt = Cat['dec']
        
        except:
            Ra_opt = Cat['RA']
            Dec_opt = Cat['DEC']
        ###############################################################
        # Find the center of the image
        alm_world= np.array([[Ra_opt, Dec_opt,0,0]])    
        alm_pixcrd = wcs_alma_header.wcs_world2pix(alm_world, 0) # WCS transform
        alma_x= int(alm_pixcrd[0,0]) # X pixel
        alma_y= int(alm_pixcrd[0,1]) # Y pixel
        
        position = np.array([alma_x, alma_y])
        
        pixscale = abs(header['CDELT1']*3600)
        
        cont_wcs= wcs.WCS(header).celestial
        new_size = storage['Phys_size']
        cutout = Cutout2D(cont, position, new_size/pixscale, wcs=cont_wcs,mode='partial') 
        
        cont_c_d = cutout.data
        cont_c_w = cutout.wcs
        #cont_c_d = cont
        
        rms = np.std(cont_c_d- cont_c_d.mean(axis=0))
        
        ax.contour(cont_c_d,transform= ax.get_transform(cont_c_w) ,colors='white',levels=(2*rms, 3*rms, 5*rms), alpha=0.5)
        
        #




def new_ALMA_plot(storage,f, axloc, prj='triple'):
    
    #afile = storage['Cat_entry']['ALMA']
    ID = storage['X-ray ID']
    
    #center_global = storage['HST_center_im'].copy()#storage['HST_center_glob'].copy()
    #center_HST = storage['HST_center_glob'].copy()
    #center_HST = center_HST[0]
   
    #if afile=='':
    #    print ('No ALMA data')
    
    if prj=='triple':
        folder = 'KMOS_SIN'
    
    elif prj=='4QSO':   
        folder = 'Four_Quasars'
        
        
    import glob
    afile = glob.glob(PATH+folder+'/ALMA/Final_set/'+ID+'_l*')
    
    if len(afile)>0:
        
        
        cont_file =glob.glob(PATH+folder+'/ALMA/Final_set/'+ID+'_l_clean.pbcor.*')[0]   
    
        cont= (pyfits.getdata(cont_file)[0,0,:,:])
        
        header = pyfits.getheader(cont_file)
        if (ID[:3]=='XID') | (ID[:3]=='ALE'):
            header['CRVAL1'] = header['CRVAL1'] - 0.195/3600  # Based on Elbaz. Mine +0.19
            header['CRVAL2'] = header['CRVAL2'] + 0.23/3600 
            
            print ('Taking into the account the CDFS offset')
            
        
        wcs_alma_header=  wcs.WCS(header)
        Cat = storage['Cat_entry']
        
        try:
            Ra_opt = Cat['ra']
            Dec_opt = Cat['dec']
        except:
            Ra_opt = Cat['RA']
            Dec_opt = Cat['DEC']
        ###############################################################
        # Find the center of the image
        alm_world= np.array([[Ra_opt, Dec_opt,0,0]])    
        alm_pixcrd = wcs_alma_header.wcs_world2pix(alm_world, 0) # WCS transform
        alma_x= int(alm_pixcrd[0,0]) # X pixel
        alma_y= int(alm_pixcrd[0,1]) # Y pixel
        
        position = np.array([alma_x, alma_y])
  
        pixscale = abs(header['CDELT1']*3600)
        
        cont_wcs= wcs.WCS(header).celestial
        new_size = storage['Phys_size']
        
        cutout = Cutout2D(cont, position, new_size/pixscale, wcs=cont_wcs,mode='partial') 
        
        cont_c_d = cutout.data
        cont_c_w = cutout.wcs
        
        
        ax = f.add_subplot(axloc, projection=cont_c_w) 
        rms = np.std(cont_c_d- cont_c_d.mean(axis=0))
        
        ax.imshow(cont_c_d, origin='low', vmin=0, vmax=3*rms)
        ax.set_autoscale_on(False)
        
        
        ax.contour(cont_c_d, transform= ax.get_transform(cont_c_w) ,colors='red',levels=(2*rms, 3*rms, 5*rms), alpha=0.5)
        
        return ax
         
        
        
def Spax_fit_plot_sig(storage, mode, z, ID, binning, instrument='KMOS',add='', addsave= '', prj='triple'):
    popt = storage['Median_stack_white_Center_data']
    spax = storage['Signal_mask'].copy()
    spax = spax[0,:,:]
    
    center =  storage['Median_stack_white_Center_data'][1:3].copy()
    
    header_cube = storage['header']
    cube_wcs= wcs.WCS(header_cube).celestial    
       
     
    g = plt.figure(figsize=(12,8))
    
    if instrument == 'KMOS':
        ins = ''
    elif instrument =='Sinfoni':
        ins = '_sin'
    if mode =='OIII':       
        Image_narrow = pyfits.getdata(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+binning+'_spaxel_fit'+ins+add+'.fits')
        
    elif mode =='H':
        Image_narrow = pyfits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+binning+'_spaxel_fit'+ins+add+'.fits')
        
    Im_flux_pl = Image_narrow[0]
    Im_offset_pl = Image_narrow[1]
    Im_width_pl = Image_narrow[2]
    
    y = 0.38
    x = 0.0154
    dx= 0.252
    lx = 0.215
    ly = 0.03
    
    # Flux Map
    mask_nan =  np.ma.masked_invalid(Im_flux_pl).mask
   
    spax_tot = np.logical_or( mask_nan, spax)
    
    
    
    masked_flux = np.ma.array(data= Im_flux_pl, mask= spax_tot)
    min_flux = np.ma.min(masked_flux)
    max_flux = np.ma.max(masked_flux)
    
    ax1 = g.add_subplot(2,4,6, projection=cube_wcs)


    flx_m =ax1.imshow(Im_flux_pl,vmin=min_flux, vmax= max_flux,origin='low',cmap='plasma')
    ax1.tick_params(direction='in')
    new_ALMA_contour_plot(storage, ax1, prj=prj)
    
    ax1.plot(center[0], center[1], marker='*', color='limegreen',markersize=8)
    
    if (ID=='lid_1565') & (mode=='H'):
        ax1.set_xlim(50-13, 50+13)
        ax1.set_ylim(42-13, 42+13)
    
    if (ID=='XID_36') & (mode=='OIII'):
        ax1.set_xlim(49-13, 49+13)
        ax1.set_ylim(45-13, 45+13)

    
    axcbar0 = plt.axes([ x+ dx, y , lx, ly]) #plt.axes([0.055+ 0.2469,0.397,0.189,0.03])
    axcbar0.tick_params(direction='in')        
    cbar0 = g.colorbar(flx_m,cax=axcbar0 ,orientation='horizontal', ticks= [min_flux,(min_flux+max_flux)/2, max_flux])
    axcbar0.tick_params(axis='y',left='off',labelleft='off',right='off',labelright='off')
    axcbar0.tick_params(axis='x',labelbottom='off',top='on',labeltop='on')
    
    #image_axis(ax, storage)
    
    # Velocity map
    
    masked_off = np.ma.array(data= Im_offset_pl, mask= spax_tot)
    min_off = np.ma.min(masked_off)
    
    if min_off< -300:
        min_off=-100
    
    max_off = np.ma.max(masked_off)
     
    #min_off = -100
    if (ID=='XID_587')&( mode=='OIII'):
        max_off = 250
        min_off = -100
    
    if (ID=='XID_587')&( mode=='H'):
        min_off = -120
        max_off = 250
        
    
    if (ID=='XID_208')&( mode=='OIII'):
        min_off = -100
    
    if (ID=='ALESS_75')&( mode=='H'):
        min_off = -50
        max_off = 50
    
    if (ID=='ALESS_75')&( mode=='OIII'):
        min_off = -100
        max_off = 100
        
    
    if (ID=='XID_614')&( mode=='H'):
        min_off = 0
        max_off = 250
    
    if (ID=='XID_427')&( mode=='H'):
        min_off = -200
        max_off = 170
    
    if (ID=='ALESS_75')&( mode=='H'):
        min_off = -30
        max_off = 400
    
    if (ID=='lid_1565')&( mode=='H'):
        min_off = -400
        max_off = 300
    
    if (ID=='XID_36')&( mode=='OIII'):        
        max_off = 200
        min_off = -200
        
    
    # ,projection=cube_wcs
    ax = g.add_subplot(2,4,7)
    
    offset_m = ax.imshow(Im_offset_pl, vmin=min_off, vmax= max_off, origin='low', cmap='coolwarm')
    ax.tick_params(direction='in')
    
    ax.plot(center[0], center[1], marker='*', color='limegreen',markersize=8)
   
    axcbar1 = plt.axes([ x+ dx*2, y , lx, ly]) #plt.axes([0.046+(0.2465)*2,0.405,0.185,0.03])
    axcbar1.tick_params(direction='in')
    cbar1 = g.colorbar(offset_m,cax=axcbar1 ,orientation='horizontal', ticks= [min_off,0,max_off])
    axcbar1.tick_params(axis='y',left='off',labelleft='off',right='off',labelright='off')
    axcbar1.tick_params(axis='x',labelbottom='off',top='on',labeltop='on')
    
    if (ID=='lid_1565') & (mode=='H'):
        ax.set_xlim(50-13, 50+13)
        ax.set_ylim(42-13, 42+13)
    
    if (ID=='XID_36') & (mode=='OIII'):
        ax.set_xlim(49-13, 49+13)
        ax.set_ylim(45-13, 45+13)
    #image_axis(ax, storage)
    
    # FWHM map
    masked_wid = np.ma.array(data= Im_width_pl, mask= spax_tot)
    min_wid = np.ma.min(masked_wid)
    max_wid = np.ma.max(masked_wid)
    
    if max_wid>800:
        max_wid=800
    
    ax = g.add_subplot(2,4,8, projection=cube_wcs)
    
    if (ID=='XID_587')&( mode=='OIII'):
        max_wid = 800
        
           
    if (ID=='XID_587')&( mode=='H'):        
        max_wid = 900
    
    if (ID=='XID_36')&( mode=='OIII'):        
        max_wid = 2500
        min_wid = 1600
    
    if (ID=='lid_1565')&( mode=='H'):        
        max_wid = 1200
        min_wid = 500
        
    width_m = ax.imshow(abs(Im_width_pl), vmin=min_wid, vmax=max_wid, origin='low') 
    ax.plot(center[0], center[1], marker='*', color='limegreen',markersize=8)
    
    ax.tick_params(direction='in')
    
    if (ID=='lid_1565') & (mode=='H'):
        ax.set_xlim(50-13, 50+13)
        ax.set_ylim(42-13, 42+13)
        
    if (ID=='XID_36') & (mode=='OIII'):
        ax.set_xlim(49-13, 49+13)
        ax.set_ylim(45-13, 45+13)
        
        
    axcbar2 = plt.axes([ x+ dx*3, y , lx, ly])#plt.axes([0.046+(0.246)*3,0.405,0.185,0.03])
    axcbar2.tick_params(direction='in')
    cbar2 = g.colorbar(width_m,cax=axcbar2 ,orientation='horizontal', ticks= [min_wid, (min_wid+max_wid)/2 ,max_wid])
    axcbar2.tick_params(axis='y',left='off',labelleft='off',right='off',labelright='off')
    axcbar2.tick_params(axis='x',labelbottom='off',top='on',labeltop='on')
    
    #image_axis(ax, storage)
    
    # SNR maps    
    ax2 = g.add_subplot(2,4,5, projection=cube_wcs) #g.add_axes([X_image_l, Y_image_l, ly_image_l, lx_image_l], projection=cube_wcs)
    SNR_m = ax2.imshow(Image_narrow[3], vmin=3, vmax=10, origin='low')
    ax2.tick_params(direction='in')
    
    if (ID=='lid_1565') & (mode=='H'):
        ax2.set_xlim(50-13, 50+13)
        ax2.set_ylim(42-13, 42+13)
        
    if (ID=='XID_36') & (mode=='OIII'):
        ax2.set_xlim(49-13, 49+13)
        ax2.set_ylim(45-13, 45+13)
    
    axcbar2 = plt.axes([ x, y , lx, ly])
    axcbar2.tick_params(direction='in')
    cbar2 = g.colorbar(SNR_m,cax=axcbar2 ,orientation='horizontal', ticks= [3,5,8,10])
    axcbar2.tick_params(axis='y',left='off',labelleft='off',right='off',labelright='off')
    axcbar2.tick_params(axis='x',labelbottom='off',top='on',labeltop='on')
    new_ALMA_contour_plot(storage, ax2, prj=prj)
    ##################
    # ax0 HST    
    try:
        storage, ax = new_HST_image_plot(storage, g, '241')
        ax.set_title(ID)
        
    except:
        print ('NO HST')
        ax = g.add_subplot(241)
        ax.set_title(ID)
    
    new_ALMA_contour_plot(storage, ax, prj=prj)
    #new_ALMA_CO_cont(storage,ax)
    
    ################
    # ALMA map 
    try:
        ax =  new_ALMA_plot(storage, g, '244', prj=prj)
        
    except:
        print ('No ALMA')
        ax = g.add_subplot(244)
        ax.text(0.3,0.5, 'NO ALMA on file')
    
    ##################
    # ax1 Continuum Map
    popt = storage['Median_stack_white_Center_data']    
    cont_map = storage['Median_stack_white'].copy()   
    
    shp = storage['dim']
    
    ax = g.add_subplot(242, projection=cube_wcs)
    ax.set_title('z='+str(z))
    
    ax.imshow(cont_map, vmin=0, vmax= cont_map[int(popt[2]), int(popt[1])], origin='low')
    ax.plot(float(shp[1])/2, float(shp[0])/2, 'b*')
    ax.tick_params(direction='in')
    
    ax.plot(center[0], center[1], marker='*', color='limegreen',markersize=8)
    
    new_ALMA_contour_plot(storage, ax, prj=prj)
        
    ###################
    # 1D spectrum
    ###################
    ax = g.add_subplot(243) 
    ax.tick_params(direction='in')
    
    if mode=='OIII':
        wave = storage['obs_wave'].copy()
        flux = storage['1D_spectrum'].copy()
    
        out = storage['1D_fit_OIII_mul']
        outs= storage['1D_fit_OIII_sig']
         
        fit_loc = np.where((wave>4900)&(wave<5050))[0]   
    
        plotting_OIII(wave, flux, ax, out, 'mul',z)
        
        wave = storage['obs_wave'].copy()
        ax.plot(wave[fit_loc], outs.eval(x=wave[fit_loc]), 'k--')
        #ax.set_ylim(np.ma.min(flux*1.2),  np.max(out.eval(x=wave[fit_loc]-5006.84))*1.2)
        ax.set_xlim(4930, 5050)

    elif mode=='H':
        wave = storage['obs_wave'].copy()#*1e4/(1+z)
        flux = storage['1D_spectrum'].copy()
        
        out = storage['1D_fit_Halpha_mul'] 
        outs= storage['1D_fit_Halpha_sig'] 
        fit_loc = np.where((wave>6562.8-300)&(wave<6562.8+300))[0]    
        
        plotting_Halpha(wave, flux, ax, out, 'mul',z)
        
        
    g.tight_layout()
    
    #######################################
    # Saving all the info
    #######################################
    
    save = {'Phys_size': storage['Phys_size']}
    
    try:
        save['World_cent']  = storage['HST_center_glob']
    
    except:
        print ('Missing some info')
        
    save['Cube_cent'] = storage['Median_stack_white_Center_data']
    
    from IFU_tools_QSO import flux_measure
    fl = flux_measure(storage, mode)
    
    
    save['flx_map'] = np.array([min_flux, max_flux])
    save['off_map'] = np.array([min_off, max_off])
    save['wid_map'] = np.array([min_wid, max_wid])
    
    if mode=='OIII':
        save['SNR'] = storage['1D_fit_OIII_SNR']
        save['flx'] = fl
        
        

    elif mode == 'H':
        save['Hal_SNR'] = storage['1D_fit_Halpha_SNR']
        save['Hal_flx'] = fl
        
        

    save_obj(save, 'KMOS_SIN/Results_storage/Props/'+ID+'_'+mode+ins+addsave)
        
 
    

def image_axis(axes, storage, tickin=0):
    
    header  = storage['header']
    
    try:
        arc = np.round(1./(header['CD2_2']*3600))   
    
    except:
        arc = np.round(1./(header['CDELT2']*3600))   
        
    shapes = storage['dim']
    center =  storage['Median_stack_white_Center_data'][1:3].copy()
    
    
    axes.plot(center[0], center[1], 'r*',markersize=8)
    
    x_tick_loc_min = np.array([center[0]])
    x_tick_name_min = np.array([0])
    x_tick_loc_max = np.array([center[0]])
    x_tick_name_max =  np.array([0])
    
    y_tick_loc_min = np.array([center[1]])
    y_tick_name_min =  np.array([0])
    y_tick_loc_max = np.array([center[1]])
    y_tick_name_max =  np.array([0])
    
    for i in range(20):
        x_tick_loc_min = np.append(x_tick_loc_min, x_tick_loc_min[-1]-arc/2)
        x_tick_name_min = np.append(x_tick_name_min,x_tick_name_min[-1]-0.5)
        x_tick_loc_max = np.append(x_tick_loc_max, x_tick_loc_max[-1]+arc/2)
        x_tick_name_max = np.append(x_tick_name_max,x_tick_name_max[-1]+0.5)
        
        y_tick_loc_min = np.append(y_tick_loc_min, y_tick_loc_min[-1]-arc/2)
        y_tick_name_min = np.append(y_tick_name_min,y_tick_name_min[-1]-0.5)
        y_tick_loc_max = np.append(y_tick_loc_max, y_tick_loc_max[-1]+arc/2)
        y_tick_name_max = np.append(y_tick_name_max,y_tick_name_max[-1]+0.5)
    
    x_tick_loc = np.append(x_tick_loc_min, x_tick_loc_max)
    y_tick_loc = np.append(y_tick_loc_min, y_tick_loc_max)
  
    x_tick_name = np.append(x_tick_name_min, x_tick_name_max)   
    y_tick_name = np.append(y_tick_name_min, y_tick_name_max)
    
    usex = np.where((x_tick_loc>0)&(x_tick_loc< shapes[0]))[0]    
    x_tick_loc = x_tick_loc[usex]
    x_tick_name = x_tick_name[usex]
    
    usey = np.where((y_tick_loc>0)&(y_tick_loc< shapes[1]))[0]
    y_tick_loc = y_tick_loc[usey]
    y_tick_name = y_tick_name[usey]
    
    axes.set_xticks(x_tick_loc)
    axes.set_xticklabels(x_tick_name,rotation='vertical')
    axes.set_xlabel('arcseconds')
    
    axes.set_yticks(y_tick_loc)
    axes.set_yticklabels(y_tick_name)
    axes.set_ylabel('arcseconds')
    
    
    
  
    
def plot_vel_slices(storage, mode, binning):
    z = storage['z_guess']
    ID = storage['X-ray ID']
    if mode =='OIII':
        Image_cube = pyfits.getdata(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+binning+'_vel_slices.fits')
        
        out_old = storage['1D_fit_OIII_sig']
        center = out_old.params['o3r_center'].value
        
        out = storage['1D_fit_OIII_mul']
        outs= storage['1D_fit_OIII_sig']
                
        
    elif mode =='H':
        Image_cube = pyfits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+binning+'_vel_slices.fits')
        
        out = storage['1D_fit_Halpha_mul']
        center = out.params['Han_center'].value
        
        outs= storage['1D_fit_Halpha_sig']
        
        
    print (np.shape(Image_cube))
    velocity_r = np.loadtxt(PATH+'KMOS_SIN/Results_storage/Vel_slices.txt')*1000
    
    fig = plt.figure(figsize= (11,8))

    cl = 0
    rw = 0

    wave = storage['obs_wave'].copy()#*1e4/(1+z)
    flux = storage['1D_spectrum'].copy()
    popt = storage['Median_stack_white_Center_data']
    
    
    for i in range(8):
        if i==0:
            ax = fig.add_axes([0.05, 0.65, 0.22,0.22])
            image_axis(ax, storage)
        
            ax.imshow(storage['Median_stack_white'], vmin=0, vmax= storage['Median_stack_white'][int(popt[2]), int(popt[1])], origin='low')
            
            center =  storage['Median_stack_white_Center_data'][1:3].copy()
            ax.plot(center[0], center[1], marker='*', color='limegreen',markersize=8)
    
            ax.set_title(storage['X-ray ID'])
            
        elif i>0:    
            ax1 = fig.add_axes([0.052 + cl*0.245,0.83-0.5*rw,0.178,0.15] )
            
            if mode =='OIII':
                fit_loc = np.where((wave>4900)&(wave<5030))[0]       
                plotting_OIII(wave, flux, ax1, out, 'mul',z, title=0)
                ax1.set_xlim(4930,5020)
            
            elif mode=='H':
                fit_loc = np.where((wave>6562.8-100)&(wave<6562.8+100))[0]       
                plotting_Halpha(wave, flux, ax1, out, 'mul',z, title=0)
                ax1.set_xlim(6562.8-100,6562.8+100)
                
            
            cnt = outs.params['o3r_center'].value
            
            ax1.plot(wave[fit_loc], outs.eval(x=wave[fit_loc]), 'k--')            
            yi, ya =  ax1.get_ylim()
            
            
            velocity = cnt + velocity_r[i-1]/c*(cnt)
            width = 100*1000/c*cnt
            
            lmin =   (velocity- width)/(1+z)*1e4
            lmax =   (velocity+ width)/(1+z)*1e4
            
        
                  
            ax1.plot( np.array([lmin, lmin]), np.array([yi,ya]),'r--')
            ax1.plot( np.array([lmax, lmax]), np.array([yi,ya]),'r--')
        
            ax2 = fig.add_axes([0.04 + cl*0.245,0.55-0.5*rw,0.2,0.25] )
            ax2.imshow(Image_cube[i-1,:,:],vmin=0, origin='low')
        
            image_axis(ax2, storage)       
    
        cl+=1
    
        if cl==4:
            cl=0
            rw=1



def Outflow_maps(storage, Sub_stor, n_up, w, w_d, region=False):
    
    z = storage['z_guess']
    ID = storage['X-ray ID']
    g = plt.figure()
    ax_spc = plt.axes([ 0.1, 0.6 , 0.85, 0.3])
    ax_nr = plt.axes([ 0.1, 0.1 , 0.35, 0.4])  
    ax_wd = plt.axes([ 0.6, 0.1 , 0.35, 0.4])
    
    wave = storage['obs_wave'].copy()
    flux = storage['1D_spectrum'].copy()
    
    out = storage['1D_fit_OIII_mul']
    plotting_OIII(wave, flux, ax_spc, out, 'mul',z)
        
    wave = Sub_stor['rest wave'].copy()*(1+z)/1e4
    
    narrow_ind = np.where( (wave>w) & (wave< n_up) )[0] 
    
    ax_spc.fill_between(np.array([w/(1+z)*1e4,n_up/(1+z)*1e4]), np.array([0,0]), np.array([18000,18000]), color='red', alpha=0.2)    
    ax_spc.fill_between(np.array([w_d/(1+z)*1e4,w/(1+z)*1e4]), np.array([0,0]), np.array([18000,18000]), color='blue', alpha=0.2)
    
    
    
    flx_cube = Sub_stor['flux'].copy()
    flx_cube = np.ma.array(data = flx_cube.data, mask = Sub_stor['sky_clipped'], fill_value=0)  
    
    y = int(storage['Median_stack_white_Center_data'][1])
    x = int(storage['Median_stack_white_Center_data'][2])
    if region == False:
        narrow = np.ma.sum(flx_cube[narrow_ind, :,:], axis=(0))    
    
    elif region == True:
        
        narrow = np.ma.sum(flx_cube[narrow_ind, x-15:x+15, y-15:y+15], axis=(0))    
    
    spax = storage['Signal_mask'].copy()
    spax = spax[0,:,:]
    
    mask_nan =  np.ma.masked_invalid(narrow).mask   
    spax_tot = np.logical_or( mask_nan, spax) 
    masked_narrow = np.ma.array(data= narrow, mask= spax_tot)
    min_nar = np.ma.min(masked_narrow)
    max_nar = np.ma.max(masked_narrow)
   
    ax_nr.imshow(narrow, origin='low', aspect='equal', vmin=0, vmax = max_nar)
    
    if region == False:
        ax_nr.plot(storage['Median_stack_white_Center_data'][1],storage['Median_stack_white_Center_data'][2], 'ro')
    
    elif region == True:
        ax_nr.plot(15,15, 'ro')
    
    wave = Sub_stor['rest wave'].copy()*(1+z)/1e4
    wide_ind = np.where( (wave>w_d) & (wave< w) )[0] 
    
    flx_cube = Sub_stor['flux'].copy()
    flx_cube = np.ma.array(data = flx_cube.data, mask = Sub_stor['sky_clipped'], fill_value=0) 
    
    if region == False:
        wide = np.ma.sum(flx_cube[wide_ind, :,:], axis=(0))     
    
    elif region == True:
        y = int(storage['Median_stack_white_Center_data'][1])
        x = int(storage['Median_stack_white_Center_data'][2])
        wide = np.ma.sum(flx_cube[wide_ind, x-15:x+15,y-15:y+15], axis=(0))  
         
        
    spax = storage['Signal_mask'].copy()
    spax = spax[0,:,:]
    
    mask_nan =  np.ma.masked_invalid(wide).mask   
    spax_tot = np.logical_or( mask_nan, spax) 
    masked_wide = np.ma.array(data= wide, mask= spax_tot)
    min_wid = np.ma.min(masked_wide)
    max_wid= np.ma.max(masked_wide)
    
    
    
    
    ax_wd.imshow(wide, origin='low', aspect='equal', vmin=0,vmax = max_wid )
    if region == False:
        ax_wd.plot(storage['Median_stack_white_Center_data'][1],storage['Median_stack_white_Center_data'][2], 'ro')
    
    elif region == True:
        ax_wd.plot(15,15, 'ro')
        
 
    plt.savefig(PATH + 'KMOS_SIN/Results_storage/OIII_Outflow/'+ID+'_outflow_plot.pdf')
    
    
    prhdr = storage['header']
    hdu = pyfits.PrimaryHDU(wide.data, header=prhdr)
    hdulist = pyfits.HDUList([hdu])
    
    hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII_Outflow/'+ID+'_OIII_outflow.fits', overwrite=True)

 
def plot_outflow(ID,ax, linew=1):
    # Outflow
    import glob
    afile = glob.glob(PATH+'KMOS_SIN/Results_storage/OIII_Outflow/'+ID+'_OIII_outflow.fits')
    
    if len(afile)>0:
        
        outflow = pyfits.getdata(afile[0])
        out_header = pyfits.getheader(afile[0])
        out_wcs= wcs.WCS(out_header).celestial    
        
        outflow[np.isnan(outflow)] = 0
        
        if (ID=='XID_587') | (ID=='XID_751'):
            outflow[:8,:] = 0
            outflow[-8:,:] = 0
            outflow[:,:8] = 0
            outflow[:,-8:] = 0
        
        if ID=='lid_1565':
            outflow[:25,:] = 0
            outflow[-25:,:] = 0
            outflow[:,:25] = 0
            outflow[:,-25:] = 0
        
        
        outflow_sm = smooth(outflow,1.)
        rms = np.std(outflow_sm- outflow_sm.mean(axis=0))
        if ID=='lid_1565':
            rms = 200
            
        
        
        ax.contour(outflow_sm, transform= ax.get_transform(out_wcs), colors='white',levels=(2*rms, 3*rms, 5*rms), alpha=0.7, linewidth=linew)                   

'''


