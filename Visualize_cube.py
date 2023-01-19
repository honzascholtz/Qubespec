#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:23:34 2022

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from astropy.io import fits as pyfits
from astropy import wcs
from astropy.table import Table, join, vstack
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from scipy.optimize import curve_fit

import Graph_setup as gst 

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


PATH='/Users/jansen/My Drive/Astro/'
fsz = gst.graph_format()


import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.widgets
import matplotlib.cm
import numpy as np
from brokenaxes import brokenaxes
np.random.seed(8614)

from astropy.modeling.models import Gaussian2D
from scipy.ndimage import gaussian_filter1d

def generate_data(mean_x, mean_y, std_x, std_y, theta, n_pixels):
    """Generate mock data and model
    Returns
    -------
    cube : float 3-d array
        The mock data.
    modl : float 3-d array
        The mock data, without Gaussian noise added.
    """

    # Generate a realistic galaxy image.
    g2d = Gaussian2D(
        1., mean_x, mean_y, std_x, std_y, theta=theta)
    x, y = [np.linspace(-5., 5., 30) for _ in 'ab']
    xx, yy = np.meshgrid(x, y) 
    img = g2d(xx, yy)

    # Generate a realistic galaxy spectrum X-D
    spec = np.zeros(n_pixels)
    wave = np.arange(spec.size)

    spec = 10./(wave/1000.+1.)**2
    spec[:155] /= 4.
    spec = gaussian_filter1d(spec, 10)
    for w,f,s in zip((250, 1022, 1257, 2200, 2250),
                     (5, -5, 12, 37, 6),
                     (15, 18, 18, 18, 25)):
        spec += f*np.exp(-0.5*((wave-w)/s)**2)

    # Observing the model galaxy with noise.
    modl = img[:, :, None] * spec[None, None, :]
    cube = np.random.normal(
        img[:, :, None]*spec[None, None, :],    # mean
        )
    img  = np.nanmedian(cube, axis=2)

    return cube, modl


import Plotting_tools_v2 as emplot
import Halpha_OIII_models as models
import IFU_tools_class as IFU
def plotting_haloiii(results,i,j, iter_map, axes):   
    index = iter_map[j,i]
    
    if index==0:
        print('not fit at '+str(i)+' '+str(j))
    else:
        index=int(index)
    
        i,j, res_spx,chains,obs_wave,flx_spax_m,error = results[index]
        
        lists = list(res_spx.keys())
        if 'Failed fit' in lists:
           return
        
        z = res_spx['popt'][0]
        
        if 'zBLR' in lists:
            modelfce = models.Halpha_OIII_BLR
        elif 'outflow_vel' not in lists:
            modelfce = models.Halpha_OIII
        elif 'outflow_vel' in lists and 'zBLR' not in lists:
            modelfce = models.Halpha_OIII_outflow
        
        emplot.plotting_Halpha_OIII(obs_wave, flx_spax_m, axes, res_spx, modelfce)
        #axes.set_title(str(IFU.sp.SNR_calc(obs_wave, flx_spax_m, error, res_spx ,'Hn')))

    
    

class Visualize:
    def __init__(self):
        
        cube, modl = generate_data(
            0., 0., 1., 2., np.pi/6., 2500)
        
        self.cube = cube
        self.modl = modl
        
    def load_data(self, path_maps, path_res, mode='Halpha_OIII'):
        
        with pyfits.open(path_maps, memmap=False) as hdulist:
            self.map_hal = hdulist['Halpha'].data
            self.map_nii = hdulist['NII'].data
            self.map_hb = hdulist['Hbeta'].data
            self.map_oiii = hdulist['OIII'].data
            self.map_oi = hdulist['OI'].data
            self.map_siir = hdulist['SIIr'].data
            self.map_siib = hdulist['SIIb'].data
            self.map_hal_ki = hdulist['Hal_kin'].data
            self.map_oiii_ki= hdulist['OIII_kin'].data
                
            self.header = hdulist['PRIMARY'].header
        
        with open(path_res, "rb") as fp:
            self.results= pickle.load(fp)
        
        self.iter_map = np.zeros_like(self.map_hal[0])
        for index in enumerate(self.results):
            it = index[0]
            i,j, res_spx,chains,wave,flx_spax_m,error = index[1]
            self.iter_map[i,j] = int(it)
        
        
    def showme(self,z=0):
    
        fig = plt.figure(figsize=(15.6, 8))
        fig.canvas.manager.set_window_title('vicube')
        gs = fig.add_gridspec(
            2, 3, height_ratios=(1,1), width_ratios=(1,1,1))
    
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        axspec = brokenaxes(xlims=((4800,5050),(6250,6350),(6500,6800)),  hspace=.01, subplot_spec=gs[1, :])
    
        _img_ = ax0.imshow(self.map_hal[0], origin='lower')
        _img_ = ax1.imshow(self.map_hal[1], origin='lower')
        _img_ = ax2.imshow(self.map_oiii[1], origin='lower')
        
        selector  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        ax1.add_artist(selector)
        selector.set_visible(True)
        
        plotting_haloiii(self.results, 50, 50, self.iter_map, axspec)
        
    
        def update_plot(event):
            selector.set_visible(True)
            i, j = int(event.xdata), int(event.ydata)
            
            axspec = brokenaxes(xlims=((4800,5050),(6250,6350),(6500,6800)),  hspace=.01, subplot_spec=gs[1, :])
            plotting_haloiii(self.results, i, j, self.iter_map, axspec)
            selector.set_xy((i-.5, j-.5))
            
            
        def hover(event):
            if (event.inaxes != ax1):
                selector.set_visible(False)
                fig.canvas.draw_idle()
                return
            
            update_plot(event)
            fig.canvas.draw_idle()
            
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.show()
        #plt.subplots_adjust(wspace=0, hspace=0)


    
    
    
    
    
    
    