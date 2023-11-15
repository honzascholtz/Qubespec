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

from . import Graph_setup as gst

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


from . import Plotting as emplot
from . import QubeSpec as IFU

def plot_general(i,j, obs_wave, flux, error, yeval , axes):
    axes.cla()
    fluxm= flux[:,j,i]
    errorm= error[:,j,i]
    yevalm = yeval[:,j,i]
    axes.plot(obs_wave, fluxm, drawstyle='steps-mid')
    axes.plot(obs_wave, yevalm, 'r--')
    axes.plot(obs_wave, errorm, 'k:')
    axes.set_ylim(-0.01*max(yevalm), 1.1*max(yevalm))


class Visualize:
    def __init__(self):

        cube, modl = generate_data(
            0., 0., 1., 2., np.pi/6., 2500)

        self.cube = cube
        self.modl = modl

    def load_data(self, path_res, map_hdu ):
        with pyfits.open(path_res, memmap=False) as hdulist:
            self.map = []
            for its in map_hdu:
                self.map.append(hdulist[its].data[0,:,:])
            
            self.yeval = hdulist['yeval'].data
            self.flux = hdulist['flux'].data
            self.error = hdulist['error'].data

            self.header = hdulist['PRIMARY'].header
            nwave = np.shape(self.yeval)[0]
            self.obs_wave = self.header['CRVAL3'] + (np.arange(nwave) - (self.header['CRPIX3'] - 1.0))*self.header['CDELT3']



    def showme(self, xlims= ((3,5.3)), vmax=1e-15):
                                  
        fig = plt.figure(figsize=(15.6, 8))
        fig.canvas.manager.set_window_title('vicube')
        gs = fig.add_gridspec(
            2, 3, height_ratios=(1,1), width_ratios=(1,1,1))

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(gs[0, 2], sharex=ax0, sharey=ax0)
        axes = [ax0,ax1,ax2]
        axspec = fig.add_subplot(gs[1, :]) #brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])

        axes= axes[:len(self.map)]
        for ax, map,its in zip(axes,self.map, range(len(axes))):
            if its==2:
                cmap='coolwarm'
            else:
                cmap='viridis'
            _img_ = ax.imshow(map, origin='lower', cmap=cmap)
        

        selector0  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector1  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector2  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        ax0.add_artist(selector0)
        ax1.add_artist(selector1)
        ax2.add_artist(selector2)
        selector0.set_visible(True)
        selector1.set_visible(True)
        selector2.set_visible(True)

        plot_general(50,50, self.obs_wave, self.flux, self.error, self.yeval , axspec)


        def update_plot(event):
            selector0.set_visible(True)
            i, j = int(event.xdata), int(event.ydata)

            #axspec = brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])
            #axspec = fig.add_subplot(gs[1, :])
            plot_general(i,j, self.obs_wave, self.flux, self.error, self.yeval , axspec)
            selector0.set_xy((i-.5, j-.5)),selector1.set_xy((i-.5, j-.5)),selector2.set_xy((i-.5, j-.5))

        def hover(event):
            if (event.inaxes != ax0):
                selector0.set_visible(False),selector1.set_visible(False),selector2.set_visible(False)
                fig.canvas.draw_idle()
                return
            update_plot(event)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()
        #plt.subplots_adjust(wspace=0, hspace=0)
