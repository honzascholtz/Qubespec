#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 21:23:34 2022

@author: jansen
"""

#importing modules
from astropy.io import fits as pyfits
from .. import Graph_setup as gst
import numpy as np
nan= float('nan')

pi= np.pi
e= np.e
c= 3.*10**8

fsz = gst.graph_format()

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.widgets
import matplotlib.cm
from brokenaxes import brokenaxes

class Visualize:
    def __init__(self, path_res, map_hdu ):
        """
        Load the fit and the data
        Returns
        -------
        """
        with pyfits.open(path_res, memmap=False) as hdulist:
            self.map = []
            for its in map_hdu:
                self.map.append(hdulist[its].data[:,:,:])
            
            self.yeval = hdulist['yeval'].data
            self.flux = hdulist['flux'].data
            self.error = hdulist['error'].data

            self.header = hdulist['PRIMARY'].header
            nwave = np.shape(self.yeval)[0]
            self.obs_wave = self.header['CRVAL3'] + (np.arange(nwave) - (self.header['CRPIX3'] - 1.0))*self.header['CDELT3']

    def plot_general(self, i,j, axes):
        axes.cla()
        fluxm= self.flux[:,j,i]
        errorm= self.error[:,j,i]
        yevalm = self.yeval[:,j,i]
        axes.plot(self.obs_wave, fluxm, drawstyle='steps-mid')
        axes.plot(self.obs_wave, yevalm, 'r--')
        axes.plot(self.obs_wave, errorm, 'k:')
        axes.text(self.obs_wave[10], 0.9*max(yevalm), 'x='+str(i)+', y='+str(j) )

        axes.set_ylim(-0.01*max(yevalm), 1.1*max(yevalm))

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
                map_ind=0
            else:
                cmap='viridis'
                map_ind=1
            _img_ = ax.imshow(map[map_ind,:,:], origin='lower', cmap=cmap)
        
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

        self.plot_general(50,50, axspec)
        plt.tight_layout()

        def update_plot(event):
            selector0.set_visible(True)
            i, j = int(event.xdata), int(event.ydata)

            #axspec = brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])
            #axspec = fig.add_subplot(gs[1, :])
            self.plot_general(i,j , axspec)
            selector0.set_xy((i-.5, j-.5))
            selector1.set_xy((i-.5, j-.5))
            selector2.set_xy((i-.5, j-.5))

        def hover(event):
            if (event.inaxes != ax0):
                selector0.set_visible(False),selector1.set_visible(False),selector2.set_visible(False)
                fig.canvas.draw_idle()
                return
            update_plot(event)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()    