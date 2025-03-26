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
import matplotlib.cm
from brokenaxes import brokenaxes
from matplotlib.widgets import Slider, Button, RadioButtons, RangeSlider

class Viz_outreach:
    '''Class to visualize '''
    def __init__(self, path_res, map_hdu, indc = [1,0],vlim2= [-100,100], z=None ):
        """
        Load the fit and the data
        Returns
        -------
        """
        self.indc = indc
        self.z = z
        self.map_hdu_name = map_hdu
        self.vlim2 = vlim2

        with pyfits.open(path_res, memmap=False) as hdulist:
            self.map = []
            for its in map_hdu:
                self.map.append(hdulist[its].data[:,:,:])
            
            self.yeval = hdulist['yeval'].data
            self.flux = hdulist['flux'].data
            self.error = hdulist['error'].data

            names = [entry.name for entry in hdulist]
            print('HDUs:', names)
            if 'YEVAL_NAR' in names:
                print('yes')
                self.yeval_nar = hdulist['yeval_nar'].data
            if 'YEVAL_BRO' in names:
                self.yeval_bro = hdulist['yeval_bro'].data
      
            self.header = hdulist['PRIMARY'].header
            nwave = np.shape(self.yeval)[0]
            self.obs_wave = self.header['CRVAL3'] + (np.arange(nwave) - (self.header['CRPIX3'] - 1.0))*self.header['CDELT3']
        self.slice_val_ind = 2
        self.slice_val = self.obs_wave[2]

    def plot_general(self, i,j):
        ''' 
        General function to plot the updated spaxel spectrum in the plot
        '''
        self.axspec.cla()
        fluxm= self.flux[:,j,i]
        errorm= self.error[:,j,i]
        yevalm = self.yeval[:,j,i]
        self.axspec.plot(self.obs_wave, fluxm, drawstyle='steps-mid')
        self.axspec.plot(self.obs_wave, yevalm, 'r--')
        self.axspec.plot(self.obs_wave, errorm, 'k:')

        if hasattr(self, 'yeval_nar'):
            self.axspec.plot(self.obs_wave, self.yeval_nar[:,j,i], 'g--')
        if hasattr(self, 'yeval_bro'):
            self.axspec.plot(self.obs_wave, self.yeval_bro[:,j,i], 'b--')


        self.axspec.set_ylim(-0.1*max(yevalm), 1.1*max(yevalm))
        
        if self.xlims is not None:
            self.axspec.set_xlim(self.xlims[0], self.xlims[1])
            
            use = np.where( (self.obs_wave<self.xlims[1]) & (self.obs_wave>self.xlims[0]) )[0]
            self.axspec.text(self.obs_wave[use][10], 0.9*max(yevalm[use]), 'x='+str(i)+', y='+str(j) )

            if self.ylims:
                self.axspec.set_ylim(self.ylims[0], self.ylims[1]) 
            else:
                self.axspec.set_ylim(-0.1*max(yevalm[use]), 1.1*max(yevalm[use])) 
        
        
        limst = self.axspec.get_ylim()
        self.line_slice = self.axspec.vlines(self.slice_val,limst[0], limst[1], color='k', linestyle='dashed')


    def showme(self, xlims= None, vmax=1e-15, ylims=None):
        '''
        Function to initialize the whole setup and layout
        '''
        self.xlims = xlims
        self.ylims = ylims
        fig = plt.figure(figsize=(15.6, 8))
        fig.canvas.manager.set_window_title('vicube')
        gs = fig.add_gridspec(
            2, 3, height_ratios=(1,1), width_ratios=(1,1,1))

        self.ax0 = fig.add_subplot(gs[0, 0])
        self.ax1 = fig.add_subplot(gs[0, 1], sharex=self.ax0, sharey=self.ax0)
        self.ax2 = fig.add_subplot(gs[0, 2], sharex=self.ax0, sharey=self.ax0)
        self.axes = [self.ax0,self.ax1,self.ax2]

        self.ax0.set_title(self.map_hdu_name[0])
        self.ax1.set_title(self.map_hdu_name[1])
        self.ax2.set_title('Slice')
        self.axspec = fig.add_subplot(gs[1, :]) #brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])

        self.axes= self.axes[:len(self.map)]

        self.ax2.imshow(np.sum(self.flux[self.slice_val_ind-1:self.slice_val_ind+1,:,:],axis=0), origin='lower')
        k=0
        for ax, map,its in zip(self.axes,self.map, range(len(self.axes))):
            if its==1:
                cmap='coolwarm'
                map_ind= self.indc[k]
                _img_ = ax.imshow(map[map_ind,:,:], origin='lower',vmin=self.vlim2[0], vmax=self.vlim2[1], cmap=cmap)
            else:
                cmap='viridis'
                map_ind=self.indc[k]
                _img_ = ax.imshow(map[map_ind,:,:], origin='lower', cmap=cmap)
            
            
            k+=1
        
        self.xlimax = plt.axes([0.12, 0.95, 0.8, 0.03])
        self.xlim_slider = RangeSlider(self.xlimax, 'xlim', self.obs_wave[0], self.obs_wave[-1], valinit=(self.xlims[0], self.xlims[1]))
        self.xlim_slider.on_changed(self.slide_update)

        self.sliceax = plt.axes([0.12, 0.05, 0.8, 0.03])
        self.slice_slider = Slider(self.sliceax, 'wavelength', valmin=self.obs_wave[0],
                                   valmax=self.obs_wave[-1],
                                    valinit=self.obs_wave[0])
        
        self.slice_slider.on_changed(self.slide_update_slice)

        selector0  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector1  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector2  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        self.ax0.add_artist(selector0)
        self.ax1.add_artist(selector1)
        self.ax2.add_artist(selector2)
        selector0.set_visible(True)
        selector1.set_visible(True)
        selector2.set_visible(True)

        self.plot_general(50,50)
        #plt.tight_layout()

        def update_plot(event):
            '''
            Function to update the plot after a click 
            '''
            selector0.set_visible(True)
            self.i, self.j = int(event.xdata), int(event.ydata)

            #axspec = brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])
            #axspec = fig.add_subplot(gs[1, :])
            self.plot_general(self.i,self.j )
            selector0.set_xy((self.i-.5, self.j-.5))
            selector1.set_xy((self.i-.5, self.j-.5))
            selector2.set_xy((self.i-.5, self.j-.5))

        def hover(event):
            if (event.inaxes != self.ax0):
                selector0.set_visible(False),selector1.set_visible(False),selector2.set_visible(False)
                fig.canvas.draw_idle()
                return
            update_plot(event)
            fig.canvas.draw_idle()
        #fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_press_event", hover)    
        plt.show()   

    def slide_update(self, val):
        self.xlims = self.xlim_slider.val
        self.axspec.set_xlim(self.xlims[0], self.xlims[1])

    def slide_update_slice(self, val):
        self.slice_val = self.slice_slider.val
        self.slice_val_ind = np.argmin(np.abs(self.obs_wave-self.slice_val))
        self.ax2.cla()
        self.ax2.imshow(np.sum(self.flux[self.slice_val_ind-1:self.slice_val_ind+1,:,:],axis=0), origin='lower')
        self.line_slice.set_visible(False)
        limst = self.axspec.get_ylim()
        self.line_slice = self.axspec.vlines(self.slice_val,limst[0], limst[1], color='k', linestyle='dashed')

class Visualize:
    def __init__(self, path_res, map_hdu, indc = [1,1,0], z=None ):
        """
        Load the fit and the data
        Returns
        -------
        """
        self.indc = indc
        self.z = z
        self.map_hdu_name = map_hdu
        with pyfits.open(path_res, memmap=False) as hdulist:
            self.map = []
            for its in map_hdu:
                self.map.append(hdulist[its].data[:,:,:])
            
            self.yeval = hdulist['yeval'].data
            self.flux = hdulist['flux'].data
            self.error = hdulist['error'].data

            names = [entry.name for entry in hdulist]
            print('HDUs:', names)
            if 'YEVAL_NAR' in names:
                print('yes')
                self.yeval_nar = hdulist['yeval_nar'].data
            if 'YEVAL_BRO' in names:
                self.yeval_bro = hdulist['yeval_bro'].data
      
            self.header = hdulist['PRIMARY'].header
            nwave = np.shape(self.yeval)[0]
            self.obs_wave = self.header['CRVAL3'] + (np.arange(nwave) - (self.header['CRPIX3'] - 1.0))*self.header['CDELT3']

    def plot_general(self, i,j):
        self.axspec.cla()
        fluxm= self.flux[:,j,i]
        errorm= self.error[:,j,i]
        yevalm = self.yeval[:,j,i]
        self.axspec.plot(self.obs_wave, fluxm, drawstyle='steps-mid')
        self.axspec.plot(self.obs_wave, yevalm, 'r--')
        self.axspec.plot(self.obs_wave, errorm, 'k:')

        if hasattr(self, 'yeval_nar'):
            self.axspec.plot(self.obs_wave, self.yeval_nar[:,j,i], 'g--')
        if hasattr(self, 'yeval_bro'):
            self.axspec.plot(self.obs_wave, self.yeval_bro[:,j,i], 'b--')


        self.axspec.set_ylim(-0.1*max(yevalm), 1.1*max(yevalm))
        if self.z is not None:
            self.axspec.vlines(0.5008*(1+self.z),-0.1*max(yevalm), 1.1*max(yevalm), color='k', linestyle='dashed')
            self.axspec.vlines(0.6563*(1+self.z),-0.1*max(yevalm), 1.1*max(yevalm), color='k', linestyle='dashed')

        if self.xlims is not None:
            self.axspec.set_xlim(self.xlims[0], self.xlims[1])
            
            use = np.where( (self.obs_wave<self.xlims[1]) & (self.obs_wave>self.xlims[0]) )[0]
            self.axspec.text(self.obs_wave[use][10], 0.9*max(yevalm[use]), 'x='+str(i)+', y='+str(j) )

            if self.ylims:
                self.axspec.set_ylim(self.ylims[0], self.ylims[1]) 
            else:
                self.axspec.set_ylim(-0.1*max(yevalm[use]), 1.1*max(yevalm[use])) 
        
        if self.z is not None:
            limst = self.axspec.get_ylim()
            self.axspec.vlines(0.5008*(1+self.z),limst[0], limst[1], color='k', linestyle='dashed')


    def showme(self, xlims= None, vmax=1e-15, ylims=None):
        self.xlims = xlims
        self.ylims = ylims
        fig = plt.figure(figsize=(15.6, 8))
        fig.canvas.manager.set_window_title('vicube')
        gs = fig.add_gridspec(
            2, 3, height_ratios=(1,1), width_ratios=(1,1,1))

        self.ax0 = fig.add_subplot(gs[0, 0])
        self.ax1 = fig.add_subplot(gs[0, 1], sharex=self.ax0, sharey=self.ax0)
        self.ax2 = fig.add_subplot(gs[0, 2], sharex=self.ax0, sharey=self.ax0)
        self.axes = [self.ax0,self.ax1,self.ax2]

        self.ax0.set_title(self.map_hdu_name[0])
        self.ax1.set_title(self.map_hdu_name[1])
        self.ax2.set_title(self.map_hdu_name[2])
        self.axspec = fig.add_subplot(gs[1, :]) #brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])

        self.axes= self.axes[:len(self.map)]
        k=0
        for ax, map,its in zip(self.axes,self.map, range(len(self.axes))):
            if its==2:
                cmap='coolwarm'
                map_ind= self.indc[k]
            else:
                cmap='viridis'
                map_ind=self.indc[k]
            _img_ = ax.imshow(map[map_ind,:,:], origin='lower', cmap=cmap)
            k+=1
        
        self.xlimax = plt.axes([0.12, 0.03, 0.8, 0.03])
        self.xlim_slider = RangeSlider(self.xlimax, 'xlim', self.obs_wave[0], self.obs_wave[-1], valinit=(self.xlims[0], self.xlims[1]))
        self.xlim_slider.on_changed(self.slide_update)

        selector0  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector1  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        selector2  = matplotlib.patches.Rectangle(
            (14.5, 14.5), 1, 1, edgecolor='r', facecolor='none', lw=1.0)
        self.ax0.add_artist(selector0)
        self.ax1.add_artist(selector1)
        self.ax2.add_artist(selector2)
        selector0.set_visible(True)
        selector1.set_visible(True)
        selector2.set_visible(True)

        self.plot_general(50,50)
        plt.tight_layout()

        def update_plot(event):
            selector0.set_visible(True)
            i, j = int(event.xdata), int(event.ydata)

            #axspec = brokenaxes(xlims=xlims,  hspace=.01, subplot_spec=gs[1, :])
            #axspec = fig.add_subplot(gs[1, :])
            self.plot_general(i,j)
            selector0.set_xy((i-.5, j-.5))
            selector1.set_xy((i-.5, j-.5))
            selector2.set_xy((i-.5, j-.5))

        def hover(event):
            if (event.inaxes != self.ax0):
                selector0.set_visible(False),selector1.set_visible(False),selector2.set_visible(False)
                fig.canvas.draw_idle()
                return
            update_plot(event)
            fig.canvas.draw_idle()
        #fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_press_event", hover)    
        plt.show()   

    def slide_update(self, val):
        self.xlims = self.xlim_slider.val
        self.axspec.set_xlim(self.xlims[0], self.xlims[1])

    