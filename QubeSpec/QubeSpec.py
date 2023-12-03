#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:36:26 2017

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import warnings

from astropy.io import fits
from astropy.wcs import wcs
from astropy.nddata import Cutout2D

from matplotlib.backends.backend_pdf import PdfPages
import pickle
import corner
import tqdm
import os
from astropy import stats
import multiprocess as mp
from multiprocess import Pool
from brokenaxes import brokenaxes
from astropy.utils.exceptions import AstropyWarning
import astropy.constants, astropy.cosmology, astropy.units, astropy.wcs
from astropy.table import Table


nan= float('nan')

pi= np.pi
e= np.e

c= 3.*10**8

OIIIr = 5008.24
OIIIb = 4960.3
Hal = 6564.52
NII_r = 6585.27
NII_b = 6549.86
Hbe = 4862.6

SII_r = 6731
SII_b = 6718.29
import time

from .Models import FeII_templates as pth
try:
    PATH_TO_FeII = pth.__path__[0]+ '/'
    with open(PATH_TO_FeII+'/Preconvolved_FeII.txt', "rb") as fp:
        Templates= pickle.load(fp)
    has_FeII = True
    print(has_FeII)

except FileNotFoundError:
    
    from .Models.FeII_comp import *
    its = preconvolve()
    print(its)



from . import Support as sp
from . import Plotting as emplot
from . import Fitting as emfit

from .Models import Halpha_OIII_models as HaO_models
from . import Background as bkg


# ============================================================================
#  Main class
# =============================================================================
class Cube:
    def __init__(self, Full_path='', z='', ID='', flag='', savepath='', Band='', norm=1e-13):
        import importlib
        importlib.reload(emfit )

        self.z = z
        self.ID = ID
        self.instrument = flag
        self.savepath = savepath
        self.band = Band
        self.Cube_path = Full_path
        self.flux_norm= norm

        
        if self.Cube_path !='':
            

            #print (Full_path)
            if self.instrument=='KMOS':
                filemarker = fits.open(Full_path)
                self.header = filemarker[1].header # FITS header in HDU 1
                flux_temp  = filemarker[1].data/norm

                filemarker.close()  # FITS HDU file marker closed

            elif self.instrument=='Sinfoni':
                filemarker = fits.open(Full_path)
                self.header = filemarker[0].header # FITS header in HDU 1
                flux_temp  = filemarker[0].data/norm

                filemarker.close()  # FITS HDU file marker closed

            elif self.instrument=='NIRSPEC_IFU':
                with fits.open(self.Cube_path, memmap=False) as hdulist:
                    try:
                        flux_temp = hdulist['SCI'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                        error = hdulist['ERR'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                    except Exception as _exc_:
                        print(_exc_)
                        flux_temp = hdulist['SCI'].data/norm * astropy.units.Unit('Jy')*1e-6
                        error = hdulist['ERR'].data/norm * astropy.units.Unit('Jy')*1e-6

                    w = wcs.WCS(hdulist[1].header)
                    self.header = hdulist[1].header
                    cube_wcs = astropy.wcs.WCS(hdulist['SCI'].header)
                    wave = cube_wcs.all_pix2world(0., 0., np.arange(cube_wcs._naxis[2]), 0)[2]
                    wave *= astropy.units.Unit(hdulist['SCI'].header['CUNIT3'])
                    if wave.unit==astropy.units.m:
                        wave = wave.to('um')
                    else:
                        wave *= 1.e6 # Somehow, units are autoconverted to m

                    error *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                    flux_temp *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                    flux_temp = flux_temp.to('1 erg/(s cm2 AA arcsec2)')/0.01
                    error = error.to('1 erg/(s cm2 AA arcsec2)')/0.01
                    flux_temp = flux_temp.value
                    self.error_cube = error.value

            elif self.instrument=='NIRSPEC_IFU_fl':
                with fits.open(self.Cube_path, memmap=False) as hdulist:
                    flux_temp = hdulist['SCI'].data/norm*1e4
                    self.error_cube = hdulist['ERR'].data/norm*1e4
                    self.w = wcs.WCS(hdulist[1].header)
                    self.header = hdulist[1].header

            elif self.instrument=='MIRI':
                with fits.open(self.Cube_path, memmap=False) as hdulist:
                    flux_temp = hdulist['SCI'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                    error = hdulist['ERR'].data/norm * astropy.units.Unit(hdulist['SCI'].header['BUNIT'])
                    self.w = wcs.WCS(hdulist[1].header)
                    self.header = hdulist[1].header
                    cube_wcs = astropy.wcs.WCS(hdulist['SCI'].header)
                    wave = cube_wcs.all_pix2world(0., 0., np.arange(cube_wcs._naxis[2]), 0)[2]
                    wave *= astropy.units.Unit(hdulist['SCI'].header['CUNIT3'])
                    if wave.unit==astropy.units.m:
                        wave = wave.to('um')
                    else:
                        wave *= 1.e6 # Somehow, units are autoconverted to m

                    error *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                    flux_temp *= (astropy.constants.c.to('AA/s') / wave.to('AA')**2)[:, None, None]
                    flux_temp = flux_temp.to('1 erg/(s cm2 AA arcsec2)')/0.01
                    error = error.to('1 erg/(s cm2 AA arcsec2)')/0.01
                    flux_temp = flux_temp.value
                    self.error_cube = error.value

            else:
                raise Exception('Instrument flag not understood')

            self.flux = np.ma.masked_invalid(flux_temp)   #  deal with NaN

            # Number of spatial pixels
            n_xpixels = self.header['NAXIS1']
            n_ypixels = self.header['NAXIS2']
            #  Number of spectral pixels
            n_spixels = self.header['NAXIS3']
            self.dim = [n_ypixels, n_xpixels, n_spixels]


            try:
                x = self.header['CDELT1']
            except:
                self.header['CDELT1'] = self.header['CD1_1']
            try:
                x = self.header['CDELT2']
            except:
                self.header['CDELT2'] = self.header['CD2_2']
            try:
                x = self.header['CDELT3']
            except:
                self.header['CDELT3'] = self.header['CD3_3']

            self.obs_wave = self.header['CRVAL3'] + (np.arange(n_spixels) - (self.header['CRPIX3'] - 1.0))*self.header['CDELT3']

            deg_per_pix_x = abs(self.header['CDELT1'])
            arc_per_pix_x = 1.*deg_per_pix_x*3600
            Xpix = self.header['NAXIS1']
            Xph = Xpix*arc_per_pix_x

            deg_per_pix_y = abs(self.header['CDELT2'])

            arc_per_pix_y = deg_per_pix_y*3600
            Ypix = self.header['NAXIS2']
            Yph = Ypix*arc_per_pix_y
                        
            if self.instrument=='NIRSPEC_IFU_fl':
                self.instrument= 'NIRSPEC_IFU'
            self.phys_size = np.array([Xph, Yph])

        else:
            self.save_dummy = 0


    def add_res(self, line_cat):
        '''
        Add catalogue line from a astorpy table - used in KASHz
        Parameters
        ----------
        line_cat : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        self.cat = line_cat


    def mask_emission(self):
        '''
        This function masks out all the OIII and HBeta emission.
        It is a ;egacy KASHz function.

        Returns
        -------
        None.

        '''
        OIIIa=  501./1e3*(1+self.z)
        OIIIb=  496./1e3*(1+self.z)
        Hbeta=  485./1e3*(1+self.z)
        width = 300/3e5*OIIIa

        mask =  self.flux.mask.copy()

        OIIIa_loc = np.where((self.obs_wave<OIIIa+width)&(self.obs_wave>OIIIa-width))[0]
        OIIIb_loc = np.where((self.obs_wave<OIIIb+width)&(self.obs_wave>OIIIb-width))[0]
        Hbeta_loc = np.where((self.obs_wave<Hbeta+width)&(self.obs_wave>Hbeta-width))[0]

        mask[OIIIa_loc,:,:] = True
        mask[OIIIb_loc,:,:] = True
        mask[Hbeta_loc,:,:] = True
        self.em_line_mask= mask


    def mask_sky(self,sig, mode=0):
        '''
        Old function to mask_sky - very rudmumentary only used as initial masking
        for KASHz.

        Parameters
        ----------
        sig : TYPE
            DESCRIPTION.
        mode : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''

        bins = np.linspace(0,2048,5)
        bins= np.array(bins, dtype=int)
        flux = np.ma.array(data=self.flux.data.copy(), mask= self.em_line_mask.copy())
        mask =  self.em_line_mask.copy()
        dim = self.dim


        if mode=='Hsin':
            use = np.where((self.obs_wave>1.81)&(self.obs_wave<1.46))[0]
            mask[use,:,:] = True

        for i in range(dim[0]):
            for j in range(dim[1]):
                stds = np.ma.std(flux[:,i,j])
                y = np.ma.mean(flux[:,i,j])


                sky = np.where((flux[:,i,j]< (y-stds*sig)) | (flux[:,i,j]> (y+stds*sig)))[0]

                mask[sky,i,j] = True

                if i==20 and j==20:
                    print(sky)


        self.sky_line_mask_em = mask


    def collapse_white(self, plot):
        '''
        This function creates a white light image of the cube.

        Parameters
        ----------
        plot : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        try:
            flux = np.ma.array(self.flux.data, mask=self.sky_line_mask_em)
        except:
            flux = self.flux

        median = np.ma.median(flux, axis=(0))

        if self.ID== 'ALESS_75':
            use = np.where((self.obs_wave < 1.7853) &(self.obs_wave > 1.75056))[0]

            use = np.append(use, np.where((self.obs_wave < 2.34) &(self.obs_wave > 2.31715))[0] )

            median = np.ma.median(flux[use,:,:], axis=(0))


        self.Median_stack_white = median


        if plot==1:
            plt.figure()
            plt.imshow(median,  origin='lower')
            plt.colorbar()



    def find_center(self, plot=0, extra_mask=0, manual=np.array([0])):
        '''
        Input:
            Storage,
            image name to be loaded
            Plot it?
            extra mask of the weird bright features
            manual - If manual is on then It will just return the manual inputs. Use if there is no continuum detected
        '''

        shapes = self.dim
        data = self.Median_stack_white

        # Create the x inputs for the curve_fit
        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)

        import scipy.optimize as opt

        if len(manual)==1:

            # If there is no extra mask -create a dummy extra mask
            if len(np.shape(extra_mask))<1.:
                extra_mask = self.flux.mask.copy()
                extra_mask = extra_mask[0,:,:]
                extra_mask[:,:] = False

            #############
            # Masking the edges based on the pixel scale.
            edges = self.flux.mask.copy() # Mask to collapse
            edges = edges[0,:,:]
            edges[:,:] = False
            try:
                pixel_scale = 1./(self.header['CD2_2']*3600)

            except:
                pixel_scale = 1./(self.header['CDELT2']*3600)


            if pixel_scale < 7:
                edges[:,0] = True
                edges[:,1] = True
                edges[:, -1] = True
                edges[:, -2] = True
                edges[1,:] = True
                edges[0,:] = True
                edges[-1,:] = True
                edges[-2,:] = True

                print ('Masking edges based on 0.2 scale')

            else:
                edges[:,:5] = True
                edges[:, -6:] = True
                edges[:5,:] = True
                edges[-6:,:] = True

                print ('Masking edges based on 0.1 scale')

            # Combining the edge and extra mask
            comb = np.logical_or(extra_mask, edges)

            # Setting all other nan values to 0 to avoid any troubles
            data.data[np.isnan(data.data)] = 0
            data= data.data

            # combining the data and masks
            masked_im = np.ma.array(data= data, mask=comb)

            # Finding the center of the contiunuum
            loc = np.ma.where(masked_im == masked_im.max())
            print ('location of the peak on the continuum',loc)

            #plt.figure()
            #plt.title('Find center - plot')
            #plt.imshow(masked_im, origin='low')

            # Setting the
            initial_guess = (data[loc[1][0], loc[0][0]],loc[1][0],loc[0][0],1,1,0,0)

            print ('Initial guesses', initial_guess)

            try:
                dm = (x,y)
                popt, pcov = opt.curve_fit(sp.twoD_Gaussian, dm, data.ravel(), p0=initial_guess)

                er = np.sqrt(np.diag(pcov))

                print ('Cont loc ', popt[1:3])
                print ('Cont loc er', er[1:3])

            except:
                print('Failed fit to the continuum - switching to the center of the cube')
                popt = np.zeros(7)
                popt[1] = int(self.dim[0]/2); popt[2] = int(self.dim[1]/2)

            if (popt[1]<3) | (popt[2]<3):
                print('Failed fit to the continuum - switching to the center of the cube')
                popt = np.zeros(7)
                popt[1] = int(self.dim[0]/2); popt[2] = int(self.dim[1]/2)

            self.center_data = popt



        else:
            manual = np.append(data[int(manual[0]), int(manual[1])], manual)
            manual = np.append(manual, np.array([2.,2.,0.5,0. ]))
            self.center_data = manual



    def choose_pixels(self, plot, rad= 0.6, flg=1, mask_manual=[0]):
        '''

        Choosing the pixels that will collapse into the 1D spectrum. Also this mask is used later
        Parameters
        ----------
        plot : TYPE
            DESCRIPTION.
        rad : TYPE, optional
            DESCRIPTION. The default is 0.6.
        flg : TYPE, optional
            DESCRIPTION. The default is 1.
        mask_manual : TYPE, optional
            DESCRIPTION. The default is [0].

        Returns
        -------
        None.

        '''

        center =  self.center_data[1:3].copy()
        shapes = self.dim

        print ('Center of cont', center)
        print ('Extracting spectrum from diameter', rad*2, 'arcseconds')

        # Creating a mask for all spaxels.
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        print('Pixel scale:', arc)
        print ('radius ', arc*rad)


        # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc*rad:
                    mask_catch[:,ix,iy] = False

        if len(mask_manual) !=1:
            for i in range(shapes[2]):
                mask_catch[i,:,:] = mask_manual.copy()

        # 587 have special extended stuff that I dont want to mask
        if flg=='K587':
            mask_catch[:,11:29 ,3:36 ] = False
            print ('extracting flux K band XID 587')

        elif flg=='H587':
            mask_catch[:,12:29 ,2:36 ] = False
            print ('extracting flux H band XID 587')

        elif flg=='K751':
            mask_catch[:,3:31 ,5:34 ] = False
            print ('extracting flux K band XID 751')

        elif flg=='K587sin':
            mask_catch[:,20:65 ,25:50 ] = False
            print ('extracting flux H band XID 587 sinfoni')

        self.Signal_mask = mask_catch


        if plot==1:
            plt.figure()
            plt.title('Selected Spaxels for 1D spectrum + Contours from 2D Gaus')
            plt.imshow(np.ma.array(data=self.Median_stack_white, mask=self.Signal_mask[0,:,:]), origin='lower')
            plt.colorbar()

            shapes = self.dim
            x = np.linspace(0, shapes[1]-1, shapes[1])
            y = np.linspace(0, shapes[0]-1, shapes[0])
            x, y = np.meshgrid(x, y)

            data_fit = sp.twoD_Gaussian((x,y), *self.center_data)

            plt.contour(x, y, data_fit.reshape(shapes[0], shapes[1]), 8, colors='w')



    def background_sub_spec_depricated(self, center, rad=0.6, manual_mask=[],smooth=25, plot=0):
        '''
        Background subtraction used when the NIRSPEC cube has still flux in the blank field.

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        rad : TYPE, optional
            DESCRIPTION. The default is 0.6.
        plot : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        self.collapsed_bkg, self.flux = bkg.background_sub_spec_depricated(self, center, rad=rad, manual_mask=manual_mask, smooth=smooth, plot=plot)

    def background_subtraction(self, box_size=(21,21), filter_size=(5,5), sigma_clip=5,\
                source_mask=[], wave_smooth=25, wave_range=None, plot=0, detection_threshold=3, **kwargs):
        '''
        Background subtraction used when the NIRSPEC cube has still flux in the blank field.

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        rad : TYPE, optional
            DESCRIPTION. The default is 0.6.
        plot : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        ------
        None.

        '''
        self.background, self.flux, self.flux_old = bkg.background_subtraction(self, box_size=box_size, \
                                   filter_size=filter_size, sigma_clip=sigma_clip, source_mask=source_mask, wave_smooth=wave_smooth,\
                                    wave_range= wave_range, detection_threshold=detection_threshold,plot=plot, **kwargs) 
        
        
    def background_sub_spec_gnz11(self, center, rad=0.6, manual_mask=[],smooth=25, plot=0):
        '''
        Background subtraction used when the NIRSPEC cube has still flux in the blank field.

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        rad : TYPE, optional
            DESCRIPTION. The default is 0.6.
        plot : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        self.collapsed_bkg, self.flux = bkg.background_sub_spec_gnz11(self, center, rad=rad, manual_mask=manual_mask, smooth=smooth, plot=plot)


    def D1_spectra_collapse(self, plot,rad= 0.6, addsave='', err_range=[0], boundary=2.4, plot_err= 0, flg=1):
        '''
        This function collapses the Cube to form a 1D spectrum of the galaxy

        Parameters
        ----------
        plot : TYPE
            DESCRIPTION.
        addsave : TYPE, optional
            DESCRIPTION. The default is ''.
        err_range : TYPE, optional
            DESCRIPTION. The default is [0].
        boundary : TYPE, optional
            DESCRIPTION. The default is 2.4.

        Returns
        -------
        None.

        '''
        
        self.choose_pixels( plot=plot, rad= rad, flg=flg, mask_manual=[0])
        # Loading the Signal mask - selects spaxel to collapse to 1D spectrum
        mask_spax = self.Signal_mask.copy()
        # Loading mask of the sky lines an bad features in the spectrum
        mask_sky_1D = self.sky_clipped_1D.copy()

        if self.instrument=='NIRSPEC_IFU':
            total_mask = np.logical_or( mask_spax, self.sky_clipped)
        else:
            total_mask = np.logical_or( mask_spax, self.flux.mask)
        # M
        flux = np.ma.array(data=self.flux.data, mask= total_mask)

        D1_spectra = np.ma.sum(flux, axis=(1,2))
        self.D1_spectrum = np.ma.array(data = D1_spectra.data, mask=mask_sky_1D)

        if plot==1:
            plt.figure()
            plt.title('Collapsed 1D spectrum from D1_spectra_collapse fce')

            plt.plot(self.obs_wave, D1_spectra.data, drawstyle='steps-mid', color='grey')
            plt.plot(self.obs_wave, np.ma.array(data= D1_spectra, mask=self.sky_clipped_1D), drawstyle='steps-mid')

            plt.ylabel('Flux')
            plt.xlabel('Observed wavelength')
               
        self.D1_spectrum_var = np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2))

        if self.instrument =='NIRSPEC_IFU':
            print('NIRSPEC mode of error calc')
            D1_spectrum_var_er = np.sqrt(self.D1_spectrum_var)

            self.D1_spectrum_er = sp.error_scaling(self.obs_wave, D1_spectra, D1_spectrum_var_er, err_range, boundary,\
                                                   exp=plot_err)

            if plot_err==1:
                f,ax = plt.subplots(1)
                ax.plot(self.obs_wave, D1_spectrum_var_er, label='Extension')
                ax.plot(self.obs_wave, self.D1_spectrum_er, label='rescaled')

                ax.legend(loc='best')
                ax.set_xlabel(r'$\lambda_{\rm obs}$ $\mu$m')
                ax.set_ylabel('Flux density')

        else:
            print('Other mode of error calc')
            if len(err_range)==2:
                error = stats.sigma_clipped_stats(D1_spectra[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2] \
                                                        *np.ones(len(D1_spectra))
                self.D1_spectrum_er = error

            elif len(err_range)==4:
                error1 = stats.sigma_clipped_stats(D1_spectra[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2]
                error2 = stats.sigma_clipped_stats(D1_spectra[(err_range[2]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[3])],sigma=3)[2]
                
                error = np.zeros(len(D1_spectra))
                error[self.obs_wave<boundary] = error1
                error[self.obs_wave>boundary] = error2
                self.D1_spectrum_er = error
            else:
                self.D1_spectrum_er = stats.sigma_clipped_stats(self.D1_spectra,sigma=3)[2]*np.ones(len(self.D1_spectrum)) #STD_calc(wave/(1+self.z)*1e4,self.D1_spectrum, self.band)* np.ones(len(self.D1_spectrum))

        if self.ID =='cdfs_220':
            self.D1_spectrum_er = 0.05*np.ones(len(self.D1_spectrum))
        if self.ID =='cid_346':
            self.D1_spectrum_er = 0.005*np.ones(len(self.D1_spectrum))

        if self.ID =='cdfs_584':
            self.D1_spectrum_er = 0.02*np.ones(len(self.D1_spectrum))

        Save_spec = np.zeros((4,len(D1_spectra)))
        Save_spec[0,:] = self.obs_wave
        Save_spec[1,:] = self.D1_spectrum
        Save_spec[2,:] = self.D1_spectrum_er.copy()
        Save_spec[3,:] = mask_sky_1D

        np.savetxt(self.savepath+self.ID+'_'+self.band+addsave+'_1Dspectrum.txt', Save_spec)



    def mask_JWST(self, plot, threshold=1e11, spe_ma=[]):
        '''
        Masking bad pixels in JWST NIRSPEC and MIRI observations.

        Parameters
        ----------
        plot : TYPE
            DESCRIPTION.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1e11.
        spe_ma : TYPE, optional
            DESCRIPTION. The default is [].
        dtype : TYPE, optional
            DESCRIPTION. The default is bool.

        Returns
        -------
        None.

        '''

        sky_clipped =  self.flux.mask.copy()
        sky_clipped[self.error_cube>threshold] = True

        sky_clipped_1D = self.flux.mask.copy()[:,10,10].copy()
        sky_clipped_1D[:] = False
        sky_clipped_1D[spe_ma] = True
        self.sky_clipped_1D = sky_clipped_1D
        self.sky_clipped = sky_clipped
        self.Sky_stack_mask = self.flux.mask.copy()



    def stack_sky(self,plot, spe_ma=np.array([], dtype=bool), expand=0):
        '''
        Masking sky for KMOS and SINFONI based observations

        Parameters
        ----------
        plot : TYPE
            DESCRIPTION.
        spe_ma : TYPE, optional
            DESCRIPTION. The default is np.array([], dtype=bool).
        expand : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        '''
        header = self.header
        ######
        # Finding the center of the Object
        center =  self.center_data[1:3].copy()

        ######
        # Defining other variable to be used
        wave = self.obs_wave.copy()#*1e4/(1+z) # Wavelength
        flux = self.flux.copy()   # Flux

        shapes = self.dim # The shapes of the image

        mask_nan = self.flux.mask.copy()

        mask_collapse = self.flux.mask.copy() # Mask to collapse
        mask_collapse[:,:,:] = False



        header  = self.header

        try:
            Radius = np.round(1.2/(header['CD2_2']*3600))

        except:
            Radius = np.round(1.2/(header['CDELT2']*3600))


        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< Radius:
                    mask_collapse[:,ix,iy] = True

        try:
            arc = (header['CD2_2']*3600)
        except:
            arc = (header['CDELT2']*3600)


        if arc< 0.17:
            print ('Masking special corners')
            mask_collapse[:,:,0] = True
            mask_collapse[:,:,1] = True
            mask_collapse[:,:,2] = True
            mask_collapse[:,:,3] = True

            mask_collapse[:,:,-1] = True
            mask_collapse[:,:,-2] = True
            mask_collapse[:,:,-3] = True
            mask_collapse[:,:,-4] = True

            mask_collapse[:,0,:] = True
            mask_collapse[:,1,:] = True
            mask_collapse[:,2,:] = True
            mask_collapse[:,3,:] = True

            mask_collapse[:,-1,:] = True
            mask_collapse[:,-2,:] = True
            mask_collapse[:,-3,:] = True
            mask_collapse[:,-4,:] = True

        elif arc> 0.17:
            print ('Masking special corners')
            mask_collapse[:,:,0] = True
            mask_collapse[:,:,-1] = True
            mask_collapse[:,0,:] = True
            mask_collapse[:,-1,:] = True

        # For Hband sinfoni data I subtracted the frames from each other to imitate the sky frame. Therefore there are two negative objects in the cube. I am masking those as well.
        if (self.band =='Hsin'):

            mask_collapse[:,76-10:76+8 ,83-10:83+10 ] = True #mask_collapse[:,2:10 ,0:12 ] = True #
            mask_collapse[:,21-10:21+8 ,27-10:27+10 ] = True
            print ('HB89 Hsin masking of negative objects')


        mask_collapse = np.logical_or(mask_collapse, mask_nan)

        self.Sky_stack_mask = mask_collapse
        ######

        # Collapsing the spaxels into sky spectrum
        stacked_sky = np.ma.sum(np.ma.array(data=flux.data, mask=mask_collapse), axis=(1,2))

        # Creating a brand new mask to mask weird features in the sky_spectrum
        std_mask = mask_collapse[:,7,7].copy()
        std_mask[:] = False

        weird = np.array([], dtype=int)

        if self.band=='YJ':
            print ('Masking based on YJ band')
            # Actually masking the weird features: Edges + that 1.27 annoying bit
            weird = np.where((wave>1.269) &(wave<1.285))[0]
            #weird = np.where((wave>1.28499) &(wave<1.285))[0]
            weird = np.append(weird, np.where((wave<1.010))[0])
            weird = np.append(weird, np.where((wave>1.35))[0])
            weird = np.append(weird, np.where((wave>1.31168)&(wave< 1.3140375))[0])

            # Masking the stacked sky not to intefere with calculating std
            std_mask[weird] = True

        elif  self.band=='H':
            print ('Masking based on H band')
            weird = np.where((wave<1.45))[0]
            weird = np.append(weird, np.where((wave>1.85))[0])
            weird = np.append(weird, np.where((wave>1.78351) &(wave<1.7885))[0])

        elif self.band=='K':
            print ('Masking based on K band')
            weird = np.where((wave<1.945))[0]
            weird = np.append(weird, np.where((wave>2.4))[0])

        elif self.band=='Hsin':
            print ('Masking based on H band with Sinfoni')
            weird = np.where((wave>1.82))[0]
            weird = np.append(weird, np.where((wave<1.45))[0])
            print (len(wave), len(weird))

        elif self.band=='Ysin':
            print ('Masking based on Y band with Sinfoni')
            weird = np.where((wave>1.35))[0]
            weird = np.append(weird, np.where((wave<1.11))[0])
            print (len(wave), len(weird))

        elif self.band=='Ksin':
            print ('Masking based on K band with Sinfoni')
            weird = np.where((wave>2.4))[0]
            weird = np.append(weird, np.where((wave<1.945))[0])
            print (len(wave), len(weird))

        elif self.band=='HKsin':
            print ('Masking based on HK band with Sinfoni')
            weird = np.where((wave>2.4))[0]
            weird = np.append(weird, np.where((wave<1.5))[0])
            print (len(wave), len(weird))

        weird = np.append(weird, spe_ma)

        std_mask[weird] = True

        stacked_sky_mask = np.ma.array(data = stacked_sky, mask=std_mask)


        # Masking sky lines
        y = np.zeros_like(stacked_sky)

        clip_w = 1.5

        ssma = stacked_sky_mask.data[np.invert(stacked_sky_mask.mask)]
        low, hgh = sp.conf(ssma)


        sky = np.where((stacked_sky<y+ low*clip_w) | (stacked_sky> y + hgh*clip_w))[0]

        sky_clipped =  self.flux.mask.copy()
        sky_clipped = sky_clipped[:,7,7]
        sky_clipped[sky] = True          # Masking the sky features
        sky_clipped[weird] = True        # Masking the weird features


        # Storing the 1D sky line mask into cube to be used later
        mask_sky = self.flux.mask.copy()



        if (self.ID=='xuds_316') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.202 ) & (wave > 1.199))[0]] = False

        if (self.ID=='cdfs_328') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.2658 ) & (wave > 1.2571))[0]] = False

        if (self.ID=='cdfs_220') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.286 ) & (wave > 1.2571))[0]] = False

        if (self.ID=='cdfs_751') & (self.band=='K'):
            sky_clipped[np.where((wave< 1.99 ) )[0]] = False

        if (self.ID=='cdfs_485') & (self.band=='H'):
            sky_clipped[np.where((wave< 1.71577 ) & (wave > 1.71300))[0]] = False

        if (self.ID=='xuds_398') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.2976 ) & (wave > 1.26))[0]] = False
        if (self.ID=='xuds_479') & (self.band=='YJ'):
            sky_clipped[np.where((wave< 1.04806 ) & (wave > 1.03849))[0]] = False

        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                mask_sky[:,ix,iy] = sky_clipped

        # Storing the 1D and 3D sky masks
        self.sky_clipped = mask_sky
        self.sky_clipped_1D = sky_clipped

        self.Stacked_sky = stacked_sky


        np.savetxt(self.savepath+'clp.txt', sky_clipped)



        if plot==1:
            mask_spax = self.Signal_mask.copy()
            mask_sky = self.sky_clipped.copy()

            total_mask = np.logical_or(mask_spax, mask_sky)


            flux_new = np.ma.array(data=self.flux.data, mask= total_mask)
            D1_spectra_new = np.ma.sum(flux_new, axis=(1,2))

            flux_old = np.ma.array(data=self.flux.data, mask=mask_spax)
            D1_spectra_old = np.ma.sum(flux_old, axis=(1,2))
            #######
            # Plotting the spaxels used to assemble the sky

            plt.figure()
            plt.imshow(np.ma.array(data=self.Median_stack_white, mask=mask_collapse[0,:,:]), origin='lower')
            plt.colorbar()
            plt.title('Removing the Galaxy')


            #######
            # 3 panel plot
            # 1 panel - The stacked spectrum
            # 2 panel - clipped sky lines
            # 3 panel - The collpsed 1D spectrum of the galaxy - No masking
            # 4 panel - The collpsed 1D spectrum of the galaxy - Final masking

            f, (ax1,ax2, ax3, ax4) = plt.subplots(4, sharex=True,sharey=True, figsize=(10,15))

            ax1.set_title('Stacked spectrum outside the galaxy')

            ax1.plot(wave, y+np.ones_like(stacked_sky)*hgh, 'g--')
            ax1.plot(wave, y+np.ones_like(stacked_sky)*low, 'g--')

            ax1.plot(wave, y+np.ones_like(stacked_sky)*hgh*clip_w, 'r--')
            ax1.plot(wave, y+np.ones_like(stacked_sky)*low*clip_w, 'r--')

            ax1.plot(wave, (stacked_sky), drawstyle='steps-mid', color='grey')
            ax1.plot(wave, np.ma.array(data=stacked_sky,mask=std_mask), drawstyle='steps-mid')
            ax1.set_ylabel('Sky spec')

            ax1.set_ylim(np.ma.min(np.ma.array(data=stacked_sky,mask=std_mask)), np.ma.max(np.ma.array(data=stacked_sky,mask=std_mask)))

            ax2.plot(wave, np.ma.array(data=stacked_sky, mask=sky_clipped), drawstyle='steps-mid')
            ax2.set_ylabel('Clipped sky')

            ax2.set_ylim(np.ma.min(np.ma.array(data=stacked_sky,mask=std_mask)), np.ma.max(np.ma.array(data=stacked_sky,mask=std_mask)))


            ax3.set_ylabel('Old spec')
            ax3.plot(wave, D1_spectra_old, drawstyle='steps-mid')

            ax4.plot(wave, D1_spectra_new, drawstyle='steps-mid')
            ax4.set_ylabel('New spec')

            ax1.set_ylim(-0.05,0.4)

            plt.tight_layout()

    def fitting_collapse_Halpha(self, plot=1, models = 'BLR', progress=True,er_scale=1, N=6000, priors= {'z':[0, 'normal_hat', 0,0,0,0],\
                                                                                       'cont':[0,'loguniform',-3,1],\
                                                                                       'cont_grad':[0,'normal',0,0.3], \
                                                                                       'Hal_peak':[0,'loguniform',-3,1],\
                                                                                       'BLR_Hal_peak':[0,'loguniform',-3,1],\
                                                                                       'zBLR':[0, 'normal', 0,0.003],\
                                                                                        'NII_peak':[0,'loguniform',-3,1],\
                                                                                       'Nar_fwhm':[300,'uniform',100,900],\
                                                                                       'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                       'BLR_offset':[-100,'normal',0,200],\
                                                                                        'SIIr_peak':[0,'loguniform',-3,1],\
                                                                                        'SIIb_peak':[0,'loguniform',-3,1],\
                                                                                        'Hal_out_peak':[0,'loguniform',-3,1],\
                                                                                        'NII_out_peak':[0,'loguniform',-3,1],\
                                                                                        'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                        'outflow_vel':[-50,'normal', 0,300]}):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()/er_scale
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        
        if models=='BLR':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='gal')
            
            
            Fits_blr = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_blr.fitting_Halpha(model='BLR')
            
            
            
            if Fits_blr.BIC-Fits_sig.BIC <-2:
                print('Delta BIC' , Fits_blr.BIC-Fits_sig.BIC, ' ')
                print('BICM', Fits_blr.BIC)
                self.D1_fit_results = Fits_blr.props
                self.D1_fit_chain = Fits_blr.chains
                self.D1_fit_model = Fits_blr.fitted_model
                self.D1_fit_full = Fits_blr
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
                self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = Fits_blr.BIC-Fits_sig.BIC
                
                
            else:
                print('Delta BIC' , Fits_blr.BIC-Fits_sig.BIC, ' ')
                
                self.D1_fit_results = Fits_sig.props
                self.D1_fit_chain = Fits_sig.chains
                self.D1_fit_model = Fits_sig.fitted_model
                self.D1_fit_full = Fits_sig
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = Fits_blr.BIC-Fits_sig.BIC
                
            '''This is for KASHz only!
            
            if (self.ID=='cid_111') | (self.ID=='xuds_254') | (self.ID=='xuds_379') | (self.ID=='xuds_235') | (self.ID=='sxds_620')\
                | (self.ID=='cdfs_751') | (self.ID=='cdfs_704') | (self.ID=='cdfs_757') | (self.ID=='sxds_787') | (self.ID=='sxds_1093')\
                    | (self.ID=='xuds_186') | (self.ID=='cid_1445') | (self.ID=='cdfs_38')| (self.ID=='cdfs_485')\
                        | (self.ID=='cdfs_588')  | (self.ID=='cid_932')  | (self.ID=='xuds_317'):
                    
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = BICM-BICS
                
                
            if (self.ID=='xuds_168') :
                 print('Delta BIC' , BICM-BICS, ' ')
                 print('BICM', BICM)
                 self.D1_fit_results = prop_blr
                 self.D1_fit_chain = flat_samples_blr
                 self.D1_fit_model = fitted_model_blr
                 self.z = prop_blr['popt'][0]
                 
                 self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
                 self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                 self.dBIC = BICM-BICS
            '''       
        elif models=='Outflow':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='gal')
            
            
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_out.fitting_Halpha(model='outflow')
            
            
            if Fits_out.BIC-Fits_sig.BIC <-2:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                print('BICM', Fits_out.BIC)
                self.D1_fit_results = Fits_out.props
                self.D1_fit_chain = Fits_out.chains
                self.D1_fit_model = Fits_out.fitted_model
                self.D1_fit_full = Fits_out
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
                self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
                
                
            else:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                
                self.D1_fit_results = Fits_sig.props
                self.D1_fit_chain = Fits_sig.chains
                self.D1_fit_model = Fits_sig.fitted_model
                self.D1_fit_full = Fits_sig
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
                self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
                
        elif models=='Single_only':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='gal')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
        
        elif models=='Outflow_only':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='outflow')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
            
        elif models=='BLR_only':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='BLR')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
            
        elif models=='QSO_BKPL':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha(model='QSO_BKPL')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hblr')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
        
        else:
            Exception('Sorry, models keyword not understood - AGN keywords allowed: BLR, Outflow, Single_only, BLR_only, Outflow_only, QSO_BKPL')
        
        labels= list(self.D1_fit_chain.keys())[1:]
            
        fig = corner.corner(
            sp.unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print('SNR hal ', self.SNR)
        print('SNR SII ', self.SNR_sii)
        
        
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_Halpha(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error', axres=ax2)
        
        self.fit_plot = [f,ax1,ax2]
        
        if (models=='BLR') | (models=='Outflow'):
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_Halpha(wave, flux, ax1a, Fits_sig.props , Fits_sig.fitted_model)
            try:
                emplot.plotting_Halpha(wave, flux, ax2a, Fits_blr.props , Fits_blr.fitted_model)
            except:
                emplot.plotting_Halpha(wave, flux, ax2a, Fits_out.props , Fits_out.fitted_model)
            
            
    def fitting_collapse_Halpha_OIII(self, plot=1, progress=True,N=6000,models='Single_only', priors={'z':[0,'normal_hat', 0, 0.,0,0],\
                                                                                                     'cont':[0,'loguniform', -3,1],\
                                                                                                     'cont_grad':[0,'normal', 0,0.2],\
                                                                                                     'Hal_peak':[0,'loguniform', -3,1],\
                                                                                                     'NII_peak':[0,'loguniform', -3,1],\
                                                                                                     'Nar_fwhm':[300,'uniform', 200,900],\
                                                                                                     'SIIr_peak':[0,'loguniform', -3,1],\
                                                                                                     'SIIb_peak':[0,'loguniform', -3,1],\
                                                                                                     'OIII_peak':[0,'loguniform', -3,1],\
                                                                                                     'Hbeta_peak':[0,'loguniform', -3,1],\
                                                                                                     'OI_peak':[0,'loguniform', -3,1],\
                                                                                                     'outflow_fwhm':[450,'uniform', 300,900],\
                                                                                                     'outflow_vel':[-50,'normal', -50,100],\
                                                                                                     'Hal_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'NII_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'OIII_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'OI_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'Hbeta_out_peak':[0,'loguniform', -3,1],\
                                                                                                     'zBLR':[0,'normal', 0,0.003],\
                                                                                                     'BLR_fwhm':[4000,'normal', 5000,500],\
                                                                                                     'BLR_Hal_peak':[0,'loguniform', -3,1],\
                                                                                                     'BLR_Hbeta_peak':[0,'loguniform', -3,1],\
                                                                                                     }):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        if models=='Single_only':   
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha_OIII(model='gal' )
            
            
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.z = Fits_sig.props['popt'][0]
            
            self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
    
            
            self.dBIC = 3
            
            
        elif models=='Outflow_only':   
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_Halpha_OIII(model='outflow' )
            
            
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.z = Fits_sig.props['popt'][0]
            
            self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
    
            
            self.dBIC = 3
            
        elif models=='BLR':   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
             Fits_sig.fitting_Halpha_OIII(model='BLR' )
             
             
             self.D1_fit_results = Fits_sig.props
             self.D1_fit_chain = Fits_sig.chains
             self.D1_fit_model = Fits_sig.fitted_model
             self.z = Fits_sig.props['popt'][0]
             
             self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
     
             
             self.dBIC = 3

        elif models=='BLR_simple':   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
             Fits_sig.fitting_Halpha_OIII(model='BLR_simple' )
             
             
             self.D1_fit_results = Fits_sig.props
             self.D1_fit_chain = Fits_sig.chains
             self.D1_fit_model = Fits_sig.fitted_model
             self.z = Fits_sig.props['popt'][0]
             
             self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
     
             
             self.dBIC = 3

        elif models=='QSO_BKPL':   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
             Fits_sig.fitting_Halpha_OIII(model='QSO_BKPL' )
             
             
             self.D1_fit_results = Fits_sig.props
             self.D1_fit_chain = Fits_sig.chains
             self.D1_fit_model = Fits_sig.fitted_model
             self.z = Fits_sig.props['popt'][0]
             '''
             self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
             '''
             
        
        else:
            raise Exception('models variable in fitting_collapse_Halpha_OIII not understood. Outflow variables: Single_only, Outflow_only, BLR, QSO_BKPL, BLR_simple')
        labels= list(self.D1_fit_chain.keys())[1:]
            
        fig = corner.corner(
            sp.unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        
        f = plt.figure(figsize=(10,4))
        if models=='QSO_BKPL':
            baxes = brokenaxes(xlims=((4700,5050),(6200,6800)),  hspace=.01)
        else:
            baxes = brokenaxes(xlims=((4800,5050),(6400,6800)),  hspace=.01)
                
        emplot.plotting_Halpha_OIII(wave, flux, baxes, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error')                             
        baxes.set_xlabel('Restframe wavelength (ang)')
        
        g = plt.figure(figsize=(10,4))
        if models=='QSO_BKPL':
            baxes_er = brokenaxes(xlims=((4700,5050),(6200,6800)),  hspace=.01)  
        else:
            baxes_er = brokenaxes(xlims=((4800,5050),(6400,6800)),  hspace=.01)
            
        
        y_tot = self.D1_fit_model(self.obs_wave, *self.D1_fit_results['popt'])
        
        baxes_er.plot(self.obs_wave/(1+self.D1_fit_results['popt'][0])*1e4, self.D1_spectrum-y_tot)
        baxes_er.set_ylim(-5*self.D1_spectrum_er[0], 5*self.D1_spectrum_er[0])
         
        self.fit_plot = [f,baxes]
        
        
        
    def fitting_collapse_OIII(self, plot=1, models='Outflow',simple=1, template=0, Hbeta_dual=0,progress=True, N=6000,priors= {'z': [0,'normal_hat',0, 0, 0,0],\
                                                                                                    'cont':[0,'loguniform',-3,1],\
                                                                                                    'cont_grad':[0,'normal',0,0.2], \
                                                                                                    'OIII_peak':[0,'loguniform',-3,1],\
                                                                                                    'OIII_out_peak':[0,'loguniform',-3,1],\
                                                                                                    'Nar_fwhm':[300,'uniform', 100,900],\
                                                                                                    'outflow_fwhm':[700,'uniform',600,2500],\
                                                                                                    'outflow_vel':[-50,'normal',0,200],\
                                                                                                    'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbeta_fwhm':[400,'uniform',120,7000],\
                                                                                                    'Hbeta_vel':[10,'normal', 0,200],\
                                                                                                    'Hbetan_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbeta_out_peak':[0,'loguniform',-3,1],\
                                                                                                    'Hbetan_fwhm':[300,'uniform',120,700],\
                                                                                                    'Hbetan_vel':[10,'normal', 0,100],\
                                                                                                    'Fe_peak':[0,'loguniform',-3,1],\
                                                                                                    'Fe_fwhm':[3000,'uniform',2000,6000]}):
        
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        ID = self.ID
        
        
        if models=='Outflow':
            if simple==0:
                Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_sig.fitting_OIII(model='gal', template=template,Hbeta_dual=Hbeta_dual )
                
                
                Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_out.fitting_OIII(model='outflow', template=template,Hbeta_dual=Hbeta_dual )
            elif simple==1:
                Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_sig.fitting_OIII(model='gal_simple' )
                
                
                Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_out.fitting_OIII(model='outflow_simple' )
            
            if Fits_out.BIC-Fits_sig.BIC <-2:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                print('BICM', Fits_out.BIC)
                self.D1_fit_results = Fits_out.props
                self.D1_fit_chain = Fits_out.chains
                self.D1_fit_model = Fits_out.fitted_model
                self.D1_fit_full = Fits_out
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
                
                
            else:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                
                self.D1_fit_results = Fits_sig.props
                self.D1_fit_chain = Fits_sig.chains
                self.D1_fit_model = Fits_sig.fitted_model
                self.D1_fit_full = Fits_sig
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
            '''
            if (ID=='cdfs_751') | (ID=='cid_40') | (ID=='xuds_068') | (ID=='cdfs_51') | (ID=='cdfs_614')\
                | (ID=='xuds_190') | (ID=='cdfs_979') | (ID=='cdfs_301')| (ID=='cid_453') | (ID=='cid_61') | (ID=='cdfs_254')  | (ID=='cdfs_427'):
                    
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_sig
                self.D1_fit_chain = flat_samples_sig
                self.D1_fit_model = fitted_model_sig
                self.z = prop_sig['popt'][0]
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
                
                
                
            if ID=='cid_346':
                print('Delta BIC' , BICM-BICS, ' ')
                self.D1_fit_results = prop_out
                self.D1_fit_chain = flat_samples_out
                self.D1_fit_model = fitted_model_out
                self.z = prop_out['popt'][0]
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
                self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
                self.dBIC = BICM-BICS
            '''
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_OIII(wave, flux, ax1a, Fits_sig.props , Fits_sig.fitted_model)
            emplot.plotting_OIII(wave, flux, ax2a, Fits_out.props , Fits_out.fitted_model)
            
            
        elif models=='Single_only':
            if simple==0:
                Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_sig.fitting_OIII(model='gal', template=template,Hbeta_dual=Hbeta_dual )
            elif simple==1:
                Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_sig.fitting_OIII(model='gal_simple' )
               
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = 3
            
        elif models=='Outflow_only':
            if simple==0:
                Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_out.fitting_OIII(model='outflow', template=template,Hbeta_dual=Hbeta_dual )
            elif simple==1:
                Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
                Fits_out.fitting_OIII(model='outflow_simple')
               
            print('BICM', Fits_out.BIC)
            self.D1_fit_results = Fits_out.props
            self.D1_fit_chain = Fits_out.chains
            self.D1_fit_model = Fits_out.fitted_model
            self.D1_fit_full = Fits_out
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = 3
            
        elif models=='QSO':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_OIII(model='QSO', template=template)
              
               
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  3
            self.SNR_hb =  3
            self.dBIC = 3
            
            
        
        elif models=='QSO_bkp':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_OIII(model='QSO_BKPL', template=template)
              
               
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  3
            self.SNR_hb =  3
            self.dBIC = 3
            
        else:
            Exception('Sorry, models keyword not understood: QSO, QSO_BKPL, Outflow_only, Single_only, Outflow')
        
        
        labels= list(self.D1_fit_results.keys())
        print(labels)
        fig = corner.corner(
            sp.unwrap_chain(self.D1_fit_chain), 
            labels=labels[1:],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print(self.SNR)
        print(self.SNR_hb)
        
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_OIII(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model, error=error, residual='error', axres=ax2, template=template)
        
        self.fit_plot = [f,ax1,ax2]  
            
    
    def fitting_collapse_general(self,fitted_model, labels, priors, logprior, nwalkers=64,use=np.array([]), N=6000 ):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        ID = self.ID
        if len(use)==0:
            use = np.linspace(0, len(wave)-1, len(wave), dtype=int)
            
        Fits_gen = emfit.Fitting(wave[use], flux[use], error[use], z, priors=priors, N=10000)
        Fits_gen.fitting_general(fitted_model, labels, logprior, nwalkers=nwalkers)
        
        self.D1_fit_results = Fits_gen.props
        self.D1_fit_chain = Fits_gen.chains
        self.D1_fit_model = Fits_gen.fitted_model
        self.z = Fits_gen.props['popt'][0]
            
            
        fig = corner.corner(
            sp.unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})

    def fitting_collapse_ppxf(self):
        import os
        try:
            os.mkdir(self.savepath+'PRISM_1D')
        except:
            print('Folder structure already exists')
        try:
            os.mkdir(self.savepath+'PRISM_1D/prism_clear')
        except:
            print('Folder structure already exists')

        sp.jadify(self.savepath+'PRISM_1D/prism_clear/100000', 'prism_clear', self.obs_wave, self.D1_spectrum.data/(1e-7*1e4)*self.flux_norm, err=self.D1_spectrum_er.data/(1e-7*1e4)*self.flux_norm, mask=np.zeros_like(self.obs_wave),
                        overwrite=True, descr=None, author='jscholtz', verbose=False)
        import yaml
        from yaml.loader import SafeLoader

        # Open the file and load the file
        with open('/Users/jansen/My Drive/MyPython/Qubespec/QubeSpec/jadify_temp/r100_jades_deep_hst_v3.1.1_template.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)

        data['dirs']['data_dir'] = self.savepath+'PRISM_1D/'
        data['dirs']['output_dir'] = self.savepath+'PRISM_1D/'
        data['ppxf']['redshift_table'] = self.savepath+'PRISM_1D/redshift_1D.csv'

        with open(self.savepath+'/PRISM_1D/R100_1D_setup_test.yaml', 'w') as f:
            data = yaml.dump(data, f, sort_keys=False, default_flow_style=True)
        from . import jadify_temp as pth
        PATH_TO_jadify = pth.__path__[0]+ '/'
        filename = PATH_TO_jadify+ 'red_table_template.csv'
        redshift_cat = Table.read(filename)

        redshift_cat['ID'][0] = 100000
        redshift_cat['z_visinsp'][0] = self.z
        redshift_cat['z_phot'][0] = self.z
        redshift_cat['z_bagp'][0] = self.z
        redshift_cat.write(self.savepath+'PRISM_1D/redshift_1D.csv',overwrite=True)

        import nirspecxf
        id = 100000
        config100 = nirspecxf.NIRSpecConfig(self.savepath+'PRISM_1D/R100_1D_setup_manual.yaml')
        ns, _ = nirspecxf.process_object_id(id, config100)

        self.D1_ppxf = ns

    def save_fit_info(self):
        results = [self.D1_fit_results, self.dBIC, self.SNR]

        with open(self.savepath+self.ID+'_'+self.band+'_best_fit.txt', "wb") as fp:
            pickle.dump( results,fp)

        with open(self.savepath+self.ID+'_'+self.band+'_best_chain.txt', "wb") as fp:
            pickle.dump( self.D1_fit_chain,fp)


    def astrometry_correction_HST(self, img_file):
        '''
        Correcting the Astrometry of the Cube. Fits a 2D Gaussian to the HST image and assumes that the
        HST and Cube centroids are in the same location.
        '''

        img=fits.getdata(img_file)
        img_wcs= wcs.WCS(img_file).celestial
        hdr=fits.getheader(img_file)

        # Sie of the new image - same as the cube
        new_size = self.phys_size# size of the cube
        # Finding the pixel scale of the HST
        try:
            pixscale=abs(hdr['CD1_1']*3600)

        except:
            pixscale=abs(hdr['CDELT1']*3600)

        # Loading the Catalogue coordinates - Chris sometimes uses ra and sometimes RA
        Cat = self.cat
        try:
            Ra_opt = Cat['ra']
            Dec_opt = Cat['dec']

        except:
            Ra_opt = Cat['RA']
            Dec_opt = Cat['DEC']

        # Finding the position of the Galaxy in pix scale
        opt_world= np.array([[Ra_opt,Dec_opt]])
        opt_pixcrd = img_wcs.wcs_world2pix(opt_world, 0) # WCS transform
        opt_x= (opt_pixcrd[0,0]) # X pixel
        opt_y= (opt_pixcrd[0,1]) # Y pixel

        position = np.array([opt_x, opt_y])

        # Cutting out an image from the bigger image
        cutout = Cutout2D(img, position, new_size/pixscale, wcs=img_wcs,mode='partial')

        # Extracting the new image and the new wcs solution
        img=(cutout.data).copy()
        img_wcs=cutout.wcs

        # To avoid weird things on side of the stamps
        img[np.isnan(img)] = 0

        # Finding the dimensions of the new image - need to create new XY grid for the 2D Gaussian
        shapes = np.array(np.shape(img))

        # Finding the initial condition- location of the maximum of the image
        loc = np.where(img == img.max()) # Assumes that our AGN is the brightest thing in the image

        initial_guess = (np.max(img),loc[1][0],loc[0][0],5. , 5. ,0,0)

        # XID 522 in KMOS is just damn aweful
        if self.ID == 'XID_522':
            print( 'XID_522: Changing initial conditions for Astrometry corrections')
            initial_guess = (img[34,32],34,32,5. , 5. ,0,0)

            print (initial_guess)


        #print 'Initial guesses', initial_guess
        import scipy.optimize as opt

        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)

        popt, pcov = opt.curve_fit(sp.twoD_Gaussian, (x, y), img.ravel(), p0=initial_guess)
        er = np.sqrt(np.diag(pcov))
        print ('HST Cont', popt[1:3])
        print ('HST Cont er', er[1:3])
        popt_HST  = popt.copy()
        # Extrating the XY coordinates of the center. Will be plotted later to check accuracy
        center_C= popt[1:3]

        center_C1 = np.zeros(2)
        center_C1[0] = center_C[1]
        center_C1[1] = center_C[0]

        # Finding the Accurate HST based optical coordinates
        center_global =  img_wcs.wcs_pix2world(np.array([center_C]), 0)[0]

        cube_center = self.center_data[1:3]

        # Old Header for plotting purposes
        Header_cube_old = self.header.copy()

        # New Header
        Header_cube_new = self.header.copy()
        Header_cube_new['CRPIX1'] = cube_center[0]+1
        Header_cube_new['CRPIX2'] = cube_center[1]+1
        Header_cube_new['CRVAL1'] = center_global[0]
        Header_cube_new['CRVAL2'] = center_global[1]

        # Saving new coordinates and the new header
        self.HST_cent = center_global
        self.header = Header_cube_new

        ###############
        # DO check plots:
        f  = plt.figure(figsize=(10,7))
        ax = f.add_subplot(121, projection=img_wcs)

        rms = np.std(img - img.mean(axis=0))

        ax.imshow((img), origin='low')#, vmin=0, vmax=3*rms)
        ax.set_autoscale_on(False)
        ax.plot(center_C[0], center_C[1], 'r*')


        cube_wcs= wcs.WCS(Header_cube_new).celestial
        cont_map = self.Median_stack_white
        popt_cube = self.center_data

        x = np.linspace(0, shapes[1]-1, shapes[1])
        y = np.linspace(0, shapes[0]-1, shapes[0])
        x, y = np.meshgrid(x, y)

        data_fit = sp.twoD_Gaussian((x,y), *popt_cube)

        ax.contour( data_fit.reshape(shapes[0], shapes[1]), levels=(max(data_fit)*0.68,max(data_fit)*0.98),transform= ax.get_transform(cube_wcs), colors='r')

        cube_wcs= wcs.WCS(Header_cube_old).celestial

        ax.contour( data_fit.reshape(shapes[0], shapes[1]), levels=(max(data_fit)*0.68,max(data_fit)*0.98),transform= ax.get_transform(cube_wcs), colors='g')

        popt = cube_center
        cube_wcs= wcs.WCS(Header_cube_new).celestial

        ax = f.add_subplot(122, projection=cube_wcs)
        ax.imshow(cont_map, vmin=0, vmax= cont_map[int(popt[1]), int(popt[0])], origin='low')

        ax.plot(popt[0], popt[1],'r*')

        ax.contour(img, transform= ax.get_transform(img_wcs), colors='red',levels=(popt_HST[0]*0.68,popt_HST[0]*0.95), alpha=0.9)

        ax.set_xlim(40,60)
        ax.set_ylim(40,57)

    def astrometry_correction(self, Coord):
        '''
        Correcting the Astrometry of the Cube. Enter new Ra and Dec coordinates.
        '''

        cube_center = self.center_data[1:3]

        # Old Header for plotting purposes
        Header_cube_old = self.header.copy()

        # New Header
        Header_cube_new = self.header.copy()
        Header_cube_new['CRPIX1'] = cube_center[0]+1
        Header_cube_new['CRPIX2'] = cube_center[1]+1
        Header_cube_new['CRVAL1'] = Coord[0]
        Header_cube_new['CRVAL2'] = Coord[1]

        self.header = Header_cube_new

    def astrometry_correction_GAIA(self, path_to_gaia):
        '''
        Correcting the Astrometry of the Cube. Fits a 2D Gaussian to the HST image and assumes that the
        HST and Cube centroids are in the same location.
        '''

        Gaia = Table.read(path_to_gaia , format='ipac')

        cube_center = self.center_data[1:3]

        # Old Header for plotting purposes
        Header_cube_old = self.header.copy()

        # New Header
        Header_cube_new = self.header.copy()
        Header_cube_new['CRPIX1'] = cube_center[0]+1
        Header_cube_new['CRPIX2'] = cube_center[1]+1
        Header_cube_new['CRVAL1'] = float(Gaia['ra'])
        Header_cube_new['CRVAL2'] = float(Gaia['dec'])

        center_global= np.array([float(Gaia['ra']), float(Gaia['dec'])])

        # Saving new coordinates and the new header
        self.HST_center_glob = center_global
        self.header = Header_cube_new

    def report(self):
        '''
        Reporting the full results of D1_fit results

        Returns
        -------
        None.

        '''

        results = self.D1_fit_results
        print('')
        print('Model: ', results['name'])
        print('SNR of the line: ', self.SNR)
        print('dBIC of ', self.dBIC)

        for key in results.keys():

            if key == 'name' or key =='popt':
                print('')
            else:
                print(key, results[key])

    def unwrap_cube_prism(self, rad=0.4, add='',instrument='NIRSPEC', mask_manual=0, binning_pix=1, err_range=[0], boundary=2.4):
        '''
        Unwrapping the cube.

        Parameters
        ----------
        rad : TYPE, optional
            DESCRIPTION. The default is 0.4.
        sp_binning : TYPE, optional
            DESCRIPTION. The default is 'Nearest'.
        instrument : TYPE, optional
            DESCRIPTION. The default is 'KMOS'.
        add : TYPE, optional
            DESCRIPTION. The default is ''.
        mask_manual : TYPE, optional
            DESCRIPTION. The default is 0.
        binning_pix : TYPE, optional
            DESCRIPTION. The default is 1.
        err_range : TYPE, optional
            DESCRIPTION. The default is [0].
        boundary : TYPE, optional
            DESCRIPTION. The default is 2.4.

        Returns
        -------
        None.

        '''
        flux = self.flux.copy()
        Mask= self.sky_clipped_1D
        shapes = self.dim

        ThD_mask = self.sky_clipped.copy()
        Spax_mask = self.Sky_stack_mask[0,:,:]

# =============================================================================
#   Unwrapping the cube
# =============================================================================
        try:
            arc = (self.header['CD2_2']*3600)

        except:
            arc = (self.header['CDELT2']*3600)

        if instrument=='NIRSPEC':
            upper_lim = 0
            step = 1
            step = binning_pix
        if instrument=='NIRSPEC05':
            upper_lim = 0
            step = 2
            step = binning_pix
        x = range(shapes[0]-upper_lim)
        y = range(shapes[1]-upper_lim)


        print(rad/arc)
        h, w = self.dim[:2]
        center= self.center_data[1:3]

        mask = sp.create_circular_mask(h, w, center= center, radius= rad/arc)
        mask = np.invert(mask)
        try:
            if mask_manual.all !=1:
                mask=mask_manual
        except:
            print('Circular mask')

        Spax_mask = np.logical_or(np.invert(Spax_mask),mask)
        if self.instrument=='NIRSPEC_IFU':
            Spax_mask = mask.copy()
        import os
        try:
            os.mkdir(self.savepath+'PRISM_spaxel/prism_clear')
        except:
            print('Making directory failed. Maybe it exists already')
        for i in tqdm.tqdm(x):
            #i= i+step

            for j in y:
                #j=j+step
                if Spax_mask[i,j]==False:
                    #print i,j

                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False

                    if self.instrument=='NIRSPEC_IFU':
                        total_mask = np.logical_or(Spax_mask_pick, self.sky_clipped)
                        flx_spax_t = np.ma.array(data=flux.data,mask=total_mask)

                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=self.sky_clipped_1D)
                        nspaxel= np.sum(np.logical_not(total_mask[22,:,:]))
                        Var_er = np.sqrt(np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2))/nspaxel)

                        error = sp.error_scaling(self.obs_wave, flx_spax_m, Var_er, err_range, boundary,\
                                                   exp=0)

                    sp.jadify(self.savepath+'PRISM_spaxel/prism_clear/00'+str(i)+str(j), 'prism_clear', self.obs_wave, flx_spax_m.data/(1e-7*1e4)*self.flux_norm, err=error/(1e-7*1e4)*self.flux_norm, mask=np.zeros_like(self.obs_wave),
                        overwrite=True, descr=None, author='jscholtz', verbose=False)
                    
        

    def unwrap_cube(self, rad=0.4, sp_binning='Nearest', instrument='KMOS', add='', mask_manual=0, binning_pix=1, err_range=[0], boundary=2.4):
        '''
        Unwrapping the cube.

        Parameters
        ----------
        rad : TYPE, optional
            DESCRIPTION. The default is 0.4.
        sp_binning : TYPE, optional
            DESCRIPTION. The default is 'Nearest'.
        instrument : TYPE, optional
            DESCRIPTION. The default is 'KMOS'.
        add : TYPE, optional
            DESCRIPTION. The default is ''.
        mask_manual : TYPE, optional
            DESCRIPTION. The default is 0.
        binning_pix : TYPE, optional
            DESCRIPTION. The default is 1.
        err_range : TYPE, optional
            DESCRIPTION. The default is [0].
        boundary : TYPE, optional
            DESCRIPTION. The default is 2.4.

        Returns
        -------
        None.

        '''
        flux = self.flux.copy()
        Mask= self.sky_clipped_1D
        shapes = self.dim


        ThD_mask = self.sky_clipped.copy()
        z = self.z
        wv_obs = self.obs_wave.copy()

        msk = Mask.copy()
        Spax_mask = self.Sky_stack_mask[0,:,:]

        if (self.ID=='XID_587'):
            print ('Masking special corners')
            Spax_mask[:,0] = False
            Spax_mask[:,1] = False

            Spax_mask[:,-1] = False
            Spax_mask[:,-2] = False

            Spax_mask[0,:] = False
            Spax_mask[1,:] = False

            Spax_mask[-1,:] = False
            Spax_mask[-2,:] = False

# =============================================================================
#   Unwrapping the cube
# =============================================================================
        try:
            arc = (self.header['CD2_2']*3600)

        except:
            arc = (self.header['CDELT2']*3600)


        if arc> 0.17:
            upper_lim = 2
            step = 1

        elif arc< 0.17:
            upper_lim = 3
            step = 2
        if instrument=='NIRSPEC':
            upper_lim = 0
            step = 1
            step = binning_pix
        if instrument=='NIRSPEC05':
            upper_lim = 0
            step = 2
            step = binning_pix
        x = range(shapes[0]-upper_lim)
        y = range(shapes[1]-upper_lim)


        print(rad/arc)
        h, w = self.dim[:2]
        center= self.center_data[1:3]

        mask = sp.create_circular_mask(h, w, center= center, radius= rad/arc)
        mask = np.invert(mask)
        try:
            if mask_manual.all !=1:
                mask=mask_manual
        except:
            print('Circular mask')

        Spax_mask = np.logical_or(np.invert(Spax_mask),mask)
        if self.instrument=='NIRSPEC_IFU':
            Spax_mask = mask.copy()

        plt.figure()
        plt.imshow(np.ma.array(data=self.Median_stack_white, mask=Spax_mask), origin='lower')

        Unwrapped_cube = []
        for i in tqdm.tqdm(x):
            #i= i+step

            for j in y:
                #j=j+step
                if Spax_mask[i,j]==False:
                    #print i,j

                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    if sp_binning=='Nearest':
                        Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                    if sp_binning=='Single':
                        Spax_mask_pick[:, i, j] = False

                    if self.instrument=='NIRSPEC_IFU':
                        total_mask = np.logical_or(Spax_mask_pick, self.sky_clipped)
                        flx_spax_t = np.ma.array(data=flux.data,mask=total_mask)

                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=self.sky_clipped_1D)
                        nspaxel= np.sum(np.logical_not(total_mask[22,:,:]))
                        Var_er = np.sqrt(np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2))/nspaxel)

                        error = sp.error_scaling(self.obs_wave, flx_spax_m, Var_er, err_range, boundary,\
                                                   exp=0)

                    else:
                        flx_spax_t = np.ma.array(data=flux.data,mask=Spax_mask_pick)
                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)

                        error = stats.sigma_clipped_stats(flx_spax_m,sigma=3)[2] * np.ones(len(flx_spax))

                    Unwrapped_cube.append([i,j,flx_spax_m, error,wv_obs, z])


        print(len(Unwrapped_cube))
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "wb") as fp:
            pickle.dump(Unwrapped_cube, fp)

    def Spaxel_fitting_OIII_MCMC_mp(self,Ncores=(mp.cpu_count() - 1), priors= {'cont':[0,-3,1],\
                                                  'cont_grad':[0,-0.01,0.01], \
                                                  'OIII_peak':[0,-3,1],\
                                                  'OIII_out_peak':[0,-3,1],\
                                                  'OIII_fwhm':[300,100,900],\
                                                  'OIII_out':[700,600,2500],\
                                                  'outflow_vel':[-200,-900,600],\
                                                  'Hbeta_peak':[0,-3,1],\
                                                  'Hbeta_fwhm':[200,120,800],\
                                                  'Hbeta_vel':[10,-200,200],\
                                                  'Hbetan_peak':[0,-3,1],\
                                                  'Hbetan_fwhm':[300,120,700],\
                                                  'Hbetan_vel':[10,-100,100],\
                                                  'Fe_peak':[0,-3,2],\
                                                  'Fe_fwhm':[3000,2000,6000]}):


        import pickle
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')


        if Ncores<1:
            Ncores=1
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)
        #for i in range(len(Unwrapped_cube)):
            #results.append( emfit.Fitting_OIII_unwrap(Unwrapped_cube[i], self.obs_wave, self.z))

        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_OIII_unwrap, Unwrapped_cube )

        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

    def Spaxel_fitting_OIII_2G_MCMC_mp(self,add='', Ncores=(mp.cpu_count() - 1), priors= {'cont':[0,-3,1],\
                                                                    'cont_grad':[0,-0.01,0.01], \
                                                                    'OIII_peak':[0,-3,1],\
                                                                    'OIII_out_peak':[0,-3,1],\
                                                                    'OIII_fwhm':[300,100,900],\
                                                                    'OIII_out':[900,600,2500],\
                                                                    'outflow_vel':[-200,-900,600],\
                                                                    'Hbeta_peak':[0,-3,1],\
                                                                    'Hbeta_fwhm':[200,120,7000],\
                                                                    'Hbeta_vel':[10,-200,200],\
                                                                    'Hbetan_peak':[0,-3,1],\
                                                                    'Hbetan_fwhm':[300,120,700],\
                                                                    'Hbetan_vel':[10,-100,100],\
                                                                    'Fe_peak':[0,-3,2],\
                                                                    'Fe_fwhm':[3000,2000,6000]}):
        import pickle
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)

        print('import of the unwrap cube - done')
        print(len(Unwrapped_cube))

        if Ncores<1:
            Ncores=1
        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_OIII_2G_unwrap, Unwrapped_cube )

        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII_2G'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

    def Spaxel_fitting_Halpha_MCMC_mp(self, add='',Ncores=(mp.cpu_count() - 1),priors={'cont':[0,-3,1],\
                                                     'cont_grad':[0,-0.01,0.01], \
                                                     'Hal_peak':[0,-3,1],\
                                                     'BLR_peak':[0,-3,1],\
                                                     'NII_peak':[0,-3,1],\
                                                     'Nar_fwhm':[300,100,900],\
                                                     'BLR_fwhm':[4000,2000,9000],\
                                                     'zBLR':[-200,-900,600],\
                                                     'SIIr_peak':[0,-3,1],\
                                                     'SIIb_peak':[0,-3,1],\
                                                     'Hal_out_peak':[0,-3,1],\
                                                     'NII_out_peak':[0,-3,1],\
                                                     'outflow_fwhm':[600,300,1500],\
                                                     'outflow_vel':[-50, -300,300]}):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)


        if Ncores<1:
            Ncores=1

        with Pool(Ncores) as pool:
            cube_res = pool.map(emfit.Fitting_Halpha_unwrap, Unwrapped_cube)


        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))

    def Spaxel_fitting_Halpha_OIII_MCMC_mp(self,add='',Ncores=(mp.cpu_count() - 2),models='Single', priors= {'cont':[0,-3,1],\
                                                          'cont_grad':[0,-10.,10], \
                                                          'OIII_peak':[0,-3,1],\
                                                           'Nar_fwhm':[300,100,900],\
                                                           'OIII_vel':[-100,-600,600],\
                                                           'Hbeta_peak':[0,-3,1],\
                                                           'Hal_peak':[0,-3,1],\
                                                           'NII_peak':[0,-3,1],\
                                                           'Nar_fwhm':[300,150,900],\
                                                           'SIIr_peak':[0,-3,1],\
                                                           'SIIb_peak':[0,-3,1],\
                                                           'OI_peak':[0,-3,1]}):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')

        #results = []
        #for i in range(len(Unwrapped_cube)):
        #    results.append( emfit.Fitting_Halpha_unwrap(Unwrapped_cube[i]))
        #cube_res = results

        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump( priors,fp)

        if Ncores<1:
            Ncores=1
        if models=='Single':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_unwrap, Unwrapped_cube)
        elif models=='BLR':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_AGN_unwrap, Unwrapped_cube)

        elif models=='outflow_both':
            with Pool(Ncores) as pool:
                cube_res =pool.map(emfit.Fitting_Halpha_OIII_outflowboth_unwrap, Unwrapped_cube)
        else:
            raise Exception('models variable not understood. Options are: Single, BLR and outflow_both')

        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))

    def Spaxel_ppxf(self, ncpu=2):
        import glob
        import yaml
        from yaml.loader import SafeLoader

        # Open the file and load the file
        with open('/Users/jansen/My Drive/MyPython/Qubespec/QubeSpec/jadify_temp/r100_jades_deep_hst_v3.1.1_template.yaml') as f:
            data = yaml.load(f, Loader=SafeLoader)

        data['dirs']['data_dir'] = self.savepath+'PRISM_spaxel/'
        data['dirs']['output_dir'] = self.savepath+'PRISM_spaxel/'
        data['ppxf']['redshift_table'] = self.savepath+'PRISM_1D/redshift_1D.csv'

        with open(self.savepath+'/PRISM_spaxel/R100_1D_setup_test.yaml', 'w') as f:
            data = yaml.dump(data, f, sort_keys=False, default_flow_style=True)
        from . import jadify_temp as pth
        PATH_TO_jadify = pth.__path__[0]+ '/'
        filename = PATH_TO_jadify+ 'red_table_template.csv'
        redshift_cat = Table.read(filename)
        
        files = glob.glob(self.savepath+'PRISM_spaxel/prism_clear/*.fits')
        
        IDs= np.array([], dtype=int)
        for i, file in enumerate(files):
            comp = file.split('/')
            IDs = np.append( IDs, int(comp[-1][:6]))
        redshift_cat_mod = Table()
        redshift_cat_mod['ID'] = IDs
        redshift_cat_mod['z_visinsp'] = np.ones_like(len(IDs))*self.z
        redshift_cat_mod['z_phot'] = np.ones_like(len(IDs))*self.z
        redshift_cat_mod['z_bagp'] = np.ones_like(len(IDs))*self.z
        redshift_cat_mod['flag'] = np.zeros_like(IDs, dtype='<U6')
        redshift_cat_mod['flag'][:] = redshift_cat['flag'][0]
        redshift_cat_mod.write(self.savepath+'PRISM_spaxel/redshift_spaxel.csv',overwrite=True)
        
        import nirspecxf
        config100 = nirspecxf.NIRSpecConfig(self.savepath+'PRISM_spaxel/R100_1D_setup_manual.yaml')
        #xid = IDs[3]
        #ns, _ = nirspecxf.process_object_id(id, config100)
        nirspecxf.process_multi(ncpu, IDs, config100)
        #for i, id in enumerate(IDs):
        #    print(i)
        #    ns, _ = nirspecxf.process_object_id(id, config100)
        print('Fitting done, merging results')
        nirspecxf.data_prods.merge_em_lines_tables(
            self.savepath+'PRISM_spaxel/res/*R100_em_lines.fits',
            self.savepath+'PRISM_spaxel/spaxel_R100_ppxf_emlines.fits')
        

    def Spaxel_fitting_general_MCMC_mp(self,fitted_model, labels, priors, logprior, nwalkers=64,use=np.array([]), N=10000, add='',Ncores=(mp.cpu_count() - 2), **kwargs):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        data= {'priors':priors}
        data['fitted_model'] = fitted_model
        data['labels'] = labels
        data['logprior'] = logprior
        data['nwalkers'] = nwalkers
        data['use'] = use
        data['N'] = N
        
        
        with open(os.getenv("HOME")+'/priors.pkl', "wb") as fp:
            pickle.dump(data,fp)     
        
        if Ncores<1:
            Ncores=1
        
        progress = kwargs.get('progress', True)
        progress = tqdm.tqdm if progress else lambda x, total=0: x
        debug = kwargs.get('debug', False)
        if debug:
            warnings.warn(
                '\u001b[5;33mDebug mode - no multiprocessing!\033[0;0m',
                UserWarning)
            cube_res = list(progress(
                map(emfit.Fitting_general_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
        else:
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        emfit.Fitting_general_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
               
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_general'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))

    def Spaxel_fitting_general_toptup(self, to_fit ,fitted_model, labels, priors, logprior, nwalkers=64,use=np.array([]), N=10000, add='',Ncores=(mp.cpu_count() - 2), **kwargs):
        import pickle
        start_time = time.time()
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_general'+add+'.txt', "rb") as fp:
            Cube_res= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        data= {'priors':priors}
        data['fitted_model'] = fitted_model
        data['labels'] = labels
        data['logprior'] = logprior
        data['nwalkers'] = nwalkers
        data['use'] = use
        data['N'] = N

        for i, row in enumerate(Cube_res):
            y,x, res = row
            if to_fit[0]==x and to_fit[1]==y:

                flx_spax_m, error, wave = res.fluxs, res.error, res.wave
                z = self.z
                use = data['use'] 
    
                Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=data['N'],progress=True, priors=data['priors'])
                Fits_sig.fitting_general(data['fitted_model'], data['labels'], data['logprior'], nwalkers=data['nwalkers'])
                Fits_sig.fitted_model = 0
      
            
                Cube_res[i]  = [x,y,Fits_sig ]

                f,ax = plt.subplots(1, figsize=(10,5))
                ax.plot(Fits_sig.wave, Fits_sig.flux, drawstyle='steps-mid')
                ax.plot(Fits_sig.wave, Fits_sig.yeval, 'r--')

                ax.text(Fits_sig.wave[10], 0.9*max(Fits_sig.yeval), 'x='+str(x)+', y='+str(y) )

                break
       
               
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_general'+add+'.txt', "wb") as fp:
            pickle.dump( Cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))
        
    def Map_creation_OIII(self,SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='',):
        z0 = self.z
        failed_fits=0
        wvo3 = 5008.24*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        try:
            with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII_2G'+add+'.txt', "rb") as fp:
                results= pickle.load(fp)
        except:
            with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_OIII'+add+'.txt', "rb") as fp:
                results= pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_oiii = np.zeros((4,self.dim[0], self.dim[1]))
        map_oiii[:,:,:] = np.nan

        map_oiii_ki = np.zeros((5,self.dim[0], self.dim[1]))
        map_oiii_ki[:,:,:] = np.nan
        # =============================================================================
        #        Filling these maps
        # =============================================================================
        f,ax= plt.subplots(1)
        from . import Plotting_tools_v2 as emplot

        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_OIII_fit_detection_only.pdf')


        for row in tqdm.tqdm(range(len(results))):
            try:
                i,j, Fits = results[row]
            except:
                print('Loading old fits? I am sorry no longer compatible...')

            if str(type(Fits)) == "<class 'dict'>":
                failed_fits+=1
                continue

            fitted_model = Fits.fitted_model
                
           
            
            z = Fits.props['popt'][0]
            SNR = sp.SNR_calc(Fits.wave, Fits.fluxs, Fits.error, Fits.props, 'OIII')
            flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits.props, Fits.chains, 'OIIIt', self.flux_norm)

            map_oiii[0,i,j]= SNR

            if SNR>SNR_cut:
                map_oiii[1,i,j] = flux_oiii.copy()
                map_oiii[2,i,j] = p16_oiii.copy()
                map_oiii[3,i,j] = p84_oiii.copy()


                map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(fitted_model, Fits.props, 0, z=self.z)#res_spx['Nar_fwhm'][0]

                p = ax.get_ylim()[1]

                ax.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(map_oiii_ki[1,i,j],2)) )
            else:


                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(5008.24*(1+self.z)/1e4)/dl
                map_oiii[3,i,j] = SNR_cut*Fits.error[1]*dl*np.sqrt(n)
                

            
            if SNR>SNR_cut:
                try:
                    emplot.plotting_OIII(self.obs_wave, Fits.fluxs, ax, Fits.props, Fits.fitted_model)
                except:
                    print(Fits.props, Fits.fitted_model)
                    break
                ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
                plt.tight_layout()
                Spax.savefig()
                ax.clear()

        Spax.close()

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x = int(self.center_data[1]); y= int(self.center_data[2])
        f = plt.figure( figsize=(10,10))

        IFU_header = self.header

        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600


        Offsets_low = -self.center_data[1:3][::-1]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3][::-1]

        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])

        lim_sc = lim*arc_per_pix

        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])

        flx = ax1.imshow(map_oiii[1,:,:],vmax=map_oiii[1,y,x], origin='lower', extent= lim_sc)
        ax1.set_title('Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')

        #lims =
        #emplot.overide_axes_labels(f, axes[0,0], lims)


        vel = ax2.imshow(map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
        ax2.set_title('v50')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')


        fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0], vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('W80 map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')

        snr = ax4.imshow(map_oiii[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')

        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y



        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)
        
        oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
        oiii_kin_hdu = fits.ImageHDU(map_oiii_ki, name='OIII_kin')

        hdulist = fits.HDUList([primary_hdu,oiii_hdu,oiii_kin_hdu ])

        hdulist.writeto(self.savepath+self.ID+'_OIII_fits_maps_2G.fits', overwrite=True)

    def Map_creation_Halpha(self, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, add=''):
        z0 = self.z

        wvo3 = 6563*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
            results= pickle.load(fp)

        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_vel = np.zeros(self.dim[:2])
        map_vel[:,:] = np.nan

        map_fwhm = np.zeros(self.dim[:2])
        map_fwhm[:,:] = np.nan

        map_flux = np.zeros(self.dim[:2])
        map_flux[:,:] = np.nan

        map_snr = np.zeros(self.dim[:2])
        map_snr[:,:] = np.nan

        map_nii = np.zeros(self.dim[:2])
        map_nii[:,:] = np.nan

        # =============================================================================
        #        Filling these maps
        # =============================================================================
        gf,ax= plt.subplots(1)
        from . import Plotting_tools_v2 as emplot

        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_Halpha_fit_detection_only.pdf')


        for row in range(len(results)):

            i,j, res_spx = results[row]
            i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]

            z = res_spx['popt'][0]
            SNR = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hn')
            map_snr[i,j]= SNR
            if SNR>SNR_cut:

                map_vel[i,j] = ((6563*(1+z)/1e4)-wvo3)/wvo3*3e5
                map_fwhm[i,j] = res_spx['popt'][5]
                map_flux[i,j] = sp.flux_calc(res_spx, 'Hat',self.flux_norm)
                map_nii[i,j] = sp.flux_calc(res_spx, 'NIIt', self.flux_norm)


            emplot.plotting_Halpha(self.obs_wave, flx_spax_m, ax, res_spx, emfit.H_models.Halpha, error=error)
            ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))

            if res_spx['Hal_peak'][0]<3*error[0]:
                ax.set_ylim(-error[0], 5*error[0])
            if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
                ax.set_ylim(-error[0], 5*error[0])
            Spax.savefig()
            ax.clear()
        plt.close(gf)
        Spax.close()

        self.Flux_map = map_flux
        self.Vel_map = map_vel
        self.FWHM_map = map_fwhm
        self.SNR_map = map_snr

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        x = int(self.center_data[1]); y= int(self.center_data[2])
        f = plt.figure( figsize=(10,10))

        IFU_header = self.header

        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600


        Offsets_low = -self.center_data[1:3][::-1]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3][::-1]

        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])

        lim_sc = lim*arc_per_pix

        ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
        ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
        ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
        ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])

        if flux_max==0:
            flx_max = map_flux[y,x]
        else:
            flx_max = flux_max

        print(lim_sc)
        flx = ax1.imshow(map_flux,vmax=flx_max, origin='lower', extent= lim_sc)
        ax1.set_title('Halpha Flux map')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(flx, cax=cax, orientation='vertical')
        cax.set_ylabel('Flux (arbitrary units)')
        ax1.set_xlabel('RA offset (arcsecond)')
        ax1.set_ylabel('Dec offset (arcsecond)')

        #lims =
        #emplot.overide_axes_labels(f, axes[0,0], lims)


        vel = ax2.imshow(map_vel, cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
        ax2.set_title('Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')

        cax.set_ylabel('Velocity (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')


        fw = ax3.imshow(map_fwhm,vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')

        cax.set_ylabel('FWHM (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')

        snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')

        cax.set_ylabel('SNR')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')

        fnii,axnii = plt.subplots(1)
        axnii.set_title('[NII] map')
        fw= axnii.imshow(map_nii, vmax=flx_max ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fnii.colorbar(fw, cax=cax, orientation='vertical')

        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y

        Line_info = np.zeros((5,self.dim[0],self.dim[1]))
        Line_info[0,:,:] = map_flux
        Line_info[1,:,:] = map_vel
        Line_info[2,:,:] = map_fwhm
        Line_info[3,:,:] = map_snr
        Line_info[4,:,:] = map_nii

        prhdr = hdr
        hdu = fits.PrimaryHDU(Line_info, header=prhdr)
        hdulist = fits.HDUList([hdu])


        hdulist.writeto(self.savepath+self.ID+'_Halpha_fits_maps.fits', overwrite=True)

        return f

    def Map_creation_ppxf(self, info, add=''):
        flux_table = Table.read(self.savepath+'PRISM_spaxel/spaxel_R100_ppxf_emlines.fits')
        info_keys = list(info.keys())
        for key in info_keys:
            map_flx = np.zeros((2,self.dim[0], self.dim[1]))
            map_flx[:,:,:] = np.nan
            
            for k, row in tqdm.tqdm(enumerate(flux_table)):
                ID = str(row['ID'])
                i,j = int(ID[:2]),int(ID[2:])
                map_flx[0,i,j] = (row[key+'_flux'] if row[key+'_flux']>row[key+'_flux_upper'] else np.nan)
                map_flx[0,i,j] = (row[key+'_flux_upper']/3 if row[key+'_flux']>row[key+'_flux_upper'] else np.nan)
            
            info[key]['flux_map'] = map_flx
        
        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=self.header)
        hdus = [primary_hdu]
        for key in info_keys:
            hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))
        

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.savepath+self.ID+'_ppxf_fits_maps'+add+'.fits', overwrite=True)


    def Map_creation_Halpha_OIII(self, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='',modelfce = HaO_models.Halpha_OIII):
        z0 = self.z
        failed_fits=0
        wv_hal = 6564.52*(1+z0)/1e4
        wv_oiii = 5008.24*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
            results= pickle.load(fp)

        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================

        map_hal = np.zeros((4,self.dim[0], self.dim[1]))
        map_hal[:,:,:] = np.nan

        map_nii = np.zeros((4,self.dim[0], self.dim[1]))
        map_nii[:,:,:] = np.nan

        map_hb = np.zeros((4,self.dim[0], self.dim[1]))
        map_hb[:,:,:] = np.nan

        map_oiii = np.zeros((4,self.dim[0], self.dim[1]))
        map_oiii[:,:,:] = np.nan

        map_siir = np.zeros((4,self.dim[0], self.dim[1]))
        map_siir[:,:,:] = np.nan

        map_siib = np.zeros((4,self.dim[0], self.dim[1]))
        map_siib[:,:,:] = np.nan

        map_hal_ki = np.zeros((4,self.dim[0], self.dim[1]))
        map_hal_ki[:,:,:] = np.nan

        map_nii_ki = np.zeros((4,self.dim[0], self.dim[1]))
        map_nii_ki[:,:,:] = np.nan

        map_oiii_ki = np.zeros((5,self.dim[0], self.dim[1]))
        map_oiii_ki[:,:,:] = np.nan
        # =============================================================================
        #        Filling these maps
        # =============================================================================


        from . import Plotting_tools_v2 as emplot

        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_Halpha_OIII_fit_detection_only.pdf')

        from . import Halpha_OIII_models as HO_models
        for row in tqdm.tqdm(range(len(results))):

            try:
                i,j, res_spx,chains,wave,flx_spax_m,error = results[row]
            except:
                i,j, res_spx,chains= results[row]
                i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]

            lists = list(res_spx.keys())
            if 'Failed fit' in lists:
                failed_fits+=1
                continue

            z = res_spx['popt'][0]
        
        
            if 'zBLR' in lists:
                modelfce = HO_models.Halpha_OIII_BLR
            elif 'outflow_vel' not in lists:
                modelfce = HO_models.Halpha_OIII
            elif 'outflow_vel' in lists and 'zBLR' not in lists:
                modelfce = HO_models.Halpha_OIII_outflow

# =============================================================================
#             Halpha
# =============================================================================

            flux_hal, p16_hal,p84_hal = sp.flux_calc_mcmc(res_spx, chains, 'Hat', self.flux_norm)
            SNR_hal = flux_hal/p16_hal
            map_hal[0,i,j]= SNR_hal

            SNR_hal = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hn')
            SNR_oiii = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'OIII')
            SNR_nii = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'NII')
            SNR_hb = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'Hb')

            if SNR_hal>SNR_cut:
                map_hal[1,i,j] = flux_hal.copy()
                map_hal[2,i,j] = p16_hal.copy()
                map_hal[3,i,j] = p84_hal.copy()

                if 'Hal_out_peak' in list(res_spx.keys()):
                    map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = sp.W80_Halpha_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]

                else:
                    map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = sp.W80_Halpha_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]


                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                map_hal[3,i,j] = -SNR_cut*error[-1]*dl*np.sqrt(n)




# =============================================================================
#             Plotting
# =============================================================================
            f = plt.figure(figsize=(10,4))
            baxes = brokenaxes(xlims=((4800,5050),(6500,6800)),  hspace=.01)
            emplot.plotting_Halpha_OIII(self.obs_wave, flx_spax_m, baxes, res_spx, modelfce)

            #if res_spx['Hal_peak'][0]<3*error[0]:
            #    baxes.set_ylim(-error[0], 5*error[0])
            #if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
            #    baxes.set_ylim(-error[0], 5*error[0])

            SNRs = np.array([SNR_hal])

# =============================================================================
#             NII
# =============================================================================
            #SNR = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'NII')
            flux_NII, p16_NII,p84_NII = sp.flux_calc_mcmc(res_spx, chains, 'NIIt', self.flux_norm)

            map_nii[0,i,j]= SNR_nii
            if SNR_nii>SNR_cut:
                map_nii[1,i,j] = flux_NII.copy()
                map_nii[2,i,j] = p16_NII.copy()
                map_nii[3,i,j] = p84_NII.copy()

                if 'NII_out_peak' in list(res_spx.keys()):
                    map_nii_ki[2,i,j], map_nii_ki[3,i,j],map_nii_ki[1,i,j],map_nii_ki[0,i,j], = sp.W80_NII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]

                else:
                    map_nii_ki[2,i,j], map_nii_ki[3,i,j],map_nii_ki[1,i,j],map_nii_ki[0,i,j], = sp.W80_NII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]

            else:
                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                map_nii[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
# =============================================================================
#             OIII
# =============================================================================
            flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(res_spx, chains, 'OIIIt', self.flux_norm)

            map_oiii[0,i,j]= SNR_oiii

            if SNR_oiii>SNR_cut:
                map_oiii[1,i,j] = flux_oiii.copy()
                map_oiii[2,i,j] = p16_oiii.copy()
                map_oiii[3,i,j] = p84_oiii.copy()


                if 'OIII_out_peak' in list(res_spx.keys()):
                    map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]

                else:
                    map_oiii_ki[0,i,j] = ((5008.24*(1+z)/1e4)-wv_oiii)/wv_oiii*3e5
                    map_oiii_ki[1,i,j] = res_spx['Nar_fwhm'][0]
                    map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(modelfce, res_spx, 0, z=self.z)#res_spx['Nar_fwhm'][0]
                p = baxes.get_ylim()[1][1]

                baxes.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(map_oiii_ki[1,i,j],2)) )
            else:


                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(5008.24*(1+self.z)/1e4)/dl
                map_oiii[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)

# =============================================================================
#             Hbeta
# =============================================================================
            flux_hb, p16_hb,p84_hb = sp.flux_calc_mcmc(res_spx, chains, 'Hbeta', self.flux_norm)

            map_hb[0,i,j]= SNR_hb.copy()
            if SNR_hb>SNR_cut:
                map_hb[1,i,j] = flux_hb.copy()
                map_hb[2,i,j] = p16_hb.copy()
                map_hb[3,i,j] = p84_hb.copy()

            else:

                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(4860*(1+self.z)/1e4)/dl
                map_hb[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)

# =============================================================================
#           SII
# =============================================================================
            fluxr, p16r,p84r = sp.flux_calc_mcmc(res_spx, chains, 'SIIr', self.flux_norm)
            fluxb, p16b,p84b = sp.flux_calc_mcmc(res_spx, chains, 'SIIb', self.flux_norm)

            SNR_SII = sp.SNR_calc(self.obs_wave, flx_spax_m, error, res_spx, 'SII')

            if SNR_SII>SNR_cut:
                map_siir[0,i,j] = SNR_SII.copy()
                map_siib[0,i,j] = SNR_SII.copy()

                map_siir[1,i,j] = fluxr.copy()
                map_siir[2,i,j] = p16r.copy()
                map_siir[3,i,j] = p84r.copy()

                map_siib[1,i,j] = fluxb.copy()
                map_siib[2,i,j] = p16b.copy()
                map_siib[3,i,j] = p84b.copy()

            else:

                dl = self.obs_wave[1]-self.obs_wave[0]
                n = width_upper/3e5*(6731*(1+self.z)/1e4)/dl
                map_siir[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
                map_siib[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)




            baxes.set_title('xy='+str(j)+' '+ str(i) + ', SNR = '+ str(np.round([SNR_hal, SNR_oiii, SNR_nii, SNR_SII],1)))
            baxes.set_xlabel('Restframe wavelength (ang)')
            baxes.set_ylabel(r'$10^{-16}$ ergs/s/cm2/mic')
            wv0 = 5008.24*(1+z0)
            wv0 = wv0/(1+z)
            baxes.vlines(wv0, 0,10, linestyle='dashed', color='k')
            Spax.savefig()
            plt.close(f)

        print('Failed fits', failed_fits)
        Spax.close()
# =============================================================================
#         Calculating Avs
# =============================================================================
        Av = sp.Av_calc(map_hal[1,:,:],map_hb[1,:,:])
# =============================================================================
#         Plotting maps
# =============================================================================
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        x = int(self.center_data[1]); y= int(self.center_data[2])
        IFU_header = self.header
        deg_per_pix = IFU_header['CDELT2']
        arc_per_pix = deg_per_pix*3600

        Offsets_low = -self.center_data[1:3][::-1]
        Offsets_hig = self.dim[0:2] - self.center_data[1:3][::-1]

        lim = np.array([ Offsets_low[0], Offsets_hig[0],
                         Offsets_low[1], Offsets_hig[1] ])

        lim_sc = lim*arc_per_pix

        if flux_max==0:
            flx_max = map_hal[y,x]
        else:
            flx_max = flux_max

        
        print(lim_sc)

# =============================================================================
#         Plotting Stuff
# =============================================================================
        f,axes = plt.subplots(6,3, figsize=(10,20))
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
        fw= axes[1,1].imshow(map_nii[1,:,:] ,origin='lower', extent= lim_sc)
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
        fw= axes[2,1].imshow(map_hb[1,:,:] ,origin='lower', extent= lim_sc)
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
        fw= axes[3,1].imshow(map_oiii[1,:,:] ,origin='lower', extent= lim_sc)
        divider = make_axes_locatable(axes[3,1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')
        axes[3,1].set_xlabel('RA offset (arcsecond)')
        axes[3,1].set_ylabel('Dec offset (arcsecond)')

        # =============================================================================
        # OIII  velocity
        ax2 = axes[2,2]
        vel = ax2.imshow(map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
        ax2.set_title('OIII Velocity offset map')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')

        cax.set_ylabel('Velocity (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')

        # =============================================================================
        # OIII fwhm
        ax3 = axes[3,2]
        fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('OIII FWHM map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')

        cax.set_ylabel('FWHM (km/s)')
        ax2.set_xlabel('RA offset (arcsecond)')
        ax2.set_ylabel('Dec offset (arcsecond)')

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


        plt.tight_layout()

        self.map_hal = map_hal

        hdr = self.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y

        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=self.header)
        hal_hdu = fits.ImageHDU(map_hal, name='Halpha')
        nii_hdu = fits.ImageHDU(map_nii, name='NII')
        nii_kin_hdu = fits.ImageHDU(map_nii_ki, name='NII_kin')
        hbe_hdu = fits.ImageHDU(map_hb, name='Hbeta')
        oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')

        siir_hdu = fits.ImageHDU(map_siir, name='SIIr')
        siib_hdu = fits.ImageHDU(map_siib, name='SIIb')

        hal_kin_hdu = fits.ImageHDU(map_hal_ki, name='Hal_kin')
        oiii_kin_hdu = fits.ImageHDU(map_oiii_ki, name='OIII_kin')
        Av_hdu = fits.ImageHDU(Av, name='Av')

        hdulist = fits.HDUList([primary_hdu, hal_hdu, nii_hdu, nii_kin_hdu, hbe_hdu, oiii_hdu,hal_kin_hdu,siir_hdu,oiii_kin_hdu, siib_hdu, Av_hdu ])
        hdulist.writeto(self.savepath+self.ID+'_Halpha_OIII_fits_maps'+add+'.fits', overwrite=True)

        return f
    
    def Map_creation_general_comparison(self,info,path1, path2, SNR_cut = 3 ,deltabic=10, add='',\
                             brokenaxes_xlims= ((2.820,3.45),(3.75,4.05),(5,5.3)) ):
        z0 = self.z
        failed_fits=0
        
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(path1, "rb") as fp:
            results1= pickle.load(fp)
        
        with open(path2, "rb") as fp:
            results2 = pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================

        Result_cube = np.zeros_like(self.flux.data)
        Result_cube_data = self.flux.data
        Result_cube_error = self.error_cube.data
        
        info_keys = list(info.keys())
        
        for key in info_keys:
            map_flx = np.zeros((4,self.dim[0], self.dim[1]))
            map_flx[:,:,:] = np.nan
                
            info[key]['flux_map'] = map_flx
            
            if info[key]['kin'] ==1:
                map_ki = np.zeros((5,self.dim[0], self.dim[1]))
                map_ki[:,:,:] = np.nan

                info[key]['kin_map'] = map_ki
        # =============================================================================
        #        Filling these maps
        # =============================================================================

        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_general_fit_detection_only_comp'+add+'.pdf')

        Results2_map = np.full((2,len(results2)), fill_value=np.nan)
        for i,row in enumerate(Result_cube):
            Results2_map[:,i] = row[0], row[1]
        '''
        for row in tqdm.tqdm(range(len(results1))):

            try:
                i,j, Fits = results1[row]
            except:
                print('Loading old fits? I am sorry no longer compatible...')

            if str(type(Fits)) == "<class 'dict'>":
                failed_fits+=1
                continue

            Result_cube_data[:,i,j] = Fits.fluxs.data
            try:
                Result_cube_error[:,i,j] = Fits.error.data
            except:
                lds=0
            Result_cube[:,i,j] = Fits.yeval

            for key in info_keys:
                
                SNR= sp.SNR_calc(self.obs_wave, Fits.fluxs, Fits.error, Fits.props, 'general',\
                                 wv_cent = info[key]['wv'],\
                                 peak_name = key+'_peak', \
                                     fwhm_name = info[key]['fwhm'])
                
                info[key]['flux_map'][0,i,j] = SNR
                
                if SNR>SNR_cut:
                    flux, p16,p84 = sp.flux_calc_mcmc(Fits.props, Fits.chains, 'general', self.flux_norm,\
                                                      wv_cent = info[key]['wv'],\
                                                      peak_name = key+'_peak', \
                                                          fwhm_name = info[key]['fwhm'])
                    
                    info[key]['flux_map'][1,i,j] = flux
                    info[key]['flux_map'][2,i,j] = p16
                    info[key]['flux_map'][3,i,j] = p84

                    if info[key]['kin'] ==1:
                        info[key]['kin_map'][0,i,j] = (np.median(Fits.chains['z'])-self.z)/(1+self.z)*3e5
                        info[key]['kin_map'][1,i,j] = np.median(Fits.chains[info[key]['fwhm']])
                     
                else:
                    dl = self.obs_wave[1]-self.obs_wave[0]
                    n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                    info[key]['flux_map'][3,i,j] = -SNR_cut*Fits.error[-1]*dl*np.sqrt(n)

# =============================================================================
#             Plotting
# =============================================================================
            f = plt.figure( figsize=(20,6))

            ax = brokenaxes(xlims=brokenaxes_xlims,  hspace=.01)
            
            ax.plot(Fits.wave, Fits.fluxs.data, drawstyle='steps-mid')
            y= Fits.yeval
            ax.plot(self.obs_wave,  y, 'r--')
            
            ax.set_xlabel('wavelength (um)')
            ax.set_ylabel('Flux density')
            
            ax.set_ylim(-2*Fits.error[0], 1.2*max(y))
            ax.set_title('xy='+str(j)+' '+ str(i) )

            Spax.savefig()
            plt.close(f)

        print('Failed fits', failed_fits)
        Spax.close()

# =============================================================================
#         Plotting maps
# =============================================================================
        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=self.header)
        hdus = [primary_hdu]
        hdus.append(fits.ImageHDU(Result_cube_data, name='flux'))
        hdus.append(fits.ImageHDU(Result_cube_error, name='error'))
        hdus.append(fits.ImageHDU(Result_cube, name='yeval'))

        for key in info_keys:
            hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))

        for key in info_keys:
            if info[key]['kin'] ==1:
                hdus.append(fits.ImageHDU(info[key]['kin_map'], name=key+'_kin'))

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.savepath+self.ID+'_general_fits_maps'+add+'.fits', overwrite=True)

        return f
    '''
    def Map_creation_general(self,info, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='',modelfce = None,\
                             brokenaxes_xlims= ((2.820,3.45),(3.75,4.05),(5,5.3)) ):
        z0 = self.z
        failed_fits=0
        
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw_general'+add+'.txt', "rb") as fp:
            results= pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================

        Result_cube = np.zeros_like(self.flux.data)
        Result_cube_data = self.flux.data
        Result_cube_error = self.error_cube.data
        
        info_keys = list(info.keys())
        
        for key in info_keys:
            map_flx = np.zeros((4,self.dim[0], self.dim[1]))
            map_flx[:,:,:] = np.nan
                
            info[key]['flux_map'] = map_flx
            
            if info[key]['kin'] ==1:
                map_ki = np.zeros((5,self.dim[0], self.dim[1]))
                map_ki[:,:,:] = np.nan

                info[key]['kin_map'] = map_ki
        # =============================================================================
        #        Filling these maps
        # =============================================================================

        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_general_fit_detection_only'+add+'.pdf')

        for row in tqdm.tqdm(range(len(results))):

            try:
                i,j, Fits = results[row]
            except:
                print('Loading old fits? I am sorry no longer compatible...')

            if str(type(Fits)) == "<class 'dict'>":
                failed_fits+=1
                continue

            Result_cube_data[:,i,j] = Fits.fluxs.data
            try:
                Result_cube_error[:,i,j] = Fits.error.data
            except:
                lds=0
            Result_cube[:,i,j] = Fits.yeval

            for key in info_keys:
                
                SNR= sp.SNR_calc(self.obs_wave, Fits.fluxs, Fits.error, Fits.props, 'general',\
                                 wv_cent = info[key]['wv'],\
                                 peak_name = key+'_peak', \
                                     fwhm_name = info[key]['fwhm'])
                
                info[key]['flux_map'][0,i,j] = SNR
                
                if SNR>SNR_cut:
                    flux, p16,p84 = sp.flux_calc_mcmc(Fits.props, Fits.chains, 'general', self.flux_norm,\
                                                      wv_cent = info[key]['wv'],\
                                                      peak_name = key+'_peak', \
                                                          fwhm_name = info[key]['fwhm'])
                    
                    info[key]['flux_map'][1,i,j] = flux
                    info[key]['flux_map'][2,i,j] = p16
                    info[key]['flux_map'][3,i,j] = p84

                    if info[key]['kin'] ==1:
                        info[key]['kin_map'][0,i,j] = (np.median(Fits.chains['z'])-self.z)/(1+self.z)*3e5
                        info[key]['kin_map'][1,i,j] = np.median(Fits.chains[info[key]['fwhm']])
                     
                else:
                    dl = self.obs_wave[1]-self.obs_wave[0]
                    n = width_upper/3e5*(6564.52**(1+self.z)/1e4)/dl
                    info[key]['flux_map'][3,i,j] = -SNR_cut*Fits.error[-1]*dl*np.sqrt(n)

# =============================================================================
#             Plotting
# =============================================================================
            f = plt.figure( figsize=(20,6))

            ax = brokenaxes(xlims=brokenaxes_xlims,  hspace=.01)
            
            ax.plot(Fits.wave, Fits.fluxs.data, drawstyle='steps-mid')
            y= Fits.yeval
            ax.plot(self.obs_wave,  y, 'r--')
            
            ax.set_xlabel('wavelength (um)')
            ax.set_ylabel('Flux density')
            
            ax.set_ylim(-2*Fits.error[0], 1.2*max(y))
            ax.set_title('xy='+str(j)+' '+ str(i) )

            Spax.savefig()
            plt.close(f)

        print('Failed fits', failed_fits)
        Spax.close()

# =============================================================================
#         Plotting maps
# =============================================================================
        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=self.header)
        hdus = [primary_hdu]
        hdus.append(fits.ImageHDU(Result_cube_data, name='flux'))
        hdus.append(fits.ImageHDU(Result_cube_error, name='error'))
        hdus.append(fits.ImageHDU(Result_cube, name='yeval'))

        for key in info_keys:
            hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))

        for key in info_keys:
            if info[key]['kin'] ==1:
                hdus.append(fits.ImageHDU(info[key]['kin_map'], name=key+'_kin'))

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(self.savepath+self.ID+'_general_fits_maps'+add+'.fits', overwrite=True)

        return f
     

    def Regional_Spec(self, center=[30,30], rad=0.4, err_range=None, manual_mask=np.array([]), boundary=None):
        '''
        Extracting regional spectra to be fitted.

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        rad : TYPE
            DESCRIPTION.
        err_range : TYPE, optional
            DESCRIPTION. The default is None.
        manual_mask : TYPE, optional
            DESCRIPTION. The default is None.
        boundary : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        D1_spectrum : TYPE
            DESCRIPTION.
        D1_spectrum_er : TYPE
            DESCRIPTION.
        mask_catch : TYPE
            DESCRIPTION.

        '''

        shapes = self.dim

        print ('Center of cont', center)
        print ('Extracting spectrum from diameter', rad*2, 'arcseconds')

        # Creating a mask for all spaxels.
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        print('Pixel scale:', arc)
        print ('radius ', arc*rad)
        
        if len(manual_mask)==0:
            # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
            for ix in range(shapes[0]):
                for iy in range(shapes[1]):
                    dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                    if dist< arc*rad:
                        mask_catch[:,ix,iy] = False
        else:
            for ix in range(shapes[0]):
                for iy in range(shapes[1]):
                    
                    if manual_mask[ix,iy]==False:
                        mask_catch[:,ix,iy] = False
            

        # Loading mask of the sky lines an bad features in the spectrum
        mask_sky_1D = self.sky_clipped_1D.copy()

        if self.instrument=='NIRSPEC_IFU':
            total_mask = np.logical_or( mask_catch, self.sky_clipped)
        else:
            total_mask = np.logical_or( mask_catch, self.flux.mask)
        # M
        flux = np.ma.array(data=self.flux.data, mask= total_mask)

        D1_spectrum = np.ma.sum(flux, axis=(1,2))
        D1_spectrum = np.ma.array(data = D1_spectrum.data, mask=mask_sky_1D)

        if self.instrument =='NIRSPEC_IFU':
            print('NIRSPEC mode of error calc')
            D1_spectrum_var_er = np.sqrt(np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2)))
            
            if len(err_range)==2:
                error = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2]
                
                average_var = stats.sigma_clipped_stats(D1_spectrum_var_er[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[1]
                D1_spectrum_er = D1_spectrum_var_er*(error/average_var)
                
            elif len(err_range)==4:
                error1 = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2]
                error2 = stats.sigma_clipped_stats(D1_spectrum[(err_range[2]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[3])],sigma=3)[2]
                
                average_var1 = stats.sigma_clipped_stats(D1_spectrum_var_er[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[1]
                average_var2 = stats.sigma_clipped_stats(D1_spectrum_var_er[(err_range[2]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[3])],sigma=3)[1]
                
                error = np.zeros(len(D1_spectrum))
                error[self.obs_wave<boundary] = D1_spectrum_var_er[self.obs_wave<boundary]*(error1/average_var1)
                error[self.obs_wave>boundary] = D1_spectrum_var_er[self.obs_wave>boundary]*(error2/average_var2)
                D1_spectrum_er = error
            else:
                error = stats.sigma_clipped_stats(D1_spectrum,sigma=3)[2]
                
                average_var = stats.sigma_clipped_stats(D1_spectrum,sigma=3)[1]
                D1_spectrum_er = D1_spectrum_var_er/(error/average_var)
            
            D1_spectrum_er[self.D1_spectrum_er==0] = np.mean(self.D1_spectrum_er)*5
        else:
            print('Other mode of error calc')
            if len(err_range)==2:
                error = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2] \
                                                        *np.ones(len(D1_spectrum))
                D1_spectrum_er = error

            elif len(err_range)==4:
                error1 = stats.sigma_clipped_stats(D1_spectrum[(err_range[0]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[1])],sigma=3)[2]
                error2 = stats.sigma_clipped_stats(D1_spectrum[(err_range[2]<self.obs_wave) \
                                                            &(self.obs_wave<err_range[3])],sigma=3)[2]
                
                error = np.zeros(len(D1_spectrum))
                error[self.obs_wave<boundary] = error1
                error[self.obs_wave>boundary] = error2
                D1_spectrum_er = error
            else:
                D1_spectrum_er = stats.sigma_clipped_stats(self.D1_spectrum,sigma=3)[2]*np.ones(len(self.D1_spectrum)) #STD_calc(wave/(1+self.z)*1e4,self.D1_spectrum, self.band)* np.ones(len(self.D1_spectrum))


        return D1_spectrum, D1_spectrum_er, mask_catch

    def PSF_matching(self, psf_fce=sp.NIRSpec_IFU_PSF, wv_ref=5.2, theta=None):
        from astropy.modeling.models import Gaussian2D 
        from photutils.psf import create_matching_kernel
        from photutils.psf import TopHatWindow, CosineBellWindow, HanningWindow, TukeyWindow, SplitCosineBellWindow
        from astropy.convolution import convolve, convolve_fft  
        from astropy.convolution import Gaussian2DKernel

        #y, x = np.mgrid[0:104, 0:98]
        #gf_match = Gaussian2D(100, 50, 50, psf_fce(wv_ref)[0], psf_fce(wv_ref)[1], theta=107)
        #gmatch = gf_match(x, y)
        #gmatch /= gmatch.sum()
        #window = TopHatWindow(0.6)
        #window = CosineBellWindow(alpha=0.99)
        theta = self.header['PA_V3']-138
        psf_matched = self.flux.copy()
        error_matched = self.error_cube.copy()
        for its in tqdm.tqdm(enumerate(self.obs_wave)):
            i, wave = its
            #gf_loc = Gaussian2D(100, 50, 50, PSF(wave)[0], PSF(wave)[1], theta=theta)
            #g_loc = gf_loc(x,y)
            #g_loc /=g_loc.sum()
            #kernel = create_matching_kernel(g_loc, gmatch, window=window)
            sigma = np.sqrt(psf_fce(wv_ref)**2 - psf_fce(wave)**2)/0.05
            if wave<wv_ref:
                kernel = Gaussian2DKernel( sigma[0],sigma[1], theta=theta)

                psf_matched[i,:,:] = convolve_fft(psf_matched[i,:,:], kernel)
                error_matched[i,:,:] = convolve_fft(error_matched[i,:,:], kernel)

        primary_hdu = fits.PrimaryHDU(np.zeros(1), header=self.header)

        hdus = [primary_hdu,fits.ImageHDU(psf_matched.data, name='SCI', header=self.header), fits.ImageHDU(error_matched, name='ERR', header=self.header)]
        hdulist = fits.HDUList(hdus)
        hdulist.writeto( self.cube_path[-4] +'psf_matched.fits', overwrite=True)

    def ppxf_fitting(self):
        x=1

    
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
        
        
        