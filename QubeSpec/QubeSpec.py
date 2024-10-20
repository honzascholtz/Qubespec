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
from brokenaxes import brokenaxes

from astropy.utils.exceptions import AstropyWarning
import astropy.constants, astropy.cosmology, astropy.units, astropy.wcs
from astropy.table import Table
import os

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



from . import Utils as sp
from . import Plotting as emplot
from . import Fitting as emfit
from . import Dust as dst

from .Models import Halpha_OIII_models as HaO_models
from . import Background as bkg
from . import Maps as Maps
from . import Spaxel_fitting as Spaxel


def Shortcut(QubeSpec_setup):
    """Quick function which does most of the prep of a of the data for fitting. 

    :param QubeSpec_setup: dict - setup file for QubeSpec

    :return: QubeSpec Cube object
    """

    print('Loading the data')
    obj = Cube( Full_path = QubeSpec_setup['file'],\
                z =  QubeSpec_setup['z'], \
                ID =  QubeSpec_setup['ID'] ,\
                flag =  QubeSpec_setup['instrument'] ,\
                savepath = QubeSpec_setup['save_path'] ,\
                Band = 'NIRSPEC',\
                norm = QubeSpec_setup['norm'])
    
    print('Masking JWST')
    obj.mask_JWST(0, threshold= QubeSpec_setup['mask_threshold'], spe_ma=QubeSpec_setup['mask_channels'])

    print('Performing background subtraction')
    if any(QubeSpec_setup['Source_mask']) !=None:
        print('Loading source mask from file')
        source_bkg = sp.QFitsview_mask(QubeSpec_setup['Source_mask']) # Loading background mask
    
    obj.background_subtraction( source_mask=source_bkg, wave_range=QubeSpec_setup['line_map_wavelength'], plot=1) # Doing 

    print('PSF matching')
    obj.PSF_matching(PSF_match = QubeSpec_setup['PSF_match'],\
                    wv_ref= QubeSpec_setup['PSF_match_wv'])
    
    print('Creating Continuum map')
    obj.collapse_white(1)

    print('Extracting 1D collapsed spectrum')
    obj.find_center(1, manual=QubeSpec_setup['Object_center'])
    obj.D1_spectra_collapse(1, addsave='',rad=QubeSpec_setup['Aperture_extraction'], err_range=QubeSpec_setup['err_range'], boundary=QubeSpec_setup['err_boundary'], plot_err=1)
    

    return obj

# ============================================================================
#  Main class
# =============================================================================
class Cube:
    """Main Class for QubeSpec
    """
    def __init__(self, Full_path='', z='', ID='', flag='', savepath='', Band='', norm=1e-13,):
        """The main class for QubeSpex

        Args:
            Full_path (str, optional): _description_. Defaults to ''.
            z (str, optional): _description_. Defaults to ''.
            ID (str, optional): _description_. Defaults to ''.
            flag (str, optional): _description_. Defaults to ''.
            savepath (str, optional): _description_. Defaults to ''.
            Band (str, optional): _description_. Defaults to ''.
            norm (_type_, optional): _description_. Defaults to 1e-13.

        Raises:
            Exception: _description_
        """
        import importlib
        importlib.reload(emfit )

        self.z = z
        self.ID = ID
        self.instrument = flag
        self.savepath = savepath
        self.band = Band
        self.Cube_path = Full_path
        self.flux_norm= norm

        if not os.path.isdir(self.savepath+'Diagnostics'):
            os.mkdir(self.savepath+'Diagnostics')

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

    def divider(self):
        return np.nan

    background_subtraction = bkg.background_subtraction
    background_subtraction_depricated = bkg.background_sub_spec_depricated
    
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
        Legacy: This function masks out all the OIII and HBeta emission.
        It is a ;egacy KASHz function.

        Returns
        -------
        None.

        '''
        OIIIa=  OIIIr/1e4*(1+self.z)
        OIIIb=  OIIIb/1e4*(1+self.z)
        Hbeta=  Hbeta/1e4*(1+self.z)
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

        self.Median_stack_white =  np.ma.median(self.flux, axis=(0))

        if plot==1:
            plt.figure()
            plt.imshow(self.Median_stack_white,  origin='lower')
            plt.colorbar()
            plt.savefig(self.savepath+'Diagnostics/Median_cont_image.pdf')



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
            plt.savefig(self.savepath+'Diagnostics/1D_spectrum_Selected_pixel.pdf')


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
            plt.savefig(self.savepath+'Diagnostics/1D_spectrum.pdf')
               
        self.D1_spectrum_var = np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2))

        if self.instrument =='NIRSPEC_IFU':
            print('NIRSPEC mode of error calc')
            nspaxel= np.sum(np.logical_not(total_mask[150,:,:]))

            D1_spectrum_var_er = np.sqrt(self.D1_spectrum_var)#/nspaxel)
            
            self.D1_spectrum_er = sp.error_scaling(self.obs_wave, D1_spectra, D1_spectrum_var_er, err_range, boundary,\
                                                   exp=plot_err)

            if plot_err==1:
                f,ax = plt.subplots(1)
                ax.plot(self.obs_wave, D1_spectrum_var_er, label='Extension')
                ax.plot(self.obs_wave, self.D1_spectrum_er, label='rescaled')

                ax.legend(loc='best')
                ax.set_xlabel(r'$\lambda_{\rm obs}$ $\mu$m')
                ax.set_ylabel('Flux density')
                plt.savefig(self.savepath+'Diagnostics/1D_spectrum_error_scaling.pdf')

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



    def mask_JWST(self, plot=0, threshold=100, spe_ma=[]):
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
        self.masking_threshold = threshold
        self.channel_mask = spe_ma
        sky_clipped =  self.flux.mask.copy()
        median_error = np.nanmedian(self.error_cube)
        std_error = np.nanmedian(self.error_cube)
        limit = threshold*std_error+median_error
        sky_clipped[self.error_cube>limit] = True

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

            plt.savefig(self.savepath+'Diagnostics/Sky_spectrum.pdf')

            plt.tight_layout()

    def fitting_collapse_Halpha(self, plot=1, models = 'BLR', progress=True, sampler ='emcee',er_scale=1, N=6000, priors= {'z': [0,'normal_hat',0, 0, 0,0]}):
        priors_update = priors.copy()
        priors= {'z':[0, 'normal_hat', 0,0,0,0],\
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
                'outflow_vel':[-50,'normal', 0,300]}

        
        for name in list(priors_update.keys()):
            priors[name] = priors_update[name]

        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()/er_scale
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        
        if models=='BLR':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha(model='gal')
            
            
            Fits_blr = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
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
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha(model='gal')
            
            
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_out.fitting_Halpha(model='outflow')
            
            
            if Fits_out.BIC-Fits_sig.BIC <-2:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                print('BICM', Fits_out.BIC)
                self.D1_fit_results = Fits_out.props
                self.D1_fit_chain = Fits_out.chains
                self.D1_fit_model = Fits_out.fitted_model
                self.D1_fit_full = Fits_out
                
                self.z = self.D1_fit_results['popt'][0]
                
                self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
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
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha(model='gal')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
        
        elif models=='Outflow_only':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha(model='outflow')
        
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.dBIC = 3
            
        elif models=='BLR_only':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
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
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
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
        
        emplot.plotting_Halpha(self.D1_fit_full, ax1, errors=True, residual='error', axres=ax2)
        
        self.fit_plot = [f,ax1,ax2]
        
        if (models=='BLR') | (models=='Outflow'):
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_Halpha(Fits_sig, ax1a)
            try:
                emplot.plotting_Halpha(Fits_blr, ax2a)
            except:
                emplot.plotting_Halpha(Fits_out, ax2a)
        
        plt.savefig(self.savepath+'Diagnostics/1D_spectrum_Halpha_fit.pdf')

            
            
    def fitting_collapse_Halpha_OIII(self, plot=1, progress=True,N=6000,sampler='emcee', models='Single_only', priors= {'z': [0,'normal_hat',0, 0, 0,0]}):
        priors_update = priors.copy()
        priors={'z':[0,'normal_hat', 0, 0.,0,0],\
            'cont':[0,'loguniform', -3,1],\
            'cont_grad':[0,'normal', 0,0.2],\
            'Hal_peak':[0,'loguniform', -3,1],\
            'NII_peak':[0,'loguniform', -3,1],\
            'Nar_fwhm':[300,'uniform', 200,900],\
            'SIIr_peak':[0,'loguniform', -3,1],\
            'SIIb_peak':[0,'loguniform', -3,1],\
            'OIII_peak':[0,'loguniform', -3,1],\
            'Hbeta_peak':[0,'loguniform', -3,1],\
            'outflow_fwhm':[450,'uniform', 300,1500],\
            'outflow_vel':[-50,'normal', -50,100],\
            'Hal_out_peak':[0,'loguniform', -3,1],\
            'NII_out_peak':[0,'loguniform', -3,1],\
            'OIII_out_peak':[0,'loguniform', -3,1],\
            'Hbeta_out_peak':[0,'loguniform', -3,1],\
            'zBLR':[0,'normal', 0,0.003],\
            'BLR_fwhm':[4000,'normal', 5000,500],\
            'BLR_Hal_peak':[0,'loguniform', -3,1],\
            'BLR_Hbeta_peak':[0,'loguniform', -3,1] }

        for name in list(priors_update.keys()):
            priors[name] = priors_update[name]
            
        
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        if models=='Single_only':   
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha_OIII(model='gal' )
            
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig

            self.z = Fits_sig.props['popt'][0]
            
            self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
    
            self.dBIC = 3
            
            
        elif models=='Outflow_only':   
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_Halpha_OIII(model='outflow' )
            
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.z = Fits_sig.props['popt'][0]
            self.D1_fit_full = Fits_sig

            self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
            self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
            self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
    
            
            self.dBIC = 3
            
        elif (models=='BLR') | (models=='BLR_only'):   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
             Fits_sig.fitting_Halpha_OIII(model='BLR' )
             
             self.D1_fit_results = Fits_sig.props
             self.D1_fit_chain = Fits_sig.chains
             self.D1_fit_model = Fits_sig.fitted_model
             self.z = Fits_sig.props['popt'][0]
             self.D1_fit_full = Fits_sig

             self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
             
             self.dBIC = 3

        elif models=='BLR_simple':   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
             Fits_sig.fitting_Halpha_OIII(model='BLR_simple' )
             
             self.D1_fit_results = Fits_sig.props
             self.D1_fit_chain = Fits_sig.chains
             self.D1_fit_model = Fits_sig.fitted_model
             self.z = Fits_sig.props['popt'][0]
             self.D1_fit_full = Fits_sig

             self.SNR_hal =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hn')
             self.SNR_sii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'SII')
             self.SNR_OIII =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
             self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
             self.SNR_nii =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'NII')
     
             self.dBIC = 3

        elif models=='QSO_BKPL':   
             Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
             Fits_sig.fitting_Halpha_OIII(model='QSO_BKPL' )
             
             self.D1_fit_full = Fits_sig
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
                
        emplot.plotting_Halpha_OIII(self.D1_fit_full, baxes, errors=True, residual='error')                             
        baxes.set_xlabel('Restframe wavelength (ang)')

        plt.savefig(self.savepath+'Diagnostics/1D_spectrum_Halpha_OIII_fit.pdf')

        
        g = plt.figure(figsize=(10,4))
        if models=='QSO_BKPL':
            baxes_er = brokenaxes(xlims=((4700,5050),(6200,6800)),  hspace=.01)  
        else:
            baxes_er = brokenaxes(xlims=((4800,5050),(6400,6800)),  hspace=.01)
            
        
        y_tot = self.D1_fit_model(self.obs_wave, *self.D1_fit_results['popt'])
        
        baxes_er.plot(self.obs_wave/(1+self.D1_fit_results['popt'][0])*1e4, self.D1_spectrum-y_tot)
        baxes_er.set_ylim(-5*self.D1_spectrum_er[0], 5*self.D1_spectrum_er[0])
         
        self.fit_plot = [f,baxes]
        
    def fitting_collapse_OIII(self, plot=1, models='Outflow',simple=1, Fe_template=0,progress=True,sampler='emcee', N=6000,priors= {'z': [0,'normal_hat',0, 0, 0,0]}):
        priors_update = priors.copy()
        priors= {'z': [0,'normal_hat',0, 0, 0,0],\
                'cont':[0,'loguniform',-3,1],\
                'cont_grad':[0,'normal',0,0.2], \
                'OIII_peak':[0,'loguniform',-3,1],\
                'OIII_out_peak':[0,'loguniform',-3,1],\
                'Nar_fwhm':[300,'uniform', 100,900],\
                'BLR_fwhm':[5000,'uniform', 2000,9000],\
                'outflow_fwhm':[700,'uniform',600,2500],\
                'outflow_vel':[-50,'normal',0,200],\
                'Hbeta_peak':[0,'loguniform',-3,1],\
                'BLR_Hbeta_peak':[0,'loguniform',-3,1],\
                'Hbeta_out_peak':[0,'loguniform',-3,1],\
                'zBLR': [0,'normal_hat',0, 0, 0,0],\
                'Fe_peak':[0,'loguniform',-3,1],\
                'Fe_fwhm':[3000,'uniform',2000,6000]}
        
        for name in list(priors_update.keys()):
            priors[name] = priors_update[name]

        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
    
        if models=='Outflow':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_OIII(model='gal')
                
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_out.fitting_OIII(model='outflow')
            
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
            emplot.plotting_OIII(Fits_sig, ax1a)
            emplot.plotting_OIII(Fits_out, ax2a)
            
            
        elif models=='Single_only':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_OIII(model='gal' )
               
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.SNR =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'OIII')
            self.SNR_hb =  sp.SNR_calc(wave, flux, error, self.D1_fit_results, 'Hb')
            self.dBIC = 3
            
        elif models=='Outflow_only':
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_out.fitting_OIII(model='outflow', Fe_template=Fe_template )
                
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
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_OIII(model='BLR_simple', Fe_template=Fe_template )
                
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_out.fitting_OIII(model='BLR_outflow', Fe_template=Fe_template )
            
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
           
        elif models=='QSO_bkp':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors, sampler=sampler)
            Fits_sig.fitting_OIII(model='QSO_BKPL',Fe_template=Fe_template)
                
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
        
        
        self.D1_fit_full.corner()
        
        print(self.SNR)
        print(self.SNR_hb)
        
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_OIII(self.D1_fit_full, ax1, errors=True, residual='error', axres=ax2, template=Fe_template)
        plt.savefig(self.savepath+'Diagnostics/1D_spectrum_OIII_fit.pdf')

        self.fit_plot = [f,ax1,ax2]  
    
    def fitting_collapse_optical(self, plot=1, models='Outflow', progress=True, N=6000,priors= {'z': [0,'normal_hat',0, 0, 0,0]}):
        
        priors= {'z': [0,'normal_hat',0, 0, 0,0],\
                'cont':[0,'loguniform',-3,1],\
                'cont_grad':[0,'normal',0,0.2], \
                'OIII_peak':[0,'loguniform',-3,1],\
                'OIII_out_peak':[0,'loguniform',-3,1],\
                'Nar_fwhm':[300,'uniform', 100,900],\
                'BLR_fwhm':[5000,'uniform', 2000,9000],\
                'outflow_fwhm':[700,'uniform',600,2500],\
                'outflow_vel':[-50,'normal',0,200],\
                'Hbeta_peak':[0,'loguniform',-3,1],\
                'BLR_Hbeta_peak':[0,'loguniform',-3,1],\
                'Hbeta_out_peak':[0,'loguniform',-3,1],\
                'zBLR': [0,'normal_hat',0, 0, 0,0],\
                'Fe_peak':[0,'loguniform',-3,1],\
                'Fe_fwhm':[3000,'uniform',2000,6000]}
        
        for name in list(priors.keys()):
            priors[name] = priors[name]

        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
    
        if models=='Outflow':
            
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_optical(model='gal')
                
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_out.fitting_optical(model='outflow')
            
            if Fits_out.BIC-Fits_sig.BIC <-2:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                print('BICM', Fits_out.BIC)
                self.D1_fit_results = Fits_out.props
                self.D1_fit_chain = Fits_out.chains
                self.D1_fit_model = Fits_out.fitted_model
                self.D1_fit_full = Fits_out
                
                self.z = self.D1_fit_results['popt'][0]
                
                
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
                
            else:
                print('Delta BIC' , Fits_out.BIC-Fits_sig.BIC, ' ')
                
                self.D1_fit_results = Fits_sig.props
                self.D1_fit_chain = Fits_sig.chains
                self.D1_fit_model = Fits_sig.fitted_model
                self.D1_fit_full = Fits_sig
                
                self.z = self.D1_fit_results['popt'][0]
                
                
                self.dBIC = Fits_out.BIC-Fits_sig.BIC
            
            g, (ax1a,ax2a) = plt.subplots(2)
            emplot.plotting_optical(Fits_sig, ax1a)
            emplot.plotting_optical(Fits_out, ax2a)
            
            
        elif models=='Single_only':
            Fits_sig = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_sig.fitting_optical(model='gal' )
               
            self.D1_fit_results = Fits_sig.props
            self.D1_fit_chain = Fits_sig.chains
            self.D1_fit_model = Fits_sig.fitted_model
            self.D1_fit_full = Fits_sig
            
            self.z = self.D1_fit_results['popt'][0]
            
            self.dBIC = 3
            
        elif models=='Outflow_only':
            Fits_out = emfit.Fitting(wave, flux, error, self.z,N=N,progress=progress, priors=priors)
            Fits_out.fitting_optical(model='outflow' )
                
            print('BICM', Fits_out.BIC)
            self.D1_fit_results = Fits_out.props
            self.D1_fit_chain = Fits_out.chains
            self.D1_fit_model = Fits_out.fitted_model
            self.D1_fit_full = Fits_out
            
            self.z = self.D1_fit_results['popt'][0]
            self.dBIC = 3
             
        else:
            Exception('Sorry, models keyword not understood: Outflow_only, Single_only, Outflow')
        
        
        self.D1_fit_full.corner()
      
        f, (ax1, ax2) = plt.subplots(2, 1,  gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        plt.subplots_adjust(hspace=0)
        ax1.yaxis.tick_left()
        ax2.yaxis.tick_left()
        
        emplot.plotting_optical(self.D1_fit_full, ax1, errors=True, residual='error', axres=ax2)
        plt.savefig(self.savepath+'Diagnostics/1D_spectrum_optical_fit.pdf')

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

        f,ax = plt.subplots(1)

        ax.plot(self.wave, self.flux, drawstyle='steps-mid')
        ax.plot(self.wave, Fits_gen.yeval, 'r--')
        plt.savefig(self.savepath+'Diagnostics/1D_spectrum_general_fit.pdf')

            
            
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

        flux = self.D1_spectrum.data/(1e-7*1e4)*self.flux_norm
        obs_wave = self.obs_wave
        errors = self.D1_spectrum_er.data/(1e-7*1e4)*self.flux_norm
        import spectres as spres
        from . import jadify_temp as pth
        PATH_TO_jadify = pth.__path__[0]+ '/'
        with fits.open(PATH_TO_jadify+ 'temp_prism_clear_v3.0_extr3_1D.fits', memmap=False) as hdulist:
            wave_new = hdulist['wavelength'].data*1e6

        flux_new, error_new = spres.spectres( wave_new , obs_wave,flux,errors,fill=np.nan, verbose=False)
    
        sp.jadify(self.savepath+'PRISM_1D/prism_clear/100000', 'prism_clear', wave_new, flux_new, err=error_new, mask=np.zeros_like(wave_new),
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
        redshift_cat.write(self.savepath+'/PRISM_1D/redshift_1D.csv',overwrite=True)

        import nirspecxf
        id = 100000
        config100 = nirspecxf.NIRSpecConfig(self.savepath+'PRISM_1D/R100_1D_setup_manual.yaml')
        ns, _ = nirspecxf.process_object_id(id, config100)

        self.D1_ppxf = ns


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
                        if nspaxel==0:
                            nspaxel=1 
                        Var_er = np.sqrt(np.ma.sum(np.ma.array(data=self.error_cube.data, mask= total_mask)**2, axis=(1,2))/nspaxel)

                        error = sp.error_scaling(self.obs_wave, flx_spax_m, Var_er, err_range, boundary,\
                                                   exp=0)

                    sp.jadify(self.savepath+'PRISM_spaxel/prism_clear/00'+str(i)+str(j), 'prism_clear', self.obs_wave, flx_spax_m.data/(1e-7*1e4)*self.flux_norm, err=error/(1e-7*1e4)*self.flux_norm, mask=np.zeros_like(self.obs_wave),
                        overwrite=True, descr=None, author='jscholtz', verbose=False)
                    
        

    def unwrap_cube(self, rad=0.4,mask_manual=0, sp_binning='Nearest', add='', binning_pix=1, err_range=[0], boundary=2.4,instrument='NIRSPEC05'):
        """ Unwrapping the cube to prep it for spaxel-by-spaxel fitting. Saves the output as a pickle .txt object. 


        Parameters
        ----------
    
        rad : float
            radius of the circular aperture to select spaxel to fit (only used if mask_manual is not supplied)

        mask_manual: 2d - array
            2D array - with True value for spaxel you want ot fit. Ideal to select with QFitsView and load with QubeSpec.sp.Qfitsview_mask function

        sp_binning : str
            spatial binning - 'Single' - no binning, 'Nearest' - bins within 0.1 arcsec radius

        binning_pix: int -
            If sp_biiing = 'Nearest', how many pixels to bin over 1,2,3

        add: str - optional 
            add additional string to the saved file name for version/variations/names of companions. 

        err_range : list - optional
            same as for extracting 1D spectra and for the errors.

        boundary: float - optinal
            same as for extracting 1D spectra and for the errors.

        instrument: str
            not used anymore. Doesnt do anything. Will be remove soon.
            
           
        """
           
        flux = self.flux.copy()
        Mask= self.sky_clipped_1D
        shapes = self.dim


        ThD_mask = self.sky_clipped.copy()
        z = self.z
        wv_obs = self.obs_wave.copy()

        msk = Mask.copy()
        Spax_mask = self.Sky_stack_mask[0,:,:]

# =============================================================================
#   Unwrapping the cube
# =============================================================================
        try:
            arc = (self.header['CD2_2']*3600)

        except:
            arc = (self.header['CDELT2']*3600)

        upper_lim = 0
        step = binning_pix
    
        x = range(shapes[0]-upper_lim)
        y = range(shapes[1]-upper_lim)

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
            for j in y:
                if Spax_mask[i,j]==False:
                    
                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    if sp_binning=='Nearest':
                        Spax_mask_pick[:, i-step:i+step, j-step:j+step] = False
                    if sp_binning=='Single':
                        Spax_mask_pick[:, i, j] = False

                    if self.instrument=='NIRSPEC_IFU':
                        total_mask = np.logical_or(Spax_mask_pick, self.sky_clipped)
                        flx_spax_t = np.ma.array(data=flux.data,mask=total_mask)

                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=self.sky_clipped_1D)
                        nspaxel= np.sum(np.logical_not(total_mask[22,:,:]))
                        
                        Var_er = np.sqrt(np.ma.sum(np.ma.array(data=self.error_cube, mask= total_mask)**2, axis=(1,2))/nspaxel)
                        error = sp.error_scaling(self.obs_wave, flx_spax_m, Var_er, err_range, boundary,\
                                                   exp=0)
                        try:
                            error[error.mask==True] = np.ma.median(error)
                        except:
                            pass
                    else:
                        flx_spax_t = np.ma.array(data=flux.data,mask=Spax_mask_pick)
                        flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
                        flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)

                        error = stats.sigma_clipped_stats(flx_spax_m,sigma=3)[2] * np.ones(len(flx_spax))

                    Unwrapped_cube.append([i,j,flx_spax_m, error,wv_obs, z])


        print(len(Unwrapped_cube))
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube'+add+'.txt', "wb") as fp:
            pickle.dump(Unwrapped_cube, fp)
     

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

        # Creating a mask for all spaxels.
        mask_catch = self.flux.mask.copy()
        mask_catch[:,:,:] = True
        header  = self.header
        #arc = np.round(1./(header['CD2_2']*3600))
        arc = np.round(1./(header['CDELT2']*3600))
        
        
        if len(manual_mask)==0:
            print ('Center of cont', center)
            print ('Extracting spectrum from diameter', rad*2, 'arcseconds')
            print('Pixel scale:', arc)
            print ('radius ', arc*rad)
            # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
            for ix in range(self.dim[0]):
                for iy in range(self.dim[1]):
                    dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                    if dist< arc*rad:
                        mask_catch[:,ix,iy] = False
        else:
            print('Selecting spaxel based on the supplied mask.')
            for ix in range(self.dim[0]):
                for iy in range(self.dim[1]):
                    
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

            D1_spectrum_er = sp.error_scaling(self.obs_wave, D1_spectrum, D1_spectrum_var_er, err_range, boundary,\
                                                   exp=0)
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

    def PSF_matching(self, PSF_match=True, psf_fce=sp.NIRSpec_IFU_PSF, wv_ref=0, theta=None):
        from astropy.convolution import convolve_fft  
        from astropy.convolution import Gaussian2DKernel
        if PSF_match==True:
            print('Now PSF matching')
            if wv_ref == 0:
                wv_ref, = self.obs_wave[-1]
        
            theta = self.header['PA_V3']-138
            psf_matched = self.flux.copy()
            error_matched = self.error_cube.copy()
            for its in tqdm.tqdm(enumerate(self.obs_wave[ self.obs_wave<wv_ref])):
                i, wave = its
                
                sigma = np.sqrt(psf_fce(wv_ref)**2 - psf_fce(wave)**2)/0.05
                if wave<wv_ref:
                    kernel = Gaussian2DKernel( sigma[0],sigma[1], theta=theta)

                    psf_matched[i,:,:] = convolve_fft(psf_matched[i,:,:], kernel)
                    error_matched[i,:,:] = convolve_fft(error_matched[i,:,:], kernel)

            primary_hdu = fits.PrimaryHDU(np.zeros(1), header=self.header)

            hdus = [primary_hdu,fits.ImageHDU(psf_matched.data, name='SCI', header=self.header), fits.ImageHDU(error_matched, name='ERR', header=self.header)]
            hdulist = fits.HDUList(hdus)
            hdulist.writeto( self.Cube_path[:-4] +'psf_matched.fits', overwrite=True)

            psf_matched[np.isnan(self.flux.data)] = np.nan


            self.flux = np.ma.masked_invalid(psf_matched.data.copy())
            self.error = error_matched.copy()
            
            self.mask_JWST(plot=0, threshold=self.masking_threshold, spe_ma= self.channel_mask)
        else:
            print('You asked to do PSF matching, but PSF_match keyword is False. I am skipping PSF matching.')
        

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
        
        
        