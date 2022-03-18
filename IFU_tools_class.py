#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:36:26 2017

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import wcs
from astropy.nddata import Cutout2D

import Fitting_tools_mcmc as emfit
import Plotting_tools_v2 as emplot
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import emcee
import corner

from astropy import stats

nan= float('nan')

pi= np.pi
e= np.e

c= 3.*10**8
h= 6.62*10**-34
k= 1.38*10**-23

Ken98= (4.5*10**-44)
Conversion2Chabrier=1.7 # Also Madau
Calzetti12= 2.8*10**-44
arrow = u'$\u2193$' 


PATH='/Users/jansen/My Drive/Astro/'

PATH_store = PATH+'KASHz/'

OIIIr = 5008.
OIIIb = 4960
Hal = 6562.8   
NII_r = 6583.
NII_b = 6548.

# =============================================================================
# Useful function 
# =============================================================================
def gauss(x,k,mu,sig):
    expo= -((x-mu)**2)/(2*sig*sig)
    
    y= k* e**expo
    
    return y

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def conf(aray):
    sorted_array= np.array(sorted(aray))
    leng= (float(len(aray))/100)*16
    leng= int(leng)
    
    
    hgh = sorted_array[-leng]
    low = sorted_array[leng]
    
    return low, hgh

def twoD_Gaussian(dm, amplitude, xo, yo, sigma_x, sigma_y, theta, offset): 
    x, y = dm
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def Av_calc(Falpha, Fbeta):
    
    Av = 1.964*4.12*np.log10(Falpha/Fbeta/2.86)
    
    return Av

def Flux_cor(Flux, Av, lam= 0.6563):
    
    
    Ebv = Av/4.12
    Ahal = 3.325*Ebv
    
    F = Flux*10**(0.4*Ahal)
    
    return F



def prop_calc(results):  
    labels = list(results.keys())[1:]
    res_plt = []
    res_dict = {'name': results['name']}
    for lbl in labels:
        
        array = results[lbl]
        
        p50,p16,p84 = np.percentile(array, (50,16,84))
        p16 = p50-p16
        p84 = p84-p50
        
        res_plt.append(p50)
        res_dict[lbl] = np.array([p50,p16,p84])
        
    res_dict['popt'] = res_plt
    return res_dict
        

def SNR_calc(wave,flux, std, sol, mode):
    wave = wave[np.invert(flux.mask)]
    flux = flux.data[np.invert(flux.mask)]
    
    wv_or = wave.copy()
    
    if mode =='OIII':
        center = OIIIr*(1+sol[0])/1e4
        if len(sol)==5:
            fwhm = sol[4]/3e5*center
            
            model = flux- (wave*sol[2]+sol[1]) #gauss(wave, sol[3], center, fwhm/2.35) # emfit.OIII(wave,  *sol)
        elif len(sol)==8:
            fwhms = sol[5]/3e5*center
            fwhm = sol[6]/3e5*center
            
            center = OIIIr*(1+sol[0])/1e4
            centerw = OIIIr*(1+sol[0])/1e4 + sol[7]/3e5*center
            
            model = emfit.OIII_outflow(wave,  *sol)- (wave*sol[2]+sol[1]) #gauss(wave, sol[3], center, fwhms/2.35) + gauss(wave, sol[4], centerw, fwhm/2.35)  #emfit.OIII_outflow(wave,  *sol)- wave*sol[2]+sol[1]
            
        
    elif mode =='Hn':
        center = Hal*(1+sol[0])/1e4
        if len(sol)==8:
            fwhm = sol[5]/3e5*center
            model = gauss(wave, sol[3], center, fwhm/2.35)
        elif len(sol)==11:
            fwhm = sol[6]/3e5*center*2
            model = gauss(wave, sol[3], center, fwhm/2.35)
    
    elif mode =='Hblr':
        center = Hal*(1+sol[0])/1e4
        
        fwhm = sol[7]/3e5*center
        model = gauss(wave, sol[4], center, fwhm/2.35)
            
    elif mode =='NII':
        center = NII_r*(1+sol[0])/1e4
        if len(sol)==8:
            fwhm = sol[5]/3e5*center
            model = gauss(wave, sol[4], center, fwhm/2.35)
        elif len(sol)==11:
            fwhm = sol[6]/3e5*center
            model = gauss(wave, sol[5], center, fwhm/2.35)
            
      
    else:
        raise Exception('Sorry mode in SNR_calc not understood')
    
    use = np.where((wave< center+fwhm*2)&(wave> center-fwhm*2))[0]   
    flux_l = model[use]
    
    n = len(use)
    SNR = (sum(flux_l)/np.sqrt(n)) * (1./std)
    if SNR < 0:
        SNR=0
    
    return SNR  

def BIC_calc(wave,fluxm,error, model, results, mode):
    popt = results['popt']
    z= popt[0]
    
    if mode=='OIII':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>4900*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = sum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    if mode=='Halpha':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>(6562.8-600)*(1+z)/1e4)&(wave<(6562.8+600)*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = sum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    return chi2, BIC

def unwrap_chain(res):
    keys = list(res.keys())[1:]
    
    chains = np.zeros(( len(res[keys[0]]), len(keys) ))
    
    for i in range(len(keys)):
        
        chains[:,i] = res[keys[i]]
        
    return chains
    

def flux_calc(res, mode):
    
    if mode=='OIIIt':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if len(res['popt'])==8:
            o3 = 5008*(1+res['z'][0])/1e4
            
            o3n = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
        
            o3w = gauss(wave, res['OIIIw_peak'][0], o3, res['OIIIw_fwhm'][0]/2.355/3e5*o3  )*1.333
            
            model = o3n+o3w
        elif len(res['popt'])==5:
            o3 = 5008*(1+res['z'][0])/1e4
            
            model = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333

    elif mode=='OIIIn':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if len(res['popt'])==8:
            o3 = 5008*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
        
        
        elif len(res['popt'])==5:
            o3 = 5008*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIIIn_peak'][0], o3, res['OIIIn_fwhm'][0]/2.355/3e5*o3  )*1.333
    
    elif mode=='OIIIw':
        wave = np.linspace(4900, 5100,300)*(1+res['z'][0])/1e4
        if len(res['popt'])==8:
            o3 = 5008*(1+res['z'][0])/1e4
            model = gauss(wave, res['OIIIw_peak'][0], o3, res['OIIIw_fwhm'][0]/2.355/3e5*o3  )*1.333
        elif len(res['popt'])==5:
            model = np.zeros_like(wave)
    
    elif mode=='Han':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        hn = 6565*(1+res['z'][0])/1e4
        
        model = gauss(wave, res['Hal_peak'][0], hn, res['Nar_fwhm'][0]/2.355/3e5*hn  )
    
    elif mode=='Hblr':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        hn = 6565*(1+res['z'][0])/1e4
        
        model = gauss(wave, res['BLR_peak'][0], hn, res['BLR_fwhm'][0]/2.355/3e5*hn  )
    
    elif mode=='NII':
        wave = np.linspace(6300,6700,300)*(1+res['z'][0])/1e4
        nii = 6583*(1+res['z'][0])/1e4
        model = gauss(wave, res['NII_peak'][0], nii, res['Nar_fwhm'][0]/2.355/3e5*nii  )*1.333
        
    import scipy.integrate as scpi
        
    Flux = scpi.simps(model, wave)*1e-13
        
    return Flux
        
        
    
# ============================================================================
#  Main class
# =============================================================================
class Cube:
    
    def __init__(self, Full_path, z, ID, flag, savepath, Band):
        import importlib
        importlib.reload(emfit )
    
        filemarker = fits.open(Full_path)
        
        print (Full_path)
        if flag=='KMOS':
            
            header = filemarker[1].header # FITS header in HDU 1
            flux_temp  = filemarker[1].data/1.0e-13   
                             
            filemarker.close()  # FITS HDU file marker closed
        
        elif flag=='Sinfoni':
            header = filemarker[0].header # FITS header in HDU 1
            flux_temp  = filemarker[0].data/1.0e-13*1e4
                             
            filemarker.close()  # FITS HDU file marker closed
        
        else:
            print ('Instrument Flag is not understood!')
        
        flux  = np.ma.masked_invalid(flux_temp)   #  deal with NaN
       
        # Number of spatial pixels
        n_xpixels = header['NAXIS1']
        n_ypixels = header['NAXIS2']
        dim = [n_ypixels, n_xpixels]
    
        #  Number of spectral pixels
        n_spixels = header['NAXIS3']
        dim = [n_ypixels, n_xpixels, n_spixels]
        
            
        try:
            x = header['CDELT1']
        except:
            header['CDELT1'] = header['CD1_1']
        
        
        try:
            x = header['CDELT2']
        except:
            header['CDELT2'] = header['CD2_2']
            
        
        try:
            x = header['CDELT3']
        except:
            header['CDELT3'] = header['CD3_3']
        
        wave = header['CRVAL3'] + (np.arange(n_spixels) - (header['CRPIX3'] - 1.0))*header['CDELT3']
         
        
        try:
            deg_per_pix_x = abs(header['CDELT1'])
            
        except:
            deg_per_pix_x = header['CDELT1']
            
            
        arc_per_pix_x = 1.*deg_per_pix_x*3600
        Xpix = header['NAXIS1']
        Xph = Xpix*arc_per_pix_x
        
        try:
            deg_per_pix_y = abs(header['CDELT2'])
        
        except:
            deg_per_pix_y = header['CDELT2']
            header['CD2_2'] = header['CDELT2']
            
        arc_per_pix_y = deg_per_pix_y*3600
        Ypix = header['NAXIS2']
        Yph = Ypix*arc_per_pix_y
        
        self.dim = dim
        self.z = z
        self.obs_wave = wave
        self.flux = flux
        self.ID = ID
        self.telescope = flag
        self.savepath = savepath
        self.header= header
        self.phys_size = np.array([Xph, Yph])
        self.band = Band
        
    
    def add_res(self, line_cat):
        
        self.cat = line_cat
        

    def mask_emission(self):
        '''This function masks out all the OIII and HBeta emission
        '''
        z= self.z
        OIIIa=  501./1e3*(1+z)
        OIIIb=  496./1e3*(1+z)
        Hbeta=  485./1e3*(1+z)
        width = 0.006
    
        mask =  self.flux.mask.copy()
        wave= self.obs_wave
    
        OIIIa_loc = np.where((wave<OIIIa+width)&(wave>OIIIa-width))[0]
        OIIIb_loc = np.where((wave<OIIIb+width)&(wave>OIIIb-width))[0]
        Hbeta_loc = np.where((wave<Hbeta+width)&(wave>Hbeta-width))[0]
    
        mask[OIIIa_loc,:,:] = True
        mask[OIIIb_loc,:,:] = True
        mask[Hbeta_loc,:,:] = True
        
        self.em_line_mask= mask
        

    def mask_sky(self,sig, mode=0):
            
        bins = np.linspace(0,2048,5)
        bins= np.array(bins, dtype=int)
        
        flux = np.ma.array(data=self.flux.data.copy(), mask= self.em_line_mask.copy())    
        mask =  self.em_line_mask.copy()  
        dim = self.dim
        wave = self.obs_wave
        x=0
        
        if mode=='Hsin':
            use = np.where((wave>1.81)&(wave<1.46))[0]
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
        
        flux = np.ma.array(data=flux.data, mask=mask)
        
        noise_spax = np.ma.std(flux, axis=(0))
                


    def collapse_white(self, plot):
        flux = np.ma.array(self.flux.data, mask=self.sky_line_mask_em) 
        
        median = np.ma.median(flux, axis=(0))  
        
        ID = self.ID
        if ID== 'ALESS_75':
            wave = self.obs_wave.copy()
            
            use = np.where((wave < 1.7853) &(wave > 1.75056))[0]
            
            use = np.append(use, np.where((wave < 2.34) &(wave > 2.31715))[0] )
            
            median = np.ma.median(flux[use,:,:], axis=(0))  
            
       
        self.Median_stack_white = median
        
      
        if plot==1:
            plt.figure()
            plt.imshow(median,  origin='lower')
            plt.colorbar()
            
        

    def find_center(self, plot, extra_mask=0, manual=np.array([0])):
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
            dm = (x,y)
            popt, pcov = opt.curve_fit(twoD_Gaussian, dm, data.ravel(), p0=initial_guess)
            
            er = np.sqrt(np.diag(pcov))
        
            print ('Cont loc ', popt[1:3])
            print ('Cont loc er', er[1:3])
            self.center_data = popt 
            
            
            
        else:
            manual = np.append(data[int(manual[0]), int(manual[1])], manual)
            manual = np.append(manual, np.array([2.,2.,0.5,0. ]))
            self.center_data = manual
        
        
    
    def choose_pixels(self, plot, rad= 0.6, flg=1):
        ''' Choosing the pixels that will collapse into the 1D spectrum. Also this mask is used later
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
        print ('radius ', arc*rad)
        
        
        # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc*rad:
                    mask_catch[:,ix,iy] = False
        
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
    
            data_fit = twoD_Gaussian((x,y), *self.center_data)
    
            plt.contour(x, y, data_fit.reshape(shapes[0], shapes[1]), 8, colors='w')
            
    def D1_spectra_collapse(self, plot, addsave=''):
        '''
        This function collapses the Cube to form a 1D spectrum of the galaxy
        '''
        # Loading the Signal mask - selects spaxel to collapse to 1D spectrum 
        mask_spax = self.Signal_mask.copy()
        # Loading mask of the sky lines an bad features in the spectrum 
        mask_sky_1D = self.sky_clipped_1D.copy()
        
        
        total_mask = mask_spax 
        # M
        flux = np.ma.array(data=self.flux.data, mask= total_mask) 
        
        D1_spectra = np.ma.sum(flux, axis=(1,2))
        D1_spectra = np.ma.array(data = D1_spectra.data, mask=mask_sky_1D)
        
        wave= self.obs_wave
        
        if plot==1:
            plt.figure()
            plt.title('Collapsed 1D spectrum from D1_spectra_collapse fce')
            
            plt.plot(wave, D1_spectra, drawstyle='steps-mid', color='grey')
            plt.plot(wave, np.ma.array(data= D1_spectra, mask=self.sky_clipped_1D), drawstyle='steps-mid')
            
        
            plt.ylabel('Flux')
            plt.xlabel('Observed wavelength')
       
        
        self.D1_spectrum = D1_spectra
        self.D1_spectrum_er = stats.sigma_clipped_stats(D1_spectra,sigma=3)[2]*np.ones(len(self.D1_spectrum)) #STD_calc(wave/(1+self.z)*1e4,self.D1_spectrum, self.band)* np.ones(len(self.D1_spectrum))
        
        Save_spec = np.zeros((4,len(D1_spectra)))
        
        Save_spec[0,:] = wave
        Save_spec[1,:] = self.D1_spectrum
        Save_spec[2,:] = self.D1_spectrum_er.copy()
        Save_spec[3,:] = mask_sky_1D
        

        
        np.savetxt(self.savepath+self.ID+'_'+self.band+addsave+'_1Dspectrum.txt', Save_spec)
        



    def stack_sky(self,plot, spe_ma=np.array([], dtype=bool), expand=0):
        header = self.header
        ID = self.ID
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
            weird = np.where((wave>1.25) &(wave<1.285))[0]
            weird = np.append(weird, np.where((wave<1.020))[0])
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
        print (len(ssma))
        low, hgh = conf(ssma)   
        
        
        sky = np.where((stacked_sky<y+ low*clip_w) | (stacked_sky> y + hgh*clip_w))[0]
        
        sky_clipped =  self.flux.mask.copy()
        sky_clipped = sky_clipped[:,7,7]
        sky_clipped[sky] = True          # Masking the sky features
        sky_clipped[weird] = True        # Masking the weird features
            
          
        # Storing the 1D sky line mask into cube to be used later
        mask_sky = self.flux.mask.copy()
        

        
        #if (storage['X-ray ID']=='HB89') & (Band=='Hsin'):
        #    sky_clipped[np.where((wave< 1.72344 ) & (wave > 1.714496))[0]] = False
        
        
            
        
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                mask_sky[:,ix,iy] = sky_clipped
        
        # Storing the 1D and 3D sky masks
        self.sky_clipped = mask_sky
        self.sky_clipped_1D = sky_clipped 
        
        self.Stacked_sky = stacked_sky
        
        
        np.savetxt(self.savepath, sky_clipped)
        

        
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
    
    def fitting_collapse_Halpha(self, plot, broad = 1):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        fl = flux.data
        msk = flux.mask
        
        flux = np.ma.array(data=fl, mask = msk)
        
        
        flat_samples_sig, fitted_model_sig = emfit.fitting_Halpha(wave,flux,error,z, BLR=0)

        prop_sig = prop_calc(flat_samples_sig)
        
        
        y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
        chi2S = sum(((flux.data-y_model_sig)/error)**2)
        BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
        
        flat_samples_blr, fitted_model_blr = emfit.fitting_Halpha(wave,flux,error,z, BLR=1)
        prop_blr = prop_calc(flat_samples_blr)
        
        chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'Halpha')
        chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_blr, prop_blr, 'Halpha')
        
        
        
        
        if BICM-BICS <-2:
            print('Delta BIC' , BICM-BICS, ' ')
            print('BICM', BICM)
            self.D1_fit_results = prop_blr
            self.D1_fit_chain = flat_samples_blr
            self.D1_fit_model = fitted_model_blr
            
            self.z = prop_blr['popt'][0]
            
            self.SNR =  SNR_calc(wave, flux, error[0], self.D1_fit_results['popt'], 'Hblr')
            self.dBIC = BICM-BICS
            labels=('z', 'cont','cont_grad', 'Hal_peak','BLR_peak', 'NII_peak', 'Nar_fwhm', 'BLR_fwhm', 'BLR_offset', 'SIIr_peak', 'SIIb_peak')
        else:
            print('Delta BIC' , BICM-BICS, ' ')
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            
            self.SNR =  SNR_calc(wave, flux, error[0], self.D1_fit_results['popt'], 'Hn')
            self.dBIC = BICM-BICS
            labels=('z', 'cont','cont_grad', 'Hal_peak', 'NII_peak', 'Nar_fwhm', 'SIIr_peak', 'SIIb_peak')
            
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print(self.SNR)
        
        f, ax1 = plt.subplots(1)
        
        emplot.plotting_Halpha(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model)
        
        
        
    def fitting_collapse_OIII(self,  plot, outflow=0):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=0)
        prop_sig = prop_calc(flat_samples_sig)
        
        y_model_sig = fitted_model_sig(wave, *prop_sig['popt'])
        chi2S = sum(((flux.data-y_model_sig)/error)**2)
        BICS = chi2S+ len(prop_sig['popt'])*np.log(len(flux))
        
        flat_samples_out, fitted_model_out = emfit.fitting_OIII(wave,flux,error,z, outflow=1)
        prop_out = prop_calc(flat_samples_out)
        
        chi2S, BICS = BIC_calc(wave, flux, error, fitted_model_sig, prop_sig, 'OIII')
        chi2M, BICM = BIC_calc(wave, flux, error, fitted_model_out, prop_out, 'OIII')
        
        
        if BICM-BICS <-2:
            print('Delta BIC' , BICM-BICS, ' ')
            self.D1_fit_results = prop_out
            self.D1_fit_chain = flat_samples_out
            self.D1_fit_model = fitted_model_out
            self.z = prop_out['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error[0], self.D1_fit_results['popt'], 'OIII')
            self.dBIC = BICM-BICS
            
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIw_peak', 'OIIIn_fwhm', 'OIIIw_fwhm', 'out_vel')
            
            
            
            
        
        else:
            print('Delta BIC' , BICM-BICS, ' ')
            self.D1_fit_results = prop_sig
            self.D1_fit_chain = flat_samples_sig
            self.D1_fit_model = fitted_model_sig
            self.z = prop_sig['popt'][0]
            self.SNR =  SNR_calc(wave, flux, error[0], self.D1_fit_results['popt'], 'OIII')
            self.dBIC = BICM-BICS
            
            labels=('z', 'cont','cont_grad', 'OIIIn_peak', 'OIIIn_fwhm')
            
            
        fig = corner.corner(
            unwrap_chain(self.D1_fit_chain), 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12})
        
        print(self.SNR)
        
        
        f, ax1 = plt.subplots(1)
        
        emplot.plotting_OIII(wave, flux, ax1, self.D1_fit_results ,self.D1_fit_model)
        

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
        
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), img.ravel(), p0=initial_guess)
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
        
        
        data_fit = twoD_Gaussian((x,y), *popt_cube)
        
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
        


    def astrometry_correction_GAIA(self, path_to_gaia):
        '''
        Correcting the Astrometry of the Cube. Fits a 2D Gaussian to the HST image and assumes that the 
        HST and Cube centroids are in the same location. 
        '''
        
        from astropy.table import Table
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
                
     
    def unwrap_cube(self, rad=0.4, sp_binning='Nearest', instrument='KMOS', add=''):
        flux = self.flux.copy()
        Mask= self.sky_clipped_1D
        shapes = self.dim
        ID = self.ID
        
        ThD_mask = self.sky_clipped.copy()
        z = self.z
        wv_obs = self.obs_wave.copy()
        
        Residual = np.zeros_like(flux).data
        Model = np.zeros_like(flux).data
            
          
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
            
        x = range(shapes[0]-upper_lim)
        y = range(shapes[1]-upper_lim) 
        
        h, w = self.dim[:2]
        center= self.center_data[1:3]
        
        mask = create_circular_mask(h, w, center= center, radius= rad/arc)
        mask = np.invert(mask)
        
        Spax_mask = np.logical_or(np.invert(Spax_mask),mask)
        
        plt.figure()
        
        plt.imshow(np.ma.array(data=self.Median_stack_white, mask=Spax_mask), origin='lower')
        
        Unwrapped_cube = []        
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                j=j+step
                if Spax_mask[i,j]==False:
                    #print i,j
                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                    
                    #Spax_mask_pick= np.logical_or(Spax_mask_pick, ThD_mask)
                    flx_spax_t = np.ma.array(data=flux,mask=Spax_mask_pick)
                    
                    flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                    flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)                
                    error = stats.sigma_clipped_stats(flx_spax_m,sigma=3)[2] * np.ones(len(flx_spax))
                    
                    Unwrapped_cube.append([i,j,flx_spax_m, error])
                    if i==20 and j==20:
                        plt.figure()
                        plt.plot(wv_obs, flx_spax_m, color='grey', drawstyle='steps-mid', alpha=0.2)
                         
                        fluxplt = flx_spax_m.data[np.invert(flx_spax_m.mask)]
                        wv_obs = wv_obs[np.invert(flx_spax_m.mask)]
                          
                          
                        plt.plot(wv_obs, fluxplt, drawstyle='steps-mid')

        
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "wb") as fp:
            pickle.dump(Unwrapped_cube, fp)      
        
    
    
    def Spaxel_fitting_OIII_MCMC(self):
        import pickle
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        results = []
        
        for i in range(len(Unwrapped_cube)):
            
            row = Unwrapped_cube[i]
            
            results.append( emfit.Fitting_OIII_unwrap(row, self.obs_wave, self.z))
        
        self.spaxel_fit_raw = results
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw.txt', "wb") as fp:
            pickle.dump( results,fp)     
    
    def Spaxel_fitting_OIII_lqs(self):
        import pickle
        import Fitting_tools_lqs as emfit_lqs
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        results = []
        
        for i in range(len(Unwrapped_cube)):
            
            row = Unwrapped_cube[i]
            
            results.append( emfit_lqs.Fitting_OIII_unwrap(row, self.obs_wave, self.z))
        
        self.spaxel_fit_raw = results
        
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw.txt', "wb") as fp:
            pickle.dump( results,fp)     
   
    
    def Map_creation_OIII(self):
        z0 = self.z
    
        wvo3 = 5008*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        with open(self.savepath+self.ID+'_'+self.band+'_spaxel_fit_raw.txt', "rb") as fp:
            results= pickle.load(fp)
            
        with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
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
        # =============================================================================
        #        Filling these maps  
        # =============================================================================
        f,ax= plt.subplots(1)
        import Plotting_tools_v2 as emplot
        
        Spax = PdfPages(self.savepath+self.ID+'_Spaxel_OIII_fit_detection_only.pdf')
        
        
        for row in range(len(results)):
            i,j, res_spx = results[row]
            i,j, flx_spax_m, error = Unwrapped_cube[row]
            
            z = res_spx['popt'][0]
            SNR = SNR_calc(self.obs_wave, flx_spax_m, error[0], res_spx['popt'], 'OIII')
            map_snr[i,j]= SNR
            if SNR>3:
            
                map_vel[i,j] = ((5008*(1+z)/1e4)-wvo3)/wvo3*3e5
                map_fwhm[i,j] = res_spx['popt'][4]
                map_flux[i,j] = flux_calc(res_spx, 'OIIIt')
                
                
                emplot.plotting_OIII(self.obs_wave, flx_spax_m, ax, res_spx, emfit.OIII)
                ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
                Spax.savefig()  
                ax.clear()
          
        Spax.close() 
        
        self.Flux_map = map_flux
        self.Vel_map = map_vel
        self.FWHM_map = map_fwhm
        self.SNR_map = map_snr
        
        f, axes = plt.subplots(2,2, figsize=(10,10))
        axes[0,0].imshow(map_flux,vmax=5e-18, origin='lower')
        axes[0,0].set_title('Flux map')
        
        
        axes[1,0].imshow(map_vel, cmap='coolwarm', origin='lower', vmin=-200, vmax=200)
        axes[1,0].set_title('Velocity offset map')
        
        
        axes[0,1].imshow(map_fwhm,vmin=100, origin='lower')
        axes[0,1].set_title('FWHM map')
        
        axes[1,1].imshow(map_snr,vmin=3, vmax=20, origin='lower')
        axes[1,1].set_title('SNR map')
        
        
# =============================================================================
# Old and to be developed code        
# =============================================================================
def Spaxel_fitting_OIII_mp(self):
    import pickle
    with open(self.savepath+self.ID+'_'+self.band+'_Unwrapped_cube.txt', "rb") as fp:
        Unwrapped_cube= pickle.load(fp)
        
    print('import of the unwrap cube - done')
    

    ## DONT FORGET TO RUN THE FOLLOWING COMMAND IN TERMINAL
    ## ipcluster start -n 4
    ## where 4 is the number of parallel processes.
    from ipyparallel import Client
    import itertools
    
    cs = Client()
    view = cs[:]
    view.activate()
    
    with view.sync_imports():
        import numpy as np
        from astropy.io import fits as pyfits
        from astropy import wcs
        from astropy.table import Table, join, vstack
        from matplotlib.backends.backend_pdf import PdfPages
        import pickle
        from scipy.optimize import curve_fit

        import emcee
        import corner
        import Fitting_tools_mcmc as emfit
    
        
        
    ## This is how you declare variables in all the nodes at once...
    
    view.push(dict(c = 3.0e8, N=2000))
    #b3 = view.map_sync(fitting_data_leastsq,fluxes_full[:N_max],Noise_sig[:N_max],itertools.repeat(wavelengths_full,N_max))
    #flat_samples_sig, fitted_model_sig = emfit.fitting_OIII(wave,flux,error,z, outflow=0)
    
    b3 = view.map_sync(emfit.Fitting_OIII_unwrap, Unwrapped_cube, itertools.repeat(self.obs_wave, self.z))     

'''

def W68_mes(out, mode, plot):
    import scipy.integrate as scpi
    
    if mode=='OIII':
        N= 5000
        
        cent = out.params['o3rn_center'].value
        
        bound1 =  cent + 4000/3e5*cent
        bound2 =  cent - 4000/3e5*cent
              
        wvs = np.linspace(bound2, bound1, N)
        
        try:            
            y = out.eval_components(x=wvs)['o3rw_'] + out.eval_components(x=wvs)['o3rn_']
        
        except:
            y = out.eval_components(x=wvs)['o3rn_']
                            
        Int = np.zeros(N-1)
        
        for i in range(N-1):
            
            Int[i] = scpi.simps(y[:i+1], wvs[:i+1]) * 1e-13
                    
        Int = Int/max(Int)

        ind16 = np.where( (Int>0.16*0.992)& (Int<0.16/0.992) )[0][0]            
        ind84 = np.where( (Int>0.84*0.995)& (Int<0.84/0.995) )[0][0] 
        ind50 = np.where( (Int>0.5*0.992)& (Int<0.5/0.992) )[0][0]            
        
        wv10 = wvs[ind16]
        wv90 = wvs[ind84]
        wv50 = wvs[ind50]
        
        v10 = (wv10-cent)/cent*3e5
        v90 = (wv90-cent)/cent*3e5
        v50 = (wv50-cent)/cent*3e5
        
        
        w80 = v90-v10
        
        
        if plot==1:
            f,ax1 = plt.subplots(1)
            ax1.plot(wvs,y, 'k--')
        
        
            g, ax2 = plt.subplots(1)
            ax2.plot(wvs[:-1], Int)
        
            ax2.plot(np.array([bound2,bound1]), np.array([0.9,0.9]), 'r--')
            ax2.plot(np.array([bound2,bound1]), np.array([0.1,0.1]), 'r--')
        
            ax2.plot(np.array([cent,cent]), np.array([0,1]), 'b--')
        
            ax1.plot(np.array([wv10,wv10]), np.array([0, max(y)]), 'r--')
            ax1.plot(np.array([wv90,wv90]), np.array([0, max(y)]), 'r--')
            
    
    elif mode=='CO':
        N= 5000
        
        cent = out.params['COn_center'].value
        
        bound1 =  cent + 4000/3e5*cent
        bound2 =  cent - 4000/3e5*cent
              
        wvs = np.linspace(bound2, bound1, N)
        
        try:            
            y = out.eval_components(x=wvs)['COn_'] + out.eval_components(x=wvs)['COb_']
        
        except:
            y = out.eval_components(x=wvs)['COn_']
                            
        Int = np.zeros(N-1)
        
        for i in range(N-1):
            
            Int[i] = scpi.simps(y[:i+1], wvs[:i+1]) * 1e-13
                    
        Int = Int/max(Int)
        try:
            
            ind10 = find_nearest(Int, 0.16)          
               
        
        except:
            print( np.where( (Int>0.16*0.991)& (Int<0.16/0.991) )[0]  )
            plt.figure()
            plt.plot(wvs[:-1], Int)
           
            
            
        ind90 = np.where( (Int>0.84*0.995)& (Int<0.84/0.995) )[0][0] 
        ind50 = np.where( (Int>0.5*0.992)& (Int<0.5/0.992) )[0][0] 
        
        wv10 = wvs[ind10]
        wv90 = wvs[ind90]
        wv50 = wvs[ind50]
        
        v10 = (wv10-cent)/cent*3e5
        v90 = (wv90-cent)/cent*3e5
        v50 = (wv50-cent)/cent*3e5
        
        
        w80 = v90-v10
        
        
        if plot==1:
            f,ax1 = plt.subplots(1)
            ax1.plot(wvs,y, 'k--')
        
        
            g, ax2 = plt.subplots(1)
            ax2.plot(wvs[:-1], Int)
        
            ax2.plot(np.array([bound2,bound1]), np.array([0.9,0.9]), 'r--')
            ax2.plot(np.array([bound2,bound1]), np.array([0.1,0.1]), 'r--')
        
            ax2.plot(np.array([cent,cent]), np.array([0,1]), 'b--')
        
            ax1.plot(np.array([wv10,wv10]), np.array([0, max(y)]), 'r--')
            ax1.plot(np.array([wv90,wv90]), np.array([0, max(y)]), 'r--')
    
    return v10,v90, w80, v50




def Sub_QSO(storage_H):
    ID = storage_H['X-ray ID']
    z = storage_H['z_guess']
    flux = storage_H['flux'].copy()
    Mask = storage_H['sky_clipped']
    Mask_1D = storage_H['sky_clipped_1D']
    out = storage_H['1D_fit_Halpha_mul']
    
    wv_obs = storage_H['obs_wave'].copy()
    
    rst_w = storage_H['obs_wave'].copy()*1e4/(1+storage_H['z_guess'])
        
    center =  storage_H['Median_stack_white_Center_data'][1:3].copy()
    sig = storage_H['Signal_mask'][0,:,:].copy()
    sig[:,:] = True
    
    shapes = storage_H['dim']
    # This choose spaxel within certain radius. Then sets it to False since we dont mask those pixels
    for ix in range(shapes[0]):
        for iy in range(shapes[1]):
            dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
            if dist< 6:
                sig[ix,iy] = False
        
    Mask_em = Mask_1D.copy()
    Mask_em[:] = False
        
    wmin = out.params['Nb_center'].value - 1*out.params['Nb_sigma'].value
    wmax = out.params['Nr_center'].value + 1*out.params['Nr_sigma'].value

    
        
    em_nar = np.where((wv_obs>wmin) & (wv_obs<wmax))[0]        
    Mask_em[em_nar] = True
    
    
    
    wmin = (6718.-5)*(1+z)/1e4
    wmax = (6732.+5)*(1+z)/1e4
    
    em_nar = np.where((wv_obs>wmin) & (wv_obs<wmax))[0]        
    Mask_em[em_nar] = True

       
    comb = np.logical_or(Mask_1D, Mask_em)
    
    shapes = storage_H['dim']
    Cube = np.zeros((shapes[2], shapes[0], shapes[1]))
    C_ms = Cube[:,:,:].copy()
    C_ms[:,:,:] = False
    Cube = np.ma.array(data=Cube, mask = C_ms)
    
    BLR_map = np.zeros((shapes[0], shapes[1]))
    
    
    
    x = range(shapes[0])
    y = range(shapes[1])
    
    #x = np.linspace(30,60,31)
    #y = np.linspace(30,60,31)
    
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)
        
    ls,ax = plt.subplots(1)
    pdf_plots = PdfPages(PATH+'Four_Quasars/Graphs/'+ID+'_SUB_QSO.pdf') 
    #import progressbar
    for i in x:# progressbar.progressbar(x):
        print (i,'/',x[-1])
        for j in y:
            flx_pl = flux[:,i,j]
            flxm = np.ma.array(data=flx_pl, mask=comb)
            
                
            error = STD_calc(wv_obs/(1+z)*1e4,flxm, 'H')* np.ones(len(flxm))
            
            plt.figure
            
            plt.plot(wv_obs, flxm.data)
            
            try:
                
            
                out = emfit.sub_QSO(wv_obs, flxm, error,z, storage_H['1D_fit_Halpha_mul'])            
                sums = flxm.data-(out.eval_components(x=wv_obs)['Ha_']+out.eval_components(x=wv_obs)['linear'])
                Cube[:,i,j] = sums
                
                BLR_map[i,j] = np.sum(out.eval_components(x=wv_obs)['Ha_'])
                    
                    
                if sig[i,j]==False:
                    ax.plot(wv_obs, flxm.data, label='data')
                    ax.plot(wv_obs,out.eval_components(x=wv_obs)['Ha_']+out.eval_components(x=wv_obs)['linear'], label='model')
                    ax.plot(wv_obs, sums, label='Res')
                        
                    ax.legend(loc='best')
                    wmin = out.params['Ha_center'].value - 3*out.params['Ha_sigma'].value
                    wmax = out.params['Ha_center'].value + 3*out.params['Ha_sigma'].value
                        
                    ax.set_xlim(wmin,wmax)
                        
                    pdf_plots.savefig()
                    
                    ax.clear()
            except:
                Cube[:,i,j] = flxm.data
                
                BLR_map[i,j] =0
                
                print (i,j, ' BLR sub fail')
                
                
                
    pdf_plots.close()

    Cube_ready = np.ma.array(data= Cube, mask= storage_H['flux'].copy().mask)               
    storage_new = storage_H.copy()   
    storage_new['flux'] = Cube_ready
    storage_new['BLR_map'] = BLR_map
        
        
    prhdr = storage_H['header']
    hdu = fits.PrimaryHDU(Cube_ready.data, header=prhdr)
    hdulist = fits.HDUList([hdu])
    
    hdulist.writeto(PATH+'Four_Quasars/Sub_QSO/'+ID+'.fits', overwrite=True)
    
    return storage_new


'''