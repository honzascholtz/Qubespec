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
    expo= -((x-mu)**2)/(sig*sig)
    
    y= k* e**expo
    
    return y

def create_circular_mask(h, w, center=None, radius1 = 1, radius2 = 2):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
   
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = ( dist_from_center <= radius2) & ( dist_from_center >= radius1)
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
        if len(sol)==4:
            fwhm = sol[3]/3e5*center
            
            model = gauss(wave, sol[2], center, fwhm/2.35)
        elif len(sol)==7:
            fwhm = sol[4]/3e5*center
            
            model = emfit.OIII_outflow(wave,  *sol)-sol[1]
            
        
    elif mode =='Hn':
        center = Hal*(1+sol[0])/1e4
        if len(sol)==5:
            fwhm = sol[4]/3e5*center
            model = gauss(wave, sol[2], center, fwhm/2.35)
        elif len(sol)==8:
            fwhm = sol[4]/3e5*center
            model = gauss(wave, sol[2], center, fwhm/2.35)
    
    elif mode =='Hblr':
        center = Hal*(1+sol[0])/1e4
        
        fwhm = sol[6]/3e5*center
        model = gauss(wave, sol[3], center, fwhm/2.35)
            
    elif mode =='NII':
        center = NII_r*(1+sol[0])/1e4
        if len(sol)==5:
            fwhm = sol[4]/3e5*center
            model = gauss(wave, sol[3], center, fwhm/2.35)
        elif len(sol)==8:
            fwhm = sol[4]/3e5*center
            model = gauss(wave, sol[4], center, fwhm/2.35)
            
      
    else:
        raise Exception('Sorry mode in in SNR_calc not understood')
    
    use = np.where((wave< center+fwhm)&(wave> center-fwhm))[0]   
    flux_l = model[use]
    n = len(use)
    
    SNR = (sum(flux_l)/np.sqrt(n)) * (1./std)
    if SNR < 0:
        SNR=0
    
    return SNR  
    
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
        
        D1_s = np.ma.sum(np.ma.array(data=self.flux.data, mask= mask_spax) , axis=(1,2))
        
        wave= self.obs_wave
        
        if plot==1:
            plt.figure()
            plt.title('Collapsed 1D spectrum from D1_spectra_collapse fce')
            
            plt.plot(wave, D1_s, drawstyle='steps-mid', color='grey')
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
            arc = np.round(1.3/(header['CD2_2']*3600))
        
        except:
            arc = np.round(1.3/(header['CDELT2']*3600))
            
        
        for ix in range(shapes[0]):
            for iy in range(shapes[1]):
                dist = np.sqrt((ix- center[1])**2+ (iy- center[0])**2)
                if dist< arc:
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
        ms = flux.mask
        
        #SII_ms = ms.copy()
        #SII_ms[:] = False
        #SII_ms[np.where(((wave*1e4/(1+z))<6741)&((wave*1e4/(1+z))> 6712))[0]] = True
        
        #msk = np.logical_or(SII_ms,ms)  
        msk = ms
        
        flux = np.ma.array(data=fl, mask = msk)
        
        fit_loc = np.where(((wave*1e4/(1+z))>6562.8-100)&((wave*1e4/(1+z))<6562.8+100))[0]
        
        flat_samples, fitted_model = emfit.fitting_Halpha(wave,flux,error,z, BLR=broad)
        
        self.D1_fit_chain = flat_samples
        self.D1_fit_model = fitted_model
        
        
    
        
    def fitting_collapse_OIII(self,  plot, outflow=0):
        wave = self.obs_wave.copy()
        flux = self.D1_spectrum.copy()
        error = self.D1_spectrum_er.copy()
        z = self.z
        
        flat_samples, fitted_model = emfit.fitting_OIII(wave,flux,error,z, outflow=outflow)
        
        prop = prop_calc(flat_samples)
        
        self.D1_fit_results = prop
        self.D1_fit_chain = flat_samples
        self.D1_fit_model = fitted_model
        
        print(SNR_calc(wave, flux, error[0], prop['popt'], 'OIII'))
        
        popt = prop['popt']
        
        f, ax1 = plt.subplots(1)
        
        emplot.plotting_OIII(wave, flux, ax1, prop,fitted_model)
        

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
        
    def Spaxel_fit_OIII(self, plot=0, sp_binning='Nearest', instrument='KMOS', add=''):
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
        Unwrapped_cube = []        
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                
                j=j+step
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
                
        
        
        '''
        
        #############################
        # Binning Spaxel Fitting
        #############################
        if sp_binning== 'Nearest':
            header = storage['header']
            
            try:
                arc = (header['CD2_2']*3600)
            
            except:
                arc = (header['CDELT2']*3600)
            
            Line_info[:,:,:] = np.nan
            #Line_info = fits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_Nearest_spaxel_fit.fits')
            
            if arc> 0.17:            
                upper_lim = 2            
                step = 1
                
                if localised==1:
                    popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                    
                    x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                    x =np.array(x, dtype=int)
                    y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                    y =np.array(y, dtype=int)
                
                else:
                    x = range(shapes[0]-upper_lim)
                    y = range(shapes[1]-upper_lim)               
                
            elif arc< 0.17:            
                upper_lim = 3 
                step = 2
                
                if localised==1:
                    popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                    
                    x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                    x =np.array(x, dtype=int)
                    y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                    y =np.array(y, dtype=int)
                
                                   
                else:
                    x = range(shapes[0]-upper_lim)
                    y = range(shapes[1]-upper_lim)
            
            
            #x = np.array([20])
            #y = np.array([20])
            for i in x: #progressbar.progressbar(x):
                i= i+step
                print (i,'/',len(x))
                for j in y:
                    
                    j=j+step
                    #print i,j
                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                    
                    #Spax_mask_pick= np.logical_or(Spax_mask_pick, ThD_mask)
                    flx_spax_t = np.ma.array(data=flux,mask=Spax_mask_pick)
                    
                    flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                    flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)                
                    error = STD_calc(wv_obs/(1+z)*1e4,flx_spax, mode)* np.ones(len(flx_spax))
              
                    SNR, Line_info,out, suc = Spaxel_fit_wrap_sig(storage, Line_info, wv_obs, flx_spax_m, error, mode,i,j, broad )
                    
                    if out !=1:
                        try:
                            
                            out = out[0]
                        except:
                            out=out
                    
                    
                        Residual[:,i,j] = flx_spax.data - out.eval(x=wv_obs)
                        Model[:,i,j] =  out.eval(x=wv_obs)   
                    
                    
                    
                                   
                    prhdr = storage['header']
                    hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                    hdulist = fits.HDUList([hdu])
        
                    if mode == 'OIII':  
                        if instrument =='KMOS':
                            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
                        
                        elif instrument=='Sinfoni':
                            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
                            
            
            
                    elif mode == 'H':
                        if instrument =='KMOS':
                            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
                        
                        elif instrument=='Sinfoni':
                            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
                            
                            
                    
                    if (plot==1) &(suc==1):
                        
                        if (Spax_mask[i,j]==True) :
                            ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                            
                                                  
                            ax1.plot(wv_obs, flx_spax_m.data, color='grey', drawstyle='steps-mid')                       
                            ax1.plot(wv_obs[np.invert(flx_spax_m.mask)], flx_spax_m.data[np.invert(flx_spax_m.mask)], drawstyle='steps-mid')                   
                            ax1.plot(wv_obs, out.eval(x=wv_obs), 'r--')
            
                            if mode=='H':
                                ax1.set_xlim(6400.*(1+z)/1e4, 6700.*(1+z)/1e4)
                                
                                ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Han_'], color='orange', linestyle='dashed')
                                ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Haw_'], color='blue', linestyle='dashed')
                                ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nr_'], color='green', linestyle='dashed')
                                ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nb_'], color='limegreen', linestyle='dashed')
                        
                                Hal_cm = 6562.*(1+z)/1e4
                        
                                #print (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5 
                        
                            elif mode=='OIII':
                                cen = out.params['o3r_center'].value
                                wid = out.params['o3r_fwhm'].value
                                use = np.where((wv_obs< cen+wid)&(wv_obs> cen-wid))[0]
                                
                                ax1.plot(wv_obs, out.eval(x=wv_obs), 'k--')
                                ax1.plot(wv_obs[use], out.eval_components(x= wv_obs[use])['o3r_'], 'r--')
                                ax1.set_xlim(4900.*(1+z)/1e4, 5100.*(1+z)/1e4)
                            
                            #Spax.savefig()    
                            ax1.clear()
    '''


    
            


def Spaxel_fit_wrap_sig(storage, Line_info, obs_wv, flx_spax_m, error, mode,i,j ,broad):
    obs_wave = storage['obs_wave']
    
    z = storage['z_guess']
    
    if mode=='H':
        out_list = []
        chi_list = np.array([])
        Hal_cm = 6562.8*(1+z)/1e4
        
        suc=1
        
        D_out = storage['1D_fit_Halpha_mul']
        loc = D_out.params['Han_center'].value 
            
              
        dec = D_out
            
        
        try:
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=250.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec,init_sig=370.,   wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=450.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
            
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'H',z, mul=1)
            
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
            
                    
                    
            if SNR>3:
                Line_info[0,i,j] =   flux_measure_ind(out,obs_wave, 'H', use='BPT')  #out.params['Han_height'].value
                Line_info[1,i,j] = -((loc-out.params['Han_center'].value)/Hal_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5  
                Line_info[5,i,j] =  out.params['Haw_amplitude'].value
                
        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                Line_info[5,i,j] = out.params['Haw_amplitude'].value
        
        
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR=0
            out=1
                               
    elif mode =='OIII':
        out_old = storage['1D_fit_OIII_sig']
        
        
        OIII_cm = 5006.9*(1+z)/1e4
        width = (out_old.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
        
        out_list = []
        chi_list = np.array([])
        
        try:
            suc = 1
            
            
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.2)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.5)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.66)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.4)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
                    
            D_out = storage['1D_fit_OIII_sig']
            loc = D_out.params['o3r_center'].value 
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'OIII',z)
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
                    
                    
            if SNR>3:
                Line_info[0,i,j] = flux_measure_ind(out,obs_wave, 'OIIIs', use='tot')
                Line_info[1,i,j] = -((loc- out.params['o3r_center'].value)/OIII_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
                        
                        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                   
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR = 0
            out =1 
    
    return SNR, Line_info,out,suc

'''

def Spaxel_fit_wrap_mul(storage, Line_info, obs_wv, flx_spax_m, error, Residual, mode,i,j ,broad):
    
    z = storage['z_fit']
                               
    if mode =='OIII':
        outo = storage['1D_fit_OIII_mul']
        
        
        OIII_cm = 5006.9*(1+z)/1e4
        #width = (outo.params['o3rn_fwhm'].value/OIII_cm)*2.9979e5
        
        out_list = []
        chi_list = np.array([])
        
        if 1==1:
            suc = 1
            
            out, chi2 = emfit.fitting_OIII_Hbeta_qso_mul(obs_wv,flx_spax_m, error,z, decompose=outo, chir=1)
            
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[0]
                    
            D_out = storage['1D_fit_OIII_mul']
            loc = D_out.params['o3rn_center'].value 
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'OIII',z, mul=1)
            Line_info[9,i,j] = SNR  
                    
                    
            if SNR>3:
                Line_info[0,i,j] = out.params['o3rn_center'].value
                Line_info[1,i,j] = out.params['o3rn_sigma'].value
                Line_info[2,i,j] = out.params['o3rn_amplitude'].value
                
                Line_info[3,i,j] = out.params['o3rw_center'].value
                Line_info[4,i,j] = out.params['o3rw_sigma'].value
                Line_info[5,i,j] = out.params['o3rw_amplitude'].value
                
                Line_info[6,i,j] = out.params['Hbn_center'].value
                Line_info[7,i,j] = out.params['Hbn_sigma'].value
                Line_info[8,i,j] = out.params['Hbn_amplitude'].value
                
                Line_info[8,i,j] = out.params['Hbw_amplitude'].value
                Line_info[9,i,j] = out.params['Hbw_center'].value
                Line_info[10,i,j] = out.params['Hbw_sigma'].value
                Line_info[11,i,j] = out.params['Hbw_a1'].value
                Line_info[12,i,j] = out.params['Hbw_a2'].value   

                Line_info[13,i,j] = out.params['slope'].value
                Line_info[14,i,j] = out.params['intercept'].value
                
                Line_info[15,i,j] = SNR
                
                Residual[:,i,j] = flx_spax_m.data - out.eval(x=obs_wv)
                
                                    
            else:
                Line_info[:14,i,j] = np.nan
                
                   
        else:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[:,i,j] = np.nan
            
            SNR = 0
            out =1 
    
    return SNR, Line_info,out,suc, Residual


def Spaxel_fit_mul(storage, mode, plot, sp_binning, localised = 0, broad=1, instrument='KMOS', add=''):
    flux = storage['flux'].copy()
    Mask= storage['sky_clipped_1D']
    shapes = storage['dim']
    
    ThD_mask = storage['sky_clipped'].copy()
    z = storage['z_guess']
    wv_obs = storage['obs_wave'].copy()
        
      
    ms = Mask.copy()
    Spax_mask = storage['Sky_stack_mask'][0,:,:]  
    
    if mode=='OIII':
        msk = ms  
    
    ID = storage['X-ray ID']    
    Line_info = np.zeros((16, shapes[0], shapes[1]))
    
    Residual = flux.data.copy()
    
    Residual_OIII = Residual.copy()
    
    if plot==1:
        f, (ax1) = plt.subplots(1)
        
        ax1.set_xlabel('Rest Wavelegth (ang)')
        ax1.set_ylabel('Flux')
            
        
        if sp_binning=='Nearest':
            Spax = PdfPages(PATH+'Four_Quasars/Graphs/Spax_fit/Spaxel_Nearest_'+ID+'_'+mode+add+'.pdf')
            
            
        elif sp_binning=='Individual':
            Spax = PdfPages(PATH+'Four_Quasars/Graphs/Spax_fit/Spaxel_Nearest_'+ID+'_'+mode+add+'.pdf')
            
    
    
    
    import Plotting_tools as emplot
    #############################
    # Binning Spaxel Fitting
    #############################
    if 1==1:
        header = storage['header']
        
        try:
            arc = (header['CD2_2']*3600)
        
        except:
            arc = (header['CDELT2']*3600)
        
        Line_info[:,:,:] = np.nan
        #Line_info = fits.getdata(PATH+'Four_Quasars/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit.fits')
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                y =np.array(y, dtype=int)
            
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)               
            
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                y =np.array(y, dtype=int)
                

                
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)
        
        #import progressbar
        #x = np.array([47])
        #y = np.array([50])
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i-x[0]+1-step,'/',len(x))
            for j in y:
                
                j=j+step
                if sp_binning=='Nearest':
                    
                    #print i,j
                    Spax_mask_pick = ThD_mask.copy()
                    Spax_mask_pick[:,:,:] = True
                    Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                    
                    #Spax_mask_pick= np.logical_or(Spax_mask_pick, ThD_mask)
                    flx_spax_t = np.ma.array(data=flux,mask=Spax_mask_pick)              
                    flx_spax = np.ma.median(flx_spax_t, axis=(1,2)) 
                
                elif sp_binning =='Individual':
                    flx_spax = flux[:,i,j].copy()
                    
                    
                flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)                
                error = STD_calc(wv_obs/(1+z)*1e4,flx_spax, mode)* np.ones(len(flx_spax))
                
                if broad==1:
                    broad= storage['1D_fit_OIII_mul']
          
                SNR, Line_info,out, suc, Residual = Spaxel_fit_wrap_mul(storage, Line_info, wv_obs, flx_spax_m, error, Residual , mode,i,j, broad )
                
                Residual_OIII[:,i,j] = flx_spax_m.data - (out.eval_components(x=wv_obs)['linear'] + out.eval_components(x=wv_obs)['Hbn_'] +  out.eval_components(x=wv_obs)['Hbw_']    ) 
                
                prhdr = storage['header']
                hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                hdulist = fits.HDUList([hdu])
                
                prhdrr = storage['header']
                hdur = fits.PrimaryHDU(Residual, header=prhdrr)
                hdulistr = fits.HDUList([hdur])
                
                prhdrro = storage['header']
                hduro = fits.PrimaryHDU(Residual_OIII, header=prhdrr)
                hdulistro = fits.HDUList([hduro])
    
                if mode == 'OIII':
                    hdulist.writeto(PATH+'Four_Quasars/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit.fits', overwrite=True)
                    hdulistr.writeto(PATH+'Four_Quasars/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_residual.fits', overwrite=True)
                    hdulistro.writeto(PATH+'Four_Quasars/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_residual_OIII.fits', overwrite=True)
                
                if (plot==1) &(suc==1):
                    
                    if (Spax_mask[i,j]==True) :
                        ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                                                                    
                        emplot.plotting_OIII_Hbeta(wv_obs, flx_spax_m, ax1, out, 'mul',z, title=0)
     
             
                        Spax.savefig()    
                        ax1.clear()
    
    if plot==1:
        Spax.close()
    
    prhdr = storage['header']
    hdu = fits.PrimaryHDU(Line_info, header=prhdr)
    hdulist = fits.HDUList([hdu])
    
    if mode == 'OIII':
        hdulist.writeto(PATH+'Four_Quasars/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit.fits', overwrite=True)
     
        
    return storage





       
def Upper_lim_calc(wave,flux,error,z, mode, width_test, fit_u):   
        
    if (mode =='Hb') :        
        ffit = fit_u.eval_components(x= wave)['Hbn_'] + fit_u.eval_components(x= wave)['Hbw_']
        cent = 4865.*(1+z)/1e4
        width_test = width_test/3e5* (4862.*(1+z)/1e4)/2.35 #2.35 is to convert between FWHM and sigma
        
        
    
    if (mode =='Hbs') :        
        ffit = fit_u.eval_components(x= wave)['Hb_'] + fit_u.eval_components(x= wave)['Hb_']
        cent = 4865.*(1+z)/1e4
        width_test = width_test/3e5* (4862.*(1+z)/1e4)/2.35 #2.35 is to convert between FWHM and sigma
        
        
        
    
    flux_sub  = flux- ffit
 
    N_at = 40
    peak_test = np.linspace(1.5, 10, N_at)* error.data[0]    

    for i in range(N_at):
        flux_test = flux_sub + gauss(wave, peak_test[i], cent, width_test)

        if mode=='Hb':
            New,c = emfit.fitting_Hbeta_mul(wave, flux_test, error,z, decompose= np.array([cent, 10, 0, width_test]))                                  
            SNR,chi2 = SNR_calc(flux_test, wave, New, 'Hb',z, mul=1)
            #print 'SNR is ' ,SNR
        
        if mode=='Hbs':
            New,c = emfit.fitting_Hbeta_mul(wave, flux_test, error,z, decompose= np.array([cent, 10, 0, width_test]))                                  
            SNR,chi2 = SNR_calc(flux_test, wave, New, 'Hb',z, mul=1)
            #print 'SNR is ' ,SNR
        
        if SNR>3.:
            break
    f,(ax2) = plt.subplots(1)
 
    ax2.plot(wave/(1+z)*1e4, flux_test.data, color='grey',drawstyle='steps-mid', label='Unmasked')
    
    #Substracted
    ax2.plot(wave[np.invert(flux_sub.mask)]/(1+z)*1e4, flux_sub[np.invert(flux_sub.mask)], color='blue',drawstyle='steps-mid', label='Substracted')
    
    # Added
    ax2.plot(wave[np.invert(flux_test.mask)]/(1+z)*1e4, flux_test[np.invert(flux_test.mask)], color='orange',drawstyle='steps-mid', label='Upper lim')
    
    #Original
    ax2.plot(wave[np.invert(flux.mask)]/(1+z)*1e4, flux[np.invert(flux.mask)], color='red',drawstyle='steps-mid', label='Original')
    
    ax2.set_xlim(4862-40, 4862+40)
    ax2.set_ylim(-0.1, 0.5)
    ax2.legend(loc='best')
    
    if mode =='Hb':
        t=1
        #emplot.plotting_Hbeta(wave, flux_test, ax2, New,z, 'mul', title=1)
    if mode=='Hbs':
        Flux_up = flux_measure_ind(New,wave, 'Hb' , use='BPT')
    else:
        Flux_up = flux_measure_ind(New,wave, mode , use='BPT')
        
    print ('Flux upper limit is ',Flux_up)
    
    return Flux_up
 


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   

def W80_mes(out, mode, plot):
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

        ind10 = np.where( (Int>0.1*0.992)& (Int<0.1/0.992) )[0][0]            
        ind90 = np.where( (Int>0.9*0.995)& (Int<0.9/0.995) )[0][0] 
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
            
            ind10 = find_nearest(Int, 0.1)          
               
        
        except:
            print( np.where( (Int>0.1*0.991)& (Int<0.1/0.991) )[0]  )
            plt.figure()
            plt.plot(wvs[:-1], Int)
           
            
            
        ind90 = np.where( (Int>0.9*0.995)& (Int<0.9/0.995) )[0][0] 
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






def Spaxel_fit_wrap_sig_2L(storage, Line_info, obs_wv, flx_spax_m, error, mode,i,j ,broad):
    obs_wave = storage['obs_wave']
    
    z = storage['z_guess']
    
    if mode=='H':
        out_list = []
        chi_list = np.array([])
        Hal_cm = 6562.8*(1+z)/1e4
        
        suc=1
        
        D_out = storage['1D_fit_Halpha_mul']
        loc = D_out.params['Han_center'].value 
            
              
        dec = D_out
            
        
        try:
            out,chi2 = emfit.fitting_Halpha_mul_LBQS_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=250.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_LBQS_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec,init_sig=370.,   wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_LBQS_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=450.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
            
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'H',z, mul=1)
            
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
            
                    
                    
            if SNR>3:
                Line_info[0,i,j] =   flux_measure_ind(out,obs_wave, 'H', use='BPT')  #out.params['Han_height'].value
                Line_info[1,i,j] = -((loc-out.params['Han_center'].value)/Hal_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5  
                Line_info[5,i,j] =  out.params['Haw_amplitude'].value
                
        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                Line_info[5,i,j] = out.params['Haw_amplitude'].value
        
        
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR=0
            out=1
                               
    elif mode =='OIII':
        out_old = storage['1D_fit_OIII_sig']
        
        
        OIII_cm = 5006.9*(1+z)/1e4
        width = (out_old.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
        
        out_list = []
        chi_list = np.array([])
        
        try:
            suc = 1
            
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.2)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.5)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.66)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.4)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
                    
            D_out = storage['1D_fit_OIII_sig']
            loc = D_out.params['o3r_center'].value 
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'OIII',z)
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
                    
                    
            if SNR>3:
                Line_info[0,i,j] = flux_measure_ind(out,obs_wave, 'OIIIs', use='tot')
                Line_info[1,i,j] = -((loc- out.params['o3r_center'].value)/OIII_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
                        
                        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                   
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR = 0
            out =1 
    
    return SNR, Line_info,out,suc


def Spaxel_fit_sig_2L(storage, mode, plot, sp_binning, localised = 0, broad=1, instrument='KMOS', add=''):
    flux = storage['flux'].copy()
    Mask= storage['sky_clipped_1D']
    shapes = storage['dim']
    
    ThD_mask = storage['sky_clipped'].copy()
    z = storage['z_guess']
    wv_obs = storage['obs_wave'].copy()
    
    Residual = np.zeros_like(flux).data
    Model = np.zeros_like(flux).data
        
      
    ms = Mask.copy()
    Spax_mask = storage['Sky_stack_mask'][0,:,:]
    
    if (storage['X-ray ID']=='XID_587'):
        print ('Masking special corners')
        Spax_mask[:,0] = False
        Spax_mask[:,1] = False
        
        Spax_mask[:,-1] = False
        Spax_mask[:,-2] = False
        
        Spax_mask[0,:] = False
        Spax_mask[1,:] = False
        
        Spax_mask[-1,:] = False
        Spax_mask[-2,:] = False
        
    
    if mode =='H':
        SII_ms = ms.copy()
        SII_ms[:] = False
        SII_ms[np.where((wv_obs<6741.*(1+z)/1e4)&(wv_obs> 6712*(1+z)/1e4))[0]] = True
    
        msk = np.logical_or(SII_ms, ms)    
    
    elif mode=='OIII':
        msk = ms  
    
    ID = storage['X-ray ID']    
    Line_info = np.zeros((6, shapes[0], shapes[1]))
    
    if plot==1:
        f, (ax1) = plt.subplots(1)
        
        ax1.set_xlabel('Rest Wavelegth (ang)')
        ax1.set_ylabel('Flux')
        if sp_binning=='Individual':
            
            if instrument =='KMOS':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Individual/Spaxel_'+ID+'_'+mode+add+'.pdf')
            
            elif instrument=='Sinfoni':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Individual/Spaxel_'+ID+'_'+mode+'_sin'+add+'.pdf')
            
        
        elif sp_binning=='Nearest':
            if instrument =='KMOS':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Nearest/Spaxel_Nearest_'+ID+'_'+mode+add+'.pdf')
            
            elif instrument=='Sinfoni':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Nearest/Spaxel_Nearest_'+ID+'_'+mode+'_sin'+add+'.pdf')
                
    
    
    
    import Plotting_tools as emplot
    #############################
    # Binning Spaxel Fitting
    #############################
    if sp_binning== 'Nearest':
        header = storage['header']
        
        try:
            arc = (header['CD2_2']*3600)
        
        except:
            arc = (header['CDELT2']*3600)
        
        Line_info[:,:,:] = np.nan
        #Line_info = fits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_Nearest_spaxel_fit.fits')
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                y =np.array(y, dtype=int)
            
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)               
            
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                y =np.array(y, dtype=int)
            
                               
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)
        
        
        #x = np.array([20])
        #y = np.array([20])
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                
                j=j+step
                #print i,j
                Spax_mask_pick = ThD_mask.copy()
                Spax_mask_pick[:,:,:] = True
                Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                
                #Spax_mask_pick= np.logical_or(Spax_mask_pick, ThD_mask)
                flx_spax_t = np.ma.array(data=flux,mask=Spax_mask_pick)
                
                flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)                
                error = STD_calc(wv_obs/(1+z)*1e4,flx_spax, mode)* np.ones(len(flx_spax))
          
                SNR, Line_info,out, suc = Spaxel_fit_wrap_sig_2L(storage, Line_info, wv_obs, flx_spax_m, error, mode,i,j, broad )
                
                if out !=1:
                    try:
                        
                        out = out[0]
                    except:
                        out=out
                
                
                    Residual[:,i,j] = flx_spax.data - out.eval(x=wv_obs)
                
                
                               
                prhdr = storage['header']
                hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                hdulist = fits.HDUList([hdu])
    
                if mode == 'OIII':  
                    if instrument =='KMOS':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
                    
                    elif instrument=='Sinfoni':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
                        
        
        
                elif mode == 'H':
                    if instrument =='KMOS':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'2L.fits', overwrite=True)
                    
                    elif instrument=='Sinfoni':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'2L.fits', overwrite=True)
                        
                        
                
                if (plot==1) &(suc==1):
                    
                    if (Spax_mask[i,j]==True) :
                        ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                        
                                              
                        ax1.plot(wv_obs, flx_spax_m.data, color='grey', drawstyle='steps-mid')                       
                        ax1.plot(wv_obs[np.invert(flx_spax_m.mask)], flx_spax_m.data[np.invert(flx_spax_m.mask)], drawstyle='steps-mid')                   
                        ax1.plot(wv_obs, out.eval(x=wv_obs), 'r--')
        
                        if mode=='H':
                            ax1.set_xlim(6400.*(1+z)/1e4, 6700.*(1+z)/1e4)
                            
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Han_'], color='orange', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Haw_'], color='blue', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nr_'], color='green', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nb_'], color='limegreen', linestyle='dashed')
                    
                            Hal_cm = 6562.*(1+z)/1e4
                    
                            #print (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5 
                    
                        elif mode=='OIII':
                            cen = out.params['o3r_center'].value
                            wid = out.params['o3r_fwhm'].value
                            use = np.where((wv_obs< cen+wid)&(wv_obs> cen-wid))[0]
                            
                            ax1.plot(wv_obs, out.eval(x=wv_obs), 'k--')
                            ax1.plot(wv_obs[use], out.eval_components(x= wv_obs[use])['o3r_'], 'r--')
                            ax1.set_xlim(4900.*(1+z)/1e4, 5100.*(1+z)/1e4)
                        
                        #Spax.savefig()    
                        ax1.clear()
    #############################
    # Individual Spaxel Fitting
    #############################
    elif sp_binning== 'Individual':
        
        header = storage['header']
        
        try:
            arc = (header['CD2_2']*3600)
        
        except:
            arc = (header['CDELT2']*3600)
        
        Line_info[:,:,:] = np.nan
        #Line_info = fits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_Nearest_spaxel_fit.fits')
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                y =np.array(y, dtype=int)
            
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)               
            
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                y =np.array(y, dtype=int)
            
                #x = np.linspace(35, 65, 31) 
                #x =np.array(x, dtype=int)
                #y = np.linspace(35, 65, 31) 
                #y =np.array(y, dtype=int)
                #x = np.array([45])-2
                #y = np.array([47])-2
                
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)
        
        
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                
                
                flx_spax = flux[:,i,j]
                flx_spax_m = np.ma.array(data=flx_spax, mask= msk)
                error =   STD_calc(wv_obs*(1+z)/1e4,flx_spax, mode)* np.ones(len(flx_spax))
            
                SNR, Line_info,out, suc = Spaxel_fit_wrap_sig_2L(storage, Line_info, wv_obs, flx_spax_m, error, mode,i,j, broad )
                
                prhdr = storage['header']
                hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                
                hdulist = fits.HDUList([hdu])
    
                if mode == 'OIII':            
                    hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
        
        
                elif mode == 'H':     
                    hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
        
                if (plot==1) &(suc==1): 
                    if (Spax_mask[i,j]==True) :
                        ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                        
                                              
                        ax1.plot(wv_obs, flx_spax_m.data, color='grey', drawstyle='steps-mid')                       
                        ax1.plot(wv_obs[np.invert(flx_spax_m.mask)], flx_spax_m.data[np.invert(flx_spax_m.mask)], drawstyle='steps-mid')                   
                        ax1.plot(wv_obs, out.eval(x=wv_obs), 'r--')
        
                        if mode=='H':
                            ax1.set_xlim(6400.*(1+z)/1e4, 6700.*(1+z)/1e4)
                            
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Han_'], color='orange', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Haw_'], color='blue', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nr_'], color='green', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nb_'], color='limegreen', linestyle='dashed')
                    
                            Hal_cm = 6562.*(1+z)/1e4
                    
                            #print (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5 
                    
                        elif mode=='OIII':
                            cen = out.params['o3r_center'].value
                            wid = out.params['o3r_fwhm'].value
                            use = np.where((wv_obs< cen+wid)&(wv_obs> cen-wid))[0]
                            
                            ax1.plot(wv_obs, out.eval(x=wv_obs), 'k--')
                            ax1.plot(wv_obs[use], out.eval_components(x= wv_obs[use])['o3r_'], 'r--')
                            ax1.set_xlim(4900.*(1+z)/1e4, 5100.*(1+z)/1e4)
                        
                        Spax.savefig()    
                        ax1.clear()
    if plot==1:
        Spax.close()
    
    prhdr = storage['header']
    hdu = fits.PrimaryHDU(Line_info, header=prhdr)
    hdulist = fits.HDUList([hdu])
    
    hdu_res = fits.PrimaryHDU(Residual, header=prhdr)
    hdulist_res = fits.HDUList([hdu_res])
    
    hdu_mod = fits.PrimaryHDU(Model, header=prhdr)
    hdulist_mod = fits.HDUList([hdu_mod])
    
    if mode == 'OIII':
        if instrument =='KMOS':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
            
            
                    
        elif instrument=='Sinfoni':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
            
                        
        
        
    elif mode == 'H':
        if instrument =='KMOS':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
            
                
                
        elif instrument=='Sinfoni':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
            print ('Saving Halpha Sinfoni')
        
        hdulist_res.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'_res_2L.fits', overwrite=True)
        hdulist_mod.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'_mod_2L.fits', overwrite=True)
        
    return storage




def Spaxel_fit_wrap_sig_HB(storage, Line_info, obs_wv, flx_spax_m, error, mode,i,j ,broad):
    obs_wave = storage['obs_wave']
    
    z = storage['z_guess']
    
    if mode=='H':
        out_list = []
        chi_list = np.array([])
        Hal_cm = 6562.8*(1+z)/1e4
        
        suc=1
        
        D_out = storage['1D_fit_Halpha_mul']
        loc = D_out.params['Han_center'].value 
            
              
        dec = D_out
            
        
        try:
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=250.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec,init_sig=370.,   wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out,chi2 = emfit.fitting_Halpha_mul_testing(obs_wv,flx_spax_m ,error,z,broad=broad, decompose=  dec, init_sig=450.,  wvnet=loc)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
            
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'H',z, mul=1)
            
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
            
                    
                    
            if SNR>3:
                Line_info[0,i,j] =   flux_measure_ind(out,obs_wave, 'H', use='BPT')  #out.params['Han_height'].value
                Line_info[1,i,j] = -((loc-out.params['Han_center'].value)/Hal_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5  
                Line_info[5,i,j] =  out.params['Haw_amplitude'].value
                
        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                Line_info[5,i,j] = out.params['Haw_amplitude'].value
        
        
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR=0
            out=1
                               
    elif mode =='OIII':
        out_old = storage['1D_fit_OIII_sig']
        
        
        OIII_cm = 5006.9*(1+z)/1e4
        width = (out_old.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
        
        out_list = []
        chi_list = np.array([])
        
        try:
            suc = 1
            
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.2)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*1.5)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.66)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width*0.4)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            out, chi2 = emfit.fitting_OIII_sig(obs_wv,flx_spax_m ,error,z, init_sig=width)
            out_list.append(out)
            chi_list = np.append(chi_list, chi2)
            
            
            best = np.argmin(chi_list)
            out = out_list[best]
                    
            D_out = storage['1D_fit_OIII_sig']
            loc = D_out.params['o3r_center'].value 
            
            SNR,dat = SNR_calc(flx_spax_m,obs_wv, out, 'OIII',z)
            Line_info[3,i,j] = SNR  
            Line_info[4,i,j] = dat
                    
                    
            if SNR>3:
                Line_info[0,i,j] = flux_measure_ind(out,obs_wave, 'OIIIs', use='tot')
                Line_info[1,i,j] = -((loc- out.params['o3r_center'].value)/OIII_cm)*2.9979e5
                Line_info[2,i,j] = (out.params['o3r_fwhm'].value/OIII_cm)*2.9979e5
                        
                        
            else:
                Line_info[0,i,j] = np.nan
                Line_info[1,i,j] = np.nan
                Line_info[2,i,j] = np.nan
                   
        except:
            print ('Spaxel fit fail')
            suc=0                    
            Line_info[0,i,j] = np.nan
            Line_info[1,i,j] = np.nan
            Line_info[2,i,j] = np.nan
            Line_info[3,i,j] = np.nan
            SNR = 0
            out =1 
    
    return SNR, Line_info,out,suc


def Spaxel_fit_sig_HB(storage, mode, plot, sp_binning, localised = 0, broad=1, instrument='KMOS', add=''):
    flux = storage['flux'].copy()
    Mask= storage['sky_clipped_1D']
    shapes = storage['dim']
    
    ThD_mask = storage['sky_clipped'].copy()
    z = storage['z_guess']
    wv_obs = storage['obs_wave'].copy()
    
    Residual = np.zeros_like(flux).data
    Model = np.zeros_like(flux).data
        
      
    ms = Mask.copy()
    Spax_mask = storage['Sky_stack_mask'][0,:,:]
    
    if (storage['X-ray ID']=='XID_587'):
        print ('Masking special corners')
        Spax_mask[:,0] = False
        Spax_mask[:,1] = False
        
        Spax_mask[:,-1] = False
        Spax_mask[:,-2] = False
        
        Spax_mask[0,:] = False
        Spax_mask[1,:] = False
        
        Spax_mask[-1,:] = False
        Spax_mask[-2,:] = False
        
    
    if mode =='H':
        #SII_ms = ms.copy()
        #SII_ms[:] = False
        #SII_ms[np.where((wv_obs<6741.*(1+z)/1e4)&(wv_obs> 6712*(1+z)/1e4))[0]] = True
    
        #msk = np.logical_or(SII_ms, ms)    
        msk=ms
    elif mode=='OIII':
        msk = ms  
    
    ID = storage['X-ray ID']    
    Line_info = np.zeros((6, shapes[0], shapes[1]))
    
    if plot==1:
        f, (ax1) = plt.subplots(1)
        
        ax1.set_xlabel('Rest Wavelegth (ang)')
        ax1.set_ylabel('Flux')
        if sp_binning=='Individual':
            
            if instrument =='KMOS':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Individual/Spaxel_'+ID+'_'+mode+add+'.pdf')
            
            elif instrument=='Sinfoni':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Individual/Spaxel_'+ID+'_'+mode+'_sin'+add+'.pdf')
            
        
        elif sp_binning=='Nearest':
            if instrument =='KMOS':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Nearest/Spaxel_Nearest_'+ID+'_'+mode+add+'.pdf')
            
            elif instrument=='Sinfoni':
                Spax = PdfPages(PATH+'KMOS_SIN/Graphs/Spax_fit/Nearest/Spaxel_Nearest_'+ID+'_'+mode+'_sin'+add+'.pdf')
                
    
    
    
    import Plotting_tools as emplot
    #############################
    # Binning Spaxel Fitting
    #############################
    if sp_binning== 'Nearest':
        header = storage['header']
        
        try:
            arc = (header['CD2_2']*3600)
        
        except:
            arc = (header['CDELT2']*3600)
        
        Line_info[:,:,:] = np.nan
        #Line_info = fits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_Nearest_spaxel_fit.fits')
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                y =np.array(y, dtype=int)
            
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)               
            
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                y =np.array(y, dtype=int)
            
                               
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)
        
        
        #x = np.array([20])
        #y = np.array([20])
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                
                j=j+step
                #print i,j
                Spax_mask_pick = ThD_mask.copy()
                Spax_mask_pick[:,:,:] = True
                Spax_mask_pick[:, i-step:i+upper_lim, j-step:j+upper_lim] = False
                
                #Spax_mask_pick= np.logical_or(Spax_mask_pick, ThD_mask)
                flx_spax_t = np.ma.array(data=flux,mask=Spax_mask_pick)
                
                flx_spax = np.ma.median(flx_spax_t, axis=(1,2))                
                flx_spax_m = np.ma.array(data = flx_spax.data, mask=msk)                
                error = STD_calc(wv_obs/(1+z)*1e4,flx_spax, mode)* np.ones(len(flx_spax))
          
                SNR, Line_info,out, suc = Spaxel_fit_wrap_sig_HB(storage, Line_info, wv_obs, flx_spax_m, error, mode,i,j, broad )
                
                if out !=1:
                    try:
                        
                        out = out[0]
                    except:
                        out=out
                
                
                    Residual[:,i,j] = flx_spax.data - out.eval(x=wv_obs)
                    Model[:,i,j] =  out.eval(x=wv_obs)   
                
                
                               
                prhdr = storage['header']
                hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                hdulist = fits.HDUList([hdu])
    
                if mode == 'OIII':  
                    if instrument =='KMOS':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
                    
                    elif instrument=='Sinfoni':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
                        
        
        
                elif mode == 'H':
                    if instrument =='KMOS':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'HB.fits', overwrite=True)
                    
                    elif instrument=='Sinfoni':
                        hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'HB.fits', overwrite=True)
                        
                        
                
                if (plot==1) &(suc==1):
                    
                    if (Spax_mask[i,j]==True) :
                        ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                        
                                              
                        ax1.plot(wv_obs, flx_spax_m.data, color='grey', drawstyle='steps-mid')                       
                        ax1.plot(wv_obs[np.invert(flx_spax_m.mask)], flx_spax_m.data[np.invert(flx_spax_m.mask)], drawstyle='steps-mid')                   
                        ax1.plot(wv_obs, out.eval(x=wv_obs), 'r--')
        
                        if mode=='H':
                            ax1.set_xlim(6400.*(1+z)/1e4, 6700.*(1+z)/1e4)
                            
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Han_'], color='orange', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Haw_'], color='blue', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nr_'], color='green', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nb_'], color='limegreen', linestyle='dashed')
                    
                            Hal_cm = 6562.*(1+z)/1e4
                    
                            #print (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5 
                    
                        elif mode=='OIII':
                            cen = out.params['o3r_center'].value
                            wid = out.params['o3r_fwhm'].value
                            use = np.where((wv_obs< cen+wid)&(wv_obs> cen-wid))[0]
                            
                            ax1.plot(wv_obs, out.eval(x=wv_obs), 'k--')
                            ax1.plot(wv_obs[use], out.eval_components(x= wv_obs[use])['o3r_'], 'r--')
                            ax1.set_xlim(4900.*(1+z)/1e4, 5100.*(1+z)/1e4)
                        
                        #Spax.savefig()    
                        ax1.clear()
    #############################
    # Individual Spaxel Fitting
    #############################
    elif sp_binning== 'Individual':
        
        header = storage['header']
        
        try:
            arc = (header['CD2_2']*3600)
        
        except:
            arc = (header['CDELT2']*3600)
        
        Line_info[:,:,:] = np.nan
        #Line_info = fits.getdata(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_Nearest_spaxel_fit.fits')
        
        if arc> 0.17:            
            upper_lim = 2            
            step = 1
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[0]-4, popt[0]+4, 9) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[1]-4, popt[1]+4, 9) 
                y =np.array(y, dtype=int)
            
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)               
            
        elif arc< 0.17:            
            upper_lim = 3 
            step = 2
            
            if localised==1:
                popt =  storage['Median_stack_white_Center_data'][1:3].copy()
                
                x = np.linspace(popt[1]-12, popt[1]+12, 25) 
                x =np.array(x, dtype=int)
                y = np.linspace(popt[0]-12, popt[0]+12, 25) 
                y =np.array(y, dtype=int)
            
                #x = np.linspace(35, 65, 31) 
                #x =np.array(x, dtype=int)
                #y = np.linspace(35, 65, 31) 
                #y =np.array(y, dtype=int)
                #x = np.array([45])-2
                #y = np.array([47])-2
                
            else:
                x = range(shapes[0]-upper_lim)
                y = range(shapes[1]-upper_lim)
        
        
        for i in x: #progressbar.progressbar(x):
            i= i+step
            print (i,'/',len(x))
            for j in y:
                
                
                flx_spax = flux[:,i,j]
                flx_spax_m = np.ma.array(data=flx_spax, mask= msk)
                error =   STD_calc(wv_obs*(1+z)/1e4,flx_spax, mode)* np.ones(len(flx_spax))
            
                SNR, Line_info,out, suc = Spaxel_fit_wrap_sig_2L(storage, Line_info, wv_obs, flx_spax_m, error, mode,i,j, broad )
                
                prhdr = storage['header']
                hdu = fits.PrimaryHDU(Line_info, header=prhdr)
                
                hdulist = fits.HDUList([hdu])
    
                if mode == 'OIII':            
                    hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
        
        
                elif mode == 'H':     
                    hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
        
                if (plot==1) &(suc==1): 
                    if (Spax_mask[i,j]==True) :
                        ax1.set_title('Spaxel '+str(i)+', '+str(j)+' in Obj with SNR = "%.3f' % SNR )
                        
                                              
                        ax1.plot(wv_obs, flx_spax_m.data, color='grey', drawstyle='steps-mid')                       
                        ax1.plot(wv_obs[np.invert(flx_spax_m.mask)], flx_spax_m.data[np.invert(flx_spax_m.mask)], drawstyle='steps-mid')                   
                        ax1.plot(wv_obs, out.eval(x=wv_obs), 'r--')
        
                        if mode=='H':
                            ax1.set_xlim(6400.*(1+z)/1e4, 6700.*(1+z)/1e4)
                            
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Han_'], color='orange', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Haw_'], color='blue', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nr_'], color='green', linestyle='dashed')
                            ax1.plot(wv_obs, out.eval_components(x=wv_obs)['Nb_'], color='limegreen', linestyle='dashed')
                    
                            Hal_cm = 6562.*(1+z)/1e4
                    
                            #print (out.params['Han_fwhm'].value/Hal_cm)*2.9979e5 
                    
                        elif mode=='OIII':
                            cen = out.params['o3r_center'].value
                            wid = out.params['o3r_fwhm'].value
                            use = np.where((wv_obs< cen+wid)&(wv_obs> cen-wid))[0]
                            
                            ax1.plot(wv_obs, out.eval(x=wv_obs), 'k--')
                            ax1.plot(wv_obs[use], out.eval_components(x= wv_obs[use])['o3r_'], 'r--')
                            ax1.set_xlim(4900.*(1+z)/1e4, 5100.*(1+z)/1e4)
                        
                        Spax.savefig()    
                        ax1.clear()
    if plot==1:
        Spax.close()
    
    prhdr = storage['header']
    hdu = fits.PrimaryHDU(Line_info, header=prhdr)
    hdulist = fits.HDUList([hdu])
    
    hdu_res = fits.PrimaryHDU(Residual, header=prhdr)
    hdulist_res = fits.HDUList([hdu_res])
    
    hdu_mod = fits.PrimaryHDU(Model, header=prhdr)
    hdulist_mod = fits.HDUList([hdu_mod])
    
    if mode == 'OIII':
        if instrument =='KMOS':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'.fits', overwrite=True)
            
            
                    
        elif instrument=='Sinfoni':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/OIII/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
            
                        
        
        
    elif mode == 'H':
        if instrument =='KMOS':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'_HB.fits', overwrite=True)
            
                
                
        elif instrument=='Sinfoni':
            hdulist.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit_sin'+add+'.fits', overwrite=True)
            print ('Saving Halpha Sinfoni')
        
        hdulist_res.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'_res_HB.fits', overwrite=True)
        hdulist_mod.writeto(PATH+'KMOS_SIN/Results_storage/Halpha/'+ID+'_'+sp_binning+'_spaxel_fit'+add+'_mod_HB.fits', overwrite=True)
        
    return storage
'''