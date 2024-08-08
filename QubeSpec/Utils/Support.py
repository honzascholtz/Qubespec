"""
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


OIIIr = 5008.24
OIIIb = 4960.3
Hal = 6564.52
NII_r = 6585.27
NII_b = 6549.86
Hbe = 4862.6

SII_r = 6731
SII_b = 6718.29

from astropy.modeling.powerlaws import PowerLaw1D

def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)

    y= k* e**expo

    return y

def find_nearest(array, value):
    """
    Find the index of the element in an array that is closest to a given value.

    Parameters:
        array (array-like): The input array.
        value (float or int): The value to find the closest element to.

    Returns:
        int: The index of the element in the array that is closest to the given value.

    Example:
        >>> array = [1, 2, 3, 4, 5]
        >>> value = 3.7
        >>> find_nearest(array, value)
        3
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_circular_mask(h, w, center=None, radius=None):
    """ Creates a circular mask input - size of the array (height, width), optional center of the circular
    aperture and the radius in pixels
	
	"""

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return mask

def error_calc(array):
    """ calculates, 50th, 16th and 84 percintile of an array
	
	"""
    p50,p16,p84 = np.percentile(array, (50,16,84))
    p16 = p50-p16
    p84 = p84-p50
    
    return p50, p16,p84

def conf(aray):
    """ Legacy - old version of finding 16th and 84th percintile
	
	"""
    
    sorted_array= np.array(sorted(aray))
    leng= (float(len(aray))/100)*16
    leng= int(leng)
    
    
    hgh = sorted_array[-leng]
    low = sorted_array[leng]
    
    return low, hgh

def twoD_Gaussian(dm, amplitude, xo, yo, sigma_x, sigma_y, theta, offset): 
    """ 2D Gaussian array used to find center of emission 
	
	"""
    x, y = dm
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def smooth(image,sm):
    """ Gaussian 2D smoothning for maps
	
	"""
    from astropy.convolution import Gaussian2DKernel
    from scipy.signal import convolve as scipy_convolve
    from astropy.convolution import convolve
    
    gauss_kernel = Gaussian2DKernel(sm)

    con_im = convolve(image, gauss_kernel)
    
    #con_im = con_im#*image/image
    
    return con_im  

def prop_calc(results): 
    """ Legacy - Now part of Fitting class.  Take the dictionary with the results chains and calculates the values 
    and 1 sigma confidence interval
	
	"""
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
        

def SNR_calc(wave,flux, error, dictsol, mode, wv_cent=5008, peak_name='', fwhm_name=''):
    """ Calculates the SNR of a line
    wave - observed wavelength
    flux - flux of the spectrum
    error - error on the spectrum
    dictsol - spectral fitting results in dictionary form 
    mode - which emission line do you want to caluclate the SNR for: OIII, Hn, Hblr, NII,
           Hb, SII
	
	"""
    sol = dictsol['popt']
    wave = wave[np.invert(flux.mask)]
    flux = flux.data[np.invert(flux.mask)]
    keys = list(dictsol.keys())
    
    if mode=='general':
        center = wv_cent*(1+dictsol['z'][0])/1e4
        fwhm = dictsol[fwhm_name][0]
        contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
        model = gauss(wave, dictsol[peak_name][0],center, fwhm)
    elif mode =='OIII':
        center = OIIIr*(1+sol[0])/1e4
        if 'outflow_fwhm' in keys:
            fwhms = sol[5]/3e5*center
            fwhm = dictsol['outflow_fwhm'][0]
            
            center = OIIIr*(1+sol[0])/1e4
            centerw = OIIIr*(1+sol[0])/1e4 + sol[7]/3e5*center
            
            contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
            model = flux-contfce 
        elif 'Nar_fwhm' in keys:
            fwhm = dictsol['Nar_fwhm'][0]
            
            center = OIIIr*(1+sol[0])/1e4
            
            contfce = PowerLaw1D.evaluate(wave, sol[1],center, alpha=sol[2])
            model = flux-contfce 
        else:   
            fwhm = sol[4]/3e5*center
            model = flux- PowerLaw1D.evaluate(wave,sol[1],center, alpha=sol[2])        
            
    elif mode =='Hn':
        center = Hal*(1+sol[0])/1e4

        fwhm = dictsol['Nar_fwhm'][0]
        model = gauss(wave, dictsol['Hal_peak'][0], center, fwhm)
    
    elif mode =='Hblr':
        center = Hal*(1+sol[0])/1e4
        
        fwhm = sol[7]/3e5*center
        model = gauss(wave, sol[4], center, fwhm/2.35)
            
    elif mode =='NII':
        center = NII_r*(1+sol[0])/1e4
        fwhm = dictsol['Nar_fwhm'][0]
        model = gauss(wave, dictsol['NII_peak'][0], center, fwhm)
    
    elif mode =='Hb':
        center = Hbe*(1+sol[0])/1e4
        if 'Hbetan_fwhm' in keys:
            fwhm = dictsol['Hbetan_fwhm'][0]
            model = gauss(wave, dictsol['Hbetan_peak'][0], center, fwhm)
        elif 'Nar_fwhm' in keys:
            fwhm = dictsol['Nar_fwhm'][0]
            model = gauss(wave, dictsol['Hbeta_peak'][0], center, fwhm)
        else:
            fwhm = dictsol['Hbeta_fwhm'][0]
            model = gauss(wave, dictsol['Hbeta_peak'][0], center, fwhm)
    
    elif mode =='SII':
        center = SII_r*(1+sol[0])/1e4
        
        fwhm = dictsol['Nar_fwhm'][0]
        try:
            model_r = gauss(wave, dictsol['SIIr_peak'][0], center, fwhm) 
            model_b = gauss(wave, dictsol['SIIb_peak'][0], center, fwhm) 
        except:
            model_r = gauss(wave, dictsol['SIIr_peak'][0], center, fwhm) 
            model_b = gauss(wave, dictsol['SIIb_peak'][0], center, fwhm) 
        
        model = model_r + model_b
        
        center = 6724*(1+sol[0])/1e4
        fwhm = fwhm/3e5*center
        use = np.where((wave< center+fwhm*1)&(wave> center-fwhm*1))[0]   
        flux_l = model[use]
        std = error[use]
        
        n = len(use)
        SNR = np.nansum(flux_l)/np.sqrt(np.nansum(std**2))
        
        if SNR < 0:
            SNR=0
        
        return SNR
    
    else:
        raise Exception('Sorry mode in SNR_calc not understood')
    fwhm = fwhm/3e5*center
    use = np.where((wave< center+fwhm*1)&(wave> center-fwhm*1))[0] 
    flux_l = model[use]
    std = error[use]
    
    n = len(use)
    SNR =np.nansum(flux_l)/np.sqrt(np.nansum(std**2))
    if SNR < 0:
        SNR=0
    
    return SNR  

def BIC_calc(wave,fluxm,error, model, results, mode, template=0):
    """ calculates BIC
	
	"""
    popt = results['popt']
    z= popt[0]
    
    if mode=='OIII':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>4800*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        if template==0:
            y_model = model(wave, *popt)
        else:
            y_model = model(wave, *popt, template)
        chi2 = np.nansum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    if mode=='Halpha':
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>(6564.52-200)*(1+z)/1e4)&(wave<(6564.52+300)*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = np.nansum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
    
    if mode=='Halpha_OIII':
        fit_loc = np.where((wave>4700*(1+z)/1e4)&(wave<5100*(1+z)/1e4))[0]
        fit_loc = np.append(fit_loc, np.where((wave>(6300-50)*(1+z)/1e4)&(wave<(6300+50)*(1+z)/1e4))[0])
        fit_loc = np.append(fit_loc, np.where((wave>(6564.52-170)*(1+z)/1e4)&(wave<(6564.52+170)*(1+z)/1e4))[0])
        
        flux = fluxm.data[np.invert(fluxm.mask)]
        wave = wave[np.invert(fluxm.mask)]
        error = error[np.invert(fluxm.mask)]
        
        fit_loc = np.where((wave>(6564.52-200)*(1+z)/1e4)&(wave<(6564.52+300)*(1+z)/1e4))[0]
        
        flux = flux[fit_loc]
        wave = wave[fit_loc]
        error = error[fit_loc]
        
        y_model = model(wave, *popt)
        chi2 = np.nansum(((flux-y_model)/error)**2)
        BIC = chi2+ len(popt)*np.log(len(flux))
       
    return chi2, BIC

def unwrap_chain(res):
    """
    Unwraps a dictionary of chains into a 2D numpy array.

    Parameters:
    - res (dict): A dictionary containing chains as values, where each chain is a 1D numpy array.

    Returns:
    - chains (ndarray): A 2D numpy array where each column represents a chain from the input dictionary.

    Example:
    >>> res = {'chain1': np.array([1, 2, 3]), 'chain2': np.array([4, 5, 6])}
    >>> unwrap_chain(res)
    array([[1., 4.],
        [2., 5.],
        [3., 6.]])
    """
    keys = list(res.keys())[1:]  
    chains = np.zeros(( len(res[keys[0]]), len(keys) ))  
    for i in range(len(keys)):  
        chains[:,i] = res[keys[i]]    
    return chains

def QFitsview_mask(filepath):
    mask_load = pyfits.getdata(filepath)
    mask = ~mask_load
    mask[mask==-1] = 1
    mask[mask==-2] = 0
    return mask

def flux_calc_general(wv_cent, res, fwhm_name, peak_name):
    """
    Calculate the flux using the general formula.

    Parameters:
        wv_cent (float): The central wavelength.
        res (dict): A dictionary containing the necessary parameters.
        fwhm_name (str): The name of the key in the 'res' dictionary that corresponds to the full width at half maximum (FWHM).
        peak_name (str): The name of the key in the 'res' dictionary that corresponds to the peak value.

    Returns:
        float: The calculated flux.

    """
    mu = wv_cent*(1+res['z'][0])/1e4
    if type(fwhm_name)==str:
        FWHM = res[fwhm_name][0]
    else: 
        FWHM = fwhm_name
    a = 1./(2*(FWHM/3e5*mu/2.35482)**2)
    return res[peak_name][0]*np.sqrt(np.pi/a)


def flux_calc(res, mode, norm=1e-13, wv_cent=5008, peak_name='', fwhm_name='', ratio_name=''):
    """
    Calculate the flux for different emission lines based on the given parameters.

    Parameters:
        res (dict): A dictionary containing the necessary parameters.
        mode (str): The mode of calculation. Possible values are:
            - 'general': Calculate the flux using the general formula.
            - 'OIIIt': Calculate the flux for OIII emission line with outflow component.
            - 'OIIIn': Calculate the flux for OIII emission line without outflow component.
            - 'OIIIw': Calculate the flux for the outflow component of OIII emission line.
            - 'Hat': Calculate the flux for Halpha emission line with outflow component.
            - 'Han': Calculate the flux for Halpha emission line without outflow component.
            - 'Hal_BLR': Calculate the flux for the broad line region (BLR) component of Halpha emission line.
            - 'NIIt': Calculate the flux for NII emission line with outflow component.
            - 'NII': Calculate the flux for NII emission line without outflow component.
            - 'NIIo': Calculate the flux for the outflow component of NII emission line.
            - 'Hbeta': Calculate the flux for Hbeta emission line.
            - 'Hbe_BLR': Calculate the flux for the broad line region (BLR) component of Hbeta emission line.
            - 'Hbetaw': Calculate the flux for the wing component of Hbeta emission line.
            - 'Hbetan': Calculate the flux for the narrow component of Hbeta emission line.
            - 'SIIr': Calculate the flux for SII red emission line.
            - 'SIIb': Calculate the flux for SII blue emission line.
        norm (float, optional): The normalization factor for the calculated flux. Default is 1e-13.
        wv_cent (float, optional): The central wavelength for the calculation. Default is 5008.
        peak_name (str, optional): The name of the key in the 'res' dictionary that corresponds to the peak value. Required for 'general', 'OIIIt', 'OIIIn', 'OIIIw', 'Hat', 'Han', 'Hal_BLR', 'NIIt', 'NII', 'NIIo', 'Hbeta', 'Hbe_BLR', 'Hbetaw', 'Hbetan' modes.
        fwhm_name (str, optional): The name of the key in the 'res' dictionary that corresponds to the full width at half maximum (FWHM). Required for 'general', 'OIIIt', 'OIIIn', 'OIIIw', 'Hat', 'Han', 'Hal_BLR', 'NIIt', 'NII', 'NIIo', 'Hbeta', 'Hbe_BLR', 'Hbetaw', 'Hbetan' modes.
        ratio_name (str, optional): The name of the key in the 'res' dictionary that corresponds to the ratio value. Required for 'general' mode.

    Returns:
        float: The calculated flux.

    Raises:
        Exception: If the mode is not understood.

    """
    keys = list(res.keys())
    
    if mode=='general':
        if ratio_name=='':
            ratio=1
        else:
            ratio=res[ratio_name][0]
        flx =  ratio*flux_calc_general(wv_cent, res, fwhm_name, peak_name)
        return flx*norm
    
    elif mode=='OIIIt':
        flx =  flux_calc_general(OIIIr, res, 'Nar_fwhm', 'OIII_peak')
        if 'OIII_out_peak' in keys:  
            flx +=  flux_calc_general(OIIIr, res, 'outflow_fwhm', 'OIII_out_peak')      
        return flx*norm
            
            
    elif mode=='OIIIn':
        flx =  flux_calc_general(OIIIr, res, 'Nar_fwhm', 'OIII_peak')
        return flx*norm
        
    elif mode=='OIIIw':
        if 'OIII_out_peak' in keys:
            flx =  flux_calc_general(OIIIr, res, 'outflow_fwhm', 'OIII_out_peak')  
            return flx*norm
        else:
            return 0
    
    elif mode=='Hat':
        flx =  flux_calc_general(Hal, res, 'Nar_fwhm', 'Hal_peak')
        if 'outflow_fwhm' in list(res.keys()):
            flx +=  flux_calc_general(Hal, res, 'outflow_fwhm', 'Hal_out_peak')
        return flx*norm
    
    elif mode=='Han':
        flx = flux_calc_general(Hal, res, 'Nar_fwhm', 'Hal_peak')
        return flx*norm

    elif mode=='Haw':
        flx = flux_calc_general(Hal, res, 'Nar_fwhm', 'Hal_out_peak')
        return flx*norm
        
    elif mode=='Hal_BLR':
        if 'BLR_fwhm' in keys:
            flx = flux_calc_general(Hal, res, 'BLR_fwhm', 'BLR_Hal_peak')
            return flx*norm
            
        elif 'BLR_alp1' in keys:
            from ..Models.QSO_models import BKPLG
            wave = np.linspace(6300,6700,700)*(1+res['z'][0])/1e4
            model = BKPLG(wave, res['BLR_peak'][0], Hal*(1+res['z'][0])/1e4, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
            
        else:
            return 0
    
    elif mode=='NIIt':
        flx = flux_calc_general(NII_r, res, 'Nar_fwhm', 'NII_peak')
        
        if 'outflow_fwhm' in list(res.keys()):
            flx +=  flux_calc_general(NII_r, res, 'outflow_fwhm', 'NII_out_peak')
        return flx*norm
    
    elif mode=='NII':
        flx = flux_calc_general(NII_r, res, 'Nar_fwhm', 'NII_peak')
        return flx*norm
              
    elif mode=='NIIo':
        flx = flux_calc_general(NII_r, res, 'outflow_fwhm', 'NII_out_peak')
        return flx*norm
        
    elif mode=='Hbeta':     
        try:
            flx = flux_calc_general(Hbe, res, 'Nar_fwhm', 'Hbeta_peak')
        except:
            flx = flux_calc_general(Hbe, res, 'Hbeta_fwhm', 'Hbeta_peak')
        return flx*norm
    
    elif mode=='Hbe_BLR':
        if 'BLR_fwhm' in keys:
            flx = flux_calc_general(Hbe, res, 'BLR_fwhm', 'BLR_Hbeta_peak')
            return flx*norm
        
        elif 'BLR_alp1' in keys:
            wave = np.linspace(4800,4900,700)*(1+res['z'][0])/1e4
            from ..Models.QSO_models import BKPLG
            model = BKPLG(wave, res['BLR_peak'][0], Hbe, res['BLR_sig'][0], res['BLR_alp1'][0], res['BLR_alp2'][0])
        else:
            return 0 
        
    elif mode=='Hbetaw':
        flx = flux_calc_general(Hbe, res, 'Hbeta_fwhm', 'Hbeta_peak')
        return flx*norm
    elif mode=='Hbetan':
        flx = flux_calc_general(Hbe, res, 'Hbetan_fwhm', 'Hbetan_peak')
        return flx*norm  
    
    elif mode=='SIIr':
        flx = flux_calc_general(6732, res, 'Nar_fwhm', 'SIIr_peak')
        return flx*norm
        
    elif mode=='SIIb':
        flx = flux_calc_general(6718, res, 'Nar_fwhm', 'SIIb_peak') 
        return flx*norm
    
    else:
        raise Exception('Sorry mode in Flux calc not understood')
        
    import scipy.integrate as scpi
    
    Flux = scpi.simps(model, wave)*norm
        
    return Flux

import random
def flux_calc_mcmc(fit_obj, mode, norm=1, N=2000, wv_cent=5008, peak_name='', fwhm_name='', ratio_name=''):
    """
    Calculates flux and 68% confidence iterval. 

    Parameters
    ----------

        fit_obj - object
            Fitting class object
        
        mode - string
            modes: general, OIIIn, OIIIw, OIIIt, Han, NII, Hbeta, SIIr, SIIb
        
        norm - value
            normalization used in the QubeSpec cube class 

        N - int
            number of sampling of the chains
        
        wv_cent - float
            rest-frame wavelength in ang of the line if mode='general'
        
        peak_name - string
          if mode='general' name of the peak name to use

        fwhm_name - string
            if mode='general' name of the fwhm name to use

        ratio_name - string
            if mode='general' name of the ratio to use (e.g. in [OII])

    Returns
    -------

    array of median value and +- 1sigma
    """
    chains = fit_obj.chains
    res = fit_obj.props
    labels = list(chains.keys())

    popt = np.zeros_like(res['popt'])
    Fluxes = []
    res_new = {'name': res['name']}
    
    Nchain = len(chains['z'])
    itere = np.arange(Nchain/2,Nchain,1, dtype=int)
        
    for j in itere:
        #sel = random.randint(Nchain/2,N-1)
        for i in range(len(popt)): 
            
            popt[i] = chains[labels[i+1]][j]
            res_new[labels[i+1]] = [popt[i], 0,0 ]
        
        res_new['popt'] = popt
        Fluxes.append(flux_calc(res_new, mode,norm, wv_cent=wv_cent, peak_name=peak_name, fwhm_name=fwhm_name, ratio_name=ratio_name))
    
    p50,p16,p84 = np.percentile(Fluxes, (50,16,84))
    p16 = p50-p16
    p84 = p84-p50
    return p50, p16, p84


def vel_kin_percentiles(self, peak_names, fwhm_names, vel_names,rest_wave,vel_percentiles=[], z=0, error_range=[50,16,84], N=100):
    """_summary_

    :param peak_names: _description_
    :param fwhm_names: _description_
    :param vel_names: _description_
    :param rest_wave: _description_
    :param vel_percentiles: _description_, defaults to []
    :param z: _description_, defaults to 0
    :param error_range: _description_, defaults to [50,16,84]
    :param N: _description_, defaults to 100
    
    :return: _description_
    """
    if z==0:
        z = self.props['popt'][0]
    
    import scipy.integrate as scpi
    import scipy.interpolate as intp
    
    cent =  rest_wave*(1+z)/1e4
    bound1 =  cent + 3000/3e5*cent
    bound2 =  cent - 3000/3e5*cent
    Ni = 6000
    wvs = np.linspace(bound2, bound1, Ni)
    vels = np.linspace(-3000,3000, Ni)

    velnames = ['z']
    for j in vel_names:
        velnames.append(j)

    Nchain = len(self.chains['z'])
    itere = np.arange(Nchain/2,Nchain,1, dtype=int)
    itere = np.random.choice(itere, size=N, replace=False)

    vel_percentiles = np.append( np.array([10,90,50]), vel_percentiles)

    vel_res = np.zeros((len(vel_percentiles), N))
    peak_vel = np.zeros(N)
    if N==1:
        y = np.zeros_like(wvs)
        for i, name in enumerate(vel_names):
            
            if name =='z':
                wv_cent = rest_wave*(1+self.props['z'][0])/1e4
            else:
                wv_cent = rest_wave*(1+self.props['z'][0])/1e4
                wv_cent = wv_cent + self.props[vel_names[i]][0]/3e5*wv_cent

            y += gauss(wvs, self.props[peak_names[i]][0], wv_cent, self.props[fwhm_names[i]][0])
    
        vel_peak = vels[np.argmax(y)]
        flux_cum = np.cumsum(y)
        flux_cum = flux_cum/np.max(flux_cum)
        for k, value in enumerate(vel_percentiles):
            vel_res[k] = vels[find_nearest(flux_cum, value/100)]

        res = {}
        res['vel_peak'] = vel_peak
        res['w80'] = vel_res[1]-vel_res[0]

        for i in range(len(vel_percentiles)):
            res['v'+str(int(vel_percentiles[i]))] = vel_res[i]

        return res
        
    else:
        for ind, itr in enumerate(itere):
            y = np.zeros_like(wvs)
            for i, name in enumerate(velnames):
                if name =='z':
                    wv_cent = rest_wave*(1+self.chains['z'][itr])/1e4
                else:
                    wv_cent = rest_wave*(1+self.chains['z'][itr])/1e4
                    wv_cent = wv_cent + self.chains[velnames[i]][itr]/3e5*wv_cent

                y += gauss(wvs, self.chains[peak_names[i]][itr], wv_cent, self.chains[fwhm_names[i]][itr])
            peak_vel[ind] = vels[np.argmax(y)]

            #if ind==0:

            #    plt.figure()
            #    plt.plot(wvs, y)
            
            flux_cum = np.cumsum(y)
            flux_cum = flux_cum/np.max(flux_cum)
            for k, value in enumerate(vel_percentiles):
                vel_res[k,ind] = vels[find_nearest(flux_cum, value/100)]
        
        W80 = vel_res[1,:] - vel_res[0,:]

        res = {}
        res['vel_peak'] = np.percentile(peak_vel, error_range)
        res['w80'] = np.percentile(W80, error_range)

        for i in range(len(vel_percentiles)):
            res['v'+str(int(vel_percentiles[i]))] = np.percentile(vel_res[i,:], error_range)


        return res
        

        
def W80_OIII_calc(Fits, N=100,z=0):
    """_summary_

    :param Fits: _description_
    :param N: _description_, defaults to 100
    :return: _description_
    """
    sol = Fits.props
    
    if 'outflow_fwhm' in sol:

        if 'OIII_out_peak' in sol:

            kin_res = vel_kin_percentiles(Fits, ['OIII_peak', 'OIII_out_peak'],\
                                ['Nar_fwhm', 'outflow_fwhm'], ['outflow_vel'], \
                                    rest_wave=5008,  N=N,error_range=[50,16,84],z=z)
        
        if 'OIIIw_peak' in sol:

            kin_res = vel_kin_percentiles(Fits, ['OIII_peak', 'OIIIw_peak'],\
                                ['Nar_fwhm', 'outflow_fwhm'], ['outflow_vel'], \
                                    rest_wave=5008,  N=N,error_range=[50,16,84],z=z)
            
    else:
        kin_res = vel_kin_percentiles(Fits, ['OIII_peak'],\
                                ['Nar_fwhm'], [], \
                                    rest_wave=5008,  N=N,error_range=[50,16,84],z=z)

    vel_peak = kin_res['vel_peak']
    v10 = kin_res['v10']
    v90 = kin_res['v90']
    w80 = kin_res['w80']
    v50 = kin_res['v50']
    if N==1:        
        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}
    
    else:
        v10[1] = v10[1] - v10[0]
        v10[2] = v10[2]- v10[0]

        v50[1] = v50[1] - v50[0]
        v50[2] = v50[2]- v50[0]

        v90[1] = v90[1] - v90[0]
        v90[2] = v90[2]- v90[0]
            
        w80[1] = w80[1] - w80[0]
        w80[2] = w80[2] - w80[0]

        vel_peak[1] = vel_peak[1] - vel_peak[2]
        vel_peak[2] = vel_peak[2] - vel_peak[1]

        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}



def W80_Halpha_calc(Fits, N=100,z=0):
    """_summary_

    :param Fits: _description_
    :param N: _description_, defaults to 100
    :return: _description_
    """
    sol = Fits.props
    
    if 'outflow_fwhm' in sol:
        kin_res = vel_kin_percentiles(Fits, ['Hal_peak', 'Hal_out_peak'],\
                            ['Nar_fwhm', 'outflow_fwhm'], ['outflow_vel'], \
                                rest_wave=6564,  N=N,error_range=[50,16,84],z=z)
        
        
            
    else:
        kin_res = vel_kin_percentiles(Fits, ['Hal_peak'],\
                                ['Nar_fwhm'], [], \
                                    rest_wave=6564,  N=N,error_range=[50,16,84],z=z)

    vel_peak = kin_res['vel_peak']
    v10 = kin_res['v10']
    v90 = kin_res['v90']
    w80 = kin_res['w80']
    v50 = kin_res['v50']

    if N==1:        
        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}
    
    else:
        v10[1] = v10[1] - v10[0]
        v10[2] = v10[2]- v10[0]

        v50[1] = v50[1] - v50[0]
        v50[2] = v50[2]- v50[0]

        v90[1] = v90[1] - v90[0]
        v90[2] = v90[2]- v90[0]
            
        w80[1] = w80[1] - w80[0]
        w80[2] = w80[2] - w80[0]

        vel_peak[1] = vel_peak[1] - vel_peak[2]
        vel_peak[2] = vel_peak[2] - vel_peak[1]

        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}




def W80_NII_calc(Fits, N=100,z=0):
    """_summary_

    :param Fits: _description_
    :param N: _description_, defaults to 100
    :return: _description_
    """
    sol = Fits.props
    
    if 'outflow_fwhm' in sol:

        kin_res = vel_kin_percentiles(Fits, ['NII_peak', 'NII_out_peak'],\
                            ['Nar_fwhm', 'outflow_fwhm'], ['outflow_vel'], \
                                rest_wave=6564,  N=N,error_range=[50,16,84],z=z)
              
    else:
        kin_res = vel_kin_percentiles(Fits, ['NII_peak'],\
                                ['Nar_fwhm'], [], \
                                    rest_wave=6564,  N=N,error_range=[50,16,84],z=z)
    vel_peak = kin_res['vel_peak']
    v10 = kin_res['v10']
    v90 = kin_res['v90']
    w80 = kin_res['w80']
    v50 = kin_res['v50']

    if N==1:        
        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}
    
    else:
        v10[1] = v10[1] - v10[0]
        v10[2] = v10[2]- v10[0]

        v50[1] = v50[1] - v50[0]
        v50[2] = v50[2]- v50[0]

        v90[1] = v90[1] - v90[0]
        v90[2] = v90[2]- v90[0]
            
        w80[1] = w80[1] - w80[0]
        w80[2] = w80[2] - w80[0]

        vel_peak[1] = vel_peak[1] - vel_peak[2]
        vel_peak[2] = vel_peak[2] - vel_peak[1]

        return {'vel_peak':vel_peak,'v10':v10,'v90':v90,'w80':w80,'v50':v50}


def flux_to_lum(flux,redshift):
    from astropy.cosmology import WMAP9 as cosmo
    import astropy.units as u 
    lum = (flux*u.erg/u.s/u.cm**2) * 4*np.pi*(cosmo.luminosity_distance(redshift))**2  
    return lum.to(u.erg/u.s)

def jadify(object_name, disp_filt, wave, flux, err=None, mask=None, verbose=True,
    overwrite=True, descr=None, author=None):
    """
    object_name : int or str **do not use "_" characters
        unique identifier. Preferrably up to eight digits. String is acceptable but
        frowned upon.
    disp_filt : string
        one of 'prism_clear', 'g140m_f070lp', 'g235m_f170lp', 'g395m_f290lp'
    wave : array or any other iterable
        in [um]
    flux : array or any other iterable. Same shape as `wave`
        in [W/m^3]
    err  : array or any other iterable. Same shape as `wave`; [optional]
        in [W/m^3]
    mask : array or any other iterable. Same sahpe as `wave`; [optional]
        0 means good pixel; 1 bad pixel
    verbose : bool
        Suppress messages
    overwrite : bool
        Force overwrite existing fits file. Ask for confirmation otherwise.
    descr : str [optional]
        Any text to add to the comments. Added to the primary HDU under 'comments'
    author : str [optional]
        Only if you want to. Added to the primary HDU under 'author'
    """
    import os
    from astropy.io import fits
    if verbose:
        print('Reminder (suppress with verbose=False):\n'
            '`wave`: [um]; in observed frame\n'
            '`flux`: [W/m^3]; 1 erg/(s cm^2 AA) = 1e7 W/m^3\n'
            '`err` : [W/m^3]; optional. If not given all 0\n'
            '`mask`: [0=good, 1=bad]; optional. If not given all 0\n'
            )
    if any(wave<0.5) or any(wave>6):
        print(f'{wave} wavelength vector seems strange. Are you sure it is in um?')
    weird_wave_disp = (
        (disp_filt=='g140m_f070lp' and (any(wave<0.5) or any(wave>2))) # Not in G140M
        or (disp_filt=='g235m_f170lp' and (any(wave<1.5) or any(wave>3.5))) # Not in G235M
        or (disp_filt=='g395m_f290lp' and (any(wave<2.5) or any(wave>6)))   # Not in G395M
        )
    if weird_wave_disp:
        print(f'{wave} wavelength vector seems strange for {disp_filt}. Are you sure it the right disperser/filter combination?')
        
    # Open dummy file.
    output_filename = f'{object_name}_{disp_filt}_v3.0_1D.fits'
    from .. import jadify_temp as pth

    PATH_TO_jadify = pth.__path__[0]+ '/'
    filename = PATH_TO_jadify+ 'temp_prism_clear_v3.0_extr3_1D.fits'

    with fits.open(filename) as hdu:
        hdu['DATA'].data = flux
        hdu['ERR'].data  = (err if err is not None else np.zeros_like(flux))
        hdu['DIRTY_Data'].data = flux
        hdu['DIRTY_QUALITY'].data = (mask if mask is not None else np.zeros(flux.size, dtype=int))
        hdu['WAVELENGTH'].data = wave/1e6
        hdu['GTO_FLAG'].data = np.zeros_like(flux)
        hdu['GTO_OVERLAPPING'].data = np.zeros_like(flux)

        if descr is not None:
            hdu[0].header['COMMENT'] = str(descr)

        if os.path.isfile(output_filename) and (overwrite is False):
            proceed = input(f'{output_filename} exists and {overwrite=}. Enter y to overwrite\n')
            if proceed is not 'y': 
                print('Aborted by user')
                return
         
        # Never reach this if overwrite=False and file exists
        hdu.writeto(output_filename, overwrite=True)


def NIRSpec_IFU_PSF(wave):
    """
    Calculate the Point Spread Function (PSF) for the NIRSpec IFU.

    Parameters:
        wave (float): The wavelength of the light.

    Returns:
        numpy.ndarray: An array containing the two components of the PSF, sigma1 and sigma2.

    Notes:
    This function calculates the PSF based on the formula from D'Eugenio et al 2023 - stellar kinematics.
    The PSF is calculated using two components, sigma1 and sigma2, which depend on the wavelength of the light.
    The formula for sigma1 is 0.12 + 1.9 * wave * e^(-24.4/wave).
    The formula for sigma2 is 0.09 + 0.2 * wave * e^(-12.5/wave).
    The function returns an array containing the values of sigma1 and sigma2.
    """
    # From D'Eugenio et al 2023 - stellar kinematics
    sigma1= 0.12 + 1.9*wave * e**(-24.4/wave)
    sigma2= 0.09 + 0.2*wave * e**(-12.5/wave)                     
    return np.array([sigma1,sigma2])

def pickle_load(file_path):
    import pickle
    with open(file_path, "rb") as fp:
        return pickle.load(fp)

def pickle_save(file_path, stuff):
    import pickle
    with open(file_path, "wb") as fp:
        pickle.dump(stuff, fp)

def error_scaling(obs_wave,flux, error_var, err_range, boundary, exp=0):
    error= np.zeros_like(flux)
    from astropy import stats

    #if (np.any(np.array(err_range)<obs_wave[0])==True) | (np.any(np.array(err_range)>obs_wave[-1])==True):
    #    raise Exception('err range values are out of range of the wavelength range of the data')
    
    #if (boundary<obs_wave[0]) | (boundary>obs_wave[-1]):
    #    raise Exception('Err boundary value are out of range of the wavelength range of the data')


    if len(err_range)==2:
        error1 = stats.sigma_clipped_stats(flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3, mask = flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])].mask)[2]
        
        average_var1 = stats.sigma_clipped_stats(error_var[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3, mask = error[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])].mask)[1]
    
        error = error_var*(error1/average_var1)

    elif len(err_range)==4:
        error1 = stats.sigma_clipped_stats(flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3, mask = flux[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])].mask)[2]
        error2 = stats.sigma_clipped_stats(flux[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])],sigma=3, mask = flux[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])].mask)[2]
        
        average_var1 = stats.sigma_clipped_stats(error_var[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])],sigma=3, mask = error[(err_range[0]<obs_wave) \
                                                    &(obs_wave<err_range[1])].mask)[1]
        average_var2 = stats.sigma_clipped_stats(error_var[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])],sigma=3, mask = error[(err_range[2]<obs_wave) \
                                                    &(obs_wave<err_range[3])].mask)[1]
        
        error[obs_wave<boundary] = error_var[obs_wave<boundary]*(error1/average_var1)
        error[obs_wave>boundary] = error_var[obs_wave>boundary]*(error2/average_var2)
    else:
        error1 = stats.sigma_clipped_stats(flux,sigma=3)[2]
                
        average_var1 = stats.sigma_clipped_stats(flux,sigma=3)[1]
        error = error_var/(error1/average_var1)
            
    error[error==0] = np.mean(error)*10
    try:
        error[error.mask==True] = np.ma.mean(error)*10
        error = error.data
    except:
        lsk=0

    if exp==1:
        try:
            print('Error rescales are: ', error1/average_var1, error2/average_var2 )
        except:
            print('Error rescale is: ', error1/average_var1 )
    return error

def where(array, lmin, lmax):
    use = np.where( (array>lmin) & (array<lmax))
    return use

def header_to_2D(header):
    from astropy.io import fits
    hdu = fits.PrimaryHDU()
    new_header = hdu.header  # show the all of the header cards

    new_header['NAXIS'] = 2
    new_header['WCSAXES'] = 2
    list_cp = ['BITPIX', 'NAXIS1', 'NAXIS2', 'CRPIX1', 'CRVAL1', 'CTYPE1', 'CUNIT1', 'CDELT1', 'CRPIX2', 'CRVAL2', 'CTYPE2', 'CUNIT2', 'CDELT2',\
               'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',  ]
    
    for item in list_cp:
        try:
            new_header[item] = header[item]
        except:
            ils=0
    return new_header

def DS9_region_mask(filepath, header):
    from regions import Regions
    region_sky  = Regions.read(filepath, format='ds9')
    try:
        new_header = header_to_2D(header)
    except:
        new_header= header
    try:
        region_pix = region_sky[0].to_pixel(wcs.WCS(new_header))
    except:
        region_pix = region_sky.to_pixel(wcs.WCS(new_header))
    mask = region_pix.to_mask(mode='center',).to_image([new_header['NAXIS2'],new_header['NAXIS1']])
    return ~np.array(mask, dtype=bool)

def MSA_load(Full_path):
    with pyfits.open(Full_path, memmap=False) as hdulist:
        flux_orig = hdulist['DATA'].data*1e-7*1e4*1e15
        obs_wave = hdulist['wavelength'].data*1e6
        error =  hdulist['ERR'].data*1e-7*1e4*1e15

        flux = np.ma.masked_invalid(flux_orig.copy())
    
    return obs_wave, flux, error