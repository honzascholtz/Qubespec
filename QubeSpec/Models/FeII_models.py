# =============================================================================
# FeII code
# =============================================================================
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.interpolate import interp1d
from astropy.io import fits as pyfits
import numpy as np
import pickle

from . import FeII_templates as pth
PATH_TO_FeII = pth.__path__[0]+ '/'

#Loading the template
Veron_d = pyfits.getdata(PATH_TO_FeII+ 'Veron-cetty_2004.fits')
Veron_hd = pyfits.getheader(PATH_TO_FeII+'Veron-cetty_2004.fits')
Veron_wv = np.arange(Veron_hd['CRVAL1'], Veron_hd['CRVAL1']+ Veron_hd['NAXIS1'])

Tsuzuki = np.loadtxt(PATH_TO_FeII+'FeII_Tsuzuki_opttemp.txt')
Tsuzuki_d = Tsuzuki[:,1]
Tsuzuki_wv = Tsuzuki[:,0]

BG92 = np.loadtxt(PATH_TO_FeII+'bg92.con')
BG92_d = BG92[:,1]
BG92_wv = BG92[:,0]

def find_nearest(array, value):
    """ Find the location of an array closest to a value

	"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

with open(PATH_TO_FeII+'Preconvolved_FeII.txt', "rb") as fp:
    Templates= pickle.load(fp)

def FeII_Veron(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['Veron_dat'][:,index]

    fce = interp1d(Veron_wv*(1+z)/1e4, convolved , kind='cubic',fill_value=0, bounds_error=False)

    return fce(wave)

def FeII_Tsuzuki(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['Tsuzuki_dat'][:,index]

    fce = interp1d(Tsuzuki_wv*(1+z)/1e4, convolved , kind='cubic',fill_value=0, bounds_error=False)

    return fce(wave)

def FeII_BG92(wave,z, FWHM_feii):

    index = find_nearest(Templates['FWHMs'],FWHM_feii)
    convolved = Templates['BG92_dat'][:,index]

    fce = interp1d(BG92_wv*(1+z)/1e4, convolved , kind='cubic',fill_value=0, bounds_error=False)

    return fce(wave)