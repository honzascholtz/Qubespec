#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:23:35 2022

@author: jansen
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt; plt.ioff()

from astropy.io import fits as pyfits
from astropy.table import Table, join, vstack
from astropy.modeling.powerlaws import PowerLaw1D

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

from . import FeII_templates as pth
PATH_TO_FeII = pth.__path__[0]+ '/'

from . import FeII_models as Fem

def gauss(x, k, mu,FWHM):
    sig = FWHM/3e5*mu/2.35482
    expo= -((x-mu)**2)/(2*sig*sig)
    y= k* e**expo
    return y

def OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak):
    OIIIr = 5008.24*(1+z)/1e4
    OIIIb = OIIIr- (48.*(1+z)/1e4)

    Hbeta = 4862.6*(1+z)/1e4
    OIII_nar = gauss(x, OIIIn_peak, OIIIr,OIII_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, OIII_fwhm)

    Hbeta_wv = Hbeta 
    Hbeta_nar = gauss(x, Hbeta_peak, Hbeta_wv, OIII_fwhm )
    contm = PowerLaw1D.evaluate(x, cont, OIIIr, alpha=cont_grad)
    return contm+ OIII_nar + Hbeta_nar

def OIII_outflow(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak):

    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
    z_out = z+ out_vel/3e5*(1+z)
    y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)

    return y

def OIII_gal_BLR(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak,\
                  zBLR, Hbeta_blr_peak, BLR_fwhm):
    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)

    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )
    return y

def OIII_outflow_BLR(x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak,\
                      zBLR, Hbeta_blr_peak, BLR_fwhm):

    y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
    z_out = z+ out_vel/3e5*(1+z)
    y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)
    
    Hbeta_blr = 4862.6*(1+zBLR)/1e4
    y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

    return y


class OIII_fe_models:
    def __init__(self, template):
        self.template = template

    def OIII_gal_BLR_Fe(self, x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak,\
                    zBLR, Hbeta_blr_peak, BLR_fwhm,\
                        FeII_peak, FeII_fwhm):
        
        y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)

        Hbeta_blr = 4862.6*(1+zBLR)/1e4
        y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

        if self.template=='BG92':
            FeII_fce = Fem.FeII_BG92
        if self.template=='Tsuzuki':
            FeII_fce = Fem.FeII_Tsuzuki
        if self.template=='Veron':
            FeII_fce = Fem.FeII_Veron

        y += FeII_peak*FeII_fce(x, z, FeII_fwhm)

        return y

    def OIII_outflow_BLR_Fe(self, x, z, cont,cont_grad, OIIIn_peak, OIIIw_peak, OIII_fwhm, OIII_out, out_vel, Hbeta_peak, Hbeta_out_peak,\
                        zBLR, Hbeta_blr_peak, BLR_fwhm,\
                            FeII_peak, FeII_fwhm):

        y = OIII_gal(x, z, cont, cont_grad, OIIIn_peak,  OIII_fwhm, Hbeta_peak)
        z_out = z+ out_vel/3e5*(1+z)
        y += OIII_gal(x, z_out, 0, 0, OIIIw_peak,  OIII_out, Hbeta_out_peak)
        
        Hbeta_blr = 4862.6*(1+zBLR)/1e4
        y += gauss(x, Hbeta_blr_peak, Hbeta_blr, BLR_fwhm )

        if self.template=='BG92':
            FeII_fce = Fem.FeII_BG92
        if self.template=='Tsuzuki':
            FeII_fce = Fem.FeII_Tsuzuki
        if self.template=='Veron':
            FeII_fce = Fem.FeII_Veron

        y += FeII_peak*FeII_fce(x, z, FeII_fwhm)

        return y
