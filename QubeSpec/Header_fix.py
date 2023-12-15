#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:24:16 2023

@author: jansen
"""


import shutil
from astropy.io import fits

def astronometry_cor(JWST_FILE, new_val):
    altered_file=JWST_FILE.replace(JWST_FILE,\
                               JWST_FILE[:-4]+'_shifted.fits')
    shutil.copyfile(JWST_FILE, altered_file)

    [CRV1,CRV2,CRP1,CRP2]=new_val
    fits.setval(altered_file, 'CRPIX1', value=float(CRP1),ext=1)
    fits.setval(altered_file, 'CRPIX2', value=float(CRP2),ext=1)
    fits.setval(altered_file, 'CRVAL1', value=float(CRV1),ext=1)
    fits.setval(altered_file, 'CRVAL2', value=float(CRV2),ext=1)

