import tqdm
from . import Fitting as emfit
import os
import multiprocess as mp
from multiprocess import Pool
import numpy as np
from astropy.table import Table
import time
import warnings
import matplotlib.pyplot as plt 
from astropy.utils.exceptions import AstropyWarning

from . import Fitting as emfit
import pickle

import numba
from . import Support as sp


from . import QubeSpec

import time


class Halpha_OIII:
    def __init__(self):
        self.init = 'ok'

    def Spaxel_fitting(self, Cube,add='',Ncores=(mp.cpu_count() - 2),models='Single',priors= {'z':[0, 'normal', 0,0.003],\
                                                                                        'cont':[0,'loguniform',-4,1],\
                                                                                        'cont_grad':[0,'normal',0,0.3], \
                                                                                        'Hal_peak':[0,'loguniform',-3,1],\
                                                                                        'BLR_Hal_peak':[0,'loguniform',-3,1],\
                                                                                        'NII_peak':[0,'loguniform',-3,1],\
                                                                                        'Nar_fwhm':[300,'uniform',100,900],\
                                                                                        'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                        'zBLR':[0, 'normal', 0,0.003],\
                                                                                            'SIIr_peak':[0,'loguniform',-3,1],\
                                                                                            'SIIb_peak':[0,'loguniform',-3,1],\
                                                                                            'Hal_out_peak':[0,'loguniform',-3,1],\
                                                                                            'NII_out_peak':[0,'loguniform',-3,1],\
                                                                                            'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                            'outflow_vel':[-50,'normal', 0,300],\
                                                                                            'OIII_peak':[0,'loguniform',-3,1],\
                                                                                            'OIII_out_peak':[0,'loguniform',-3,1],\
                                                                                            'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                            'Hbeta_out_peak':[0,'loguniform',-3,1],\
                                                                                            'SIIr_peak':[0,'loguniform', -3,1],\
                                                                                            'SIIb_peak':[0,'loguniform', -3,1],\
                                                                                            'BLR_Hbeta_peak':[0,'loguniform', -3,1]}, **kwargs):
                                        
                                        
                                        
                                        
        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')

        self.priors = priors

        if Ncores<1:
            Ncores=1
        
        progress = kwargs.get('progress', True)
        progress = tqdm.tqdm if progress else lambda x, total=0: x

        if models=='Single':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_OIII_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_OIII_AGN_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR_simple':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_OIII_AGN_simple_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))

        elif models=='outflow_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_OIII_outflowboth_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
        
        elif models=='BLR_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_OIII_BLR_both_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                

        else:
            raise Exception('models variable not understood. Options are: Single, BLR, BLR_simple, BLR_both and outflow_both')

        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))
    
    def Spaxel_fitting_Halpha_OIII_toptup(self, Cube, to_fit ,add='',progress=True, Ncores=(mp.cpu_count() - 2),models='Single',priors= {'z':[0, 'normal', 0,0.003],\
                                                                                       'cont':[0,'loguniform',-4,1],\
                                                                                       'cont_grad':[0,'normal',0,0.3], \
                                                                                       'Hal_peak':[0,'loguniform',-3,1],\
                                                                                       'BLR_Hal_peak':[0,'loguniform',-3,1],\
                                                                                       'NII_peak':[0,'loguniform',-3,1],\
                                                                                       'Nar_fwhm':[300,'uniform',100,900],\
                                                                                       'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                       'zBLR':[0, 'normal', 0,0.003],\
                                                                                        'SIIr_peak':[0,'loguniform',-3,1],\
                                                                                        'SIIb_peak':[0,'loguniform',-3,1],\
                                                                                        'Hal_out_peak':[0,'loguniform',-3,1],\
                                                                                        'NII_out_peak':[0,'loguniform',-3,1],\
                                                                                        'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                        'outflow_vel':[-50,'normal', 0,300],\
                                                                                        'OIII_peak':[0,'loguniform',-3,1],\
                                                                                        'OIII_out_peak':[0,'loguniform',-3,1],\
                                                                                        'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                        'Hbeta_out_peak':[0,'loguniform',-3,1],\
                                                                                        'SIIr_peak':[0,'loguniform', -3,1],\
                                                                                        'SIIb_peak':[0,'loguniform', -3,1],\
                                                                                        'BLR_Hbeta_peak':[0,'loguniform', -3,1]}, **kwargs):
        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
            Cube_res= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        

        for i, row in enumerate(Cube_res):
            y,x, res = row
            if to_fit[0]==x and to_fit[1]==y:
                flx_spax_m, error, wave = res.fluxs, res.error, res.wave

                if models=='Single':
                    Fits_sig = emfit.Fitting(wave, flx_spax_m, error, Cube.z,N=10000,progress=progress, priors=priors)
                    Fits_sig.fitting_Halpha_OIII(model='gal' )
                    Fits_sig.fitted_model = 0
                    Cube_res[i]  = [x,y,Fits_sig ]
                
                elif models=='BLR':
                    Fits_sig = emfit.Fitting(wave, flx_spax_m, error, Cube.z,N=10000,progress=progress, priors=priors)
                    Fits_sig.fitting_Halpha_OIII(model='BLR' )
                    Fits_sig.fitted_model = 0
                    Cube_res[i]  = [x,y,Fits_sig ]

                elif models=='outflow_both':
                    Fits_sig = emfit.Fitting(wave, flx_spax_m, error, Cube.z,N=10000,progress=progress, priors=priors)
                    Fits_sig.fitting_Halpha_OIII(model='gal' )
                    Fits_sig.fitted_model = 0
                    
                    
                    Fits_out = emfit.Fitting(wave, flx_spax_m, error, Cube.z,N=10000,progress=progress, priors=priors)
                    Fits_out.fitting_Halpha_OIII(model='outflow' )
                    Fits_out.fitted_model = 0

                    Cube_res[i]  = [x,y,Fits_sig, Fits_out]
                    

                f,ax = plt.subplots(1, figsize=(10,5))
                ax.plot(Fits_sig.wave, Fits_sig.flux, drawstyle='steps-mid')
                ax.plot(Fits_sig.wave, Fits_sig.yeval, 'r--')

                ax.text(Fits_sig.wave[10], 0.9*max(Fits_sig.yeval), 'x='+str(x)+', y='+str(y) )

                break
        
                
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general'+add+'.txt', "wb") as fp:
            pickle.dump( Cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))



    def Fitting_Halpha_OIII_unwrap(self, lst, progress=False):
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha_OIII(model='gal' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_OIII_AGN_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha_OIII(model='BLR' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_OIII_AGN_simple_unwrap(self, lst, progress=False):
       
        i,j,flx_spax_m, error, wave, z = lst

        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha_OIII(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_OIII_outflowboth_unwrap(self, lst, progress=False):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha_OIII(model='gal' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_Halpha_OIII(model='outflow' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res

    def Fitting_Halpha_OIII_BLR_both_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha_OIII(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_Halpha_OIII(model='BLR' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res


class OIII:
    def __init__(self):
        self.status = 'ok'

    def Spaxel_fitting(self, Cube,models='Single',add='',template=0, Ncores=(mp.cpu_count() - 1), priors= {'z':[0, 'normal', 0,0.003],\
                                                                                        'cont':[0,'loguniform',-4,1],\
                                                                                        'cont_grad':[0,'normal',0,0.3], \
                                                                                        'Nar_fwhm':[300,'uniform',100,900],\
                                                                                        'BLR_fwhm':[4000,'uniform', 2000,9000],\
                                                                                        'zBLR':[0, 'normal', 0,0.003],\
                                                                                            'outflow_fwhm':[600,'uniform', 300,1500],\
                                                                                            'outflow_vel':[-50,'normal', 0,300],\
                                                                                            'OIII_peak':[0,'loguniform',-3,1],\
                                                                                            'OIII_out_peak':[0,'loguniform',-3,1],\
                                                                                            'Hbeta_peak':[0,'loguniform',-3,1],\
                                                                                            'Hbeta_out_peak':[0,'loguniform',-3,1],\
                                                                                            'BLR_Hbeta_peak':[0,'loguniform', -3,1]}, **kwargs):

        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')

        self.priors = priors
        self.template = template

        if Ncores<1:
            Ncores=1
        
        progress = kwargs.get('progress', True)
        progress = tqdm.tqdm if progress else lambda x, total=0: x

        if models=='Single':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_OIII_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_OIII_AGN_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR_simple':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_OIII_AGN_simple_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))

        elif models=='outflow_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_OIII_outflowboth_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
        
        elif models=='BLR_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_OIII_BLR_both_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                

        else:
            raise Exception('models variable not understood. Options are: Single, BLR, BLR_simple, BLR_both and outflow_both')

        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_OIII'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))

    def Fitting_OIII_unwrap(self, lst, progress=False):
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_OIII(model='gal' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_OIII_AGN_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors, template=self.template)
            Fits_sig.fitting_OIII(model='BLR' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_OIII_AGN_simple_unwrap(self, lst, progress=False):
       
        i,j,flx_spax_m, error, wave, z = lst

        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors, template= self.template)
            Fits_sig.fitting_OIII(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_OIII_outflowboth_unwrap(self, lst, progress=False):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_OIII(model='gal' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_OIII(model='outflow' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res

    def Fitting_OIII_BLR_both_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_OIII(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_OIII(model='BLR' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res
        
    
    def Fitting_OIII_unwrap(self,lst):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        
        Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, priors=self.priors)
        Fits_sig.fitting_OIII(model='gal_simple')
        Fits_sig.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig]
                    
        return cube_res
    
    def Fitting_OIII_outflowboth_unwrap(self,lst, progress=False):
    
        i,j,flx_spax_m, error, wave, z = lst
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_OIII(model='gal' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_OIII(model='outflow' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res
                    
        return cube_res
    
class Halpha:
    def __init__(self):
        self.status = 'ok'

    def Spaxel_fitting(self, Cube,models='Single', add='',Ncores=(mp.cpu_count() - 1),priors={'cont':[0,-3,1],\
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
                                                        'outflow_vel':[-50, -300,300]}, **kwargs):
        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)

        print('import of the unwrap cube - done')

        self.priors = priors

        if Ncores<1:
            Ncores=1
        
        progress = kwargs.get('progress', True)
        progress = tqdm.tqdm if progress else lambda x, total=0: x

        if models=='Single':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_AGN_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        elif models=='BLR_simple':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_AGN_simple_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))

        elif models=='outflow_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_outflowboth_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
        
        elif models=='BLR_both':
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_Halpha_BLR_both_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                

        else:
            raise Exception('models variable not understood. Options are: Single, BLR, BLR_simple, BLR_both and outflow_both')

        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_OIII'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)

        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))

    def Fitting_Halpha_unwrap(self, lst, progress=False):
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha(model='gal' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_AGN_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha(model='BLR' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_AGN_simple_unwrap(self, lst, progress=False):
       
        i,j,flx_spax_m, error, wave, z = lst

        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            cube_res  = [i,j, Fits_sig]
            
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
        return cube_res

    def Fitting_Halpha_outflowboth_unwrap(self, lst, progress=False):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha(model='gal' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_Halpha(model='outflow' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res

    def Fitting_Halpha_BLR_both_unwrap(self, lst, progress=False):
        
        i,j,flx_spax_m, error, wave, z = lst
        try:
            Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_sig.fitting_Halpha(model='BLR_simple' )
            Fits_sig.fitted_model = 0
            
            Fits_out = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=progress, priors=self.priors)
            Fits_out.fitting_Halpha(model='BLR' )
            Fits_out.fitted_model = 0
            
            cube_res  = [i,j,Fits_sig, Fits_out ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        return cube_res
        
    
    def Fitting_Halpha_unwrap(self,lst):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
            priors= pickle.load(fp) 
        
        Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, priors=priors)
        Fits_sig.fitting_Halpha(model='gal')
        Fits_sig.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig]
                    
        return cube_res
    
    def Fitting_Halpha_outflowboth_unwrap(self,lst):
    
        i,j,flx_spax_m, error, wave, z = lst
        
        with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
            priors= pickle.load(fp) 
        
        Fits_sig = emfit.Fitting(wave, flx_spax_m, error, z,N=10000,progress=False, priors=priors)
        Fits_sig.fitting_OIII(model='gal')
        Fits_sig.fitted_model = 0
        
        cube_res  = [i,j, Fits_sig]
                    
        return cube_res

class general:
    def __init__(self):
        self.status = 'ok'

    def Spaxel_fitting_general_MCMC_mp(self, Cube,fitted_model, labels, priors, logprior, nwalkers=64,use=np.array([]), N=10000, add='',Ncores=(mp.cpu_count() - 2), **kwargs):
        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
            Unwrapped_cube= pickle.load(fp)
            
        print('import of the unwrap cube - done')
        
        self.priors= priors
        self.fitted_model = fitted_model
        self.labels = labels
        self.logprior = logprior
        self.nwalkers = nwalkers
        self.use = use
        self.N = N     
        
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
                self.Fitting_general_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube))
        else:
            with Pool(Ncores) as pool:
                cube_res = list(progress(
                    pool.imap(
                        self.Fitting_general_unwrap, Unwrapped_cube),
                    total=len(Unwrapped_cube)))
                
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general'+add+'.txt', "wb") as fp:
            pickle.dump( cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))


    def Spaxel_fitting_general_toptup(self, Cube, to_fit ,fitted_model, labels, priors, logprior, nwalkers=64,use=np.array([]), N=10000, add='',Ncores=(mp.cpu_count() - 2), **kwargs):
        import pickle
        start_time = time.time()
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general'+add+'.txt', "rb") as fp:
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
                z = Cube.z
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
        
                
        with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general'+add+'.txt', "wb") as fp:
            pickle.dump( Cube_res,fp)  
        
        print("--- Cube fitted in %s seconds ---" % (time.time() - start_time))
    
    def Fitting_general_unwrap(self, lst, progress=False):
        with open(os.getenv("HOME")+'/priors.pkl', "rb") as fp:
            data= pickle.load(fp) 

        i,j,flx_spax_m, error, wave, z = lst
        

        if len(use)==0:
            use = np.linspace(0, len(wave)-1, len(wave), dtype=int)

        try:
            Fits_sig = emfit.Fitting(wave[self.use], flx_spax_m[self.use], error[self.use], z,N=self.N,progress=progress, priors=self.priors)
            Fits_sig.fitting_general(self.fitted_model, self.labels, self.logprior, nwalkers=self.nwalkers)
            Fits_sig.fitted_model = 0
        
                
            cube_res  = [i,j,Fits_sig ]
        except Exception as _exc_:
            print(_exc_)
            cube_res = [i,j, {'Failed fit':0}, {'Failed fit':0}]
            print('Failed fit')
        
        return cube_res

def Spaxel_ppxf(Cube, ncpu=2):
    import glob
    import yaml
    from yaml.loader import SafeLoader

    # Open the file and load the file
    with open('/Users/jansen/My Drive/MyPython/Qubespec/QubeSpec/jadify_temp/r100_jades_deep_hst_v3.1.1_template.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)

    data['dirs']['data_dir'] = Cube.savepath+'PRISM_spaxel/'
    data['dirs']['output_dir'] = Cube.savepath+'PRISM_spaxel/'
    data['ppxf']['redshift_table'] = Cube.savepath+'PRISM_1D/redshift_1D.csv'

    with open(Cube.savepath+'/PRISM_spaxel/R100_1D_setup_test.yaml', 'w') as f:
        data = yaml.dump(data, f, sort_keys=False, default_flow_style=True)
    from . import jadify_temp as pth
    PATH_TO_jadify = pth.__path__[0]+ '/'
    filename = PATH_TO_jadify+ 'red_table_template.csv'
    redshift_cat = Table.read(filename)
    
    files = glob.glob(Cube.savepath+'PRISM_spaxel/prism_clear/*.fits')
    
    IDs= np.array([], dtype=int)
    for i, file in enumerate(files):
        comp = file.split('/')
        IDs = np.append( IDs, int(comp[-1][:6]))
    redshift_cat_mod = Table()
    redshift_cat_mod['ID'] = IDs
    redshift_cat_mod['z_visinsp'] = np.ones_like(len(IDs))*Cube.z
    redshift_cat_mod['z_phot'] = np.ones_like(len(IDs))*Cube.z
    redshift_cat_mod['z_bagp'] = np.ones_like(len(IDs))*Cube.z
    redshift_cat_mod['flag'] = np.zeros_like(IDs, dtype='<U6')
    redshift_cat_mod['flag'][:] = redshift_cat['flag'][0]
    redshift_cat_mod.write(Cube.savepath+'PRISM_spaxel/redshift_spaxel.csv',overwrite=True)
    
    import nirspecxf
    config100 = nirspecxf.NIRSpecConfig(Cube.savepath+'PRISM_spaxel/R100_1D_setup_manual.yaml')
    #xid = IDs[3]
    #ns, _ = nirspecxf.process_object_id(id, config100)
    nirspecxf.process_multi(ncpu, IDs, config100)
    #for i, id in enumerate(IDs):
    #    print(i)
    #    ns, _ = nirspecxf.process_object_id(id, config100)
    print('Fitting done, merging results')
    nirspecxf.data_prods.merge_em_lines_tables(
        Cube.savepath+'PRISM_spaxel/res/*R100_em_lines.fits',
        Cube.savepath+'PRISM_spaxel/spaxel_R100_ppxf_emlines.fits')