.. _spaxel_fitting:

Spaxel-by-Spaxel fitting
=======================================

Now we are going to perform the spaxel-by-spaxel fitting. The whole process is split across 3 seperate steps and we will slowly explore all of them:

#. "Unwrapping the cube": Extracting all of the spaxel spectra, binning them if required, and estimating the uncertainties of the flux 
#. "Spaxel fitting": fitting each of the spectra extracted above
#. "Map creation": Post processing of the fits to each of the specta: 



1) Unwrapping the cube
----------------------

.. automethod:: QubeSpec.Cube.unwrap_cube



.. code:: ipython3

    #importing modules
    import numpy as np
    import matplotlib.pyplot as plt; plt.ioff()
    
    
    c= 3e8
    h= 6.62*10**-34
    k= 1.38*10**-23
    
    %load_ext autoreload
    %autoreload 2
    
    import QubeSpec as IFU
    import QubeSpec.Plotting as emplot
    import QubeSpec.Fitting as emfit


    Cube = IFU.Cube()
    Cube.load('/Users/jansen/Test.txt')

    mask_spaxel = IFU.sp.QFitsview_mask(QubeSpec_setup['Spaxel_mask'])
    
    plt.figure()
    plt.imshow(mask_spaxel, cmap='gray', origin='lower')
    plt.show()

    Unwrapping = False
    if Unwrapping==True:
        Cube.unwrap_cube(instrument='NIRSPEC05',mask_manual=mask_spaxel, \
                         err_range=QubeSpec_setup['err_range'],\
                         boundary=QubeSpec_setup['err_boundary'],\
                         add='',\
                         sp_binning= QubeSpec_setup['Spaxel_Binning']) 
    plt.show()

.. image:: Spaxel_fitting_files/Spaxel_fitting_5_0.png



2) Fitting spaxel-by-spaxel
----------------------

In order to fit all of the spaxels we need to use the ``QubeSpec.Spaxel`` module. Similar to the fitting the 1D collapsed spectra,
there are pre written functions/classes to allow fit the basic Halpha, [OIII] and Halpha+[OIII] and of course full custom functions. 
The four classes classes are:

* ``QubeSpec.Spaxel.Halpha`` - class to fit Halpha complex
* ``QubeSpec.Spaxel.OIII`` - class to fit [OIII] complex
* ``QubeSpec.Spaxel.Halpha_OIII`` - class to fit Halpha+[OIII] complex
* ``QubeSpec.Spaxel.general`` - class to fit any custom function. 

Within each class there are couple of standarized models and combination of models to fit (``models='Single'``):

* ``Single`` - single Gaussian per emission line
* ``BLR`` - BLR + outflow
* ``BLR_simple`` - BLR (no outflow)
* ``outflow_both`` - Fits single model and outflow models and then we decide later which fit to use 
* ``BLR_both`` - BLR and BLR_simple and then we decide later which fit to use 

Below is the full description of the fitting function:

.. automethod:: QubeSpec.Spaxel.Halpha_OIII.Spaxel_fitting

And below is an example of how to trigger it: 

.. code:: ipython3
    dvmax = 1000/3e5*(1+Cube.z)
    dvstd = 200/3e5*(1+Cube.z)
    priors={'z':[Cube.z,'normal_hat', Cube.z, dvstd, Cube.z-dvmax, Cube.z+dvmax]}
    priors['cont']=[0,'loguniform',-7,1]
    priors['Nar_fwhm']=[300,'uniform',100,400]

    priors['cont_grad']= [0,'normal',0,0.3]
    priors['Hal_peak']=[0,'loguniform',-5,1]
    priors['NII_peak']=[0,'loguniform',-5,1]
    priors['SIIr_peak']=[0,'loguniform',-5,1]
    priors['SIIb_peak']= [0,'loguniform',-7,1]
    priors['OIII_peak']=[0,'loguniform',-7,1]
    priors['Hbeta_peak']=[0,'loguniform',-6,1]

    priors['OIII_out_peak']=[0,'loguniform',-6,1]
    priors['Hbeta_out_peak']=[0,'loguniform',-6,1]
    priors['Hal_out_peak']=[0,'loguniform',-6,1]
    priors['NII_out_peak']=[0,'loguniform',-6,1]
    priors['outflow_fwhm']=[800,'uniform', 600,2000]
    priors['outflow_vel']=[-50,'normal', 0,300]

    priors['BLR_Hbeta_peak']=[0,'loguniform', -6,1]
    priors['BLR_Hal_peak']=[0,'loguniform',-6,1]
    priors['zBLR']=[0, 'normal', 0,0.003]
    priors['BLR_fwhm']=[4000,'uniform', 2000,9000]

    Spaxel = True
    if Spaxel==True: 
        if __name__ == '__main__':
            spx = IFU.Spaxel.Halpha_OIII()
            spx.Spaxel_fitting(Cube, models='outflow_both',add='_test', Ncores=QubeSpec_setup['ncpu'], priors=priors)


Please not couple of things. First in the ``priors``, we have set the low boundary of the ``_peak`` and ``cont`` to quite low (-6 or 1e-6).
This allows for pretty low values when fitting spaxel spectrum with very low fluxes in them. Secondly, please make sure that you run the Spaxel
fitting in the ``if __name__ == '__main__':`` or the multiprocess code will freak out. 

In order to fit a custom function, we need to define similar things as for the general fitting in 1D spectrum fitting. Actually, I highly recommend
that we use the exact function, labels, priors, etc. This way will make sure that things work on a single spectrum before we fit all of the spaxel. 
See example below:


.. code:: ipython3

    def gauss(x, k, mu,FWHM):
        sig = FWHM/3e5*mu/2.35482
        expo= -((x-mu)**2)/(2*sig*sig)
    
        y= k* e**expo
    
        return y
    from astropy.modeling.powerlaws import PowerLaw1D
    
    def Full_optical(x, z, cont,cont_grad,  Hal_peak, NII_peak, OIIIn_peak, Hbeta_peak, Hgamma_peak, Hdelta_peak, NeIII_peak, OII_peak, OII_rat,OIIIc_peak, HeI_peak,HeII_peak, Nar_fwhm):
        # Halpha side of things
        Hal_wv = 6564.52*(1+z)/1e4
        NII_r = 6585.27*(1+z)/1e4
        NII_b = 6549.86*(1+z)/1e4
        
        OIIIr = 5008.24*(1+z)/1e4
        OIIIb = 4960.3*(1+z)/1e4
        Hbeta = 4862.6*(1+z)/1e4
    
        Hal_nar = gauss(x, Hal_peak, Hal_wv, Nar_fwhm)
        NII_nar_r = gauss(x, NII_peak, NII_r, Nar_fwhm)
        NII_nar_b = gauss(x, NII_peak/3, NII_b, Nar_fwhm)
        
        Hgamma_wv = 4341.647191*(1+z)/1e4
        Hdelta_wv = 4102.859855*(1+z)/1e4
        
        Hgamma_nar = gauss(x, Hgamma_peak, Hgamma_wv, Nar_fwhm)
        Hdelta_nar = gauss(x, Hdelta_peak, Hdelta_wv, Nar_fwhm)
        
        
        # [OIII] side of things
        OIIIr = 5008.24*(1+z)/1e4
        OIIIb = 4960.3*(1+z)/1e4
        Hbeta = 4862.6*(1+z)/1e4
    
        OIII_nar = gauss(x, OIIIn_peak, OIIIr, Nar_fwhm) + gauss(x, OIIIn_peak/3, OIIIb, Nar_fwhm)
        Hbeta_nar = gauss(x, Hbeta_peak, Hbeta, Nar_fwhm)
        
        NeIII = gauss(x, NeIII_peak, 3869.68*(1+z)/1e4, Nar_fwhm ) + gauss(x, 0.322*NeIII_peak, 3968.68*(1+z)/1e4, Nar_fwhm)
        
        OII = gauss(x, OII_peak, 3727.1*(1+z)/1e4, Nar_fwhm )  + gauss(x, OII_rat*OII_peak, 3729.875*(1+z)/1e4, Nar_fwhm) 
        
        OIIIc = gauss(x, OIIIc_peak, 4364.436*(1+z)/1e4, Nar_fwhm )
        HeI = gauss(x, HeI_peak, 3889.73*(1+z)/1e4, Nar_fwhm )
        HeII = gauss(x, HeII_peak, 4686.0*(1+z)/1e4, Nar_fwhm )
    
        contm = PowerLaw1D.evaluate(x, cont,Hal_wv, alpha=cont_grad)
    
        return contm+Hal_nar+NII_nar_r+NII_nar_b + OIII_nar + Hbeta_nar + Hgamma_nar + Hdelta_nar + NeIII+ OII + OIIIc+ HeI+HeII
    
    labels= ['z', 'cont','cont_grad',  'Hal_peak', 'NII_peak', 'OIII_peak', 'Hbeta_peak','Hgamma_peak', 'Hdelta_peak','NeIII_peak','OII_peak','OII_rat','OIIIaur_peak', 'HeI_peak','HeII_peak', 'Nar_fwhm']

    dvmax = 1000/3e5*(1+Cube.z)
    dvstd = 200/3e5*(1+Cube.z)
    priors={'z':[Cube.z,'normal_hat', Cube.z, dvstd, Cube.z-dvmax, Cube.z+dvmax]}
    
    priors['cont']=[0.001,'loguniform', -4,1]
    priors['cont_grad']=[0.1,'normal', 0,0.2]
    priors['Hal_peak']=[0.1,'loguniform', -4,1]
    priors['NII_peak']=[0.4,'loguniform', -4,1]
    priors['Nar_fwhm']=[300,'uniform', 200,900]
    priors['OIII_peak']=[0.1,'loguniform', -4,1]
    priors['OI_peak']=[0.01,'loguniform', -4,1]
    priors['HeI_peak']=[0.01,'loguniform', -4,1]
    priors['Hbeta_peak']=[0.02,'loguniform', -4,1]
    priors['Hgamma_peak'] = [0.02,'loguniform',-4,1]
    priors['Hdelta_peak'] = [0.01,'loguniform',-4,1]
    priors['NeIII_peak'] = [0.01,'loguniform',-4,1]
    priors['OII_peak'] = [0.01,'loguniform',-4,1]
    priors['OII_rat']=[1,'uniform', 0.2,4]
    priors['OIIIaur_peak']=[0.01,'loguniform', -4,1]

Please notice above in the priors that we have intenationally put the initial conditions for the ``_peak`` to be ~5-10 smaller than in the 1D spectra case.
Secondly, the lower boundaries for the ``_peak`` are also smaller. 

Below is the full description of the ``Spaxel_fitting`` function.

.. automethod:: QubeSpec.Spaxel.general.Spaxel_fitting

And here is the example to run it. Ass you can see we have supplied all of the same info as for fitting a 1D spectrum and the same variabls 
as for pre written models. 

.. code:: ipython3

    Spaxel = False
    if Spaxel==True: 
        if __name__ == '__main__':
            spx = IFU.Spaxel.general()
            spx.Spaxel_fitting_general_MCMC_mp(Cube, Full_optical,labels, priors, emfit.logprior_general_scipy, add='', Ncores=QubeSpec_setup['ncpu'])


3) Map creation
----------------------

During the Spaxel-by-Spaxel fitting above, we only create ``QubeSpec.Fitting.Fitting`` class instance for every spaxel and save it into a text document (pickling it).
However, we dont actually extract any useful information (such as fluxes, velocities, velocity widths, etc). As such, we need to post process all of the fitting results.

To post process the results, we will usethe ``QubeSpec.Maps`` module. As usual there are pre written function to post process the results for the usual emission line combination
and general functions. 

.. automethod:: QubeSpec.Maps.Map_creation_Halpha_OIII

This is the same for ``QubeSpec.Maps.Map_creation_Halpha`` and ``QubeSpec.Maps.Map_creation_OIII``

For the general fit, you need supply it additional information to extract all of the emission lines. 

.. automethod:: QubeSpec.Maps.Map_creation_general

most importantly you need to supply the ``info`` dictionary containing the information needed to extract the emission lines. 

The shape of the ``info`` dictionary should be as below:

.. code:: ipython3

    OIII_kins = {'fwhms':['Nar_fwhm','outflow_fwhm'], 'vels':['outflow_vel'], 'peaks':['OIII_peak', 'OIII_out_peak']}
    Hal_kins = {'fwhms':['Nar_fwhm'], 'vels':[], 'peaks':['Hal_peak']}

    info = {'Hal': {'wv':6563,'fwhm':'Nar_fwhm','kin':Hal_kins}}
    info['NII'] = {'wv':6583, 'fwhm':'Nar_fwhm',}
    info['OIII'] = {'wv':5008, 'fwhm':'Nar_fwhm', 'kin': OIII_kins}
    info['OIII_out'] = {'wv':5008, 'fwhm':'outflow_fwhm',}
    info['Hbeta'] = {'wv':4861, 'fwhm':'Nar_fwhm',}
    info['Hgamma'] = {'wv':4341.647, 'fwhm':'Nar_fwhm',}
    info['Hdelta'] = {'wv':4102.859, 'fwhm':'Nar_fwhm',}
    info['NeIII'] = {'wv':3869.68, 'fwhm':'Nar_fwhm',}
    info['OII'] = {'wv':3727.1, 'fwhm':'Nar_fwhm',}
    info['OIIIaur'] = {'wv':4363, 'fwhm':'Nar_fwhm',}
    info['HeI'] = {'wv':3889, 'fwhm':'Nar_fwhm',}
    info['params'] = ['z','outflow_vel', 'outflow_fwhm']

    fmaps = IFU.Maps.Map_creation_general(Cube, info, SNR_cut=4., add='_test' )

Each entry contains another dictionary with:

* ``'wv'`` - rest-frame wavelength of the emission line
* ``'fwhm'`` - name of the FWHM variable associated with that particular emission line component 
* ``'kin'`` -  If you want to recover the kinematics of the line or multiple components of the same line. The ``'kin'`` should contain a dictionary with the name of the peaks, FWHMs and velocities to get the v10,w80,v90 and peak velocity
* ``'params'`` - please put with a list of variables you would like to directly extract from the chains. 



We can then run the post processing suc this: 

.. code:: ipython3  
    fmaps = IFU.Maps.Map_creation_general(Cube, info,flux_max=1e-18, SNR_cut=4., width_upper=300,\
                brokenaxes_xlims= ((1.75,2.1),(2.2,2.4), (3,3.2)) )

    plt.show()

Visually inspecting the fits
-----------------------------------------

Once we fit all of the spaxels, we can visually inspect the maps and each of the fits. We need to use the
 ``QubeSpec.Visulizations`` module which initialize a UI to fit. To initialize it we need to supply the path to the 
 fits file containing the Spaxel maps and list of three fits extentions we want to show. 

.. code:: ipython3
    import numpy as np
    import QubeSpec.Visulizations as viz

    PATH='/Users/jansen/JADES/GA_NIFS/'

    Viz = viz.Visualize(PATH+'Results/GS551/GS551_R2700_general_fits_maps.fits',\
                         ['HAL','OIII','OIII_kin'])
    
    Viz.showme()



Something didnt fit right? lets refit it.
-----------------------------------------

There is a decent chat that not all (800?) fits are going to be perfect on the first try. Actually, it is quite likely. 
Therefore, I have written few "topup" functions. These function have the same syntax as the e.g. ``QubeSpec.Spaxel.Halpha_OIII.Spaxel_fitting``
but they also include variable ``to_fit`` with should contain a list of pair of coordinates to refit: 

.. code:: ipython3
    spx.toptup(Cube, to_fit = [59,48], fitted_model = Full_optical, labels=labels, priors=priors, logprior= emfit.logprior_general_scipy)

this will replace the fits with new ones. However, please remember to regenerate all of the maps. 