
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import tqdm
from astropy.io import fits

def background_sub_spec_depricated(Cube, center, rad=0.6, manual_mask=[],smooth=25, plot=0):
    '''
    Background subtraction used when the NIRSPEC cube has still flux in the blank field.

    Parameters
    ----------
    center : TYPE
        DESCRIPTION.
    rad : TYPE, optional
        DESCRIPTION. The default is 0.6.
    plot : TYPE, optional
        DESCRIPTION. The default is 0
    Returns
    -------
    None.

    '''

    # Creating a mask for all spaxels.
    shapes = Cube.dim
    mask_catch = Cube.flux.mask.copy()
    mask_catch[:,:,:] = True
    header  = Cube.header
    #arc = np.round(1./(header['CD2_2']*3600))
    arc = np.round(1./(header['CDELT2']*3600))

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

    mask_spax = mask_catch.copy()
    # Loading mask of the sky lines an bad features in the spectrum
    mask_sky_1D = Cube.sky_clipped_1D.copy()
    total_mask = np.logical_or( mask_spax, Cube.sky_clipped)

    background = np.ma.array(data=Cube.flux.data, mask= total_mask)
    backgroerr =  np.ma.array(data=Cube.error_cube, mask= total_mask)
    weights = 1/backgroerr**2; weights /= np.ma.sum(weights, axis=(1,2))[:, None, None]                       
    
    master_background_sum = np.ma.sum(background*weights, axis=(1,2))
    Sky = master_background_sum
    '''
    Sky = np.ma.median(flux, axis=(1,2))
    Sky = np.ma.array(data = Sky.data, mask=mask_sky_1D)
    '''

    Sky_smooth = medfilt(Sky, smooth)

    for ix in range(shapes[0]):
        for iy in range(shapes[1]):
            Cube.flux[:,ix,iy] = Cube.flux[:,ix,iy] - Sky_smooth

    if plot==1:
        plt.figure()
        plt.title('Median Background spectrum')

        plt.plot(Cube.obs_wave, np.ma.array(data= Sky_smooth , mask=Cube.sky_clipped_1D), drawstyle='steps-mid')


        plt.ylabel('Flux')
        plt.xlabel('Observed wavelength')

    return Sky_smooth, Cube.flux


def background_subtraction(Cube, box_size=(21,21), filter_size=(5,5), sigma_clip=5,\
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
    from photutils.background import Background2D, MedianBackground
    from astropy.stats import SigmaClip

    n_wave, n_x, n_y = Cube.flux.shape
    Cube.background = np.full((n_wave, n_x, n_y), np.nan)
    Cube.coverage_mask = Cube.flux.data==np.nan
    Cube.coverage_mask = Cube.coverage_mask[100,:,:]
    if len(source_mask) !=0:
        print('Using supplied source mask')
        source_mask_temp = source_mask.copy()
        source_mask[source_mask_temp==0] = True
        source_mask[source_mask_temp==1] = False
    else:
        from .detection import Detection as dtn
        if any(wave_range):
            print('Using sextractor to find the source. ')
            obj, seg = dtn.source_detection(Cube.flux, Cube.error_cube, Cube.obs_wave, wave_range=wave_range,noise_type='nominal', detection_threshold=detection_threshold)
            source_mask = Cube.coverage_mask.copy()
            source_mask[seg !=0] = True
            source_mask[seg ==0] = False       
        else:
            raise Exception('Define wave_range or source_mask ')
            

    for _wave_,_image_ in tqdm.tqdm(enumerate(Cube.flux)):
        mask = ~np.isfinite(_image_)
        mask = mask if source_mask is None else mask | source_mask

        #plt.figure()
        #plt.imshow(mask, origin='lower')
        #plt.show()
        try:
            background2d = Background2D(
                    _image_, box_size, filter_size=filter_size, mask=mask,
                    coverage_mask=Cube.coverage_mask, sigma_clip=SigmaClip(sigma=sigma_clip),
                    bkg_estimator=MedianBackground(), **kwargs)
            
            Cube.background[_wave_,:,:] = background2d.background
        except Exception as _exc_:
            print(_exc_)
            background2d = np.full(_image_.shape, np.nan)

            Cube.background[_wave_,:,:] = background2d

    # For wavelength slices where all spaxels were invalid, interpolate linearly
    # between nearby wavelengths.
    wave_mask = np.all(np.isnan(Cube.background), axis=(1,2))
    wave_indx = np.linspace(0., 1, n_wave) # Dummy variable.

    if np.any(wave_mask):
        for i in range(n_x):
            for j in range(n_y):
                if Cube.coverage_mask[i, j]:
                        continue
                Cube.background[wave_mask, i, j] = np.interp(
                        wave_indx[wave_mask], wave_indx[~wave_mask],
                        Cube.background[~wave_mask, i, j])
                
    
    from scipy import signal
    if wave_smooth:
        Cube.backgrond = signal.medfilt(Cube.background, (wave_smooth, 1, 1))
    Cube.flux_old = Cube.flux.copy()
    Cube.flux = Cube.flux-Cube.background

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=Cube.header)
    hdus = [primary_hdu]
    hdus.append(fits.ImageHDU(Cube.background.data, name='background'))
    hdus.append(fits.ImageHDU(Cube.flux.data, name='flux_bkg'))

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(Cube.savepath+'/'+Cube.ID+'BKG.fits', overwrite=True)

    if plot==1:
        f, ax = plt.subplots(1)
        ax.plot(Cube.obs_wave, np.median(Cube.background[:, int(n_x/2)-5:int(n_x/2)+5, int(n_y/2)-5:int(n_y/2)+5], axis=(1,2)), drawstyle='steps-mid')
        ax.set_xlabel('obs_wave')
        ax.set_ylabel('Flux density')
    
    return Cube.background, Cube.flux, Cube.flux_old

def background_sub_spec_gnz11(Cube, center, rad=0.6, manual_mask=[],smooth=25, plot=0):
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
    # Creating a mask for all spaxels.
    shapes = Cube.dim
    mask_catch = Cube.flux.mask.copy()
    mask_catch[:,:,:] = True
    header  = Cube.header
    #arc = np.round(1./(header['CD2_2']*3600))
    arc = np.round(1./(header['CDELT2']*3600))

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

    mask_spax = mask_catch.copy()
    # Loading mask of the sky lines an bad features in the spectrum
    mask_sky_1D = Cube.sky_clipped_1D.copy()
    total_mask = np.logical_or( mask_spax, Cube.sky_clipped)

    background = np.ma.array(data=Cube.flux.data, mask= total_mask)
    backgroerr =  np.ma.array(data=Cube.error_cube, mask= total_mask)
    weights = 1/backgroerr**2; weights /= np.ma.sum(weights, axis=(1,2))[:, None, None]                       
    
    master_background_sum = np.ma.sum(background*weights, axis=(1,2))
    Sky = master_background_sum
    '''
    Sky = np.ma.median(flux, axis=(1,2))
    Sky = np.ma.array(data = Sky.data, mask=mask_sky_1D)
    '''
    from scipy.signal import medfilt
    Sky_smooth = medfilt(Sky, smooth)

    Cube.collapsed_bkg = Sky_smooth
    
    use = np.where( (Cube.obs_wave<1.38) & (Cube.obs_wave>1.30) )[0]
    white_image = np.ma.median(Cube.flux[use, :,:], axis=(0))
    white_bkg = np.ma.median(Sky_smooth[use])
    norm = white_image/white_bkg

    plt.figure()
    plt.imshow(norm,vmin=0.5,vmax=1.5, origin='lower')
    plt.colorbar()
    Cube.flux_orig = Cube.flux.copy()
    for ix in range(shapes[0]):
        for iy in range(shapes[1]):
            Cube.flux[:,ix,iy] = Cube.flux[:,ix,iy] - Sky_smooth*norm[ix,iy]

    return Cube.collapsed_bkg, Cube.flux

    