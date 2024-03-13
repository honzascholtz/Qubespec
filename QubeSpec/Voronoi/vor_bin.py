from os import path
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astropy import stats
from astropy.io import fits

from matplotlib.backends.backend_pdf import PdfPages
from brokenaxes import brokenaxes

from .. import Utils as sp
from .. import Plotting as emplot
from .. import Fitting as emfit

from ..Models import Halpha_OIII_models as HaO_models
import pickle

def Voronoi_binning(Map_SNR, Map_noise, target_SNR, plot=1,quiet=0):
    """
    Perform Voronoi binning on a 2D map.

    Parameters:
    - Map_SNR (ndarray): 2D array representing the signal-to-noise ratio map.
    - Map_noise (ndarray): 2D array representing the noise map.
    - target_SNR (float): Target signal-to-noise ratio.
    - plot (int, optional): Flag to enable/disable plotting. Default is 1 (enabled).
    - quiet (int, optional): Flag to enable/disable progress bar. Default is 0 (disabled).

    Returns:
    - Map_seg (ndarray): 2D array representing the segmented map.

    Note:
    - The input maps (Map_SNR and Map_noise) should have the same shape.
    - The output segmented map (Map_seg) will have the same shape as the input maps.

    """
    shapes = Map_SNR.shape
    x = []
    y = []
    SNR = []
    noise = []
    for i in range(shapes[1]):
        for j in range(shapes[0]):
            if np.isfinite(Map_SNR[j,i]):
                x.append(i)
                y.append(j)
                SNR.append(Map_SNR[j,i])
                noise.append(Map_noise[j,i])

    x = np.array(x)
    y = np.array(y)
    SNR = np.array(SNR)
    noise = np.array(noise)
    signal = noise*SNR
    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, x_gen, y_gen, x_bar, y_bar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, target_SNR, plot=plot, quiet=quiet, pixelsize=1)
    
    Map_seg = np.full_like(Map_SNR, np.nan)
    for k,(seg,i,j) in enumerate(zip(binNum,x,y)):
        Map_seg[j,i] = seg

    return Map_seg

def unwrap_voronoi(Cube, Map_seg, add='', err_range=[0], boundary=2.4 ):
    """
    Unwraps the Voronoi segments in a given Cube object and saves the results.

    Parameters:
    -----------
    Cube : object
        The Cube object containing the data.
    Map_seg : array-like
        The segmentation map.
    add : str, optional
        Additional string to append to the saved file name. Default is an empty string.
    err_range : list, optional
        The range of error values to consider. Default is [0].
    boundary : float, optional
        The boundary value for error scaling. Default is 2.4.

    Returns:
    --------
    None

    Notes:
    ------
    The Cube object should have the following attributes:
    - Sky_stack_mask: A 2D mask array indicating the sky stack.
    - instrument: A string indicating the instrument used.
    - lux: A 3D data array containing the flux values.
    - sky_clipped: A 2D mask array indicating the clipped sky.
    - sky_clipped_1D: A 1D mask array indicating the clipped sky.
    - error_cube: A 3D data array containing the error values.
    - obs_wave: A 1D array containing the observed wavelengths.
    - z: A float indicating the redshift value.
    - savepath: A string indicating the path to save the results.
    - ID: A string indicating the ID of the Cube object.
    - band: A string indicating the band of the Cube object.

    Examples:
    ---------
    >>> unwrap_voronoi(Cube, Map_seg)
    """

    itere = np.unique(Map_seg)
    segs= itere[np.isfinite(itere)]
    Spax_mask = Cube.Sky_stack_mask[0,:,:]

    Unwrapped_cube = []
    for seg in tqdm.tqdm(segs):
        Spax_mask[:,:] = True
        Spax_mask[Map_seg==seg] = False


        if Cube.instrument=='NIRSPEC_IFU':
            total_mask = np.logical_or(Spax_mask, Cube.sky_clipped)
            flx_spax_t = np.ma.array(data=Cube.flux.data,mask=total_mask)

            flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
            flx_spax_m = np.ma.array(data = flx_spax.data, mask=Cube.sky_clipped_1D)
            nspaxel= np.sum(np.logical_not(total_mask[22,:,:]))
            Var_er = np.sqrt(np.ma.sum(np.ma.array(data=Cube.error_cube.data, mask= total_mask)**2, axis=(1,2))/nspaxel)

            error = sp.error_scaling(Cube.obs_wave, flx_spax_m, Var_er, err_range, boundary,\
                                        exp=0)

        else:
            flx_spax_t = np.ma.array(data=Cube.flux.data,mask=Spax_mask)
            flx_spax = np.ma.median(flx_spax_t, axis=(1,2))
            flx_spax_m = np.ma.array(data = flx_spax.data, mask=Cube.sky_clipped_1D)

            error = stats.sigma_clipped_stats(flx_spax_m,sigma=3)[2] * np.ones(len(flx_spax))

        Unwrapped_cube.append([seg,seg,flx_spax_m, error,Cube.obs_wave, Cube.z])


    print(len(Unwrapped_cube))
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube_voronoi'+add+'.txt', "wb") as fp:
        pickle.dump(Unwrapped_cube, fp)



def fill_cube(array, fill, mask):
    shapes = mask.shape
    for j in range(shapes[1]):
        for i in range(shapes[0]):
           if mask[i,j]==True:
               array[:,i,j] = fill
    return array

def Map_creation_general(Cube,info, Map_seg, SNR_cut = 3 ,  width_upper=300,add='',\
                            brokenaxes_xlims= ((2.820,3.45),(3.75,4.05),(5,5.3)) ):
    """ Function to post process fits. The function will load the fits results and determine which model is more likely,
        based on BIC. It will then calculate the W80 of the emission lines, V50 etc and create flux maps, velocity maps eyc.,
        Afterwards it saves all of it as .fits file. 

        Parameters
        ----------
    
        Cube : QubeSpec.Cube class instance
            Cube class from the main part of the QubeSpec. 
        
        info : dict
            dictionary containing information on what to extract. 

        SNR_cut : float
            SNR cutoff to detect emission lines 
        
        add : str
            additional string to use to load the results and save maps/pdf
        
        brokenaxes_xlims: list
            list of wavelength ranges to use for broken axes when plotting

            
        """
    z0 = Cube.z
    failed_fits=0
    
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general_voronoi'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================
    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data
    
    info_keys = list(info.keys())
    
    for key in info_keys:
        if key=='params':
            info[key] = {'extract':info[key]}
            for param in info[key]['extract']:
                info[key][param] = np.full((3, Cube.dim[0], Cube.dim[1]),np.nan)

        else:
            map_flx = np.zeros((4,Cube.dim[0], Cube.dim[1]))
            map_flx[:,:,:] = np.nan
                
            info[key]['flux_map'] = map_flx
            
            if 'kin' in list(info[key]):
                info[key]['W80'] = np.full((3, Cube.dim[0], Cube.dim[1]),np.nan)
                info[key]['peak_vel'] = np.full((3, Cube.dim[0], Cube.dim[1]),np.nan)

                info[key]['v10'] = np.full((3, Cube.dim[0], Cube.dim[1]),np.nan)
                info[key]['v90'] = np.full((3, Cube.dim[0], Cube.dim[1]),np.nan)


    BIC_map = np.zeros((Cube.dim[0], Cube.dim[1]))
    BIC_map[:,:] = np.nan

    chi2_map = np.zeros((Cube.dim[0], Cube.dim[1]))
    chi2_map[:,:] = np.nan
    # =============================================================================
    #        Filling these maps
    # =============================================================================

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_general_fit_voronoi_detection_only'+add+'.pdf')

    for row in tqdm.tqdm(range(len(results))):

        try:
            i,j, Fits = results[row]
        except:
            ls=0
        if str(type(Fits)) == "<class 'dict'>":
            failed_fits+=1
            continue

        mask = Map_seg==i        
        #Result_cube_data[:,mask] = Fits.fluxs.data
        Result_cube_data = fill_cube(Result_cube_data, Fits.fluxs.data, mask)
        try:
            Result_cube_error= fill_cube(Result_cube_error, Fits.error.data, mask)
        except:
            lds=0
        
        Result_cube= fill_cube(Result_cube, Fits.yeval, mask)
        try:
            chi2_map[mask], BIC_map[mask] = Fits.chi2, Fits.BIC
        except:
            chi2_map[mask], BIC_map[mask] = 0,0

        for key in info_keys:
            if key=='params':
                for param in info[key]['extract']:
                    info[key][param] =  fill_cube(info[key][param], np.percentile(Fits.chains[param], (16,50,84)), mask) 
            else:
                if 'kin' not in key:
                    SNR= sp.SNR_calc(Cube.obs_wave, Fits.fluxs, Fits.error, Fits.props, 'general',\
                                        wv_cent = info[key]['wv'],\
                                        peak_name = key+'_peak', \
                                            fwhm_name = info[key]['fwhm'])
                    
                    info[key]['flux_map'][0,mask] = SNR
                    
                    if SNR>SNR_cut:
                        flux, p16,p84 = sp.flux_calc_mcmc(Fits, 'general', Cube.flux_norm,\
                                                            wv_cent = info[key]['wv'],\
                                                            peak_name = key+'_peak', \
                                                                fwhm_name = info[key]['fwhm'])
                        
                        info[key]['flux_map'][1,mask] = flux
                        info[key]['flux_map'][2,mask] = p16
                        info[key]['flux_map'][3,mask] = p84

                        if 'kin' in list(info[key]):
                            kins_par = sp.vel_kin_percentiles(Fits, peak_names=info[key]['kin']['peaks'], \
                                                                                fwhm_names=info[key]['kin']['fwhms'],\
                                                                                vel_names=info[key]['kin']['vels'],\
                                                                                rest_wave=info[key]['wv'],\
                                                                                N=100,z=Cube.z)
                    
                            info[key]['W80'] = fill_cube(info[key]['peak_vel'], kins_par['w80'], mask)
                            info[key]['peak_vel'] = fill_cube(info[key]['peak_vel'], kins_par['vel_peak'], mask)

                            info[key]['v10'] = fill_cube(info[key]['peak_vel'],  kins_par['v10'], mask)
                            info[key]['v90'] = fill_cube(info[key]['peak_vel'], kins_par['v90'], mask)
                            
                    else:
                        flux, p16,p84 = sp.flux_calc_mcmc(Fits, 'general', Cube.flux_norm,\
                                                            wv_cent = info[key]['wv'],\
                                                            peak_name = key+'_peak', \
                                                                fwhm_name = info[key]['fwhm'])
                        info[key]['flux_map'][2,mask] = p16


# =============================================================================
#             Plotting
# =============================================================================
        f = plt.figure( figsize=(20,6))

        ax = brokenaxes(xlims=brokenaxes_xlims,  hspace=.01)
        
        ax.plot(Fits.wave, Fits.fluxs.data, drawstyle='steps-mid')
        y= Fits.yeval
        ax.plot(Cube.obs_wave,  y, 'r--')
        
        ax.set_xlabel('wavelength (um)')
        ax.set_ylabel('Flux density')
        
        ax.set_ylim(-2*Fits.error[0], 1.2*max(y))
        ax.set_title('xy='+str(j)+' '+ str(i) )

        Spax.savefig()
        plt.close(f)

    print('Failed fits', failed_fits)
    Spax.close()

# =============================================================================
#         Plotting maps
# =============================================================================
    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=Cube.header)
    hdus = [primary_hdu]
    hdus.append(fits.ImageHDU(Result_cube_data, name='flux'))
    hdus.append(fits.ImageHDU(Result_cube_error, name='error'))
    hdus.append(fits.ImageHDU(Result_cube, name='yeval'))

    for key in info_keys:
        if key=='params':
            for param in info[key]['extract']:
                hdus.append(fits.ImageHDU(info[key][param], name=param))
            
        else: 
            hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))
            
            if 'kin' in list(info[key]):
                hdus.append(fits.ImageHDU(info[key]['peak_vel'], name=key+'_peakvel'))
                hdus.append(fits.ImageHDU(info[key]['W80'], name=key+'_W80'))
                hdus.append(fits.ImageHDU(info[key]['v10'], name=key+'_v10'))
                hdus.append(fits.ImageHDU(info[key]['v90'], name=key+'_v90'))


    hdus.append(fits.ImageHDU(chi2_map, name='chi2'))
    hdus.append(fits.ImageHDU(BIC_map, name='BIC'))
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(Cube.savepath+Cube.ID+'_general_fits_voronoi_maps'+add+'.fits', overwrite=True)

    return f