from os import path
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astropy import stats
from .. import Utils as sp
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