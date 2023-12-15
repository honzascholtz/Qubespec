# coding: utf-8
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from astropy.stats import biweight_scale

import sep

__all__ = ('source_detection',)



def __source_detection__(image, noise, mask=None, detection_threshold=3,
    deblend_cont=0.005, segmentation_map=True):
    """
    image  : 2-d array
    noise  : 2-d array
    """

    obj, seg = sep.extract(image, detection_threshold,
        err=noise, mask=mask, deblend_cont=deblend_cont,
        segmentation_map=segmentation_map)

    return obj, seg


def source_detection(flux, error, obs_wave, wave_range=(-np.inf, np.inf),
    noise_type='nominal', plot=True, **kwargs):

    """Use `sep` to detect sources in the datacube.

    flux:
    error:
    obs_wave
    wave_range : 2-elements [wavelength]
        use only the specified wavelength range for creating the detection image.
        The default uses all wavelength elements (`white-light image`).
    noise_type : str, optional
        If `uniform`, use the robust standard deviation of the background-subtracted
        image. If `nominal`, use the formal uncertainty on the detection image; the
        latter may be underestimated!
    kwargs : optional
        any valid keyword for `__source_detection__`.

    Return
    ------

    obj : structured array
        Each row contains data for an individual detection.
    seg : 2-d int array
        A segmentation map: 1 for spaxels belonging to the first source in `obj`,
        2 for the second source, etc.

    Note: the 1-d, 2-d and 3-d masks of the datacube are always applied. The
    principle is that these masks contain invalid entries.
    """

    mask1d = (obs_wave<wave_range[0]) | (obs_wave>wave_range[1])

    mask3d = flux.mask | mask1d[:, None, None] # Also apply 3-d DQ mask.
    spec_norm_factor = np.nanmedian(flux.data[~mask3d])
    spec_norm = flux / spec_norm_factor

    image = np.nanmedian(np.where(mask3d, np.nan, spec_norm), axis=0)

    # This is onyl a rough background for object detection, not the final background.
    mask2d = flux==np.nan
    mask2d = mask2d[100,:,:]
    rough_background = sep.Background(image, mask=mask2d)
    rough_background = rough_background.back()
    image -= rough_background

    if noise_type=='uniform':
        print('Uniform noise estimate for source detection')
        noise = biweight_scale(image[np.isfinite(image)])
        noise = np.full_like(image, noise)
    elif noise_type=='nominal':
        vars_norm = (error/spec_norm_factor)**2
        noise = np.nansum(np.where(mask3d, np.nan, vars_norm), axis=0) # Variance of sum.
        noise /= np.nansum(np.where(mask3d, 0, 1), axis=0) # Variance of mean
        noise = np.sqrt(noise * np.pi / 2.) # From variance of mean to of median.
    else:
        raise ValueError(f'{noise_type=} not supported')

    obj, seg = __source_detection__(image, noise, mask=mask2d, **kwargs)

    if plot:
        
        fig, (ax0, ax1, ax2,ax3) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 5))
        vmin, vmax = np.nanpercentile(image + rough_background, (1, 98))

        ax0.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap='Greys')
        ax1.imshow(rough_background, origin='lower', vmin=vmin, vmax=vmax, cmap='Greys')
        ax2.imshow(seg, origin='lower', cmap='nipy_spectral')

        ax3.imshow(image, origin='lower', cmap='nipy_spectral')
        ax3.imshow(seg, origin='lower', cmap='nipy_spectral', alpha=0.5)
        for _ax_ in (ax0, ax1, ax2,ax3):
            _ax_.plot(obj['xcpeak'], obj['ycpeak'], color='firebrick',
                marker='.', mec='none', alpha=0.7, ms=12, ls='none')
        
        

    return obj, seg
