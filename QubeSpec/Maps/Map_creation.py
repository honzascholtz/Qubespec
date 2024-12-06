from astropy.io import fits
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from brokenaxes import brokenaxes

from .. import Utils as sp
from .. import Plotting as emplot
from .. import Fitting as emfit

from ..Models import Halpha_OIII_models as HaO_models


def Map_creation_OIII(Cube,SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100],dbic=12, flux_max=0, width_upper=300,add='',):
    """ Function to post process fits. The function will load the fits results and determine which model is more likely,
        based on BIC. It will then calculate the W80 of the emission lines, V50 etc and create flux maps, velocity maps eyc.,
        Afterwards it saves all of it as .fits file. 

        Parameters
        ----------
    
        Cube : QubeSpec.Cube class instance
            Cube class from the main part of the QubeSpec. 

        SNR_cut : float
            SNR cutoff to detect emission lines 

        fwhmrange : list
            list of the two values to use as vmin and vmax in imshow of FWHM range

        velrange : list
            list of the two values to use as vmin and vmax in imshow of velocity range
        
        width_upper : float
            FWHM value used in the flux upper limit calculation.
        
        dbic : float
            delta bic to decide which model to use. 
        
        add : str
            additional string to use to load the results and save maps/pdf
            
        """
    z0 = Cube.z
    failed_fits=0
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_OIII'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================
    map_oiii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_outflow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_outflow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hb_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)


    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data

    Result_cube_narrow = np.zeros_like(Cube.flux.data)
    Result_cube_broad = np.zeros_like(Cube.flux.data)
    # =============================================================================
    #        Filling these maps
    # =============================================================================
    f,ax= plt.subplots(1)

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_OIII_fit_detection_only'+add+'.pdf')


    for row in tqdm.tqdm(range(len(results))):
        if len(results[row])==3:
            i,j, Fits= results[row]
            if str(type(Fits)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

        else:
            i,j, Fits_sig, Fits_out= results[row]

            if str(type(Fits_sig)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

            if (Fits_sig.BIC-Fits_out.BIC) >dbic:
                Fits = Fits_out
            else:
                Fits = Fits_sig
                
        if 'outflow_fwhm' in list(Fits.props.keys()):
            flag= 'outflow'
        else:
            flag='single'
            
        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval

        z = Fits.props['popt'][0]
        SNR = sp.SNR_calc(Fits.wave, Fits.fluxs, Fits.error, Fits.props, 'OIII')
        flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits, 'OIIIt', Cube.flux_norm)

        map_oiii[0,i,j]= SNR

        if SNR>SNR_cut:
            map_oiii[1,i,j] = flux_oiii.copy()
            map_oiii[2,i,j] = p16_oiii.copy()
            map_oiii[3,i,j] = p84_oiii.copy()


            kins_par = sp.W80_OIII_calc( Fits, z=Cube.z, N=100)

            map_oiii_w80[:,i,j] = kins_par['w80']
            map_oiii_v10[:,i,j] = kins_par['v10']
            map_oiii_v90[:,i,j] = kins_par['v90']
            map_oiii_v50[:,i,j] = kins_par['v50']
            map_oiii_vel[:,i,j] = kins_par['vel_peak']

            map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
            map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

            map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
            map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            Result_cube_narrow[:,i,j] = 0

            if flag=='outflow':
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])


            p = ax.get_ylim()[1]

            ax.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(kins_par['w80'][0],2)) )
        else:
            map_oiii[2,i,j] = p16_oiii.copy()
            map_oiii[3,i,j] = p84_oiii.copy()
            

        
        if SNR>SNR_cut:
            try:
                emplot.plotting_OIII(Fits, ax)
            except:
                print(Fits.props)
                break
            ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))
            plt.tight_layout()
            Spax.savefig()
            ax.clear()

    Spax.close()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x = int(Cube.center_data[1]); y= int(Cube.center_data[2])
    f = plt.figure( figsize=(10,10))

    IFU_header = Cube.header

    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600


    Offsets_low = -Cube.center_data[1:3][::-1]
    Offsets_hig = Cube.dim[0:2] - Cube.center_data[1:3][::-1]

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                        Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix

    ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
    ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
    ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
    ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])

    flx = ax1.imshow(map_oiii[1,:,:],vmax=map_oiii[1,y,x], origin='lower', extent= lim_sc)
    ax1.set_title('Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')

    #lims =
    #emplot.overide_axes_labels(f, axes[0,0], lims)


    vel = ax2.imshow(map_oiii_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
    ax2.set_title('v50')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')


    fw = ax3.imshow(map_oiii_w80[0,:,:],vmin=fwhmrange[0], vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('W80 map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    snr = ax4.imshow(map_oiii[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax4.set_title('SNR map')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')

    f.savefig(Cube.savepath+'Diagnostics/OIII_maps.pdf')

    hdr = Cube.header.copy()

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)

    hdu_data=fits.ImageHDU(Result_cube_data, name='flux')
    hdu_err = fits.ImageHDU(Result_cube_error, name='error')
    hdu_yeval = fits.ImageHDU(Result_cube, name='yeval')
    hdu_resid = fits.ImageHDU(Result_cube_data-Result_cube, name='residuals')


    oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
    oiii_w80 = fits.ImageHDU(map_oiii_w80, name='OIII_w80')
    oiii_v10 = fits.ImageHDU(map_oiii_v10, name='OIII_v10')
    oiii_v90 = fits.ImageHDU(map_oiii_v90, name='OIII_v90')
    oiii_v50 = fits.ImageHDU(map_oiii_v50, name='OIII_v50')
    oiii_vel = fits.ImageHDU(map_oiii_vel, name='OIII_vel')

    outflow_fwhm = fits.ImageHDU(map_outflow_fwhm, name='outflow_fwhm')
    outflow_vel = fits.ImageHDU(map_outflow_vel, name='outflow_vel')

    Nar_vel = fits.ImageHDU(map_narrow_vel, name='narrow_vel')
    Nar_fwhm = fits.ImageHDU(map_narrow_fwhm, name='narrow_fwhm')


    hdulist = fits.HDUList([primary_hdu,hdu_data, hdu_err, hdu_yeval,hdu_resid,\
                            oiii_hdu,oiii_w80, oiii_v10, oiii_v90, oiii_vel, oiii_v50 ,Nar_vel, Nar_fwhm, outflow_fwhm,outflow_vel])

    hdulist.writeto(Cube.savepath+Cube.ID+'_OIII_fits_maps'+add+'.fits', overwrite=True)

def Map_creation_Halpha(Cube, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100],dbic=10, flux_max=0, add=''):
    """ 
     Function to post process fits. The function will load the fits results and determine which model is more likely,
        based on BIC. It will then calculate the W80 of the emission lines, V50 etc and create flux maps, velocity maps eyc.,
        Afterwards it saves all of it as .fits file. 

        Parameters
        ----------
    
        Cube : QubeSpec.Cube class instance
            Cube class from the main part of the QubeSpec. 

        SNR_cut : float
            SNR cutoff to detect emission lines 

        fwhmrange : list
            list of the two values to use as vmin and vmax in imshow of FWHM range

        velrange : list
            list of the two values to use as vmin and vmax in imshow of velocity range
        
        width_upper : float
            FWHM value used in the flux upper limit calculation.
        
        dbic : float
            delta bic to decide which model to use. 
        
        add : str
            additional string to use to load the results and save maps/pdf
            
    """
    z0 = Cube.z

    wvo3 = 6563*(1+z0)/1e4
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_Halpha'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================
    map_hal = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)


    map_nii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siir = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siib = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_outflow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_outflow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hb_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_nii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data
    # =============================================================================
    #        Filling these maps
    # =============================================================================
    gf,ax= plt.subplots(1)

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_Halpha_fit_detection_only'+add+'.pdf')

    failed_fits = 0
    for row in tqdm.tqdm(range(len(results))):
        if len(results[row])==3:
            i,j, Fits= results[row]
            if str(type(Fits)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

        else:
            i,j, Fits_sig, Fits_out= results[row]

            if str(type(Fits_sig)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

            if (Fits_sig.BIC-Fits_out.BIC) >dbic:
                Fits = Fits_out
            else:
                Fits = Fits_sig
        
        if 'outflow_fwhm' in list(Fits.props.keys()):
            flag= 'outflow'
        else:
            flag='single'

        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval

        res_spx = Fits.props
        flx_spax_m = Fits.fluxs
        error = Fits.error
        z = res_spx['popt'][0]
        SNR = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn')
        map_hal[0,i,j] = SNR
        if SNR>SNR_cut:
            map_hal[0,i,j] = SNR
            map_hal[1:,i,j] = sp.flux_calc_mcmc(Fits, 'Hat',Cube.flux_norm)

            kins_par = sp.W80_Halpha_calc( Fits, z=Cube.z, N=100)

            map_hal_w80[:,i,j] = kins_par['w80']
            map_hal_v10[:,i,j] = kins_par['v10']
            map_hal_v90[:,i,j] = kins_par['v90']
            map_hal_v50[:,i,j] = kins_par['v50']
            map_hal_vel[:,i,j] = kins_par['vel_peak']

            map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
            map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

            map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
            map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            if flag=='outflow':
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

        else:
            flux_hal, p16_hal, p84_hal = sp.flux_calc_mcmc(Fits, 'Hat',Cube.flux_norm)
            map_hal[2,i,j] = p16_hal.copy()
            map_hal[3,i,j] = p84_hal.copy()


        SNR_n2 = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')
        map_nii[0,i,j] = SNR_n2
        if SNR_n2>SNR_cut:
            map_nii[1:,i,j] = sp.flux_calc_mcmc(Fits, 'NIIt', Cube.flux_norm)
        else:
            flux_nii, p16_nii, p84_nii = sp.flux_calc_mcmc(Fits, 'NIIt',Cube.flux_norm)
            map_nii[2,i,j] = p16_nii.copy()
            map_nii[3,i,j] = p84_nii.copy()

        emplot.plotting_Halpha(Fits, ax, errors=True)
        ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))

        if res_spx['Hal_peak'][0]<3*error[0]:
            ax.set_ylim(-error[0], 5*error[0])
        if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
            ax.set_ylim(-error[0], 5*error[0])
        Spax.savefig()
        ax.clear()
    plt.close(gf)
    Spax.close()

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x = int(Cube.center_data[1]); y= int(Cube.center_data[2])
    f = plt.figure( figsize=(10,10))

    IFU_header = Cube.header

    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600


    Offsets_low = -Cube.center_data[1:3][::-1]
    Offsets_hig = Cube.dim[0:2] - Cube.center_data[1:3][::-1]

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                        Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix

    ax1 = f.add_axes([0.1, 0.55, 0.38,0.38])
    ax2 = f.add_axes([0.1, 0.1, 0.38,0.38])
    ax3 = f.add_axes([0.55, 0.1, 0.38,0.38])
    ax4 = f.add_axes([0.55, 0.55, 0.38,0.38])

    if flux_max==0:
        flx_max = map_hal[1,y,x]
    else:
        flx_max = flux_max

    print(lim_sc)
    flx = ax1.imshow(map_hal[1,:,:],vmax=flx_max, origin='lower', extent= lim_sc)
    ax1.set_title('Halpha Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    cax.set_ylabel('Flux (arbitrary units)')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    #lims =
    #emplot.overide_axes_labels(f, axes[0,0], lims)


    vel = ax2.imshow(map_hal_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')


    fw = ax3.imshow(map_hal_w80[0,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    snr = ax4.imshow(map_hal[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax4.set_title('SNR map')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')

    cax.set_ylabel('SNR')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    fnii,axnii = plt.subplots(1)
    axnii.set_title('[NII] map')
    fw= axnii.imshow(map_nii[1,:,:], vmax=flx_max ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fnii.colorbar(fw, cax=cax, orientation='vertical')

    f.savefig(Cube.savepath+'Diagnostics/Halpha_maps.pdf')


    hdr = Cube.header.copy()

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)

    hdu_data=fits.ImageHDU(Result_cube_data, name='flux')
    hdu_err = fits.ImageHDU(Result_cube_error, name='error')
    hdu_yeval = fits.ImageHDU(Result_cube, name='yeval')
    hdu_resid = fits.ImageHDU(Result_cube_data-Result_cube, name='residuals')

    hal_hdu = fits.ImageHDU(map_hal, name='Hal')
    hal_w80 = fits.ImageHDU(map_hal_w80, name='Hal_w80')
    hal_v10 = fits.ImageHDU(map_hal_v10, name='Hal_v10')
    hal_v90 = fits.ImageHDU(map_hal_v90, name='Hal_v90')
    hal_v50 = fits.ImageHDU(map_hal_v50, name='Hal_v50')
    hal_vel = fits.ImageHDU(map_hal_vel, name='Hal_vel')

    nii_hdu = fits.ImageHDU(map_nii, name='Hal')

    hdulist = fits.HDUList([primary_hdu,hdu_data, hdu_err, hdu_yeval,hdu_resid,\
                            hal_hdu,hal_w80, hal_v10, hal_v90, hal_vel, hal_v50, nii_hdu ])

    hdulist.writeto(Cube.savepath+Cube.ID+'_Halpha_fits_maps'+add+'.fits', overwrite=True)

    return f


def Map_creation_Halpha_OIII(Cube, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100],dbic=10, flux_max=0, width_upper=300,add=''):
    """ Function to post process fits. The function will load the fits results and determine which model is more likely,
        based on BIC. It will then calculate the W80 of the emission lines, V50 etc and create flux maps, velocity maps etc.,
        Afterwards it saves all of it as .fits file. 

        Parameters
        ----------
    
        Cube : QubeSpec.Cube class instance
            Cube class from the main part of the QubeSpec. 

        SNR_cut : float
            SNR cutoff to detect emission lines 

        fwhmrange : list
            list of the two values to use as vmin and vmax in imshow of FWHM range

        velrange : list
            list of the two values to use as vmin and vmax in imshow of velocity range
        
        width_upper : float
            FWHM value used in the flux upper limit calculation.
        
        dbic : float
            delta bic to decide which model to use. 
        
        add : str
            additional string to use to load the results and save maps/pdf
            
    """
    z0 = Cube.z
    failed_fits=0
    wv_hal = 6564.52*(1+z0)/1e4
    wv_oiii = 5008.24*(1+z0)/1e4
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_Halpha_OIII'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================

    map_oiii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hb = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_nii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siir = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siib = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_outflow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_outflow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_nar = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hb_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_nii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    Result_cube_narrow = np.zeros_like(Cube.flux.data)
    Result_cube_broad = np.zeros_like(Cube.flux.data)

    from ..Models import Halpha_OIII_models as HO
    # =============================================================================
    #        Filling these maps
    # =============================================================================
    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_Halpha_OIII_fit_detection_only'+add+'.pdf')

    from ..Models import Halpha_OIII_models as HO_models
    for row in tqdm.tqdm(range(len(results))):
        
        if len(results[row])==3:
            i,j, Fits= results[row]
            if str(type(Fits)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

        else:
            i,j, Fits_sig, Fits_out= results[row]
            if str(type(Fits_sig)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

            if (Fits_sig.BIC-Fits_out.BIC) >dbic:
                Fits = Fits_out
                
            else:
                Fits = Fits_sig
                
        
        if 'outflow_fwhm' in list(Fits.props.keys()):
            flag= 'outflow'
        else:
            flag='single'


        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval

        z = Fits.props['popt'][0]
        res_spx = Fits.props
        chains = Fits.chains
        flx_spax_m = Fits.fluxs
        error = Fits.error
        lists= Fits.props.keys()
    

# =============================================================================
#             Halpha
# =============================================================================
        #print(sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn'))
        flux_hal, p16_hal,p84_hal = sp.flux_calc_mcmc(Fits, 'Hat', Cube.flux_norm)
        SNR_hal = flux_hal/p16_hal
        map_hal[0,i,j]= SNR_hal

        #SNR_hal = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn')
        #SNR_oiii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'OIII')
        #SNR_nii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')

        if SNR_hal>SNR_cut:
            map_hal[1,i,j] = flux_hal
            map_hal[2,i,j] = p16_hal
            map_hal[3,i,j] = p84_hal

            kins_par= sp.W80_Halpha_calc( Fits, z=Cube.z, N=100)

            map_hal_w80[:,i,j] = kins_par['w80']
            map_hal_v10[:,i,j] = kins_par['v10']
            map_hal_v90[:,i,j] = kins_par['v90']
            map_hal_v50[:,i,j] = kins_par['v50']
            map_hal_vel[:,i,j] = kins_par['vel_peak']

            map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
            map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

            map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
            map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])
            ptp = [Fits.props['z'][0], Fits.props['cont'][0], Fits.props['cont_grad'][0],\
                   Fits.props['Hal_peak'][0], Fits.props['NII_peak'][0], Fits.props['Nar_fwhm'][0],\
                     Fits.props['SIIr_peak'][0],Fits.props['SIIb_peak'][0], Fits.props['OIII_peak'][0],\
                         Fits.props['Hbeta_peak'][0] ]

            Result_cube_narrow[:,i,j] = HO.Halpha_OIII(Cube.obs_wave, *ptp)

            if flag=='outflow':
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])
                zout =  Fits.props['z'][0]+ Fits.props['outflow_vel'][0]/3e5*(1+Fits.props['z'][0])
                ptp = [zout, 0, 0,\
                   Fits.props['Hal_out_peak'][0], Fits.props['NII_out_peak'][0], Fits.props['outflow_fwhm'][0],\
                     0,0, Fits.props['OIII_out_peak'][0],\
                         Fits.props['Hbeta_out_peak'][0] ]

                Result_cube_broad[:,i,j] = HO.Halpha_OIII(Cube.obs_wave, *ptp)



        else:     
            map_hal[2,i,j] = p16_hal.copy()
            map_hal[3,i,j] = p84_hal.copy()

# =============================================================================
#             Plotting
# =============================================================================
        f = plt.figure(figsize=(10,4))
        baxes = brokenaxes(xlims=((4800,5050),(6500,6800)),  hspace=.01)
        emplot.plotting_Halpha_OIII(Fits,  baxes)

        #if res_spx['Hal_peak'][0]<3*error[0]:
        #    baxes.set_ylim(-error[0], 5*error[0])
        #if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
        #    baxes.set_ylim(-error[0], 5*error[0])

        SNRs = np.array([SNR_hal])

# =============================================================================
#             NII
# =============================================================================
        #SNR = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')
        flux_NII, p16_NII,p84_NII = sp.flux_calc_mcmc(Fits, 'NIIt', Cube.flux_norm)
        SNR_nii = flux_NII/p16_NII
        map_nii[0,i,j]= SNR_nii
        if SNR_nii>SNR_cut:
            map_nii[1,i,j] = flux_NII
            map_nii[2,i,j] = p16_NII
            map_nii[3,i,j] = p84_NII

        else:
            map_nii[2,i,j] = p16_NII.copy()
            map_nii[3,i,j] = p84_NII.copy()
# =============================================================================
#             OIII
# =============================================================================
        flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits, 'OIIIt', Cube.flux_norm)
        SNR_oiii = flux_oiii/p16_oiii
        map_oiii[0,i,j]= SNR_oiii

        if SNR_oiii>SNR_cut:
            map_oiii[1,i,j] = flux_oiii
            map_oiii[2,i,j] = p16_oiii
            map_oiii[3,i,j] = p84_oiii

            kins_par = sp.W80_OIII_calc( Fits, z=Cube.z, N=100)

            map_oiii_w80[:,i,j] = kins_par['w80']
            map_oiii_v10[:,i,j] = kins_par['v10']
            map_oiii_v90[:,i,j] = kins_par['v90']
            map_oiii_v50[:,i,j] = kins_par['v50']
            map_oiii_vel[:,i,j] = kins_par['vel_peak']

            flux_oiiin, p16_oiiin,p84_oiiin = sp.flux_calc_mcmc(Fits, 'OIIIn', Cube.flux_norm)
            map_oiii_nar[1,i,j],map_oiii_nar[2,i,j],map_oiii_nar[3,i,j] = flux_oiiin, p16_oiiin,p84_oiiin

            if (map_narrow_fwhm[0,i,j]==np.nan):
                map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
                map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
                map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            if (flag=='outflow') & (map_outflow_fwhm[0,i,j]==np.nan):
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])
            if (flag=='outflow'):
                flux_oiiiw, p16_oiiiw,p84_oiiiw = sp.flux_calc_mcmc(Fits, 'OIIIw', Cube.flux_norm)
                map_oiii_out[1,i,j],map_oiii_out[2,i,j],map_oiii_out[3,i,j] = flux_oiiiw, p16_oiiiw,p84_oiiiw

            p = baxes.get_ylim()[0][1]
            baxes.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(kins_par['w80'][0],2)) )
        else:
            map_oiii[2,i,j] = p16_oiii.copy()
            map_oiii[3,i,j] = p84_oiii.copy()

# =============================================================================
#             Hbeta
# =============================================================================
        flux_hb, p16_hb,p84_hb = sp.flux_calc_mcmc(Fits, 'Hbeta', Cube.flux_norm)
        SNR_hb = flux_hb/ p16_hb
        map_hb[0,i,j]= SNR_hb
        if SNR_hb>SNR_cut:
            map_hb[1,i,j] = flux_hb
            map_hb[2,i,j] = p16_hb
            map_hb[3,i,j] = p84_hb

        else:
            map_hb[2,i,j] = p16_hb.copy()
            map_hb[3,i,j] = p84_hb.copy()
# =============================================================================
#           SII
# =============================================================================
        fluxr, p16r,p84r = sp.flux_calc_mcmc(Fits, 'SIIr', Cube.flux_norm)
        fluxb, p16b,p84b = sp.flux_calc_mcmc(Fits, 'SIIb', Cube.flux_norm)

        SNR_SII = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'SII')

        if SNR_SII>SNR_cut:
            map_siir[0,i,j] = SNR_SII.copy()
            map_siib[0,i,j] = SNR_SII.copy()

            map_siir[1,i,j] = fluxr
            map_siir[2,i,j] = p16r
            map_siir[3,i,j] = p84r

            map_siib[1,i,j] = fluxb
            map_siib[2,i,j] = p16b
            map_siib[3,i,j] = p84b

        else:
            map_siir[2,i,j] = p16r
            map_siib[2,i,j] = p16b
            map_siir[3,i,j] = p84r
            map_siib[3,i,j] = p84b

        baxes.set_title('xy='+str(j)+' '+ str(i) + ', SNR = '+ str(np.round([SNR_hal, SNR_oiii, SNR_nii, SNR_SII],1)))
        baxes.set_xlabel('Restframe wavelength (ang)')
        baxes.set_ylabel(r'$10^{-16}$ ergs/s/cm2/mic')
        wv0 = 5008.24*(1+z0)
        wv0 = wv0/(1+z)
        baxes.vlines(wv0, 0,10, linestyle='dashed', color='k')
        Spax.savefig()
        plt.close(f)

    print('Failed fits', failed_fits)
    Spax.close()

# =============================================================================
#         Plotting maps
# =============================================================================
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    x = int(Cube.center_data[1]); y= int(Cube.center_data[2])
    IFU_header = Cube.header
    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600

    Offsets_low = -Cube.center_data[1:3][::-1]
    Offsets_hig = Cube.dim[0:2] - Cube.center_data[1:3][::-1]

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                        Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix

    if flux_max==0:
        flx_max = map_hal[1,y,x]
    else:
        flx_max = flux_max

    
    print(lim_sc)

# =============================================================================
#         Plotting Stuff
# =============================================================================
    f,axes = plt.subplots(6,3, figsize=(10,20))
    ax1 = axes[0,0]
    # =============================================================================
    # Halpha SNR
    snr = ax1.imshow(map_hal[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax1.set_title('Hal SNR map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha flux
    ax1 = axes[0,1]
    flx = ax1.imshow(map_hal[1,:,:],vmax=flx_max, origin='lower', extent= lim_sc)
    ax1.set_title('Halpha Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    cax.set_ylabel('Flux (arbitrary units)')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha  velocity
    ax2 = axes[0,2]
    vel = ax2.imshow(map_hal_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Hal Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha fwhm
    ax3 = axes[1,2]
    fw = ax3.imshow(map_hal_w80[0,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('Hal FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('W80 (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] SNR
    axes[1,0].set_title('[NII] SNR')
    fw= axes[1,0].imshow(map_nii[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,0].set_xlabel('RA offset (arcsecond)')
    axes[1,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] flux
    axes[1,1].set_title('[NII] map')
    fw= axes[1,1].imshow(map_nii[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,1].set_xlabel('RA offset (arcsecond)')
    axes[1,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta] SNR
    axes[2,0].set_title('Hbeta SNR')
    fw= axes[2,0].imshow(map_hb[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,0].set_xlabel('RA offset (arcsecond)')
    axes[2,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta flux
    axes[2,1].set_title('Hbeta map')
    fw= axes[2,1].imshow(map_hb[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,1].set_xlabel('RA offset (arcsecond)')
    axes[2,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] SNR
    axes[3,0].set_title('[OIII] SNR')
    fw= axes[3,0].imshow(map_oiii[0,:,:],vmin=3, vmax=20,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,0].set_xlabel('RA offset (arcsecond)')
    axes[3,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] flux
    axes[3,1].set_title('[OIII] map')
    fw= axes[3,1].imshow(map_oiii[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,1].set_xlabel('RA offset (arcsecond)')
    axes[3,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # OIII  velocity
    ax2 = axes[2,2]
    vel = ax2.imshow(map_oiii_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('OIII Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # OIII fwhm
    ax3 = axes[3,2]
    fw = ax3.imshow(map_oiii_w80[0,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('OIII W80 map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # SII SNR
    ax3 = axes[5,0]
    ax3.set_title('[SII] SNR')
    fw = ax3.imshow(map_siir[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # SII Ratio
    ax3 = axes[5,1]
    ax3.set_title('[SII]r/[SII]b')
    fw = ax3.imshow(map_siir[1,:,:]/map_siib[1,:,:] ,vmin=0.3, vmax=1.5, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    f.savefig(Cube.savepath+'Diagnostics/Halpha_OIII_maps.pdf')

    plt.tight_layout()

    Cube.map_hal = map_hal

    hdr = Cube.header.copy()

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)

    hdu_data=fits.ImageHDU(Result_cube_data, name='flux')
    hdu_err = fits.ImageHDU(Result_cube_error, name='error')
    hdu_yeval = fits.ImageHDU(Result_cube, name='yeval')
    hdu_resid = fits.ImageHDU(Result_cube_data-Result_cube, name='residuals')

    hdu_nar = fits.ImageHDU(Result_cube_narrow, name='yeval_nar')
    hdu_bro = fits.ImageHDU(Result_cube_broad, name='yeval_bro')

    hal_hdu = fits.ImageHDU(map_hal, name='Hal')
    hal_w80 = fits.ImageHDU(map_hal_w80, name='Hal_w80')
    hal_v10 = fits.ImageHDU(map_hal_v10, name='Hal_v10')
    hal_v90 = fits.ImageHDU(map_hal_v90, name='Hal_v90')
    hal_v50 = fits.ImageHDU(map_hal_v50, name='Hal_v50')
    hal_vel = fits.ImageHDU(map_hal_vel, name='Hal_vel')

    nii_hdu = fits.ImageHDU(map_nii, name='NII')

    oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
    oiii_w80 = fits.ImageHDU(map_oiii_w80, name='OIII_w80')
    oiii_v10 = fits.ImageHDU(map_oiii_v10, name='OIII_v10')
    oiii_v90 = fits.ImageHDU(map_oiii_v90, name='OIII_v90')
    oiii_v50 = fits.ImageHDU(map_oiii_v50, name='OIII_v50')
    oiii_vel = fits.ImageHDU(map_oiii_vel, name='OIII_vel')

    oiii_nar = fits.ImageHDU(map_oiii_nar, name='OIII_nar')
    oiii_out = fits.ImageHDU(map_oiii_out, name='OIII_out')

    outflow_fwhm = fits.ImageHDU(map_outflow_fwhm, name='outflow_fwhm')
    outflow_vel = fits.ImageHDU(map_outflow_vel, name='outflow_vel')

    Nar_vel = fits.ImageHDU(map_narrow_vel, name='narrow_vel')
    Nar_fwhm = fits.ImageHDU(map_narrow_fwhm, name='narrow_fwhm')

    hb_hdu = fits.ImageHDU(map_hb, name='Hbeta')

    hdulist = fits.HDUList([primary_hdu,hdu_data,hdu_err, hdu_yeval, hdu_nar, hdu_bro, hdu_resid,\
                            oiii_hdu,oiii_w80, oiii_v10, oiii_v90, oiii_vel, oiii_v50, oiii_nar, oiii_out,\
                            hal_hdu,hal_w80, hal_v10, hal_v90, hal_vel, hal_v50, nii_hdu, hb_hdu,Nar_vel, Nar_fwhm, outflow_fwhm,outflow_vel ])
    
   
    hdulist.writeto(Cube.savepath+Cube.ID+'_Halpha_OIII_fits_maps'+add+'.fits', overwrite=True)

    return f

def Map_creation_Halpha_OIII_SNR(Cube, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add=''):
    """ Function to post process fits. The function will load the fits results and determine which model is more likely,
        based on BIC. It will then calculate the W80 of the emission lines, V50 etc and create flux maps, velocity maps etc.,
        Afterwards it saves all of it as .fits file. 

        Parameters
        ----------
    
        Cube : QubeSpec.Cube class instance
            Cube class from the main part of the QubeSpec. 

        SNR_cut : float
            SNR cutoff to detect emission lines 

        fwhmrange : list
            list of the two values to use as vmin and vmax in imshow of FWHM range

        velrange : list
            list of the two values to use as vmin and vmax in imshow of velocity range
        
        width_upper : float
            FWHM value used in the flux upper limit calculation.
        
        dbic : float
            delta bic to decide which model to use. 
        
        add : str
            additional string to use to load the results and save maps/pdf
            
    """
    z0 = Cube.z
    failed_fits=0
    wv_hal = 6564.52*(1+z0)/1e4
    wv_oiii = 5008.24*(1+z0)/1e4

    from ..Models import Halpha_OIII_models as HO
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_Halpha_OIII'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================

    map_oiii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_oiii_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hal_w80 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v10 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v90 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_v50 = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_hb = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_nii = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siir = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_siib = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    map_outflow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_outflow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_fwhm = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)
    map_narrow_vel = np.full((3,Cube.dim[0], Cube.dim[1]), np.nan)

    map_oiii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hal_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_hb_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)
    map_nii_out = np.full((4,Cube.dim[0], Cube.dim[1]), np.nan)

    # =============================================================================
    #        Filling these maps
    # =============================================================================
    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data

    Result_cube_narrow = np.zeros_like(Cube.flux.data)
    Result_cube_broad = np.zeros_like(Cube.flux.data)

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_Halpha_OIII_fit_detection_only'+add+'.pdf')

    from ..Models import Halpha_OIII_models as HO_models
    for row in tqdm.tqdm(range(len(results))):
        
        if len(results[row])==3:
            i,j, Fits= results[row]
            if str(type(Fits)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

        else:
            i,j, Fits_sig, Fits_out= results[row]
            if str(type(Fits_sig)) != "<class 'QubeSpec.Fitting.fits_r.Fitting'>":
                failed_fits+=1
                continue

            Fits = Fits_out
        
        if 'outflow_fwhm' in list(Fits.props.keys()):
            flag= 'outflow'
        else:
            flag='single'


        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval

        z = Fits.props['popt'][0]
        res_spx = Fits.props
        chains = Fits.chains
        flx_spax_m = Fits.fluxs
        error = Fits.error
        lists= Fits.props.keys()
    

# =============================================================================
#             Halpha
# =============================================================================
        #print(sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn'))
        flux_hal, p16_hal,p84_hal = sp.flux_calc_mcmc(Fits, 'Hat', Cube.flux_norm)
        SNR_hal = flux_hal/p16_hal
        map_hal[0,i,j]= SNR_hal

        #SNR_hal = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn')
        #SNR_oiii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'OIII')
        #SNR_nii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')

        if SNR_hal>SNR_cut:
            map_hal[1,i,j] = flux_hal
            map_hal[2,i,j] = p16_hal
            map_hal[3,i,j] = p84_hal

            kins_par= sp.W80_Halpha_calc( Fits, z=Cube.z, N=100)

            map_hal_w80[:,i,j] = kins_par['w80']
            map_hal_v10[:,i,j] = kins_par['v10']
            map_hal_v90[:,i,j] = kins_par['v90']
            map_hal_v50[:,i,j] = kins_par['v50']
            map_hal_vel[:,i,j] = kins_par['vel_peak']

            map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
            map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

            map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
            map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            ptp = [Fits.props['z'][0], Fits.props['cont'][0], Fits.props['cont_grad'][0],\
                   Fits.props['Hal_peak'][0], Fits.props['NII_peak'][0], Fits.props['Nar_fwhm'][0],\
                     Fits.props['SIIr_peak'][0],Fits.props['SIIb_peak'][0], Fits.props['OIII_peak'][0],\
                         Fits.props['Hbeta_peak'][0] ]

            Result_cube_narrow[:,i,j] = HO.Halpha_OIII(Cube.obs_wave, *ptp)

            flux_brd = sp.flux_calc_mcmc(Fits, 'Haw', Cube.flux_norm)
            if flux_brd[0]/flux_brd[1]>3:
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])


        else:     
            map_hal[2,i,j] = p16_hal.copy()
            map_hal[3,i,j] = p84_hal.copy()

# =============================================================================
#             Plotting
# =============================================================================
        f = plt.figure(figsize=(10,4))
        baxes = brokenaxes(xlims=((4800,5050),(6500,6800)),  hspace=.01)
        emplot.plotting_Halpha_OIII(Fits,  baxes)

        #if res_spx['Hal_peak'][0]<3*error[0]:
        #    baxes.set_ylim(-error[0], 5*error[0])
        #if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
        #    baxes.set_ylim(-error[0], 5*error[0])

        SNRs = np.array([SNR_hal])

# =============================================================================
#             NII
# =============================================================================
        #SNR = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')
        flux_NII, p16_NII,p84_NII = sp.flux_calc_mcmc(Fits, 'NIIt', Cube.flux_norm)
        SNR_nii = flux_NII/p16_NII
        map_nii[0,i,j]= SNR_nii
        if SNR_nii>SNR_cut:
            map_nii[1,i,j] = flux_NII
            map_nii[2,i,j] = p16_NII
            map_nii[3,i,j] = p84_NII

        else:
            map_nii[2,i,j] = p16_NII.copy()
            map_nii[3,i,j] = p84_NII.copy()
# =============================================================================
#             OIII
# =============================================================================
        flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits, 'OIIIt', Cube.flux_norm)
        SNR_oiii = flux_oiii/p16_oiii
        map_oiii[0,i,j]= SNR_oiii

        if SNR_oiii>SNR_cut:
            map_oiii[1,i,j] = flux_oiii
            map_oiii[2,i,j] = p16_oiii
            map_oiii[3,i,j] = p84_oiii

            kins_par = sp.W80_OIII_calc( Fits, z=Cube.z, N=100)

            map_oiii_w80[:,i,j] = kins_par['w80']
            map_oiii_v10[:,i,j] = kins_par['v10']
            map_oiii_v90[:,i,j] = kins_par['v90']
            map_oiii_v50[:,i,j] = kins_par['v50']
            map_oiii_vel[:,i,j] = kins_par['vel_peak']

            if (map_narrow_fwhm[0,i,j]==np.nan):
                map_narrow_fwhm[:,i,j] = np.percentile(Fits.chains['Nar_fwhm'], (16,50,84) )
                map_narrow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_narrow_vel[:,i,j] = np.percentile((Fits.chains['z']-z0)/(1+z0)*3e5, (16,50,84) )
                map_narrow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            flux_brd = sp.flux_calc_mcmc(Fits, 'OIIIw', Cube.flux_norm)
            if (flux_brd[0]/flux_brd[1]>3) & (map_outflow_fwhm[0,i,j]==np.nan): #if (flag=='outflow') & (map_outflow_fwhm[0,i,j]==np.nan):
                map_outflow_fwhm[:,i,j] = np.percentile(Fits.chains['outflow_fwhm'], (16,50,84) )
                map_outflow_fwhm[1:,i,j] = abs(map_narrow_fwhm[1:,i,j]-map_narrow_fwhm[0,i,j])

                map_outflow_vel[:,i,j] = np.percentile(Fits.chains['outflow_vel'], (16,50,84) )
                map_outflow_vel[1:,i,j] = abs(map_narrow_vel[1:,i,j]-map_narrow_vel[0,i,j])

            p = baxes.get_ylim()[0][1]
            baxes.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(kins_par['w80'][0],2)) )
        else:
            map_oiii[2,i,j] = p16_oiii.copy()
            map_oiii[3,i,j] = p84_oiii.copy()

# =============================================================================
#             Hbeta
# =============================================================================
        flux_hb, p16_hb,p84_hb = sp.flux_calc_mcmc(Fits, 'Hbeta', Cube.flux_norm)
        SNR_hb = flux_hb/ p16_hb
        map_hb[0,i,j]= SNR_hb
        if SNR_hb>SNR_cut:
            map_hb[1,i,j] = flux_hb
            map_hb[2,i,j] = p16_hb
            map_hb[3,i,j] = p84_hb

        else:
            map_hb[2,i,j] = p16_hb.copy()
            map_hb[3,i,j] = p84_hb.copy()
# =============================================================================
#           SII
# =============================================================================
        fluxr, p16r,p84r = sp.flux_calc_mcmc(Fits, 'SIIr', Cube.flux_norm)
        fluxb, p16b,p84b = sp.flux_calc_mcmc(Fits, 'SIIb', Cube.flux_norm)

        SNR_SII = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'SII')

        if SNR_SII>SNR_cut:
            map_siir[0,i,j] = SNR_SII.copy()
            map_siib[0,i,j] = SNR_SII.copy()

            map_siir[1,i,j] = fluxr
            map_siir[2,i,j] = p16r
            map_siir[3,i,j] = p84r

            map_siib[1,i,j] = fluxb
            map_siib[2,i,j] = p16b
            map_siib[3,i,j] = p84b

        else:
            map_siir[2,i,j] = p16r
            map_siib[2,i,j] = p16b
            map_siir[3,i,j] = p84r
            map_siib[3,i,j] = p84b

        baxes.set_title('xy='+str(j)+' '+ str(i) + ', SNR = '+ str(np.round([SNR_hal, SNR_oiii, SNR_nii, SNR_SII],1)))
        baxes.set_xlabel('Restframe wavelength (ang)')
        baxes.set_ylabel(r'$10^{-16}$ ergs/s/cm2/mic')
        wv0 = 5008.24*(1+z0)
        wv0 = wv0/(1+z)
        baxes.vlines(wv0, 0,10, linestyle='dashed', color='k')
        Spax.savefig()
        plt.close(f)

    print('Failed fits', failed_fits)
    Spax.close()

# =============================================================================
#         Plotting maps
# =============================================================================
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    x = int(Cube.center_data[1]); y= int(Cube.center_data[2])
    IFU_header = Cube.header
    deg_per_pix = IFU_header['CDELT2']
    arc_per_pix = deg_per_pix*3600

    Offsets_low = -Cube.center_data[1:3][::-1]
    Offsets_hig = Cube.dim[0:2] - Cube.center_data[1:3][::-1]

    lim = np.array([ Offsets_low[0], Offsets_hig[0],
                        Offsets_low[1], Offsets_hig[1] ])

    lim_sc = lim*arc_per_pix

    if flux_max==0:
        flx_max = map_hal[1,y,x]
    else:
        flx_max = flux_max

    
    print(lim_sc)

# =============================================================================
#         Plotting Stuff
# =============================================================================
    f,axes = plt.subplots(6,3, figsize=(10,20))
    ax1 = axes[0,0]
    # =============================================================================
    # Halpha SNR
    snr = ax1.imshow(map_hal[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax1.set_title('Hal SNR map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha flux
    ax1 = axes[0,1]
    flx = ax1.imshow(map_hal[1,:,:],vmax=flx_max, origin='lower', extent= lim_sc)
    ax1.set_title('Halpha Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    cax.set_ylabel('Flux (arbitrary units)')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha  velocity
    ax2 = axes[0,2]
    vel = ax2.imshow(map_hal_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Hal Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Halpha fwhm
    ax3 = axes[1,2]
    fw = ax3.imshow(map_hal_w80[0,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('Hal FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('W80 (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] SNR
    axes[1,0].set_title('[NII] SNR')
    fw= axes[1,0].imshow(map_nii[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,0].set_xlabel('RA offset (arcsecond)')
    axes[1,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [NII] flux
    axes[1,1].set_title('[NII] map')
    fw= axes[1,1].imshow(map_nii[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[1,1].set_xlabel('RA offset (arcsecond)')
    axes[1,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta] SNR
    axes[2,0].set_title('Hbeta SNR')
    fw= axes[2,0].imshow(map_hb[0,:,:],vmin=3, vmax=10,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,0].set_xlabel('RA offset (arcsecond)')
    axes[2,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # Hbeta flux
    axes[2,1].set_title('Hbeta map')
    fw= axes[2,1].imshow(map_hb[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[2,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[2,1].set_xlabel('RA offset (arcsecond)')
    axes[2,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] SNR
    axes[3,0].set_title('[OIII] SNR')
    fw= axes[3,0].imshow(map_oiii[0,:,:],vmin=3, vmax=20,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,0].set_xlabel('RA offset (arcsecond)')
    axes[3,0].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # [OIII] flux
    axes[3,1].set_title('[OIII] map')
    fw= axes[3,1].imshow(map_oiii[1,:,:] ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(axes[3,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')
    axes[3,1].set_xlabel('RA offset (arcsecond)')
    axes[3,1].set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # OIII  velocity
    ax2 = axes[2,2]
    vel = ax2.imshow(map_oiii_vel[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('OIII Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # OIII fwhm
    ax3 = axes[3,2]
    fw = ax3.imshow(map_oiii_w80[0,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('OIII W80 map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    # =============================================================================
    # SII SNR
    ax3 = axes[5,0]
    ax3.set_title('[SII] SNR')
    fw = ax3.imshow(map_siir[0,:,:],vmin=3, vmax=10, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    # =============================================================================
    # SII Ratio
    ax3 = axes[5,1]
    ax3.set_title('[SII]r/[SII]b')
    fw = ax3.imshow(map_siir[1,:,:]/map_siib[1,:,:] ,vmin=0.3, vmax=1.5, origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    f.savefig(Cube.savepath+'Diagnostics/Halpha_OIII_maps.pdf')



    plt.tight_layout()

    Cube.map_hal = map_hal

    hdr = Cube.header.copy()

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)

    hdu_data=fits.ImageHDU(Result_cube_data, name='flux')
    hdu_err = fits.ImageHDU(Result_cube_error, name='error')
    hdu_yeval = fits.ImageHDU(Result_cube, name='yeval')
    hdu_resid = fits.ImageHDU(Result_cube_data-Result_cube, name='residuals')
    hdu_nar = fits.ImageHDU(Result_cube_narrow, name='yeval_nar')
    hdu_bro = fits.ImageHDU(Result_cube_broad, name='yeval_bro')
    hal_hdu = fits.ImageHDU(map_hal, name='Hal')
    hal_w80 = fits.ImageHDU(map_hal_w80, name='Hal_w80')
    hal_v10 = fits.ImageHDU(map_hal_v10, name='Hal_v10')
    hal_v90 = fits.ImageHDU(map_hal_v90, name='Hal_v90')
    hal_v50 = fits.ImageHDU(map_hal_v50, name='Hal_v50')
    hal_vel = fits.ImageHDU(map_hal_vel, name='Hal_vel')

    nii_hdu = fits.ImageHDU(map_nii, name='NII')

    oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
    oiii_w80 = fits.ImageHDU(map_oiii_w80, name='OIII_w80')
    oiii_v10 = fits.ImageHDU(map_oiii_v10, name='OIII_v10')
    oiii_v90 = fits.ImageHDU(map_oiii_v90, name='OIII_v90')
    oiii_v50 = fits.ImageHDU(map_oiii_v50, name='OIII_v50')
    oiii_vel = fits.ImageHDU(map_oiii_vel, name='OIII_vel')

    outflow_fwhm = fits.ImageHDU(map_outflow_fwhm, name='outflow_fwhm')
    outflow_vel = fits.ImageHDU(map_outflow_vel, name='outflow_vel')

    Nar_vel = fits.ImageHDU(map_narrow_vel, name='narrow_vel')
    Nar_fwhm = fits.ImageHDU(map_narrow_fwhm, name='narrow_fwhm')

    hb_hdu = fits.ImageHDU(map_hb, name='Hbeta')

    hdulist = fits.HDUList([primary_hdu,hdu_data,hdu_err, hdu_yeval, hdu_resid,\
                            oiii_hdu,oiii_w80, oiii_v10, oiii_v90, oiii_vel, oiii_v50,\
                            hal_hdu,hal_w80, hal_v10, hal_v90, hal_vel, hal_v50, nii_hdu, hb_hdu,Nar_vel, Nar_fwhm, outflow_fwhm,outflow_vel ])
    
   
    hdulist.writeto(Cube.savepath+Cube.ID+'_Halpha_OIII_fits_maps_snr'+add+'.fits', overwrite=True)

    return f

def Map_creation_general(Cube,info, SNR_cut = 3 , width_upper=300,add='',\
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
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_general'+add+'.txt', "rb") as fp:
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

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_general_fit_detection_only'+add+'.pdf')

    for row in tqdm.tqdm(range(len(results))):

        try:
            i,j, Fits = results[row]
        except:
            ls=0
        if str(type(Fits)) == "<class 'dict'>":
            failed_fits+=1
            continue

        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval
        try:
            chi2_map[i,j], BIC_map[i,j] = Fits.chi2, Fits.BIC
        except:
            chi2_map[i,j], BIC_map[i,j] = 0,0

        for key in info_keys:
            if key=='params':
                for param in info[key]['extract']:
                    info[key][param][:,i,j] = np.percentile(Fits.chains[param], (16,50,84))
            
            else:
                if 'kin' not in key:
                    SNR= sp.SNR_calc(Cube.obs_wave, Fits.fluxs, Fits.error, Fits.props, 'general',\
                                        wv_cent = info[key]['wv'],\
                                        peak_name = key+'_peak', \
                                            fwhm_name = info[key]['fwhm'])
                    
                    info[key]['flux_map'][0,i,j] = SNR
                    
                    if SNR>SNR_cut:
                        flux, p16,p84 = sp.flux_calc_mcmc(Fits, 'general', Cube.flux_norm,\
                                                            wv_cent = info[key]['wv'],\
                                                            peak_name = key+'_peak', \
                                                                fwhm_name = info[key]['fwhm'])
                        
                        info[key]['flux_map'][1,i,j] = flux
                        info[key]['flux_map'][2,i,j] = p16
                        info[key]['flux_map'][3,i,j] = p84

                        if 'kin' in list(info[key]):
                            kins_par = sp.vel_kin_percentiles(Fits, peak_names=info[key]['kin']['peaks'], \
                                                                                fwhm_names=info[key]['kin']['fwhms'],\
                                                                                vel_names=info[key]['kin']['vels'],\
                                                                                rest_wave=info[key]['wv'],\
                                                                                N=100,z=Cube.z)
                    
                            info[key]['W80'][:,i,j] = kins_par['w80']
                            info[key]['peak_vel'][:,i,j] = kins_par['vel_peak']

                            info[key]['v10'][:,i,j] = kins_par['v10']
                            info[key]['v90'][:,i,j] = kins_par['v90']
                            
                    else:
                        flux, p16,p84 = sp.flux_calc_mcmc(Fits, 'general', Cube.flux_norm,\
                                                            wv_cent = info[key]['wv'],\
                                                            peak_name = key+'_peak', \
                                                                fwhm_name = info[key]['fwhm'])
                        info[key]['flux_map'][2,i,j] = p16


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
    hdus.append(fits.ImageHDU(Result_cube_data-Result_cube, name='residuals'))


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
    hdulist.writeto(Cube.savepath+Cube.ID+'_general_fits_maps'+add+'.fits', overwrite=True)

    return f

def Map_creation_ppxf(Cube, info, add=''):
    flux_table = Table.read(Cube.savepath+'PRISM_spaxel/spaxel_R100_ppxf_emlines.fits')
    info_keys = list(info.keys())
    for key in info_keys:
        map_flx = np.zeros((2,Cube.dim[0], Cube.dim[1]))
        map_flx[:,:,:] = np.nan
        
        for k, row in tqdm.tqdm(enumerate(flux_table)):
            ID = str(row['ID'])
            i,j = int(ID[:2]),int(ID[2:])
            map_flx[0,i,j] = (row[key+'_flux'] if row[key+'_flux']>row[key+'_flux_upper'] else np.nan)
            map_flx[0,i,j] = (row[key+'_flux_upper']/3 if row[key+'_flux']>row[key+'_flux_upper'] else np.nan)
        
        info[key]['flux_map'] = map_flx
    
    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=Cube.header)
    hdus = [primary_hdu]
    for key in info_keys:
        hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))
    

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(Cube.savepath+Cube.ID+'_ppxf_fits_maps'+add+'.fits', overwrite=True)