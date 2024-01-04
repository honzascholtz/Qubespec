from astropy.io import fits
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from brokenaxes import brokenaxes

from . import Support as sp
from . import Plotting as emplot
from . import Fitting as emfit

from .Models import Halpha_OIII_models as HaO_models


def Map_creation_OIII(Cube,SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='',):
        z0 = Cube.z
        failed_fits=0
        wvo3 = 5008.24*(1+z0)/1e4
        # =============================================================================
        #         Importing all the data necessary to post process
        # =============================================================================
        try:
            with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_OIII_2G'+add+'.txt', "rb") as fp:
                results= pickle.load(fp)
        except:
            with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw_OIII'+add+'.txt', "rb") as fp:
                results= pickle.load(fp)

        # =============================================================================
        #         Setting up the maps
        # =============================================================================
        map_oiii = np.zeros((4,Cube.dim[0], Cube.dim[1]))
        map_oiii[:,:,:] = np.nan

        map_oiii_ki = np.zeros((5,Cube.dim[0], Cube.dim[1]))
        map_oiii_ki[:,:,:] = np.nan
        # =============================================================================
        #        Filling these maps
        # =============================================================================
        f,ax= plt.subplots(1)
        from . import Plotting_tools_v2 as emplot

        Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_OIII_fit_detection_only.pdf')


        for row in tqdm.tqdm(range(len(results))):
            try:
                i,j, Fits = results[row]
            except:
                print('Loading old fits? I am sorry no longer compatible...')

            if str(type(Fits)) == "<class 'dict'>":
                failed_fits+=1
                continue

            fitted_model = Fits.fitted_model
                
           
            
            z = Fits.props['popt'][0]
            SNR = sp.SNR_calc(Fits.wave, Fits.fluxs, Fits.error, Fits.props, 'OIII')
            flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits, 'OIIIt', Cube.flux_norm)

            map_oiii[0,i,j]= SNR

            if SNR>SNR_cut:
                map_oiii[1,i,j] = flux_oiii.copy()
                map_oiii[2,i,j] = p16_oiii.copy()
                map_oiii[3,i,j] = p84_oiii.copy()


                map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(fitted_model, Fits.props, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]

                p = ax.get_ylim()[1]

                ax.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(map_oiii_ki[1,i,j],2)) )
            else:


                dl = Cube.obs_wave[1]-Cube.obs_wave[0]
                n = width_upper/3e5*(5008.24*(1+Cube.z)/1e4)/dl
                map_oiii[3,i,j] = SNR_cut*Fits.error[1]*dl*np.sqrt(n)
                

            
            if SNR>SNR_cut:
                try:
                    emplot.plotting_OIII(Cube.obs_wave, Fits.fluxs, ax, Fits.props, Fits.fitted_model)
                except:
                    print(Fits.props, Fits.fitted_model)
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


        vel = ax2.imshow(map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0], vmax=velrange[1], extent= lim_sc)
        ax2.set_title('v50')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(vel, cax=cax, orientation='vertical')


        fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0], vmax=fwhmrange[1], origin='lower', extent= lim_sc)
        ax3.set_title('W80 map')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(fw, cax=cax, orientation='vertical')

        snr = ax4.imshow(map_oiii[0,:,:],vmin=3, vmax=20, origin='lower', extent= lim_sc)
        ax4.set_title('SNR map')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(snr, cax=cax, orientation='vertical')

        hdr = Cube.header.copy()
        hdr['X_cent'] = x
        hdr['Y_cent'] = y



        primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=hdr)
        
        oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')
        oiii_kin_hdu = fits.ImageHDU(map_oiii_ki, name='OIII_kin')

        hdulist = fits.HDUList([primary_hdu,oiii_hdu,oiii_kin_hdu ])

        hdulist.writeto(Cube.savepath+Cube.ID+'_OIII_fits_maps_2G.fits', overwrite=True)

def Map_creation_Halpha(Cube, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, add=''):
    z0 = Cube.z

    wvo3 = 6563*(1+z0)/1e4
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_spaxel_fit_raw'+add+'.txt', "rb") as fp:
        results= pickle.load(fp)

    with open(Cube.savepath+Cube.ID+'_'+Cube.band+'_Unwrapped_cube'+add+'.txt', "rb") as fp:
        Unwrapped_cube= pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================
    map_vel = np.zeros(Cube.dim[:2])
    map_vel[:,:] = np.nan

    map_fwhm = np.zeros(Cube.dim[:2])
    map_fwhm[:,:] = np.nan

    map_flux = np.zeros(Cube.dim[:2])
    map_flux[:,:] = np.nan

    map_snr = np.zeros(Cube.dim[:2])
    map_snr[:,:] = np.nan

    map_nii = np.zeros(Cube.dim[:2])
    map_nii[:,:] = np.nan

    # =============================================================================
    #        Filling these maps
    # =============================================================================
    gf,ax= plt.subplots(1)
    from . import Plotting_tools_v2 as emplot

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_Halpha_fit_detection_only.pdf')


    for row in range(len(results)):

        i,j, res_spx = results[row]
        i,j, flx_spax_m, error,wave,z = Unwrapped_cube[row]

        z = res_spx['popt'][0]
        SNR = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn')
        map_snr[i,j]= SNR
        if SNR>SNR_cut:

            map_vel[i,j] = ((6563*(1+z)/1e4)-wvo3)/wvo3*3e5
            map_fwhm[i,j] = res_spx['popt'][5]
            map_flux[i,j] = sp.flux_calc(res_spx, 'Hat',Cube.flux_norm)
            map_nii[i,j] = sp.flux_calc(res_spx, 'NIIt', Cube.flux_norm)


        emplot.plotting_Halpha(Cube.obs_wave, flx_spax_m, ax, res_spx, emfit.H_models.Halpha, error=error)
        ax.set_title('x = '+str(j)+', y='+ str(i) + ', SNR = ' +str(np.round(SNR,2)))

        if res_spx['Hal_peak'][0]<3*error[0]:
            ax.set_ylim(-error[0], 5*error[0])
        if (res_spx['SIIr_peak'][0]>res_spx['Hal_peak'][0]) & (res_spx['SIIb_peak'][0]>res_spx['Hal_peak'][0]):
            ax.set_ylim(-error[0], 5*error[0])
        Spax.savefig()
        ax.clear()
    plt.close(gf)
    Spax.close()

    Cube.Flux_map = map_flux
    Cube.Vel_map = map_vel
    Cube.FWHM_map = map_fwhm
    Cube.SNR_map = map_snr

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
        flx_max = map_flux[y,x]
    else:
        flx_max = flux_max

    print(lim_sc)
    flx = ax1.imshow(map_flux,vmax=flx_max, origin='lower', extent= lim_sc)
    ax1.set_title('Halpha Flux map')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(flx, cax=cax, orientation='vertical')
    cax.set_ylabel('Flux (arbitrary units)')
    ax1.set_xlabel('RA offset (arcsecond)')
    ax1.set_ylabel('Dec offset (arcsecond)')

    #lims =
    #emplot.overide_axes_labels(f, axes[0,0], lims)


    vel = ax2.imshow(map_vel, cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
    ax2.set_title('Velocity offset map')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(vel, cax=cax, orientation='vertical')

    cax.set_ylabel('Velocity (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')


    fw = ax3.imshow(map_fwhm,vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    snr = ax4.imshow(map_snr,vmin=3, vmax=20, origin='lower', extent= lim_sc)
    ax4.set_title('SNR map')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(snr, cax=cax, orientation='vertical')

    cax.set_ylabel('SNR')
    ax2.set_xlabel('RA offset (arcsecond)')
    ax2.set_ylabel('Dec offset (arcsecond)')

    fnii,axnii = plt.subplots(1)
    axnii.set_title('[NII] map')
    fw= axnii.imshow(map_nii, vmax=flx_max ,origin='lower', extent= lim_sc)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fnii.colorbar(fw, cax=cax, orientation='vertical')

    hdr = Cube.header.copy()
    hdr['X_cent'] = x
    hdr['Y_cent'] = y

    Line_info = np.zeros((5,Cube.dim[0],Cube.dim[1]))
    Line_info[0,:,:] = map_flux
    Line_info[1,:,:] = map_vel
    Line_info[2,:,:] = map_fwhm
    Line_info[3,:,:] = map_snr
    Line_info[4,:,:] = map_nii

    prhdr = hdr
    hdu = fits.PrimaryHDU(Line_info, header=prhdr)
    hdulist = fits.HDUList([hdu])


    hdulist.writeto(Cube.savepath+Cube.ID+'_Halpha_fits_maps.fits', overwrite=True)

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


def Map_creation_Halpha_OIII(Cube, SNR_cut = 3 , fwhmrange = [100,500], velrange=[-100,100], flux_max=0, width_upper=300,add='', dbic=10):
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

    map_hal = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_hal[:,:,:] = np.nan

    map_nii = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_nii[:,:,:] = np.nan

    map_hb = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_hb[:,:,:] = np.nan

    map_oiii = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_oiii[:,:,:] = np.nan

    map_siir = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_siir[:,:,:] = np.nan

    map_siib = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_siib[:,:,:] = np.nan

    map_hal_ki = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_hal_ki[:,:,:] = np.nan

    map_nii_ki = np.zeros((4,Cube.dim[0], Cube.dim[1]))
    map_nii_ki[:,:,:] = np.nan

    map_oiii_ki = np.zeros((5,Cube.dim[0], Cube.dim[1]))
    map_oiii_ki[:,:,:] = np.nan
    # =============================================================================
    #        Filling these maps
    # =============================================================================
    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data

    from . import Plotting as emplot

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_Halpha_OIII_fit_detection_only.pdf')

    from .Models import Halpha_OIII_models as HO_models
    for row in tqdm.tqdm(range(len(results))):

        try:
            i,j, res_spx,chains,wave,flx_spax_m,error = results[row]

        except:
            if len(results[row])==3:
                i,j, Fits= results[row]
                if str(type(Fits)) != "<class 'QubeSpec.Fitting.Fitting'>":
                    failed_fits+=1
                    continue

            else:
                i,j, Fits_sig, Fits_out= results[row]

                if str(type(Fits_sig)) != "<class 'QubeSpec.Fitting.Fitting'>":
                    failed_fits+=1
                    continue

                if (Fits_sig.BIC-Fits_out.BIC) >10:
                    Fits = Fits_out
                else:
                    Fits = Fits_sig

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
    
        from .Models import Halpha_OIII_models as HO_models
        if 'zBLR' in lists:
            modelfce = HO_models.Halpha_OIII_BLR
        elif 'outflow_vel' not in lists:
            modelfce = HO_models.Halpha_OIII
        elif 'outflow_vel' in lists and 'zBLR' not in lists:
            modelfce = HO_models.Halpha_OIII_outflow

# =============================================================================
#             Halpha
# =============================================================================

        flux_hal, p16_hal,p84_hal = sp.flux_calc_mcmc(Fits, 'Hat', Cube.flux_norm)
        SNR_hal = flux_hal/p16_hal
        map_hal[0,i,j]= SNR_hal

        SNR_hal = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hn')
        SNR_oiii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'OIII')
        SNR_nii = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'NII')
        SNR_hb = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'Hb')

        if SNR_hal>SNR_cut:
            map_hal[1,i,j] = flux_hal.copy()
            map_hal[2,i,j] = p16_hal.copy()
            map_hal[3,i,j] = p84_hal.copy()

            if 'Hal_out_peak' in list(res_spx.keys()):
                map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = sp.W80_Halpha_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]

            else:
                map_hal_ki[2,i,j], map_hal_ki[3,i,j],map_hal_ki[1,i,j],map_hal_ki[0,i,j] = sp.W80_Halpha_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]


            dl = Cube.obs_wave[1]-Cube.obs_wave[0]
            n = width_upper/3e5*(6564.52**(1+Cube.z)/1e4)/dl
            map_hal[3,i,j] = -SNR_cut*error[-1]*dl*np.sqrt(n)




# =============================================================================
#             Plotting
# =============================================================================
        f = plt.figure(figsize=(10,4))
        baxes = brokenaxes(xlims=((4800,5050),(6500,6800)),  hspace=.01)
        emplot.plotting_Halpha_OIII(Cube.obs_wave, flx_spax_m, baxes, res_spx, modelfce)

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

        map_nii[0,i,j]= SNR_nii
        if SNR_nii>SNR_cut:
            map_nii[1,i,j] = flux_NII.copy()
            map_nii[2,i,j] = p16_NII.copy()
            map_nii[3,i,j] = p84_NII.copy()

            if 'NII_out_peak' in list(res_spx.keys()):
                map_nii_ki[2,i,j], map_nii_ki[3,i,j],map_nii_ki[1,i,j],map_nii_ki[0,i,j], = sp.W80_NII_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]

            else:
                map_nii_ki[2,i,j], map_nii_ki[3,i,j],map_nii_ki[1,i,j],map_nii_ki[0,i,j], = sp.W80_NII_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]

        else:
            dl = Cube.obs_wave[1]-Cube.obs_wave[0]
            n = width_upper/3e5*(6564.52**(1+Cube.z)/1e4)/dl
            map_nii[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
# =============================================================================
#             OIII
# =============================================================================
        flux_oiii, p16_oiii,p84_oiii = sp.flux_calc_mcmc(Fits, 'OIIIt', Cube.flux_norm)

        map_oiii[0,i,j]= SNR_oiii

        if SNR_oiii>SNR_cut:
            map_oiii[1,i,j] = flux_oiii.copy()
            map_oiii[2,i,j] = p16_oiii.copy()
            map_oiii[3,i,j] = p84_oiii.copy()


            if 'OIII_out_peak' in list(res_spx.keys()):
                map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]

            else:
                map_oiii_ki[0,i,j] = ((5008.24*(1+z)/1e4)-wv_oiii)/wv_oiii*3e5
                map_oiii_ki[1,i,j] = res_spx['Nar_fwhm'][0]
                map_oiii_ki[2,i,j], map_oiii_ki[3,i,j],map_oiii_ki[1,i,j],map_oiii_ki[0,i,j], = sp.W80_OIII_calc_single(modelfce, res_spx, 0, z=Cube.z)#res_spx['Nar_fwhm'][0]
            p = baxes.get_ylim()[1][1]

            baxes.text(4810, p*0.9 , 'OIII W80 = '+str(np.round(map_oiii_ki[1,i,j],2)) )
        else:


            dl = Cube.obs_wave[1]-Cube.obs_wave[0]
            n = width_upper/3e5*(5008.24*(1+Cube.z)/1e4)/dl
            map_oiii[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)

# =============================================================================
#             Hbeta
# =============================================================================
        flux_hb, p16_hb,p84_hb = sp.flux_calc_mcmc(Fits, 'Hbeta', Cube.flux_norm)

        map_hb[0,i,j]= SNR_hb.copy()
        if SNR_hb>SNR_cut:
            map_hb[1,i,j] = flux_hb.copy()
            map_hb[2,i,j] = p16_hb.copy()
            map_hb[3,i,j] = p84_hb.copy()

        else:

            dl = Cube.obs_wave[1]-Cube.obs_wave[0]
            n = width_upper/3e5*(4860*(1+Cube.z)/1e4)/dl
            map_hb[3,i,j] = SNR_cut*error[1]*dl*np.sqrt(n)

# =============================================================================
#           SII
# =============================================================================
        fluxr, p16r,p84r = sp.flux_calc_mcmc(Fits, 'SIIr', Cube.flux_norm)
        fluxb, p16b,p84b = sp.flux_calc_mcmc(Fits, 'SIIb', Cube.flux_norm)

        SNR_SII = sp.SNR_calc(Cube.obs_wave, flx_spax_m, error, res_spx, 'SII')

        if SNR_SII>SNR_cut:
            map_siir[0,i,j] = SNR_SII.copy()
            map_siib[0,i,j] = SNR_SII.copy()

            map_siir[1,i,j] = fluxr.copy()
            map_siir[2,i,j] = p16r.copy()
            map_siir[3,i,j] = p84r.copy()

            map_siib[1,i,j] = fluxb.copy()
            map_siib[2,i,j] = p16b.copy()
            map_siib[3,i,j] = p84b.copy()

        else:

            dl = Cube.obs_wave[1]-Cube.obs_wave[0]
            n = width_upper/3e5*(6731*(1+Cube.z)/1e4)/dl
            map_siir[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)
            map_siib[3,i,j] = SNR_cut*error[-1]*dl*np.sqrt(n)




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
    vel = ax2.imshow(map_hal_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
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
    fw = ax3.imshow(map_hal_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('Hal FWHM map')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(fw, cax=cax, orientation='vertical')

    cax.set_ylabel('FWHM (km/s)')
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
    vel = ax2.imshow(map_oiii_ki[0,:,:], cmap='coolwarm', origin='lower', vmin=velrange[0],vmax=velrange[1], extent= lim_sc)
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
    fw = ax3.imshow(map_oiii_ki[1,:,:],vmin=fwhmrange[0],vmax=fwhmrange[1], origin='lower', extent= lim_sc)
    ax3.set_title('OIII FWHM map')
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


    plt.tight_layout()

    Cube.map_hal = map_hal

    hdr = Cube.header.copy()
    hdr['X_cent'] = x
    hdr['Y_cent'] = y

    primary_hdu = fits.PrimaryHDU(np.zeros((3,3,3)), header=Cube.header)
    hdu_data=fits.ImageHDU(Result_cube_data, name='flux')
    hdu_err = fits.ImageHDU(Result_cube_error, name='error')
    hdu_yeval = fits.ImageHDU(Result_cube, name='yeval')

    hal_hdu = fits.ImageHDU(map_hal, name='Halpha')
    nii_hdu = fits.ImageHDU(map_nii, name='NII')
    nii_kin_hdu = fits.ImageHDU(map_nii_ki, name='NII_kin')
    hbe_hdu = fits.ImageHDU(map_hb, name='Hbeta')
    oiii_hdu = fits.ImageHDU(map_oiii, name='OIII')

    siir_hdu = fits.ImageHDU(map_siir, name='SIIr')
    siib_hdu = fits.ImageHDU(map_siib, name='SIIb')

    hal_kin_hdu = fits.ImageHDU(map_hal_ki, name='Hal_kin')
    oiii_kin_hdu = fits.ImageHDU(map_oiii_ki, name='OIII_kin')

    hdulist = fits.HDUList([primary_hdu, hdu_data, hdu_err, hdu_yeval, hal_hdu, nii_hdu, nii_kin_hdu, hbe_hdu, oiii_hdu,hal_kin_hdu,siir_hdu,oiii_kin_hdu, siib_hdu ])
    hdulist.writeto(Cube.savepath+Cube.ID+'_Halpha_OIII_fits_maps'+add+'.fits', overwrite=True)

    return f

def Map_creation_general_comparison(Cube,info,path1, path2, SNR_cut = 3 ,deltabic=10, add='',\
                            brokenaxes_xlims= ((2.820,3.45),(3.75,4.05),(5,5.3)) ):
    z0 = Cube.z
    failed_fits=0
    
    # =============================================================================
    #         Importing all the data necessary to post process
    # =============================================================================
    with open(path1, "rb") as fp:
        results1= pickle.load(fp)
    
    with open(path2, "rb") as fp:
        results2 = pickle.load(fp)

    # =============================================================================
    #         Setting up the maps
    # =============================================================================

    Result_cube = np.zeros_like(Cube.flux.data)
    Result_cube_data = Cube.flux.data
    Result_cube_error = Cube.error_cube.data
    
    info_keys = list(info.keys())
    
    for key in info_keys:
        map_flx = np.zeros((4,Cube.dim[0], Cube.dim[1]))
        map_flx[:,:,:] = np.nan
            
        info[key]['flux_map'] = map_flx
        
        if info[key]['kin'] ==1:
            map_ki = np.zeros((5,Cube.dim[0], Cube.dim[1]))
            map_ki[:,:,:] = np.nan

            info[key]['kin_map'] = map_ki
    # =============================================================================
    #        Filling these maps
    # =============================================================================

    Spax = PdfPages(Cube.savepath+Cube.ID+'_Spaxel_general_fit_detection_only_comp'+add+'.pdf')

    Results2_map = np.full((2,len(results2)), fill_value=np.nan)
    for i,row in enumerate(Result_cube):
        Results2_map[:,i] = row[0], row[1]
    '''
    for row in tqdm.tqdm(range(len(results1))):

        try:
            i,j, Fits = results1[row]
        except:
            print('Loading old fits? I am sorry no longer compatible...')

        if str(type(Fits)) == "<class 'dict'>":
            failed_fits+=1
            continue

        Result_cube_data[:,i,j] = Fits.fluxs.data
        try:
            Result_cube_error[:,i,j] = Fits.error.data
        except:
            lds=0
        Result_cube[:,i,j] = Fits.yeval

        for key in info_keys:
            
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

                if info[key]['kin'] ==1:
                    info[key]['kin_map'][0,i,j] = (np.median(Fits.chains['z'])-Cube.z)/(1+Cube.z)*3e5
                    info[key]['kin_map'][1,i,j] = np.median(Fits.chains[info[key]['fwhm']])
                    
            else:
                dl = Cube.obs_wave[1]-Cube.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+Cube.z)/1e4)/dl
                info[key]['flux_map'][3,i,j] = -SNR_cut*Fits.error[-1]*dl*np.sqrt(n)

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
        hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))

    for key in info_keys:
        if info[key]['kin'] ==1:
            hdus.append(fits.ImageHDU(info[key]['kin_map'], name=key+'_kin'))

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(Cube.savepath+Cube.ID+'_general_fits_maps'+add+'.fits', overwrite=True)

    return f
'''
def Map_creation_general(Cube,info, SNR_cut = 3 , width_upper=300,add='',\
                            brokenaxes_xlims= ((2.820,3.45),(3.75,4.05),(5,5.3)) ):
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
        map_flx = np.zeros((4,Cube.dim[0], Cube.dim[1]))
        map_flx[:,:,:] = np.nan
            
        info[key]['flux_map'] = map_flx
        
        if info[key]['kin'] ==1:
            map_ki = np.zeros((5,Cube.dim[0], Cube.dim[1]))
            map_ki[:,:,:] = np.nan

            info[key]['kin_map'] = map_ki
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

        for key in info_keys:
            
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

                if info[key]['kin'] ==1:
                    info[key]['kin_map'][0,i,j] = (np.median(Fits.chains['z'])-Cube.z)/(1+Cube.z)*3e5
                    info[key]['kin_map'][1,i,j] = np.median(Fits.chains[info[key]['fwhm']])
                    
            else:
                dl = Cube.obs_wave[1]-Cube.obs_wave[0]
                n = width_upper/3e5*(6564.52**(1+Cube.z)/1e4)/dl
                info[key]['flux_map'][3,i,j] = -SNR_cut*Fits.error[-1]*dl*np.sqrt(n)

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
        hdus.append(fits.ImageHDU(info[key]['flux_map'], name=key))

    for key in info_keys:
        if info[key]['kin'] ==1:
            hdus.append(fits.ImageHDU(info[key]['kin_map'], name=key+'_kin'))

    hdulist = fits.HDUList(hdus)
    hdulist.writeto(Cube.savepath+Cube.ID+'_general_fits_maps'+add+'.fits', overwrite=True)

    return f