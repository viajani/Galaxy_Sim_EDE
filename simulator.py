#!/usr/bin/python

# Copyright (c) 2012-2015 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""
Short description
"""

import sys
import os
import math
import numpy
import logging
import time
import galsim
import pandas as pd
from astropy.table import Table


######## DEFINE FUNCTIONS ##########

def patch_selection(cat, patchsize):
    ''' Takes a catalog and and slices it into N patches of a given patchsize
        The patches are returned in a list of len N
    '''
    ra_min = np.amin(cat['RA_MAG'])
    ra_max = np.amax(cat['RA_MAG'])
    dec_min = np.amin(cat['DEC_MAG'])
    dec_max = np.amax(cat['DEC_MAG'])

    dra = ra_max-ra_min
    ddec = dec_max-dec_min

    # build the grid of center points of each tile
    grid_ra = np.arange(ra_min, ra_max, patchsize)
    grid_dec = np.arange(dec_min, dec_max, patchsize)

    patches = []
    for i in range(len(grid_ra)-1):
        for j in range(len(grid_dec)-1):
            cond = (cat['RA_MAG'] > grid_ra[i]) & (cat['RA_MAG'] < grid_ra[i+1]) & (cat['DEC_MAG'] > grid_dec[j]) & (cat['DEC_MAG'] < grid_dec[j+1])   
            patches.append(cat[cond])
    
    return patches

def get_ell_12(q, fi):
    ''' Computes the two components of the ellipticity
        given the axis ration q=b/a and the angle fi
        Returns ell1 and ell2
    '''
    ell_compl = (1-q)/(1+q) * np.exp(2j*np.radians(fi))
    ell1 = np.abs(ell_compl) * np.cos(2*np.radians(fi))
    ell2 = np.abs(ell_compl) * np.sin(2*np.radians(fi))
    return ell1, ell2

def main(argv):
    """
    Make a simple image with a few galaxies.
      - Only galaxies.  No stars.
      - PSF is Airy (Euclid-like)
      - Each galaxy is single sersic.
      - Noise is Gaussian using a specified sky value
    """
    timei = time.time()
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("simulator")

###DEFINE CATALOGUE PARAMETERS###
    
    # loading the full catalogue
#     cat_path=argv[1]
    cat_path = 'data/EUCLID_TU_CATsel_v1.fits'
    cat_table=Table.read(cat_path, format='fits')
#     cat_data=cat_table.to_pandas()
    
    # slicing the catalogue into patches
    patchsize = 1/60 #(size of the patch is 1x1 arcmin)
    patches = patch_selection(cat_table, patchsize)

    # work with one patch at a time
    nobj=[]
    RAall=[]
    DECall=[]
    magall=[]
    rhDall=[]
    rhBall=[]
    nsersicall=[]
    ell1all=[]
    ell2all=[]
    shear1all=[]
    shear2all=[]
    diskangleall=[]

    # work with one patch at a time, store the needed quantities for every patch in a list
    for patch in patches:
        nobj.append(len(patch))

        RAall.append(patch['RA_MAG'])
        DECall.append(patch['DEC_MAG'])
        magall.append(patch['VIS'])
        rhDall.append(patch['DISK_LENGTH'])
        rhBall.append(patch['BULGE_LENGTH'])
        # nsersicall at the moment is not read. it will be assigned either 1 or 4 if galaxy is disk or elliptical
        # not sure if 'DISK_ANGLE' is in fact the fi angle that computes the ellipticity compontents
        ell1, ell2 = get_ell_12(patch['DISK_AXIS_RATIO'], patch['DISK_ANGLE'])
        ell1all.append(ell1)
        ell2all.append(ell2)
        shear1all.append(patch['GAMMA1'])
        shear2all.append(patch['GAMMA2'])
        diskangleall.append(patch['DISK_ANGLE'])
    
    xsize = 128                      # pixels
    ysize = 128                      # pixels
    

######
    
#     shear1=np.random.uniform(-0.05, 0.05,1)
#     shear2=np.random.uniform(-0.05, 0.05,1)
    
    
# ###DEFINE IMAGE PARAMETERS###

#     image_prop_path=argv[2]
#     image_prop_table=Table.read(image_prop_path, format='fits')
#     image_data=image_prop_table.to_pandas()

#     random_seed = 8241574

#     pixel_scale = image_data.pixel_scale               # arcsec / pixel  (size units in input catalog are pixels)
#     image_size = image_data.image_size               # pixels

#     t_exp = image_data.t_exp #s
#     gain = image_data.gain #e-/ADU
#     readoutnoise = image_data.readoutnoise #e-
#     sky_bkg = image_data.sky_bkg #mag/arcsec2
    
#     ZP=image_data.ZP #mag

#     F_sky = pixel_scale**(2)*t_exp*10**(-(sky_bkg-ZP)/2.5) #e-/pixel
#     noise_variance = ( numpy.sqrt( ( (readoutnoise)**2 + F_sky ) ) *1/gain )**2 #e- -> ADU by dividing sigma by gain ; sigma = 4.9ADU
######


###DEFINE IMAGE PARAMETERS###
    num = argv[1] #number to appear in the image name

    random_seed = 8241574

    pixel_scale = 0.1               # arcsec / pixel  (size units in input catalog are pixels)
    xsize = 128                      # pixels
    ysize = 128                      # pixels
    image_size = np.int(60/0.1)               # pixels

    t_exp = 3*565 #s
    gain = 3.1 #e-/ADU
    readoutnoise = 4.2 #e-
    sky_bkg = 22.35 #mag/arcsec2
    
    ZP=24.0 #mag

    F_sky = pixel_scale**(2)*t_exp*10**(-(sky_bkg-ZP)/2.5) #e-/pixel
    noise_variance = ( numpy.sqrt( ( (readoutnoise)**2 + F_sky ) ) *1/gain )**2 #e- -> ADU by dividing sigma by gain ; sigma = 4.9ADU
######


   

###DISPLAY IMAGE INFO###
    logger.info('\nStarting simulator using:')
    logger.info('    - pixel scale = %.2f arcsec',pixel_scale)
    logger.info('    - Image size = %.0f pixels', image_size)
    logger.info('    - Image ZP = %.2f mag', ZP)
    logger.info('    - Image exposure time = %.0f s', t_exp)
    logger.info('    - Image gain = %.2f e-/ADU', gain)
    
    logger.info('\n    - Sky background = %.2f mag/arcsec2', sky_bkg)
    logger.info('    - Read-out noise = %.1f e-', readoutnoise)
    logger.info('    - Gaussian noise (sigma = %.2f ADU/pixel)', numpy.sqrt(noise_variance))

    logger.info('\n    - Airy PSF (lam=600,700,800, diam=1.2, obscuration=0.3)')
    logger.info('    - Sersic galaxies')
    logger.info('    - Number of galaxies = %.0f\n', nobj)

######


###MAKE THE WCS COORDINATES (test11)###
    # Make a slightly non-trivial WCS.  We'll use a slightly rotated coordinate system
    # and center it at the image center.
    theta = 0.17 * galsim.degrees
    #dudx = math.cos(theta.rad()) * pixel_scale
    #dudy = -math.sin(theta.rad()) * pixel_scale
    #dvdx = math.sin(theta.rad()) * pixel_scale
    #dvdy = math.cos(theta.rad()) * pixel_scale
    dudx = numpy.cos(theta) * pixel_scale
    dudy = -numpy.sin(theta) * pixel_scale
    dvdx = numpy.sin(theta) * pixel_scale
    dvdy = numpy.cos(theta) * pixel_scale

#     image_center = full_image.true_center
#     affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

#     # We can also put it on the celestial sphere to give it a bit more realism.
#     # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
#     # that plane onto the sky using a given point as the tangent point.  The tangent 
#     # point should be given as a CelestialCoord.
#     sky_center = galsim.CelestialCoord(ra=3.544151*galsim.hours, dec=-27.791371*galsim.degrees)
#     # The third parameter, units, defaults to arcsec, but we make it explicit here.
#     # It sets the angular units of the (u,v) intermediate coordinate system.

#     wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
#     full_image.wcs = wcs

######


###TUNE THE SPEED OF FFT###
    #slightly decrease the precision on fourrier and convolution to speed up.
    #Taken from Jarvis discussion https://github.com/GalSim-developers/GalSim/issues/566
    gsparams = galsim.GSParams(xvalue_accuracy=2.e-4, kvalue_accuracy=2.e-4,
                           maxk_threshold=5.e-3, folding_threshold=1.e-2)
######


###BUILD PSF###
    psf = galsim.Airy(lam=800, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3) + galsim.Airy(lam=700, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3) + galsim.Airy(lam=600, diam=1.2, obscuration=0.3, scale_unit=galsim.arcsec,flux=1./3)
    # ###uncomment to write the PSF
    # logger.info('\nWriting PSF')
    # image = galsim.ImageF(xsize,ysize,scale=pixel_scale)
    # psf.drawImage(image=image)
    # image.write('psf_nonoise.fits')
    
    # rng = galsim.BaseDeviate(random_seed)
    # noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance)*1./1000)
    # image.addNoise(noise)
    # image.write('psf_onethousands_noise.fits')
    
    # logger.info('PSF written in psf_nonoise.fits and psf_onethousands_noise.fits')
    # ###
#######


###PAINT GALAXIES###
    timeigal = time.time()
    logger.info('\n\nStarting to simulate galaxies')

    # go over the patches
    for p in range(len(patches)):    
#     for p in range(5):    
        ###CREATE OUTPUT IMAGES###    
        file_name ='output/sim_patch-%s_nonoise.fits' %(p)
        file_name_noise ='output/sim_patch-%s_noise.fits' %(p)
        full_image = galsim.ImageF(image_size, image_size)
        full_image.setOrigin(1,1)
        image_center = full_image.true_center
        affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)
        ra_cent = np.mean(RAall[p])
        dec_cent = np.mean(DECall[p])
        sky_center = galsim.CelestialCoord(ra=ra_cent*galsim.degrees, dec=dec_cent*galsim.degrees)
        wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
        full_image.wcs = wcs
        logger.info('Image %r and %r created',file_name,file_name_noise)
        
        # at the moment the center of the image and its wcs are the same for each patch. Realistically this should change
        ######

        for k in range(nobj[p]):
            #Read galaxy parameters from catalog
            RA = RAall[p][k]
            DEC = DECall[p][k]
            # from RA,DEC get pixel position on the image
            world_pos = galsim.CelestialCoord(RA*galsim.degrees, DEC*galsim.degrees)
            image_pos = wcs.toImage(world_pos)
            
            ## if disk galaxy, rh is the disk length and sersic is 1
            ## if elliptical galaxy, rh is bulge lenght and sersic is 4
            if rhDall != 0:
                half_light_radius = rhDall[p][k]
                nsersic=1
            else:
                half_light_radius = rhBall[p][k]
                nsersic=4
                
            mag = magall[p][k]
            ell1 = ell1all[p][k]
            ell2 = ell2all[p][k]
            # the shear for a galaxy is the mean shear of the patch
            shear1 = np.mean(shear1all[p])
            shear2 = np.mean(shear2all[p])
            diskangle = diskangleall[p][k]
                        
            ### DISPLAY INFO FOR GALAXIES IN PATCH###
            logger.info('    - Patch %.0f', p)
            logger.info('    - Galaxy %.0f', k)
            logger.info('    - position RA,DEC %.3f,%.3f', RA, DEC)
            logger.info('    - position X,Y %.0f,%.0f', image_pos.x, image_pos.y)
            logger.info('    - magnitude %.2f', mag)
            logger.info('    - half-light radius %.2f', half_light_radius)
            logger.info('    - sersic index %.2f', nsersic)
            logger.info('    - ellipticity %.4f,%.4f\n', ell1,ell2)


            #Galaxy is a sersic profile:
            fluxflux = t_exp/gain*10**(-(mag-ZP)/2.5)
            gal = galsim.Sersic(n=nsersic, half_light_radius=half_light_radius, flux=fluxflux, gsparams=gsparams, trunc=half_light_radius*4.5)
            gal = gal.shear(e1=ell1, e2=ell2)
            gal = gal.shear(g1=shear1, g2=shear2)


            #Rotate galaxy
            ang = diskangle
            gal = gal.rotate(theta=ang*galsim.degrees)

            #convolve galaxy with PSF
            final = galsim.Convolve([psf, gal])

            #offset the center for pixelization (of random fraction of half a pixel)
            ud = galsim.UniformDeviate(random_seed+k)
            x_nominal = image_pos.x+0.5
            y_nominal = image_pos.y+0.5
            ix_nominal = int(math.floor(x_nominal+0.5))
            iy_nominal = int(math.floor(y_nominal+0.5))
            dx = (x_nominal - ix_nominal)*(2*ud()-1)
            dy = (y_nominal - iy_nominal)*(2*ud()-1)
            offset = galsim.PositionD(dx,dy)

            #draw galaxy
            image = galsim.ImageF(xsize,ysize,scale=pixel_scale)
            final.drawImage(image=image,wcs=wcs.local(image_pos), offset=offset)
            image.setCenter(ix_nominal,iy_nominal)

            #add stamps to single image
            bounds = image.bounds & full_image.bounds
            full_image[bounds] += image[bounds]

        timegal = time.time()
        logger.info('%d galaxies computed in t=%.2f s',k+1,timegal-timeigal)
    ######

    ###WRITE THE FITS FILE BEFORE NOISE###
        full_image.write(file_name)
        logger.info('Image without noise written to fits file %r',file_name)
    ######

    ###ADD NOISE###
        #add Gaussian noise
        rng = galsim.BaseDeviate(random_seed)
        noise = galsim.GaussianNoise(rng, sigma=math.sqrt(noise_variance))
        full_image.addNoise(noise)
    ######

    ###WRITE THE FITS FILE WITH NOISE###
        full_image.write(file_name_noise)
        logger.info('Image with noise written to fits file %r',file_name_noise)
    ######


    timef = time.time()
    tot_time = timef-timegal
    logger.info('Noise added and image written to files in t=%.2f s',tot_time)

    tot_time = timef-timei
    logger.info('\nFull simulation run in t=%.2f s',tot_time)

if __name__ == "__main__":
    main(sys.argv)
