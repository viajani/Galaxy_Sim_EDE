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
    
    
    cat_path=argv[1]
    cat_table=Table.read(cat_path, format='fits')
    cat_data=cat_table.to_pandas()
    
    nobj = len(cat_data)
   
    RAall = cat_data.RA
    DECall = cat_data.DEC 
    magall = cat_data.mag
    rhall = cat_data.rh
    nsersicall = cat_data.nsersic #numpy.random.random(nobj)*10.
    ell1all = cat_data.ell1 
    ell2all = cat_data.ell2
      
    
    xsize = 128                      # pixels
    ysize = 128                      # pixels
    

######
    
    shear1=np.random.uniform(-0.05, 0.05,1)
    shear2=np.random.uniform(-0.05, 0.05,1)
    
    
###DEFINE IMAGE PARAMETERS###

    image_prop_path=argv[2]
    image_prop_table=Table.read(image_prop_path, format='fits')
    image_data=image_prop_table.to_pandas()

    random_seed = 8241574

    pixel_scale = image_data.pixel_scale               # arcsec / pixel  (size units in input catalog are pixels)
    image_size = image_data.image_size               # pixels

    t_exp = image_data.t_exp #s
    gain = image_data.gain #e-/ADU
    readoutnoise = image_data.readoutnoise #e-
    sky_bkg = image_data.sky_bkg #mag/arcsec2
    
    ZP=image_data.ZP #mag

    F_sky = pixel_scale**(2)*t_exp*10**(-(sky_bkg-ZP)/2.5) #e-/pixel
    noise_variance = ( numpy.sqrt( ( (readoutnoise)**2 + F_sky ) ) *1/gain )**2 #e- -> ADU by dividing sigma by gain ; sigma = 4.9ADU
######


   

###DISPLAY INFO###
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

    for k in range(nobj):
        logger.info('    - Galaxy %.0f', k)
        logger.info('    - position %.0f,%.0f', xall[k],yall[k])
        logger.info('    - magnitude %.2f', magall[k])
        logger.info('    - half-light radius %.2f', rhall[k])
        logger.info('    - sersic index %.2f', nsersicall[k])
        logger.info('    - ellipticity %.4f,%.4f\n', ell1all[k],ell2all[k])
######

###CREATE OUTPUT IMAGES###    
    #Create 1st image (bright galaxies only)
    file_name ='sim_%s_nonoise.fits' %(nobj)
    file_name_noise ='sim_%s_noise.fits' %(nobj)
    full_image = galsim.ImageF(image_size, image_size)
    full_image.setOrigin(1,1)
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
    image_center = full_image.true_center
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=full_image.true_center)

    # We can also put it on the celestial sphere to give it a bit more realism.
    # The TAN projection takes a (u,v) coordinate system on a tangent plane and projects
    # that plane onto the sky using a given point as the tangent point.  The tangent 
    # point should be given as a CelestialCoord.
    sky_center = galsim.CelestialCoord(ra=3.544151*galsim.hours, dec=-27.791371*galsim.degrees)
    # The third parameter, units, defaults to arcsec, but we make it explicit here.
    # It sets the angular units of the (u,v) intermediate coordinate system.

    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    full_image.wcs = wcs

    logger.info('Image %r and %r created',file_name,file_name_noise)
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

    for k in range(nobj):

        #Read galaxy parameters from catalog
        RA = RAall[k]
        DEC = DECall[k]
        
        world_pos = galsim.CelestialCoord(RA*galsim.degrees, DEC*galsim.degrees)
        image_pos = wcs.toImage(world_pos)
        
        mag = magall[k]
        half_light_radius = rhall[k]
        nsersic = nsersicall[k]
        ell1 = ell1all[k]
        ell2 = ell2all[k]
        
        #Get position on sky
        #image_pos = galsim.PositionD(x,y)
        #world_pos = affine.toWorld(image_pos)
        #image_pos = wcs.toImage(world_pos)

                        
        #Galaxy is a sersic profile:
        fluxflux = t_exp/gain*10**(-(mag-ZP)/2.5)
        gal = galsim.Sersic(n=nsersic, half_light_radius=half_light_radius, flux=fluxflux, gsparams=gsparams, trunc=half_light_radius*4.5)
        gal = gal.shear(e1=ell1, e2=ell2)
        gal = gal.shear(g1=shear1, g2=shear2)

        
                    
        #Rotate galaxy
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
