import os, sys, glob, time
import numpy as np
import astropy.io.fits as pyfits
from pire.obslog import ObservingLog
import pire.irreduce as red


def tmass():
    from astropy import table
    t = table.Table()
    tab = t.read('fp_2mass.fp_psc3888.tbl.txt', format = 'ascii.ipac')
    tab.write('tmass.reg', format='ascii', include_names=['ra', 'dec'])
    tabarr = np.array(tab)
    return tabarr
    
def reduce_J(k, scale, outdir='.'):

    band = 'J'
    clobber = True
    
    dirname = '../Nov15_data/'
    pattern = '*.slp.fits.gz'
    log = ObservingLog(dirname, pattern)
    log.select('filter', band)

    print(scale)
    
    # Figure out which images are the `on` sources
    ontarg = log.imnames[np.abs(log.get('INSTAZ')) < 30.0]
    flat = red.make_flat(log, ontarg, flatscale=scale)
    
    for im in ontarg:
        hdr = log._headers[im].copy()
        hdr['SKYSC'] = scale
        hdr['FLATSC'] = scale
        skysub, histry, _ = red.skysub(im, log, k, fitpar='obstime', skyscale=scale)
        hdr.add_history(histry)
        flattened = skysub / (flat / np.median(flat))
        hdr.add_history('Divided by flat field at {} using flats from {} scaling'.format(time.ctime(), scale))

        # Hack
        hdr['CRPIX1'] = hdr['CRPIX1'] - 20
        
        pyfits.writeto(os.path.join(outdir, im.replace('.fits.gz', '.final.fits')),
                       flattened, header=hdr, clobber=clobber)
        pyfits.writeto('flat_{}_{}.fits'.format(scale, band), flat / np.median(flat), clobber=clobber)
        
        print(im)


def reduce_K(k, scale, etime=14.753, outdir='.'):
    clobber = True
    band = 'K'
    klog = ObservingLog('../Nov15_data/', '*dcorr*fits.gz')
    klog.select('filter', band)
    klog.select('etime', etime)

    # Figure out which images are the `on` sources
    ontarg = klog.imnames[np.abs(klog.get('INSTAZ')) < 30.0]
    flat = red.make_flat(klog, ontarg, flatscale=scale)
    
    for im in ontarg:
        hdr = klog._headers[im].copy()
        hdr['SKYSC'] = scale
        hdr['FLATSC'] = scale
        skysub, histry, _ = red.skysub(im, klog, k, fitpar='obstime', skyscale=scale)
        hdr.add_history(histry)
        flattened = skysub / (flat / np.median(flat))
        hdr.add_history('Divided by flat field at {} using flats from {} scaling'.format(time.ctime(), scale))

        # Hack
        #hdr['CRPIX1'] = hdr['CRPIX1'] - 20
        
        pyfits.writeto(os.path.join(outdir, im.replace('.fits.gz', '.final.fits')),
                       flattened, header=hdr, clobber=clobber)
        pyfits.writeto('flat_{}_{}.fits'.format(scale, band), flat / np.median(flat), clobber=clobber)

if __name__ =="__main__":
    k = 5
    scale = 'mult'
    reduce_J(k, scale)

    k = 6
    reduce_K(k, scale)
