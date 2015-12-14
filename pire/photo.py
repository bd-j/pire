import os, sys, glob, time, subprocess
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
import astropy.io.fits as pyfits
from sklearn.neighbors import KDTree
import matplotlib.pyplot  as pl

# Default sextractor files
piredir, f = os.path.split(__file__)
dd = piredir+'../data/default.'
default = {'sex': dd+'sex', 'cat': dd+'params'}


def imcoords(imname, catalog, colnames, origin=1):
    ra = catalog[colnames['ra']]
    dec = catalog[colnames['dec']]
    hdr = pyfits.getheader(imname)
    wcs = WCS(hdr)
    x, y = wcs.wcs_world2pix(ra, dec, origin)
    sel = (x > 0) & (x < hdr['naxis1']) & (y > 0) & (y < hdr['naxis2'])
    return x[sel], y[sel], catalog[sel]


def match(x1, y1, x2=None, y2=None, k=5, kdt=None):
    X2 = np.vstack([x2, y2]).T
    X1 = np.vstack([x1, y1]).T
    if kdt is None:
        kdt = KDTree(X2, leaf_size=30, metric='euclidean')
    dists, inds = kdt.query(X1, k=k, return_distance=True)
    return dists, inds, kdt


def run_command(cmd):
    """Open a child process, and return its exit status and stdout.
    """
    child = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = [s for s in child.stdout]
    err = [s for s in child.stderr]
    w = child.wait()
    return os.WEXITSTATUS(w), out, err


def sextract(imname, sexpars=default['sex'], catpars=default['cat'],
             **kwargs):
    """Run SExtractor on ``imname`` and return the name of the resulting
    catalog.
    """
    sname = imname.replace('.fits','.cat')
    
    command = ("sex {} -c {} -PARAMETERS_NAME {} "
               "-CATALOG_NAME {}").format(imname, sexpars, catpars, sname)
    for k, v in list(kwargs.items()):
        command += " -{} {}".format(k, v)
    ex, err, out = run_command(command)
    if ex > 0:
        print('Sextractor died with error message {}'.format(err))
        sys.exit()
    return sname

if __name__ == "__main__":

    imname = 'SAO-11_NGC185_A.3246.slp.final.fits'
    sname = sextract(imname)
    
    tab = Table.read(sname, format='ascii.sextractor')
    sex = np.array(tab)
    scols = {'ra': 'ALPHA_J2000', 'dec':'DELTA_J2000'}

    tab = Table.read('fp_2mass.fp_psc3888.tbl.txt', format = 'ascii.ipac')
    tmass = np.array(tab)
    tcols = {'ra': 'ra', 'dec':'dec'}

    tx, ty, tcat = imcoords(imname, tmass, tcols)
    dist, sinds, kdt = match(tx, ty, sex['X_IMAGE'], sex['Y_IMAGE'], k=5)
    dy = (sex[sinds[:,0]]['Y_IMAGE'] - ty)
    dx = (sex[sinds[:,0]]['X_IMAGE'] - tx)

    nx, ny = 2.0, 2
    good = (np.abs(dx - dx.mean()) < nx * dx.std()) & (np.abs(dy - dy.mean()) < ny * dy.std())

    smatch = sex[sinds[good,0]]
    tmatch = tcat[good]
    dm = smatch['MAG_APER'] - tmatch['j_m']

    pl.plot(tmatch['j_m'], smatch['MAG_APER'], 'o') 
