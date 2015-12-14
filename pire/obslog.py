import os, glob, sys
import numpy as np
import sklearn.cluster
import astropy.io.fits as pyfits


hdrtags = {'cra':'RA', 'cdec': 'DEC', 'ra':'CRVAL1', 'dec': 'CRVAL2', 'detpa': 'POSANGLE',
           'azoff': 'INSTAZ', 'eloff': 'INSTEL',
           'filter':'FILTID1', 'etime':'EXPTIME', 'obstime': 'HEADTIME'}


class ObservingLog(object):

    def __init__(self, dirname='.', pattern='*.fits.gz',
                 hdrnamemap=hdrtags, ndither=3):
        self.hdrpar = hdrnamemap
        self.dirname = dirname
        self.pattern = pattern
        self.ndither = ndither
        self.reset(reread=True)

    def reset(self, reread=False):
        if reread:
            self._headers = all_metadata(self.dirname, self.pattern)
        self.imnames = np.sort(self._headers.keys())

    def get(self, tag, imnames=None, **extras):
        """An ndarray of the desired FITS parameter.
        """
        if imnames is None:
            imnames = self.imnames
        imnames = np.atleast_1d(imnames)
        par = self.hdrpar.get(tag, tag)
        vals = np.array([self._headers[im][par] for im in imnames])
        return vals
    
    def resort(self, tag, **extras):
        """Sort the images by 
        """
        x = self.get(tag)
        self.imnames = self.imnames[x.argsort()]

    def select(self, tag, value, **extras):
        """Change the imname list to only include objects with a particular tag
        value (e.g. 'filter' == 'J')
        """
        valid = self.get(tag) == value
        self.imnames = self.imnames[valid]
        
    def imind(self, imname, **extras):
        """Find the index of the image with name ``imname``
        """
        return (list(self.imnames)).index(imname)

    @property
    def dithergroup(self):
        """Find ``n`` groups of observations (based on ra and dec).  ``n`` is
        set by the ``ndither`` attribute.
        """
        ra = self.get('ra')
        dec = self.get('dec')
        return dither_group(self.ndither, ra, dec)

    def image_data(self, imnames):
        imnames = np.atleast_1d(imnames)
        dat = []
        for im in imnames:
            n = os.path.join(self.dirname, im)
            dat.append(pyfits.getdata(n))
        return dat

    def knearest(self, imname, k, metric='obstime', dithersep=True, **extras):
        """Find the k nearest images (in the space defined by ``metric``
        """
        x = self.get(metric)
        ind = self.imind(imname)
        if dithersep:
            dgroup = self.dithergroup.copy()
            valid = dgroup != dgroup[ind]
        else:
            valid = np.ones(len(x), dtype=bool)
        diff = np.abs(x[ind] - x[valid])
        oo = np.argsort(diff)
        knames = self.imnames[valid][oo][:k]
        return knames

    @property
    def hkeys(self):
        return list(np.unique(self._headers.values()[0].keys()))


def all_metadata(dirname, pattern='*.fits.gz'):
    files = glob.glob(os.path.join(dirname, pattern))
    info = {}
    for i,f in enumerate(files):
        key = os.path.basename(f) 
        info[key] = pyfits.getheader(f)

    return info


def dither_group(n, ra, dec):
    km = sklearn.cluster.KMeans(n)
    dither = km.fit_predict(np.vstack([ra, dec]).T)
    return dither


if __name__ == "__main__":
    pass
