import numpy as np
import time

def skysub(imname, log, k, fitpar=None, **extras):
    obj = log.image_data(imname)[0]
    knames = log.knearest(imname, k, **extras)
    d = np.array(log.image_data(knames))

    if fitpar is None:
        sky = simple_sky(d)
    else:
        x = log.get(fitpar, imnames=knames)
        xt = log.get(fitpar, imnames=imname)
        sky = fancy_sky(d, x, xt, **extras)

    skysub = obj - sky
    history = ("Sky subtraction performed at {} using {} nearest "
               "(in time) off-target images.".format(time.ctime(), k))
    return skysub, history, sky
    

def simple_sky(d, x=None):
    med = np.median(d, axis=0)
    return med


def fancy_sky(d, x, xtarg, skyscale='mult', **extras):
    print(skyscale)
    norm = np.median(d, axis=[1,2])
    xhat = x - x.mean()
    nhat = norm - norm.mean()
    slope = np.mean(nhat / xhat)
    ntarg = slope * (xtarg - x.mean()) + norm.mean()

    # use the nearest neighbor for the overall sky level
    ntarg = norm[np.argmin(np.abs(xhat))]
    
    if skyscale == 'mult':
        # Treat differences in sky level as multiplicative, divide them off,
        # take the median, and multiply by the expected level at xtarg
        med = np.median(d / norm[:, None, None], axis=0) * ntarg
    elif skyscale == 'add':
        # Treat differences in sky level as additive, subtract them off, take
        # the median, and add the expected offset at xtarg
        med = np.median(d - nhat[:, None, None], axis=0) + (ntarg - norm.mean())

    return med

def make_flat(log, ontarg, nlo=3, nhi=5, flatscale='mult'):
    iofftarg = list([i not in ontarg for i in log.imnames])
    offtarg = log.imnames[np.array(iofftarg, dtype=bool)]
    d = np.array(log.image_data(offtarg))
    norm = np.median(d, axis=[1,2])
    if flatscale == 'mult':
        # Treat differences in sky level as multiplicative, divide them off,
        # and sort
        sorted_stack = np.sort(d / norm[:,None, None], axis=0)
    elif flatscale == 'add':
        # Treat differences in sky level as additive, subtract them off,
        # and sort
        sorted_stack = np.sort(d - norm[:, None, None], axis=0) + norm.mean()
    
    avclipped = np.mean(sorted_stack[nlo:-nhi, :, :], axis=0)
    return avclipped

def get_fwhm(y):
    from scipy.optimize import curve_fit
    n = len(y)
    x = np.arange(n) - n/2.0
    p0 = [1., 0., 1., 0.0]
    coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
    return coeff

def gauss(x, *p):
    A, mu, sigma, base = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + base
