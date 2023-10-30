# PS6 - Grad Compuatational Physics 
# By Leen Alrawas

import timeit
import_module = '''
import numpy as np
import matplotlib.pyplot as plt
import math
import astropy
import astropy.io
from astropy.io import fits
import numpy.linalg as linalg
import timeit
from statistics import mean
'''

test_for_C = '''

hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

normalization_factors = [] #should have size 9713
fluxNorm = [] #should have size 4001
fluxNormalized = [] #should have size 9713
for i in range(len(flux)):
	factor = 1/sum(flux[i])
	normalization_factors.append(factor)
	fluxNorm = factor * flux[i]
	fluxNormalized.append(fluxNorm)
	
mean_spectrum_list = [] 
new_fluxes = []
for i in range(len(flux)):
	mean_spectrum = fluxNormalized[i].mean()
	mean_spectrum_list.append(mean_spectrum)
	residuals = fluxNormalized[i] - mean_spectrum
	new_fluxes.append(residuals)

C = np.array(new_fluxes)
Ct = C.transpose()
Covar = Ct.dot(C) 
eigenvaluesCo, eigenvectorsCo = np.linalg.eig(Covar)
idx = eigenvaluesCo.argsort()[::-1]  #sort eigenvectors based on eigenvalues 
eigenvaluesCo = eigenvaluesCo[idx]
eigenvectorsCo = eigenvectorsCo[:,idx]
'''

test_for_SVD = '''
hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data

normalization_factors = [] #should have size 9713
fluxNorm = [] #should have size 4001
fluxNormalized = [] #should have size 9713
for i in range(len(flux)):
	factor = 1/sum(flux[i])
	normalization_factors.append(factor)
	fluxNorm = factor * flux[i]
	fluxNormalized.append(fluxNorm)

mean_spectrum_list = [] 
new_fluxes = []
for i in range(len(flux)):
	mean_spectrum = fluxNormalized[i].mean()
	mean_spectrum_list.append(mean_spectrum)
	residuals = fluxNormalized[i] - mean_spectrum
	new_fluxes.append(residuals)

C = np.array(new_fluxes)
(U, w, VT) = linalg.svd(C)
V = VT.transpose() #these are the eigenvectors
eigvecs_svd = V
eigvals_svd = w**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]
'''

print("Time for Covariance matrix: ",timeit.timeit(stmt=test_for_C, setup=import_module, number=1))
print("Time for SVD: ",timeit.timeit(stmt=test_for_SVD, setup=import_module, number=1))