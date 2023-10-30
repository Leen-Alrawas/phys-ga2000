# PS6 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import astropy
from astropy import *
import astropy.io
from astropy.io import fits
import numpy.linalg as linalg
import timeit
from statistics import mean 

# PROBLEM 1

hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data
logwave2 = 10**(logwave)*10**(-1) #the wavelength in nm

#check dimensions of the list 
print("wavelength dim", len(logwave))
print("flux dim", len(flux), len(flux[0]))

#the plot using given units
plt.plot(logwave, flux[0], label="galaxy 1")
plt.plot(logwave, flux[1], label="galaxy 2")
plt.plot(logwave, flux[2], label="galaxy 3")
plt.legend()
plt.xlabel('$Log_{10}(\lambda(A))$')
plt.ylabel('Flux ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot.png")
plt.show()

#the plot using nm for the wavelength
plt.plot(logwave2, flux[0], label="galaxy 1")
plt.plot(logwave2, flux[1], label="galaxy 2")
plt.plot(logwave2, flux[2], label="galaxy 3")
plt.legend()
plt.xlabel('$\lambda(nm)$')
plt.ylabel('Flux ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot2.png")
plt.show()


normalization_factors = [] #should have size 9713
fluxNorm = [] #should have size 4001
fluxNormalized = [] #should have size 9713
for i in range(len(flux)):
	factor = 1/sum(flux[i])
	normalization_factors.append(factor)
	fluxNorm = factor * flux[i]
	fluxNormalized.append(fluxNorm)
#print(len(normalization_factors),len(fluxNorm),len(fluxNormalized[0]),len(fluxNormalized))	

mean_spectrum_list = [] 
new_fluxes = []
for i in range(len(flux)):
	mean_spectrum = fluxNormalized[i].mean()
	mean_spectrum_list.append(mean_spectrum)
	residuals = fluxNormalized[i] - mean_spectrum
	new_fluxes.append(residuals)
print("new flux dim", len(new_fluxes), len(new_fluxes[0])) #should give 9713 and 4001


#the plot using residuals 
plt.plot(logwave, new_fluxes[0], label="galaxy 1")
plt.plot(logwave, new_fluxes[1], label="galaxy 2")
plt.plot(logwave, new_fluxes[2], label="galaxy 3")
plt.legend()
plt.xlabel('$Log_{10}(\lambda(A))$')
plt.title('Normalized Fluxes ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.ylabel('Residuals of Normalized Flux ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot_normal2.png")
plt.show()


C = np.array(new_fluxes)
Ct = C.transpose()
Covar = Ct.dot(C) 
print("Covariance matrix dim", len(Covar), len(Covar[0])) #should be a 4001 * 4001 matrix 
print("R dim", C.shape) #should be 9713 * 4001
eigenvaluesCo, eigenvectorsCo = np.linalg.eig(Covar)
idx = eigenvaluesCo.argsort()[::-1]  #sort eigenvectors based on eigenvalues 
eigenvaluesCo = eigenvaluesCo[idx]
eigenvectorsCo = eigenvectorsCo[:,idx]
print("eigenvectors dim", len(eigenvectorsCo[0]),len(eigenvectorsCo)) #length of eigenvector and number of eigenvectors 
print("eigenvalues of Covariance matrix", eigenvaluesCo)


#condition number = highest eigenvalue / lowest eigenvalue
#eigenvectors plot
plt.plot(eigenvectorsCo[0], label="eigenvector 1")
plt.plot(eigenvectorsCo[1], label="eigenvector 2")
plt.plot(eigenvectorsCo[2], label="eigenvector 3")
plt.plot(eigenvectorsCo[3], label="eigenvector 4")
plt.plot(eigenvectorsCo[4], label="eigenvector 5")
plt.legend()
#plt.xlabel('$Log_{10}(\lambda(A))$')
plt.ylabel('Eigenvectors of Covariance Matrix')
plt.savefig("Eigenvectors_Co.png")
plt.show()


(U, w, VT) = linalg.svd(C)
V = VT.transpose() #these are the eigenvectors
eigvecs_svd = V
eigvals_svd = w**2
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]
condition_number_SVD = max(eigvals_svd)/min(eigvals_svd)
condition_number_Co = max(eigenvaluesCo)/min(eigenvaluesCo)
print("eigenvectors dim", len(V),len(V[0])) #length of eigenvector and number of eigenvectors 
print("eigenvalues using SVD", eigvals_svd)


#eigenvectors plot
plt.plot(eigvecs_svd[0], label="eigenvector 1")
plt.plot(eigvecs_svd[1], label="eigenvector 2")
plt.plot(eigvecs_svd[2], label="eigenvector 3")
plt.plot(eigvecs_svd[3], label="eigenvector 4")
plt.plot(eigvecs_svd[4], label="eigenvector 5")
plt.legend()
#plt.xlabel('$Log_{10}(\lambda(A))$')
plt.ylabel('Eigenvectors Using SVD')
plt.savefig("Eigenvectors_Co_SVD.png")
plt.show()


print("Condition number: Using Covariance matrix =", condition_number_Co)
print("Condition number: Using SVD =", condition_number_SVD)

used_eigenvectors = eigenvectorsCo[:,:5] #first five eigenvectors 
reduced_wavelength_data = np.dot(used_eigenvectors.transpose(),C.transpose())
weights = reduced_wavelength_data.transpose()
print("weights dim", weights.shape) #should be 9713 * 5
approximate_spectra = np.dot(used_eigenvectors, reduced_wavelength_data).transpose()

plt.plot(logwave, approximate_spectra[0], label="approximation")
plt.plot(logwave, new_fluxes[0], label="data")
plt.legend()
plt.xlabel('$Log_{10}(\lambda(A))$')
plt.title('The first galaxy')
plt.ylabel('Normalized Fluxes ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot_new.png")
plt.show()

plt.plot(logwave, approximate_spectra[1], label="approximation")
plt.plot(logwave, new_fluxes[1], label="data")
plt.legend()
plt.xlabel('$Log_{10}(\lambda(A))$')
plt.title('The second galaxy')
plt.ylabel('Normalized Fluxes ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot_new2.png")
plt.show()

#map back to original spectra 
new_appr_spectra = []
for i in range(len(flux)):
	new_appr = (approximate_spectra[i] + mean_spectrum_list[i]) / (normalization_factors[i])
	new_appr_spectra.append(new_appr)
	
plt.plot(logwave, new_appr_spectra[0], label="approximation")
plt.plot(logwave, flux[0], label="original data")
plt.legend()
plt.xlabel('$Log_{10}(\lambda(A))$')
plt.title('The first galaxy')
plt.ylabel('Fluxes ($10^{-17}*erg*s^{-1}*cm^{-2}*A^{-1}$)')
plt.savefig("spectra_plot_orig.png")
plt.show()	

plt.plot(weights[0], weights[1])
plt.xlabel('$c_0$')
plt.ylabel('$c_1$')
plt.savefig("c0 and c1")
plt.show()

plt.plot(weights[0], weights[2])
plt.xlabel('$c_0$')
plt.ylabel('$c_2$')
plt.savefig("c0 and c2")
plt.show()

def residuals(N, galaxy=0): #for the first galaxy by default
	used_eigenvectors = eigenvectorsCo[:,:N]  
	reduced_wavelength_data = np.dot(used_eigenvectors.transpose(),C.transpose())
	weights = reduced_wavelength_data.transpose()
	approximate_spectra = np.dot(used_eigenvectors, reduced_wavelength_data).transpose()
	residuals_sqr = (new_fluxes - approximate_spectra)**2
	return mean(residuals_sqr[galaxy])

residuals_list = []
for i in range(20):
	residuals_list.append(residuals(i))
	
plt.plot(range(20), residuals_list, label="first galaxy")
plt.legend()
plt.xlabel('$N_c$')
plt.title('The first galaxy')
plt.ylabel('Mean of Squared Residuals (approximation - spectra)')
plt.savefig("sqr_residuals.png")
plt.show()

residuals_list = []
for i in range(20):
	residuals_list.append(residuals(i,1))
	
plt.plot(range(20), residuals_list, label="second galaxy")
plt.legend()
plt.xlabel('$N_c$')
plt.title('The second galaxy')
plt.ylabel('Mean of Squared Residuals (approximation - spectra)')
plt.savefig("sqr_residuals_2.png")
plt.show()

used_eigenvectors = eigenvectorsCo[:,:20]  
reduced_wavelength_data = np.dot(used_eigenvectors.transpose(),C.transpose())
weights = reduced_wavelength_data.transpose()
approximate_spectra = np.dot(used_eigenvectors, reduced_wavelength_data).transpose()
fractional_error = (new_fluxes - approximate_spectra)**2

print("fractional_error", fractional_error)


