# PS5 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import gaussxw
from gaussxw import gaussxw
from numpy import polynomial 
from numpy import loadtxt
import scipy
import numpy.linalg as linalg

# PROBLEM 1
print("\n")
print("PROBLEM 1\n")

def gamma_integrand(input, a):
	return input**(a-1) * np.exp(-input)

x1 = np.linspace(0, 5, 1000)
y1 = gamma_integrand(x1, 2)
y2 = gamma_integrand(x1, 3)
y3 = gamma_integrand(x1, 4)

plt.plot(x1, y1, label="a=2")
plt.plot(x1, y2, label="a=3")
plt.plot(x1, y3, label="a=4")
#plt.ylim(-100, 100)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Gamma Integrand')
plt.savefig("gamma_integrand.png")
plt.show()

# First, using Gauss–Laguerre quadrature method (of weight e^-x)
def integrand2(input, a):
	return input**(a-1)	

def gamma1(a, N=100): # Gauss–Laguerre quadrature method
	x,w = polynomial.laguerre.laggauss(N)
	s = 0.0
	for k in range(N):
		s += w[k]*integrand2(x[k], a)
	return s

'''
y4 = gamma1(x1)
plt.plot(x1, y4)
plt.ylim(0, 25)
plt.xlabel('X-axis')
plt.ylabel('Gamma Function')
plt.title('Using Gauss–Laguerre Quadrature')
plt.savefig("gamma_function_1.png")
plt.show()
'''

# You can demonstrate the infinity at x=0 in a better way
# using the identity Gamma(x)=Gamma(x+1)/x

def gamma2(a, N=10): # Gauss–Laguerre quadrature method with an identity
	x,w = polynomial.laguerre.laggauss(N)
	s = 0.0
	for k in range(N):
		s += w[k]*integrand2(x[k], a+1)
	return s/a

y5 = gamma2(x1)
'''
plt.plot(x1, y5)#, label="a=2")
plt.ylim(0, 25)
#plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Gamma Function')
plt.title('Using Gauss–Laguerre Quadrature & Modified for x=0')
plt.savefig("gamma_function_2.png")
plt.show()	
'''
	
def integrand3(z,a): # the function of the integrand after change of variable
	# Using c = a-1 & x = -z*c/(z-1)
	return (a*z/(1-z))**(a-1) * np.exp(-a*z/(1-z)) * a/((1-z)**2)

def gamma_gaussian(n, N=10): # Gaussian Quadrature method
	# Set the Integration Limits
	a = 0
	b = 1
	# Calculate the sample points and weights, then map them
	x,w = gaussxw(N)
	xp = 0.5*(b-a)*x + 0.5*(b+a)
	wp = 0.5*(b-a)*w
	# Perform the integration
	s = 0.0
	for k in range(N):
		s += wp[k]*integrand3(xp[k], n)
	return s
		
y6 = gamma_gaussian(x1)
plt.plot(x1, y6, label="Gaussian Quadrature")
plt.plot(x1, y5, label="Gauss–Laguerre Quadrature")
plt.ylim(0, 25)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Gamma Function')
plt.savefig("gamma_function_4.png")
plt.show()

print("\n Using Gaussian Quadrature\n")
print("Gamma(3/2) =", gamma_gaussian(1.5), " Error =", abs(scipy.special.gamma(1.5)-gamma_gaussian(1.5)),"\n")
print("Gamma(3) =", gamma_gaussian(3)," Error =", scipy.special.gamma(3)-gamma_gaussian(3),"\n")
print("Gamma(6) =", gamma_gaussian(6)," Error =", abs(scipy.special.gamma(6)-gamma_gaussian(6)),"\n")
print("Gamma(10) =", gamma_gaussian(10)," Error =", scipy.special.gamma(10)-gamma_gaussian(10),"\n")

print("\n Using Gauss–Laguerre Quadrature\n")
print("Gamma(3/2) =", gamma2(1.5), " Error =", scipy.special.gamma(1.5)-gamma2(1.5),"\n")
print("Gamma(3) =", gamma2(3)," Error =", abs(scipy.special.gamma(3)-gamma2(3)),"\n")
print("Gamma(6) =", gamma2(6)," Error =", scipy.special.gamma(6)-gamma2(6),"\n")
print("Gamma(10) =", gamma2(10)," Error =", abs(scipy.special.gamma(10)-gamma2(10)),"\n")


# PROBLEM 2
print("\n")
print("PROBLEM 2\n")

DataIn = loadtxt('signal.dat')
time0, signal = loadtxt('signal.dat', unpack=True)

#signal plot
plt.plot(time0, signal, '.')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig("signal.png")
plt.show()	

#rescale time
time = (time0 - time0.mean())/time0.std()

#fitting a 3 degree polynomial
A = np.zeros((len(time), 4))
A[:, 0] = 1.
A[:, 1] = time 
A[:, 2] = time**2
A[:, 3] = time**3

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
Ainv = vt.transpose().dot(np.diag(1./w)).dot(u.transpose())
coeff = Ainv.dot(signal)
signal_model = A.dot(coeff)

plt.plot(time0, signal, '.', label='data')
plt.plot(time0, signal_model, '.', label='3-degree polynomial')
plt.xlabel('time')
plt.ylabel('signal')
plt.legend()
plt.savefig("cubic_model.png")
plt.show()

residuals = signal_model - signal
plt.plot(time0, residuals, '.')
plt.xlabel('time')
plt.ylabel('residuals between actual signal and its cubic model')
plt.savefig("residuals.png")
plt.show()

#fitting a 10 degree polynomial
B = np.zeros((len(time), 11))
B[:, 0] = 1.
B[:, 1] = time 
B[:, 2] = time**2
B[:, 3] = time**3
B[:, 4] = time**4
B[:, 5] = time**5
B[:, 6] = time**6
B[:, 7] = time**7
B[:, 8] = time**8
B[:, 9] = time**9
B[:, 10] = time**10

(u2, w2, vt2) = np.linalg.svd(B, full_matrices=False)
Binv = vt2.transpose().dot(np.diag(1./w2)).dot(u2.transpose())
coeff2 = Binv.dot(signal)
signal_model2 = B.dot(coeff2)

plt.plot(time0, signal, '.', label='data')
plt.plot(time0, signal_model2, '.', label='10-degree polynomial')
plt.xlabel('time')
plt.ylabel('signal')
plt.legend()
plt.savefig("degree10_model.png")
plt.show()

condition_number = max(w2)/min(w2)
print("Condition number for 10-degree polynomial =", condition_number)

#fitting a combination of sin & cos functions
period = 0.5 * max(time)
Periodic = np.zeros((len(time), 21))
Periodic[:, 0] = 1.
Periodic[:, 1] = np.sin((1/period)*time) 
Periodic[:, 2] = np.cos((1/period)*time) 
Periodic[:, 3] = np.sin((1/(period-0.1*period))*time) 
Periodic[:, 4] = np.cos((1/(period-0.1*period))*time)
Periodic[:, 5] = np.sin((1/(period-0.2*period))*time) 
Periodic[:, 6] = np.cos((1/(period-0.2*period))*time)
Periodic[:, 7] = np.sin((1/(period-0.3*period))*time) 
Periodic[:, 8] = np.cos((1/(period-0.3*period))*time)
Periodic[:, 9] = np.sin((1/(period-0.4*period))*time) 
Periodic[:, 10] = np.cos((1/(period-0.4*period))*time)
Periodic[:, 11] = np.sin((1/(period-0.5*period))*time) 
Periodic[:, 12] = np.cos((1/(period-0.5*period))*time)
Periodic[:, 13] = np.sin((1/(period-0.6*period))*time) 
Periodic[:, 14] = np.cos((1/(period-0.6*period))*time)
Periodic[:, 15] = np.sin((1/(period-0.7*period))*time) 
Periodic[:, 16] = np.cos((1/(period-0.7*period))*time)
Periodic[:, 17] = np.sin((1/(period-0.8*period))*time) 
Periodic[:, 18] = np.cos((1/(period-0.8*period))*time)
Periodic[:, 19] = np.sin((1/(period-0.9*period))*time) 
Periodic[:, 20] = np.cos((1/(period-0.9*period))*time)


(u3, w3, vt3) = np.linalg.svd(Periodic, full_matrices=False)
Periodicinv = vt3.transpose().dot(np.diag(1./w3)).dot(u3.transpose())
coeff3 = Periodicinv.dot(signal)
signal_model3 = Periodic.dot(coeff3)

plt.plot(time0, signal, '.', label='data')
plt.plot(time0, signal_model3, '.', label='Sin and Cos model')
plt.xlabel('time')
plt.ylabel('signal')
plt.legend()
plt.savefig("SinCos_model.png")
plt.show()

''' deleted part
Periodic2 = np.zeros((len(time), 3))
Periodic2[:, 0] = 1.
Periodic2[:, 1] = np.sin((1/1)*time) 
Periodic2[:, 2] = np.cos((1/1)*time) 


(u4, w4, vt4) = np.linalg.svd(Periodic2, full_matrices=False)
print(w4)
Periodic2inv = vt4.transpose().dot(np.diag(1./w4)).dot(u4.transpose())
coeff4 = Periodic2inv.dot(signal)
signal_model4 = Periodic2.dot(coeff4)

plt.plot(time, signal, '.', label='data')
plt.plot(time, signal_model4, '.', label='Sin and Cos model')
plt.xlabel('time')
plt.ylabel('signal')
plt.legend()
plt.savefig("SinCos_model2.png")
plt.show()
'''









