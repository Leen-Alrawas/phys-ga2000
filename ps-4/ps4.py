# PS4 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import gaussxw
from gaussxw import gaussxw

# PROBLEM 1
print("\n")
print("PROBLEM 1\n")

# Constants
V = 0.001
p = 6.022 * (10**(28))
theta = 428
kb = 1.380649 * (10**(-23))
Coeff = 9 * V * p * kb * (theta**(-3))
	
def f(x): # the function of the integrand 
	return x**4 * math.exp(x) * ((math.exp(x)-1)**(-2))
	
def Integral(T,N=50):
	# Set the Integration Limits
	a = 0.0
	b = theta * (T**(-1))
	# Calculate the sample points and weights, then map them
	x,w = gaussxw(N)
	xp = 0.5*(b-a)*x + 0.5*(b+a)
	wp = 0.5*(b-a)*w
	# Perform the integration
	s = 0.0
	for k in range(N):
		s += wp[k]*f(xp[k])
	return s 
	
def Cv(T,N=50): # specific heat calculation 
	return Coeff * T * T * T * Integral(T,N)	

x = range(5, 500, 5) # generate the plot points by varying T
Cv_list = []
for i in x: 
	Cv_list.append(Cv(i))
	
plt.plot(x, Cv_list)
plt.xlabel('Temperature(K)')
plt.ylabel('Heat Capacity Cv(J/(K*kg))')
plt.title('Heat Capacity of a Solid as a function of the Temperature')
plt.savefig("Heat_capacity.png")
plt.show()

x2 = range(5, 100, 1) # generate the plot points by varying N
Cv_list2 = []
for i in x2: 
	Cv_list2.append(Cv(50,i))
	
plt.plot(x2, Cv_list2)
plt.xlabel('Number of Points N')
plt.ylabel('Heat Capacity Cv(J/(K*kg))')
plt.title('Heat Capacity of a Solid evaluated using N points')
plt.savefig("Heat_capacity2.png")
plt.show()

# PROBLEM 2
print("\n")
print("PROBLEM 2\n")

# Constants
m = 1
Coeff2 = math.sqrt(8*m)

def g(x, a): # the function of the integrand 
	return (math.sqrt(a**4 - x**4))**(-1)
	
def Integral2(a ,N=20):
	# Set the Integration Limits
	b1 = 0.0
	b2 = a
	# Calculate the sample points and weights, then map them
	x,w = gaussxw(N)
	xp = 0.5*(b2-b1)*x + 0.5*(b2+b1)
	wp = 0.5*(b2-b1)*w
	# Perform the integration
	s = 0.0
	for k in range(N):
		s += wp[k]*g(xp[k],a)
	return s 	
	
def Period(a,N=20): # period calculation 
	return Coeff2 * Integral2(a,N)	

domain = [n * 0.1 for n in range(1, 20)] # generate the plot points by varying a
rangelist = []
for i in domain: 
	rangelist.append(Period(i))
	
plt.plot(domain, rangelist)
plt.xlabel('Amplitude a(m)')
plt.ylabel('Period T(s)')
plt.title('Period of an anharmonic oscillator as a function of the amplitude')
plt.savefig("harmonic_oscillator.png")
plt.show()


# PROBLEM 3
print("\n")
print("PROBLEM 3\n")

from functools import lru_cache
@lru_cache(maxsize=None) # this saves all previous calculated H(x) to avoid repeated calculations
def H(x, n): # recursive relation for Hermite Polynomials
	if(n==0):
		return x+1-x
	elif(n==1):
		return 2*x
	else:
		return 2*x*H(x, n-1)-2*(n-1)*H(x, n-2)

x = [n * 0.2 for n in range(-20, 20)] # lists of points to be plotted for n=0,1,2,3
rangey0 = []
rangey1 = []
rangey2 = []
rangey3 = []
for i in x: 
	rangey0.append(H(i, 0))
	rangey1.append(H(i, 1))
	rangey2.append(H(i, 2))
	rangey3.append(H(i, 3))
#x = np.linspace(-4.0, 4.0, 100)

# plotting Hn(x) for n=0,1,2,3 and x=[-4,4]
plt.plot(x, rangey0, label="n=0")
plt.plot(x, rangey1, label="n=1")
plt.plot(x, rangey2, label="n=2")
plt.plot(x, rangey3, label="n=3")
plt.ylim(-100, 100)
plt.legend()
plt.xlabel('X-Axis')
plt.ylabel('Hermite Polynomials Hn(x)')
plt.savefig("Hermite_Polynomials.png")
plt.show()	

xx = [n * 0.1 for n in range(-100, 101)]
#xx = [n * 0.1 for n in range(-40, 41)]
rangey30 = []
for i in xx: 
	rangey30.append(H(i, 30))

# plotting Hn(x) for n=30 and x=[-4,4]
plt.plot(xx, rangey30)	
plt.xlabel('X-Axis')
plt.ylabel('Hermite Polynomial H(x), n=30')
plt.savefig("Hermite_Polynomials_n=30_2.png")
plt.show()	

def integrand(x,n): # the function of the integrand after change of variable  
	return math.exp(-(x/(1-x**2))**2) * ((H(x/(1-x**2), n))**2) * ((1-x**2)**(-4)) * (x**2 + x**4)
	
def Integral_qm(n,N=100): # Gaussian Quadrature method
	# Set the Integration Limits
	a = -1
	b = 1
	# Calculate the sample points and weights, then map them
	x,w = gaussxw(N)
	xp = 0.5*(b-a)*x + 0.5*(b+a)
	wp = 0.5*(b-a)*w
	# Perform the integration
	s = 0.0
	for k in range(N):
		s += wp[k]*integrand(xp[k], n)
	return s 
	
def root_mean_square(n): # the vaue of the root mean square with the coefficent
	return math.sqrt(Integral_qm(n) * ( (2**n) * math.factorial(n) * math.sqrt(math.pi))**(-1))

print("root mean square using Gaussian Quadrature =", root_mean_square(5))
print("percentage error is =", abs((root_mean_square(5)-math.sqrt(5.5))*100/math.sqrt(5.5)))

def integrand2(x,n): # the function of the integrand for Gauss-Hermite Quadrature - no change of domain needed 
	return ((H(x, n))**2) * (x**2) 
	
#import scipy
#from scipy import special
from numpy import polynomial
def Integral_method2(n,N=10): # Gauss-Hermite Quadrature method
	x,w = polynomial.hermite.hermgauss(N)
	s = 0.0
	for k in range(N):
		s += w[k]*integrand2(x[k], n)
	return s 

def root_mean_square2(n): # the vaue of the root mean square with the coefficent
	return math.sqrt(Integral_method2(n) * ( (2**n) * math.factorial(n) * math.sqrt(math.pi))**(-1))
	
print("root mean square using Gauss-Hermite Quadrature =", root_mean_square2(5))
print("percentage error is =", abs((root_mean_square2(5)-math.sqrt(5.5))*100/math.sqrt(5.5)))	
	
	

