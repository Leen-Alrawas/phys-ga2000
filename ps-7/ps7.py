# PS7 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import jax
import jaxopt
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize as optimize
import pandas as pd
jax.config.update("jax_enable_x64", True)

# PROBLEM 1
print("\n")
print("PROBLEM 1\n")

def objective(x):
    return (x-0.3)**2 * np.exp(x) # The function to be minimized
    
def golden_section_search(f=None, astart=None, bstart=None, cstart=None, tol=1.e-16, maxiter=100):
    gsection = (3. - np.sqrt(5)) / 2
    a = astart
    b = bstart
    c = cstart
    niter = 0
    while((np.abs(c - a) > tol) & (niter < maxiter)):
        # split the larger interval
        if((b - a) > (c - b)):
            x = b
            b = b - gsection * (b - a)
        else:
            x = b + gsection * (c - b)
        fb = f(b)
        fx = f(x)
        # choose the new minimum 
        if(fb < fx):
            c = x # the interval is now smaller
        else:
            a = b # the minimum is now at x 
            b = x 
        niter += niter     
    return(b)
    
def parabolic_step(f=None, a=None, b=None, c=None):
    fa = f(a)
    fb = f(b)
    fc = f(c)
    denom = (b - a) * (fb - fc) - (b -c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b -c)**2 * (fb - fa)
    # If singular, just return b 
    if(np.abs(denom) < 1.e-15):
        x = b
    else:
        x = b - 0.5 * numer / denom
    return(x)

def parabolic_minimize(f=None, astart=None, bstart=None, cstart=None,
                       tol=1.e-16, maxiter=10000):
    a = astart
    b = bstart
    c = cstart
    bold = b + 2. * tol
    niter = 0
    while((np.abs(bold - b) > tol) & (niter < maxiter)):
        bold = b
        b = parabolic_step(f=f, a=a, b=b, c=c)
        if(b < bold):
            c = bold
        else:
            a = bold
        niter = niter + 1
    return(b)    

# Brent's method with quadratic interpolation and golden section search
def brent_method(f, a, b, c, tol=1e-16, maxiter=100):
	bold = b + 2. * tol
	niter = 0
	diff = c - a
	diff_older = diff
	while((np.abs(bold - b) > tol) & (niter < maxiter)):
		bold = b
		b = parabolic_step(f=f, a=a, b=b, c=c)	
		if ((b < a or b > c) or abs(b - bold) > diff_older):
			print ("changed to golden")
			return golden_section_search(f=f, astart=a, bstart=bold, cstart=c)
			break
		diff_older = diff	
		diff = abs (b - bold)	
		if(b < bold):
			c = bold
		else:
			a = bold
		niter = niter + 1
	return(b)   

# Initial interval [a, b]
a = -1.0
b = 0.5
c = 1.0

optimal_solution_scipy = scipy.optimize.brent(objective, brack=(a, b, c), tol=1.0e-16)
print("Optimal solution - Scipy:", optimal_solution_scipy)

# Perform Brent's method
optimal_solution = brent_method(objective, a, b, c)
optimal_value = objective(optimal_solution)

print("Optimal solution - Brent's:", optimal_solution)
print("difference =", abs(optimal_solution - optimal_solution_scipy))

# PROBLEM 2
print("\n")
print("PROBLEM 2\n")

def model(x, params=[2., 1.]): # example with b0 = 2 and b1 = 1 
    b0 = params[0]
    b1 = params[1]
    m = 1. / (1. + np.exp(-b0 - b1 * x))
    return(m)

data = pd.read_csv('survey.csv') 
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
xs = xs[x_sort]
ys = ys[x_sort]

plt.scatter(xs, ys, s=3)
plt.xlabel('x')
plt.ylabel('data')
plt.show()

def log_likelihood(params, *args):
	b0 = params[0]
	b1 = params[1]
	xs = args[0]
	ys = args[1]
	p = model(xs, params=params)
	for i in range(len(p)):
		if (p[i] == 0.):
			p[i] = 1e-16
		elif (p[i] == 1.):	
			p[i] = 1. - 1e-16
	element = ys * np.log10(p/(1-p)) + np.log10(1-p) 
	#summ = element.sum()
	summ = np.sum(np.array(element), axis = -1)
	return -summ

gradient = np.gradient(log_likelihood)

# Covariance matrix of parameters
def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance

#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt(np.diag(covariance))

pst = np.array([0.5, 0.5]) #initial array
xpath = [pst]

# Both methods give the same answer
r = optimize.minimize(log_likelihood, pst, args=(xs, ys))
#r = optimize.minimize(log_likelihood, pst, jac=gradient,
#                      args=(xs, ys), method='BFGS', tol=1e-6)

hess_inv = r.hess_inv # inverse of hessian matrix
var = r.fun/(len(ys)-len(pst)) 
dFit = error( hess_inv,  var)
xpath = np.array(xpath)

print('Optimal parameters and error:\n\tp: ' , r.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n ' , Covariance( hess_inv,  var))

b0, b1 = r.x
m = model(xs, [b0, b1])
plt.scatter(xs, ys, s=3, label="data")
plt.scatter(xs, m, s=3, label="model")
plt.legend()
plt.xlabel('age')
plt.ylabel('data (yes=1/no=0)')
plt.savefig("logistic_model.png")
plt.show()
