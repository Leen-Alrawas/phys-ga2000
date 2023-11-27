# PS9 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import banded as bd

from matplotlib import animation
from IPython.display import HTML

# set all constants in the problem
hbar = 1.054571817e-34	
m = 9.109e-31
L = 1.0e-8
x0 = L / 2
sigma = 1.0e-10
k = 5e+10
TimeStep = 1.0e-18
GridSize = 1000
a = L / GridSize
b1 = 1.0 - TimeStep * 1.0j * hbar / (2 * m * a * a)
b2 = TimeStep * 1.0j * hbar / (4 * m * a * a)
a1 = 1.0 + TimeStep * 1.0j * hbar / (2 * m * a * a)
a2 = -b2

# initialize psi 
def initialize(m, L, x0, sigma, k, TimeStep, TimePoints=10, GridSize=1000):

	psi = np.zeros(GridSize, dtype = np.csingle)
	x = np.linspace(0, L, GridSize)
	t = np.zeros(TimePoints)
	psi = np.exp( -( x - x0 )**2 / ( 2 * sigma**2 )) * np.exp(1.0j * k * x)
		
	A = np.zeros((GridSize, GridSize), dtype = np.csingle)

	for i in range(GridSize - 1):
		A[i, i] = a1
		A[i , i+1] = a2
		A[i+1, i] = a2
		
	return psi, A, t, x
	
# update quantities using the linear system 	
def UpdateQuantities(psi, A, GridSize=1000):
	
	v = np.zeros(GridSize, dtype = np.csingle)
	v[0] = b1 * psi[0] + b2 * psi[1]
	v[GridSize - 1] = b2 * psi[GridSize - 2] + b1 * psi[GridSize - 1]
	
	for i in list(np.arange(1, GridSize - 1)):
		v[i] = b1 * psi[i] + b2 * (psi[i+1] + psi[i-1])
	
	# this banded functions causes error! I just used the regular solver 
	#psi = bd.banded(A, v, 999, 999)
	psi = np.linalg.solve(A, v)
	
	# fix boundry conditions at 0
	psi[0] = 0
	psi[len(psi)-1] = 0

	return psi

# a function to make plots of x against the real part of psi 
def makePlot(x, y, figname, figtitle):
	plt.plot(x, np.real(y))
	plt.title(figtitle)
	plt.xlabel('X(m)')
	plt.ylabel('Re(Psi)')
	plt.savefig(figname)
	#plt.show()

# initialize the system 
currentTime = 0
psi, A, t, x = initialize(m, L, x0, sigma, k, TimeStep)
figname = 'PlotAtTime=' + str(currentTime) + '.png'
figtitle = 'Time=' + str(currentTime) 
makePlot(x, psi, figname, figtitle)

# the system eveloves across 4000 timestep each of e-18 sec
for k in range(4000):
	psi = UpdateQuantities(psi, A)
	currentTime += TimeStep
	if k%500==0:
		print(k)
		figname = 'PlotAtTime=' + str(currentTime) + '.png'
		figtitle = 'Time=' + str(currentTime) + ' sec'
		makePlot(x, psi, figname, figtitle)
	k += 1

	




