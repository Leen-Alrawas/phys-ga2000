# PS3 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand
import matplotlib.cm as cm
import matplotlib as matplotlib
from random import random
 
# PROBLEM 1
print("\n")
print("PROBLEM 1\n")

def f(x): # the function f(x)
	return x*(x-1)

def Df(x, sigma): # the derivative of the function f with small sigma
	return (f(x + sigma) - f(x))/(sigma)
	
def error(x, sigma): # the error with the analytical solution
	print ("for x =",x," and sigma =","%10.4E" % sigma, ", the error =","%10.8E" % abs((2*x - 1)-Df(x, sigma)))
	
for i in range(2,14,2):
	error(1,10**(-i))


# PROBLEM 2
print("\n")
print("PROBLEM 2\n")
print("Matrix Multiplication in Python")

# files to save the computation time 
#file_object = open("timings1.txt", "a")
#file_object = open("timings2.txt", "a")

import time
from numpy import zeros
from numpy import ones

timings1 = []
timings2 = []

for N in range(10,200,5): # for different matrix dimensions
	C = zeros([N,N] ,float)
	A = ones([N,N] ,float)
	B = ones([N,N] ,float)
	
	start_manual = time.time() # multiplying two matrices manually
	
	for i in range(N):
		for j in range(N):
			for k in range(N):
				C[i,j] += A[i,k]*B[k,j]
	
	end_manual = time.time()
	#with open('timings1.txt', 'a') as file: file.write(str(end_manual - start_manual))
	#with open('timings1.txt', 'a') as file: file.write('\n')
	timings1.append(end_manual - start_manual)
		
	start_numpy = time.time()
	
	np.dot(A, B) # multiplying two matrices using numpy commands
	
	end_numpy = time.time()
	#with open('timings2.txt', 'a') as file: file.write(str(end_numpy - start_numpy))
	#with open('timings2.txt', 'a') as file: file.write('\n')
	timings2.append(end_numpy - start_numpy)	

Ncubic_list = []
for i in range(10,200,5): # scaling as N^3
	Ncubic_list.append(i**3/500000)
	
#file_object.close()
plt.plot(range(10,200,5), timings1, '.r-', label="Using Nested For Loops")
plt.plot(range(10,200,5), timings2, '.b-', label="Using dot()")
plt.plot(range(10,200,5), Ncubic_list, '.g', label="N^3 Scaled", linestyle="--")
plt.legend()
plt.xlabel('Matrix Dimension N')
plt.ylabel('Computation Time')
plt.title('Matrix Multiplication In Python')
plt.savefig("matrix_multiplication.png")
plt.show()


from numpy import arange
from pylab import plot,xlabel,ylabel,show, legend, savefig


# PROBLEM 3
print("\n")
print("PROBLEM 3\n")	

# Constants
NBI213 = 10000                 # Number of Bi-213 atoms
NPb = 0                        # Number of lead atoms
NBI209 = 0                     # Number of Bi-209 atoms
NTl = 0                        # Number of Tl atoms
tauPb = 3.3*60                 # Half life of lead in seconds
tauTl = 2.2*60                 # Half life of Tl in seconds
tauBI213 = 46*60               # Half life of Bi-213 in seconds
h = 1.0                        # Size of time-step in seconds
prob_Pb = 1 - 2**(-h/tauPb)        # Probability of decay in one step for Pb
prob_Tl = 1 - 2**(-h/tauTl)        # Probability of decay in one step for Tl
prob_BI213 = 1 - 2**(-h/tauBI213)  # Probability of decay in one step for Bi-213
tmax = 20000                   # Total time
tpoints = arange(0.0,tmax,h)   # List of time plot points

# Lists of atoms plot points
BI213points = []
Pbpoints = []
BI209points = []
Tlpoints = []

# Main loop
for t in tpoints:
	
	# add number of each atom to atoms lists
	BI213points.append(NBI213)
	Pbpoints.append(NPb)
	BI209points.append(NBI209)
	Tlpoints.append(NTl)
	
	if (NBI213+NPb+NBI209+NTl) != 10000: # total number should be conserved at all times
		print("ERROR! Number of atoms not conserved")
		break
	
	# Calculate the number of atoms that decay for each type
	decayPb = 0 # loop for decay of Pb
	for i in range(NPb):
		if random() < prob_Pb:
			decayPb += 1
	NPb -= decayPb
	NBI209 += decayPb
	
	decayTl = 0 # loop for decay of Tl
	for i in range(NTl):
		if random() < prob_Tl:
			decayTl += 1
	NTl -= decayTl
	NPb += decayTl
	
	decay_to_Tl = 0 # loop for decay of Bi-213
	decay_to_Pb = 0
	for i in range(NBI213):
		if random() < prob_BI213: # Probability that atom Bi-213 decays
			if random() < 0.0209: # Proability the decay was to Tl
				decay_to_Tl += 1
			else:                 # Proability the decay was to Pb
				decay_to_Pb += 1
	NBI213 -= (decay_to_Pb + decay_to_Tl)
	NTl += decay_to_Tl
	NPb += decay_to_Pb

print("Maximum number of Tl atoms at any instant =", max(Tlpoints))

# Make the graph
plot(tpoints,Tlpoints, label="Tl-209")
plot(tpoints,Pbpoints, label="Pb-209")
plot(tpoints,BI209points, label="Bi-209")
plot(tpoints,BI213points, label="Bi-213")
legend()
xlabel("Time(s)")
ylabel("Number of atoms")
savefig("nuclear_decay.png")
show()


# PROBLEM 4
print("\n")
print("PROBLEM 4\n")

print("The decay 1000 atoms of Tl-208 to Pb-208")
from numpy import sort

# Constants
tau = 3.053*60           # Half life of thallium in seconds

listx = []
flipped_range = []
for i in range(1000): # the decay of 1000 Tl atoms 
	x = -1*(tau*math.log(1-random())) / (math.log(2)) # the times x at which an atom Tl decayed 
	flipped = 1000 - i 
	listx.append(x) 
	flipped_range.append(flipped)

sorted_x = sort(np.array(listx)).tolist()

plot(sorted_x,range(1000), label="Pb-208") 
plot(sorted_x,flipped_range, label="Tl-208")
legend()
xlabel ("Time(s)")
ylabel("Number of atoms")
savefig("nuclear_decay2.png")
show()
























