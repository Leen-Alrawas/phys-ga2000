# PS2 - Grad Compuatational Physics 
# By Leen Alrawas

import timeit
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand
import matplotlib.cm as cm
import matplotlib as matplotlib
 
# PROBLEM 1
print("\n")
print("PROBLEM 1\n")
print("The Errors representing 100.98763 in float32 calculated manually and by Python:")
print("%10.3E" % (abs((np.float64(((0.15586328506469727+1)/2 + 1)*2**6)-100.98763)*100)))
print("%10.3E" %((np.float32(100.98763)-100.98763)*100))

# PROBLEM 2
print("\n")
print("PROBLEM 2\n")

L = 100

# METHOD 1
print("Using for loops")

def potential(x,y,z):
	if x**2+y**2+z**2 != 0:
		if ((x+y+z)%2 == 0):
			return 1/(math.sqrt(x**2+y**2+z**2))
		else:
			return -1/(math.sqrt(x**2+y**2+z**2))
	else: 
		return 0 

sum = 0 
for i in range(-L,L+1):
	for j in range(-L,L+1):
		for k in range(-L,L+1):
			sum += potential(i,j,k)
print(sum)		

# Another approach using for loops
"""
sum2 = np.sum(np.array([potential(i,j,k)
for i in range(-L,L+1)
for j in range(-L,L+1)
for k in range(-L,L+1)]))
print(sum2)	
"""


# METHOD 2
print("Using lists and maps")

# step 1: we create all possible combinations of (i,j,k) and save them in a list of lists
result = []
list(map(lambda a: list(map(lambda b: list(map(lambda c: result.append((a, b, c)), range(-L,L+1))), range(-L,L+1))), range(-L,L+1)))

# step 2: define the potential on a list [i,j,k] 
def potential2(listt):
	if listt[0]**2+listt[1]**2+listt[2]**2 != 0:
		if ((listt[0]+listt[1]+listt[2])%2 == 0):
			return 1/(math.sqrt(listt[0]**2+listt[1]**2+listt[2]**2))
		else:
			return -1/(math.sqrt(listt[0]**2+listt[1]**2+listt[2]**2))
	else: 
		return 0

# step 3: add the individual potentials of each [i,j,k], to add them we convert the list to an array first 
sum2 = np.sum(np.array(list(map(potential2, result))))
print(sum2)


# PROBLEM 3
print("\n")
print("PROBLEM 3\n")

# recursive relation to determine if a number belongs to Mandelbrot set 
# the only input needed is the first two cx & cy, e.g. print(is_Mandelbrot(0,1))
def is_Mandelbrot(cx,cy,x=0,y=0,start=0): # cx is the real part and cy is the imaginary part
	#print(x,y,start) 
	if((x**2 + y**2) > 4):
		return False #white
	elif(start > 100):
		return True #black
	else:
		return is_Mandelbrot(cx,cy,x**2-y**2+cx,2*x*y+cy,start+1)

N = 200

arrayx=np.random.uniform(size = N, low = -2, high = 2) 
arrayy=np.random.uniform(size = N, low = -2, high = 2) 

for i in range(N):
	if(is_Mandelbrot(arrayx[i], arrayy[i])==True):
		color_ = 'black'
	else:
		color_= 'white'
	plt.scatter(arrayx[i], arrayy[i], c = color_, s = 10, linewidths=0)
plt.xlabel('Real part x')
plt.ylabel('Imaginary part y')
plt.title('Mandelbrot Set for x+iy')
plt.savefig("Mandelbrot_plot.png")	


# for the colorful graph, we can alter the function as follows
def is_Mandelbrot_v2(cx,cy,x=0,y=0,start=0): # cx is the real part and cy is the imaginary part
	#print(x,y,start) 
	if((x**2 + y**2) > 4):
		return start # now it returns the number of iterations before the point leaves the Mandelbrot set
	elif(start > 100):
		return 0 # if the point belongs to the set, the returned value if zero 
	else:
		return is_Mandelbrot_v2(cx,cy,x**2-y**2+cx,2*x*y+cy,start+1)

iterations_list = [] # an empty list to store the number of iterations 

for i in range(N): # apply the new function to a set of random points and store the number of iterations
	iterations = is_Mandelbrot_v2(arrayx[i], arrayy[i])
	iterations_list.append(iterations)
	#print(max(iterations_list))
	
# a map functuion that gives color depending on value between 0 and the max number of iterations reached plus some constant	
def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=max(iterations_list)+20):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color	
    
for i in range(N):
	if(iterations_list[i] == 0): # to a get a better visual image, the points that belong to the set are assigned 
		                         # a higher value than the max number reached in the iterations
		iterations_list[i] = max(iterations_list)+20
	plt.scatter(arrayx[i], arrayy[i], c = color_map_color(iterations_list[i]), s = 10, linewidths=0)    
plt.xlabel('Real part x')
plt.ylabel('Imaginary part y')
plt.title('Mandelbrot Set for x+iy')
plt.savefig("Mandelbrot_color.png")	























