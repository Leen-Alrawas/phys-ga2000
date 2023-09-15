# PS2 - Grad Compuatational Physics 
# By Leen Alrawas

import timeit
import_module = '''
import numpy as np
import matplotlib.pyplot as plt
import math
'''

test_for_loop = '''

# PROBLEM 2

L = 200

# METHOD 1

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
'''

test_map = '''

# METHOD 2

L = 200

result = []
list(map(lambda a: list(map(lambda b: list(map(lambda c: result.append((a, b, c)), range(-L,L+1))), range(-L,L+1))), range(-L,L+1)))

def potential2(listt):
	if listt[0]**2+listt[1]**2+listt[2]**2 != 0:
		if ((listt[0]+listt[1]+listt[2])%2 == 0):
			return 1/(math.sqrt(listt[0]**2+listt[1]**2+listt[2]**2))
		else:
			return -1/(math.sqrt(listt[0]**2+listt[1]**2+listt[2]**2))
	else: 
		return 0

sum2 = np.sum(np.array(list(map(potential2, result))))
'''

print("Time in for loops method: ",timeit.timeit(stmt=test_for_loop, setup=import_module, number=1))
print("Time in maps method: ",timeit.timeit(stmt=test_map, setup=import_module, number=1))