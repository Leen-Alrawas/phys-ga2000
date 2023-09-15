# PS2 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import rand
 
# PROBLEM 4
print("\n")
print("PROBLEM 4\n")

# parts a and b 
'''
def quadratic_1(a, b, c):
	sqrtt = math.sqrt(b**2 - 4*a*c)
	return ((-b+sqrtt)/(2*a), (-b-sqrtt)/(2*a))

print(quadratic_1(0.001,1000,0.001))
#print("(%10.3E,%10.3E)" %(quadratic(0.001,1000,0.001)))

def quadratic_2(a, b, c):
	sqrtt = math.sqrt(b**2 - 4*a*c)
	return ((2*c)/(-b-sqrtt), (2*c)/(-b+sqrtt))
	
print(quadratic_2(0.001,1000,0.001))
'''

def quadratic(a, b, c):
	sqrtt = math.sqrt(b**2 - 4*a*c)
	if(abs(b - sqrtt) < 0.001):
		x1 = (-b-sqrtt)/(2*a)
		return ((c)/(a*x1), x1)
	else:
		x1 = (-b+sqrtt)/(2*a)
		return (x1, (c)/(a*x1))

'''
print(quadratic(0.001,1000,0.001))
print(quadratic(0.001,-1000,0.001))
'''