# PS8 - Grad Compuatational Physics 
# By Leen Alrawas

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.fft import fft, ifft


# PROBLEM 1
print("\n")
print("PROBLEM 1\n")

# plot the signal 
def plot_waveform(signal, figname):
	plt.plot(signal)
	plt.title('Waveform')
	plt.xlabel('Sample')
	plt.ylabel('Amplitude')
	plt.savefig(figname)
	plt.show()

def plot_fft_magnitudes(signal, figname, num_coefficients=10000, rate=44100):
    fft_result = fft(signal) # the fast fourier transform of the signal
    magnitudes = np.abs(fft_result)[:num_coefficients] # the first 10,000 magnitudes
    # fftfreq Returns the Discrete Fourier Transform sample frequencies
    frequencies = np.fft.fftfreq(len(fft_result), d=1/rate)[:num_coefficients]

    plt.plot(frequencies, magnitudes)
    plt.title('FFT Magnitudes')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.savefig(figname)
    plt.show()
    
    # check that the inverse gives the siganl back
    reconstructed_waveform = ifft(fft_result).real
    plt.plot(reconstructed_waveform)
    plt.title('Reconstructed Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()

# the piano signal
waveform_piano = np.loadtxt("piano.txt", dtype=float) 
plot_waveform(waveform_piano, "piano")    
plot_fft_magnitudes(waveform_piano, "piano_fft")

# the trumpet signal
waveform_trumpet = np.loadtxt("trumpet.txt", dtype=float) 
plot_waveform(waveform_trumpet, "trumpet")    
plot_fft_magnitudes(waveform_trumpet, "trumpet_fft")


# PROBLEM 2
print("\n")
print("PROBLEM 2\n")

# using the multivariable version of the fourth-order Runge-Kutta

# define the equations
def f(r,t, sigma=10, rr=28, bb=float(8/3)):
	x = r[0]
	y = r[1]
	z = r[2]
	fx = sigma * (y - x)
	fy = rr*x - y - x*z
	fz = x*y - bb*z
	return np.array([fx,fy,fz] ,float)

# define the interval
a = 0.0
b = 50.0
N = 1000
h = (b-a)/N

tpoints = np.arange(a,b,h)
xpoints = []
ypoints = []
zpoints = []

# apply runge-kutta method using initial conditions
r = np.array([0.0,1.0,0.0] ,float)
for t in tpoints:
	xpoints.append(r[0])
	ypoints.append(r[1])
	zpoints.append(r[2])
	k1 = h*f(r,t)
	k2 = h*f(r+0.5*k1,t+0.5*h)
	k3 = h*f(r+0.5*k2,t+0.5*h)
	k4 = h*f(r+k3,t+h)
	r += (k1+2*k2+2*k3+k4)/6

plt.plot(tpoints,ypoints)
plt.xlabel('t')
plt.ylabel('y')
plt.savefig("y-t plot")
plt.show()

plt.plot(xpoints,zpoints)
plt.xlabel('x')
plt.ylabel('z')
plt.savefig("z-x plot")
plt.show()
