import numpy as np
import matplotlib.pyplot as plt

mean = 0
SD = 3

# the normalized Gaussian curve 
def Gaussian_dist(value):
	return ((1 / (np.sqrt(2 * np.pi) * SD)) * np.exp(-0.5 * (1 / SD * (value - mean))**2))

x = np.linspace(-10, 10, 1000)
y = Gaussian_dist(x)

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Normalized Gaussian Distribution \n with (mean,SD)=(0,3)')
plt.savefig("gaussian.png")
plt.show()

# OR by creating data that is normally distributed
# values = np.random.normal(mean, SD, 100000)
# plt.hist(values, 1000)
# plt.show()