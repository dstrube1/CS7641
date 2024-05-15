#numpyTest.py
#from https://www.youtube.com/watch?v=qsIrQi0fzbY&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=17
import numpy as np

print("I can import numpy: " + np.__version__)
#numpy: 1.18.1

import sklearn
print("I can import sklearn: " + sklearn.__version__)
#sklearn: 0.23.2

import scipy
print("I can import scipy: " + scipy.__version__)
#scipy: 1.4.1

import matplotlib
print("I can import matplotlib: " + matplotlib.__version__)
#matplotlib: 3.1.2

#a = np.array([1,2,3,4])
#print(a)

import time

max = 1_000_000

a = np.random.rand(max)
b = np.random.rand(max)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print("Vectorized: " + str(1000 * (toc - tic)) + "ms")

c = 0
tic = time.time()
for i in range(max):
	c += a[i] * b[i]
toc = time.time()

print("Non-Vectorized: " + str(1000 * (toc - tic)) + "ms")

import matplotlib.pyplot as plt

