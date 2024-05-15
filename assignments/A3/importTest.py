#importTest.py

import matplotlib
print("matplotlib: " + matplotlib.__version__)
#matplotlib: 3.3.2

#import mlrose_hiive as mlrose
#print("mlrose_hiive: 2.1.3")# + mlrose.__version__)
#no __version__ attribute :( 
#2.1.3?

import numpy as np
print("numpy: " + np.__version__)
#numpy: 1.19.2

#import random 
#print("random: " + random.__version__)
#no version because it comes with python

import sklearn
print("sklearn: " + sklearn.__version__)
#0.23.2

#import scipy
#print("scipy: " + scipy.__version__)
#scipy: 1.5.2

#import time
#print("time: " + time.__version__)
#no version because it comes with python

#to get jupyter version:
#jupyter --version

#import tqdm
#print("tqdm: " + tqdm.__version__)
#4.50.2
"""from tqdm import tqdm
x = 0
for i in tqdm(range(10000)):
	for j in range(100):
		for k in range(100):
			x = j + k
print("done looping")"""
#Useful at command line, not so much in Jupyter notebooks

import hyperopt 
print("hyperopt: " + hyperopt.__version__)
#import hpsklearn 
#from hpsklearn import HyperoptEstimator
#print("hpsklearn: ")

#NFLT probability approximation
def prob_approx():
    p = 0.001
    X1s = []
    X2s = []
    Ys = []
    y = 1
    while p <= 1:
        for n in range(10):
            x1 = 1 - math.pow((1-p), n)
            x2 = n * p
            X1s.append(x1)
            X2s.append(x2)
            Ys.append(y)
            y += 1
        p += p 
        #this doesn't work as well as above:
        #p += 0.001
    plt.plot(X1s, Ys)
    plt.title("1 - (1 - p)^n")
    plt.xlabel("result")
    plt.ylabel("iteration")
    plt.show()
    
    plt.plot(X2s, Ys)
    plt.title("n * p")
    plt.xlabel("result")
    plt.ylabel("iteration")
    plt.show()
#prob_approx()