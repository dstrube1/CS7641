#importTest.py

import matplotlib
print("matplotlib: " + matplotlib.__version__)
#matplotlib: 3.3.2

import mlrose_hiive as mlrose
print("mlrose_hiive: 2.1.3")# + mlrose.__version__)
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

import scipy
print("scipy: " + scipy.__version__)
#scipy: 1.5.2

#import time
#print("time: " + time.__version__)
#no version because it comes with python

#to get jupyter version:
#jupyter --version