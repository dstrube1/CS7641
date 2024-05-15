#Problem_Set_1_Q2_util.py
#Code common to Problem Set 1 Question 2a & 2b

#Starting with https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a

import numpy as np

def unit_step(s):
	#Heavyside Step function. s must be a scalar
	if s >= 0:
		return 1
	else:
		return 0
	
def perceptron(x, w, b):
    #Function implemented by a perceptron with weight vector w and bias b 
	s = np.dot(w, x) + b
	y = unit_step(s)
	return y

def NOT_percep(x):
	return perceptron(x, w=-1, b=0.5)
	
def AND_percep(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)

def OR_percep(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

if __name__ == '__main__':
	print("This is a utility file, not intended to be run directly.")