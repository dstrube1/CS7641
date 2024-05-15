#Problem_Set_1_Q2b.py
#Design a two-layer network of perceptrons that implements 𝐴 ⊕ 𝐵 (⊕ is XOR).

#Starting with https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a

import Problem_Set_1_Q2_util as util
import numpy as np

def A_AND_NOT_B_percep(A, B):
	return util.AND_percep(np.array([A, util.NOT_percep(B)]))

def NOT_A_AND_B_percep(A, B):
	return util.AND_percep(np.array([util.NOT_percep(A), B]))
    
def A_XOR_B_percep(A, B):
	#Doing this a little differently from how the towardsdatascience article does it
	#A XOR B = (A∧¬B) ∨(¬A∧B)
	#See this for details:
	#http://www.inf.unibz.it/~zini/ML/slides/ml_2012_lab_05_solutions.pdf
	A_AND_NOT_B = A_AND_NOT_B_percep(A, B)
	NOT_A_AND_B = NOT_A_AND_B_percep(A, B)
	return util.OR_percep(np.array([A_AND_NOT_B, NOT_A_AND_B]))

for A in range(2):
	for B in range(2):
		print("A: " + str(A) + "; B: " + str(B) + "; A⊕B: " + str(A_XOR_B_percep(A, B)))
