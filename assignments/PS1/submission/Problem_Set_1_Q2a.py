#Problem_Set_1_Q2a.py
#Design a two-input perceptron that implements the boolean function 𝐴∧¬𝐵.

#Starting with https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a

import Problem_Set_1_Q2_util as util
import numpy as np
    
def A_AND_NOT_B_percep(A, B):
	return util.AND_percep(np.array([A, util.NOT_percep(B)]))

for A in range(2):
	for B in range(2):
		print("A: " + str(A) + "; B: " + str(B) + "; A∧¬B: " + str(A_AND_NOT_B_percep(A, B)))