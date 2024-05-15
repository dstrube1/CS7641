#optimizeQuiz1.py

import math

bestX = 0
bestY = 0
for x in range (1,101):
	testY = math.pow((x%6),2)%7 - math.sin(x)
	print("x: " + str(x) + " = " + str(testY))
	if testY > bestY:
		bestY = testY
		bestX = x
		
print("bestX: " + str(bestX))

"""
this method fails:
bestX = 0
bestY = 0
for x in range (-1000001,1000001):
	testY = -(x^4) + (1000 * (x^3)) - (20 * (x^2)) + (4*x) - 6
	if testY > bestY:
		bestY = testY
		bestX = x
		
print("bestX: " + str(bestX))

must take derivative, graph it, find highest point:
somewhere around 750
"""
