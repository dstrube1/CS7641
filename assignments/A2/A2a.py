#A2a.py
'''
four local random search algorithms:

1- randomized hill climbing
2- simulated annealing
3- genetic algorithm
4- MIMIC

four search techniques to these three optimization problems:

N-Queens optimization problem
https://mlrose.readthedocs.io/en/stable/source/fitness.html#mlrose.fitness.Queens
For the N-Queens optimization problem, I took a 9-Queen setup and used it for each of the four search techniques.

Knapsack optimization problem
https://mlrose.readthedocs.io/en/stable/source/fitness.html#mlrose.fitness.Knapsack
For the Knapsack optimization problem, 

Max-k color optimization problem
https://mlrose.readthedocs.io/en/stable/source/fitness.html#mlrose.fitness.MaxKColor

3- SixPeaks
https://mlrose.readthedocs.io/en/stable/source/fitness.html#mlrose.fitness.SixPeaks
^MIMIC
'''

import mlrose_hiive as mlrose
import numpy as np

#N-Queens
# Define initial state
init_state = np.array(range(9))

fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness, maximize=False, \
	max_val=len(init_state))
# Define decay schedule
schedule = mlrose.ExpDecay()

# Solve problem using simulated annealing
best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule = schedule, \
            max_attempts = 10, max_iters = 1000, init_state = init_state, random_state = 1)

print("best_state:")
print(best_state)

print("best_fitness:")
print(best_fitness)

#Knapsack
init_state = np.array([1, 0, 2, 1, 0])
weights = [10, 5, 2, 8, 15]
values = [1, 2, 3, 4, 5]
max_weight_pct = 0.6
fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness, maximize=True, \
	max_val=sum(weights))

best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule = schedule, \
            max_attempts = 10, max_iters = 1000, init_state = init_state, random_state = 1)

print("best_state:")
print(best_state)

print("best_fitness:")
print(best_fitness)

#Max-k color
init_state = np.array([0, 1, 0, 1, 1])
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
fitness = mlrose.MaxKColor(edges)

problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness, maximize=True, \
	max_val=len(init_state))

best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule = schedule, \
            max_attempts = 10, max_iters = 1000, init_state = init_state, random_state = 1)

print("best_state:")
print(best_state)

print("best_fitness:")
print(best_fitness)

"""
This is good to know how to do, but bad for reproducibility for this assignment
import random
def getRandomBits(size):
	array = []
	for i in range(size):
		array.append(random.randint(0, 1))
	return array

rando_array = getRandomBits(100)
print(rando_array)
"""
init_state = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
fitness = mlrose.SixPeaks(t_pct=0.1)
problem = mlrose.DiscreteOpt(length=len(init_state), fitness_fn=fitness, maximize=True, \
	max_val=2)

_, best_fitness, fitness_curve = mlrose.simulated_annealing(problem, schedule = schedule, \
            max_attempts = 10, max_iters = 300, init_state = init_state, random_state = 1, curve=True)

print("best_fitness:")
print(best_fitness)

print("fitness_curve:")
print(fitness_curve)

