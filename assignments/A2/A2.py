#A2.py
#Starting point for A2

import pandas as pd
import numpy as np

#import sklearn
#A1 dataset:
#https://www.openml.org/d/1489

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

dataset = fetch_openml(name='phoneme') 
X = dataset.data #matrix
y = dataset.target #vector

#Split data into train and test subsets:
X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize feature data
scaler = MinMaxScaler()
# One hot encode target values
one_hot = OneHotEncoder()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Note, since the targets are already split up into Class 1 or 2, 
#this OneHotEncoding may be unnecessary / redundant, 
#but still, it's good practice as a beginner
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

'''
#Exploring the data
print("target_names:")
print(dataset.target_names) 
print("targets:")
print(np.unique(dataset.target))
print("data shape:")
print(X.shape) 
#print("description:")
#print(dataset.DESCR) 

print("first line of X_train:")
for x_line in X_train:
	print(x_line)
	break

print("first line of X_train_scaled:")
for x_line in X_train_scaled:
	print(x_line)
	break

print("first line of y_train, and count:")
printed_first = False
i = 1;
for x_line in y_train_hot:
	if not printed_first:
		print(x_line)
	printed_first = True
	i += 1
print("count: " + str(i))

print("first line of y_train_hot, and count:")
printed_first = False
i = 1;
for x_line in y_train_hot:
	if not printed_first:
		print(x_line)
	printed_first = True
	i += 1
print("count: " + str(i))

data_csv = pd.read_csv('https://www.openml.org/data/get_csv/1592281/php8Mz7BG')
print("head from csv:")
print(data_csv.head())
print("tail:")
print(data_csv.tail())

print("head from X_train_scaled:")
data_scaled = pd.DataFrame(X_train_scaled)
#more pd stuff: https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html
print(data_scaled.head())
print("tail:")
print(data_scaled.tail())
'''

#On to mlrose
import mlrose_hiive as mlrose
#print("mlrose")#: " + mlrose.__version__)
#no __version__ attribute :( 
#2.1.3?

nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[9], activation='tanh', algorithm='random_hill_climb', max_iters=1000, bias=True, curve=True, is_classifier=True, learning_rate=10.0, early_stopping=True, clip_max=5, max_attempts=100, random_state=0)

nn_model1.fit(X_train_scaled, y_train_hot)

from sklearn.metrics import accuracy_score

# Predict labels for train set and assess accuracy
y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

print("y_train_accuracy: ")
print(y_train_accuracy)

#Once optimals are found:
# Predict labels for test set and assess accuracy
y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

print("y_test_accuracy:")
print(y_test_accuracy)

"""
fitness = mlrose.Queens()
# Define alternative N-Queens fitness function for maximization problem
def queens_max(state):

	# Initialize counter
	fitness_cnt = 0
	
	# For all pairs of queens
	for i in range(len(state) - 1):
		for j in range(i + 1, len(state)):
			# Check for horizontal, diagonal-up and diagonal-down attacks
			if (state[j] != state[i]) \
				and (state[j] != state[i] + (j - i)) \
				and (state[j] != state[i] - (j - i)):
				# If no attacks, then increment counter
				fitness_cnt += 1

	return fitness_cnt

# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(queens_max)
problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)
# Define decay schedule
schedule = mlrose.ExpDecay()

# Define initial state
init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])

# Solve problem using simulated annealing
best_state, best_fitness, _ = mlrose.simulated_annealing(problem, schedule = schedule,
	max_attempts=10, max_iters=1000,init_state = init_state,random_state = 1)

print(best_state)

print(best_fitness)

print(fitness)

#import scipy
#print("scipy: " + scipy.__version__)
#scipy: 1.5.2

#import matplotlib
#print("matplotlib: " + matplotlib.__version__)
#matplotlib: 3.3.2

#https://mlrose.readthedocs.io/en/stable/source/algorithms.html

#randomized hill climbing
#mlrose.random_hill_climb(None)

#simulated annealing
#mlrose.simulated_annealing(None)
#geometric decay, arithmetic decay or exponential decay.

#genetic algorithm
#mlrose.genetic_alg(None)

#MIMIC
#mlrose.mimic(None)

three optimization problem domains

1- highlight advantages of your genetic algorithm
2- of simulated annealing, 
3- of MIMIC

4-peaks 
k-color 
integer-string optimization problems, such as N-Queens and the Knapsack problem; continuous-valued optimization problems, such as the neural network weight problem; and tour optimization problems, such as the Travelling Salesperson problem

Pre-defined fitness functions exist for solving the: One Max, Flip Flop, Four Peaks, Six Peaks, Continuous Peaks, Knapsack, Travelling Salesperson, N-Queens and Max-K Color optimization problems

https://mlrose.readthedocs.io/en/stable/source/tutorial1.html
"""
