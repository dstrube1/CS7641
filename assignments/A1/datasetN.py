#datasetN.py

import numpy as np

from sklearn.datasets import fetch_openml

#current_dataset = 'phoneme'
#https://www.openml.org/d/1489
#url = "https://www.openml.org/data/get_csv/1592281/php8Mz7BG"

current_dataset = 'credit-g'
#https://www.openml.org/d/31
url = "https://www.openml.org/data/get_csv/31/dataset_31_credit-g.arff"

dataset1 = fetch_openml(name=current_dataset) 
X_ds1 = dataset1.data #matrix
y_ds1 = dataset1.target #vector

print(current_dataset + " target names:")
print(dataset1.target_names) 
print(current_dataset + " targets:")
print(np.unique(dataset1.target))
print(current_dataset + " data shape:")
print(X_ds1.shape) 

print(current_dataset + " description:")
print(dataset1.DESCR) 

import pandas as pd
data = pd.read_csv(url)
print(data.head())
for headerCol in data:
	print(data[headerCol])
	print(type(data[headerCol][0]))
	thisType = type(data[headerCol][0])
	print(len(data[headerCol]))
	for i in range(len(data[headerCol])):
		if type(data[headerCol][i]) != thisType:
			#Problem at row 8192?
			print("Problem with headerCol " + headerCol + " at row " + str(i) +"; headerCol type: " + str(thisType) + "; this row type: " + str(type(data[headerCol][i])) +"; value: " + str(data[headerCol][i]))			
		print(data[headerCol][i])
		break
	break
	
print("done")

