#1 Verify imports
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np

from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import time

import warnings

#allow plots to appear within the notebook:
%matplotlib inline

print("All imports are good")

###############################################################################

#2 Get the data
dataset1 = fetch_openml(name='phoneme') 
X_ds1 = dataset1.data #matrix
y_ds1 = dataset1.target #vector
X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1 = \
    train_test_split(X_ds1, y_ds1, test_size=0.2, random_state=0) 

dataset2 = fetch_openml(name='credit-g') 
X_ds2 = dataset2.data #matrix
y_ds2 = dataset2.target #vector
X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2 = \
    train_test_split(X_ds2, y_ds2, test_size=0.2, random_state=0) 

#Other global variables:
cv_ds1 = 0
cv_ds2 = 0
current_dataset = ''

print("Data gotten, and other global variables initialized")

###############################################################################

#3 Functions
def decisionTree(X_Train, y_Train):
    dtc = tree.DecisionTreeClassifier(random_state=0)
    scores = []
    cross_val_sizes = range(2,101)
    best_score = 0
    cv_ds = 0
    
    tic = time.time()
    
    for i in cross_val_sizes:
        score = cross_val_score(dtc, X_Train, y_Train, cv=i).mean()
        scores.append(score)
        if score > best_score:
            best_score = score
            cv_ds = i
            print(str(i) + " : " + str(score))
    
    toc = time.time()
    
    print("Decision Trees " + current_dataset + " best score = " + str(best_score) + " at cv = " + str(cv_ds) \
         + " in "+ str(int(toc - tic)) + " seconds")
    plt.plot(cross_val_sizes, scores)
    plt.xlabel("Cross validation size - dataset 1 (phoneme)")
    plt.ylabel("Cross validation score")
    
    return cv_ds, dtc

def decisionTreePrePruning(X_Train, y_Train, cv_ds):
    max_depth_sizes = range(2,21)
    scores = []
    best_score = 0
    max_depth_ds = 0
    
    tic = time.time()
    
    for md in max_depth_sizes:
        dtc = RandomForestClassifier(random_state=0, max_depth=md)
        score = cross_val_score(dtc, X_Train, y_Train, cv=cv_ds).mean()
        scores.append(score)
        if score > best_score:
            best_score = score
            max_depth_ds = md
            print(str(md) + " : " + str(score))
            
    toc = time.time()
    
    print("Decision Trees " + current_dataset + " with pre-pruning best score = " \
        + str(best_score) + " at max_depth = " + str(max_depth_ds) + " in "+ str(int(toc - tic)) + " seconds")
    plt.plot(max_depth_sizes, scores)
    plt.xlabel("Max depth size on RandomForestClassifier")
    plt.ylabel("Score")
    
    dtc = RandomForestClassifier(random_state=0, max_depth=max_depth_ds)
    
    return dtc

def decisionTreePostPruning(X_Train, y_Train, cv_ds):
    ccp_alpha_sizes = range(0,11)
    scores = []
    best_score = 0
    ccp_alpha_ds = 0

    tic = time.time()
    
    for ccpa in ccp_alpha_sizes:
        dtc = RandomForestClassifier(random_state=0, ccp_alpha=ccpa)
        score = cross_val_score(dtc, X_Train, y_Train, cv=cv_ds).mean()
        scores.append(score)
        if score > best_score:
            best_score = score
            ccp_alpha_ds = ccpa
            print(str(ccpa) + " : " + str(score))
    
    toc = time.time()
    
    print("Decision Trees " + current_dataset + " with post-pruning best score = " \
        + str(best_score) + " at ccp_alpha = " + str(ccp_alpha_ds) + " in "+ str(int(toc - tic)) + " seconds")
    plt.plot(ccp_alpha_sizes, scores)
    plt.xlabel("CCP alpha size on RandomForestClassifier")
    plt.ylabel("Score")
    
    dtc = RandomForestClassifier(random_state=0, ccp_alpha=ccp_alpha_ds)
    
    return dtc

def neuralNetworks(X_Train, y_Train, cv_ds):
    solvers = ['lbfgs','sgd','adam'] 
    best_score = 0
    solver_ds = ''
    
    tic = time.time()
    
    for my_solver in solvers:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                mlp = MLPClassifier(solver=my_solver, alpha=100.0, hidden_layer_sizes=(5, 2), random_state=0)
                score = cross_val_score(mlp, X_Train, y_Train, cv=cv_ds).mean()
                if score > best_score:
                    best_score = score
                    solver_ds = my_solver
                    print(my_solver + " : " + str(score))
            except Warning:
                #Ignore and skip 
                pass
    
    toc = time.time()
    
    print("Neural Network " + current_dataset + " optimum solver: "+ solver_ds \
          + " in "+ str(int(toc - tic)) + " seconds")
    print("")#newline to separate this run from the next
    
    my_alpha = 100.0
    best_score = 0
    alpha_ds = 0.0

    tic = time.time()
    
    while my_alpha <= 1e5:
        mlp = MLPClassifier(solver=solver_ds, alpha=my_alpha, hidden_layer_sizes=(5, 2), random_state=0)
        score = cross_val_score(mlp, X_Train, y_Train, cv=cv_ds).mean()
        if score > best_score:
            best_score = score
            alpha_ds = my_alpha
            print(str(my_alpha) + " : " + str(score))

        my_alpha *= 10 
    
    toc = time.time()
    
    print("Neural Network " + current_dataset + " optimum alpha order of magnitude: "+ str(alpha_ds) \
          + " in "+ str(int(toc - tic)) + " seconds")
    print("")#newline to separate this run from the next

    scores = []
    As = []
    Bs = []
    my_hidden_layer = (1,1)
    best_score = 0
    hiddle_layer_ds = (0,0)
    
    tic = time.time()
    
    for a in range(1,11):
        for b in range(1,11):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    mlp = MLPClassifier(solver=solver_ds, alpha=alpha_ds, \
                        hidden_layer_sizes=my_hidden_layer, random_state=0)
                    score = cross_val_score(mlp, X_Train, y_Train, cv=cv_ds).mean()
                    scores.append(score)
                    As.append(a)
                    Bs.append(b)
                    if score > best_score:
                        best_score = score
                        hiddle_layer_ds = my_hidden_layer
                        print(str(my_hidden_layer) + " : " + str(score))
                except Warning:
                    #Ignore and skip 
                    pass
            my_hidden_layer = (a,b)
        my_hidden_layer = (a,b)
    
    toc = time.time()
    
    print("Neural Network " + current_dataset + " optimum hidden layer: " + str(hiddle_layer_ds) \
          + " in "+ str(int(toc - tic)) + " seconds")
    
    """ 3d line plot:
    fig = plt.figure()
    axes = plt.axes(projection='3d')
    line = axes.plot3D(As,Bs,scores,'green')
    """
    
    #3d scatter plot:
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.scatter(As, Bs, scores)
    
    axes.set_xlabel('As')
    axes.set_ylabel('Bs')
    axes.set_zlabel('Scores')
    plt.show()
    
    mlp = MLPClassifier(solver=solver_ds, alpha=alpha_ds,  \
        hidden_layer_sizes=hiddle_layer_ds, random_state=0)
    
    return mlp

def boosting(X_Train, y_Train, cv_ds):
    scores = []
    n_range = range(1,101)
    best_score = 0
    n_estimators_ds = 0
    
    tic = time.time()
    
    for i in n_range:
        boost = AdaBoostClassifier(n_estimators=i, random_state=0)
        score = cross_val_score(boost, X_Train, y_Train, cv=cv_ds).mean()
        if (score > best_score):
            best_score = score
            n_estimators_ds = i
            print("n_estimators = " + str(i) + " : " + str(score))
        scores.append(score)
    
    toc = time.time()
    
    print("Boosting " + current_dataset + " optimum n_estimators: " + str(n_estimators_ds) \
        + " with score = " + str(best_score) + " in "+ str(int(toc - tic)) + " seconds")
    
    plt.plot(n_range, scores)
    plt.xlabel("Boosting n_estimators")
    plt.ylabel("Score")
    
    boost = AdaBoostClassifier(n_estimators=n_estimators_ds, random_state=0)
    
    return boost

def supportVectorMachines(X_Train, y_Train, cv_ds):
    SVMs = ['SVC','NuSVC', 'LinearSVC']
    my_SVMs = []
    my_SVMs.append(svm.SVC(random_state=0))
    my_SVMs.append(svm.NuSVC(random_state=0))
    my_SVMs.append(svm.LinearSVC(random_state=0))
    scores = []
    best_score = 0
    i = 0
    SVM_ds = ''
    chosen_svm_ds = None

    tic = time.time()
    
    for mySVM in my_SVMs:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                score = cross_val_score(mySVM, X_Train, y_Train, cv=cv_ds).mean()
                print(SVMs[i] + " score: " + str(score))
                if (score > best_score):
                    best_score = score
                    SVM_ds = SVMs[i]
                    chosen_svm_ds = mySVM
            except Warning:
                #Ignore and skip 
                score = 0
        scores.append(score)
        i += 1

    toc = time.time()
    
    print("Support Vector Machines " + current_dataset + " optimum SVM: " + \
        SVM_ds + " with score = " + str(best_score) + " in "+ str(int(toc - tic)) + " seconds")
    
    plt.plot(SVMs, scores)
    plt.xlabel("Support Vector Machine")
    plt.ylabel("Score")
    
    return chosen_svm_ds

def kNearestNeighbors(X_Train, y_Train, cv_ds):
    scores = []
    k_range = range(1,31)
    best_score = 0
    k_ds = 0
    
    tic = time.time()
    
    for i in k_range:
        knn_estimator = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn_estimator, X_Train, y_Train, cv=cv_ds).mean()
        scores.append(score)
        if score > best_score:
            best_score = score
            k_ds = i
            print("k = " + str(i) + " : " + str(score))
    
    toc = time.time()
    
    print("Best score:" + str(best_score) + " with k = " + str(k_ds) + " in "+ str(int(toc - tic)) + " seconds")
    plt.plot(k_range, scores)
    plt.xlabel("K-Nearest Neighbors")
    plt.ylabel("Score")
    
    knn_estimator = KNeighborsClassifier(n_neighbors=k_ds)
    
    return knn_estimator

def confusionMatrix(X_Train, X_Test, y_Train, y_Test, clf):
    clf.fit(X_Train, y_Train)
    plot_confusion_matrix(clf, X_Test, y_Test)  
    plt.show() 
    
def plot_learning_curve(estimator, X, y, cv_n_splits):
    
    train_sizes=np.linspace(.1, 1.0, 5)
    ylim=(0.7, 1.01)
    _, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes[0].set_title("Learning Curves")

    axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    
    cv = ShuffleSplit(n_splits=cv_n_splits, test_size=0.2, random_state=0)
    n_jobs=-1
    
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()
    
print("Functions have been defined")
    
###############################################################################

#4 Explore Dataset 1
current_dataset = "Dataset 1"
print(current_dataset + " target names:")
print(dataset1.target_names) 
print(current_dataset + " targets:")
print(np.unique(dataset1.target))
print(current_dataset + " data shape:")
print(X_ds1.shape) 
#print(current_dataset + " description:")
#print(dataset1.DESCR) 

#4.1 Decision Trees
cv_ds1, clf = decisionTree(X_Train_ds1, y_Train_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.1.1 Decision Tree with pre-pruning
clf = decisionTreePrePruning(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.1.2 Decision Tree with post-pruning
clf = decisionTreePostPruning(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.2 Neural Networks
clf = neuralNetworks(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.3 Boosting
clf = boosting(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.4 Support Vector Machines
clf = supportVectorMachines(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#4.5 K-Nearest Neighbors
clf = kNearestNeighbors(X_Train_ds1, y_Train_ds1, cv_ds1)
confusionMatrix(X_Train_ds1, X_Test_ds1, y_Train_ds1, y_Test_ds1, clf)
plot_learning_curve(clf, X_ds1, y_ds1, cv_ds1)

###############################################################################

#5 Explore Dataset 2
current_dataset = "Dataset 2"
print(current_dataset + " target names:")
print(dataset2.target_names) 
print(current_dataset + " targets:")
print(np.unique(dataset2.target))
print(current_dataset + " data shape:")
print(X_ds2.shape) 
#print(current_dataset + " description:")
#print(dataset2.DESCR) 

#5.1 Decision Trees
cv_ds2, clf = decisionTree(X_Train_ds2, y_Train_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.1.1 Decision Tree with pre-pruning
clf = decisionTreePrePruning(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.1.2 Decision Tree with post-pruning 
clf = decisionTreePostPruning(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.2 Neural Networks
clf = neuralNetworks(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.3 Boosting
clf = boosting(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.4 Support Vector Machines
clf = supportVectorMachines(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

#5.5 K-Nearest Neighbors
clf = kNearestNeighbors(X_Train_ds2, y_Train_ds2, cv_ds2)
confusionMatrix(X_Train_ds2, X_Test_ds2, y_Train_ds2, y_Test_ds2, clf)
plot_learning_curve(clf, X_ds2, y_ds2, cv_ds1)

###############################################################################

