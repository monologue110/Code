#########################################################
## Stat 413 - Homework 3
## Author: Luxi Li
## Date : 2/18/2018
## Description: This script implements an adaboost classifier
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names,
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to
## double-check your work, but MAKE SURE TO COMMENT OUT ALL
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "os.chdir" anywhere
## in your code. If you do, I will be unable to grade your
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################

import math
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split


def prepare_data(valid_digits = np.array((6, 5))):

    ## valid_digits is a vector containing the digits
    ## we wish to classify.
    ## Do not change anything inside of this function
    if len(valid_digits) != 2:
        raise Exception("Error: you must specify exactly 2 digits for classification!")

    data = ds.load_digits()
    labels = data['target']
    features = data['data']

    X = features[(labels == valid_digits[0]) | (labels == valid_digits[1]), :]
    Y = labels[(labels == valid_digits[0]) | (labels == valid_digits[1]),]

    X = np.asarray(list(map(lambda k: X[k,:]/X[k,:].max(), range(0,len(X)))))

    Y[Y == valid_digits[0]] = 0
    Y[Y == valid_digits[1]] = 1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_test = Y_test.reshape((len(Y_test), 1))

    return X_train, Y_train, X_test, Y_test


##########################
## Function 1: Adaboost ##
##########################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Use Adaboost to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations=200):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## num_iterations: Number of iterations.

    ## Function should learn the parameters of an Adaboost classifier.

    n = X_train.shape[0]
    p = X_train.shape[1]
    threshold = 0.8

    X_train1 = 2 * (X_train > threshold) - 1
    Y_train = 2 * Y_train - 1

    X_test1 = 2 * (X_test > threshold) - 1
    Y_test = 2 * Y_test - 1

    beta = np.repeat(0., p).reshape((p, 1))
    w = np.repeat(1. / n, n).reshape((n, 1))

    weak_results = np.multiply(Y_train, X_train1) > 0

    acc_train = np.repeat(0., num_iterations, axis=0)
    acc_test = np.repeat(0., num_iterations, axis=0)

    #######################
    ## FILL IN CODE HERE ##
    #######################

    for it in range(num_iterations):
        weighted_weak_results = np.multiply(w, weak_results)
        weighted_accuracy = np.sum(weighted_weak_results, axis = 0)
        e = 1 - weighted_accuracy
        j = np.argmin(e)
        dbeta = 0.5* math.log((1 - e[j]) / e[j])
        beta[j] = beta[j] + dbeta
        w = np.multiply(w, np.exp(-np.multiply(np.multiply(Y_train, X_train1[:, j:(j+1)]), dbeta)))
        w = w/np.sum(w)
        acc_train[it] = np.mean(np.multiply(np.sign(np.dot(X_train1, beta)), Y_train) >0)
        #print acc_train[it]
        #acc_test[it] = np.mean(np.sign(np.dot(X_test1, beta)) == Y_test)
        acc_test[it] = np.mean(np.multiply(np.sign(np.dot(X_test1, beta)), Y_test) > 0 )






    ## Function should output 3 things:
    ## 1. The learned parameters of the adaboost classifier, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    return beta, acc_train, acc_test


############################################################################
## Testing your functions and visualize the results here##
############################################################################

X_train, Y_train, X_test, Y_test = prepare_data()

beta, acc_train, acc_test = my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations = 300)
#print(beta)
#print(acc_train)
#print(acc_test)
print('median accuracy when number of iterations is 300:')
print('test accuracy:')
print(np.median(acc_test))
print('train accuracy:')
print(np.median(acc_train))


beta, acc_train, acc_test = my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations = 100)
print('accuracy in iterations of 1000:')
print('maximum accuracy in testing data:')
print(np.max(acc_test))
print('maximum accuracy in training data:')
print(np.max(acc_train))
print('median accuracy in testing data:')
print(np.median(acc_test))
print('median accuracy in training data:')
print(np.median(acc_train))


print(acc_train)
print(acc_test)

####################################################
## Optional examples (comment out your examples!) ##
####################################################




