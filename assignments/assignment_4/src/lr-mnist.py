#!/usr/bin/env python

'''
Import libraries
'''
#path tools
import sys,os
sys.path.append(os.path.join(".."))

import argparse
import numpy as np
import utils.classifier_utils as clf_util
#neural networks with numpy
from utils.neuralnetwork import NeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



'''
Main function
'''

def main():
    # Create an argument parser 
    ap = argparse.ArgumentParser(description = "[INFO] Classify MNIST data and print performance")
    
    # add arguments for train size 
    ap.add_argument("-trs", "--train_size", 
                    required = False, 
                    default = 0.8, 
                    type = float, 
                    help = "define size of train with (-trs), default is 0.8")
             
    # add arguments for test size
    ap.add_argument("-tst", "--test_size", 
                    required = False, 
                    default = 0.2, 
                    type = float, 
                    help = "define size of test with (-tst), default is 0.2")
    
    args = vars(ap.parse_args())
    
                    
    train_size = args["train_size"]
    test_size = args["test_size"]
                    
                    
    '''
    Logistic regression
    '''
    #Fetch data 
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

    #Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Create training data and test dataset 
    X_train, X_test, y_train, y_test = train_test_split(X, # our data
                                                        y, # our labels
                                                        train_size=train_size, 
                                                        test_size=test_size)
    #scaling the features 
    #scaling the features
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0

    # train a logistic regression model
    clf = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train_scaled, y_train)
    #Predict test data
    y_pred = clf.predict(X_test_scaled)

    cm = metrics.classification_report(y_test, y_pred)
    print(cm)
    
if __name__ == "__main__":
    main()