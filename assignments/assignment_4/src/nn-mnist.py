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
from sklearn.preprocessing import LabelBinarizer


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
    # add argument for epoch iterations
    ap.add_argument("-ep", "--epoch", 
                    required = False, 
                    default = 100, 
                    type = float, 
                    help = "define iterations of epoch with (-ep), default is 100")
    
    ap.add_argument("-hl1", "--hidden_layer1", 
                    required = False, 
                    default = 32, 
                    type = float, 
                    help = "define number of hidden layer 1 with (-hl1), default is 32")
     
    ap.add_argument("-hl2", "--hidden_layer2", 
                    required = False, 
                    default = 32, 
                    type = float, 
                    help = "define number of hidden layer 2 with (-hl2), default is 16")
    args = vars(ap.parse_args())
    
                    
    train_size = args["train_size"]
    test_size = args["test_size"]
    epoch = args["epoch"]
    hl1 = args["hidden_layer1"]
    hl2 = args["hidden_layer2"]
                    
    '''
    Neural Network model
    '''
    ## We could use the same dataset as with the regression logistic model, but this however takes way too long which is why this commented 
    #Fetch data 
    #X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

    #Convert to numpy arrays
    #X = np.array(X)
    #y = np.array(y)

    # MinMax regularization
    #X = (X - X.min())/(X.max() - X.min())
    # Create training data and test dataset 
    #X_train, X_test, y_train, y_test = train_test_split(X, # our data
                                                     #   y, # our labels
                                                     #   train_size=train_size, 
                                                     #   test_size=test_size)
    #Instead we use the dataset from sklearn            
    digits = datasets.load_digits()
    # Convert to floats
    data = digits.data.astype("float")
    # split data
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                  digits.target, 
                                                  test_size=test_size,
                                                  train_size=train_size)
    
    # convert labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    
    # train network
    print("[INFO] training network...")
    nn = NeuralNetwork([X_train.shape[1], hl1, hl2, 10])
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=epoch)
    
    # evaluate network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    
if __name__ == "__main__":
    main()