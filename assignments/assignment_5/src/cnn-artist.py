''' 
Import libraries
'''
# data tools
import os
#sys.path.append(os.path.join(".."))
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd
import re

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

''' 
Plot history model 
'''

def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("../output/performance.png")

'''
Load data samples
'''
def main(): 
    #Path to training folder with painters
    training_dir = os.path.join("..", "data", "training", "small_training")

    #Names as a string
    label_names = []
    #Training labels
    trainY = []

    #For the labels we find the painters 
    i = 0
    for folder in Path(training_dir).glob("*"):
        #find the painters name with regex
        painter = re.findall(r"(?!.*/).+", str(folder)) #re.findall returns a list 
        #append the list with painters
        label_names.append(painter[0])


        for img in folder.glob("*"):
            trainY.append(i)

        i +=1

    #Labels for validation
    #Path to training folder with painters
    validation_dir = os.path.join("..", "data", "validation", "small_validation")

    #test labels
    testY = []


    i = 0
    for folder in Path(validation_dir).glob("*"):
        for img in folder.glob("*"):
            testY.append(i)
        i +=1

    # integers to one-hot vectors
    lb = LabelBinarizer()
    # transform labels to binary codes 
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # Change size of images in small_training
    # The path were the images are located
    filepath = os.path.join("..","data", "training", "small_training")

    #create the list
    trainX=[]

    #loop through all the folders
    for folder in Path(filepath).glob("*"):
        #loop through all the files
        for file in Path(folder).glob("*"):
            #read images
            image_open = cv2.imread(str(file))
            #save dimensions
            dim = (120, 120)
            #resize images
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            #append images to trainX
            trainX.append(resize_image.astype("float") / 255.)

            # Change size of images in validation_training
    # The path were the images are located
    filepath = os.path.join("..","data", "validation", "small_validation")

    testX=[]
    # Same as before but with the validation images
    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            dim = (120, 120)
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            testX.append(resize_image.astype("float") / 255.)

    #Convert to numpy
    testX = np.array(testX)
    trainX = np.array(trainX)

    '''
    LeNet Model
    '''
    # define model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape=(120, 120, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))

    #Compile the model
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    plot_model(model, to_file = "../output/plot_model.png", show_shapes=True, show_layer_names=True)

    # train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=32,
                  epochs=20,
                  verbose=1)

    plot_history(H,20)

    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))

if __name__ == "__main__":
    main()