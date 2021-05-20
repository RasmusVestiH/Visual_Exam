import sys,os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wget
from pathlib import Path
import re
import pydot

#Import shutil and random
import shutil
import random
from shutil import copyfile

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow import * 
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,
                                     GlobalMaxPool1D,
                                     MaxPooling2D, 
                                     Activation,
                                     Dropout,
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2


def main():
    #Path to data
    input_file = os.path.join("..", "data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")
    
    #-----------------------------------Cleaning the data:-----------------------------------------
    
    #specifying genres based on mixed genres (Example: Western|Drama -> Western etc.)
    data["Genre"]=data.Genre.str.replace(r'(^.*Animation.*$)', 'Animation')
    data["Genre"]=data.Genre.str.replace(r'(^.*Western.*$)', 'Western')
    data["Genre"]=data.Genre.str.replace(r'(^.*Sci-Fi.*$)', 'Sci-Fi')
    data["Genre"] = data["Genre"].replace(to_replace='Drama|Romance', value='Romance', regex=False)
    
    #Getting all unique cathegories from the dataset:
    unique_data = data.Genre.unique()

    #Iterating through the unique cathegories.
    unique_cathegories = []
    for cat in unique_data:

        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            unique_cathegories.append(cat)
       
    one_Genre_df = data[data.Genre.isin(unique_cathegories)]
    #Resetting the indexes of the new df.
    one_Genre_df = one_Genre_df.reset_index()
    # Replace the whitespaces in the titles with a underscore
    one_Genre_df["Title"] = one_Genre_df["Title"].str.replace(pat=" ", repl="_")
    
    # Making a final dataframe with the genres intended to be used in the LeNet CNN Model
    final_df = one_Genre_df[one_Genre_df.Genre.isin(["Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western", "Romance", "Animation"])]
    final_df = final_df.reset_index()
    final_df["Genre"].value_counts()
    
    #-----------------------------------Creating trainY/X and testY/X:-------------------------------
    #Path to training folder with painters
    training_dir = os.path.join("..", "data","poster_data", "train")

    #Names as a string
    label_names = []
    #Training labels
    trainY = []

    #For the labels we find the painters 
    i = 0
    for folder in Path(training_dir).glob("*"):
        #find the painters name with regex
        movie = re.findall(r"(?!.*/).+.*", str(folder)) #re.findall returns a list 
        #append the list with posters
        label_names.append(movie[0])


        for img in folder.glob("*"):
            trainY.append(i)

        i +=1
    
    #Labels for test
    #Same procedure as above but this time with the test folder instead of train. 
    test_dir = os.path.join("..", "data","poster_data", "test")

    #test labels
    testY = []


    i = 0
    for folder in Path(test_dir).glob("*"):
        for img in folder.glob("*"):
            testY.append(i)
        i +=1
        
       
    #Create trainX and change dimensions to 256, 160 of all images in training
    # The path were the images are located
    filepath = os.path.join("..","data", "poster_data", "train")

    #create the list
    trainX=[]

    #loop through all the folders
    for folder in Path(filepath).glob("*"):
        #loop through all the files
        for file in Path(folder).glob("*"):
            #read images
            image_open = cv2.imread(str(file))
            #save dimensions
            dim = (256, 160) #(268,182 = original dimensions)
            #resize images
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            #append images to trainX
            trainX.append(resize_image.astype("float") / 255.)

    # Same as above but with the testX
    # The path were the images are located
    filepath = os.path.join("..","data", "poster_data", "test")

    testX=[]
    # Same as before but with the validation images
    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            dim = (256, 160)
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            testX.append(resize_image.astype("float") / 255.)
           
       
    # integers to one-hot vectors
    lb = LabelBinarizer()
    # transform labels to binary codes 
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    #Convert to numpy
    testY = np.array(testY)
    trainY = np.array(trainY)
    #Convert to numpy
    testX = np.array(testX)
    trainX = np.array(trainX)
    
    #------------------------------------Plotting the LeNet CNN model--------------------------
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
        fig.savefig("../output/LeNet_performance.png")
    #Clear session before to make sure we don't continue feeding the model
    tf.keras.backend.clear_session()
    
    #Define model as being Sequential and add layers
    model = Sequential()
    # First set of CONV => RELU => POOL
    model.add(Conv2D(64, (3, 3),  
                     padding="same", 
                     input_shape=(160, 256, 3))) #The shape of all the posters with width, height and dimensions
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #View the summary
    model.summary()

    #Second set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), #NB: the filter is set to 32 and the kernel to 3x3
                    padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #Dropout to challenge the model
    model.add(Dropout(0.2))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(16)) 
    model.add(Activation("relu"))

    # Softmax classifier
    model.add(Dense(8))  #NB: the filter is set to 8 which is the number of unique labels
    model.add(Activation("softmax"))

    # Compile model
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Save model summary as model_architecture.png
    plot_model(model, to_file = "model_architecture.png", show_shapes=True, show_layer_names=True)

    # Train the model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=10,
                  epochs=10, #NB: could be set as a paramenter with argparse
                  verbose=1)

    # Plot and save history via the earlier defined function
    plot_history(H, 10) #NB: epochs(10) could be set as a paramenter

    # Save model summary as model_architecture.png
    plot_model(model, to_file = "../output/LeNet_model_architecture.png", show_shapes=True, show_layer_names=True)
    
    predictions = model.predict(testX, batch_size=10)
    print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names))
    
    #Writing results to txt file.
    txt_file = open("../output/CNN_model_result.txt", "a")
    txt_file.write(classification_report(testY.argmax(axis=1),
                                     predictions.argmax(axis=1),
                                     target_names=label_names))
    txt_file.close()
if __name__ == "__main__":
    main()
