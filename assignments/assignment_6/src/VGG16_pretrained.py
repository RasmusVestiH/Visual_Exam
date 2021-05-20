#Import shutil and random
import os, sys
import shutil
import random
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd
import re

# tf tools
import tensorflow as tf
from tensorflow.keras.utils import plot_model

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers import SGD, Adam

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow import * 
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     GlobalMaxPool1D,
                                     MaxPooling2D, 
                                     Activation,
                                     Dropout,
                                     Flatten, 
                                     Dense)

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import L2

def main():
    #Path to data
    input_file = os.path.join("..", "data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")

    #specifying genres based on mixed genres (Example: Western|Drama -> Western etc.)
    data["Genre"]=data.Genre.str.replace(r'(^.*Animation.*$)', 'Animation')
    data["Genre"]=data.Genre.str.replace(r'(^.*Western.*$)', 'Western')
    data["Genre"]=data.Genre.str.replace(r'(^.*Sci-Fi.*$)', 'Sci-Fi')
    data["Genre"] = data["Genre"].replace(to_replace='Drama|Romance', value='Romance', regex=False)

    #Getting all unique cathegories from the dataset:
    unique_data = data.Genre.unique()

    #Iterating through the unique cathegories.
    unique_categories = []
    for cat in unique_data:
        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            unique_categories.append(cat)

    one_Genre_df = data[data.Genre.isin(unique_categories)]

    # Resetting the indexes of the new df.
    one_Genre_df = one_Genre_df.reset_index()

    # Replace the whitespaces in the titles with a underscore
    one_Genre_df["Title"] = one_Genre_df["Title"].str.replace(pat=" ", repl="_")
    
    #Making the final dataframe 
    final_df = one_Genre_df[one_Genre_df.Genre.isin(["Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western", "Romance", "Animation"])]
    final_df = final_df.reset_index()
    final_df["Genre"].value_counts()

    #Path to training folder with painters
    training_dir = os.path.join("..", "data","poster_data", "train")

    #Names as a string
    label_names = []
    #Training labels
    trainY = []

    #For the labels we find the movies 
    i = 0
    for folder in Path(training_dir).glob("*"):
        #find the painters name with regex
        movie = re.findall(r"(?!.*/).+.*", str(folder)) #re.findall returns a list 
        #append the list with movies
        label_names.append(movie[0])


        for img in folder.glob("*"):
            trainY.append(i)

        i +=1



    # Labels for validation
    # Path to training folder with painters
    test_dir = os.path.join("..", "data","poster_data", "test")

    # test labels
    testY = []


    i = 0
    for folder in Path(test_dir).glob("*"):
        for img in folder.glob("*"):
            testY.append(i)
        i +=1


    # Change size of images in small_training
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
            dim = (32, 32) #(268,182 = original dimensions)
            #resize images
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            #append images to trainX
            trainX.append(resize_image.astype("float") / 255.)



    # Change size of images in validation_training
    # The path were the images are located
    filepath = os.path.join("..","data", "poster_data", "test")

    testX=[]
    
    # Same as before but with the validation images
    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            dim = (32, 32)
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            testX.append(resize_image.astype("float") / 255.)

    # integers to one-hot vectors
    lb = LabelBinarizer()
    # transform labels to binary codes 
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # Convert to numpy
    testY = np.array(testY)
    trainY = np.array(trainY)
    #Convert to numpy
    testX = np.array(testX)
    trainX = np.array(trainX)

    #-----------------------------Run the VGG16 model-------------------------
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
        fig.savefig("../output/pretrained_performance.png")
    # load model without classifier layers
    model = VGG16(include_top=False, 
                    pooling='avg',
                    input_shape=(32, 32, 3))


    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False



    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(10, 
                activation='relu')(flat1)
    output = Dense(8, 
                activation='softmax')(class1)

    # define new model
    model = Model(inputs=model.inputs, 
                    outputs=output)


    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9)
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Save model summary
    plot_model(model, to_file = "../output/pretrained_model_architecture.png", show_shapes=True, show_layer_names=True)

    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=128,
                  epochs=10,
                  verbose=1)

    #View the summary
    model.summary()

    plot_history(H, 10)
    
    # Print the classification report
    predictions = model.predict(testX, batch_size=10)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
    #Writing results to txt file.
    txt_file = open("../output/pretrained_clas_rep.txt", "a")
    txt_file.write(classification_report(testY.argmax(axis=1),
                                         predictions.argmax(axis=1),
                                         target_names=label_names))         
    txt_file.close()
    
if __name__ == "__main__":
    main()