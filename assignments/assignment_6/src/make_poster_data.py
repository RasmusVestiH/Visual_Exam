import sys,os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wget
from pathlib import Path
import re

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


def main(): 
    #Path to data
    input_file = os.path.join("..", "data", "MovieGenre.csv")

    #Reading data
    data = pd.read_csv(input_file, encoding = "ISO-8859-1")
    
    #-----------------------------------Importing the data:-----------------------------------------
    
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

    try:
        os.mkdir("../data/poster_data")
    except FileExistsError:
        print(f"poster_data already exists.")
    #Making the folders for the data

    #Iterating through the unique cathegories.
    final_categories = []
    for cat in final_df["Genre"].unique():

        #We find the Genres, that is only described by cathegory
        # "|" indicate that the movie has more than one genre.
        if "|" not in str(cat):
            final_categories.append(cat)

    print(final_categories)


    #----------------------Creating sub folders:-------------------


    #Running through a list with unique cathegories
    for folder in final_categories:

        #Creating the path to where the folder should be created
        path = "../data/poster_data/" + str(folder)

        #Creating directory
        try:
            os.mkdir(path)
            print(f"Folder:{folder} was created.")

        #If the folder already exists print that it already exists.
        except FileExistsError:
            print(f"{folder} already exists.")


    #---------------Downloading posters and saving them to the different folders:------------------
    
    #Running the loop as many times as the length of the data set
    for i in range(len(final_df)):

        subfolder = final_df["Genre"][i]

        #Creating name of poster files
        filename = "../data/poster_data/"+ str(subfolder)+ "/" + str(final_df["Title"][i]) + ".jpg"
        print(filename)

        #Accessing the links for the posters
        image_url = final_df["Poster"][i]
        #print(image_url)

        #Error handling.
        #If the poster does not exist: pass, and move on to the next file.
        try:
            image_filename = wget.download(image_url, filename)
        except:
            #print("There was an error")
            pass

    #--------------------Create traning and test folders:----------------------------------------- 

    #Path of the original folder
    rootdir= '../data/poster_data' 

    #List sub directories/the genres
    classes = ['Animation', 'Horror', 'Romance', 'Comedy', 'Drama', 'Documentary','Western', 'Thriller']

    #As the thriller genre had 335 posters to work with I decided to go with 240 posters for the training dataset and 60 for the test

    #Iterate through them
    for i in classes:
        #Create a new training folder with the name of the iterated sub dir
        os.makedirs(rootdir +'/train/' + i)
        #Create a new test folder with the name of the iterated sub dir
        os.makedirs(rootdir +'/test/' + i)

        source = rootdir + '/' + i
        
        # Create a variable of all listed posters
        allFileNames = os.listdir(source)
        
        # Use Sklearns train_test_split function to separate the posters into datasets
        train_FileNames, test_FileNames = train_test_split(allFileNames, test_size=60, train_size=240)

        #Create Labels based on the source which is the subfolder. 
        train_FileNames = [source+'/'+ name for name in train_FileNames]
        test_FileNames = [source+'/' + name for name in test_FileNames]

        # Save the files in the the data folders
        for name in train_FileNames:
            shutil.copy(name, rootdir +'/train/' + i)

        for name in test_FileNames:
                    shutil.copy(name, rootdir +'/test/' + i)

if __name__ == "__main__":
    main()