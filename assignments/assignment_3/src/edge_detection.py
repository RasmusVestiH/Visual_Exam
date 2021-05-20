
'''
Import libraries
'''
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
from PIL import Image #we load in the the PIL package that makes it easy to crop and save the image

'''
Defining the main function which will be the script that runs when executed
'''
def main(): 
    ## First we load in the image with openCV
    fname = os.path.join("..","data","jefferson_img.jpg")
    image = cv2.imread(fname)

    # Then we define the height and width of the image which we use to make a rectangle surrounding the letters on the image
    (widthX, heightY) = image.shape[1], image.shape[0]
    cv2.rectangle(image, (widthX//4,heightY//4),((widthX//4)*3,(heightY//8)*7), (0,255,0), 4)
    
    # Here I used jimshow to show, but as this runs in the terminal it won't be neccessary

    '''
    Cropping the image
    '''
    im = Image.open(r"../data/jefferson_img.jpg") #defining the im with our image
  
    # Setting the points for cropped image which we calculated above
    left = widthX//4
    top = heightY//4
    right = (widthX//4)*3
    bottom = (heightY//8)*7
  
    
    #Making the ROI with the dimensions above (basically draw a line where the image is getting cropped below) 
    im0 = cv2.rectangle(image, (left,top), (right,bottom), (255,0,0), 3)
    cv2.imwrite("../output/image_with_ROI.jpg", im0)
    
    # Cropped image of above dimension 
    # It will not change orginal image as we save it as im1
    im1 = im.crop((left, top, right, bottom)) 

    # Then we save it as the image_cropped in the data folder
    im1.save("../output/image_cropped.jpg")
    
    '''
    Loading the cropped image
    '''
    fname = os.path.join("..","output","image_cropped.jpg") #Now we load the cropped image
    cropped_image = cv2.imread(fname) #defining it the same way as with the original image


    '''
    Blurred Image
    '''
    #Now we make the blurred image and use the canny edge to then create contours around the letters
    blurred = cv2.GaussianBlur(cropped_image, (5,5), 0) #we set the kernel of 5,5

    canny = cv2.Canny(blurred, 100, 150) #I tried messing around with the setting of the blur but figured this was the best

    (contours, _) = cv2.findContours(canny.copy(), #using a copy of the canny image to find the countours
                    cv2.RETR_EXTERNAL, #filtering internal structures hieararchially
                    cv2.CHAIN_APPROX_SIMPLE)

    image_text = cv2.drawContours(cropped_image.copy(), # draw contours on original
                        contours,      # our list of contours
                        -1,            # which contours to draw
                        (0,255,0),     # contour color
                        2)  
    cv2.imwrite("../output/image_letters.jpg", image_text)
    print("The results are located in the output folder")

# At last we save the image with the green contours surrounding the letters and a bit of noise around. 
# Defining the behavior for console
if __name__ == "__main__":
    main()

