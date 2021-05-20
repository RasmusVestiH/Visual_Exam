# Assignment 6 - CNNs on Movie Posters (Self assigned project) 
## Description
**Disclaimer: This assignment has been worked out in the group consisting of Emil Buus Thomsen, Johanne Brandhøj Würtz and Rasmus Vesti Hansen** 

For this assignment we tried to make a LeNet Model classify movie genres by images of move posters.

## Running it
For this I have created a bash script called "run_script.sh" this will set up the virtual environment, download the images for the movie posters and split these into smaller datasets with unique genres. After this it will run the LeNet CNNs model that will train itself on the data and the pretrained model VGG16. These models will the try to figure out which genre a movie is based on the poster. Afterwards outputs will be created based on the figures, accuracy and models.  
