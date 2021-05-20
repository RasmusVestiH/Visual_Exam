# Visual Analytics - Assignment 4: Classification Benchmarks

## Description
**Disclaimer: This assignment has been worked out in the group consisting of Emil Buus Thomsen, Johanne Brandhøj Würtz, Christoffer Kramer and Rasmus Vesti Hansen**

In this assignment we were working with neural networks and logistic regression as classification models. For this we worked with mnist dataset which contains of large amount pictures of handdrawn numbers. 


## Running
For this assignment it will be possible to run the bash script: "run_scripth.sh" and it would automatically run both models in the terminal, print the results and save the results in the output folder. For this assignment we also included some argparse functions to change the setup of the model. The default train size is 0.8 (80%) and the test size is 0.2 (20%). This means that it will be possible to change the ratios to look at different results with e.g. "-trs = (0.7)" and "-tst = (0.3)" after running the .py scripts in the /src folder. I have also kept the original notebook script in this repo in case one would see the structure.  