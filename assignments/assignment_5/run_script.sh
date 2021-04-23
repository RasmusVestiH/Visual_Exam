#!/usr/bin/env bash

#Environment name
VENVNAME=as5env

#Create and Activate environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip
pip install ipython
pip install matplotlib
pip install opencv-python
pip install pandas
pip install pydot

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

#run script
python cnn-artist.py 


#deactivate environment
deactivate
