#!/usr/bin/env bash

#Environment name
VENVNAME=as3env

#create and activate the virtual environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt


#navigate to src folder
cd src

#run script
python3 edge_detection.py

#deactivate environment
deactivate
