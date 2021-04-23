#!/usr/bin/env bash

#Environment name
VENVNAME=as4env

#create and activate the virtual environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip
pip install numpy
pip install pandas
pip install seaborn
pip install sklearn

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

#run script
python3 lr-mnist.py $@
python3 nn-mnist.py $@

#deactivate environment
deactivate
