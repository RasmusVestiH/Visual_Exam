#!/usr/bin/env bash

VENVNAME=as5env 

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

# problems when installing from requirements.txt
pip install ipython
pip install matplotlib
pip install opencv-python
pip install pandas
pip install pydot

test -f requirements.txt && pip install requirements.txt 

deactivate
echo "build $VENVNAME"