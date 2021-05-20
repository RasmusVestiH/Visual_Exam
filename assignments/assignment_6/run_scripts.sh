#!/usr/bin/env bash

#Environment name
VENVNAME=as6env

#Create and Activate environment
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip
pip install ipython
pip install opencv-python
pip install pandas
pip install sklearn
pip install tensorflow
pip install pydot

# problems when installing from requirements.txt
test -f requirements.txt && pip install -r requirements.txt

#navigate to src folder
cd src

#run script
python make_poster_data.py   $@
python CNN_movie_posters.py  $@
python VGG16_pretrained.py   $@



#deactivate environment
deactivate
