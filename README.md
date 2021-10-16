# Our object detection
## Requirements
### Windows:
* Python 3.9
* Virtualenv
* Microsoft C++ Build Tools
### Linux:
* Python 3.9
* Virtualenv
* Linux package: libpython3-dev (or python3-dev)
## Getting started
### Setting up python virtual environment
Windows
```
python -m venv env
```
Linux (Ubuntu)
```
virtualenv env
```
### Installing required python modules
Windows
```
python -m pip install -r requirements.txt
```
On linux you don't have to
### Project setup
```
python setup.py
python checksetup.py
python image_collect.py
```
After collecting images and xml, copy them from _/Tensorflow/workspace/images/collectedimages_ to _/Tensorflow/workspace/images/train_ and _/Tensorflow/workspace/images/test_
```
python model_train.py -s <num_of_steps> # default 2000
python detect.py
```