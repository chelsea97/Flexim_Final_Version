
# Flexim_Final_Version
## Table of Directory
* [Flexim User](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_User)
* [Flexim Object](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_object)
## General info
There are two components to this project: Flexim User and Flexim object. Flexim object directory describes research project component that employs Flexim to emulate objective similarity metrics, such as Euclidean distance and Dynamic Time Wrapping. The Flexim User directory corresponds to a research component that uses Flexim to emulate user-defined similarity metrics. User can load dataset and interact with Flexim's Graphical User Interface to train neural network model to capture user-defined similarity metric.
## Setup
Train the data encoder and clusterer based on user-selected dataset *dataset.csv*
```
$ cd Flexim_User
$ python st500_emulate_preprocess.py dataset.csv
$ python GUI_new.py
```
## Run application
After running GUI_new.py, user can see corresponding Graphical User Interface.
```
$ click load button
$ load dataset from local directory (When a dataset is loaded, the user can view the corresponding plot in the upper-left corner of the graphical user interface.)
$ click start button (Figure 2 illustrates various linear and non-linear transformations and their corresponding parameter values.)
```
![gui](https://user-images.githubusercontent.com/28042893/212168889-9af1a342-12a7-4d35-a552-afe30f886fbf.png)
