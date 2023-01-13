
# Flexim_Final_Version
## Table of Directory
* [Flexim User](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_User)
* [Flexim Object](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_object)
## General info
This project has two components, Flexim_User and Flexim_object. Flexim_object directory describe component of research project that uses Flexim to emulate objective similarity metrics, such as Euclidean distance, Dynamic Time Wrapping. The other directory Flexim_User corresponds to research component that uses Flexim to emulate user-defined similarity metrics. For component of emulating user-defined similarity , user can load dataset and interact with Flexim's Graphical User Interface to train corresponding neural network model in order to capture user-defined similarity metric. 
## Setup
To train user-defined similarity function,install it locally:
```
$ cd Flexim_User
$ python st500_emulate_preprocess.py dataset.csv
$ python GUI_new.py
```
## Run application
After running GUI_new.py, user can see corresponding Graphical User Interface.
```
$ click load button
$ load dataset from local directory (after loading dataset, user can see corresponding plot on the upperleft position of graphical user interface)
$ click start button (user can see different linear/non-linear transformation parameter's value which is shown on following figures)
```
![gui](https://user-images.githubusercontent.com/28042893/212168889-9af1a342-12a7-4d35-a552-afe30f886fbf.png)
