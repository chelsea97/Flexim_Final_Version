# Flexim_Final_Version
## Table of Directory
* [Flexim User](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_User)
* [Flexim Object](https://github.com/chelsea97/Flexim_Final_Version/tree/main/Flexim_object)
## General info
This project has two components, one main component is using Flexim to emulate objective similarity metrics, such as Euclidean distance, Dynamic Time Wrapping. The other main component is using Flexim to emulate user-defined similarity metrics. For this component, user can load dataset and interact with Flexim's Graphical User Interface to train neural network model. 
## Setup
To train user-defined similarity function,install it locally:
```
$ cd Flexim_User
$ python st500_emulate_preprocess.py dataset.csv
$ python GUI_new.py
## Run application
After running GUI_new.py, user can see corresponding Graphical User Interface.
Then click load button, load dataset from your local directory and you will see corresponding plot of dataset.
![emulate_user_part](https://user-images.githubusercontent.com/28042893/212110199-f83bbcb7-9266-4ad4-a6e4-ea9b6b374512.png)
