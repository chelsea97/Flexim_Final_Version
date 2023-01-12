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
After running GUI_new.py, user can see corresponding Graphical User Interface.
![image](https://raw.githubusercontent.com/chelsea97/Flexim_Final_Version/main/emulate_user_part.png)
