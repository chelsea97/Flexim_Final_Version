from array import array
from cProfile import label
import math
import multiprocessing
import numbers
from random import shuffle
from unicodedata import decimal
from unittest import result, skip
from Functions import Functions
from tkinter import *
from tkinter import filedialog
import os
import copy
import pandas as pd
from generator import datagenerator
import numpy as np
import random
import time as time
from Active import Active as ac
import tensorflow as tf
#import GUI as gui
from numpy import dot
from numpy.linalg import norm
from AE import AE_Model, kmeans_label
import ray
from generator import ac_generate
from AE import get_cluster_label
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from NN import NN as nn
import joblib
from pattern_search import Pattern_Search as ps
from sklearn.model_selection import train_test_split
import sys
import argparse
import Core_Test
from pattern_search import Pattern_Search

def neural_network_train(instance,label,sample_size,validation_data,class_weight):
    neural_network = nn(instance,label,sample_size,validation_data,class_weight)
    neural_network_model = neural_network.model
    return neural_network_model

def nn_predict(model,X):
    predict_score = model.predict(X)
    return predict_score

def pattern_search_show(query_start,kvalue,dataset,sample_size,model):
    ps = Pattern_Search(dataset,sample_size)
    query_pattern = dataset[query_start:query_start+sample_size,0]
    stride_size = sample_size // 10
    score_cosine = ps.find_pattern_cosine(query_pattern,stride_size,kvalue)
    score_dtw = ps.find_pattern_dtw(query_pattern,stride_size,kvalue)
    score_flexim = ps.find_pattern_model_regression(query_pattern,stride_size,kvalue,model)
    score_cosine_index_start,score_cosine_index_end = ps.record_index(score_cosine)
    print("score cosine start is {}".format(score_cosine_index_start))
    print("score cosine end is {}".format(score_cosine_index_end))
    score_dtw_index_start,score_dtw_index_end = ps.record_index(score_dtw)
    print("score dtw start is {}".format(score_dtw_index_start))
    print("score dtw end is {}".format(score_dtw_index_end))
    score_flexim_index_start,score_flexim_index_end = ps.record_index(score_flexim)
    print("score flexim start is {}".format(score_flexim_index_start))
    print("score flexim end is {}".format(score_flexim_index_end))
    
    

if __name__ == '__main__':
    csv_file = sys.argv[1]
    train_data_file = sys.argv[2]
    train_data_label_file = sys.argv[3]
    validate_data_file = sys.argv[4]
    validate_data_label_file = sys.argv[5]
    sample_size = 50
    temp = pd.read_csv("./" + csv_file)
    temp = temp["close"].values
    scale = MinMaxScaler()
    dataset = scale.fit_transform(temp.reshape(-1,1))
    with open(train_data_file) as file_name:
        total_train_table = np.loadtxt(train_data_file,delimiter = ",")
    with open(train_data_label_file) as file_name:
        total_label_table = np.loadtxt(train_data_label_file,delimiter = ",").reshape(-1,1)
    with open(validate_data_file) as file_name:
        total_validate_table = np.loadtxt(validate_data_file,delimiter = ",")
    with open(validate_data_label_file) as file_name:
        predict_label_valid = np.loadtxt(validate_data_label_file,delimiter = ",").reshape(-1,1)
    print("shape of train data is {}".format(total_train_table.shape))
    print("shape of train data label is {}".format(total_label_table.shape))
    print("shape of validate data is {}".format(total_validate_table.shape))
    print("shape of validate data label is {}".format(predict_label_valid.shape))
    neg = total_label_table[total_label_table < 0.5].shape[0]
    pos = total_label_table[total_label_table >= 0.5].shape[0]
    total = neg + pos
    weight_for_0 = (1/neg)*(total/2)
    weight_for_1 = (1/pos)*(total/2)
    class_weight = {0:weight_for_0,1:weight_for_1}
    nn_model = neural_network_train(total_train_table,total_label_table,sample_size,(total_validate_table,predict_label_valid),class_weight)
    pattern_search_show(50,5,dataset,sample_size,nn_model)
    
    
    
    
    
    
    