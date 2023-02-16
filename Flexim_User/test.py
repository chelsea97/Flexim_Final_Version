from array import array
from cProfile import label
from random import shuffle
from unicodedata import decimal
from unittest import result, skip
from Functions import Functions
from tkinter import *
from tkinter import filedialog
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
from generator import ac_generate
from AE import get_cluster_label
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from NN import NN as nn
from pattern_search import Pattern_Search as ps
from sklearn.model_selection import train_test_split
import sys
from pattern_search import Pattern_Search
import matplotlib.pyplot as plt
from GUI_show import GUI_show

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
    return score_cosine_index_start,score_dtw_index_start,score_flexim_index_start
    
def transform(original_data,sample_size):
    convert_list = []
    for i in range(len(original_data)):
        original_row = original_data[i,0:sample_size].tolist()
        transform_row = original_data[i,sample_size:2*sample_size].tolist()
        row_list = []
        for j in range(len(original_row)):
            row_list.append(original_row.pop(0))
            row_list.append(transform_row.pop(0))
        convert_list.append(row_list)
    convert_list = np.array(convert_list)
    return convert_list

def plot_compare_curve(dataset,sample_size,query_start,cosine_result_start,dtw_result_start,flexim_result_start,k):
    #scale = np.arange(0,len(dataset))
    fig,ax = plt.subplots(4,1)
    t = np.arange(query_start,query_start+sample_size)
    ax[0].plot(t, dataset[t], linewidth=2.5,color='r')
    ax[0].set_ylabel('query pattern',fontsize = 14)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].grid(True)
    #ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
    t1 = np.arange(cosine_result_start[k-1],cosine_result_start[k-1]+sample_size)
    #scale = np.arange(0,1.1,0.1)
    ax[1].plot(t1, dataset[t1],linewidth=2.5)
    #ax[1].set(ylabel='L2')
    ax[1].set_ylabel('Cosine',fontsize = 14)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    #ax[1].set_yticks(dataset[t1],font_size = 0.2)
    #ax[2].set_yticks(scale)
    ax[1].grid(True)
    #ax[2].axvspan(result_start[0], result_end[0],color='green', alpha=0.2)
    t2 = np.arange(dtw_result_start[k-1], dtw_result_start[k-1]+sample_size)
    #scale = np.arange(0, 1.1, 0.1)
    ax[2].plot(t2, dataset[t2],linewidth=2.5,color = 'purple')
    ax[2].set_ylabel('DTW',fontsize = 14)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    #ax[2].set_xticks(t2,font_size = 0.2)
    #ax[2].set_yticks(dataset[t2],font_size = 0.2)
    #ax[2].set_yticks(scale)
    ax[2].grid(True)
    #ax[2].axvspan(result_start[i], result_end[i],color='blue', alpha=0.2)
    t3 = np.arange(flexim_result_start[k-1], flexim_result_start[k-1]+sample_size)
    #scale = np.arange(0, 1.1, 0.1)
    ax[3].plot(t3, dataset[t3],linewidth=2.5,color = 'seagreen')
    ax[3].set_xlabel('time', fontsize = 14)
    ax[3].set_ylabel('Flexim',fontsize = 14)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    #ax[3].set_xticks(t3,font_size = 0.2)
    #ax[3].set_yticks(dataset[t3],font_size = 0.2)
    #ax[2].set_yticks(scale)
    ax[3].grid(True)
    fig.savefig("compare.png")
    fig.tight_layout()
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=0.1)
    plt.show()
    
            
if __name__ == '__main__':
    csv_file = sys.argv[1]
    sample_size = int(sys.argv[2])
    query_start = int(sys.argv[3])
    kvalue = int(sys.argv[4])
    temp = pd.read_csv("./" + csv_file)
    temp = temp["close"].values
    scale = MinMaxScaler()
    dataset = scale.fit_transform(temp.reshape(-1,1))
    # with open(train_data_file) as file_name:
    #     total_train_table = np.loadtxt(train_data_file,delimiter = ",")
    # total_train_table = transform(total_train_table,sample_size)
    # with open(train_data_label_file) as file_name:
    #     total_label_table = np.loadtxt(train_data_label_file,delimiter = ",").reshape(-1,1)
    # with open(validate_data_file) as file_name:
    #     total_validate_table = np.loadtxt(validate_data_file,delimiter = ",")
    # total_validate_table = transform(total_validate_table,sample_size)
    # with open(validate_data_label_file) as file_name:
    #     predict_label_valid = np.loadtxt(validate_data_label_file,delimiter = ",").reshape(-1,1)
    # print("shape of train data is {}".format(total_train_table.shape))
    # print("shape of train data label is {}".format(total_label_table.shape))
    # print("shape of validate data is {}".format(total_validate_table.shape))
    # print("shape of validate data label is {}".format(predict_label_valid.shape))
    # neg = total_label_table[total_label_table < 0.5].shape[0]
    # pos = total_label_table[total_label_table >= 0.5].shape[0]
    # total = neg + pos
    # weight_for_0 = (1/neg)*(total/2)
    # weight_for_1 = (1/pos)*(total/2)
    # class_weight = {0:weight_for_0,1:weight_for_1}
    #nn_model = neural_network_train(total_train_table,total_label_table,sample_size,(total_validate_table,predict_label_valid),class_weight)
    #nn_model.save('saved_model/my_model_test')
    print("load model")
    nn_model = tf.keras.models.load_model('saved_model/mx_iter_5/my_model_4')
    score_cosine_index_start,score_dtw_index_start,score_flexim_index_start = pattern_search_show(query_start,kvalue,dataset,sample_size,nn_model)
    gui_show = GUI_show(dataset,sample_size,query_start,score_cosine_index_start,score_dtw_index_start,score_flexim_index_start)
    
    
    
    
    
    
    
    
