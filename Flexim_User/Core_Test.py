# Test
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

class Core_Test:
    def __init__(self, caller, delta=0.1, n_points=1000):
        self.caller         = caller
        self.n_points       = n_points
        self.label = IntVar(self.caller.root)
        self.sample_size = 10
        self.data           = None
        self.transformed    = None
        self.original       = None
        self.params_non_lin = []
        self.params_lin     = []
        self.__init_non_lin__()
        self.__init_lin__()
        self.current_cluster = 1
        #self.generated_data = np.array([[0 for _ in self.params_non_lin] + [0 for _ in self.params_lin] + [1]]) #？？？generate data part, why combine non linear parameter and linear parameter together
        self.f_fetch_done   = False
        self.fetch_started  = False
        self.alphas_counter = 0
        self.active_counter = 0
        self.current_interv = None
        self.side           = 0
        self.delta          = delta
        self.current_setting= []
        self.limits         = {}
        self.unlabeled       = None
        self.labels         = []
        self.training_done  = False
        self.cluster_center = []
        self.cluster_label = 0
        self.cluster_index_dictionary = {}
        self.dictionary = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.alpha_dictionary = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.train_autoencoder = False
        self.cluster_number = 0
        self.label_query_list = []
        self.nece_label_time = 0
        self.sample_index = []
        self.generate_naive_data = False
        

    def __init_non_lin__(self):
        self.F = Functions(self, {'sample_size': self.sample_size}) #???
        funcs = list(filter(lambda f: not f.startswith('__'),dir(Functions))) #return functions which satisfies lambda
        for f in funcs:
            _, rng, name = eval('self.F.'+f+'()') #？？？
            vr = DoubleVar(self.caller.root) #send message via variable, construct a float variable
            self.params_non_lin.append((f, rng, name, vr))
            
    def __init_lin__(self):       #???
        names = ['a','b','c','d','e','f']
        for n in names:
            vr = DoubleVar(self.caller.root)
            self.params_lin.append(('_',(-1,1), n, vr))
            
    def __init__cluster_index_dictionary__(self):
        for i in range(1,self.cluster_number+1):
            self.cluster_index_dictionary[i] = []
            
    def __init__limit_dictionary__(self):
        for i in range(1,self.cluster_number+1):
            self.limits[i] = {}
        
    
    def __restart_fetch__(self):
        self.fetch_started =False
        self.alphas_counter=0
        self.side          =0
        self.training_done = False
        self.f_fetch_done = False
        self.unlabeled = None

    def __first_fetch__(self, last_label):
        #print(self.current_interv, self.label.get())
        #First fetch implements binary search
        cluster_one_hot = to_categorical(
            self.current_cluster-1, num_classes=self.cluster_number).astype('int').tolist()
        if not self.fetch_started:
            self.fetch_started = True
            #initialize the left side
            #self.current setting is relevant to GUI part
            self.current_interv = [-1,0]
            self.current_setting= cluster_one_hot + [0 for _ in range(len(self.params_lin))], \
                                  cluster_one_hot + [0 for _ in range(len(self.params_non_lin))]
            self.current_setting[0][self.cluster_number+self.alphas_counter]=-1
            self.alpha          = -1
            #self.current_setting.astype('float32')
            return self.current_setting
        else: 
            #check the interval if within the boundary and side is 1 stop, update the alphas and return;
            if abs(self.current_interv[1] - self.current_interv[0]) <= self.delta:
                #control GUI button side. If self.side == 0, button move in the region (-1,0) else  
                if self.side == 0:  
                    self.limits[self.current_cluster][self.alphas_counter]=[self.current_interv[0], 0]
                    self.side=1
                    self.current_interv         = [0,1]
                    temp                        = cluster_one_hot+[0] * len(self.params_lin), cluster_one_hot+[0]*len(self.params_non_lin)
                    temp[0][self.cluster_number+self.alphas_counter]= 1
                    self.alpha                  = 1
                    return temp
                    #return self.__first_fetch__(last_label)
                else:
                    self.limits[self.current_cluster][self.alphas_counter][1]=self.current_interv[1]
                    self.fetch_started = False
                    self.side=0
                    #self.__restart_fetch__()
                    if self.alphas_counter < len(self.params_lin) - 1:
                        self.alphas_counter        += 1
                        self.fetch_started          = False
                    else:
                        self.f_fetch_done = True
                        #return self.fetch(last_label)
                        return self.current_setting
                return self.__first_fetch__(last_label)
            else:
                self.current_setting= cluster_one_hot + [0] * len(self.params_lin), cluster_one_hot + [0] * len(self.params_non_lin)
                #print("self.label.get() is {}".format(self.label.get()))
                if self.label.get():
                    #print('label 1')
                    if self.side==0:
                        self.current_interv[1]=self.alpha
                        self.alpha=(self.current_interv[1]+self.current_interv[0])/2
                    else:
                        self.current_interv[0]=self.alpha
                        self.alpha=(self.current_interv[1]+self.current_interv[0])/2
                else:
                    #print('label 0')
                    if self.side==0:
                        self.current_interv[0]=self.alpha
                        self.alpha=(self.current_interv[1]+self.current_interv[0])/2
                    else:
                        self.current_interv[1]=self.alpha
                        self.alpha=(self.current_interv[1]+self.current_interv[0])/2
                self.current_setting[0][self.cluster_number+self.alphas_counter]=self.alpha
                #print("new interval")
                #print(self.current_interv)
                return self.current_setting
            
    def __active_fetch__(self):
        labeled = unlabeled = None
        if self.unlabeled is None:
            labeled, unlabeled = self.gen_data_points()
            print('label is {}'.format(labeled))
            print('unlabeled is {}'.format(unlabeled))
            print('label shape is {}'.format(labeled.shape))
            print('unlabeled shape is {}'.format(unlabeled.shape))
            if self.current_cluster == 1:
                self.active_lr   = ac(unlabeled, labeled)   #construct active learning instance with unlabel and label data
            else:
                # num_training_data = len(labeled)
                # training_indices = np.random.randint(
                #     low=0, high=num_training_data, size=4*num_training_data // 5)
                # self.active_lr.learner.teach(X = labeled[training_indices,0:-1],y = labeled[training_indices,-1])
                # self.active_lr.labeled_data = np.delete(labeled,training_indices,axis = 0)
                # self.active_lr.mx_iter = 3
                # self.active_lr.un_labled_data = unlabeled
                self.active_lr.train_label, self.active_lr.test_label = train_test_split(labeled,test_size = 0.2,random_state = 42,shuffle = None)
                self.active_lr.train_label, self.active_lr.valid_label = train_test_split(self.active_lr.train_label,test_size = 0.2, random_state = 44, shuffle = None)
                self.active_lr.learner.teach(X = self.active_lr.train_label[:,0:-1],y = self.active_lr.train_label[:,-1])
                self.active_lr.mx_iter = 1
                self.active_lr.un_label_data = unlabeled
            global query_list
            self.unlabeled = self.active_lr.query() #use query method to get unlabeled uncertainty data

            
            print("initial active learning score is {}".format(self.active_lr.learner.score(
                self.active_lr.valid_label[:, 0:-1], self.active_lr.valid_label[:, -1].astype('int64'))))
        
        #self.active_counter作用记录被标记为label的data,这样好判断是否附上label value在此循环中
        if self.active_counter >= self.unlabeled.shape[0] and self.nece_label_time < self.active_lr.mx_iter:
            print("implement necessary training")
            print("label time is {}".format(self.nece_label_time))
            print(self.unlabeled.shape)
            labels = np.array(self.labels)
            print(labels.shape)
            print("necessary training label is {}".format(labels))
            self.labels = []
            sub_query_list = np.concatenate((self.unlabeled,labels.reshape(-1,1)),axis = 1)
            #global query_list
            if self.nece_label_time == 0:
                #query_list = sub_query_list
                query_list = np.concatenate((sub_query_list,self.active_lr.valid_label),axis = 0)
                #print("query list is {}".format(query_list))
            else:
                query_list = np.concatenate((query_list,sub_query_list),axis = 0)
            shuffle(query_list)
            self.active_lr.learner.teach(X = self.unlabeled, y = labels)
            print("******* query list is {}".format(query_list))
            print("!!!!!!!! query list shape is {}".format(query_list.shape))
            print("after training, active learning score is {}".format(self.active_lr.learner.score(
                query_list[:, 0:-1], query_list[:, -1].astype('int64'))))
            self.active_counter = 0
            self.unlabeled = self.active_lr.query()
            self.nece_label_time += 1
            if self.active_lr.learner.score(
                    query_list[:, 0:-1], query_list[:, -1].astype('int64')) >= 0.85:
                self.training_done = True
                print("accuracy reaches 85%")
            
        
        if self.active_counter >= self.unlabeled.shape[0] and self.nece_label_time == self.active_lr.mx_iter:
            print("training")
            print(self.unlabeled.shape)
            labels     = np.array(self.labels)
            print(labels.shape)
            self.labels=[]
            print("label is {}".format(labels))
            #self.training_done=self.active_lr.train(X=self.unlabeled, Y=labels)
            self.active_lr.learner.teach(X = self.unlabeled, y = labels)
            sub_query_list = np.concatenate((self.unlabeled,labels.reshape(-1,1)),axis = 1)
            query_list = np.concatenate((query_list,sub_query_list),axis = 0)
            print("after training, active learning score is {}".format(self.active_lr.learner.score(
                query_list[:, 0:-1], query_list[:, -1].astype('int64'))))
            self.active_counter = 0
            self.unlabeled = self.active_lr.query()
            self.nece_label_time = 0
            self.training_done = True

        
        self.labels.append(self.label.get())
        #self.active_lr.active_learning(self.active_lr.get_final_learner(),self.unlabeled,self.labels,self.unlabeled)
        #[ 0.109  0.007 -0.149 -0.082  0.659  0.21  -0.929  0.936] 
        #(array([ 0.109,  0.007, -0.149, -0.082,  0.659,  0.21 ]), array([-0.929]))
        res = self.unlabeled[self.active_counter][0:len(self.params_lin)+self.cluster_number], \
              self.unlabeled[self.active_counter][len(self.params_lin)+self.cluster_number:]
        #res apprear when do active learning
        print('res:')
        print(res) 
        self.active_counter+=1
        return res

    def fetch(self, last_label=True):
        #After click sample button, program will jump to this part and fetch data to label
        # While the first fetch not done yet, call __first_fetch__() function to generate single distortion params
        #print(self.limits)
        print("self.training_done value is {}".format(self.training_done))
        if self.training_done:
            if self.current_cluster == self.cluster_number:
                #self.caller.deactivate_buttons()
                if not(self.generate_naive_data):
                    self.alpha_array = self.gen_data_complex_points(100000)
                # query_index,query_instance = self.active_lr.get_final_learner().query(self.alpha_array)
                    np.random.shuffle(self.alpha_array)
                    self.un_final_train_complex,self.un_final_test_complex = train_test_split(self.alpha_array,test_size = 0.2,random_state = 50, shuffle = None)
                    self.un_final_train_complex,self.un_final_valid_complex = train_test_split(self.un_final_train_complex,test_size = 0.2, random_state = 50, shuffle = None)
                    print("finish split")
                    self.nece_label_time = 0
                    print("finish generate naive data")
                    print("enter validation active learning process")
                    query_index, self.unlabeled = self.active_lr.learner.query(self.un_final_train_complex)
                    self.un_final_train_complex = np.delete(self.un_final_train_complex,query_index,axis = 0)
                    print("validation process query data is {}".format(self.unlabeled))
                    self.generate_naive_data = True
                    self.active_learning_validate = False
                if self.active_counter >= self.unlabeled.shape[0] and self.nece_label_time < self.active_lr.mx_iter:
                    print("enter condition 1")
                    labels = np.array(self.labels)
                    self.labels = []
                    sub_query_list = np.concatenate((self.unlabeled,labels.reshape(-1,1)),axis = 1)
                    if self.nece_label_time == 0:
                        self.query_list = copy.deepcopy(sub_query_list)
                    else:
                        self.query_list = np.concatenate((self.query_list,sub_query_list),axis = 0)
                    self.active_lr.learner.teach(X = self.unlabeled,y = labels)
                    self.active_counter = 0
                    query_index, self.unlabeled = self.active_lr.learner.query(self.un_final_train_complex)
                    self.un_final_train_complex = np.delete(self.un_final_train_complex,query_index,axis = 0)
                    self.nece_label_time += 1
                    if self.active_lr.learner.score(self.query_list[:,0:-1],self.query_list[:,-1]) >= 0.85:
                        self.active_learning_validate = True
                        print("accuracy reaches 85%")
                            
                if self.active_counter >= self.unlabeled.shape[0] and self.nece_label_time == self.active_lr.mx_iter:
                    print("enter condition 2")
                    labels = np.array(self.labels)
                    self.labels = []
                    self.active_lr.learner.teach(X = self.unlabeled, y = labels)
                    sub_query_list = np.concatenate((self.unlabeled,labels.reshape(-1,1)),axis = 1)
                    self.query_list = np.concatenate((self.query_list,sub_query_list),axis = 0)
                    self.active_counter = 0
                    self.nece_label_time = 0
                    print("finish active learning in validation process")
                    self.active_learning_validate = True
                if not(self.active_learning_validate):
                    print("enter condition 3")
                    print("get label is {}".format(self.label.get()))
                    self.labels.append(self.label.get())
                    result_lin,result_non_lin = self.unlabeled[self.active_counter][0:len(self.params_lin)+self.cluster_number],self.unlabeled[self.active_counter][len(self.params_lin)+self.cluster_number:]
                    print("validation part query process")
                    self.active_counter += 1
                    for alpha,v in zip(result_lin[self.cluster_number:],self.params_lin):
                        v[-1].set(alpha)
                    cluster = np.where(result_lin[0:self.cluster_number] == 1)[0][0]
                    print("cluster is {}".format(cluster))
                    original_index = self.sample_index[cluster-1]
                    self.original = self.data[original_index:original_index+self.sample_size]
                    print("change original function")
                #need to filter out not timestamp sorted samples
                else:
                    self.caller.deactivate_buttons()
                    predict_label_valid = self.active_lr.predict(self.un_final_valid_complex).reshape(-1,1)
                    self.validate_array = np.concatenate((self.un_final_valid_complex,predict_label_valid),axis = 1)
                    predict_label_test = self.active_lr.predict(self.un_final_test_complex).reshape(-1,1)
                    self.test_array = np.concatenate((self.un_final_test_complex,predict_label_test),axis = 1)
                    predict_label_train = self.active_lr.predict(self.un_final_train_complex).reshape(-1,1)
                    self.train_array = np.concatenate((self.un_final_train_complex,predict_label_train),axis = 1)
                    print("append query list into train array")
                    self.train_array = np.concatenate((self.train_array,self.query_list),axis = 0)
                    self.original_train_array,self.transform_train_array = self.transform_complex_points_2(self.train_array)
                    self.original_valid_array,self.transform_valid_array = self.transform_complex_points_2(self.validate_array)
                    self.original_test_array,self.transform_test_array = self.transform_complex_points_2(self.test_array)
                    
                    self.caller.activate_data_generator(3000*self.cluster_number)
                    self.total_original_table = np.concatenate((self.original_train_array,self.final_original_result_table),axis = 0)
                    self.total_transform_table = np.concatenate((self.transform_train_array,self.final_transform_result_table),axis = 0)
                    self.total_label_table = np.concatenate((predict_label_train,self.final_label_result_table),axis = 0)
                     
                    #self.total_train_table = np.concatenate((self.transform_train_array,self.final_naive_result_table),axis = 0)
                    neg = self.total_label_table[self.total_label_table < 0.5].shape[0]
                    pos = self.total_label_table[self.total_label_table >= 0.5].shape[0]
                    total = neg + pos
                    print("neg number is {}".format(neg))
                    print("pos number is {}".format(pos))
                    weight_for_0 = (1/neg)*(total / 2.0)
                    weight_for_1 = (1/pos)*(total / 2.0)
                    class_weight = {0:weight_for_0,1:weight_for_1}
                    # print("train data instance shape is {}".format(self.total_train_table[:,0:-1].shape))
                    # print("train data label shape is {}".format(self.total_train_table[:,-1].shape))
                    self.neural_network_train(self.total_original_table,self.total_transform_table,self.total_label_table,self.sample_size,self.original_valid_array,self.transform_valid_array,predict_label_valid,class_weight=class_weight)
                    self.pattern_search_show(query_start=50,kvalue=5)
            else:
                self.current_cluster += 1
                self.resample()
                for v in self.params_lin:
                    v[-1].set(0)
                self.__restart_fetch__()
                self.fetch()
                
        if not self.f_fetch_done:
            result_lin, result_non_lin = self.__first_fetch__(last_label)
        else:
            print('arrived here!')
            #result_lin, result_non_lin = [0]*len(self.params_lin), [0]*len(self.params_non_lin)
            # active learning to be called here
            result_lin, result_non_lin = self.__active_fetch__()
        print("result_lin is {}".format(result_lin))
        #print("self.params_lin is {}".format(self.params_lin))
        for alpha, v in zip(result_lin[self.cluster_number:], self.params_lin):
            v[-1].set(alpha)
            
    def pattern_search_show(self,query_start,kvalue):
        print("search process begin")
        self.ps = ps(self.data,self.sample_size)
        query_start = query_start
        query_end = query_start + self.ps.sample_size
        query_pattern = self.ps.dataset[query_start:query_end,0]
        stride_size = self.sample_size // 10
        print("search cosine method begin")
        score_cosine = self.ps.find_pattern_cosine(query_pattern,stride_size,kvalue)
        print("search cosine method finish")
        print("search dtw method begin")
        score_dtw = self.ps.find_pattern_dtw(query_pattern,stride_size,kvalue)
        print("search dtw method finish")
        print("search flexim method begin")
        score_flexim = self.ps.find_pattern_model_regression(
            query_pattern, stride_size, kvalue, self.neural_network_model)
        print("score flexim is {}".format(score_flexim))
        print("search flexim method finish")
        score_cosine_index_start,score_cosine_index_end = self.ps.record_index(score_cosine)
        print("score cosine start is {}".format(score_cosine_index_start))
        print("score cosine end is {}".format(score_cosine_index_end))
        score_dtw_index_start,score_dtw_index_end = self.ps.record_index(score_dtw)
        print("score dtw start is {}".format(score_dtw_index_start))
        print("score dtw end is {}".format(score_dtw_index_end))
        score_flexim_index_start,score_flexim_index_end = self.ps.record_index(score_flexim)
        print("score flexim start is {}".format(score_flexim_index_start))
        print("score flexim end is {}".format(score_flexim_index_end))
        #self.ps.plot_compare_curve(query_start,query_start+self.ps.sample_size,score_cosine_index_start,score_cosine_index_end,1)
        #self.ps.plot_compare_curve(query_start,query_start+self.ps.sample_size,score_dtw_index_start,score_dtw_index_end,2)
        #self.ps.plot_compare_curve(query_start,query_start+self.ps.sample_size,score_flexim_index_start,score_flexim_index_end,3)
        
        
    def gen_naive_params(self,positive):
        # naives = []
        # m         = len(self.params_non_lin) + len(self.params_lin)
        # if positive:
        #     naives.append([cluster_label]+[0]*m+[1])
        # for i in range(1,len(self.params_lin)):
        #     temp = [cluster_label]+[0]*m + [1]
        #     print("temp is {}".format(temp))
        #     #limit of each dimension
        #     temp[i]=self.limits[i][0]-(0 if positive else self.delta)
        #     naives.append(temp)
        #     temp = [cluster_label]+[0]*m + [1 if positive else 0]
        #     temp[i]=self.limits[i][1]+(0 if positive else self.delta)
        #     naives.append(temp)
        # return naives
        num = 0
        naives = []
        m = len(self.params_lin)
        cluster_one_hot = to_categorical(self.current_cluster-1, num_classes=self.cluster_number).astype('int').tolist()
        if positive:
            temp = cluster_one_hot + [0]*m + [1]
            naives.append(temp)
        num += 1
        while num < self.n_points//4 -12:
            for i in range(len(self.params_lin)):
                gap = self.limits[self.current_cluster][i][1] - self.limits[self.current_cluster][i][0]
                temp = cluster_one_hot + [0]*m + [1 if positive else 0]
                #temp[i+len(cluster_one_hot)] = self.limits[self.current_cluster][i][0]-(0 if positive else self.delta)
                temp[i + len(cluster_one_hot)] = self.limits[self.current_cluster][i][0] - (np.random.uniform(
                    0, -1*gap) if positive else np.random.uniform(0.01, self.limits[self.current_cluster][i][0]+1 - 0.01))
                #print("temp is {}".format(temp))
                naives.append(temp)
                temp = cluster_one_hot + [0]*m + [1 if positive else 0]
                temp[i+len(cluster_one_hot)] = self.limits[self.current_cluster][i][1]+(np.random.uniform(0,-1*gap) if positive else np.random.uniform(0.01,1-self.limits[self.current_cluster][i][1]-0.01))
                naives.append(temp)
                num += 2  
        return naives
    
    def gen_data_naive_points(self,root,progress_bar,instance_number):
        # data_table = np.random.uniform(size = (instance_number,len(self.params_lin)+len(self.params_non_lin)+1))
        # for i in range(len(data_table)):
        #     data_instance = data_table[i,0:-1]
        #     label = np.argmax(self.active_lr.predict(data_instance.reshape(1,-1)),axis = 1)
        #     print("label is {}".format(label))
        #     data_table[i,-1] = label
        #     if i % (instance_number//100) == 0:
        #         progress_bar['value'] += 1
        #         root.update()
        #         time.sleep(1)
        # return data_table
        data = self.data
        generator1 = datagenerator.remote(data,self.sample_size)
        generator2 = datagenerator.remote(data,self.sample_size)
        generator3 = datagenerator.remote(data,self.sample_size)
        generator4 = datagenerator.remote(data,self.sample_size)
        next_result_id = []
        next_result_id.append(generator1.naive_negative_shift.remote(instance_number//4))
        next_result_id.append(generator2.naive_negative_rotate.remote(instance_number//4))
        next_result_id.append(generator3.naive_positive_shift.remote(instance_number//4))
        next_result_id.append(generator4.naive_positive_rotate.remote(instance_number//4))
        self.naive_result_table = ray.get(next_result_id)
        while(len(self.naive_result_table) == 0):
            print("still in data generation process")
        
        self.final_naive_result_table = np.concatenate((self.naive_result_table[0],self.naive_result_table[1]),axis = 0)
        self.final_naive_result_table = np.concatenate((self.final_naive_result_table,self.naive_result_table[2]),axis = 0)
        self.final_naive_result_table = np.concatenate((self.final_naive_result_table,self.naive_result_table[3]),axis = 0)
        self.final_original_result_table = self.final_naive_result_table[:,0:self.sample_size]
        self.final_transform_result_table = self.final_naive_result_table[:,self.sample_size:2*self.sample_size]
        self.final_label_result_table = self.final_naive_result_table[:,-1]
        progress_bar['value'] += 100
        root.update()
        time.sleep(1)
        print("finish generate naive data")
        ray.shutdown()
        # generator1.write_table(next_result_table[0],number = 1)
        # generator2.write_table(next_result_table[1],number = 2)
        # generator3.write_table(next_result_table[2],number = 3)
        # generator4.write_table(next_result_table[3],number = 4)
        # generator5.write_table(next_result_table[4],number = 5)
        
    def gen_complex_points_test(self,instance_number):
        complex_data_list = []
        for index in range(1,self.cluster_number+1):
            low_limits = np.array([self.limits[index][i][0] for i in self.limits[index].keys()])
            high_limits = np.array([self.limits[index][i][1] for i in self.limits[index].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(size = (instance_number // self.cluster_number,len(low_limits)))
            unlabeled = (unlabeled) * diff + low_limits
            unlabeled = np.round_(unlabeled,decimals = 3)
            cluster_one_hot = to_categorical(index,num_classes = self.cluster_number)
            cluster_index_table_unlabel = np.array([cluster_one_hot for _ in range(len(unlabeled))])
            unlabeled = np.append(cluster_index_table_unlabel,unlabeled,axis = 1)
            complex_data_list.append(unlabeled)
        alpha_array = np.concatenate((complex_data_list[0],complex_data_list[1]),axis = 0)
        for i in range(2,self.cluster_number):
            alpha_array = np.concatenate((alpha_array,complex_data_list[i]),axis = 0)
        transformed_table = np.random.uniform(size = (len(array),2*self.sample_size + 1))
        

    def gen_data_unlabel_complex_points(self,root,progress_bar,instance_number):
        complex_data_list = []
        for index in range(1, self.cluster_number+1):
            low_limits = np.array([self.limits[index][i][0]
                                  for i in self.limits[index].keys()])
            high_limits = np.array([self.limits[index][i][1]
                                   for i in self.limits[index].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(
                size=(instance_number//self.cluster_number, len(low_limits)))
            unlabeled = (unlabeled * diff) + low_limits
            unlabeled = np.round_(unlabeled, decimals=3)
            cluster_one_hot = to_categorical(
                index-1, num_classes=self.cluster_number)
            cluster_index_table_unlabel = np.array(
                [cluster_one_hot for _ in range(len(unlabeled))])
            unlabeled = np.append(
                cluster_index_table_unlabel, unlabeled, axis=1)
            complex_data_list.append(unlabeled)
            progress_bar['value'] += 100 // self.cluster_number
            root.update()
            time.sleep(1)
        alpha_array = np.concatenate(
            (complex_data_list[0], complex_data_list[1]), axis=0)
        for i in range(2, self.cluster_number):
            self.alpha_array = np.concatenate(
                (alpha_array, complex_data_list[i]), axis=0)
        if(progress_bar['value'] != 100 and progress_bar['value'] >= 80):
            progress_bar['value'] = 100
        


    def gen_data_complex_points(self,instance_number):
        complex_data_list = []
        for index in range(1,self.cluster_number+1):
            low_limits = np.array([self.limits[index][i][0] for i in self.limits[index].keys()])
            high_limits = np.array([self.limits[index][i][1] for i in self.limits[index].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(size = (instance_number//self.cluster_number,len(low_limits)))
            unlabeled = (unlabeled * diff) + low_limits
            unlabeled = np.round_(unlabeled,decimals = 3)
            cluster_one_hot = to_categorical(index-1,num_classes = self.cluster_number)
            cluster_index_table_unlabel = np.array([cluster_one_hot for _ in range(len(unlabeled))])
            unlabeled = np.append(cluster_index_table_unlabel,unlabeled,axis = 1)
            complex_data_list.append(unlabeled)
            # progress_bar['value'] += 100 // self.cluster_number
            # root.update()
            # time.sleep(1)
        alpha_array = np.concatenate((complex_data_list[0],complex_data_list[1]),axis = 0)
        print("alpha array shape is {}".format(alpha_array.shape))
        for i in range(2,self.cluster_number):
            print("complex data shape is {}".format(complex_data_list[i].shape))
            alpha_array = np.concatenate((alpha_array,complex_data_list[i]),axis = 0)
        # if(progress_bar['value']!=100 and progress_bar['value']>=80):
        #     progress_bar['value'] = 100
        #predict_label = np.argmax(self.active_lr.predict(alpha_array),axis = 1)
        #predict_label = self.active_lr.predict(alpha_array).reshape(-1,1)
        # self.alpha_array = np.concatenate((alpha_array,predict_label),axis = 1)
        #self.transform_complex_points(self.alpha_array)
        
        return alpha_array
        # result_id = []
        # result_id.append(ray.put(ac_generate(limits,params_non_lin,instance_number//5,active_lr)))
        # result_id.append(ray.put(ac_generate(limits,params_non_lin,instance_number//5,active_lr)))
        # result_id.append(ray.put(ac_generate(limits,params_non_lin,instance_number//5,active_lr)))
        # result_id.append(ray.put(ac_generate(limits,params_non_lin,instance_number//5,active_lr)))
        # result_id.append(ray.put(ac_generate(limits,params_non_lin,instance_number//5,active_lr)))
        # result_table = ray.get(result_id)
        # progress_bar['value'] += 50
        # root.update()
        # time.sleep(1)
        # #print("{}  result table is {}".format(0, result_table[0]))
        # print("result table shape is {}".format(result_table[0].shape))
    def transform_complex_unlabel_points(self,alpha_array):
        without_label_array = alpha_array
        transform_unlabel_table = np.random.uniform(
            size=(len(alpha_array), 2*self.sample_size))
        for i in range(len(without_label_array)):
            cluster = np.argmax(
                without_label_array[i, 0:self.cluster_number], axis=0) + 1
            # original_pattern_index = self.cluster_index_dictionary[cluster][np.random.randint(
            #     0, len(self.cluster_index_dictionary[cluster]))]
            original_pattern_index = self.sample_index[cluster-1]
            original_pattern = self.data[original_pattern_index:
                                         original_pattern_index+self.sample_size]
            transform = self.complex_transform(
                original_pattern, without_label_array[i, self.cluster_number:])[:, 1].reshape(-1, 1)
            transform_unlabel_table[i,
                                 0:self.sample_size] = original_pattern[:, 0]
            transform_unlabel_table[i, self.sample_size:2 *
                                 self.sample_size] = transform[:, 0]
        return transform_unlabel_table
        
        
    def transform_complex_points(self, alpha_array):
        without_label_array = alpha_array[:,0:-1]
        transform_table = np.random.uniform(size=(len(alpha_array),2*self.sample_size+1))
        terrible_timestamp = []
        for i in range(len(without_label_array)):
            cluster = np.argmax(without_label_array[i,0:self.cluster_number],axis = 0) + 1
            #original_pattern_index = self.cluster_index_dictionary[cluster][np.random.randint(0,len(self.cluster_index_dictionary[cluster]))]
            original_pattern_index = self.sample_index[cluster-1]
            original_pattern = self.data[original_pattern_index:original_pattern_index+self.sample_size]
            #print("original pattern shape is {}".format(original_pattern.shape))
            #transform = self.complex_transform(original_pattern,without_label_array[i,self.cluster_number:])[:,1].reshape(-1,1)
            transform_pattern = self.complex_transform(original_pattern,without_label_array[i,self.cluster_number:])
            #print("transform pattern shape is {}".format(transform_pattern.shape))
            timestamp = transform_pattern[:,0]
            transform_pattern = transform_pattern[:,1]
            #print("timestamp is {}".format(timestamp))
            #print("transform pattern is {}".format(transform_pattern[:,1]))
            sorted_timestamp = sorted(timestamp)
            if (timestamp != sorted_timestamp).any():
                terrible_timestamp.append(i)
            #print("original pattern is {}".format(original_pattern))
            #print("transform pattern is {}".format(transform_pattern))
            transitive = np.dstack((original_pattern[:,0].tolist(),transform_pattern))
            #print("transitive is {}".format(transitive))
            transitive = transitive[0].reshape(-1,).tolist()
            #print("transitive is {}".format(transitive))
            # transform_table[i,0:self.sample_size] = original_pattern[:,0]
            # transform_table[i,self.sample_size:-1] = transform_pattern[:,1]
            # transform_table[i,-1] = alpha_array[i,-1]
            transform_table[i,0:-1] = transitive
            transform_table[i,-1] = alpha_array[i,-1]
        transform_table = np.delete(transform_table,terrible_timestamp,axis = 0)
        return transform_table
    
    def transform_complex_points_2(self,alpha_array):
        without_label_array = alpha_array[:,0:-1]
        original_table = np.random.uniform(size = (len(alpha_array),self.sample_size))
        transform_table = np.random.uniform(size = (len(alpha_array),self.sample_size))
        terrible_timestamp = []
        for i in range(len(without_label_array)):
            cluster = np.argmax(without_label_array[i,0:self.cluster_number],axis = 0) + 1
            original_pattern_index = self.sample_index[cluster-1]
            original_pattern = self.data[original_pattern_index:original_pattern_index + self.sample_size]
            transform_pattern = self.complex_transform(original_pattern,without_label_array[i,self.cluster_number:])
            timestamp = transform_pattern[:,0]
            transform_pattern = transform_pattern[:,1]
            sorted_timestamp = sorted(timestamp)
            if (timestamp != sorted_timestamp).any():
                terrible_timestamp.append(i)
            original_table[i,:] = original_pattern[:,0]
            transform_table[i,:] = transform_pattern
        original_table = np.delete(original_table,terrible_timestamp,axis = 0)
        transform_table = np.delete(transform_table,terrible_timestamp,axis = 0)
        return original_table,transform_table
            
        
        
                
    def neural_network_train(self,instance,label,sample_size,validation_data,class_weight):
        print("begin train model")
        self.neural_network = nn(instance,label,sample_size,validation_data,class_weight)
        self.neural_network_model = self.neural_network.model
        print("finish train model")
    
    def neural_network_predict(self,X):
        similarity_score = self.neural_network.nn_predict(X)
        return similarity_score
        
    def complex_transform(self,original,alpha_set):
        a,b,c,d,e,f = (e for e in alpha_set)
        x = np.concatenate([np.array([range(0, original.shape[0])]).reshape((-1, 1)), original])
        x = x.reshape((2,-1))
        A = np.array([[a+1,b*original.shape[0]],[c/10,d+1]])
        B = np.array([[e * self.original.shape[0], f]])
        transform = np.matmul(A,x).T + B
        return transform

    def sorted_timestamp(self,transform_pattern):
        sorted_timestamp = np.sort(transform_pattern[:,0])
        #if np.all()
    #generate label and unlabel data based on previous alpha chose
    #After finishing user label process, then we generate data points
    def gen_data_points(self):
        positives = self.gen_naive_params(positive=True)
        negatives = self.gen_naive_params(positive=False)
        #print("positive data is {}".format(positives))
        #print("negative data is {}".format(negatives))
        #print("data format is {}".format(positives.shape))
        '''
        m         = len(self.params_non_lin) + len(self.params_lin)
        # create positive examples
        positives.append([0]*m+[1])
        for i in range(len(self.params_lin)):
            temp = [0]*m + [1]
            temp[i]=self.limits[i][0]
            positives.append(temp)
            temp = [0]*m + [1]
            temp[i]=self.limits[i][1]
            positives.append(temp)
        # create negative examples
        for i in range(len(self.params_lin)):
            temp = [0]*m + [0]
            temp[i]=self.limits[i][0]-self.delta
            negatives.append(temp)
            temp = [0]*m + [0]
            temp[i]=self.limits[i][1] + self.delta
            negatives.append(temp)
        '''
        #rray = np.array(positives + negatives)
        #concentrate positive and negative data together and round to 3
        labeled = np.array(positives + negatives)
        labeled = np.round_(labeled,decimals = 3)
        # create the pool
        # low_limits = np.array([self.limits[self.current_cluster][i][0] for i in self.limits[self.current_cluster].keys()] +
        #                       [-1 for j in range(len(self.params_non_lin))])
        # high_limits= np.array([self.limits[self.current_cluster][i][1] for i in self.limits[self.current_cluster].keys()] +
        #                       [1 for j in range(len(self.params_non_lin))])
        low_limits = np.array([self.limits[self.current_cluster][i][0] for i in self.limits[self.current_cluster].keys()])
        high_limits= np.array([self.limits[self.current_cluster][i][1] for i in self.limits[self.current_cluster].keys()])
        diff       = high_limits - low_limits

        unlabeled= np.random.uniform(size=(self.n_points, len(low_limits)))
        unlabeled= (unlabeled*diff) + low_limits
        #unlabeled= np.concatenate([unlabeled, np.zeros((unlabeled.shape[0], len(self.params_non_lin)))], axis=1)
        unlabeled= np.round_(unlabeled,decimals=3)
        cluster_one_hot = to_categorical(self.current_cluster-1,num_classes = self.cluster_number).astype(int)
        cluster_index_table_unlabel = np.array([cluster_one_hot for _ in range(len(unlabeled))])
        #unlabeled = np.append(cluster_index_table_unlabel,unlabeled,axis = 1)
        unlabeled = np.concatenate((cluster_index_table_unlabel,unlabeled),axis = 1)
        return labeled, unlabeled
    
    def transform_list(self,original_list,cluster_label):
        transfer_list = []
        for i in range(len(original_list)):
            b = [cluster_label]
            for j in range(len(original_list[i])):
                b.append(original_list[i][j])
            transfer_list.append(b)
        return transfer_list
    
    # def generate_AE_data(self,instance_number):
    #     dataset = self.data
    #     batch_size = self.sample_size
    #     AE_data_table = np.ones((instance_number,batch_size),dtype = np.float32)
    #     self.AE_index_table = np.ones((instance_number,1))
    #     for i in range(instance_number):
    #         start_point = np.random.randint(0,len(dataset)-1-batch_size)
    #         data = dataset[start_point:start_point+batch_size]
    #         AE_data_table[i,:] = data.values.reshape(1,batch_size)
    #         self.AE_index_table[i,0] = start_point
    #     ae = AE_Model(input_dim = batch_size,hidden_dim=batch_size//5,latent_dim=batch_size//10,dataset=AE_data_table)
    #     self.AutoEncoder = ae.model
    #     labels,self.kmeans = get_cluster_label(kmax=10,test_data=AE_data_table,autoencoder=self.AutoEncoder)
    #     #self.AE_index_table = np.concatenate()
    #     self.train_autoencoder = True
    #     #print("maximal cluster number is {}".format(self.max_k))
        
    def collect_cluster_index(self,instance_number):
        np.random.seed(6)
        for i in range(instance_number):
            start_point = np.random.randint(0,len(self.data)-1-self.sample_size)
            data = self.data[start_point:start_point + self.sample_size,0]
            encode_data = self.encoder_model(data.reshape(1,-1))
            cluster = self.kmeans_model.predict(encode_data)[0] + 1
            self.cluster_index_dictionary[cluster].append(start_point)

    
    def instance_cluster_label(self,data,kmeans):
        encode_data = self.encoder_model(np.array(data).reshape(1,-1))
        #print("encode_data is {}".format(encode_data))
        encode_data = np.array(encode_data)[0].tolist()
        cluster_label = kmeans.predict(np.reshape(encode_data,(1,-1)))[0] + 1
        print("instance cluster label is {}".format(cluster_label))
        return cluster_label
    
    def change_dictionary(self):
        self.dictionary[self.current_cluster].append(self.current_setting)
        

    def get_label(self):
        return self.label
    
    def submit(self):
        #print("run sumbit")
        cluster_one_hot = to_categorical(self.current_cluster-1, num_classes=self.cluster_number).astype('int').tolist()
        #print("cluster one hot is {}".format(cluster_one_hot))
        #print("self.generate data is {}".format(self.generated_data))
        #self.generated_data = np.concatenate([np.array([[e[-1].get() for e in self.params_non_lin]+[e[-1].get() for e in self.params_lin]+[int(self.label.get())]]), self.generated_data])
        #np.savetxt('generated.csv', X=self.generated_data,fmt='%3d', delimiter=',')
        self.generated_data = np.concatenate([np.array([[e for e in cluster_one_hot]+[e[-1].get() for e in self.params_lin]+[int(self.label.get())]]), self.generated_data])
        self.fetch()
        self.action()

    def save(self):
        #np.savetxt('generated.csv', X=self.generated_data,fmt='%3d', delimiter=',')
        np.savetxt('generated.csv', X = self.generated_data,fmt='%.4f',delimiter=',')
        #self.set_alphas([(random.random()*2)-1 for i in range(len(self.params_lin))], [(random.random()*2)-1 for i in range(len(self.params_non_lin))])
        #self.action()
        
    def action(self):
        if self.original is not None:
            transform = self.original
            for par in self.params_non_lin:
                # I am confused about this part
                transform,_,_ = eval('self.F.'+par[0])(transform,alpha=par[-1].get())
            a, b, c, d, e, f = (e[-1].get() for e in self.params_lin)
            x = np.concatenate([np.array([range(0, transform.shape[0])]).reshape((-1,1)), transform])
            x = x.reshape((2, -1))
            #print("x is {}".format(x))
            #why define it in this way
            A = np.array([[a+1, b * self.original.shape[0]], [c/10, d+1]])
            #print("A is {}".format(A))
            B = np.array([[e * self.original.shape[0], f]])
            #print("B is {}".format(B))
            transform = np.matmul(A, x).T + B
            #print("transform is {}".format(transform))
            timestamp = transform[:,0]
            #print("timestamp is {}".format(timestamp))
            sorted_timestamp = sorted(timestamp)
            #print("sorted timestamp is {}".format(sorted_timestamp))
            if (timestamp != sorted_timestamp).any():
                print("skip this one")
                self.submit()
            else:
                print("normal one")
                self.caller.plot_sample(self.original, transform)
            
    def search(self, var):
        for i,e in enumerate(self.params_non_lin):
            if var == e[-1]:
                return i
    
    #     
    def retransform(self,transform):
        print("run retransform")
        gap = ((transform[-1,0] - transform[0,0])/len(transform))
        #print("transform[0,0] is {}".format(transform[0,0]))
        #print("transform[-1,0] is {}".format(transform[-1,0]))
        if transform[0,0] >50 or transform[-1,0] < 0:
            print("run first condition")
            gap = 50/len(transform)
            change_transform = np.random.rand(len(transform),2)
            for i in range(len(change_transform)):
                change_transform[i,0] = 0+i*gap
                change_transform[i,1] = random.uniform(-0.5,0.5)
            return change_transform
        if transform[0,0]-gap>0:
            print("run second condition")
            if gap == 0:
                gap = 1
            n = int((transform[0,0] - 0)//gap)
            change_transform = np.random.rand(int(n),2)
            for i in range(0,n-1):
                change_transform[i,0] = 0+i*gap
                change_transform[i, 1] = random.uniform(-0.5, 0.5)
            # change_transform = np.concatenate((change_transform,transform),axis = 0)
            change_transform[n-1,0] = 0 + (n-1)*gap
            change_transform[n-1,1] = transform[0,1]
            return change_transform
        if transform[-1,0] + gap <50:
            print("run third condition")
            if gap == 0:
                gap = 1
            n = int((50 - int(transform[-1,0]))//gap)
            change_transform = np.random.rand(n,2)
            change_transform[0, 0] = transform[-1,0]
            change_transform[0, 1] = transform[-1,1]
            for i in range(1,n):
                change_transform[i,0] = transform[-1,0]+ i*gap
                change_transform[i,1] = random.uniform(-0.5,0.5)
            # change_transform = np.concatenate((transform,change_transform),axis = 0)
            return change_transform
        else:
            return transform
        
    def retransform_2(self,transform):
        gap = ((transform[-1,0]-transform[0,0])/len(transform))
        #print("len of transform is {}".format(len(transform)))
        #print("gap is {}".format(gap))
        if gap == 0:
            gap = 1
        if transform[0,0] >= 50 or transform[-1,0] <= 0:
            print("retransform_2 condition 1")
            gap = 50 // len(transform)
            change_transform = np.random.rand(len(transform),2)
            for i in range(len(change_transform)):
                change_transform[i,0] = 0+ i*gap
                change_transform[i,1] = random.uniform(-0.5,0.5)
            return change_transform
        
        elif transform[0,0] - gap >= 0 and transform[-1,0] > 50:
            print("retransform_2 condition 2")
            n = int((transform[0,0]-0)//gap)
            change_transform = np.random.rand(n,2)
            for i in range(0,n-1):
                change_transform[i,0] = 0 + i*gap
                change_transform[i,1] = random.uniform(-0.5,0.5)
            change_transform[n-1,0] = 0 + (n-1)*gap
            change_transform[n-1, 1] = transform[0, 1]
            for i in range(len(transform)-1,-1,-1):
                if transform[i,0] > 50:
                    index_value = i
                    break
            transform = np.delete(transform,slice(index_value,len(transform)+1),0)
            change_transform = np.concatenate((change_transform,transform),axis = 0)
            return change_transform
        
        elif transform[-1, 0] + gap <= 50 and transform[0,0] < 0:
            print("retransform_2 condition 3")
            n = int((50 - int(transform[-1, 0]))//gap)
            change_transform = np.random.rand(n, 2)
            change_transform[0, 0] = transform[-1, 0]
            change_transform[0, 1] = transform[-1, 1]
            for i in range(1, n):
                change_transform[i, 0] = transform[-1, 0] + i*gap
                change_transform[i, 1] = random.uniform(-0.5, 0.5)
            for i in range(0,len(transform)):
                if transform[i,0] < 0:
                    index_value = i
                    break
            transform = np.delete(transform,slice(0,index_value+1),0)
            change_transform = np.concatenate((transform,change_transform),axis = 0)               
            # change_transform = np.concatenate((transform,change_transform),axis = 0)
            return change_transform
        
        #figure out problem
        elif transform[0,0] >= 0 and transform[-1,0] <= 50:
            print("retransform_2 condition 4")
            n = int((transform[0, 0]-0)//gap)
            m = int((50 - transform[-1,0])//gap)
            if n!=0 and m == 0:
                change_transform_n = np.random.rand(n,2)
                for i in range(n):
                    change_transform_n[i,0] = 0 + i*gap
                    change_transform_n[i, 1] = random.uniform(-0.5, 0.5)
                change_transform = np.concatenate((change_transform_n,transform),axis = 0)
                return change_transform
            elif m!=0 and n == 0:
                change_transform_m = np.random.rand(m,2)
                for j in range(m):
                    change_transform_m[j,0] = transform[-1,0] + j*gap
                    change_transform_m[j,1] = random.uniform(-0.5,0.5)
                change_transform = np.concatenate((transform,change_transform_m),axis = 0)
                return change_transform
            elif m!= 0 and n!=0:
                change_transform_n = np.random.rand(n, 2)
                for i in range(n):
                    change_transform_n[i, 0] = 0 + i*gap
                    change_transform_n[i, 1] = random.uniform(-0.5, 0.5)
                change_transform = np.concatenate((change_transform_n, transform), axis=0)
                change_transform_m = np.random.rand(m, 2)
                for j in range(m):
                    change_transform_m[j, 0] = transform[-1, 0] + j*gap
                    change_transform_m[j, 1] = random.uniform(-0.5, 0.5)
                change_transform = np.concatenate((transform, change_transform_m), axis=0)
                return change_transform
            else:
                change_transform = transform
                return change_transform
            
        elif transform[0,0] < 0 and transform[-1,0] > 50:
            print("retransform_2 condition 5")
            for i in range(1,len(transform)):
                if transform[i,0] >=0:
                    index_value_1 = i
                    break
            for j in range(len(transform)-1,-1,-1):
                if transform[j,0] <=50:
                    index_value_2 = j
                    break
            transform = np.delete(transform,slice(i,j+1),0)
            return transform
        else:
            #print("retransform_2 condition 6")
            return transform 
        
        
    def remove_unsort(self,para):
        problem_index_list = []
        for i in range(len(para)):
            cluster_number_instance = np.argmax(para[i,0:self.cluster_number],axis = 0)+1
            current_index = self.sample_index[cluster_number_instance]
            #original_pattern = 
    
            
    def transform(self, param, func):
        # update the transformed variable and redraw the results
        # apply non-linears in order
        # then apply linears
        pass

    def file_opener(self):
        path = filedialog.askopenfile(initialdir=os.getcwd() + "/")
        print("path name is {}".format(path.name))
        #temp = pd.read_csv(path.name, index_col=None, header=None)
        temp = pd.read_csv(path.name)
        temp = temp["close"].values
        directory_name = path.name[0:-4]
        file = open(directory_name+".txt","r")
        # with open(directory_name+".txt",'r') as file:
        #     lines = file.readlines()
        lines = file.readlines()
        self.sample_size = int(lines[0].strip('\n'))
        self.cluster_number = int(lines[2].strip('\n'))
        print("sample size is {}".format(self.sample_size))
        print("cluster number is {}".format(self.cluster_number))
        cluster_one_hot = to_categorical(self.current_cluster-1, num_classes=self.cluster_number).astype('int').tolist()
        self.generated_data = np.array([[e for e in cluster_one_hot]+[0 for _ in self.params_lin]+[1]])
        self.encoder_model = tf.keras.models.load_model(directory_name + "/encoder/")
        self.decoder_mdoel = tf.keras.models.load_model(directory_name + "/decoder/")
        self.kmeans_model = joblib.load(directory_name + "/kmeans_model.m")
        scaler = MinMaxScaler()
        #self.data = (temp-mn)/(mx-mn)
        self.data = scaler.fit_transform(temp.reshape(-1,1))
        print("data type of dataset is {}".format(type(self.data)))
        self.caller.do_plot(self.data, None, 0)
        self.caller.activate_buttons()

    def sample(self):
        self.__init__cluster_index_dictionary__()
        self.__init__limit_dictionary__()
        self.collect_cluster_index(instance_number=30000)
        #print("cluster dictionary is {}".format(self.cluster_index_dictionary))
        self.caller.__slides__()
        self.caller.activate_scales()
        interval = len(self.cluster_index_dictionary[self.current_cluster])
        np.random.seed(15)
        print("random index is {}".format(np.random.randint(0, interval)))
        i = self.cluster_index_dictionary[self.current_cluster][np.random.randint(0,interval)]
        print("index i is {}".format(i))
        j = i + self.sample_size
        print("add start index into sample index set")
        self.sample_index.append(i)
        self.original = self.data[i:j] #data shape is (470,1)
        self.caller.plot_data(self.data)
        self.caller.draw_sample_window(i, j)
        self.caller.plot_sample(self.original, None)
        self.action()
        print("first sample already finished")
        
    def resample(self):
        print("resample process begin")
        self.caller.__slides__()
        self.caller.activate_scales()
        interval = len(self.cluster_index_dictionary[self.current_cluster])
        np.random.seed(30)
        i = self.cluster_index_dictionary[self.current_cluster][np.random.randint(0, interval)]
        j = i + self.sample_size
        self.sample_index.append(i)
        self.original = self.data[i:j]  # data shape is (470,1)
        self.caller.plot_data(self.data)
        self.caller.draw_sample_window(i,j)
        self.caller.plot_sample(self.original,None)
        self.action()
        print("resample has already finished")

    def get_linears(self):
        return [e[1:] for e in self.params_lin]
    
    def get_non_linears(self):
        return [e[1:] for e in self.params_non_lin]

    def set_alphas(self, alpha_lin, alpha_non_lin):
        for i, par in enumerate(self.params_lin):
            par[-1].set(alpha_lin[i])
        for i, par in enumerate(self.params_non_lin):
            par[-1].set(alpha_non_lin[i])
