from array import array
from cProfile import label
from curses import window
from email import generator
from multiprocessing.sharedctypes import Value
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import ray
#from Active_modify import Active
from AE import AE_Model
import multiprocessing
from multiprocessing import cpu_count
import time
from multiprocessing import Pool
from dtaidistance import dtw
ray.init()
@ray.remote
class generator(object):
    def __init__(self, dataset,window_size,max_distance):
        self.dataset = dataset
        self.window_size = window_size
        self.max_distance = max_distance
        print("has change code 3")
        
        
    def test1(self,x):
        count = 0
        for i in range(x):
            count+=1
        return count
    
    def test2(self,y,z):
        count = 0
        for i in range(y):
            count+= z
        return count
            
    def read_file(self,dataurl):
        df = pd.read_csv(dataurl, header=None)
        df = df.iloc[:, 1:8]
        df = df.drop([0], axis=0)
        df = df.values.astype('float32')
        df = df.reshape(-1,)
        return df
        
        
    def naive_positive(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size+1),dtype = float)
        insert_number = 0
        while insert_number < instance_number:
            begin_index = random.randint(0,len(dataset)-0.1*window_size)
            original_pattern = dataset[begin_index:begin_index+window_size]
            compare_index = int(begin_index+0.1*window_size)
            compare_pattern = dataset[compare_index:compare_index+window_size]
            result_table[insert_number,0:window_size] = original_pattern
            result_table[insert_number,window_size:2*window_size] = compare_pattern
            result_table[2*window_size] = 1
            insert_number += 1
        return result_table
    
    def naive_negative(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number, 2*window_size+1), dtype=float)
        insert_number = 0
        while insert_number < instance_number:
            begin_index = random.randint(window_size+1, len(dataset))
            original_pattern = dataset[begin_index:begin_index+window_size]
            compare_index = int(begin_index+window_size)
            compare_pattern = dataset[compare_index:compare_index+window_size]
            result_table[insert_number, 0:window_size] = original_pattern
            result_table[insert_number, window_size:2*window_size] = compare_pattern
            result_table[2*window_size] = 0
            insert_number += 1
        return result_table
    
    def naive_positive_shift(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size+1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = 0.01,high = 0.03)
            original_pattern_up,original_pattern_down = self.shift(original_pattern,distance)
            result_table[2*insert_number,0:window_size] = original_pattern
            result_table[2*insert_number,window_size:-1] = original_pattern_up
            result_table[2*insert_number,-1] = 1
            result_table[2*insert_number + 1,0:window_size] = original_pattern
            result_table[2*insert_number + 1,window_size:-1] = original_pattern_down
            result_table[2*insert_number + 1,-1] = 1
            insert_number += 2
        return result_table
    
    def naive_negative_shift(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size+1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = 0.5,high = 1)
            original_pattern_up,original_pattern_down = self.shift(original_pattern,distance)
            result_table[2*insert_number,0:window_size] = original_pattern
            result_table[2*insert_number,window_size:-1] = original_pattern_up
            result_table[2*insert_number,-1] = 0
            result_table[2*insert_number + 1,0:window_size] = original_pattern
            result_table[2*insert_number + 1,window_size:-1] = original_pattern_down
            result_table[2*insert_number + 1,-1] = 0
            insert_number += 2
        return result_table
    
    def naive_negative_rotate(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size + 1),dtype = float)
        insert_number = 0
        np.random.seed(8)
        while insert_number < (instance_number):
            begin_index = random.randint(
                0.5*window_size, len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size, 0]
            rotate_pattern = -1 * original_pattern
            result_table[insert_number,0:window_size] = original_pattern
            result_table[insert_number,window_size:-1] = rotate_pattern
            result_table[insert_number,-1] = 0
            insert_number += 1
        return result_table
            
    def write_table(self,table,number):
        pd.DataFrame(table).to_csv('naive_negative'+number+'.csv')
        
    def transform(self,original,transform_function):
        if original is not None:
            transform = original
            a,b,c,d,e,f = (e for e in transform_function)
            x = np.concatenate([np.array(range(0,transform.shape[0])).reshape((-1,1)),transform])
            x = x.reshape((2,-1))
            A = np.array([[a+1,b*original.shape[0]],[c/10,d+1]])
            B = np.array([[e * original.shape[0],f]])
            transform = np.matmul(A,x).T + B
        return transform
    
    def shift(self,pattern,distance):
        pattern_upshift = [pattern[i] + distance for i in range(len(pattern))]
        pattern_downshift = [pattern[i] - distance for i in range(len(pattern))]
        return pattern_upshift,pattern_downshift            
        
def ac_generate(limits,params_non_lin,instance_number,ac):
    low_limits = np.array([limits[i][0] for i in limits.keys()] + [-1 for j in range(len(params_non_lin))])
    high_limits = np.array([limits[i][1] for i in limits.keys()] + [1 for j in range(len(params_non_lin))])
    diff = high_limits - low_limits
    unlabeled = np.random.uniform(size = (instance_number,len(low_limits)))
    unlabeled = (unlabeled*diff)+ low_limits
    unlabeled = np.round_(unlabeled,decimals = 3)
    label = ac.predict(unlabeled)
    ac_data = np.hstack((unlabeled,label))
    return ac_data
    
        
        
    #generate complex positive and negative
    # def complex_pn(self,active:Active,autoencoder:AE_Model,alpha_set,cluster_centers_):
    #     dataset = self.dataset
    #     instance_number = self.instance_number
    #     window_size = self.window_size
    #     result_table = np.ones((instance_number, 2*window_size+1), dtype=float)
    #     insert_number = 0
    #     while insert_number<instance_number:
    #         begin_index = random.randint(0,120000)
    #         original_pattern = dataset[begin_index:begin_index+window_size]
    #         encode_input_data = autoencoder.model.encoder(original_pattern)
    #         cluster_label = autoencoder.get_cluster_label_singe(encode_input_data, cluster_centers_)
    #         label = active.learner.predict_label_single(alpha_set, cluster_label)
    #         # I instantiate variable of class defined in your side
    #         transform_data_pattern = transform(input_data, alpha_set)
    #         result_table[insert_number,0:window_size] = original_pattern
    #         result_table[insert_number,window_size:2*window_size] = transform_data_pattern
    #         result_table[insert_number,2*window_size] = label
    #         insert_number += 1
    #     return result_table
def positive(instance_number,window_size):
     # begin_index = random.randint(0,100000)
     # original_pattern = self.dataset[begin_index:begin_index+window_size]
    dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")
    result_table = np.ones((instance_number, 2*window_size+1), dtype=float)
    insert_number = 0
    while insert_number < instance_number:
        begin_index = random.randint(0, 120000)
        original_pattern = dataset[begin_index:begin_index+window_size]
        k_value = random.randint(-20, 20)
        compare_index = begin_index + k_value*window_size
        compare_pattern = dataset[compare_index:compare_index+window_size]
        if(((compare_pattern.all() != original_pattern.all()))):
            cos_sim = dot(np.transpose(compare_pattern), original_pattern) / \
                (norm(compare_pattern)*norm(original_pattern))
            if cos_sim >= 0.8:
                result_table[insert_number,0:window_size] = original_pattern
                result_table[insert_number, window_size:2 *window_size] = compare_pattern
                result_table[insert_number, 2*window_size] = 1
                insert_number += 1
    return result_table

#test_part
def read_data_csv(self,dataurl):
    temp = pd.read_csv("./"+dataurl, index_col=None, header=None)
    min = temp.min()
    max = temp.max()
    data = (temp-min)/(max-min)
    return data
