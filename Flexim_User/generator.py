from array import array
from cProfile import label
from email import generator
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
ray.init()
@ray.remote
class datagenerator(object):
    def __init__(self, dataset,window_size):
        self.dataset = dataset
        self.window_size = window_size
        
        
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
    
    def naive_positive_shift(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size+1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = -0.003,high = 0.003)
            original_pattern_up,original_pattern_down = self.shift(original_pattern,distance)
            result_table[2*insert_number,0:window_size] = original_pattern
            result_table[2*insert_number,window_size:-1] = original_pattern_up
            result_table[2*insert_number,-1] = 1
            result_table[2*insert_number + 1,0:window_size] = original_pattern
            result_table[2*insert_number + 1,window_size:-1] = original_pattern_down
            result_table[2*insert_number + 1,-1] = 1
            insert_number += 2
        return result_table
    
    def naive_positive_shift_2(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        original_result_table = np.ones((instance_number,window_size),dtype = float)
        transform_result_table = np.ones((instance_number,window_size),dtype = float)
        label_table = np.ones((instance_number,1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = -0.003,high = 0.003)
            transform_pattern_up,transform_pattern_down = self.shift(original_pattern,distance)
            original_result_table[2*insert_number,:] = original_pattern
            transform_result_table[2*insert_number,:] = transform_pattern_up
            original_result_table[2*insert_number+1,:] = original_pattern
            transform_result_table[2*insert_number+1,:] = transform_pattern_down
            insert_number += 2
        return original_result_table,transform_result_table,label_table
    
    def naive_negative_shift(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size+1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = -0.3,high = 0.3)
            original_pattern_up,original_pattern_down = self.shift(original_pattern,distance)
            result_table[2*insert_number,0:window_size] = original_pattern
            result_table[2*insert_number,window_size:-1] = original_pattern_up
            result_table[2*insert_number,-1] = 0
            result_table[2*insert_number + 1,0:window_size] = original_pattern
            result_table[2*insert_number + 1,window_size:-1] = original_pattern_down
            result_table[2*insert_number + 1,-1] = 0
            insert_number += 2
        return result_table
    
    def naive_negative_shift_2(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        original_result_table = np.ones((instance_number,window_size),dtype = float)
        transform_result_table = np.ones((instance_number,window_size),dtype = float)
        label_table = np.zeros((instance_number,1),dtype = float)
        insert_number = 0
        np.random.seed(6)
        while insert_number < (instance_number)//2:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            distance = np.random.uniform(low = -0.3,high = 0.3)
            transform_pattern_up,transform_pattern_down = self.shift(original_pattern,distance)
            original_result_table[2*insert_number,:] = original_pattern
            transform_result_table[2*insert_number,:] = transform_pattern_up
            original_result_table[2*insert_number+1,:] = original_pattern
            transform_result_table[2*insert_number+1,:] = transform_pattern_down
            insert_number += 2
        return original_result_table,transform_result_table,label_table
    
    def naive_positive_rotate(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        result_table = np.ones((instance_number,2*window_size + 1),dtype = float)
        insert_number = 0
        np.random.seed(8)
        while insert_number < (instance_number):
            begin_index = random.randint(
                0.5*window_size, len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size, 0]
            rotate_pattern = original_pattern
            result_table[insert_number,0:window_size] = original_pattern
            result_table[insert_number,window_size:-1] = rotate_pattern
            result_table[insert_number,-1] = 0
            insert_number += 1
        return result_table
    
    def naive_positive_rotate_2(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        original_result_table = np.ones((instance_number,window_size),dtype = float)
        transform_result_table = np.ones((instance_number,window_size),dtype = float)
        label_table = np.ones((instance_number,1),dtype = float)
        insert_number = 0
        np.random.seed(8)
        while insert_number < (instance_number):
            begin_index = random.randint(
                0.5*window_size, len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size, 0]
            rotate_pattern = original_pattern
            original_result_table[insert_number,:] = original_pattern
            transform_result_table[insert_number,:] = rotate_pattern
            insert_number += 1
        return original_result_table,transform_result_table,label_table
    
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
            rotate_place = random.randint(1,window_size // 5 + 1)
            rotate_pattern = np.roll(original_pattern,rotate_place)
            #rotate_pattern = -1 * original_pattern
            result_table[insert_number,0:window_size] = original_pattern
            result_table[insert_number,window_size:-1] = rotate_pattern
            result_table[insert_number,-1] = 0
            insert_number += 1
        return result_table
    
    def naive_negative_rotate_2(self,instance_number):
        dataset = self.dataset
        window_size = self.window_size
        original_result_table = np.ones((instance_number,window_size),dtype = float)
        transform_result_table = np.ones((instance_number,window_size),dtype = float)
        label_table = np.zeros((instance_number,1),dtype = float)
        insert_number = 0
        np.random.seed(8)
        while insert_number < instance_number:
            begin_index = random.randint(0.5*window_size,len(dataset) - 1.5*window_size)
            original_pattern = dataset[begin_index:begin_index + window_size,0]
            rotate_place = random.randint(1,window_size // 5 + 1)
            rotate_pattern = np.roll(original_pattern,rotate_place)
            original_result_table[insert_number,:] = original_pattern
            transform_result_table[insert_number,:] = rotate_pattern
            insert_number += 1
        return original_result_table,transform_result_table,label_table
            
    
    def shift(self,pattern,distance):
        pattern_upshift = [pattern[i] + distance for i in range(len(pattern))]
        pattern_downshift = [pattern[i] - distance for i in range(len(pattern))]
        return pattern_upshift,pattern_downshift  
    # def naive_positive(self,instance_number):
    #     dataset = self.dataset.values
    #     window_size = self.window_size
    #     # begin_index = random.randint(0,100000)
    #     # original_pattern = self.dataset[begin_index:begin_index+window_size]
    #     result_table = np.ones((instance_number,2*window_size+1),dtype = float)
    #     insert_number = 0
    #     while insert_number<instance_number:
    #         begin_index = random.randint(0.5*window_size,len(dataset)-window_size)
    #         original_pattern = dataset[begin_index:begin_index+window_size]
    #         k_value = random.randint(-5,5)
    #         compare_index = begin_index + k_value*0.1*window_size
    #         compare_pattern = dataset[compare_index:compare_index+window_size]
    #         #print("len of compare pattern is {}".format(len(compare_pattern)))
    #         #print("compare pattern is {}".format(compare_pattern))
    #         #if(((compare_pattern[0]!=original_pattern[0]))):
    #         cos_sim = dot(np.transpose(compare_pattern),original_pattern)/(norm(compare_pattern)*norm(original_pattern))
    #         if cos_sim >= 0.8:
    #                 result_table[insert_number,0:window_size] = original_pattern
    #                 result_table[insert_number,window_size:2*window_size] = compare_pattern
    #                 result_table[insert_number,2*window_size] = 1
    #                 insert_number += 1
    #     return result_table
    
    # def naive_negative(self, instance_number):
    #     print("begin to run naive_negative")
    #     dataset = self.dataset.values
    #     #print("dataset is {}".format(dataset))
    #     window_size = self.window_size
    #     result_table = np.ones((instance_number,2*window_size+1),dtype = float)
    #     insert_number = 0
    #     while insert_number<instance_number:
    #         begin_index = random.randint(0,len(dataset)-window_size)
    #         original_pattern = dataset[begin_index:begin_index+window_size,0]
    #         compare_index = random.randint(0,len(dataset)-window_size)
    #         compare_pattern = dataset[compare_index:compare_index+window_size,0]
    #         #print("len of compare pattern is {}".format(len(compare_pattern)))
    #         cos_sim = dot(np.transpose(compare_pattern),original_pattern)/(norm(compare_pattern)*norm(original_pattern))
    #         if cos_sim <=0.2:
    #             result_table[insert_number,0:window_size] = original_pattern
    #             result_table[insert_number,window_size:2*window_size] = compare_pattern
    #             result_table[insert_number,2*window_size] = 0
    #             insert_number += 1
    #     return result_table 
   
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


#test_part
def read_data_csv(dataurl):
    df = pd.read_csv(dataurl, header=None)
    df = df.iloc[:, 1:8]
    df = df.drop([0], axis=0)
    df = df.values.astype('float32')
    df = df.reshape(-1,)
    return df


# data = read_data_csv(
#     "/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")
# generator1 = datagenerator.remote(data, 50)
# generator2 = datagenerator.remote(data, 50)
# generator3 = datagenerator.remote(data, 50)
# generator4 = datagenerator.remote(data, 50)
# generator5 = datagenerator.remote(data, 50)
# generator_id = []
# generator_id.append(generator1)
# generator_id.append(generator2)
# generator_id.append(generator3)
# generator_id.append(generator4)
# generator_id.append(generator5)
# result_id = []
# # for i in range(5):
# #     result_id.append(generator1.naive_positive.remote(5000//5))
# #     result_id.append(generator2.naive_negative.remote(5000//5))
# #     result_id.append(generator3.naive_positive.remote(5000//5))
# #     result_id.append(generator4.naive_negative.remote(5000//5))
# #     result_id.append(generator5.naive_positive.remote(5000//5))
# #             # progress_bar['value'] += 50
# #             # root.update()
# #             # time.sleep(1)
# #     print("run result_id[0]".format(result_id[0]))
# result_id.append(generator_id[0].naive_negative.remote(1000))
# print("result table is {}".format(ray.get(result_id[0])))
#print("result 1 is {}".format(ray.get(result_id[0])))

# if __name__ == '__main__':
#     dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")


# generator1 = datagenerator.remote()
# generator2 = datagenerator.remote()
# generator3 = datagenerator.remote()
# generator4 = datagenerator.remote()
# generator5 = datagenerator.remote()

# result_id = []
# generator_id = []
# generator_id.append(generator1)
# generator_id.append(generator2)
# generator_id.append(generator3)
# generator_id.append(generator4)
# generator_id.append(generator5)
    
# for i in range(5):
#     result_id.append(generator_id[i].naive_positive.remote())
# tic = time.time()
# results = ray.get(result_id)
# toc = time.time()
# print("ray time elapsed is {}".format(toc-tic))
# print("all ray process result shape is {}".format(np.shape(results)))
# print("ray process 1 result shape is {}".format(np.shape(ray.get(result_id[0]))))

# generator_test = datagenerator.remote(dataset, window_size=50, instance_number=1000)
# tic = time.time()
# result = ray.get(generator_test.naive_positive.remote())
# toc = time.time()
# print("ray time elapsed is {}".format(toc-tic))
# print("result is {}".format(np.shape(results)))

# tic = time.time()
# result = positive(instance_number=500, window_size=50)
# toc = time.time()
# print("time elapse is {}".format(toc-tic))
# print("result is {}".format(np.shape(result)))


# tic = time.time()
# result = positive(instance_number=2500, window_size=50)
# toc = time.time()
# print("time elapse is {}".format(toc-tic))
# print("result shape is {}".format(np.shape(result)))

# tic = time.time()
# generator.test2()
# toc = time.time()
# print("test 2 run Done is {:.4f} seconds".format(toc-tic))







    #python multiprocess module 
    # generator1 = datagenerator(dataset,50)
    # generator2 = datagenerator(dataset, 50)
    # generator3 = datagenerator(dataset, 50)
    # generator4 = datagenerator(dataset, 50)
    # generator5 = datagenerator(dataset, 50)
    # generatorx = datagenerator(dataset, 50)
    # tic = time.time()
    # p1 = multiprocessing.Process(target=generator1.test2,args=(200000000,))
    # p2 = multiprocessing.Process(target=generator2.test2, args=(200000000,))
    # p3 = multiprocessing.Process(target=generator3.test2, args=(200000000,))
    # p4 = multiprocessing.Process(target=generator4.test2, args=(200000000,))
    # p5 = multiprocessing.Process(target=generator5.test2, args=(200000000,))
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # toc = time.time()
    # print("multiprocess time is {}".format(toc-tic))
    
    # tic = time.time()
    # p = Pool(processes=5)
    # data = p.starmap(generator1.test2, [(200000000,10) for _ in range(5)])
    # p.close()
    # print(data)
    # toc = time.time()
    # print("multiprocess time is {}".format(toc-tic))
    
    
    # tic = time.time()
    # result = generatorx.test2(200000000,10)
    # toc = time.time()
    # print("result is {}".format(result))
    # print("single process time is {}".format(toc-tic))
                
                

        
        
