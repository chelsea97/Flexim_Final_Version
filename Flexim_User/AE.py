from base64 import encode
from cgi import test
from random import shuffle
from tabnanny import verbose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import math
from multiprocessing import Pool
import multiprocessing
import ray
#from Cluster import find_k_DBSCAN, find_k_kmeans, plot_DSBCAN, plot_cluster,plot_find_k,cluster_effect
#construct encoder and decoder
class AutoEncoder(tf.keras.Model):
    def __init__(self,input_dim,hidden_dim,latent_dim):
        super(AutoEncoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential()
        #self.encoder.add(tf.keras.layers.InputLayer(input_shape=(1,self.input_dim)))
        self.encoder.add(tf.keras.layers.BatchNormalization(axis = -1))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.encoder.add(tf.keras.layers.Dense(self.latent_dim,activation = 'relu'))
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'),
            tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'),
            tf.keras.layers.Dense(self.input_dim)
        ])
    
    
    #get decoded array based on input array via process of encode and decode
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # def get_model(self,input_data):
    #     self.model = tf.keras.Model(input_data)
    #     self.model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    #     self.model.fit(input_data, input_data, epochs=3000,
    #                    batch_size=40, shuffle=True)
    #     return self.model

#you should create object under of this class to actually get label 
class AE_Model(tf.keras.Model):
    def __init__(self,input_dim,hidden_dim,latent_dim,dataset):
        super(AE_Model,self).__init__()
        # batch_number = len(dataset)//input_dim
        # input_data = dataset[0:batch_number*input_dim].reshape(batch_number,input_dim)
        self.input_dim = input_dim
        self.model = AutoEncoder(input_dim,hidden_dim,latent_dim)
        self.model.compile(optimizer='adam', loss=losses.MeanSquaredError())
        self.model.fit(dataset,dataset, epochs=5,shuffle=True)
        
    def get_model(self):
        return self.model
    
    #find appropriate best kmeans number


    # def find_k_kmeans(self,kmax, input_data):
    #     max_score = 0
    #     sil = []
    #     for k in range(2, kmax+1):
    #         print("run {} cluster".format(k))
    #         kmeans = KMeans(n_clusters=k).fit(input_data)
    #         labels = kmeans.labels_
    #         current_score = silhouette_score(input_data, labels, metric='euclidean')
    #         if current_score > max_score:
    #             max_score = current_score
    #             max_k = k
    #     return max_k
@ray.remote
def calculate_silhouette_score(input_data, k):
    kmeans = KMeans(n_clusters=k).fit(input_data)
    labels = kmeans.labels_
    current_score = silhouette_score(input_data, labels, metric='euclidean')
    current_score = round(current_score,3)
    print("get {} cluster score {}".format(k,current_score))
    return current_score

#multiprocess method
# def find_k_kmeans(self, kmax, input_data):
#     max_score = 0
#     p = Pool(processes=kmax-1)
#     result = p.starmap(self.calculate_silhouette_score, [
#                        (input_data, k) for k in range(2, kmax+1)])
#     print("result is {}".format(result))
#     p.close()
#     max_k = result.index(max(result))
#     return max_k
    
    #return label of each data sample user inputs, [1,1,2,2,3] indicates that cluster number of first data sample is 1, cluster number of second data sample is 1, cluster number of third data sample is 2 etc.
    #just call get_cluster_label directly with transmit dataset and maximal cluster number you can accept like at most 10 clusters and window_size
def find_k_kmeans(kmax,input_data):
    max_score = 0
    result_ids = []
    for i in range(3,kmax+1):
        result_ids.append(calculate_silhouette_score.remote(input_data, i))
    results = ray.get(result_ids)
    print("result is {}".format(results))
    max_k = results.index(max(results))+3
    return max_k
    
@ray.remote
def collect_cluster_index(data_table,dictionary,k_value):
    list = dictionary[k_value]
    for j in len(data_table):
        if data_table[j,1] == k_value:
            list.append(data_table[j,0])

def collect_cluster_index_total(data_table,dictionary,max_k):
    result_ids = []
    for i in range(1,max_k+1,1):
        result_ids.append(collect_cluster_index.remote(data_table,dictionary,i))
            
            
        



def get_cluster_label(kmax, test_data,autoencoder):
    # batch_number = len(test_data)//self.input_dim
    # test_data = test_data[0:batch_number * self.input_dim].reshape(batch_number, self.input_dim)
    print("run kmeans cluster")
    encoded_data = autoencoder.encoder(test_data)
    max_k = find_k_kmeans(kmax, encoded_data)
    kmeans = KMeans(n_clusters=max_k).fit(encoded_data)
    labels = [i+1 for i in kmeans.labels_]
    cluster_centers = kmeans.cluster_centers_
    print("kmeans cluster finishes")
    return labels, kmeans, max_k
    

def kmeans_label(self, data, kmeans):
    cluster_label = kmeans.predict(data)
    cluster_label += 1
    return cluster_label
    # def match_cluster_label(self,data,cluster_centers_):
    #     distance = 10000000000
    #     cluster_label = 0
    #     cluster_label_list = []
    #     for j in len(data):
    #         data_sample = data[j,:]
    #         for i in range(len(cluster_centers_)):
    #             if(distance<math.dist(data,cluster_centers_[i,:])):
    #                 distance = math.dist(data_sample, cluster_centers_[i, :])
    #                 cluster_label = i+1
    #         cluster_label_list.append(cluster_label)
    #     return cluster_label
            
        
        
    

# def read_data_csv(dataurl):
#     df = pd.read_csv(dataurl, header=None)
#     df = df.iloc[:, 1:8]
#     df = df.drop([0], axis=0)
#     df = df.values.astype('float32')
#     df = df.reshape(-1,)
#     return df

# #generate random test dataset and encode them 
# def get_encode_data(autoencoder,dataset,test_number,input_dim,latent_dim):
#     begin_index = random.randint(110000,121000)
#     test_array = dataset[begin_index:begin_index+input_dim*test_number].astype('float32').reshape(test_number,input_dim)
#     encoder_test_array = np.ones((test_number,latent_dim),dtype = float)
#     for i in range(len(encoder_test_array)):
#         encoder_test_array[i, :] = autoencoder.encoder(test_array[i, :].reshape(1, -1))
#     return test_array,encoder_test_array


    

    
    



#test example
# input_dim = 40
# hidden_dim = 20
# latent_dim = 5
# test_number = 150
# dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")
# AE_model = AE_Model(input_dim,hidden_dim,latent_dim,dataset)


# autoencoder.compile(optimizer = 'Adam',loss = losses.MeanSquaredError())
# dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")
# print("length of dataset is {}".format(len(dataset)))
# #x_train = np.random.rand(30000,50).astype('float32')
# x_train = np.reshape(dataset[0:120000],(-1, input_dim))
# autoencoder.fit(x_train, x_train, epochs=3000,batch_size=40,shuffle = True)
# kmax = 20
# test_array,encoder_test_array = get_encode_data(autoencoder,test_number,input_dim,5)
# max_k,sil = find_k_kmeans(kmax,encoder_test_array)
# cluster_effect(max_k,encoder_test_array,test_array,[3,4])




