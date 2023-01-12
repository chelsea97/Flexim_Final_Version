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
import joblib
from sklearn.preprocessing import MinMaxScaler
# from test_emulate import AE_train_table
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
        self.encoder.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.encoder.add(tf.keras.layers.Dense(self.latent_dim,activation = 'relu'))
        self.decoder = tf.keras.Sequential([
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
    def __init__(self,input_dim,hidden_dim,latent_dim,train_data,valid_data):
        super(AE_Model,self).__init__()
        # batch_number = len(dataset)//input_dim
        # input_data = dataset[0:batch_number*input_dim].reshape(batch_number,input_dim)
        self.input_dim = input_dim
        self.model = AutoEncoder(input_dim,hidden_dim,latent_dim)
        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
        self.model.fit(train_data,train_data, epochs=10,validation_data = (valid_data,valid_data))
        #self.save_encoder_model(parameter_directory)
        #self.save_encoder_model(parameter_directory)
        
    def get_model(self):
        return self.model
    
    def save_encoder_model(self,parameter_directory):
        self.path = parameter_directory + "/encoder/"
        self.model.encoder.save(self.path)
    
    def save_decoder_model(self,parameter_directory):
        self.path = parameter_directory + "/decoder/"
        self.model.decoder.save(self.path)
        
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
    current_score = round(current_score, 3)
    print("get {} cluster score {}".format(k, current_score))
    return (current_score, kmeans)

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
    for i in range(3, kmax+1):
        result_ids.append(calculate_silhouette_score.remote(input_data, i))
    results = ray.get(result_ids)
    #k_means = results.index(max(results[0]))[1]
    largest_score = max(results[i][0] for i in range(len(results)))
    for k in range(len(results)):
        if results[k][0] == largest_score:
            index = k
    kmeans = results[index][1]
    return kmeans
    
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
    kmeans = find_k_kmeans(kmax, encoded_data)
    labels = [i+1 for i in kmeans.labels_]
    cluster_centers = kmeans.cluster_centers_
    print("kmeans cluster finishes")
    return labels, kmeans


def kmeans_label(self, data, kmeans):
    cluster_label = kmeans.predict(data)
    cluster_label += 1
    return cluster_label

def save_kmeans_model(kmeans_model,data_directory):
    joblib.dump(kmeans_label,data_directory + "/kmeans_model.m")
    
def read_data_csv(dataurl):
    temp = pd.read_csv("./"+dataurl,index_col= None, header=None)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(temp)
    return data

def load_dataset(argument):
    dataurl = argument + ".csv"
    dataset = read_data_csv(dataurl)
    return dataset
