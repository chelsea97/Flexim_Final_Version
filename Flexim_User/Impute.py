from base64 import encode
from cgi import test
from pyexpat import model
from random import shuffle
from tabnanny import verbose
import matplotlib.pyplot as plt
from AE import AE_Model
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

class AutoEncoder(tf.keras.Model):
    def __init__(self,input_dim,hidden_dim,latent_dim):
        super(AutoEncoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.network = tf.keras.Sequential()
        #self.encoder.add(tf.keras.layers.InputLayer(input_shape=(1,self.input_dim)))
        self.network.add(tf.keras.layers.BatchNormalization(axis = -1))
        self.network.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.network.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.network.add(tf.keras.layers.Dense(self.hidden_dim,activation = 'relu'))
        self.network.add(tf.keras.layers.Dense(self.latent_dim))
    
    #get decoded array based on input array via process of encode and decode
    def call(self,x):
        impute = self.network(x)
        return impute
    
def get_encode_data(autoencoder,dataset,test_number,input_dim,latent_dim):
    begin_index = random.randint(110000,121000)
    test_array = dataset[begin_index:begin_index+input_dim*test_number].astype('float32').reshape(test_number,input_dim)
    encoder_test_array = np.ones((test_number,latent_dim),dtype = float)
    for i in range(len(encoder_test_array)):
        encoder_test_array[i, :] = autoencoder.encoder(test_array[i, :].reshape(1, -1))
    return test_array,encoder_test_array

def get_nested_encode_data(dataset,test_number,input_dim,latent_dim):
    begin_index = random.randint(0,3000)
    test_array = dataset[begin_index:begin_index+input_dim*test_number].astype('float32').reshape(test_number,input_dim)
    encode_test_array = np.ones((test_number,latent_dim),dtype = float)
    for i in range(len(test_array)-1):
        encode_test_array[i,:] = test_array[i+1,0:latent_dim]
    encode_test_array[test_number-1,:] = dataset[begin_index + input_dim*test_number:begin_index+input_dim*test_number+latent_dim] 
    return test_array,encode_test_array

def get_nested_encode_data_list(dataset,series_number,input_dim,latent_dim):
    data_list = []
    for i in range(series_number):
        test_array,encode_test_array = get_nested_encode_data(dataset,test_number,input_dim,latent_dim)
        data_list.append([test_array,encode_test_array])
    return data_list

def read_data_csv(dataurl):
    df = pd.read_csv(dataurl,header=None)
    df = df.iloc[:,1:8]
    df = df.drop([0],axis = 0)
    df = df.values.astype('float32')
    df = df.reshape(-1,)
    return df

def plot_figure(original_array,predict_array):
    t = [i for i in range(original_array.shape[1])]
    fig,axs = plt.subplots()
    axs.plot(t,original_array[0,:],label = 'original_array')
    axs.plot(t,predict_array[0,:],label = 'impute_array')
    axs.set_xlim(0,10)
    plt.legend()
    fig.tight_layout()
    plt.show()
    
    
    

def test_effect(autoencoder,dataset,input_dim,latent_dim):
    begin_index = random.randint(124000,128000)
    test_array = dataset[begin_index:begin_index+input_dim].astype('float32')
    original_array = dataset[begin_index+input_dim:begin_index+input_dim+latent_dim].reshape(1,-1)
    predict_array = autoencoder.call(test_array.reshape(1,-1))
    plot_figure(original_array,predict_array)
    
    
    

input_dim = 40
hidden_dim = 20
latent_dim = 10
test_number = 3000
dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016Dataset/measure1_smartphone_sens.csv")
print(len(dataset))
model = AutoEncoder(input_dim,hidden_dim,latent_dim)
model.compile(optimizer = 'Adam',loss = losses.MeanSquaredError())
test_array,encode_test_array = get_nested_encode_data(dataset,test_number,input_dim,latent_dim)
model.fit(test_array,encode_test_array,epochs = 300, batch_size = 40, shuffle = True)
test_effect(model,dataset,input_dim,latent_dim)


