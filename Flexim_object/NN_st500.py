from audioop import avg
from distutils.command.build import build
from gc import callbacks
from multiprocessing import set_forkserver_preload
from pyexpat import model
from random import shuffle
from tabnanny import verbose
from unicodedata import name
import tensorflow as tf
#import active_learning
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from keras.layers import LeakyReLU
import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import ray
import joblib
from sklearn.preprocessing import MinMaxScaler

class NN:
    def __init__(self,instance,label,
                 sample_size,validation_data,class_weight,params):
        print("use bias is False")
        self.instance = instance
        self.label = label
        self.sample_size = sample_size
        self.validation_data = validation_data
        gamma = params[2]
        label_smoothing = params[3]
        print("gamma is {}".format(gamma))
        print("label smoothing is {}".format(label_smoothing))
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(32,activation = 'relu'),
        #     tf.keras.layers.Dense(16,activation = 'relu'),
        #     tf.keras.layers.Dense(8,activation = 'relu'),
        #     tf.keras.layers.Dense(4,activation = 'relu'),
        #     # tf.keras.layers.Dense(2,activation = 'softmax')
        #     tf.keras.layers.Dense(2,activation = 'softmax')  
        # ])
        # self.model.compile(
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     #loss = tf.keras.losses.MeanAbsoluteError(),
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        #     metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
        #     #metrics = [tf.keras.metrics.MeanAbsolteError(name = 'accuracy')]
        # )
        # self.model.fit(self.instance, to_categorical(self.label,num_classes = 2),
        #                epochs=2, batch_size=20)
        
        '''
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Reshape((1,2*self.sample_size)))
        self.model.add(tf.keras.layers.Conv1D(32,1,activation = 'relu',input_shape = (1,2*self.sample_size)))
        self.model.add(tf.keras.layers.Conv1D(16,1,activation = 'relu'))
        self.model.add(tf.keras.layers.Conv1D(8,1,activation = 'relu'))
        self.model.add(tf.keras.layers.Conv1D(4,1,activation = 'relu'))
        self.model.add(tf.keras.layers.GlobalAveragePooling1D())
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(2,activation = 'softmax'))
        '''
        
        
        # self.model = tf.keras.Sequential()
        # self.model.add(tf.keras.layers.Reshape((self.sample_size,2),input_shape = (2*sample_size,)))
        # self.model.add(tf.keras.layers.Conv1D(
        #     32, 10, activation='relu', input_shape=(self.sample_size, 2)))
        # #self.model.add(tf.keras.layers.BatchNormalization())
        # self.model.add(tf.keras.layers.MaxPool1D(3))
        # self.model.add(tf.keras.layers.Conv1D(64,10,activation = 'relu'))
        # #self.model.add(tf.keras.layers.BatchNormalization())
        # self.model.add(tf.keras.layers.GlobalAveragePooling1D())
        # self.model.add(tf.keras.layers.Dropout(0.5))
        # self.model.add(tf.keras.layers.Dense(1,activation = 'sigmoid'))
        print("new version 1")
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Reshape((self.sample_size,2),input_shape = (2*sample_size,)))
        self.model.add(tf.keras.layers.Conv1D(8,5,activation = 'relu',input_shape = (self.sample_size,2)))
        #self.model.add(tf.keras.layers.Conv1D(256,5,activation = 'relu'))
        self.model.add(tf.keras.layers.MaxPool1D(3))
        self.model.add(tf.keras.layers.Conv1D(16,5,activation = 'relu'))
        #self.model.add(tf.keras.layers.Conv1D(512,5,activation = 'relu'))
        self.model.add(tf.keras.layers.GlobalAveragePooling1D())
        #self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias = False))
        
        self.model.compile(
            loss = tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma,label_smoothing = label_smoothing),
            #loss = tf.keras.losses.CosineSimilarity(axis = -1),
            #loss = tf.keras.losses.BinaryCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate= 1e-3),
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
                tf.keras.metrics.Precision(name = 'precision'),
                tf.keras.metrics.Recall(name = 'recall')
            ]
        )
        
        self.model.fit(self.instance, self.label, epochs=10,
                       validation_data=self.validation_data, shuffle=True, class_weight=class_weight)
        
        
        # self.model.fit(self.instance,to_categorical(self.label,num_classes = 2),epochs = 2,batch_size = 20)
        #self.save_train_model_2(dataset)
    
    def nn_calculate(self,prediction_class,y_new):
        print("Accuracy is {}".format(accuracy_score(y_new,prediction_class)))
        print("Precision is {}".format(precision_score(y_new,prediction_class,average = 'micro')))
        print("Recall is {}".format(recall_score(y_new, prediction_class, average='micro')))
        
    def nn_predict(self,X):
        predict_score = self.model.predict(X)
        return predict_score
    
    def nn_predict_classify(self,X):
        predict_score = self.model.predict(X)
        predict_label = np.argmax(predict_score,axis = 1)
        predict_prob = np.array(predict_score[:,1])
        return predict_label,predict_prob
