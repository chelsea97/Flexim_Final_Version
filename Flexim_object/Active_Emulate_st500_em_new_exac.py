from cgi import test
from ensurepip import bootstrap
from opcode import stack_effect
from operator import index
from pyexpat import features
from random import random, shuffle
from re import X
from statistics import mode
from typing import final
import numpy as np
from modAL.models import ActiveLearner
from scikeras.wrappers import KerasClassifier
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.batch import select_cold_start_instance
from modAL.uncertainty import classifier_uncertainty
from AE import AE_Model, AutoEncoder
from tensorflow.keras import layers, losses
from dtaidistance import dtw
from numpy import dot
from numpy.linalg import norm
from keras.utils.np_utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from modAL.models import Committee
from modAL.disagreement import max_disagreement_sampling
from modAL.disagreement import vote_entropy_sampling
from sklearn.ensemble import StackingClassifier
from sklearn.utils.estimator_checks import check_estimator
from modAL.uncertainty import uncertainty_sampling

class Active_Emulate_Test:
    def __init__(self, window_size, dataset, cluster_index_dictionary, cluster_number, unlabel, labeled_data,params,params_2,max_distance,accuracy_threshold,mx_iter=5):
	    # main part of the implementation goes here including
        # the initialization and definition of neural net andone
        # active learning components. As an example:
        #self.unlabled_data=
        self.iter_counter = 0
        self.mx_iter = mx_iter
        self.window_size = window_size
        self.dataset = dataset
        self.cluster_index_dictionary = cluster_index_dictionary
        self.cluster_number = cluster_number
        self.nn_estimator_model_l2 = tf.keras.models.load_model('./'+str(1)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        #print('load l2 model')
        #self.nn_estimator_model_l2.trainable = False
        #print('freeze all layers of l2 model')
        #self.nn_estimator_model_dtw = tf.keras.models.load_model('./'+str(3)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        #print('load dtw model')
        self.nn_estimator_model_cosine = tf.keras.models.load_model('./'+str(2)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        #print('load cosine model')
        #self.nn_estimator_model_cosine.trainable = False
        #print('freeze all layers of cosine model')
        self.nn_estimator_model_blank = self.keras_model(labeled_data,params,params_2)
        #self.nn_estimator_model_2 = self.keras_model(labeled_data,params,params_2)
        #self.nn_estimator_model.fit(labeled_data[:,0:-1],labeled_data[:,-1],epochs = 1,shuffle = True)
        #self.nn_estimator_model.fit(labeled_data[:,0:-1],labeled_data[:,-1],epochs = 5,shuffle = True)
        batch_size = 5
        print("batch size is {}".format(batch_size))
        # preset_batch = partial(
        #      uncertainty_batch_sampling, n_instances=batch_size)
        #preset_batch = partial(max_disagreement_sampling,n_instances = batch_size)
        preset_batch = partial(uncertainty_batch_sampling,n_instances = batch_size)
        num_training_data = len(labeled_data)
        
        # training_indices = np.random.choice(
        #     num_training_data, size=4 * num_training_data // 5, replace=False)
        self.train_label,self.test_label = train_test_split(labeled_data,test_size = 0.2, random_state = 42,shuffle = None)
        self.train_unlabel,self.test_unlabel = train_test_split(unlabel,test_size = 0.2,random_state = 45,shuffle = None)
        self.train_unlabel,self.valid_unlabel = train_test_split(self.train_unlabel,test_size = 0.2,random_state = 47,shuffle = None)
        
        #self.learner_1 = ActiveLearner(estimator=KerasClassifier(self.nn_estimator_model), query_strategy = preset_batch,
        #                              X_training=self.train_label[:, 0:-1], y_training=self.train_label[:, -1], verbose=1)
        self.rf = RandomForestClassifier()
        #self.nn_estimator_model_blank.fit(self.train_label[:,0:-1], self.train_label[:, -1])
        estimator_list = [('l2 model',KerasClassifier(self.nn_estimator_model_l2)),('cosine',KerasClassifier(self.nn_estimator_model_cosine)),('blank',KerasClassifier(self.nn_estimator_model_blank))]
        estimator_list_length = len(estimator_list)
        self.nn_stack_estimator = self.keras_model_2(estimator_list_length,params)
        #print("construct estimator model 2")
        #print("check estimator is {}".format(check_estimator(StackingClassifier(estimators = estimator_list, final_estimator = LogisticRegression()))))
        
        
        self.learner = ActiveLearner(estimator = StackingClassifier(estimators = estimator_list, final_estimator = LogisticRegression(),stack_method = 'predict_proba'),query_strategy = preset_batch, X_training = self.train_label[:,0:-1], y_training = self.train_label[:,-1])
        #self.learner = ActiveLearner(estimator = StackingClassifier(estimators = estimator_list, final_estimator = KerasClassifier(self.nn_stack_estimator),stack_method = 'predict_proba'),query_strategy = preset_batch, X_training = self.train_label[:,0:-1], y_training = self.train_label[:,-1]) 
        #self.learner = ActiveLearner(estimator = KerasClassifier(self.nn_estimator_model_blank),query_strategy = preset_batch,X_training=self.train_label[:,0:-1], y_training=self.train_label[:,-1])
        self.learner.teach(X = self.train_label[:,0:-1], y = self.train_label[:,-1])
        #self.committee = Committee(learner_list = [self.learner_1,self.learner_2],query_strategy = preset_batch)
        
        #self.learner_3 = 
        # self.learner = ActiveLearner(estimator=RandomForestClassifier(), query_strategy=preset_batch,
        #                         X_training=labeled_data[training_indices, 0:-1], y_training=labeled_data[training_indices, -1])
        
        #self.learner.teach(X=self.train_label[:, 0:-1],y=self.train_label[:, -1])
        
        #self.learner.fit(X = labeled_data[training_indices,0:-1],y = labeled_data[training_indices,-1],epochs = 10)
        #learner_list.append(self.learner)
        #self.learner = ActiveLearner(estimator=self.nn_estimator_model, query_strategy=preset_batch,X_training=labeled_data[training_indices, 0:-1], y_training=labeled_data[training_indices, -1])
        self.query_time = 0
        self.max_distance = max_distance
        self.accuracy_threshold = accuracy_threshold
        print("accuracy threshold is {}".format(accuracy_threshold))

    def train(self, X, Y):
        print('train')
        #print(X.shape)
        #rint(Y.shape)
        if self.iter_counter >= self.mx_iter:
          return True
        self.iter_counter += 1
        #self.learner.teach(X=X, y=Y,bootstrap = True,only_new = True)
        #print("X shape is {}".format(X.shape))
        #print("Y shape is {}".format(Y.shape))
        #self.committee.teach(X=X, y=Y)
        self.learner.teach(X = X, y = Y)
        #self.learner.teach(X=X, y=Y,only_new = True)
        #self.learner.fit(X=X,y=Y,epochs = 10)
        #self.nn_estimator_model.fit(X,Y,epochs = 5,shuffle = True)
        return False
    
    def train_weight(self,X,Y,class_weight):
        print('train based on weight')
        if self.iter_counter >= self.mx_iter:
            return True
        self.iter_counter += 1
        self.learner.teach(X = X,y = Y, class_weight= class_weight)
        
        
        
        
    def query(self):
        query_index, query_instance = self.learner.query(self.train_unlabel)
        return query_instance

    def get_estimator_model(self):
        return self.nn_estimator_model

    # def get_committee(self):
    #     return self.committee
    
    def get_learner(self):
        return self.learner

    def active_learning_emulate(self, original, num,learner):
        stop_state = False
        print("run into active learning emulate process")
        print("active learning initial score is {}".format(round(learner.score(
            self.test_label[:, 0:-1].astype('float32'), self.test_label[:, -1].astype('float32')), 3)))
        print("emulate train unlabel shape is {}".format(
            self.train_unlabel.shape))
        self.valid_unlabel = self.get_label_query_instance(original,self.valid_unlabel,num)
        # for time in range(3):
        #     query_index, query_instance = learner.query(
        #         self.train_unlabel)
        #     self.query_time += 1
        #     #print("active learning query instance is {}".format(query_instance))
        #     self.train_unlabel = np.delete(
        #         self.train_unlabel, query_index, axis=0)
        #     label_query_instance = self.get_label_query_instance(original,query_instance,num)
        #     if(len(label_query_instance) == 0):
        #         break
        #     np.random.shuffle(label_query_instance)
        #     stop_state = self.train(
        #         label_query_instance[:, 0:-1], label_query_instance[:, -1])
        #     print("{} time active learning score is {}".format(self.query_time,round(learner.score(
        #         self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3)))
            
        # while round(learner.score(self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3) < 0.85 and self.iter_counter < self.mx_iter:
        #     query_index, query_instance = learner.query(
        #         self.train_unlabel)
        #     self.query_time += 1
        #     #print("active learning query instance is {}".format(query_instance))
        #     self.train_unlabel = np.delete(
        #         self.train_unlabel, query_index, axis=0)
        #     label_query_instance = self.get_label_query_instance(original,query_instance,num)
        #     if(len(label_query_instance) == 0):
        #         break
        #     np.random.shuffle(label_query_instance)
        #     stop_state = self.train(
        #         label_query_instance[:, 0:-1], label_query_instance[:, -1])
        #     print("{} time active learning score is {}".format(self.query_time,round(learner.score(
        #         self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3)))
        # print("emulate process finishes")
        
        while self.iter_counter < self.mx_iter:
            query_index, query_instance = learner.query(
                self.train_unlabel)
            self.query_time += 1
            #print("active learning query instance is {}".format(query_instance))
            self.train_unlabel = np.delete(
                self.train_unlabel, query_index, axis=0)
            label_query_instance = self.get_label_query_instance(original,query_instance,num)
            if(len(label_query_instance) == 0):
                break
            stop_state = self.train(label_query_instance[:,0:-1],label_query_instance[:,-1])
            print("{} time active learning score is {}".format(self.query_time,round(learner.score(
                self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3)))
        print("emulate process finishes")
        
        # while round(learner.score(self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3) < 0.85 :
        #     query_index, query_instance = learner.query(
        #         self.train_unlabel)
        #     self.query_time += 1
        #     #print("active learning query instance is {}".format(query_instance))
        #     self.train_unlabel = np.delete(
        #         self.train_unlabel, query_index, axis=0)
        #     label_query_instance = self.get_label_query_instance(original,query_instance,num)
        #     if(len(label_query_instance) == 0):
        #         break
        #     np.random.shuffle(label_query_instance)
        #     stop_state = self.train(
        #         label_query_instance[:, 0:-1], label_query_instance[:, -1])
        #     print("{} time active learning score is {}".format(self.query_time,round(learner.score(
        #         self.valid_unlabel[:, 0:-1].astype('float32'), self.valid_unlabel[:, -1].astype('float32')), 3)))
        # print("emulate process finishes")




    def transform(self, original, query_instance):
        transform_result = []
        problem_label_index = []
        for i in range(len(query_instance)):
            select_sample = original
            single_query_instance = query_instance[i, self.cluster_number:]
            transform = self.transform_single_sample(
                select_sample, single_query_instance)
            sort_transform_timestamp = np.sort(transform[:, 0])
            if np.all(sort_transform_timestamp == transform[:, 0]):
                temporal_list = []
                temporal_list[0:len(select_sample)
                              ] = select_sample[:, 0].tolist()
                temporal_list[len(select_sample):] = transform[:, 1].tolist()
                transform_result.append(temporal_list)
            else:
                problem_label_index.append(i)
        query_instance = np.delete(query_instance, problem_label_index, axis=0)
        transform_result = np.array(transform_result)
        print("query instance shape is {}".format(query_instance.shape))
        print("transform result shape is {}".format(transform_result.shape))
        return transform_result,query_instance

    def normalize(self, pattern):
        return (pattern - np.min(pattern))/(np.max(pattern)-np.min(pattern))

    def transform_single_sample(self, sample, single_query_instance):
        a, b, c, d, e, f = (e for e in single_query_instance)
        x = np.concatenate(
            [np.array([range(0, sample.shape[0])]).reshape((-1, 1)), sample])
        x = x.reshape((2, -1))
        A = np.array([[a+1, b*sample.shape[0]], [c/10, d+1]])
        B = np.array([[e * sample.shape[0], f]])
        transform = np.matmul(A, x).T + B
        return transform

    def get_label_query_instance(self,original,query_instance,num):
        print("query instance shape is {}".format(query_instance.shape))
        transform_unlabel_train,query_instance = self.transform(
            original, query_instance)
        label_table = self.get_label(transform_unlabel_train.astype('float32'),num)
        #label_query_instance = np.concatenate((query_instance,label_table.reshape(-1,1)),axis = 1)
        label_query_instance = np.concatenate(
            (query_instance, label_table.reshape(-1, 1)), axis=1)
        return label_query_instance

    def get_label(self, transform_result, num):
        label_table = []
        for i in range(len(transform_result)):
            original_pattern = transform_result[i,
                                                0:transform_result.shape[1]//2]
            transform_pattern = transform_result[i,
                                                 transform_result.shape[1]//2:transform_result.shape[1]]
            if num == 1:
                #print("active learning L2")
                l2_distance = np.linalg.norm(
                    original_pattern - transform_pattern)
                similarity = round(1 - (l2_distance/self.max_distance), 3)
                #print("get label part l2 similarity score {}".format(similarity))
                if similarity >= self.accuracy_threshold:
                    label = 1
                else:
                    label = 0
            elif num == 2:
                #print("active learning cosine similarity")
                #print("original_pattern shape is {}".format(original_pattern.shape))
                similarity = (np.dot(np.transpose(
                    original_pattern), transform_pattern)/(np.linalg.norm(original_pattern) * np.linalg.norm(transform_pattern)))
                #print("similarity is {}".format(similarity))
                similarity = round(similarity/self.max_distance, 3)
                #print("get label part cosine similarity score {}".format(similarity))
                if similarity >= self.accuracy_threshold:
                    label = 1
                else:
                    label = 0
            elif num == 3:
                #print("active learning dtw")
                original_pattern = np.array(original_pattern,dtype = np.double)
                transform_pattern = np.array(transform_pattern,dtype = np.double)
                dtw_distance = dtw.distance_fast(
                    original_pattern, transform_pattern)
                similarity = round(1 - (dtw_distance/self.max_distance), 3)
                #print("get label part dtw similarity score {}".format(similarity))
                if similarity >= self.accuracy_threshold:
                    label = 1
                else:
                    label = 0
            label_table.append(label)
        label_table = np.array(label_table)
        return label_table

    def normalize(self, pattern):
        return (pattern - np.min(pattern))/(np.max(pattern)-np.min(pattern))
    
    #make experiment that using binaryfocalcrossentropy method, get predict_proba value, compare with real score condition
    def keras_model(self,labeled_data,params,params_2):
        lr = params[0]
        fl = params[1]
        sl = params[2]
        gamma = params_2[0]
        label_smoothing = params_2[1]
        print("active learning keras model run")
        print("gamma is {}".format(gamma))
        print("label smoothing is {}".format(label_smoothing))
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2**fl, activation='relu',
                  input_dim=labeled_data[:, 0:-1].shape[1]))
        model.add(tf.keras.layers.Dense(2**sl, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma,label_smoothing = label_smoothing),
                      #loss = tf.keras.losses.BinaryCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=10**lr),
                      metrics=['accuracy'])
        return model

    # def keras_model_2(self,estimator_length,params):
    #     lr = params[0]
    #     fl = 3
    #     sl = 5
    #     gamma = 0
    #     label_smoothing = 0
    #     print("active learning keras model run")
    #     print("gamma is {}".format(gamma))
    #     print("label smoothing is {}".format(label_smoothing))
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(2**fl, activation='relu',
    #               input_dim=estimator_length))
    #     model.add(tf.keras.layers.Dense(2**sl, activation='relu'))
    #     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #     model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(gamma = gamma,label_smoothing = label_smoothing),
    #                   #loss = tf.keras.losses.BinaryCrossentropy(),
    #                   optimizer=tf.keras.optimizers.Adam(learning_rate=10**lr),
    #                   metrics=['accuracy'])
    #     return model
    