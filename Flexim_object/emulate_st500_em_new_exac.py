from ensurepip import bootstrap
from generator_st500 import generator
import tensorflow as tf
import time as time
import joblib
import pandas as pd
import numpy as np
from Active_Emulate_st500_em_new_exac import Active_Emulate_Test as ace
from NN_st500 import NN as nn
import copy
import ray
from scipy import spatial
from dtaidistance import dtw
#from scipy.spatial import distance_fast
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from numpy import dot
import random
from numpy.linalg import norm
import sys
from modAL.models import Committee
from sklearn.utils import shuffle
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.batch import select_cold_start_instance
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import wasserstein_distance
import joblib


class Emulate_TestD:
    def __init__(self, dataset, parameter_directionary, similarity_measure, cluster_number, sample_size, similarity_threshold, delta = 0.1):
        self.parameter_directionary = parameter_directionary
        self.delta = delta
        self.sample_size = sample_size
        self.dataset = dataset
        self.transformed = None
        self.original = None
        self.params_non_lin = []
        self.f_fetch_done = False
        self.fetch_started = False
        self.alphas_counter = 1
        self.active_counter = 0
        self.current_interv = None
        self.side = 0
        self.delta = delta
        self.current_setting = []
        self.limits = {}
        self.unlabled = None
        self.labels = []
        self.training_done = False
        self.cluster_center = []
        self.cluster_label = 0
        self.cluster_dictionary = {}  # define each parameter's interval for each cluster
        self.cluster_index_dictionary = {}  # define index of sample from each cluster
        self.sample_index_dictionary = {}
        self.train_cluster = 1
        self.cluster_number = cluster_number
        self.train_autoencoder = False
        self.similarity_measure = similarity_measure
        self.label_query_list_dict = {}
        self.learner_list = list()
        self.total_query_time = 0
        self.average_l1_distance_flexim = 0.0
        self.average_l1_distance_real = 0.0
        self.similarity_threshold = similarity_threshold

    #load encoder model
    def load_encoder_model(self, parameter_directory):
        self.encoder_model = tf.keras.models.load_model(
            parameter_directory+"/encoder/")

    #load decoder model
    def load_decoder_model(self, parameter_directory):
        self.decoder_model = tf.keras.models.load_model(
            parameter_directory+"/decoder/")

    def load_ae_model(self, parameter_directory):
        self.ae_model = tf.keras.models.load_model(
            parameter_directory+"/model")

    def load_kmeans_model(self, parameter_directionary):
        self.kmeans_model = joblib.load(
            "./"+parameter_directionary+"/kmeans_model.m")

    #initial parameter of linear combination

    def __init_lin__(self):
        self.params_lin = []
        names = ['a', 'b', 'c', 'd', 'e', 'f']
        for n in names:
            self.params_lin.append((n, (-1, 1)))

    #initial parameter of non-linear combination
    # def __init_non_lin__(self):
    #     self.F = Functions(self, {'sample_size': self.sample_size.get()})
    #     funcs = list(filter(lambda f: not f.startswith('__'), dir(Functions)))
    #     for f in funcs:
    #         _, rng, name = eval('self.F.'+f+'()')
    #         self.params_non_lin.append((f, rng, name))

    def __init_both_cluster_directory__(self, cluster_number):
        for i in range(cluster_number):
            self.cluster_dictionary[i+1] = copy.deepcopy(self.params_lin)
            self.cluster_index_dictionary[i+1] = []
            self.sample_index_dictionary[i+1] = []

    def __init_limit_dictionary__(self):
        for i in range(1, self.cluster_number+1):
            self.limits[i] = {}

    #fetch again

    def __restart_fetch__(self, num,params):
        self.fetch_started = False
        self.alphas_counter = 1
        self.training_done = False
        self.f_fetch_done = False
        self.unlabled = None
        self.__fetch__(num,params)

    def load_parameter(self,params_2):
        self.params_2 = params_2
        
    #fetch samples based on active learning sample
    def __active_fetch__(self, num,params):
        labeled, unlabeled = self.gen_naive_complex_alpha_parameter(
            n_points=800)
        #print("labeled is {}".format(labeled[0:10, :]))
        #print("unlabel is {}".format(unlabeled[0:10, :]))
        #print("self.limits is {}".format(self.limits))
        if self.train_cluster == 1:
            print("initial active learner model")
            self.active_lr = ace(self.sample_size, self.dataset,
                                 self.cluster_index_dictionary, self.cluster_number, unlabeled, labeled,params,self.params_2,self.max_distance,self.similarity_threshold)
            self.active_lr.active_learning_emulate(self.original,
                                                   num, self.active_lr.get_learner())
            self.test_unlabel_list = self.active_lr.test_unlabel
            
        else:
            print("continue training active learner mdoel")
            self.active_lr.query_time = 0
            np.random.seed(6)
            self.active_lr.train_label, self.active_lr.test_label = train_test_split(
                labeled, test_size=0.2, random_state=42, shuffle=None)
            self.active_lr.train_unlabel, self.active_lr.test_unlabel = train_test_split(
                unlabeled, test_size=0.2, random_state=45, shuffle=None)
            self.active_lr.train_unlabel, self.active_lr.valid_unlabel = train_test_split(
                self.active_lr.train_unlabel, test_size=0.2, random_state=47, shuffle=None)
            print("teach active learning model")
            # self.active_lr.committee.teach(
            #     X=self.active_lr.train_label[:,0:-1], y=self.active_lr.train_label[:,-1])
            # self.active_lr.learner_1.teach(X=self.active_lr.train_label[:,0:-1], y=self.active_lr.train_label[:,-1])
            # self.active_lr.learner_2.teach(X=self.active_lr.train_label[:,0:-1], y=self.active_lr.train_label[:,-1])
            # self.active_lr.learner_3.teach(X=self.active_lr.train_label[:,0:-1], y=self.active_lr.train_label[:,-1])
            #self.active_lr.learner_4.teach(X=self.active_lr.train_label[:,0:-1], y=self.active_lr.train_label[:,-1])
            
            self.active_lr.get_learner().teach(X = self.active_lr.train_label[:,0:-1], y = self.active_lr.train_label[:,-1])
            self.active_lr.iter_counter = 0
            print("train unlabel shape is {}".format(self.active_lr.train_unlabel.shape))
            #self.active_lr.nn_estimator_model.fit(labeled[training_indices, 0:-1], labeled[training_indices, -1], epochs=1, shuffle = True)
            self.active_lr.active_learning_emulate(self.original,
                                                   num, self.active_lr.get_learner())
            self.test_unlabel_list = np.concatenate((self.test_unlabel_list,self.active_lr.test_unlabel),axis = 0)
        self.label_query_list_dict[self.train_cluster] = self.active_lr.test_unlabel
        self.training_done = True
        self.total_query_time += self.active_lr.query_time
        print("query time is {}".format(self.active_lr.query_time))
        print("first stage total query time is {}".format(self.total_query_time))
        self.__fetch__(num,params)

    #binary search for each cluster
    def __first_fetch__(self, num,params):
        #print("self.original pattern is {}".format(self.original))
        print("new1")
        print("{} binary fetch".format(self.train_cluster))
        for _ in range(len(self.params_lin)):
            self.current_interv = [-1, 0]
            self.alpha = -1
            self.limits[self.train_cluster][self.alphas_counter] = [
                self.current_interv[0], 0]
            current_setting = [0 for _ in range(len(self.params_lin))]
            current_setting[self.alphas_counter-1] = self.alpha
            #print("self.transform pattern is {}".format(self.transform_pattern))
            while abs(self.current_interv[1] - self.current_interv[0]) > self.delta:
                self.transform_pattern = self.transform_first_fetch(
                    current_setting)
                #self.transform_pattern = self.transform_first_fetch(current_setting)[:,-1].reshape(-1,1)
                #self.transform_pattern = self.transform_first_fetch(current_setting)
                if not self.sorted_timestamp(self.transform_pattern):
                    self.label = 0
                    #print("timestamp is not sorted")
                else:
                    transform_pattern = self.transform_pattern[
                        :, -1].reshape(-1, 1)
                #print("first fetch first type transform pattern is {}".format(self.transform_pattern))
                    self.get_label(self.original, transform_pattern, num)
                if self.label:
                    self.current_interv[1] = self.alpha
                    self.alpha = (
                        self.current_interv[1]+self.current_interv[0])/2
                else:
                    self.current_interv[0] = self.alpha
                    self.alpha = (
                        self.current_interv[1]+self.current_interv[0])/2
                current_setting[self.alphas_counter-1] = self.alpha
            self.limits[self.train_cluster][self.alphas_counter] = [
                self.current_interv[0], 0]
            self.current_interv = [0, 1]
            self.alpha = 1
            current_setting[self.alphas_counter-1] = self.alpha
            while abs(self.current_interv[1] - self.current_interv[0]) > self.delta:
                self.transform_pattern = self.transform_first_fetch(
                    current_setting)
                if not self.sorted_timestamp(self.transform_pattern):
                    self.label = 0
                    #print("timestamp is not sorted")
                else:
                    transform_pattern = self.transform_pattern[:, -
                                                               1].reshape(-1, 1)
                    self.get_label(self.original, transform_pattern, num)
                if self.label:
                    #print("run this if")
                    self.current_interv[0] = self.alpha
                    self.alpha = (
                        self.current_interv[1]+self.current_interv[0])/2
                else:
                    #print("run this else")
                    self.current_interv[1] = self.alpha
                    self.alpha = (
                        self.current_interv[1] + self.current_interv[0])/2
                current_setting[self.alphas_counter-1] = self.alpha
            self.limits[self.train_cluster][self.alphas_counter][1] = self.current_interv[1]
            self.alphas_counter += 1
        self.f_fetch_done = True
        self.__fetch__(num,params)

    #Do binary search on each cluster and then do active learning fetch
    def __fetch__(self, num,params):
        print("self.training_done is {}".format(self.training_done))
        print("simi kind is {}".format(num))
        #print("similarity measure is {}".format(self.similarity_measure))
        if self.training_done:
            if self.train_cluster == self.cluster_number:
                print("finish all cluster binary and active search")
                print("make modification new new")
                finish_status = False
                #self.active_lr.get_estimator_model().save("./" + str()+"estimator"+"/")
                if self.similarity_measure == 2:
                    result_table,finish_status = self.gen_naive_data_1(10,self.similarity_measure,finish_status)
                else:
                    result_table,finish_status = self.gen_naive_data_1(3000*self.cluster_number,self.similarity_measure,finish_status)
                if finish_status:
                    print("result table shape is {}".format(result_table.shape))
                    np.random.seed(6)
                    if self.similarity_measure == 2:
                        final_complex_params_data = self.gen_complex_params_training_3(
                            num_points=self.cluster_number * 50000)
                        print("run new data generation method")
                    else:
                        final_complex_params_data = self.gen_complex_params_training_2(
                            num_points=self.cluster_number * 50000)
                    np.random.shuffle(final_complex_params_data)
                    train_complex,test_complex = train_test_split(final_complex_params_data,test_size = 0.2,random_state = 50,shuffle = None)
                    train_complex,valid_complex = train_test_split(train_complex,test_size = 0.2,random_state = 50,shuffle = None)
                    # print("complex params label data is {}".format(
                    #     final_complex_params_data))
                    print("finish split")
                    train_complex_data, problem_list_train,transform_label_train = self.gen_label(
                        train_complex, num)
                    print("finish generate label")
                    train_complex_data_score,problem_list_train_score,transform_label_train_score = self.gen_score(train_complex,num)
                    validate_complex_data_score,problem_list_validate_score,transform_label_valid_score = self.gen_score(valid_complex,num)
                    
                    train_complex_data = np.delete(train_complex_data,problem_list_train_score,axis = 0)
                    train_complex_data_active = train_complex_data
                    train_complex = np.delete(train_complex,problem_list_train_score,axis = 0)
                    valid_complex = np.delete(
                        valid_complex, problem_list_validate_score, axis=0)
                    test_complex_data, problem_list_test,transform_label_test = self.gen_label(test_complex, num)
                    transform_label_test = np.delete(transform_label_test,problem_list_test,axis = 0)
                    validate_complex_data, problem_list_validate,transform_label_valid = self.gen_label(valid_complex, num)
                    transform_label_train = np.delete(transform_label_train,problem_list_train_score,axis = 0)
                    
                    transform_label_train_score = np.delete(transform_label_train_score,problem_list_train_score,axis = 0)
                    transform_label_valid_score = np.delete(transform_label_valid_score,problem_list_validate_score,axis = 0)
                    print("train data shape is {}".format(train_complex_data.shape))
                    print("validate data shape is {}".format(validate_complex_data.shape))
                    #self.active_lr.learner.teach(
                    #    train_data[:, 0:-1].astype('float32'), train_data[:, -1].astype('float32'), only_new=True)
                    time = 0
                    np.random.shuffle(self.test_unlabel_list)
                    print("initial active learning accuracy is {}".format(round(self.active_lr.get_learner().score(
                        test_complex_data[:, 0:-1].astype('float32'), test_complex_data[:, -1].astype('float32')), 2)))
                    
                    # while(round(self.active_lr.get_learner().score(validate_complex_data[:, 0:-1].astype('float32'), validate_complex_data[:, -1].astype('float32')), 2) < 0.85):
                    #     if time >= self.active_lr.mx_iter:
                    #         break
                    #     print("further learning teach, generate more data")
                    #     query_index, query_instance = self.active_lr.get_learner().query(
                    #         train_complex_data_active[:, 0:-1])
                    #     print("train complex shape is {}".format(train_complex_data_active.shape))
                    #     print("query index shape is {}".format(query_index.shape))
                    #     self.active_lr.get_learner().teach(
                    #         train_complex_data_active[query_index, 0:-1].astype('float32'), train_complex_data_active[query_index, -1], bootstrap=True)
                    #     train_complex_data_active = np.delete(
                    #         train_complex_data_active, query_index, axis=0)
                    #     time += 1
                    #     print("{} time fur active learning score is {}".format(time, round(self.active_lr.get_learner().score(
                    #         validate_complex_data[:, 0:-1].astype('float32'), validate_complex_data[:, -1].astype('float32')), 3)))
                    
                    while(time < self.active_lr.mx_iter):
                        print("further learning teach, generate more data")
                        query_index, query_instance = self.active_lr.get_learner().query(
                            train_complex_data_active[:, 0:-1])
                        print("train complex shape is {}".format(train_complex_data_active.shape))
                        print("query index shape is {}".format(query_index.shape))
                        self.active_lr.get_learner().teach(
                            train_complex_data_active[query_index, 0:-1].astype('float32'), train_complex_data_active[query_index, -1], bootstrap=True)
                        train_complex_data_active = np.delete(
                            train_complex_data_active, query_index, axis=0)
                        time += 1
                        print("{} time fur active learning score is {}".format(time, round(self.active_lr.get_learner().score(
                            validate_complex_data[:, 0:-1].astype('float32'), validate_complex_data[:, -1].astype('float32')), 3)))
                    
                    print("second stage total query time is {}".format(
                            time))
                    self.total_query_time += time
                    print("total query time is {}".format(self.total_query_time))
                    self.final_train_accuracy = round(self.active_lr.get_learner().score(train_complex_data[:,0:-1].astype('float32'),train_complex_data[:,-1].astype('float32')),2)
                    self.final_accuracy = round(self.active_lr.get_learner().score(
                        test_complex_data[:, 0:-1].astype('float32'), test_complex_data[:, -1].astype('float32')),2)
                    print("final train AL accuracy is {}".format(self.final_train_accuracy))
                    print("final test AL accuracy is {}".format(self.final_accuracy))
                    # print("accuracy threshold is {}".format(self.similarity_threshold))
                    
                    # self.active_lr.nn_estimator_model.save('./'+str(simi_kind)+str(5)+'/saved_model'+str(self.active_lr.mx_iter)+'/'+'model_1')
                    # joblib.dump(self.active_lr.rf,'./'+str(simi_kind)+str(5)+'/saved_model'+str(self.active_lr.mx_iter)+'/'+'random forest.joblib')
                    
                    # print("complex active data label is {}".format(complex_active_data_label[100:200,0]))
                    # accuracy = np.sum(
                    #     complex_human_label_data[:, -1] == complex_active_data_label[:, 0])/50000
                    # scomplex_active_data_label = np.argmax(complex_active_data_label,axis = 1).reshape(-1,1)
                    # final_train_data = np.concatenate((naive_data_list,complex_train_data),axis = 0)
                    
                    
                    ## continue here ###
                    # print("finish generating naive data and complex data")
                    # print("train complex is {}".format(train_complex.shape))
                    # print("valid complex is {}".format(valid_complex.shape))
                    flexim_label_train = self.active_lr.get_learner().predict(train_complex).reshape(-1,1)
                    flexim_label_train_prob = self.active_lr.get_learner().predict_proba(train_complex)[:,1].reshape(-1,1)
                    transform_label_train_flexim = np.concatenate((transform_label_train[:,0:-1],flexim_label_train),axis = 1)
                    flexim_label_valid = self.active_lr.get_learner().predict(valid_complex).reshape(-1,1)
                    flexim_label_valid_prob = self.active_lr.get_learner().predict_proba(valid_complex)[:,1].reshape(-1,1)
                    transform_label_valid_flexim = np.concatenate((transform_label_valid[:,0:-1],flexim_label_valid),axis = 1)
                    flexim_predict_prob = np.concatenate((flexim_label_train_prob,flexim_label_valid_prob),axis = 0)
                    real_prob = np.concatenate(
                        (transform_label_train_score[:, -1], transform_label_valid_score[:, -1]), axis=0).reshape(-1,1)
                    #wasserstein_distance_1 = wasserstein_distance(flexim_predict_prob.tolist(),real_prob.tolist())
                    #print("wassersetin distance_fast before adding naive data is {}".format(wasserstein_distance_1))
                    scaler = MinMaxScaler()
                    print("draw predict_prob data distribution figure")
                    print("flexim predict prob shape is {}".format(flexim_predict_prob.shape))
                    print("real prob shape is {}".format(real_prob.shape))
                    
                    # fig, axs = plt.subplots(2, 1)
                    # axs[0].hist(flexim_predict_prob,bins = 50,label = 'flexim predict')
                    # axs[1].hist(real_prob,bins = 50,label = 'real data label')
                    # axs[0].legend()
                    # axs[1].legend()
                    # fig.tight_layout()
                    # plt.show()
                    
                    #flexim_label_predict = np.concatenate((flexim_label_train,flexim_label_valid),axis = 0)
                    #print("first 10 instance label predict is {}".format(flexim_label_predict[0:10,:]))
                    #score = self.just_get_score(result_table).reshape(-1, 1)
                    score = result_table[:,-1].reshape(-1,1)
                    transform_label_train = np.concatenate((transform_label_train,result_table),axis = 0)
                    transform_label_train_flexim = np.concatenate((transform_label_train_flexim,result_table),axis = 0)
                    flexim_predict_prob = np.concatenate((flexim_predict_prob,score),axis = 0)
                    flexim_predict_prob = scaler.fit_transform(flexim_predict_prob)
                    # print("real prob shape is {}".format(real_prob.shape))
                    # print("score shape is {}".format(score.shape))
                    real_prob = np.concatenate((real_prob,score),axis = 0)
                    real_prob = scaler.fit_transform(real_prob)
                    #wasserstein_distance_2 = wasserstein_distance(flexim_predict_prob.tolist(),real_prob.tolist())
                    # print("real train data shape is {}".format(transform_label_train.shape))
                    # print("flexim train data shape is {}".format(transform_label_train_flexim.shape))
                    #print("wasserstein_distance after add naive data is {}".format(wasserstein_distance_2))
                    
                    
                    # print("draw predict_proba data distribution figure after adding naive data")
                    # fig, axs = plt.subplots(2, 1)
                    # axs[0].hist(flexim_predict_prob,bins = 50,label='flexim predict')
                    # axs[1].hist(real_prob,bins = 50,label='real data label')
                    # axs[0].legend()
                    # axs[1].legend()
                    # axs[0].set_title('add naive data sample')
                    # axs[1].set_title('add naive data sample')
                    # fig.tight_layout()
                    # plt.show()
                    
                    
                    
                    print("***** begin training *****")
                    neg_flexim = transform_label_train_flexim[transform_label_train_flexim[:,-1] < 0.5].shape[0]
                    pos_flexim = transform_label_train_flexim[transform_label_train_flexim[:,-1] >= 0.5].shape[0]
                    total_flexim = neg_flexim + pos_flexim
                    print("neg number flexim is {}".format(neg_flexim))
                    print("pos number flexim is {}".format(pos_flexim))
                    weight_for_0_flexim = (1/neg_flexim)*(total_flexim) / 2.0
                    weight_for_1_flexim = (1/pos_flexim)*(total_flexim) / 2.0
                    #print("transform_label_valid_flexim shape is {}".format(transform_label_valid_flexim.shape))
                    scaler = MinMaxScaler()
                    # transform_label_train_flexim[:, -1] = scaler.fit_transform(
                    #     transform_label_train_flexim[:, -1])
                    # transform_label_valid_flexim[:, -1] = scaler.fit_transform(
                    #     transform_label_valid_flexim[:, -1])
                    class_weight_flexim = {0: weight_for_0_flexim, 1: weight_for_1_flexim}
                    self.neural_network_train(transform_label_train_flexim[:, 0:-1], transform_label_train_flexim[:, -1], (transform_label_valid_flexim[:, 0:-1], transform_label_valid_flexim[:,-1]), class_weight=class_weight_flexim)
                    #self.logistic_regression(transform_label_train_flexim[:,0:-1],transform_label_train_flexim[:,-1],(transform_label_valid_flexim[:,0:-1],transform_label_valid_flexim[:,-1]), class_weight=class_weight_flexim)

                    neg = transform_label_train[transform_label_train[:, -1] < 0.5].shape[0]
                    pos = transform_label_train[transform_label_train[:, -1] >= 0.5].shape[0]
                    total = neg + pos
                    print("neg number is {}".format(neg))
                    print("pos number is {}".format(pos))
                    weight_for_0 = (1/neg)*(total) / 2.0
                    weight_for_1 = (1/pos)*(total) / 2.0
                    class_weight = {0: weight_for_0, 1: weight_for_1}
                    # scaler = MinMaxScaler()
                    # transform_label_train[:, -1] = scaler.fit_transform(
                    #     transform_label_train[:, -1])
                    # transform_label_valid[:, -1] = scaler.fit_transform(
                    #     transform_label_valid[:, -1])
                    self.neural_network_train_real(
                        transform_label_train[:, 0:-1], transform_label_train[:, -1], (transform_label_valid[:, 0:-1], transform_label_valid[:, -1]), class_weight=class_weight)
                    print("change to regression")
                    
                    #self.validate_result(transform_label_valid_score[:,0:-1],num)
                    test_complex = np.delete(test_complex,problem_list_test,axis = 0)
                    test_data, problem_index_list = self.gen_complex_active_data(test_complex)
                    #print("test data shape is {}".format(test_data.shape))
                    #self.test_result_regression(test_data,self.similarity_measure)
                    self.test_result(test_data,self.similarity_measure)
                    
            else:
                print("run next cluster")
                self.train_cluster += 1
                self.resample()
                self.__init_lin__()
                self.__restart_fetch__(num,params)
        elif not self.f_fetch_done:
            print("run first fetch")
            self.__first_fetch__(num,params)
        else:
            print("run active learning fetch for current cluster")
            self.__active_fetch__(num,params)
            
    def validate_result(self,validate_data,simi_kind):
        print("run validate result")
        scaler = MinMaxScaler()
        score_list_flexim = np.round_(self.neural_network_flexim_predict(validate_data)[:,0],3)
        score_list_flexim = scaler.fit_transform(score_list_flexim.reshape(-1,1))
        #print("shape of score_list_flexim is {}".format(score_list_flexim.shape))
        print("score list flexim min and max is {} {}".format(np.min(score_list_flexim),np.max(score_list_flexim)))
        print("score list flexim std is {}".format(np.std(score_list_flexim)))
        score_list_mreal = np.round_(self.neural_network_real_predict(validate_data)[:,0],3)
        score_list_mreal = scaler.fit_transform(
                score_list_mreal.reshape(-1, 1))
        print("score list mreal min and max is {} {}".format(np.min(score_list_mreal),np.max(score_list_mreal)))
        print("score list mreal std is {}".format(np.std(score_list_mreal)))
        score_list_real,score_label_real = self.test_similarity_measure(validate_data,simi_kind)
        score_list_real = scaler.fit_transform(score_list_real.reshape(-1,1))
        print("score list real min and max is {} {}".format(np.min(score_list_real),np.max(score_list_real)))
        print("score list real std is {}".format(np.std(score_list_real)))
        self.wasserstein_distance_flexim = wasserstein_distance(score_list_flexim[:,0].tolist(),score_list_real[:,0].tolist())
        self.wasserstein_distance_real = wasserstein_distance(score_list_mreal[:,0].tolist(),score_list_real[:,0].tolist())
        print("wasserstein flexim is {}".format(self.wasserstein_distance_flexim))
        print("wasserstein real is {}".format(self.wasserstein_distance_real))
        self.plot_figure_result(score_list_flexim,score_list_mreal,score_list_real)        
            
            
    def test_result(self,test_data,simi_kind):
        scaler = MinMaxScaler()
        score_list_flexim = np.round_(self.neural_network_flexim_predict(test_data)[:, 0], 3)
        score_list_flexim = scaler.fit_transform(
                score_list_flexim.reshape(-1, 1))
        #print("score list flexim min and max is {} {}".format(np.min(score_list_flexim),np.max(score_list_flexim)))
        #print("score list flexim std is {}".format(np.std(score_list_flexim)))
        score_list_mreal = np.round_(self.neural_network_real_predict(test_data)[:,0],3)
        score_list_mreal = scaler.fit_transform(
                score_list_mreal.reshape(-1, 1))
        #print("score list mreal min and max is {} {}".format(np.min(score_list_mreal),np.max(score_list_mreal)))
        #print("score list mreal std is {}".format(np.std(score_list_mreal)))
        score_list_real,score_label_real = self.test_similarity_measure(test_data,simi_kind)
        score_list_real = scaler.fit_transform(score_list_real.reshape(-1,1))
        #print("score list real min and max is {} {}".format(np.min(score_list_real),np.max(score_list_real)))
        #print("score list real std is {}".format(np.std(score_list_real)))
        # self.average_l1_distance_flexim = np.linalg.norm((score_list_real - score_list_flexim),ord = 1) / (test_data.shape[0])
        # self.average_l1_distance_real = np.linalg.norm(
        #     (score_list_real - score_list_mreal), ord=1) / (test_data.shape[0])
        self.wasserstein_distance_flexim = wasserstein_distance(score_list_flexim[:,0].tolist(),score_list_real[:,0].tolist())
        print("emd flexim is {}".format(self.wasserstein_distance_flexim))
        #print("wasserstein distance_fast flexim is {}".format(wasserstein_distance_flexim))
        #print("wassersetin distance_fast real is {}".format(wasserstein_distance_real))
        
        #np.savetxt('flexim.txt',score_list_flexim,delimiter = ',')
        #np.savetxt('mreal.txt',score_list_mreal,delimiter = ',')
        #np.savetxt('real.txt',score_list_real,delimiter = ',')
        self.plot_figure_result(score_list_flexim,score_list_real)
        
    def test_result_regression(self,test_data,simi_kind):
        score_list_flexim = np.round_(self.logistic_regression_predict(test_data)[:,0],3)
        print("score list flexim min and max is {} {}".format(
            np.min(score_list_flexim), np.max(score_list_flexim)))
        print("score list flexim std is {}".format(np.std(score_list_flexim)))
        score_list_real, score_label_real = self.test_similarity_measure(
            test_data, simi_kind)
        print("score list real min and max is {} {}".format(
            np.min(score_list_real), np.max(score_list_real)))
        print("score list real std is {}".format(np.std(score_list_real)))
        self.average_l1_distance_flexim = np.linalg.norm(
            (score_list_real - score_list_flexim), ord=1) / (test_data.shape[0])
        print("flexim average l1 error is {}".format(
            self.average_l1_distance_flexim))
        self.plot_figure_result(
            score_list_flexim, score_list_real)
        
        
        
    def plot_figure_result(self,score_list_flexim,score_list_real):
        fig, axs = plt.subplots(2, 1)
        axs[0].hist(score_list_flexim,bins = 50,range = (0,1),label = 'flexim predict')
        axs[0].set_yticks([0.0,0.2,0.4,0.6,0.8,1,0])
        axs[1].hist(score_list_real,bins = 50,range = (0,1), label = 'nn real')
        axs[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1, 0])
        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        plt.show()
        
    def gen_label(self,data,num):
        complex_human_label_data, problem_index_list = self.gen_label_complex(
            data, num)
        train_data = np.concatenate(
            (data, complex_human_label_data[:, -1].reshape(-1, 1)), axis=1)
        return train_data,problem_index_list,complex_human_label_data
    
    def gen_score(self,data,num):
        complex_human_score_data, problem_index_list = self.gen_score_complex(data,num)
        train_data = np.concatenate((data,complex_human_score_data[:,-1].reshape(-1,1)),axis = 1)
        return train_data,problem_index_list,complex_human_score_data
    
    def just_get_score(self,result_table):
        result = []
        problem_index = []
        for i in range(len(result_table)):
            original_pattern = result_table[i,0:self.sample_size]
            transform_pattern = result_table[i,self.sample_size:-1]
            score = self.get_score(original_pattern,transform_pattern,self.similarity_measure)
            result.append(score)
        result = np.array([result])
        return result
            
        
    #generate label complex parameter set based on
    def gen_query_label(self, query_dict, delta, num_inst):
        query_sample_list = query_dict[1]
        for i in range(2, self.train_cluster+1):
            query_sample_list = np.concatenate(
                (query_sample_list, query_dict[i]), axis=0)
        print("query sample list shape is {}".format(query_sample_list.shape))
        query_label_list = []
        batch_size = num_inst // len(query_sample_list)
        for i in range(len(query_sample_list)):
            label_query_sample = list(query_sample_list[i, :])
            cluster_number = np.argmax(
                label_query_sample[0:self.cluster_number], axis=0)+1
            original_pattern_index = self.sample_index_dictionary[cluster_number][0]
            original_pattern = self.dataset[original_pattern_index:
                                            original_pattern_index + self.sample_size]
            num = 0
            while num < batch_size:
                for j in range(self.cluster_number, len(label_query_sample)-1):
                    change_query_sample_1 = copy.deepcopy(label_query_sample)
                    change_query_sample_2 = copy.deepcopy(label_query_sample)
                    # print("label query sample[j] is {}".format(
                    #     label_query_sample[j]))
                    # print("self limits[0] is {}".format(
                    #     self.limits[cluster_number][j-self.cluster_number+1][0]))
                    # print("self limits[1] is {}".format(
                    #     self.limits[cluster_number][j-self.cluster_number+1][1]))
                    if label_query_sample[j] - delta > self.limits[cluster_number][j-self.cluster_number+1][0]:
                        change_query_sample_1[j] = np.random.uniform(
                            label_query_sample[j] + 0.01, label_query_sample[j] + delta)
                    else:
                        change_query_sample_1[j] = self.limits[cluster_number][j -
                                                                               self.cluster_number+1][0]
                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_1[self.cluster_number:-1])
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_1[-1] = 0

                    if label_query_sample[j] + delta < self.limits[cluster_number][j-self.cluster_number+1][1]:
                        change_query_sample_2[j] = np.random.uniform(
                            label_query_sample[j] + 0.01, label_query_sample[j] + delta)
                    else:
                        change_query_sample_2[j] = self.limits[cluster_number][j -
                                                                               self.cluster_number+1][1]
                    
                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_2[self.cluster_number:-1])
                    
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_2[-1] = 0

                    query_label_list.append(change_query_sample_1)
                    query_label_list.append(change_query_sample_2)
                    num += 2
                    if num > batch_size:
                        break
        np.random.shuffle(query_label_list)
        query_label_table = np.round_(np.array(query_label_list), 3)
        print("query label table is {}".format(query_label_table))
        print("query label table shape is {}".format(query_label_table.shape))
        return query_label_table

    def gen_query_disk(self, query_dict, delta, num_inst):
        query_sample_list = query_dict[1]
        for i in range(2, self.train_cluster + 1):
            query_sample_list = np.concatenate(
                (query_sample_list, query_dict[i]), axis=0)
        #print("query sample list shape is {}".format(query_sample_list.shape))
        query_label_list = []
        batch_size = num_inst // len(query_sample_list)
        pi = round(np.pi, 3)
        for i in range(len(query_sample_list)):
            label_query_sample = list(query_sample_list[i, :])
            cluster_number = np.argmax(
                label_query_sample[0:self.cluster_number], axis=0)+1
            original_pattern_index = self.sample_index_dictionary[cluster_number][0]
            original_pattern = self.dataset[original_pattern_index:
                                            original_pattern_index + self.sample_size]
            num = 0
            while num < batch_size:
                for j in range(self.cluster_number, len(label_query_sample)-2):
                    change_query_sample_1 = copy.deepcopy(label_query_sample)
                    change_query_sample_2 = copy.deepcopy(label_query_sample)
                    theta = np.random.random() * pi/2
                    r = delta
                    x_r = r * np.cos(theta)
                    y_r = r * np.sin(theta)
                    # print("label query sample[j] is {}".format(
                    #     label_query_sample[j]))
                    # print("self limits[0] is {}".format(
                    #     self.limits[cluster_number][j-self.cluster_number+1][0]))
                    # print("self limits[1] is {}".format(
                    #     self.limits[cluster_number][j-self.cluster_number+1][1]))
                    if label_query_sample[j] - x_r > self.limits[cluster_number][j-self.cluster_number+1][0]:
                        change_query_sample_1[j] = np.random.uniform(
                            label_query_sample[j] + 0.01, label_query_sample[j] + x_r)
                    else:
                        change_query_sample_1[j] = self.limits[cluster_number][j -
                                                                               self.cluster_number+1][0]
                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_1[self.cluster_number:-1])
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_1[-1] = 0

                    if label_query_sample[j] + x_r < self.limits[cluster_number][j-self.cluster_number+1][1]:
                        change_query_sample_2[j] = np.random.uniform(
                            label_query_sample[j] + 0.01, label_query_sample[j] + x_r)
                    else:
                        change_query_sample_2[j] = self.limits[cluster_number][j -
                                                                               self.cluster_number+1][1]

                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_2[self.cluster_number:-1])
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_2[-1] = 0

                    if label_query_sample[j+1] - y_r > self.limits[cluster_number][j+1-self.cluster_number+1][0]:
                        change_query_sample_1[j+1] = np.random.uniform(
                            label_query_sample[j+1] + 0.01, label_query_sample[j+1] + y_r)
                    else:
                        change_query_sample_1[j+1] = self.limits[cluster_number][j+1 -
                                                                                 self.cluster_number+1][0]
                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_1[self.cluster_number:-1])
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_1[-1] = 0

                    if label_query_sample[j+1] + y_r < self.limits[cluster_number][j+1-self.cluster_number+1][1]:
                        change_query_sample_2[j+1] = np.random.uniform(
                            label_query_sample[j+1] + 0.01, label_query_sample[j+1] + y_r)
                    else:
                        change_query_sample_2[j+1] = self.limits[cluster_number][j+1 -
                                                                                 self.cluster_number+1][1]
                    transform_pattern = self.transform(
                        original_pattern.reshape(-1, 1), change_query_sample_1[self.cluster_number:-1])
                    if not self.sorted_timestamp(transform_pattern):
                        change_query_sample_2[-1] = 0

                    query_label_list.append(change_query_sample_1)
                    query_label_list.append(change_query_sample_2)
                    num += 2
                    if num > batch_size:
                        break
        np.random.shuffle(query_label_list)
        query_label_table = np.round_(np.array(query_label_list), 3)
        print("last 20 query disk label table is {}".format(
            query_label_table[-20:-1, :]))
        print("query disk label table shape is {}".format(query_label_table.shape))
        return query_label_table

    #get similarity score based on original function and transform function and specific similarity measure method

    def get_label(self, original, transform, num):
        sim = 0
        if num == 1:
            l2_distance = np.linalg.norm(original-transform,ord = 2)
            sim = round(1 - (l2_distance/self.max_distance), 3)
            self.label = int(sim >= self.similarity_threshold)
        elif num == 2:
            sim = (np.dot(np.transpose(original),transform)/(np.linalg.norm(original)* np.linalg.norm(transform)))[0][0]
            sim = round(sim/self.max_distance,3)
            self.label = int(sim >= self.similarity_threshold)
        else:
            original = np.array(original[:,0],dtype = np.double)
            transform = np.array(transform[:,0],dtype = np.double)
            #print("original is {}".format(original))
            #print("transform is {}".format(transform))
            dtw_similarity = dtw.distance_fast(original, transform)
            sim = round(1 - (dtw_similarity/self.max_distance), 3)
            self.label = int(sim >= self.similarity_threshold)
        
    def get_score(self,original,transform,num):
        if num == 1:
            l2_distance = np.linalg.norm(original - transform,ord = 2)
            sim = round(1 - (l2_distance/self.max_distance), 3)
        elif num == 2:
            sim = (np.dot(np.transpose(original), transform) /
                   (np.linalg.norm(original) * np.linalg.norm(transform)))
            sim = round(sim/self.max_distance, 3)
        else:
            original = np.array(original,dtype = np.double)
            transform = np.array(transform,dtype = np.double)
            #print("original is {}".format(original))
            #print("transform is {}".format(transform))
            dtw_similarity = dtw.distance_fast(original, transform)
            sim = round(1 - (dtw_similarity/self.max_distance), 3)
        return sim

    #assign cluster label to encode data
    def instance_cluster_label(self, data, kmeans):
        encode_data = self.encoder_model(data)
        encode_data = np.array(encode_data)[0].tolist()
        cluster_label = kmeans.predict(np.reshape(encode_data, (1, -1)))+1
        return cluster_label

    def add_cluster_dict(self, instance_number=300):
        AE_data_table = np.ones(
            (instance_number, self.sample_size), dtype=np.float32)
        AE_index_table = np.ones((instance_number, 1))
        np.random.seed(1)
        for i in range(instance_number):
            start_point = np.random.randint(0, len(self.dataset)-1-self.sample_size)
            data = self.dataset[start_point:start_point + self.sample_size]
            AE_data_table[i, :] = data.reshape(1, self.sample_size)
            AE_index_table[i, 0] = start_point
        encode_result = self.encoder_model(AE_data_table)
        kmeans_label = np.array(
            [self.kmeans_model.predict(encode_result)]).reshape(-1, 1)
        kmeans_label = kmeans_label + 1
        AE_index_table = np.concatenate((AE_index_table, kmeans_label), axis=1)
        for i in range(len(AE_index_table)):
            self.cluster_index_dictionary[AE_index_table[i, -1]
                                          ].append(AE_index_table[i, 0])

    def get_max_distance(self,instance_number):
        distance_list = []
        np.random.seed(3)
        if self.similarity_measure == 2:
            self.max_distance = 1
        else:
            for i in range(instance_number):
                start_point = np.random.randint(
                    0, len(self.dataset) - 1 - self.sample_size)
                data = self.dataset[start_point:start_point + self.sample_size]
                #print("data is {}".format(data))
                start_point_2 = np.random.randint(
                    0, len(self.dataset) - 1 - self.sample_size)
                data_2 = self.dataset[start_point_2:start_point_2 +
                                        self.sample_size]
                #print("data_2 is {}".format(data_2))
                if self.similarity_measure == 1:
                    l2_distance = np.linalg.norm(data - data_2, ord=2)
                    #print("l2 distance is {}".format(l2_distance))
                    distance_list.append(l2_distance)
                elif self.similarity_measure == 3:
                    #print("data shape is {}".format(data[:,0].shape))
                    data = np.array(data[:,0],dtype = np.double)
                    data_2 = np.array(data_2[:,0],dtype = np.double)
                    dtw_distance = dtw.distance_fast(data,data_2)
                    distance_list.append(dtw_distance)
            print("length of distance list is {}".format(len(distance_list)))
            self.max_distance = 2*round(max(distance_list), 3)
            print("self.max_distance is {}".format(self.max_distance))
        
            
        
    # def sample(self):
    #     i = np.random.randint(
    #         0, self.dataset.shape[0] - int(self.sample_size))
    #     j = i + int(self.sample_size)
    #     self.original = self.dataset[i:j]
    #     cluster_label = self.instance_cluster_label(np.transpose(self.original), self.kmeans_model)
    #     while(cluster_label != self.train_cluster):
    #         i = np.random.randint(0, self.dataset.shape[0] - self.sample_size)
    #         j = i + int(self.sample_size)
    #         self.original = self.dataset[i:j]
    #         cluster_label = self.instance_cluster_label(np.transpose(self.original), self.kmeans_model)
    #     self.cluster_index_dictionary[self.train_cluster].append(i)
    #     print("has already got sample")

    def sample(self):
        np.random.seed(2)
        random_value = np.random.randint(
            len(self.cluster_index_dictionary[self.train_cluster]))
        print("random value is {}".format(random_value))
        start_point = int(
            self.cluster_index_dictionary[self.train_cluster][random_value])
        print("start point is {}".format(start_point))
        self.original = self.dataset[start_point:start_point +
                                     int(self.sample_size)]
        print("has already got sample")
        self.sample_index_dictionary[self.train_cluster].append(start_point)

    #resample from whole dataset
    # def resample(self):
    #     i = np.random.randint(0,self.dataset.shape[0] - int(self.sample_size))
    #     j = i + int(self.sample_size)
    #     self.original = self.dataset[i:j]
    #     cluster_label = self.instance_cluster_label(np.transpose(self.original),self.kmeans_model)
    #     while(cluster_label!=self.train_cluster):
    #         i = np.random.randint(0,self.dataset.shape[0] - int(self.sample_size))
    #         j = i + int(self.sample_size)
    #         self.original = self.dataset[i:j]
    #         cluster_label = self.instance_cluster_label(np.transpose(self.original),self.kmeans_model)
    #     self.cluster_index_dictionary[self.train_cluster].append(i)
    #     print("has already resampled")

    def resample(self):
        np.random.seed(3)
        random_value = np.random.randint(
            len(self.cluster_index_dictionary[self.train_cluster]))
        print("random value is {}".format(random_value))
        start_point = int(
            self.cluster_index_dictionary[self.train_cluster][random_value])
        print("start point is {}".format(start_point))
        self.original = self.dataset[start_point:start_point +
                                     int(self.sample_size)]
        print("has already resample")
        self.sample_index_dictionary[self.train_cluster].append(start_point)

    #This funcion is prepared for further data generation part, generate naive positive and negative them
    def gen_naive_data_1(self, instance_number, num, finish_status):
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
        data = self.dataset
        generator1 = generator.remote(data, self.sample_size,self.max_distance)
        generator2 = generator.remote(data, self.sample_size,self.max_distance)
        generator3 = generator.remote(data, self.sample_size,self.max_distance)
        generator4 = generator.remote(data, self.sample_size,self.max_distance)
        generator5 = generator.remote(data,self.sample_size,self.max_distance)
        next_result_id = []

        if num == 2:
            print("generate cosine naive positive and negative data new")
            next_result_id.append(
                generator1.naive_negative_rotate.remote(instance_number//2))
            next_result_id.append(
                generator2.naive_positive_shift.remote(instance_number//2))
            # next_result_id.append(
            #     generator4.naive_positive_shift.remote(instance_number//5))
        elif num == 3:
            print("generate DTW naive positive and negative data")
            next_result_id.append(
                generator1.naive_negative_shift.remote(instance_number//2))
            next_result_id.append(
                generator2.naive_positive_shift.remote(instance_number//2))
        elif num == 1:
            print("generate L2 naive positive and negative data")
            # next_result_id.append(
            #     generator1.naive_negative_l2.remote(instance_number//2))
            # next_result_id.append(
            #     generator2.naive_positive_l2.remote(instance_number//4))
            next_result_id.append(generator2.naive_negative_shift.remote(instance_number // 4))
            next_result_id.append(
                generator3.naive_positive_shift.remote(instance_number//4))
        next_result_table = ray.get(next_result_id)
        print("length of next_result_table is {}".format(len(next_result_table)))
        while(len(next_result_table) == 0):
            print("data generation")
        next_result_final_table = np.concatenate(
            (next_result_table[0], next_result_table[1]), axis=0)
        for i in range(2, len(next_result_table)):
            next_result_final_table = np.concatenate(
                (next_result_final_table, next_result_table[i]), axis=0)
        # print("next_result_table shape is {}".format(next_result_table[0].shape))
        # generator1.write_table(next_result_table[0], number=1)
        # generator2.write_table(next_result_table[1], number=2)
        # generator3.write_table(next_result_table[2], number=3)
        # generator4.write_table(next_result_table[3], number=4)
        # generator5.write_table(next_result_table[4], number=5)
        finish_status = True
        return next_result_final_table, finish_status

    #This function is also prepared for further data generation part
    def gen_naive_data_2(self, naives):
        result_table = np.random.random(
            size=(len(naives), 2*self.sample_size+1))
        for i in range(len(result_table)):
            cluster_number_instance = np.argmax(
                naives[i, 0:self.cluster_number], axis=0) + 1
            current_index = int(self.cluster_index_dictionary[cluster_number_instance][np.random.randint(
                len(self.cluster_index_dictionary[cluster_number_instance]))])
            original_pattern = self.dataset[current_index:current_index+self.sample_size]
            transform_pattern = self.transform(
                original_pattern.reshape(-1, 1), naives[i, self.cluster_number:-1])[:, 1].reshape(-1, 1)
            label = naives[i, -1]
            result_table[i, 0:self.sample_size] = original_pattern[:, 0]
            result_table[i, self.sample_size:-1] = transform_pattern[:, 0]
            result_table[i, -1] = label
        return result_table

    def gen_label_complex(self, complex, num):
        result_table = np.random.random(
            size=(len(complex), 2*self.sample_size + 1))
        problem_index_list = []
        for i in range(len(result_table)):
            cluster_number_instance = np.argmax(
                complex[i, 0:self.cluster_number], axis=0) + 1

            # current_index = int(self.cluster_index_dictionary[cluster_number_instance][np.random.randint(
            #     len(self.cluster_index_dictionary[cluster_number_instance]))])
            current_index = self.sample_index_dictionary[cluster_number_instance][0]
            original_pattern = self.dataset[current_index:current_index+self.sample_size]
            #transform_pattern = self.transform(original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])[:,1].reshape(-1,1)
            transform_pattern = self.transform(
                original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])
            # if not self.sorted_timestamp(transform_pattern):
            #     label = 0
            # else:
            if self.sorted_timestamp(transform_pattern):
                transform_pattern = transform_pattern[:, 1].reshape(-1, 1)
                label = self.assign_label(
                    original_pattern, transform_pattern, num)
            else:
                #print("fuck timestamp")
                label = 0
                transform_pattern = transform_pattern[:, 1].reshape(-1, 1)
                problem_index_list.append(i)
            result_table[i, 0:self.sample_size] = original_pattern[:, 0]
            result_table[i, self.sample_size:-1] = transform_pattern[:, 0]
            result_table[i, -1] = label
        return result_table, problem_index_list
    
    def gen_score_complex(self,complex,num):
        result_table = np.random.random(
            size=(len(complex), 2*self.sample_size + 1))
        problem_index_list = []
        for i in range(len(result_table)):
            cluster_number_instance = np.argmax(
                complex[i, 0:self.cluster_number], axis=0) + 1

            # current_index = int(self.cluster_index_dictionary[cluster_number_instance][np.random.randint(
            #     len(self.cluster_index_dictionary[cluster_number_instance]))])
            current_index = self.sample_index_dictionary[cluster_number_instance][0]
            original_pattern = self.dataset[current_index:current_index+self.sample_size]
            #transform_pattern = self.transform(original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])[:,1].reshape(-1,1)
            transform_pattern = self.transform(
                original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])
            # if not self.sorted_timestamp(transform_pattern):
            #     label = 0
            # else:
            if self.sorted_timestamp(transform_pattern):
                transform_pattern = transform_pattern[:, 1].reshape(-1, 1)
                score = self.assign_score(
                    original_pattern, transform_pattern, num)
                if score < 0:
                    problem_index_list.append(i)
            else:
                #print("fuck timestamp")
                score = 0
                transform_pattern = transform_pattern[:, 1].reshape(-1, 1)
                problem_index_list.append(i)
            result_table[i, 0:self.sample_size] = original_pattern[:, 0]
            result_table[i, self.sample_size:-1] = transform_pattern[:, 0]
            result_table[i, -1] = score
        return result_table, problem_index_list
    

    def gen_complex_data(self, complex):
        result_table = np.random.random(
            size=(len(complex), 2*self.sample_size))
        for i in range(len(result_table)):
            cluster_number_instance = np.argmax(
                complex[i, 0:self.cluster_number], axis=0) + 1
            # current_index = int(self.cluster_index_dictionary[cluster_number_instance][np.random.randint(
            #     len(self.cluster_index_dictionary[cluster_number_instance]))])
            current_index = self.sample_index_dictionary[cluster_number_instance][0]
            original_pattern = self.dataset[current_index:current_index+self.sample_size]
            transform_pattern = self.transform(
                original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])[:, 1].reshape(-1, 1)
            result_table[i, 0:self.sample_size] = original_pattern[:, 0]
            result_table[i, self.sample_size:] = transform_pattern[:, 0]
        return result_table

    def gen_complex_active_data(self, complex):
        print("run gen_complex_active_data")
        complex = np.round_(complex, decimals=3).astype('float32')
        print("generate complex data")
        result_table = np.random.random(
            size=(len(complex), 2*self.sample_size))
        problem_index_list = []
        for i in range(len(result_table)):
            cluster_number_instance = np.argmax(
                complex[i, 0:self.cluster_number], axis=0) + 1
            # current_index = int(self.cluster_index_dictionary[cluster_number_instance][np.random.randint(
            #     len(self.cluster_index_dictionary[cluster_number_instance]))])
            current_index = self.sample_index_dictionary[cluster_number_instance][0]
            original_pattern = self.dataset[current_index:current_index+self.sample_size]
            transform_pattern = self.transform(
                original_pattern.reshape(-1, 1), complex[i, self.cluster_number:])
            if not self.sorted_timestamp(transform_pattern):
               problem_index_list.append(i)
               #print("not sorted timestamp {}".format(transform_pattern[:,0]))
            transform_pattern = transform_pattern[:, 1].reshape(-1, 1)
            #print(" generate complex active data label is {}".format(label))
            result_table[i, 0:self.sample_size] = original_pattern[:, 0]
            result_table[i, self.sample_size:] = transform_pattern[:, 0]
        #print("result table shape is {}".format(result_table.shape))
        print("len of problem index list is {}".format(len(problem_index_list)))
        return result_table, problem_index_list

    #generate naive and complex alpha parameter sets after binary search for each cluster. Also prepare for each cluster active learning
    def gen_naive_complex_alpha_parameter(self, n_points):
        print("generate naive complex alpha parameter")
        np.random.seed(4)
        positive = self.gen_naive_params_current_cluster(
            positive=True, points=n_points//8)
        negative = self.gen_naive_params_current_cluster(
            positive=False, points=n_points//8)
        labeled = np.round_(np.concatenate((positive, negative), axis=0), 3)
        low_limits = np.array([self.limits[self.train_cluster][i][0]
                              for i in self.limits[self.train_cluster].keys()])
        high_limits = np.array([self.limits[self.train_cluster][i][1]
                               for i in self.limits[self.train_cluster].keys()])
        diff = high_limits - low_limits
        unlabeled = np.random.uniform(size=(3*n_points//4, len(low_limits)))
        unlabeled = (unlabeled*diff) + low_limits
        unlabeled = np.round_(unlabeled, decimals=3)
        cluster_index_table_unlabeled = np.array([to_categorical(
            self.train_cluster-1, num_classes=self.cluster_number) for _ in range(len(unlabeled))]).reshape(len(unlabeled), -1)
        unlabeled = np.concatenate(
            (cluster_index_table_unlabeled, unlabeled), axis=1)
        return labeled, unlabeled

    #This gen_naive function actually generate parameters for producing naive positive or negative label for every cluster
    def gen_naive_params(self, positive, num_instances):
        naives = []
        m = len(self.params_lin)
        number = 0
        # for i in range(1, self.cluster_number+1, 1):
        #     cluster_index = to_categorical(
        #         i-1, num_classes=self.cluster_number)
        #     if positive:
        #         temp = [0]*m + [1]
        #         temp = np.concatenate((cluster_index, temp), axis=0)
        #         naives.append(temp)
        while number < num_instances:
            for i in range(1, self.cluster_number+1, 1):
                cluster_index = to_categorical(
                    i-1, num_classes=self.cluster_number)
                for j in range(len(self.params_lin)):
                    temp = [0]*m + [1 if positive else 0]
                    temp[j] = self.limits[i][j+1][0] - (0 if positive else np.random.uniform(
                        0.01, self.limits[i][j+1][0]+1-0.01))
                    temp = np.concatenate((cluster_index, temp), axis=0)
                    naives.append(temp)
                    temp = [0]*m + [1 if positive else 0]
                    temp[j] = self.limits[i][j+1][1] + (0 if positive else np.random.uniform(
                        0.01, 1-self.limits[i][j+1][1]-0.01))
                    temp = np.concatenate((cluster_index, temp), axis=0)
                    naives.append(temp)
                    number += 2
                    if number > num_instances:
                        break
        np.random.shuffle(naives)
        naives = np.array(naives)
        print("finish generating naive parameters for each cluster")
        return naives

    #This gen_naive function actually generate parameters for producing naive positive or negative label for current cluster
    def gen_naive_params_current_cluster(self, positive, points):
        np.random.uniform(5)
        num = 0
        naives = []
        m = len(self.params_lin)
        cluster_index = to_categorical(
            (self.train_cluster)-1, num_classes=self.cluster_number)
        if positive:
            temp = [0]*m + [1]
            temp = np.concatenate((cluster_index, temp), axis=0)
            naives.append(temp)
        num += 1
        while num <= points:
            for j in range(len(self.params_lin)):
                gap = self.limits[self.train_cluster][j+1][1] - \
                    self.limits[self.train_cluster][j+1][0]
                temp = [0]*m + [1 if positive else 0]
                temp[j] = self.limits[self.train_cluster][j+1][0] - (np.random.uniform(
                    0, -1*gap) if positive else np.random.uniform(0.01, self.limits[self.train_cluster][j+1][0]+1-0.01))
                temp = np.concatenate((cluster_index, temp), axis=0)
                naives.append(temp)
                temp = [0]*m + [1 if positive else 0]
                temp[j] = self.limits[self.train_cluster][j+1][1] + (np.random.uniform(
                    0, -1*gap) if positive else np.random.uniform(0.01, 1-self.limits[self.train_cluster][j+1][1]-0.01))
                temp = np.concatenate((cluster_index, temp), axis=0)
                naives.append(temp)
                num += 2
                if num > points:
                    break
        naives = np.array(naives)
        print("finish generating naive parameters for current cluster")
        return naives

    #generates complex alpha parameter sets for each cluster
    def gen_complex_params(self, num_points):
        np.random.seed(10)
        result_table = np.random.uniform(
            size=(num_points, self.cluster_number + len(self.limits[1].keys())))
        for cluster in range(1, self.cluster_number+1, 1):
            cluster_index = to_categorical(
                cluster-1, num_classes=self.cluster_number)
            low_limits = np.array([self.limits[cluster][i][0]
                                  for i in self.limits[cluster].keys()])
            print("length of low limits is {}".format(len(low_limits)))
            high_limits = np.array([self.limits[cluster][i][1]
                                   for i in self.limits[cluster].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(
                size=(num_points//self.cluster_number, len(self.limits[1].keys())))
            unlabeled = (unlabeled * diff) + low_limits
            unlabeled = np.round_(unlabeled, decimals=3)
            cluster_table = np.array(
                [cluster_index for _ in range(len(unlabeled))])
            unlabeled = np.concatenate((cluster_table, unlabeled), axis=1)
            result_table[(cluster-1)*(num_points//self.cluster_number):cluster*(num_points//self.cluster_number), :] = unlabeled
        print("finish generating complex parameters for all clusters and predict label by using active learner")
        return result_table
    
    #generate complex alpha parameter sets for final training part 
    def gen_complex_params_training(self,num_points):
        print("version 1")
        np.random.seed(10)
        result_table = np.random.uniform(size = (num_points,self.cluster_number + len(self.limits[1].keys())))
        for cluster in range(1,self.cluster_number + 1,1):
            cluster_index = to_categorical(
                cluster-1, num_classes=self.cluster_number)
            low_limits = np.array([self.limits[cluster][i][0]
                                  for i in self.limits[cluster].keys()])
            print("length of low limits is {}".format(len(low_limits)))
            high_limits = np.array([self.limits[cluster][i][1]
                                   for i in self.limits[cluster].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(
                size=(num_points//self.cluster_number, len(self.limits[1].keys())))
            #unlabeled = (unlabeled * diff) + low_limits
            unlabeled = np.round_(unlabeled, decimals=3)
            cluster_table = np.array(
                [cluster_index for _ in range(len(unlabeled))])
            unlabeled = np.concatenate((cluster_table, unlabeled), axis=1)
            result_table[(cluster-1)*(num_points//self.cluster_number):cluster*(num_points//self.cluster_number), :] = unlabeled
        print("finish generating complex parameters for all clusters and predict label by using active learner")
        return result_table
    
    def gen_complex_params_training_2(self,num_points):
        np.random.seed(11)
        result_table = np.random.uniform(
            size=(num_points, self.cluster_number + len(self.limits[1].keys())))
        d_n = 0
        while d_n < num_points: 
            cluster = np.random.randint(low = 1,high = self.cluster_number + 1)
            cluster_index = to_categorical(cluster -1, num_classes = self.cluster_number)
            low_limits = np.array([self.limits[cluster][i][0]
                               for i in self.limits[cluster].keys()])
            #print("length of low limits is {}".format(len(low_limits)))
            high_limits = np.array([self.limits[cluster][i][1]
                                for i in self.limits[cluster].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(
                size=(1, len(self.limits[1].keys())))
            unlabeled = (unlabeled * diff) + low_limits
            cluster_table = np.array([cluster_index])
            unlabeled = np.concatenate((cluster_table, unlabeled), axis=1)
            unlabeled = np.round_(unlabeled, decimals=3)
            result_table[d_n,:] = unlabeled
            d_n += 1
        print("finish generating complex parameters for all clusters and predict label by using active learner")
        print("shape of table is {}".format(result_table.shape))
        return result_table
        
    def gen_complex_params_training_3(self, num_points):
        np.random.seed(11)
        result_table = np.random.uniform(
            size=(num_points, self.cluster_number + len(self.limits[1].keys())))
        d_n = 0
        while d_n < num_points:
            cluster = np.random.randint(low=1, high=self.cluster_number + 1)
            cluster_index = to_categorical(
                cluster - 1, num_classes=self.cluster_number)
            low_limits = np.array([self.limits[cluster][i][0] 
                                   for i in self.limits[cluster].keys()])
            #print("length of low limits is {}".format(len(low_limits)))
            high_limits = np.array([self.limits[cluster][i][1]
                                    for i in self.limits[cluster].keys()])
            diff = high_limits - low_limits
            unlabeled = np.random.uniform(
                size=(1, len(self.limits[1].keys())))
            unlabeled = (unlabeled * diff) + low_limits
            cluster_table = np.array([cluster_index])
            unlabeled = np.concatenate((cluster_table, unlabeled), axis=1)
            unlabeled = np.round_(unlabeled, decimals=3)
            result_table[d_n, :] = unlabeled
            d_n += 1
        print("finish generating complex parameters for all clusters and predict label by using active learner")
        print("shape of table 3 is {}".format(result_table.shape))
        return result_table
                    
    #generate predict label parameter set based on label and cluster number
    def gen_predict_label_parameter(self, num_points):
        np.random.seed(20)
        class_vector = [np.random.randint(
            self.cluster_number) for _ in range(num_points)]
        one_hot_vector = to_categorical(
            class_vector, num_classes=self.cluster_number)
        class_vector_new = [x+1 for x in class_vector]
        parameter_table = np.round_(np.random.random(
            size=(num_points, len(self.limits[1].keys()))), decimal=3)
        for i in range(len(class_vector_new)):
            class_number = class_vector_new[i]
            low_limits = [self.limits[class_number][j][0]
                          for j in self.limits[class_number].keys()]
            high_limits = [self.limits[class_number][j][1]
                           for j in self.limits[class_number].keys()]
            parameter_table[i, self.cluster_number:-1] = [np.round(np.random.uniform(
                low_limits[i]+0.001, high_limits[i]), 3) for i in range(len(low_limits))]
        parameter_table = np.concatenate(
            (one_hot_vector, parameter_table), axis=1)
        predict_label = self.active_lr.learner.predict(parameter_table)
        label_parameter_table = np.concatenate(
            (parameter_table, predict_label), axis=1)
        return label_parameter_table

    def transform_first_fetch(self, current_setting):
        if self.original is not None:
            transform = self.original
            a, b, c, d, e, f = (e for e in current_setting)
            x = np.concatenate(
                [np.array([range(0, transform.shape[0])]).reshape((-1, 1)), transform])
            x = x.reshape((2, -1))
            A = np.array([[a+1, b*self.original.shape[0]], [c/10, d+1]])
            B = np.array([[e * self.original.shape[0], f]])
            transform = np.matmul(A, x).T + B
            return transform

    def transform(self, original, transform_function):
        if original is not None:
            transform = original
            #print("shape of transform is {}".format(transform.shape))
            a, b, c, d, e, f = (e for e in transform_function)
            x = np.concatenate(
                [np.array([range(0, transform.shape[0])]).reshape((-1, 1)), transform])
            x = x.reshape((2, -1))
            A = np.array([[a+1, b*original.shape[0]], [c/10, d+1]])
            B = np.array([[e * original.shape[0], f]])
            transform = np.matmul(A, x).T + B
        return transform

    def normalize(self, pattern):
        return (pattern - np.min(pattern))/(np.max(pattern)-np.min(pattern))

    def get_linears(self):
        return [e[1:] for e in self.params_lin]

    def get_non_linears(self):
        return [e[1:] for e in self.params_non_lin]

    def set_alphas(self, alpha_lin, alpha_non_lin):
        for i, par in enumerate(self.params_lin):
            par[-1].set(alpha_lin[i])
        for i, par in enumerate(self.params_non_lin):
            par[-1].set(alpha_non_lin[i])

    def sorted_timestamp(self, transform_pattern):
        sorted_timestamp = np.sort(transform_pattern[:, 0])
        #print("transform pattern timestamp is {}".format(transform_pattern[:,0]))
        #print("sorted timestamp is {}".format(sorted_timestamp))
        #print("value is {}".format(
        #    sorted_timestamp == transform_pattern[:, 0]))
        #print("boolean value is {}".format(
        #    np.all(sorted_timestamp == transform_pattern[:, 0])))
        if np.all(sorted_timestamp == transform_pattern[:, 0]):
            return True
        else:
            return False

    #train neural network based on input instance and label
    def neural_network_sima_predict(self,X):
        similarity_score = self.neural_network_sima.nn_predict(X)
        return similarity_score
    
    def logistic_regression(self,instance,label,validation_data,class_weight):
        self.logistic_model = LogisticRegression(class_weight=class_weight,random_state=1,max_iter = 200)
        self.logistic_model.fit(instance,label)
        
    def logistic_regression_predict(self,X):
        logistic_similarity_score = self.logistic_model.predict_proba(X)
        return logistic_similarity_score
    
    def logistic_regression_real(self,instance,label,validation_data,class_weight):
        self.logistic_model_real = LogisticRegression(
            class_weight=class_weight, random_state=1, max_iter=200)
        self.logistic_model_real.fit(instance,label)
        
    def logistic_regression_real_predict(self,X):
        logistic_similarity_score = self.logistic_model_real.predict_proba(X)
        return logistic_similarity_score
    
    def neural_network_train(self, instance, label, validation_data,class_weight):
        self.neural_network = nn(
            instance, label, self.sample_size, validation_data,class_weight,self.params_2)
        self.neural_network_model = self.neural_network.model
    
    def neural_network_train_real(self,instance,label,validation_data,class_weight):
        self.neural_network_real = nn(instance,label,self.sample_size,validation_data,class_weight,self.params_2)
        self.neural_network_model_real = self.neural_network_real.model

    #use neural network to predict similarity score
    def neural_network_flexim_predict(self, X):
        similarity_score = self.neural_network.nn_predict(X)
        return similarity_score
    
    def neural_network_real_predict(self,X):
        similarity_score_real = self.neural_network_real.nn_predict(X)
        return similarity_score_real


    def assign_label(self, original_pattern, transform_pattern, num):
        if num == 1:
            l2_distance = np.linalg.norm(original_pattern-transform_pattern,ord = 2)
            similarity = round(1 - (l2_distance/self.max_distance), 3)
            if similarity >= self.similarity_threshold:
                label = 1
            else:
                label = 0
        elif num == 2:
            sim = (np.dot(np.transpose(original_pattern), transform_pattern) /
                   (np.linalg.norm(original_pattern) * np.linalg.norm(transform_pattern)))[0][0]
            #print("sim is {}".format(sim))
            sim = round(sim/self.max_distance, 3)
            if sim >= self.similarity_threshold:
                label = 1
            else:
                label = 0
        elif num == 3:
            original_pattern = np.array(original_pattern[:,0],dtype = np.double)
            transform_pattern = np.array(transform_pattern[:,0], dtype = np.double)
            dtw_distance = dtw.distance_fast(original_pattern, transform_pattern)
            similarity = round(1 - (dtw_distance/self.max_distance), 3)
            #print("similarity is {}".format(similarity))
            if similarity >= self.similarity_threshold:
                label = 1
            else:
                label = 0
        return label
    
    #may generate negative value
    def assign_score(self,original_pattern,transform_pattern,num):
        if num == 1:
            l2_distance = np.linalg.norm(original_pattern-transform_pattern,ord = 2)
            similarity = round(1 - (l2_distance/self.max_distance), 3)
        elif num == 2:
            similarity = (np.dot(np.transpose(original_pattern), transform_pattern) /
                          (np.linalg.norm(original_pattern) * np.linalg.norm(transform_pattern)))[0][0]
            similarity = round(similarity/self.max_distance, 3)
        elif num == 3:
            original_pattern = np.array(original_pattern[:,0],dtype = np.double)
            transform_pattern = np.array(transform_pattern[:,0],dtype = np.double)
            dtw_distance = dtw.distance_fast(original_pattern, transform_pattern)
            similarity = round(1 - (dtw_distance/self.max_distance), 3)
        return similarity

    def test_similarity_measure(self, X, num):
        print("measure similarity score based on simi kind you choose")
        score_list = []
        score_label = []
        for i in range(len(X)):
            original_pattern = X[i, 0:self.sample_size]
            #print("original pattern shape is {}".format(original_pattern.shape))
            transform_pattern = X[i, self.sample_size:2*self.sample_size]
            #print("transform pattern shape is {}".format(transform_pattern.shape))
            if num == 1:
                l2_distance = np.linalg.norm(
                    original_pattern-transform_pattern,ord = 2)
                similarity = round(1 - (l2_distance/self.max_distance), 3)
                if similarity < 0:
                    similarity = 0
                score_list.append(similarity)
                # print("original pattern is {}".format(original_pattern))
                # print("transform pattern is {}".format(transform_pattern))
                if similarity >= self.similarity_threshold:
                    label = 1
                else:
                    label = 0
                score_label.append(label)
            elif num == 2:
                cos_sim = (np.dot(np.transpose(original_pattern), transform_pattern) /
                       (np.linalg.norm(original_pattern) * np.linalg.norm(transform_pattern)))
                cos_sim = round(cos_sim/self.max_distance, 3)
                score_list.append(cos_sim)
                if cos_sim >= self.similarity_threshold:
                    label = 1
                else:
                    label = 0
                score_label.append(label)
            elif num == 3:
                original_pattern = np.array(original_pattern,dtype = np.double)
                transform_pattern = np.array(transform_pattern,dtype = np.double)
                dtw_distance = dtw.distance_fast(
                    original_pattern, transform_pattern)
                #max_l2_distance = round(dtw.distance_fast([1 for _ in range(self.sample_size)], [0 for _ in range(self.sample_size)]),1)
                similarity = round(1 - (dtw_distance/self.max_distance), 3)
                score_list.append(similarity)
                if similarity >= self.similarity_threshold:
                    label = 1
                else:
                    label = 0
                score_label.append(label)
        score_list = np.array(score_list)
        return score_list, score_label

    def train_committee(self, X, Y, iter_counter, mx_iter):
        print("train active_lr.get_final_learner()")
        if iter_counter >= mx_iter:
            return True
        iter_counter += 1
        print("X shape is {}".format(X.shape))
        print("Y shape is {}".format(Y.shape))
        self.active_lr.get_final_learner().teach(X=X, y=Y, only_new=True)

    def active_learning_emulate_total(self, labeled_data, mx_iter=5):
        stop_state = False
        iter_counter = 0
        print("run into active learning active_lr.get_final_learner() emulate process")
        print("active learning initial score is {}".format(round(self.active_lr.get_final_learner().score(
            labeled_data[:, 0:-1].astype('float32'), labeled_data[:, -1].astype('float32')), 3)))
        print("number of active_lr.get_final_learner() is {}".format(len(self.learner_list)))
        while round(self.active_lr.get_final_learner().score(labeled_data[:, 0:-1].astype('float32'), labeled_data[:, -1].astype('float32')), 3) < 0.85 and not stop_state:
            query_index, query_instance = self.active_lr.get_final_learner().query(
                labeled_data[:, 0:-1])
            #print("active learning query instance is {}".format(query_instance))
            query_instance_label = labeled_data[query_index, -1]
            # print("query instance shape is {}".format(np.shape(query_instance)))
            # print("query instance label is {}".format(
            #     np.shape(query_instance_label)))
            labeled_data = np.delete(
                labeled_data, query_index, axis=0)
            #query_instance,transform_result,problem_label_instance = self.transform(query_instance,self.dataset,self.window_size)
            #print("transform result is {}".format(transform_result))
            # if len(problem_label_instance) != 0:
            #     label_query_instance = np.concatenate((label_query_instance, problem_label_instance), axis=0)
            # else:
            #     label_query_instance = label_query_instance
            stop_state = self.train_committee(
                query_instance, query_instance_label, iter_counter, mx_iter)
            print("active learning score is {}".format(round(self.active_lr.get_final_learner().score(
                labeled_data[:, 0:-1].astype('float32'), labeled_data[:, -1].astype('float32')), 3)))
            #print("stop state is {}".format(stop_state))
        print("emulate process finishes")


def read_data_csv(dataurl):
    temp = pd.read_csv("./"+dataurl)
    temp = temp["close"].values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(temp.reshape(-1,1))
    return data


def load_dataset(argument):
    dataurl = argument
    dataset = read_data_csv(dataurl)
    return dataset

if __name__ == "__main__":
    run_dataset = sys.argv[1]
    parameter_directory = run_dataset[0:-4]
    dataset = load_dataset(run_dataset)
    file = open("./"+parameter_directory+".txt", "r")
    lines = file.readlines()
    sample_size = int(lines[0].strip('\n'))
    simi_kind = 3
    cluster_number = int(lines[2].strip('\n'))
    params = [-2.16,6,9]
    params_2 = [20,0,20,0.8]
    similarity_threshold = 0.6
    emulate = Emulate_TestD(dataset, parameter_directory,
                            simi_kind, cluster_number, sample_size,similarity_threshold)
    #load autoencoder encoder modela
    emulate.load_encoder_model(parameter_directory)
    #load kmeans model
    emulate.load_kmeans_model(parameter_directory)
    emulate.__init_lin__()
    emulate.__init_both_cluster_directory__(cluster_number)
    emulate.__init_limit_dictionary__()
    emulate.add_cluster_dict()
    emulate.get_max_distance(instance_number=300000)
    emulate.sample()
    emulate.load_parameter(params_2)
    #print("self.original is {}".format(emulate.original))
    emulate.__fetch__(simi_kind,params)
