from cgi import test
from random import random, shuffle
from tabnanny import verbose
import numpy as np
from modAL.models import ActiveLearner
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from AE import AE_Model, AutoEncoder
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

class Active:
    def __init__(self, un_labled_data, labeled_data,mx_iter=1):
	    # main part of the implementation goes here including
        # the initialization and definition of neural net and
        # active learning components. As an example:
        self.un_labled_data=un_labled_data
        self.iter_counter = 0
        self.mx_iter = mx_iter
        self.un_label_data = un_labled_data
        batch_size = 5
        preset_batch = partial(uncertainty_batch_sampling,n_instances = batch_size)
        num_training_data = len(labeled_data)
        self.train_label, self.test_label = train_test_split(labeled_data,test_size = 0.2,random_state = 42,shuffle = None)
        self.train_label, self.valid_label = train_test_split(self.train_label,test_size = 0.2, random_state = 44, shuffle = None)
        #training_indices = np.random.randint(low = 0,high = num_training_data,size = 4*num_training_data //5)
        self.nn_estimator_model = self.keras_model(labeled_data)
        # self.nn_estimator_model.fit(
        #     labeled_data[training_indices, 0:-1], labeled_data[training_indices, -1])
        self.nn_estimator_model_l2 = tf.keras.models.load_model("./"+str(1)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        self.nn_estimator_model_dtw = tf.keras.models.load_model("./"+str(3)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        self.nn_estimator_model_cosine = tf.keras.models.load_model("./"+str(2)+str(5)+'/saved_model'+str(60)+'/'+'model_1')
        #estimator_list = [('l2 model',KerasClassifier(self.nn_estimator_model_l2))]
        estimator_list = [('l2 model',KerasClassifier(self.nn_estimator_model_l2)),('cosine',KerasClassifier(self.nn_estimator_model_cosine)),('dtw',KerasClassifier(self.nn_estimator_model_dtw))]
        #self.nn_estimator_model.fit(labeled_data[training_indices,0:-1],labeled_data[training_indices,-1])
        #self.learner = ActiveLearner(estimator = KerasClassifier(self.nn_estimator_model), query_strategy=preset_batch, X_training=self.train_label[:,0:-1], y_training=self.train_label[:,-1],verbose = 1)
        #self.learner = ActiveLearner(estimator = KerasClassifier(self.nn_estimator_model_dtw),query_strategy = preset_batch,X_training = self.train_label[:,0:-1],y_training = self.train_label[:,-1])
        self.learner = ActiveLearner(estimator = StackingClassifier(estimators = estimator_list,final_estimator = LogisticRegression(),stack_method = 'predict_proba'),query_strategy = preset_batch, X_training = self.train_label[:,0:-1], y_training = self.train_label[:,-1])
        self.learner.teach(X = self.train_label[:,0:-1],y = self.train_label[:,-1])
        self.query_time = 0
        
    def train(self, X, Y):
        print('train')
        print(X.shape)
        print(Y.shape)
        if self.iter_counter >= self.mx_iter:
          return True
        self.iter_counter += 1
        self.learner.teach(X=X, y=Y)
        return False
    
    def query(self): #
        print("un_labled_data shape is {}".format(self.un_labled_data.shape))
        query_index, query_instance = self.learner.query(self.un_labled_data)
        self.un_labled_data = np.delete(self.un_labled_data, query_index, axis=0)
        print("remain unlabel data is {}".format(self.un_labled_data.shape[0]))
        return query_instance

    def get_final_learner(self):
        return self.learner
    
    def get_remain_label_data(self):
        return self.labeled_data

    def get_model(self):
        return self.nn_estimator_model
    
    def active_learning(self,data,labels,unlabel_data):
        stop_state = False
        # For score part just uses 1/5 of whole data
        for time in range(3):
            query_index,query_instance = self.learner.query(unlabel_data)
            self.query_time += 1
            self.train_unlabel = np.delete(self.train_unlabel,query_index,axis = 0)
            #label_query_instance = 
            
            
        while self.learner.score(data, labels) < 0.85 and not stop_state:
            query_index, query_instance = self.learner.query(unlabel_data)
            # label_query_instance =
            stop_state = self.train(query_instance, self.label_query_instance)
            unlabel_data = np.delete(unlabel_data, query_index, axis=0)
            print("active learning score is {}".format(self.learner.score(data,labels)))
                
    def predict(self,data):
        label = self.learner.predict(data)
        return label
    

    def keras_model(self,labeled_data):
        lr = 2.16
        fl = 6
        sl = 9
        gamma = 0
        label_smoothing = 0
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
    
    
    
    # def autoencoder(self,label_instance):
    #     input_dim = label_instance.shape[1]
    #     hidden_dim = (1/2)*input_dim
    #     latent_dim = (1/10)*input_dim
    #     autoencoder = AutoEncoder(input_dim, hidden_dim, latent_dim)
    #     autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    #     autoencoder.fit(label_instance, label_instance,epochs=3000, batch_size=40, shuffle=True)
    #     return autoencoder
    
    # def get_cluster_label(self,dataset,instance):
    #     input_dim = dataset.shape[1]
    #     hidden_dim = (1/2)*input_dim
    #     latent_dim = (1/10)*input_dim
    #     ae = AE_Model(input_dim,hidden_dim,latent_dim)
    #     cluster_label, cluster_centers = ae.get_cluster_label(10,dataset)
    #     assign_cluster_label = ae.match_cluster_label(instance,cluster_centers)
    #     return assign_cluster_label
    
    
    
           
    
#transform original input_data to one-hot vector
def transform_data(input_data,label_set):
    n_labels = max(label_set)
    one_hot_vector = np.eye(n_labels)[label_set]
    transfer_data = np.ones([input_data.shape[0],n_labels+1+len(input_data)])
    for i in range(one_hot_vector.shape[0]):
        transfer_data[i,0:n_labels+2] = one_hot_vector[i,:]
        transfer_data[i,n_labels+1:-1] = input_data[i,:]
    return transfer_data        
