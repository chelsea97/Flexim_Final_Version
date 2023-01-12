from re import X
import pandas as pd
data = pd.read_csv("./"+"all_stocks_5yr.csv",index_col = None)
data = data["close"].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="1"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import skopt
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from modAL.batch import uncertainty_batch_sampling
import pickle
#from Active_Emulate_TestD_gp import Active_Emulate_Test as ace
#from NN import NN as nn
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from dtaidistance import dtw
from numpy import dot
from numpy.linalg import norm
import sys
from sklearn.utils import shuffle
import copy
import emulate_st500_em_new_write_exac as et
import importlib
import Active_Emulate_st500_em_new_exac as at
from skopt import dump, load
parameter_directory = "all_stocks_5yr"
file = open("./"+parameter_directory+".txt","r")
lines = file.readlines()
sample_size = int(lines[0].strip('\n'))
simi_kind = 3
cluster_number = int(lines[2].strip('\n'))
similarity_threshold = 0.9
#emulate single new
def obj(params_2):
    #print("simi_kind is {}".format(simi_kind))
    params = [-2.16,6,9]
    emulate = et.Emulate_TestD(data,parameter_directory,simi_kind,cluster_number,sample_size,similarity_threshold)
    emulate.load_encoder_model(parameter_directory)
    emulate.load_kmeans_model(parameter_directory)
    emulate.__init_lin__()
    emulate.__init_both_cluster_directory__(cluster_number)
    emulate.__init_limit_dictionary__()
    emulate.add_cluster_dict()
    emulate.get_max_distance(instance_number = 30000)
    emulate.sample()
    emulate.load_parameter(params_2)
    emulate.__fetch__(simi_kind,params)
    print("test accuracy is {}".format(emulate.final_accuracy))
    print("accuracy threshold is {}".format(emulate.similarity_threshold))
    print("wass distance is {}".format(emulate.final_accuracy))
    print('#######################################################################################')
    return emulate.final_accuracy
SPACE = [
   skopt.space.Integer(0,50,name = 'gamma 1'),
   skopt.space.Real(0,1,name = 'label_smoothing 1'),
   skopt.space.Integer(0,50,name = 'gamma 2'),
   skopt.space.Real(0,1,name = 'label_smoothing 2'),
]
res = skopt.gp_minimize(obj,SPACE,n_calls = 50,random_state = 44,n_jobs = 5)
dump(res,'result'+str(similarity_threshold)+'.pkl')
res_loaded = load('result'+str(similarity_threshold)+'.pkl')
print("value is {}".format(res_loaded.fun))
print("x is {}".format(res_loaded.x))
