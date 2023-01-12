from turtle import color, distance
from unittest import result
import numpy as np
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from itertools import count
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
import math
import heapq
#a400,10,6 has problem
class Pattern_Search:
    def __init__(self,dataset,sample_size):
        self.sample_size = sample_size
        self.dataset = dataset
    
    def plot_compare_curve(self,query_start, query_end, result_start, result_end, num):
        scale = np.arange(0, len(self.dataset))
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(scale, self.dataset)
        ax[0].set(xlabel='time', ylabel='original_dataset')
        ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
        if num == 1:
            ax[0].axvspan(result_start[0], result_end[0], color='blue', alpha=0.2)
        elif num == 2:
            ax[0].axvspan(result_start[0], result_end[0], color='green', alpha=0.2)
        elif num == 3:
            ax[0].axvspan(result_start[0], result_end[0], color='purple', alpha=0.2)
        t = np.arange(query_start, query_end)
        ax[1].plot(t, self.dataset[t], color='r')
        ax[1].set(xlabel='time', ylabel='query pattern')
        ax[1].grid(True)
        #ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
        if(num == 1):
                t = np.arange(result_start[0],result_end[0])
                scale = np.arange(0,1.1,0.1)
                ax[2].plot(t, self.dataset[t])
                ax[2].set(xlabel='time', ylabel='cosine_similarity')
                ax[2].set_yticks(scale)
                ax[2].grid(True)
                #ax[2].axvspan(result_start[0], result_end[0],color='green', alpha=0.2)
        elif(num == 2):
                t = np.arange(result_start[0], result_end[0])
                scale = np.arange(0, 1.1, 0.1)
                ax[2].plot(t, self.dataset[t])
                ax[2].set(xlabel='time', ylabel='DTW')
                ax[2].set_yticks(scale)
                ax[2].grid(True)
                #ax[2].axvspan(result_start[i], result_end[i],color='blue', alpha=0.2)
        else:  
                t = np.arange(result_start[0], result_end[0])
                scale = np.arange(0, 1.1, 0.1)
                ax[2].plot(t, self.dataset[t])
                ax[2].set(xlabel='time', ylabel='Flexim')
                ax[2].set_yticks(scale)
                ax[2].grid(True)
        fig.savefig("compare_{}.png".format(num))
        fig.tight_layout()
        plt.show()
    
    
    def cosine_similarity(self,array1,array2):
        cos_sim = dot(np.transpose(array1), array2) / (norm(array1) * norm(array2))
        return cos_sim
    
    def detect_duplicate(self,similar_pattern,query_pattern):
        #print("run function")
        #print("size of duplicate similar_pattern is {}".format(np.shape(similar_pattern)))
        #print("size of duplicate query_pattern is {}".format(np.shape(query_pattern)))
        #detect number of duplicate elements
        length = len(set(similar_pattern) & set(query_pattern))
        #print("the length of list is {}".format(length))
        if(length) <= 0.4 * self.sample_size:
            return True
        else:
            return False
        
    def up_downsample(similar_pattern,query_pattern):
        similar_pattern = signal.resample(similar_pattern, len(query_pattern), axis=0)
        
    def find_pattern_cosine(self,query_pattern,stride_size,kvalue):
        length = len(self.dataset)
        N = int(length/stride_size)
        score_list = []
        tiebreaker = count()
        similarity_score = 0
        select_number = 0
        for batch in range(N):
            #use window to get pattern data
            similar_pattern = self.dataset[batch*stride_size:self.sample_size+batch*stride_size,0].tolist()
            #print("length of similar pattern {}".format(similar_pattern))
            #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
            #check whether window period is larger then length of self.dataset
            if self.sample_size+batch*stride_size > length:
                similar_pattern = self.dataset[batch*stride_size:length]
            #if number of duplicate elements is smaller than specific value
            if(len(similar_pattern)!=len(query_pattern)):
                #print("original similar pattern {}".format(similar_pattern))
                similar_pattern = signal.resample(similar_pattern, len(query_pattern), axis=0)
                #print("similar_pattern changed {}".format(similar_pattern))
                #print("if statemnet pattern matching is {}".format(similar_pattern))
            else:
                #print("happy")
                if(self.detect_duplicate(similar_pattern,query_pattern)):
                    #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
                    similarity_score = cosine_similarity(np.array(similar_pattern).reshape(1,-1),np.array(query_pattern).reshape(1,-1))[0][0]
                    #print("score is {}".format(similarity_score))
                    #print("similarity_score is".format(similarity_score))
                    #print("similarity_score is {}".format(similarity_score))
                    if similarity_score >= 0:
                        #print("batch is {}".format(batch))
                        if select_number<kvalue:
                            heapq.heappush(score_list, (similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size,similar_pattern))
                                #print("during push process the score_list is {}".format(np.array(score_list).shape))
                            select_number+=1
                        else:
                            #print("during process the score_list is {}".format(np.array(score_list).shape))
                            #print("run this part")
                                # check whether it should be pushed to heap
                            #print('score_list is {}'.format(score_list))
                            #print("score_list is {}".format(score_list))
                            if similarity_score > heapq.nsmallest(1, score_list)[0][0]:
                                heapq.heapreplace(score_list, (similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
                #print("else statemnet pattern matching is {}".format(similar_pattern))
        #extract k smallest value from heap
        heap = heapq.nlargest(kvalue,score_list)
        return heap
    
    def find_pattern_dtw(self,query_pattern,stride_size, kvalue):
        length = len(self.dataset)
        N = int(length/stride_size)
        score_list = []
        tiebreaker = count()
        similarity_score = 0
        select_number = 0
        for batch in range(N):
            #use window to get pattern data
            similar_pattern = self.dataset[batch * stride_size:self.sample_size+batch*stride_size,0].tolist()
            #print("length of similar pattern {}".format(similar_pattern))
            #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
            #check whether window period is larger then length of self.dataset
            if self.sample_size+batch*stride_size > length:
                similar_pattern = self.dataset[batch*stride_size:length]
            #if number of duplicate elements is smaller than specific value
            if(len(similar_pattern) != len(query_pattern)):
                #print("original similar pattern {}".format(similar_pattern))
                similar_pattern = signal.resample(
                    similar_pattern, len(query_pattern), axis=0)
                #print("similar_pattern changed {}".format(similar_pattern))
                #print("if statemnet pattern matching is {}".format(similar_pattern))
            else:
                #print("happy")
                if(self.detect_duplicate(similar_pattern, query_pattern)):
                    #print('yes')
                    #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
                    similarity_score = dtw.distance(query_pattern,similar_pattern)
                    print("DTW similarity score is {}".format(similarity_score))
                    #print("similarity_score is".format(similarity_score))
                    #print("similarity_score is {}".format(similarity_score))
                    if similarity_score >= 0:
                        #print("batch is {}".format(batch))
                        if select_number < kvalue:
                            heapq.heappush(
                                score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
                            #print("during push process the score_list is {}".format(np.array(score_list).shape))
                            #print("score_list is {}".format(score_list))
                            select_number += 1
                        else:
                            #print("during process the score_list is {}".format(np.array(score_list).shape))
                            #print("run this part")
                            # check whether it should be pushed to heap
                            #print('score_list is {}'.format(score_list))
                            #print("score_list is {}".format(score_list))
                            #print("heapq queue is {}".format(heapq.nsmallest(1, score_list)))
                            if (-1)*similarity_score > heapq.nsmallest(1, score_list)[0][0]:
                                heapq.heapreplace(
                                    score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
                                #print("score_list 2 is {}".format(score_list))
                #print("else statemnet pattern matching is {}".format(similar_pattern))
        #extract k smallest value from heap
        heap = heapq.nlargest(kvalue, score_list)
        return heap
    
    def find_pattern_model_regression(self,query_pattern,stride_size, kvalue, model):
        length = len(self.dataset)
        N = int(length/stride_size)
        score_list = []
        tiebreaker = count()
        similarity_score = 0
        select_number = 0
        for batch in range(N):
            #use window to get pattern data
            similar_pattern = self.dataset[batch *
                                    stride_size:self.sample_size+batch*stride_size,0].tolist()
            #print("length of similar pattern {}".format(similar_pattern))
            #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
            #check whether window period is larger then length of self.dataset
            if self.sample_size+batch*stride_size > length:
                similar_pattern = self.dataset[batch*stride_size:length]
            #if number of duplicate elements is smaller than specific value
            if(len(similar_pattern) != len(query_pattern)):
                #print("original similar pattern {}".format(similar_pattern))
                similar_pattern = signal.resample(
                    similar_pattern, len(query_pattern), axis=0)
                #print("similar_pattern changed {}".format(similar_pattern))
                #print("if statemnet pattern matching is {}".format(similar_pattern))
            else:
                #print("happy")
                if(self.detect_duplicate(similar_pattern, query_pattern)):
                    #print('yes')
                    #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
                    append_pattern = np.append(query_pattern, similar_pattern)
                    append_pattern = append_pattern.reshape(1,-1)
                    #print(append_pattern.shape)
                    similarity_score = model.predict(append_pattern,verbose = 0)[0][0]
                    #print("similarity score of Flexim is {}".format(similarity_score))
                    #print("similarity_score is".format(similarity_score))
                    #print("similarity_score is {}".format(similarity_score))
                    if similarity_score > 0:
                        #print("batch is {}".format(batch))
                        if select_number < kvalue:
                            heapq.heappush(score_list, (similarity_score, next(tiebreaker),
                                        batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
                            select_number += 1
                            #print("during push process the score_list is {}".format(np.array(score_list).shape))
                            #print("score_list is {}".format(score_list))
                        else:
                        #     #print("during process the score_list is {}".format(np.array(score_list).shape))
                        #     #print("run this part")
                        #     # check whether it should be pushed to heap
                        #     #print('score_list is {}'.format(score_list))
                        #     #print("score_list is {}".format(score_list))
                            if similarity_score > heapq.nsmallest(1, score_list)[0][0]:
                                heapq.heapreplace(
                                    score_list, (similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
                            #print("score_list 2 is {}".format(score_list))
                #print("else statemnet pattern matching is {}".format(similar_pattern))
        #extract k largest value from heap
        heap = heapq.nlargest(kvalue, score_list)
        return heap
    
    def record_index(self,result):
        start_list = []
        end_list = []
        for i in range(len(result)):
            start_list.append(result[i][2])
            end_list.append(result[i][3])
        return start_list,end_list
        
    
    
    
        
# def read_data(dataurl):
#     list = []
#     with open(dataurl) as  data_file:
#         for line in data_file:
#             list.append(line.split()) #split sentence by blank
#     #reshape array
#     array = np.array(list, dtype=float)
#     #delete NaN
#     array = array[np.isfinite(array)]
#     #print("array is {}".format(array))
#     # array = array.dropna(axis = 1).astype(float)  # 丢弃有缺失值的列
#     for i in range(len(array)):
#         if array[i] == "NaN":
#             print("NaN value still exists")
#     return array
    
# def read_data_csv(dataurl):
#     df = pd.read_csv(dataurl,header=None)
#     df = df.loc[:,1]
#     df = df.drop([0],axis = 0)
#     df = df.values.astype('float64')
#     return df

# def plot_figure(self.dataset):
#     t = np.arange(1,len(self.dataset)+1)
#     #self.dataset = self.dataset.values.astype('float64')
#     fig,ax = plt.subplots()
#     ax.plot(t,self.dataset)
#     ax.set(xlabel = 'time',ylabel = 'AccelerationX')
#     ax.grid()
#     fig.savefig("test.png")
#     plt.show()

# def sin_fun(self.dataset,begin,end):
#     for i in range(begin,end+1):
#         self.dataset[i] = math.sin(i)
#     return self.dataset

# def cos_fun(self.dataset,begin,end):
#     for i in range(begin, end+1):
#         self.dataset[i] = math.cos(i)
#     return self.dataset

# def tan_fun(self.dataset,begin,end):
#     for i in range(begin,end+1):
#         self.dataset[i] = math.tan(i)
#     return self.dataset
     
    
    
    
# def plot_compare(self.dataset, query_start, query_end, result_start, result_end, num):
#     scale = np.arange(0,len(self.dataset))
#     fig, ax = plt.subplots(3, 1)
#     ax[0].plot(scale,self.dataset)
#     ax[0].set(xlabel='time', ylabel='AccelerationX')
#     ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
#     t = np.arange(query_start, query_end)
#     ax[1].plot(t, self.dataset[t],color = 'r')
#     ax[1].set(xlabel='time', ylabel='AccelerationX')
#     ax[1].grid(True)
#     #ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
#     if(num == 1):
#         for i in range(len(result_start)):
#             ax[2].plot(scale, self.dataset)
#             ax[2].plot(t, self.dataset[t], color='m')
#             ax[2].set(xlabel='time', ylabel='cosine_similarity')
#             ax[2].grid(True)
#             ax[2].axvspan(result_start[i], result_end[i], color='green', alpha=0.2)
#     elif(num == 2):
#         for i in range(len(result_start)):
#             ax[2].plot(scale, self.dataset)
#             ax[2].plot(t, self.dataset[t],color = 'r')
#             ax[2].set(xlabel='time', ylabel='DTW')
#             ax[2].grid(True)
#             ax[2].axvspan(result_start[i], result_end[i],color='blue', alpha=0.2)
#     else:
#         for i in range(len(result_start)):
#             ax[2].plot(scale, self.dataset)
#             ax[2].plot(t, self.dataset[t],color = 'b')
#             ax[2].set(xlabel='time', ylabel='Flexim')
#             ax[2].grid(True)
#             ax[2].axvspan(result_start[i], result_end[i],color='yellow', alpha=0.2)
            
#     fig.savefig("score_list_{}.png".format(num))
#     fig.tight_layout()
#     plt.show()
    
    
# def plot_compare_curve(self.dataset, query_start, query_end, result_start, result_end, num):
#     scale = np.arange(0, len(self.dataset))
#     fig, ax = plt.subplots(3, 1)
#     ax[0].plot(scale, self.dataset)
#     ax[0].set(xlabel='time', ylabel='AccelerationX')
#     ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
#     t = np.arange(query_start, query_end)
#     ax[1].plot(t, self.dataset[t], color='r')
#     ax[1].set(xlabel='time', ylabel='AccelerationX')
#     ax[1].grid(True)
#     #ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
#     if(num == 1):
#             t = np.arange(result_start[0],result_end[0])
#             ax[2].plot(t, self.dataset[t])
#             ax[2].set(xlabel='time', ylabel='cosine_similarity')
#             ax[2].grid(True)
#             #ax[2].axvspan(result_start[0], result_end[0],color='green', alpha=0.2)
#     elif(num == 2):
#             t = np.arange(result_start[0], result_end[0])
#             ax[2].plot(t, self.dataset[t])
#             ax[2].set(xlabel='time', ylabel='DTW')
#             ax[2].grid(True)
#             #ax[2].axvspan(result_start[i], result_end[i],color='blue', alpha=0.2)
#     else:  
#             t = np.arange(result_start[0], result_end[0])
#             ax[2].plot(t, self.dataset[t])
#             ax[2].set(xlabel='time', ylabel='Flexim')
#             ax[2].grid(True)
#     fig.savefig("compare_{}.png".format(num))
#     fig.tight_layout()
#     plt.show()
    
    

# # def cosine_similarity(array1,array2):
# #     cos_sim = dot(array1, array2) / (norm(array1) * norm(array2))
# #     return cos_sim

# def detect_duplicate(similar_pattern,query_pattern,self.sample_size):
#     #print("run function")
#     #print("size of duplicate similar_pattern is {}".format(np.shape(similar_pattern)))
#     #print("size of duplicate query_pattern is {}".format(np.shape(query_pattern)))
#     #detect number of duplicate elements
#     length = len(set(similar_pattern) & set(query_pattern))
#     #print("the length of list is {}".format(length))
#     if(length) <= 0.4*self.sample_size:
#         return True
#     else:
#         return False

# def up_downsample(similar_pattern,query_pattern):
#     similar_pattern = signal.resample(similar_pattern, len(query_pattern), axis=0)
        

                
 
# def find_pattern(query_pattern,self.dataset,self.sample_size,stride_size,kvalue):
#     length = len(self.dataset)
#     N = int(length/stride_size)
#     score_list = []
#     tiebreaker = count()
#     similarity_score = 0
#     select_number = 0
#     for batch in range(N):
#         #use window to get pattern data
#         similar_pattern = self.dataset[batch*stride_size:self.sample_size+batch*stride_size]
#         #print("length of similar pattern {}".format(similar_pattern))
#         #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
#         #check whether window period is larger then length of self.dataset
#         if self.sample_size+batch*stride_size > length:
#             similar_pattern = self.dataset[batch*stride_size:length]
#         #if number of duplicate elements is smaller than specific value
#         if(len(similar_pattern)!=len(query_pattern)):
#             #print("original similar pattern {}".format(similar_pattern))
#             similar_pattern = signal.resample(similar_pattern, len(query_pattern), axis=0)
#             #print("similar_pattern changed {}".format(similar_pattern))
#             #print("if statemnet pattern matching is {}".format(similar_pattern))
#         else:
#             #print("happy")
#             if(detect_duplicate(similar_pattern,query_pattern,self.sample_size)):
#                 #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
#                 similarity_score = cosine_similarity(np.array(similar_pattern).reshape(1,-1),np.array(query_pattern).reshape(1,-1))[0][0]
#                 #print("score is {}".format(similarity_score))
#                 #print("similarity_score is".format(similarity_score))
#                 #print("similarity_score is {}".format(similarity_score))
#                 if similarity_score >= 0:
#                     #print("batch is {}".format(batch))
#                     if select_number<kvalue:
#                         heapq.heappush(score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size,similar_pattern))
#                             #print("during push process the score_list is {}".format(np.array(score_list).shape))
#                         select_number+=1 
#                     else:
#                         #print("during process the score_list is {}".format(np.array(score_list).shape))
#                         #print("run this part")
#                             # check whether it should be pushed to heap
#                         #print('score_list is {}'.format(score_list))
#                         #print("score_list is {}".format(score_list))
#                         if (-1)*similarity_score > heapq.nsmallest(1, score_list)[0][0]:
#                             heapq.heappushpop(score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#             #print("else statemnet pattern matching is {}".format(similar_pattern))
#     #extract k smallest value from heap
#     heap = heapq.nlargest(kvalue,score_list)
#     return heap



# def find_pattern_dtw(query_pattern, self.dataset, self.sample_size, stride_size, kvalue):
#     length = len(self.dataset)
#     N = int(length/stride_size)
#     score_list = []
#     tiebreaker = count()
#     similarity_score = 0
#     select_number = 0
#     for batch in range(N):
#         #use window to get pattern data
#         similar_pattern = self.dataset[batch * stride_size:self.sample_size+batch*stride_size]
#         #print("length of similar pattern {}".format(similar_pattern))
#         #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
#         #check whether window period is larger then length of self.dataset
#         if self.sample_size+batch*stride_size > length:
#             similar_pattern = self.dataset[batch*stride_size:length]
#         #if number of duplicate elements is smaller than specific value
#         if(len(similar_pattern) != len(query_pattern)):
#             #print("original similar pattern {}".format(similar_pattern))
#             similar_pattern = signal.resample(
#                 similar_pattern, len(query_pattern), axis=0)
#             #print("similar_pattern changed {}".format(similar_pattern))
#             #print("if statemnet pattern matching is {}".format(similar_pattern))
#         else:
#             #print("happy")
#             if(detect_duplicate(similar_pattern, query_pattern, self.sample_size)):
#                 #print('yes')
#                 #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
#                 similarity_score = dtw.distance(query_pattern,similar_pattern)
#                 #print("similarity_score is".format(similarity_score))
#                 #print("similarity_score is {}".format(similarity_score))
#                 if similarity_score >= 0:
#                     #print("batch is {}".format(batch))
#                     if select_number < kvalue:
#                         heapq.heappush(
#                             score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                         #print("during push process the score_list is {}".format(np.array(score_list).shape))
#                         #print("score_list is {}".format(score_list))
#                         select_number += 1
#                     else:
#                         #print("during process the score_list is {}".format(np.array(score_list).shape))
#                         #print("run this part")
#                         # check whether it should be pushed to heap
#                         #print('score_list is {}'.format(score_list))
#                         #print("score_list is {}".format(score_list))
#                         #print("heapq queue is {}".format(heapq.nsmallest(1, score_list)))
#                         if (-1)*similarity_score > heapq.nsmallest(1, score_list)[0][0]:
#                             heapq.heappushpop(
#                                 score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                             #print("score_list 2 is {}".format(score_list))
#             #print("else statemnet pattern matching is {}".format(similar_pattern))
#     #extract k smallest value from heap
#     heap = heapq.nlargest(kvalue, score_list)
#     return heap

# def find_pattern_model(query_pattern,self.dataset,self.sample_size,stride_size,kvalue,model):
#     length = len(self.dataset)
#     N = int(length/stride_size)
#     score_list = []
#     tiebreaker = count()
#     similarity_score = 0
#     select_number = 0
#     for batch in range(N):
#         #use window to get pattern data
#         similar_pattern = self.dataset[batch *
#                                   stride_size:self.sample_size+batch*stride_size]
#         #print("length of similar pattern {}".format(similar_pattern))
#         #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
#         #check whether window period is larger then length of self.dataset
#         if self.sample_size+batch*stride_size > length:
#             similar_pattern = self.dataset[batch*stride_size:length]
#         #if number of duplicate elements is smaller than specific value
#         if(len(similar_pattern) != len(query_pattern)):
#             #print("original similar pattern {}".format(similar_pattern))
#             similar_pattern = signal.resample(similar_pattern, len(query_pattern), axis=0)
#             #print("similar_pattern changed {}".format(similar_pattern))
#             #print("if statemnet pattern matching is {}".format(similar_pattern))
#         else:
#             #print("happy")
#             if(detect_duplicate(similar_pattern, query_pattern, self.sample_size)):
#                 #print('yes')
#                 #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
#                 append_pattern = np.append(query_pattern,similar_pattern)
#                 append_pattern = append_pattern.reshape(1,len(append_pattern))
#                 #print(append_pattern.shape)
#                 similarity_score = nn.nn_multi_predict_single(model,append_pattern)
#                 #print("similarity_score is".format(similarity_score))
#                 #print("similarity_score is {}".format(similarity_score))
#                 if similarity_score == 1:
#                     #print("batch is {}".format(batch))
#                     if select_number < kvalue:
#                         heapq.heappush(score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                         select_number += 1
#                         #print("during push process the score_list is {}".format(np.array(score_list).shape))
#                         #print("score_list is {}".format(score_list))
#                     # else:
#                     #     #print("during process the score_list is {}".format(np.array(score_list).shape))
#                     #     #print("run this part")
#                     #     # check whether it should be pushed to heap
#                     #     #print('score_list is {}'.format(score_list))
#                     #     #print("score_list is {}".format(score_list))
#                     #     if (-1)*similarity_score > heapq.nsmallest(1, score_list)[0][0]:
#                     #         heapq.heappushpop(
#                     #             score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                             #print("score_list 2 is {}".format(score_list))
#             #print("else statemnet pattern matching is {}".format(similar_pattern))
#     #extract k smallest value from heap
#     heap = heapq.nlargest(kvalue, score_list)
#     return heap

# def find_pattern_model_regression(query_pattern, self.dataset, self.sample_size, stride_size, kvalue, model):
#     length = len(self.dataset)
#     N = int(length/stride_size)
#     score_list = []
#     tiebreaker = count()
#     similarity_score = 0
#     for batch in range(N):
#         #use window to get pattern data
#         similar_pattern = self.dataset[batch *
#                                   stride_size:self.sample_size+batch*stride_size]
#         #print("length of similar pattern {}".format(similar_pattern))
#         #print("size of similar pattern is {}".format(np.shape(similar_pattern)))
#         #check whether window period is larger then length of self.dataset
#         if self.sample_size+batch*stride_size > length:
#             similar_pattern = self.dataset[batch*stride_size:length]
#         #if number of duplicate elements is smaller than specific value
#         if(len(similar_pattern) != len(query_pattern)):
#             #print("original similar pattern {}".format(similar_pattern))
#             similar_pattern = signal.resample(
#                 similar_pattern, len(query_pattern), axis=0)
#             #print("similar_pattern changed {}".format(similar_pattern))
#             #print("if statemnet pattern matching is {}".format(similar_pattern))
#         else:
#             #print("happy")
#             if(detect_duplicate(similar_pattern, query_pattern, self.sample_size)):
#                 #print('yes')
#                 #print("len of similar_pattern of main function {}".format(len(similar_pattern)))
#                 append_pattern = np.append(query_pattern, similar_pattern)
#                 append_pattern = append_pattern.reshape(1, len(append_pattern))
#                 #print(append_pattern.shape)
#                 similarity_score = nn.predict_single(model, append_pattern)
#                 #print("similarity_score is".format(similarity_score))
#                 print("similarity_score is {}".format(similarity_score))
#                 if similarity_score == 1:
#                     #print("batch is {}".format(batch))
#                     if batch <= kvalue:
#                         heapq.heappush(score_list, (-1*similarity_score, next(tiebreaker),
#                                        batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                         #print("during push process the score_list is {}".format(np.array(score_list).shape))
#                         #print("score_list is {}".format(score_list))
#                     # else:
#                     #     #print("during process the score_list is {}".format(np.array(score_list).shape))
#                     #     #print("run this part")
#                     #     # check whether it should be pushed to heap
#                     #     #print('score_list is {}'.format(score_list))
#                     #     #print("score_list is {}".format(score_list))
#                     #     if (-1)*similarity_score > heapq.nsmallest(1, score_list)[0][0]:
#                     #         heapq.heappushpop(
#                     #             score_list, (-1*similarity_score, next(tiebreaker), batch*stride_size, self.sample_size+batch*stride_size, similar_pattern))
#                         #print("score_list 2 is {}".format(score_list))
#             #print("else statemnet pattern matching is {}".format(similar_pattern))
#     #extract k smallest value from heap
#     heap = heapq.nlargest(kvalue, score_list)
#     return heap




# def record_index(result):
#     start_list = []
#     end_list = []
#     for i in range(len(result)):
#         start_list.append(result[i][2])
#         end_list.append(result[i][3])
#     return start_list,end_list


        
                
    
# def main():
#     self.dataset = read_data_csv("/Users/wangzhuo/Documents/Brown CS Research/research/Amir/Flexim/Ipin2016self.dataset/measure1_smartphone_sens.csv")
#    # print(type(self.dataset))
#     #print("self.dataset: {}".format(self.dataset))
#     #print("size of self.dataset is {}".format(np.shape(self.dataset)))
#     self.dataset = self.dataset[0:200]
#     self.sample_size = 10
#     start = 5
#     end = start + self.sample_size
#     self.dataset = sin_fun(self.dataset,start,end)
#     query_pattern = self.dataset[start:end]
#     stride_size = 2
#     score_list_1 = find_pattern(query_pattern,self.dataset,self.sample_size,stride_size,10)
#     score_list_2 = find_pattern_dtw(query_pattern,self.dataset,self.sample_size,stride_size,10)
#     #print("method 1 score list is : {}".format(score_list_1))
#     #print("method 2 score list is : {}".format(score_list_2))
#     score_1_index_start,score_1_index_end = record_index(score_list_1)
#     score_2_index_start,score_2_index_end = record_index(score_list_2)
    
#     # df1 = pd.DataFrame(score_list_1)
#     # df1.to_csv('score_list_1.csv')
#     # df2 = pd.DataFrame(score_list_2)
#     # df2.to_csv('score_list_2.csv')
    
    
#     plot_compare(self.dataset,start,end,score_1_index_start,score_1_index_end,1)
#     plot_compare(self.dataset,start,end,score_2_index_start,score_2_index_end,2)
    
#     plot_compare_curve(self.dataset,start,end,score_1_index_start,score_1_index_end,1)
#     plot_compare_curve(self.dataset,start,end,score_2_index_start,score_2_index_end,2)
#     print(score_list_1)
#     print(score_list_2)
# if __name__ == '__main__':
# 	main()
 
 