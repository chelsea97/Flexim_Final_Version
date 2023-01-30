import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def plot_compare_curve(dataset,query_start, query_end, result_start, result_end, num,index):
    scale = np.arange(0, len(dataset))
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(scale, dataset)
    ax[0].set(xlabel='time', ylabel='original_dataset')
    ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
    if num == 1:
        ax[0].axvspan(result_start[index], result_end[index], color='blue', alpha=0.2)
    elif num == 2:
        ax[0].axvspan(result_start[index], result_end[index], color='green', alpha=0.2)
    elif num == 3:
        ax[0].axvspan(result_start[index], result_end[index], color='purple', alpha=0.2)
    t = np.arange(query_start, query_end)
    ax[1].plot(t, dataset[t], color='r')
    ax[1].set(xlabel='time', ylabel='query pattern')
    ax[1].grid(True)
    #ax[0].axvspan(query_start, query_end, color='red', alpha=0.2)
    if(num == 1):
            t = np.arange(result_start[index],result_end[index])
            #scale = np.arange(0,1.1,0.1)
            ax[2].plot(t, dataset[t])
            ax[2].set(xlabel='time', ylabel='cosine_similarity')
            #ax[2].set_yticks(scale)
            ax[2].grid(True)
            #ax[2].axvspan(result_start[0], result_end[0],color='green', alpha=0.2)
    elif(num == 2):
            t = np.arange(result_start[index], result_end[index])
            #scale = np.arange(0, 1.1, 0.1)
            ax[2].plot(t, dataset[t])
            ax[2].set(xlabel='time', ylabel='DTW')
            #ax[2].set_yticks(scale)
            ax[2].grid(True)
            #ax[2].axvspan(result_start[i], result_end[i],color='blue', alpha=0.2)
    else:  
            t = np.arange(result_start[index], result_end[index])
            #scale = np.arange(0, 1.1, 0.1)
            ax[2].plot(t, dataset[t])
            ax[2].set(xlabel='time', ylabel='Flexim')
            #ax[2].set_yticks(scale)
            ax[2].grid(True)
    fig.savefig("compare_{}.png".format(num))
    fig.tight_layout()
    plt.show()
    
