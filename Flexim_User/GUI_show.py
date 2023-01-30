from thinker import *
import thinker as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from tkinter import ttk
from Functions import Functions as F
import numpy as np
from Core_Test import *
from pattern_search import plot_compare_curve
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class GUI_Result:
    def __init__(self):
        self.root = TK()
        self.core = Core_Test()
        self.scales = []
        self.labels = []
        self.layout__result()
        self.display__result()
    def layout__result(self):
        self.root.wm_title("Time Series Distortion Tool")
        s = ttk.Style()
        s.configure('TFrame',background = 'blue')
        self.frame00 = ttk.Frame(self.root,style = 'TFrame',width = 500,height = 640)
        self.frame00.grid(row = 0, column = 0, rowspan = 3, sticky = 'NEWS')
        self.frame00['borderwidth'] = 2
        self.frame00['relief'] = 'sunken'
        
        self.frame01 = ttk.Frame(self.root,style = 'TFrame',width = 550,height = 50)
        self.frame01.grid(row = 0,column=1,columnspan=2,sticky = 'NEW')
        self.frame01['borderwidth'] = 2
        self.frame01['relief'] = 'sunken'
    def _buttons_(self):
        self.check_bttn = ttk.Button(master=self.frame01,text = 'check result', command = lambda: self.display__result(),state=DISABLED)
        self.check_bttn.pack()
        
        
    def display__result(self,result_start_cosine,result_start_dtw,result_start_flexim,sample_size,type,dataset):
        fig, ax = plt.subplots(len(result_start_flexim,3))
        for i in range(3):
            if i == 0:
                similarity_metrics = "cosine"
                result_start = result_start_cosine
                result_end = result_start + sample_size
            elif i == 1:
                similarity_metrics = "dynamic time wrapping"
                result_start = result_start_dtw
                result_end = result_start_dtw + sample_size
            elif i == 2:
                similarity_metrics = "flexim"
                result_start = result_start_flexim
                result_end = result_start_flexim + sample_size
            for j in range(len(result_start)):
                t = np.arange(result_start[j],result_end[j])
                scale = np.arange(0,1.1,0.1)
                ax[i][j].plot(t,dataset[t])
                ax[i][j].plot(xlabel = 'time',ylabel = similarity_metrics)
                ax[i][j].set_yticks(scale)
                ax[i][j].grid(True)
        fig.savefig("compare_{}.png".format(type))
        fig.tight_layout()
        plt.show()
        canvas = FigureCanvasTkAgg(fig,master = self.root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas,self.root)
        toolbar.update()
        canvas.get_tk_widget().pack()
        
                   
            