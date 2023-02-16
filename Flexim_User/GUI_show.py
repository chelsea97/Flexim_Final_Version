import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from tkinter import filedialog
import os
import numpy as np
from tkinter import *
from sklearn.preprocessing import MinMaxScaler

class GUI_show:
    def __init__(self,dataset,sample_size,query_start,cosine_result_start,dtw_result_start,flexim_result_start):
        self.root = Tk()
        self.scales = []
        # self.label_cosine = IntVar(self.root)
        # self.label_flexim = IntVar(self)
        self.check_button_list = {1:[],2:[],3:[]}
        self.labels = {1:[],2:[],3:[]}
        self.total_number = {"cosine":[],"dtw":[],"flexim":[]}
        self.dataset = dataset
        self.sample_size = sample_size
        self.cosine_result_start = cosine_result_start
        self.dtw_result_start = dtw_result_start
        self.flexim_result_start = flexim_result_start
        self.query_start = query_start
        self.__layout__()
        self.__display__()
        
    def __layout__(self):
        self.root.wm_title("Search Result")
        self.root.columnconfigure(0,weight=1)
        self.root.rowconfigure(0,weight=1)
        self.content = ttk.Frame(self.root,padding=(3,3,12,12))
        self.content.grid(column=0,row = 0,sticky=(N, S, E, W))
        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=3)
        self.content.columnconfigure(2, weight=3)
        self.content.columnconfigure(3, weight=3)
        self.content.columnconfigure(4, weight=3)
        self.content.rowconfigure(1,weight=1)
        self.frame00 = ttk.Frame(self.content,borderwidth = 5,relief = "ridge",width = 500, height = 600)
        self.frame00.grid(column=0,row = 0,columnspan = 3,rowspan=2,sticky=(N, S, E, W))
        #self.frame00.grid(row = 0,column=0,rowspan=3,sticky='NEWS')
        # self.root.columnconfigure(0, weight=4)
        # self.root.rowconfigure(0, weight=4)
        self.__buttons__()
        
    def __display__(self):
        while True:
            try:
                self.root.mainloop()
                break
            except UnicodeDecodeError:
                pass
        
    def __buttons__(self):
        self.submit_button = ttk.Button(self.root,text = 'Submit',command = lambda: self.submit())
        self.submit_button.grid(row = 3, column = 0)
        
        self.show_query_button = ttk.Button(master = self.root,text = 'Search_Query',command=lambda: self._show_query_pattern_())
        self.show_query_button.grid(row = 3, column = 1)
        
        self.show_cosine_button = ttk.Button(master = self.root,text = 'Search_Cosine',command = lambda: self._show_result_cosine_())
        self.show_cosine_button.grid(row = 3, column = 2)
        
        self.show_dtw_button = ttk.Button(master = self.root,text = 'Search_DTW',command = lambda: self._show_result_dtw_())
        self.show_dtw_button.grid(row = 3, column = 3)
        
        self.show_flexim_button = ttk.Button(master = self.root,text='Search_Flexim',command = lambda: self._show_result_flexim_())
        self.show_flexim_button.grid(row = 3, column = 4)
        
        self.show_button = ttk.Button(master = self.root, text = 'Show_Result',command = lambda: self.show_table(),width = 10,style = 'W.TButton')
        self.show_button.grid(row = 3,column = 5)
        style = ttk.Style()
        style.configure('W.TButton',background = 'white',borderwidth = 0,relief = 'flat',highlightcolor ='white')
        self.activate_buttons()
        
            
    def _show_query_pattern_(self):
        fig, ax = plt.subplots(1,1)
        t = np.arange(self.query_start,self.query_start + self.sample_size)
        ax.plot(t,self.dataset[t],linewidth = 2.5,color = 'purple')
        ax.set_ylabel('query pattern',fontsize = 14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True)
        fig.tight_layout()
        plt.subplots_adjust(left=None,bottom=None,right=None,wspace=None,hspace=0.1)
        self.canvas_2 = FigureCanvasTkAgg(fig,master = self.frame00)
        self.canvas_2.draw()
        self.canvas_2.get_tk_widget().grid(row = 0,column = 0,sticky = 'NEWS')
    
    def _show_result_cosine_(self):
        self.names_cosine = locals()
        scale = np.arange(0,len(self.dataset))
        self.cosine_labels = []
        fig,ax = plt.subplots(len(self.cosine_result_start),1)
        #l_cosine = ttk.Scrollbar(fig,orient=VERTICAL)
        for i in range(len(self.cosine_result_start)):
            t = np.arange(self.cosine_result_start[i],self.cosine_result_start[i] + self.sample_size)
            ax[i].plot(t,self.dataset[t],linewidth = 2.5,color = 'red')
            ax[i].set_ylabel("cosine"+str(i),fontsize = 14)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].grid(True)
            self.names_cosine['self.cosine_label'+str(i+1)] = IntVar(self.frame00)
            self.names_cosine['self.check_cosine'+str(i+1)] = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.names_cosine['self.cosine_label'+str(i+1)])
            self.names_cosine['self.check_cosine'+str(i+1)].grid(row = 10+i, column = 5,sticky = 'W')
            
            
            # self.cosine_label_i = IntVar(self.frame00)
            # self.check_cosine = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.cosine_label_i)
            # self.check_cosine.grid(row = 5*i+8,column = 5,sticky = 'W')
            
            
            #self.labels[i+1].append(int(self.get_label().get())) 
        # l_cosine = Scrollbar(self.frame00,orient=VERTICAL)
        # l_cosine.pack(side=RIGHT,fill = Y)  
        fig.tight_layout()
        plt.subplots_adjust(left = None,bottom=None,right = None,top = None, wspace = None, hspace=0.1)
        #fig.config(yscrollcommand = l_cosine.set)
        self.canvas_3 = FigureCanvasTkAgg(fig,master = self.frame00)
        self.canvas_3.draw()
        self.canvas_3.get_tk_widget().grid(row = 10, column = 0,sticky = 'NEWS')
        
    def _show_result_dtw_(self):
        self.names_dtw = locals()
        scale = np.arange(0,len(self.dataset))
        self.dtw_labels = []
        fig,ax = plt.subplots(len(self.dtw_result_start),1)
        for i in range(len(self.dtw_result_start)):
            t = np.arange(self.dtw_result_start[i],self.dtw_result_start[i] + self.sample_size)
            ax[i].plot(t,self.dataset[t],linewidth = 2.5,color = 'green')
            ax[i].set_ylabel("dtw"+str(i),fontsize = 14)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].grid(True)
            self.names_dtw['self.dtw_label'+str(i+1)] = IntVar(self.frame00)
            # # exec(f'self.check_cosine_{i+1} = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.get_label)')
            # # exec(f'self.check_cosine_{i+1}.grid(row = 50,column = {i+20}',sticky = 'NEWS')
            self.names_dtw['self.check_dtw'+str(i+1)] = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.names_dtw['self.dtw_label'+str(i+1)])
            self.names_dtw['self.check_dtw'+str(i+1)].grid(row = 5*i+8, column = 15,sticky = 'NEWS')
            
            # self.dtw_label_i = IntVar(self.frame00)
            # self.check_dtw = ttk.Checkbutton(master=self.frame00,text = "Match?",variable=self.dtw_label_i)
            # self.check_dtw.grid(row = 5*i + 8,column=15,sticky="W")
            
        fig.tight_layout()
        plt.subplots_adjust(left = None,bottom=None,right = None,top = None, wspace = None, hspace=0.1)
        self.canvas_4 = FigureCanvasTkAgg(fig,master = self.frame00)
        self.canvas_4.draw()
        self.canvas_4.get_tk_widget().grid(row = 10, column = 10,sticky = 'NEWS')
    
    def _show_result_flexim_(self):
        self.names_flexim = locals()
        scale = np.arange(0,len(self.dataset))
        self.flexim_labels = []
        fig,ax = plt.subplots(len(self.flexim_result_start),1)
        for i in range(len(self.flexim_result_start)):
            t = np.arange(self.flexim_result_start[i],self.flexim_result_start[i] + self.sample_size)
            ax[i].plot(t,self.dataset[t],linewidth = 2.5,color = 'blue')
            ax[i].set_ylabel("flexim"+str(i),fontsize = 14)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].grid(True)
            self.names_flexim['self.flexim_label'+str(i+1)] = tk.IntVar(self.frame00)
            # # exec(f'self.check_cosine_{i+1} = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.get_label)')
            # # exec(f'self.check_cosine_{i+1}.grid(row = 50,column = {i+20}',sticky = 'NEWS')
            self.names_flexim['self.check_flexim'+str(i+1)] = ttk.Checkbutton(master=self.frame00,text="Match?",variable=self.names_flexim['self.flexim_label'+str(i+1)])
            self.names_flexim['self.check_flexim'+str(i+1)].grid(row = 5*i+8, column = 25,sticky = 'NEWS')
            
            # self.flexim_label_i = IntVar(self.frame00)
            # self.check_flexim = ttk.Checkbutton(master = self.frame00,text = "Match?",variable=self.flexim_label_i)
            # self.check_flexim.grid(row = 5*i + 8,column=25,sticky='NEWS')
        fig.tight_layout()
        plt.subplots_adjust(left = None,bottom=None,right = None,top = None, wspace = None, hspace=0.1)
        self.canvas_5 = FigureCanvasTkAgg(fig,master = self.frame00)
        self.canvas_5.draw()
        self.canvas_5.get_tk_widget().grid(row = 10, column = 20,sticky = 'NEWS')
    
    def get_label(self):
        return self.label
    
    def activate_buttons(self):
        self.submit_button.configure(state = ACTIVE)
        self.show_query_button.configure(state = ACTIVE)
        self.show_cosine_button.configure(state = ACTIVE)
        self.show_dtw_button.configure(state = ACTIVE)
        self.show_flexim_button.configure(state = ACTIVE)
        
        
        
    def submit(self):
        for i in range(len(self.cosine_result_start)):
            self.cosine_labels.append(int(self.names_cosine['self.cosine_label'+str(i+1)].get()))
            self.flexim_labels.append(int(self.names_flexim['self.flexim_label'+str(i+1)].get()))
            self.dtw_labels.append(int(self.names_dtw['self.dtw_label'+str(i+1)].get()))
        print("cosine label is {}".format(self.cosine_labels))
        print("dtw label is {}".format(self.dtw_labels))
        print("flexim label is {}".format(self.flexim_labels))
        self.total_flexim = sum(self.flexim_labels)
        self.total_cosine = sum(self.cosine_labels)
        self.total_dtw = sum(self.dtw_labels)
        self.total_number["cosine"] = self.total_cosine
        self.total_number["dtw"] = self.total_dtw
        self.total_number["flexim"] = self.total_flexim
        self.result_list = [('cosine',self.total_cosine),('dtw',self.total_dtw),('flexim',self.total_flexim)]
        print(self.result_list[2][1])   
        
    def show_table(self):
        for i in range(3):
            for j in range(2):
                self.e = Entry(self.content,width = 20, fg = 'blue',font=('Arial',16,'bold'))
                self.e.grid(row = i+10, column = j+10)
                self.e.insert(END,self.result_list[i][j])
        
        
        
if __name__ == '__main__':
    temp = pd.read_csv("./"+"all_stocks_5yr.csv")
    temp = temp["close"].values
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(temp.reshape(-1,1))
    sample_size = 50
    score_cosine_start_1 = [271650,420520,178050,208595,77005]
    score_dtw_start_1 = [446825,404325,304210,580365,499480]
    score_flexim_start_1 = [46845,76590,395890,395890,76600]
    query_start = 4000
    gui_show = GUI_show(dataset,sample_size,query_start,score_cosine_start_1,score_dtw_start_1,score_flexim_start_1)
    
    
        

        
                
