from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from tkinter import filedialog
import os
from tkinter import ttk
from Functions import Functions as F
import pandas as pd
import numpy as np
from Core_Test import *
import threading
from sklearn.preprocessing import MinMaxScaler


class GUI:
    def __init__(self):
        self.ROUNDING_CONST = 3
        self.root           = Tk()
        self.core           = Core_Test(self)
        self.scales         = []
        self.labels         = []
        self.__layout__()
        self.__display__()
    def __layout__(self):
        #self.root.geometry('1050x740+300+50')
        self.root.wm_title("Time Series Distortion Tool")
        #self.sample_size = StringVar()
        #self.root.register(self.validate())
        s = ttk.Style()
        s.configure('TFrame', background='blue')
        self.frame00 = ttk.Frame(self.root, style='TFrame', width=500, height=640); #self.frame00.place(x=0, y=0, width=500, height=770)
        self.frame00.grid(row=0, column=0, rowspan=3, sticky='NEWS')
        self.frame00['borderwidth'] = 2
        self.frame00['relief']      = 'sunken'

        '''
        self.frame2 = ttk.Frame(self.root, style='TFrame', width=450, height=750); #self.frame2.place(x=500, y=0, width=450, height=750)
        self.frame2.grid(row=2, column=1, sticky='WNS')
        self.frame2['borderwidth'] = 2
        self.frame2['relief']      = 'sunken'
        '''
        self.frame01 = ttk.Frame(self.root, style='TFrame', width=550, height=50); #self.frame01.place(x=500, y=585, width=550, height=200)
        self.frame01.grid(row=0, column=1, columnspan=2, sticky='NEWS')
        self.frame01['borderwidth'] = 2
        self.frame01['relief']      = 'sunken'

        self.frame02 = ttk.Frame(self.root, style='TFrame', width=500, height=150)
        self.frame02.grid(row=3, column=0, sticky='NEWS')
        self.frame02['relief'] = 'sunken'

        self.frame11 = ttk.Frame(self.root, style='TFrame', width=225, height=600); #self.frame11.place(x=500, y=85, width=225, height=500)
        self.frame11.grid(row=1, column=1, rowspan=1, sticky='NEWS')
        self.frame11['borderwidth'] = 2
        self.frame11['relief']      = 'sunken'

        self.frame12 = ttk.Frame(self.root, style='TFrame', width=225, height=600); #self.frame12.place(x=725, y=85, width=225, height=500)
        self.frame12.grid(row=1, column=2, rowspan=1,sticky='NEWS')
        self.frame12['borderwidth'] = 2
        self.frame12['relief']      = 'sunken'

        self.frame22 = ttk.Frame(self.root, style='TFrame', width=450, height=150); #self.frame12.place(x=725, y=85, width=225, height=500)
        self.frame22.grid(row=3, column=1, rowspan=2,sticky='NEWS')
        self.frame22['borderwidth'] = 2
        self.frame22['relief']      = 'sunken'

        self.root.columnconfigure(0, weight=4)
        self.root.rowconfigure(0, weight=4)
        for i in range(1,3):
            self.root.columnconfigure(i, weight=1)
            self.root.rowconfigure(i, weight=1)
        '''
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(1, weight=1)
        '''
        figure = plt.Figure(figsize=(5,5),facecolor='#ececec')
        self.canvas = FigureCanvasTkAgg(figure, self.root)#self.frame00)
        #self.canvas.get_tk_widget().place(x=0,y=0,width=495,height=640)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, columnspan=1, sticky='NEWS')
        self.ax = [figure.add_subplot(2,1,i) for i in range(1,3)]
        self.ax[0].tick_params(axis='x', labelsize=8)
        self.ax[0].tick_params(axis='y', labelsize=8)
        self.ax[1].tick_params(axis='x', labelsize=8)
        self.ax[1].tick_params(axis='y', labelsize=8)
        #self.sample_size.set('50')
        self.__buttons__()
        #self.__slides__()

    def __buttons__(self):
        for i in range(3):
            self.frame01.columnconfigure(i, weight=2)
            #self.frame01.rowconfigure(i, weight=1)
        self.load_bttn = ttk.Button(master=self.frame01, text='Load Data', command= lambda: self.load_bttn_handler())
        #self.load_bttn.pack()


        self.load_bttn.grid(row=0, column=1, padx=225, columnspan=1, sticky='NEWS')
        self.sample_bttn = ttk.Button(master=self.frame01, text='Start', command= lambda: self.core.sample(),
                                      state=DISABLED)
        #self.sample_bttn.pack()
        self.sample_bttn.grid(row=1, column=1, padx=225, columnspan=1, sticky='NEWS')
        self.win_label = ttk.Label(self.frame01, text="Width:")
        #self.win_label.place(x=180, y=55)
        self.win_label.grid(row=2, column=1, padx=225, sticky='NEWS')
        #self.sample_txt    = ttk.Entry(master=self.frame01, textvariable=self.core.sample_size, width=4, validate='all',
        #                               validatecommand=self.validate)
        #self.sample_txt.place(x=225, y=54)
        #self.sample_txt.grid(row=3, column=1, padx=225, sticky='NW')
        style = ttk.Style()

        style.configure('W.TButton', background='white', borderwidth=0, relief='flat', highlightcolor='white')
        #ttk.Label(master=self.frame02, text="   ").pack(side=BOTTOM)
        #ttk.Label(master=self.frame02, text="   ").pack(side=BOTTOM)

        for i in range(3):
            self.frame02.columnconfigure(i, weight=2)

        self.check = ttk.Checkbutton(master=self.frame02, text="Match?", variable=self.core.get_label())
        #self.check.pack(side=BOTTOM)


        self.save_bttn = ttk.Button(master=self.frame02, text='Save All', command=lambda: self.core.save(),
                                   width=10, style='W.TButton', state=DISABLED)
        self.submit_bttn = ttk.Button(master=self.frame02, text='Submit', command=lambda: self.core.submit(),
                                      width=10, style='W.TButton', state=DISABLED)
        #self.submit_bttn.pack(side=BOTTOM)
        self.check.grid(row=0, column=1, padx=250, columnspan=1, sticky='NEWS')
        self.submit_bttn.grid(row=1, column=1, padx=250, columnspan=1, sticky='NEWS')
        self.save_bttn.grid(row=2, column=1, padx=250, columnspan=1, sticky='NEWS')
        #self.save_bttn.pack(side=BOTTOM)




    def __slides__(self):
        for s in self.scales:
            s.pack_forget() 
        for l in self.labels:
            l.pack_forget()
        funcs = self.core.get_non_linears()
        #Nonlinear transformations parameters
        label = ttk.Label(master=self.frame11, text='N/linear Transformation Parameters')
        label.pack()
        self.labels.append(label)

        for e in funcs:
            r ,l, v = e
            spacer = ttk.Label(master=self.frame11, text="   ")
            self.labels.append(spacer)
            spacer.pack()
            slider = tk.Scale(master=self.frame11, orient=HORIZONTAL, command=lambda x: self.core.action(), label=l,
                              from_=r[0], to=r[1], length=200, troughcolor='#ffcccb', relief='sunken',
                              activebackground='red', variable=v, state=DISABLED, resolution=0.001)
            self.scales.append(slider)
            slider.pack()
        #Linear transformations parameters
        label = ttk.Label(master=self.frame12, text='Linear Transformation Parameters')
        self.labels.append(label)
        label.pack()
        parameters = self.core.get_linears()

        for p in parameters:
            r ,l, v = p
            spacer = ttk.Label(master=self.frame12, text="   ")
            self.labels.append(spacer)
            spacer.pack()
            slider = tk.Scale(master=self.frame12, orient=HORIZONTAL, command=lambda x: self.core.action(), label=l,
                              length=200, relief='sunken', troughcolor='#add8e6', activebackground='blue', variable=v,
                              from_=r[0], to=r[1], state=DISABLED, resolution=0.001)
            self.scales.append(slider)
            slider.pack()
            
    def activate_data_generator(self,instance_number):
        s = ttk.Style()
        s.configure("TProgressbar", thickness=100)
        self.pb = ttk.Progressbar(
            self.frame01,
            orient='horizontal',
            mode='determinate',
            length=280,
            style="TProgressbar"
        )
        self.pb.place(x=85, y=85)
        self.core.gen_data_naive_points(self.root,self.pb,instance_number)
        
    def activate_data_generator_unlabel(self,instance_number):
        s = ttk.Style()
        s.configure("TProgressbar", thickness=100)
        self.pb = ttk.Progressbar(
            self.frame01,
            orient='horizontal',
            mode='determinate',
            length=280,
            style="TProgressbar"
        )
        self.pb.place(x=85, y=85)
        self.core.gen_data_unlabel_complex_points(
            self.root, self.pb, instance_number)
           
    def deactivate_buttons(self):
        self.sample_bttn.configure(state=DISABLED)
        self.submit_bttn.configure(state=DISABLED)
        self.save_bttn.configure(state=DISABLED)

    def activate_buttons(self):
        self.sample_bttn.configure(state=ACTIVE)
        self.submit_bttn.configure(state=ACTIVE)

    def activate_scales(self):
        self.save_bttn.configure(state=ACTIVE)
        #for s in self.scales:
        #    s.configure(state=ACTIVE)

    def plot_data(self, original):
        self.ax[0].clear()
        self.ax[0].plot(original)
        #self.ax[0].set_ylim([0,1])
        self.canvas.draw()
    def plot_sample(self, original, transform):
        self.ax[1].clear()
        self.ax[1].plot(original, ls='--')
        #self.ax[1].set_ylim([-1.5,1.5])
        self.ax[1].set_xlim([-self.core.sample_size, 2*self.core.sample_size])
        if transform is not None:
            #print(transform)
            if transform.shape[1] == 0:
                self.ax[1].plot(transform)
            else:
                change_transform = self.core.retransform_2(transform)
                self.ax[1].plot(change_transform[:,0],change_transform[:,1],color = 'green')
                print("plot change transform")
                self.ax[1].plot(transform[:, 0],transform[:, 1], color='orange')
                print("plot tr")
        self.ax[1].axvspan(self.core.sample_size, 1000, alpha=0.2, hatch='x', snap=True, color='black')
        self.ax[1].axvspan(-1, -1000, alpha=0.2, hatch='x', color='black')
        self.canvas.draw()
        
    def plot_random_sample(self):
        self.ax[1].clear()
        self.ax[1].set_ylim([-2, 2])
        self.ax[1].set_xlim([-self.core.sample_size, 2*self.core.sample_size])
        random_sample = np.random.rand(self.core.sample_size,2)
        for i in range(len(random_sample)):
            random_sample[i,0] = i
            random_sample[i,1] = np.random.uniform(-0.5,0.5)
        self.ax[1].plot(random_sample[:,0],random_sample[:,1],color = 'blue')
        self.ax[1].axvspan(self.core.sample_size, 1000,
                           alpha=0.2, hatch='x', snap=True, color='black')
        self.ax[1].axvspan(-1, -1000, alpha=0.2, hatch='x', color='black')
        self.canvas.draw()
            
        

    def draw_sample_window(self, i, j):
        self.ax[0].axvspan(i, j, color='red', alpha=0.5)
    '''
    def validate(self):
        num = self.core.window_size#self.sample_size.get()
        if len(num) == 0:
            return True
        try:
            int(num)
            return True
        except:
            self.sample_size.set(num[0:-1])
            return False
    '''
    def __display__(self):
        while True:
            try:
                self.root.mainloop()
                break
            except UnicodeDecodeError:
                pass
    def load_bttn_handler(self):
        self.core.__restart_fetch__()
        self.core.file_opener()
        '''
        self.file_opener()
        self.sample_bttn.configure(state=ACTIVE)
        self.do_plot(self.data, None, 0)
        '''
    def file_opener_old(self):
        path = filedialog.askopenfile(initialdir=os.getcwd() + "/")
        temp = pd.read_csv(path.name, index_col=0)
        # mn   = temp.min()
        # mx   = temp.max()
        # self.data = (temp-mn)/(mx-mn)
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(temp)

    def do_plot(self, original, transformed, ind):
        if original is None:
            return
        self.ax[ind].clear()
        self.ax[ind].plot(original)
        #self.ax[ind].set_ylim([0,1.1])
        if transformed is not None:
            self.ax[ind].plot(transformed)
        self.canvas.draw()

gui = GUI()
