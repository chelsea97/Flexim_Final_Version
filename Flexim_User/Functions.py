import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class Functions:
    def __init__(self, core, params):
        self.core   = core
        self.params = params

    def func1(self, original=None, alpha=None):
        def translate(org_alpha):
            real_alpha = org_alpha # if conversion is necessary then translate it.
        name = 'Smoothen/Add Noise'
        alpha_range = (-100,self.params['sample_size']-1)
        transformed = None
        if original is not None:
            original = original.reshape((-1,))
            if alpha > 2:
                transformed = savgol_filter(original, alpha if alpha % 2 else alpha + 1, 2)#(alpha * np.arange(0,original.shape[0]))
            elif alpha < 0:
                transformed = original + np.random.normal(0, -alpha/1000, original.shape[0])#.reshape(original.shape)
            else:
                transformed = original
            transformed = transformed.reshape((-1, 1))
            assert transformed.shape[1] == 1
        return transformed, alpha_range, name


    #@staticmethod
    def func2(self, original=None,alpha=None):
        name = 'Min/Max_cap'
        result = None
        if original is not None:
            if alpha > 0:
                result = np.minimum(original, original.max()-(alpha/100))# + ((alpha/1000) * np.arange(0,original.shape[0]).reshape((-1,1)))
            elif alpha < 0:
                result = np.maximum(original, original.min()+(abs(alpha)/100))
            else:
                result = original
            assert result.shape[1] == 1
        return result, (-100, 100), name
    '''
    #@staticmethod
    def func3(self, original=None, alpha=None):
        name = 'Min_cap'
        alpha_range = (-100,100)
        result = None
        if original is not None:
            result = np.maximum(original, original.min()+(alpha/100))# + ((alpha/1000) * np.arange(0,original.shape[0]).reshape((-1,1)))
            assert result.shape[1] == 1
        return result, alpha_range, name
    '''
    '''
    @staticmethod
    def func3(original=None, alpha=None):
        name = 'Shift'
        alpha_range = (0,50)
        transformed = None
        if original is not None:
            transformed = original
        return transformed, alpha_range, name
    '''
    '''
    @staticmethod
    def func4(original=None, alpha=None):
        name = 'Noise'
        alpha_range = (0,100)
        transformed = None
        if original is not None:
            #np.random.normal(0, alpha/1000, original.shape[0])
            transformed = original + np.random.normal(0, alpha/1000, original.shape[0]).reshape(original.shape)
        return transformed, alpha_range, name
    '''

