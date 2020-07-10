# -*- coding: utf-8 -*-
"""
Description: synthetic eletronic health record data (SEHRD)
Author: Haley Hunter-Zinck
Date: June 23, 2020
"""

from medgan_reimp import medgan
#from corgan import corgan
#from medwgan import medwgan
#from ppgan import ppgan

class sehrd(object):
    
    def __init__(self):
        self.label_medgan = 'medgan'
         
    """
    def lookup():
        print(label_medgan)
    """
    
    def train(self, x, method, n_epoch, batch_size):
        if method == self.label_medgan:
            obj = medgan(data_type='binary', input_dim=x.shape[1])
        else:
            print('Method ' + method + ' not recognized.  Returning None.')
            return None
            
        obj.train(x=x, n_epoch=n_epoch, batch_size=batch_size)
        
        return obj
    
    
    def generate(self, obj, n_sample):
        if n_sample <= 0:
            print('Number of samples requested (n) must be greater than 0.  Returning None.')
            return None
        
        synth_x = obj.generate(n_sample)
        
        return(synth_x)
