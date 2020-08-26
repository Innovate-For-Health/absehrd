#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Haley Hunter-Zinck
Date: August 11, 2020
"""

from re import search
import numpy as np
import pandas as pd
import feather

class preprocessor:

    def get_file_type(file_name, debug=False):
        
        file_type = ''
        
        if search("\.npy$", file_name):
            file_type = 'npy'
        elif search("\.feather$", file_name):
            file_type = 'feather'
        elif search("\.csv$", file_name):
            file_type = 'csv'
        elif search("\.tsv$", file_name):
            file_type = 'tsv'
        else:
            file_type = None
            
        if debug:
            print(file_type)
            
        return file_type
    
    def get_default_header(n, prefix='C'):
        
        header = np.full(shape=n, fill_value='', dtype='<U'+str(len(str(n-1))+len(prefix)))
        for i in range(n):
            header[i] = prefix + str(i).zfill(len(str(n-1)))
            
        return header
    
    def read_file(self, file_name, skip_header=0, debug=False):
        
        arr = None
        header = []
        
        if(self.get_file_type(file_name) == 'npy'):
            if debug:
                print('Reading npy file ', file_name, '...')
            arr = np.load(file_name)
        
        elif(self.get_file_type(file_name) == 'feather'):
            if debug:
                print('Reading feather file ', file_name, '...')
            df = feather.read_dataframe(file_name)
            arr = df.to_numpy()
        
        elif(self.get_file_type(file_name) == 'csv'):
            if debug:
                print('Reading csv file ', file_name, '...')
            arr = np.genfromtxt(file_name, delimiter=',', skip_header=skip_header)
            if skip_header > 0:
                header = np.genfromtxt(file_name, delimiter=',',max_rows=1, skip_header=0)
        
        elif(self.get_file_type(file_name) == 'tsv'):
            if debug:
                print('Reading tsv file ', file_name, '...')
            arr = np.genfromtxt(file_name, delimiter='\t', skip_header=skip_header)
            if skip_header > 0:
                header = np.genfromtxt(file_name, delimiter=',',max_rows=1, skip_header=0)
                
        else:
            if debug:
                print('Unknown file type for file', file_name)
                print("Returning None")
            return None
        
        if debug:
            print('  Number of samples:', arr.shape[0])
            print('  Number of features:', arr.shape[1])
            
        if len(header) == 0:
            header = self.get_default_header(arr.shape[1])
        
        return {'x':arr, 'header':header}
    
    def is_numeric(x):
        for ele in x:
            try:
                f = float(ele)
            except: 
                return False
        return True
    
    def get_minority_class(x):
        
        vs = np.unique(x)
        v0 = 0
        v1 = 0
        
        for ele in x:
            if(ele == vs[0]):
                v0+=1
            else:
                v1+=1
        
        if(v0 <= v1):
            return(vs[0])
        return(vs[1])
        
    def get_variable_type(self, x, label=None, custom_categorical=(),
                               custom_continuous=(), custom_count=(),
                               max_uniq=10, max_diff=1e-10):
        
        n_uniq = len(set(x))
        
        if label is not None and label in custom_categorical:
            return 'categorical'
        
        if label is not None and label in custom_continuous:
            return 'continuous'
        
        if label is not None and label in custom_count:
            return 'count'
        
        if n_uniq == 1:
            return 'constant'
        
        if n_uniq == 2:
            return 'binary'
        
        if n_uniq > max_uniq and self.is_numeric(x):
            
            for ele in x:
                ele = float(ele)
                if abs(ele - round(ele)) > max_diff:
                    return 'continuous'
            return 'count'
                
        return("categorical")
                
    
    def get_metadata(self, x, header):
        
        names = ['label','type', 'min', 'max', 'zero', 'one']
        formats = ['<U100','<U11','float64', 'float64',str(x.dtype),str(x.dtype)]
        m = np.recarray(shape=x.shape[1], names=names, formats=formats)
        
        for j in range(x.shape[1]):
            
            m[j]['label'] = header[j]
            
            m[j]['type'] = self.get_variable_type(self, x[:,j], label=header[j])
            
            if(m[j]['type'] == 'binary'):
                m[j]['one'] = self.get_minority_class(x[:,j])
                m[j]['zero'] = set(np.unique(x[:,j])).difference([m[j]['one']]).pop()
            
            if(m[j]['type'] == 'constant'):
                m[j]['zero'] = x[0,j]
            
            if(m[j]['type'] == 'continuous' or m[j]['type'] == 'count'):
                m[j]['min'] = np.min(x[:,j].astype(np.float))
                m[j]['max'] = np.max(x[:,j].astype(np.float))
            
        return m
    
    def get_one_hot_encoding(x, label, delim="__"):
                
        u_label = []
        x_hot = []
        
        u_value = np.unique(x)
        u_label = [label + delim] * len(u_value)
        x_hot = np.zeros(shape=(len(x), len(u_value)), dtype=int)
        
        for j in range(len(u_label)):
            u_label[j] += str(u_value[j])
            
            for i in range(len(x)):
                if(x[i] == u_value[j]):
                    x_hot[i,j] = 1
            
            
            
        return {'x':x_hot, 'header':u_label}
    
    def scale(x):
        
        x_min = np.min(x)
        x_max = np.max(x)
        s = np.zeros(len(x))
        
        for i in range(len(x)):
            s[i] = (x[i] - x_min) / (x_max - x_min)
        
        return(s)
    
    def unscale(s, min_value, max_value):
        
        x = np.zeros(len(s))
        
        for i in range(len(s)):
            x[i] = s[i] * (max_value - min_value) + min_value
        return x
    
    def get_discretized_matrix(self, x, m, header, delim='__', debug=False):
        
        d_x = np.empty(shape=0)
        d_header = np.empty(shape=0)
        
        for j in range(x.shape[1]):
            
            c_type = m[j]['type']
            s_j = []
            
            if c_type == 'constant':
                s_j = np.zeros(shape=x.shape[0])
                d_header = np.append(d_header, header[j])
                
            elif c_type == 'continuous' or c_type == 'count':
                x_j = x[:,j].astype(np.float)
                s_j = self.scale(x_j)
                d_header = np.append(d_header, header[j])
                
            elif c_type == "categorical":
                res = self.get_one_hot_encoding(x=x[:,j], label=header[j])
                s_j = res['x']
                d_header = np.append(d_header, res['header'])
                
            elif c_type == "binary":
                s_j = np.zeros(shape=len(x), dtype=int)
                for i in range(len(x)):
                    if x[i,j] == m[j]['one']:
                        s_j[i] = 1
                d_header = np.append(d_header, header[j])
                
            else:
                print("Warning: variable type not recognized.  Appending as is.")
                s_j = x[:,j]
                d_header = np.append(d_header, header[j])
            
            if len(d_x) == 0:
                d_x = s_j
            else:
                d_x = np.column_stack((d_x, s_j))
                
            if debug:
                print('Dimensions of matrix are now', d_x.shape)
        
        
        return {'x':d_x, 'header':d_header}
    
    def unravel_one_hot_encoding(s, header, delim="__"):
        
        values = []
        x = []
       
        for i in range(len(header)):
            values = np.append(values, header[i].split(sep=delim)[1])
            
        for i in range(len(s)):
            
            idx = np.where(s[i,:]==np.max(s[i,:]))[0]
            if len(idx) > 1:
                idx = idx[np.random.randint(low=0,high=len(idx), size=1)]
            x = np.append(x, values[idx])
            
        return x
    
    def restore_matrix(self, s, m, header, delim='__'):
        
        x_prime = []
        variable_names = []
        header_prime = m['label']
       
        for i in range(len(header)):
            variable_names = np.append(variable_names, header[i].split(sep=delim)[0])
        
        for j in range(len(m)):
            
            x_j = []
            idx = np.where(variable_names == m[j]['label'])[0]
            s_j = s[:,idx]
            
            if m[j]['type'] == 'constant':
                x_j = np.full(shape=len(s), fill_value=m[j]['zero'])
            
            elif m[j]['type'] == 'continuous' or m[j]['type'] == 'count':
                min_value = float(m[j]["min"])
                max_value = float(m[j]["max"])
                x_j = self.unscale(s=s_j, min_value=min_value, max_value=max_value)
                
                if m[j]['type'] == 'count':
                  x_j = np.round(x_j)
            
            elif m[j]['type'] == 'categorical':
                x_j = self.unravel_one_hot_encoding(s=s_j, header=header[idx])
            
            elif m[j]['type'] == 'binary':
                
                x_j = np.full(shape=len(s), fill_value=m[j]['zero'])
                
                for i in range(len(s_j)):
                    if s_j[i] == 1:
                        x_j[i] = m[j]['one']
                        
            if len(x_prime) == 0:
                x_prime = x_j
            else: 
                x_prime = np.column_stack((x_prime, x_j))
                
        return {'x':x_prime, 'header':header_prime}
            