from re import search
import numpy as np
import pandas as pd
import feather

class preprocessor:
    
    def __init__(self, missing_value):
         self.missing_value=missing_value

    def get_file_type(self, file_name, debug=False):
        
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
    
    def get_default_header(self, n, prefix='C'):
        
        header = np.full(shape=n, fill_value='', dtype='<U'+str(len(str(n-1))+len(prefix)))
        for i in range(n):
            header[i] = prefix + str(i).zfill(len(str(n-1)))
            
        return header
    
    def read_file(self, file_name, n_header=0, debug=False):
        
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
            header = df.columns
        
        elif(self.get_file_type(file_name) == 'csv'):
            if debug:
                print('Reading csv file ', file_name, '...')
            
            df = pd.read_csv(file_name)
            arr = df.to_numpy()
            if n_header > 0:
                header = df.columns
        
        elif(self.get_file_type(file_name) == 'tsv'):
            if debug:
                print('Reading tsv file ', file_name, '...')
            arr = np.genfromtxt(file_name, delimiter='\t', skip_header=n_header)
            if n_header > 0:
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
    
    def is_numeric(self, x):
        for ele in x:
            try:
                f = float(ele)
            except: 
                return False
        return True
    
    def remove_na(self, x):
        
        if self.is_numeric(x):
            d = x[np.where(x.astype(float) != float(self.missing_value))]
        else:
            d = x[np.where(x != str(self.missing_value))]
        
        return d
    
    def get_minority_class(self, x):
        
        d = self.remove_na(x)
        vs = np.unique(d)
        v0 = 0
        v1 = 0
        
        for ele in d:
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
        
        d = self.remove_na(x)
        n_uniq = len(set(d))
        
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
        
        if n_uniq > max_uniq and self.is_numeric(d):
            
            for ele in d:
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
            
            d = self.remove_na(x[:,j])
            
            m[j]['label'] = header[j]
            
            m[j]['type'] = self.get_variable_type(d, label=header[j])
            
            if(m[j]['type'] == 'binary'):
                m[j]['one'] = self.get_minority_class(d)
                m[j]['zero'] = set(np.unique(d)).difference([m[j]['one']]).pop()
            
            if(m[j]['type'] == 'constant'):
                m[j]['zero'] = np.unique(d)[0]
            
            if(m[j]['type'] == 'continuous' or m[j]['type'] == 'count'):
                m[j]['min'] = np.min(d.astype(np.float))
                m[j]['max'] = np.max(d.astype(np.float))
            
        return m
    
    def get_one_hot_encoding(self, x, label, delim="__"):
                
        u_value = np.unique(x)
        u_label = [label + delim] * len(u_value)
        x_hot = np.zeros(shape=(len(x), len(u_value)), dtype=int)
        
        for j in range(len(u_label)):
            u_label[j] += str(u_value[j])
            
            for i in range(len(x)):
                if(x[i] == u_value[j]):
                    x_hot[i,j] = 1
            
        return {'x':x_hot, 'header':u_label}
    
    def scale(self, x):
        
        d = self.remove_na(x=x)
        x_min = np.min(d)
        x_max = np.max(d)
        s = np.zeros(shape=len(x))
        
        for i in range(len(x)):
            if x[i] != self.missing_value:
                s[i] = (x[i] - x_min) / (x_max - x_min)
            else:
                s[i] = self.missing_value
        
        return s
    
    def unscale(self, s, min_value, max_value):
        
        x = np.zeros(len(s))
        
        for i in range(len(s)):
            x[i] = s[i] * (max_value - min_value) + min_value
        return x
    
    def get_missing_column(self, x):
        
        y = np.zeros(shape=x.shape)
        idx = np.array([], dtype=int)
        
        for i in range(len(x)):
            if x[i] == self.missing_value or x[i] == str(self.missing_value):
                idx = np.append(idx, i)
            
        y[idx] = 1
        
        return y
    
    def get_discretized_matrix(self, x, m, header, delim='__', debug=False):
        
        d_x = np.empty(shape=0)
        d_header = np.empty(shape=0)
        
        for j in range(x.shape[1]):
            
            c_type = m[j]['type']
            x_j = x[:,j]
            s_j = []
            
            contains_missing = len(self.remove_na(x_j)) < len(x_j)
            
            if c_type == 'constant':
                
                if contains_missing:
                    s_j = np.column_stack((np.zeros(shape=x.shape[0]),
                                           self.get_missing_column(x_j)))
                    d_header = np.append(d_header, 
                                         (header[j]+delim+header[j], 
                                          header[j]+delim+str(self.missing_value)))
                else:
                    s_j = np.zeros(shape=x.shape[0])
                    d_header = np.append(d_header, header[j])
                
            elif c_type == 'continuous' or c_type == 'count':
                
                x_j = x_j.astype(np.float)
                s_j = self.scale(x_j)
                
                if contains_missing:
                    idx = np.where(s_j == self.missing_value)[0]
                    s_j[idx] = np.random.uniform(low=0, high=1, size=len(idx))
                    s_j = np.column_stack((s_j,
                                       self.get_missing_column(x_j)))
                    d_header = np.append(d_header, 
                                         (header[j]+delim+header[j], 
                                          header[j]+delim+str(self.missing_value)))
                else:
                    d_header = np.append(d_header, header[j])
                
            elif c_type == 'categorical':
                
                res = self.get_one_hot_encoding(x=x_j, label=header[j])
                s_j = res['x']
                d_header = np.append(d_header, res['header'])
                
            elif c_type == 'binary':
                
                s_j = np.zeros(shape=len(x_j), dtype=int)
                for i in range(len(x)):
                    if x[i,j] == m[j]['one']:
                        s_j[i] = 1
                
                if contains_missing:
                    f_j = s_j
                    idx = np.where(x_j == str(self.missing_value))[0]
                    f_j[idx] = np.random.randint(low=0,high=2,size=len(idx))
                    s_j = np.column_stack((f_j, self.get_missing_column(x_j)))
                    d_header = np.append(d_header, 
                                         (header[j]+delim+header[j], 
                                          header[j]+delim+str(self.missing_value)))
                else:
                    d_header = np.append(d_header, header[j])
                
            else:
                print("Warning: variable type not recognized.  Appending as is.")
                s_j = x_j
                d_header = np.append(d_header, header[j])
            
            if len(d_x) == 0:
                d_x = s_j
            else:
                d_x = np.column_stack((d_x, s_j))
                
            if debug:
                print('Dimensions of matrix are now', d_x.shape)
        
        
        return {'x':d_x, 'header':d_header}
    
    def unravel_one_hot_encoding(self, s, header, delim="__"):
        
        values = []
        x = []
       
        for i in range(len(header)):
            values = np.append(values, header[i].split(sep=delim)[1])
            
        for i in range(len(s)):
            
            s_i = s[i,:].astype(float)
            
            idx = np.where(s_i==np.max(s_i))[0]
            if len(idx) > 1:
                idx = idx[np.random.randint(low=0,high=len(idx), size=1)]
            x = np.append(x, values[idx])
            
        return x
    
    def restore_matrix(self, s, m, header, delim='__'):
        
        x_prime = []
        variable_names = []
        variable_values = []
        header_prime = m['label']
       
        for i in range(len(header)):
            
            splt = header[i].split(sep=delim)
            variable_names = np.append(variable_names, splt[0])
            
            if len(splt) > 1:
                variable_values = np.append(variable_values, splt[1])
            else:
                variable_values = np.append(variable_values, '')
        
        for j in range(len(m)):
            
            x_j = []
            idx_missing = []
            idx_col = np.where(variable_names == m[j]['label'])[0]
            
            if m[j]['type'] != 'categorical' and len(idx_col) > 1:
                for k in np.where(variable_names == m[j]['label'])[0]:
                    if variable_values[k] == str(self.missing_value):
                        idx_missing = np.where(s[:,k] == 1)[0]
                    else:
                        idx_col = k
                        
            s_j = s[:,idx_col]
            
            if m[j]['type'] == 'constant':
                x_j = np.full(shape=len(s), fill_value=m[j]['zero'])
                x_j[idx_missing] = self.missing_value
            
            elif m[j]['type'] == 'continuous' or m[j]['type'] == 'count':
                min_value = float(m[j]["min"])
                max_value = float(m[j]["max"])
                x_j = self.unscale(s=s_j.astype('float'), min_value=min_value, max_value=max_value)
                
                if m[j]['type'] == 'count':
                  x_j = np.round(x_j)
                  
                x_j[idx_missing] = self.missing_value
            
            elif m[j]['type'] == 'categorical':
                x_j = self.unravel_one_hot_encoding(s=s_j, header=header[idx_col])
            
            elif m[j]['type'] == 'binary':
                
                x_j = np.full(shape=len(s), fill_value=m[j]['zero'], dtype='O')
                
                for i in range(len(s_j)):
                    if s_j[i] == 1:
                        x_j[i] = m[j]['one']
                        
                x_j[idx_missing] = self.missing_value
                        
            if len(x_prime) == 0:
                x_prime = x_j
            else: 
                x_prime = np.column_stack((x_prime, x_j))
                
        return {'x':x_prime, 'header':header_prime}
            