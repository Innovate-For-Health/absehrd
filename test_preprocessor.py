"""
Description: automated tests for SEHRD functions.
Author: Haley Hunter-Zinck
Date: August 11, 2020
use %reset in terminal to clear all variables
"""

from preprocessor import preprocessor
import feather
import numpy as np
import pandas as pd
import os
from re import search

class tester_pre(object):

    def test_get_file_type():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        rec_names = ['test.npy', 'test.feather', 'test.csv', 'test.tsv',
                     'hello..csv']
        unr_names = ['test.pdf', 'test.npy.pdf', 'feather.myfeather', 
                     'mycsv.txt', 'test.tsv.tsv.not', 'test.csvcsv']
        
        for file_name in rec_names:
            if pre.get_file_type(file_name, debug=False) is None:
                return False
            
        for file_name in unr_names:
            if pre.get_file_type(file_name, debug=False) is not None:
                return False
            
        return True
    
    def test_read_file():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        arr = np.random.randint(low=0,high=2, size=(10,2))
        
        # npy
        file_name = 'test.npy'
        np.save(file_name, arr)
        res = pre.read_file(file_name, debug=False)
        os.remove(file_name)
        if res is None:
            return False
        
        # feather
        file_name = 'test.feather'
        feather.write_dataframe(pd.DataFrame(arr), file_name)
        res = pre.read_file(file_name, debug=False)
        os.remove(file_name)
        if res is None:
            return False
        
        # csv
        file_name = 'test.csv'
        np.savetxt(file_name, arr, delimiter=',')
        res = pre.read_file(file_name, debug=False)
        os.remove(file_name)
        if res is None:
            return False
        
        # tsv
        file_name = 'test.tsv'
        np.savetxt(file_name, arr, delimiter='\t')
        res = pre.read_file(file_name, debug=False)
        os.remove(file_name)
        if res is None:
            return False
        
        # bad file
        file_name = 'test.txt'
        np.savetxt(file_name, arr, delimiter=' ')
        res = pre.read_file(file_name, debug=False)
        os.remove(file_name)
        if res is not None:
            return False
        
        return True
    
    def test_is_numeric():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        x = [1,5,2,5,5.5,7.2]
        if not pre.is_numeric(x):
            return False
        
        y = ['A','B','5','10.2']
        if pre.is_numeric(y):
            return False
        
        z = ['1.2', '3.3', '5.0']
        if not pre.is_numeric(z):
            return False
        
        return True
    
    def test_remove_na():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        v1 = np.array(['hello', missing_value, 'world'])
        v2 = np.array([3,2,5,missing_value,5,2,1])
        
        d1 = pre.remove_na(v1)
        d2 = pre.remove_na(v2)
        
        g1 = np.array(['hello', 'world'])
        g2 = np.array([3,2,5,5,2,1])
    
        if len(d1) != len(g1):
            return False
        
        if len(d2) != len(g2):
            return False
        
        for i in range(len(g1)):
            if d1[i] != g1[i]:
                return False
            
        for i in range(len(g2)):
            if d2[i] != g2[i]:
                return False
        
        return True
    
    def test_get_variable_type():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        # count
        x = np.random.randint(low=0,high=11,size=1000)
        if pre.get_variable_type(x=x, label='my_feature') != 'count':
            return False
        
        # categorical
        x = np.random.choice(['hello','world','oi','terra','hi','goodbye','tchau'],1000)
        if pre.get_variable_type(x=x, label='my_feature') != 'categorical':
            return False
        
        # binary
        x = np.random.randint(low=0,high=2,size=1000)
        if pre.get_variable_type(x=x, label='my_feature') != 'binary':
            return False
        
        # constant
        x = np.random.randint(low=0,high=1,size=1000)
        if pre.get_variable_type(x=x, label='my_feature') != 'constant':
            return False
        
        # continuous
        # binary
        x = np.random.random(1000)
        if pre.get_variable_type(x=x, label='my_feature') != 'continuous':
            return False
        
        return True
    
    def test_get_minority_class():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n0 = 20
        n1 = 17
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        if(pre.get_minority_class(x) != choices[1]):
            return False
        
        n0 = 20
        n1 = 20
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        if(pre.get_minority_class(x) != choices[0]):
            return False

        n0 = 4
        n1 = 40
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        if(pre.get_minority_class(x) != choices[0]):
            return False

        return True
    
    def test_get_metadata():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        threshold = 1e-5
        
        names = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        formats = ['<U5']*len(names)
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        
        g_type = ['constant','binary','binary','categorical','count','continuous']
        if(m['type'].tolist() != g_type):
            return False
        
        g_min = [0,0,0,0,count_min,np.min(v_continuous)]
        for i,ele in enumerate(g_min):
            if abs(ele - m[i]['min']) > threshold:
                return False
            
        g_max = [0,0,0,0,count_max,np.max(v_continuous)]
        for i,ele in enumerate(g_max):
            if abs(ele - m[i]['max']) > threshold:
                return False
            
        g_zero = [constant_value,'0',binary_A,'','','']
        if m['zero'].tolist() != g_zero:
            return False
            
        g_one = ['','1',binary_B,'','','']
        if m['one'].tolist() != g_one:
            return False
        
        return True
    
    def test_get_default_header():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        g5 = ['C0','C1','C2','C3','C4']
        h5 = pre.get_default_header(n=5, prefix='C')
        if h5.tolist() != g5:
            return False
        
        g10 = ['col0','col1','col2','col3','col4','col5','col6','col7','col8',
               'col9']
        h10 = pre.get_default_header(n=10, prefix='col')
        if h10.tolist() != g10:
            return False
        
        g11 = ['col00','col01','col02','col03','col04','col05','col06','col07',
               'col08','col09', 'col10']
        h11 = pre.get_default_header(n=10, prefix='col')
        if h10.tolist() != g10:
            return False
        
        return True
    
    def test_get_one_hot_encoding():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        delim = "___"
        label="var"
        
        x = np.array(['Y','X','X','Z','Y'])
        res = pre.get_one_hot_encoding(x=x, label=label, delim=delim)
        g_header = [label+delim+'X', label+delim+'Y', label+delim+'Z']
        g_x_hot = np.array([(0,1,0),(1,0,0),(1,0,0),(0,0,1),(0,1,0)]) 
        
        if res['header'] != g_header:
            return False
        
        if res['x'].tolist() != g_x_hot.tolist():
            return False
        
        x = np.array(['Y','X','X',missing_value,'Y'])
        res = pre.get_one_hot_encoding(x=x, label=label, delim=delim)
        g_header = [label+delim+str(missing_value), label+delim+'X', label+delim+'Y']
        g_x_hot = np.array([(0,0,1),(0,1,0),(0,1,0),(1,0,0),(0,0,1)]) 
        
        if res['header'] != g_header:
            return False
        
        if res['x'].tolist() != g_x_hot.tolist():
            return False
        
        return True
    
    def test_scale():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        x = np.array([0,1,2,3,4,5,6,7,8,9,10])
        g_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        s = pre.scale(x)
        
        for i in range(len(s)):
            if s[i] != g_s[i]:
                return False
        
        return True
    
    def test_get_missing_column():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        x1 = np.array([1,2,3,missing_value, 4])
        g1 = np.array([0,0,0,1,0])
        c1 = pre.get_missing_column(x1)
        
        for i in range(len(x1)):
            if g1[i] != c1[i]:
                return False
            
        x2 = np.array([1,2,3,missing_value, 4])
        g2 = np.array([0,0,0,1,0])
        c2 = pre.get_missing_column(x2)
        
        for i in range(len(x2)):
            if g2[i] != c2[i]:
                return False
            
        x3 = np.array(['hello',missing_value, 'world'])
        g3 = np.array([0,1,0])
        c3 = pre.get_missing_column(x3)
        
        for i in range(len(x3)):
            if g3[i] != c3[i]:
                return False
        
        return True
    
    def test_get_discretized_matrix():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        
        names = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        
        res = pre.get_discretized_matrix(x, m, names)
        n_col = (len(names) - 1) + len(categorical_values) 
        
        if len(res['header']) != n_col:
            return False

        if res['x'].shape[1] != n_col:
            return False
        
        return True
    
    def test_get_discretized_matrix_missing():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        n_missing = 7
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        
        names = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                 
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        for j in range(x.shape[1]):
            idx = np.random.randint(low=0, high=n, size=n_missing)
            x[idx,j] = missing_value
        x = x.astype('O')
        
        m = pre.get_metadata(x, header=names)
        
        res = pre.get_discretized_matrix(x, m, names)
        n_col = 2 * (len(names) - 1) + (len(categorical_values) + 1)
        
        if(len(res['header']) != n_col):
            return False

        if(res['x'].shape[1] != n_col):
            return False
               
        return True
    
    def test_unscale():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        min_value=0
        max_value=10
        s = [0,0.1,0.2,0.3,0.4,1]
        g = [0, 1, 2, 3, 4, 10]
        x = pre.unscale(s, min_value, max_value)
        
        for i in range(len(s)):
            if x[i] != g[i]:
                return False
        
        return True
    
    def test_unravel_one_hot_encoding():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        s = np.array([[0,0,1], [0,0,1], [0,1,0], [1,0,0], [0,1,0]])
        g = ['var3','var3','var2','var1','var2']
        header = ['col__var1', 'col__var2', 'col__var3']
        x = pre.unravel_one_hot_encoding(s, header)
        
        for i in range(len(x)):
            if x[i] != g[i]:
                return False
        
        return True
    
    def test_restore_matrix():
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        threshold = 1e-5
        
        header = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        formats = ['<U5']*len(header)
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        m = pre.get_metadata(x, header)
        d = pre.get_discretized_matrix(x, m, header)
        r = pre.restore_matrix(d['x'], m, d['header'])
        
        # check header
        for i in range(len(header)):
            if r['header'][i] != header[i]:
                return False
        
        # check matrix values
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if m[j]['type'] == 'count' or m[j]['type'] == 'continuous':
                    if abs(float(x[i,j]) - float(r['x'][i,j])) > threshold:
                        return False
                else:
                    if x[i,j] != r['x'][i,j]:
                        return False
        
        return True
    
    def test_restore_matrix_missing():
        
        # numeric missing value
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        n_missing = 7
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        threshold = 1e-5
        
        header = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                 
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        for j in range(x.shape[1]):
            idx = np.random.randint(low=0, high=n, size=n_missing)
            x[idx,j] = missing_value
        x = x.astype('O')
        
        m = pre.get_metadata(x, header)
        d = pre.get_discretized_matrix(x, m, header)
        r = pre.restore_matrix(d['x'], m, d['header'])
        
        # check header
        for i in range(len(header)):
            if r['header'][i] != header[i]:
                return False
        
        # check matrix values
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if m[j]['type'] == 'count' or m[j]['type'] == 'continuous':
                    if abs(float(x[i,j]) - float(r['x'][i,j])) > threshold:
                        return False
                else:
                    if x[i,j] != r['x'][i,j] and x[i,j] != str(r['x'][i,j]):
                        return False
                    
                    
        # string missing value
                    
        missing_value = 'NULL'
        pre = preprocessor(missing_value=missing_value)
        
        n = 1000
        n_missing = 7
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        threshold = 1e-5
        
        header = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                 
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        for j in range(x.shape[1]):
            idx = np.random.randint(low=0, high=n, size=n_missing)
            x[idx,j] = missing_value
        x = x.astype('O')
        
        m = pre.get_metadata(x, header)
        d = pre.get_discretized_matrix(x, m, header)
        r = pre.restore_matrix(d['x'], m, d['header'])
        
        # check header
        for i in range(len(header)):
            if r['header'][i] != header[i]:
                return False
        
        # check matrix values
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if m[j]['type'] == 'count' or m[j]['type'] == 'continuous':
                    
                    if str(x[i,j]) == missing_value and str(r['x'][i,j]) != missing_value:
                        print('i=',i,' j=',j)
                        return False
                    if str(x[i,j]) != missing_value and str(r['x'][i,j]) == missing_value:
                        print('i=',i,' j=',j)
                        return False
                    if str(x[i,j]) != missing_value and abs(float(x[i,j]) - float(r['x'][i,j])) > threshold:
                        print('i=',i,' j=',j)
                        return False
                else:
                    if x[i,j] != r['x'][i,j] and x[i,j] != str(r['x'][i,j]):
                        print('i=',i,' j=',j)
                        return False
                        
        
        return True
    
    def test_is_iterable():
        
        pre = preprocessor(missing_value='NULL')
        
        if pre.is_iterable(3):
            return False
        
        if not pre.is_iterable(np.array([1,2,3,4,5])):
            return False
        
        return True


    def test_get_na_idx():
        
        missing_value = 'NULL'
        pre = preprocessor(missing_value=missing_value)
        
        x = np.array([4,5,2,5,3])
        g_idx = np.array([])
        r_idx = pre.get_na_idx(x)
        g_idx.sort()
        r_idx.sort()
        
        if len(g_idx) != len(r_idx):
            return False
        
        for i in range(len(g_idx)):
            if g_idx[i] != r_idx[i]:
                return False
            
        
        x = np.array([4, 5, 2, 5, missing_value, 3])
        g_idx = np.array([4])
        r_idx = pre.get_na_idx(x)
        g_idx.sort()
        r_idx.sort()
        
        if len(g_idx) != len(r_idx):
            return False
        
        for i in range(len(g_idx)):
            if g_idx[i] != r_idx[i]:
                return False
            
            
        x = np.array(['hello','world','earth','tierra',missing_value,'blah','foo',missing_value])
        g_idx = np.array([4,7])
        r_idx = pre.get_na_idx(x)
        g_idx.sort()
        r_idx.sort()
        
        if len(g_idx) != len(r_idx):
            return False
        
        for i in range(len(g_idx)):
            if g_idx[i] != r_idx[i]:
                return False

        
        return True
    
def main():
    
    buffer = "\t\t\t\t\t"
    
    print('Testing preprocessor.get_file_type()', end='...\t'+buffer)
    print('PASS') if tester_pre.test_get_file_type() else print('FAIL')
        
    print('Testing preprocessor.read_file()', end="...\t\t"+buffer)
    print('PASS') if tester_pre.test_read_file() else print('FAIL')
         
    print('Testing preprocessor.is_numeric()', end="...\t\t"+buffer)
    print('PASS') if tester_pre.test_is_numeric() else print('FAIL')
    
    print('Testing preprocessor.remove_na()', end="...\t\t"+buffer)
    print('PASS') if tester_pre.test_remove_na() else print('FAIL')
        
    print('Testing preprocessor.get_variable_type()', end="..."+buffer)
    print('PASS') if tester_pre.test_get_variable_type() else print('FAIL')
        
    print('Testing preprocessor.get_minority_class()', end="..."+buffer)
    print('PASS') if tester_pre.test_get_minority_class() else print('FAIL')
        
    print('Testing preprocessor.get_metadata()', end="...\t\t"+buffer)
    print('PASS') if tester_pre.test_get_metadata() else print('FAIL')
        
    print('Testing preprocessor.get_default_header()', end="..."+buffer)
    print('PASS') if tester_pre.test_get_default_header() else print('FAIL')
        
    print('Testing preprocessor.get_one_hot_encoding()', end="..."+buffer)
    print('PASS') if tester_pre.test_get_one_hot_encoding() else print('FAIL')
    
    print('Testing preprocessor.scale()', end="...\t\t\t"+buffer)
    print('PASS') if tester_pre.test_scale() else print('FAIL')
    
    print('Testing preprocessor.get_missing_column())', end="..."+buffer)
    print('PASS') if tester_pre.test_get_missing_column() else print('FAIL')
  
    print('Testing preprocessor.get_discretized_matrix()', end="...\t\t\t\t")
    print('PASS') if tester_pre.test_get_discretized_matrix() else print('FAIL')
    
    print('Testing preprocessor.get_discretized_matrix_missing()', end="...\t\t")
    print('PASS') if tester_pre.test_get_discretized_matrix_missing() else print('FAIL')
    
    print('Testing preprocessor.unscale()', end="...\t\t\t"+buffer)
    print('PASS') if tester_pre.test_unscale() else print('FAIL')
    
    print('Testing preprocessor.unravel_one_hot_encoding()', end="...\t\t\t\t")
    print('PASS') if tester_pre.test_unravel_one_hot_encoding() else print('FAIL')
    
    print('Testing preprocessor.restore_matrix()', end="...\t"+buffer)
    print('PASS') if tester_pre.test_restore_matrix() else print('FAIL')
    
    print('Testing preprocessor.restore_matrix_missing()', end="...\t\t\t\t")
    print('PASS') if tester_pre.test_restore_matrix_missing() else print('FAIL')
    
    print('Testing preprocessor.is_iterable()', end="...\t\t"+buffer)
    print('PASS') if tester_pre.test_is_iterable() else print('FAIL')
    
if __name__ == "__main__":
    main()
