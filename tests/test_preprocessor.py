# NOTE: must set PYTHONPATH variable for pytest to recognize local modules
# export PYTHONPATH=/my/path/to/modules
# OR
# export PYTHONPATH=$(pwd)

import numpy as np
import os
import pandas as pd
import pickle
from pyarrow import feather

# sehrd modules
from preprocessor import Preprocessor

class TestPreprocessor:
    
    # create toy datasets    
    
    def create_binary_matrix(self, n, m):
        arr = np.random.randint(low=0,high=2, size=(n,m))
        return(arr)
    
    def create_header(self, m):
        header = []
        for i in range(m):
            header = np.append(header, 'col'+str(i))
        return(header)
    
    def create_multimodal_object(self, n=1000):
        
        count_min = 5
        count_max = 19
        constant_value = 'helloworld'
        binary_A = 'A'
        binary_B = 'B'
        categorical_values = ['X','Y','Z']
        
        header = np.array(['constant','binary01', 'binaryAB', 'categorical','count','continuous'])
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        return({'x':x, 'header':header})

    def test_get_file_type_npy(self):
        pre = Preprocessor('null')
        file_name = 'file.npy'
        assert pre.get_file_type(file_name=file_name) == 'npy'
        
    def test_get_file_type_pkl(self):
        pre = Preprocessor('null')
        file_name = 'file.pkl'
        assert pre.get_file_type(file_name=file_name) == 'pkl'
        
    def test_get_file_type_tsv(self):
        pre = Preprocessor('null')
        file_name = 'file.tsv'
        assert pre.get_file_type(file_name=file_name) == 'tsv'
        
    def test_get_file_type_csv(self):
        pre = Preprocessor('null')
        file_name = 'file.csv'
        assert pre.get_file_type(file_name=file_name) == 'csv'
        
    def test_get_file_type_feather(self):
        pre = Preprocessor('null')
        file_name = 'file.feather'
        assert pre.get_file_type(file_name=file_name) == 'feather'
        
    def test_get_file_type_wrong1(self):
        pre = Preprocessor('null')
        file_name = 'file.pdf'
        assert pre.get_file_type(file_name=file_name) is None
        
    def test_get_file_type_wrong2(self):
        pre = Preprocessor('null')
        file_name = 'test.npy.pdf'
        assert pre.get_file_type(file_name=file_name) is None
        
    def test_get_file_type_wrong3(self):
        pre = Preprocessor('null')
        file_name = 'feather.myfeather'
        assert pre.get_file_type(file_name=file_name) is None
        
    def test_get_file_type_wrong4(self):
        pre = Preprocessor('null')
        file_name = 'mycsv.txt'
        assert pre.get_file_type(file_name=file_name) is None
        
    def test_get_file_type_wrong5(self):
        pre = Preprocessor('null')
        file_name = 'test.tsv.tsv.not'
        assert pre.get_file_type(file_name=file_name) is None
        
    def test_get_file_type_wrong6(self):
        pre = Preprocessor('null')
        file_name = 'test.csvcsv'
        assert pre.get_file_type(file_name=file_name) is None

    def test_get_default_header(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        g5 = ['C0','C1','C2','C3','C4']
        h5 = pre.get_default_header(n_col=5, prefix='C')
        
        assert (g5 == h5).all()

    def test_read_file_npy(self):
        
        n = 10
        m = 2
        pre = Preprocessor('none')
        file_name = 'test.npy'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.save(file_name, np.row_stack((header,arr)))
        res = pre.read_file(file_name, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_pkl(self):
        
        n = 10
        m = 2
        pre = Preprocessor('none')
        file_name = 'test.pkl'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        data_obj = pd.DataFrame(data=arr, columns=header)
        
        with open(file_name, 'wb') as file_obj:
            pickle.dump(data_obj, file_obj, pickle.HIGHEST_PROTOCOL)
        
        res = pre.read_file(file_name, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_feather(self):
        
        n = 10
        m = 2
        pre = Preprocessor('none')
        file_name = 'test.feather'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)

        feather.write_feather(pd.DataFrame(arr, columns=header), file_name)
        res = pre.read_file(file_name, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_csv(self):
        
        n = 10
        m = 2
        pre = Preprocessor('none')
        file_name = 'test.csv'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.savetxt(file_name, arr, delimiter=',', header=','.join(header))
        res = pre.read_file(file_name, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_tsv(self):
        
        n = 10
        m = 2
        pre = Preprocessor('none')
        file_name = 'test.tsv'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.savetxt(file_name, arr, delimiter='\t', header='\t'.join(header))
        res = pre.read_file(file_name, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n

    def test_read_file_bad1(self):
        
        pre = Preprocessor('none')
        file_name = 'test.csv'
        res = pre.read_file(file_name, has_header=True)
        
        assert res is None
        
    def test_read_file_bad2(self):
        
        pre = Preprocessor('none')
        file_name = 'test.txt'
        res = pre.read_file(file_name, has_header=True)
        
        assert res is None
        
    def test_is_iterable_no(self):
        
        pre = Preprocessor(missing_value='NULL')
        assert not pre.is_iterable(3)
        
    def test_is_iterable_yes(self):
        
        pre = Preprocessor(missing_value='NULL')
        assert pre.is_iterable(np.array([1,2,3,4,5]))

    def test_is_numeric_yes1(self):
        
        pre = Preprocessor('none')
        assert pre.is_numeric([1,5,2,5,5.5,7.2])
        
    def test_is_numeric_yes2(self):
        
        pre = Preprocessor('none')
        assert pre.is_numeric(['1.2', '3.3', '5.0'])

    def test_is_numeric_yes3(self):

        pre = Preprocessor('none')
        assert pre.is_numeric(np.random.random(size=(10,7)))
        
    def test_is_numeric_no1(self):
        
        pre = Preprocessor('none')
        assert not pre.is_numeric(['A','B','5','10.2'])
        
    def test_remove_na_1(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_2(self):
        
        missing_value = '-999999'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_3(self):
        
        missing_value = 'NULL'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_4(self):
       
        missing_value = -99999
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1])
        
        assert (g==d).all()
        
    def test_remove_na_5(self):
       
        missing_value = '-99999'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1], dtype=str)
        
        assert (g==d).all()
        
    def test_remove_na_6(self):
       
        missing_value = 'NULL'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1], dtype=str)
        
        assert (g==d).all()
        
    def test_remove_na_7(self):
       
        missing_value = 'nan'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1], dtype=str)
        
        assert (g==d).all()
        
    def test_remove_na_8(self):
       
        missing_value = np.nan
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1])
        
        assert (g==d).all()
    
    def test_remove_na_9(self):
       
        missing_value = 'nan'
        pre = Preprocessor(missing_value=missing_value)
        
        v = np.full(6,'nan')
        d = pre.remove_na(v)
        
        assert len(d) == 0

    def test_get_variable_type_count(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.random.randint(low=0,high=11,size=1000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'count'

    def test_get_variable_type_categorical(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.random.choice(['hello','world','oi','terra','hi','goodbye','tchau'],1000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'categorical'
        
    def test_get_variable_type_binary(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.random.randint(low=0,high=2,size=1000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'binary'
        
    def test_get_variable_type_constant_num(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.zeros(shape=1000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'constant'
        
    def test_get_variable_type_constant_str(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.full(fill_value='hello world', shape=10000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'constant'
  

    def test_get_variable_type_continuous(self):
        
        pre = Preprocessor(missing_value=-999999)
        x = np.random.random(1000)
        var_type = pre.get_variable_type(arr=x, label='my_feature')
        
        assert var_type == 'continuous'
        
    def test_get_minority_class_1(self):
        
        pre = Preprocessor(missing_value=-999999)
        n0 = 20
        n1 = 17
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[1]
            
    def test_get_minority_class_2(self):
        
        pre = Preprocessor(missing_value=-999999)
        n0 = 20
        n1 = 20
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[0]

    def test_get_minority_class_3(self):
        
        pre = Preprocessor(missing_value=-999999)
        n0 = 4
        n1 = 40
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[0]
    
    def test_get_metadata_type(self):
        
        pre = Preprocessor(missing_value= -999999)
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_type = ['constant','binary','binary','categorical','count','continuous']
        assert m['type'].tolist() == g_type
       
    def test_get_metadata_min(self):
        
        pre = Preprocessor(missing_value= -999999)
        threshold = 1e-5
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        header = obj['header']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        min_count = np.min((x[:,np.where(header=='count')[0][0]]).astype(int))
        min_continuous = np.min((x[:,np.where(header=='continuous')[0][0]]).astype(float))
        g_min = [0,0,0,0,min_count, min_continuous]
        
        assert np.allclose(m['min'], g_min, atol=threshold)
        
    def test_get_metadata_max(self):
        
        pre = Preprocessor(missing_value= -999999)
        threshold = 1e-5
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        header = obj['header']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        max_count = np.max((x[:,np.where(header=='count')[0][0]]).astype(int))
        max_continuous = np.max((x[:,np.where(header=='continuous')[0][0]]).astype(float))
        g_min = [0,0,0,0,max_count, max_continuous]
        
        assert np.allclose(m['max'], g_min, atol=threshold)
        
    def test_get_metadata_zero(self):
        
        pre = Preprocessor(missing_value= -999999)
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_zero = ['helloworld','0','A','','','']
        
        assert (m['zero'] == g_zero).all()
        
    def test_get_metadata_one(self):
        
        pre = Preprocessor(missing_value= -999999)
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_one = ['','1','B','','','']
        
        assert (m['one'] == g_one).all()
        
    def test_get_metadata_unique(self):
        
        pre = Preprocessor(missing_value= -999999)
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_unique = ['','','','X,Y,Z','','']
        
        assert (m['unique'] == g_unique).all()
        
    def test_get_metadata_missing(self):
        
        missing_value = -999999
        idx = 1
        pre = Preprocessor(missing_value= -999999)
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        x[np.random.randint(low=0,high=len(x), size=len(x)),idx] = str(missing_value)
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_missing = [False, True, False, False, False, False]
        
        assert m['missing'].tolist() == g_missing
        
    def test_get_one_hot_encoding(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        label="var"
        
        x = np.array(['Y','X','X','Z','Y'])
        res = pre.get_one_hot_encoding(arr=x, label=label)
        g_x_hot = np.array([(0,1,0),(1,0,0),(1,0,0),(0,0,1),(0,1,0)]) 
        
        assert (res['x'] == g_x_hot).all()
        
    def test_get_one_hot_encoding_missing(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        label="var"
        
        x = np.array(['Y','X','X',missing_value,'Y'])
        res = pre.get_one_hot_encoding(arr=x, label=label)
        g_x_hot = np.array([(0,0,1),(0,1,0),(0,1,0),(1,0,0),(0,0,1)]) 
        
        assert (res['x'] == g_x_hot).all()

    def test_scale(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        x = np.array([0,1,2,3,4,5,6,7,8,9,10])
        g_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        s = pre.scale(x)
        
        assert (s == g_s).all()
        
    def test_unscale(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        min_value=0
        max_value=10
        s = [0,0.1,0.2,0.3,0.4,1]
        g = [0, 1, 2, 3, 4, 10]
        x = pre.unscale(s, min_value, max_value)
        
        assert (x == g).all()

    def test_get_missing_column(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        x1 = np.array([1,2,3,missing_value, 4])
        g1 = np.array([0,0,0,1,0])
        c1 = pre.get_missing_column(x1)
        
        assert (g1 == c1).all()
        
        
    def test_get_na_idx_full(self):
    
        missing_value = 'NULL'
        pre = Preprocessor(missing_value=missing_value)
        
        x = np.array([4,5,2,5,3])
        r_idx = pre.get_na_idx(x)
        
        assert len(r_idx) == 0
        
    def test_get_na_idx_missing_1(self):
    
        missing_value = 'NULL'
        pre = Preprocessor(missing_value=missing_value)
        
        x = np.array([4,5,missing_value,2,5,3])
        r_idx = pre.get_na_idx(x)
        
        assert r_idx[0] == 2
        
    def test_get_na_idx_missing_2(self):
    
        missing_value = 'NULL'
        pre = Preprocessor(missing_value=missing_value)
        
        x = np.array(['hello','world','earth','tierra',missing_value,'blah','foo',missing_value])
        g_idx = np.array([4,7])
        r_idx = pre.get_na_idx(x)
        
        assert r_idx.sort() == g_idx.sort()
        
    def test_get_discretized_matrix_full(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        obj_f = self.create_multimodal_object(n=1000)
        m = pre.get_metadata(obj_f['x'], pre.get_default_header(obj_f['x'].shape[1]))
        
        obj_d = pre.get_discretized_matrix(arr=obj_f['x'], meta=m, 
                    header=obj_f['header'], require_missing=False)
        
        assert len(obj_d['header']) == 8

    def test_get_discretized_matrix_missing_1(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        obj_f = self.create_multimodal_object(n=1000)
        m = pre.get_metadata(obj_f['x'], pre.get_default_header(obj_f['x'].shape[1]))
        
        obj_d = pre.get_discretized_matrix(arr=obj_f['x'], meta=m, 
                    header=obj_f['header'], require_missing=True)
        
        assert obj_d['x'].shape[1] == 14
        
    def test_get_discretized_matrix_missing_2(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        obj_f = self.create_multimodal_object(n=1000)
        m = pre.get_metadata(obj_f['x'], pre.get_default_header(obj_f['x'].shape[1]))
        
        obj_d = pre.get_discretized_matrix(arr=obj_f['x'], meta=m, 
                    header=obj_f['header'], require_missing=True)
        
        assert len(obj_d['header']) == 14
                
    def test_unravel_one_hot_encoding(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        unique_values = ['val1', 'val2', 'val3']
        
        s = np.array([[0,0,1], [0,0,1], [0,1,0], [1,0,0], [0,1,0]])
        g = ['val3','val3','val2','val1','val2']
        x = pre.unravel_one_hot_encoding(arr=s, unique_values=unique_values)
        
        assert (x == g).all()
        
    def test_restore_matrix_1(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        header = ['col1','col2']
        x = np.random.randint(low=0, high=2, size=(5,2)).astype(str)
        v = np.full(shape=len(header), fill_value=False)
        
        m = pre.get_metadata(arr=x, header=header)
        obj_d = pre.get_discretized_matrix(arr=x, meta=m, 
                header=header, require_missing=True)
        obj_r = pre.restore_matrix(arr=obj_d['x'], meta=m, header=obj_d['header'])
        
        # check header
        for i in range(len(header)):
            v[i] = (header[i] == obj_r['header'][i])
        
        assert v.all()
        
    def test_restore_matrix_2(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        threshold = 1e-5
        
        header = ['col1','col2']
        x = np.random.randint(low=0, high=2, size=(5,2)).astype(str)
        v = np.full(shape=x.shape, fill_value=False)
        
        m = pre.get_metadata(arr=x, header=header)
        obj_d = pre.get_discretized_matrix(arr=x, meta=m, 
                header=header, require_missing=True)
        obj_r = pre.restore_matrix(arr=obj_d['x'], meta=m, header=obj_d['header'])
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if m[j]['type'] == 'count' or m[j]['type'] == 'continuous':
                    if abs(float(x[i,j]) - float(obj_r['x'][i,j])) < threshold:
                        v[i,j] = True
                else:
                    if x[i,j] == obj_r['x'][i,j]:
                        v[i,j] = True
        
        assert v.all()
   
    def test_restore_matrix_3(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        
        obj_f = self.create_multimodal_object(n=1000)
        v = np.full(shape=len(obj_f['header']), fill_value=False)
        
        m = pre.get_metadata(obj_f['x'], obj_f['header'])
        obj_d = pre.get_discretized_matrix(arr=obj_f['x'], meta=m, 
                header=obj_f['header'], require_missing=True)
        obj_r = pre.restore_matrix(arr=obj_d['x'], meta=m, header=obj_d['header'])
        
        # check header
        for i in range(len(obj_f['header'])):
            v[i] = obj_f['header'][i] == obj_r['header'][i]
                
        assert v.all()
               
    def test_restore_matrix_4(self):
        
        missing_value = -999999
        pre = Preprocessor(missing_value=missing_value)
        threshold = 1e-5
        
        obj_f = self.create_multimodal_object(n=1000)
        v = np.full(shape=obj_f['x'].shape, fill_value=False)
        
        m = pre.get_metadata(obj_f['x'], obj_f['header'])
        obj_d = pre.get_discretized_matrix(arr=obj_f['x'], meta=m, 
                header=obj_f['header'], require_missing=True)
        obj_r = pre.restore_matrix(arr=obj_d['x'], meta=m, header=obj_d['header'])
        
        for i in range(obj_f['x'].shape[0]):
            for j in range(obj_f['x'].shape[1]):
                if m[j]['type'] == 'count' or m[j]['type'] == 'continuous':
                    if abs(float(obj_f['x'][i,j]) - float(obj_r['x'][i,j])) < threshold:
                        v[i,j] = True
                else:
                    if obj_f['x'][i,j] == obj_r['x'][i,j]:
                        v[i,j] = True
        
        assert v.all()
