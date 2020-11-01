# NOTE: must set PYTHONPATH variable for pytest to recognize local modules
# export PYTHONPATH=/my/path/to/modules
# OR
# export PYTHONPATH=$(pwd)

import numpy as np
import os
import feather
import pandas as pd

# sehrd modules
from preprocessor import preprocessor

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
        
        header = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
        v_constant = np.full(shape=n, fill_value=constant_value)
        v_binary01 = np.concatenate((np.full(shape=n-1, fill_value=0), np.array([1])))
        v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
        v_categorical = np.random.choice(categorical_values, size=n)
        v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
        v_continuous = np.random.random(size=n)
                
        x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
        return({'x':x, 'header':header})

    def test_get_file_type_npy(self):
        pre = preprocessor('null')
        file_name = 'file.npy'
        assert pre.get_file_type(file_name=file_name, debug=False) == 'npy'
        
    def test_get_file_type_tsv(self):
        pre = preprocessor('null')
        file_name = 'file.tsv'
        assert pre.get_file_type(file_name=file_name, debug=False) == 'tsv'
        
    def test_get_file_type_csv(self):
        pre = preprocessor('null')
        file_name = 'file.csv'
        assert pre.get_file_type(file_name=file_name, debug=False) == 'csv'
        
    def test_get_file_type_feather(self):
        pre = preprocessor('null')
        file_name = 'file.feather'
        assert pre.get_file_type(file_name=file_name, debug=False) == 'feather'
        
    def test_get_file_type_wrong1(self):
        pre = preprocessor('null')
        file_name = 'file.pdf'
        assert pre.get_file_type(file_name=file_name, debug=False) is None
        
    def test_get_file_type_wrong2(self):
        pre = preprocessor('null')
        file_name = 'test.npy.pdf'
        assert pre.get_file_type(file_name=file_name, debug=False) is None
        
    def test_get_file_type_wrong3(self):
        pre = preprocessor('null')
        file_name = 'feather.myfeather'
        assert pre.get_file_type(file_name=file_name, debug=False) is None
        
    def test_get_file_type_wrong4(self):
        pre = preprocessor('null')
        file_name = 'mycsv.txt'
        assert pre.get_file_type(file_name=file_name, debug=False) is None
        
    def test_get_file_type_wrong5(self):
        pre = preprocessor('null')
        file_name = 'test.tsv.tsv.not'
        assert pre.get_file_type(file_name=file_name, debug=False) is None
        
    def test_get_file_type_wrong6(self):
        pre = preprocessor('null')
        file_name = 'test.csvcsv'
        assert pre.get_file_type(file_name=file_name, debug=False) is None

    def test_read_file_npy(self):
        
        n = 10
        m = 2
        pre = preprocessor('none')
        file_name = 'test.npy'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.save(file_name, np.row_stack((header,arr)))
        res = pre.read_file(file_name, debug=False, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_feather(self):
        
        n = 10
        m = 2
        pre = preprocessor('none')
        file_name = 'test.feather'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)

        feather.write_dataframe(pd.DataFrame(arr, columns=header), file_name)
        res = pre.read_file(file_name, debug=False, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
        
    def test_read_file_feather(self):
        
        n = 10
        m = 2
        pre = preprocessor('none')
        file_name = 'test.feather'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)

        feather.write_dataframe(pd.DataFrame(arr, columns=header), file_name)
        res = pre.read_file(file_name, debug=False, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_csv(self):
        
        n = 10
        m = 2
        pre = preprocessor('none')
        file_name = 'test.csv'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.savetxt(file_name, arr, delimiter=',', header=','.join(header))
        res = pre.read_file(file_name, debug=False, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n
        
    def test_read_file_tsv(self):
        
        n = 10
        m = 2
        pre = preprocessor('none')
        file_name = 'test.tsv'
        
        header = self.create_header(m)
        arr = self.create_binary_matrix(n,m)
        
        np.savetxt(file_name, arr, delimiter='\t', header='\t'.join(header))
        res = pre.read_file(file_name, debug=False, has_header=True)
        os.remove(file_name)
        
        assert res is not None and len(res['header']) == m and len(res['x']) == n

    def test_read_file_bad1(self):
        
        pre = preprocessor('none')
        file_name = 'test.csv'
        res = pre.read_file(file_name, debug=False, has_header=True)
        
        assert res is None
        
    def test_read_file_bad2(self):
        
        pre = preprocessor('none')
        file_name = 'test.txt'
        res = pre.read_file(file_name, debug=False, has_header=True)
        
        assert res is None
        
    def test_is_numeric_yes(self):
        
        pre = preprocessor('none')
        assert pre.is_numeric([1,5,2,5,5.5,7.2])
        
    def test_is_numeric_no1(self):
        
        pre = preprocessor('none')
        assert not pre.is_numeric(['A','B','5','10.2']) 
        
    def test_is_numeric_no2(self):
        
        pre = preprocessor('none')
        assert pre.is_numeric(['1.2', '3.3', '5.0'])
        
    def test_remove_na_1(self):
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_2(self):
        
        missing_value = '-999999'
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_3(self):
        
        missing_value = 'NULL'
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array(['hello', missing_value, 'world'])
        d = pre.remove_na(v)
        g = np.array(['hello', 'world'])
    
        assert (g==d).all()
        
    def test_remove_na_4(self):
       
        missing_value = -99999
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1])
        
        assert (g==d).all()
        
    def test_remove_na_5(self):
       
        missing_value = '-99999'
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1], dtype=str)
        
        assert (g==d).all()
        
    def test_remove_na_6(self):
       
        missing_value = 'NULL'
        pre = preprocessor(missing_value=missing_value)
        
        v = np.array([3,2,5,missing_value,5,2,1])
        d = pre.remove_na(v)
        g = np.array([3,2,5,5,2,1], dtype=str)
        
        assert (g==d).all()

    def test_get_variable_type_count(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.random.randint(low=0,high=11,size=1000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'count'

    def test_get_variable_type_categorical(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.random.choice(['hello','world','oi','terra','hi','goodbye','tchau'],1000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'categorical'
        
    def test_get_variable_type_binary(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.random.randint(low=0,high=2,size=1000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'binary'
        
    def test_get_variable_type_constant_num(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.zeros(shape=1000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'constant'
        
    def test_get_variable_type_constant_str(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.full(fill_value='hello world', shape=10000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'constant'
  

    def test_get_variable_type_continuous(self):
        
        pre = preprocessor(missing_value=-999999)
        x = np.random.random(1000)
        var_type = pre.get_variable_type(x=x, label='my_feature')
        
        assert var_type == 'continuous'
        
    def test_get_minority_class_1(self):
        
        pre = preprocessor(missing_value=-999999)
        n0 = 20
        n1 = 17
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[1]
            
    def test_get_minority_class_2(self):
        
        pre = preprocessor(missing_value=-999999)
        n0 = 20
        n1 = 20
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[0]

    def test_get_minority_class_3(self):
        
        pre = preprocessor(missing_value=-999999)
        n0 = 4
        n1 = 40
        choices = ['hello','world']
        x = np.concatenate((np.full(n0,choices[0]), np.full(n1,choices[1])))
        assert pre.get_minority_class(x) == choices[0]
    
    def test_get_metadata_type(self):
        
        pre = preprocessor(missing_value= -999999)
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        g_type = ['constant','binary','binary','categorical','count','continuous']
        assert (m['type'].tolist() == g_type).all()
        
    def test_get_metadata_min(self):
        
        pre = preprocessor(missing_value= -999999)
        threshold = 1e-5
        
        obj = self.create_multimodal_object(n=1000)
        x = obj['x']
        header = obj['header']
        
        m = pre.get_metadata(x, pre.get_default_header(x.shape[1]))
        min_count = np.min(x[:,np.where(header=='count')])
        min_continuous = np.min(x[:,np.where(header=='continuous')[0][0]])
        g_min = [0,0,0,0,min_count, min_continuous]
        
        assert (np.abs(g_min - m['min']) < threshold).all()
        
        
