# NOTE: must set PYTHONPATH variable for pytest to recognize local modules
# export PYTHONPATH=/my/path/to/modules

import numpy as np
import os
import feather
import pandas as pd

# sehrd modules
from preprocessor import preprocessor

class TestPreprocessor:
    
    def create_binary_matrix(self, n, m):
        arr = np.random.randint(low=0,high=2, size=(n,m))
        return(arr)
    
    def create_header(self, m):
        header = []
        for i in range(m):
            header = np.append(header, 'col'+str(i))
        return(header)

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
        
    def test_remove_na(self):
        
        missing_value = -999999
        pre = preprocessor(missing_value=missing_value)
        
        v1 = np.array(['hello', missing_value, 'world'])
        d1 = pre.remove_na(v1)
        g1 = np.array(['hello', 'world'])
    
        assert (g1==d1).any()
    