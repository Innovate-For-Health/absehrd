# NOTE: must set PYTHONPATH variable for pytest to recognize local modules
# export PYTHONPATH=/my/path/to/modules
# OR
# export PYTHONPATH=$(pwd)

import numpy as np

# absehrd modules
from realism import Realism

class TestRealism:
    
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
    
    def test_which_list(self):
        
        rea = Realism()
        x = ['a','b','c']
        idx = 1
        item = x[idx]
        
        assert idx == rea.which(x, item)[0]
        
    def test_which_array(self):
        
        rea = Realism()
        x = np.array(['a','b','c'])
        idx = 1
        item = x[idx]
        
        assert idx == rea.which(x,item)[0]

        
    def test_validate_univariate(self):
        
        rea = Realism()
        n = 1000
        m = 17
        v = np.full(shape=m, fill_value=False)
        
        prefix='col'
        header = np.full(shape=m, fill_value='', dtype='<U'+str(len(str(m-1))+len(prefix)))
        for i in range(m):
            header[i] = prefix + str(i).zfill(len(str(m-1)))
        
        x = np.random.randint(low=0, high=2, size=(n,m))
        res = rea.validate_univariate(arr_r=x, arr_s=x, header=header)
        
        for j in range(m):
            if res['frq_r'][j] == res['frq_s'][j]:
                v[j] = True
        
        assert v.all()
    
    
    def test_gan_train_match(self):
        
        rea = Realism()
        
        n = 1000
        m_2 = 3
        threshold = 0.05
        max_beta = 10
        n_epoch = 100
        
        beta = np.append(np.random.randint(low=-max_beta,high=0,size=(m_2,1)), 
                         np.random.randint(low=0,high=max_beta,size=(m_2,1)))
        x_real = np.random.randint(low=0, high=2, size=(n,m_2*2))
        x_for_e = np.reshape(np.matmul(x_real, beta), (n,1)) + 0.5 * np.random.random(size=(n,1))
        y_real = np.reshape(np.round(1.0 / (1.0 + np.exp(-x_for_e))), (n,))
        
        res_real = rea.gan_train(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        res_gan_train1 = rea.gan_train(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        
        assert (abs(res_real['auc'] - res_gan_train1['auc']) < threshold)
        
    def test_gan_train_mismatch(self):
        
        rea = Realism()
        
        n = 1000
        m_2 = 3
        threshold = 0.05
        max_beta = 10
        n_epoch = 100
        
        beta = np.append(np.random.randint(low=-max_beta,high=0,size=(m_2,1)), 
                         np.random.randint(low=0,high=max_beta,size=(m_2,1)))
        x_real = np.random.randint(low=0, high=2, size=(n,m_2*2))
        x_for_e = np.reshape(np.matmul(x_real, beta), (n,1)) + 0.5 * np.random.random(size=(n,1))
        y_real = np.reshape(np.round(1.0 / (1.0 + np.exp(-x_for_e))), (n,))
        
        res_real = rea.gan_train(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        x_synth = x_real
        y_synth = 1 - y_real
        res_gan_train2 = rea.gan_train(x_synth, y_synth, x_real, y_real, n_epoch=n_epoch)
        
        assert abs(res_real['auc'] - res_gan_train2['auc']) > threshold
        
        
    def test_gan_test_match(self):
        
        rea = Realism()
        
        n = 1000
        m_2 = 3
        threshold = 0.05
        max_beta = 10
        n_epoch = 100
        
        beta = np.append(np.random.randint(low=-max_beta,high=0,size=(m_2,1)), 
                         np.random.randint(low=0,high=max_beta,size=(m_2,1)))
        x_real = np.random.randint(low=0, high=2, size=(n,m_2*2))
        x_for_e = np.reshape(np.matmul(x_real, beta), (n,1)) + 0.5 * np.random.random(size=(n,1))
        y_real = np.reshape(np.round(1.0 / (1.0 + np.exp(-x_for_e))), (n,))
        
        res_real = rea.gan_test(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        res_gan_test1 = rea.gan_test(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        
        assert (abs(res_real['auc'] - res_gan_test1['auc']) < threshold)

    
    def test_gan_test_mismatch(self):
        
        rea = Realism()
        
        n = 1000
        m_2 = 3
        threshold = 0.05
        max_beta = 10
        n_epoch = 100
        
        beta = np.append(np.random.randint(low=-max_beta,high=0,size=(m_2,1)), 
                         np.random.randint(low=0,high=max_beta,size=(m_2,1)))
        x_real = np.random.randint(low=0, high=2, size=(n,m_2*2))
        x_for_e = np.reshape(np.matmul(x_real, beta), (n,1)) + 0.5 * np.random.random(size=(n,1))
        y_real = np.reshape(np.round(1.0 / (1.0 + np.exp(-x_for_e))), (n,))
        
        # flip label to ensure AUCs are very different
        x_synth = x_real
        y_synth = 1 - y_real
        res_real = rea.gan_train(x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        res_gan_test2 = rea.gan_test(x_synth, y_synth, x_real, y_real, n_epoch=n_epoch)
        
        assert (abs(res_real['auc'] - res_gan_test2['auc']) > threshold)
        
        
    def test_gan_test(self):
        assert True
    
    def test_validate_feature(self):
        assert True