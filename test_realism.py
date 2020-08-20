"""
Description: automated tests for SEHRD realism functions.
Author: Haley Hunter-Zinck
Date: August 14, 2020
"""

from preprocessor import preprocessor as pre
from realism import realism as rea
import numpy as np

class tester_rea(object):
    
    def test_validate_univariate():
        
        n = 1000
        m = 4
        
        header = np.full(shape=m, fill_value='col')
        for j in range(m):
            header[j] = header[j] + str(j)
            
        s = np.random.randint(low=0, high=2, size=(n,m))
        r = s
        
        res = rea.validate_univariate(r, s, header)
        
        for j in range(m):
            if res['frq_r'][j] != res['frq_s'][j]:
                return False
        
        return True
    
    def test_gan_train(debug=False):
        
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
        
        res_real = rea.gan_train(rea, x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        res_gan_train1 = rea.gan_train(rea, x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        
        if debug:
            print('percentage 1 for y_real: ', str(float(sum(y_real)) / n * 100))
            print('auc real: '+ str(res_real['auc']))
            print('auc gan-train #1: ', str(res_gan_train1['auc']))
        
        if (abs(res_real['auc'] - res_gan_train1['auc']) > threshold):
            return False
        
        # flip label to ensure AUCs are very different
        x_synth = x_real
        y_synth = 1 - y_real
        res_gan_train2 = rea.gan_train(rea, x_synth, y_synth, x_real, y_real, n_epoch=n_epoch)
        
        if debug:
            print('auc real: '+ str(res_real['auc']))
            print('auc gan-train #2: ', str(res_gan_train2['auc']))

        if (abs(res_real['auc'] - res_gan_train2['auc']) <= threshold):
            return False
        
        return True
    
    def test_gan_test(debug=False):
        
        if debug:
            print()
        
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
        
        res_real = rea.gan_test(rea, x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        res_gan_test1 = rea.gan_test(rea, x_synth=x_real, y_synth=y_real, 
                                      x_real=x_real, y_real=y_real, n_epoch=n_epoch)
        
        if debug:
            print('percentage 1 for y_real: ', str(float(sum(y_real)) / n * 100))
            print('auc real: '+ str(res_real['auc']))
            print('auc gan-train #1: ', str(res_gan_test1['auc']))
        
        if (abs(res_real['auc'] - res_gan_test1['auc']) > threshold):
            return False
        
        # flip label to ensure AUCs are very different
        x_synth = x_real
        y_synth = 1 - y_real
        res_gan_test2 = rea.gan_test(rea, x_synth, y_synth, x_real, y_real, n_epoch=n_epoch)
        
        if debug:
            print('auc real: '+ str(res_real['auc']))
            print('auc gan-train #2: ', str(res_gan_test2['auc']))

        if (abs(res_real['auc'] - res_gan_test2['auc']) <= threshold):
            return False
        
        return True
    
def main():
    
    buffer = "\t\t"
    
    print('Testing realism.validate_univariate()', end='...'+buffer)
    print('PASS') if tester_rea.test_validate_univariate() else print('FAIL')
    
    print('Testing realism.gan_train()', end='...\t\t'+buffer)
    print('PASS') if tester_rea.test_gan_train(debug=False) else print('FAIL')
    
    print('Testing realism.gan_test()', end='...\t\t\t'+buffer)
    print('PASS') if tester_rea.test_gan_test(debug=False) else print('FAIL')
        
if __name__ == "__main__":
    main()
