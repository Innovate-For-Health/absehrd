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
    
def main():
    
    buffer = "\t\t"
    
    print('Testing realism.validate_univariate()', end='...\t'+buffer)
    print('PASS') if tester_rea.test_validate_univariate() else print('FAIL')
        
if __name__ == "__main__":
    main()
