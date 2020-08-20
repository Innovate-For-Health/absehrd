import numpy as np
from privacy import privacy

class tester_pri(object):
    
    def test_nearest_neighbor():
        
        threshold=1e-5
        metric = "euclidean"
        x = np.array([[1,1],[4,4], [5,4]])
        
        nn_dist = privacy.nearest_neighbors(privacy, a=x, metric=metric)
        if(abs(nn_dist[0] - privacy.euclidean(x[0,:], x[1,:])) > threshold): 
            return False
        if(abs(nn_dist[1] - privacy.euclidean(x[1,:], x[2,:])) > threshold): 
            return False
        if(abs(nn_dist[2] - privacy.euclidean(x[2,:], x[1,:])) > threshold): 
            return False
        
        return True
    
    def test_assess_memorization():
        
        n = 10
        m = 3
        x_real = np.random.random(size=(n,m))
        x_synth = np.random.random(size=(n,m))
        res = privacy.assess_memorization(privacy, x_real, x_synth, 'euclidean')
        
        if(np.mean(res['real']) >= np.mean(res['rand'])):
            return False
        
        return True
    
def main():
    
    buffer = "\t\t"
    
    print('Testing privacy.nearest_neighbor()', end='...\t' + buffer)
    print('PASS') if tester_pri.test_nearest_neighbor() else print('FAIL')
    
    print('Testing privacy.assess_memorization)', end='...' + buffer)
    print('PASS') if tester_pri.test_assess_memorization() else print('FAIL')
        
if __name__ == "__main__":
    main()
