import numpy as np
from privacy import privacy

class tester_pri(object):
    
    def test_distance():
        
        pri = privacy()
        
        metric = "euclidean"
        a = np.array([[1,1]])
        b = np.array([[2,1]])
        
        
        d = pri.distance(a=a,b=b,metric=metric)
        
        if d != 1:
            return False
        
        metric = "euclidean"
        a = np.array([1,1])
        b = np.array([[2,1]])
    
        d = pri.distance(a=a,b=b,metric=metric)
        
        if d != 1:
            return False
        
        return True
        
    
    def test_nearest_neighbor():
        
        threshold=1e-5
        metric = "euclidean"
        x = np.array([[1,1],[4,4], [5,4]])
        pri = privacy()
        
        nn_dist = pri.nearest_neighbors(a=x, metric=metric)
        if(abs(nn_dist[0] - pri.distance(x[0,:], x[1,:], metric)) > threshold): 
            return False
        if(abs(nn_dist[1] - pri.distance(x[1,:], x[2,:], metric)) > threshold): 
            return False
        if(abs(nn_dist[2] - pri.distance(x[2,:], x[1,:], metric)) > threshold): 
            return False
        
        return True
    
    def test_assess_memorization():
        
        n = 10
        m = 3
        pri = privacy()
        
        x_real = np.random.random(size=(n,m))
        x_synth = np.random.random(size=(n,m))
        res = pri.assess_memorization(x_real, x_synth, 'euclidean')
        
        if(np.mean(res['real']) >= np.mean(res['rand'])):
            return False
        
        return True
    
def main():
    
    buffer = "\t\t"
    
    print('Testing privacy.distance()', end='...\t\t\t' + buffer)
    print('PASS') if tester_pri.test_distance() else print('FAIL')
    
    print('Testing privacy.nearest_neighbor()', end='...\t' + buffer)
    print('PASS') if tester_pri.test_nearest_neighbor() else print('FAIL')
    
    print('Testing privacy.assess_memorization)', end='...' + buffer)
    print('PASS') if tester_pri.test_assess_memorization() else print('FAIL')
        
if __name__ == "__main__":
    main()
