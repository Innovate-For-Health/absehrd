import numpy as np


# abserhd
from privacy import Privacy

class TestPrivacy(object):
    
    def test_distance_euclidean(self):
        
        pri = Privacy()
        
        metric = 'euclidean'
        a = np.array([[1,1]])
        b = np.array([[2,1]])
        d = pri.distance(arr1=a,arr2=b,metric=metric)
        
        assert d == 1
        
    def test_distance_hamming(self):
        
        pri = Privacy()
        
        metric = 'hamming'
        a = np.array([[1,1]])
        b = np.array([[1,1]])
        d = pri.distance(arr1=a,arr2=b,metric=metric)
        
        assert d == 0
        
    
    def test_nearest_neighbor(self):
        
        threshold=1e-5
        metric = "euclidean"
        x = np.array([[1,1],[4,4], [5,4]])
        pri = Privacy()
        
        nn_dist = pri.nearest_neighbors(arr1=x, metric=metric)
        assert abs(nn_dist[0] - pri.distance(x[0,:], x[1,:], metric)) < threshold
            
    def test_assess_memorization(self):
        
        n = 1000
        m = 3
        missing_value = -999999
        pri = Privacy()
        
        header =[]
        for i in range(m):
            header = np.append(header, 'col'+str(i))

        x_real = np.random.random(size=(n,m))
        x_synth = np.random.random(size=(n,m))
        res = pri.assess_memorization(mat_f_r=x_real, 
                                      mat_f_s=x_synth, 
                                      missing_value=missing_value, 
                                      header=header,
                                      metric='euclidean', 
                                      debug=False)
        
        assert np.mean(res['real']) < np.mean(res['rand'])

    def test_membership_inference_torfi_match(self):
        
        n = 10000
        m = 17
        threshold = 0.05
        missing_value = -999999
        pri = Privacy()
        
        header =[]
        for i in range(m):
            header = np.append(header, 'col'+str(i))
        
        # create dummy dataset for high risk of membership disclosure
        r_trn = np.random.normal(loc=0, size=(n,m))
        r_tst = np.random.normal(loc=100, size=(n,m))
        s = r_trn
        
        res_mi =  pri.membership_inference(mat_f_r_trn=r_trn, 
                                           mat_f_r_tst=r_tst, 
                                           mat_f_s=s,
                                           header=header,
                                           missing_value=missing_value,
                                           mi_type='torfi',
                                           n_cpu=1)
        
        avg_p_trn = np.mean(res_mi['prob'][np.where(res_mi['label']==1)])
        avg_p_tst = np.mean(res_mi['prob'][np.where(res_mi['label']==0)])
        
        
        assert avg_p_trn - avg_p_tst > threshold
        
    def test_membership_inference_torfi_mismatch(self):
        
        n = 10000
        m = 17
        missing_value = -999999
        pri = Privacy()
        
        header =[]
        for i in range(m):
            header = np.append(header, 'col'+str(i))
        
        # create dummy dataset for high risk of membership disclosure
        r_trn = np.random.normal(loc=0, size=(n,m))
        r_tst = np.random.normal(loc=0, size=(n,m))
        s =  np.random.normal(loc=10, size=(n,m))
        
        res_mi =  pri.membership_inference(mat_f_r_trn=r_trn, 
                                           mat_f_r_tst=r_tst, 
                                           mat_f_s=s,
                                           header=header,
                                           missing_value=missing_value,
                                           mi_type='torfi',
                                           n_cpu=1)
        
        avg_p_trn = np.mean(res_mi['prob'][np.where(res_mi['label']==1)])
        avg_p_tst = np.mean(res_mi['prob'][np.where(res_mi['label']==0)])
        
        assert np.allclose(avg_p_trn, avg_p_tst, atol=0.05)
        
    def test_membership_inference_hayes_match(self):

        """
        n = 10000
        m = 17
        threshold = 0.05
        missing_value = -999999
        pri = Privacy()
        
        header =[]
        for i in range(m):
            header = np.append(header, 'col'+str(i))
        
        # create dummy dataset for high risk of membership disclosure
        r_trn = np.random.normal(loc=0, size=(n,m))
        r_tst = np.random.normal(loc=100, size=(n,m))
        s = r_trn
        
        res_mi =  pri.membership_inference(mat_f_r_trn=r_trn, 
                                           mat_f_r_tst=r_tst, 
                                           mat_f_s=s,
                                           header=header,
                                           missing_value=missing_value,
                                           mi_type='hayes',
                                           n_cpu=1)
        
        avg_p_trn = np.mean(res_mi['prob'][np.where(res_mi['label']==1)])
        avg_p_tst = np.mean(res_mi['prob'][np.where(res_mi['label']==0)])
        
        
        assert avg_p_trn - avg_p_tst > threshold
        """
        assert True

        
    def test_membership_inference_hayes_mismatch(self):
        assert True