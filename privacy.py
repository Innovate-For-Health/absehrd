import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm

class privacy(object):
    
    """
    def nearest_neighbors(x, y=None, metric='euclidean'):
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X=x, y=y)
        distances, indices = nbrs.kneighbors(X=x, return_distance=True)
        
        return distances[:,1]
    """
    
    def euclidean(self, a,b):
        
        d = 0
        
        for i in range(len(a)):
            d += (a[i] - b[i])**2
            
        return math.sqrt(d)
    
    def distance(self, a, b, metric):
        
        if metric == 'euclidean':
            return self.euclidean(a,b)
        
        return None
    
    def nearest_neighbors(self, a, b=None, metric='euclidean'):
        
        d_min = np.full(shape=len(a), fill_value=float('inf'))
        
        if(b is None):
            
            for i in range(len(a)):
                d_i = np.full(shape=len(a), fill_value=float('inf'))
                
                for j in range(len(a)):
                    if i != j:
                        d_i[j] = self.distance(a[i,:], a[j,:], metric)
                    
                d_min[i] = np.min(d_i)
                
        else:
            
            for i in range(len(a)):
                d_i = np.full(shape=len(b), fill_value=float('inf'))
                
                for j in range(len(b)):
                    d_i[j] = self.distance(a[i,:], b[j,:], metric)
                    
                d_min[i] = np.min(d_i)
        
        return d_min
    
    def assess_memorization(self, x_real, x_synth, metric='euclidean'):
        
        # real to real
        nn_real = self.nearest_neighbors(a=x_real, metric=metric)
        
        # real to synth
        nn_synth = self.nearest_neighbors(a=x_real, b=x_synth, metric=metric)
        
        # real to probabilistically sampled
        x_prob = np.full(shape=x_real.shape, fill_value=0)
        for j in range(x_real.shape[1]):
            x_prob[:,j] = np.random.binomial(n=1, p=np.mean(x_real[:,j]), size=x_real.shape[0])
        nn_prob = self.nearest_neighbors(a=x_real, b=x_prob, metric=metric)
        
        # real to noise
        x_rand = np.random.randint(low=0, high=2, size=x_real.shape)
        nn_rand = self.nearest_neighbors(a=x_real, b=x_rand, metric=metric)
        
        return {'real':nn_real, 'synth':nn_synth, 'prob':nn_prob, 'rand':nn_rand}
    
    def membership_inference(self, r_trn, r_tst, a_trn, s_trn, a_tst, s_tst, model_type='lr'):
        
        x_as_trn = np.row_stack((a_trn,s_trn))
        y_as_trn = np.append(np.zeros(len(a_trn)), np.ones(len(s_trn)))
        
        x_as_tst = np.row_stack((a_tst,s_tst))
        y_as_tst = np.append(np.zeros(len(a_tst)), np.ones(len(s_tst)))
        
        x_rr_tst = np.row_stack((r_trn,r_tst))
        y_rr_tst = np.append(np.zeros(len(r_trn)), np.ones(len(r_tst)))
        
        if model_type == 'lr':
            clf = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1')
        elif model_type == 'svm':
            clf = svm.SVC(kernel='rbf', gamma='scale', probability=True)
        else:
            print('Error: ', model_type, ' is not recognized.')
            return None
        
        model = clf.fit(X=x_as_trn, y=y_as_trn)
        p_as = model.predict_proba(x_as_tst)[:,1]
        p_rr = model.predict_proba(x_rr_tst)[:,1]
        
        auc_as = metrics.roc_auc_score(y_true=y_as_tst, y_score=p_as)
        auc_rr = metrics.roc_auc_score(y_true=y_rr_tst, y_score=p_rr)
        
        return {'prob_rr':p_rr, 'prob_as':p_as, 
                'auc_as':auc_as, 'auc_rr':auc_rr}
        