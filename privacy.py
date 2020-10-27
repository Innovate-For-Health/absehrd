import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import DistanceMetric
import random

class privacy(object):
    
    """
    def nearest_neighbors(x, y=None, metric='euclidean'):
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X=x, y=y)
        distances, indices = nbrs.kneighbors(X=x, return_distance=True)
        
        return distances[:,1]
    """
    
    def distance(self, a, b, metric='euclidean'):
        
        dist = DistanceMetric.get_metric(metric)
        
        if len(a.shape) == 1:
            a = np.reshape(a, newshape=(1,len(a)))
        if len(b.shape) == 1:
            b = np.reshape(b, newshape=(1,len(b)))            
            
        return dist.pairwise(X=a, Y=b)
    
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
    
    def membership_inference_hayes(self, r_trn, r_tst, a_trn, s_trn, a_tst, s_tst, model_type='lr'):
        
        # real auxilliary and synthetic training set
        x_as_trn = np.row_stack((a_trn,s_trn))
        y_as_trn = np.append(np.zeros(len(a_trn)), np.ones(len(s_trn)))
        
        # real auxilliary and synthetic test set
        x_as_tst = np.row_stack((a_tst,s_tst))
        y_as_tst = np.append(np.zeros(len(a_tst)), np.ones(len(s_tst)))
        
        # real dataset (r_trn used to train the generator, r_tst held out)
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
    
    def membership_inference_torfi(self, r_trn, r_tst, s, n_sample=100, threshold=1e-3):
        
        precision = 0
        recall = 0
        p = []
        
        idx_trn = random.sample(range(len(r_trn)), min(n_sample, len(r_trn)))
        idx_tst = random.sample(range(len(r_tst)), min(n_sample, len(r_tst)))
        p = np.zeros(shape=len(idx_trn)+len(idx_tst))
        y = np.append(np.ones(len(idx_trn)), np.zeros(len(idx_tst)))
        x = np.row_stack((r_trn[idx_trn,:], r_tst[idx_tst,:]))
        
        for i in range(len(x)):
            for j in range(len(s)):
                if self.distance(a=r_trn[idx_trn[i],:], b=s[j,:], metric="cosine") < threshold:
                    p[i] = 1
                    
        precision = metrics.precision_score(y_true=y, y_pred=p)
        recall = metrics.recall_score(y_true=y, y_pred=p)
        accuracy = metrics.accuracy_score(y_true=y, y_pred=p)
        
        return {'precision':precision, 'recall':recall, 'accuracy':accuracy}
    
    def membership_inference(self, r_trn, r_tst, a_trn, s_trn, a_tst, s_tst, model_type='lr', mi_type='hayes'):
        
        if mi_type == 'hayes':
            return self.membership_inference_hayes(r_trn, r_tst, a_trn, s_trn, a_tst, s_tst, model_type)
        
        if mi_type == 'torfi':
            return self.membership_inference_torfi(r_trn, r_tst, np.row_stack((s_trn,s_tst)))
        
        return None
    
    def plot(self, res, analysis, file_pdf):
        
        fontsize = 6
        
        f = plt.figure()
        
        if analysis == 'nearest_neighbors':
        
            plt.hist((res['real'], res['synth'], 
                      res['prob'], res['rand']),
                     bins=30, 
                     label = ['Real-real','Real-synthetic','Real-probabilistic','Real-random'])
            plt.set_xlabel(dist_metric.capitalize()+' distance', fontsize=fontsize)
            plt.set_ylabel('Number of samples', fontsize=fontsize)
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
           
        elif analysis == 'membership_inference':
            
            # TODO: figure out what to plot for mem inf
            placeholder = None
            
        else:
            msg = 'Warning: plot for analysis \'' + analysis + 
            '\' not currently implemented in privacy::plot().' 
 
           
        plt.show()
        f.savefig(file_pdf, bbox_inches='tight')    
        return True
    
    def summarize(self, res, analysis, n_decimal=2):
        
        msg = ''
        n_decimal = 2
         
        if analysis == 'nearest_neighbors':
            msg = 'Mean nearest neighbor distance: ' +
                    '  > Real-real: ' +
                    str(np.round(np.mean(res['real']),n_decimal)) +
                    '  > Real-synthetic: ' +
                    str(np.round(np.mean(res['synth']),n_decimal)) +
                    '  > Real-probabilistic: ' +
                    str(np.round(np.mean(res['prob']),n_decimal)) +
                    '  > Real-random: ' +
                    str(np.round(np.mean(res['rand']),n_decimal))
        
        elif analysis == 'membership_inference':
            msg = 'AUC for auxiliary-synthetic: ' + res['auc_as'] +
                'AUC for real train-test: ' + res['auc_rr']
        else 
            msg = 'Warning: summary message for analysis \'' + analysis + 
            '\' not currently implemented in privacy::summarize().' 
        
        return msg
