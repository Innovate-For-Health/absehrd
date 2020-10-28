import numpy as np
from sklearn import metrics
from sklearn.neighbors import DistanceMetric
import random
import matplotlib.pyplot as plt

# sehrd modules
from validator import Validator
from corgan import corgan
from corgan import Discriminator

class privacy(Validator):
    
    """
    def nearest_neighbors(x, y=None, metric='euclidean'):
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X=x, y=y)
        distances, indices = nbrs.kneighbors(X=x, return_distance=True)
        
        return distances[:,1]
    """
    
    def scale(self, x, invert=False):
        
        s =  (x - np.min(x)) / (np.max(x) - np.min(x))
        
        if invert:
            return 1 - s
        
        return s
    
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
        
        return {'real':nn_real, 'synth':nn_synth, 'prob':nn_prob, 'rand':nn_rand, 'metric':metric}
    
    def membership_inference_hayes(self, r_trn, r_tst, s, n_cpu):
        
        cor = corgan()
        
        # evaluation set
        x = np.row_stack((r_tst,r_trn))
        y = np.append(np.zeros(len(r_tst)), np.ones(len(r_trn)))
        
        # train shadow GAN
        gan_shadow = cor.train(x=s)
        
        # load shadow discriminator
        minibatch_averaging = gan_shadow['parameter_dict']['minibatch_averaging']
        feature_size = gan_shadow['parameter_dict']['feature_size']
        d_shadow = Discriminator(minibatch_averaging=minibatch_averaging, 
                                 feature_size=feature_size)
        d_shadow.load_state_dict(gan_shadow['Discriminator_state_dict'])
        d_shadow.eval()
        
        # calculate probabilities from shadow discriminator
        prob = d_shadow(x)
        
        roc = metrics.roc_curve(y_true=y, y_score=prob)
        auc = metrics.roc_auc_score(y_true=y, y_score=prob)
        
        return {'prob': prob, 'roc':roc, 'auc':auc}
    
    def membership_inference_torfi(self, r_trn, r_tst, s, n_sample=100, threshold=1e-3):
        
        idx_trn = random.sample(range(len(r_trn)), min(n_sample, len(r_trn)))
        idx_tst = random.sample(range(len(r_tst)), min(n_sample, len(r_tst)))
        d = np.zeros(shape=len(idx_trn)+len(idx_tst))
        y = np.append(np.ones(len(idx_trn)), np.zeros(len(idx_tst)))
        x = np.row_stack((r_trn[idx_trn,:], r_tst[idx_tst,:]))
        
        # store distance not label
        for i in range(len(x)):
            for j in range(len(s)):
                d[i] = self.distance(a=r_trn[idx_trn[i],:], b=s[j,:], metric="cosine")
        
        # scale distances to get 'probabilities'
        prob = self.scale(d, invert=True)
        
        # calculate performance metrics 
        roc = metrics.roc_curve(y_true=y, y_score=prob)
        auc = metrics.roc_auc_score(y_true=y, y_score=prob)
        
        return {'prob': prob, 'roc':roc, 'auc':auc}
        
    def membership_inference(self, r_trn, r_tst, s, mi_type='hayes', n_cpu=1):
        
        if mi_type == 'hayes':
            return self.membership_inference_hayes(r_trn, r_tst, s, n_cpu=n_cpu)
        
        if mi_type == 'torfi':
            return self.membership_inference_torfi(r_trn, r_tst, s)
        
        return None
    
    def plot(self, res, analysis, file_pdf):
        
        fontsize = 6
        
        f = plt.figure()
        
        if analysis == 'nearest_neighbors':
        
            # plot distributions of distances
            plt.hist((res['real'], res['synth'], 
                      res['prob'], res['rand']),
                     bins=30, 
                     label = ['Real-real','Real-synthetic','Real-probabilistic','Real-random'])
            plt.set_xlabel(res['metric'].capitalize()+' distance', fontsize=fontsize)
            plt.set_ylabel('Number of samples', fontsize=fontsize)
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
           
        elif analysis == 'membership_inference':
            
            # plot ROC curve 
            plt.plot(res['roc'][0], res['roc'][1], label="Real")
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.set_xlabel('False positive rate', fontsize=fontsize)
            plt.set_ylabel('True positive rate', fontsize=fontsize)
            plt.title('AUC = ' + res['auc'])
            
        else:
            msg = 'Warning: plot for analysis \'' + analysis + \
                '\' not currently implemented in privacy::plot().' 
            print(msg)

        plt.show()
        f.savefig(file_pdf, bbox_inches='tight')    
        return True
    
    def summarize(self, res, analysis, n_decimal=2):
        
        msg = ''
        n_decimal = 2
         
        if analysis == 'nearest_neighbors':
            msg = 'Mean nearest neighbor distance: ' + \
                    '  > Real-real: ' + \
                    str(np.round(np.mean(res['real']),n_decimal)) + \
                    '  > Real-synthetic: ' + \
                    str(np.round(np.mean(res['synth']),n_decimal)) + \
                    '  > Real-probabilistic: ' + \
                    str(np.round(np.mean(res['prob']),n_decimal)) + \
                    '  > Real-random: ' + \
                    str(np.round(np.mean(res['rand']),n_decimal))
        
        elif analysis == 'membership_inference':
            msg = 'AUC for auxiliary-synthetic: ' + res['auc_as'] + \
                'AUC for real train-test: ' + res['auc_rr']
        else:
            msg = 'Warning: summary message for analysis \'' + analysis + \
            '\' not currently implemented in privacy::summarize().' 
        
        return msg
